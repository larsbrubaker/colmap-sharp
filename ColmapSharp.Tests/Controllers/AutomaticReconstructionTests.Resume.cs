// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AutomaticReconstructionTests (continued): C#-only tests, not ports. They pin how a new
// controller resumes over a workspace an earlier run left behind, the part of the dense
// stages' skip rule that docs/CPP_DIVERGENCES.md entry 135 changes:
// - With Texture on, a finished model keeps its fused.ply and mesh untouched (PatchMatch,
//   fusion and meshing are skipped) but is textured again, since the texture lives in memory.
// - With Texture off, a finished model is skipped entirely, as in COLMAP.
// - A mesh file that is not a PLY mesh (the zero-byte file an interrupted Poisson run leaves,
//   since PoissonMeshing.Run creates it before reconstructing) counts as missing, so the model
//   is re-meshed from its fused.ply instead of aborting every resume.
//   The re-mesh is byte-identical to the first run's mesh (docs/CPP_DIVERGENCES.md entry 137).
// - Cancelling during meshing leaves no partial mesh behind.
// - Texturing progress ends at 100% even for an empty mesh.
// One first run (the textured scene of AutomaticReconstructionTests.CSharpOnly.cs, Delaunay
// meshing) feeds every phase to keep the runtime down; each later controller skips extraction
// and matching and reads the sparse model back from the workspace.

using ColmapSharp.Controllers;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class AutomaticReconstructionTests
{
	[Test]
	public async Task CSharpOnly_ResumeReusesDenseResultsAndRecoversPartialMesh()
	{
		const int NumViews = 4;
		string testDir = CreateTestDir();
		string workspacePath = Path.Combine(testDir, "workspace");
		Directory.CreateDirectory(workspacePath);
		var images = new InMemoryImageSource();
		for (int i = 0; i < NumViews; ++i)
		{
			images.Add($"view{i}.png", RenderTexturedScene(i, NumViews, width: 200, height: 150));
		}

		AutomaticReconstructionOptions Options(bool resume) => new()
		{
			WorkspacePath = workspacePath,
			Images = images,
			Data = AutomaticReconstructionOptions.DataType.Individual,
			Quality = AutomaticReconstructionOptions.QualityLevel.Low,
			Dense = true,
			RandomSeed = 1,
			Mesher = AutomaticReconstructionOptions.MesherType.Delaunay,
			Extraction = !resume,
			Matching = !resume,
		};

		string densePath = Path.Combine(workspacePath, "dense", "0");
		string fusedPath = Path.Combine(densePath, "fused.ply");
		string meshPath = Path.Combine(densePath, "meshed-delaunay.ply");
		string poissonPath = Path.Combine(densePath, "meshed-poisson.ply");

		(AutomaticReconstructionController Controller, ReportLog Log) RunController(AutomaticReconstructionOptions options)
		{
			var log = new ReportLog();
			var controller = new AutomaticReconstructionController(options, new ReconstructionManager()) { Progress = log };
			controller.Setup();
			controller.Run();
			return (controller, log);
		}

		// The first run.
		int firstTextured = RunController(Options(resume: false)).Controller.TexturedMeshes.Count;
		(long, DateTime) fusedStamp = Stamp(fusedPath);
		(long, DateTime) meshStamp = Stamp(meshPath);
		byte[] firstMeshBytes = File.ReadAllBytes(meshPath);

		// A resume with Texture on: dense results untouched, the model textured again.
		(AutomaticReconstructionController textureResume, ReportLog textureLog) = RunController(Options(resume: true));
		bool textureResumeKeptFused = Stamp(fusedPath) == fusedStamp;
		bool textureResumeKeptMesh = Stamp(meshPath) == meshStamp;
		int textureResumeTextured = textureResume.TexturedMeshes.Count;
		int textureResumeFaces = textureResume.TexturedMeshes.Count == 1 ? textureResume.TexturedMeshes[0].Mesh.Faces.Count : -1;
		string textureResumeStages = string.Join(" | ", textureLog.Stages());

		// A resume with Texture off: the model skipped entirely (only the dense heading).
		AutomaticReconstructionOptions noTexture = Options(resume: true);
		noTexture.Texture = false;
		(AutomaticReconstructionController plainResume, ReportLog plainLog) = RunController(noTexture);
		bool plainResumeKeptFused = Stamp(fusedPath) == fusedStamp;
		bool plainResumeKeptMesh = Stamp(meshPath) == meshStamp;
		int plainResumeTextured = plainResume.TexturedMeshes.Count;
		int plainResumeDenseReports = plainLog.Count(AutomaticReconstructionController.DenseStage);
		string plainResumeStages = string.Join(" | ", plainLog.Stages());

		// A resume over a zero-byte mesh: re-meshed from the kept fused.ply, then textured.
		File.WriteAllBytes(meshPath, []);
		(AutomaticReconstructionController remeshResume, ReportLog remeshLog) = RunController(Options(resume: true));
		bool remeshKeptFused = Stamp(fusedPath) == fusedStamp;
		long remeshMeshLength = new FileInfo(meshPath).Length;
		bool remeshMatchesFirstRun = File.ReadAllBytes(meshPath).AsSpan().SequenceEqual(firstMeshBytes);
		int remeshFaces = Ply.ReadPlyMesh(meshPath).Mesh.Faces.Count;
		int remeshTextured = remeshResume.TexturedMeshes.Count;
		string remeshStages = string.Join(" | ", remeshLog.Stages());

		// Cancelling during Poisson meshing, which creates its output file before it starts.
		AutomaticReconstructionOptions poisson = Options(resume: true);
		poisson.Mesher = AutomaticReconstructionOptions.MesherType.Poisson;
		using var cancellation = new CancellationTokenSource();
		var cancelLog = new ReportLog(onReport: p =>
		{
			if (p.Stage == AutomaticReconstructionController.MeshingStage)
			{
				cancellation.Cancel();
			}
		});
		var cancelled = new AutomaticReconstructionController(poisson, new ReconstructionManager())
		{
			Progress = cancelLog,
			CancellationToken = cancellation.Token,
		};
		cancelled.Setup();
		bool threwCancelled = false;
		try
		{
			cancelled.Run();
		}
		catch (OperationCanceledException)
		{
			threwCancelled = true;
		}

		bool poissonMeshLeft = File.Exists(poissonPath);
		bool cancelReachedMeshing = cancelLog.Count(AutomaticReconstructionController.MeshingStage) > 0;

		// Poisson meshing run through: COLMAP's default trim leaves this small cloud an empty
		// mesh (see AutomaticReconstructionTests.CSharpOnly.cs), which MeshTextureMapping
		// returns early for; texturing progress must still end at 100%.
		(AutomaticReconstructionController emptyResume, ReportLog emptyLog) = RunController(poisson);
		int emptyFaces = Ply.ReadPlyMesh(poissonPath).Mesh.Faces.Count;
		int emptyTextured = emptyResume.TexturedMeshes.Count;
		ControllerProgress? lastTexturing = emptyLog.Last(AutomaticReconstructionController.TexturingStage);

		try
		{
			Directory.Delete(testDir, recursive: true);
		}
		catch (IOException)
		{
			// A leftover temp folder is harmless.
		}

		await Assert.That(firstTextured).IsEqualTo(1);

		await Assert.That(textureResumeKeptFused).IsTrue();
		await Assert.That(textureResumeKeptMesh).IsTrue();
		await Assert.That(textureResumeTextured).IsEqualTo(1);
		await Assert.That(textureResumeFaces).IsGreaterThan(0);
		await Assert.That(textureResumeStages).IsEqualTo(string.Join(" | ", new[]
		{
			AutomaticReconstructionController.SparseStage,
			AutomaticReconstructionController.DenseStage,
			AutomaticReconstructionController.TexturingStage,
		}));

		await Assert.That(plainResumeKeptFused).IsTrue();
		await Assert.That(plainResumeKeptMesh).IsTrue();
		await Assert.That(plainResumeTextured).IsEqualTo(0);
		await Assert.That(plainResumeDenseReports).IsEqualTo(1);
		await Assert.That(plainResumeStages).IsEqualTo(string.Join(" | ", new[]
		{
			AutomaticReconstructionController.SparseStage,
			AutomaticReconstructionController.DenseStage,
		}));

		await Assert.That(remeshKeptFused).IsTrue();
		await Assert.That(remeshMeshLength).IsGreaterThan(0);
		await Assert.That(remeshMatchesFirstRun).IsTrue();
		await Assert.That(remeshFaces).IsGreaterThan(0);
		await Assert.That(remeshTextured).IsEqualTo(1);
		await Assert.That(remeshStages).IsEqualTo(string.Join(" | ", new[]
		{
			AutomaticReconstructionController.SparseStage,
			AutomaticReconstructionController.DenseStage,
			AutomaticReconstructionController.MeshingStage,
			AutomaticReconstructionController.TexturingStage,
		}));

		await Assert.That(cancelReachedMeshing).IsTrue();
		await Assert.That(threwCancelled).IsTrue();
		await Assert.That(poissonMeshLeft).IsFalse();

		await Assert.That(emptyFaces).IsEqualTo(0);
		await Assert.That(emptyTextured).IsEqualTo(1);
		await Assert.That(lastTexturing).IsNotNull();
		await Assert.That(lastTexturing!.Value.Done).IsEqualTo(lastTexturing.Value.Total);
		await Assert.That(lastTexturing.Value.Total).IsGreaterThan(0);
	}

	private static (long Length, DateTime LastWrite) Stamp(string path)
	{
		var info = new FileInfo(path);
		return (info.Exists ? info.Length : -1, info.Exists ? info.LastWriteTimeUtc : default);
	}

	// Records every report in order, and runs onReport on each. Synchronous and
	// locked: the stages report from worker threads.
	private sealed class ReportLog(Action<ControllerProgress>? onReport = null) : IProgress<ControllerProgress>
	{
		private readonly List<ControllerProgress> reports = [];

		public void Report(ControllerProgress value)
		{
			lock (reports)
			{
				reports.Add(value);
			}

			onReport?.Invoke(value);
		}

		public List<string> Stages()
		{
			lock (reports)
			{
				return [.. reports.Select(r => r.Stage).Distinct()];
			}
		}

		public int Count(string stage)
		{
			lock (reports)
			{
				return reports.Count(r => r.Stage == stage);
			}
		}

		public ControllerProgress? Last(string stage)
		{
			lock (reports)
			{
				return reports.Where(r => r.Stage == stage).Select(r => (ControllerProgress?)r).LastOrDefault();
			}
		}
	}
}
