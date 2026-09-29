// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AutomaticReconstructionTests (continued): a C#-only test, not a port. It pins
// docs/CPP_DIVERGENCES.md entry 140: ReconstructionManager.Write puts the model with the most
// 3D points in sparse/0, and the dense stage must index its models the same way, so dense/<i>
// is built from sparse/<i> both on the first run (models in memory in mapper build order) and
// on a resume (models read back from sparse/<i>). In COLMAP the first run walks the models in
// build order, so with two or more models dense/<i> can belong to another model than
// sparse/<i>, and a resume then mixes one model's depth maps with another's.
//
// The mapper cannot be made to build a smaller model first reliably on a scene this small, so
// the first run is handed two models in build order [small, large] through the injected
// ReconstructionManager, with an empty database: the incremental mapper then finds no images
// and adds nothing, and the controller's own RunMapper writes sparse/ and runs dense as it
// would after a real mapper run. The small model is the textured scene's model with most of
// its 3D points deleted, so the two models are told apart by their point counts.
//
// CSharpOnly_ResumeReadsSparseModelsInNumericOrder pins the other half of entry 140: a resume
// reads sparse/<i> by the number i (COLMAP sorts the directory names as strings, so with 11
// models sparse/10 would become model 2).

using ColmapSharp.Controllers;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class AutomaticReconstructionTests
{
	[Test]
	public async Task CSharpOnly_DenseModelsFollowSparseOrderOnFirstRunAndResume()
	{
		const int NumViews = 4;
		string testDir = CreateTestDir();
		var images = new InMemoryImageSource();
		for (int i = 0; i < NumViews; ++i)
		{
			images.Add($"view{i}.png", SyntheticObjectScene.RenderTexturedSphereOnWall(i, NumViews, width: 200, height: 150));
		}

		AutomaticReconstructionOptions Options(string workspacePath, bool extractAndMatch, bool dense) => new()
		{
			WorkspacePath = workspacePath,
			Images = images,
			Data = AutomaticReconstructionOptions.DataType.Individual,
			Quality = AutomaticReconstructionOptions.QualityLevel.Low,
			Dense = dense,
			RandomSeed = 1,
			Mesher = AutomaticReconstructionOptions.MesherType.Delaunay,
			Extraction = extractAndMatch,
			Matching = extractAndMatch,
		};

		AutomaticReconstructionController RunController(AutomaticReconstructionOptions options, ReconstructionManager manager)
		{
			var controller = new AutomaticReconstructionController(options, manager);
			controller.Setup();
			controller.Run();
			return controller;
		}

		// One sparse model of the scene, the source of both test models.
		string sourceWorkspace = Path.Combine(testDir, "source");
		Directory.CreateDirectory(sourceWorkspace);
		RunController(Options(sourceWorkspace, extractAndMatch: true, dense: false), new ReconstructionManager());
		string sourceModel = Path.Combine(sourceWorkspace, "sparse", "0");

		var large = new Reconstruction();
		large.Read(sourceModel);
		var small = new Reconstruction();
		small.Read(sourceModel);
		foreach (ulong point3DId in small.Point3DIds().OrderBy(id => id).Skip(small.NumPoints3D / 3))
		{
			small.DeletePoint3D(point3DId);
		}

		int largePoints = large.NumPoints3D;
		int smallPoints = small.NumPoints3D;

		// The first run: models in build order [small, large].
		string workspacePath = Path.Combine(testDir, "workspace");
		Directory.CreateDirectory(workspacePath);
		var firstManager = new ReconstructionManager();
		firstManager.Add();
		firstManager.Set(0, small);
		firstManager.Add();
		firstManager.Set(1, large);
		AutomaticReconstructionController first = RunController(Options(workspacePath, extractAndMatch: false, dense: true), firstManager);

		int[] sparsePoints = [.. Enumerable.Range(0, 2).Select(i => ModelPoints(Path.Combine(workspacePath, "sparse", $"{i}")))];
		int[] firstDensePoints = [.. Enumerable.Range(0, 2).Select(i => ModelPoints(Path.Combine(workspacePath, "dense", $"{i}", "sparse")))];
		int[] firstMemoryPoints = [.. Enumerable.Range(0, firstManager.Size).Select(i => firstManager.Get(i).NumPoints3D)];
		Dictionary<string, byte[]> firstProducts = AllDenseProducts(WorkspaceFiles(workspacePath));
		int[] firstTexturedIdx = [.. first.TexturedMeshes.Select(m => m.ModelIdx)];
		int[] firstTexturedFaces = [.. first.TexturedMeshes.Select(m => m.Mesh.Faces.Count)];

		// The resume: dense/ removed, the models read back from sparse/<i>.
		Directory.Delete(Path.Combine(workspacePath, "dense"), recursive: true);
		var resumeManager = new ReconstructionManager();
		AutomaticReconstructionController resume = RunController(Options(workspacePath, extractAndMatch: false, dense: true), resumeManager);

		int[] resumeDensePoints = [.. Enumerable.Range(0, 2).Select(i => ModelPoints(Path.Combine(workspacePath, "dense", $"{i}", "sparse")))];
		int[] resumeMemoryPoints = [.. Enumerable.Range(0, resumeManager.Size).Select(i => resumeManager.Get(i).NumPoints3D)];
		Dictionary<string, byte[]> resumeProducts = AllDenseProducts(WorkspaceFiles(workspacePath));
		int[] resumeTexturedIdx = [.. resume.TexturedMeshes.Select(m => m.ModelIdx)];
		int[] resumeTexturedFaces = [.. resume.TexturedMeshes.Select(m => m.Mesh.Faces.Count)];
		string differences = Differences(resumeProducts, firstProducts);

		DeleteTestDir(testDir);

		await Assert.That(largePoints).IsGreaterThan(smallPoints);
		await Assert.That(smallPoints).IsGreaterThan(0);

		// sparse/0 is the larger model, as ReconstructionManager.Write orders them.
		await Assert.That(Join(sparsePoints)).IsEqualTo(Join(new[] { largePoints, smallPoints }));

		// dense/<i> is built from sparse/<i> on both runs, and the in-memory order agrees.
		await Assert.That(Join(firstDensePoints)).IsEqualTo(Join(sparsePoints));
		await Assert.That(Join(resumeDensePoints)).IsEqualTo(Join(sparsePoints));
		await Assert.That(Join(firstMemoryPoints)).IsEqualTo(Join(sparsePoints));
		await Assert.That(Join(resumeMemoryPoints)).IsEqualTo(Join(sparsePoints));

		// Both runs write the same dense products for each model.
		await Assert.That(firstProducts.Keys.Any(k => k.EndsWith("fused.ply", StringComparison.Ordinal))).IsTrue();
		await Assert.That(differences).IsEqualTo("");

		// TexturedMeshes[].ModelIdx names the same dense/<i> on both runs.
		await Assert.That(Join(firstTexturedIdx)).IsEqualTo(Join(new[] { 0, 1 }));
		await Assert.That(Join(resumeTexturedIdx)).IsEqualTo(Join(firstTexturedIdx));
		await Assert.That(Join(resumeTexturedFaces)).IsEqualTo(Join(firstTexturedFaces));
	}

	[Test]
	public async Task CSharpOnly_ResumeReadsSparseModelsInNumericOrder()
	{
		const int NumModels = 11;
		string testDir = CreateTestDir();
		string workspacePath = Path.Combine(testDir, "workspace");

		// sparse/<i> holds i + 1 points, so a model's point count names its directory.
		for (int i = 0; i < NumModels; ++i)
		{
			var reconstruction = new Reconstruction();
			for (int j = 0; j <= i; ++j)
			{
				reconstruction.AddPoint3D(new ColmapSharp.LinearAlgebra.Vector3d(j, i, 0), new Track());
			}

			string modelPath = Path.Combine(workspacePath, "sparse", $"{i}");
			Directory.CreateDirectory(modelPath);
			reconstruction.Write(modelPath);
		}

		var manager = new ReconstructionManager();
		var controller = new AutomaticReconstructionController(new AutomaticReconstructionOptions
		{
			WorkspacePath = workspacePath,
			Images = new InMemoryImageSource(),
			Extraction = false,
			Matching = false,
			Dense = false,
		}, manager);
		controller.Setup();
		controller.Run();

		int[] readPoints = [.. Enumerable.Range(0, manager.Size).Select(i => manager.Get(i).NumPoints3D)];

		DeleteTestDir(testDir);

		await Assert.That(Join(readPoints)).IsEqualTo(Join(Enumerable.Range(1, NumModels)));
	}

	// The dense products of every model: DenseProducts' filter (no undistorted sparse model,
	// written in undistortion order, and no .cfg) over dense/0 and dense/1.
	private static Dictionary<string, byte[]> AllDenseProducts(Dictionary<string, byte[]> files) =>
		files.Where(f => f.Key.StartsWith("dense" + Path.DirectorySeparatorChar, StringComparison.Ordinal)
				&& !f.Key.Contains(Path.DirectorySeparatorChar + "sparse" + Path.DirectorySeparatorChar, StringComparison.Ordinal)
				&& !f.Key.EndsWith(".cfg", StringComparison.Ordinal))
			.ToDictionary(f => f.Key, f => f.Value, StringComparer.Ordinal);

	// Order-sensitive text for a sequence, so an assertion shows both sides.
	private static string Join(IEnumerable<int> values) => string.Join(", ", values);

	// The number of 3D points of the model in path, or -1 when there is none.
	private static int ModelPoints(string path)
	{
		if (!Directory.Exists(path))
		{
			return -1;
		}

		var reconstruction = new Reconstruction();
		reconstruction.Read(path);
		return reconstruction.NumPoints3D;
	}
}
