// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouettePlacementPipelineTests: C#-only (not a port; COLMAP has no silhouette placement).
// AutomaticReconstructionOptions.SilhouettePlacement wired into the controller
// (Controllers/AutomaticReconstruction.SilhouettePlacement.cs; divergence 144): an Object-mode
// run on a small TexturedSphere whose middle frames have their texture painted flat, so feature
// matching cannot place them but their outlines are intact. The run must place some of them from
// their outlines, finish every dense stage and give a mesh; a resumed workspace must not place a
// model twice; and the option is refused outside Object mode, without the sparse stage, and for
// photos not marked as frames in filming order.
//
// Known intrinsics and the scene's true masks (host masks, so no segmentation) keep it fast and
// the outcome independent of self-calibration. Tier C (outcome). Measured: features register
// 9/16, placement adds 7 (frames 6-9 and 0-2). The bars are only that something is placed and the
// pipeline finishes: a sphere's outline does not pin its rotation, so the placed poses are not
// accurate here (divergence 144 has the numbers).

using ColmapSharp.Controllers;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Mvs.Testing.Benchmark;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public class SilhouettePlacementPipelineTests
{
	private const int NumFrames = 16;

	// The frames whose texture is painted flat (a gap in the middle of the capture).
	private static readonly int[] FlatFrames = [6, 7, 8, 9];

	[Test]
	public async Task CSharpOnly_ObjectRunPlacesFramesFromOutlinesAndMeshes()
	{
		SyntheticObjectScene scene = GapScene();
		string workspace = NewWorkspace();
		try
		{
			SyntheticBenchmarkOptions options = Options(workspace, dense: true);
			// Texturing too: the placed frames have no points, and every dense step must take them.
			options.Texture = true;
			SyntheticBenchmarkResult result = SyntheticBenchmark.Run(scene, options);
			string summary = Marker(workspace);
			int placed = PlacedFrom(summary);
			Console.WriteLine($"{summary.Trim()}; registered {result.Metrics.NumRegistered}/{NumFrames}, surface {result.Metrics.SurfaceSource}, closed {result.MeshClosed}");

			await Assert.That(placed).IsGreaterThan(0);
			// The placed frames are in the model the dense stage read (sparse/0 was rewritten).
			var model = new Reconstruction();
			model.Read(Path.Combine(workspace, "sparse", "0"));
			int flatRegistered = FlatFrames.Count(k => model.FindImageWithName(SyntheticBenchmark.FrameName(k)) is { HasPose: true });
			await Assert.That(flatRegistered).IsGreaterThan(0);
			await Assert.That(result.Metrics.NumRegistered).IsEqualTo(model.NumRegImages);
			await Assert.That(result.Metrics.SurfaceSource).IsEqualTo("mesh");
			await Assert.That(result.MeshClosed).IsTrue();
			await Assert.That(File.Exists(Path.Combine(workspace, "dense", "0", "meshed-poisson-textured", "mesh.ply"))).IsTrue();
		}
		finally
		{
			Directory.Delete(workspace, recursive: true);
		}
	}

	[Test]
	public async Task CSharpOnly_ResumedModelIsNotPlacedTwice()
	{
		SyntheticObjectScene scene = GapScene();
		string workspace = NewWorkspace();
		try
		{
			SyntheticBenchmark.Run(scene, Options(workspace, dense: false));
			string sparse0 = Path.Combine(workspace, "sparse", "0");
			string first = Marker(workspace);
			var before = new Reconstruction();
			before.Read(sparse0);
			await Assert.That(PlacedFrom(first)).IsGreaterThan(0);

			var reports = new List<ControllerProgress>();
			SyntheticBenchmarkOptions resume = Options(workspace, dense: false);
			resume.Progress = new SyncProgress(reports.Add);
			SyntheticBenchmark.Run(scene, resume);

			var after = new Reconstruction();
			after.Read(sparse0);
			await Assert.That(reports.Any(r => r.Message == AutomaticReconstructionController.SilhouettePlacementMessage)).IsFalse();
			await Assert.That(Marker(workspace)).IsEqualTo(first);
			await Assert.That(after.NumRegImages).IsEqualTo(before.NumRegImages);
		}
		finally
		{
			Directory.Delete(workspace, recursive: true);
		}
	}

	[Test]
	public async Task CSharpOnly_SilhouettePlacementIsRefusedWhereItCannotRun()
	{
		string workspace = NewWorkspace();
		try
		{
			AutomaticReconstructionOptions Options(
				AutomaticReconstructionOptions.SubjectType subject, bool sparse, bool? timeOrdered = true) => new()
			{
				WorkspacePath = workspace,
				Images = new InMemoryImageSource(),
				Subject = subject,
				Sparse = sparse,
				FramesAreTimeOrdered = timeOrdered,
				SilhouettePlacement = true,
			};

			var scene = Assert.Throws<ArgumentException>(
				() => new AutomaticReconstructionController(Options(AutomaticReconstructionOptions.SubjectType.Scene, true), new ReconstructionManager()));
			await Assert.That(scene!.Message).Contains("Set Subject to Object");

			var noSparse = Assert.Throws<ArgumentException>(
				() => new AutomaticReconstructionController(Options(AutomaticReconstructionOptions.SubjectType.Object, false), new ReconstructionManager()));
			await Assert.That(noSparse!.Message).Contains("Turn Sparse on");

			// Individual data (the default) is not taken as filming order unless the host says so.
			var unordered = Assert.Throws<ArgumentException>(
				() => new AutomaticReconstructionController(Options(AutomaticReconstructionOptions.SubjectType.Object, true, null), new ReconstructionManager()));
			await Assert.That(unordered!.Message).Contains("Set FramesAreTimeOrdered to true");

			_ = new AutomaticReconstructionController(Options(AutomaticReconstructionOptions.SubjectType.Object, true), new ReconstructionManager());
		}
		finally
		{
			Directory.Delete(workspace, recursive: true);
		}
	}

	private static SyntheticBenchmarkOptions Options(string workspace, bool dense)
	{
		var options = new SyntheticBenchmarkOptions
		{
			MapperSeed = 1,
			Subject = AutomaticReconstructionOptions.SubjectType.Object,
			SilhouettePlacement = true,
			// Exhaustive matching (Individual data) of frames in filming order, as the demo runs.
			FramesAreTimeOrdered = true,
			KnownIntrinsics = true,
			UseTrueMasks = true,
			Dense = dense,
			PoissonDepth = 8,
			WorkspacePath = workspace,
		};
		options.Surface.SampleCount = 2000;
		return options;
	}

	// A small TexturedSphere with FlatFrames' object pixels set to that frame's mean object color:
	// the outline stays, the texture SIFT matches on is gone.
	private static SyntheticObjectScene GapScene()
	{
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(
			SyntheticObjectKind.TexturedSphere, NumFrames, 240, 180, seed: 1, motionDuration: 0.27);
		foreach (int k in FlatFrames)
		{
			Bitmap frame = scene.Frames[k];
			byte[] data = frame.RowMajorData, mask = scene.Masks[k].RowMajorData;
			int channels = frame.Channels;
			var sums = new long[channels];
			long count = 0;
			for (int p = 0; p < mask.Length; p++)
			{
				if (mask[p] >= 128)
				{
					count++;
					for (int c = 0; c < channels; c++)
					{
						sums[c] += data[p * channels + c];
					}
				}
			}

			for (int p = 0; p < mask.Length; p++)
			{
				if (mask[p] >= 128)
				{
					for (int c = 0; c < channels; c++)
					{
						data[p * channels + c] = (byte)(sums[c] / count);
					}
				}
			}
		}

		return scene;
	}

	private static string NewWorkspace()
	{
		string workspace = Path.Combine(Path.GetTempPath(), "colmap-sharp-silhouette-placement-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(workspace);
		return workspace;
	}

	private static string Marker(string workspace) =>
		File.ReadAllText(Path.Combine(workspace, "sparse", "0", AutomaticReconstructionController.SilhouettePlacementMarkerName));

	// The count out of "Placed N more photo(s) from their outlines (...)"; -1 for any other summary.
	private static int PlacedFrom(string summary)
	{
		string[] words = summary.Split(' ');
		return words.Length > 1 && words[0] == "Placed" && int.TryParse(words[1], out int n) ? n : -1;
	}

	private sealed class SyncProgress(Action<ControllerProgress> report) : IProgress<ControllerProgress>
	{
		public void Report(ControllerProgress value) => report(value);
	}
}
