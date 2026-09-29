// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchControllerTests: C#-only tests (COLMAP has no patch_match_test.cc) for
// ColmapSharp/Mvs/PatchMatchController.cs: reading patch-match.cfg into problems
// ("__all__", "__auto__, N", explicit lists, comments), the "__auto__" ranking and its tie
// order (divergence 84), loading one problem's inputs (SetUpProblem,
// including the used-image order of entry 85), and Run: the files it writes, the skip of
// finished problems, the photometric-then-geometric passes, a model in memory,
// cancellation, and the warnings it logs. The workspace is a synthetic COLMAP reconstruction
// written to disk, with grey bitmaps served from memory. Tier A bookkeeping (the maps
// themselves are PatchMatchRunTests' concern).

using ColmapSharp.Mathematics;
using ColmapSharp.Mvs;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public partial class PatchMatchControllerTests
{
	private const int CameraWidth = 20;
	private const int CameraHeight = 10;

	/// <summary>
	/// A four-image workspace from the synthetic dataset. With unequal overlap, image 1 loses
	/// 3 and image 2 loses 6 of its observations, so image 0 shares 15, 13 and 20 points
	/// with images 1, 2 and 3, and image 1 shares 15, 11 and 15 with images 0, 2 and 3.
	/// </summary>
	private sealed class Fixture
	{
		public Fixture(bool unequalOverlap = true, bool withPoints = true)
		{
			TempDir = MvsTestUtils.CreateTestDir();
			Directory.CreateDirectory(Path.Combine(TempDir, "sparse"));
			Directory.CreateDirectory(Path.Combine(TempDir, "stereo", "depth_maps"));
			Directory.CreateDirectory(Path.Combine(TempDir, "stereo", "normal_maps"));

			var syntheticOptions = new SyntheticDatasetOptions
			{
				NumRigs = 1,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 4,
				NumPoints3D = 20,
				CameraWidth = CameraWidth,
				CameraHeight = CameraHeight,
				CameraModelId = CameraModelId.Pinhole,
				CameraParams = [25, 25, 10, 5],
			};
			var reconstruction = new Reconstruction();
			Synthetic.SynthesizeDataset(syntheticOptions, reconstruction);

			List<uint> imageIds = reconstruction.RegImageIds();
			if (unequalOverlap)
			{
				DeleteObservations(reconstruction, imageIds[1], 3);
				DeleteObservations(reconstruction, imageIds[2], 6);
			}

			if (!withPoints)
			{
				reconstruction.DeleteAllPoints2DAndPoints3D();
			}

			reconstruction.Write(Path.Combine(TempDir, "sparse"));

			foreach (uint imageId in imageIds)
			{
				string name = reconstruction.Image(imageId).Name;
				Names.Add(name);
				var bitmap = new Bitmap(CameraWidth, CameraHeight, asRgb: false);
				bitmap.Fill(new BitmapColor<byte>(7));
				Bitmaps.Add(Path.Combine(TempDir, "images", name), bitmap);
			}
		}

		public string TempDir { get; }

		/// <summary>Image names in model index order (the registered image order).</summary>
		public List<string> Names { get; } = new();

		public MvsTestUtils.InMemoryBitmapSource Bitmaps { get; } = new();

		public void WriteConfig(params string[] lines) =>
			File.WriteAllText(Path.Combine(TempDir, "stereo", "patch-match.cfg"), string.Join("\n", lines) + "\n");

		public PatchMatchController Open(PatchMatchOptions options, string configPath = "")
		{
			var controller = new PatchMatchController(options, TempDir, "COLMAP", "", Bitmaps, configPath);
			controller.ReadWorkspace();
			controller.ReadProblems();
			return controller;
		}

		public void WriteMaps(int imageIdx, string inputType)
		{
			string fileName = Names[imageIdx] + "." + inputType + ".bin";
			var depthMap = new Mat<float>(CameraWidth, CameraHeight, 1);
			depthMap.Fill(imageIdx + 1.0f);
			depthMap.Write(Path.Combine(TempDir, "stereo", "depth_maps", fileName));
			var normalMap = new Mat<float>(CameraWidth, CameraHeight, 3);
			normalMap.Fill(-1.0f);
			normalMap.Write(Path.Combine(TempDir, "stereo", "normal_maps", fileName));
		}

		private static void DeleteObservations(Reconstruction reconstruction, uint imageId, int count)
		{
			ColmapSharp.Scene.Image image = reconstruction.Image(imageId);
			int deleted = 0;
			for (int point2DIdx = 0; point2DIdx < image.Points2D.Count && deleted < count; point2DIdx++)
			{
				if (image.Points2D[point2DIdx].HasPoint3D)
				{
					reconstruction.DeleteObservation(imageId, (uint)point2DIdx);
					deleted++;
				}
			}
		}
	}

	private static PatchMatchOptions PhotometricOptions() => new() { GeomConsistency = false, MinTriangulationAngle = 0 };

	[Test]
	public async Task ReadProblems_AllExplicitAndComments()
	{
		var fixture = new Fixture();
		fixture.WriteConfig(
			"# a comment",
			fixture.Names[0],
			"__all__",
			"",
			"  " + fixture.Names[2] + "  ",
			fixture.Names[3] + "; " + fixture.Names[1] + ",, ",
			"# trailing reference without a source line",
			fixture.Names[1]);

		PatchMatchController controller = fixture.Open(PhotometricOptions());
		await Assert.That(controller.Problems.Count).IsEqualTo(2);
		await Assert.That(controller.Problems[0].RefImageIdx).IsEqualTo(0);
		await Assert.That(controller.Problems[0].SrcImageIdxs).IsEquivalentTo([1, 2, 3], CollectionOrdering.Matching);
		await Assert.That(controller.Problems[1].RefImageIdx).IsEqualTo(2);
		await Assert.That(controller.Problems[1].SrcImageIdxs).IsEquivalentTo([3, 1], CollectionOrdering.Matching);
	}

	[Test]
	public async Task ReadProblems_AutoRanksBySharedPoints()
	{
		var fixture = new Fixture();
		fixture.WriteConfig(fixture.Names[0], "__auto__, 2", fixture.Names[1], "__auto__, 10", fixture.Names[2], "__auto__, -1");

		PatchMatchController controller = fixture.Open(PhotometricOptions());
		await Assert.That(controller.Problems.Count).IsEqualTo(3);

		// The premise (see Fixture).
		List<SortedDictionary<int, int>> shared = controller.Workspace.GetModel().ComputeSharedPoints();
		await Assert.That(shared[0].Values.ToArray()).IsEquivalentTo([15, 13, 20], CollectionOrdering.Matching);
		await Assert.That(shared[1].Values.ToArray()).IsEquivalentTo([15, 11, 15], CollectionOrdering.Matching);

		// Image 0: images 3 (20 shared points) and 1 (15).
		await Assert.That(controller.Problems[0].SrcImageIdxs).IsEquivalentTo([3, 1], CollectionOrdering.Matching);

		// Image 1 shares 15 points with images 0 and 3 (a tie, lower index first) and 11
		// with image 2.
		await Assert.That(controller.Problems[1].SrcImageIdxs).IsEquivalentTo([0, 3, 2], CollectionOrdering.Matching);

		// std::stoll("-1") converted to size_t is the maximum count, so every image is used.
		await Assert.That(controller.Problems[2].SrcImageIdxs.Count).IsEqualTo(3);
	}

	[Test]
	public async Task ReadProblems_AutoTiesGoToLowerImageIndex()
	{
		var fixture = new Fixture(unequalOverlap: false);
		fixture.WriteConfig(fixture.Names[2], "__auto__, 2");

		PatchMatchController controller = fixture.Open(PhotometricOptions());

		// The premise: image 2 shares 19 points with images 0 and 3 and 17 with image 1.
		SortedDictionary<int, int> shared = controller.Workspace.GetModel().ComputeSharedPoints()[2];
		await Assert.That(shared[0]).IsEqualTo(19);
		await Assert.That(shared[1]).IsEqualTo(17);
		await Assert.That(shared[3]).IsEqualTo(19);

		await Assert.That(controller.Problems[0].SrcImageIdxs).IsEquivalentTo([0, 3], CollectionOrdering.Matching);
	}

	[Test]
	public async Task ReadProblems_AutoDropsSmallTriangulationAngles()
	{
		// No pair of images reaches a 179 degree triangulation angle, so the reference image
		// is left without sources and ignored.
		var fixture = new Fixture();
		fixture.WriteConfig(fixture.Names[0], "__auto__, 3", fixture.Names[1], fixture.Names[0]);

		var options = PhotometricOptions();
		options.MinTriangulationAngle = 179;
		PatchMatchController controller = fixture.Open(options);
		await Assert.That(controller.Problems.Count).IsEqualTo(1);
		await Assert.That(controller.Problems[0].RefImageIdx).IsEqualTo(1);
	}

	[Test]
	public async Task ReadProblems_UnknownImageThrows()
	{
		var fixture = new Fixture();
		fixture.WriteConfig("missing.png", "__all__");
		await Assert.That(() => fixture.Open(PhotometricOptions())).Throws<Exception>();

		fixture.WriteConfig(fixture.Names[0], "missing.png");
		await Assert.That(() => fixture.Open(PhotometricOptions())).Throws<Exception>();
	}

	[Test]
	public async Task ReadProblems_ConfigPathOverride()
	{
		var fixture = new Fixture();
		fixture.WriteConfig(fixture.Names[0], "__all__");
		string configPath = Path.Combine(fixture.TempDir, "custom.cfg");
		File.WriteAllText(configPath, fixture.Names[3] + "\n" + fixture.Names[0] + "\n");

		PatchMatchController controller = fixture.Open(PhotometricOptions(), configPath);
		await Assert.That(controller.Problems.Count).IsEqualTo(1);
		await Assert.That(controller.Problems[0].RefImageIdx).IsEqualTo(3);
		await Assert.That(controller.Problems[0].SrcImageIdxs).IsEquivalentTo([0], CollectionOrdering.Matching);
	}

	[Test]
	public async Task SetUpProblem_Photometric()
	{
		var fixture = new Fixture();
		fixture.WriteConfig(fixture.Names[1], fixture.Names[3] + ", " + fixture.Names[0]);
		PatchMatchOptions options = PhotometricOptions();
		options.FilterMinNumConsistent = 5;
		PatchMatchController controller = fixture.Open(options);

		(PatchMatchOptions problemOptions, PatchMatch.Problem problem) = controller.SetUpProblem(options, 0);

		// The depth range comes from the sparse model, sigma_spatial from the window radius
		// and filter_min_num_consistent is capped by the number of source images.
		(float min, float max) = controller.Workspace.GetModel().ComputeDepthRanges()[1];
		await Assert.That(problemOptions.DepthMin).IsEqualTo((double)min);
		await Assert.That(problemOptions.DepthMax).IsEqualTo((double)max);
		await Assert.That(problemOptions.DepthMin).IsGreaterThan(0.0);
		await Assert.That(problemOptions.SigmaSpatial).IsEqualTo(5.0);
		await Assert.That(problemOptions.FilterMinNumConsistent).IsEqualTo(2);
		await Assert.That(options.DepthMin).IsEqualTo(-1.0);

		await Assert.That(problem.RefImageIdx).IsEqualTo(1);
		await Assert.That(problem.SrcImageIdxs).IsEquivalentTo([3, 0], CollectionOrdering.Matching);
		await Assert.That(problem.Images!.Count).IsEqualTo(4);
		await Assert.That(problem.DepthMaps!.Count).IsEqualTo(0);
		foreach (int imageIdx in new[] { 0, 1, 3 })
		{
			Bitmap bitmap = problem.Images[imageIdx].GetBitmap();
			await Assert.That(bitmap.IsGrey).IsTrue();
			await Assert.That(bitmap.Width).IsEqualTo(CameraWidth);
			await Assert.That(bitmap.GetPixel(3, 4)!.Value.R).IsEqualTo((byte)7);
		}

		await Assert.That(problem.Images[2].GetBitmap().IsEmpty).IsTrue();

		// The controller's own problem gets the (here unchanged) source list, not the inputs.
		await Assert.That(controller.Problems[0].Images).IsNull();
		await Assert.That(controller.Problems[0].SrcImageIdxs).IsEquivalentTo([3, 0], CollectionOrdering.Matching);

		new PatchMatch(problemOptions, problem).Check();
	}

	[Test]
	public async Task SetUpProblem_KeepsGivenDepthRange()
	{
		var fixture = new Fixture();
		fixture.WriteConfig(fixture.Names[0], "__all__");
		PatchMatchOptions options = PhotometricOptions();
		options.DepthMin = 0.5;
		options.DepthMax = 7;
		options.SigmaSpatial = 2.5;
		PatchMatchController controller = fixture.Open(options);

		(PatchMatchOptions problemOptions, _) = controller.SetUpProblem(options, 0);
		await Assert.That(problemOptions.DepthMin).IsEqualTo(0.5);
		await Assert.That(problemOptions.DepthMax).IsEqualTo(7.0);
		await Assert.That(problemOptions.SigmaSpatial).IsEqualTo(2.5);
	}

	[Test]
	public async Task SetUpProblem_GeometricReadsPhotometricMaps()
	{
		var fixture = new Fixture();
		fixture.WriteConfig(fixture.Names[2], fixture.Names[0] + ", " + fixture.Names[1]);
		for (int imageIdx = 0; imageIdx < 3; imageIdx++)
		{
			fixture.WriteMaps(imageIdx, "photometric");
		}

		var options = new PatchMatchOptions { MinTriangulationAngle = 0 };
		PatchMatchController controller = fixture.Open(options);
		(PatchMatchOptions problemOptions, PatchMatch.Problem problem) = controller.SetUpProblem(options, 0);

		await Assert.That(problem.DepthMaps!.Count).IsEqualTo(4);
		await Assert.That(problem.NormalMaps!.Count).IsEqualTo(4);
		for (int imageIdx = 0; imageIdx < 3; imageIdx++)
		{
			await Assert.That(problem.DepthMaps[imageIdx].Get(1, 2)).IsEqualTo(imageIdx + 1.0f);
			await Assert.That(problem.NormalMaps[imageIdx].Get(1, 2, 0)).IsEqualTo(-1.0f);
		}

		await Assert.That(problem.DepthMaps[3].GetWidth()).IsEqualTo(0);
		new PatchMatch(problemOptions, problem).Check();
	}

	[Test]
	public async Task SetUpProblem_MissingFiles()
	{
		var fixture = new Fixture();
		fixture.WriteConfig(fixture.Names[0], "__all__");
		for (int imageIdx = 0; imageIdx < 4; imageIdx++)
		{
			if (imageIdx != 2)
			{
				fixture.WriteMaps(imageIdx, "photometric");
			}
		}

		// Without AllowMissingFiles, reading the missing maps fails.
		var options = new PatchMatchOptions { MinTriangulationAngle = 0 };
		PatchMatchController controller = fixture.Open(options);
		await Assert.That(() => controller.SetUpProblem(options, 0)).Throws<Exception>();

		// With it, the source image without maps is dropped.
		options.AllowMissingFiles = true;
		controller = fixture.Open(options);
		(_, PatchMatch.Problem problem) = controller.SetUpProblem(options, 0);
		await Assert.That(problem.SrcImageIdxs).IsEquivalentTo([1, 3], CollectionOrdering.Matching);
	}

	[Test]
	public async Task SetUpProblem_GeometricPassStartsFromFilteredSources()
	{
		// COLMAP writes the filtered source list back into the controller's problem, so the
		// geometric pass sees only the sources the photometric pass kept, and
		// filter_min_num_consistent is capped by that smaller count.
		var fixture = new Fixture();
		fixture.WriteConfig(fixture.Names[0], fixture.Names[1] + ", " + fixture.Names[2]);
		fixture.Bitmaps.Remove(Path.Combine(fixture.TempDir, "images", fixture.Names[2]));
		fixture.WriteMaps(0, "photometric");
		fixture.WriteMaps(1, "photometric");

		var options = new PatchMatchOptions { MinTriangulationAngle = 0, AllowMissingFiles = true };
		PatchMatchController controller = fixture.Open(options);

		PatchMatchOptions photometricOptions = options.Clone();
		photometricOptions.GeomConsistency = false;
		photometricOptions.Filter = false;
		(_, PatchMatch.Problem photometricProblem) = controller.SetUpProblem(photometricOptions, 0);
		await Assert.That(photometricProblem.SrcImageIdxs).IsEquivalentTo([1], CollectionOrdering.Matching);
		await Assert.That(controller.Problems[0].SrcImageIdxs).IsEquivalentTo([1], CollectionOrdering.Matching);

		(PatchMatchOptions geometricOptions, PatchMatch.Problem geometricProblem) = controller.SetUpProblem(options, 0);
		await Assert.That(geometricProblem.SrcImageIdxs).IsEquivalentTo([1], CollectionOrdering.Matching);
		await Assert.That(geometricOptions.FilterMinNumConsistent).IsEqualTo(1);
	}

	[Test]
	public async Task SetUpProblem_NoSparseDepthRangeThrows()
	{
		// An image with no points has no depth range; the depth range must then be set.
		var fixture = new Fixture(unequalOverlap: false, withPoints: false);
		fixture.WriteConfig(fixture.Names[0], "__all__");

		PatchMatchOptions options = PhotometricOptions();
		PatchMatchController controller = fixture.Open(options);
		await Assert.That(() => controller.SetUpProblem(options, 0)).Throws<ArgumentException>();
	}

	private static PatchMatchOptions RunOptions(bool geomConsistency) => new()
	{
		MinTriangulationAngle = 0,
		NumIterations = 1,
		WindowRadius = 2,
		NumSamples = 3,
		GeomConsistency = geomConsistency,
		Filter = geomConsistency,
		WriteConsistencyGraph = geomConsistency,
	};

	private static string MapPath(Fixture fixture, string kind, int imageIdx, string type) =>
		Path.Combine(fixture.TempDir, "stereo", kind, fixture.Names[imageIdx] + "." + type + ".bin");

	[Test]
	public async Task Run_PhotometricWritesMapsAndSkipsExisting()
	{
		var fixture = new Fixture();
		fixture.WriteConfig(fixture.Names[0], "__all__", fixture.Names[1], fixture.Names[0] + ", " + fixture.Names[2]);
		var reports = new List<ColmapSharp.Controllers.ControllerProgress>();
		var controller = new PatchMatchController(RunOptions(false), fixture.TempDir, "COLMAP", "", fixture.Bitmaps);
		controller.Run(progress: new SynchronousProgress<ColmapSharp.Controllers.ControllerProgress>(reports));

		foreach (int imageIdx in new[] { 0, 1 })
		{
			var depthMap = new DepthMap();
			depthMap.Read(MapPath(fixture, "depth_maps", imageIdx, "photometric"));
			await Assert.That(depthMap.GetWidth()).IsEqualTo(CameraWidth);
			var normalMap = new NormalMap();
			normalMap.Read(MapPath(fixture, "normal_maps", imageIdx, "photometric"));
			await Assert.That(normalMap.GetDepth()).IsEqualTo(3);
		}

		await Assert.That(File.Exists(MapPath(fixture, "depth_maps", 2, "photometric"))).IsFalse();
		await Assert.That(reports.Select(r => r.Done).ToArray()).IsEquivalentTo([1, 2], CollectionOrdering.Matching);
		await Assert.That(reports[0].Total).IsEqualTo(2);
		await Assert.That(reports[0].Message).IsEqualTo(fixture.Names[0]);

		// A problem whose outputs exist is skipped: a marker map survives a second run.
		var marker = new Mat<float>(CameraWidth, CameraHeight, 1);
		marker.Fill(42.0f);
		marker.Write(MapPath(fixture, "depth_maps", 0, "photometric"));
		new PatchMatchController(RunOptions(false), fixture.TempDir, "COLMAP", "", fixture.Bitmaps).Run();
		var reread = new DepthMap();
		reread.Read(MapPath(fixture, "depth_maps", 0, "photometric"));
		await Assert.That(reread.Get(3, 4)).IsEqualTo(42.0f);
	}

	[Test]
	public async Task Run_GeometricComputesPhotometricFirst()
	{
		var fixture = new Fixture();
		Directory.CreateDirectory(Path.Combine(fixture.TempDir, "stereo", "consistency_graphs"));
		fixture.WriteConfig(fixture.Names[0], "__all__", fixture.Names[1], "__all__", fixture.Names[2], "__all__", fixture.Names[3], "__all__");
		var reports = new List<ColmapSharp.Controllers.ControllerProgress>();
		var controller = new PatchMatchController(RunOptions(true), fixture.TempDir, "COLMAP", "", fixture.Bitmaps);
		controller.Run(progress: new SynchronousProgress<ColmapSharp.Controllers.ControllerProgress>(reports));

		for (int imageIdx = 0; imageIdx < 4; ++imageIdx)
		{
			await Assert.That(File.Exists(MapPath(fixture, "depth_maps", imageIdx, "photometric"))).IsTrue();
			await Assert.That(File.Exists(MapPath(fixture, "normal_maps", imageIdx, "photometric"))).IsTrue();
			await Assert.That(File.Exists(MapPath(fixture, "depth_maps", imageIdx, "geometric"))).IsTrue();
			await Assert.That(File.Exists(MapPath(fixture, "normal_maps", imageIdx, "geometric"))).IsTrue();
			await Assert.That(File.Exists(MapPath(fixture, "consistency_graphs", imageIdx, "geometric"))).IsTrue();
		}

		// As in COLMAP, the photometric pass keeps write_consistency_graph, so it writes
		// graphs too, but they are empty: filtering is off there.
		var photometricGraph = new ConsistencyGraph();
		photometricGraph.Read(MapPath(fixture, "consistency_graphs", 0, "photometric"));
		await Assert.That(photometricGraph.GetImageIdxs(4, 5).Length).IsEqualTo(0);
		await Assert.That(reports.Count).IsEqualTo(8);
		await Assert.That(reports[0].Stage).IsEqualTo("PatchMatch photometric");
		await Assert.That(reports[7].Stage).IsEqualTo("PatchMatch geometric");
	}

	[Test]
	public async Task Run_WithModelInMemory()
	{
		var fixture = new Fixture();
		fixture.WriteConfig(fixture.Names[2], "__all__");
		var model = new Model();
		model.ReadFromCOLMAP(fixture.TempDir);

		// The sparse folder is not read.
		Directory.Delete(Path.Combine(fixture.TempDir, "sparse"), recursive: true);
		new PatchMatchController(RunOptions(false), model, fixture.TempDir, fixture.Bitmaps).Run();
		await Assert.That(File.Exists(MapPath(fixture, "depth_maps", 2, "photometric"))).IsTrue();
		await Assert.That(model.Images[0].GetBitmap().IsEmpty).IsTrue();
	}

	[Test]
	public async Task Run_StopsWithoutErrorWhenCancelled()
	{
		var fixture = new Fixture();
		fixture.WriteConfig(fixture.Names[0], "__all__");
		using var cancellation = new CancellationTokenSource();
		cancellation.Cancel();
		new PatchMatchController(RunOptions(false), fixture.TempDir, "COLMAP", "", fixture.Bitmaps).Run(cancellation.Token);
		await Assert.That(File.Exists(MapPath(fixture, "depth_maps", 0, "photometric"))).IsFalse();
	}

	[Test]
	public async Task Run_CancelledMidProblemWritesNothingForIt()
	{
		// Cancelling while a problem's PatchMatch runs aborts that problem (divergence 122):
		// no exception escapes, it writes no maps, and the finished problem's maps stay.
		var fixture = new Fixture();
		fixture.WriteConfig(fixture.Names[0], "__all__", fixture.Names[1], "__all__");
		using var cancellation = new CancellationTokenSource();
		var controller = new PatchMatchController(RunOptions(false), fixture.TempDir, "COLMAP", "", fixture.Bitmaps);
		int started = 0;
		controller.ProblemRunning = problemIdx =>
		{
			started++;
			if (problemIdx == 1)
			{
				cancellation.Cancel();
			}
		};

		controller.Run(cancellation.Token);

		await Assert.That(started).IsEqualTo(2);
		await Assert.That(File.Exists(MapPath(fixture, "depth_maps", 0, "photometric"))).IsTrue();
		await Assert.That(File.Exists(MapPath(fixture, "normal_maps", 0, "photometric"))).IsTrue();
		await Assert.That(File.Exists(MapPath(fixture, "depth_maps", 1, "photometric"))).IsFalse();
		await Assert.That(File.Exists(MapPath(fixture, "normal_maps", 1, "photometric"))).IsFalse();
	}

	[Test]
	[NotInParallel(nameof(Log))]
	public async Task Warnings_GoToTheLog()
	{
		IReadOnlyList<(LogLevel Level, string Message)> messages;
		using (var capture = new LogCapture())
		{
			var fixture = new Fixture();
			fixture.WriteConfig(fixture.Names[0], "__auto__, 3", fixture.Names[1], fixture.Names[0] + ", " + fixture.Names[2]);
			fixture.Bitmaps.Remove(Path.Combine(fixture.TempDir, "images", fixture.Names[2]));
			var options = PhotometricOptions();
			options.MinTriangulationAngle = 179;
			options.AllowMissingFiles = true;
			PatchMatchController controller = fixture.Open(options);
			controller.SetUpProblem(options, 0);
			messages = capture.Messages;
		}

		await Assert.That(messages).Contains(m => m.Level == LogLevel.Warning && m.Message.StartsWith("Ignoring reference image ", StringComparison.Ordinal));
		await Assert.That(messages).Contains(m => m.Level == LogLevel.Warning && m.Message.StartsWith("Skipping source image 2: ", StringComparison.Ordinal));
	}

	private sealed class SynchronousProgress<T>(List<T> reports) : IProgress<T>
	{
		public void Report(T value) => reports.Add(value);
	}

	[Test]
	public async Task ParseStoll_MatchesStdStoll()
	{
		await Assert.That(PatchMatchController.ParseStoll(" 20")).IsEqualTo(20L);
		await Assert.That(PatchMatchController.ParseStoll("+7x")).IsEqualTo(7L);
		await Assert.That(PatchMatchController.ParseStoll("-1")).IsEqualTo(-1L);
		await Assert.That(() => PatchMatchController.ParseStoll("abc")).Throws<FormatException>();
		await Assert.That(() => PatchMatchController.ParseStoll("99999999999999999999")).Throws<OverflowException>();
	}
}
