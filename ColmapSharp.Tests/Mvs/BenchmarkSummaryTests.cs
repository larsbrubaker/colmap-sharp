// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BenchmarkSummaryTests: C#-only tests (not ports; COLMAP has no reconstruction benchmark).
// They pin ColmapSharp/Mvs/Testing/Benchmark/BenchmarkSummary.cs and BenchmarkEvaluator's
// bookkeeping: failed runs are counted and make the worst null instead of improving the mean;
// the regression check follows each metric's direction and its kind's allowance (counts, scores
// in [0, 1], errors), flags more failed runs and a worst that went null, and refuses reports made with different settings; and Evaluate maps images to
// frames by name (not id or order), skips unknown names and scores the largest model.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Mvs.Testing.Benchmark;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class BenchmarkSummaryTests
{
	private const string Config = """{"frames": 40, "width": 480, "mapper_seeds": [1, 2, 3]}""";

	[Test]
	public async Task CSharpOnly_FailedRunsAreCountedAndMakeTheWorstNull()
	{
		BenchmarkMetrics good = new() { NumFrames = 10, NumRegistered = 10, Surface = new SurfaceScores { FScore = 0.8 } };
		BenchmarkMetrics worse = new() { NumFrames = 10, NumRegistered = 6, Surface = new SurfaceScores { FScore = 0.4 } };

		// A run that did not align: its pose errors are NaN.
		BenchmarkMetrics failed = new() { NumFrames = 10, NumRegistered = 2, Surface = new SurfaceScores { FScore = 0 } };

		List<MetricSummary> summary = BenchmarkSummary.Summarize([good, worse, failed]);
		MetricSummary registered = summary.First(s => s.Name == "registered_fraction");
		MetricSummary f = summary.First(s => s.Name == "f_score");
		MetricSummary rotation = summary.First(s => s.Name == "rotation_error_median_deg");

		await Assert.That(Math.Abs(registered.Mean - 0.6)).IsLessThan(1e-12);
		await Assert.That(registered.Worst).IsEqualTo(0.2);
		await Assert.That(registered.Failed).IsEqualTo(0);
		await Assert.That(f.Worst).IsEqualTo(0.0);
		await Assert.That(rotation.Failed).IsEqualTo(3);
		await Assert.That(double.IsNaN(rotation.Worst)).IsTrue();
		await Assert.That(rotation.HigherIsBetter).IsFalse();
	}

	[Test]
	public async Task CSharpOnly_RegressionsFollowEachMetricsDirection()
	{
		string baseline = Report(("f_score", 0.5, 0.4, 0), ("accuracy_pct", 2.0, 3.0, 0));

		// Better in both directions: no regression.
		await Assert.That(BenchmarkSummary.Regressions(baseline, Report(("f_score", 0.6, 0.5, 0), ("accuracy_pct", 1.0, 2.0, 0)), 0.05).Count).IsEqualTo(0);

		// Within tolerance: max(0.01, 0.05 * 2.0) = 0.1 for accuracy, an absolute 0.02 for F.
		await Assert.That(BenchmarkSummary.Regressions(baseline, Report(("f_score", 0.49, 0.4, 0), ("accuracy_pct", 2.09, 3.0, 0)), 0.05).Count).IsEqualTo(0);

		// Worse in each direction.
		List<string> worse = BenchmarkSummary.Regressions(baseline, Report(("f_score", 0.47, 0.4, 0), ("accuracy_pct", 2.2, 3.0, 0)), 0.05);
		await Assert.That(worse.Count).IsEqualTo(2);
		await Assert.That(worse.Any(r => r.Contains("f_score"))).IsTrue();
		await Assert.That(worse.Any(r => r.Contains("accuracy_pct"))).IsTrue();
	}

	// Drops measured on real benchmark runs that the old max(t, t x |baseline|) allowance let
	// through: each must be flagged.
	[Test]
	[Arguments("f_score", 0.463, 0.414)]
	[Arguments("registered_fraction", 0.85, 0.80)]
	[Arguments("rotation_error_median_deg", 0.377, 0.427)]
	public async Task CSharpOnly_MeasuredMissesAreRegressions(string metric, double before, double after)
	{
		List<string> regressions = BenchmarkSummary.Regressions(Report((metric, before, before, 0)), Report((metric, after, after, 0)), 0.05);
		await Assert.That(regressions.Count).IsEqualTo(1);
		await Assert.That(regressions[0]).Contains(metric);
	}

	[Test]
	public async Task CSharpOnly_EachMetricKindHasItsOwnAllowance()
	{
		// One frame lost in one of the three seeds' runs of 40 frames is flagged; an unchanged
		// fraction is not.
		string registered = Report(("registered_fraction", 0.85, 0.8, 0));
		await Assert.That(BenchmarkSummary.Regressions(registered, Report(("registered_fraction", 0.85 - (1.0 / 120), 0.8, 0)), 0.05).Count).IsEqualTo(1);
		await Assert.That(BenchmarkSummary.Regressions(registered, Report(("registered_fraction", 0.85, 0.8, 0)), 0.05).Count).IsEqualTo(0);

		// So is one more model in one run.
		string models = Report(("num_models", 1.0, 1.0, 0));
		await Assert.That(BenchmarkSummary.Regressions(models, Report(("num_models", 4.0 / 3, 2.0, 0)), 0.05).Count).IsEqualTo(1);

		// IoU: an absolute 0.02.
		string iou = Report(("silhouette_iou_mean", 0.9, 0.85, 0));
		await Assert.That(BenchmarkSummary.Regressions(iou, Report(("silhouette_iou_mean", 0.885, 0.85, 0)), 0.05).Count).IsEqualTo(0);
		await Assert.That(BenchmarkSummary.Regressions(iou, Report(("silhouette_iou_mean", 0.875, 0.85, 0)), 0.05).Count).IsEqualTo(1);

		// An error metric at 0 has the absolute floor, not a zero allowance.
		string zero = Report(("position_error_median_pct", 0.0, 0.0, 0));
		await Assert.That(BenchmarkSummary.Regressions(zero, Report(("position_error_median_pct", 0.005, 0.005, 0)), 0.05).Count).IsEqualTo(0);
		await Assert.That(BenchmarkSummary.Regressions(zero, Report(("position_error_median_pct", 0.02, 0.02, 0)), 0.05).Count).IsEqualTo(1);
	}

	[Test]
	public async Task CSharpOnly_MoreFailedRunsOrANullWorstIsARegression()
	{
		string baseline = Report(("rotation_error_median_deg", 3.0, 4.0, 0));

		// A seed stopped aligning: the mean over the rest improved, but that is a regression.
		List<string> failed = BenchmarkSummary.Regressions(baseline, Report(("rotation_error_median_deg", 2.0, double.NaN, 1)), 0.05);
		await Assert.That(failed.Any(r => r.Contains("failed runs 0 -> 1"))).IsTrue();
		await Assert.That(failed.Any(r => r.Contains("worst 4 -> null"))).IsTrue();

		// No run computes it any more.
		List<string> none = BenchmarkSummary.Regressions(baseline, Report(("rotation_error_median_deg", double.NaN, double.NaN, 3)), 0.05);
		await Assert.That(none.Any(r => r.Contains("mean 3 -> null"))).IsTrue();
	}

	[Test]
	public async Task CSharpOnly_ReportsWithDifferentSettingsAreNotCompared()
	{
		string baseline = Report(("f_score", 0.5, 0.4, 0));
		string other = Report(("f_score", 0.5, 0.4, 0)).Replace("\"width\": 480", "\"width\": 240");
		await Assert.That(() => BenchmarkSummary.Regressions(baseline, other, 0.05)).Throws<InvalidOperationException>();
	}

	[Test]
	public async Task CSharpOnly_EvaluateMapsFramesByNameAndScoresTheLargestModel()
	{
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(SyntheticObjectKind.TexturedSphere, 5, 64, 48, seed: 2);
		string[] names = [.. Enumerable.Range(0, 5).Select(SyntheticBenchmark.FrameName)];

		// A one-image model first, then the largest: image ids out of frame order, and an image
		// whose name is not a frame (it must be ignored, not scored against some frame).
		Reconstruction small = Model(scene, [(1, names[1], scene.CamFromWorld[1])]);
		Reconstruction large = Model(scene,
		[
			(1, names[3], scene.CamFromWorld[3]),
			(2, names[0], scene.CamFromWorld[0]),
			(3, "unrelated.png", new Rigid3d(Quaterniond.Identity, new Vector3d(5, 5, 5))),
			(4, names[2], scene.CamFromWorld[2]),
		]);

		BenchmarkMetrics m = BenchmarkEvaluator.Evaluate(scene, names, [small, large], BenchmarkMesh.FromScene(scene), null);
		await Assert.That(BenchmarkEvaluator.SelectModel([small, large])).IsEqualTo(1);
		await Assert.That(m.NumModels).IsEqualTo(2);
		await Assert.That(m.NumRegistered).IsEqualTo(3);
		await Assert.That(m.Pose.Aligned).IsTrue();
		await Assert.That(m.Pose.MaxRotationDeg).IsLessThan(1e-6);
		await Assert.That(m.Pose.MaxPositionPct).IsLessThan(1e-6);
		await Assert.That(Math.Abs(m.FocalRatio - 1)).IsLessThan(1e-12);
		await Assert.That(m.SurfaceSource).IsEqualTo("mesh");
		await Assert.That(m.Surface.FScore).IsEqualTo(1.0);
		await Assert.That(m.SilhouetteIouMin).IsGreaterThan(0.95);
	}

	// A reconstruction with the scene's camera and the given (id, name, pose) images registered.
	private static Reconstruction Model(SyntheticObjectScene scene, (uint Id, string Name, Rigid3d Pose)[] images)
	{
		var reconstruction = new Reconstruction();
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Pinhole, 1, scene.Camera.Width, scene.Camera.Height);
		camera.Params = [.. scene.Camera.Params];
		reconstruction.AddCameraWithTrivialRig(camera);
		foreach ((uint id, string name, Rigid3d pose) in images)
		{
			var image = new ColmapSharp.Scene.Image { ImageId = id, Name = name };
			image.SetCameraId(1);
			reconstruction.AddImageWithTrivialFrame(image, pose);
		}

		return reconstruction;
	}

	// A one-case report with the given (metric, mean, worst, failed) summaries; NaN is null.
	private static string Report(params (string Name, double Mean, double Worst, int Failed)[] metrics)
	{
		static string Number(double v) => double.IsFinite(v) ? v.ToString("R", System.Globalization.CultureInfo.InvariantCulture) : "null";
		string summary = string.Join(", ", metrics.Select(m =>
			$"\"{m.Name}\": {{\"mean\": {Number(m.Mean)}, \"worst\": {Number(m.Worst)}, \"failed\": {m.Failed}}}"));
		return "{\"config\": " + Config + ", \"cases\": [{\"case\": \"realistic\", \"scene\": \"TexturedSphere\", \"scene_seed\": 1, \"summary\": {" + summary + "}}]}";
	}
}
