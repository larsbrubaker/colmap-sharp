// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BenchmarkEndToEndTests: C#-only (not a port). One tiny run of the whole benchmark path,
// SyntheticBenchmark.Run on a small TexturedSphere scene: photos in, AutomaticReconstruction
// with dense stages, the surface read back, and every metric computed. It pins that the path
// works and its numbers are sane, not how good the reconstruction is - that is the runner's job
// (ColmapSharp.Benchmarks, benchmarks/baseline.json). The bounds are loose on purpose: at this
// frame count the scene turns about 20 degrees per frame over its whole pendulum (the
// benchmark's "fast" motion) and only a few of its 16 frames register. Runs in about 15-20
// seconds in the test host.

using ColmapSharp.Controllers;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Mvs.Testing.Benchmark;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class BenchmarkEndToEndTests
{
	[Test]
	public async Task CSharpOnly_TinyTexturedSphereRunGivesSaneMetrics()
	{
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(SyntheticObjectKind.TexturedSphere, 16, 240, 180, seed: 1);
		var options = new SyntheticBenchmarkOptions { MapperSeed = 1 };
		options.Surface.SampleCount = 5000;

		SyntheticBenchmarkResult result = SyntheticBenchmark.Run(scene, options);
		BenchmarkMetrics m = result.Metrics;

		await Assert.That(m.NumFrames).IsEqualTo(16);
		await Assert.That(m.NumModels).IsGreaterThanOrEqualTo(1);
		await Assert.That(m.NumRegistered).IsGreaterThanOrEqualTo(3);
		await Assert.That(m.NumRegistered).IsLessThanOrEqualTo(16);
		await Assert.That(m.RegisteredFraction).IsEqualTo(m.NumRegistered / 16.0);

		// Enough cameras for an alignment, and its errors are finite and ordered.
		await Assert.That(m.Pose.Aligned).IsTrue();
		await Assert.That(m.Pose.NumPoses).IsEqualTo(m.NumRegistered);
		await Assert.That(m.Pose.MedianRotationDeg).IsLessThanOrEqualTo(m.Pose.MaxRotationDeg);
		await Assert.That(m.Pose.MaxRotationDeg).IsLessThanOrEqualTo(180.0);
		await Assert.That(m.Pose.MedianPositionPct).IsLessThanOrEqualTo(m.Pose.MaxPositionPct);

		// Surface and silhouette scores are in range whatever the surface was.
		await Assert.That(m.Surface.FScore).IsBetween(0.0, 1.0);
		await Assert.That(m.Surface.Precision).IsBetween(0.0, 1.0);
		await Assert.That(m.Surface.Recall).IsBetween(0.0, 1.0);
		await Assert.That(m.Surface.ExcludedFraction).IsBetween(0.0, 1.0);
		if (m.SurfaceSource == "mesh")
		{
			await Assert.That(m.SilhouetteIouMean).IsBetween(0.0, 1.0);
			await Assert.That(m.SilhouetteIouMin).IsLessThanOrEqualTo(m.SilhouetteIouMean);
		}

		// Every metric has a name, and the timer saw the controller's first three stages.
		await Assert.That(m.Values().Select(v => v.Name).Distinct().Count()).IsEqualTo(m.Values().Count);
		string[] stages = [.. result.StageSeconds.Select(s => s.Stage)];
		await Assert.That(stages.Take(3).SequenceEqual([
			FeatureExtraction.ExtractionStage,
			FeatureMatching.MatchingStage,
			AutomaticReconstructionController.SparseStage,
		])).IsTrue();

		// The stages read their times just after the total was taken, hence the slack.
		await Assert.That(result.StageSeconds.Sum(s => s.Seconds)).IsLessThanOrEqualTo(result.TotalSeconds + 0.5);
	}
}
