// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VideoTrackMatchingTests: C#-only (divergence 143; QUALITY_PLAN stage 2b). The whole video
// path with and without KLT tracks on the benchmark's TexturedSphere (40 frames, 480x360,
// realistic motion as in ColmapSharp.Benchmarks, true masks, self-calibrating, mapper seed 1,
// sparse only). Measured when written (seeds 1 and 2): 34/40 registered without tracks, 40/40
// with; on the 34 frames both place, median rotation error 1.16 -> 0.83 and 1.47 -> 0.83
// degrees. (With known intrinsics the tracks make the common frames worse, 0.33 -> 0.61
// degrees: KLT drift, see divergence 143. That case is not pinned here because it fails.)
// The masks matter: without them the tracker also follows the still wall. About 25 s.

using ColmapSharp.Controllers;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Mvs.Testing.Benchmark;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public class VideoTrackMatchingTests
{
	private static SyntheticBenchmarkResult Run(SyntheticObjectScene scene, bool tracking)
	{
		var options = new SyntheticBenchmarkOptions
		{
			MapperSeed = 1,
			Dense = false,
			Data = AutomaticReconstructionOptions.DataType.Video,
			VideoTracking = tracking,
			UseTrueMasks = true,
		};
		return SyntheticBenchmark.Run(scene, options);
	}

	[Test]
	public async Task CSharpOnly_TracksRegisterMoreSphereFramesWithoutWorsePoses()
	{
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(SyntheticObjectKind.TexturedSphere, 40, 480, 360, 1, motionDuration: 0.27);
		SyntheticBenchmarkResult sift = Run(scene, tracking: false);
		SyntheticBenchmarkResult tracked = Run(scene, tracking: true);

		await Assert.That(tracked.Metrics.NumRegistered).IsGreaterThan(sift.Metrics.NumRegistered);
		await Assert.That(tracked.Metrics.NumRegistered).IsGreaterThanOrEqualTo(38);
		await Assert.That(tracked.Metrics.NumModels).IsEqualTo(1);

		// On the frames both runs place, the tracked poses are no worse.
		List<int> common = [.. sift.RegisteredPoses.Keys.Where(tracked.RegisteredPoses.ContainsKey).Order()];
		await Assert.That(common.Count).IsEqualTo(sift.Metrics.NumRegistered);
		PoseScores siftPose = PoseMetrics.Compute([.. common.Select(k => sift.RegisteredPoses[k])], [.. common.Select(k => scene.CamFromWorld[k])], 1);
		PoseScores trackedPose = PoseMetrics.Compute([.. common.Select(k => tracked.RegisteredPoses[k])], [.. common.Select(k => scene.CamFromWorld[k])], 1);
		await Assert.That(trackedPose.Aligned).IsTrue();
		await Assert.That(trackedPose.MedianRotationDeg).IsLessThanOrEqualTo(siftPose.MedianRotationDeg);
	}
}
