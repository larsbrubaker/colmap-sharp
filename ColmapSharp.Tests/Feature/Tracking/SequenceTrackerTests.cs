// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SequenceTrackerTests: C#-only tests (no COLMAP counterpart) of Feature/Tracking/SequenceTracker
// (docs/QUALITY_PLAN.md stage 2a) on the benchmark's TexturedSphere scene
// (Mvs/Testing/SyntheticObjectScene) with realistic motion: 480x360, 40 frames over 0.27 of the
// pendulum swing, tracked with the true foreground masks. The tracks are checked against the
// truth by triangulating each from the true poses and reprojecting it: a track that drifts or
// jumps cannot be explained by one 3D point.

using ColmapSharp.Feature.Tracking;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature.Tracking;

public class SequenceTrackerTests
{
	private static readonly Lazy<SyntheticObjectScene> Scene = new(() => SyntheticObjectScene.Generate(
		SyntheticObjectKind.TexturedSphere, numFrames: 40, width: 480, height: 360, seed: 7, motionDuration: 0.27));

	[Test]
	public async Task TexturedSphere_TracksSurviveAndAgreeWithTheTruth()
	{
		SyntheticObjectScene scene = Scene.Value;
		IReadOnlyList<FeatureTrack> tracks = SequenceTracker.Run(scene.Frames, scene.Masks);

		int[] lengths = [.. tracks.Select(t => t.Length).Order()];
		int median = lengths[lengths.Length / 2];
		double reprojection = MedianReprojectionError(scene, tracks, minLength: 5, out int checkedTracks);
		Console.WriteLine($"tracks={tracks.Count} medianLength={median} longest={lengths[^1]} checked={checkedTracks} medianReprojection={reprojection:F3}px");

		// Measured: 1117 tracks, median length 6, 619 of at least 5 frames, the longest 36,
		// median reprojection 0.50 px. The bars sit just outside those numbers so a regression
		// shows. Without illumination compensation (KltOptions.CompensateIllumination) the
		// reprojection was 1.38 px: the sphere turns under a lab-fixed light, so each surface
		// point's brightness changes between frames and plain KLT lags the motion by ~7%. With
		// ambient-only lighting plain KLT's per-pair error drops to 0.25 px, which is what the
		// compensated tracker nearly reaches (0.34 px).
		await Assert.That(median).IsGreaterThanOrEqualTo(3);
		await Assert.That(checkedTracks).IsGreaterThan(500);
		await Assert.That(lengths[^1]).IsGreaterThanOrEqualTo(30);
		await Assert.That(reprojection).IsLessThan(0.6);
	}

	[Test]
	public async Task SequentialAndParallel_GiveIdenticalTracks()
	{
		SyntheticObjectScene scene = Scene.Value;
		SequenceTrackerOptions Options(bool parallel) => new() { Klt = new KltOptions { Parallel = parallel } };

		IReadOnlyList<FeatureTrack> parallel = SequenceTracker.Run(scene.Frames.Take(12), scene.Masks.Take(12), Options(true));
		IReadOnlyList<FeatureTrack> sequential = SequenceTracker.Run(scene.Frames.Take(12), scene.Masks.Take(12), Options(false));

		await Assert.That(parallel.Count).IsEqualTo(sequential.Count);
		for (int i = 0; i < parallel.Count; ++i)
		{
			await Assert.That(parallel[i].Observations.SequenceEqual(sequential[i].Observations)).IsTrue();
		}
	}

	[Test]
	public async Task Run_HonoursCancellationAndReportsProgress()
	{
		SyntheticObjectScene scene = Scene.Value;
		var reported = new List<int>();
		using var cts = new CancellationTokenSource();
		var progress = new SynchronousProgress(n =>
		{
			reported.Add(n);
			if (n == 3)
			{
				cts.Cancel();
			}
		});

		await Assert.That(() => SequenceTracker.Run(scene.Frames, null, null, progress, cts.Token))
			.Throws<OperationCanceledException>();
		await Assert.That(reported.SequenceEqual([1, 2, 3])).IsTrue();
	}

	// Triangulates every track of at least minLength frames from the true poses (normalized
	// image rays through Camera.CamFromImg) and returns the median reprojection error in pixels
	// over all of their observations.
	private static double MedianReprojectionError(
		SyntheticObjectScene scene, IReadOnlyList<FeatureTrack> tracks, int minLength, out int checkedTracks)
	{
		Camera camera = scene.Camera;
		var errors = new List<double>();
		checkedTracks = 0;
		foreach (FeatureTrack track in tracks.Where(t => t.Length >= minLength))
		{
			var poses = new Matrix3x4d[track.Length];
			var rays = new Vector2d[track.Length];
			for (int i = 0; i < track.Length; ++i)
			{
				TrackObservation o = track.Observations[i];
				poses[i] = scene.CamFromWorld[o.Frame].ToMatrix();
				rays[i] = camera.CamFromImg(new Vector2d(o.X, o.Y))!.Value;
			}

			if (!Triangulation.TriangulateMultiViewPoint(poses, rays, out Vector3d point))
			{
				continue;
			}

			++checkedTracks;
			foreach (TrackObservation o in track.Observations)
			{
				Vector2d? projected = camera.ImgFromCam(scene.CamFromWorld[o.Frame] * point);
				errors.Add(projected is Vector2d p
					? Math.Sqrt((p.X - o.X) * (p.X - o.X) + (p.Y - o.Y) * (p.Y - o.Y))
					: double.PositiveInfinity);
			}
		}

		errors.Sort();
		return errors.Count == 0 ? double.PositiveInfinity : errors[errors.Count / 2];
	}

	// Progress<T> posts to the thread pool; the test needs the callback on the calling thread.
	private sealed class SynchronousProgress(Action<int> report) : IProgress<int>
	{
		public void Report(int value) => report(value);
	}
}
