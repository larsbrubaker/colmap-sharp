// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VideoTrackFeatureTests: C#-only (divergence 143; QUALITY_PLAN stage 2b). The database side of
// the video-tracking intake on a tiny hand-made database (three 64x64 images, hand-built
// tracks): appending track keypoints and descriptors, re-runs, cancellation, and merging track
// matches into descriptor matches. Milliseconds; the end-to-end gain is VideoTrackMatchingTests.

using ColmapSharp.Controllers;
using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Feature.Tracking;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public class VideoTrackFeatureTests
{
	// SIFT feature counts of the three images; the middle one has none.
	private static readonly int[] SiftCounts = [3, 0, 2];

	private sealed record Fixture(InMemoryDatabase Database, uint[] ImageIds, Bitmap Frame, TrackKeyframeMatches Matches);

	private static Fixture Build()
	{
		var frame = new Bitmap(64, 64, asRgb: false);
		var rng = new Random(5);
		for (int i = 0; i < frame.RowMajorData.Length; ++i)
		{
			frame.RowMajorData[i] = (byte)rng.Next(256);
		}

		var database = new InMemoryDatabase();
		uint cameraId = database.WriteCamera(Camera.CreateFromModelName(1, "SIMPLE_PINHOLE", 60, 64, 64));
		var ids = new uint[3];
		for (int i = 0; i < 3; ++i)
		{
			var image = new Image { Name = $"f{i}.png" };
			image.SetCameraId(cameraId);
			ids[i] = database.WriteImage(image);
			var keypoints = Enumerable.Range(0, SiftCounts[i]).Select(k => new FeatureKeypoint(5 + k, 5 + k)).ToList();
			database.WriteKeypoints(ids[i], keypoints);
			var data = new RowMajorMatrix<byte>(SiftCounts[i], 128);
			for (int r = 0; r < data.Rows; ++r)
			{
				data.Row(r).Fill((byte)(r + 1));
			}

			database.WriteDescriptors(ids[i], SiftCounts[i] == 0 ? new FeatureDescriptors() : new FeatureDescriptors(FeatureExtractorType.Sift, data));
		}

		// Ten tracks through all three frames, each keyframe a frame.
		var tracks = new List<FeatureTrack>();
		for (int t = 0; t < 10; ++t)
		{
			var track = new FeatureTrack(t);
			for (int f = 0; f < 3; ++f)
			{
				track.Observations.Add(new TrackObservation(f, 20.5 + (2 * t) + f, 30.5 + t));
			}

			tracks.Add(track);
		}

		var options = new VideoTrackingOptions { MinPairMatches = 1 };
		return new Fixture(database, ids, frame, TrackMatcher.BuildMatches(tracks, [0, 1, 2], options));
	}

	private static VideoTrackFeatures Append(Fixture f, CancellationToken token = default, Func<uint, Bitmap>? read = null) =>
		VideoTrackMatching.AppendTrackFeatures(f.Database, f.ImageIds, f.Matches, read ?? (_ => f.Frame), new VideoTrackingOptions(), cancellationToken: token);

	[Test]
	public async Task CSharpOnly_AppendKeepsKeypointsAndDescriptorsInStep()
	{
		Fixture f = Build();
		VideoTrackFeatures features = Append(f);

		await Assert.That(features.KeypointOffsets.SequenceEqual(SiftCounts)).IsTrue();
		for (int i = 0; i < 3; ++i)
		{
			uint id = f.ImageIds[i];
			await Assert.That(f.Database.NumKeypointsForImage(id)).IsEqualTo(SiftCounts[i] + 10);
			await Assert.That(f.Database.NumDescriptorsForImage(id)).IsEqualTo(SiftCounts[i] + 10);
			await Assert.That(f.Database.ReadDescriptors(id).Data.Cols).IsEqualTo(128);
			List<FeatureKeypoint> keypoints = f.Database.ReadKeypoints(id);
			await Assert.That(keypoints[SiftCounts[i]].X).IsEqualTo((float)f.Matches.Keypoints[i][0].X);

			// The SIFT rows are untouched.
			RowMajorMatrix<byte> data = f.Database.ReadDescriptors(id).Data;
			for (int r = 0; r < SiftCounts[i]; ++r)
			{
				await Assert.That(data[r, 0]).IsEqualTo((byte)(r + 1));
			}
		}
	}

	[Test]
	public async Task CSharpOnly_ReRunDoesNotAppendTwice()
	{
		Fixture f = Build();
		Append(f);
		VideoTrackFeatures again = Append(f);

		await Assert.That(again.KeypointOffsets.SequenceEqual(SiftCounts)).IsTrue();
		for (int i = 0; i < 3; ++i)
		{
			await Assert.That(f.Database.NumKeypointsForImage(f.ImageIds[i])).IsEqualTo(SiftCounts[i] + 10);
			await Assert.That(f.Database.NumDescriptorsForImage(f.ImageIds[i])).IsEqualTo(SiftCounts[i] + 10);
		}
	}

	[Test]
	public async Task CSharpOnly_CancelLeavesTheDatabaseUntouched()
	{
		Fixture f = Build();
		using var cts = new CancellationTokenSource();
		Bitmap ReadThenCancel(uint id)
		{
			cts.Cancel();
			return f.Frame;
		}

		await Assert.That(() => Append(f, cts.Token, ReadThenCancel)).Throws<OperationCanceledException>();
		for (int i = 0; i < 3; ++i)
		{
			await Assert.That(f.Database.NumKeypointsForImage(f.ImageIds[i])).IsEqualTo(SiftCounts[i]);
			await Assert.That(f.Database.NumDescriptorsForImage(f.ImageIds[i])).IsEqualTo(SiftCounts[i]);
		}
	}

	[Test]
	public async Task CSharpOnly_MergeAddsTrackMatchesWithoutClashes()
	{
		Fixture f = Build();
		VideoTrackFeatures features = Append(f);
		uint a = f.ImageIds[0], c = f.ImageIds[2];
		int offA = features.KeypointOffsets[0], offC = features.KeypointOffsets[2];

		// Descriptor matches for (0, 2): SIFT 0 <-> SIFT 0, and SIFT 1 <-> track 0's keypoint
		// in image 2, which the track match (track 0 in image 0 <-> track 0 in image 2) clashes with.
		f.Database.WriteMatches(a, c, [new FeatureMatch(0, 0), new FeatureMatch(1, (uint)offC)]);
		f.Database.WriteTwoViewGeometry(a, c, new TwoViewGeometry());

		int verified = VideoTrackMatching.MergeTrackMatches(f.Database, features, new TwoViewGeometryOptions());
		await Assert.That(verified).IsEqualTo(3);

		List<FeatureMatch> merged = f.Database.ReadMatches(a, c);
		await Assert.That(merged.Count).IsEqualTo(2 + 9);
		await Assert.That(merged.Take(2).Select(m => (m.Point2DIdx1, m.Point2DIdx2)).SequenceEqual([(0u, 0u), (1u, (uint)offC)])).IsTrue();
		await Assert.That(merged.Any(m => m.Point2DIdx1 == (uint)offA)).IsFalse();
		await Assert.That(merged.Select(m => m.Point2DIdx1).Distinct().Count()).IsEqualTo(merged.Count);
		await Assert.That(merged.Select(m => m.Point2DIdx2).Distinct().Count()).IsEqualTo(merged.Count);
		await Assert.That(f.Database.ExistsTwoViewGeometry(a, c)).IsTrue();

		// Merging again finds nothing new.
		await Assert.That(VideoTrackMatching.MergeTrackMatches(f.Database, features, new TwoViewGeometryOptions())).IsEqualTo(0);
		await Assert.That(f.Database.ReadMatches(a, c).Count).IsEqualTo(11);
	}
}
