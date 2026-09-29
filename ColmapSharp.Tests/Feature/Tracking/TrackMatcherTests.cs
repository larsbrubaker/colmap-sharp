// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// TrackMatcherTests: C#-only (TrackMatcher is not a COLMAP port; QUALITY_PLAN stage 2b).
// Keyframe selection and the track -> keypoint -> match conversion on hand-built tracks, and the
// fixed-scale track descriptors on a textured image.

using ColmapSharp.Feature;
using ColmapSharp.Feature.Tracking;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature.Tracking;

public class TrackMatcherTests
{
	// A track from frame `start` for `length` frames moving `step` px a frame in x.
	private static FeatureTrack Track(int id, int start, int length, double x0, double step)
	{
		var t = new FeatureTrack(id);
		for (int i = 0; i < length; ++i)
		{
			t.Observations.Add(new TrackObservation(start + i, x0 + (step * i), 50));
		}

		return t;
	}

	[Test]
	public async Task CSharpOnly_KeyframesFollowParallax()
	{
		// 10 tracks over 20 frames moving 2 px a frame; min parallax 0.03 * 200 = 6 px -> every
		// 3rd frame, and the gap cap never binds.
		var tracks = Enumerable.Range(0, 10).Select(i => Track(i, 0, 20, 10 * i, 2)).ToList();
		var options = new VideoTrackingOptions { MaxKeyframeGap = 8 };
		List<int> keys = TrackMatcher.SelectKeyframes(tracks, 20, 200, options);
		await Assert.That(keys.SequenceEqual([0, 3, 6, 9, 12, 15, 18, 19])).IsTrue();
	}

	[Test]
	public async Task CSharpOnly_KeyframeGapIsCapped()
	{
		// Still points: no parallax, so the cap decides.
		var tracks = Enumerable.Range(0, 10).Select(i => Track(i, 0, 20, 10 * i, 0)).ToList();
		var options = new VideoTrackingOptions { MaxKeyframeGap = 5 };
		List<int> keys = TrackMatcher.SelectKeyframes(tracks, 20, 200, options);
		await Assert.That(keys.SequenceEqual([0, 5, 10, 15, 19])).IsTrue();
	}

	[Test]
	public async Task CSharpOnly_KeyframeBeforeTracksDie()
	{
		// Tracks 0-9 live frames 0-4; tracks 10-19 start at frame 3. At frame 5 none of frame 0's
		// tracks is alive, so frame 4 (the last that still shares them) becomes a keyframe.
		var tracks = Enumerable.Range(0, 10).Select(i => Track(i, 0, 5, 10 * i, 0))
			.Concat(Enumerable.Range(10, 10).Select(i => Track(i, 3, 10, 10 * i, 0))).ToList();
		var options = new VideoTrackingOptions { MaxKeyframeGap = 20 };
		List<int> keys = TrackMatcher.SelectKeyframes(tracks, 13, 200, options);
		await Assert.That(keys[1]).IsEqualTo(4);
		await Assert.That(keys[^1]).IsEqualTo(12);
	}

	[Test]
	public async Task CSharpOnly_MatchesPairSharedTracksWithoutDuplicates()
	{
		// Track 0 spans keyframes 0-2, track 1 keyframes 1-3, track 2 only keyframe 0 (no
		// keypoint), track 3 keyframes 0 and 3.
		var tracks = new List<FeatureTrack>
		{
			Track(0, 0, 21, 0, 0),
			Track(1, 10, 21, 1, 0),
			Track(2, 0, 3, 2, 0),
			Track(3, 0, 31, 3, 0),
		};
		var options = new VideoTrackingOptions { MatchNeighbors = 2, MinPairMatches = 1 };
		TrackKeyframeMatches m = TrackMatcher.BuildMatches(tracks, [0, 10, 20, 30], options);

		await Assert.That(m.Keypoints[0].Select(k => k.TrackIndex).SequenceEqual([0, 3])).IsTrue();
		await Assert.That(m.Keypoints[1].Select(k => k.TrackIndex).SequenceEqual([0, 1, 3])).IsTrue();
		await Assert.That(m.Keypoints[3].Select(k => k.TrackIndex).SequenceEqual([1, 3])).IsTrue();

		// Pairs up to 2 keyframes apart: (0,1) (0,2) (1,2) (1,3) (2,3); (0,3) is 3 apart.
		await Assert.That(m.Pairs.Select(p => (p.KeyframeA, p.KeyframeB)).SequenceEqual([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])).IsTrue();
		foreach (KeyframePairMatches pair in m.Pairs)
		{
			await Assert.That(pair.Matches.Select(x => x.Point2DIdx1).Distinct().Count()).IsEqualTo(pair.Matches.Count);
			await Assert.That(pair.Matches.Select(x => x.Point2DIdx2).Distinct().Count()).IsEqualTo(pair.Matches.Count);
			foreach (FeatureMatch match in pair.Matches)
			{
				// Each match joins the same track in both keyframes.
				int trackA = m.Keypoints[pair.KeyframeA][(int)match.Point2DIdx1].TrackIndex;
				int trackB = m.Keypoints[pair.KeyframeB][(int)match.Point2DIdx2].TrackIndex;
				await Assert.That(trackA).IsEqualTo(trackB);
			}
		}

		// (0,1) shares tracks 0 and 3: keypoints 0,1 in keyframe 0 and 0,2 in keyframe 1.
		KeyframePairMatches first = m.Pairs[0];
		await Assert.That(first.Matches.Select(x => (x.Point2DIdx1, x.Point2DIdx2)).SequenceEqual([(0u, 0u), (1u, 2u)])).IsTrue();
	}

	// Pins "the SIFT matcher can match track keypoints": at the extractor's own octave-0
	// keypoints whose scale is near the track descriptors' fixed sigma, the track descriptor's
	// nearest extractor descriptor (largest dot product, as the SIFT matcher ranks them) is
	// that same keypoint's.
	[Test]
	[Arguments(SiftNormalization.L1Root)]
	[Arguments(SiftNormalization.L2)]
	public async Task CSharpOnly_TrackDescriptorsMatchTheExtractorsAtTheSameKeypoint(SiftNormalization normalization)
	{
		const int Size = 160;
		var rng = new Random(11);
		var noise = new double[Size * Size];
		for (int i = 0; i < noise.Length; ++i)
		{
			noise[i] = rng.NextDouble();
		}

		// Blurred noise: blobs a few pixels across, so octave 0 has keypoints near sigma 2.5.
		var bitmap = new Bitmap(Size, Size, asRgb: false);
		for (int y = 0; y < Size; ++y)
		{
			for (int x = 0; x < Size; ++x)
			{
				double sum = 0;
				int n = 0;
				for (int dy = -2; dy <= 2; ++dy)
				{
					for (int dx = -2; dx <= 2; ++dx)
					{
						int xx = Math.Clamp(x + dx, 0, Size - 1), yy = Math.Clamp(y + dy, 0, Size - 1);
						sum += noise[(yy * Size) + xx];
						n++;
					}
				}

				bitmap.RowMajorData[(y * Size) + x] = (byte)(255 * sum / n);
			}
		}

		var options = new FeatureExtractionOptions();
		options.Sift.FirstOctave = 0;
		options.Sift.Normalization = normalization;
		var extractor = new SiftCpuFeatureExtractor(options);
		var keypoints = new List<FeatureKeypoint>();
		var descriptors = new FeatureDescriptors();
		extractor.Extract(bitmap, keypoints, descriptors);

		double sigma = 1.6 * Math.Pow(2.0, 2.0 / 3.0);
		var candidates = Enumerable.Range(0, keypoints.Count)
			.Where(i => Math.Abs(keypoints[i].ComputeScale() - sigma) < 0.1 * sigma)
			.Where(i => keypoints[i].X > 20 && keypoints[i].Y > 20 && keypoints[i].X < Size - 20 && keypoints[i].Y < Size - 20)
			.ToList();
		await Assert.That(candidates.Count).IsGreaterThanOrEqualTo(5);

		(_, var track) = TrackMatcher.ComputeDescriptors(
			bitmap, [.. candidates.Select(i => new TrackKeypoint(i, keypoints[i].X, keypoints[i].Y))], normalization);
		int hits = 0;
		for (int c = 0; c < candidates.Count; ++c)
		{
			int best = -1;
			long bestDot = -1;
			for (int r = 0; r < descriptors.Data.Rows; ++r)
			{
				long dot = 0;
				for (int d = 0; d < 128; ++d)
				{
					dot += track[c, d] * descriptors.Data[r, d];
				}

				if (dot > bestDot)
				{
					bestDot = dot;
					best = r;
				}
			}

			FeatureKeypoint k = keypoints[candidates[c]];
			if (keypoints[best].X == k.X && keypoints[best].Y == k.Y)
			{
				hits++;
			}
		}

		await Assert.That(hits).IsGreaterThanOrEqualTo((int)Math.Ceiling(0.8 * candidates.Count));
	}

	[Test]
	public async Task CSharpOnly_TrackDescriptorsAreNormalizedSift()
	{
		var bitmap = new Bitmap(64, 64, asRgb: false);
		var rng = new Random(3);
		for (int i = 0; i < bitmap.RowMajorData.Length; ++i)
		{
			bitmap.RowMajorData[i] = (byte)rng.Next(256);
		}

		(List<FeatureKeypoint> keypoints, var descriptors) = TrackMatcher.ComputeDescriptors(
			bitmap, [new TrackKeypoint(0, 32.5, 32.5), new TrackKeypoint(1, 20.5, 40.5)]);
		await Assert.That(keypoints.Count).IsEqualTo(2);
		await Assert.That(descriptors.Rows).IsEqualTo(2);
		await Assert.That(descriptors.Cols).IsEqualTo(128);
		await Assert.That(keypoints[0].X).IsEqualTo(32.5f);
		for (int r = 0; r < 2; ++r)
		{
			await Assert.That(descriptors.Row(r).ToArray().Any(b => b != 0)).IsTrue();
		}
	}
}
