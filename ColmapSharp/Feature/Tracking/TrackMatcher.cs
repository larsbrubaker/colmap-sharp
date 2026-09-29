// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// TrackMatcher: turns KLT video tracks (SequenceTracker) into keyframes, keypoints and feature
// matches that COLMAP's two-view verification and mapper can use (docs/QUALITY_PLAN.md stage
// 2b). Not a COLMAP port; COLMAP has no video tracker. The keyframe rule is the usual one of
// keyframe-based visual odometry (e.g. Klein and Murray, PTAM, ISMAR 2007; Mur-Artal et al.,
// ORB-SLAM, T-RO 2015), written from the idea only: a new keyframe once the tracked points
// have moved far enough since the last one (parallax), or too many of its tracks have ended,
// or too many frames have passed.
//
// Each track seen in two or more keyframes becomes one keypoint in every keyframe that sees
// it, and every pair of keyframes up to MatchNeighbors apart gets one match per shared track.
// A track is at most one keypoint per keyframe, so a pair never matches a point twice. The
// descriptor of a track keypoint is VLFeat's SIFT descriptor (VlSiftFilter) at one fixed scale
// in the first full-resolution octave, oriented by its dominant gradient, so descriptor
// matching (sequential, exhaustive, loop closure) can use these points as well.
//
// The database side (appending the keypoints and merging the matches before verification) is
// Controllers/VideoTrackMatching.cs. Everything here depends only on the tracks, the frames and
// the options.

using ColmapSharp.Feature.VLFeat;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Feature.Tracking;

/// <summary>Options for <see cref="TrackMatcher"/> and the video-tracking intake.</summary>
public sealed class VideoTrackingOptions
{
	/// <summary>How points are tracked through the video.</summary>
	public SequenceTrackerOptions Tracker { get; set; } = new();

	/// <summary>
	/// A new keyframe once the median displacement of the tracks shared with the last keyframe
	/// reaches this fraction of the larger image side.
	/// </summary>
	public double MinParallaxFraction { get; set; } = 0.03;

	/// <summary>A new keyframe at the latest this many frames after the last one.</summary>
	public int MaxKeyframeGap { get; set; } = 8;

	/// <summary>
	/// A new keyframe (the frame before, when that is not the last keyframe) once fewer than
	/// this fraction of the last keyframe's tracks are still alive.
	/// </summary>
	public double MinSharedFraction { get; set; } = 0.5;

	/// <summary>Match each keyframe with the next this many keyframes.</summary>
	public int MatchNeighbors { get; set; } = 3;

	/// <summary>Write a keyframe pair only when it shares at least this many tracks.</summary>
	public int MinPairMatches { get; set; } = 8;

	/// <summary>Write a keyframe pair only when its frames are at most this far apart (0: no cap).</summary>
	public int MaxPairFrameGap { get; set; }

	/// <summary>
	/// Leave a pair's matches alone when descriptor matching already verified at least this
	/// many inliers for it (0: always add the tracks).
	/// </summary>
	public int MaxSiftInliers { get; set; }

	/// <summary>Append the track keypoints' SIFT descriptors so descriptor matching sees them too.</summary>
	public bool DescribeTracks { get; set; } = true;

	/// <summary>
	/// Cut tracks into pieces of at most this many frames before matching (0: whole tracks), so
	/// KLT drift does not pile up along one 3D point.
	/// </summary>
	public int MaxTrackLength { get; set; }
}

/// <summary>A track's keypoint in one keyframe: the track's index and its position there.</summary>
public readonly record struct TrackKeypoint(int TrackIndex, double X, double Y);

/// <summary>The matches of one keyframe pair, as indices into each keyframe's <see cref="TrackKeypoint"/> list.</summary>
public sealed record KeyframePairMatches(int KeyframeA, int KeyframeB, IReadOnlyList<FeatureMatch> Matches);

/// <summary>The keyframes chosen from a video, their track keypoints and the matches between them.</summary>
public sealed record TrackKeyframeMatches(
	IReadOnlyList<int> Keyframes,
	IReadOnlyList<IReadOnlyList<TrackKeypoint>> Keypoints,
	IReadOnlyList<KeyframePairMatches> Pairs);

/// <summary>Keyframes, keypoints, matches and descriptors from video tracks.</summary>
public static class TrackMatcher
{
	/// <summary>
	/// Picks keyframes (ascending frame indices, always the first and last frame) from
	/// <paramref name="tracks"/> over <paramref name="numFrames"/> frames whose larger side is
	/// <paramref name="imageSize"/> pixels.
	/// </summary>
	public static List<int> SelectKeyframes(
		IReadOnlyList<FeatureTrack> tracks, int numFrames, int imageSize, VideoTrackingOptions options)
	{
		Check.That(numFrames > 0);
		Check.That(options.MaxKeyframeGap >= 1);
		List<int>[] byFrame = TracksByFrame(tracks, numFrames);
		double minParallax = options.MinParallaxFraction * imageSize;
		var keyframes = new List<int> { 0 };
		int key = 0;
		var displacements = new List<double>();
		for (int f = 1; f < numFrames; ++f)
		{
			displacements.Clear();
			foreach (int t in byFrame[key])
			{
				if (TryPosition(tracks[t], f, out double x, out double y))
				{
					TryPosition(tracks[t], key, out double kx, out double ky);
					double dx = x - kx, dy = y - ky;
					displacements.Add(Math.Sqrt((dx * dx) + (dy * dy)));
				}
			}

			int chosen = -1;
			if (f - key >= options.MaxKeyframeGap)
			{
				chosen = f;
			}
			else if (displacements.Count < options.MinSharedFraction * byFrame[key].Count || displacements.Count == 0)
			{
				// Tracks are dying: keep the last frame that still shared enough with the keyframe.
				chosen = f - 1 > key ? f - 1 : f;
			}
			else
			{
				displacements.Sort();
				if (displacements[displacements.Count / 2] >= minParallax)
				{
					chosen = f;
				}
			}

			if (chosen < 0)
			{
				continue;
			}

			keyframes.Add(chosen);
			key = chosen;
			if (chosen < f)
			{
				// Re-test f against the new keyframe.
				--f;
			}
		}

		if (keyframes[^1] != numFrames - 1)
		{
			keyframes.Add(numFrames - 1);
		}

		return keyframes;
	}

	/// <summary>
	/// The track keypoints of each keyframe (tracks seen in two or more keyframes, in track
	/// order) and the matches between keyframes up to MatchNeighbors apart that share at least
	/// MinPairMatches tracks.
	/// </summary>
	public static TrackKeyframeMatches BuildMatches(
		IReadOnlyList<FeatureTrack> tracks, IReadOnlyList<int> keyframes, VideoTrackingOptions options)
	{
		Check.That(options.MatchNeighbors >= 1);
		int n = keyframes.Count;
		var seenIn = new int[tracks.Count];
		for (int t = 0; t < tracks.Count; ++t)
		{
			foreach (int k in keyframes)
			{
				if (TryPosition(tracks[t], k, out _, out _))
				{
					seenIn[t]++;
				}
			}
		}

		var keypoints = new List<TrackKeypoint>[n];
		// Per keyframe, track index -> keypoint index.
		var local = new Dictionary<int, int>[n];
		for (int i = 0; i < n; ++i)
		{
			keypoints[i] = [];
			local[i] = [];
			for (int t = 0; t < tracks.Count; ++t)
			{
				if (seenIn[t] >= 2 && TryPosition(tracks[t], keyframes[i], out double x, out double y))
				{
					local[i][t] = keypoints[i].Count;
					keypoints[i].Add(new TrackKeypoint(t, x, y));
				}
			}
		}

		var pairs = new List<KeyframePairMatches>();
		for (int a = 0; a < n; ++a)
		{
			for (int b = a + 1; b < n && b - a <= options.MatchNeighbors; ++b)
			{
				var matches = new List<FeatureMatch>();
				foreach (TrackKeypoint kp in keypoints[a])
				{
					if (local[b].TryGetValue(kp.TrackIndex, out int idxB))
					{
						matches.Add(new FeatureMatch((uint)local[a][kp.TrackIndex], (uint)idxB));
					}
				}

				if (matches.Count >= options.MinPairMatches
					&& (options.MaxPairFrameGap <= 0 || keyframes[b] - keyframes[a] <= options.MaxPairFrameGap))
				{
					pairs.Add(new KeyframePairMatches(a, b, matches));
				}
			}
		}

		return new TrackKeyframeMatches(keyframes, keypoints, pairs);
	}

	/// <summary>
	/// SIFT keypoints and descriptors (UBC order, normalized as <paramref name="normalization"/>
	/// says, 8-bit, like COLMAP's CPU SIFT extractor) for <paramref name="points"/> in <paramref name="frame"/>, at the scale
	/// of the first full-resolution octave's level 1 (sigma about 2.5 px), each oriented by its
	/// dominant gradient. Positions are COLMAP's continuous image coordinates.
	/// </summary>
	public static (List<FeatureKeypoint> Keypoints, RowMajorMatrix<byte> Descriptors) ComputeDescriptors(
		Bitmap frame, IReadOnlyList<TrackKeypoint> points, SiftNormalization normalization = SiftNormalization.L1Root)
	{
		const int Levels = 3;
		const int Level = 1;
		float[] grey = ImagePyramid.ToGrey(frame);
		for (int i = 0; i < grey.Length; ++i)
		{
			grey[i] /= 255.0f;
		}

		var sift = new VlSiftFilter(frame.Width, frame.Height, 1, Levels, 0);
		sift.ProcessFirstOctave(grey);
		double sigma = sift.Sigma0 * Math.Pow(2.0, (double)Level / Levels);

		var keypoints = new List<FeatureKeypoint>(points.Count);
		var vlfeat = new RowMajorMatrix<byte>(points.Count, VlSiftFilter.DescriptorLength);
		Span<double> angles = stackalloc double[4];
		var desc = new float[VlSiftFilter.DescriptorLength];
		for (int i = 0; i < points.Count; ++i)
		{
			// VLFeat's pixel centers are at integers; COLMAP's at +0.5.
			double x = points[i].X - 0.5, y = points[i].Y - 0.5;
			var k = new VlSiftKeypoint
			{
				O = 0,
				IX = (int)(x + 0.5),
				IY = (int)(y + 0.5),
				IS = Level,
				X = (float)x,
				Y = (float)y,
				S = Level,
				Sigma = (float)sigma,
			};
			int numAngles = sift.CalcKeypointOrientations(angles, k);
			double angle = numAngles > 0 ? angles[0] : 0.0;
			Array.Clear(desc);
			sift.CalcKeypointDescriptor(desc, k, angle);
			keypoints.Add(new FeatureKeypoint((float)points[i].X, (float)points[i].Y, (float)sigma, (float)angle));
			if (IsZero(desc))
			{
				// VLFeat left it untouched (out of bounds): keep an all-zero descriptor, which
				// matches nothing well, rather than normalizing 0/0.
				continue;
			}

			switch (normalization)
			{
				case SiftNormalization.L2:
					FeatureUtils.L2NormalizeRow(desc);
					break;
				case SiftNormalization.L1Root:
					FeatureUtils.L1RootNormalizeRow(desc);
					break;
				default:
					throw new ArgumentOutOfRangeException(nameof(normalization));
			}

			FeatureUtils.ToUnsignedByte(desc, vlfeat.Row(i));
		}

		return (keypoints, SiftCpuFeatureExtractor.TransformVLFeatToUBCFeatureDescriptors(vlfeat));
	}

	/// <summary>
	/// <paramref name="tracks"/> cut into consecutive pieces of at most
	/// <paramref name="maxLength"/> frames (each a new track; ids count up in order).
	/// </summary>
	public static List<FeatureTrack> SplitTracks(IReadOnlyList<FeatureTrack> tracks, int maxLength)
	{
		Check.That(maxLength >= 1);
		var pieces = new List<FeatureTrack>();
		foreach (FeatureTrack track in tracks)
		{
			for (int start = 0; start < track.Observations.Count; start += maxLength)
			{
				var piece = new FeatureTrack(pieces.Count);
				int end = Math.Min(start + maxLength, track.Observations.Count);
				for (int i = start; i < end; ++i)
				{
					piece.Observations.Add(track.Observations[i]);
				}

				pieces.Add(piece);
			}
		}

		return pieces;
	}

	/// <summary>Where <paramref name="track"/> is in <paramref name="frame"/>, if it is observed there.</summary>
	public static bool TryPosition(FeatureTrack track, int frame, out double x, out double y)
	{
		int i = frame - track.Observations[0].Frame;
		if (i < 0 || i >= track.Observations.Count)
		{
			x = y = 0;
			return false;
		}

		TrackObservation o = track.Observations[i];
		x = o.X;
		y = o.Y;
		return true;
	}

	private static bool IsZero(float[] v)
	{
		foreach (float f in v)
		{
			if (f != 0)
			{
				return false;
			}
		}

		return true;
	}

	private static List<int>[] TracksByFrame(IReadOnlyList<FeatureTrack> tracks, int numFrames)
	{
		var byFrame = new List<int>[numFrames];
		for (int f = 0; f < numFrames; ++f)
		{
			byFrame[f] = [];
		}

		for (int t = 0; t < tracks.Count; ++t)
		{
			foreach (TrackObservation o in tracks[t].Observations)
			{
				Check.That(o.Frame < numFrames);
				byFrame[o.Frame].Add(t);
			}
		}

		return byFrame;
	}
}
