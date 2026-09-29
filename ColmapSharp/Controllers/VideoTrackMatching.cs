// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VideoTrackMatching: the database side of the video-tracking intake (docs/QUALITY_PLAN.md
// stage 2b). Not a COLMAP port. AutomaticReconstructionController calls it for video data with
// AutomaticReconstructionOptions.VideoTracking on, around its descriptor matching:
//
// - AddTrackFeatures (after extraction, before matching) runs SequenceTracker over the frames in
//   name order (the order ImageReader gives them ids), picks keyframes and appends each
//   keyframe's track keypoints and their SIFT descriptors (Feature/Tracking/TrackMatcher.cs)
//   after that image's SIFT features, so the descriptor matchers see them too.
// - MergeTrackMatches (after matching) adds each keyframe pair's track matches to whatever the
//   descriptor matcher found for the pair (a feature already matched in the pair keeps its
//   descriptor match), then re-runs two-view verification on the union. Tracks add matches;
//   they never replace descriptor matches.
//
// The pairs are verified one after another, each with a fresh PRNG for an unseeded run (the
// matchers' rule, FeatureMatcherController.WithFreshPrng), so the result does not depend on
// thread count.

using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Feature.Tracking;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>What <see cref="VideoTrackMatching.AddTrackFeatures"/> wrote, for <see cref="VideoTrackMatching.MergeTrackMatches"/>.</summary>
/// <param name="Matches">The keyframes, their track keypoints and the pair matches (keypoint indices local to the tracks).</param>
/// <param name="KeyframeImageIds">The database image id of each keyframe.</param>
/// <param name="KeypointOffsets">Per keyframe, the index of its first track keypoint in the image's keypoints.</param>
/// <param name="MaxSiftInliers">VideoTrackingOptions.MaxSiftInliers.</param>
public sealed record VideoTrackFeatures(
	TrackKeyframeMatches Matches, IReadOnlyList<uint> KeyframeImageIds, IReadOnlyList<int> KeypointOffsets, int MaxSiftInliers = 0);

/// <summary>Video tracks as keypoints and matches in a <see cref="Database"/>.</summary>
public static class VideoTrackMatching
{
	/// <summary>The progress stage of the tracking pass.</summary>
	public const string TrackingStage = "Video tracking";

	/// <summary>
	/// Tracks the database's images (in name order, read from <paramref name="images"/>, with
	/// masks named as ImageReader names them when <paramref name="masks"/> is given), then
	/// appends every keyframe's track keypoints and descriptors to its features
	/// (<see cref="AppendTrackFeatures"/>). <paramref name="normalization"/> is the SIFT
	/// extractor's, so the track descriptors compare with the image's own.
	/// </summary>
	public static VideoTrackFeatures AddTrackFeatures(
		Database database,
		IImageSource images,
		IImageSource? masks,
		VideoTrackingOptions options,
		SiftNormalization normalization = SiftNormalization.L1Root,
		IProgress<ControllerProgress>? progress = null,
		CancellationToken cancellationToken = default)
	{
		List<Image> ordered = [.. database.ReadAllImages().OrderBy(image => image.Name, StringComparer.Ordinal)];
		Check.That(ordered.Count > 0, "no images to track");
		var tracker = new SequenceTracker(options.Tracker);
		int width = 0, height = 0;
		for (int i = 0; i < ordered.Count; ++i)
		{
			cancellationToken.ThrowIfCancellationRequested();
			Bitmap frame = ReadFrame(images, ordered[i].Name);
			width = frame.Width;
			height = frame.Height;
			tracker.AddFrame(frame, masks is null ? null : ReadMask(masks, ordered[i].Name));
			progress?.Report(new ControllerProgress(TrackingStage, i + 1, ordered.Count, ordered[i].Name));
		}

		List<int> keyframes = TrackMatcher.SelectKeyframes(tracker.Tracks, ordered.Count, Math.Max(width, height), options);
		IReadOnlyList<FeatureTrack> tracks = options.MaxTrackLength > 0
			? TrackMatcher.SplitTracks(tracker.Tracks, options.MaxTrackLength)
			: tracker.Tracks;
		TrackKeyframeMatches matches = TrackMatcher.BuildMatches(tracks, keyframes, options);
		uint[] keyframeImageIds = [.. keyframes.Select(k => ordered[k].ImageId)];
		var names = ordered.ToDictionary(image => image.ImageId, image => image.Name);
		return AppendTrackFeatures(
			database, keyframeImageIds, matches, id => ReadFrame(images, names[id]), options, normalization, cancellationToken);
	}

	/// <summary>
	/// Appends each keyframe's track keypoints (<paramref name="matches"/>; keyframe k is image
	/// <paramref name="keyframeImageIds"/>[k]) and their descriptors after the image's own
	/// features. Everything is computed before anything is written, so a cancelled call leaves
	/// the database as it was. An image whose keypoints already end with its track keypoints (a
	/// re-run) is left alone and its offset points at them, so running twice appends once. Only
	/// the keyframes that gain keypoints are rewritten.
	/// </summary>
	public static VideoTrackFeatures AppendTrackFeatures(
		Database database,
		IReadOnlyList<uint> keyframeImageIds,
		TrackKeyframeMatches matches,
		Func<uint, Bitmap> readFrame,
		VideoTrackingOptions options,
		SiftNormalization normalization = SiftNormalization.L1Root,
		CancellationToken cancellationToken = default)
	{
		Check.Eq(keyframeImageIds.Count, matches.Keyframes.Count);
		var offsets = new int[keyframeImageIds.Count];
		var writes = new List<(uint ImageId, List<FeatureKeypoint> Keypoints, FeatureDescriptors Descriptors)>();
		for (int k = 0; k < keyframeImageIds.Count; ++k)
		{
			cancellationToken.ThrowIfCancellationRequested();
			uint imageId = keyframeImageIds[k];
			IReadOnlyList<TrackKeypoint> track = matches.Keypoints[k];
			List<FeatureKeypoint> keypoints = database.ReadKeypoints(imageId);
			if (track.Count == 0)
			{
				offsets[k] = keypoints.Count;
				continue;
			}

			if (EndsWithTrack(keypoints, track))
			{
				offsets[k] = keypoints.Count - track.Count;
				continue;
			}

			offsets[k] = keypoints.Count;
			(List<FeatureKeypoint> trackKeypoints, RowMajorMatrix<byte> trackDescriptors) =
				TrackMatcher.ComputeDescriptors(readFrame(imageId), track, normalization);
			if (!options.DescribeTracks)
			{
				// All-zero rows: every similarity with them is 0, so descriptor matching never
				// picks them, and the rows still line up with the keypoints.
				trackDescriptors = new RowMajorMatrix<byte>(trackDescriptors.Rows, trackDescriptors.Cols);
			}

			keypoints.AddRange(trackKeypoints);
			writes.Add((imageId, keypoints, Append(database.ReadDescriptors(imageId), trackDescriptors)));
		}

		cancellationToken.ThrowIfCancellationRequested();
		foreach ((uint imageId, List<FeatureKeypoint> keypoints, FeatureDescriptors descriptors) in writes)
		{
			database.UpdateKeypoints(imageId, keypoints);
			database.UpdateDescriptors(imageId, descriptors);
		}

		return new VideoTrackFeatures(matches, keyframeImageIds, offsets, options.MaxSiftInliers);
	}

	// Whether keypoints end with exactly the track keypoints' positions (as stored, in float).
	private static bool EndsWithTrack(List<FeatureKeypoint> keypoints, IReadOnlyList<TrackKeypoint> track)
	{
		int start = keypoints.Count - track.Count;
		if (start < 0)
		{
			return false;
		}

		for (int i = 0; i < track.Count; ++i)
		{
			if (keypoints[start + i].X != (float)track[i].X || keypoints[start + i].Y != (float)track[i].Y)
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>
	/// Adds the track matches of every keyframe pair to its descriptor matches and re-verifies
	/// the pair. Returns how many pairs were (re)verified.
	/// </summary>
	public static int MergeTrackMatches(
		Database database,
		VideoTrackFeatures features,
		TwoViewGeometryOptions geometryOptions,
		IProgress<ControllerProgress>? progress = null,
		CancellationToken cancellationToken = default)
	{
		int done = 0;
		IReadOnlyList<KeyframePairMatches> pairs = features.Matches.Pairs;
		for (int p = 0; p < pairs.Count; ++p)
		{
			cancellationToken.ThrowIfCancellationRequested();
			KeyframePairMatches pair = pairs[p];
			uint id1 = features.KeyframeImageIds[pair.KeyframeA];
			uint id2 = features.KeyframeImageIds[pair.KeyframeB];
			int offset1 = features.KeypointOffsets[pair.KeyframeA];
			int offset2 = features.KeypointOffsets[pair.KeyframeB];

			if (features.MaxSiftInliers > 0 && database.ExistsTwoViewGeometry(id1, id2)
				&& database.ReadTwoViewGeometry(id1, id2).InlierMatches.Count >= features.MaxSiftInliers)
			{
				continue;
			}

			List<FeatureMatch> merged = database.ExistsMatches(id1, id2) ? database.ReadMatches(id1, id2) : [];
			var used1 = new HashSet<uint>(merged.Select(m => m.Point2DIdx1));
			var used2 = new HashSet<uint>(merged.Select(m => m.Point2DIdx2));
			int added = 0;
			foreach (FeatureMatch m in pair.Matches)
			{
				uint idx1 = (uint)(offset1 + (int)m.Point2DIdx1);
				uint idx2 = (uint)(offset2 + (int)m.Point2DIdx2);
				if (used1.Add(idx1) && used2.Add(idx2))
				{
					merged.Add(new FeatureMatch(idx1, idx2));
					added++;
				}
			}

			if (added == 0 && database.ExistsTwoViewGeometry(id1, id2))
			{
				continue;
			}

			if (database.ExistsMatches(id1, id2))
			{
				database.DeleteMatches(id1, id2);
			}

			if (database.ExistsTwoViewGeometry(id1, id2))
			{
				database.DeleteTwoViewGeometry(id1, id2);
			}

			database.WriteMatches(id1, id2, merged);
			Image image1 = database.ReadImage(id1);
			Image image2 = database.ReadImage(id2);
			Camera camera1 = database.ReadCamera(image1.CameraId);
			Camera camera2 = database.ReadCamera(image2.CameraId);
			Vector2d[] points1 = FeatureUtils.FeatureKeypointsToPointsVector(database.ReadKeypoints(id1));
			Vector2d[] points2 = FeatureUtils.FeatureKeypointsToPointsVector(database.ReadKeypoints(id2));
			TwoViewGeometry geometry = FeatureMatcherController.WithFreshPrng(geometryOptions, () =>
				TwoViewGeometryEstimation.EstimateTwoViewGeometry(camera1, points1, camera2, points2, merged, geometryOptions));
			database.WriteTwoViewGeometry(id1, id2, geometry);
			done++;
			progress?.Report(new ControllerProgress(TrackingStage, p + 1, pairs.Count, $"{image1.Name} - {image2.Name}"));
		}

		return done;
	}

	private static FeatureDescriptors Append(FeatureDescriptors existing, RowMajorMatrix<byte> rows)
	{
		RowMajorMatrix<byte> old = existing.Data;
		// An image without features may hold an empty matrix of any width; the track's width wins.
		Check.That(old.Rows == 0 || old.Cols == rows.Cols, "track descriptors need SIFT features");
		var data = new RowMajorMatrix<byte>(old.Rows + rows.Rows, rows.Cols);
		for (int r = 0; r < old.Rows; ++r)
		{
			old.Row(r).CopyTo(data.Row(r));
		}

		for (int r = 0; r < rows.Rows; ++r)
		{
			rows.Row(r).CopyTo(data.Row(old.Rows + r));
		}

		return new FeatureDescriptors(FeatureExtractorType.Sift, data);
	}

	private static Bitmap ReadFrame(IImageSource images, string name) =>
		images.Read(name) ?? throw new InvalidOperationException($"Could not read {name} for video tracking.");

	// ImageReader's mask naming: "<image name>.png", else the name with its extension replaced.
	// Masks are 8-bit grey, 255 = object; an RGB mask is reduced to grey.
	private static Bitmap? ReadMask(IImageSource masks, string imageName)
	{
		string name = imageName + ".png";
		if (!masks.Exists(name))
		{
			int lastDot = imageName.LastIndexOf('.');
			name = lastDot >= 0 ? imageName[..lastDot] + ".png" : name;
		}

		Bitmap? mask = masks.Exists(name) ? masks.Read(name) : null;
		return mask is null || mask.IsGrey ? mask : mask.CloneAsGrey();
	}
}
