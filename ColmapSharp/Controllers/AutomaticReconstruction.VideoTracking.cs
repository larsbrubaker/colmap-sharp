// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AutomaticReconstructionController, video tracking: C#-only (divergence 143), not in COLMAP.
// With AutomaticReconstructionOptions.VideoTracking on and video data, the matching stage adds
// KLT tracks (Controllers/VideoTrackMatching.cs): keyframe track keypoints and descriptors
// before the descriptor matcher runs, and the tracks' keyframe matches, re-verified, after it.
// The rest of the controller is AutomaticReconstruction.cs.

using ColmapSharp.Feature;

namespace ColmapSharp.Controllers;

/// <content>Video tracking around the matching stage.</content>
public sealed partial class AutomaticReconstructionController
{
	private bool UsesVideoTracking =>
		options.VideoTracking
		&& options.Data == AutomaticReconstructionOptions.DataType.Video
		&& options.Feature == AutomaticReconstructionOptions.FeatureType.Sift;

	// Null when tracking is off. On a resumed run the tracks are found again (the tracker is
	// deterministic); AppendTrackFeatures sees they are already there and does not append them
	// twice, and MergeTrackMatches skips the pairs whose track matches are already merged.
	private VideoTrackFeatures? AddVideoTrackFeatures()
	{
		if (!UsesVideoTracking || database.NumImages() < 2)
		{
			return null;
		}

		// Reported under the matching stage, so hosts see only the controller's stages.
		return VideoTrackMatching.AddTrackFeatures(
			database, options.Images!, options.Masks, options.VideoTrackingOptions,
			optionManager.FeatureExtraction.Sift.Normalization, Under(FeatureMatching.MatchingStage), CancellationToken);
	}

	private void MergeVideoTrackMatches(VideoTrackFeatures? features)
	{
		if (features is null)
		{
			return;
		}

		VideoTrackMatching.MergeTrackMatches(
			database, features, optionManager.TwoViewGeometry, Under(FeatureMatching.MatchingStage), CancellationToken);
	}
}
