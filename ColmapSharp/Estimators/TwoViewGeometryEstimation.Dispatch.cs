// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometryEstimation.Dispatch: the entry point of colmap/estimators/two_view_geometry.cc
// - EstimateTwoViewGeometry, which routes a pair to the forced-homography, one-sided focal,
// spherical, shared focal, calibrated or uncalibrated path by what is known about its
// cameras - and EstimateMultipleTwoViewGeometries (recursive multi-model estimation).
// TwoViewGeometryEstimation.cs holds the options, shared helpers and file notes.

using ColmapSharp.Feature;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

public static partial class TwoViewGeometryEstimation
{
	// COLMAP's LOG_FIRST_N(WARNING, 1): each of these warnings is logged once per process.
	private static int forceHUseNonPinholeWarned;
	private static int fisheyeWithoutFocalPriorWarned;

	// Whether a camera's intrinsics are known: it either carries a focal prior, or is
	// spherical and has no focal length to estimate in the first place.
	private static bool IsCameraCalibrated(Camera camera) => camera.IsSpherical || camera.HasPriorFocalLength;

	/// <summary>
	/// Estimate two-view geometry from calibrated or uncalibrated image pair, depending on
	/// whether a prior focal length is given or not. Port of colmap::EstimateTwoViewGeometry.
	/// </summary>
	/// <param name="camera1">Camera of first image.</param>
	/// <param name="points1">Feature points in first image.</param>
	/// <param name="camera2">Camera of second image.</param>
	/// <param name="points2">Feature points in second image.</param>
	/// <param name="matches">Feature matches between first and second image (not modified;
	/// C++ takes them by value).</param>
	/// <param name="options">Two-view geometry estimation options.</param>
	public static TwoViewGeometry EstimateTwoViewGeometry(
		Camera camera1,
		IReadOnlyList<Vector2d> points1,
		Camera camera2,
		IReadOnlyList<Vector2d> points2,
		IReadOnlyList<FeatureMatch> matches,
		TwoViewGeometryOptions options)
	{
		if (options.FilterStationaryMatches)
		{
			var filteredMatches = new List<FeatureMatch>(matches);
			FilterStationaryMatches(options.StationaryMatchesMaxError, points1, points2, filteredMatches);
			matches = filteredMatches;
		}

		if (options.MultipleModels)
		{
			TwoViewGeometryOptions multipleModelOptions = options.Clone();
			// Set to false to prevent recursive calls to this function.
			multipleModelOptions.MultipleModels = false;
			// Set to false to prevent redundant filtering of stationary matches.
			multipleModelOptions.FilterStationaryMatches = false;
			return EstimateMultipleTwoViewGeometries(camera1, points1, camera2, points2, matches, multipleModelOptions);
		}

		if (options.ForceHUse)
		{
			// In image coordinates, a homography relates two views of a plane only under a
			// pinhole projection. Fisheye and spherical models map the plane non-linearly, so
			// the estimated homography would be meaningless; such pairs are marked degenerate
			// (with a warning logged once, as in COLMAP).
			if (!camera1.IsPerspectivePinhole || !camera2.IsPerspectivePinhole)
			{
				if (Interlocked.Exchange(ref forceHUseNonPinholeWarned, 1) == 0)
				{
					Log.Warning(
						"Ignoring force_H_use for non-pinhole cameras, as a homography does not relate their images of a plane. "
						+ "Such pairs are marked as degenerate.");
				}

				return new TwoViewGeometry { Config = TwoViewGeometry.ConfigurationType.Degenerate };
			}

			return EstimateCalibratedHomography(camera1, points1, camera2, points2, matches, options);
		}

		if (IsCameraCalibrated(camera1) != IsCameraCalibrated(camera2)
			&& (IsCameraCalibrated(camera1) ? camera2 : camera1).IsPerspectivePinhole)
		{
			// Exactly one side is known, either from a focal prior or because it is spherical
			// and has no focal at all. Recover the other side's focal rather than discard the
			// known intrinsics. Only the uncalibrated side must be a pinhole projection; the
			// calibrated side enters as rays, so any model is admissible there.
			return EstimateOneSidedFocalTwoViewGeometry(camera1, points1, camera2, points2, matches, options);
		}

		if (camera1.IsSpherical || camera2.IsSpherical)
		{
			// No pinhole image plane, so the fundamental matrix is not meaningful. Mixed
			// spherical/uncalibrated pairs are caught above.
			return EstimateSphericalTwoViewGeometry(camera1, points1, camera2, points2, matches, options);
		}

		if (camera1.CameraId == camera2.CameraId && !camera1.HasPriorFocalLength && camera1.IsPerspectivePinhole)
		{
			// A single shared unknown focal. Multi-focal models (e.g. PINHOLE) are seeded
			// isotropically (fx = fy = f) and refined later by bundle adjustment; distortion
			// is absorbed by the epipolar fit, as in the fundamental-matrix path.
			return EstimateSharedFocalTwoViewGeometry(camera1, points1, points2, matches, options);
		}

		if (camera1.HasPriorFocalLength && camera2.HasPriorFocalLength)
		{
			// Both focals are known, so the pair reduces to the relative pose.
			return EstimateCalibratedTwoViewGeometry(camera1, points1, camera2, points2, matches, options);
		}

		if (!camera1.IsPerspectivePinhole || !camera2.IsPerspectivePinhole)
		{
			// Without a focal-length prior, the only remaining option is the
			// fundamental-matrix path below, which assumes a pinhole projection that a fisheye
			// camera does not have. The calibrated path above does handle fisheye, as it works
			// on bearing vectors. As in COLMAP, a warning is logged once and the pair is marked degenerate.
			if (Interlocked.Exchange(ref fisheyeWithoutFocalPriorWarned, 1) == 0)
			{
				Log.Warning(
					"Marking fisheye pairs without a focal length prior as degenerate, as their focal length cannot be "
					+ "recovered from a fundamental matrix. Provide a focal length prior to register these pairs.");
			}

			return new TwoViewGeometry { Config = TwoViewGeometry.ConfigurationType.Degenerate };
		}

		// Two independent unknown focals, recovered from the fundamental matrix.
		return EstimateUncalibratedTwoViewGeometry(camera1, points1, camera2, points2, matches, options);
	}

	// Recursively estimate multiple configurations by removing the previous set of inliers
	// from the matches until not enough inliers are found.
	// Port of the internal colmap::EstimateMultipleTwoViewGeometries.
	private static TwoViewGeometry EstimateMultipleTwoViewGeometries(
		Camera camera1,
		IReadOnlyList<Vector2d> points1,
		Camera camera2,
		IReadOnlyList<Vector2d> points2,
		IReadOnlyList<FeatureMatch> matches,
		TwoViewGeometryOptions options)
	{
		IReadOnlyList<FeatureMatch> remainingMatches = matches;
		var geometries = new List<TwoViewGeometry>();
		while (true)
		{
			TwoViewGeometry geometry = EstimateTwoViewGeometry(camera1, points1, camera2, points2, remainingMatches, options);
			if (geometry.Config == TwoViewGeometry.ConfigurationType.Degenerate)
			{
				break;
			}

			remainingMatches = ExtractOutlierMatches(remainingMatches, geometry.InlierMatches);

			if (!options.MultipleIgnoreWatermark || geometry.Config != TwoViewGeometry.ConfigurationType.Watermark)
			{
				geometries.Add(geometry);
			}
		}

		if (geometries.Count == 0)
		{
			return new TwoViewGeometry { Config = TwoViewGeometry.ConfigurationType.Degenerate };
		}

		if (geometries.Count == 1)
		{
			return geometries[0];
		}

		var multiGeometry = new TwoViewGeometry { Config = TwoViewGeometry.ConfigurationType.Multiple };
		foreach (TwoViewGeometry geometry in geometries)
		{
			multiGeometry.InlierMatches.AddRange(geometry.InlierMatches);
		}

		return multiGeometry;
	}
}
