// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometryEstimation.Calibrated: the calibrated (both focals known) and spherical
// (omnidirectional camera) paths of colmap/estimators/two_view_geometry.cc -
// EstimateCalibratedTwoViewGeometry and the internal EstimateSphericalTwoViewGeometry.
// TwoViewGeometryEstimation.cs holds the options, shared helpers and file notes.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Scene;

namespace ColmapSharp.Estimators;

public static partial class TwoViewGeometryEstimation
{
	private static void ExtractMatchedCamRaysWithJac(
		Camera camera1,
		IReadOnlyList<Vector2d> points1,
		Camera camera2,
		IReadOnlyList<Vector2d> points2,
		IReadOnlyList<FeatureMatch> matches,
		out Vector2d[] matchedImgPoints1,
		out Vector2d[] matchedImgPoints2,
		out CamRayWithJac[] matchedCamRays1WithJac,
		out CamRayWithJac[] matchedCamRays2WithJac)
	{
		matchedImgPoints1 = new Vector2d[matches.Count];
		matchedImgPoints2 = new Vector2d[matches.Count];
		matchedCamRays1WithJac = new CamRayWithJac[matches.Count];
		matchedCamRays2WithJac = new CamRayWithJac[matches.Count];
		for (int i = 0; i < matches.Count; ++i)
		{
			int idx1 = (int)matches[i].Point2DIdx1;
			int idx2 = (int)matches[i].Point2DIdx2;
			matchedImgPoints1[i] = points1[idx1];
			matchedImgPoints2[i] = points2[idx2];
			matchedCamRays1WithJac[i] = camera1.CamRayFromImgWithJac(points1[idx1]) ?? CamRayWithJac.Zero;
			matchedCamRays2WithJac[i] = camera2.CamRayFromImgWithJac(points2[idx2]) ?? CamRayWithJac.Zero;
		}
	}

	private static RansacReport<Matrix3d, MEstimatorSupportMeasurer.Support> EstimateEssentialMatrix(
		RansacOptions ransacOptions, CamRayWithJac[] camRays1WithJac, CamRayWithJac[] camRays2WithJac)
	{
		return new LoRansac<EssentialMatrixTangentSampsonEstimator, EssentialMatrixTangentSampsonEstimator,
			CamRayWithJac, CamRayWithJac, Matrix3d, MEstimatorSupportMeasurer, MEstimatorSupportMeasurer.Support, RandomSampler>(
			ransacOptions, default, default, new MEstimatorSupportMeasurer()).Estimate(camRays1WithJac, camRays2WithJac);
	}

	/// <summary>
	/// Estimate two-view geometry for an image pair where at least one camera is
	/// omnidirectional (no pinhole image plane, e.g. EQUIRECTANGULAR) and both sides have
	/// known intrinsics. The fundamental matrix is not geometrically meaningful for such
	/// cameras, so the pair is classified from the bearing-based essential matrix and a
	/// ray-space homography. Port of the internal colmap::EstimateSphericalTwoViewGeometry.
	/// </summary>
	internal static TwoViewGeometry EstimateSphericalTwoViewGeometry(
		Camera camera1,
		IReadOnlyList<Vector2d> points1,
		Camera camera2,
		IReadOnlyList<Vector2d> points2,
		IReadOnlyList<FeatureMatch> matches,
		TwoViewGeometryOptions options)
	{
		Util.Check.That(options.Check(), "options.Check()");

		var geometry = new TwoViewGeometry();

		ulong minNumInliers = ToSizeT(options.MinNumInliers);
		if ((ulong)matches.Count < minNumInliers)
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		// For a mixed spherical/perspective pair the perspective camera's bearings come from
		// CamFromImg and degrade if its focal length is off, but an essential matrix can still
		// often be estimated as long as the focal length is not completely wrong. Rather than
		// gate on a focal prior, we attempt the estimation and let the inlier count below
		// decide whether the result is degenerate.

		// For omnidirectional cameras the bearing rays are the only valid representation; the
		// raw image points are kept only for watermark detection.
		ExtractMatchedCamRaysWithJac(camera1, points1, camera2, points2, matches,
			out Vector2d[] matchedImgPoints1, out Vector2d[] matchedImgPoints2,
			out CamRayWithJac[] matchedCamRays1WithJac, out CamRayWithJac[] matchedCamRays2WithJac);

		// Only the bearing-based essential matrix is meaningful: the fundamental matrix and
		// homography assume a pinhole image plane that an omnidirectional camera does not have.
		// The tangent Sampson residual is in pixels, so the threshold is used unscaled.
		RansacOptions ransacOptions = WithMinInlierRatioOverride(options);
		var eReport = EstimateEssentialMatrix(ransacOptions, matchedCamRays1WithJac, matchedCamRays2WithJac);
		geometry.E = eReport.Model;

		// Detect the planar/panoramic degeneracy: under pure rotation E vanishes and its pose
		// decomposition is meaningless, which is the usual capture mode for a 360 degree
		// camera. Kept in ray space, as spherical cameras have no K.
		var matchedCamRays1 = new Vector3d[matches.Count];
		var matchedCamRays2 = new CamRayWithImgPoint[matches.Count];
		for (int i = 0; i < matches.Count; ++i)
		{
			matchedCamRays1[i] = matchedCamRays1WithJac[i].Ray;
			matchedCamRays2[i] = new CamRayWithImgPoint(matchedCamRays2WithJac[i].Ray, matchedImgPoints2[i]);
		}

		// Budget the search for the ratio the homography must beat to be selected below,
		// rather than the default. See EstimateCalibratedTwoViewGeometry.
		RansacOptions hRansacOptions = ransacOptions;
		hRansacOptions.MinInlierRatio = Math.Max(
			ransacOptions.MinInlierRatio,
			options.MaxHInlierRatio * (double)eReport.Support.NumInliers / matches.Count);

		var hEstimator = new HomographyMatrixRayEstimator(camera2);
		var hReport = new LoRansac<HomographyMatrixRayEstimator, HomographyMatrixRayEstimator, Vector3d, CamRayWithImgPoint, Matrix3d,
			MEstimatorSupportMeasurer, MEstimatorSupportMeasurer.Support, RandomSampler>(
			hRansacOptions, hEstimator, hEstimator, new MEstimatorSupportMeasurer()).Estimate(matchedCamRays1, matchedCamRays2);
		if (hReport.Success)
		{
			geometry.H = hReport.Model;
		}

		if ((!eReport.Success || (ulong)eReport.Support.NumInliers < minNumInliers)
			&& (!hReport.Success || (ulong)hReport.Support.NumInliers < minNumInliers))
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		bool[] bestInlierMask = eReport.InlierMask;
		int numInliers = eReport.Support.NumInliers;
		double hEInlierRatio = (double)hReport.Support.NumInliers / eReport.Support.NumInliers;
		if (eReport.Success && (ulong)eReport.Support.NumInliers >= minNumInliers
			&& hEInlierRatio <= options.MaxHInlierRatio)
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Calibrated;
		}
		else
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.PlanarOrPanoramic;
			if (hReport.Support.NumInliers > numInliers)
			{
				numInliers = hReport.Support.NumInliers;
				bestInlierMask = hReport.InlierMask;
			}
		}

		geometry.InlierMatches = ExtractInlierMatches(matches, numInliers, bestInlierMask);

		// Check inlier ratio threshold.
		if (options.MinInlierRatio > 0)
		{
			double inlierRatio = (double)numInliers / matches.Count;
			if (inlierRatio < options.MinInlierRatio)
			{
				geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
				return geometry;
			}
		}

		if (options.DetectWatermark
			&& DetectWatermarkMatches(camera1, matchedImgPoints1, camera2, matchedImgPoints2, numInliers, bestInlierMask, options))
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Watermark;
		}

		if (options.ComputeRelativePose)
		{
			EstimateTwoViewGeometryPose(camera1, points1, camera2, points2, geometry);
		}

		return geometry;
	}

	/// <summary>
	/// Estimate two-view geometry from calibrated image pair.
	/// Port of colmap::EstimateCalibratedTwoViewGeometry.
	/// </summary>
	/// <param name="camera1">Camera of first image.</param>
	/// <param name="points1">Feature points in first image.</param>
	/// <param name="camera2">Camera of second image.</param>
	/// <param name="points2">Feature points in second image.</param>
	/// <param name="matches">Feature matches between first and second image.</param>
	/// <param name="options">Two-view geometry estimation options.</param>
	public static TwoViewGeometry EstimateCalibratedTwoViewGeometry(
		Camera camera1,
		IReadOnlyList<Vector2d> points1,
		Camera camera2,
		IReadOnlyList<Vector2d> points2,
		IReadOnlyList<FeatureMatch> matches,
		TwoViewGeometryOptions options)
	{
		Util.Check.That(options.Check(), "options.Check()");

		if (camera1.IsSpherical || camera2.IsSpherical)
		{
			return EstimateSphericalTwoViewGeometry(camera1, points1, camera2, points2, matches, options);
		}

		var geometry = new TwoViewGeometry();

		ulong minNumInliers = ToSizeT(options.MinNumInliers);
		if ((ulong)matches.Count < minNumInliers)
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		ExtractMatchedCamRaysWithJac(camera1, points1, camera2, points2, matches,
			out Vector2d[] matchedImgPoints1, out Vector2d[] matchedImgPoints2,
			out CamRayWithJac[] matchedCamRays1WithJac, out CamRayWithJac[] matchedCamRays2WithJac);

		// Estimate epipolar models.
		RansacOptions ransacOptions = WithMinInlierRatioOverride(options);

		// The tangent Sampson residual is in pixels, matching the fundamental matrix and
		// homography paths below, so all three share the same unscaled pixel threshold. This
		// also removes the former CamFromImgThreshold conversion, whose single per-camera
		// focal length is only exact at the principal point.
		var eReport = EstimateEssentialMatrix(ransacOptions, matchedCamRays1WithJac, matchedCamRays2WithJac);
		geometry.E = eReport.Model;

		var fReport = EstimateFundamentalMatrix(options, ransacOptions, matchedImgPoints1, matchedImgPoints2);
		geometry.F = fReport.Model;

		// Estimate planar or panoramic model. Budget the estimation as above. The competing
		// model depends on the branch taken below so the homography must reach the smallest
		// count of the two.
		RansacOptions hRansacOptions = ransacOptions;
		hRansacOptions.MinInlierRatio = Math.Max(
			ransacOptions.MinInlierRatio,
			options.MaxHInlierRatio * Math.Min(eReport.Support.NumInliers, fReport.Support.NumInliers) / matches.Count);
		// Undistorted pinhole cameras keep the pixel estimator, where the two are
		// algebraically equivalent.
		var hReport = camera1.IsUndistorted() && camera2.IsUndistorted()
			? EstimateHomographyMatrix(hRansacOptions, matchedImgPoints1, matchedImgPoints2)
			: EstimateHomographyMatrixFromRays(hRansacOptions, camera1, matchedImgPoints1, camera2, matchedImgPoints2);
		geometry.H = hReport.Model;

		if ((!eReport.Success && !fReport.Success && !hReport.Success)
			|| ((ulong)eReport.Support.NumInliers < minNumInliers
				&& (ulong)fReport.Support.NumInliers < minNumInliers
				&& (ulong)hReport.Support.NumInliers < minNumInliers))
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		// Determine inlier ratios of different models.
		double eFInlierRatio = (double)eReport.Support.NumInliers / fReport.Support.NumInliers;
		double hFInlierRatio = (double)hReport.Support.NumInliers / fReport.Support.NumInliers;
		double hEInlierRatio = (double)hReport.Support.NumInliers / eReport.Support.NumInliers;

		bool[] bestInlierMask;
		int numInliers;

		if (eReport.Success && eFInlierRatio > options.MinEFInlierRatio
			&& (ulong)eReport.Support.NumInliers >= minNumInliers)
		{
			// Calibrated configuration.

			// Always use the model with maximum matches.
			if (eReport.Support.NumInliers >= fReport.Support.NumInliers)
			{
				numInliers = eReport.Support.NumInliers;
				bestInlierMask = eReport.InlierMask;
			}
			else
			{
				numInliers = fReport.Support.NumInliers;
				bestInlierMask = fReport.InlierMask;
			}

			if (hEInlierRatio > options.MaxHInlierRatio)
			{
				geometry.Config = TwoViewGeometry.ConfigurationType.PlanarOrPanoramic;
				if (hReport.Support.NumInliers > numInliers)
				{
					numInliers = hReport.Support.NumInliers;
					bestInlierMask = hReport.InlierMask;
				}
			}
			else
			{
				geometry.Config = TwoViewGeometry.ConfigurationType.Calibrated;
			}
		}
		else if (fReport.Success && (ulong)fReport.Support.NumInliers >= minNumInliers)
		{
			// Uncalibrated configuration.
			numInliers = fReport.Support.NumInliers;
			bestInlierMask = fReport.InlierMask;

			if (hFInlierRatio > options.MaxHInlierRatio)
			{
				geometry.Config = TwoViewGeometry.ConfigurationType.PlanarOrPanoramic;
				if (hReport.Support.NumInliers > numInliers)
				{
					numInliers = hReport.Support.NumInliers;
					bestInlierMask = hReport.InlierMask;
				}
			}
			else
			{
				geometry.Config = TwoViewGeometry.ConfigurationType.Uncalibrated;
			}
		}
		else if (hReport.Success && (ulong)hReport.Support.NumInliers >= minNumInliers)
		{
			numInliers = hReport.Support.NumInliers;
			bestInlierMask = hReport.InlierMask;
			geometry.Config = TwoViewGeometry.ConfigurationType.PlanarOrPanoramic;
		}
		else
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		geometry.InlierMatches = ExtractInlierMatches(matches, numInliers, bestInlierMask);

		// Check inlier ratio threshold.
		if (options.MinInlierRatio > 0)
		{
			double inlierRatio = (double)numInliers / matches.Count;
			if (inlierRatio < options.MinInlierRatio)
			{
				geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
				return geometry;
			}
		}

		if (options.DetectWatermark
			&& DetectWatermarkMatches(camera1, matchedImgPoints1, camera2, matchedImgPoints2, numInliers, bestInlierMask, options))
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Watermark;
		}

		if (options.ComputeRelativePose)
		{
			EstimateTwoViewGeometryPose(camera1, points1, camera2, points2, geometry);
		}

		return geometry;
	}
}
