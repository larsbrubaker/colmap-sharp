// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometryEstimation.Focal: the paths of colmap/estimators/two_view_geometry.cc that
// recover an unknown focal length along with the relative pose -
// EstimateSharedFocalTwoViewGeometry (Solvers/RelativePoseSharedFocal.cs) and
// EstimateOneSidedFocalTwoViewGeometry (Solvers/RelativePoseOneSidedFocal.cs), each checked
// against a homography for planar/panoramic degeneracies. TwoViewGeometryEstimation.cs
// holds the options, shared helpers and file notes.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Scene;

namespace ColmapSharp.Estimators;

public static partial class TwoViewGeometryEstimation
{
	/// <summary>
	/// Estimate two-view geometry from an image pair captured by a single, uncalibrated
	/// camera with an unknown but shared focal length. Runs PoseLib's 6-point shared-focal
	/// relative-pose solver (with nonlinear local optimization) against a homography model to
	/// reject planar/panoramic degeneracies. On success the geometry has the Uncalibrated
	/// configuration with E, F, and the estimated shared camera in Camera1/Camera2 set.
	/// Both images are assumed to reference the same pinhole-projection camera (perspective,
	/// non-fisheye); <paramref name="camera"/> provides the principal point. A single
	/// isotropic focal length is recovered; multi-focal models are seeded fx = fy = f and
	/// refined later. Port of colmap::EstimateSharedFocalTwoViewGeometry.
	/// </summary>
	public static TwoViewGeometry EstimateSharedFocalTwoViewGeometry(
		Camera camera,
		IReadOnlyList<Vector2d> points1,
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

		// Extract corresponding points. The shared-focal solver operates on
		// principal-point-centered image points (u - cx, v - cy), while the homography and
		// watermark checks operate on raw image points.
		Vector2d principalPoint = camera.PrincipalPoint();
		ExtractMatchedImagePoints(points1, points2, matches, out Vector2d[] matchedImgPoints1, out Vector2d[] matchedImgPoints2);
		var matchedCenteredPoints1 = new Vector2d[matches.Count];
		var matchedCenteredPoints2 = new Vector2d[matches.Count];
		for (int i = 0; i < matches.Count; ++i)
		{
			matchedCenteredPoints1[i] = matchedImgPoints1[i] - principalPoint;
			matchedCenteredPoints2[i] = matchedImgPoints2[i] - principalPoint;
		}

		RansacOptions ransacOptions = WithMinInlierRatioOverride(options);

		// Shared-focal relative pose. Residuals are pixel-space squared Sampson error, so the
		// pixel threshold in ransacOptions is used unscaled.
		var sfReport = new LoRansac<RelativePoseSharedFocalEstimator, RelativePoseSharedFocalEstimator, Vector2d, Vector2d,
			RelativePoseSharedFocalEstimator.Model, MEstimatorSupportMeasurer, MEstimatorSupportMeasurer.Support, RandomSampler>(
			ransacOptions, default, default, new MEstimatorSupportMeasurer()).Estimate(matchedCenteredPoints1, matchedCenteredPoints2);

		// Estimate a homography to detect planar/panoramic degeneracies, where two-view focal
		// recovery is ill-posed and the 6-point solver returns a meaningless focal length.
		// Budget the estimation as in EstimateUncalibratedTwoViewGeometry.
		RansacOptions hRansacOptions = ransacOptions;
		hRansacOptions.MinInlierRatio = Math.Max(
			ransacOptions.MinInlierRatio,
			options.MaxHInlierRatio * sfReport.Support.NumInliers / matches.Count);
		var hReport = EstimateHomographyMatrix(hRansacOptions, matchedImgPoints1, matchedImgPoints2);
		geometry.H = hReport.Model;

		if ((!sfReport.Success && !hReport.Success)
			|| ((ulong)sfReport.Support.NumInliers < minNumInliers && (ulong)hReport.Support.NumInliers < minNumInliers))
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		double hSfInlierRatio = (double)hReport.Support.NumInliers / sfReport.Support.NumInliers;

		bool[] bestInlierMask;
		int numInliers;

		if (sfReport.Success && (ulong)sfReport.Support.NumInliers >= minNumInliers
			&& hSfInlierRatio <= options.MaxHInlierRatio)
		{
			// Shared-focal configuration. Labeled Uncalibrated (the focal is estimated, not a
			// trusted prior); the estimated intrinsics are surfaced via Camera1/Camera2 so
			// consumers can distinguish it from a plain uncalibrated pair and seed the
			// recovered focal.
			numInliers = sfReport.Support.NumInliers;
			bestInlierMask = sfReport.InlierMask;
			geometry.Config = TwoViewGeometry.ConfigurationType.Uncalibrated;
			geometry.E = sfReport.Model.E;
			// Store the shared camera with the estimated focal length; both views share it, so
			// Camera1 and Camera2 are identical (separate copies, as C++ copies by value).
			Camera estimatedCamera = camera.Clone();
			estimatedCamera.SetFocalLength(sfReport.Model.Focal);
			geometry.Camera1 = estimatedCamera;
			geometry.Camera2 = estimatedCamera.Clone();
			// Also expose F = K^-T E K^-1 (K = diag(f, f, 1) at the principal point), which
			// view graph calibration requires from every Uncalibrated pair to calibrate focal
			// lengths. It is also the only epipolar model persisted to the database, which has
			// no columns for the estimated intrinsics.
			double focal = sfReport.Model.Focal;
			var k = new Matrix3d(focal, 0, principalPoint.X, 0, focal, principalPoint.Y, 0, 0, 1);
			Matrix3d kInv = k.Inverse();
			geometry.F = kInv.Transpose() * sfReport.Model.E * kInv;
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

		// If the focal is unidentifiable (parallel or isosceles-intersecting optical axes),
		// drop the estimated intrinsics so the pair degrades to a plain uncalibrated pair (F is
		// valid).
		if (geometry.Camera1 is not null)
		{
			ExtractInlierCamRays(geometry.Camera1, points1, geometry.Camera2!, points2, geometry.InlierMatches,
				out Vector3d[] inlierCamRays1, out Vector3d[] inlierCamRays2);
			var validIndices = new List<int>();
			EssentialMatrix.PoseFromEssentialMatrix(geometry.E!.Value, inlierCamRays1, inlierCamRays2, out Rigid3d cam2FromCam1, validIndices);
			if (validIndices.Count == 0 || !RelativePoseSharedFocalEstimator.IsFocalIdentifiable(cam2FromCam1))
			{
				geometry.E = null;
				geometry.Camera1 = null;
				geometry.Camera2 = null;
			}
		}

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
			&& DetectWatermarkMatches(camera, matchedImgPoints1, camera, matchedImgPoints2, numInliers, bestInlierMask, options))
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Watermark;
		}

		if (options.ComputeRelativePose)
		{
			EstimateTwoViewGeometryPose(camera, points1, camera, points2, geometry);
		}

		return geometry;
	}

	/// <summary>
	/// Estimate two-view geometry when exactly one of the two cameras has a known focal
	/// length, by jointly recovering the relative pose and the other camera's focal
	/// (LO-RANSAC over a minimal 6-point one-sided focal solver) against a homography model
	/// to reject planar/panoramic degeneracies. When the epipolar model wins, the geometry
	/// has the Uncalibrated configuration with E and F set, and the estimated camera in
	/// whichever of Camera1/Camera2 is the uncalibrated image; the other stays unset.
	/// Exactly one camera must be calibrated (a focal prior, or spherical), and the
	/// uncalibrated one must use a pinhole projection.
	/// Port of colmap::EstimateOneSidedFocalTwoViewGeometry.
	/// </summary>
	public static TwoViewGeometry EstimateOneSidedFocalTwoViewGeometry(
		Camera camera1,
		IReadOnlyList<Vector2d> points1,
		Camera camera2,
		IReadOnlyList<Vector2d> points2,
		IReadOnlyList<FeatureMatch> matches,
		TwoViewGeometryOptions options)
	{
		Util.Check.That(options.Check(), "options.Check()");
		// Exactly one side must be calibrated, otherwise this is the calibrated or the
		// shared-focal/uncalibrated problem.
		Util.Check.That(IsCameraCalibrated(camera1) != IsCameraCalibrated(camera2), "IsCameraCalibrated(camera1) != IsCameraCalibrated(camera2)");

		// The solver requires the uncalibrated view first. If it is the second one, run with
		// the roles swapped and map the result back via Invert().
		if (IsCameraCalibrated(camera1))
		{
			var swappedMatches = new List<FeatureMatch>(matches.Count);
			foreach (FeatureMatch match in matches)
			{
				swappedMatches.Add(new FeatureMatch(match.Point2DIdx2, match.Point2DIdx1));
			}

			TwoViewGeometry swapped = EstimateOneSidedFocalTwoViewGeometry(camera2, points2, camera1, points1, swappedMatches, options);
			swapped.Invert();
			return swapped;
		}

		var geometry = new TwoViewGeometry();

		ulong minNumInliers = ToSizeT(options.MinNumInliers);
		if ((ulong)matches.Count < minNumInliers)
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		// The solver takes the uncalibrated view as principal-point-centered image points and
		// the calibrated view as bearing rays with their unprojection Jacobians, which undoes
		// its distortion exactly and lets the residual be measured in that view's pixels. The
		// homography and watermark checks use raw image points.
		Vector2d principalPoint1 = camera1.PrincipalPoint();
		var matchedImgPoints1 = new Vector2d[matches.Count];
		var matchedImgPoints2 = new Vector2d[matches.Count];
		var matchedCenteredPoints1 = new Vector2d[matches.Count];
		var matchedCamRays2WithJac = new CamRayWithJac[matches.Count];
		for (int i = 0; i < matches.Count; ++i)
		{
			int idx1 = (int)matches[i].Point2DIdx1;
			int idx2 = (int)matches[i].Point2DIdx2;
			matchedImgPoints1[i] = points1[idx1];
			matchedImgPoints2[i] = points2[idx2];
			matchedCenteredPoints1[i] = points1[idx1] - principalPoint1;
			matchedCamRays2WithJac[i] = camera2.CamRayFromImgWithJac(points2[idx2]) ?? CamRayWithJac.Zero;
		}

		RansacOptions ransacOptions = WithMinInlierRatioOverride(options);

		// One-sided focal relative pose. Residuals are squared tangent Sampson errors in
		// pixels, so the pixel threshold in ransacOptions applies unscaled, matching the
		// essential matrix, fundamental matrix and homography paths.
		var focalReport = new LoRansac<RelativePoseOneSidedFocalEstimator, RelativePoseOneSidedFocalEstimator, Vector2d, CamRayWithJac,
			RelativePoseOneSidedFocalEstimator.Model, MEstimatorSupportMeasurer, MEstimatorSupportMeasurer.Support, RandomSampler>(
			ransacOptions, default, default, new MEstimatorSupportMeasurer()).Estimate(matchedCenteredPoints1, matchedCamRays2WithJac);

		// Homography, to detect planar/panoramic degeneracies where the epipolar geometry is
		// ill-constrained and the recovered focal is meaningless. Only meaningful if the
		// calibrated view has a pinhole image plane; for a spherical one it is skipped, as in
		// the spherical path. A default report has no inliers, so the checks below then
		// behave as a failed estimate.
		bool hasImagePlane = !camera2.IsSpherical;
		// Budget the estimation as in EstimateUncalibratedTwoViewGeometry.
		RansacOptions hRansacOptions = ransacOptions;
		hRansacOptions.MinInlierRatio = Math.Max(
			ransacOptions.MinInlierRatio,
			options.MaxHInlierRatio * focalReport.Support.NumInliers / matches.Count);
		// C++ constructs the LORANSAC object unconditionally; its constructor only checks the
		// options, which EstimateHomographyMatrix does as well when it runs.
		hRansacOptions.Check();
		var hReport = hasImagePlane
			? EstimateHomographyMatrix(hRansacOptions, matchedImgPoints1, matchedImgPoints2)
			: new RansacReport<Matrix3d, MEstimatorSupportMeasurer.Support>();
		if (hReport.Success)
		{
			// Only set on success: a failed report's model is meaningless, and the swap path
			// above would invert it.
			geometry.H = hReport.Model;
		}

		if ((!focalReport.Success && !hReport.Success)
			|| ((ulong)focalReport.Support.NumInliers < minNumInliers && (ulong)hReport.Support.NumInliers < minNumInliers))
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		double hFocalInlierRatio = (double)hReport.Support.NumInliers / focalReport.Support.NumInliers;

		bool[] bestInlierMask;
		int numInliers;

		if (focalReport.Success && (ulong)focalReport.Support.NumInliers >= minNumInliers
			&& hFocalInlierRatio <= options.MaxHInlierRatio)
		{
			// The focal is estimated rather than a trusted prior, hence Uncalibrated, and is
			// surfaced via Camera1 so consumers can tell this apart from a plain uncalibrated
			// pair. Camera2 stays unset: its intrinsics were an input, not an estimate.
			numInliers = focalReport.Support.NumInliers;
			bestInlierMask = focalReport.InlierMask;
			geometry.Config = TwoViewGeometry.ConfigurationType.Uncalibrated;
			geometry.E = focalReport.Model.E;
			Camera estimatedCamera1 = camera1.Clone();
			estimatedCamera1.SetFocalLength(focalReport.Model.Focal);
			geometry.Camera1 = estimatedCamera1;
			if (hasImagePlane)
			{
				// Also expose F, so that epipolar consumers unaware of the estimated focal can
				// use this config directly. It assumes a pinhole image plane on both sides, so
				// it is only an approximation for a distorted second camera; the exact geometry
				// is carried by E plus the rays. A spherical second view has no calibration
				// matrix at all, so no F is published.
				Matrix3d k1Inv = estimatedCamera1.CalibrationMatrix().Inverse();
				Matrix3d k2Inv = camera2.CalibrationMatrix().Inverse();
				geometry.F = k2Inv.Transpose() * focalReport.Model.E * k1Inv;
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
