// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometryEstimation: colmap/estimators/two_view_geometry.h and .cc - robust
// verification of an image pair's matches into a Scene/TwoViewGeometry (E/F/H, its
// configuration, inlier matches and optionally the relative pose). Built on the minimal
// solvers in Estimators/Solvers and LO-RANSAC (Optim/LoRansac.cs) with COLMAP's
// MEstimatorSupportMeasurer. Split by responsibility:
// - this file: options, shared helpers, the uncalibrated and forced-homography paths,
//   watermark and stationary-match filtering, TwoViewGeometryFromKnownRelativePose;
// - TwoViewGeometryEstimation.Dispatch.cs: EstimateTwoViewGeometry and the multiple-model
//   estimation;
// - TwoViewGeometryEstimation.Calibrated.cs: the calibrated and spherical paths;
// - TwoViewGeometryEstimation.Focal.cs: the shared-focal and one-sided-focal paths;
// - TwoViewGeometryEstimation.Pose.cs: relative pose decomposition
//   (EstimateTwoViewGeometryPose) and MaybeDecomposeRelativePoses.
// Tests: ColmapSharp.Tests/Estimators/TwoViewGeometryEstimationTests.cs
// (two_view_geometry_test.cc 1:1, see its header for the cases that wait).
//
// Tier C (outcome) for the RANSAC-driven estimators; Tier B for the pose decomposition and
// TwoViewGeometryFromKnownRelativePose (decompositions, no randomness).
//
// Not yet ported (its dependency is not): EstimateRigTwoViewGeometries (needs
// estimators/generalized_pose).
//
// Translation notes:
// - size_t inlier counts are ints here (RansacReport's Support.NumInliers); the size_t casts
//   of `min_num_inliers` are kept through ToSizeT so a negative option compares as in C++.
// - LOG(INFO)/LOG(WARNING) messages are dropped until the library decides how to surface
//   them (PORTING_PLAN.md); the Timer in MaybeDecomposeRelativePoses only fed the log.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>Port of colmap::TwoViewGeometryOptions.</summary>
public sealed class TwoViewGeometryOptions
{
	/// <summary>Minimum number of inliers for non-degenerate two-view geometry.</summary>
	public int MinNumInliers { get; set; } = 15;

	/// <summary>
	/// Minimum ratio of inliers to total matches for non-degenerate geometry. Disabled by
	/// default, only effective when &gt; 0.
	/// </summary>
	public double MinInlierRatio { get; set; }

	/// <summary>
	/// In case both cameras are calibrated, the calibration is verified by estimating an
	/// essential and fundamental matrix and comparing their fractions of number of inliers.
	/// If the essential matrix produces a similar number of inliers
	/// (MinEFInlierRatio * F_num_inliers), the calibration is assumed to be correct.
	/// </summary>
	public double MinEFInlierRatio { get; set; } = 0.95;

	/// <summary>
	/// In case an epipolar geometry can be verified, it is checked whether the geometry
	/// describes a planar scene or panoramic view (pure rotation) described by a homography.
	/// This is a degenerate case, since epipolar geometry is only defined for a moving camera.
	/// If the inlier ratio of a homography comes close to the inlier ratio of the epipolar
	/// geometry, a planar or panoramic configuration is assumed.
	/// </summary>
	public double MaxHInlierRatio { get; set; } = 0.8;

	/// <summary>
	/// In case of valid two-view geometry, it is checked whether the geometry describes a pure
	/// translation in the border region of the image. If more than a certain ratio of inlier
	/// points conform with a pure image translation, a watermark is assumed.
	/// </summary>
	public double WatermarkMinInlierRatio { get; set; } = 0.7;

	/// <summary>
	/// Watermark matches have to be in the border region of the image, defined as a fraction
	/// of the image diagonal around the image borders.
	/// </summary>
	public double WatermarkBorderSize { get; set; } = 0.1;

	/// <summary>
	/// Whether to enable watermark detection. A watermark causes a pure translation in the
	/// image space with inliers in the border region.
	/// </summary>
	public bool DetectWatermark { get; set; } = true;

	/// <summary>Whether to ignore watermark models in multiple model estimation.</summary>
	public bool MultipleIgnoreWatermark { get; set; } = true;

	/// <summary>Maximum translational error of matched points to be considered inliers of a watermark.</summary>
	public double WatermarkDetectionMaxError { get; set; } = 4.0;

	/// <summary>
	/// Whether to filter stationary matches. This is useful when a camera is rigidly mounted
	/// on a moving vehicle and the vehicle itself is visible.
	/// </summary>
	public bool FilterStationaryMatches { get; set; }

	/// <summary>Maximum displacement for points to be considered stationary matches.</summary>
	public double StationaryMatchesMaxError { get; set; } = 4.0;

	/// <summary>In case the user asks for it, only going to estimate a Homography between both cameras.</summary>
	public bool ForceHUse { get; set; }

	/// <summary>
	/// Use DEGENSAC (Chum et al., CVPR 2005) for the fundamental matrix instead of plain
	/// LO-RANSAC, making estimation robust to a dominant scene plane.
	/// </summary>
	public bool UseDegensac { get; set; }

	/// <summary>
	/// Locally optimize the fundamental matrix by nonlinearly minimizing the Sampson error over
	/// the inlier set, instead of refitting the linear eight-point algorithm. The refinement
	/// optimizes the same residual RANSAC scores and keeps the model rank 2 throughout, so no
	/// singular value has to be truncated afterwards. On by default.
	/// </summary>
	public bool UseSampsonRefinement { get; set; } = true;

	/// <summary>Whether to compute the relative pose between the two views.</summary>
	public bool ComputeRelativePose { get; set; }

	/// <summary>
	/// Recursively estimate multiple configurations by removing the previous set of inliers
	/// from the matches until not enough inliers are found. Inlier matches are concatenated
	/// and the configuration type is Multiple if multiple models could be estimated. Note that
	/// in case the model type is Multiple, only InlierMatches is initialized.
	/// </summary>
	public bool MultipleModels { get; set; }

	/// <summary>Options used to robustly estimate the geometry.</summary>
	public RansacOptions RansacOptions = new()
	{
		MaxError = 4.0,
		Confidence = 0.999,
		MinNumTrials = 100,
		MaxNumTrials = 10000,
		MinInlierRatio = 0.25,
	};

	/// <summary>A copy (C++ copies the options struct by value).</summary>
	public TwoViewGeometryOptions Clone() => (TwoViewGeometryOptions)MemberwiseClone();

	/// <summary>
	/// Port of TwoViewGeometryOptions::Check: false if an option is out of range (COLMAP's
	/// CHECK_OPTION logs and returns false; callers wrap it in a throwing check).
	/// </summary>
	public bool Check()
	{
		return MinNumInliers >= 0
			&& MinEFInlierRatio >= 0 && MinEFInlierRatio <= 1
			&& MaxHInlierRatio >= 0 && MaxHInlierRatio <= 1
			&& WatermarkMinInlierRatio >= 0 && WatermarkMinInlierRatio <= 1
			&& WatermarkBorderSize >= 0 && WatermarkBorderSize <= 1
			&& RansacOptions.MaxError > 0
			&& RansacOptions.MinInlierRatio >= 0 && RansacOptions.MinInlierRatio <= 1
			&& RansacOptions.Confidence >= 0 && RansacOptions.Confidence <= 1
			&& RansacOptions.MinNumTrials <= RansacOptions.MaxNumTrials
			&& RansacOptions.RandomSeed >= -1;
	}
}

/// <summary>
/// Two-view geometry estimation. Port of the free functions of
/// colmap/estimators/two_view_geometry.h.
/// </summary>
public static partial class TwoViewGeometryEstimation
{
	private static ulong ToSizeT(int value) => unchecked((ulong)(long)value);

	private static RansacReport<Matrix3d, MEstimatorSupportMeasurer.Support> EstimateHomographyMatrix(
		RansacOptions ransacOptions, Vector2d[] points1, Vector2d[] points2)
	{
		return new LoRansac<HomographyMatrixEstimator, HomographyMatrixEstimator, Vector2d, Vector2d, Matrix3d,
			MEstimatorSupportMeasurer, MEstimatorSupportMeasurer.Support, RandomSampler>(
			ransacOptions, default, default, new MEstimatorSupportMeasurer()).Estimate(points1, points2);
	}

	// Robustly estimate the fundamental matrix, either with plain LO-RANSAC or with DEGENSAC
	// (robust to a dominant scene plane), depending on the options. The DEGENSAC report
	// already has the plain report's type (C++ converts it with ToFundamentalMatrixReport).
	private static RansacReport<Matrix3d, MEstimatorSupportMeasurer.Support> EstimateFundamentalMatrix(
		TwoViewGeometryOptions options, RansacOptions ransacOptions, Vector2d[] points1, Vector2d[] points2)
	{
		if (options.UseDegensac)
		{
			var degensacOptions = new FundamentalMatrixDegensacOptions
			{
				Ransac = ransacOptions,
				UseSampsonRefinement = options.UseSampsonRefinement,
			};
			return FundamentalMatrixDegensac.EstimateFundamentalMatrixDegensac(points1, points2, degensacOptions);
		}

		// The two local estimators differ only in how they refit the inlier set, so both
		// instantiations share the same report type, which depends on the hypothesis
		// estimator alone.
		if (options.UseSampsonRefinement)
		{
			return new LoRansac<FundamentalMatrixSevenPointEstimator, FundamentalMatrixSampsonEstimator, Vector2d, Vector2d, Matrix3d,
				MEstimatorSupportMeasurer, MEstimatorSupportMeasurer.Support, RandomSampler>(
				ransacOptions, default, default, new MEstimatorSupportMeasurer()).Estimate(points1, points2);
		}

		return new LoRansac<FundamentalMatrixSevenPointEstimator, FundamentalMatrixEightPointEstimator, Vector2d, Vector2d, Matrix3d,
			MEstimatorSupportMeasurer, MEstimatorSupportMeasurer.Support, RandomSampler>(
			ransacOptions, default, default, new MEstimatorSupportMeasurer()).Estimate(points1, points2);
	}

	// Robustly estimate the pixel-space homography of a distorted camera pair. A world plane
	// relates image points projectively only under a pinhole projection, so the estimate is
	// made on bearing rays, where it holds for any central camera, and conjugated back by the
	// calibration matrices.
	private static RansacReport<Matrix3d, MEstimatorSupportMeasurer.Support> EstimateHomographyMatrixFromRays(
		RansacOptions ransacOptions, Camera camera1, Vector2d[] points1, Camera camera2, Vector2d[] points2)
	{
		Util.Check.Eq(points1.Length, points2.Length);

		var camRays1 = new Vector3d[points1.Length];
		var camRays2 = new CamRayWithImgPoint[points2.Length];
		for (int i = 0; i < points1.Length; ++i)
		{
			camRays1[i] = camera1.CamRayFromImg(points1[i]) ?? Vector3d.Zero;
			camRays2[i] = new CamRayWithImgPoint(camera2.CamRayFromImg(points2[i]) ?? Vector3d.Zero, points2[i]);
		}

		var estimator = new HomographyMatrixRayEstimator(camera2);
		var rayReport = new LoRansac<HomographyMatrixRayEstimator, HomographyMatrixRayEstimator, Vector3d, CamRayWithImgPoint, Matrix3d,
			MEstimatorSupportMeasurer, MEstimatorSupportMeasurer.Support, RandomSampler>(
			ransacOptions, estimator, estimator, new MEstimatorSupportMeasurer()).Estimate(camRays1, camRays2);

		// The estimator maps rays to rays, so publish K2 H K1^-1 to keep the stored homography
		// in pixel space. K carries no distortion, so for a distorted camera that is the
		// homography in its virtual pinhole frame.
		return new RansacReport<Matrix3d, MEstimatorSupportMeasurer.Support>
		{
			Success = rayReport.Success,
			NumTrials = rayReport.NumTrials,
			Support = rayReport.Support,
			InlierMask = rayReport.InlierMask,
			Model = camera2.CalibrationMatrix() * rayReport.Model * camera1.CalibrationMatrix().Inverse(),
		};
	}

	private static List<FeatureMatch> ExtractInlierMatches(
		IReadOnlyList<FeatureMatch> matches, int numInliers, bool[] inlierMask)
	{
		var inlierMatches = new List<FeatureMatch>(numInliers);
		for (int i = 0; i < matches.Count; ++i)
		{
			if (inlierMask[i])
			{
				inlierMatches.Add(matches[i]);
			}
		}

		// C++ sizes the vector to num_inliers up front; the mask always carries exactly that
		// many set entries, which this check pins.
		Util.Check.Eq(inlierMatches.Count, numInliers);
		return inlierMatches;
	}

	/// <summary>The matches not in <paramref name="inlierMatches"/>, in their original order.</summary>
	internal static List<FeatureMatch> ExtractOutlierMatches(
		IReadOnlyList<FeatureMatch> matches, IReadOnlyList<FeatureMatch> inlierMatches)
	{
		Util.Check.Ge(matches.Count, inlierMatches.Count);

		var inlierMatchesSet = new HashSet<(uint, uint)>(inlierMatches.Count);
		foreach (FeatureMatch match in inlierMatches)
		{
			inlierMatchesSet.Add((match.Point2DIdx1, match.Point2DIdx2));
		}

		var outlierMatches = new List<FeatureMatch>(matches.Count - inlierMatches.Count);
		foreach (FeatureMatch match in matches)
		{
			if (!inlierMatchesSet.Contains((match.Point2DIdx1, match.Point2DIdx2)))
			{
				outlierMatches.Add(match);
			}
		}

		return outlierMatches;
	}

	private static void ExtractMatchedImagePoints(
		IReadOnlyList<Vector2d> points1,
		IReadOnlyList<Vector2d> points2,
		IReadOnlyList<FeatureMatch> matches,
		out Vector2d[] matchedImgPoints1,
		out Vector2d[] matchedImgPoints2)
	{
		matchedImgPoints1 = new Vector2d[matches.Count];
		matchedImgPoints2 = new Vector2d[matches.Count];
		for (int i = 0; i < matches.Count; ++i)
		{
			matchedImgPoints1[i] = points1[(int)matches[i].Point2DIdx1];
			matchedImgPoints2[i] = points2[(int)matches[i].Point2DIdx2];
		}
	}

	private static RansacOptions WithMinInlierRatioOverride(TwoViewGeometryOptions options)
	{
		RansacOptions ransacOptions = options.RansacOptions;
		if (options.MinInlierRatio > 0)
		{
			ransacOptions.MinInlierRatio = options.MinInlierRatio;
		}

		return ransacOptions;
	}

	/// <summary>Port of the internal colmap::EstimateCalibratedHomography (force_H_use).</summary>
	internal static TwoViewGeometry EstimateCalibratedHomography(
		Camera camera1,
		IReadOnlyList<Vector2d> points1,
		Camera camera2,
		IReadOnlyList<Vector2d> points2,
		IReadOnlyList<FeatureMatch> matches,
		TwoViewGeometryOptions options)
	{
		var geometry = new TwoViewGeometry();

		ulong minNumInliers = ToSizeT(options.MinNumInliers);
		if ((ulong)matches.Count < minNumInliers)
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		ExtractMatchedImagePoints(points1, points2, matches, out Vector2d[] matchedImgPoints1, out Vector2d[] matchedImgPoints2);

		// Estimate planar or panoramic model. Estimated on image points rather than rays: the
		// caller only guarantees a pinhole projection here, not a focal length prior, so no
		// rays can be built.
		RansacOptions ransacOptions = WithMinInlierRatioOverride(options);
		var hReport = EstimateHomographyMatrix(ransacOptions, matchedImgPoints1, matchedImgPoints2);
		geometry.H = hReport.Model;

		if (!hReport.Success || (ulong)hReport.Support.NumInliers < minNumInliers)
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		geometry.Config = TwoViewGeometry.ConfigurationType.PlanarOrPanoramic;

		geometry.InlierMatches = ExtractInlierMatches(matches, hReport.Support.NumInliers, hReport.InlierMask);
		if (options.DetectWatermark
			&& DetectWatermarkMatches(camera1, matchedImgPoints1, camera2, matchedImgPoints2,
				hReport.Support.NumInliers, hReport.InlierMask, options))
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
	/// Port of the internal colmap::EstimateUncalibratedTwoViewGeometry: two independent
	/// unknown focals, recovered from the fundamental matrix.
	/// </summary>
	internal static TwoViewGeometry EstimateUncalibratedTwoViewGeometry(
		Camera camera1,
		IReadOnlyList<Vector2d> points1,
		Camera camera2,
		IReadOnlyList<Vector2d> points2,
		IReadOnlyList<FeatureMatch> matches,
		TwoViewGeometryOptions options)
	{
		var geometry = new TwoViewGeometry();

		ulong minNumInliers = ToSizeT(options.MinNumInliers);
		if ((ulong)matches.Count < minNumInliers)
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		ExtractMatchedImagePoints(points1, points2, matches, out Vector2d[] matchedImgPoints1, out Vector2d[] matchedImgPoints2);

		// Estimate epipolar model.
		var fReport = EstimateFundamentalMatrix(options, options.RansacOptions, matchedImgPoints1, matchedImgPoints2);
		geometry.F = fReport.Model;

		// Estimate planar or panoramic model. Estimated on image points rather than rays, as
		// the intrinsics are unknown here and no rays can be built without them. The
		// fundamental matrix above shares that frame, so the comparison below stays
		// self-consistent.

		// Budget the search for the inlier ratio that the homography must reach to be
		// selected below, since a weaker one is discarded anyway.
		RansacOptions hRansacOptions = options.RansacOptions;
		hRansacOptions.MinInlierRatio = Math.Max(
			options.RansacOptions.MinInlierRatio,
			options.MaxHInlierRatio * fReport.Support.NumInliers / matches.Count);
		var hReport = EstimateHomographyMatrix(hRansacOptions, matchedImgPoints1, matchedImgPoints2);
		geometry.H = hReport.Model;

		if ((!fReport.Success && !hReport.Success)
			|| ((ulong)fReport.Support.NumInliers < minNumInliers && (ulong)hReport.Support.NumInliers < minNumInliers))
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		// Determine inlier ratios of different models.
		double hFInlierRatio = (double)hReport.Support.NumInliers / fReport.Support.NumInliers;

		bool[] bestInlierMask = fReport.InlierMask;
		int numInliers = fReport.Support.NumInliers;
		if (hFInlierRatio > options.MaxHInlierRatio)
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.PlanarOrPanoramic;
			if (hReport.Support.NumInliers >= fReport.Support.NumInliers)
			{
				numInliers = hReport.Support.NumInliers;
				bestInlierMask = hReport.InlierMask;
			}
		}
		else
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Uncalibrated;
		}

		geometry.InlierMatches = ExtractInlierMatches(matches, numInliers, bestInlierMask);

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
	/// Detect if inlier matches are caused by a watermark, where a watermark causes a pure
	/// translation in the border of the image. Port of colmap::DetectWatermarkMatches.
	/// </summary>
	public static bool DetectWatermarkMatches(
		Camera camera1,
		IReadOnlyList<Vector2d> points1,
		Camera camera2,
		IReadOnlyList<Vector2d> points2,
		int numInliers,
		bool[] inlierMask,
		TwoViewGeometryOptions options)
	{
		Util.Check.That(options.Check(), "options.Check()");

		// Check if inlier points in border region and extract inlier matches.

		// Width and height are size_t in COLMAP: the squares are exact integers.
		double diagonal1 = Math.Sqrt((double)(((ulong)camera1.Width * (ulong)camera1.Width) + ((ulong)camera1.Height * (ulong)camera1.Height)));
		double diagonal2 = Math.Sqrt((double)(((ulong)camera2.Width * (ulong)camera2.Width) + ((ulong)camera2.Height * (ulong)camera2.Height)));
		double borderSize1 = options.WatermarkBorderSize * diagonal1;
		var box1Min = new Vector2d(borderSize1, borderSize1);
		var box1Max = new Vector2d(camera1.Width - borderSize1, camera1.Height - borderSize1);
		double borderSize2 = options.WatermarkBorderSize * diagonal2;
		var box2Min = new Vector2d(borderSize2, borderSize2);
		var box2Max = new Vector2d(camera2.Width - borderSize2, camera2.Height - borderSize2);

		var inlierPoints1 = new Vector2d[numInliers];
		var inlierPoints2 = new Vector2d[numInliers];

		int numMatchesInBorder = 0;

		int j = 0;
		for (int i = 0; i < inlierMask.Length; ++i)
		{
			if (inlierMask[i])
			{
				inlierPoints1[j] = points1[i];
				inlierPoints2[j] = points2[i];
				++j;

				if (!BoxContains(box1Min, box1Max, points1[i]) && !BoxContains(box2Min, box2Max, points2[i]))
				{
					++numMatchesInBorder;
				}
			}
		}

		double matchesInBorderRatio = (double)numMatchesInBorder / numInliers;

		if (matchesInBorderRatio < options.WatermarkMinInlierRatio)
		{
			return false;
		}

		// Check if matches follow a translational model.

		RansacOptions ransacOptions = options.RansacOptions;
		ransacOptions.MaxError = options.WatermarkDetectionMaxError;
		ransacOptions.MinInlierRatio = options.WatermarkMinInlierRatio;

		var ransac = new LoRansac<TranslationTransformEstimator2d, TranslationTransformEstimator2d, Vector2d, Vector2d, Vector2d>(
			ransacOptions, default, default);
		var report = ransac.Estimate(inlierPoints1, inlierPoints2);

		double inlierRatio = (double)report.Support.NumInliers / numInliers;

		return inlierRatio >= options.WatermarkMinInlierRatio;
	}

	// Eigen::AlignedBox2d::contains: min <= p <= max in every coordinate (so an empty box,
	// min > max, contains nothing).
	private static bool BoxContains(Vector2d min, Vector2d max, Vector2d p) =>
		min.X <= p.X && p.X <= max.X && min.Y <= p.Y && p.Y <= max.Y;

	/// <summary>
	/// Remove matches that are caused by static content that has the same position in both
	/// images. Port of colmap::FilterStationaryMatches.
	/// </summary>
	public static void FilterStationaryMatches(
		double maxError, IReadOnlyList<Vector2d> points1, IReadOnlyList<Vector2d> points2, List<FeatureMatch> matches)
	{
		double maxErrorSquared = maxError * maxError;
		// RemoveAll keeps the survivors' order, as std::remove_if does.
		matches.RemoveAll(match =>
			(points1[(int)match.Point2DIdx1] - points2[(int)match.Point2DIdx2]).SquaredNorm <= maxErrorSquared);
	}

	/// <summary>
	/// Compute two-view geometry from known relative pose and input matches.
	/// Port of colmap::TwoViewGeometryFromKnownRelativePose.
	/// </summary>
	public static TwoViewGeometry TwoViewGeometryFromKnownRelativePose(
		Camera camera1,
		IReadOnlyList<Vector2d> points1,
		Camera camera2,
		IReadOnlyList<Vector2d> points2,
		Rigid3d cam2FromCam1,
		IReadOnlyList<FeatureMatch> matches,
		int minNumInliers = 15,
		double maxError = 4.0)
	{
		Util.Check.Ge(minNumInliers, 0);
		Util.Check.Gt(maxError, 0);

		var geometry = new TwoViewGeometry();
		int numMatches = matches.Count;

		if (numMatches < minNumInliers)
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		// Score in pixels with the tangent Sampson error, matching the design of
		// EstimateCalibratedTwoViewGeometry, so that a pair filtered here and a pair verified
		// there are held to the same threshold.
		var matchedCamRays1WithJac = new CamRayWithJac[numMatches];
		var matchedCamRays2WithJac = new CamRayWithJac[numMatches];
		for (int i = 0; i < numMatches; ++i)
		{
			matchedCamRays1WithJac[i] = camera1.CamRayFromImgWithJac(points1[(int)matches[i].Point2DIdx1]) ?? CamRayWithJac.Zero;
			matchedCamRays2WithJac[i] = camera2.CamRayFromImgWithJac(points2[(int)matches[i].Point2DIdx2]) ?? CamRayWithJac.Zero;
		}

		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
		var residuals = new List<double>(numMatches);
		EssentialMatrix.ComputeSquaredTangentSampsonError(matchedCamRays1WithJac, matchedCamRays2WithJac, e, residuals);
		var inlierMatches = new List<FeatureMatch>();
		double squaredMaxError = maxError * maxError;
		for (int i = 0; i < numMatches; ++i)
		{
			if (residuals[i] <= squaredMaxError)
			{
				inlierMatches.Add(matches[i]);
			}
		}

		if (inlierMatches.Count < minNumInliers)
		{
			geometry.Config = TwoViewGeometry.ConfigurationType.Degenerate;
			return geometry;
		}

		geometry.Config = TwoViewGeometry.ConfigurationType.Calibrated;
		geometry.Cam2FromCam1 = cam2FromCam1;
		geometry.E = e;
		geometry.InlierMatches = inlierMatches;
		return geometry;
	}
}
