// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometryEstimation.Pose: relative pose recovery from a verified two-view geometry
// in colmap/estimators/two_view_geometry.cc - EstimateTwoViewGeometryPose (decomposing the
// selected E, F or H via Geometry/EssentialMatrix.cs and HomographyMatrix.cs) and
// MaybeDecomposeRelativePoses (the same for every pair of a DatabaseCache, refitting a
// matrix an older database did not store). TwoViewGeometryEstimation.cs holds the notes.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

public static partial class TwoViewGeometryEstimation
{
	private static void ExtractInlierCamRays(
		Camera camera1,
		IReadOnlyList<Vector2d> points1,
		Camera camera2,
		IReadOnlyList<Vector2d> points2,
		IReadOnlyList<FeatureMatch> inlierMatches,
		out Vector3d[] inlierCamRays1,
		out Vector3d[] inlierCamRays2)
	{
		inlierCamRays1 = new Vector3d[inlierMatches.Count];
		inlierCamRays2 = new Vector3d[inlierMatches.Count];
		for (int i = 0; i < inlierMatches.Count; ++i)
		{
			FeatureMatch match = inlierMatches[i];
			inlierCamRays1[i] = camera1.CamRayFromImg(points1[(int)match.Point2DIdx1]) ?? Vector3d.Zero;
			inlierCamRays2[i] = camera2.CamRayFromImg(points2[(int)match.Point2DIdx2]) ?? Vector3d.Zero;
		}
	}

	private static bool EstimateTwoViewGeometryPoseFromCamRays(
		Camera camera1,
		Camera camera2,
		Vector3d[] inlierCamRays1,
		Vector3d[] inlierCamRays2,
		TwoViewGeometry geometry)
	{
		var points3D = new List<Vector3d>();

		// Omnidirectional cameras (no focal length, e.g. EQUIRECTANGULAR) have no calibration
		// matrix, so they never reach the fundamental-matrix branch below. The homography
		// branch handles them by decomposing through the identity, their homography already
		// being in ray space.
		Rigid3d cam2FromCam1;
		var validIndices = new List<int>();
		// Decompose the model the solver selected. A calibrated pair, or one whose intrinsics
		// a solver estimated (Camera1/Camera2 set), selected E, with its rays already
		// calibrated accordingly. A plain uncalibrated pair selected F and recovers E from it
		// with the current calibration below. Note E is stored for every pair, so its presence
		// does not imply it is the selected model.
		if (geometry.Config == TwoViewGeometry.ConfigurationType.Calibrated
			|| geometry.Camera1 is not null || geometry.Camera2 is not null)
		{
			Matrix3d e = geometry.E ?? throw new InvalidOperationException("Check failed: geometry->E.has_value()");
			EssentialMatrix.PoseFromEssentialMatrix(e, inlierCamRays1, inlierCamRays2, out cam2FromCam1, validIndices);
			if (validIndices.Count == 0)
			{
				return false;
			}
		}
		else if (geometry.Config == TwoViewGeometry.ConfigurationType.Uncalibrated)
		{
			Matrix3d f = geometry.F ?? throw new InvalidOperationException("Check failed: geometry->F.has_value()");
			Matrix3d e = EssentialMatrix.EssentialFromFundamentalMatrix(camera2.CalibrationMatrix(), f, camera1.CalibrationMatrix());
			EssentialMatrix.PoseFromEssentialMatrix(e, inlierCamRays1, inlierCamRays2, out cam2FromCam1, validIndices);
			if (validIndices.Count == 0)
			{
				return false;
			}
		}
		else if (geometry.Config == TwoViewGeometry.ConfigurationType.Planar
			|| geometry.Config == TwoViewGeometry.ConfigurationType.Panoramic
			|| geometry.Config == TwoViewGeometry.ConfigurationType.PlanarOrPanoramic)
		{
			Matrix3d h = geometry.H ?? throw new InvalidOperationException("Check failed: geometry->H.has_value()");
			// The decomposition removes the calibration first. The spherical path stores its
			// homography in ray space, having no calibration matrix, so such a pair passes the
			// identity on both sides.
			bool isRaySpace = camera1.IsSpherical || camera2.IsSpherical;
			Matrix3d k1 = isRaySpace ? Matrix3d.Identity : camera1.CalibrationMatrix();
			Matrix3d k2 = isRaySpace ? Matrix3d.Identity : camera2.CalibrationMatrix();
			HomographyMatrix.PoseFromHomographyMatrix(h, k1, k2, inlierCamRays1, inlierCamRays2, out cam2FromCam1, out _, points3D);
			if (geometry.Config == TwoViewGeometry.ConfigurationType.PlanarOrPanoramic)
			{
				geometry.Config = cam2FromCam1.Translation.SquaredNorm < 1e-12
					? TwoViewGeometry.ConfigurationType.Panoramic
					: TwoViewGeometry.ConfigurationType.Planar;
			}

			if (geometry.Config == TwoViewGeometry.ConfigurationType.Panoramic)
			{
				geometry.TriAngle = 0;
			}

			if (geometry.Config == TwoViewGeometry.ConfigurationType.Planar && points3D.Count == 0)
			{
				return false;
			}
		}
		else
		{
			return false;
		}

		geometry.Cam2FromCam1 = cam2FromCam1;

		if (validIndices.Count > 0)
		{
			// Essential-matrix paths (CALIBRATED / UNCALIBRATED): the triangulation angle is the
			// parallax between the surviving corresponding rays, which needs no explicit
			// triangulation.
			Quaterniond cam1FromCam2Rotation = cam2FromCam1.Rotation.Inverse();
			var triAngles = new double[validIndices.Count];
			for (int i = 0; i < validIndices.Count; ++i)
			{
				int idx = validIndices[i];
				double angle = Triangulation.CalculateAngleBetweenVectors(
					inlierCamRays1[idx], cam1FromCam2Rotation * inlierCamRays2[idx]);
				triAngles[i] = Math.Min(angle, Math.PI - angle);
			}

			geometry.TriAngle = MathUtils.Median(triAngles.AsSpan());
		}
		else if (points3D.Count > 0)
		{
			// Homography path (PLANAR): the triangulation angle is computed from the 3D points
			// recovered by PoseFromHomographyMatrix.
			Vector3d projCenter1 = Vector3d.Zero;
			Vector3d projCenter2 = cam2FromCam1.TgtOriginInSrc();
			geometry.TriAngle = MathUtils.Median(
				Triangulation.CalculateTriangulationAngles(projCenter1, projCenter2, points3D).AsSpan());
		}

		return true;
	}

	/// <summary>
	/// Estimate relative pose for two-view geometry; sets Cam2FromCam1 and TriAngle (and
	/// resolves PlanarOrPanoramic into Planar or Panoramic). Port of
	/// colmap::EstimateTwoViewGeometryPose.
	/// </summary>
	/// <param name="camera1">Camera of first image.</param>
	/// <param name="points1">Feature points in first image.</param>
	/// <param name="camera2">Camera of second image.</param>
	/// <param name="points2">Feature points in second image.</param>
	/// <param name="geometry">The verified geometry, updated in place.</param>
	public static bool EstimateTwoViewGeometryPose(
		Camera camera1,
		IReadOnlyList<Vector2d> points1,
		Camera camera2,
		IReadOnlyList<Vector2d> points2,
		TwoViewGeometry geometry)
	{
		// We need a valid epipolar geometry to estimate the relative pose.
		if (geometry.Config != TwoViewGeometry.ConfigurationType.Calibrated
			&& geometry.Config != TwoViewGeometry.ConfigurationType.Uncalibrated
			&& geometry.Config != TwoViewGeometry.ConfigurationType.Planar
			&& geometry.Config != TwoViewGeometry.ConfigurationType.Panoramic
			&& geometry.Config != TwoViewGeometry.ConfigurationType.PlanarOrPanoramic)
		{
			return false;
		}

		if (geometry.InlierMatches.Count == 0)
		{
			return false;
		}

		// Calibrate rays with the intrinsics the solver estimated where available (its focal,
		// not the camera's stale default), else the given cameras.
		Camera effectiveCamera1 = geometry.Camera1 ?? camera1;
		Camera effectiveCamera2 = geometry.Camera2 ?? camera2;
		ExtractInlierCamRays(effectiveCamera1, points1, effectiveCamera2, points2, geometry.InlierMatches,
			out Vector3d[] inlierCamRays1, out Vector3d[] inlierCamRays2);
		return EstimateTwoViewGeometryPoseFromCamRays(camera1, camera2, inlierCamRays1, inlierCamRays2, geometry);
	}

	private static void ExtractInlierImagePoints(
		IReadOnlyList<Vector2d> points1,
		IReadOnlyList<Vector2d> points2,
		IReadOnlyList<FeatureMatch> inlierMatches,
		out Vector2d[] inlierPoints1,
		out Vector2d[] inlierPoints2) =>
		ExtractMatchedImagePoints(points1, points2, inlierMatches, out inlierPoints1, out inlierPoints2);

	// Fits the E/F/H matrix matching geometry.Config from the existing inlier matches when it
	// was not persisted in the database (e.g. databases from older COLMAP versions that stored
	// only the configuration). The 8-point estimators are used because the inlier matches have
	// already been filtered by RANSAC at storage time (inlier counts are typically well above
	// 8); pairs with fewer inliers would yield unreliable geometry and are returned as a
	// failed fit. Returns false if the inlier set is too small or the fit returned no model;
	// true if the relevant matrix is present (already or now).
	private static bool MaybeFitMissingTwoViewGeometryMatrix(
		IReadOnlyList<Vector2d> points1,
		IReadOnlyList<Vector2d> points2,
		Vector3d[] inlierCamRays1,
		Vector3d[] inlierCamRays2,
		TwoViewGeometry geometry)
	{
		var models = new List<Matrix3d>();
		switch (geometry.Config)
		{
			case TwoViewGeometry.ConfigurationType.Calibrated:
			{
				if (geometry.E is not null)
				{
					return true;
				}

				var validCamRays1 = new List<Vector3d>(inlierCamRays1.Length);
				var validCamRays2 = new List<Vector3d>(inlierCamRays2.Length);
				for (int i = 0; i < inlierCamRays1.Length; ++i)
				{
					if (!IsZero(inlierCamRays1[i]) && !IsZero(inlierCamRays2[i]))
					{
						validCamRays1.Add(inlierCamRays1[i]);
						validCamRays2.Add(inlierCamRays2[i]);
					}
				}

				if (validCamRays1.Count < EssentialMatrixEightPointEstimator.MinNumSamples)
				{
					return false;
				}

				default(EssentialMatrixEightPointEstimator).Estimate(
					System.Runtime.InteropServices.CollectionsMarshal.AsSpan(validCamRays1),
					System.Runtime.InteropServices.CollectionsMarshal.AsSpan(validCamRays2),
					models);
				if (models.Count == 0)
				{
					return false;
				}

				geometry.E = models[0];
				return true;
			}

			case TwoViewGeometry.ConfigurationType.Uncalibrated:
			{
				if (geometry.F is not null)
				{
					return true;
				}

				ExtractInlierImagePoints(points1, points2, geometry.InlierMatches, out Vector2d[] inlierPoints1, out Vector2d[] inlierPoints2);
				if (inlierPoints1.Length < FundamentalMatrixEightPointEstimator.MinNumSamples)
				{
					return false;
				}

				default(FundamentalMatrixEightPointEstimator).Estimate(inlierPoints1, inlierPoints2, models);
				if (models.Count == 0)
				{
					return false;
				}

				geometry.F = models[0];
				return true;
			}

			case TwoViewGeometry.ConfigurationType.Planar:
			case TwoViewGeometry.ConfigurationType.Panoramic:
			case TwoViewGeometry.ConfigurationType.PlanarOrPanoramic:
			{
				if (geometry.H is not null)
				{
					return true;
				}

				ExtractInlierImagePoints(points1, points2, geometry.InlierMatches, out Vector2d[] inlierPoints1, out Vector2d[] inlierPoints2);
				if (inlierPoints1.Length < HomographyMatrixEstimator.MinNumSamples)
				{
					return false;
				}

				default(HomographyMatrixEstimator).Estimate(inlierPoints1, inlierPoints2, models);
				if (models.Count == 0)
				{
					return false;
				}

				geometry.H = models[0];
				return true;
			}

			default:
				return false;
		}
	}

	// Eigen's isZero(): every coefficient within dummy_precision (1e-12) of zero.
	private static bool IsZero(Vector3d v) =>
		Math.Abs(v.X) <= LinearAlgebraConstants.DummyPrecision
		&& Math.Abs(v.Y) <= LinearAlgebraConstants.DummyPrecision
		&& Math.Abs(v.Z) <= LinearAlgebraConstants.DummyPrecision;

	/// <summary>
	/// Decompose relative poses from two-view geometries in the database cache and update the
	/// results in-memory. Skips pairs that already have a relative pose or have invalid
	/// two-view geometries (Undefined, Degenerate, Watermark, Multiple). The recovered
	/// translation is normalized to unit length. Port of colmap::MaybeDecomposeRelativePoses.
	/// </summary>
	public static void MaybeDecomposeRelativePoses(DatabaseCache databaseCache)
	{
		IReadOnlyDictionary<uint, Camera> cameras = databaseCache.Cameras;
		IReadOnlyDictionary<uint, Image> images = databaseCache.Images;
		CorrespondenceGraph correspondenceGraph = databaseCache.CorrespondenceGraph;

		foreach (ulong pairId in correspondenceGraph.ImagePairs())
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);

			TwoViewGeometry twoViewGeometry =
				correspondenceGraph.ExtractTwoViewGeometry(imageId1, imageId2, extractInlierMatches: true);

			if (twoViewGeometry.Cam2FromCam1 is not null)
			{
				continue;
			}

			bool isInvalid = twoViewGeometry.Config == TwoViewGeometry.ConfigurationType.Undefined
				|| twoViewGeometry.Config == TwoViewGeometry.ConfigurationType.Degenerate
				|| twoViewGeometry.Config == TwoViewGeometry.ConfigurationType.Watermark
				|| twoViewGeometry.Config == TwoViewGeometry.ConfigurationType.Multiple;

			if (isInvalid)
			{
				continue;
			}

			// COLMAP counts these pairs as failed decompositions for its log line only.
			if (twoViewGeometry.InlierMatches.Count == 0)
			{
				continue;
			}

			Image image1 = images[imageId1];
			Image image2 = images[imageId2];
			Camera camera1 = cameras[image1.CameraId];
			Camera camera2 = cameras[image2.CameraId];

			var points1 = new List<Vector2d>((int)image1.NumPoints2D);
			foreach (Point2D point in image1.Points2D)
			{
				points1.Add(point.Xy);
			}

			var points2 = new List<Vector2d>((int)image2.NumPoints2D);
			foreach (Point2D point in image2.Points2D)
			{
				points2.Add(point.Xy);
			}

			// Calibrate rays with the intrinsics the solver estimated where available (its
			// focal, not the camera's stale default), else the given cameras.
			Camera effectiveCamera1 = twoViewGeometry.Camera1 ?? camera1;
			Camera effectiveCamera2 = twoViewGeometry.Camera2 ?? camera2;

			ExtractInlierCamRays(effectiveCamera1, points1, effectiveCamera2, points2, twoViewGeometry.InlierMatches,
				out Vector3d[] inlierCamRays1, out Vector3d[] inlierCamRays2);

			if (!MaybeFitMissingTwoViewGeometryMatrix(points1, points2, inlierCamRays1, inlierCamRays2, twoViewGeometry))
			{
				continue;
			}

			bool success = EstimateTwoViewGeometryPoseFromCamRays(camera1, camera2, inlierCamRays1, inlierCamRays2, twoViewGeometry);

			if (success && twoViewGeometry.Cam2FromCam1 is Rigid3d cam2FromCam1)
			{
				double norm = cam2FromCam1.Translation.Norm;
				if (norm > 1e-12)
				{
					twoViewGeometry.Cam2FromCam1 = new Rigid3d(cam2FromCam1.Rotation, cam2FromCam1.Translation / norm);
				}

				correspondenceGraph.UpdateTwoViewGeometry(imageId1, imageId2, twoViewGeometry);
			}
		}
	}
}
