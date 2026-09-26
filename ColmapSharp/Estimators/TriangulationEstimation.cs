// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TriangulationEstimation: colmap/estimators/triangulation.h/.cc - robust multi-view point
// triangulation: TriangulationEstimator (two-view DLT or multi-view DLT on bearings, with
// cheirality and triangulation-angle checks) inside LO-RANSAC over all view pairs
// (CombinationSampler). The DLT solvers are Geometry/Triangulation.cs; the residuals are
// Scene/Projection.cs. Consumers: the incremental triangulator and mapper (Phase 10). COLMAP
// has no triangulation_test.cc; the C#-only tests are in
// ColmapSharp.Tests/Estimators/TriangulationEstimationTests.cs.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>
/// Triangulation estimator to estimate 3D point from multiple observations. The
/// triangulation must satisfy the following constraints: sufficient triangulation angle
/// between observation pairs, and all observations must satisfy the cheirality constraint.
/// An observation is composed of an image measurement and the corresponding camera pose and
/// calibration. Port of colmap::TriangulationEstimator.
/// </summary>
public readonly struct TriangulationEstimator
	: IEstimator<TriangulationEstimator.PointData, TriangulationEstimator.PoseData, Vector3d>,
	ILocalEstimator<TriangulationEstimator.PointData, TriangulationEstimator.PoseData, Vector3d>
{
	private readonly double _minTriAngle;
	private readonly ResidualType _residualType;

	/// <summary>An estimator with the given minimum triangulation angle (radians) and residual.</summary>
	public TriangulationEstimator(double minTriAngle, ResidualType residualType)
	{
		Check.Ge(minTriAngle, 0);
		_minTriAngle = minTriAngle;
		_residualType = residualType;
	}

	/// <summary>The residual RANSAC scores with.</summary>
	public enum ResidualType
	{
		/// <summary>Squared angle between the observed ray and the ray to the point.</summary>
		AngularError,

		/// <summary>Squared reprojection error in pixels.</summary>
		ReprojectionError,
	}

	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 2;

	/// <summary>
	/// Estimate a 3D point from the observations; appends it if the cheirality and
	/// triangulation angle checks pass.
	/// </summary>
	public void Estimate(ReadOnlySpan<PointData> pointData, ReadOnlySpan<PoseData> poseData, List<Vector3d> models)
	{
		Check.Ge(pointData.Length, 2);
		Check.Eq(pointData.Length, poseData.Length);

		models.Clear();

		Vector3d xyz;
		if (pointData.Length == 2)
		{
			// More efficient closed-form solution for the two-view case.
			bool allCamsPerspective = Check.NotNull(poseData[0].Camera).IsPerspective
				&& Check.NotNull(poseData[1].Camera).IsPerspective;
			if (allCamsPerspective)
			{
				if (!Triangulation.TriangulatePoint(
					poseData[0].CamFromWorld,
					poseData[1].CamFromWorld,
					pointData[0].CamRay.HNormalized(),
					pointData[1].CamRay.HNormalized(),
					out xyz))
				{
					return;
				}
			}
			else if (!Triangulation.TriangulatePoint(
				poseData[0].CamFromWorld, poseData[1].CamFromWorld, pointData[0].CamRay, pointData[1].CamRay, out xyz))
			{
				return;
			}
		}
		else
		{
			var camsFromWorld = new Matrix3x4d[pointData.Length];
			var camRays = new Vector3d[pointData.Length];
			for (int i = 0; i < pointData.Length; ++i)
			{
				camsFromWorld[i] = poseData[i].CamFromWorld;
				camRays[i] = pointData[i].CamRay;
			}

			if (!Triangulation.TriangulateMultiViewPoint(camsFromWorld, camRays, out xyz))
			{
				return;
			}
		}

		// Cheirality. Perspective cameras require positive depth (the point in front
		// of the local +Z axis). Omnidirectional cameras (e.g. EQUIRECTANGULAR) have
		// no single front, but the point must still lie in the half-space the
		// observed bearing points toward.
		for (int i = 0; i < poseData.Length; ++i)
		{
			if (poseData[i].Camera!.IsPerspective)
			{
				if (!Projection.HasPointPositiveDepth(poseData[i].CamFromWorld, xyz))
				{
					return;
				}
			}
			else if ((poseData[i].CamFromWorld * xyz.Homogeneous()).Dot(pointData[i].CamRay) <= 0.0)
			{
				return;
			}
		}

		// Require a sufficient triangulation angle for at least one pair of views.
		for (int i = 0; i < poseData.Length; ++i)
		{
			for (int j = 0; j < i; ++j)
			{
				if (Triangulation.CalculateTriangulationAngle(poseData[i].ProjCenter, poseData[j].ProjCenter, xyz) >= _minTriAngle)
				{
					models.Add(xyz);
					return;
				}
			}
		}
	}

	/// <summary>LO-RANSAC's local estimate: TriangulationEstimator has no Refine, so this re-estimates.</summary>
	public void EstimateLocal(ReadOnlySpan<PointData> pointData, ReadOnlySpan<PoseData> poseData, in Vector3d initialModel, List<Vector3d> models) =>
		Estimate(pointData, poseData, models);

	/// <summary>Calculate residuals in terms of squared reprojection or angular error.</summary>
	public void Residuals(ReadOnlySpan<PointData> pointData, ReadOnlySpan<PoseData> poseData, in Vector3d xyz, Span<double> residuals)
	{
		Check.Eq(pointData.Length, poseData.Length);

		for (int i = 0; i < pointData.Length; ++i)
		{
			if (_residualType == ResidualType.ReprojectionError)
			{
				residuals[i] = Projection.CalculateSquaredReprojectionError(
					pointData[i].ImgPoint, xyz, poseData[i].CamFromWorld, poseData[i].Camera!);
			}
			else if (_residualType == ResidualType.AngularError)
			{
				double angularError = Projection.CalculateAngularReprojectionError(
					pointData[i].CamRay, xyz, poseData[i].CamFromWorld);
				residuals[i] = angularError * angularError;
			}
		}
	}

	/// <summary>An image observation. Port of colmap::TriangulationEstimator::PointData.</summary>
	public struct PointData
	{
		/// <summary>Image observation in pixels. Only needs to be set for REPROJECTION_ERROR.</summary>
		public Vector2d ImgPoint;

		/// <summary>
		/// Unit bearing vector in the camera frame (Camera.CamRayFromImg). The canonical
		/// observation representation for all camera models, including omnidirectional
		/// (EQUIRECTANGULAR) back-hemisphere rays that the 2D normalized representation
		/// cannot encode.
		/// </summary>
		public Vector3d CamRay;

		/// <summary>An observation from its pixel and bearing.</summary>
		public PointData(Vector2d imgPoint, Vector3d camRay)
		{
			ImgPoint = imgPoint;
			CamRay = camRay;
		}
	}

	/// <summary>The pose and camera of an observation. Port of colmap::TriangulationEstimator::PoseData.</summary>
	public struct PoseData
	{
		/// <summary>The projection matrix for the image of the observation.</summary>
		public Matrix3x4d CamFromWorld;

		/// <summary>The projection center for the image of the observation.</summary>
		public Vector3d ProjCenter;

		/// <summary>The camera for the image of the observation.</summary>
		public Camera? Camera;

		/// <summary>Pose data from its projection matrix, center and camera.</summary>
		public PoseData(Matrix3x4d camFromWorld, Vector3d projCenter, Camera camera)
		{
			CamFromWorld = camFromWorld;
			ProjCenter = projCenter;
			Camera = camera;
		}
	}
}

/// <summary>Port of colmap::EstimateTriangulationOptions.</summary>
public sealed class EstimateTriangulationOptions
{
	/// <summary>Creates the options with COLMAP's defaults.</summary>
	public EstimateTriangulationOptions()
	{
		RansacOptions.MaxError = MathUtils.DegToRad(2.0);
		RansacOptions.Confidence = 0.9999;
		RansacOptions.MinInlierRatio = 0.02;
		RansacOptions.MaxNumTrials = 10000;
	}

	/// <summary>Minimum triangulation angle in radians.</summary>
	public double MinTriAngle { get; set; }

	/// <summary>The employed residual type.</summary>
	public TriangulationEstimator.ResidualType ResidualType { get; set; } = TriangulationEstimator.ResidualType.AngularError;

	/// <summary>RANSAC options for TriangulationEstimator.</summary>
	public RansacOptions RansacOptions = new();

	/// <summary>Checks the options.</summary>
	public void Check()
	{
		Util.Check.Ge(MinTriAngle, 0.0);
		RansacOptions.Check();
	}
}

/// <summary>Port of the free function of colmap/estimators/triangulation.h.</summary>
public static class TriangulationEstimation
{
	/// <summary>
	/// Robustly estimate 3D point from observations in multiple views using RANSAC and a
	/// subsequent non-linear refinement using all inliers. Returns true if the estimated
	/// number of inliers has more than two views. Port of colmap::EstimateTriangulation.
	/// </summary>
	public static bool EstimateTriangulation(
		EstimateTriangulationOptions options,
		IReadOnlyList<Vector2d> points,
		IReadOnlyList<Rigid3d> camsFromWorld,
		IReadOnlyList<Camera> cameras,
		out bool[] inlierMask,
		out Vector3d xyz)
	{
		Check.Ge(points.Count, 2);
		Check.Eq(points.Count, camsFromWorld.Count);
		Check.Eq(points.Count, cameras.Count);
		options.Check();
		inlierMask = [];
		xyz = default;

		var pointData = new TriangulationEstimator.PointData[points.Count];
		var poseData = new TriangulationEstimator.PoseData[points.Count];
		for (int i = 0; i < points.Count; ++i)
		{
			// Unit bearing in the camera frame. CamRayFromImg yields a valid ray for
			// any camera model, including omnidirectional (EQUIRECTANGULAR)
			// back-hemisphere observations that CamFromImg cannot represent. Fall back
			// to a defined forward bearing (+Z) if unprojection fails, so downstream
			// normalize() in the DLT never sees a zero vector (which would produce
			// NaNs).
			pointData[i] = new TriangulationEstimator.PointData(
				points[i], cameras[i].CamRayFromImg(points[i]) ?? Vector3d.UnitZ);
			poseData[i] = new TriangulationEstimator.PoseData(
				camsFromWorld[i].ToMatrix(), camsFromWorld[i].TgtOriginInSrc(), cameras[i]);
		}

		// Robustly estimate track using LORANSAC.
		var estimator = new TriangulationEstimator(options.MinTriAngle, options.ResidualType);
		var ransac = new LoRansac<TriangulationEstimator, TriangulationEstimator, TriangulationEstimator.PointData,
			TriangulationEstimator.PoseData, Vector3d, InlierSupportMeasurer, InlierSupportMeasurer.Support, CombinationSampler>(
			options.RansacOptions, estimator, estimator, new InlierSupportMeasurer());
		RansacReport<Vector3d, InlierSupportMeasurer.Support> report = ransac.Estimate(pointData, poseData);
		if (!report.Success)
		{
			return false;
		}

		inlierMask = report.InlierMask;
		xyz = report.Model;

		return report.Success;
	}
}
