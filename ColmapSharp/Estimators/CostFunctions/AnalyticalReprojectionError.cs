// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AnalyticalReprojectionError: the analytic-Jacobian half of
// colmap/estimators/cost_functions/reprojection_error.h -
// AnalyticalReprojErrorCostFunction and AnalyticalReprojErrorConstantPoseCostFunction, the
// cost functions bundle adjustment actually uses for ReprojErrorCostFunctor and
// ReprojErrorConstantPoseCostFunctor (CameraCostFunctions.cs picks them, as COLMAP's
// CreateCameraCostFunction does for every model with ImgFromCamWithJac, which in COLMAP
// 4.2.0 is every model). They chain the camera model's analytic projection Jacobian
// (Sensor/*CameraModels.Jacobian.cs) with QuaternionUtils.QuaternionRotatePointWithJac
// instead of evaluating Jets, which is the bulk of BA's residual cost. Their autodiff
// counterparts are in ReprojectionError.cs; the tests check the two agree
// (ReprojectionErrorTests, through Solver/GradientChecker.cs).
//
// Evaluation allocates nothing: the intermediate Jacobians live on the stack, and an
// instance holds only read-only state, so it may be evaluated from several threads.
// Tier B: the 2x3 by 3xN products sum left to right; Eigen may pair them for SIMD.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>
/// Full reprojection error cost function with analytical Jacobians. Blocks:
/// point3D_in_world (3), cam_from_world (7), camera params.
/// Port of colmap::AnalyticalReprojErrorCostFunction.
/// </summary>
public sealed class AnalyticalReprojErrorCostFunction<TModel> : CostFunction
	where TModel : struct, ICameraModel<TModel>
{
	private readonly Vector2d _point2D;

	/// <summary>Creates the cost function for one observation.</summary>
	public AnalyticalReprojErrorCostFunction(Vector2d point2D)
		: base(2, 3, 7, TModel.NumParams)
	{
		_point2D = point2D;
	}

	/// <inheritdoc/>
	public override bool Evaluate(
		ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
	{
		int numParams = TModel.NumParams;
		ReadOnlySpan<double> point3DInWorld = parameters[0].AsSpan(0, 3);
		ReadOnlySpan<double> camFromWorld = parameters[1].AsSpan(0, 7);
		ReadOnlySpan<double> cameraParams = parameters[2].AsSpan(0, numParams);

		Span<double> jPoint = AnalyticalReprojection.JacobianOrEmpty(jacobians, 0, 2 * 3);
		Span<double> jPose = AnalyticalReprojection.JacobianOrEmpty(jacobians, 1, 2 * 7);
		Span<double> jParams = AnalyticalReprojection.JacobianOrEmpty(jacobians, 2, 2 * numParams);

		Span<double> jRpQuat = stackalloc double[12];
		Span<double> jUvw = stackalloc double[6];

		Vector3d rotated = QuaternionUtils.QuaternionRotatePointWithJac(
			camFromWorld, point3DInWorld, jPose.IsEmpty ? default : jRpQuat);
		var pointInCam = new Vector3d(rotated.X + camFromWorld[4], rotated.Y + camFromWorld[5], rotated.Z + camFromWorld[6]);

		if (!TModel.ImgFromCamWithJac(
			cameraParams,
			pointInCam.X,
			pointInCam.Y,
			pointInCam.Z,
			out double x,
			out double y,
			jParams,
			(!jPoint.IsEmpty || !jPose.IsEmpty) ? jUvw : default))
		{
			residuals[..2].Clear();
			jPose.Clear();
			jPoint.Clear();
			jParams.Clear();
			return true;
		}

		residuals[0] = x - _point2D.X;
		residuals[1] = y - _point2D.Y;

		// No-op for non-periodic models. The offset is locally constant, so the analytic
		// Jacobians below are unaffected.
		ReprojectionErrors.WrapEquirectangularHorizontalSeam<TModel, Real>(Real.Cast(cameraParams), Real.CastWritable(residuals));

		if (!jPoint.IsEmpty)
		{
			Matrix3d r = new Quaterniond(camFromWorld[3], camFromWorld[0], camFromWorld[1], camFromWorld[2]).ToRotationMatrix();
			AnalyticalReprojection.MultiplyByRotation(jUvw, r, jPoint);
		}

		if (!jPose.IsEmpty)
		{
			for (int row = 0; row < 2; row++)
			{
				double a0 = jUvw[row * 3], a1 = jUvw[row * 3 + 1], a2 = jUvw[row * 3 + 2];
				for (int col = 0; col < 4; col++)
				{
					jPose[row * 7 + col] = a0 * jRpQuat[col] + a1 * jRpQuat[4 + col] + a2 * jRpQuat[8 + col];
				}

				jPose[row * 7 + 4] = a0;
				jPose[row * 7 + 5] = a1;
				jPose[row * 7 + 6] = a2;
			}
		}

		return true;
	}
}

/// <summary>
/// Reprojection error cost function with analytical Jacobians for a fixed camera pose
/// (variable point and camera calibration). Analytical counterpart of
/// <see cref="ReprojErrorConstantPoseCostFunctor{TModel}"/>. As in that functor, the fixed
/// pose is stored as a precomputed rotation matrix and translation; besides the faster
/// matrix-vector transform, the rotation matrix is reused directly for the point Jacobian,
/// avoiding a quaternion-to-matrix conversion on every evaluation. Blocks:
/// point3D_in_world (3), camera params.
/// Port of colmap::AnalyticalReprojErrorConstantPoseCostFunction.
/// </summary>
public sealed class AnalyticalReprojErrorConstantPoseCostFunction<TModel> : CostFunction
	where TModel : struct, ICameraModel<TModel>
{
	private readonly Vector2d _point2D;
	private readonly Matrix3d _camFromWorldRotation;
	private readonly Vector3d _camFromWorldTranslation;

	/// <summary>Creates the cost function for one observation and the fixed pose.</summary>
	public AnalyticalReprojErrorConstantPoseCostFunction(Vector2d point2D, Rigid3d camFromWorld)
		: base(2, 3, TModel.NumParams)
	{
		_point2D = point2D;
		_camFromWorldRotation = camFromWorld.Rotation.ToRotationMatrix();
		_camFromWorldTranslation = camFromWorld.Translation;
	}

	/// <inheritdoc/>
	public override bool Evaluate(
		ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
	{
		int numParams = TModel.NumParams;
		ReadOnlySpan<double> p = parameters[0].AsSpan(0, 3);
		ReadOnlySpan<double> cameraParams = parameters[1].AsSpan(0, numParams);

		Span<double> jPoint = AnalyticalReprojection.JacobianOrEmpty(jacobians, 0, 2 * 3);
		Span<double> jParams = AnalyticalReprojection.JacobianOrEmpty(jacobians, 1, 2 * numParams);
		Span<double> jUvw = stackalloc double[6];

		Matrix3d r = _camFromWorldRotation;
		Vector3d t = _camFromWorldTranslation;
		double u = r[0, 0] * p[0] + r[0, 1] * p[1] + r[0, 2] * p[2] + t.X;
		double v = r[1, 0] * p[0] + r[1, 1] * p[1] + r[1, 2] * p[2] + t.Y;
		double w = r[2, 0] * p[0] + r[2, 1] * p[1] + r[2, 2] * p[2] + t.Z;

		if (!TModel.ImgFromCamWithJac(cameraParams, u, v, w, out double x, out double y, jParams, jPoint.IsEmpty ? default : jUvw))
		{
			residuals[..2].Clear();
			jPoint.Clear();
			jParams.Clear();
			return true;
		}

		residuals[0] = x - _point2D.X;
		residuals[1] = y - _point2D.Y;

		// No-op for non-periodic models. The offset is locally constant, so the analytic
		// Jacobian below is unaffected.
		ReprojectionErrors.WrapEquirectangularHorizontalSeam<TModel, Real>(Real.Cast(cameraParams), Real.CastWritable(residuals));

		if (!jPoint.IsEmpty)
		{
			AnalyticalReprojection.MultiplyByRotation(jUvw, r, jPoint);
		}

		return true;
	}
}

/// <summary>Helpers shared by the analytic reprojection cost functions.</summary>
internal static class AnalyticalReprojection
{
	/// <summary>The requested Jacobian block as a span, or empty when it is not wanted.</summary>
	public static Span<double> JacobianOrEmpty(ReadOnlySpan<ArraySegment<double>> jacobians, int block, int size) =>
		jacobians.IsEmpty || jacobians[block].Array is null ? default : jacobians[block].AsSpan(0, size);

	/// <summary>Row-major 2x3 result = a (row-major 2x3) * r.</summary>
	public static void MultiplyByRotation(ReadOnlySpan<double> a, in Matrix3d r, Span<double> result)
	{
		for (int row = 0; row < 2; row++)
		{
			double a0 = a[row * 3], a1 = a[row * 3 + 1], a2 = a[row * 3 + 2];
			for (int col = 0; col < 3; col++)
			{
				result[row * 3 + col] = a0 * r[0, col] + a1 * r[1, col] + a2 * r[2, col];
			}
		}
	}
}
