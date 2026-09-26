// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SampsonError: colmap/estimators/cost_functions/sampson_error.h - the scalar-generic
// (double or Jet, Solver/Scalar.cs) essential matrix of a relative pose, the signed Sampson
// and tangent Sampson errors, and the two Ceres autodiff cost functors built on them
// (SampsonErrorCostFunctor, TangentSampsonErrorCostFunctor, wrapped by
// Solver/AutoDiffCostFunction.cs). The TinySolver functors of TinyRelativePoseSampsonError.cs
// reuse the generic pieces; the relative pose refinement of estimators/pose.cc will use
// TangentSampsonErrorCostFunctor. Tests:
// ColmapSharp.Tests/Estimators/CostFunctions/SampsonErrorTests.cs (sampson_error_test.cc).
//
// Translation notes:
// - C++ templates the whole formula on T, but every COLMAP caller casts double data
//   (points, rays, unprojection Jacobians) to T; only E depends on the parameters. So the
//   measurement arguments stay double here and meet the T-valued E through IScalar's mixed
//   operators, which Ceres defines to give the same value and derivative as a constant Jet.
// - Eigen::Matrix<T, 3, 3> becomes Matrix3{T} below: a generic functor cannot stackalloc
//   a span of T (IAutoDiffFunctor only constrains T to struct), so the 3x3 is a value type.
// - The zero-denominator test compares the value part, as Ceres' Jet operator== does.
//
// Tier: scalar formulas (Tier A up to the grouping of Eigen's 3-term inner products; the
// tests pin them within COLMAP's tolerances).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>
/// A 3 x 3 matrix over a scalar type T (double via Real, or a Jet): the C# stand-in for
/// Eigen::Matrix&lt;T, 3, 3&gt; in the generic cost functions.
/// </summary>
public readonly struct Matrix3<T>(T m00, T m01, T m02, T m10, T m11, T m12, T m20, T m21, T m22)
	where T : struct, IScalar<T>
{
	/// <summary>Row 0.</summary>
	public readonly T M00 = m00, M01 = m01, M02 = m02;

	/// <summary>Row 1.</summary>
	public readonly T M10 = m10, M11 = m11, M12 = m12;

	/// <summary>Row 2.</summary>
	public readonly T M20 = m20, M21 = m21, M22 = m22;

	/// <summary>Entry (row, col).</summary>
	public T this[int row, int col] => (row, col) switch
	{
		(0, 0) => M00,
		(0, 1) => M01,
		(0, 2) => M02,
		(1, 0) => M10,
		(1, 1) => M11,
		(1, 2) => M12,
		(2, 0) => M20,
		(2, 1) => M21,
		(2, 2) => M22,
		_ => throw new ArgumentOutOfRangeException(nameof(row)),
	};

	/// <summary>The matrix product a b, each entry summed left to right.</summary>
	public static Matrix3<T> operator *(in Matrix3<T> a, in Matrix3<T> b) => new(
		a.M00 * b.M00 + a.M01 * b.M10 + a.M02 * b.M20,
		a.M00 * b.M01 + a.M01 * b.M11 + a.M02 * b.M21,
		a.M00 * b.M02 + a.M01 * b.M12 + a.M02 * b.M22,
		a.M10 * b.M00 + a.M11 * b.M10 + a.M12 * b.M20,
		a.M10 * b.M01 + a.M11 * b.M11 + a.M12 * b.M21,
		a.M10 * b.M02 + a.M11 * b.M12 + a.M12 * b.M22,
		a.M20 * b.M00 + a.M21 * b.M10 + a.M22 * b.M20,
		a.M20 * b.M01 + a.M21 * b.M11 + a.M22 * b.M21,
		a.M20 * b.M02 + a.M21 * b.M12 + a.M22 * b.M22);
}

/// <summary>The scalar-generic free functions of sampson_error.h.</summary>
public static class SampsonErrors
{
	/// <summary>
	/// Builds the essential matrix E = [t]_x R from a relative pose given in the Rigid3d
	/// parameter layout [qx, qy, qz, qw, tx, ty, tz]. Templated on the scalar so it works
	/// under autodiff. Port of colmap::EssentialMatrixFromPoseParams.
	/// </summary>
	public static Matrix3<T> EssentialMatrixFromPoseParams<T>(ReadOnlySpan<T> cam2FromCam1)
		where T : struct, IScalar<T>
	{
		Matrix3<T> r = QuaternionToRotationMatrix(cam2FromCam1[0], cam2FromCam1[1], cam2FromCam1[2], cam2FromCam1[3]);

		// Matrix representation of the cross product t x R.
		T zero = T.FromDouble(0);
		var tX = new Matrix3<T>(
			zero, -cam2FromCam1[6], cam2FromCam1[5],
			cam2FromCam1[6], zero, -cam2FromCam1[4],
			-cam2FromCam1[5], cam2FromCam1[4], zero);
		return tX * r;
	}

	/// <summary>
	/// Signed Sampson error under an essential/fundamental matrix. point1/point2 are points
	/// on the image plane, taken as 2D because the error is not invariant to the scale of the
	/// homogeneous representative, so only (x, y, 1) is correct here. For rays use
	/// <see cref="TangentSampsonError{T}"/>. Returns 0 when the denominator vanishes.
	/// Port of colmap::SampsonError.
	/// </summary>
	public static T SampsonError<T>(in Matrix3<T> e, Vector2d point1, Vector2d point2)
		where T : struct, IScalar<T>
	{
		// E * point1.homogeneous() and point2.homogeneous() dotted with E's columns.
		T line0 = e.M00 * point1.X + e.M01 * point1.Y + e.M02;
		T line1 = e.M10 * point1.X + e.M11 * point1.Y + e.M12;
		T line2 = e.M20 * point1.X + e.M21 * point1.Y + e.M22;
		T num = point2.X * line0 + point2.Y * line1 + line2;
		T denom0 = point2.X * e.M00 + point2.Y * e.M10 + e.M20;
		T denom1 = point2.X * e.M01 + point2.Y * e.M11 + e.M21;
		T denomNorm = T.Sqrt(denom0 * denom0 + denom1 * denom1 + line0 * line0 + line1 * line1);
		if (T.ScalarPart(denomNorm) == 0)
		{
			return T.FromDouble(0);
		}

		return num / denomNorm;
	}

	/// <summary>
	/// Signed tangent Sampson error of one correspondence under E, in pixels, using the
	/// unprojection Jacobians J_ray1 = d(ray1)/d(pixel1) and J_ray2. Returns 0 when the
	/// denominator vanishes. See EssentialMatrix.ComputeSquaredTangentSampsonError.
	/// Port of colmap::TangentSampsonError.
	/// </summary>
	public static T TangentSampsonError<T>(in Matrix3<T> e, Vector3d camRay1, in Matrix3x2d jRay1, Vector3d camRay2, in Matrix3x2d jRay2)
		where T : struct, IScalar<T>
	{
		T eRay1X = e.M00 * camRay1.X + e.M01 * camRay1.Y + e.M02 * camRay1.Z;
		T eRay1Y = e.M10 * camRay1.X + e.M11 * camRay1.Y + e.M12 * camRay1.Z;
		T eRay1Z = e.M20 * camRay1.X + e.M21 * camRay1.Y + e.M22 * camRay1.Z;
		T etRay2X = e.M00 * camRay2.X + e.M10 * camRay2.Y + e.M20 * camRay2.Z;
		T etRay2Y = e.M01 * camRay2.X + e.M11 * camRay2.Y + e.M21 * camRay2.Z;
		T etRay2Z = e.M02 * camRay2.X + e.M12 * camRay2.Y + e.M22 * camRay2.Z;
		T num = camRay2.X * eRay1X + camRay2.Y * eRay1Y + camRay2.Z * eRay1Z;

		// [J_ray1^T E^T ray2; J_ray2^T E ray1].
		T d0 = jRay1[0, 0] * etRay2X + jRay1[1, 0] * etRay2Y + jRay1[2, 0] * etRay2Z;
		T d1 = jRay1[0, 1] * etRay2X + jRay1[1, 1] * etRay2Y + jRay1[2, 1] * etRay2Z;
		T d2 = jRay2[0, 0] * eRay1X + jRay2[1, 0] * eRay1Y + jRay2[2, 0] * eRay1Z;
		T d3 = jRay2[0, 1] * eRay1X + jRay2[1, 1] * eRay1Y + jRay2[2, 1] * eRay1Z;
		T denomNorm = T.Sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3);
		if (T.ScalarPart(denomNorm) == 0)
		{
			return T.FromDouble(0);
		}

		return num / denomNorm;
	}

	/// <summary>
	/// R(q) of the quaternion with Eigen coefficients (x, y, z, w), Eigen's
	/// toRotationMatrix, with the same grouping as Quaterniond.ToRotationMatrix.
	/// </summary>
	public static Matrix3<T> QuaternionToRotationMatrix<T>(T x, T y, T z, T w)
		where T : struct, IScalar<T>
	{
		T x2 = 2.0 * x, y2 = 2.0 * y, z2 = 2.0 * z;
		T xx = x2 * x, yy = y2 * y, zz = z2 * z;
		T xy = x2 * y, xz = x2 * z, yz = y2 * z;
		T wx = x2 * w, wy = y2 * w, wz = z2 * w;
		return new Matrix3<T>(
			1.0 - (yy + zz), xy - wz, xz + wy,
			xy + wz, 1.0 - (xx + zz), yz - wx,
			xz - wy, yz + wx, 1.0 - (xx + yy));
	}
}

/// <summary>
/// Refines a relative pose by the Sampson error of image-plane point correspondences. See
/// <see cref="SampsonErrors.SampsonError{T}"/>. The pose is [qx, qy, qz, qw, tx, ty, tz]
/// with the translation on the unit sphere, so it needs a sphere manifold on tvec. For
/// calibrated rays with unprojection Jacobians use <see cref="TangentSampsonErrorCostFunctor"/>,
/// which is pixel-accurate for any central model. Port of colmap::SampsonErrorCostFunctor.
/// </summary>
public readonly struct SampsonErrorCostFunctor(Vector2d point1, Vector2d point2) : IAutoDiffFunctor
{
	private readonly Vector2d _point1 = point1;
	private readonly Vector2d _point2 = point2;

	/// <summary>The autodiff cost function: 1 residual, one 7-parameter block.</summary>
	public static AutoDiffCostFunction<SampsonErrorCostFunctor, Grad7> Create(Vector2d point1, Vector2d point2) =>
		new(new SampsonErrorCostFunctor(point1, point2), 1, 7);

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		Matrix3<T> e = SampsonErrors.EssentialMatrixFromPoseParams(parameters);
		residuals[0] = SampsonErrors.SampsonError(e, _point1, _point2);
		return true;
	}
}

/// <summary>
/// Refines a relative pose by the pixel-unit tangent Sampson error of calibrated ray
/// correspondences with unprojection Jacobians. See
/// <see cref="SampsonErrors.TangentSampsonError{T}"/>. Pose layout matches
/// <see cref="SampsonErrorCostFunctor"/>. Pixel-accurate for any central model.
/// Port of colmap::TangentSampsonErrorCostFunctor.
/// </summary>
public readonly struct TangentSampsonErrorCostFunctor(CamRayWithJac camRay1WithJac, CamRayWithJac camRay2WithJac)
	: IAutoDiffFunctor
{
	private readonly CamRayWithJac _camRay1WithJac = camRay1WithJac;
	private readonly CamRayWithJac _camRay2WithJac = camRay2WithJac;

	/// <summary>The autodiff cost function: 1 residual, one 7-parameter block.</summary>
	public static AutoDiffCostFunction<TangentSampsonErrorCostFunctor, Grad7> Create(
		CamRayWithJac camRay1WithJac, CamRayWithJac camRay2WithJac) =>
		new(new TangentSampsonErrorCostFunctor(camRay1WithJac, camRay2WithJac), 1, 7);

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		Matrix3<T> e = SampsonErrors.EssentialMatrixFromPoseParams(parameters);
		residuals[0] = SampsonErrors.TangentSampsonError(
			e, _camRay1WithJac.Ray, _camRay1WithJac.Jacobian, _camRay2WithJac.Ray, _camRay2WithJac.Jacobian);
		return true;
	}
}
