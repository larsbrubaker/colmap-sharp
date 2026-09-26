// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TinyRelativePoseSampsonError: the three relative-pose functors of
// colmap/estimators/cost_functions/tiny_sampson_error.h (the fourth, the fundamental matrix
// functor, is TinySampsonError.cs):
// - TinyTangentSampsonErrorCostFunctor: pixel-space tangent Sampson error of a calibrated
//   relative pose, closed-form Jacobian. Refiner of EssentialMatrixTangentSampsonEstimator
//   (Estimators/Solvers/EssentialMatrixEstimators.cs).
// - TinyFocalSampsonErrorCostFunctor: pose plus a shared log-focal, autodiff only (through
//   Solver/TinySolverAutoDiffFunction.cs). Refiner of relpose_shared_focal.
// - TinyOneSidedFocalTangentSampsonErrorCostFunctor: pose plus the first view's log-focal,
//   closed-form Jacobian and an autodiff form that pins it. Refiner of
//   relpose_one_sided_focal.
// All build E = [t]_x R from the Rigid3d parameter layout [qx, qy, qz, qw, tx, ty, tz]; the
// generic (autodiff) forms reuse SampsonError.cs. Tests:
// ColmapSharp.Tests/Estimators/CostFunctions/TinySampsonErrorTests.cs
// (tiny_sampson_error_test.cc).
//
// Tier C in use (an iterative solver's cost); the residuals are scalar formulas.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>
/// Tangent Sampson (pixel-space) cost functor for TinySolver refinement of a two-view
/// relative pose from calibrated rays with unprojection Jacobians. E = [t]_x R is built from
/// the pose [qx, qy, qz, qw, tx, ty, tz] (Rigid3d params). The solver applies the manifold.
/// Pixel-accurate for any central model, matching EssentialMatrixTangentSampsonEstimator's
/// score. The residual is r = num / sqrt(denom). The 7-parameter Jacobian dr/d[q,t] is
/// computed in closed form (dr/dE contracted with dE/d[q,t]), ~3x faster than autodiff
/// here. Port of colmap::TinyTangentSampsonErrorCostFunctor.
/// </summary>
/// <remarks>Holds the arrays by reference, as the C++ holds its vectors.</remarks>
public readonly struct TinyTangentSampsonErrorCostFunctor(CamRayWithJac[] camRays1WithJac, CamRayWithJac[] camRays2WithJac)
	: ITinySolverFunction
{
	private readonly CamRayWithJac[] _camRays1WithJac = camRays1WithJac;
	private readonly CamRayWithJac[] _camRays2WithJac = camRays2WithJac;

	/// <summary>C++'s NUM_PARAMETERS.</summary>
	public static int NumParameters => 7;

	/// <inheritdoc/>
	public int NumResiduals => _camRays1WithJac.Length;

	/// <summary>
	/// Residuals and, unless <paramref name="jacobian"/> is empty, the column-major
	/// NumResiduals x 7 Jacobian.
	/// </summary>
	public bool Evaluate(ReadOnlySpan<double> parameters, Span<double> residuals, Span<double> jacobian)
	{
		Matrix3d r = RelativePoseJacobians.RotationFromParams(parameters);
		Matrix3d tX = RelativePoseJacobians.CrossFromParams(parameters);
		Matrix3d e = tX * r;
		bool wantJacobian = !jacobian.IsEmpty;

		Span<Matrix3d> dE = stackalloc Matrix3d[7];
		if (wantJacobian)
		{
			RelativePoseJacobians.EssentialDerivatives(parameters, tX, r, dE);
		}

		Matrix3d eTranspose = e.Transpose();
		int n = _camRays1WithJac.Length;
		for (int i = 0; i < n; ++i)
		{
			Vector3d ray1 = _camRays1WithJac[i].Ray;
			Vector3d ray2 = _camRays2WithJac[i].Ray;
			Matrix3x2d j1 = _camRays1WithJac[i].Jacobian;
			Matrix3x2d j2 = _camRays2WithJac[i].Jacobian;
			Vector3d eRay1 = e * ray1;
			Vector3d etRay2 = eTranspose * ray2;
			double num = ray2.Dot(eRay1);
			Vector2d a = RelativePoseJacobians.TransposeTimes(j1, etRay2);
			Vector2d b = RelativePoseJacobians.TransposeTimes(j2, eRay1);
			double denom = (a.X * a.X + a.Y * a.Y) + (b.X * b.X + b.Y * b.Y);
			double sqrtDenom = Math.Sqrt(denom);
			if (sqrtDenom == 0.0)
			{
				residuals[i] = 0.0;
				if (wantJacobian)
				{
					for (int l = 0; l < 7; ++l)
					{
						jacobian[i + l * n] = 0.0;
					}
				}

				continue;
			}

			residuals[i] = num / sqrtDenom;
			if (wantJacobian)
			{
				// dr/dE = (1/sqrt_denom) ray2 ray1^T
				//         - (num/denom^1.5) (ray2 (J1 a)^T + (J2 b) ray1^T).
				Vector3d j1a = RelativePoseJacobians.Times(j1, a);
				Vector3d j2b = RelativePoseJacobians.Times(j2, b);
				double coef = num / (denom * sqrtDenom);
				Matrix3d drdE = (1.0 / sqrtDenom) * RelativePoseJacobians.Outer(ray2, ray1) -
					coef * (RelativePoseJacobians.Outer(ray2, j1a) + RelativePoseJacobians.Outer(j2b, ray1));
				for (int l = 0; l < 7; ++l)
				{
					jacobian[i + l * n] = RelativePoseJacobians.CwiseProductSum(drdE, dE[l]);
				}
			}
		}

		return true;
	}
}

/// <summary>
/// Sampson-error cost functor for TinySolver refinement of a two-view relative pose
/// <i>and</i> a shared, unknown focal length. The pose is [qx, qy, qz, qw, tx, ty, tz] and
/// the focal is appended as an eighth parameter optimized in log-space (log_f), which keeps
/// it strictly positive and gives a scale-invariant step. The essential matrix is built from
/// the pose, then converted to the fundamental matrix implied by the focal so the Sampson
/// error is measured in <i>pixel</i> space:
/// F = diag(1/f, 1/f, 1) * E * diag(1/f, 1/f, 1), f = exp(log_f).
/// The inputs are therefore principal-point-centered image points (u - cx, v - cy), not
/// calibrated rays. The 6-DoF manifold (rotation on SO(3), translation on the unit sphere,
/// log-focal) is applied by the solver. Differentiated by
/// <see cref="TinySolverAutoDiffFunction{TFunctor, TGrad}"/> with Grad8 (C++'s
/// AutoDiffFunction typedef; see <see cref="CreateAutoDiffFunction"/>).
/// Port of colmap::TinyFocalSampsonErrorCostFunctor.
/// </summary>
/// <remarks>Holds the point arrays by reference, as the C++ holds its vectors.</remarks>
public readonly struct TinyFocalSampsonErrorCostFunctor(Vector2d[] points1, Vector2d[] points2) : IAutoDiffFunctor
{
	private readonly Vector2d[] _points1 = points1;
	private readonly Vector2d[] _points2 = points2;

	/// <summary>C++'s NUM_PARAMETERS.</summary>
	public static int NumParameters => 8;

	/// <summary>The number of residuals, one per correspondence.</summary>
	public int NumResiduals => _points1.Length;

	/// <summary>C++'s AutoDiffFunction wrapper, ready for TinySolver.</summary>
	public TinySolverAutoDiffFunction<TinyFocalSampsonErrorCostFunctor, Grad8> CreateAutoDiffFunction() =>
		new(this, NumResiduals);

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		// Build E once from the pose, then scale it into the pixel-space fundamental matrix
		// F = diag(inv_f, inv_f, 1) * E * diag(inv_f, inv_f, 1).
		Matrix3<T> e = SampsonErrors.EssentialMatrixFromPoseParams(parameters);
		T invF = T.Exp(-parameters[7]);
		T invF2 = invF * invF;
		var f = new Matrix3<T>(
			e.M00 * invF2, e.M01 * invF2, e.M02 * invF,
			e.M10 * invF2, e.M11 * invF2, e.M12 * invF,
			e.M20 * invF, e.M21 * invF, e.M22);
		for (int i = 0; i < _points1.Length; ++i)
		{
			residuals[i] = SampsonErrors.SampsonError(f, _points1[i], _points2[i]);
		}

		return true;
	}
}

/// <summary>
/// Tangent Sampson (pixel-space) cost functor for TinySolver refinement of a two-view
/// relative pose and a <i>single</i> unknown focal length, where the second view is already
/// calibrated (the semi-calibrated, or "one-sided focal", configuration). The pose is
/// [qx, qy, qz, qw, tx, ty, tz] and the unknown focal of the <i>first</i> view is appended as
/// an eighth parameter optimized in log-space. Following ray2^T E b1 = 0 with
/// b1 = K1inv * point1, the mixed epipolar matrix is M = E * diag(1/f1, 1/f1, 1), i.e. only
/// columns 0-1 are scaled, and ray2^T M point1 = 0. The residual is the tangent Sampson
/// error of that constraint, in pixels:
/// C = ray2^T M point1, dC/dpx1 = (M^T ray2).head&lt;2&gt;(), dC/dpx2 = J2^T (M point1),
/// r = C / sqrt(||dC/dpx1||^2 + ||dC/dpx2||^2).
/// Both denominator terms are gradients of the same scalar with respect to pixels, so they
/// share units and combine directly. J2 = d(ray2)/d(pixel2) represents any central model
/// exactly. img_points1 are principal-point-centered points of the uncalibrated view; keeping
/// them as raw pixels makes the measurement Jacobian d(x, y, 1)/d(x, y) the constant [I2; 0],
/// so f1 enters only through M (unprojecting to a bearing would need the mixed second
/// derivative d^2(ray1)/d(pixel1)d(f1), which no camera model exposes). For the same reason
/// the Jacobians hold only while the second view's intrinsics are fixed, as they are here.
/// The 8-parameter Jacobian is computed in closed form, dr/dM contracted with
/// dM/d[q, t, log_f1]; the generic form (<see cref="Evaluate{T}"/>, through
/// <see cref="CreateAutoDiffFunction"/>) only serves to pin it in tests.
/// Port of colmap::TinyOneSidedFocalTangentSampsonErrorCostFunctor.
/// </summary>
/// <remarks>Holds the arrays by reference, as the C++ holds its vectors.</remarks>
public readonly struct TinyOneSidedFocalTangentSampsonErrorCostFunctor(Vector2d[] imgPoints1, CamRayWithJac[] camRays2WithJac)
	: ITinySolverFunction, IAutoDiffFunctor
{
	// Measurement Jacobian d(x, y, 1) / d(x, y) of the uncalibrated view.
	private static readonly Matrix3x2d J1 = new(1, 0, 0, 1, 0, 0);

	private readonly Vector2d[] _imgPoints1 = imgPoints1;
	private readonly CamRayWithJac[] _camRays2WithJac = camRays2WithJac;

	/// <summary>C++'s NUM_PARAMETERS.</summary>
	public static int NumParameters => 8;

	/// <inheritdoc/>
	public int NumResiduals => _imgPoints1.Length;

	/// <summary>C++'s AutoDiffFunction wrapper.</summary>
	public TinySolverAutoDiffFunction<TinyOneSidedFocalTangentSampsonErrorCostFunctor, Grad8> CreateAutoDiffFunction() =>
		new(this, NumResiduals);

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		// Build E once from the pose, then apply the unknown focal to the columns to obtain
		// the matrix relating view-1 image points to view-2 rays.
		Matrix3<T> e = SampsonErrors.EssentialMatrixFromPoseParams(parameters);
		T invF1 = T.Exp(-parameters[7]);
		var m = new Matrix3<T>(
			e.M00 * invF1, e.M01 * invF1, e.M02,
			e.M10 * invF1, e.M11 * invF1, e.M12,
			e.M20 * invF1, e.M21 * invF1, e.M22);
		for (int i = 0; i < _imgPoints1.Length; ++i)
		{
			residuals[i] = SampsonErrors.TangentSampsonError(
				m, _imgPoints1[i].Homogeneous(), J1, _camRays2WithJac[i].Ray, _camRays2WithJac[i].Jacobian);
		}

		return true;
	}

	/// <summary>
	/// Residuals and, unless <paramref name="jacobian"/> is empty, the column-major
	/// NumResiduals x 8 Jacobian.
	/// </summary>
	public bool Evaluate(ReadOnlySpan<double> parameters, Span<double> residuals, Span<double> jacobian)
	{
		Matrix3d r = RelativePoseJacobians.RotationFromParams(parameters);
		Matrix3d tX = RelativePoseJacobians.CrossFromParams(parameters);
		double invF1 = Math.Exp(-parameters[7]);
		Matrix3d m = ScaleLeftCols(tX * r, invF1);
		bool wantJacobian = !jacobian.IsEmpty;

		Span<Matrix3d> dM = stackalloc Matrix3d[8];
		if (wantJacobian)
		{
			RelativePoseJacobians.EssentialDerivatives(parameters, tX, r, dM);

			// The pose derivatives are of E, so carry them through the same column scaling
			// that turns E into M.
			for (int l = 0; l < 7; ++l)
			{
				dM[l] = ScaleLeftCols(dM[l], invF1);
			}

			// d(E * diag(1/f1, 1/f1, 1)) / d(log_f1) negates the scaled columns and leaves
			// the third one, which carries no focal, unchanged.
			dM[7] = new Matrix3d(-m[0, 0], -m[0, 1], 0, -m[1, 0], -m[1, 1], 0, -m[2, 0], -m[2, 1], 0);
		}

		Matrix3d mTranspose = m.Transpose();
		int n = _imgPoints1.Length;
		for (int i = 0; i < n; ++i)
		{
			Vector3d point1 = _imgPoints1[i].Homogeneous();
			Vector3d ray2 = _camRays2WithJac[i].Ray;
			Matrix3x2d j2 = _camRays2WithJac[i].Jacobian;
			Vector3d mPoint1 = m * point1;
			double num = ray2.Dot(mPoint1);

			// Constraint gradients in view-1 and view-2 pixels. The former needs no Jacobian,
			// as d(x, y, 1)/d(x, y) merely selects the first two rows.
			Vector3d mtRay2 = mTranspose * ray2;
			var g1 = new Vector2d(mtRay2.X, mtRay2.Y);
			Vector2d g2 = RelativePoseJacobians.TransposeTimes(j2, mPoint1);
			double denom = (g1.X * g1.X + g1.Y * g1.Y) + (g2.X * g2.X + g2.Y * g2.Y);
			double sqrtDenom = Math.Sqrt(denom);
			if (sqrtDenom == 0.0)
			{
				residuals[i] = 0.0;
				if (wantJacobian)
				{
					for (int l = 0; l < 8; ++l)
					{
						jacobian[i + l * n] = 0.0;
					}
				}

				continue;
			}

			residuals[i] = num / sqrtDenom;
			if (wantJacobian)
			{
				// dr/dM = (1/sqrt_denom) ray2 point1^T
				//         - (num/denom^1.5) (ray2 [g1; 0]^T + (J2 g2) point1^T).
				var j1g1 = new Vector3d(g1.X, g1.Y, 0.0);
				Vector3d j2g2 = RelativePoseJacobians.Times(j2, g2);
				double coef = num / (denom * sqrtDenom);
				Matrix3d drdM = (1.0 / sqrtDenom) * RelativePoseJacobians.Outer(ray2, point1) -
					coef * (RelativePoseJacobians.Outer(ray2, j1g1) + RelativePoseJacobians.Outer(j2g2, point1));
				for (int l = 0; l < 8; ++l)
				{
					jacobian[i + l * n] = RelativePoseJacobians.CwiseProductSum(drdM, dM[l]);
				}
			}
		}

		return true;
	}

	// M.leftCols<2>() *= s.
	private static Matrix3d ScaleLeftCols(in Matrix3d a, double s) => new(
		a[0, 0] * s, a[0, 1] * s, a[0, 2],
		a[1, 0] * s, a[1, 1] * s, a[1, 2],
		a[2, 0] * s, a[2, 1] * s, a[2, 2]);
}

/// <summary>
/// The closed-form pieces the analytic tiny Sampson functors share: R(q), [t]_x, dE/d[q, t]
/// and the small products Eigen spells inline.
/// </summary>
internal static class RelativePoseJacobians
{
	/// <summary>R of the Eigen-order quaternion parameters[0..4] (Eigen::Map&lt;const Quaterniond&gt;).</summary>
	public static Matrix3d RotationFromParams(ReadOnlySpan<double> parameters) =>
		new Quaterniond(parameters[3], parameters[0], parameters[1], parameters[2]).ToRotationMatrix();

	/// <summary>[t]_x of the translation parameters[4..7].</summary>
	public static Matrix3d CrossFromParams(ReadOnlySpan<double> parameters) => new(
		0, -parameters[6], parameters[5],
		parameters[6], 0, -parameters[4],
		-parameters[5], parameters[4], 0);

	/// <summary>
	/// dE/d[qx, qy, qz, qw, tx, ty, tz] of E = [t]_x R(q) into the first 7 entries of
	/// <paramref name="dE"/>, assuming unit q.
	/// </summary>
	public static void EssentialDerivatives(ReadOnlySpan<double> parameters, in Matrix3d tX, in Matrix3d r, Span<Matrix3d> dE)
	{
		double x = parameters[0], y = parameters[1], z = parameters[2], w = parameters[3];
		dE[0] = tX * new Matrix3d(0, 2 * y, 2 * z, 2 * y, -4 * x, -2 * w, 2 * z, 2 * w, -4 * x);   // dR/dqx
		dE[1] = tX * new Matrix3d(-4 * y, 2 * x, 2 * w, 2 * x, 0, 2 * z, -2 * w, 2 * z, -4 * y);  // dR/dqy
		dE[2] = tX * new Matrix3d(-4 * z, -2 * w, 2 * x, 2 * w, -4 * z, 2 * y, 2 * x, 2 * y, 0);  // dR/dqz
		dE[3] = tX * new Matrix3d(0, -2 * z, 2 * y, 2 * z, 0, -2 * x, -2 * y, 2 * x, 0);          // dR/dqw
		dE[4] = new Matrix3d(0, 0, 0, 0, 0, -1, 0, 1, 0) * r;  // dE/dtx
		dE[5] = new Matrix3d(0, 0, 1, 0, 0, 0, -1, 0, 0) * r;  // dE/dty
		dE[6] = new Matrix3d(0, -1, 0, 1, 0, 0, 0, 0, 0) * r;  // dE/dtz
	}

	/// <summary>J^T v for a 3x2 J.</summary>
	public static Vector2d TransposeTimes(in Matrix3x2d j, Vector3d v) => new(
		j[0, 0] * v.X + j[1, 0] * v.Y + j[2, 0] * v.Z,
		j[0, 1] * v.X + j[1, 1] * v.Y + j[2, 1] * v.Z);

	/// <summary>J v for a 3x2 J.</summary>
	public static Vector3d Times(in Matrix3x2d j, Vector2d v) => new(
		j[0, 0] * v.X + j[0, 1] * v.Y,
		j[1, 0] * v.X + j[1, 1] * v.Y,
		j[2, 0] * v.X + j[2, 1] * v.Y);

	/// <summary>a b^T.</summary>
	public static Matrix3d Outer(Vector3d a, Vector3d b) => new(
		a.X * b.X, a.X * b.Y, a.X * b.Z,
		a.Y * b.X, a.Y * b.Y, a.Y * b.Z,
		a.Z * b.X, a.Z * b.Y, a.Z * b.Z);

	/// <summary>a.cwiseProduct(b).sum(), summed in column-major order.</summary>
	public static double CwiseProductSum(in Matrix3d a, in Matrix3d b)
	{
		double sum = a[0, 0] * b[0, 0];
		for (int k = 1; k < 9; ++k)
		{
			int row = k % 3;
			int col = k / 3;
			sum += a[row, col] * b[row, col];
		}

		return sum;
	}
}
