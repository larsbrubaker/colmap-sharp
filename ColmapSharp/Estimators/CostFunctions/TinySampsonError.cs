// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TinySampsonError: the TinyFundamentalSampsonErrorCostFunctor of
// colmap/estimators/cost_functions/tiny_sampson_error.h - the Sampson-error cost of a
// fundamental matrix in Bartoli and Sturm's minimal SVD parameterization, with a closed-form
// Jacobian, for Optim/TinySolver.cs over the manifold of TinyManifold.cs. Its caller is
// RefineFundamentalMatrixSampson in Estimators/Solvers/FundamentalMatrixEstimators.cs.
// Tests: ColmapSharp.Tests/Estimators/CostFunctions/TinySampsonErrorTests.cs (the
// TinyFundamentalSampsonErrorCostFunctor cases of tiny_sampson_error_test.cc).
//
// The header's other three functors (TinyTangentSampsonErrorCostFunctor,
// TinyFocalSampsonErrorCostFunctor, TinyOneSidedFocalTangentSampsonErrorCostFunctor) serve
// the essential-matrix and focal-length refiners and build on sampson_error.h's Jet-templated
// SampsonError / TangentSampsonError / EssentialMatrixFromPoseParams; they are ported with
// those.
//
// Tier C in use (an iterative solver's cost); the residuals themselves are scalar formulas
// whose only Eigen dependencies are 3x3 products and a quaternion-to-matrix conversion.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>
/// Sampson-error cost functor for TinySolver refinement of a fundamental matrix,
/// parameterized by the SVD factorization of Bartoli and Sturm, "Non-Linear Estimation of
/// the Fundamental Matrix With Minimal Parameters", PAMI 2004:
/// F = U diag(1, sigma, 0) V^T = U0 V0^T + sigma U1 V1^T, where U = R(qU), V = R(qV) and
/// Uk, Vk are their k-th columns. The 9 ambient parameters are [qU (xyzw), qV (xyzw), sigma];
/// the solver applies the 7-DoF manifold, so rank 2 holds exactly at every iterate. Fixing
/// the largest singular value to 1 is free, as the Sampson error is scale invariant.
/// The residual r = num / sqrt(denom) is the signed Sampson error whose square
/// ComputeSquaredSampsonError scores. Port of colmap::TinyFundamentalSampsonErrorCostFunctor.
/// </summary>
/// <remarks>Holds the point arrays by reference, as the C++ holds its vectors.</remarks>
public readonly struct TinyFundamentalSampsonErrorCostFunctor(Vector2d[] points1, Vector2d[] points2)
	: ITinySolverFunction
{
	private readonly Vector2d[] _points1 = points1;
	private readonly Vector2d[] _points2 = points2;

	/// <summary>C++'s NUM_PARAMETERS.</summary>
	public static int NumParameters => 9;

	/// <inheritdoc/>
	public int NumResiduals => _points1.Length;

	/// <summary>The fundamental matrix implied by the 9 ambient parameters.</summary>
	public static Matrix3d FundamentalFromParams(ReadOnlySpan<double> parameters) =>
		FundamentalFromParams(parameters, out _, out _);

	/// <summary>
	/// The fundamental matrix implied by the 9 ambient parameters, with the two rotations,
	/// which only the Jacobian needs.
	/// </summary>
	public static Matrix3d FundamentalFromParams(ReadOnlySpan<double> parameters, out Matrix3d u, out Matrix3d v)
	{
		// Eigen::Map<const Quaterniond>: coefficients stored x, y, z, w.
		u = new Quaterniond(parameters[3], parameters[0], parameters[1], parameters[2]).ToRotationMatrix();
		v = new Quaterniond(parameters[7], parameters[4], parameters[5], parameters[6]).ToRotationMatrix();
		return Outer(u.Col(0), v.Col(0)) + parameters[8] * Outer(u.Col(1), v.Col(1));
	}

	/// <summary>
	/// Residuals and, unless <paramref name="jacobian"/> is empty, the column-major
	/// NumResiduals x 9 Jacobian.
	/// </summary>
	public bool Evaluate(ReadOnlySpan<double> parameters, Span<double> residuals, Span<double> jacobian)
	{
		double sigma = parameters[8];
		Matrix3d f = FundamentalFromParams(parameters, out Matrix3d u, out Matrix3d v);
		bool wantJacobian = !jacobian.IsEmpty;

		Span<Matrix3d> dF = stackalloc Matrix3d[9];
		if (wantJacobian)
		{
			Span<Vector3d> dU0 = stackalloc Vector3d[4];
			Span<Vector3d> dU1 = stackalloc Vector3d[4];
			Span<Vector3d> dV0 = stackalloc Vector3d[4];
			Span<Vector3d> dV1 = stackalloc Vector3d[4];
			RotationCol01Derivatives(parameters[..4], dU0, dU1);
			RotationCol01Derivatives(parameters.Slice(4, 4), dV0, dV1);
			for (int l = 0; l < 4; ++l)
			{
				dF[l] = Outer(dU0[l], v.Col(0)) + sigma * Outer(dU1[l], v.Col(1));
				dF[4 + l] = Outer(u.Col(0), dV0[l]) + sigma * Outer(u.Col(1), dV1[l]);
			}

			dF[8] = Outer(u.Col(1), v.Col(1));
		}

		Matrix3d fTranspose = f.Transpose();
		int n = _points1.Length;
		for (int i = 0; i < n; ++i)
		{
			Vector3d point1 = _points1[i].Homogeneous();
			Vector3d point2 = _points2[i].Homogeneous();
			Vector3d fPoint1 = f * point1;
			Vector3d ftPoint2 = fTranspose * point2;
			double num = point2.Dot(fPoint1);

			// Only the first two components of each constraint gradient enter, as
			// the homogeneous third coordinate is not a free variable.
			double denom = (fPoint1.X * fPoint1.X + fPoint1.Y * fPoint1.Y) +
				(ftPoint2.X * ftPoint2.X + ftPoint2.Y * ftPoint2.Y);
			double sqrtDenom = Math.Sqrt(denom);
			if (sqrtDenom == 0.0)
			{
				residuals[i] = 0.0;
				if (wantJacobian)
				{
					for (int l = 0; l < 9; ++l)
					{
						jacobian[i + l * n] = 0.0;
					}
				}

				continue;
			}

			residuals[i] = num / sqrtDenom;
			if (wantJacobian)
			{
				// dr/dF = (1/sqrt_denom) point2 point1^T
				//         - (num/denom^1.5) (g2 point1^T + point2 g1^T), where g1 and
				// g2 are F^T point2 and F point1 with the third component dropped.
				var g1 = new Vector3d(ftPoint2.X, ftPoint2.Y, 0.0);
				var g2 = new Vector3d(fPoint1.X, fPoint1.Y, 0.0);
				double coef = num / (denom * sqrtDenom);
				Matrix3d drdF = (1.0 / sqrtDenom) * Outer(point2, point1) -
					coef * (Outer(g2, point1) + Outer(point2, g1));
				for (int l = 0; l < 9; ++l)
				{
					jacobian[i + l * n] = CwiseProductSum(drdF, dF[l]);
				}
			}
		}

		return true;
	}

	/// <summary>
	/// Derivatives of the first two columns of R(q) w.r.t. the quaternion coefficients
	/// (x, y, z, w), assuming unit q. The factorization uses no other column.
	/// </summary>
	private static void RotationCol01Derivatives(ReadOnlySpan<double> q, Span<Vector3d> dcol0, Span<Vector3d> dcol1)
	{
		double x = q[0], y = q[1], z = q[2], w = q[3];
		dcol0[0] = new Vector3d(0, 2 * y, 2 * z);        // dR.col(0)/dqx
		dcol0[1] = new Vector3d(-4 * y, 2 * x, -2 * w);  // dR.col(0)/dqy
		dcol0[2] = new Vector3d(-4 * z, 2 * w, 2 * x);   // dR.col(0)/dqz
		dcol0[3] = new Vector3d(0, 2 * z, -2 * y);       // dR.col(0)/dqw
		dcol1[0] = new Vector3d(2 * y, -4 * x, 2 * w);   // dR.col(1)/dqx
		dcol1[1] = new Vector3d(2 * x, 0, 2 * z);        // dR.col(1)/dqy
		dcol1[2] = new Vector3d(-2 * w, -4 * z, 2 * y);  // dR.col(1)/dqz
		dcol1[3] = new Vector3d(-2 * z, 0, 2 * x);       // dR.col(1)/dqw
	}

	// a b^T.
	private static Matrix3d Outer(Vector3d a, Vector3d b) => new(
		a.X * b.X, a.X * b.Y, a.X * b.Z,
		a.Y * b.X, a.Y * b.Y, a.Y * b.Z,
		a.Z * b.X, a.Z * b.Y, a.Z * b.Z);

	// a.cwiseProduct(b).sum(), summed in column-major order.
	private static double CwiseProductSum(in Matrix3d a, in Matrix3d b)
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
