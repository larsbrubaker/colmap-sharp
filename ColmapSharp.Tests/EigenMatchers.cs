// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// EigenMatchers: the EigenMatrixNear matcher of colmap/util/eigen_matchers.h, which the
// ported geometry tests use as EXPECT_THAT(a, EigenMatrixNear(b, tol)). Test
// infrastructure shared by every test file that ports such a check (first users:
// Geometry/Rigid3dTests.cs, Sim3dTests.cs, GpsTests.cs, BboxTests.cs, Scene/FrameTests.cs; the VectorXd overload
// serves Optim/SparseCholeskyTests.cs and LeastAbsoluteDeviationsTests.cs).
//
// Semantics, as in COLMAP: if rhs is zero (Eigen's isZero(), every coefficient within
// dummy_precision = 1e-12 of 0), lhs matches when lhs.norm() <= tol, because isApprox is
// not well-defined against zero; otherwise lhs.isApprox(rhs, tol), a relative test
// ||lhs - rhs|| <= tol * min(||lhs||, ||rhs||).

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Tests;

/// <summary>
/// Port of COLMAP's EigenMatrixNear matcher for the LinearAlgebra types.
/// </summary>
internal static class EigenMatchers
{
	/// <summary>EigenMatrixNear(rhs, tol) applied to lhs.</summary>
	public static bool EigenMatrixNear(Vector3d lhs, Vector3d rhs, double tol = LinearAlgebraConstants.DummyPrecision)
	{
		if (IsZero([rhs.X, rhs.Y, rhs.Z]))
		{
			return lhs.Norm <= tol;
		}

		return lhs.IsApprox(rhs, tol);
	}

	/// <summary>EigenMatrixNear(rhs, tol) applied to lhs (e.g. quaternion coeffs()).</summary>
	public static bool EigenMatrixNear(Vector4d lhs, Vector4d rhs, double tol = LinearAlgebraConstants.DummyPrecision)
	{
		if (IsZero([rhs.X, rhs.Y, rhs.Z, rhs.W]))
		{
			return lhs.Norm <= tol;
		}

		return lhs.IsApprox(rhs, tol);
	}

	/// <summary>EigenMatrixNear(rhs, tol) applied to lhs.</summary>
	public static bool EigenMatrixNear(Matrix6d lhs, Matrix6d rhs, double tol = LinearAlgebraConstants.DummyPrecision)
	{
		Span<double> coefficients = stackalloc double[36];
		rhs.CopyToColumnMajor(coefficients);
		if (IsZero(coefficients))
		{
			return lhs.Norm() <= tol;
		}

		return lhs.IsApprox(rhs, tol);
	}

	/// <summary>EigenMatrixNear(rhs, tol) applied to lhs; a length mismatch never matches.</summary>
	public static bool EigenMatrixNear(VectorXd lhs, VectorXd rhs, double tol = LinearAlgebraConstants.DummyPrecision)
	{
		if (lhs.Length != rhs.Length)
		{
			return false;
		}

		if (IsZero(rhs.AsSpan()))
		{
			return lhs.Norm() <= tol;
		}

		return lhs.IsApprox(rhs, tol);
	}

	// Eigen's isZero() with the default precision: every |coefficient| <= 1e-12 * 1.
	private static bool IsZero(ReadOnlySpan<double> coefficients)
	{
		foreach (double value in coefficients)
		{
			if (!(Math.Abs(value) <= LinearAlgebraConstants.DummyPrecision))
			{
				return false;
			}
		}

		return true;
	}
}
