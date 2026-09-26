// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// EigenMatchers: the EigenMatrixNear matcher of colmap/util/eigen_matchers.h, which the
// ported geometry tests use as EXPECT_THAT(a, EigenMatrixNear(b, tol)). Test
// infrastructure shared by every test file that ports such a check (first users:
// Geometry/Rigid3dTests.cs, Sim3dTests.cs, GpsTests.cs, BboxTests.cs, Scene/FrameTests.cs, and
// the pose, essential/homography matrix, triangulation and normalization tests; the VectorXd overload
// serves Optim/SparseCholeskyTests.cs and LeastAbsoluteDeviationsTests.cs; the MatrixXd
// overload Estimators/CostFunctions/TinyManifoldTests.cs and QuaternionUtilsTests.cs; the float
// RowMajorMatrix overload Controllers/PairingTests.cs).
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
	public static bool EigenMatrixNear(Vector2d lhs, Vector2d rhs, double tol = LinearAlgebraConstants.DummyPrecision)
	{
		if (IsZero([rhs.X, rhs.Y]))
		{
			return lhs.Norm <= tol;
		}

		return lhs.IsApprox(rhs, tol);
	}

	/// <summary>EigenMatrixNear(rhs, tol) applied to lhs.</summary>
	public static bool EigenMatrixNear(Matrix3d lhs, Matrix3d rhs, double tol = LinearAlgebraConstants.DummyPrecision)
	{
		Span<double> coefficients = stackalloc double[9];
		rhs.CopyToColumnMajor(coefficients);
		if (IsZero(coefficients))
		{
			return lhs.Norm() <= tol;
		}

		return lhs.IsApprox(rhs, tol);
	}

	/// <summary>EigenMatrixNear(rhs, tol) applied to lhs (Solvers/AffineTransformTests).</summary>
	public static bool EigenMatrixNear(Matrix2x3d lhs, Matrix2x3d rhs, double tol = LinearAlgebraConstants.DummyPrecision)
	{
		if (IsZero([rhs[0, 0], rhs[1, 0], rhs[0, 1], rhs[1, 1], rhs[0, 2], rhs[1, 2]]))
		{
			return lhs.Norm() <= tol;
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

	/// <summary>EigenMatrixNear(rhs, tol) applied to lhs; a shape mismatch never matches.</summary>
	public static bool EigenMatrixNear(MatrixXd lhs, MatrixXd rhs, double tol = LinearAlgebraConstants.DummyPrecision)
	{
		if (lhs.Rows != rhs.Rows || lhs.Cols != rhs.Cols)
		{
			return false;
		}

		if (IsZero(rhs.AsSpan()))
		{
			return lhs.Norm() <= tol;
		}

		return lhs.IsApprox(rhs, tol);
	}

	/// <summary>
	/// EigenMatrixNear(rhs, tol) for a float Eigen::RowMajorMatrixXf (first user:
	/// Controllers/PairingTests.cs). The default tol is Eigen's float dummy_precision, 1e-5,
	/// which is also the isZero() precision; the arithmetic is float like Eigen's. A shape
	/// mismatch never matches.
	/// </summary>
	public static bool EigenMatrixNear(RowMajorMatrix<float> lhs, RowMajorMatrix<float> rhs, float tol = 1e-5f)
	{
		if (lhs.Rows != rhs.Rows || lhs.Cols != rhs.Cols)
		{
			return false;
		}

		float lhsSquaredNorm = 0;
		float rhsSquaredNorm = 0;
		float diffSquaredNorm = 0;
		bool rhsIsZero = true;
		for (int i = 0; i < lhs.Data.Length; ++i)
		{
			lhsSquaredNorm += lhs.Data[i] * lhs.Data[i];
			rhsSquaredNorm += rhs.Data[i] * rhs.Data[i];
			float diff = lhs.Data[i] - rhs.Data[i];
			diffSquaredNorm += diff * diff;
			rhsIsZero &= Math.Abs(rhs.Data[i]) <= 1e-5f;
		}

		if (rhsIsZero)
		{
			return MathF.Sqrt(lhsSquaredNorm) <= tol;
		}

		return diffSquaredNorm <= tol * tol * Math.Min(lhsSquaredNorm, rhsSquaredNorm);
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
