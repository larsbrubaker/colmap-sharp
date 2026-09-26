// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Householder: the elementary reflector H = I - tau * v * v^T (v[0] = 1) shared by
// HouseholderQR and ColPivHouseholderQR, plus the routines that apply a sequence of them
// and accumulate Q. Written from Golub & Van Loan, "Matrix Computations", 4th ed., §5.1
// (Householder reflections, Algorithm 5.1.1) and §5.2 (factored-form representation), with
// the sign convention of LAPACK's dlarfg (LAPACK Users' Guide, 3rd ed., §5.4; LAPACK is
// BSD-3, and only its published convention is used here, no code): for x = (alpha, tail),
//   beta = -sign(alpha) * ||x||,  tau = (beta - alpha) / beta,  v = (1, tail / (alpha - beta)),
// so H x = (beta, 0, ..., 0). A zero tail gives tau = 0 (H = I) and beta = alpha. Eigen
// (MPL-2.0) is not ported; it documents the same representation (matrixQR() with the
// essential parts below the diagonal plus hCoeffs()), and numpy's qr (LAPACK dgeqrf) gives
// the same signs, which DecompositionOracleTests pins.
//
// Arithmetic: the tail's squared norm and every v^T c are left-to-right sums seeded with
// the first term (the folder contract in Vector3d.cs); no FMA.

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Householder reflector helpers over column-major spans.
/// </summary>
internal static class Householder
{
	/// <summary>
	/// Turns x = (alpha, tail) into a reflector in place: x[0] becomes beta, the tail becomes
	/// the essential part of v. Returns tau.
	/// </summary>
	public static double MakeInPlace(Span<double> x)
	{
		double alpha = x[0];
		Span<double> tail = x[1..];
		double tailSquaredNorm = VectorXd.Dot(tail, tail);
		if (tailSquaredNorm == 0)
		{
			return 0;
		}

		double norm = Math.Sqrt(alpha * alpha + tailSquaredNorm);
		double beta = alpha >= 0 ? -norm : norm;
		double scale = alpha - beta;
		for (int i = 0; i < tail.Length; i++)
		{
			tail[i] /= scale;
		}

		x[0] = beta;
		return (beta - alpha) / beta;
	}

	/// <summary>
	/// Applies H = I - tau * v v^T (v = (1, essential)) from the left to one column segment c
	/// of the same length as v.
	/// </summary>
	public static void ApplyLeft(ReadOnlySpan<double> essential, double tau, Span<double> c)
	{
		if (tau == 0)
		{
			return;
		}

		double w = c[0];
		for (int i = 0; i < essential.Length; i++)
		{
			w += essential[i] * c[i + 1];
		}

		double tw = tau * w;
		c[0] -= tw;
		for (int i = 0; i < essential.Length; i++)
		{
			c[i + 1] -= essential[i] * tw;
		}
	}

	/// <summary>
	/// Q = H_0 H_1 ... H_{k-1} (m x m) from packed reflectors: reflector j has its essential
	/// part in rows j+1.. of column j of <paramref name="packed"/> and acts on rows j..m-1.
	/// Built backwards from the identity, so each H_j only touches rows and columns j..m-1.
	/// </summary>
	public static MatrixXd AccumulateQ(MatrixXd packed, ReadOnlySpan<double> tau)
	{
		int m = packed.Rows;
		var q = MatrixXd.Identity(m);
		for (int j = tau.Length - 1; j >= 0; j--)
		{
			ReadOnlySpan<double> essential = packed.ColumnSpan(j)[(j + 1)..];
			for (int c = j; c < m; c++)
			{
				ApplyLeft(essential, tau[j], q.ColumnSpan(c)[j..]);
			}
		}

		return q;
	}

	/// <summary>Applies Q^T = H_{k-1} ... H_0 to a vector in place.</summary>
	public static void ApplyQTranspose(MatrixXd packed, ReadOnlySpan<double> tau, Span<double> x)
	{
		for (int j = 0; j < tau.Length; j++)
		{
			ApplyLeft(packed.ColumnSpan(j)[(j + 1)..], tau[j], x[j..]);
		}
	}

	/// <summary>
	/// Applies Q = H_0 H_1 ... H_{k-1} to a vector in place (the last reflector first), so
	/// Q times a few columns never needs the m x m Q.
	/// </summary>
	public static void ApplyQ(MatrixXd packed, ReadOnlySpan<double> tau, Span<double> x)
	{
		for (int j = tau.Length - 1; j >= 0; j--)
		{
			ApplyLeft(packed.ColumnSpan(j)[(j + 1)..], tau[j], x[j..]);
		}
	}

	/// <summary>The upper-triangular part of the packed factor (Eigen's matrixQR() triangularView&lt;Upper&gt;).</summary>
	public static MatrixXd UpperPart(MatrixXd packed)
	{
		var r = new MatrixXd(packed.Rows, packed.Cols);
		for (int c = 0; c < packed.Cols; c++)
		{
			for (int row = 0; row <= Math.Min(c, packed.Rows - 1); row++)
			{
				r[row, c] = packed[row, c];
			}
		}

		return r;
	}

	/// <summary>
	/// Back substitution R(0:n, 0:n) z = y(0:n) with the upper triangle of the packed factor.
	/// </summary>
	public static void BackSubstitute(MatrixXd packed, int n, Span<double> y)
	{
		for (int i = n - 1; i >= 0; i--)
		{
			double sum = y[i];
			for (int k = i + 1; k < n; k++)
			{
				sum -= packed[i, k] * y[k];
			}

			y[i] = sum / packed[i, i];
		}
	}
}
