// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PartialPivLU: LU factorization with partial (row) pivoting of a square matrix,
// P * A = L * U, the replacement for Eigen::PartialPivLU / matrix.partialPivLu(). COLMAP
// uses it to solve small square systems (affine and homography estimators, the 5-point
// essential solver's 10x10 block, the camera-model Newton step in sensor/models.h) and,
// through MatrixXd.Determinant/Inverse, for dynamic determinants and inverses.
//
// Algorithm: right-looking Gaussian elimination with partial pivoting, Golub & Van Loan,
// "Matrix Computations", 4th ed., Algorithm 3.4.1 (the same scheme as LAPACK's dgetf2).
// Written from the book; Eigen (MPL-2.0) is not ported. The pivot is the first entry of
// largest magnitude in the column, the L multipliers are computed by division, and a zero
// pivot is left in place (U is singular; like Eigen, solving then yields inf/NaN rather
// than an error). Tier B: Eigen blocks the factorization for large matrices, so results
// agree within rounding, not bit for bit.

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// LU with partial pivoting, P * A = L * U. Replacement for Eigen::PartialPivLU.
/// </summary>
public sealed class PartialPivLU
{
	private readonly MatrixXd _lu;

	// _permutation[i] is the row of A that ended up as row i of P * A.
	private readonly int[] _permutation;
	private readonly int _determinantSign;

	/// <summary>Factorizes a square matrix. The input is not modified.</summary>
	public PartialPivLU(MatrixXd a)
	{
		if (a.Rows != a.Cols)
		{
			throw new ArgumentException($"PartialPivLU needs a square matrix, got {a.Rows}x{a.Cols}.", nameof(a));
		}

		int n = a.Rows;
		_lu = a.Clone();
		_permutation = new int[n];
		_determinantSign = FactorInPlace(_lu.AsSpan(), n, _permutation);
	}

	/// <summary>
	/// The factorization kernel over a column-major n x n span, factorized in place into the
	/// packed L\U form; <paramref name="permutation"/> (length n) receives the row order.
	/// Returns the permutation's sign. Allocation-free, so the minimal solvers
	/// (Estimators/Solvers) can run it on stackalloc buffers once per RANSAC hypothesis.
	/// </summary>
	internal static int FactorInPlace(Span<double> lu, int n, Span<int> permutation)
	{
		for (int i = 0; i < n; i++)
		{
			permutation[i] = i;
		}

		int sign = 1;
		for (int k = 0; k < n; k++)
		{
			int pivot = k;
			double best = Math.Abs(lu[k * n + k]);
			for (int i = k + 1; i < n; i++)
			{
				double v = Math.Abs(lu[k * n + i]);
				if (v > best)
				{
					best = v;
					pivot = i;
				}
			}

			if (pivot != k)
			{
				for (int c = 0; c < n; c++)
				{
					(lu[c * n + k], lu[c * n + pivot]) = (lu[c * n + pivot], lu[c * n + k]);
				}

				(permutation[k], permutation[pivot]) = (permutation[pivot], permutation[k]);
				sign = -sign;
			}

			double diagonal = lu[k * n + k];
			if (diagonal == 0)
			{
				continue;
			}

			for (int i = k + 1; i < n; i++)
			{
				lu[k * n + i] /= diagonal;
			}

			for (int c = k + 1; c < n; c++)
			{
				double ukc = lu[c * n + k];
				for (int i = k + 1; i < n; i++)
				{
					lu[c * n + i] -= lu[k * n + i] * ukc;
				}
			}
		}

		return sign;
	}

	/// <summary>
	/// Solves A x = b for one right-hand side from the packed factors of
	/// <see cref="FactorInPlace"/>. Allocation-free.
	/// </summary>
	internal static void SolveInPlace(
		ReadOnlySpan<double> lu, int n, ReadOnlySpan<int> permutation, ReadOnlySpan<double> b, Span<double> x)
	{
		for (int i = 0; i < n; i++)
		{
			x[i] = b[permutation[i]];
		}

		// Forward substitution with the unit lower-triangular L.
		for (int i = 1; i < n; i++)
		{
			double sum = x[i];
			for (int k = 0; k < i; k++)
			{
				sum -= lu[k * n + i] * x[k];
			}

			x[i] = sum;
		}

		// Back substitution with U.
		for (int i = n - 1; i >= 0; i--)
		{
			double sum = x[i];
			for (int k = i + 1; k < n; k++)
			{
				sum -= lu[k * n + i] * x[k];
			}

			x[i] = sum / lu[i * n + i];
		}
	}

	/// <summary>
	/// The packed factors (Eigen's matrixLU()): U on and above the diagonal, the strictly
	/// lower part of the unit lower-triangular L below it. A copy.
	/// </summary>
	public MatrixXd MatrixLU() => _lu.Clone();

	/// <summary>
	/// The row permutation as indices: row i of P * A is row PermutationIndices[i] of A.
	/// A copy.
	/// </summary>
	public int[] PermutationIndices() => (int[])_permutation.Clone();

	/// <summary>The permutation matrix P with P * A = L * U.</summary>
	public MatrixXd PermutationP()
	{
		int n = _permutation.Length;
		var p = new MatrixXd(n, n);
		for (int i = 0; i < n; i++)
		{
			p[i, _permutation[i]] = 1;
		}

		return p;
	}

	/// <summary>The unit lower-triangular factor L.</summary>
	public MatrixXd MatrixL()
	{
		int n = _lu.Rows;
		var l = MatrixXd.Identity(n);
		for (int c = 0; c < n; c++)
		{
			for (int r = c + 1; r < n; r++)
			{
				l[r, c] = _lu[r, c];
			}
		}

		return l;
	}

	/// <summary>The upper-triangular factor U.</summary>
	public MatrixXd MatrixU()
	{
		int n = _lu.Rows;
		var u = new MatrixXd(n, n);
		for (int c = 0; c < n; c++)
		{
			for (int r = 0; r <= c; r++)
			{
				u[r, c] = _lu[r, c];
			}
		}

		return u;
	}

	/// <summary>det(A): the permutation sign times the product of U's diagonal, left to right.</summary>
	public double Determinant()
	{
		double det = _determinantSign;
		for (int i = 0; i < _lu.Rows; i++)
		{
			det *= _lu[i, i];
		}

		return det;
	}

	/// <summary>Solves A x = b.</summary>
	public VectorXd Solve(VectorXd b)
	{
		var x = new MatrixXd(b.Length, 1);
		b.AsSpan().CopyTo(x.AsSpan());
		return Solve(x).Col(0);
	}

	/// <summary>Solves A X = B column by column.</summary>
	public MatrixXd Solve(MatrixXd b)
	{
		int n = _lu.Rows;
		if (b.Rows != n)
		{
			throw new ArgumentException($"Right-hand side has {b.Rows} rows, expected {n}.", nameof(b));
		}

		var x = new MatrixXd(n, b.Cols);
		for (int j = 0; j < b.Cols; j++)
		{
			SolveInPlace(_lu.AsSpan(), n, _permutation, b.ColumnSpan(j), x.ColumnSpan(j));
		}

		return x;
	}

	/// <summary>A^-1, by solving against the identity.</summary>
	public MatrixXd Inverse() => Solve(MatrixXd.Identity(_lu.Rows));
}
