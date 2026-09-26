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

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

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
	internal static int FactorInPlace(Span<double> lu, int n, Span<int> permutation) =>
		FactorAugmentedInPlace(lu, n, 0, permutation);

	/// <summary>
	/// <see cref="FactorInPlace"/> on the n x (n + extraCols) column-major span [A | B]:
	/// A becomes its packed L\U factors and B is carried through the same row swaps and
	/// elimination, ending as L^-1 P B - the forward-substituted right-hand sides, ready for
	/// <see cref="BackSubstituteTrailingRows"/>. Each entry of B receives b - l(i,0) y(0) -
	/// l(i,1) y(1) - ... in the same order as <see cref="SolveInPlace"/>'s forward
	/// substitution, so the results are bit-identical to factoring and then solving each
	/// column. Returns the permutation's sign. Allocation-free.
	/// </summary>
	internal static int FactorAugmentedInPlace(Span<double> lu, int n, int extraCols, Span<int> permutation)
	{
		for (int i = 0; i < n; i++)
		{
			permutation[i] = i;
		}

		int totalCols = n + extraCols;
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
				for (int c = 0; c < totalCols; c++)
				{
					(lu[c * n + k], lu[c * n + pivot]) = (lu[c * n + pivot], lu[c * n + k]);
				}

				(permutation[k], permutation[pivot]) = (permutation[pivot], permutation[k]);
				sign = -sign;
			}

			int below = n - k - 1;
			Span<double> multipliers = lu.Slice(k * n + k + 1, below);
			double diagonal = lu[k * n + k];
			int firstUpdated = k + 1;
			if (diagonal == 0)
			{
				// U is singular; the column is left as it is, but the right-hand sides still
				// get the forward-substitution step (a solve would subtract l * y too).
				firstUpdated = n;
			}
			else
			{
				for (int i = 0; i < multipliers.Length; i++)
				{
					multipliers[i] /= diagonal;
				}
			}

			// The rank-1 update, column by column. Each entry is a - l * u exactly as in the
			// textbook loop; the refs only drop the bounds checks from the O(n^3) part.
			ref double l = ref MemoryMarshal.GetReference(multipliers);
			for (int c = firstUpdated; c < totalCols; c++)
			{
				double ukc = lu[c * n + k];
				SubtractScaled(ref MemoryMarshal.GetReference(lu.Slice(c * n + k + 1, below)), ref l, ukc, below);
			}
		}

		return sign;
	}

	/// <summary>
	/// y[i] -= x[i] * a for i in [0, length), unrolled by four. Every entry gets exactly the
	/// one multiply and one subtract of the plain loop (no FMA), so results are unchanged.
	/// </summary>
	private static void SubtractScaled(ref double y, ref double x, double a, int length)
	{
		int i = 0;
		for (; i + 4 <= length; i += 4)
		{
			Unsafe.Add(ref y, i) -= Unsafe.Add(ref x, i) * a;
			Unsafe.Add(ref y, i + 1) -= Unsafe.Add(ref x, i + 1) * a;
			Unsafe.Add(ref y, i + 2) -= Unsafe.Add(ref x, i + 2) * a;
			Unsafe.Add(ref y, i + 3) -= Unsafe.Add(ref x, i + 3) * a;
		}

		for (; i < length; i++)
		{
			Unsafe.Add(ref y, i) -= Unsafe.Add(ref x, i) * a;
		}
	}

	/// <summary>
	/// The back-substitution half of <see cref="SolveTrailingRowsInPlace"/>: with y =
	/// L^-1 P b in <paramref name="x"/>, finishes x[firstRow..n-1] of A x = b (rows above
	/// firstRow keep y). Back substitution of row i reads only the rows below it.
	/// </summary>
	internal static void BackSubstituteTrailingRows(ReadOnlySpan<double> lu, int n, Span<double> x, int firstRow)
	{
		for (int i = n - 1; i >= firstRow; i--)
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
	/// Solves A x = b for one right-hand side from the packed factors of
	/// <see cref="FactorInPlace"/>. Allocation-free.
	/// </summary>
	internal static void SolveInPlace(
		ReadOnlySpan<double> lu, int n, ReadOnlySpan<int> permutation, ReadOnlySpan<double> b, Span<double> x)
	{
		SolveTrailingRowsInPlace(lu, n, permutation, b, x, 0);
	}

	/// <summary>
	/// <see cref="SolveInPlace"/> when only x[firstRow..n-1] is wanted: the forward
	/// substitution runs in full, the back substitution stops at firstRow (x above it is
	/// left holding L^-1 P b). The wanted entries are bit-identical to a full solve, because
	/// back substitution of row i reads only rows below it. Allocation-free; b and x must
	/// not overlap.
	/// </summary>
	internal static void SolveTrailingRowsInPlace(
		ReadOnlySpan<double> lu, int n, ReadOnlySpan<int> permutation, ReadOnlySpan<double> b, Span<double> x, int firstRow)
	{
		for (int i = 0; i < n; i++)
		{
			x[i] = b[permutation[i]];
		}

		// Forward substitution with the unit lower-triangular L, column-oriented: x[i]
		// still receives x[i] - l(i,0) x[0] - l(i,1) x[1] - ... in the same order as the
		// row-oriented sum, but walking L down its contiguous columns.
		ref double xRef = ref MemoryMarshal.GetReference(x);
		for (int k = 0; k < n - 1; k++)
		{
			double xk = Unsafe.Add(ref xRef, k);
			ref double column = ref MemoryMarshal.GetReference(lu.Slice(k * n, n));
			for (int i = k + 1; i < n; i++)
			{
				Unsafe.Add(ref xRef, i) -= Unsafe.Add(ref column, i) * xk;
			}
		}

		// Back substitution with U.
		BackSubstituteTrailingRows(lu, n, x, firstRow);
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
