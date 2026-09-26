// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// FullPivLU: LU with complete pivoting, P A Q = L U, the replacement for Eigen::FullPivLU.
// COLMAP calls rank() (estimators/solvers/similarity_transform.h rejects 3 x N point sets
// of rank below the sample count); Ceres' line search polynomial fit
// (Solver/CeresPolynomial.cs) calls setThreshold(0).solve(b), which is Solve here.
//
// Algorithm: Gaussian elimination with complete pivoting, Golub & Van Loan, "Matrix
// Computations", 4th ed., Algorithm 3.4.3: at step k the entry of largest magnitude in the
// remaining submatrix is swapped to (k, k) (ties: the first in column-major order).
// Elimination stops once the remaining submatrix is exactly zero. Written from G&VL; Eigen
// (MPL-2.0) is not ported.
//
// Solve (G&VL section 3.4.4): c = P b, forward substitution with the unit lower factor, back
// substitution with the leading r x r block of U over the r pivots above the threshold, and
// x = Q [y; 0], so the coordinates past the rank are zero (Eigen's documented behavior).
//
// Rank: Eigen's documented default threshold. A pivot counts as nonzero when its magnitude
// is strictly greater than min(rows, cols) * epsilon * the largest pivot (the first one).
// Tier B.

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// LU decomposition with complete pivoting; exposes the numerical rank. Replacement for
/// Eigen::FullPivLU.
/// </summary>
public sealed class FullPivLU
{
	private readonly double[] _pivots;
	private readonly MatrixXd _lu;
	private readonly int[] _rowTranspositions;
	private readonly int[] _colTranspositions;

	/// <summary>Factorizes a matrix of any shape. The input is not modified.</summary>
	public FullPivLU(MatrixXd a)
	{
		MatrixXd lu = a.Clone();
		int m = lu.Rows;
		int n = lu.Cols;
		int size = Math.Min(m, n);
		var pivots = new List<double>(size);
		_rowTranspositions = new int[size];
		_colTranspositions = new int[size];
		for (int k = 0; k < size; k++)
		{
			int pivotRow = k;
			int pivotCol = k;
			double largest = 0;
			for (int c = k; c < n; c++)
			{
				for (int r = k; r < m; r++)
				{
					double magnitude = Math.Abs(lu[r, c]);
					if (magnitude > largest)
					{
						largest = magnitude;
						pivotRow = r;
						pivotCol = c;
					}
				}
			}

			if (largest == 0)
			{
				break;
			}

			_rowTranspositions[k] = pivotRow;
			_colTranspositions[k] = pivotCol;
			SwapRows(lu, k, pivotRow);
			SwapCols(lu, k, pivotCol);
			double pivot = lu[k, k];
			pivots.Add(pivot);
			for (int r = k + 1; r < m; r++)
			{
				double factor = lu[r, k] / pivot;
				lu[r, k] = factor;
				for (int c = k + 1; c < n; c++)
				{
					lu[r, c] -= factor * lu[k, c];
				}
			}
		}

		_pivots = pivots.ToArray();
		for (int k = _pivots.Length; k < size; k++)
		{
			_rowTranspositions[k] = k;
			_colTranspositions[k] = k;
		}

		_lu = lu;
		DiagonalSize = size;
	}

	/// <summary>min(rows, cols).</summary>
	public int DiagonalSize { get; }

	/// <summary>
	/// Numerical rank: the number of pivots with magnitude strictly greater than
	/// min(rows, cols) * epsilon * the largest pivot.
	/// </summary>
	public int Rank()
	{
		if (_pivots.Length == 0)
		{
			return 0;
		}

		double threshold = DiagonalSize * LinearAlgebraConstants.MachineEpsilon * Math.Abs(_pivots[0]);
		int rank = 0;
		foreach (double pivot in _pivots)
		{
			if (Math.Abs(pivot) > threshold)
			{
				rank++;
			}
		}

		return rank;
	}

	/// <summary>
	/// Solves A x = b (least squares is not attempted: rows past the rank are ignored),
	/// counting a pivot as nonzero when its magnitude is strictly greater than
	/// <paramref name="threshold"/> times the largest pivot's; the unknowns beyond that rank
	/// are zero. Eigen's setThreshold(threshold).solve(b).
	/// </summary>
	public VectorXd Solve(VectorXd b, double threshold)
	{
		int m = _lu.Rows;
		int n = _lu.Cols;
		Util.Check.Eq(b.Length, m);
		int rank = 0;
		double limit = _pivots.Length == 0 ? 0.0 : threshold * Math.Abs(_pivots[0]);
		foreach (double pivot in _pivots)
		{
			if (Math.Abs(pivot) > limit)
			{
				rank++;
			}
		}

		var x = new VectorXd(n);
		if (rank == 0)
		{
			return x;
		}

		// c = P b.
		var c = new double[m];
		b.AsSpan().CopyTo(c);
		for (int k = 0; k < DiagonalSize; k++)
		{
			(c[k], c[_rowTranspositions[k]]) = (c[_rowTranspositions[k]], c[k]);
		}

		// L y = c over the leading DiagonalSize rows (unit diagonal).
		for (int i = 0; i < DiagonalSize; i++)
		{
			double sum = c[i];
			for (int j = 0; j < i; j++)
			{
				sum -= _lu[i, j] * c[j];
			}

			c[i] = sum;
		}

		// U z = y over the leading rank x rank block.
		for (int i = rank - 1; i >= 0; i--)
		{
			double sum = c[i];
			for (int j = i + 1; j < rank; j++)
			{
				sum -= _lu[i, j] * c[j];
			}

			c[i] = sum / _lu[i, i];
		}

		// x = Q [z; 0]: undo the column transpositions in reverse.
		var y = new double[n];
		for (int i = 0; i < rank; i++)
		{
			y[i] = c[i];
		}

		for (int k = DiagonalSize - 1; k >= 0; k--)
		{
			(y[k], y[_colTranspositions[k]]) = (y[_colTranspositions[k]], y[k]);
		}

		y.CopyTo(x.AsSpan());
		return x;
	}

	private static void SwapRows(MatrixXd m, int a, int b)
	{
		if (a == b)
		{
			return;
		}

		for (int c = 0; c < m.Cols; c++)
		{
			(m[a, c], m[b, c]) = (m[b, c], m[a, c]);
		}
	}

	private static void SwapCols(MatrixXd m, int a, int b)
	{
		if (a == b)
		{
			return;
		}

		for (int r = 0; r < m.Rows; r++)
		{
			(m[r, a], m[r, b]) = (m[r, b], m[r, a]);
		}
	}
}
