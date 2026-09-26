// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// FullPivLU: LU with complete pivoting, P A Q = L U, the replacement for Eigen::FullPivLU.
// COLMAP only calls rank() (estimators/solvers/similarity_transform.h rejects 3 x N point
// sets of rank below the sample count), so that is all this class exposes.
//
// Algorithm: Gaussian elimination with complete pivoting, Golub & Van Loan, "Matrix
// Computations", 4th ed., Algorithm 3.4.3: at step k the entry of largest magnitude in the
// remaining submatrix is swapped to (k, k) (ties: the first in column-major order).
// Elimination stops once the remaining submatrix is exactly zero. Written from G&VL; Eigen
// (MPL-2.0) is not ported.
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

	/// <summary>Factorizes a matrix of any shape. The input is not modified.</summary>
	public FullPivLU(MatrixXd a)
	{
		MatrixXd lu = a.Clone();
		int m = lu.Rows;
		int n = lu.Cols;
		int size = Math.Min(m, n);
		var pivots = new List<double>(size);
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
