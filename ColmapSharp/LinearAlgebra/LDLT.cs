// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// LDLT: pivoted "robust Cholesky" A = P^T L D L^T P of a symmetric positive or negative
// semidefinite matrix, the replacement for Eigen::LDLT (COLMAP: optim/tiny_solver.h
// solves its Levenberg-Marquardt normal equations with it and checks info()). Sibling of
// LLT (unpivoted, positive definite only), which defines ComputationInfo.
//
// Algorithm: outer-product LDL^T with symmetric (diagonal) pivoting: at step k the
// remaining diagonal entry of largest magnitude is swapped to position k (rows and
// columns together), then the trailing block is updated with the rank-1 term. Golub &
// Van Loan, "Matrix Computations", 4th ed., §4.2.9 (outer-product Cholesky with diagonal
// pivoting) applied to the LDL^T form of §4.1.2; Higham, "Accuracy and Stability of
// Numerical Algorithms", 2nd ed., §10.3. Written from those texts; Eigen (MPL-2.0) is not
// ported. Eigen documents the same factorization shape (P^T L D L^* P with a unit lower L,
// "robust Cholesky with pivoting"), so the factors correspond; Tier B.
//
// Only the lower triangle of the input is read, like Eigen's default (Lower). Choices of
// ours, not taken from Eigen's code: the first diagonal entry of largest magnitude wins a
// tie; a zero pivot whose column below is also zero (the semidefinite case) leaves its L
// column zero and Solve maps that component to 0; Info is NumericalIssue when a pivot is
// NaN or infinite, or when a zero pivot has a nonzero entry below it (no diagonally
// pivoted LDL^T exists, e.g. [[0, 1], [1, 0]]; the factors are then not usable).

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Pivoted LDL^T factorization A = P^T L D L^T P. Replacement for Eigen::LDLT&lt;MatrixXd&gt;.
/// </summary>
public sealed class LDLT
{
	// Packed: D on the diagonal, the strictly lower part of the unit lower L below it.
	private readonly MatrixXd _ldl;

	// _transpositions[k] = the index swapped with k at step k (Eigen's transpositionsP()).
	private readonly int[] _transpositions;

	/// <summary>Factorizes a symmetric matrix, reading its lower triangle.</summary>
	public LDLT(MatrixXd a)
	{
		if (a.Rows != a.Cols)
		{
			throw new ArgumentException($"LDLT needs a square matrix, got {a.Rows}x{a.Cols}.", nameof(a));
		}

		int n = a.Rows;

		// Work on a full symmetric copy built from the lower triangle; the trailing block
		// is kept symmetric so the row-and-column swaps stay consistent.
		_ldl = new MatrixXd(n, n);
		for (int c = 0; c < n; c++)
		{
			for (int r = c; r < n; r++)
			{
				_ldl[r, c] = a[r, c];
				_ldl[c, r] = a[r, c];
			}
		}

		_transpositions = new int[n];
		Info = ComputationInfo.Success;
		Span<double> m = _ldl.AsSpan();
		for (int k = 0; k < n; k++)
		{
			int pivot = k;
			double best = Math.Abs(m[k * n + k]);
			for (int i = k + 1; i < n; i++)
			{
				double v = Math.Abs(m[i * n + i]);
				if (v > best)
				{
					best = v;
					pivot = i;
				}
			}

			_transpositions[k] = pivot;
			if (pivot != k)
			{
				SwapSymmetric(m, n, k, pivot);
			}

			double d = m[k * n + k];
			if (!double.IsFinite(d))
			{
				Info = ComputationInfo.NumericalIssue;
			}

			if (d == 0)
			{
				// A zero pivot is only consistent when its whole column below is zero too
				// (a semidefinite matrix); then that column of L is zero. Otherwise, e.g.
				// [[0, 1], [1, 0]], no diagonally pivoted LDL^T exists: report it so callers
				// like tiny_solver reject the step instead of using a wrong solve.
				for (int i = k + 1; i < n; i++)
				{
					if (m[k * n + i] != 0)
					{
						Info = ComputationInfo.NumericalIssue;
					}

					m[k * n + i] = 0;
				}

				continue;
			}

			// l(i, k) = a(i, k) / d, then A(k+1:, k+1:) -= l * d * l^T (lower part).
			for (int i = k + 1; i < n; i++)
			{
				m[k * n + i] /= d;
			}

			for (int c = k + 1; c < n; c++)
			{
				double dlc = m[k * n + c] * d;
				for (int r = c; r < n; r++)
				{
					// Mirror into the upper part so later symmetric swaps see a symmetric
					// trailing block.
					double updated = m[c * n + r] - m[k * n + r] * dlc;
					m[c * n + r] = updated;
					m[r * n + c] = updated;
				}
			}
		}

		// Clear the strictly upper part so MatrixLDLT() is unambiguous.
		for (int c = 1; c < n; c++)
		{
			for (int r = 0; r < c; r++)
			{
				m[c * n + r] = 0;
			}
		}
	}

	/// <summary>Success, or NumericalIssue when a pivot was NaN or infinite, or zero with a nonzero entry below it.</summary>
	public ComputationInfo Info { get; }

	/// <summary>The diagonal of D, Eigen's vectorD().</summary>
	public VectorXd VectorD()
	{
		var d = new VectorXd(_ldl.Rows);
		for (int i = 0; i < d.Length; i++)
		{
			d[i] = _ldl[i, i];
		}

		return d;
	}

	/// <summary>The unit lower-triangular factor L.</summary>
	public MatrixXd MatrixL()
	{
		var l = _ldl.Clone();
		for (int i = 0; i < l.Rows; i++)
		{
			l[i, i] = 1;
		}

		return l;
	}

	/// <summary>The transpositions, Eigen's transpositionsP(): step k swapped k with Transpositions()[k].</summary>
	public int[] Transpositions() => (int[])_transpositions.Clone();

	/// <summary>The permutation matrix P with A = P^T L D L^T P.</summary>
	public MatrixXd PermutationP()
	{
		var p = MatrixXd.Identity(_ldl.Rows);
		for (int k = 0; k < _transpositions.Length; k++)
		{
			SwapRows(p, k, _transpositions[k]);
		}

		return p;
	}

	/// <summary>True when every entry of D is &gt;= 0 (Eigen's isPositive()).</summary>
	public bool IsPositive()
	{
		for (int i = 0; i < _ldl.Rows; i++)
		{
			if (_ldl[i, i] < 0)
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>True when every entry of D is &lt;= 0 (Eigen's isNegative()).</summary>
	public bool IsNegative()
	{
		for (int i = 0; i < _ldl.Rows; i++)
		{
			if (_ldl[i, i] > 0)
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>Solves A x = b.</summary>
	public VectorXd Solve(VectorXd b)
	{
		int n = _ldl.Rows;
		if (b.Length != n)
		{
			throw new ArgumentException($"Right-hand side has {b.Length} rows, expected {n}.", nameof(b));
		}

		var x = b.Clone();
		SolveInPlace(x.AsSpan());
		return x;
	}

	/// <summary>Solves A X = B column by column.</summary>
	public MatrixXd Solve(MatrixXd b)
	{
		if (b.Rows != _ldl.Rows)
		{
			throw new ArgumentException($"Right-hand side has {b.Rows} rows, expected {_ldl.Rows}.", nameof(b));
		}

		var x = b.Clone();
		for (int j = 0; j < x.Cols; j++)
		{
			SolveInPlace(x.ColumnSpan(j));
		}

		return x;
	}

	private void SolveInPlace(Span<double> x)
	{
		int n = _ldl.Rows;
		ReadOnlySpan<double> m = _ldl.AsSpan();
		for (int k = 0; k < n; k++)
		{
			(x[k], x[_transpositions[k]]) = (x[_transpositions[k]], x[k]);
		}

		for (int i = 1; i < n; i++)
		{
			double sum = x[i];
			for (int k = 0; k < i; k++)
			{
				sum -= m[k * n + i] * x[k];
			}

			x[i] = sum;
		}

		for (int i = 0; i < n; i++)
		{
			double d = m[i * n + i];
			x[i] = d == 0 ? 0 : x[i] / d;
		}

		for (int i = n - 1; i >= 0; i--)
		{
			double sum = x[i];
			for (int k = i + 1; k < n; k++)
			{
				sum -= m[i * n + k] * x[k];
			}

			x[i] = sum;
		}

		for (int k = n - 1; k >= 0; k--)
		{
			(x[k], x[_transpositions[k]]) = (x[_transpositions[k]], x[k]);
		}
	}

	private static void SwapSymmetric(Span<double> m, int n, int a, int b)
	{
		for (int c = 0; c < n; c++)
		{
			(m[c * n + a], m[c * n + b]) = (m[c * n + b], m[c * n + a]);
		}

		for (int r = 0; r < n; r++)
		{
			(m[a * n + r], m[b * n + r]) = (m[b * n + r], m[a * n + r]);
		}
	}

	private static void SwapRows(MatrixXd p, int a, int b)
	{
		for (int c = 0; c < p.Cols; c++)
		{
			(p[a, c], p[b, c]) = (p[b, c], p[a, c]);
		}
	}
}
