// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// LLT: Cholesky factorization A = L * L^T of a symmetric positive-definite matrix, the
// replacement for Eigen::LLT / matrix.llt() (COLMAP: cost_functions/utils.h takes
// cov.inverse().llt().matrixL() as a square-root information matrix). Also defines
// ComputationInfo, shared with LDLT. Sibling of LDLT (the pivoted, semidefinite-tolerant
// variant) and PartialPivLU.
//
// Algorithm: column-oriented ("gaxpy") Cholesky, Golub & Van Loan, "Matrix Computations",
// 4th ed., Algorithm 4.2.2. Written from the book; Eigen (MPL-2.0) is not ported. Like
// Eigen's default (Lower) LLT, only the lower triangle of the input is read. A
// non-positive pivot stops the factorization and reports NumericalIssue, which is Eigen's
// documented info() contract. Tier B.

namespace ColmapSharp.LinearAlgebra;

/// <summary>Outcome of a decomposition, Eigen's ComputationInfo.</summary>
public enum ComputationInfo
{
	/// <summary>The decomposition succeeded.</summary>
	Success,

	/// <summary>The input did not have the required properties (e.g. not positive definite).</summary>
	NumericalIssue,
}

/// <summary>
/// Cholesky factorization A = L L^T. Replacement for Eigen::LLT&lt;MatrixXd&gt;.
/// </summary>
public sealed class LLT
{
	private readonly MatrixXd _l;

	/// <summary>Factorizes a symmetric positive-definite matrix, reading its lower triangle.</summary>
	public LLT(MatrixXd a)
	{
		if (a.Rows != a.Cols)
		{
			throw new ArgumentException($"LLT needs a square matrix, got {a.Rows}x{a.Cols}.", nameof(a));
		}

		int n = a.Rows;
		_l = new MatrixXd(n, n);
		Span<double> l = _l.AsSpan();
		ReadOnlySpan<double> src = a.AsSpan();
		Info = ComputationInfo.Success;
		for (int j = 0; j < n; j++)
		{
			// v = A(j:n, j) - L(j:n, 0:j) * L(j, 0:j)^T
			for (int i = j; i < n; i++)
			{
				double v = src[j * n + i];
				for (int k = 0; k < j; k++)
				{
					v -= l[k * n + i] * l[k * n + j];
				}

				l[j * n + i] = v;
			}

			double pivot = l[j * n + j];
			if (!(pivot > 0))
			{
				Info = ComputationInfo.NumericalIssue;
				return;
			}

			double root = Math.Sqrt(pivot);
			l[j * n + j] = root;
			for (int i = j + 1; i < n; i++)
			{
				l[j * n + i] /= root;
			}
		}
	}

	/// <summary>Success, or NumericalIssue when the matrix is not positive definite.</summary>
	public ComputationInfo Info { get; }

	/// <summary>The lower-triangular factor L (a copy).</summary>
	public MatrixXd MatrixL() => _l.Clone();

	/// <summary>The upper-triangular factor U = L^T.</summary>
	public MatrixXd MatrixU() => _l.Transpose();

	/// <summary>Solves A x = b by forward then backward substitution.</summary>
	public VectorXd Solve(VectorXd b)
	{
		int n = _l.Rows;
		if (b.Length != n)
		{
			throw new ArgumentException($"Right-hand side has {b.Length} rows, expected {n}.", nameof(b));
		}

		var x = b.Clone();
		SolveInPlace(_l.AsSpan(), n, x.AsSpan());
		return x;
	}

	/// <summary>Solves A X = B column by column.</summary>
	public MatrixXd Solve(MatrixXd b)
	{
		int n = _l.Rows;
		if (b.Rows != n)
		{
			throw new ArgumentException($"Right-hand side has {b.Rows} rows, expected {n}.", nameof(b));
		}

		var x = b.Clone();
		for (int j = 0; j < x.Cols; j++)
		{
			SolveInPlace(_l.AsSpan(), n, x.ColumnSpan(j));
		}

		return x;
	}

	private static void SolveInPlace(ReadOnlySpan<double> l, int n, Span<double> x)
	{
		for (int i = 0; i < n; i++)
		{
			double sum = x[i];
			for (int k = 0; k < i; k++)
			{
				sum -= l[k * n + i] * x[k];
			}

			x[i] = sum / l[i * n + i];
		}

		for (int i = n - 1; i >= 0; i--)
		{
			double sum = x[i];
			for (int k = i + 1; k < n; k++)
			{
				sum -= l[i * n + k] * x[k];
			}

			x[i] = sum / l[i * n + i];
		}
	}
}
