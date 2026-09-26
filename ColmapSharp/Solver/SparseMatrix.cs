// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/sparse_matrix.h and
// internal/ceres/dense_sparse_matrix.cc (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The Jacobian as the trust-region minimizer sees it: a linear operator it can scale by
// columns, multiply with and take column norms of. DenseSparseMatrix (a dense column-major
// matrix behind that interface, for DENSE_QR and DENSE_NORMAL_CHOLESKY) lives here;
// BlockSparseMatrix.cs is the block-structured one the sparse and Schur solvers use.
// Every product is a left-to-right sum in a fixed order, so results do not depend on the
// thread count.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Solver;

/// <summary>ceres::internal::SparseMatrix: the minimizer's view of a Jacobian.</summary>
internal abstract class SparseMatrix
{
	/// <summary>Number of rows (residuals).</summary>
	public abstract int NumRows { get; }

	/// <summary>Number of columns (tangent parameters).</summary>
	public abstract int NumCols { get; }

	/// <summary>Sets every stored value to zero, keeping the structure.</summary>
	public abstract void SetZero();

	/// <summary>y += A x.</summary>
	public abstract void RightMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y);

	/// <summary>y += A' x.</summary>
	public abstract void LeftMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y);

	/// <summary>x[j] = squared norm of column j.</summary>
	public abstract void SquaredColumnNorm(Span<double> x);

	/// <summary>A = A * diag(scale).</summary>
	public abstract void ScaleColumns(ReadOnlySpan<double> scale);
}

/// <summary>ceres::internal::DenseSparseMatrix: a dense Jacobian.</summary>
internal sealed class DenseSparseMatrix : SparseMatrix
{
	/// <summary>Creates a zero rows x cols matrix.</summary>
	public DenseSparseMatrix(int rows, int cols) => Matrix = new MatrixXd(rows, cols);

	/// <summary>The column-major storage.</summary>
	public MatrixXd Matrix { get; }

	/// <inheritdoc/>
	public override int NumRows => Matrix.Rows;

	/// <inheritdoc/>
	public override int NumCols => Matrix.Cols;

	/// <inheritdoc/>
	public override void SetZero() => Matrix.AsSpan().Clear();

	/// <inheritdoc/>
	public override void RightMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y)
	{
		int rows = NumRows;
		for (int c = 0; c < NumCols; c++)
		{
			ReadOnlySpan<double> column = Matrix.ColumnSpan(c);
			double xc = x[c];
			for (int r = 0; r < rows; r++)
			{
				y[r] += column[r] * xc;
			}
		}
	}

	/// <inheritdoc/>
	public override void LeftMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y)
	{
		int rows = NumRows;
		for (int c = 0; c < NumCols; c++)
		{
			ReadOnlySpan<double> column = Matrix.ColumnSpan(c);
			double sum = 0.0;
			for (int r = 0; r < rows; r++)
			{
				sum += column[r] * x[r];
			}

			y[c] += sum;
		}
	}

	/// <inheritdoc/>
	public override void SquaredColumnNorm(Span<double> x)
	{
		for (int c = 0; c < NumCols; c++)
		{
			double sum = 0.0;
			foreach (double v in Matrix.ColumnSpan(c))
			{
				sum += v * v;
			}

			x[c] = sum;
		}
	}

	/// <inheritdoc/>
	public override void ScaleColumns(ReadOnlySpan<double> scale)
	{
		for (int c = 0; c < NumCols; c++)
		{
			Span<double> column = Matrix.ColumnSpan(c);
			double s = scale[c];
			for (int r = 0; r < column.Length; r++)
			{
				column[r] *= s;
			}
		}
	}
}
