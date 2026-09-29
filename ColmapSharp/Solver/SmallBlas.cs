// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/small_blas.h (the "Naive" loop kernels) and
// internal/ceres/invert_psd_matrix.h (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The products dispatch the bundle adjustment shapes (2 or 3 inner rows, inner dimension 3)
// to SmallBlasFixed.cs, which unrolls the k loop but keeps its order, so every shape rounds
// exactly as the loops here do.
//
// The small dense kernels of the Schur solvers (SchurEliminator.cs,
// ImplicitSchurComplement.cs, BlockRandomAccessMatrix.cs). Every matrix is row-major. Each
// output element is a sum over k in increasing order into a temporary, then stored with the
// operation (=, +=, -=), which is how Ceres' naive kernels round; Ceres switches to Eigen's
// kernels when all sizes are compile-time constants (its template specializations such as
// <2, 3, 6>), whose summation order is Eigen's and may differ in the last bits
// (divergence 35).
//
// InvertUpperPsd is InvertPSDMatrix with assume_full_rank = true for dynamic sizes:
// selfadjointView<Upper>().llt().solve(Identity), a Cholesky factorization of the upper
// triangle (Golub & Van Loan, "Matrix Computations", 4th ed., Algorithm 4.2.2, written from
// the book like LinearAlgebra/LLT.cs; Eigen is not ported) and two triangular solves per
// column of the identity.

namespace ColmapSharp.Solver;

/// <summary>Small dense row-major kernels (Ceres' small_blas.h) and PSD block inversion.</summary>
internal static class SmallBlas
{
	/// <summary>
	/// C(startRow.., startCol..) op= A' B for A (rowsA x colsA) and B (rowsA x colsB);
	/// <paramref name="operation"/> is 1 (+=), -1 (-=) or 0 (=). C has row stride
	/// <paramref name="colStrideC"/>.
	/// </summary>
	public static void MatrixTransposeMatrixMultiply(
		ReadOnlySpan<double> a,
		int rowsA,
		int colsA,
		ReadOnlySpan<double> b,
		int colsB,
		Span<double> c,
		int startRowC,
		int startColC,
		int colStrideC,
		int operation)
	{
		// The bundle adjustment shapes: residual blocks of 2 rows (E'E, E'F, F'F) and the
		// 3-row E'F blocks times (E'E)^-1.
		if (rowsA == 2)
		{
			SmallBlasFixed.MatrixTransposeMatrixMultiply2(a, colsA, b, colsB, c, (startRowC * colStrideC) + startColC, colStrideC, operation);
			return;
		}

		if (rowsA == 3)
		{
			SmallBlasFixed.MatrixTransposeMatrixMultiply3(a, colsA, b, colsB, c, (startRowC * colStrideC) + startColC, colStrideC, operation);
			return;
		}

		MatrixTransposeMatrixMultiplyNaive(a, rowsA, colsA, b, colsB, c, startRowC, startColC, colStrideC, operation);
	}

	/// <summary>
	/// The loop kernel behind <see cref="MatrixTransposeMatrixMultiply"/> for every shape;
	/// the fixed-size kernels must reproduce it bit for bit (SmallBlasFixedTests).
	/// </summary>
	internal static void MatrixTransposeMatrixMultiplyNaive(
		ReadOnlySpan<double> a,
		int rowsA,
		int colsA,
		ReadOnlySpan<double> b,
		int colsB,
		Span<double> c,
		int startRowC,
		int startColC,
		int colStrideC,
		int operation)
	{
		for (int row = 0; row < colsA; row++)
		{
			int index = ((row + startRowC) * colStrideC) + startColC;
			for (int col = 0; col < colsB; col++)
			{
				double tmp = 0.0;
				for (int k = 0; k < rowsA; k++)
				{
					tmp += a[(k * colsA) + row] * b[(k * colsB) + col];
				}

				Store(c, index + col, tmp, operation);
			}
		}
	}

	/// <summary>C(startRow.., startCol..) op= A B for A (rowsA x colsA) and B (colsA x colsB).</summary>
	public static void MatrixMatrixMultiply(
		ReadOnlySpan<double> a,
		int rowsA,
		int colsA,
		ReadOnlySpan<double> b,
		int colsB,
		Span<double> c,
		int startRowC,
		int startColC,
		int colStrideC,
		int operation)
	{
		// b_i' (E'E)^-1 times an E'F block with a 3-dimensional E block (points).
		if (colsA == 3)
		{
			SmallBlasFixed.MatrixMatrixMultiplyInner3(a, rowsA, b, colsB, c, (startRowC * colStrideC) + startColC, colStrideC, operation);
			return;
		}

		MatrixMatrixMultiplyNaive(a, rowsA, colsA, b, colsB, c, startRowC, startColC, colStrideC, operation);
	}

	/// <summary>The loop kernel behind <see cref="MatrixMatrixMultiply"/> for every shape.</summary>
	internal static void MatrixMatrixMultiplyNaive(
		ReadOnlySpan<double> a,
		int rowsA,
		int colsA,
		ReadOnlySpan<double> b,
		int colsB,
		Span<double> c,
		int startRowC,
		int startColC,
		int colStrideC,
		int operation)
	{
		for (int row = 0; row < rowsA; row++)
		{
			int index = ((row + startRowC) * colStrideC) + startColC;
			for (int col = 0; col < colsB; col++)
			{
				double tmp = 0.0;
				for (int k = 0; k < colsA; k++)
				{
					tmp += a[(row * colsA) + k] * b[(k * colsB) + col];
				}

				Store(c, index + col, tmp, operation);
			}
		}
	}

	/// <summary>c op= A b for A (rowsA x colsA).</summary>
	public static void MatrixVectorMultiply(
		ReadOnlySpan<double> a, int rowsA, int colsA, ReadOnlySpan<double> b, Span<double> c, int operation)
	{
		for (int row = 0; row < rowsA; row++)
		{
			double tmp = 0.0;
			for (int col = 0; col < colsA; col++)
			{
				tmp += a[(row * colsA) + col] * b[col];
			}

			Store(c, row, tmp, operation);
		}
	}

	/// <summary>c op= A' b for A (rowsA x colsA).</summary>
	public static void MatrixTransposeVectorMultiply(
		ReadOnlySpan<double> a, int rowsA, int colsA, ReadOnlySpan<double> b, Span<double> c, int operation)
	{
		for (int col = 0; col < colsA; col++)
		{
			double tmp = 0.0;
			for (int row = 0; row < rowsA; row++)
			{
				tmp += a[(row * colsA) + col] * b[row];
			}

			Store(c, col, tmp, operation);
		}
	}

	private static void Store(Span<double> c, int index, double tmp, int operation)
	{
		if (operation > 0)
		{
			c[index] += tmp;
		}
		else if (operation < 0)
		{
			c[index] -= tmp;
		}
		else
		{
			c[index] = tmp;
		}
	}

	/// <summary>
	/// inverse = M^-1 for the symmetric positive definite n x n row-major M, reading only
	/// its upper triangle. <paramref name="scratch"/> needs n * n doubles and may not alias
	/// either matrix; <paramref name="inverse"/> may alias <paramref name="m"/>. A matrix that
	/// is not positive definite yields NaN (Eigen's LLT leaves unspecified values there; NaN
	/// makes the step invalid, which is what the minimizer does with such garbage).
	/// </summary>
	public static void InvertUpperPsd(ReadOnlySpan<double> m, int n, Span<double> inverse, Span<double> scratch)
	{
		// L is stored row-major in scratch: L(i, j) = scratch[i * n + j], i >= j, with
		// A(i, j) = M(j, i) for i >= j (the lower triangle of the symmetric matrix the upper
		// triangle describes).
		Span<double> l = scratch[..(n * n)];
		for (int j = 0; j < n; j++)
		{
			for (int i = j; i < n; i++)
			{
				double v = m[(j * n) + i];
				for (int k = 0; k < j; k++)
				{
					v -= l[(i * n) + k] * l[(j * n) + k];
				}

				l[(i * n) + j] = v;
			}

			double pivot = l[(j * n) + j];
			if (!(pivot > 0))
			{
				inverse[..(n * n)].Fill(double.NaN);
				return;
			}

			double root = Math.Sqrt(pivot);
			l[(j * n) + j] = root;
			for (int i = j + 1; i < n; i++)
			{
				l[(i * n) + j] /= root;
			}
		}

		// Column c of the inverse solves L L' x = e_c.
		for (int c = 0; c < n; c++)
		{
			for (int i = 0; i < n; i++)
			{
				double v = i == c ? 1.0 : 0.0;
				for (int k = 0; k < i; k++)
				{
					v -= l[(i * n) + k] * inverse[(k * n) + c];
				}

				inverse[(i * n) + c] = v / l[(i * n) + i];
			}

			for (int i = n - 1; i >= 0; i--)
			{
				double v = inverse[(i * n) + c];
				for (int k = i + 1; k < n; k++)
				{
					v -= l[(k * n) + i] * inverse[(k * n) + c];
				}

				inverse[(i * n) + c] = v / l[(i * n) + i];
			}
		}
	}
}
