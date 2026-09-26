// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/small_blas_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// BLAS tests (Ceres' test, not COLMAP's): ColmapSharp/Solver/SmallBlas.cs, same sizes,
// strides, start offsets, initial values and the 5 * epsilon tolerance, against products
// computed with MatrixXd.
// - Ceres runs each product through four entry points: the dispatching kernel and the
//   "Naive" loop kernel, each with static and dynamic sizes. The port has one kernel (the
//   naive loops, dynamic sizes), so all four test names run it on their sizes.
// - The C row stride of Ceres' kernels (row_stride_c) only bounds the matrix; the port's
//   kernels take the column stride alone, so row_stride_c sizes the test matrix as in Ceres.
// - MatrixTransposeVectorMultiply fills A and b with Eigen's setRandom(); the port draws the
//   same range, [-1, 1], from System.Random with a fixed seed.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class BLAS
{
	private const double Tolerance = 5.0 * 2.220446049250313E-16;

	// Row-major data of the matrix with entries i + j + 1 (Ceres' initMatrix).
	private static double[] InitMatrix(int rows, int cols)
	{
		var m = new double[rows * cols];
		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				m[(i * cols) + j] = i + j + 1;
			}
		}

		return m;
	}

	private static MatrixXd ToMatrix(double[] rowMajor, int rows, int cols) => MatrixXd.FromRowMajor(rows, cols, rowMajor);

	private static double Norm(MatrixXd m) => Math.Sqrt(m.AsSpan().ToArray().Sum(v => v * v));

	// C op= product into rows [startRow, startRow + product.Rows) and the matching columns.
	private static void ApplyReference(MatrixXd c, MatrixXd product, int startRow, int startCol, int operation)
	{
		for (int r = 0; r < product.Rows; r++)
		{
			for (int k = 0; k < product.Cols; k++)
			{
				double p = product[r, k];
				c[startRow + r, startCol + k] = operation > 0 ? c[startRow + r, startCol + k] + p
					: operation < 0 ? c[startRow + r, startCol + k] - p
					: p;
			}
		}
	}

	private delegate void Kernel(double[] a, int rowA, int colA, double[] b, int colB, double[] c, int startRow, int startCol, int colStride, int op);

	// TestMatrixFunctions (transpose = false) and TestMatrixTransposeFunctions (true).
	private static async Task TestMatrixFunctions(int kRowA, int kColA, int kColB, bool transpose)
	{
		double[] a = InitMatrix(kRowA, kColA);
		int kRowB = transpose ? kRowA : kColA;
		double[] b = InitMatrix(kRowB, kColB);
		MatrixXd aMatrix = ToMatrix(a, kRowA, kColA);
		MatrixXd product = transpose ? aMatrix.TransposeTimes(ToMatrix(b, kRowB, kColB)) : aMatrix * ToMatrix(b, kRowB, kColB);
		int kRowC = transpose ? kColA : kRowA;
		Kernel kernel = transpose
			? (a1, r, c, b1, cb, c1, sr, sc, cs, op) => SmallBlas.MatrixTransposeMatrixMultiply(a1, r, c, b1, cb, c1, sr, sc, cs, op)
			: (a1, r, c, b1, cb, c1, sr, sc, cs, op) => SmallBlas.MatrixMatrixMultiply(a1, r, c, b1, cb, c1, sr, sc, cs, op);

		for (int rowStrideC = kRowC; rowStrideC < 3 * kRowC; ++rowStrideC)
		{
			for (int colStrideC = kColB; colStrideC < 3 * kColB; ++colStrideC)
			{
				foreach (int operation in new[] { 1, -1, 0 })
				{
					double[] c = [.. Enumerable.Repeat(1.0, rowStrideC * colStrideC)];
					MatrixXd cRef = MatrixXd.Constant(rowStrideC, colStrideC, 1.0);
					for (int startRowC = 0; startRowC + kRowC < rowStrideC; ++startRowC)
					{
						for (int startColC = 0; startColC + kColB < colStrideC; ++startColC)
						{
							ApplyReference(cRef, product, startRowC, startColC, operation);
							kernel(a, kRowA, kColA, b, kColB, c, startRowC, startColC, colStrideC, operation);
							await Assert.That(Norm(cRef - ToMatrix(c, rowStrideC, colStrideC))).IsEqualTo(0.0).Within(Tolerance);
						}
					}
				}
			}
		}
	}

	[Test]
	public Task MatrixMatrixMultiply_5_3_7() => TestMatrixFunctions(5, 3, 7, false);

	[Test]
	public Task MatrixMatrixMultiply_5_3_7_Dynamic() => TestMatrixFunctions(5, 3, 7, false);

	[Test]
	public Task MatrixMatrixMultiply_1_1_1() => TestMatrixFunctions(1, 1, 1, false);

	[Test]
	public Task MatrixMatrixMultiply_1_1_1_Dynamic() => TestMatrixFunctions(1, 1, 1, false);

	[Test]
	public Task MatrixMatrixMultiply_9_9_9() => TestMatrixFunctions(9, 9, 9, false);

	[Test]
	public Task MatrixMatrixMultiply_9_9_9_Dynamic() => TestMatrixFunctions(9, 9, 9, false);

	[Test]
	public Task MatrixMatrixMultiplyNaive_5_3_7() => TestMatrixFunctions(5, 3, 7, false);

	[Test]
	public Task MatrixMatrixMultiplyNaive_5_3_7_Dynamic() => TestMatrixFunctions(5, 3, 7, false);

	[Test]
	public Task MatrixMatrixMultiplyNaive_1_1_1() => TestMatrixFunctions(1, 1, 1, false);

	[Test]
	public Task MatrixMatrixMultiplyNaive_1_1_1_Dynamic() => TestMatrixFunctions(1, 1, 1, false);

	[Test]
	public Task MatrixMatrixMultiplyNaive_9_9_9() => TestMatrixFunctions(9, 9, 9, false);

	[Test]
	public Task MatrixMatrixMultiplyNaive_9_9_9_Dynamic() => TestMatrixFunctions(9, 9, 9, false);

	[Test]
	public Task MatrixTransposeMatrixMultiply_5_3_7() => TestMatrixFunctions(5, 3, 7, true);

	[Test]
	public Task MatrixTransposeMatrixMultiply_5_3_7_Dynamic() => TestMatrixFunctions(5, 3, 7, true);

	[Test]
	public Task MatrixTransposeMatrixMultiply_1_1_1() => TestMatrixFunctions(1, 1, 1, true);

	[Test]
	public Task MatrixTransposeMatrixMultiply_1_1_1_Dynamic() => TestMatrixFunctions(1, 1, 1, true);

	[Test]
	public Task MatrixTransposeMatrixMultiply_9_9_9() => TestMatrixFunctions(9, 9, 9, true);

	[Test]
	public Task MatrixTransposeMatrixMultiply_9_9_9_Dynamic() => TestMatrixFunctions(9, 9, 9, true);

	[Test]
	public Task MatrixTransposeMatrixMultiplyNaive_5_3_7() => TestMatrixFunctions(5, 3, 7, true);

	[Test]
	public Task MatrixTransposeMatrixMultiplyNaive_5_3_7_Dynamic() => TestMatrixFunctions(5, 3, 7, true);

	[Test]
	public Task MatrixTransposeMatrixMultiplyNaive_1_1_1() => TestMatrixFunctions(1, 1, 1, true);

	[Test]
	public Task MatrixTransposeMatrixMultiplyNaive_1_1_1_Dynamic() => TestMatrixFunctions(1, 1, 1, true);

	[Test]
	public Task MatrixTransposeMatrixMultiplyNaive_9_9_9() => TestMatrixFunctions(9, 9, 9, true);

	[Test]
	public Task MatrixTransposeMatrixMultiplyNaive_9_9_9_Dynamic() => TestMatrixFunctions(9, 9, 9, true);

	private static async Task TestMatrixVector(bool transpose, Func<int, double> fill)
	{
		for (int numRowsA = 1; numRowsA < 10; ++numRowsA)
		{
			for (int numColsA = 1; numColsA < 10; ++numColsA)
			{
				double[] a = [.. Enumerable.Range(0, numRowsA * numColsA).Select(fill)];
				int bLength = transpose ? numRowsA : numColsA;
				int cLength = transpose ? numColsA : numRowsA;
				double[] b = [.. Enumerable.Range(0, bLength).Select(fill)];
				MatrixXd aMatrix = ToMatrix(a, numRowsA, numColsA);
				VectorXd product = transpose ? aMatrix.TransposeTimes(new VectorXd(b)) : aMatrix * new VectorXd(b);
				foreach (int operation in new[] { 1, -1, 0 })
				{
					double[] c = [.. Enumerable.Repeat(1.0, cLength)];
					VectorXd cRef = operation > 0 ? VectorXd.Ones(cLength) + product
						: operation < 0 ? VectorXd.Ones(cLength) - product
						: product;
					if (transpose)
					{
						SmallBlas.MatrixTransposeVectorMultiply(a, numRowsA, numColsA, b, c, operation);
					}
					else
					{
						SmallBlas.MatrixVectorMultiply(a, numRowsA, numColsA, b, c, operation);
					}

					await Assert.That((cRef - new VectorXd(c)).Norm()).IsEqualTo(0.0).Within(Tolerance);
				}
			}
		}
	}

	[Test]
	public Task MatrixVectorMultiply() => TestMatrixVector(false, _ => 1.0);

	[Test]
	public Task MatrixTransposeVectorMultiply()
	{
		var random = new Random(1);
		return TestMatrixVector(true, _ => (2.0 * random.NextDouble()) - 1.0);
	}
}
