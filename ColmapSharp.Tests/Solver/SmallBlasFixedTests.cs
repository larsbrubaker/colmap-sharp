// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SmallBlasFixedTests (C#-only; no Ceres counterpart): the unrolled kernels of
// ColmapSharp/Solver/SmallBlasFixed.cs, reached through SmallBlas' dispatch, give the same
// bits as SmallBlas' naive loops for every operation, on random data with signed zeros and
// strided, offset output blocks.

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class SmallBlasFixedTests
{
	private static double[] RandomValues(Random random, int count)
	{
		var values = new double[count];
		for (int i = 0; i < count; i++)
		{
			// Every seventh value is a negative zero, to pin the 0.0 + ... start of each sum.
			values[i] = i % 7 == 3 ? -0.0 : (random.NextDouble() - 0.5) * Math.Pow(10, random.Next(-3, 4));
		}

		return values;
	}

	private static bool SameBits(double[] x, double[] y) =>
		x.Length == y.Length && x.Zip(y).All(p => BitConverter.DoubleToInt64Bits(p.First) == BitConverter.DoubleToInt64Bits(p.Second));

	[Test]
	public async Task MatrixTransposeMatrixMultiply_FixedRowsMatchNaive()
	{
		var random = new Random(3);
		foreach (int rowsA in new[] { 2, 3 })
		{
			foreach ((int colsA, int colsB) in new[] { (3, 3), (3, 6), (6, 6), (6, 3), (1, 4), (4, 1) })
			{
				for (int operation = -1; operation <= 1; operation++)
				{
					double[] a = RandomValues(random, rowsA * colsA);
					double[] b = RandomValues(random, rowsA * colsB);
					const int StartRow = 1;
					const int StartCol = 2;
					int stride = colsB + 3;
					double[] fixedC = RandomValues(random, (colsA + StartRow) * stride);
					double[] naiveC = [.. fixedC];
					SmallBlas.MatrixTransposeMatrixMultiply(a, rowsA, colsA, b, colsB, fixedC, StartRow, StartCol, stride, operation);
					SmallBlas.MatrixTransposeMatrixMultiplyNaive(a, rowsA, colsA, b, colsB, naiveC, StartRow, StartCol, stride, operation);
					await Assert.That(SameBits(fixedC, naiveC)).IsTrue();
				}
			}
		}
	}

	[Test]
	public async Task MatrixMatrixMultiply_FixedInnerDimensionMatchesNaive()
	{
		var random = new Random(5);
		foreach ((int rowsA, int colsB) in new[] { (3, 3), (6, 6), (6, 3), (1, 4), (4, 1) })
		{
			for (int operation = -1; operation <= 1; operation++)
			{
				double[] a = RandomValues(random, rowsA * 3);
				double[] b = RandomValues(random, 3 * colsB);
				const int StartRow = 2;
				const int StartCol = 1;
				int stride = colsB + 2;
				double[] fixedC = RandomValues(random, (rowsA + StartRow) * stride);
				double[] naiveC = [.. fixedC];
				SmallBlas.MatrixMatrixMultiply(a, rowsA, 3, b, colsB, fixedC, StartRow, StartCol, stride, operation);
				SmallBlas.MatrixMatrixMultiplyNaive(a, rowsA, 3, b, colsB, naiveC, StartRow, StartCol, stride, operation);
				await Assert.That(SameBits(fixedC, naiveC)).IsTrue();
			}
		}
	}
}
