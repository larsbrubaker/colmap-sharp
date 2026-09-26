// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SparseMatrixTests (C#-only; COLMAP has no test for Eigen's sparse module): SparseMatrixCsc
// in ColmapSharp/LinearAlgebra. Every operation is checked against the same operation on
// the dense MatrixXd built from the same entries, plus the CSC invariants (sorted, unique
// rows per column) and Eigen's setFromTriplets duplicate summing.

using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Assertions.Enums;
using TUnit.Core;

namespace ColmapSharp.Tests.LinearAlgebra;

public class SparseMatrixTests
{
	internal static SparseMatrixCsc RandomSparse(Random random, int rows, int cols, double density)
	{
		var triplets = new List<SparseTriplet>();
		for (int j = 0; j < cols; j++)
		{
			for (int i = 0; i < rows; i++)
			{
				if (random.NextDouble() < density)
				{
					triplets.Add(new SparseTriplet(i, j, random.NextDouble() * 2 - 1));
				}
			}
		}

		return SparseMatrixCsc.FromTriplets(rows, cols, triplets);
	}

	private static double MaxAbsDiff(MatrixXd a, MatrixXd b)
	{
		double max = 0;
		for (int j = 0; j < a.Cols; j++)
		{
			for (int i = 0; i < a.Rows; i++)
			{
				max = Math.Max(max, Math.Abs(a[i, j] - b[i, j]));
			}
		}

		return max;
	}

	private static async Task AssertCscInvariants(SparseMatrixCsc m)
	{
		ReadOnlySpan<int> colPtr = m.ColPtr;
		ReadOnlySpan<int> rows = m.RowIndices;
		bool ok = colPtr[0] == 0 && colPtr[m.Cols] == m.NonZeros;
		for (int j = 0; j < m.Cols && ok; j++)
		{
			for (int p = colPtr[j] + 1; p < colPtr[j + 1]; p++)
			{
				ok &= rows[p - 1] < rows[p];
			}
		}

		await Assert.That(ok).IsTrue();
	}

	[Test]
	public async Task FromTriplets_SumsDuplicatesAndSortsRows()
	{
		var m = SparseMatrixCsc.FromTriplets(3, 2,
		[
			new(2, 0, 1.0), new(0, 0, 2.0), new(2, 0, 3.0), new(1, 1, 4.0), new(0, 1, 0.0),
		]);
		using (Assert.Multiple())
		{
			await Assert.That(m.NonZeros).IsEqualTo(4);
			await Assert.That(m.ColPtr.ToArray()).IsEquivalentTo(new[] { 0, 2, 4 }, CollectionOrdering.Matching);
			await Assert.That(m.RowIndices.ToArray()).IsEquivalentTo(new[] { 0, 2, 0, 1 }, CollectionOrdering.Matching);
			await Assert.That(m.Values.ToArray()).IsEquivalentTo(new[] { 2.0, 4.0, 0.0, 4.0 }, CollectionOrdering.Matching);
			await Assert.That(m[2, 0]).IsEqualTo(4.0);
			await Assert.That(m[1, 0]).IsEqualTo(0.0);
			await Assert.That(() => SparseMatrixCsc.FromTriplets(2, 2, [new(2, 0, 1.0)])).Throws<ArgumentOutOfRangeException>();
		}
	}

	[Test]
	public async Task TransposeAndProducts_MatchDense()
	{
		var random = new Random(7);
		SparseMatrixCsc a = RandomSparse(random, 13, 9, 0.3);
		SparseMatrixCsc b = RandomSparse(random, 9, 11, 0.3);
		MatrixXd da = a.ToDense(), db = b.ToDense();
		var x = new VectorXd(9);
		var y = new VectorXd(13);
		for (int i = 0; i < 9; i++)
		{
			x[i] = random.NextDouble();
		}

		for (int i = 0; i < 13; i++)
		{
			y[i] = random.NextDouble();
		}

		SparseMatrixCsc at = a.Transpose();
		SparseMatrixCsc ab = a * b;
		SparseMatrixCsc ata = a.TransposeTimesSelf();
		await AssertCscInvariants(at);
		await AssertCscInvariants(ab);
		await AssertCscInvariants(ata);
		using (Assert.Multiple())
		{
			await Assert.That(MaxAbsDiff(at.ToDense(), da.Transpose())).IsEqualTo(0.0);
			await Assert.That(MaxAbsDiff(ab.ToDense(), da * db)).IsLessThan(1e-14);
			await Assert.That(MaxAbsDiff(ata.ToDense(), da.Transpose() * da)).IsLessThan(1e-14);
			await Assert.That((a * x - da * x).MaxAbs()).IsLessThan(1e-14);
			await Assert.That((a.TransposeMultiply(y) - da.Transpose() * y).MaxAbs()).IsLessThan(1e-14);
		}
	}

	[Test]
	public async Task TriangularPart_KeepsOneTriangleWithDiagonal()
	{
		var random = new Random(3);
		SparseMatrixCsc a = RandomSparse(random, 7, 5, 0.5).TransposeTimesSelf();
		MatrixXd lower = a.TransposeTimesSelf(SymmetricPart.Lower).ToDense();
		MatrixXd full = a.TransposeTimesSelf().ToDense();
		MatrixXd upper = a.TriangularPart(SymmetricPart.Upper).ToDense();
		bool ok = true;
		for (int j = 0; j < 5; j++)
		{
			for (int i = 0; i < 5; i++)
			{
				ok &= lower[i, j] == (i >= j ? full[i, j] : 0.0);
				ok &= upper[i, j] == (i <= j ? a[i, j] : 0.0);
			}
		}

		await Assert.That(ok).IsTrue();
	}

	[Test]
	public async Task CoeffRefAndAddToDiagonal()
	{
		var m = SparseMatrixCsc.FromTriplets(3, 3, [new(0, 0, 1.0), new(2, 1, 5.0)]);
		m.CoeffRef(2, 1) += 1.0;
		SparseMatrixCsc shifted = m.AddToDiagonal(0.5);
		using (Assert.Multiple())
		{
			await Assert.That(m[2, 1]).IsEqualTo(6.0);
			await Assert.That(() => m.CoeffRef(1, 1)).Throws<ArgumentException>();
			await Assert.That(shifted.NonZeros).IsEqualTo(4);
			await Assert.That(shifted[0, 0]).IsEqualTo(1.5);
			await Assert.That(shifted[1, 1]).IsEqualTo(0.5);
			await Assert.That(shifted[2, 2]).IsEqualTo(0.5);
			await Assert.That(shifted[2, 1]).IsEqualTo(6.0);
		}
	}

	[Test]
	public async Task Block_KeepsEntriesInsideWithRowAndColumnOffset()
	{
		// 4x4 with entries in and around the 2x2 block at (1, 2).
		SparseMatrixCsc a = SparseMatrixCsc.FromTriplets(4, 4,
		[
			new SparseTriplet(0, 2, 1.0), new SparseTriplet(1, 2, 2.0), new SparseTriplet(2, 3, 3.0),
			new SparseTriplet(3, 3, 4.0), new SparseTriplet(1, 1, 5.0), new SparseTriplet(2, 2, 6.0),
		]);
		SparseMatrixCsc block = a.Block(1, 2, 2, 2);
		await Assert.That(block.Rows).IsEqualTo(2);
		await Assert.That(block.Cols).IsEqualTo(2);
		await Assert.That(block.NonZeros).IsEqualTo(3);
		await Assert.That(block.ToDense().IsApprox(a.ToDense().Block(1, 2, 2, 2), 0.0)).IsTrue();
		await Assert.That(block[0, 0]).IsEqualTo(2.0);
		await Assert.That(block[1, 0]).IsEqualTo(6.0);
		await Assert.That(block[1, 1]).IsEqualTo(3.0);
	}

	[Test]
	public async Task Subtract_UnionOfNonOverlappingPatterns()
	{
		SparseMatrixCsc a = SparseMatrixCsc.FromTriplets(3, 2, [new SparseTriplet(0, 0, 1.0), new SparseTriplet(2, 1, 2.0)]);
		SparseMatrixCsc b = SparseMatrixCsc.FromTriplets(3, 2, [new SparseTriplet(1, 0, 3.0), new SparseTriplet(0, 1, 4.0)]);
		SparseMatrixCsc difference = a - b;
		await Assert.That(difference.NonZeros).IsEqualTo(4);
		await Assert.That(difference.RowIndices.ToArray()).IsEquivalentTo(new[] { 0, 1, 0, 2 }, CollectionOrdering.Matching);
		await Assert.That(difference.ToDense().IsApprox(a.ToDense() - b.ToDense(), 0.0)).IsTrue();
		await Assert.That(difference[1, 0]).IsEqualTo(-3.0);
		await Assert.That(difference[0, 1]).IsEqualTo(-4.0);
	}
}
