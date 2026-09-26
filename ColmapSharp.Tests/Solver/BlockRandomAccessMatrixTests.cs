// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/block_random_access_dense_matrix_test.cc,
// block_random_access_sparse_matrix_test.cc and block_random_access_diagonal_matrix_test.cc
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BlockRandomAccess{Dense,Sparse,Diagonal}MatrixTests (Ceres' tests, not COLMAP's):
// ColmapSharp/Solver/BlockRandomAccessMatrix.cs, same blocks, values and tolerances.
// - Ceres' GetCell returns the whole value array with the cell's (row, col) and strides;
//   the port's TryGetCell returns the cell itself and its row stride. So "row == position,
//   col == position, stride == 12" (dense) becomes "the cell starts at
//   position_i * 12 + position_j of the values, stride 12", and "row == col == 0,
//   strides == block sizes" (sparse, diagonal) becomes "stride == the column block's size".
// - The dense copies of the sparse and diagonal matrices (Ceres' ToDenseMatrix of the
//   underlying BlockSparseMatrix / CompressedRowSparseMatrix) are assembled from the stored
//   cells.
// - Vector::Random draws from System.Random with a fixed seed, same range [-1, 1].
// Not ported: BlockRandomAccessSparseMatrixTest.IntPairToInt64Overflow and Int64ToIntPair.
// They test Ceres' private bit-packed (row, col) cell key (kRowShift); the port keys cells by
// row * num_blocks + col in a long, which has no packing to overflow or unpack.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

internal static class BlockRandomAccessTestBlocks
{
	public static Block[] Blocks() => [new Block(3, 0), new Block(4, 3), new Block(5, 7)];

	public const int NumRows = 3 + 4 + 5;

	// The stored cells as a dense matrix (absent cells are zero).
	public static MatrixXd ToDense(BlockRandomAccessMatrix m, Block[] blocks)
	{
		var dense = new MatrixXd(m.NumRows, m.NumRows);
		for (int i = 0; i < blocks.Length; i++)
		{
			for (int j = 0; j < blocks.Length; j++)
			{
				if (!m.TryGetCell(i, j, out Span<double> cell, out int stride))
				{
					continue;
				}

				for (int r = 0; r < blocks[i].Size; r++)
				{
					for (int c = 0; c < blocks[j].Size; c++)
					{
						dense[blocks[i].Position + r, blocks[j].Position + c] = cell[(r * stride) + c];
					}
				}
			}
		}

		return dense;
	}

	public static double Norm(MatrixXd m) => Math.Sqrt(m.AsSpan().ToArray().Sum(v => v * v));

	// cell = value * ones (+ identity on the diagonal blocks if requested).
	public static void Fill(Span<double> cell, int stride, int rows, int cols, double value, bool addIdentity)
	{
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				cell[(r * stride) + c] = value + (addIdentity && r == c ? 1.0 : 0.0);
			}
		}
	}
}

public class BlockRandomAccessDenseMatrixTests
{
	[Test]
	public async Task GetCell()
	{
		Block[] blocks = BlockRandomAccessTestBlocks.Blocks();
		const int numRows = BlockRandomAccessTestBlocks.NumRows;
		var m = new BlockRandomAccessDenseMatrix(blocks);
		await Assert.That(m.NumRows).IsEqualTo(numRows);

		for (int i = 0; i < blocks.Length; ++i)
		{
			int rowIdx = blocks[i].Position;
			for (int j = 0; j < blocks.Length; ++j)
			{
				int colIdx = blocks[j].Position;
				bool found = m.TryGetCell(i, j, out Span<double> cell, out int stride);
				bool startsAtRowCol = m.Values.AsSpan().Overlaps(cell, out int offset) && offset == (rowIdx * numRows) + colIdx;
				await Assert.That(found).IsTrue();
				await Assert.That(startsAtRowCol).IsTrue();
				await Assert.That(stride).IsEqualTo(3 + 4 + 5);
			}
		}
	}

	[Test]
	public async Task WriteCell()
	{
		Block[] blocks = BlockRandomAccessTestBlocks.Blocks();
		const int numRows = BlockRandomAccessTestBlocks.NumRows;
		var m = new BlockRandomAccessDenseMatrix(blocks);

		// Fill the cell (i,j) with (i + 1) * (j + 1)
		for (int i = 0; i < blocks.Length; ++i)
		{
			for (int j = 0; j < blocks.Length; ++j)
			{
				m.TryGetCell(i, j, out Span<double> cell, out int stride);
				BlockRandomAccessTestBlocks.Fill(cell, stride, blocks[i].Size, blocks[j].Size, (i + 1) * (j + 1), false);
			}
		}

		// Check the values in the array are correct by going over the entries of each block
		// manually. (As in Ceres, the position checked is each block's first entry.)
		for (int i = 0; i < blocks.Length; ++i)
		{
			int rowIdx = blocks[i].Position;
			for (int j = 0; j < blocks.Length; ++j)
			{
				int colIdx = blocks[j].Position;
				for (int r = 0; r < blocks[i].Size; ++r)
				{
					for (int c = 0; c < blocks[j].Size; ++c)
					{
						int pos = (rowIdx * numRows) + colIdx;
						await Assert.That(m.Values[pos]).IsEqualTo((i + 1) * (j + 1));
					}
				}
			}
		}
	}
}

public class BlockRandomAccessSparseMatrixTests
{
	[Test]
	public async Task GetCell()
	{
		Block[] blocks = BlockRandomAccessTestBlocks.Blocks();
		const int numRows = BlockRandomAccessTestBlocks.NumRows;
		var blockPairs = new SortedSet<(int, int)>();
		int numNonzeros = 0;
		blockPairs.Add((0, 0));
		numNonzeros += blocks[0].Size * blocks[0].Size;
		blockPairs.Add((1, 1));
		numNonzeros += blocks[1].Size * blocks[1].Size;
		blockPairs.Add((1, 2));
		numNonzeros += blocks[1].Size * blocks[2].Size;
		blockPairs.Add((0, 2));
		numNonzeros += blocks[2].Size * blocks[0].Size;

		var m = new BlockRandomAccessSparseMatrix(blocks, blockPairs);
		await Assert.That(m.NumRows).IsEqualTo(numRows);

		foreach ((int rowBlockId, int colBlockId) in blockPairs)
		{
			bool found = m.TryGetCell(rowBlockId, colBlockId, out Span<double> cell, out int stride);
			int cellLength = cell.Length;
			if (found)
			{
				// Write into the block
				BlockRandomAccessTestBlocks.Fill(
					cell, stride, blocks[rowBlockId].Size, blocks[colBlockId].Size, (rowBlockId + 1) * (colBlockId + 1), false);
			}

			await Assert.That(found).IsTrue();
			await Assert.That(stride).IsEqualTo(blocks[colBlockId].Size);
			await Assert.That(cellLength).IsEqualTo(blocks[rowBlockId].Size * blocks[colBlockId].Size);
		}

		await Assert.That(m.Values.Length).IsEqualTo(numNonzeros);
		MatrixXd dense = BlockRandomAccessTestBlocks.ToDense(m, blocks);
		const double Tolerance = 1e-14;

		// (0, 0)
		await Assert.That(BlockRandomAccessTestBlocks.Norm(dense.Block(0, 0, 3, 3) - MatrixXd.Constant(3, 3, 1.0))).IsEqualTo(0.0).Within(Tolerance);

		// (1, 1)
		await Assert.That(BlockRandomAccessTestBlocks.Norm(dense.Block(3, 3, 4, 4) - MatrixXd.Constant(4, 4, 2 * 2))).IsEqualTo(0.0).Within(Tolerance);

		// (1, 2)
		await Assert.That(BlockRandomAccessTestBlocks.Norm(dense.Block(3, 3 + 4, 4, 5) - MatrixXd.Constant(4, 5, 2 * 3))).IsEqualTo(0.0).Within(Tolerance);

		// (0, 2)
		await Assert.That(BlockRandomAccessTestBlocks.Norm(dense.Block(0, 3 + 4, 3, 5) - MatrixXd.Constant(3, 5, 3 * 1))).IsEqualTo(0.0).Within(Tolerance);

		// There is nothing else in the matrix besides these four blocks.
		await Assert.That(BlockRandomAccessTestBlocks.Norm(dense))
			.IsEqualTo(Math.Sqrt(9.0 + (16.0 * 16.0) + (36.0 * 20.0) + (9.0 * 15.0))).Within(Tolerance);

		// expected_y = dense.selfadjointView<Upper>() * x
		MatrixXd symmetric = dense.Clone();
		for (int c = 0; c < numRows; c++)
		{
			for (int r = c + 1; r < numRows; r++)
			{
				symmetric[r, c] = dense[c, r];
			}
		}

		VectorXd x = VectorXd.Ones(numRows);
		VectorXd expectedY = symmetric * x;
		var actualY = new double[numRows];
		m.SymmetricRightMultiplyAndAccumulate(x.AsSpan(), actualY);
		await Assert.That((expectedY - new VectorXd(actualY)).Norm()).IsEqualTo(0.0).Within(Tolerance);
	}
}

public class BlockRandomAccessDiagonalMatrixTests
{
	private const int NumNonzeros = (3 * 3) + (4 * 4) + (5 * 5);

	private static async Task<(BlockRandomAccessDiagonalMatrix M, Block[] Blocks)> SetUp()
	{
		Block[] blocks = BlockRandomAccessTestBlocks.Blocks();
		var m = new BlockRandomAccessDiagonalMatrix(blocks);
		await Assert.That(m.NumRows).IsEqualTo(BlockRandomAccessTestBlocks.NumRows);

		for (int i = 0; i < blocks.Length; ++i)
		{
			for (int j = 0; j < blocks.Length; ++j)
			{
				bool found = m.TryGetCell(i, j, out Span<double> cell, out int stride);

				// Off diagonal entries are not present.
				if (i != j)
				{
					await Assert.That(found).IsFalse();
					continue;
				}

				BlockRandomAccessTestBlocks.Fill(cell, stride, blocks[i].Size, blocks[j].Size, (i + 1) * (j + 1), true);
				await Assert.That(found).IsTrue();
				await Assert.That(stride).IsEqualTo(blocks[j].Size);
			}
		}

		return (m, blocks);
	}

	[Test]
	public async Task MatrixContents()
	{
		(BlockRandomAccessDiagonalMatrix m, Block[] blocks) = await SetUp();
		await Assert.That(m.Values.Length).IsEqualTo(NumNonzeros);
		MatrixXd dense = BlockRandomAccessTestBlocks.ToDense(m, blocks);
		const double Tolerance = 1e-14;

		// (0,0)
		await Assert.That(BlockRandomAccessTestBlocks.Norm(dense.Block(0, 0, 3, 3) - (MatrixXd.Constant(3, 3, 1.0) + MatrixXd.Identity(3))))
			.IsEqualTo(0.0).Within(Tolerance);

		// (1,1)
		await Assert.That(BlockRandomAccessTestBlocks.Norm(dense.Block(3, 3, 4, 4) - (MatrixXd.Constant(4, 4, 2 * 2) + MatrixXd.Identity(4))))
			.IsEqualTo(0.0).Within(Tolerance);

		// (2,2)
		await Assert.That(BlockRandomAccessTestBlocks.Norm(dense.Block(7, 7, 5, 5) - (MatrixXd.Constant(5, 5, 3 * 3) + MatrixXd.Identity(5))))
			.IsEqualTo(0.0).Within(Tolerance);

		// There is nothing else in the matrix besides these four blocks.
		await Assert.That(BlockRandomAccessTestBlocks.Norm(dense))
			.IsEqualTo(Math.Sqrt((6 * 1.0) + (3 * 4.0) + (12 * 16.0) + (4 * 25.0) + (20 * 81.0) + (5 * 100.0))).Within(Tolerance);
	}

	[Test]
	public async Task RightMultiplyAndAccumulate()
	{
		(BlockRandomAccessDiagonalMatrix m, Block[] blocks) = await SetUp();
		const double Tolerance = 1e-14;
		MatrixXd dense = BlockRandomAccessTestBlocks.ToDense(m, blocks);
		var random = new Random(1);
		var x = new VectorXd([.. Enumerable.Range(0, dense.Rows).Select(_ => (2.0 * random.NextDouble()) - 1.0)]);
		VectorXd expectedY = dense * x;
		var actualY = new double[dense.Rows];
		m.RightMultiplyAndAccumulate(x.AsSpan(), actualY);
		await Assert.That((expectedY - new VectorXd(actualY)).Norm()).IsEqualTo(0.0).Within(Tolerance);
	}

	[Test]
	public async Task Invert()
	{
		(BlockRandomAccessDiagonalMatrix m, Block[] blocks) = await SetUp();
		const double Tolerance = 1e-14;
		MatrixXd dense = BlockRandomAccessTestBlocks.ToDense(m, blocks);
		MatrixXd expectedInverse = new LLT(dense).Solve(MatrixXd.Identity(dense.Rows));

		m.Invert();
		dense = BlockRandomAccessTestBlocks.ToDense(m, blocks);

		await Assert.That(BlockRandomAccessTestBlocks.Norm(expectedInverse - dense)).IsEqualTo(0.0).Within(Tolerance);
	}
}
