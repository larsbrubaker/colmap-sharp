// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/block_random_access_matrix.h,
// internal/ceres/block_random_access_dense_matrix.cc,
// internal/ceres/block_random_access_sparse_matrix.cc and
// internal/ceres/block_random_access_diagonal_matrix.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// The reduced camera matrix S the Schur eliminator (SchurEliminator.cs) accumulates into,
// addressed by (row block, column block). Only the upper block triangle is written.
// - Dense (DENSE_SCHUR): one row-major n x n array. Read as column-major, its upper
//   triangle is the lower triangle LinearAlgebra/LLT.cs factors, so the matrix goes to the
//   dense Cholesky with a plain copy.
// - Sparse (SPARSE_SCHUR): one row-major cell per block pair (i, j), i <= j, that two
//   F blocks share through an E block or a row; ToLowerCsc hands the lower triangle to the
//   sparse Cholesky.
// - Diagonal (the SCHUR_JACOBI preconditioner): only the diagonal blocks, inverted in place.
// Ceres guards each cell with a mutex for its multithreaded elimination; the elimination here
// is sequential in a fixed order (SchurEliminator.cs), so there are no locks.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Solver;

/// <summary>ceres::internal::BlockRandomAccessMatrix.</summary>
internal abstract class BlockRandomAccessMatrix
{
	/// <summary>Scalar rows (= columns).</summary>
	public abstract int NumRows { get; }

	/// <summary>
	/// The cell (<paramref name="rowBlockId"/>, <paramref name="colBlockId"/>) if it is
	/// stored: <paramref name="cell"/> starts at its (0, 0) entry and entry (r, c) is at
	/// r * <paramref name="stride"/> + c. False if the cell is not stored.
	/// </summary>
	public abstract bool TryGetCell(int rowBlockId, int colBlockId, out Span<double> cell, out int stride);

	/// <summary>Zeroes every stored value.</summary>
	public abstract void SetZero();
}

/// <summary>ceres::internal::BlockRandomAccessDenseMatrix.</summary>
internal sealed class BlockRandomAccessDenseMatrix : BlockRandomAccessMatrix
{
	private readonly int[] blockPositions;

	/// <summary>Creates a zero matrix over the given blocks.</summary>
	public BlockRandomAccessDenseMatrix(IReadOnlyList<Block> blocks)
	{
		blockPositions = new int[blocks.Count];
		int n = 0;
		for (int i = 0; i < blocks.Count; i++)
		{
			blockPositions[i] = n;
			n += blocks[i].Size;
		}

		NumRows = n;
		Values = new double[n * n];
	}

	/// <inheritdoc/>
	public override int NumRows { get; }

	/// <summary>The row-major values.</summary>
	public double[] Values { get; }

	/// <inheritdoc/>
	public override bool TryGetCell(int rowBlockId, int colBlockId, out Span<double> cell, out int stride)
	{
		stride = NumRows;
		cell = Values.AsSpan((blockPositions[rowBlockId] * NumRows) + blockPositions[colBlockId]);
		return true;
	}

	/// <inheritdoc/>
	public override void SetZero() => Array.Clear(Values);
}

/// <summary>ceres::internal::BlockRandomAccessSparseMatrix.</summary>
internal sealed class BlockRandomAccessSparseMatrix : BlockRandomAccessMatrix
{
	private readonly Block[] blocks;
	private readonly Dictionary<long, int> cellOffsets = [];

	// Cells in (row block, column block) order: the row-major layout Ceres' underlying
	// BlockSparseMatrix uses.
	private readonly (int Row, int Col, int Offset)[] cells;

	/// <summary>
	/// Creates a zero matrix over <paramref name="blocks"/> storing the cells
	/// <paramref name="blockPairs"/> (each i &lt;= j).
	/// </summary>
	public BlockRandomAccessSparseMatrix(IReadOnlyList<Block> blocks, SortedSet<(int Row, int Col)> blockPairs)
	{
		this.blocks = [.. blocks];
		int n = 0;
		foreach (Block block in blocks)
		{
			n += block.Size;
		}

		NumRows = n;
		cells = new (int, int, int)[blockPairs.Count];
		int offset = 0;
		int k = 0;
		foreach ((int row, int col) in blockPairs)
		{
			cells[k++] = (row, col, offset);
			cellOffsets.Add(Key(row, col), offset);
			offset += blocks[row].Size * blocks[col].Size;
		}

		Values = new double[offset];
	}

	/// <inheritdoc/>
	public override int NumRows { get; }

	/// <summary>The cell values, each cell row-major.</summary>
	public double[] Values { get; }

	/// <inheritdoc/>
	public override bool TryGetCell(int rowBlockId, int colBlockId, out Span<double> cell, out int stride)
	{
		if (!cellOffsets.TryGetValue(Key(rowBlockId, colBlockId), out int offset))
		{
			cell = default;
			stride = 0;
			return false;
		}

		stride = blocks[colBlockId].Size;
		cell = Values.AsSpan(offset, blocks[rowBlockId].Size * stride);
		return true;
	}

	/// <inheritdoc/>
	public override void SetZero() => Array.Clear(Values);

	/// <summary>
	/// y += S x for the symmetric S whose upper block triangle is stored: each off-diagonal
	/// cell also does the corresponding lower-triangle multiply.
	/// </summary>
	public void SymmetricRightMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y)
	{
		foreach ((int row, int col, int offset) in cells)
		{
			Block rowBlock = blocks[row];
			Block colBlock = blocks[col];
			ReadOnlySpan<double> cell = Values.AsSpan(offset, rowBlock.Size * colBlock.Size);
			SmallBlas.MatrixVectorMultiply(
				cell, rowBlock.Size, colBlock.Size, x.Slice(colBlock.Position, colBlock.Size), y.Slice(rowBlock.Position, rowBlock.Size), 1);
			if (row != col)
			{
				SmallBlas.MatrixTransposeVectorMultiply(
					cell, rowBlock.Size, colBlock.Size, x.Slice(rowBlock.Position, rowBlock.Size), y.Slice(colBlock.Position, colBlock.Size), 1);
			}
		}
	}

	/// <summary>
	/// The lower triangle of the symmetric matrix as CSC, with, for every stored entry, the
	/// index in <see cref="Values"/> it is copied from (column c of the lower triangle is row
	/// c of the stored upper triangle).
	/// </summary>
	public SparseMatrixCsc ToLowerCsc(out int[] valueSources)
	{
		var colPtr = new int[NumRows + 1];
		var rowIndices = new List<int>();
		var sources = new List<int>();
		int cellStart = 0;
		for (int i = 0; i < blocks.Length; i++)
		{
			// The cells of block row i are contiguous in `cells`, by increasing column block.
			int cellEnd = cellStart;
			while (cellEnd < cells.Length && cells[cellEnd].Row == i)
			{
				cellEnd++;
			}

			for (int r = 0; r < blocks[i].Size; r++)
			{
				for (int p = cellStart; p < cellEnd; p++)
				{
					(_, int col, int offset) = cells[p];
					int width = blocks[col].Size;
					for (int c = col == i ? r : 0; c < width; c++)
					{
						rowIndices.Add(blocks[col].Position + c);
						sources.Add(offset + (r * width) + c);
					}
				}

				colPtr[blocks[i].Position + r + 1] = rowIndices.Count;
			}

			cellStart = cellEnd;
		}

		valueSources = [.. sources];
		int[] rows = [.. rowIndices];
		return SparseMatrixCsc.FromCsc(NumRows, NumRows, colPtr, rows, new double[rows.Length]);
	}

	private long Key(int row, int col) => ((long)row * blocks.Length) + col;
}

/// <summary>ceres::internal::BlockRandomAccessDiagonalMatrix.</summary>
internal sealed class BlockRandomAccessDiagonalMatrix : BlockRandomAccessMatrix
{
	private readonly Block[] blocks;
	private readonly int[] offsets;
	private readonly double[] scratch;

	/// <summary>Creates a zero block-diagonal matrix over <paramref name="blocks"/>.</summary>
	public BlockRandomAccessDiagonalMatrix(IReadOnlyList<Block> blocks)
	{
		this.blocks = [.. blocks];
		offsets = new int[blocks.Count];
		int offset = 0;
		int n = 0;
		int maxSize = 0;
		for (int i = 0; i < blocks.Count; i++)
		{
			offsets[i] = offset;
			offset += blocks[i].Size * blocks[i].Size;
			n += blocks[i].Size;
			maxSize = Math.Max(maxSize, blocks[i].Size);
		}

		NumRows = n;
		Values = new double[offset];
		scratch = new double[maxSize * maxSize];
	}

	/// <inheritdoc/>
	public override int NumRows { get; }

	/// <summary>The diagonal blocks, each row-major.</summary>
	public double[] Values { get; }

	/// <inheritdoc/>
	public override bool TryGetCell(int rowBlockId, int colBlockId, out Span<double> cell, out int stride)
	{
		if (rowBlockId != colBlockId)
		{
			cell = default;
			stride = 0;
			return false;
		}

		stride = blocks[rowBlockId].Size;
		cell = Values.AsSpan(offsets[rowBlockId], stride * stride);
		return true;
	}

	/// <inheritdoc/>
	public override void SetZero() => Array.Clear(Values);

	/// <summary>Replaces every block by its inverse (from its upper triangle).</summary>
	public void Invert()
	{
		for (int i = 0; i < blocks.Length; i++)
		{
			int size = blocks[i].Size;
			Span<double> cell = Values.AsSpan(offsets[i], size * size);
			SmallBlas.InvertUpperPsd(cell, size, cell, scratch);
		}
	}

	/// <summary>y += M x.</summary>
	public void RightMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y)
	{
		for (int i = 0; i < blocks.Length; i++)
		{
			Block block = blocks[i];
			SmallBlas.MatrixVectorMultiply(
				Values.AsSpan(offsets[i], block.Size * block.Size),
				block.Size,
				block.Size,
				x.Slice(block.Position, block.Size),
				y.Slice(block.Position, block.Size),
				1);
		}
	}
}
