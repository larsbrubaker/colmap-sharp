// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/schur_eliminator.h and
// internal/ceres/schur_eliminator_impl.h (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Eliminates the E blocks (in bundle adjustment: the points) from the regularized normal
// equations of a block-sparse Jacobian A = [E F]:
//
//   S   = F'F - F'E (E'E)^-1 E'F          (the reduced camera matrix)
//   rhs = F'b - F'E (E'E)^-1 E'b
//
// and recovers the E variables from the F solution (back substitution). The Jacobian's rows
// must be ordered so that all rows of one E block are contiguous and come before the rows
// with no E block (SchurOrdering.cs does that); each such run is a "chunk". The diagonal
// regularization D is folded into both E'E and S.
//
// Ceres eliminates the chunks in parallel and serializes the updates of S with a mutex per
// cell, so the order in which chunks add into a cell (and so the last bits of S) depends on
// scheduling. Here the chunks run sequentially in order, which is exactly Ceres'
// single-threaded order; the result does not depend on the thread count
// (docs/CPP_DIVERGENCES.md entries 18 and 35). Only the dynamic-size eliminator is ported:
// Ceres' template specializations and SchurEliminatorForOneFBlock<2, 3, 6> compute the same
// quantities with Eigen's fixed-size kernels.

namespace ColmapSharp.Solver;

/// <summary>ceres::internal::SchurEliminator (dynamic block sizes).</summary>
internal sealed class SchurEliminator
{
	private int numEliminateBlocks;
	private int[] lhsRowLayout = [];
	private Chunk[] chunks = [];
	private int uneliminatedRowBegins;
	private double[] buffer = [];
	private double[] chunkOuterProductBuffer = [];
	private double[] ete = [];
	private double[] inverseEte = [];
	private double[] inverseScratch = [];
	private double[] g = [];
	private double[] inverseEteG = [];
	private double[] sj = [];

	// Rows of one E block: [Start, Start + Size). BufferLayout lists the F blocks the chunk
	// touches by increasing block id (Ceres' std::map iteration order) with the offset of
	// their E'F block in the buffer; CellBufferOffsets[j][c] is that offset for cell c >= 1
	// of the chunk's row j (Ceres looks it up in the map per cell).
	private sealed class Chunk(int start, int size, (int BlockId, int Offset)[] bufferLayout, int[][] cellBufferOffsets)
	{
		public int Start { get; } = start;

		public int Size { get; } = size;

		public (int BlockId, int Offset)[] BufferLayout { get; } = bufferLayout;

		public int[][] CellBufferOffsets { get; } = cellBufferOffsets;
	}

	/// <summary>The F column blocks of <paramref name="bs"/>, repositioned from 0 (the reduced system's layout).</summary>
	public static Block[] ReducedBlocks(CompressedRowBlockStructure bs, int numEliminateBlocks)
	{
		var blocks = new Block[bs.Cols.Length - numEliminateBlocks];
		for (int i = 0, position = 0; i < blocks.Length; i++)
		{
			blocks[i] = new Block(bs.Cols[numEliminateBlocks + i].Size, position);
			position += blocks[i].Size;
		}

		return blocks;
	}

	/// <summary>
	/// SchurEliminator::Init: detects the chunks of <paramref name="bs"/> and sizes the
	/// buffers. E'E is always inverted by Cholesky (Ceres' assume_full_rank_ete, which every
	/// caller sets).
	/// </summary>
	public void Init(int numEliminateBlocks, CompressedRowBlockStructure bs)
	{
		Util.Check.Gt(numEliminateBlocks, 0, "SchurComplementSolver cannot be initialized with num_eliminate_blocks = 0.");
		this.numEliminateBlocks = numEliminateBlocks;
		Block[] cols = bs.Cols;
		CompressedRow[] rows = bs.Rows;
		lhsRowLayout = new int[cols.Length - numEliminateBlocks];
		for (int i = numEliminateBlocks, lhsNumRows = 0; i < cols.Length; i++)
		{
			lhsRowLayout[i - numEliminateBlocks] = lhsNumRows;
			lhsNumRows += cols[i].Size;
		}

		int bufferSize = 1;
		int maxEBlockSize = 1;
		int maxRowBlockSize = 1;
		var chunkList = new List<Chunk>();
		int r = 0;
		while (r < rows.Length)
		{
			int chunkBlockId = rows[r].Cells[0].BlockId;
			if (chunkBlockId >= numEliminateBlocks)
			{
				break;
			}

			int eBlockSize = cols[chunkBlockId].Size;
			maxEBlockSize = Math.Max(maxEBlockSize, eBlockSize);
			var layout = new Dictionary<int, int>();
			var cellOffsets = new List<int[]>();
			int chunkBufferSize = 0;
			int size = 0;
			while (r + size < rows.Length && rows[r + size].Cells[0].BlockId == chunkBlockId)
			{
				// Iterate over the blocks in the row, ignoring the first block since it is
				// the one to be eliminated.
				Cell[] cells = rows[r + size].Cells;
				maxRowBlockSize = Math.Max(maxRowBlockSize, rows[r + size].Block.Size);
				var offsets = new int[cells.Length];
				for (int c = 1; c < cells.Length; c++)
				{
					if (!layout.TryGetValue(cells[c].BlockId, out int offset))
					{
						offset = chunkBufferSize;
						layout.Add(cells[c].BlockId, offset);
						chunkBufferSize += eBlockSize * cols[cells[c].BlockId].Size;
					}

					offsets[c] = offset;
				}

				bufferSize = Math.Max(bufferSize, chunkBufferSize);
				cellOffsets.Add(offsets);
				size++;
			}

			(int, int)[] sortedLayout = [.. layout.Select(p => (p.Key, p.Value)).OrderBy(p => p.Key)];
			chunkList.Add(new Chunk(r, size, sortedLayout, [.. cellOffsets]));
			r += size;
		}

		foreach (CompressedRow row in rows)
		{
			maxRowBlockSize = Math.Max(maxRowBlockSize, row.Block.Size);
		}

		chunks = [.. chunkList];
		Util.Check.That(chunks.Length > 0, "The Jacobian has no rows with an E block.");
		uneliminatedRowBegins = chunks[^1].Start + chunks[^1].Size;
		buffer = new double[bufferSize];

		// chunk_outer_product_buffer_ only needs to store e_block_size * f_block_size, which
		// is always less than buffer_size_.
		chunkOuterProductBuffer = new double[bufferSize];
		ete = new double[maxEBlockSize * maxEBlockSize];
		inverseEte = new double[maxEBlockSize * maxEBlockSize];
		inverseScratch = new double[maxEBlockSize * maxEBlockSize];
		g = new double[maxEBlockSize];
		inverseEteG = new double[maxEBlockSize];
		sj = new double[maxRowBlockSize];
	}

	/// <summary>
	/// SchurEliminator::Eliminate: the upper block triangle of S into <paramref name="lhs"/>
	/// (only the cells it stores) and, unless <paramref name="rhs"/> is empty, the reduced
	/// right-hand side. <paramref name="b"/> may be empty when <paramref name="rhs"/> is;
	/// an empty <paramref name="d"/> means no regularization.
	/// </summary>
	public void Eliminate(
		BlockSparseMatrix a, ReadOnlySpan<double> b, ReadOnlySpan<double> d, BlockRandomAccessMatrix lhs, Span<double> rhs)
	{
		if (lhs.NumRows > 0)
		{
			lhs.SetZero();
			rhs.Clear();
		}

		CompressedRowBlockStructure bs = a.Structure;
		Block[] cols = bs.Cols;

		// Add the diagonal to the schur complement.
		if (!d.IsEmpty)
		{
			for (int i = numEliminateBlocks; i < cols.Length; i++)
			{
				int blockId = i - numEliminateBlocks;
				if (lhs.TryGetCell(blockId, blockId, out Span<double> cell, out int stride))
				{
					for (int k = 0; k < cols[i].Size; k++)
					{
						double di = d[cols[i].Position + k];
						cell[(k * stride) + k] += di * di;
					}
				}
			}
		}

		// Eliminate y blocks one chunk at a time. For each chunk, compute the entries of the
		// normal equations and the gradient vector block corresponding to the y block and
		// then apply Gaussian elimination to them.
		foreach (Chunk chunk in chunks)
		{
			int eBlockId = bs.Rows[chunk.Start].Cells[0].BlockId;
			int eBlockSize = cols[eBlockId].Size;
			Array.Clear(buffer);
			Span<double> eteBlock = ete.AsSpan(0, eBlockSize * eBlockSize);
			SetDiagonalSquares(eteBlock, eBlockSize, d, cols[eBlockId].Position);
			Span<double> gBlock = g.AsSpan(0, eBlockSize);
			gBlock.Clear();

			// We are going to be computing S += F'F - F'E(E'E)^{-1}E'F for each Chunk.
			ChunkDiagonalBlockAndGradient(chunk, a, b, eteBlock, eBlockSize, gBlock, lhs);

			Span<double> inverse = inverseEte.AsSpan(0, eBlockSize * eBlockSize);
			SmallBlas.InvertUpperPsd(eteBlock, eBlockSize, inverse, inverseScratch);

			// For the current chunk compute and update the rhs of the reduced linear system:
			// rhs = F'b - F'E(E'E)^(-1) E'b.
			if (!rhs.IsEmpty)
			{
				Span<double> inverseG = inverseEteG.AsSpan(0, eBlockSize);
				SmallBlas.MatrixVectorMultiply(inverse, eBlockSize, eBlockSize, gBlock, inverseG, 0);
				UpdateRhs(chunk, a, b, inverseG, eBlockSize, rhs);
			}

			// S -= F'E(E'E)^{-1}E'F
			ChunkOuterProduct(cols, inverse, eBlockSize, chunk.BufferLayout, lhs);
		}

		// For rows with no e_blocks, the Schur complement update reduces to S += F'F.
		NoEBlockRowsUpdate(a, b, lhs, rhs);
	}

	/// <summary>
	/// SchurEliminator::BackSubstitute: the E variables <paramref name="y"/> (which must be
	/// zero at the E positions) from the F solution <paramref name="z"/>.
	/// </summary>
	public void BackSubstitute(
		BlockSparseMatrix a, ReadOnlySpan<double> b, ReadOnlySpan<double> d, ReadOnlySpan<double> z, Span<double> y)
	{
		CompressedRowBlockStructure bs = a.Structure;
		double[] values = a.Values;
		foreach (Chunk chunk in chunks)
		{
			int eBlockId = bs.Rows[chunk.Start].Cells[0].BlockId;
			int eBlockSize = bs.Cols[eBlockId].Size;
			Span<double> yBlock = y.Slice(bs.Cols[eBlockId].Position, eBlockSize);
			Span<double> eteBlock = ete.AsSpan(0, eBlockSize * eBlockSize);
			SetDiagonalSquares(eteBlock, eBlockSize, d, bs.Cols[eBlockId].Position);
			for (int j = 0; j < chunk.Size; j++)
			{
				CompressedRow row = bs.Rows[chunk.Start + j];
				Cell eCell = row.Cells[0];
				int rowSize = row.Block.Size;
				Span<double> s = sj.AsSpan(0, rowSize);
				b.Slice(row.Block.Position, rowSize).CopyTo(s);
				for (int c = 1; c < row.Cells.Length; c++)
				{
					int fBlockId = row.Cells[c].BlockId;
					int fBlockSize = bs.Cols[fBlockId].Size;
					SmallBlas.MatrixVectorMultiply(
						values.AsSpan(row.Cells[c].Position),
						rowSize,
						fBlockSize,
						z[lhsRowLayout[fBlockId - numEliminateBlocks]..],
						s,
						-1);
				}

				ReadOnlySpan<double> e = values.AsSpan(eCell.Position);
				SmallBlas.MatrixTransposeVectorMultiply(e, rowSize, eBlockSize, s, yBlock, 1);
				SmallBlas.MatrixTransposeMatrixMultiply(e, rowSize, eBlockSize, e, eBlockSize, eteBlock, 0, 0, eBlockSize, 1);
			}

			Span<double> inverse = inverseEte.AsSpan(0, eBlockSize * eBlockSize);
			SmallBlas.InvertUpperPsd(eteBlock, eBlockSize, inverse, inverseScratch);

			// y_block = inverse * y_block, through a temporary as Eigen evaluates products.
			Span<double> product = inverseEteG.AsSpan(0, eBlockSize);
			SmallBlas.MatrixVectorMultiply(inverse, eBlockSize, eBlockSize, yBlock, product, 0);
			product.CopyTo(yBlock);
		}
	}

	// ete = diag(D_e)^2, or zero without regularization.
	private static void SetDiagonalSquares(Span<double> ete, int size, ReadOnlySpan<double> d, int position)
	{
		ete.Clear();
		if (!d.IsEmpty)
		{
			for (int k = 0; k < size; k++)
			{
				double dk = d[position + k];
				ete[(k * size) + k] = dk * dk;
			}
		}
	}

	// For the chunk's rows: S += F'F (EBlockRowOuterProduct), ete += E'E, g += E'b and the
	// compressed E'F blocks into the buffer.
	private void ChunkDiagonalBlockAndGradient(
		Chunk chunk,
		BlockSparseMatrix a,
		ReadOnlySpan<double> b,
		Span<double> eteBlock,
		int eBlockSize,
		Span<double> gBlock,
		BlockRandomAccessMatrix lhs)
	{
		CompressedRowBlockStructure bs = a.Structure;
		double[] values = a.Values;
		for (int j = 0; j < chunk.Size; j++)
		{
			CompressedRow row = bs.Rows[chunk.Start + j];
			int rowSize = row.Block.Size;
			if (row.Cells.Length > 1)
			{
				RowOuterProduct(a, row, 1, lhs);
			}

			// Extract the e_block, ETE += E_i' E_i
			ReadOnlySpan<double> e = values.AsSpan(row.Cells[0].Position);
			SmallBlas.MatrixTransposeMatrixMultiply(e, rowSize, eBlockSize, e, eBlockSize, eteBlock, 0, 0, eBlockSize, 1);

			if (!b.IsEmpty)
			{
				// g += E_i' b_i
				SmallBlas.MatrixTransposeVectorMultiply(e, rowSize, eBlockSize, b.Slice(row.Block.Position, rowSize), gBlock, 1);
			}

			// buffer = E'F. This computation is done by iterating over the f_blocks for each
			// row in the chunk.
			int[] offsets = chunk.CellBufferOffsets[j];
			for (int c = 1; c < row.Cells.Length; c++)
			{
				int fBlockSize = bs.Cols[row.Cells[c].BlockId].Size;
				SmallBlas.MatrixTransposeMatrixMultiply(
					e,
					rowSize,
					eBlockSize,
					values.AsSpan(row.Cells[c].Position),
					fBlockSize,
					buffer.AsSpan(offsets[c]),
					0,
					0,
					fBlockSize,
					1);
			}
		}
	}

	// rhs += F' (b - E (E'E)^-1 E'b), row by row.
	private void UpdateRhs(
		Chunk chunk, BlockSparseMatrix a, ReadOnlySpan<double> b, ReadOnlySpan<double> inverseEteG, int eBlockSize, Span<double> rhs)
	{
		CompressedRowBlockStructure bs = a.Structure;
		double[] values = a.Values;
		for (int j = 0; j < chunk.Size; j++)
		{
			CompressedRow row = bs.Rows[chunk.Start + j];
			int rowSize = row.Block.Size;
			Span<double> s = sj.AsSpan(0, rowSize);
			b.Slice(row.Block.Position, rowSize).CopyTo(s);
			SmallBlas.MatrixVectorMultiply(values.AsSpan(row.Cells[0].Position), rowSize, eBlockSize, inverseEteG, s, -1);
			for (int c = 1; c < row.Cells.Length; c++)
			{
				int blockId = row.Cells[c].BlockId;
				int blockSize = bs.Cols[blockId].Size;
				SmallBlas.MatrixTransposeVectorMultiply(
					values.AsSpan(row.Cells[c].Position),
					rowSize,
					blockSize,
					s,
					rhs.Slice(lhsRowLayout[blockId - numEliminateBlocks], blockSize),
					1);
			}
		}
	}

	// S(i, j) -= b_i' (E'E)^-1 b_j for the chunk's F blocks i <= j, b_i its E'F blocks.
	// Ceres' profiling note: the bottleneck here is the memory traffic into S, not the
	// products.
	private void ChunkOuterProduct(
		Block[] cols, ReadOnlySpan<double> inverse, int eBlockSize, (int BlockId, int Offset)[] layout, BlockRandomAccessMatrix lhs)
	{
		for (int p = 0; p < layout.Length; p++)
		{
			(int blockId1, int offset1) = layout[p];
			int block1 = blockId1 - numEliminateBlocks;
			int block1Size = cols[blockId1].Size;
			Span<double> b1TransposeInverseEte = chunkOuterProductBuffer.AsSpan(0, block1Size * eBlockSize);
			SmallBlas.MatrixTransposeMatrixMultiply(
				buffer.AsSpan(offset1), eBlockSize, block1Size, inverse, eBlockSize, b1TransposeInverseEte, 0, 0, eBlockSize, 0);
			for (int q = p; q < layout.Length; q++)
			{
				(int blockId2, int offset2) = layout[q];
				int block2 = blockId2 - numEliminateBlocks;
				if (lhs.TryGetCell(block1, block2, out Span<double> cell, out int stride))
				{
					SmallBlas.MatrixMatrixMultiply(
						b1TransposeInverseEte,
						block1Size,
						eBlockSize,
						buffer.AsSpan(offset2),
						cols[blockId2].Size,
						cell,
						0,
						0,
						stride,
						-1);
				}
			}
		}
	}

	// Rows with no E block: S += F'F and rhs += F'b.
	private void NoEBlockRowsUpdate(BlockSparseMatrix a, ReadOnlySpan<double> b, BlockRandomAccessMatrix lhs, Span<double> rhs)
	{
		CompressedRowBlockStructure bs = a.Structure;
		double[] values = a.Values;
		for (int r = uneliminatedRowBegins; r < bs.Rows.Length; r++)
		{
			CompressedRow row = bs.Rows[r];
			RowOuterProduct(a, row, 0, lhs);
			if (rhs.IsEmpty)
			{
				continue;
			}

			foreach (Cell cell in row.Cells)
			{
				int blockSize = bs.Cols[cell.BlockId].Size;
				SmallBlas.MatrixTransposeVectorMultiply(
					values.AsSpan(cell.Position),
					row.Block.Size,
					blockSize,
					b.Slice(row.Block.Position, row.Block.Size),
					rhs.Slice(lhsRowLayout[cell.BlockId - numEliminateBlocks], blockSize),
					1);
			}
		}
	}

	// EBlockRowOuterProduct (first = 1, skipping the E cell) and NoEBlockRowOuterProduct
	// (first = 0): S(i, j) += F_i' F_j for the row's F cells i <= j. The diagonal block
	// multiply ignores the symmetry of the outer product, as in Ceres.
	private void RowOuterProduct(BlockSparseMatrix a, CompressedRow row, int first, BlockRandomAccessMatrix lhs)
	{
		CompressedRowBlockStructure bs = a.Structure;
		double[] values = a.Values;
		Cell[] cells = row.Cells;
		int rowSize = row.Block.Size;
		for (int i = first; i < cells.Length; i++)
		{
			int block1 = cells[i].BlockId - numEliminateBlocks;
			int block1Size = bs.Cols[cells[i].BlockId].Size;
			ReadOnlySpan<double> f1 = values.AsSpan(cells[i].Position);
			if (lhs.TryGetCell(block1, block1, out Span<double> diagonal, out int diagonalStride))
			{
				SmallBlas.MatrixTransposeMatrixMultiply(f1, rowSize, block1Size, f1, block1Size, diagonal, 0, 0, diagonalStride, 1);
			}

			for (int j = i + 1; j < cells.Length; j++)
			{
				int block2 = cells[j].BlockId - numEliminateBlocks;
				if (lhs.TryGetCell(block1, block2, out Span<double> cell, out int stride))
				{
					int block2Size = bs.Cols[cells[j].BlockId].Size;
					SmallBlas.MatrixTransposeMatrixMultiply(
						f1, rowSize, block1Size, values.AsSpan(cells[j].Position), block2Size, cell, 0, 0, stride, 1);
				}
			}
		}
	}
}
