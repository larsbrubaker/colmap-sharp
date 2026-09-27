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
// quantities with Eigen's fixed-size kernels; here SmallBlas dispatches the bundle adjustment
// shapes to unrolled kernels that keep the naive loops' summation order (SmallBlasFixed.cs).
//
// Performance without changing a bit (pinned by SchurDeterminismTests):
// - The cells of S every chunk, row and regularization touches are resolved to offsets into
//   the matrix values once per (structure, matrix) pair (CellLayout), not looked up per solve.
// - Eliminate keeps each chunk's (E'E)^-1; BackSubstitute reuses it instead of rebuilding and
//   refactoring E'E. The two computations were identical (same D, same rows in the same
//   order), so the cached inverse has the same bits.
// - Only the part of the E'F buffer a chunk uses is cleared.

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
	private double[] inverses = [];
	private CellLayout? cellLayout;
	private CompressedRowBlockStructure? structure;
	private double[] inverseScratch = [];
	private double[] g = [];
	private double[] inverseEteG = [];
	private double[] sj = [];

	// Rows of one E block: [Start, Start + Size). BufferLayout lists the F blocks the chunk
	// touches by increasing block id (Ceres' std::map iteration order) with the offset of
	// their E'F block in the buffer; CellBufferOffsets[j][c] is that offset for cell c >= 1
	// of the chunk's row j (Ceres looks it up in the map per cell).
	// BufferSize is how much of the buffer the chunk's E'F blocks use; InverseOffset is where
	// its (E'E)^-1 is kept in `inverses`.
	private sealed class Chunk(int start, int size, (int BlockId, int Offset)[] bufferLayout, int[][] cellBufferOffsets, int bufferSize, int inverseOffset)
	{
		public int Start { get; } = start;

		public int Size { get; } = size;

		public (int BlockId, int Offset)[] BufferLayout { get; } = bufferLayout;

		public int[][] CellBufferOffsets { get; } = cellBufferOffsets;

		public int BufferSize { get; } = bufferSize;

		public int InverseOffset { get; } = inverseOffset;
	}

	// A cell of S resolved to its offset in the matrix values (-1: not stored) and row stride.
	private readonly record struct CellRef(int Offset, int Stride);

	// The cells of one S the elimination writes, in the order it writes them:
	// - Diagonal[i]: cell (i, i), for the regularization.
	// - ChunkPairs[chunk]: the cells (p, q), p <= q, of the chunk's buffer layout, in the
	//   (p, q) loop order of ChunkOuterProduct.
	// - RowCells[row]: the cells (i, j), first <= i <= j, of the row's F cells in
	//   RowOuterProduct's loop order (first = 1 for chunk rows, 0 for rows with no E block).
	private sealed class CellLayout(BlockRandomAccessMatrix lhs, CellRef[] diagonal, CellRef[][] chunkPairs, CellRef[][] rowCells)
	{
		public BlockRandomAccessMatrix Lhs { get; } = lhs;

		public CellRef[] Diagonal { get; } = diagonal;

		public CellRef[][] ChunkPairs { get; } = chunkPairs;

		public CellRef[][] RowCells { get; } = rowCells;
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
		structure = bs;
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
		int inverseSize = 0;
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
			chunkList.Add(new Chunk(r, size, sortedLayout, [.. cellOffsets], chunkBufferSize, inverseSize));
			inverseSize += eBlockSize * eBlockSize;
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
		inverses = new double[inverseSize];
		cellLayout = null;
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
		Util.Check.That(ReferenceEquals(a.Structure, structure), "Eliminate needs the block structure Init was given.");
		CellLayout layout = GetCellLayout(lhs);
		double[] s = lhs.Values;
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
				CellRef cell = layout.Diagonal[i - numEliminateBlocks];
				if (cell.Offset >= 0)
				{
					for (int k = 0; k < cols[i].Size; k++)
					{
						double di = d[cols[i].Position + k];
						s[cell.Offset + (k * cell.Stride) + k] += di * di;
					}
				}
			}
		}

		// Eliminate y blocks one chunk at a time. For each chunk, compute the entries of the
		// normal equations and the gradient vector block corresponding to the y block and
		// then apply Gaussian elimination to them.
		for (int chunkIndex = 0; chunkIndex < chunks.Length; chunkIndex++)
		{
			Chunk chunk = chunks[chunkIndex];
			int eBlockId = bs.Rows[chunk.Start].Cells[0].BlockId;
			int eBlockSize = cols[eBlockId].Size;
			Array.Clear(buffer, 0, chunk.BufferSize);
			Span<double> eteBlock = ete.AsSpan(0, eBlockSize * eBlockSize);
			SetDiagonalSquares(eteBlock, eBlockSize, d, cols[eBlockId].Position);
			Span<double> gBlock = g.AsSpan(0, eBlockSize);
			gBlock.Clear();

			// We are going to be computing S += F'F - F'E(E'E)^{-1}E'F for each Chunk.
			ChunkDiagonalBlockAndGradient(chunk, a, b, eteBlock, eBlockSize, gBlock, layout, s);

			// Kept for BackSubstitute.
			Span<double> inverse = inverses.AsSpan(chunk.InverseOffset, eBlockSize * eBlockSize);
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
			ChunkOuterProduct(cols, inverse, eBlockSize, chunk.BufferLayout, layout.ChunkPairs[chunkIndex], s);
		}

		// For rows with no e_blocks, the Schur complement update reduces to S += F'F.
		NoEBlockRowsUpdate(a, b, layout, s, rhs);
	}

	/// <summary>
	/// SchurEliminator::BackSubstitute: the E variables <paramref name="y"/> (which must be
	/// zero at the E positions) from the F solution <paramref name="z"/>. It must follow an
	/// Eliminate of the same <paramref name="a"/> and <paramref name="d"/>, as it does in
	/// Ceres' SchurComplementSolver: the (E'E)^-1 blocks come from that Eliminate, which
	/// computed them from the same D and rows in the same order, so they are bit for bit the
	/// blocks Ceres rebuilds here (<paramref name="d"/> is therefore not read again).
	/// </summary>
	public void BackSubstitute(
		BlockSparseMatrix a, ReadOnlySpan<double> b, ReadOnlySpan<double> d, ReadOnlySpan<double> z, Span<double> y)
	{
		_ = d;
		CompressedRowBlockStructure bs = a.Structure;
		Util.Check.That(ReferenceEquals(bs, structure), "BackSubstitute needs the block structure Init was given.");
		double[] values = a.Values;
		foreach (Chunk chunk in chunks)
		{
			int eBlockId = bs.Rows[chunk.Start].Cells[0].BlockId;
			int eBlockSize = bs.Cols[eBlockId].Size;
			Span<double> yBlock = y.Slice(bs.Cols[eBlockId].Position, eBlockSize);
			for (int j = 0; j < chunk.Size; j++)
			{
				CompressedRow row = bs.Rows[chunk.Start + j];
				Cell eCell = row.Cells[0];
				int rowSize = row.Block.Size;
				Span<double> sRow = sj.AsSpan(0, rowSize);
				b.Slice(row.Block.Position, rowSize).CopyTo(sRow);
				for (int c = 1; c < row.Cells.Length; c++)
				{
					int fBlockId = row.Cells[c].BlockId;
					int fBlockSize = bs.Cols[fBlockId].Size;
					SmallBlas.MatrixVectorMultiply(
						values.AsSpan(row.Cells[c].Position),
						rowSize,
						fBlockSize,
						z[lhsRowLayout[fBlockId - numEliminateBlocks]..],
						sRow,
						-1);
				}

				ReadOnlySpan<double> e = values.AsSpan(eCell.Position);
				SmallBlas.MatrixTransposeVectorMultiply(e, rowSize, eBlockSize, sRow, yBlock, 1);
			}

			ReadOnlySpan<double> inverse = inverses.AsSpan(chunk.InverseOffset, eBlockSize * eBlockSize);

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

	private static CellRef Resolve(BlockRandomAccessMatrix lhs, int rowBlockId, int colBlockId)
	{
		int offset = lhs.CellOffset(rowBlockId, colBlockId, out int stride);
		return new CellRef(offset, stride);
	}

	// The cell offsets for lhs, built on its first use after Init.
	private CellLayout GetCellLayout(BlockRandomAccessMatrix lhs)
	{
		if (cellLayout is not null && ReferenceEquals(cellLayout.Lhs, lhs))
		{
			return cellLayout;
		}

		CompressedRowBlockStructure bs = structure!;
		int numFBlocks = bs.Cols.Length - numEliminateBlocks;
		var diagonal = new CellRef[numFBlocks];
		for (int i = 0; i < numFBlocks; i++)
		{
			diagonal[i] = Resolve(lhs, i, i);
		}

		var chunkPairs = new CellRef[chunks.Length][];
		for (int c = 0; c < chunks.Length; c++)
		{
			(int BlockId, int Offset)[] bufferLayout = chunks[c].BufferLayout;
			var pairs = new List<CellRef>();
			for (int p = 0; p < bufferLayout.Length; p++)
			{
				for (int q = p; q < bufferLayout.Length; q++)
				{
					pairs.Add(Resolve(lhs, bufferLayout[p].BlockId - numEliminateBlocks, bufferLayout[q].BlockId - numEliminateBlocks));
				}
			}

			chunkPairs[c] = [.. pairs];
		}

		var rowCells = new CellRef[bs.Rows.Length][];
		for (int r = 0; r < bs.Rows.Length; r++)
		{
			Cell[] cells = bs.Rows[r].Cells;
			int first = r < uneliminatedRowBegins ? 1 : 0;
			var refs = new List<CellRef>();
			for (int i = first; i < cells.Length; i++)
			{
				for (int j = i; j < cells.Length; j++)
				{
					refs.Add(Resolve(lhs, cells[i].BlockId - numEliminateBlocks, cells[j].BlockId - numEliminateBlocks));
				}
			}

			rowCells[r] = [.. refs];
		}

		cellLayout = new CellLayout(lhs, diagonal, chunkPairs, rowCells);
		return cellLayout;
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
		CellLayout layout,
		double[] s)
	{
		CompressedRowBlockStructure bs = a.Structure;
		double[] values = a.Values;
		for (int j = 0; j < chunk.Size; j++)
		{
			CompressedRow row = bs.Rows[chunk.Start + j];
			int rowSize = row.Block.Size;
			if (row.Cells.Length > 1)
			{
				RowOuterProduct(a, row, 1, layout.RowCells[chunk.Start + j], s);
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
			Span<double> sRow = sj.AsSpan(0, rowSize);
			b.Slice(row.Block.Position, rowSize).CopyTo(sRow);
			SmallBlas.MatrixVectorMultiply(values.AsSpan(row.Cells[0].Position), rowSize, eBlockSize, inverseEteG, sRow, -1);
			for (int c = 1; c < row.Cells.Length; c++)
			{
				int blockId = row.Cells[c].BlockId;
				int blockSize = bs.Cols[blockId].Size;
				SmallBlas.MatrixTransposeVectorMultiply(
					values.AsSpan(row.Cells[c].Position),
					rowSize,
					blockSize,
					sRow,
					rhs.Slice(lhsRowLayout[blockId - numEliminateBlocks], blockSize),
					1);
			}
		}
	}

	// S(i, j) -= b_i' (E'E)^-1 b_j for the chunk's F blocks i <= j, b_i its E'F blocks, whose
	// resolved cells are pairCells in that loop order. Ceres' profiling note: the bottleneck
	// here is the memory traffic into S, not the products.
	private void ChunkOuterProduct(
		Block[] cols, ReadOnlySpan<double> inverse, int eBlockSize, (int BlockId, int Offset)[] layout, CellRef[] pairCells, double[] s)
	{
		int pair = 0;
		for (int p = 0; p < layout.Length; p++)
		{
			(int blockId1, int offset1) = layout[p];
			int block1Size = cols[blockId1].Size;
			Span<double> b1TransposeInverseEte = chunkOuterProductBuffer.AsSpan(0, block1Size * eBlockSize);
			SmallBlas.MatrixTransposeMatrixMultiply(
				buffer.AsSpan(offset1), eBlockSize, block1Size, inverse, eBlockSize, b1TransposeInverseEte, 0, 0, eBlockSize, 0);
			for (int q = p; q < layout.Length; q++, pair++)
			{
				(int blockId2, int offset2) = layout[q];
				CellRef cell = pairCells[pair];
				if (cell.Offset >= 0)
				{
					SmallBlas.MatrixMatrixMultiply(
						b1TransposeInverseEte,
						block1Size,
						eBlockSize,
						buffer.AsSpan(offset2),
						cols[blockId2].Size,
						s.AsSpan(cell.Offset),
						0,
						0,
						cell.Stride,
						-1);
				}
			}
		}
	}

	// Rows with no E block: S += F'F and rhs += F'b.
	private void NoEBlockRowsUpdate(BlockSparseMatrix a, ReadOnlySpan<double> b, CellLayout layout, double[] s, Span<double> rhs)
	{
		CompressedRowBlockStructure bs = a.Structure;
		double[] values = a.Values;
		for (int r = uneliminatedRowBegins; r < bs.Rows.Length; r++)
		{
			CompressedRow row = bs.Rows[r];
			RowOuterProduct(a, row, 0, layout.RowCells[r], s);
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
	// (first = 0): S(i, j) += F_i' F_j for the row's F cells i <= j, whose resolved cells are
	// rowCells in that loop order. The diagonal block multiply ignores the symmetry of the
	// outer product, as in Ceres.
	private void RowOuterProduct(BlockSparseMatrix a, CompressedRow row, int first, CellRef[] rowCells, double[] s)
	{
		CompressedRowBlockStructure bs = a.Structure;
		double[] values = a.Values;
		Cell[] cells = row.Cells;
		int rowSize = row.Block.Size;
		int k = 0;
		for (int i = first; i < cells.Length; i++)
		{
			int block1Size = bs.Cols[cells[i].BlockId].Size;
			ReadOnlySpan<double> f1 = values.AsSpan(cells[i].Position);
			for (int j = i; j < cells.Length; j++, k++)
			{
				CellRef cell = rowCells[k];
				if (cell.Offset >= 0)
				{
					int block2Size = bs.Cols[cells[j].BlockId].Size;
					SmallBlas.MatrixTransposeMatrixMultiply(
						f1, rowSize, block1Size, values.AsSpan(cells[j].Position), block2Size, s.AsSpan(cell.Offset), 0, 0, cell.Stride, 1);
				}
			}
		}
	}
}
