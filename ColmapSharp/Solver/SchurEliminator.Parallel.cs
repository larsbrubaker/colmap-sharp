// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Written for ColmapSharp: the multithreaded chunk elimination of SchurEliminator.cs.
// Ceres (schur_eliminator_impl.h, BSD-3-Clause, see THIRD_PARTY_NOTICES.md) runs the chunks
// in parallel and locks each cell of S, so the order of additions into a cell depends on
// scheduling (docs/CPP_DIVERGENCES.md entry 35, item 1). Here the result is bit-identical to
// the sequential loop (EliminateChunksSequentially) for any thread count, because every
// value is still computed by the same operations in the same order:
// - Phase A, in parallel over the chunks of a batch: each chunk computes its E'E (with the
//   regularization), (E'E)^-1, (E'E)^-1 E'b and its E'F blocks into its own slots. None of
//   these touch S or rhs, so doing them apart from the S updates changes nothing.
// - Phase B, in parallel over contiguous ranges of F blocks p (block rows of S and rhs): a
//   worker walks the batch's chunks in order and, for each p of the chunk it owns, replays
//   exactly the updates the sequential loop makes to cells (p, *) and to rhs block p: the
//   rows' F'F products, the rows' F'(b - E (E'E)^-1 E'b) terms, then -b_p' (E'E)^-1 b_q. A
//   cell belongs to its row block, so each cell is written by one worker, in the sequential
//   order. Walking chunk-major keeps a chunk's data hot for all the blocks a worker owns;
//   a block-major replay read each chunk once per F block and was slower than one thread.
// The chunks go in batches whose E'F slots fit MaxBatchBufferSize, so the memory stays
// bounded; the batches run one after another, which keeps the per-cell order across them.

namespace ColmapSharp.Solver;

/// <summary>The two-phase parallel elimination of <see cref="SchurEliminator"/>.</summary>
internal sealed partial class SchurEliminator
{
	// For F block p: its (chunk, index in the chunk's buffer layout) entries, by chunk, at
	// [fBlockEntryStarts[p], fBlockEntryStarts[p + 1]).
	private int[] fBlockEntryStarts = [];
	private int[] fBlockEntryChunks = [];
	private int[] fBlockEntryLayoutIndices = [];

	// The entry of chunk c's layout index li is chunkEntries[chunkEntryStarts[c] + li].
	private int[] chunkEntryStarts = [];
	private int[] chunkEntries = [];

	// Phase B's workers own the F blocks [rangeStarts[t], rangeStarts[t + 1]), contiguous
	// runs of about equal work, so a chunk (whose F blocks are usually a few nearby cameras)
	// is mostly read by one or two workers while its data is hot.
	private int[] rangeStarts = [];

	// For entry k: the chunk rows with an F cell for its block, in row order, at
	// [entryRowStarts[k], entryRowStarts[k + 1]): the row, the cell's index in the row and
	// the index of the cell's first (i, j) pair in the row's CellLayout.RowCells.
	private int[] entryRowStarts = [];
	private int[] entryRows = [];
	private int[] entryCells = [];
	private int[] entryRowCellStarts = [];

	// Batches of chunks [batchStarts[k], batchStarts[k + 1]); a chunk's E'F blocks sit at
	// chunkSlots[c] in batchBuffer, its (E'E)^-1 E'b at chunkGOffsets[c] in inverseGs.
	private int[] batchStarts = [];
	private int[] chunkSlots = [];
	private int[] chunkGOffsets = [];
	private double[] batchBuffer = [];
	private double[] inverseGs = [];
	private int maxEBlockSizeForScratch;
	private int maxRowBlockSizeForScratch;
	private int maxOuterProductSize;

	// The operands of the current parallel Eliminate, as arrays (a lambda cannot capture a span).
	private BlockSparseMatrix? parallelA;
	private double[] parallelB = [];
	private double[] parallelD = [];
	private double[] parallelRhs = [];
	private bool hasB;
	private bool hasD;
	private bool hasRhs;
	private CellLayout? parallelLayout;
	private double[] parallelS = [];
	private int batchBegin;
	private int batchEnd;

	/// <summary>
	/// The most doubles of E'F blocks one batch of the parallel elimination holds; set before
	/// Init. Any value gives the same result.
	/// </summary>
	public int MaxBatchBufferSize { get; set; } = 1 << 22;

	// Sizes the parallel state after Init has found the chunks.
	private void InitParallel(int maxEBlockSize, int maxRowBlockSize, int bufferSize)
	{
		if (numThreads <= 1)
		{
			return;
		}

		maxEBlockSizeForScratch = maxEBlockSize;
		maxRowBlockSizeForScratch = maxRowBlockSize;
		maxOuterProductSize = bufferSize;
		CompressedRowBlockStructure bs = structure!;
		int numFBlocks = bs.Cols.Length - numEliminateBlocks;
		var counts = new int[numFBlocks + 1];
		foreach (Chunk chunk in chunks)
		{
			foreach ((int blockId, _) in chunk.BufferLayout)
			{
				counts[blockId - numEliminateBlocks + 1]++;
			}
		}

		for (int p = 0; p < numFBlocks; p++)
		{
			counts[p + 1] += counts[p];
		}

		fBlockEntryStarts = counts;
		fBlockEntryChunks = new int[counts[numFBlocks]];
		fBlockEntryLayoutIndices = new int[counts[numFBlocks]];
		chunkEntryStarts = new int[chunks.Length + 1];
		for (int c = 0; c < chunks.Length; c++)
		{
			chunkEntryStarts[c + 1] = chunkEntryStarts[c] + chunks[c].BufferLayout.Length;
		}

		chunkEntries = new int[chunkEntryStarts[chunks.Length]];
		int[] fill = [.. counts[..numFBlocks]];
		var entryOfBlock = new Dictionary<int, int>();
		var rowCounts = new int[counts[numFBlocks] + 1];
		var rowEntries = new List<(int Entry, int Row, int Cell, int RowCellStart)>();
		for (int c = 0; c < chunks.Length; c++)
		{
			Chunk chunk = chunks[c];
			(int BlockId, int Offset)[] layout = chunk.BufferLayout;
			entryOfBlock.Clear();
			for (int li = 0; li < layout.Length; li++)
			{
				int k = fill[layout[li].BlockId - numEliminateBlocks]++;
				fBlockEntryChunks[k] = c;
				chunkEntries[chunkEntryStarts[c] + li] = k;
				fBlockEntryLayoutIndices[k] = li;
				entryOfBlock.Add(layout[li].BlockId, k);
			}

			for (int j = 0; j < chunk.Size; j++)
			{
				Cell[] cells = bs.Rows[chunk.Start + j].Cells;
				int rowCellStart = 0;
				for (int i = 1; i < cells.Length; i++)
				{
					int k = entryOfBlock[cells[i].BlockId];
					rowCounts[k + 1]++;
					rowEntries.Add((k, chunk.Start + j, i, rowCellStart));
					rowCellStart += cells.Length - i;
				}
			}
		}

		for (int k = 0; k + 1 < rowCounts.Length; k++)
		{
			rowCounts[k + 1] += rowCounts[k];
		}

		entryRowStarts = rowCounts;
		entryRows = new int[rowEntries.Count];
		entryCells = new int[rowEntries.Count];
		entryRowCellStarts = new int[rowEntries.Count];
		int[] rowFill = [.. rowCounts[..^1]];

		// Rows are visited in chunk and row order, so each entry's rows stay in row order.
		foreach ((int entry, int row, int cell, int rowCellStart) in rowEntries)
		{
			int index = rowFill[entry]++;
			entryRows[index] = row;
			entryCells[index] = cell;
			entryRowCellStarts[index] = rowCellStart;
		}

		int numRanges = Math.Min(numFBlocks, 2 * numThreads);
		var ranges = new List<int> { 0 };
		long totalWork = entryRowStarts[^1] + fBlockEntryStarts[numFBlocks];
		long work = 0;
		for (int p = 0; p < numFBlocks; p++)
		{
			work += entryRowStarts[fBlockEntryStarts[p + 1]] - entryRowStarts[fBlockEntryStarts[p]] + fBlockEntryStarts[p + 1] - fBlockEntryStarts[p];
			if (work * numRanges >= totalWork * ranges.Count && p + 1 < numFBlocks)
			{
				ranges.Add(p + 1);
			}
		}

		ranges.Add(numFBlocks);
		rangeStarts = [.. ranges];

		var starts = new List<int> { 0 };
		chunkSlots = new int[chunks.Length];
		chunkGOffsets = new int[chunks.Length];
		int slot = 0;
		int largestBatch = 0;
		int gSize = 0;
		for (int c = 0; c < chunks.Length; c++)
		{
			if (slot > 0 && slot + chunks[c].BufferSize > MaxBatchBufferSize)
			{
				starts.Add(c);
				slot = 0;
			}

			chunkSlots[c] = slot;
			slot += chunks[c].BufferSize;
			largestBatch = Math.Max(largestBatch, slot);
			chunkGOffsets[c] = gSize;
			gSize += bs.Cols[bs.Rows[chunks[c].Start].Cells[0].BlockId].Size;
		}

		starts.Add(chunks.Length);
		batchStarts = [.. starts];
		batchBuffer = new double[largestBatch];
		inverseGs = new double[gSize];
	}

	// Phase A and B over each batch of chunks in turn.
	private void EliminateChunksInParallel(
		BlockSparseMatrix a, ReadOnlySpan<double> b, ReadOnlySpan<double> d, CellLayout layout, double[] s, Span<double> rhs)
	{
		parallelA = a;
		hasB = !b.IsEmpty;
		hasD = !d.IsEmpty;
		hasRhs = !rhs.IsEmpty;
		parallelB = CopyInto(parallelB, b);
		parallelD = CopyInto(parallelD, d);
		parallelRhs = CopyInto(parallelRhs, rhs);
		parallelLayout = layout;
		parallelS = s;
		var options = new ParallelOptions { MaxDegreeOfParallelism = numThreads };
		for (int k = 0; k + 1 < batchStarts.Length; k++)
		{
			batchBegin = batchStarts[k];
			batchEnd = batchStarts[k + 1];
			Parallel.For(
				batchBegin,
				batchEnd,
				options,
				() => new double[3 * maxEBlockSizeForScratch * maxEBlockSizeForScratch],
				(c, _, scratch) =>
				{
					ChunkPhaseA(c, scratch);
					return scratch;
				},
				_ => { });
			Parallel.For(
				0,
				rangeStarts.Length - 1,
				options,
				() => new double[maxOuterProductSize + maxRowBlockSizeForScratch],
				(p, _, scratch) =>
				{
					BlockRangePhaseB(p, scratch);
					return scratch;
				},
				_ => { });
		}

		if (hasRhs)
		{
			parallelRhs.AsSpan(0, rhs.Length).CopyTo(rhs);
		}

		parallelA = null;
		parallelLayout = null;
		parallelS = [];
	}

	private static double[] CopyInto(double[] array, ReadOnlySpan<double> values)
	{
		if (array.Length < values.Length)
		{
			array = new double[values.Length];
		}

		values.CopyTo(array);
		return array;
	}

	// E'E + D_e^2, its inverse (kept for BackSubstitute), (E'E)^-1 E'b and the E'F blocks of
	// chunk c, computed as ChunkDiagonalBlockAndGradient and the chunk loop do.
	private void ChunkPhaseA(int c, double[] scratch)
	{
		BlockSparseMatrix a = parallelA!;
		CompressedRowBlockStructure bs = a.Structure;
		double[] values = a.Values;
		Chunk chunk = chunks[c];
		Block eBlock = bs.Cols[bs.Rows[chunk.Start].Cells[0].BlockId];
		int eBlockSize = eBlock.Size;
		int eSquared = eBlockSize * eBlockSize;
		Span<double> eteBlock = scratch.AsSpan(0, eSquared);
		Span<double> inverseScratchBlock = scratch.AsSpan(eSquared, eSquared);
		Span<double> gBlock = scratch.AsSpan(2 * eSquared, eBlockSize);
		Span<double> chunkBuffer = batchBuffer.AsSpan(chunkSlots[c], chunk.BufferSize);
		chunkBuffer.Clear();
		SetDiagonalSquares(eteBlock, eBlockSize, hasD ? parallelD : [], eBlock.Position);
		gBlock.Clear();
		for (int j = 0; j < chunk.Size; j++)
		{
			CompressedRow row = bs.Rows[chunk.Start + j];
			int rowSize = row.Block.Size;
			ReadOnlySpan<double> e = values.AsSpan(row.Cells[0].Position);
			SmallBlas.MatrixTransposeMatrixMultiply(e, rowSize, eBlockSize, e, eBlockSize, eteBlock, 0, 0, eBlockSize, 1);
			if (hasB)
			{
				SmallBlas.MatrixTransposeVectorMultiply(e, rowSize, eBlockSize, parallelB.AsSpan(row.Block.Position, rowSize), gBlock, 1);
			}

			int[] offsets = chunk.CellBufferOffsets[j];
			for (int cell = 1; cell < row.Cells.Length; cell++)
			{
				int fBlockSize = bs.Cols[row.Cells[cell].BlockId].Size;
				SmallBlas.MatrixTransposeMatrixMultiply(
					e, rowSize, eBlockSize, values.AsSpan(row.Cells[cell].Position), fBlockSize, chunkBuffer[offsets[cell]..], 0, 0, fBlockSize, 1);
			}
		}

		Span<double> inverse = inverses.AsSpan(chunk.InverseOffset, eSquared);
		SmallBlas.InvertUpperPsd(eteBlock, eBlockSize, inverse, inverseScratchBlock);
		if (hasRhs)
		{
			SmallBlas.MatrixVectorMultiply(inverse, eBlockSize, eBlockSize, gBlock, inverseGs.AsSpan(chunkGOffsets[c], eBlockSize), 0);
		}
	}

	// Block rows [rangeStarts[t], rangeStarts[t + 1]) of S and rhs: the updates of the
	// batch's chunks, chunk by chunk in order.
	private void BlockRangePhaseB(int t, double[] scratch)
	{
		int first = rangeStarts[t] + numEliminateBlocks;
		int end = rangeStarts[t + 1] + numEliminateBlocks;
		for (int c = batchBegin; c < batchEnd; c++)
		{
			(int BlockId, int Offset)[] layout = chunks[c].BufferLayout;
			for (int li = 0; li < layout.Length; li++)
			{
				int blockId = layout[li].BlockId;
				if (blockId >= first && blockId < end)
				{
					ReplayChunkForBlockRow(blockId - numEliminateBlocks, chunkEntries[chunkEntryStarts[c] + li], scratch);
				}
			}
		}
	}

	private void ReplayChunkForBlockRow(int p, int entry, double[] scratch)
	{
		int c = fBlockEntryChunks[entry];
		int layoutIndex = fBlockEntryLayoutIndices[entry];
		BlockSparseMatrix a = parallelA!;
		CompressedRowBlockStructure bs = a.Structure;
		Block[] cols = bs.Cols;
		double[] values = a.Values;
		double[] s = parallelS;
		CellLayout layout = parallelLayout!;
		Chunk chunk = chunks[c];
		int eBlockSize = cols[bs.Rows[chunk.Start].Cells[0].BlockId].Size;
		int blockIdP = p + numEliminateBlocks;
		int blockSizeP = cols[blockIdP].Size;

		// S(p, *) += F_p' F_j over the chunk's rows (EBlockRowOuterProduct), and rhs block p
		// += F_p' (b - E (E'E)^-1 E'b) (UpdateRhs).
		for (int r = entryRowStarts[entry]; r < entryRowStarts[entry + 1]; r++)
		{
			CompressedRow row = bs.Rows[entryRows[r]];
			Cell[] cells = row.Cells;
			int rowSize = row.Block.Size;
			CellRef[] rowCells = layout.RowCells[entryRows[r]];
			int i = entryCells[r];
			int refIndex = entryRowCellStarts[r];
			ReadOnlySpan<double> f1 = values.AsSpan(cells[i].Position);
			for (int j2 = i; j2 < cells.Length; j2++, refIndex++)
			{
				CellRef cell = rowCells[refIndex];
				if (cell.Offset >= 0)
				{
					SmallBlas.MatrixTransposeMatrixMultiply(
						f1, rowSize, blockSizeP, values.AsSpan(cells[j2].Position), cols[cells[j2].BlockId].Size, s.AsSpan(cell.Offset), 0, 0, cell.Stride, 1);
				}
			}

			if (hasRhs)
			{
				Span<double> sRow = scratch.AsSpan(maxOuterProductSize, rowSize);
				parallelB.AsSpan(row.Block.Position, rowSize).CopyTo(sRow);
				SmallBlas.MatrixVectorMultiply(
					values.AsSpan(cells[0].Position), rowSize, eBlockSize, inverseGs.AsSpan(chunkGOffsets[c], eBlockSize), sRow, -1);
				SmallBlas.MatrixTransposeVectorMultiply(
					f1, rowSize, blockSizeP, sRow, parallelRhs.AsSpan(lhsRowLayout[p], blockSizeP), 1);
			}
		}

		// S(p, q) -= b_p' (E'E)^-1 b_q for the chunk's F blocks q >= p (ChunkOuterProduct).
		(int BlockId, int Offset)[] bufferLayout = chunk.BufferLayout;
		ReadOnlySpan<double> chunkBuffer = batchBuffer.AsSpan(chunkSlots[c], chunk.BufferSize);
		ReadOnlySpan<double> inverse = inverses.AsSpan(chunk.InverseOffset, eBlockSize * eBlockSize);
		Span<double> b1TransposeInverseEte = scratch.AsSpan(0, blockSizeP * eBlockSize);
		SmallBlas.MatrixTransposeMatrixMultiply(
			chunkBuffer[bufferLayout[layoutIndex].Offset..], eBlockSize, blockSizeP, inverse, eBlockSize, b1TransposeInverseEte, 0, 0, eBlockSize, 0);
		CellRef[] pairCells = layout.ChunkPairs[c];
		int n = bufferLayout.Length;
		int pair = (layoutIndex * n) - (layoutIndex * (layoutIndex - 1) / 2);
		for (int q = layoutIndex; q < n; q++, pair++)
		{
			CellRef cell = pairCells[pair];
			if (cell.Offset >= 0)
			{
				SmallBlas.MatrixMatrixMultiply(
					b1TransposeInverseEte,
					blockSizeP,
					eBlockSize,
					chunkBuffer[bufferLayout[q].Offset..],
					cols[bufferLayout[q].BlockId].Size,
					s.AsSpan(cell.Offset),
					0,
					0,
					cell.Stride,
					-1);
			}
		}
	}
}
