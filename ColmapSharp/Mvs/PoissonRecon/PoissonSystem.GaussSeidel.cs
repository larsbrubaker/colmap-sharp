// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonSystem.GaussSeidel: the multigrid solver's relaxation at one depth
// (thirdparty/PoissonRecon/FEMTree.System.inl: _solveSystemGS with sliced = true, i.e.
// _solveSlicedSystemGS, and _setMultiColorIndices; SparseMatrixInterface.inl: the
// multi-colored gsIteration). The depth is cut into blocks of sliceBlockSize z-slices; a moving
// window builds each block's rows (GetSliceMatrixAndProlongationConstraints, in PoissonSystem.cs)
// just before they are first relaxed and sweeps the in-memory blocks, so every block gets
// `iters` multi-colored Gauss-Seidel sweeps in the same temporally blocked order as upstream.
// Tier A against oracle/poisson_system_harness.cc (the "gs*" cases).
//
// Translation notes:
// - Residual norms (computeNorms, only for showResidual output) are not computed: Solve runs
//   with showResidual off, and nothing in the port reads them.
// - The C++ relaxes the rows of one color with ThreadPool::ParallelFor. With degree-1
//   functions two nodes of the same color (offsets equal mod 2 on every axis) are never in each
//   other's 3x3x3 row, so the order within a color cannot change the result; the port relaxes
//   them in index order.
// - PR_MODULO( a , b ) is ported as Modulo, including its a == 0 branch.

namespace ColmapSharp.Mvs.PoissonRecon;

public sealed partial class PoissonSystem
{
	// Moduli per axis of the multi-coloring: 1 - OverlapStart (2 for degree 1), so 8 colors.
	private const int ColorModulusPerAxis = 2;

	/// <summary>
	/// Relaxes the system at <paramref name="depth"/> in place in <paramref name="solution"/>
	/// (indexed by node index): each row's right-hand side is <paramref name="constraints"/>
	/// (indexed by node index) minus what the prolonged coarser solution meets, and each block of
	/// <paramref name="sliceBlockSize"/> slices gets <paramref name="iters"/> multi-colored
	/// Gauss-Seidel sweeps, walking the blocks backward and the colors forward when
	/// <paramref name="coarseToFine"/> (the prolongation phase), as upstream. The inverse diagonal
	/// is scaled by <paramref name="sorWeight"/>( node index ) (Solve's SOR functions return 1).
	/// <c>f</c> must be initialized at the depth. Returns <paramref name="iters"/>. Port of
	/// <c>_solveSystemGS</c> (sliced) without residual norms.
	/// </summary>
	public int SolveSystemGS(int depth, float[] solution, float[]? prolongedSolution, float[] constraints, int iters, bool coarseToFine, int sliceBlockSize, Func<int, float>? sorWeight = null)
	{
		if (sliceBlockSize <= 0)
		{
			throw new ArgumentOutOfRangeException(nameof(sliceBlockSize), "Only the sliced solver (sliceBlockSize >= 1; Solve uses 1) is ported.");
		}

		double[] ccStencil = f.SetStencil();
		double[][] pcStencils = f.SetParentChildStencils();
		BSplineEvaluationData evaluation = BSplineEvaluationData.For(f.Signature);
		int overlapBlockRadius = (overlapRadius + sliceBlockSize - 1) / sliceBlockSize;
		int sliceBegin = evaluation.BeginAt(depth), sliceEnd = evaluation.EndAt(depth);
		int blockBegin = (sliceBegin - (sliceBlockSize - 1)) / sliceBlockSize, blockEnd = (sliceEnd + (sliceBlockSize - 1)) / sliceBlockSize;
		int BlockFirst(int b) => StdMinMax.StdMax(b * sliceBlockSize, sliceBegin);
		int BlockLast(int b) => StdMinMax.StdMin(b * sliceBlockSize + sliceBlockSize - 1, sliceEnd - 1);
		int global = depth + tree.DepthOffset, inset = tree.LocalInset(depth);
		int SliceBegin(int slice) => sorted.Begin(global, slice + inset);
		int SliceEnd(int slice) => sorted.End(global, slice + inset);

		bool forward = !coarseToFine;
		const int residualOffset = 0;

		// Set the number of in-memory blocks required for a temporally blocked solver
		int colorModulus = overlapBlockRadius;

		// The number of in-core blocks over which we relax
		// [WARNING] If the block size is larger than one, we may be able to use fewer blocks
		int solveBlocks = StdMinMax.StdMax(0, StdMinMax.StdMin(colorModulus * iters - (colorModulus - 1), blockEnd - blockBegin));

		// The number of in-core blocks over which we either solve or compute residuals
		int matrixBlocks = StdMinMax.StdMax(1, StdMinMax.StdMin(solveBlocks + 2 * residualOffset, blockEnd - blockBegin));
		int dir = forward ? 1 : -1;
		var fullWindow = new BlockWindow(blockBegin, blockEnd);
		var residualWindow = new BlockWindow(fullWindow.Begin(forward), fullWindow.Begin(forward) - (colorModulus * iters - (colorModulus - 1)) * dir - 2 * residualOffset * dir);
		var solveWindow = new BlockWindow(fullWindow.Begin(forward) - residualOffset * dir, fullWindow.Begin(forward) - residualOffset * dir - (colorModulus * iters - (colorModulus - 1)) * dir);

		// If we are solving forward we start in a block S with S mod ColorModulus = ColorModulus-1
		// and end in a block E with E mod ColorModulus = 0
		while (Modulo(solveWindow.Begin(!forward), colorModulus) != (forward ? colorModulus - 1 : 0))
		{
			solveWindow.Shift(-dir);
			residualWindow.Shift(-dir);
		}

		int maxBlockSize = 0;
		for (BlockWindow w = residualWindow; w.End(!forward) * dir < fullWindow.End(forward) * dir; w.Shift(dir))
		{
			int b = w.Begin(!forward);
			if (fullWindow.InBlock(b))
			{
				maxBlockSize = StdMinMax.StdMax(maxBlockSize, SliceEnd(BlockLast(b)) - SliceBegin(BlockFirst(b)));
			}
		}

		var matrices = new PoissonSystemMatrix[matrixBlocks];
		var diagonals = new float[matrixBlocks][];
		var blockConstraints = new float[matrixBlocks][];
		for (int i = 0; i < matrixBlocks; i++)
		{
			matrices[i] = new PoissonSystemMatrix(neighbors.Length);
			diagonals[i] = new float[maxBlockSize];
			blockConstraints[i] = new float[maxBlockSize];
		}

		var mcIndices = new List<int>[solveBlocks][];
		for (int i = 0; i < solveBlocks; i++)
		{
			mcIndices[i] = new List<int>[ColorModulusPerAxis * ColorModulusPerAxis * ColorModulusPerAxis];
			for (int c = 0; c < mcIndices[i].Length; c++)
			{
				mcIndices[i][c] = [];
			}
		}

		for (; residualWindow.End(!forward) * dir < fullWindow.End(forward) * dir; residualWindow.Shift(dir), solveWindow.Shift(dir))
		{
			int frontSolveBlock = solveWindow.Begin(!forward);
			int residualBlock = residualWindow.Begin(!forward);

			// Get the leading matrix
			if (fullWindow.InBlock(residualBlock))
			{
				int b = residualBlock, slot = Modulo(b, matrixBlocks);
				int begin = SliceBegin(BlockFirst(b)), end = SliceEnd(BlockLast(b));
				float[] blockB = blockConstraints[slot], blockD = diagonals[slot];
				GetSliceMatrixAndProlongationConstraints(matrices[slot], blockD, depth, begin, end, prolongedSolution, blockB, ccStencil, pcStencils);
				for (int i = begin; i < end; i++)
				{
					blockB[i - begin] = constraints[i] - blockB[i - begin];
				}

				for (int i = begin; i < end; i++)
				{
					if (matrices[slot].RowSize(i - begin) != 0)
					{
						blockD[i - begin] *= sorWeight?.Invoke(i) ?? 1f;
					}
				}
			}

			// Get the leading multi-color indices
			if (iters != 0 && fullWindow.InBlock(frontSolveBlock))
			{
				int b = frontSolveBlock;
				List<int>[] colors = mcIndices[Modulo(b, solveBlocks)];
				foreach (List<int> color in colors)
				{
					color.Clear();
				}

				SetMultiColorIndices(SliceBegin(BlockFirst(b)), SliceEnd(BlockLast(b)), colors);
			}

			// Relax the system
			for (int block = solveWindow.Begin(!forward); solveWindow.InBlock(block); block -= dir * colorModulus)
			{
				if (fullWindow.InBlock(block))
				{
					int slot = Modulo(block, matrixBlocks);
					GsIteration(matrices[slot], mcIndices[Modulo(block, solveBlocks)], diagonals[slot], blockConstraints[slot], solution, SliceBegin(BlockFirst(block)), coarseToFine);
				}
			}
		}

		return iters;
	}

	// PR_MODULO( a , b ): a > 0 ? a % b : ( b - ( -a % b ) ) % b.
	private static int Modulo(int a, int b) => a > 0 ? a % b : (b - (-a % b)) % b;

	// SparseMatrixInterface::gsIteration( multiColorIndices , diagonal , b , x , forward , dReciprocal = true ):
	// per color (in order, or reversed), x[j] += ( b[j] - row j . x ) * diagonal[j]. Row columns
	// and the block's b/diagonal are relative to xOffset.
	private static void GsIteration(PoissonSystemMatrix m, List<int>[] colors, float[] diagonal, float[] b, float[] x, int xOffset, bool forward)
	{
		for (int c = 0; c < colors.Length; c++)
		{
			List<int> indices = colors[forward ? c : colors.Length - 1 - c];
			for (int k = 0; k < indices.Count; k++)
			{
				int jj = indices[k];
				float residual = b[jj];
				int size = m.RowSize(jj);
				for (int e = 0; e < size; e++)
				{
					residual -= x[xOffset + m.Column(jj, e)] * m.Value(jj, e);
				}

				x[xOffset + jj] += residual * diagonal[jj];
			}
		}
	}

	// _setMultiColorIndices: the valid nodes of [start, end) by color
	// ( off2 mod 2 ) * 4 + ( off1 mod 2 ) * 2 + off0 mod 2, as indices relative to start.
	private void SetMultiColorIndices(int start, int end, List<int>[] colors)
	{
		for (int i = start; i < end; i++)
		{
			int node = sorted.TreeNodes[i];
			if (!PoissonMultigrid.IsValidFem1Node(tree, node))
			{
				continue;
			}

			int index = 0;
			for (int dd = 0; dd < Dim; dd++)
			{
				index = index * ColorModulusPerAxis + Modulo(tree.LocalOffset(node, Dim - dd - 1), ColorModulusPerAxis);
			}

			colors[index].Add(i - start);
		}
	}

	// The C++'s BlockWindow: a half-open range of blocks walked forward or backward.
	private struct BlockWindow
	{
		private int begin;
		private int end;

		public BlockWindow(int begin, int end)
		{
			if (begin <= end)
			{
				this.begin = begin;
				this.end = end;
			}
			else
			{
				this.begin = end + 1;
				this.end = begin + 1;
			}
		}

		public void Shift(int off)
		{
			begin += off;
			end += off;
		}

		public readonly int Begin(bool forward) => forward ? begin : end - 1;

		public readonly int End(bool forward) => forward ? end : begin - 1;

		public readonly bool InBlock(int b) => b >= begin && b < end;
	}
}
