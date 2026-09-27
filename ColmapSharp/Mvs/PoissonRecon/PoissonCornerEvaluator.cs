// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonCornerEvaluator: the value and gradient of the solved implicit function at the corners
// of leaf cells, as the level-set extractor samples them (FEMTree.LevelSet.3D.inl's
// SetSliceCornerValuesAndMCIndices with corner gradients, which nonLinearFit turns on). Ports
// FEMTree::_Evaluator< Sigs , 1 > (the per-depth corner and child-corner evaluators and the
// boundary-corner stencils of _Evaluator::set, _cornerValues, Evaluate) and the
// ConstCornerSupportKey overload of FEMTree::_getCornerValues (FEMTree.Evaluation.inl). With a
// degree-1 basis and gradients the extractor always takes this boundary-corner overload
// (useBoundaryEvaluation), so the center and plain corner stencils are not built. The
// coefficients are PoissonSystem.Solve's solution and PoissonImplicitEvaluator's coarse
// coefficients. Tier A against oracle/poisson_levelset_harness.cc ("cornervalues").
//
// Translation notes: stencils are stored flat, four doubles per window entry (value, then the
// x, y and z derivatives: CumulativeDerivatives< 3 , 1 >'s order); the per-axis products are
// formed left to right from the first axis, as Evaluate does. The depth-0 parent-child stencils
// (built upstream from unset child evaluators, and never read) are skipped.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Corner values and gradients of the implicit function at leaf corners. Port of PoissonRecon's
/// <c>_Evaluator&lt; Sigs , 1 &gt;</c> with <c>_getCornerValues</c> (boundary-corner overload).
/// </summary>
public sealed class PoissonCornerEvaluator
{
	private readonly FemTree tree;
	private readonly BSplineSupportSizes support;
	private readonly int maxDepth;
	private readonly int width;
	private readonly BSplineTabulatedEvaluator[] cornerEvaluators;
	private readonly BSplineTabulatedEvaluator?[] childCornerEvaluators;

	// [depth][corner][4 * windowIndex + k] and [depth][childCorner][corner][...].
	private readonly double[][][] ccStencils;
	private readonly double[][][][] pcStencils;

	// CornerLoopData< BCornerSize + 1 >: [corner] and [corner][childCorner].
	private readonly int[][] ccIndices;
	private readonly int[][][] pcIndices;
	private readonly NeighborKey key;
	private readonly int[] childWindow;
	private readonly double[] dValues = new double[6];
	private readonly double[] exterior = new double[4];
	private readonly int[] fIdx = new int[3];
	private readonly int[] cIdx = new int[3];
	private readonly int[] childCIdx = new int[3];

	/// <summary>
	/// The evaluator for basis <paramref name="signature"/> on the finalized tree. Port of
	/// <c>_Evaluator::set( maxDepth )</c> with the ConstCornerSupportKey the extractor uses.
	/// </summary>
	public PoissonCornerEvaluator(FemTree tree, int signature)
	{
		this.tree = tree;
		maxDepth = PoissonMultigrid.MaxDepth(tree);
		support = BSplineSupportSizes.For(FemSignature.Degree(signature));
		width = support.BCornerSize + 1;
		key = new NeighborKey(tree, support.BCornerEnd, -support.BCornerStart + 1, resetOnMissing: false);
		key.Set(maxDepth + tree.DepthOffset);
		childWindow = new int[key.WindowSize];

		cornerEvaluators = new BSplineTabulatedEvaluator[maxDepth + 1];
		childCornerEvaluators = new BSplineTabulatedEvaluator?[maxDepth + 1];
		for (int d = 0; d <= maxDepth; d++)
		{
			cornerEvaluators[d] = new BSplineTabulatedEvaluator(signature, 1, BSplineSamplePoints.Corner, d);
			if (d >= 1)
			{
				childCornerEvaluators[d] = new BSplineTabulatedEvaluator(signature, 1, BSplineSamplePoints.ChildCorner, d - 1);
			}
		}

		ccIndices = new int[8][];
		pcIndices = new int[8][][];
		for (int c = 0; c < 8; c++)
		{
			ccIndices[c] = LoopIndices(axis => ((c >> axis) & 1) != 0 ? (1, width) : (0, width - 1));
			pcIndices[c] = new int[8][];
			for (int pc = 0; pc < 8; pc++)
			{
				pcIndices[c][pc] = LoopIndices(axis => ((pc >> axis) & 1) != ((c >> axis) & 1) ? (0, width) : ((c >> axis) & 1) != 0 ? (1, width) : (0, width - 1));
			}
		}

		ccStencils = new double[maxDepth + 1][][];
		pcStencils = new double[maxDepth + 1][][][];
		for (int depth = 0; depth <= maxDepth; depth++)
		{
			int center = (1 << depth) >> 1;
			ccStencils[depth] = new double[8][];
			for (int c = 0; c < 8; c++)
			{
				cIdx[0] = cIdx[1] = cIdx[2] = center;
				ccStencils[depth][c] = Stencil(depth, center, c, parentChild: false);
			}

			if (depth == 0)
			{
				continue;
			}

			pcStencils[depth] = new double[8][][];
			for (int c = 0; c < 8; c++)
			{
				pcStencils[depth][c] = new double[8][];
				for (int cc = 0; cc < 8; cc++)
				{
					for (int k = 0; k < 3; k++)
					{
						cIdx[k] = center + ((c >> k) & 1);
					}

					pcStencils[depth][c][cc] = Stencil(depth, center / 2, cc, parentChild: true);
				}
			}
		}
	}

	/// <summary>
	/// Whether a leaf's parent is interiorly supported, which selects the stencil path. Port of
	/// <c>_isInteriorlySupported( Degrees , leaf-&gt;parent )</c>.
	/// </summary>
	public bool IsInterior(int leaf)
	{
		int parent = tree.Parent(leaf);
		if (parent == FemTree.None || tree.LocalDepth(parent) < 0)
		{
			return false;
		}

		int depth = tree.LocalDepth(parent);
		return support.IsInteriorlySupported(depth, tree.LocalOffset(parent, 0)) && support.IsInteriorlySupported(depth, tree.LocalOffset(parent, 1)) && support.IsInteriorlySupported(depth, tree.LocalOffset(parent, 2));
	}

	/// <summary>
	/// Writes to <paramref name="values"/> the value and the x, y and z derivatives at corner
	/// <paramref name="corner"/> of <paramref name="node"/>, from <paramref name="solution"/>
	/// (indexed by node index) and <paramref name="coarse"/> (its coarse coefficients), with
	/// <paramref name="isInterior"/> from <see cref="IsInterior"/>. Port of
	/// <c>_getCornerValues&lt; Real , 1 &gt;( bNeighborKey , node , corner , ... )</c>.
	/// </summary>
	public void Values(int node, int corner, float[] solution, float[] coarse, bool isInterior, Span<float> values)
	{
		values[..4].Clear();
		key.GetNeighbors(node);
		int d = tree.LocalDepth(node);
		for (int k = 0; k < 3; k++)
		{
			cIdx[k] = tree.LocalOffset(node, k);
		}

		int[] window = key.Window(tree.Depth(node));
		if (isInterior)
		{
			AddInterior(values, ccIndices[corner], window, ccStencils[d][corner], solution);
		}
		else
		{
			AddExterior(values, ccIndices[corner], d, cIdx, corner, window, solution, parentChild: false);
		}

		if (d > 0)
		{
			int childCorner = tree.ChildIndexInParent(node);
			int[] parentWindow = key.Window(tree.Depth(tree.Parent(node)));
			if (isInterior)
			{
				AddInterior(values, pcIndices[corner][childCorner], parentWindow, pcStencils[d][childCorner][corner], coarse);
			}
			else
			{
				AddExterior(values, pcIndices[corner][childCorner], d, cIdx, corner, parentWindow, coarse, parentChild: true);
			}
		}

		// If there could be finer neighbors whose support overlaps the point
		if (d < maxDepth && key.GetChildNeighbors(corner, tree.Depth(node), childWindow) != 0)
		{
			if (isInterior)
			{
				AddInterior(values, ccIndices[corner], childWindow, ccStencils[d + 1][corner], solution);
			}
			else
			{
				for (int k = 0; k < 3; k++)
				{
					childCIdx[k] = (cIdx[k] << 1) | ((corner & (1 << k)) != 0 ? 1 : 0);
				}

				AddExterior(values, ccIndices[corner], d + 1, childCIdx, corner, childWindow, solution, parentChild: false);
			}
		}
	}

	private void AddInterior(Span<float> values, int[] indices, int[] window, double[] stencil, float[] coefficients)
	{
		for (int i = 0; i < indices.Length; i++)
		{
			int idx = indices[i];
			int n = window[idx];
			if (tree.IsActive(n))
			{
				float coefficient = coefficients[tree.NodeIndex(n)];
				for (int k = 0; k < 4; k++)
				{
					values[k] += coefficient * (float)stencil[4 * idx + k];
				}
			}
		}
	}

	private void AddExterior(Span<float> values, int[] indices, int depth, int[] cornerIdx, int corner, int[] window, float[] coefficients, bool parentChild)
	{
		for (int i = 0; i < indices.Length; i++)
		{
			int n = window[indices[i]];
			if (tree.IsActive(n))
			{
				for (int k = 0; k < 3; k++)
				{
					fIdx[k] = tree.LocalOffset(n, k);
				}

				CornerValues(depth, fIdx, cornerIdx, corner, parentChild, exterior);
				float coefficient = coefficients[tree.NodeIndex(n)];
				for (int k = 0; k < 4; k++)
				{
					values[k] += coefficient * (float)exterior[k];
				}
			}
		}
	}

	// The stencil around the centered cell: per window entry, _cornerValues of function
	// fCenter + i - BCornerEnd on each axis at the cell cIdx.
	private double[] Stencil(int depth, int fCenter, int corner, bool parentChild)
	{
		var stencil = new double[4 * width * width * width];
		var f = new int[3];
		var value = new double[4];
		for (int i0 = 0; i0 < width; i0++)
		{
			for (int i1 = 0; i1 < width; i1++)
			{
				for (int i2 = 0; i2 < width; i2++)
				{
					f[0] = fCenter + i0 - support.BCornerEnd;
					f[1] = fCenter + i1 - support.BCornerEnd;
					f[2] = fCenter + i2 - support.BCornerEnd;
					CornerValues(depth, f, cIdx, corner, parentChild, value);
					Array.Copy(value, 0, stencil, 4 * (((i0 * width) + i1) * width + i2), 4);
				}
			}
		}

		return stencil;
	}

	// _Evaluator::_cornerValues: per axis the value and derivative of function f at the cell's
	// back or front corner, then Evaluate's products, first axis first.
	private void CornerValues(int depth, int[] f, int[] cell, int corner, bool parentChild, double[] result)
	{
		BSplineTabulatedEvaluator evaluator = parentChild ? childCornerEvaluators[depth]! : cornerEvaluators[depth];
		for (int axis = 0; axis < 3; axis++)
		{
			int off = (corner >> axis) & 1;
			for (int dd = 0; dd <= 1; dd++)
			{
				dValues[2 * axis + dd] = evaluator.Value(f[axis], cell[axis] + off, dd);
			}
		}

		for (int k = 0; k < 4; k++)
		{
			// CumulativeDerivatives< 3 , 1 >::Factor: 0 is the value, 1 + a the derivative along a.
			double value = dValues[k == 1 ? 1 : 0];
			value *= dValues[2 + (k == 2 ? 1 : 0)];
			value *= dValues[4 + (k == 3 ? 1 : 0)];
			result[k] = value;
		}
	}

	// CornerLoopData: the window indices with [start, end) per axis, axis 0 outermost.
	private int[] LoopIndices(Func<int, (int Start, int End)> bounds)
	{
		var (s0, e0) = bounds(0);
		var (s1, e1) = bounds(1);
		var (s2, e2) = bounds(2);
		var indices = new List<int>();
		for (int i0 = s0; i0 < e0; i0++)
		{
			for (int i1 = s1; i1 < e1; i1++)
			{
				for (int i2 = s2; i2 < e2; i2++)
				{
					indices.Add(((i0 * width) + i1) * width + i2);
				}
			}
		}

		return indices.ToArray();
	}
}
