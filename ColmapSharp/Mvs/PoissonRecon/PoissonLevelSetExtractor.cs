// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonLevelSetExtractor: the marching-cubes extraction of the iso-surface from the solved
// octree, as COLMAP's poisson_meshing runs it (the whole tree, no slab boundaries, corner
// gradients on for the non-linear fit). Ports FEMTree.LevelSet.3D.inl's
// _LevelSetExtractor< ... , 3 , ... >::Extract set-up (the full depth, the coarse coefficients,
// SlabValues per depth) and its InitSlice, InitSlab and SetSliceValues steps with
// SetSliceCornerValuesAndMCIndices, and FEMTree.inl's sliced getFullDepth. The driver loop
// (slab by slab at the finest depth, PoissonLevelSetExtractor.Extract.cs) calls them in
// Extract's order. The iso-vertices on slice
// edges are in PoissonLevelSetExtractor.IsoVertices.cs and those on cross-slice (slab) edges in
// PoissonLevelSetExtractor.XSliceIsoVertices.cs, the iso-edges in
// PoissonLevelSetExtractor.IsoEdges.cs, the polygons (IsoSurface) in
// PoissonLevelSetExtractor.Polygons.cs. SetMCIndices and OverwriteCornerValues run only for
// slab boundaries, which COLMAP never passes, so they are not ported. Corner values come from
// PoissonCornerEvaluator. Tier A against oracle/poisson_levelset2_harness.cc through
// oracle/poisson_levelset6_harness.cc.
//
// Translation notes: depths here are local unless named global. Upstream sets the corner
// values in parallel; a corner's value does not depend on which leaf computes it first.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// The level-set extractor's corner and cell-index state. Port of PoissonRecon's
/// <c>_LevelSetExtractor&lt; HasData , Real , 3 , Data &gt;</c> (the steps listed in the file header).
/// </summary>
public sealed partial class PoissonLevelSetExtractor
{
	private readonly FemTree tree;
	private readonly SortedTreeNodes sorted;
	private readonly float[] coefficients;
	private readonly float[] coarseCoefficients;
	private readonly PoissonCornerEvaluator cornerEvaluator;
	private readonly float isoValue;
	private readonly float[] squareValues = new float[4];
	private readonly float[] cornerValues = new float[4];

	/// <summary>
	/// Sets up the extraction of the <paramref name="isoValue"/> level set of
	/// <paramref name="coefficients"/> (basis <paramref name="signature"/>, indexed by node index).
	/// Port of Extract's set-up with slabDepth 0 and slab [0, 1). The iso-vertices carry a
	/// density from <paramref name="density"/> (Extract's densityWeights, when COLMAP trims) and
	/// auxiliary data (colors) from <paramref name="data"/> (entries: data..., weight; Extract's
	/// data), falling back to <paramref name="zeroData"/> where the data has no weight.
	/// </summary>
	public PoissonLevelSetExtractor(
		FemTree tree,
		SortedTreeNodes sorted,
		int signature,
		float[] coefficients,
		float isoValue,
		DensityEstimator? density = null,
		SparseNodeData? data = null,
		float[]? zeroData = null)
	{
		this.tree = tree;
		this.sorted = sorted;
		this.coefficients = coefficients;
		this.isoValue = isoValue;
		this.density = density;
		this.data = data;
		// Copied, so a caller reusing its array cannot change the fallback mid-extraction.
		int channels = data == null ? 0 : data.Width - 1;
		if (zeroData != null && zeroData.Length != channels)
		{
			throw new ArgumentException($"zeroData has {zeroData.Length} entries; the data has {channels} (none without data).", nameof(zeroData));
		}

		this.zeroData = zeroData == null ? new float[channels] : (float[])zeroData.Clone();
		PoissonMultigrid.SetFem1ValidityFlags(tree, sorted, signature);
		MaxDepth = PoissonMultigrid.MaxDepth(tree);
		FullDepth = GetFullDepth(tree, FemSignature.Degree(signature), 0, 0, 1);
		coarseCoefficients = PoissonImplicitEvaluator.CoarseCoefficients(tree, sorted, signature, coefficients);
		cornerEvaluator = new PoissonCornerEvaluator(tree, signature);
		dataEvaluator = data == null ? null : new PoissonPointEvaluator(DataSignature, 0, MaxDepth);
		KeyGenerator = new LevelSetKeyGenerator(MaxDepth);
		SlabValues = new LevelSetSlabValues[MaxDepth + 1];
		for (int d = 0; d <= MaxDepth; d++)
		{
			SlabValues[d] = new LevelSetSlabValues();
		}
	}

	/// <summary>The tree's finest local depth (Extract's maxDepth, as slabDepth is 0).</summary>
	public int MaxDepth { get; }

	/// <summary>The coarsest depth at which every node is present. Port of Extract's <c>fullDepth</c>.</summary>
	public int FullDepth { get; }

	/// <summary>The per-depth slice and slab state. Port of Extract's <c>slabValues</c>.</summary>
	public LevelSetSlabValues[] SlabValues { get; }

	/// <summary>
	/// The coarsest depth at which the tree is complete over the region [begin, end) along each
	/// axis at <paramref name="depth"/>, for B-splines of <paramref name="degree"/>. Port of
	/// <c>FEMTree::getFullDepth( Degrees , depth , begin , end )</c> with the same bounds on
	/// every axis.
	/// </summary>
	public static int GetFullDepth(FemTree tree, int degree, int depth, int begin, int end)
	{
		// [NOTE] Need "+1" because _getFullDepth will test children of leaves
		int maxDepth = PoissonMultigrid.MaxDepth(tree) + 1;
		if (begin > end || begin < 0 || end > (1 << depth))
		{
			throw new ArgumentException($"Bad bounds: [{begin}, {end}) at depth {depth}");
		}

		// Push to max depth
		if (depth < maxDepth)
		{
			begin <<= maxDepth - depth;
			end <<= maxDepth - depth;
			depth = maxDepth;
		}

		int root = tree.Root;
		if (!tree.HasChildren(root))
		{
			return -1;
		}

		var support = BSplineSupportSizes.For(degree);
		int minDepth = int.MaxValue;
		for (int c = 0; c < 8; c++)
		{
			minDepth = Math.Min(minDepth, SubtreeFullDepth(tree, support, depth, begin, end, tree.FirstChild(root) + c));
		}

		return minDepth;
	}

	// _getFullDepth( Degrees , depth , begin , end , node ).
	private static int SubtreeFullDepth(FemTree tree, BSplineSupportSizes support, int depth, int begin, int end, int node)
	{
		int d = tree.LocalDepth(node);
		if (!tree.HasChildren(node))
		{
			bool childrenSupported = false;
			for (int c = 0; c < 8; c++)
			{
				bool supported = d + 1 <= depth;
				for (int dim = 0; dim < 3 && supported; dim++)
				{
					int off = (tree.LocalOffset(node, dim) << 1) | ((c >> dim) & 1);

					// [NOTE]: Changing closed interval to half open interval
					int supportStart = (off + support.SupportStart) * (1 << (depth - d - 1));
					int supportEnd = (off + support.SupportEnd + 1) * (1 << (depth - d - 1));
					if (d + 1 >= 0 && (supportStart >= end || supportEnd <= begin))
					{
						supported = false;
					}
				}

				childrenSupported |= supported;
			}

			return childrenSupported ? d : int.MaxValue;
		}

		int minDepth = int.MaxValue;
		for (int c = 0; c < 8; c++)
		{
			minDepth = Math.Min(minDepth, SubtreeFullDepth(tree, support, depth, begin, end, tree.FirstChild(node) + c));
		}

		return minDepth;
	}

	/// <summary>
	/// Indexes and resets every slice on the plane <paramref name="sliceAtMaxDepth"/>, finest
	/// depth first, down to the full depth. Port of Extract's <c>InitSlice</c>.
	/// </summary>
	public void InitSlice(int sliceAtMaxDepth)
	{
		for (int d = MaxDepth; d >= FullDepth; d--)
		{
			int dOff = MaxDepth - d;
			int slice = sliceAtMaxDepth >> dOff;
			if (sliceAtMaxDepth != slice << dOff)
			{
				break;
			}

			LevelSetSliceValues values = SlabValues[d].SliceValues(slice);
			values.CellIndices.SetSlice(tree, sorted, d + tree.DepthOffset, slice + tree.LocalInset(d));
			values.Reset(slice, computeGradients: true);
		}
	}

	/// <summary>
	/// Indexes the slab <paramref name="slabAtMaxDepth"/> and the coarser slabs containing it
	/// (all of them on the first call, otherwise up to the first that it does not start).
	/// Port of Extract's <c>InitSlab</c>.
	/// </summary>
	public void InitSlab(int slabAtMaxDepth, bool first)
	{
		int slab = slabAtMaxDepth;
		for (int d = MaxDepth; d >= FullDepth; d--, slab >>= 1)
		{
			SlabValues[d].SlabCellIndices(slab).SetSlab(tree, sorted, d + tree.DepthOffset, slab + tree.LocalInset(d));
			SlabValues[d].ResetSlab(slab);
			if ((slab & 1) != 0 && !first)
			{
				break;
			}
		}
	}

	/// <summary>
	/// Sets the corner values and marching-squares indices of the slices on the plane
	/// <paramref name="sliceAtMaxDepth"/>, finest depth first. Port of Extract's
	/// <c>SetSliceValues</c> (without a boundary).
	/// </summary>
	public void SetSliceValues(int sliceAtMaxDepth)
	{
		for (int d = MaxDepth, o = sliceAtMaxDepth; d >= FullDepth; d--, o >>= 1)
		{
			SetSliceCornerValuesAndMcIndices(d, o);
			if ((o & 1) != 0)
			{
				break;
			}
		}
	}

	/// <summary>
	/// The leaves on either side of slice <paramref name="slice"/> at <paramref name="depth"/>:
	/// each corner on the slice evaluated once (value and gradient) and pushed up to the coarser
	/// slices that share it, and each leaf's marching-squares index. Port of
	/// <c>SetSliceCornerValuesAndMCIndices</c>.
	/// </summary>
	public void SetSliceCornerValuesAndMcIndices(int depth, int slice)
	{
		if (slice > 0)
		{
			SetSliceCornerValuesAndMcIndices(depth, slice, HyperCubeDirection.Front);
		}

		if (slice < (1 << depth))
		{
			SetSliceCornerValuesAndMcIndices(depth, slice, HyperCubeDirection.Back);
		}
	}

	private void SetSliceCornerValuesAndMcIndices(int depth, int slice, HyperCubeDirection zDir)
	{
		LevelSetSliceValues sValues = SlabValues[depth].SliceValues(slice);
		LevelSetCellIndices cellIndices = sValues.CellIndices;
		int nodeSlice = slice - (zDir == HyperCubeDirection.Back ? 0 : 1) + tree.LocalInset(depth);
		int globalDepth = depth + tree.DepthOffset;
		int end = sorted.End(globalDepth, nodeSlice);
		for (int i = sorted.Begin(globalDepth, nodeSlice); i < end; i++)
		{
			int leaf = sorted.TreeNodes[i];
			if (!PoissonMultigrid.IsValidSpaceNode(tree, leaf) || tree.IsActive(tree.FirstChild(leaf)))
			{
				continue;
			}

			int leafIndex = tree.NodeIndex(leaf);
			bool isInterior = cornerEvaluator.IsInterior(leaf);
			for (int c2 = 0; c2 < 4; c2++)
			{
				int c = HyperCube.Element(3, 0, zDir, c2);
				int vIndex = cellIndices.Index(0, leafIndex, c2);
				if (sValues.CornerSet[vIndex] == 0)
				{
					cornerEvaluator.Values(leaf, c, coefficients, coarseCoefficients, isInterior, cornerValues);
					sValues.CornerValues[vIndex] = cornerValues[0];
					sValues.CornerGradients![(3 * vIndex) + 0] = cornerValues[1];
					sValues.CornerGradients[(3 * vIndex) + 1] = cornerValues[2];
					sValues.CornerGradients[(3 * vIndex) + 2] = cornerValues[3];
					sValues.CornerSet[vIndex] = 1;
				}

				squareValues[c2] = sValues.CornerValues[vIndex];

				// Push the corner up to every coarser node whose same corner it is.
				int node = leaf;
				int nodeDepth = depth;
				int nodeSliceIndex = slice;
				while (nodeDepth > FullDepth && PoissonMultigrid.IsValidSpaceNode(tree, tree.Parent(node)) && tree.ChildIndexInParent(node) == c)
				{
					node = tree.Parent(node);
					nodeDepth--;
					nodeSliceIndex >>= 1;
					LevelSetSliceValues coarse = SlabValues[nodeDepth].SliceValues(nodeSliceIndex);
					int coarseIndex = coarse.CellIndices.Index(0, tree.NodeIndex(node), c2);
					coarse.CornerValues[coarseIndex] = sValues.CornerValues[vIndex];
					if (coarse.CornerGradients != null)
					{
						Array.Copy(sValues.CornerGradients!, 3 * vIndex, coarse.CornerGradients, 3 * coarseIndex, 3);
					}

					coarse.CornerSet[coarseIndex] = 1;
				}
			}

			sValues.McIndices[i - cellIndices.NodeOffset] = (byte)HyperCube.McIndex(squareValues, isoValue);
		}
	}
}
