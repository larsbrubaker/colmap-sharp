// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonLevelSetExtractor.XSliceIsoVertices: the iso-vertices on the edges that cross a slab
// (the z edges of the leaves between two slices), completing the iso-vertices begun in
// PoissonLevelSetExtractor.IsoVertices.cs (the slice edges). Ports FEMTree.LevelSet.3D.inl's
// _LevelSetExtractor< ... , 3 , ... >::SetXSliceIsoVertices, the cross-edge form of
// GetIsoVertex, and Extract's SetSlabIsoVertices (with SetSlabBounds and InteriorSlab, which
// hold for every slab when the slab range is the whole tree, as COLMAP runs it) and the edge
// part of FinalizeSlab. Extract calls SetSlabIsoVertices( slab ) before SetSliceIso( slab+1 ),
// so a slab's cross-edge vertices are numbered before the next slice's edge vertices. Each
// vertex's key is recorded on the slab and, when the edge borders a coarser leaf, pushed to the
// coarser slabs that share it. Tier A against oracle/poisson_levelset4_harness.cc.
//
// Translation notes: the float/double mix of GetIsoVertex is kept: the end derivatives are the
// float product of the z gradient and the float slab length, the vertex z is the double
// bCoordinate + (fCoordinate-bCoordinate)*averageRoot rounded to float, and the color is
// evaluated at the float center ( s + w/2 , (bCoordinate+fCoordinate)/2 ). Upstream sets the
// vertices in a ThreadPool::ParallelFor over the slab's leaves; the port runs the leaves in
// order, as a single-threaded run does (divergence 123).

namespace ColmapSharp.Mvs.PoissonRecon;

public sealed partial class PoissonLevelSetExtractor
{
	/// <summary>
	/// Sets the iso-vertices on the cross edges of the slab <paramref name="slabAtMaxDepth"/> and
	/// of each coarser slab that it ends, finest depth first. Port of Extract's
	/// <c>SetSlabIsoVertices</c> (without a boundary).
	/// </summary>
	internal void SetSlabIsoVertices(int slabAtMaxDepth)
	{
		for (int d = MaxDepth, o = slabAtMaxDepth; d >= FullDepth; d--, o >>= 1)
		{
			// SetSlabBounds: the slab's back and front z in the unit cube.
			float bCoordinate = (o << (MaxDepth - d)) / (float)(1 << MaxDepth);
			float fCoordinate = ((o + 1) << (MaxDepth - d)) / (float)(1 << MaxDepth);
			SetXSliceIsoVertices(d, o, bCoordinate, fCoordinate);
			if ((o & 1) == 0)
			{
				break;
			}
		}
	}

	/// <summary>
	/// Moves each slab's recorded edge keys, for the slab <paramref name="slabAtMaxDepth"/> and
	/// each coarser slab that it ends, into its edge-vertex map. Port of the edge part of
	/// Extract's <c>FinalizeSlab</c>.
	/// </summary>
	internal void FinalizeSlabEdges(int slabAtMaxDepth)
	{
		for (int d = MaxDepth, o = slabAtMaxDepth; d >= FullDepth; d--, o >>= 1)
		{
			SlabValues[d].XSliceValues(o).SetEdgesFromScratch();
			if ((o & 1) == 0)
			{
				break;
			}
		}
	}

	/// <summary>
	/// The iso-vertices on the edges crossing slab <paramref name="slab"/> at
	/// <paramref name="depth"/>, whose back and front lie at z = <paramref name="bCoordinate"/>
	/// and <paramref name="fCoordinate"/>. The slices on either side must have their corner
	/// values and marching-squares indices. Port of <c>SetXSliceIsoVertices</c>.
	/// </summary>
	internal void SetXSliceIsoVertices(int depth, int slab, float bCoordinate, float fCoordinate)
	{
		LevelSetSliceValues bValues = SlabValues[depth].SliceValues(slab);
		LevelSetSliceValues fValues = SlabValues[depth].SliceValues(slab + 1);
		LevelSetXSliceValues xValues = SlabValues[depth].XSliceValues(slab);
		int globalDepth = depth + tree.DepthOffset;
		BSplineSupportSizes weightSupport = BSplineSupportSizes.For(WeightDegree);

		// Const keys, fresh per call as upstream's per-thread keys are.
		var neighborKey = new NeighborKey(tree, 1, 1, resetOnMissing: false);
		var weightKey = new NeighborKey(tree, weightSupport.SupportEnd, -weightSupport.SupportStart, resetOnMissing: false);
		var dataKey = new NeighborKey(tree, 0, 0, resetOnMissing: false);
		neighborKey.Set(globalDepth);
		weightKey.Set(globalDepth);
		dataKey.Set(globalDepth);

		int nodeSlab = slab + tree.LocalInset(depth);
		int end = sorted.End(globalDepth, nodeSlab);
		HyperCubeOverlapTable edgeFaces = HyperCubeTables.Of(3, 1, 2);
		HyperCubeOverlapTable faceCorners = HyperCubeTables.Of(3, 2, 0);
		for (int i = sorted.Begin(globalDepth, nodeSlab); i < end; i++)
		{
			int leaf = sorted.TreeNodes[i];
			if (!PoissonMultigrid.IsValidSpaceNode(tree, leaf) || tree.IsActive(tree.FirstChild(leaf)))
			{
				continue;
			}

			int mcIndex = bValues.McIndices[i - bValues.CellIndices.NodeOffset] | (fValues.McIndices[i - fValues.CellIndices.NodeOffset] << 4);
			if (!HyperCube.HasMcRoots(3, mcIndex))
			{
				continue;
			}

			int leafIndex = tree.NodeIndex(leaf);
			neighborKey.GetNeighbors(leaf);
			if (density != null)
			{
				weightKey.GetNeighbors(leaf);
			}

			if (data != null)
			{
				dataKey.GetNeighbors(leaf);
			}

			for (int c = 0; c < HyperCube.ElementNum(2, 0); c++)
			{
				int e = HyperCube.Element(3, 1, HyperCubeDirection.Cross, c);
				if (!HyperCube.HasMcRoots(1, HyperCube.ElementMcIndex(3, 1, e, mcIndex)))
				{
					continue;
				}

				int vIndex = xValues.CellIndices.Index(0, leafIndex, c);
				if (xValues.EdgeSet[vIndex] != 0)
				{
					continue;
				}

				LevelSetKey key = KeyGenerator.Key(1, depth, tree.LocalOffset(leaf, 0), tree.LocalOffset(leaf, 1), tree.LocalOffset(leaf, 2), e);
				LevelSetVertex vertex = GetIsoVertex(weightKey, dataKey, leaf, leafIndex, c, bCoordinate, fCoordinate, bValues, fValues);
				xValues.EdgeSet[vIndex] = 1;
				int vertexIndex = Vertices.Count;
				Vertices.Add(vertex);
				xValues.EdgeKeys[vIndex] = key;
				xValues.EdgeKeyValues.Add((key, vertexIndex));

				// We only need to pass the iso-vertex down if the edge it lies on is adjacent to a coarser leaf
				if (!IsNeeded(neighborKey, e, depth))
				{
					continue;
				}

				int[] faces = edgeFaces.OverlapElements[e];
				for (int k = 0; k < 2; k++)
				{
					int node = leaf;
					int nodeDepth = depth;
					int nodeSlabIndex = slab;

					// As long as we are still in the tree and the parent is also adjacent to the node
					while (PoissonMultigrid.IsValidSpaceNode(tree, tree.Parent(node)) && faceCorners.Overlap[faces[k]][tree.ChildIndexInParent(node)])
					{
						node = tree.Parent(node);
						nodeDepth--;
						nodeSlabIndex >>= 1;
						LevelSetXSliceValues coarse = SlabValues[nodeDepth].XSliceValues(nodeSlabIndex);
						coarse.EdgeKeyValues.Add((key, vertexIndex));
						if (nodeDepth >= FullDepth)
						{
							coarse.EdgeSet[coarse.CellIndices.Index(0, tree.NodeIndex(node), c)] = 1;
						}

						if (!IsNeeded(neighborKey, e, nodeDepth))
						{
							break;
						}
					}
				}
			}
		}
	}

	// GetIsoVertex( ... , node , _c , bCoordinate , fCoordinate , bValues , fValues , vertex ,
	// zeroData ) with nonLinearFit on and gradientNormals off.
	private LevelSetVertex GetIsoVertex(NeighborKey weightKey, NeighborKey dataKey, int node, int nodeIndex, int c, float bCoordinate, float fCoordinate, LevelSetSliceValues bValues, LevelSetSliceValues fValues)
	{
		int i0 = bValues.CellIndices.Index(0, nodeIndex, c);
		int i1 = fValues.CellIndices.Index(0, nodeIndex, c);
		float x0 = bValues.CornerValues[i0];
		float x1 = fValues.CornerValues[i1];
		Span<float> s = cellStart;
		Span<float> position = vertexPosition;
		tree.LocalStartAndWidth(node, s, out float w);
		HyperCubeDirection[] dirs = HyperCubeTables.Of(2, 0).Directions[c];
		int x = dirs[0] == HyperCubeDirection.Back ? 0 : 1;
		int y = dirs[1] == HyperCubeDirection.Back ? 0 : 1;
		position[0] = s[0] + (w * x);
		position[1] = s[1] + (w * y);

		// Float products, widened.
		float length = fCoordinate - bCoordinate;
		double averageRoot = AverageRoot(x0, x1, bValues.CornerGradients![(3 * i0) + 2] * length, fValues.CornerGradients![(3 * i1) + 2] * length);
		position[2] = (float)(bCoordinate + (length * averageRoot));

		// gradientNormals is off, so both corner gradients are zero and so is the blend.
		float gradient = (0f * (float)(1.0 - averageRoot)) + (0f * (float)averageRoot);
		float depth = 1f;
		if (density != null)
		{
			depth = PoissonSplat.GetSampleDepthAndWeight(tree, density, node, position, weightKey).Depth;
		}

		float[] dataValue = data == null ? [] : DataAt(dataKey, s[0] + (w / 2), s[1] + (w / 2), (bCoordinate + fCoordinate) / 2);
		return new LevelSetVertex(position[0], position[1], position[2], gradient, gradient, gradient, depth, dataValue);
	}
}
