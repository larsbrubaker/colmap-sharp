// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonLevelSetExtractor.IsoEdges: the iso-edges, the segments of the iso-curve on each face
// of the leaves, named by the keys of the edges their ends lie on. It follows the iso-vertices
// (PoissonLevelSetExtractor.IsoVertices.cs on slice edges, .XSliceIsoVertices.cs on cross-slice
// edges), whose edge keys it reads. Ports FEMTree.LevelSet.3D.inl's _LevelSetExtractor< ... ,
// 3 , ... >::CopyFinerSliceIsoEdgeKeys, CopyFinerXSliceIsoEdgeKeys, SetSliceIsoEdges and
// SetXSliceIsoEdges, and Extract's iso-edge loop of SetSliceIso, its SetSlabIsoEdges and the
// vertex-pair and face parts of FinalizeSlice and FinalizeSlab (the edge parts are
// FinalizeSliceEdges / FinalizeSlabEdges).
//
// What each step does:
// - CopyFiner(X)SliceIsoEdgeKeys: a coarse edge (not a leaf's) is split into two finer edges.
//   If exactly one of them has an iso-vertex, the coarse edge takes its key, so a coarser
//   leaf's face can name it; if both do, the two vertices are paired (vKeyValues), and the pair
//   pushed to the coarser slices whose edge contains this one, so the polygon step can walk
//   from one to the other.
// - Set(X)SliceIsoEdges: each leaf sets the iso-edges of a face it owns (unless the neighbor
//   across the face is finer, whose faces come from its own leaves), and pushes them up, keyed
//   by face, to the coarser slices or slabs whose face contains this one, stopping at the first
//   ancestor whose neighbor across the face is finer.
// The polygon step that reads them (SetLevelSet, IsoSurface) is
// PoissonLevelSetExtractor.Polygons.cs. Tier A against oracle/poisson_levelset5_harness.cc;
// none of its runs reaches a vertex pair (both halves of a coarse edge crossed), so the pair
// branches are pinned only by reading.
//
// Translation notes:
// - Upstream runs each step in a ThreadPool::ParallelFor over the leaves, with per-thread key
//   lists merged in thread order at finalize; the port runs the leaves in order, one list, as
//   a single-threaded run does (docs/CPP_DIVERGENCES.md, entry 123). The face and edge flags
//   are set by one owner each, so the arrays do not depend on the order either way; only the
//   order of the recorded lists would.
// - CopyFinerXSliceIsoEdgeKeys pushes a pair of cross-edge vertices first to the slab and then,
//   for its coarser ancestors, to the slices (sliceScratch) at the same parity, not to the
//   coarser slabs; that is what upstream does, and it is kept.

namespace ColmapSharp.Mvs.PoissonRecon;

public sealed partial class PoissonLevelSetExtractor
{
	/// <summary>
	/// Sets the iso-edges of the slices on the plane <paramref name="sliceAtMaxDepth"/>, finest
	/// depth first, first copying each coarser slice's edge keys from the finer one. Port of the
	/// iso-edge loop of Extract's <c>SetSliceIso</c> (without a boundary).
	/// </summary>
	public void SetSliceIsoEdges(int sliceAtMaxDepth)
	{
		for (int d = MaxDepth, o = sliceAtMaxDepth; d >= FullDepth; d--, o >>= 1)
		{
			if (d < MaxDepth)
			{
				CopyFinerSliceIsoEdgeKeys(d, o);
			}

			SetSliceIsoEdges(d, o);
			if ((o & 1) != 0)
			{
				break;
			}
		}
	}

	/// <summary>
	/// Sets the iso-edges on the cross faces of the slab <paramref name="slabAtMaxDepth"/> and
	/// of each coarser slab that it ends, finest depth first, first copying each coarser slab's
	/// cross-edge keys from the finer ones. Port of Extract's <c>SetSlabIsoEdges</c> (without a
	/// boundary; InteriorSlab holds for every slab of the whole tree).
	/// </summary>
	public void SetSlabIsoEdges(int slabAtMaxDepth)
	{
		for (int d = MaxDepth, o = slabAtMaxDepth; d >= FullDepth; d--, o >>= 1)
		{
			if (d < MaxDepth)
			{
				CopyFinerXSliceIsoEdgeKeys(d, o);
			}

			SetXSliceIsoEdges(d, o);
			if ((o & 1) == 0)
			{
				break;
			}
		}
	}

	/// <summary>
	/// Moves each slice's recorded vertex pairs, edge keys and face iso-edges on the plane
	/// <paramref name="sliceAtMaxDepth"/> into its maps. Port of Extract's <c>FinalizeSlice</c>
	/// (without a boundary).
	/// </summary>
	public void FinalizeSlice(int sliceAtMaxDepth)
	{
		FinalizeSliceEdges(sliceAtMaxDepth);
		for (int d = MaxDepth, o = sliceAtMaxDepth; d >= FullDepth; d--, o >>= 1)
		{
			LevelSetSliceValues values = SlabValues[d].SliceValues(o);
			values.SetVertexPairsFromScratch();
			values.SetFacesFromScratch();
			if ((o & 1) != 0)
			{
				break;
			}
		}
	}

	/// <summary>
	/// Moves the recorded vertex pairs, edge keys and face iso-edges of the slab
	/// <paramref name="slabAtMaxDepth"/> and each coarser slab that it ends into their maps.
	/// Port of Extract's <c>FinalizeSlab</c> (without a boundary).
	/// </summary>
	public void FinalizeSlab(int slabAtMaxDepth)
	{
		FinalizeSlabEdges(slabAtMaxDepth);
		for (int d = MaxDepth, o = slabAtMaxDepth; d >= FullDepth; d--, o >>= 1)
		{
			LevelSetXSliceValues values = SlabValues[d].XSliceValues(o);
			values.SetVertexPairsFromScratch();
			values.SetFacesFromScratch();
			if ((o & 1) == 0)
			{
				break;
			}
		}
	}

	/// <summary>
	/// Gives the edges of slice <paramref name="slice"/> at <paramref name="depth"/> that are
	/// split in the finer slice the key of their one finer iso-vertex, or pairs their two.
	/// Port of <c>CopyFinerSliceIsoEdgeKeys( ... , depth , fullDepth , slice , ... )</c>.
	/// </summary>
	public void CopyFinerSliceIsoEdgeKeys(int depth, int slice)
	{
		if (slice > 0)
		{
			CopyFinerSliceIsoEdgeKeys(depth, slice, HyperCubeDirection.Front);
		}

		if (slice < (1 << depth))
		{
			CopyFinerSliceIsoEdgeKeys(depth, slice, HyperCubeDirection.Back);
		}
	}

	private void CopyFinerSliceIsoEdgeKeys(int depth, int slice, HyperCubeDirection zDir)
	{
		LevelSetSliceValues pValues = SlabValues[depth].SliceValues(slice);
		LevelSetSliceValues cValues = SlabValues[depth + 1].SliceValues(slice << 1);
		HyperCubeOverlapTable edgeCorners = HyperCubeTables.Of(3, 1, 0);
		int globalDepth = depth + tree.DepthOffset;
		int nodeSlice = slice - (zDir == HyperCubeDirection.Back ? 0 : 1) + tree.LocalInset(depth);
		int end = sorted.End(globalDepth, nodeSlice);
		for (int i = sorted.Begin(globalDepth, nodeSlice); i < end; i++)
		{
			int node = sorted.TreeNodes[i];
			if (!PoissonMultigrid.IsValidSpaceNode(tree, node) || !tree.IsActive(tree.FirstChild(node)))
			{
				continue;
			}

			// Copy the edges that overlap the coarser edges
			for (int sliceEdge = 0; sliceEdge < HyperCube.ElementNum(2, 1); sliceEdge++)
			{
				int pIndex = pValues.CellIndices.Index(1, i, sliceEdge);
				int e = HyperCube.Element(3, 1, zDir, sliceEdge);
				int[] c = edgeCorners.OverlapElements[e];
				int child0 = tree.FirstChild(node) + c[0];
				int child1 = tree.FirstChild(node) + c[1];
				if (!PoissonMultigrid.IsValidSpaceNode(tree, child0) || !PoissonMultigrid.IsValidSpaceNode(tree, child1))
				{
					continue;
				}

				int cIndex1 = cValues.CellIndices.Index(1, tree.NodeIndex(child0), sliceEdge);
				int cIndex2 = cValues.CellIndices.Index(1, tree.NodeIndex(child1), sliceEdge);
				if (cValues.EdgeSet[cIndex1] != cValues.EdgeSet[cIndex2])
				{
					pValues.EdgeKeys[pIndex] = cValues.EdgeSet[cIndex1] != 0 ? cValues.EdgeKeys[cIndex1] : cValues.EdgeKeys[cIndex2];
					pValues.EdgeSet[pIndex] = 1;
				}
				else if (cValues.EdgeSet[cIndex1] != 0 && cValues.EdgeSet[cIndex2] != 0)
				{
					(LevelSetKey, LevelSetKey) pair = (cValues.EdgeKeys[cIndex1], cValues.EdgeKeys[cIndex2]);
					pValues.VertexPairKeyValues.Add(pair);

					int n = node;
					int nDepth = depth;
					int nSlice = slice;
					while (nDepth > FullDepth && PoissonMultigrid.IsValidSpaceNode(tree, tree.Parent(n)) && edgeCorners.Overlap[e][tree.ChildIndexInParent(n)])
					{
						n = tree.Parent(n);
						nDepth--;
						nSlice >>= 1;
						SlabValues[nDepth].SliceValues(nSlice).VertexPairKeyValues.Add(pair);
					}
				}
			}
		}
	}

	/// <summary>
	/// Gives the cross edges of slab <paramref name="slab"/> at <paramref name="depth"/> that
	/// are split in the two finer slabs the key of their one finer iso-vertex, or pairs their
	/// two. Port of <c>CopyFinerXSliceIsoEdgeKeys</c>.
	/// </summary>
	public void CopyFinerXSliceIsoEdgeKeys(int depth, int slab)
	{
		LevelSetXSliceValues pValues = SlabValues[depth].XSliceValues(slab);
		LevelSetXSliceValues cValues0 = SlabValues[depth + 1].XSliceValues((slab << 1) | 0);
		LevelSetXSliceValues cValues1 = SlabValues[depth + 1].XSliceValues((slab << 1) | 1);
		bool has0 = cValues0.Slab == ((slab << 1) | 0);
		bool has1 = cValues1.Slab == ((slab << 1) | 1);
		HyperCubeOverlapTable edgeCorners = HyperCubeTables.Of(3, 1, 0);
		int globalDepth = depth + tree.DepthOffset;
		int nodeSlab = slab + tree.LocalInset(depth);
		int end = sorted.End(globalDepth, nodeSlab);
		for (int i = sorted.Begin(globalDepth, nodeSlab); i < end; i++)
		{
			// If the node is not a leaf, inherit iso-edges from children
			int node = sorted.TreeNodes[i];
			if (!PoissonMultigrid.IsValidSpaceNode(tree, node) || !tree.IsActive(tree.FirstChild(node)))
			{
				continue;
			}

			for (int c = 0; c < HyperCube.ElementNum(2, 0); c++)
			{
				// Transform the face-corner index to a cross-edge index
				int e = HyperCube.Element(3, 1, HyperCubeDirection.Cross, c);
				int pIndex = pValues.CellIndices.Index(0, i, c);
				int child0 = tree.FirstChild(node) + HyperCube.Element(3, 0, HyperCubeDirection.Back, c);
				int child1 = tree.FirstChild(node) + HyperCube.Element(3, 0, HyperCubeDirection.Front, c);
				if (!PoissonMultigrid.IsValidSpaceNode(tree, child0) || !PoissonMultigrid.IsValidSpaceNode(tree, child1))
				{
					continue;
				}

				int cIndex0 = has0 ? cValues0.CellIndices.Index(0, tree.NodeIndex(child0), c) : -1;
				int cIndex1 = has1 ? cValues1.CellIndices.Index(0, tree.NodeIndex(child1), c) : -1;
				bool eSet0 = has0 && cValues0.EdgeSet[cIndex0] != 0;
				bool eSet1 = has1 && cValues1.EdgeSet[cIndex1] != 0;

				// If there's one zero-crossing along the edge
				if (eSet0 != eSet1)
				{
					pValues.EdgeKeys[pIndex] = eSet0 ? cValues0.EdgeKeys[cIndex0] : cValues1.EdgeKeys[cIndex1];
					pValues.EdgeSet[pIndex] = 1;
				}

				// If there's are two zero-crossings along the edge
				else if (eSet0 && eSet1)
				{
					(LevelSetKey, LevelSetKey) pair = (cValues0.EdgeKeys[cIndex0], cValues1.EdgeKeys[cIndex1]);
					pValues.VertexPairKeyValues.Add(pair);
					int n = node;
					int nDepth = depth;
					int nSlab = slab;
					while (nDepth > FullDepth && PoissonMultigrid.IsValidSpaceNode(tree, tree.Parent(n)) && edgeCorners.Overlap[e][tree.ChildIndexInParent(n)])
					{
						n = tree.Parent(n);
						nDepth--;
						nSlab >>= 1;

						// Upstream records the coarser pairs on the slice scratch (see the header).
						SlabValues[nDepth].SliceValues(nSlab).VertexPairKeyValues.Add(pair);
					}
				}
			}
		}
	}

	/// <summary>
	/// The iso-edges of the faces on slice <paramref name="slice"/> at <paramref name="depth"/>,
	/// from the leaves behind it and then those in front. Port of
	/// <c>SetSliceIsoEdges( keyGenerator , tree , depth , slice , slabValues )</c>.
	/// </summary>
	public void SetSliceIsoEdges(int depth, int slice)
	{
		if (slice > 0)
		{
			SetSliceIsoEdges(depth, slice, HyperCubeDirection.Front);
		}

		if (slice < (1 << depth))
		{
			SetSliceIsoEdges(depth, slice, HyperCubeDirection.Back);
		}
	}

	private void SetSliceIsoEdges(int depth, int slice, HyperCubeDirection zDir)
	{
		LevelSetSliceValues sValues = SlabValues[depth].SliceValues(slice);
		int globalDepth = depth + tree.DepthOffset;
		var neighborKey = new NeighborKey(tree, 1, 1, resetOnMissing: false);
		neighborKey.Set(globalDepth);

		// The neighbor across the slice: one step along z from the window's center.
		int xx = neighborKey.CenterIndex + (zDir == HyperCubeDirection.Back ? -1 : 1);
		int f = HyperCube.Element(3, 2, zDir, 0);
		bool[][] faceCorners = HyperCubeTables.Of(3, 2, 0).Overlap;
		Span<int> isoEdges = stackalloc int[2 * MarchingSquares.MaxEdges];
		int nodeSlice = slice - (zDir == HyperCubeDirection.Back ? 0 : 1) + tree.LocalInset(depth);
		int end = sorted.End(globalDepth, nodeSlice);
		for (int i = sorted.Begin(globalDepth, nodeSlice); i < end; i++)
		{
			int leaf = sorted.TreeNodes[i];
			if (!PoissonMultigrid.IsValidSpaceNode(tree, leaf) || tree.IsActive(tree.FirstChild(leaf)))
			{
				continue;
			}

			int leafIndex = tree.NodeIndex(leaf);
			int fIndex = sValues.CellIndices.Index(2, leafIndex, 0);
			if (sValues.FaceSet[fIndex] != 0)
			{
				continue;
			}

			neighborKey.GetNeighbors(leaf);
			if (IsFinerAcross(neighborKey, globalDepth, xx))
			{
				continue;
			}

			int mcIndex = sValues.McIndices[i - sValues.CellIndices.NodeOffset];
			int count = MarchingSquares.AddEdgeIndices(mcIndex, isoEdges);
			var edges = new LevelSetIsoEdge[count];
			for (int j = 0; j < count; j++)
			{
				edges[j] = new LevelSetIsoEdge(SliceEdgeKey(sValues, leafIndex, isoEdges[2 * j], slice, zDir, depth), SliceEdgeKey(sValues, leafIndex, isoEdges[(2 * j) + 1], slice, zDir, depth));
			}

			sValues.FaceSet[fIndex] = 1;
			sValues.FaceEdges[fIndex] = ToFaceEdges(edges);

			int node = leaf;
			int nDepth = depth;
			int nSlice = slice;
			while (PoissonMultigrid.IsValidSpaceNode(tree, tree.Parent(node)) && faceCorners[f][tree.ChildIndexInParent(node)])
			{
				node = tree.Parent(node);
				nDepth--;
				nSlice >>= 1;
				if (!IsFinerAcross(neighborKey, nDepth + tree.DepthOffset, xx))
				{
					LevelSetKey key = KeyGenerator.Key(2, nDepth, tree.LocalOffset(node, 0), tree.LocalOffset(node, 1), tree.LocalOffset(node, 2), f);
					SlabValues[nDepth].SliceValues(nSlice).FaceKeyValues.Add((key, edges));
				}
				else
				{
					break;
				}
			}
		}
	}

	/// <summary>
	/// The iso-edges of the faces crossing slab <paramref name="slab"/> at
	/// <paramref name="depth"/>. The slices on either side and the slab must have their edge
	/// keys. Port of <c>SetXSliceIsoEdges</c>.
	/// </summary>
	public void SetXSliceIsoEdges(int depth, int slab)
	{
		LevelSetSliceValues bValues = SlabValues[depth].SliceValues(slab);
		LevelSetSliceValues fValues = SlabValues[depth].SliceValues(slab + 1);
		LevelSetXSliceValues xValues = SlabValues[depth].XSliceValues(slab);
		int globalDepth = depth + tree.DepthOffset;
		var neighborKey = new NeighborKey(tree, 1, 1, resetOnMissing: false);
		neighborKey.Set(globalDepth);
		bool[][] faceCorners = HyperCubeTables.Of(3, 2, 0).Overlap;
		int[] antipodal = HyperCubeTables.Of(3, 2).CellOffsetAntipodal;
		Span<int> isoEdges = stackalloc int[2 * MarchingSquares.MaxEdges];
		int nodeSlab = slab + tree.LocalInset(depth);
		int end = sorted.End(globalDepth, nodeSlab);
		for (int i = sorted.Begin(globalDepth, nodeSlab); i < end; i++)
		{
			int leaf = sorted.TreeNodes[i];
			if (!PoissonMultigrid.IsValidSpaceNode(tree, leaf) || tree.IsActive(tree.FirstChild(leaf)))
			{
				continue;
			}

			int leafIndex = tree.NodeIndex(leaf);
			int mcIndex = bValues.McIndices[i - bValues.CellIndices.NodeOffset] | (fValues.McIndices[i - fValues.CellIndices.NodeOffset] << 4);
			neighborKey.GetNeighbors(leaf);

			// Iterate over the edges on the back
			for (int sliceEdge = 0; sliceEdge < HyperCube.ElementNum(2, 1); sliceEdge++)
			{
				int f = HyperCube.Element(3, 2, HyperCubeDirection.Cross, sliceEdge);
				int faceMcIndex = HyperCube.ElementMcIndex(3, 2, f, mcIndex);
				int xx = antipodal[f];
				int fIndex = xValues.CellIndices.Index(1, leafIndex, sliceEdge);
				if (xValues.FaceSet[fIndex] != 0 || IsFinerAcross(neighborKey, globalDepth, xx))
				{
					continue;
				}

				int count = MarchingSquares.AddEdgeIndices(faceMcIndex, isoEdges);
				var edges = new LevelSetIsoEdge[count];
				for (int j = 0; j < count; j++)
				{
					LevelSetKey k0 = CrossFaceEdgeKey(bValues, fValues, xValues, i, leafIndex, f, isoEdges[2 * j], slab, depth);
					LevelSetKey k1 = CrossFaceEdgeKey(bValues, fValues, xValues, i, leafIndex, f, isoEdges[(2 * j) + 1], slab, depth);
					edges[j] = new LevelSetIsoEdge(k0, k1);
				}

				xValues.FaceSet[fIndex] = 1;
				xValues.FaceEdges[fIndex] = ToFaceEdges(edges);

				int node = leaf;
				int nDepth = depth;
				int nSlab = slab;
				while (PoissonMultigrid.IsValidSpaceNode(tree, tree.Parent(node)) && faceCorners[f][tree.ChildIndexInParent(node)])
				{
					node = tree.Parent(node);
					nDepth--;
					nSlab >>= 1;
					if (IsFinerAcross(neighborKey, nDepth + tree.DepthOffset, xx))
					{
						break;
					}

					LevelSetKey key = KeyGenerator.Key(2, nDepth, tree.LocalOffset(node, 0), tree.LocalOffset(node, 1), tree.LocalOffset(node, 2), f);
					SlabValues[nDepth].XSliceValues(nSlab).FaceKeyValues.Add((key, edges));
				}
			}
		}
	}

	// Whether the neighbor at index xx of the key's window at globalDepth (the leaf's depth or
	// an ancestor's, from the key's last GetNeighbors) is subdivided, so its finer leaves set the shared face: upstream's
	// IsActiveNode( neighbor ) && IsActiveNode( neighbor->children ).
	private bool IsFinerAcross(NeighborKey neighborKey, int globalDepth, int xx)
	{
		int neighbor = neighborKey.Window(globalDepth)[xx];
		return tree.IsActive(neighbor) && tree.IsActive(tree.FirstChild(neighbor));
	}

	// The key of slice edge `sliceEdge` of the leaf, which must have an iso-vertex (its own or
	// one copied from finer). Upstream throws "Edge not set" otherwise.
	private static LevelSetKey SliceEdgeKey(LevelSetSliceValues sValues, int leafIndex, int sliceEdge, int slice, HyperCubeDirection zDir, int depth)
	{
		int idx = sValues.CellIndices.Index(1, leafIndex, sliceEdge);
		if (sValues.EdgeSet[idx] == 0)
		{
			throw new InvalidOperationException($"Edge not set: {slice - (zDir == HyperCubeDirection.Back ? 0 : 1)} / {1 << depth}");
		}

		return sValues.EdgeKeys[idx];
	}

	// The key of edge `faceEdge` of cross face f of the leaf (sorted index i): a cross edge
	// from the slab, or a back/front slice edge. Upstream throws "Edge not set" if it has no
	// iso-vertex.
	private static LevelSetKey CrossFaceEdgeKey(LevelSetSliceValues bValues, LevelSetSliceValues fValues, LevelSetXSliceValues xValues, int i, int leafIndex, int f, int faceEdge, int slab, int depth)
	{
		int e = HyperCube.FromSubCube(3, 2, f, 1, faceEdge);
		HyperCube.Factor(3, 1, e, out HyperCubeDirection dir, out int coIndex);
		byte[] eSet;
		LevelSetKey[] keys;
		int idx;
		if (dir == HyperCubeDirection.Cross)
		{
			// Cross-edge
			idx = xValues.CellIndices.Index(0, leafIndex, coIndex);
			eSet = xValues.EdgeSet;
			keys = xValues.EdgeKeys;
		}
		else
		{
			LevelSetSliceValues sValues = dir == HyperCubeDirection.Back ? bValues : fValues;
			idx = sValues.CellIndices.Index(1, i, coIndex);
			eSet = sValues.EdgeSet;
			keys = sValues.EdgeKeys;
		}

		if (eSet[idx] == 0)
		{
			throw new InvalidOperationException($"Edge not set: {slab} / {1 << depth}");
		}

		return keys[idx];
	}

	private static LevelSetFaceEdges ToFaceEdges(LevelSetIsoEdge[] edges) =>
		new(edges.Length, edges.Length > 0 ? edges[0] : LevelSetIsoEdge.Unset, edges.Length > 1 ? edges[1] : LevelSetIsoEdge.Unset);
}
