// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// LevelSetCellIndices: for one slice (a plane z = slice at one depth) or one slab (the layer
// between two slices), a dense index of every 2D cell - corner, edge, face - that the nodes
// touching it share, so each shared corner is evaluated once and each shared edge gets one
// iso-vertex. Ports FEMTree.LevelSet.inl's LevelSetExtraction::CellIndexData< 2 , 2 > with its
// _Scratch, and the set() of SliceCellIndexData< 3 > and SlabCellIndexData< 3 >: a cell is owned
// by the first (lowest incident-cube index) active node around it; owners are numbered in
// node-then-cell order. Read by LevelSetSliceValues and the extractor. Tier A against
// oracle/poisson_levelset2_harness.cc.
//
// Translation notes: the per-node index arrays are flat, node-major ([node - NodeOffset] *
// ElementNum + cell); like upstream they grow but never shrink and are not cleared on reuse
// (every entry of the span is rewritten). The scratch maps are zeroed per set, as _Scratch's
// resize does. Upstream fills them in parallel; the result does not depend on the order.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Dense indices of the 2D cells of a slice or slab. Port of PoissonRecon's
/// <c>SliceCellIndexData&lt; 3 &gt;</c> / <c>SlabCellIndexData&lt; 3 &gt;</c>.
/// </summary>
public sealed class LevelSetCellIndices
{
	private const int CellDims = 3;
	private readonly int[][] tables = new int[CellDims][];
	private readonly int[][] maps = new int[CellDims][];
	private readonly int[] counts = new int[CellDims];
	private NeighborKey? key;
	private int capacity;

	/// <summary>The sorted index of the span's first node. Port of <c>nodeOffset</c>.</summary>
	public int NodeOffset { get; private set; }

	/// <summary>The number of nodes in the span. Port of <c>size()</c>.</summary>
	public int Size { get; private set; }

	/// <summary>The number of distinct cells of each dimension (0 corners, 1 edges, 2 faces). Port of <c>counts</c>.</summary>
	public int Count(int cellDim) => counts[cellDim];

	/// <summary>
	/// The index of cell <paramref name="cell"/> (a <paramref name="cellDim"/>-element of the
	/// 2-cube) of the node with sorted index <paramref name="nodeIndex"/>. Port of
	/// <c>indices&lt; CellDim &gt;( nodeIndex )[ cell ]</c>.
	/// </summary>
	public int Index(int cellDim, int nodeIndex, int cell) => tables[cellDim][((nodeIndex - NodeOffset) * HyperCube.ElementNum(2, cellDim)) + cell];

	/// <summary>
	/// Indexes the cells of slice <paramref name="slice"/> at global depth
	/// <paramref name="depth"/>, from the nodes of the slabs on either side. Port of
	/// <c>SliceCellIndexData::set</c>.
	/// </summary>
	public void SetSlice(FemTree tree, SortedTreeNodes sorted, int depth, int slice)
	{
		Begin(tree, sorted.Begin(depth, slice - 1), sorted.End(depth, slice), depth);
		int behindEnd = sorted.End(depth, slice - 1);
		for (int i = NodeOffset; i < NodeOffset + Size; i++)
		{
			int[] neighbors = key!.GetNeighbors(sorted.TreeNodes[i]);

			// A node behind the slice sees it as its front face.
			HyperCubeDirection dir = i < behindEnd ? HyperCubeDirection.Front : HyperCubeDirection.Back;
			for (int cellDim = 0; cellDim < CellDims; cellDim++)
			{
				int elementNum = HyperCube.ElementNum(2, cellDim);
				for (int c2 = 0; c2 < elementNum; c2++)
				{
					Process(tree, neighbors, cellDim, cellDim, HyperCube.Element(3, cellDim, dir, c2), c2);
				}
			}
		}

		Finish();
	}

	/// <summary>
	/// Indexes the cells crossing slab <paramref name="slab"/> at global depth
	/// <paramref name="depth"/> (each 2D cell extruded across the slab). Port of
	/// <c>SlabCellIndexData::set</c>.
	/// </summary>
	public void SetSlab(FemTree tree, SortedTreeNodes sorted, int depth, int slab)
	{
		Begin(tree, sorted.Begin(depth, slab), sorted.End(depth, slab), depth);
		for (int i = NodeOffset; i < NodeOffset + Size; i++)
		{
			int[] neighbors = key!.GetNeighbors(sorted.TreeNodes[i]);
			for (int cellDim = 0; cellDim < CellDims; cellDim++)
			{
				int elementNum = HyperCube.ElementNum(2, cellDim);
				for (int c2 = 0; c2 < elementNum; c2++)
				{
					Process(tree, neighbors, cellDim, cellDim + 1, HyperCube.Element(3, cellDim + 1, HyperCubeDirection.Cross, c2), c2);
				}
			}
		}

		Finish();
	}

	// CellIndexData::resize, _Scratch::resize and the neighbor keys' set.
	private void Begin(FemTree tree, int begin, int end, int depth)
	{
		NodeOffset = begin;
		Size = end - begin;
		if (Size > capacity)
		{
			for (int d = 0; d < CellDims; d++)
			{
				tables[d] = new int[Size * HyperCube.ElementNum(2, d)];
				Array.Fill(tables[d], -1);
				maps[d] = new int[Size * HyperCube.ElementNum(2, d)];
			}

			capacity = Size;
		}

		for (int d = 0; d < CellDims; d++)
		{
			counts[d] = 0;
			Array.Clear(maps[d] ?? [], 0, Size * HyperCube.ElementNum(2, d));
		}

		if (key == null || key.KeyDepth != depth)
		{
			key = new NeighborKey(tree, 1, 1, resetOnMissing: false);
			key.Set(depth);
		}
	}

	// _setProcess< CellDim >: cell c2 of the 2-cube, which is element c of the node's 3-cube
	// (a K-element, K = cellDim for a slice and cellDim + 1 for a slab).
	private void Process(FemTree tree, int[] neighbors, int cellDim, int k, int c, int c2)
	{
		HyperCubeElementTable table = HyperCubeTables.Of(3, k);
		int[] cellOffset = table.CellOffset[c];
		int myIc = table.IncidentCube[c];
		for (int ic = 0; ic < myIc; ic++)
		{
			// If the neighbor exists and comes before, they own the cell
			if (tree.IsActive(neighbors[cellOffset[ic]]))
			{
				return;
			}
		}

		int elementNum = HyperCube.ElementNum(2, cellDim);
		int center = neighbors[key!.CenterIndex];
		int myCount = ((tree.NodeIndex(center) - NodeOffset) * elementNum) + c2;
		maps[cellDim][myCount] = 1;

		// Set the cell index for all nodes incident on the cell
		int[] coIndices = table.IncidentElementCoIndex[c];
		for (int ic = 0; ic < cellOffset.Length; ic++)
		{
			int neighbor = neighbors[cellOffset[ic]];
			if (tree.IsActive(neighbor))
			{
				tables[cellDim][((tree.NodeIndex(neighbor) - NodeOffset) * elementNum) + coIndices[ic]] = myCount;
			}
		}
	}

	// _setCounts and _setTables: number the owned cells in order, then map every entry.
	private void Finish()
	{
		for (int d = 0; d < CellDims; d++)
		{
			int n = Size * HyperCube.ElementNum(2, d);
			int[] map = maps[d];
			int count = 0;
			for (int i = 0; i < n; i++)
			{
				if (map[i] != 0)
				{
					map[i] = count++;
				}
			}

			counts[d] = count;
			int[] table = tables[d];
			for (int i = 0; i < n; i++)
			{
				table[i] = map[table[i]];
			}
		}
	}
}
