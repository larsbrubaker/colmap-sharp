// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// SortedTreeNodes: SortedTreeNodes<3> from thirdparty/PoissonRecon/FEMTree.h and
// FEMTree.SortedTreeNodes.inl - the active (non-ghost) nodes listed by depth and, within a
// depth, by their z slice (offset along the last axis), in pre-order within a slice. Node
// indices are then renumbered to positions in this list, which is how FEMTree's dense node
// data (constraints, solutions) is laid out and how the solver walks the tree slice by slice.
// Tier A: same order and slice starts as the C++ (oracle/poisson_tree_harness.cc).
//
// Translation note: _sliceStart[depth] (2^depth + 1 entries) becomes a jagged int array and
// treeNodes holds FemTree node handles.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// The active nodes sorted by depth and slice. Port of PoissonRecon's <c>SortedTreeNodes&lt;Dim&gt;</c>.
/// </summary>
public sealed class SortedTreeNodes
{
	private int[][] sliceStart = [];

	/// <summary>The sorted node handles (C++ <c>treeNodes</c>).</summary>
	public int[] TreeNodes { get; private set; } = [];

	/// <summary>The number of depths (the root's max depth + 1). Port of <c>levels</c>.</summary>
	public int Levels { get; private set; }

	/// <summary>The first sorted position at a depth. Port of <c>begin( depth )</c>.</summary>
	public int Begin(int depth) => sliceStart[depth][0];

	/// <summary>One past the last sorted position at a depth. Port of <c>end( depth )</c>.</summary>
	public int End(int depth) => sliceStart[depth][1 << depth];

	/// <summary>The first position of a slice (clamped to [0, 2^depth]). Port of <c>begin( depth , slice )</c>.</summary>
	public int Begin(int depth, int slice) => sliceStart[depth][slice < 0 ? 0 : (slice > (1 << depth) ? (1 << depth) : slice)];

	/// <summary>Port of <c>end( depth , slice )</c>.</summary>
	public int End(int depth, int slice) => Begin(depth, slice + 1);

	/// <summary>The total number of sorted nodes. Port of <c>size()</c>.</summary>
	public int Size => Levels != 0 ? sliceStart[Levels - 1][1 << (Levels - 1)] : 0;

	/// <summary>The number of sorted nodes at a depth. Port of <c>size( depth )</c>.</summary>
	public int SizeAt(int depth)
	{
		if (depth < 0 || depth >= Levels)
		{
			throw new ArgumentOutOfRangeException(nameof(depth), $"bad depth: 0 <= {depth} < {Levels}");
		}

		return sliceStart[depth][1 << depth] - sliceStart[depth][0];
	}

	/// <summary>
	/// Sorts the active nodes below <paramref name="root"/> and sets each one's node index to
	/// its sorted position. Port of <c>set( root )</c>.
	/// </summary>
	public void Set(FemTree tree, int root)
	{
		Sort(tree, root);
		for (int i = 0; i < Size; i++)
		{
			tree.SetNodeIndex(TreeNodes[i], i);
		}
	}

	/// <summary>
	/// Like <see cref="Set"/>, and returns map with map[new index] = the previous node index
	/// (-1 where the previous index was negative). Port of <c>reset( root , map )</c>.
	/// </summary>
	public int[] Reset(FemTree tree, int root)
	{
		Sort(tree, root);
		var map = new int[Size];
		Array.Fill(map, -1);
		for (int i = 0; i < Size; i++)
		{
			int node = TreeNodes[i];
			if (tree.NodeIndex(node) >= 0)
			{
				map[i] = tree.NodeIndex(node);
			}

			tree.SetNodeIndex(node, i);
		}

		return map;
	}

	// Port of _set: count per slice, prefix-sum, place in pre-order, then shift the starts back.
	private void Sort(FemTree tree, int root)
	{
		Levels = tree.MaxDepth(root) + 1;
		sliceStart = new int[Levels][];
		for (int l = 0; l < Levels; l++)
		{
			sliceStart[l] = new int[(1 << l) + 1];
		}

		// Count the number of nodes in each slice
		tree.ProcessNodes(root, node =>
		{
			if (!tree.IsGhost(node))
			{
				sliceStart[tree.Depth(node)][tree.Offset(node, 2) + 1]++;
			}
		});

		// Get the start index for each slice
		int levelOffset = 0;
		for (int l = 0; l < Levels; l++)
		{
			sliceStart[l][0] = levelOffset;
			for (int s = 0; s < (1 << l); s++)
			{
				sliceStart[l][s + 1] += sliceStart[l][s];
			}

			levelOffset = sliceStart[l][1 << l];
		}

		// Add the tree nodes
		TreeNodes = new int[sliceStart[Levels - 1][1 << (Levels - 1)]];
		tree.ProcessNodes(root, node =>
		{
			if (!tree.IsGhost(node))
			{
				TreeNodes[sliceStart[tree.Depth(node)][tree.Offset(node, 2)]++] = node;
			}
		});

		// Shift the slice offsets up since we incremented as we added
		for (int l = 0; l < Levels; l++)
		{
			for (int s = 1 << l; s > 0; s--)
			{
				sliceStart[l][s] = sliceStart[l][s - 1];
			}

			sliceStart[l][0] = l > 0 ? sliceStart[l - 1][1 << (l - 1)] : 0;
		}
	}
}
