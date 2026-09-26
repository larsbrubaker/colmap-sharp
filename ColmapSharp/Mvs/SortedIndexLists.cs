// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SortedIndexLists: the per-vertex adjacency lists of colmap/mvs/mesh_simplification.cc
// (VertexData::adjacent_faces / adjacent_vertices with the SortedInsert / SortedErase
// helpers), stored as one pooled int array per vertex instead of a std::vector each. Used by
// MeshSimplification.cs; the collapse priority queue is CollapseHeap.cs.
//
// Why pooled: a million-face mesh has hundreds of thousands of collapses, and each one grows
// the surviving vertex's lists and drops the removed vertex's. Arrays have power-of-two
// capacities, and a dropped or outgrown array goes back to a per-capacity free list, so the
// collapse loop reuses arrays instead of allocating once the pool has warmed up.

namespace ColmapSharp.Mvs;

/// <summary>A sorted, duplicate-free list of indices per vertex, backed by pooled arrays.</summary>
internal sealed class SortedIndexLists
{
	// Most vertices of a manifold triangle mesh have about six neighbors and six faces.
	private const int MinCapacityLog2 = 3;

	private readonly int[][] lists;
	private readonly int[] counts;
	private readonly Stack<int[]>?[] pool = new Stack<int[]>?[31];

	/// <summary>Empty lists for <paramref name="numLists"/> vertices.</summary>
	public SortedIndexLists(int numLists)
	{
		lists = new int[numLists][];
		counts = new int[numLists];
		Array.Fill(lists, Array.Empty<int>());
	}

	/// <summary>The sorted entries of list <paramref name="index"/>.</summary>
	public ReadOnlySpan<int> this[int index] => lists[index].AsSpan(0, counts[index]);

	/// <summary>Port of SortedInsert: adds <paramref name="value"/> unless it is already present.</summary>
	public void Insert(int index, int value)
	{
		int[] list = lists[index];
		int count = counts[index];
		int pos = LowerBound(list, count, value);
		if (pos < count && list[pos] == value)
		{
			return;
		}

		if (count == list.Length)
		{
			int[] grown = Rent(Math.Max(MinCapacityLog2, CapacityLog2(list.Length) + 1));
			list.AsSpan(0, count).CopyTo(grown);
			Return(list);
			lists[index] = grown;
			list = grown;
		}

		list.AsSpan(pos, count - pos).CopyTo(list.AsSpan(pos + 1));
		list[pos] = value;
		counts[index] = count + 1;
	}

	/// <summary>Port of SortedErase: removes <paramref name="value"/> if it is present.</summary>
	public void Erase(int index, int value)
	{
		int[] list = lists[index];
		int count = counts[index];
		int pos = LowerBound(list, count, value);
		if (pos < count && list[pos] == value)
		{
			list.AsSpan(pos + 1, count - pos - 1).CopyTo(list.AsSpan(pos));
			counts[index] = count - 1;
		}
	}

	/// <summary>Empties list <paramref name="index"/> and returns its array to the pool.</summary>
	public void Clear(int index)
	{
		Return(lists[index]);
		lists[index] = Array.Empty<int>();
		counts[index] = 0;
	}

	// std::lower_bound: the first position whose entry is not less than value.
	private static int LowerBound(int[] list, int count, int value)
	{
		int lo = 0;
		int hi = count;
		while (lo < hi)
		{
			int mid = (lo + hi) >>> 1;
			if (list[mid] < value)
			{
				lo = mid + 1;
			}
			else
			{
				hi = mid;
			}
		}

		return lo;
	}

	private static int CapacityLog2(int capacity) => capacity == 0 ? 0 : System.Numerics.BitOperations.Log2((uint)capacity);

	private int[] Rent(int capacityLog2)
	{
		Stack<int[]>? free = pool[capacityLog2];
		if (free is not null && free.TryPop(out int[]? array))
		{
			return array;
		}

		return new int[1 << capacityLog2];
	}

	private void Return(int[] array)
	{
		if (array.Length == 0)
		{
			return;
		}

		int log2 = CapacityLog2(array.Length);
		(pool[log2] ??= new Stack<int[]>()).Push(array);
	}
}
