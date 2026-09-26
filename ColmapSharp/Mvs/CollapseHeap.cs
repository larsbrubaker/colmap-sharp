// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause) and LLVM libc++ (Apache-2.0 WITH LLVM-exception); see
// THIRD_PARTY_NOTICES.md.
//
// CollapseHeap: the edge-collapse priority queue of colmap/mvs/mesh_simplification.cc,
// std::priority_queue<CollapseCandidate, std::vector<...>, CompareCandidateCost>, used by
// MeshSimplification.cs.
//
// COLMAP compares candidates by cost only, so among equal costs (every collapse on a flat
// region costs exactly 0) the pop order is whatever the standard library's heap produces, and
// that order decides the mesh. The macOS pycolmap wheel links libc++, so this is a port of
// libc++'s heap algorithms (__algorithm/make_heap.h, push_heap.h, pop_heap.h, sift_down.h):
// - make_heap: __sift_down from (n - 2) / 2 down to 0 (a candidate is not an arithmetic type,
//   so libc++ does not assume both children exist);
// - push: __sift_up, which stops at the first parent that does not compare less;
// - pop: __floyd_sift_down moves the hole to a leaf taking the larger child each level, then
//   the last element fills the hole and is sifted up (LLVM 15 and later).
// "less" is COLMAP's comparator, a.cost > b.cost, so the root is the cheapest candidate.
// With this, flat grids simplify byte-identically to pycolmap (MeshSimplificationOracleTests).
//
// An entry holds only the key (cost, vertices, timestamps), not the optimal position and
// color. They are recomputed when the entry is popped: a candidate is only acted on when both
// timestamps still match, and a vertex's position, color and quadric change only together
// with its timestamp, so the recomputation is bit-identical to the stored value. This keeps a
// million-face mesh's queue at 24 bytes per entry instead of 64.

namespace ColmapSharp.Mvs;

/// <summary>The key of one collapse candidate: its quadric cost and the vertex states it was computed for.</summary>
internal readonly struct CollapseKey
{
	public CollapseKey(double cost, int v1, int v2, uint timestampV1, uint timestampV2)
	{
		Cost = cost;
		V1 = v1;
		V2 = v2;
		TimestampV1 = timestampV1;
		TimestampV2 = timestampV2;
	}

	/// <summary>The quadric error of the collapse at its optimal position (never negative).</summary>
	public double Cost { get; }

	/// <summary>The vertex that survives the collapse.</summary>
	public int V1 { get; }

	/// <summary>The vertex merged into <see cref="V1"/>.</summary>
	public int V2 { get; }

	/// <summary>V1's timestamp when the candidate was computed.</summary>
	public uint TimestampV1 { get; }

	/// <summary>V2's timestamp when the candidate was computed.</summary>
	public uint TimestampV2 { get; }
}

/// <summary>Port of std::priority_queue over libc++'s heap with COLMAP's CompareCandidateCost.</summary>
internal sealed class CollapseHeap
{
	private CollapseKey[] items;
	private int count;

	/// <summary>
	/// Takes ownership of <paramref name="initial"/> and heapifies it, like the
	/// priority_queue(compare, vector&amp;&amp;) constructor (libc++ make_heap).
	/// </summary>
	public CollapseHeap(CollapseKey[] initial)
	{
		items = initial.Length == 0 ? new CollapseKey[16] : initial;
		count = initial.Length;
		if (count > 1)
		{
			for (int start = (count - 2) / 2; start >= 0; start--)
			{
				SiftDown(count, start);
			}
		}
	}

	/// <summary>The number of queued candidates.</summary>
	public int Count => count;

	/// <summary>priority_queue::push: push_back, then libc++ push_heap.</summary>
	public void Push(in CollapseKey key)
	{
		if (count == items.Length)
		{
			Array.Resize(ref items, items.Length * 2);
		}

		items[count++] = key;
		SiftUp(count);
	}

	/// <summary>priority_queue::top then pop: libc++ pop_heap, then pop_back.</summary>
	public CollapseKey Pop()
	{
		CollapseKey top = items[0];
		int len = count;
		if (len > 1)
		{
			int hole = FloydSiftDown(len);
			int last = len - 1;
			if (hole == last)
			{
				items[hole] = top;
			}
			else
			{
				items[hole] = items[last];
				items[last] = top;
				SiftUp(hole + 1);
			}
		}

		count--;
		return top;
	}

	// CompareCandidateCost: a.cost > b.cost, the heap's "less".
	private static bool Less(in CollapseKey a, in CollapseKey b) => a.Cost > b.Cost;

	// libc++ __sift_down (without __assume_both_children) over items[0, len).
	private void SiftDown(int len, int start)
	{
		int child = start;
		if (len < 2 || (len - 2) / 2 < child)
		{
			return;
		}

		child = 2 * child + 1;
		if (child + 1 < len && Less(items[child], items[child + 1]))
		{
			// The right child exists and is greater than the left child.
			child++;
		}

		// Already in heap order: start is not less than its largest child.
		if (Less(items[child], items[start]))
		{
			return;
		}

		CollapseKey top = items[start];
		do
		{
			// Not in heap order: move the largest child up.
			items[start] = items[child];
			start = child;
			if ((len - 2) / 2 < child)
			{
				break;
			}

			child = 2 * child + 1;
			if (child + 1 < len && Less(items[child], items[child + 1]))
			{
				child++;
			}
		}
		while (!Less(items[child], top));

		items[start] = top;
	}

	// libc++ __floyd_sift_down: moves the hole at the root down to a leaf, always along the
	// larger child, and returns the leaf. Requires len >= 2.
	private int FloydSiftDown(int len)
	{
		int hole = 0;
		int child = 0;
		while (true)
		{
			int childIndex = hole + child + 1;
			child = 2 * child + 1;
			if (child + 1 < len && Less(items[childIndex], items[childIndex + 1]))
			{
				childIndex++;
				child++;
			}

			items[hole] = items[childIndex];
			hole = childIndex;
			if (child > (len - 2) / 2)
			{
				return hole;
			}
		}
	}

	// libc++ __sift_up over items[0, len): the last element rises while its parent is less.
	private void SiftUp(int len)
	{
		if (len <= 1)
		{
			return;
		}

		int last = len - 1;
		len = (len - 2) / 2;
		int ptr = len;
		if (!Less(items[ptr], items[last]))
		{
			return;
		}

		CollapseKey t = items[last];
		do
		{
			items[last] = items[ptr];
			last = ptr;
			if (len == 0)
			{
				break;
			}

			len = (len - 1) / 2;
			ptr = len;
		}
		while (Less(items[ptr], t));

		items[last] = t;
	}
}
