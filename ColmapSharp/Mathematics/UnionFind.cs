// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// UnionFind: port of colmap/math/union_find.h, a disjoint-set forest keyed on arbitrary
// values (image ids, observations, strings) with path compression and no union by rank.
// ConnectedComponents.cs builds on it; later the global mapper, reconstruction clustering
// and the synthetic dataset generator use it directly. Tests:
// ColmapSharp.Tests/Mathematics/UnionFindTests.cs (union_find_test.cc 1:1).
//
// Tier A (exact): which element becomes a root is the same as COLMAP's for the same
// sequence of calls (Union always hangs root_x under root_y).
//
// Translation notes:
// - COLMAP stores parents in a NodeHashMap (boost::unordered_node_map), whose iteration
//   order is a function of boost's hash layout. Here Parents iterates in insertion order
//   (a Dictionary that never removes). See docs/CPP_DIVERGENCES.md, entry 2.
// - The C++ Find recurses; this one walks the path twice (find the root, then repoint every
//   node on the path at it), which leaves the map in the same state without recursion depth
//   limits on long chains.
// - The optional Hash template parameter becomes an optional IEqualityComparer<T>.

namespace ColmapSharp.Mathematics;

/// <summary>
/// Port of colmap::UnionFind: a disjoint-set structure over values of type
/// <typeparamref name="T"/>. Elements are inserted lazily the first time they are seen.
/// </summary>
public sealed class UnionFind<T>
	where T : notnull
{
	// Map from each element to its parent; a root is its own parent.
	private readonly Dictionary<T, T> parents;

	/// <summary>Creates an empty structure using the default equality comparer.</summary>
	public UnionFind()
		: this(null)
	{
	}

	/// <summary>Creates an empty structure keyed with <paramref name="comparer"/> (COLMAP's Hash parameter).</summary>
	public UnionFind(IEqualityComparer<T>? comparer)
	{
		parents = new Dictionary<T, T>(comparer);
	}

	/// <summary>Reserves room for <paramref name="capacity"/> elements.</summary>
	public void Reserve(int capacity)
	{
		parents.EnsureCapacity(capacity);
	}

	/// <summary>
	/// Finds the root of <paramref name="x"/>, compressing the path to it. If x is not in the
	/// structure, it is inserted as its own parent.
	/// </summary>
	public T Find(T x)
	{
		if (!parents.TryGetValue(x, out var parent))
		{
			parents.Add(x, x);
			return x;
		}

		var comparer = parents.Comparer;
		var root = x;
		while (!comparer.Equals(parent, root))
		{
			root = parent;
			parent = parents[root];
		}

		// Path compression: every node on the path now points straight at the root.
		var node = x;
		while (!comparer.Equals(node, root))
		{
			var next = parents[node];
			parents[node] = root;
			node = next;
		}

		return root;
	}

	/// <summary>
	/// Port of FindIfExists: if <paramref name="x"/> is in the structure, returns true and its
	/// stored parent (without path compression, so it is the root only after
	/// <see cref="Compress"/> or a <see cref="Find"/> of x); otherwise returns false and inserts
	/// nothing.
	/// </summary>
	public bool FindIfExists(T x, out T parent)
	{
		return parents.TryGetValue(x, out parent!);
	}

	/// <summary>Unites the sets containing <paramref name="x"/> and <paramref name="y"/>.</summary>
	public void Union(T x, T y)
	{
		var rootX = Find(x);
		var rootY = Find(y);
		if (!parents.Comparer.Equals(rootX, rootY))
		{
			parents[rootX] = rootY;
		}
	}

	/// <summary>
	/// Path-compresses all elements so each points directly to its root. Call this once
	/// after all Union operations and before iterating <see cref="Parents"/>.
	/// </summary>
	public void Compress()
	{
		// Find rewrites other entries' values, so walk a snapshot of the keys rather than
		// the live dictionary.
		var keys = new T[parents.Count];
		parents.Keys.CopyTo(keys, 0);
		foreach (var key in keys)
		{
			parents[key] = Find(key);
		}
	}

	/// <summary>All elements and their parents, in insertion order.</summary>
	public IReadOnlyDictionary<T, T> Parents => parents;
}
