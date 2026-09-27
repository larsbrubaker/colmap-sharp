// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from LLVM libc++ (Apache-2.0 WITH LLVM-exception, see THIRD_PARTY_NOTICES.md):
// <__hash_table>'s unique-key emplace path and __rehash / __do_rehash, as
// ColmapSharp/Util/LibcxxUnorderedMap.cs ports them. Modified: translated to C#, specialized to
// PoissonRecon SurfaceTrimmer.cpp's (MIT, see THIRD_PARTY_NOTICES.md) edge keys - two int
// vertex (or component) indices hashed by its BoostHash - and max_load_factor 1 (Apache-2.0
// section 4(b) notice).
//
// TrimmerEdgeHashTable: the iteration order of SurfaceTrimmer's
// std::unordered_set< HalfEdgeKey >, std::unordered_map< HalfEdgeKey , Index > and
// std::unordered_set< EdgeKey >. The island merge (PoissonSurfaceTrimmer.Islands.cs) walks
// those containers to build the component graph, and the order it adds each node's neighbors
// decides which neighbor a small component merges into and the order of the output
// triangles, so it must be libc++'s. The node list and bucket array logic is
// LibcxxUnorderedMap's, with the key widened to a pair and the hash to BoostHash; the two are
// kept separate so the int-keyed map on the solver's hot path stays specialized. Tier A,
// pinned through PoissonTreeOracleTests.Trim against oracle/poisson_trim_harness.cc.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// A hash table keyed by an ordered pair of ints that iterates in libc++'s
/// <c>std::unordered_map</c> / <c>std::unordered_set</c> order for SurfaceTrimmer.cpp's
/// <c>BoostHash</c> hasher. Insert and lookup only (no erase).
/// </summary>
internal sealed class TrimmerEdgeHashTable
{
	private const int None = -1;
	private const int Sentinel = 0;
	private int[] keys1 = new int[16];
	private int[] keys2 = new int[16];
	private int[] values = new int[16];
	private ulong[] hashes = new ulong[16];
	private int[] next = new int[16];

	// buckets[b]: the node before bucket b's first node, or None.
	private int[] buckets = [];
	private int bucketCount;

	/// <summary>An empty table with no buckets (a default-constructed container).</summary>
	public TrimmerEdgeHashTable()
	{
		next[Sentinel] = None;
	}

	/// <summary>The number of entries.</summary>
	public int Count { get; private set; }

	/// <summary>The first node in iteration order, or -1 (begin()).</summary>
	public int First => next[Sentinel];

	/// <summary>The node after <paramref name="node"/> in iteration order, or -1.</summary>
	public int Next(int node) => next[node];

	/// <summary>A node's first key.</summary>
	public int Key1(int node) => keys1[node];

	/// <summary>A node's second key.</summary>
	public int Key2(int node) => keys2[node];

	/// <summary>A node's value (unused by the sets).</summary>
	public ref int Value(int node) => ref values[node];

	/// <summary>
	/// SurfaceTrimmer.cpp's <c>BoostHash( i1 , i2 )</c> for int indices: size_t arithmetic, so
	/// each index sign-extends to 64 bits and the sums wrap.
	/// </summary>
	public static ulong BoostHash(int i1, int i2)
	{
		unchecked
		{
			ulong hash = (ulong)(long)i1 + 0x9e3779b9UL;
			hash ^= (ulong)(long)i2 + 0x9e3779b9UL + (hash << 6) + (hash >> 2);
			return hash;
		}
	}

	/// <summary>The node holding (<paramref name="k1"/>, <paramref name="k2"/>), or -1. Port of <c>find</c>.</summary>
	public int Find(int k1, int k2)
	{
		if (bucketCount == 0 || Count == 0)
		{
			return None;
		}

		ulong hash = BoostHash(k1, k2);
		ulong chash = ConstrainHash(hash, (ulong)bucketCount);
		int nd = buckets[chash];
		if (nd != None)
		{
			for (nd = next[nd]; nd != None && (hashes[nd] == hash || ConstrainHash(hashes[nd], (ulong)bucketCount) == chash); nd = next[nd])
			{
				if (hashes[nd] == hash && keys1[nd] == k1 && keys2[nd] == k2)
				{
					return nd;
				}
			}
		}

		return None;
	}

	/// <summary>
	/// Inserts the key with <paramref name="value"/> unless present, and returns its node (an
	/// existing entry keeps its value). Port of <c>__emplace_unique_key_args</c> (insert,
	/// operator[]).
	/// </summary>
	public int Insert(int k1, int k2, int value)
	{
		int existing = Find(k1, k2);
		if (existing != None)
		{
			return existing;
		}

		ulong hash = BoostHash(k1, k2);
		int nd = AllocateNode(k1, k2, value, hash);
		if ((float)(Count + 1) > bucketCount * 1.0f)
		{
			ulong bc = (ulong)bucketCount;
			ulong grown = 2 * bc + (IsHashPower2(bc) ? 0UL : 1UL);
			ulong needed = (ulong)MathF.Ceiling((float)(Count + 1) / 1.0f);
			Rehash(Math.Max(grown, needed));
		}

		ulong chash = ConstrainHash(hash, (ulong)bucketCount);
		int pn = buckets[chash];
		if (pn == None)
		{
			pn = Sentinel;
			next[nd] = next[pn];
			next[pn] = nd;
			buckets[chash] = pn;
			if (next[nd] != None)
			{
				buckets[ConstrainHash(hashes[next[nd]], (ulong)bucketCount)] = nd;
			}
		}
		else
		{
			next[nd] = next[pn];
			next[pn] = nd;
		}

		Count++;
		return nd;
	}

	private static bool IsHashPower2(ulong bc) => bc > 2 && (bc & (bc - 1)) == 0;

	private static ulong ConstrainHash(ulong h, ulong bc) => (bc & (bc - 1)) == 0 ? h & (bc - 1) : (h < bc ? h : h % bc);

	private int AllocateNode(int k1, int k2, int value, ulong hash)
	{
		int nd = Count + 1;
		if (nd >= keys1.Length)
		{
			int size = keys1.Length * 2;
			Array.Resize(ref keys1, size);
			Array.Resize(ref keys2, size);
			Array.Resize(ref values, size);
			Array.Resize(ref hashes, size);
			Array.Resize(ref next, size);
		}

		keys1[nd] = k1;
		keys2[nd] = k2;
		values[nd] = value;
		hashes[nd] = hash;
		return nd;
	}

	// __rehash<true>( n ) on the insert path, where n always exceeds the bucket count.
	private void Rehash(ulong n)
	{
		if (n == 1)
		{
			n = 2;
		}
		else if ((n & (n - 1)) != 0)
		{
			n = Util.LibcxxUnorderedMap<int>.NextPrime(n);
		}

		if (n > (ulong)bucketCount)
		{
			DoRehash((int)n);
		}
	}

	// __do_rehash<true>: relinks the node list so every bucket's nodes are contiguous.
	private void DoRehash(int count)
	{
		if (buckets.Length < count)
		{
			buckets = new int[count];
		}

		bucketCount = count;
		Array.Fill(buckets, None, 0, count);
		int pp = Sentinel;
		int cp = next[pp];
		if (cp == None)
		{
			return;
		}

		ulong bc = (ulong)count;
		ulong chash = ConstrainHash(hashes[cp], bc);
		buckets[chash] = pp;
		ulong phash = chash;
		for (pp = cp, cp = next[cp]; cp != None; cp = next[pp])
		{
			chash = ConstrainHash(hashes[cp], bc);
			if (chash == phash)
			{
				pp = cp;
			}
			else if (buckets[chash] == None)
			{
				buckets[chash] = pp;
				pp = cp;
				phash = chash;
			}
			else
			{
				int np = cp;
				next[pp] = next[np];
				next[np] = next[buckets[chash]];
				next[buckets[chash]] = cp;
			}
		}
	}
}
