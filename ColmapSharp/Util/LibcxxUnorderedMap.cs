// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from LLVM libc++ (Apache-2.0 WITH LLVM-exception, see THIRD_PARTY_NOTICES.md):
// <__hash_table> (__constrain_hash, __is_hash_power2, the unique-key emplace path and
// __rehash / __do_rehash) as shipped in the macOS SDK that Apple clang 21 uses, and
// __next_prime (compiled into the libc++ dylib) reimplemented from its documented behavior:
// the smallest prime >= n. Modified: translated to C#, specialized to int keys with
// std::hash<int> (the identity) and max_load_factor 1 (Apache-2.0 section 4(b) notice).
//
// LibcxxUnorderedMap: std::unordered_map< int , TValue >'s iteration order. The C++ standard
// leaves that order to the implementation, and some upstream code lets it reach results:
// PoissonRecon's SparseMatrix product (SparseMatrix.inl, operator*) writes each output row in
// the order its unordered_map iterates, and that entry order reaches every later float sum of
// the base-depth multigrid (PoissonSparseMatrix.Multiply). This type keeps libc++'s singly
// linked node list and bucket array (each bucket holds the node before its first entry), so
// inserts, rehashes and iteration visit nodes exactly as libc++ does. Tier A, pinned by
// LibcxxUnorderedMapTests against oracle/libcxx_unordered_map_harness.cc.
//
// Translation notes:
// - Nodes live in arrays (key, value, cached hash, next); node 0 is libc++'s __first_node_
//   sentinel. Reset() returns to a fresh (zero-bucket) map without freeing the arrays, so a
//   caller that builds one map per matrix row does not allocate per row.
// - std::hash<int> is size_t( key ): a negative key sign-extends to 64 bits, so hashes are
//   ulong and __constrain_hash works on ulong.
// - Only insertion and lookup are ported (no erase), which is all its users need.

namespace ColmapSharp.Util;

/// <summary>
/// An int-keyed hash map that iterates in libc++'s <c>std::unordered_map</c> order. Port of
/// the relevant parts of libc++'s <c>__hash_table</c>.
/// </summary>
public sealed class LibcxxUnorderedMap<TValue>
{
	private const int None = -1;
	private const int Sentinel = 0;
	private int[] keys = new int[16];
	private TValue[] values = new TValue[16];
	private ulong[] hashes = new ulong[16];
	private int[] next = new int[16];

	// buckets[b]: the node before bucket b's first node, or None.
	private int[] buckets = [];
	private int bucketCount;

	/// <summary>An empty map with no buckets (a default-constructed unordered_map).</summary>
	public LibcxxUnorderedMap()
	{
		Reset();
	}

	/// <summary>The number of entries (size()).</summary>
	public int Count { get; private set; }

	/// <summary>The number of buckets (bucket_count()).</summary>
	public int BucketCount => bucketCount;

	/// <summary>The first node in iteration order, or -1 (begin()).</summary>
	public int First => next[Sentinel];

	/// <summary>Empties the map and drops its buckets, as a freshly constructed map; the storage is kept.</summary>
	public void Reset()
	{
		next[Sentinel] = None;
		bucketCount = 0;
		Count = 0;
	}

	/// <summary>The node after <paramref name="node"/> in iteration order, or -1.</summary>
	public int Next(int node) => next[node];

	/// <summary>A node's key.</summary>
	public int Key(int node) => keys[node];

	/// <summary>A node's value.</summary>
	public ref TValue Value(int node) => ref values[node];

	/// <summary>The node holding <paramref name="key"/>, or -1. Port of <c>find( key )</c>.</summary>
	public int Find(int key)
	{
		if (bucketCount == 0 || Count == 0)
		{
			return None;
		}

		ulong hash = Hash(key);
		ulong chash = ConstrainHash(hash, (ulong)bucketCount);
		int nd = buckets[chash];
		if (nd != None)
		{
			for (nd = next[nd]; nd != None && (hashes[nd] == hash || ConstrainHash(hashes[nd], (ulong)bucketCount) == chash); nd = next[nd])
			{
				if (hashes[nd] == hash && keys[nd] == key)
				{
					return nd;
				}
			}
		}

		return None;
	}

	/// <summary>
	/// Inserts <paramref name="key"/> with <paramref name="value"/> unless present, and returns
	/// its node. Port of <c>__emplace_unique_key_args</c> (operator[], emplace, insert).
	/// </summary>
	public int Insert(int key, TValue value)
	{
		int existing = Find(key);
		if (existing != None)
		{
			return existing;
		}

		ulong hash = Hash(key);
		int nd = AllocateNode(key, value, hash);
		if ((float)(Count + 1) > bucketCount * 1.0f)
		{
			ulong bc = (ulong)bucketCount;
			ulong grown = 2 * bc + (IsHashPower2(bc) ? 0UL : 1UL);
			ulong needed = (ulong)MathF.Ceiling((float)(Count + 1) / 1.0f);
			Rehash(Math.Max(grown, needed));
		}

		ulong chash = ConstrainHash(hash, (ulong)bucketCount);

		// insert_after __bucket_list_[__chash], or __first_node if bucket is null
		int pn = buckets[chash];
		if (pn == None)
		{
			pn = Sentinel;
			next[nd] = next[pn];
			next[pn] = nd;

			// fix up __bucket_list_
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

	/// <summary>The smallest prime at least <paramref name="n"/>. Port of libc++'s <c>__next_prime</c> (its contract).</summary>
	public static ulong NextPrime(ulong n)
	{
		if (n <= 2)
		{
			return 2;
		}

		ulong candidate = n;
		while (!IsPrime(candidate))
		{
			candidate++;
		}

		return candidate;
	}

	// std::hash<int>: size_t( key ), sign-extended.
	private static ulong Hash(int key) => unchecked((ulong)(long)key);

	private static bool IsHashPower2(ulong bc) => bc > 2 && (bc & (bc - 1)) == 0;

	private static ulong ConstrainHash(ulong h, ulong bc) => (bc & (bc - 1)) == 0 ? h & (bc - 1) : (h < bc ? h : h % bc);

	private static bool IsPrime(ulong n)
	{
		if (n < 2)
		{
			return false;
		}

		for (ulong d = 2; d * d <= n; d++)
		{
			if (n % d == 0)
			{
				return false;
			}
		}

		return true;
	}

	private int AllocateNode(int key, TValue value, ulong hash)
	{
		int nd = Count + 1;
		if (nd >= keys.Length)
		{
			int size = keys.Length * 2;
			Array.Resize(ref keys, size);
			Array.Resize(ref values, size);
			Array.Resize(ref hashes, size);
			Array.Resize(ref next, size);
		}

		keys[nd] = key;
		values[nd] = value;
		hashes[nd] = hash;
		return nd;
	}

	// __rehash<true>( n ) on the insert path (n always exceeds the bucket count there, but the
	// shrink branch is kept for fidelity).
	private void Rehash(ulong n)
	{
		if (n == 1)
		{
			n = 2;
		}
		else if ((n & (n - 1)) != 0)
		{
			n = NextPrime(n);
		}

		ulong bc = (ulong)bucketCount;
		if (n > bc)
		{
			DoRehash((int)n);
		}
		else if (n < bc)
		{
			ulong needed = (ulong)MathF.Ceiling((float)Count / 1.0f);
			n = Math.Max(n, IsHashPower2(bc) ? NextHashPow2(needed) : NextPrime(needed));
			if (n < bc)
			{
				DoRehash((int)n);
			}
		}
	}

	private static ulong NextHashPow2(ulong n) => n < 2 ? n : 1UL << (64 - System.Numerics.BitOperations.LeadingZeroCount(n - 1));

	// __do_rehash<true>: relinks the node list so every bucket's nodes are contiguous, moving a
	// node that lands in an already started bucket to just after that bucket's head.
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
