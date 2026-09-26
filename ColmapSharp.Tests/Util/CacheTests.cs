// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CacheTests: the LRUCache and ThreadSafeLRUCache cases of colmap/util/cache_test.cc, 1:1
// (TEST(Suite, Name) becomes Suite_Name). Tests ColmapSharp/Util/Cache.cs.
// Pending, not skipped: the eight MemoryConstrainedLRUCache cases land with
// MemoryConstrainedLRUCache, which nothing ported uses yet.
// ConcurrentGet's gmock expectations (Load(0) and Load(1) called exactly once) become call
// counters.

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

public class CacheTests
{
	private static async Task FillFive(Func<int, int> get, Func<int> numElems, Func<int, bool> exists)
	{
		await Assert.That(numElems()).IsEqualTo(0);
		for (int i = 0; i < 5; ++i)
		{
			await Assert.That(get(i)).IsEqualTo(i);
			await Assert.That(numElems()).IsEqualTo(i + 1);
			await Assert.That(exists(i)).IsTrue();
		}
	}

	[Test]
	public async Task LRUCache_Empty()
	{
		var cache = new LRUCache<int, int>(5, key => key);
		await Assert.That(cache.NumElems).IsEqualTo(0);
		await Assert.That(cache.MaxNumElems).IsEqualTo(5);
	}

	[Test]
	public async Task LRUCache_Get()
	{
		var cache = new LRUCache<int, int>(5, key => key);
		await FillFive(cache.Get, () => cache.NumElems, cache.Exists);

		await Assert.That(cache.Get(5)).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(5);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(5)).IsTrue();

		await Assert.That(cache.Get(5)).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(5);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(5)).IsTrue();

		await Assert.That(cache.Get(6)).IsEqualTo(6);
		await Assert.That(cache.NumElems).IsEqualTo(5);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(1)).IsFalse();
		await Assert.That(cache.Exists(6)).IsTrue();
	}

	[Test]
	public async Task LRUCache_Evict()
	{
		var cache = new LRUCache<int, int>(5, key => key);
		await FillFive(cache.Get, () => cache.NumElems, cache.Exists);

		await Assert.That(cache.Get(5)).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(5);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(5)).IsTrue();

		await Assert.That(cache.Evict(0)).IsFalse();

		await Assert.That(cache.Evict(1)).IsTrue();
		await Assert.That(cache.NumElems).IsEqualTo(4);
		await Assert.That(cache.Exists(1)).IsFalse();
	}

	[Test]
	public async Task LRUCache_Pop()
	{
		var cache = new LRUCache<int, int>(5, key => key);
		await FillFive(cache.Get, () => cache.NumElems, cache.Exists);

		await Assert.That(cache.Get(5)).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(5);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(5)).IsTrue();

		for (int expected = 4; expected >= 0; --expected)
		{
			cache.Pop();
			await Assert.That(cache.NumElems).IsEqualTo(expected);
		}

		cache.Pop();
		await Assert.That(cache.NumElems).IsEqualTo(0);
	}

	[Test]
	public async Task LRUCache_Clear()
	{
		var cache = new LRUCache<int, int>(5, key => key);
		await FillFive(cache.Get, () => cache.NumElems, cache.Exists);

		cache.Clear();
		await Assert.That(cache.NumElems).IsEqualTo(0);

		await Assert.That(cache.Get(0)).IsEqualTo(0);
		await Assert.That(cache.NumElems).IsEqualTo(1);
		await Assert.That(cache.Exists(0)).IsTrue();
	}

	[Test]
	public async Task ThreadSafeLRUCache_Empty()
	{
		var cache = new ThreadSafeLRUCache<int, int>(5, key => key);
		await Assert.That(cache.NumElems).IsEqualTo(0);
		await Assert.That(cache.MaxNumElems).IsEqualTo(5);
	}

	[Test]
	public async Task ThreadSafeLRUCache_Get()
	{
		var cache = new ThreadSafeLRUCache<int, int>(5, key => key);
		await FillFive(cache.Get, () => cache.NumElems, cache.Exists);

		await Assert.That(cache.Get(5)).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(5);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(5)).IsTrue();

		await Assert.That(cache.Get(5)).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(5);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(5)).IsTrue();

		await Assert.That(cache.Get(6)).IsEqualTo(6);
		await Assert.That(cache.NumElems).IsEqualTo(5);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(1)).IsFalse();
		await Assert.That(cache.Exists(6)).IsTrue();
	}

	[Test]
	public async Task ThreadSafeLRUCache_ConcurrentGet()
	{
		object gate = new();
		bool loaded = false;
		int load0Calls = 0;
		int load1Calls = 0;

		var cache = new ThreadSafeLRUCache<int, int>(2, key =>
		{
			if (key == 0)
			{
				Interlocked.Increment(ref load0Calls);
				lock (gate)
				{
					loaded = true;
					Monitor.PulseAll(gate);
				}

				return 2;
			}

			Interlocked.Increment(ref load1Calls);
			return 4;
		});

		int result1 = 0;
		int result2 = 0;
		var thread1 = new Thread(() => result1 = cache.Get(0));
		var thread2 = new Thread(() =>
		{
			lock (gate)
			{
				while (!loaded)
				{
					Monitor.Wait(gate);
				}
			}

			result2 = cache.Get(0);
		});
		thread1.Start();
		thread2.Start();
		thread1.Join();
		thread2.Join();

		await Assert.That(result1).IsEqualTo(2);
		await Assert.That(result2).IsEqualTo(2);
		await Assert.That(cache.Get(1)).IsEqualTo(4);
		await Assert.That(load0Calls).IsEqualTo(1);
		await Assert.That(load1Calls).IsEqualTo(1);
	}

	[Test]
	public async Task ThreadSafeLRUCache_Evict()
	{
		var cache = new ThreadSafeLRUCache<int, int>(5, key => key);
		await FillFive(cache.Get, () => cache.NumElems, cache.Exists);

		await Assert.That(cache.Get(5)).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(5);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(5)).IsTrue();

		await Assert.That(cache.Evict(0)).IsFalse();

		await Assert.That(cache.Evict(1)).IsTrue();
		await Assert.That(cache.NumElems).IsEqualTo(4);
		await Assert.That(cache.Exists(1)).IsFalse();
	}

	[Test]
	public async Task ThreadSafeLRUCache_Pop()
	{
		var cache = new ThreadSafeLRUCache<int, int>(5, key => key);
		await FillFive(cache.Get, () => cache.NumElems, cache.Exists);

		await Assert.That(cache.Get(5)).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(5);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(5)).IsTrue();

		for (int expected = 4; expected >= 0; --expected)
		{
			cache.Pop();
			await Assert.That(cache.NumElems).IsEqualTo(expected);
		}

		cache.Pop();
		await Assert.That(cache.NumElems).IsEqualTo(0);
	}

	[Test]
	public async Task ThreadSafeLRUCache_Clear()
	{
		var cache = new ThreadSafeLRUCache<int, int>(5, key => key);
		await FillFive(cache.Get, () => cache.NumElems, cache.Exists);

		cache.Clear();
		await Assert.That(cache.NumElems).IsEqualTo(0);

		await Assert.That(cache.Get(0)).IsEqualTo(0);
		await Assert.That(cache.NumElems).IsEqualTo(1);
		await Assert.That(cache.Exists(0)).IsTrue();
	}

	[Test]
	public async Task LRUCache_ExceptionSafety()
	{
		// If loadFn throws, the cache should remain in a consistent state.
		int loadCount = 0;
		var cache = new LRUCache<int, int>(5, key =>
		{
			++loadCount;
			if (key == 3)
			{
				throw new InvalidOperationException("load failed");
			}

			return key * 10;
		});

		// Load some initial entries.
		await Assert.That(cache.Get(0)).IsEqualTo(0);
		await Assert.That(cache.Get(1)).IsEqualTo(10);
		await Assert.That(cache.Get(2)).IsEqualTo(20);
		await Assert.That(cache.NumElems).IsEqualTo(3);

		// Attempting to load key=3 should throw.
		Assert.Throws<InvalidOperationException>(() => cache.Get(3));

		// The cache should still be in a valid state with the original 3 entries.
		await Assert.That(cache.NumElems).IsEqualTo(3);
		await Assert.That(cache.Exists(0)).IsTrue();
		await Assert.That(cache.Exists(1)).IsTrue();
		await Assert.That(cache.Exists(2)).IsTrue();
		await Assert.That(cache.Exists(3)).IsFalse();

		// Existing entries should still be retrievable.
		await Assert.That(cache.Get(0)).IsEqualTo(0);
		await Assert.That(cache.Get(1)).IsEqualTo(10);
		await Assert.That(cache.Get(2)).IsEqualTo(20);

		// Loading a new valid key should work fine.
		await Assert.That(cache.Get(4)).IsEqualTo(40);
		await Assert.That(cache.NumElems).IsEqualTo(4);
		await Assert.That(cache.Exists(4)).IsTrue();
	}
}
