// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CacheTests.MemoryConstrained: the eight MemoryConstrainedLRUCache cases of
// colmap/util/cache_test.cc, 1:1 (Empty, Get, Pop, Evict, Clear, UpdateNumBytes,
// PopNumBytesConsistency, UpdateNumBytesConsistency). Tests ColmapSharp/Util/Cache.cs. The
// LRUCache and ThreadSafeLRUCache cases are in CacheTests.cs.

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

public partial class CacheTests
{
	private static MemoryConstrainedLRUCache<int, SizedElem> NewSizedCache(long maxNumBytes) =>
		new(maxNumBytes, key => new SizedElem(key));

	private static async Task FillFiveSized(MemoryConstrainedLRUCache<int, SizedElem> cache)
	{
		await Assert.That(cache.NumElems).IsEqualTo(0);
		for (int i = 0; i < 5; ++i)
		{
			await Assert.That(cache.Get(i).NumBytes).IsEqualTo(i);
			await Assert.That(cache.NumElems).IsEqualTo(i + 1);
			await Assert.That(cache.Exists(i)).IsTrue();
		}
	}

	[Test]
	public async Task MemoryConstrainedLRUCache_Empty()
	{
		var cache = NewSizedCache(5);
		await Assert.That(cache.NumElems).IsEqualTo(0);
		await Assert.That(cache.NumBytes).IsEqualTo(0);
		await Assert.That(cache.MaxNumBytes).IsEqualTo(5);
	}

	[Test]
	public async Task MemoryConstrainedLRUCache_Get()
	{
		var cache = NewSizedCache(10);
		await FillFiveSized(cache);

		await Assert.That(cache.Get(5).NumBytes).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(2);
		await Assert.That(cache.NumBytes).IsEqualTo(9);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(5)).IsTrue();

		await Assert.That(cache.Get(5).NumBytes).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(2);
		await Assert.That(cache.NumBytes).IsEqualTo(9);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(5)).IsTrue();

		await Assert.That(cache.Get(6).NumBytes).IsEqualTo(6);
		await Assert.That(cache.NumElems).IsEqualTo(1);
		await Assert.That(cache.NumBytes).IsEqualTo(6);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(1)).IsFalse();
		await Assert.That(cache.Exists(6)).IsTrue();

		await Assert.That(cache.Get(1).NumBytes).IsEqualTo(1);
		await Assert.That(cache.NumElems).IsEqualTo(2);
		await Assert.That(cache.NumBytes).IsEqualTo(7);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(1)).IsTrue();
		await Assert.That(cache.Exists(6)).IsTrue();
	}

	[Test]
	public async Task MemoryConstrainedLRUCache_Pop()
	{
		var cache = NewSizedCache(10);
		await FillFiveSized(cache);

		await Assert.That(cache.Get(5).NumBytes).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(2);
		await Assert.That(cache.NumBytes).IsEqualTo(9);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(4)).IsTrue();
		await Assert.That(cache.Exists(5)).IsTrue();

		cache.Pop();
		await Assert.That(cache.NumElems).IsEqualTo(1);
		await Assert.That(cache.NumBytes).IsEqualTo(5);
		await Assert.That(cache.Exists(4)).IsFalse();
		await Assert.That(cache.Exists(5)).IsTrue();

		cache.Pop();
		await Assert.That(cache.NumElems).IsEqualTo(0);
		await Assert.That(cache.NumBytes).IsEqualTo(0);
	}

	[Test]
	public async Task MemoryConstrainedLRUCache_Evict()
	{
		var cache = NewSizedCache(10);
		await FillFiveSized(cache);

		await Assert.That(cache.Get(5).NumBytes).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(2);
		await Assert.That(cache.NumBytes).IsEqualTo(9);
		await Assert.That(cache.Exists(0)).IsFalse();
		await Assert.That(cache.Exists(4)).IsTrue();
		await Assert.That(cache.Exists(5)).IsTrue();

		await Assert.That(cache.Evict(0)).IsFalse();

		await Assert.That(cache.Evict(5)).IsTrue();
		await Assert.That(cache.NumElems).IsEqualTo(1);
		await Assert.That(cache.NumBytes).IsEqualTo(4);
		await Assert.That(cache.Exists(4)).IsTrue();
		await Assert.That(cache.Exists(5)).IsFalse();
	}

	[Test]
	public async Task MemoryConstrainedLRUCache_Clear()
	{
		var cache = NewSizedCache(10);
		await FillFiveSized(cache);

		cache.Clear();
		await Assert.That(cache.NumElems).IsEqualTo(0);
		await Assert.That(cache.NumBytes).IsEqualTo(0);

		await Assert.That(cache.Get(1).NumBytes).IsEqualTo(1);
		await Assert.That(cache.NumBytes).IsEqualTo(1);
		await Assert.That(cache.NumElems).IsEqualTo(1);
		await Assert.That(cache.Exists(1)).IsTrue();
	}

	[Test]
	public async Task MemoryConstrainedLRUCache_UpdateNumBytes()
	{
		var cache = NewSizedCache(50);
		await FillFiveSized(cache);

		await Assert.That(cache.NumBytes).IsEqualTo(10);

		cache.Get(4).NumBytes = 3;
		await Assert.That(cache.NumBytes).IsEqualTo(10);
		cache.UpdateNumBytes(4);
		await Assert.That(cache.NumBytes).IsEqualTo(9);

		cache.Get(2).NumBytes = 3;
		await Assert.That(cache.NumBytes).IsEqualTo(9);
		cache.UpdateNumBytes(2);
		await Assert.That(cache.NumBytes).IsEqualTo(10);

		cache.Get(0).NumBytes = 40;
		await Assert.That(cache.NumBytes).IsEqualTo(10);
		cache.UpdateNumBytes(0);
		await Assert.That(cache.NumBytes).IsEqualTo(50);

		cache.Clear();
		await Assert.That(cache.NumBytes).IsEqualTo(0);
		await Assert.That(cache.Get(2).NumBytes).IsEqualTo(2);
		await Assert.That(cache.NumBytes).IsEqualTo(2);
	}

	[Test]
	public async Task MemoryConstrainedLRUCache_PopNumBytesConsistency()
	{
		// Pop should maintain correct num_bytes tracking.
		var cache = NewSizedCache(100);

		cache.Get(10);
		cache.Get(20);
		cache.Get(5);
		await Assert.That(cache.NumBytes).IsEqualTo(35);
		await Assert.That(cache.NumElems).IsEqualTo(3);

		// Pop should remove the LRU element (key=10, 10 bytes) and update num_bytes.
		cache.Pop();
		await Assert.That(cache.NumBytes).IsEqualTo(25);
		await Assert.That(cache.NumElems).IsEqualTo(2);

		cache.Pop();
		await Assert.That(cache.NumBytes).IsEqualTo(5);
		await Assert.That(cache.NumElems).IsEqualTo(1);

		cache.Pop();
		await Assert.That(cache.NumBytes).IsEqualTo(0);
		await Assert.That(cache.NumElems).IsEqualTo(0);

		// Pop on empty cache should be safe.
		cache.Pop();
		await Assert.That(cache.NumBytes).IsEqualTo(0);
		await Assert.That(cache.NumElems).IsEqualTo(0);
	}

	[Test]
	public async Task MemoryConstrainedLRUCache_UpdateNumBytesConsistency()
	{
		// Should correctly track byte counts even when elements shrink.
		var cache = NewSizedCache(100);

		cache.Get(10);
		cache.Get(20);
		await Assert.That(cache.NumBytes).IsEqualTo(30);

		// Shrink element from 20 bytes to 5 bytes.
		cache.Get(20).NumBytes = 5;
		cache.UpdateNumBytes(20);
		await Assert.That(cache.NumBytes).IsEqualTo(15);

		// Shrink element to 0 bytes.
		cache.Get(10).NumBytes = 0;
		cache.UpdateNumBytes(10);
		await Assert.That(cache.NumBytes).IsEqualTo(5);

		// Grow it back.
		cache.Get(10).NumBytes = 8;
		cache.UpdateNumBytes(10);
		await Assert.That(cache.NumBytes).IsEqualTo(13);
	}

	// cache_test.cc's SizedElem: its size is whatever the test sets.
	private sealed class SizedElem(long numBytes) : ICacheSizedValue
	{
		public long NumBytes { get; set; } = numBytes;
	}
}
