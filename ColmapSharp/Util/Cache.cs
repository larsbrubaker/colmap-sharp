// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Cache: LRUCache, ThreadSafeLRUCache and MemoryConstrainedLRUCache of colmap/util/cache.h.
// The SIFT CPU matcher (Feature/SiftMatcher.cs) reads its per-image descriptor indexes
// through a ThreadSafeLRUCache, as COLMAP's matching controllers do; COLMAP's MVS workspace
// caches images and depth maps in a MemoryConstrainedLRUCache. Tests:
// ColmapSharp.Tests/Util/CacheTests.cs and CacheTests.MemoryConstrained.cs (cache_test.cc).
// MemoryConstrainedLRUCache::MaxNumElems is declared in cache.h but never defined, so it
// has no port.
//
// Tier A (exact): plain bookkeeping.
//
// Translation notes:
// - std::shared_ptr<value_t> becomes the reference itself; the load function's result is
//   THROW_CHECK_NOTNULL'd in the thread-safe cache like COLMAP's.
// - std::list + NodeHashMap<key, list iterator> becomes LinkedList + Dictionary<key, node>.
//   Nothing iterates the map, so its order cannot leak.
// - ThreadSafeLRUCache's promise / shared_future per entry becomes a small Monitor-based
//   future (Entry): the first caller for a key loads it outside the lock while later
//   callers for the same key block until the same result (or the same exception) is set.
// - MemoryConstrainedLRUCache's duck-typed `value->NumBytes()` becomes the ICacheSizedValue
//   constraint, and size_t byte counts become long (a count is never negative: Pop and
//   UpdateNumBytes keep COLMAP's underflow checks).

namespace ColmapSharp.Util;

/// <summary>
/// Port of colmap::LRUCache: least recently used cache. Whenever the cache size is exceeded,
/// the least recently used (by Get) element is deleted.
/// </summary>
public sealed class LRUCache<TKey, TValue>
	where TKey : notnull
{
	private readonly int maxNumElems;
	private readonly Func<TKey, TValue> loadFn;

	// Front = most recently used.
	private readonly LinkedList<KeyValuePair<TKey, TValue>> elemsList = new();
	private readonly Dictionary<TKey, LinkedListNode<KeyValuePair<TKey, TValue>>> elemsMap = new();

	/// <summary>A cache of at most <paramref name="maxNumElems"/> values computed by <paramref name="loadFn"/>.</summary>
	public LRUCache(int maxNumElems, Func<TKey, TValue> loadFn)
	{
		this.loadFn = Check.NotNull(loadFn);
		Check.Gt(maxNumElems, 0);
		this.maxNumElems = maxNumElems;
	}

	/// <summary>The number of elements in the cache.</summary>
	public int NumElems => elemsMap.Count;

	/// <summary>The maximum number of elements.</summary>
	public int MaxNumElems => maxNumElems;

	/// <summary>Whether the element with the given key exists.</summary>
	public bool Exists(TKey key) => elemsMap.ContainsKey(key);

	/// <summary>The value of an element, either from the cache or newly computed.</summary>
	public TValue Get(TKey key)
	{
		if (elemsMap.TryGetValue(key, out var node))
		{
			elemsList.Remove(node);
			elemsList.AddFirst(node);
			return node.Value.Value;
		}

		// Call loadFn before modifying the cache data structures so that if it throws, the
		// cache remains in a consistent state.
		TValue value = loadFn(key);
		var newNode = elemsList.AddFirst(new KeyValuePair<TKey, TValue>(key, value));
		elemsMap.Add(key, newNode);
		if (elemsMap.Count > maxNumElems)
		{
			Pop();
		}

		return value;
	}

	/// <summary>Manually evicts an element; true if it was in the cache.</summary>
	public bool Evict(TKey key)
	{
		if (elemsMap.Remove(key, out var node))
		{
			elemsList.Remove(node);
			return true;
		}

		return false;
	}

	/// <summary>Pops the least recently used element.</summary>
	public void Pop()
	{
		if (elemsList.Last is { } last)
		{
			elemsMap.Remove(last.Value.Key);
			elemsList.RemoveLast();
		}
	}

	/// <summary>Clears all elements.</summary>
	public void Clear()
	{
		elemsList.Clear();
		elemsMap.Clear();
	}
}

/// <summary>
/// Port of colmap::ThreadSafeLRUCache: an <see cref="LRUCache{TKey, TValue}"/> safe to use
/// from several threads, where concurrent requests for one key load it only once.
/// </summary>
public sealed class ThreadSafeLRUCache<TKey, TValue>
	where TKey : notnull
{
	private readonly Func<TKey, TValue> loadFn;
	private readonly object cacheMutex = new();
	private readonly LRUCache<TKey, Entry> cache;

	/// <summary>A cache of at most <paramref name="maxNumElems"/> values computed by <paramref name="loadFn"/>.</summary>
	public ThreadSafeLRUCache(int maxNumElems, Func<TKey, TValue> loadFn)
	{
		cache = new LRUCache<TKey, Entry>(maxNumElems, _ => new Entry());
		this.loadFn = Check.NotNull(loadFn);
	}

	/// <summary>The number of elements in the cache.</summary>
	public int NumElems
	{
		get
		{
			lock (cacheMutex)
			{
				return cache.NumElems;
			}
		}
	}

	/// <summary>The maximum number of elements.</summary>
	public int MaxNumElems
	{
		get
		{
			lock (cacheMutex)
			{
				return cache.MaxNumElems;
			}
		}
	}

	/// <summary>Whether the element with the given key exists.</summary>
	public bool Exists(TKey key)
	{
		lock (cacheMutex)
		{
			return cache.Exists(key);
		}
	}

	/// <summary>
	/// The value of an element, either from the cache or newly computed. A load that throws
	/// evicts the entry and rethrows to every caller waiting on it.
	/// </summary>
	public TValue Get(TKey key)
	{
		bool shouldLoad = false;
		Entry entry;
		lock (cacheMutex)
		{
			entry = cache.Get(key);
			if (!entry.IsLoading)
			{
				shouldLoad = true;
				entry.IsLoading = true;
			}
		}

		if (shouldLoad)
		{
			try
			{
				TValue value = loadFn(key);
				Check.That(value is not null, "load function returned null");
				entry.SetValue(value);
			}
			catch (Exception e)
			{
				// Evict the cache entry after load failed and set the exception.
				lock (cacheMutex)
				{
					cache.Evict(key);
				}

				entry.SetException(e);
			}
		}

		return entry.Wait();
	}

	/// <summary>Manually evicts an element; true if it was in the cache.</summary>
	public bool Evict(TKey key)
	{
		lock (cacheMutex)
		{
			return cache.Evict(key);
		}
	}

	/// <summary>Pops the least recently used element.</summary>
	public void Pop()
	{
		lock (cacheMutex)
		{
			cache.Pop();
		}
	}

	/// <summary>Clears all elements.</summary>
	public void Clear()
	{
		lock (cacheMutex)
		{
			cache.Clear();
		}
	}

	// The promise / shared_future pair of COLMAP's Entry. The loader runs synchronously on
	// the first caller's thread, so a blocking wait (std::shared_future::get) is correct.
	private sealed class Entry
	{
		private readonly object gate = new();
		private bool isSet;
		private TValue? value;
		private Exception? exception;

		public bool IsLoading { get; set; }

		public void SetValue(TValue newValue)
		{
			lock (gate)
			{
				value = newValue;
				isSet = true;
				Monitor.PulseAll(gate);
			}
		}

		public void SetException(Exception e)
		{
			lock (gate)
			{
				exception = e;
				isSet = true;
				Monitor.PulseAll(gate);
			}
		}

		public TValue Wait()
		{
			lock (gate)
			{
				while (!isSet)
				{
					Monitor.Wait(gate);
				}

				if (exception is not null)
				{
					System.Runtime.ExceptionServices.ExceptionDispatchInfo.Throw(exception);
				}

				return value!;
			}
		}
	}
}

/// <summary>
/// A value a <see cref="MemoryConstrainedLRUCache{TKey, TValue}"/> can hold: the C++ template
/// requires each element to implement <c>size_t NumBytes() const</c>.
/// </summary>
public interface ICacheSizedValue
{
	/// <summary>The value's size in memory, in bytes.</summary>
	long NumBytes { get; }
}

/// <summary>
/// Port of colmap::MemoryConstrainedLRUCache: least recently used cache constrained by the
/// total memory of its elements. Whenever the memory limit is exceeded, the least recently
/// used (by Get) element is deleted, but the most recent one is always kept.
/// </summary>
public sealed class MemoryConstrainedLRUCache<TKey, TValue>
	where TKey : notnull
	where TValue : class, ICacheSizedValue
{
	private readonly long maxNumBytes;
	private readonly Func<TKey, TValue> loadFn;
	private long numBytes;

	// Front = most recently used.
	private readonly LinkedList<KeyValuePair<TKey, TValue>> elemsList = new();

	// Mapping from key to (location in list, num_bytes).
	private readonly Dictionary<TKey, (LinkedListNode<KeyValuePair<TKey, TValue>> Node, long NumBytes)> elemsMap = new();

	/// <summary>A cache of values computed by <paramref name="loadFn"/> totalling at most <paramref name="maxNumBytes"/> bytes.</summary>
	public MemoryConstrainedLRUCache(long maxNumBytes, Func<TKey, TValue> loadFn)
	{
		this.loadFn = Check.NotNull(loadFn);
		Check.Gt(maxNumBytes, 0L);
		this.maxNumBytes = maxNumBytes;
	}

	/// <summary>The size in bytes of the elements in the cache (as of their last load or UpdateNumBytes).</summary>
	public long NumBytes => numBytes;

	/// <summary>The maximum size in bytes.</summary>
	public long MaxNumBytes => maxNumBytes;

	/// <summary>The number of elements in the cache.</summary>
	public int NumElems => elemsMap.Count;

	/// <summary>Whether the element with the given key exists.</summary>
	public bool Exists(TKey key) => elemsMap.ContainsKey(key);

	/// <summary>
	/// The value of an element, either from the cache or newly computed. Loading a value that
	/// takes the cache over its limit pops least recently used elements, but never the new one.
	/// </summary>
	public TValue Get(TKey key)
	{
		if (elemsMap.TryGetValue(key, out var found))
		{
			elemsList.Remove(found.Node);
			elemsList.AddFirst(found.Node);
			return found.Node.Value.Value;
		}

		TValue value = Check.NotNull(loadFn(key));
		long valueNumBytes = value.NumBytes;

		// COLMAP looks the key up again after loading: a load function that re-entered the
		// cache may have inserted it meanwhile, in which case the new value replaces it. Like
		// COLMAP, the replaced element's bytes are not subtracted.
		var newNode = elemsList.AddFirst(new KeyValuePair<TKey, TValue>(key, value));
		if (elemsMap.Remove(key, out var stale))
		{
			elemsList.Remove(stale.Node);
		}

		elemsMap.Add(key, (newNode, valueNumBytes));

		numBytes += valueNumBytes;
		while (numBytes > maxNumBytes && elemsMap.Count > 1)
		{
			Pop();
		}

		return value;
	}

	/// <summary>Manually evicts an element; true if it was in the cache.</summary>
	public bool Evict(TKey key)
	{
		if (elemsMap.Remove(key, out var entry))
		{
			numBytes -= entry.NumBytes;
			elemsList.Remove(entry.Node);
			return true;
		}

		return false;
	}

	/// <summary>Pops the least recently used element.</summary>
	public void Pop()
	{
		if (elemsList.Last is { } last)
		{
			long lastNumBytes = elemsMap[last.Value.Key].NumBytes;
			Check.Ge(numBytes, lastNumBytes, "Unsigned underflow in MemoryConstrainedLRUCache::Pop");
			numBytes -= lastNumBytes;
			elemsMap.Remove(last.Value.Key);
			elemsList.RemoveLast();
		}
	}

	/// <summary>
	/// Re-reads the size of an element whose value changed in place (it must be in the
	/// cache), marks it most recently used, and pops elements while over the limit.
	/// </summary>
	public void UpdateNumBytes(TKey key)
	{
		if (!elemsMap.TryGetValue(key, out var entry))
		{
			// std::unordered_map::at throws std::out_of_range.
			throw new KeyNotFoundException($"MemoryConstrainedLRUCache::UpdateNumBytes: key {key} is not in the cache");
		}

		Check.Ge(numBytes, entry.NumBytes, "Unsigned underflow in MemoryConstrainedLRUCache::UpdateNumBytes");
		numBytes -= entry.NumBytes;
		long newNumBytes = Get(key).NumBytes;
		elemsMap[key] = (entry.Node, newNumBytes);
		numBytes += newNumBytes;

		while (numBytes > maxNumBytes && elemsMap.Count > 1)
		{
			Pop();
		}
	}

	/// <summary>Clears all elements.</summary>
	public void Clear()
	{
		elemsList.Clear();
		elemsMap.Clear();
		numBytes = 0;
	}
}
