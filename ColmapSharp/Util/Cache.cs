// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Cache: LRUCache and ThreadSafeLRUCache of colmap/util/cache.h. The SIFT CPU matcher
// (Feature/SiftMatcher.cs) reads its per-image descriptor indexes through a
// ThreadSafeLRUCache, as COLMAP's matching controllers do. Tests:
// ColmapSharp.Tests/Util/CacheTests.cs (the LRUCache and ThreadSafeLRUCache cases of
// cache_test.cc). MemoryConstrainedLRUCache is not ported yet: nothing ported uses it.
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
