// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// IdMap: the replacement for COLMAP's NodeHashMap<id, T> (colmap/util/hash_containers.h),
// the container Scene/Reconstruction.cs keeps its rigs, cameras, frames, images and 3D
// points in. Written here, not ported: COLMAP's map is a hash map whose iteration order is
// unspecified, and that order leaks into results (point ids assigned by Crop, the order of
// registered frames, sums over points). CLAUDE.md requires deterministic iteration, so this
// map iterates in ascending key order, always (divergence 21).
//
// Lookups go through a Dictionary (O(1), as in COLMAP). Each entry also records its slot in
// an ascending key list kept alongside:
// - Adding a key larger than every key in the list (the usual case, since COLMAP hands out
//   increasing ids) appends it. Any other add marks the list stale, and the next
//   enumeration rebuilds it with a sort.
// - Removing leaves a tombstone: the slot stays in the list, and enumeration skips slots
//   whose key is gone or has moved to another slot. When tombstones outnumber live entries
//   the list is compacted in one linear pass, which keeps the order without sorting, so
//   removal is amortized O(1). The incremental mapper deletes 3D points constantly between
//   enumerations and must never pay for a sort there.
// Enumerating while the map changes throws, like every .NET collection; take a snapshot
// (e.g. Keys.ToList()) to delete while iterating, where COLMAP uses erase(it++). Tests:
// ColmapSharp.Tests/Util/IdMapTests.cs (C#-only).

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace ColmapSharp.Util;

/// <summary>
/// An id-keyed map with O(1) lookup that enumerates in ascending key order. Replacement for
/// COLMAP's NodeHashMap with a deterministic iteration order.
/// </summary>
public sealed class IdMap<TKey, TValue> : IReadOnlyDictionary<TKey, TValue>
	where TKey : notnull, IComparable<TKey>
{
	// Compaction threshold: below this many tombstones a compaction is not worth its pass.
	private const int MinTombstonesToCompact = 64;

	private readonly Dictionary<TKey, Entry> _items = [];
	private readonly List<TKey> _sortedKeys = [];
	private bool _sortedKeysStale;
	private int _tombstones;
	private int _version;

	/// <summary>How many times enumeration had to re-sort the keys (a test hook: removals and in-order adds must not cause one).</summary>
	internal int SortCount { get; private set; }

	/// <summary>The number of entries.</summary>
	public int Count => _items.Count;

	/// <summary>The value for a key; throws <see cref="KeyNotFoundException"/> if absent.</summary>
	public TValue this[TKey key] => _items[key].Value;

	/// <summary>The keys, in ascending order.</summary>
	public IEnumerable<TKey> Keys
	{
		get
		{
			foreach (KeyValuePair<TKey, TValue> item in this)
			{
				yield return item.Key;
			}
		}
	}

	/// <summary>The values, in ascending key order.</summary>
	public IEnumerable<TValue> Values
	{
		get
		{
			foreach (KeyValuePair<TKey, TValue> item in this)
			{
				yield return item.Value;
			}
		}
	}

	/// <summary>Whether the key is present.</summary>
	public bool ContainsKey(TKey key) => _items.ContainsKey(key);

	/// <summary>Looks up a key.</summary>
	public bool TryGetValue(TKey key, [MaybeNullWhen(false)] out TValue value)
	{
		if (_items.TryGetValue(key, out Entry entry))
		{
			value = entry.Value;
			return true;
		}

		value = default;
		return false;
	}

	/// <summary>Adds the entry unless the key is present (C++ emplace(...).second).</summary>
	public bool TryAdd(TKey key, TValue value)
	{
		bool append = !_sortedKeysStale && (_sortedKeys.Count == 0 || _sortedKeys[^1].CompareTo(key) < 0);
		if (!_items.TryAdd(key, new Entry(value, append ? _sortedKeys.Count : -1)))
		{
			return false;
		}

		_version++;
		if (append)
		{
			_sortedKeys.Add(key);
		}
		else
		{
			_sortedKeysStale = true;
		}

		return true;
	}

	/// <summary>Removes the key if present (C++ erase(key)); returns whether it was.</summary>
	public bool Remove(TKey key)
	{
		if (!_items.Remove(key))
		{
			return false;
		}

		_version++;
		if (!_sortedKeysStale)
		{
			_tombstones++;
			if (_tombstones >= MinTombstonesToCompact && _tombstones > _items.Count)
			{
				Compact();
			}
		}

		return true;
	}

	/// <summary>Removes every entry.</summary>
	public void Clear()
	{
		_items.Clear();
		_sortedKeys.Clear();
		_sortedKeysStale = false;
		_tombstones = 0;
		_version++;
	}

	/// <summary>Enumerates the entries in ascending key order.</summary>
	public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
	{
		if (_sortedKeysStale)
		{
			RebuildSorted();
		}

		int version = _version;
		for (int slot = 0; slot < _sortedKeys.Count; slot++)
		{
			if (version != _version)
			{
				throw new InvalidOperationException("Collection was modified; enumeration operation may not execute.");
			}

			TKey key = _sortedKeys[slot];
			// Skip tombstones: the key was removed, or removed and re-added at another slot.
			if (_items.TryGetValue(key, out Entry entry) && entry.Slot == slot)
			{
				yield return new KeyValuePair<TKey, TValue>(key, entry.Value);
			}
		}

		if (version != _version)
		{
			throw new InvalidOperationException("Collection was modified; enumeration operation may not execute.");
		}
	}

	/// <inheritdoc/>
	System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

	// Drops the tombstones in one ordered pass; the live keys keep their relative order.
	private void Compact()
	{
		int write = 0;
		for (int slot = 0; slot < _sortedKeys.Count; slot++)
		{
			TKey key = _sortedKeys[slot];
			ref Entry entry = ref CollectionsMarshal.GetValueRefOrNullRef(_items, key);
			if (!Unsafe.IsNullRef(ref entry) && entry.Slot == slot)
			{
				entry.Slot = write;
				_sortedKeys[write++] = key;
			}
		}

		_sortedKeys.RemoveRange(write, _sortedKeys.Count - write);
		_tombstones = 0;
	}

	// After an out-of-order add: sort all live keys and renumber their slots.
	private void RebuildSorted()
	{
		_sortedKeys.Clear();
		_sortedKeys.AddRange(_items.Keys);
		_sortedKeys.Sort();
		SortCount++;
		for (int slot = 0; slot < _sortedKeys.Count; slot++)
		{
			CollectionsMarshal.GetValueRefOrNullRef(_items, _sortedKeys[slot]).Slot = slot;
		}

		_sortedKeysStale = false;
		_tombstones = 0;
	}

	private struct Entry(TValue value, int slot)
	{
		public TValue Value = value;

		// Index of the key in _sortedKeys; -1 until a rebuild places it.
		public int Slot = slot;
	}
}
