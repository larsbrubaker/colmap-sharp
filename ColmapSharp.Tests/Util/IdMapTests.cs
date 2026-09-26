// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// IdMapTests: C#-only tests of ColmapSharp/Util/IdMap.cs, the deterministic replacement for
// COLMAP's NodeHashMap (no COLMAP test covers a container). They pin the ascending-key
// iteration after out-of-order adds and removes, that changing the map while enumerating it
// throws, and the performance contract the incremental mapper needs: removals interleaved
// with enumerations never re-sort (checked through the SortCount hook, not a timer, so the
// guard cannot flake), cross-checked against a SortedDictionary, and that the key list
// stays correct across compactions.

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

public class IdMapTests
{
	[Test]
	public async Task CSharpOnly_IteratesInAscendingKeyOrder()
	{
		var map = new IdMap<uint, string>();
		await Assert.That(map.TryAdd(1, "a")).IsTrue();
		await Assert.That(map.TryAdd(5, "e")).IsTrue();
		await Assert.That(map.TryAdd(3, "c")).IsTrue();
		await Assert.That(map.TryAdd(3, "x")).IsFalse();
		await Assert.That(map.Keys.SequenceEqual(new uint[] { 1, 3, 5 })).IsTrue();
		await Assert.That(map.Values.SequenceEqual(new[] { "a", "c", "e" })).IsTrue();

		await Assert.That(map.Remove(5)).IsTrue();
		await Assert.That(map.Remove(5)).IsFalse();
		await Assert.That(map.TryAdd(7, "g")).IsTrue();
		await Assert.That(map.Remove(1)).IsTrue();
		await Assert.That(map.TryAdd(2, "b")).IsTrue();
		await Assert.That(map.Keys.SequenceEqual(new uint[] { 2, 3, 7 })).IsTrue();
		await Assert.That(map[7]).IsEqualTo("g");
		await Assert.That(map.Count).IsEqualTo(3);

		map.Clear();
		await Assert.That(map.Count).IsEqualTo(0);
		await Assert.That(map.TryAdd(4, "d")).IsTrue();
		await Assert.That(map.Keys.SequenceEqual(new uint[] { 4 })).IsTrue();
	}

	[Test]
	public async Task CSharpOnly_ModifyingWhileEnumeratingThrows()
	{
		var map = new IdMap<uint, string>();
		map.TryAdd(1, "a");
		map.TryAdd(2, "b");
		await Assert.That(() =>
		{
			foreach (uint key in map.Keys)
			{
				map.Remove(key);
			}
		}).Throws<InvalidOperationException>();
	}

	[Test]
	public async Task CSharpOnly_RemovalsBetweenEnumerationsNeverResort()
	{
		// The mapper's pattern: ids allocated increasing, points deleted constantly between
		// passes over all points.
		const int kNumAdds = 1_000_000;
		const int kRounds = 10;
		const int kRemovesPerRound = 10_000;
		var map = new IdMap<ulong, int>();
		var reference = new SortedDictionary<ulong, int>();
		for (ulong id = 1; id <= kNumAdds; id++)
		{
			map.TryAdd(id, (int)id);
			reference.Add(id, (int)id);
		}

		var random = new Random(0);
		ulong nextId = kNumAdds + 1;
		bool allMatched = true;
		for (int round = 0; round < kRounds; round++)
		{
			for (int i = 0; i < kRemovesPerRound; i++)
			{
				ulong id = (ulong)random.NextInt64(1, (long)nextId);
				allMatched &= map.Remove(id) == reference.Remove(id);
			}

			// New points keep getting increasing ids, some of them re-using nothing.
			for (int i = 0; i < 100; i++, nextId++)
			{
				map.TryAdd(nextId, (int)nextId);
				reference.Add(nextId, (int)nextId);
			}

			allMatched &= map.Count == reference.Count;
			allMatched &= map.Keys.SequenceEqual(reference.Keys);
		}

		await Assert.That(allMatched).IsTrue();
		await Assert.That(map.SortCount).IsEqualTo(0);

		// A removed id added back below the largest key is out of order: one lazy re-sort.
		ulong readded = reference.Keys.First() + 1;
		while (reference.ContainsKey(readded))
		{
			readded++;
		}

		map.TryAdd(readded, 0);
		reference.Add(readded, 0);
		await Assert.That(map.Keys.SequenceEqual(reference.Keys)).IsTrue();
		await Assert.That(map.SortCount).IsEqualTo(1);
	}

	[Test]
	public async Task CSharpOnly_CompactionKeepsOrderAndLookups()
	{
		// Removing more than half of at least 64 entries compacts the key list (drops the
		// tombstones and renumbers the slots). Afterwards enumeration, lookups and further
		// adds and removes must behave as before. (Checked with temporary instrumentation when
		// this test was written: Compact() runs once in the first phase and three more times in
		// the rounds.)
		var map = new IdMap<uint, uint>();
		var reference = new SortedDictionary<uint, uint>();
		for (uint id = 1; id <= 200; id++)
		{
			map.TryAdd(id, id * 10);
			reference.Add(id, id * 10);
		}

		bool allMatched = true;

		// Mixed positions: every id not divisible by 3 (134 of 200), from both ends inwards.
		var toRemove = reference.Keys.Where(id => id % 3 != 0).ToList();
		for (int i = 0, j = toRemove.Count - 1; i <= j; i++, j--)
		{
			allMatched &= map.Remove(toRemove[i]) == reference.Remove(toRemove[i]);
			if (i != j)
			{
				allMatched &= map.Remove(toRemove[j]) == reference.Remove(toRemove[j]);
			}
		}

		allMatched &= map.SequenceEqual(reference);

		// Further rounds after the compaction: increasing adds, removals (enough to compact
		// again several times), an out-of-order re-add of a removed id, and enumerations in
		// between.
		var random = new Random(0);
		uint nextId = 201;
		for (int round = 0; round < 20; round++)
		{
			for (int i = 0; i < 30; i++, nextId++)
			{
				allMatched &= map.TryAdd(nextId, nextId * 10) == reference.TryAdd(nextId, nextId * 10);
			}

			for (int i = 0; i < 120; i++)
			{
				uint id = (uint)random.Next(1, (int)nextId);
				allMatched &= map.Remove(id) == reference.Remove(id);
			}

			if (round % 5 == 4)
			{
				uint readded = (uint)random.Next(1, (int)nextId);
				allMatched &= map.TryAdd(readded, readded * 10) == reference.TryAdd(readded, readded * 10);
			}

			allMatched &= map.Count == reference.Count;
			allMatched &= map.SequenceEqual(reference);
			foreach ((uint id, uint value) in reference)
			{
				allMatched &= map.TryGetValue(id, out uint mapValue) && mapValue == value;
			}
		}

		await Assert.That(allMatched).IsTrue();
	}
}
