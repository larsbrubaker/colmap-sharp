// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// UnionFindTests: colmap/math/union_find_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name. Tests ColmapSharp/Mathematics/UnionFind.cs.
// Tier A (exact).

using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class UnionFindTests
{
	[Test]
	public async Task UnionFind_DefaultConstructor()
	{
		var uf = new UnionFind<int>();
		// After construction, each element should be its own parent
		using (Assert.Multiple())
		{
			await Assert.That(uf.Find(1)).IsEqualTo(1);
			await Assert.That(uf.Find(2)).IsEqualTo(2);
			await Assert.That(uf.Find(3)).IsEqualTo(3);
		}
	}

	[Test]
	public async Task UnionFind_Reserve()
	{
		var uf = new UnionFind<int>();
		uf.Reserve(10);
		using (Assert.Multiple())
		{
			await Assert.That(uf.Find(1)).IsEqualTo(1);
			await Assert.That(uf.Find(2)).IsEqualTo(2);
		}
	}

	[Test]
	public async Task UnionFind_FindSingleElement()
	{
		var uf = new UnionFind<int>();
		using (Assert.Multiple())
		{
			// Finding an element for the first time should return itself
			await Assert.That(uf.Find(42)).IsEqualTo(42);
			// Finding it again should still return itself
			await Assert.That(uf.Find(42)).IsEqualTo(42);
		}
	}

	[Test]
	public async Task UnionFind_UnionTwoElements()
	{
		var uf = new UnionFind<int>();
		uf.Union(1, 2);
		await Assert.That(uf.Find(1)).IsEqualTo(uf.Find(2));
	}

	[Test]
	public async Task UnionFind_UnionMultiplePairs()
	{
		var uf = new UnionFind<int>();
		uf.Union(1, 2);
		uf.Union(3, 4);
		uf.Union(5, 6);

		using (Assert.Multiple())
		{
			// Elements in the same set should have the same root
			await Assert.That(uf.Find(1)).IsEqualTo(uf.Find(2));
			await Assert.That(uf.Find(3)).IsEqualTo(uf.Find(4));
			await Assert.That(uf.Find(5)).IsEqualTo(uf.Find(6));

			// Elements in different sets should have different roots
			await Assert.That(uf.Find(1)).IsNotEqualTo(uf.Find(3));
			await Assert.That(uf.Find(1)).IsNotEqualTo(uf.Find(5));
			await Assert.That(uf.Find(3)).IsNotEqualTo(uf.Find(5));
		}
	}

	[Test]
	public async Task UnionFind_UnionChain()
	{
		var uf = new UnionFind<int>();
		// Create a chain: 1-2-3-4-5
		uf.Union(1, 2);
		uf.Union(2, 3);
		uf.Union(3, 4);
		uf.Union(4, 5);

		// All elements should have the same root
		var root = uf.Find(1);
		using (Assert.Multiple())
		{
			await Assert.That(uf.Find(2)).IsEqualTo(root);
			await Assert.That(uf.Find(3)).IsEqualTo(root);
			await Assert.That(uf.Find(4)).IsEqualTo(root);
			await Assert.That(uf.Find(5)).IsEqualTo(root);
		}
	}

	[Test]
	public async Task UnionFind_UnionTwoSets()
	{
		var uf = new UnionFind<int>();
		// Create two separate sets
		uf.Union(1, 2);
		uf.Union(3, 4);

		// Verify they are separate
		await Assert.That(uf.Find(1)).IsNotEqualTo(uf.Find(3));

		// Union the two sets
		uf.Union(2, 3);

		// Now all elements should have the same root
		var root = uf.Find(1);
		using (Assert.Multiple())
		{
			await Assert.That(uf.Find(2)).IsEqualTo(root);
			await Assert.That(uf.Find(3)).IsEqualTo(root);
			await Assert.That(uf.Find(4)).IsEqualTo(root);
		}
	}

	[Test]
	public async Task UnionFind_UnionSameElementTwice()
	{
		var uf = new UnionFind<int>();
		// Union an element with itself
		uf.Union(1, 1);
		await Assert.That(uf.Find(1)).IsEqualTo(1);

		// Union two elements, then union them again
		uf.Union(2, 3);
		var root = uf.Find(2);
		uf.Union(2, 3);
		// Should still have the same root
		using (Assert.Multiple())
		{
			await Assert.That(uf.Find(2)).IsEqualTo(root);
			await Assert.That(uf.Find(3)).IsEqualTo(root);
		}
	}

	[Test]
	public async Task UnionFind_PathCompression()
	{
		var uf = new UnionFind<int>();
		// Create a long chain: 1 -> 2 -> 3 -> 4 -> 5
		uf.Union(1, 2);
		uf.Union(2, 3);
		uf.Union(3, 4);
		uf.Union(4, 5);

		// Find root of element 1
		var root = uf.Find(1);

		using (Assert.Multiple())
		{
			// After path compression, finding 1 again should be efficient
			// The root should remain the same
			await Assert.That(uf.Find(1)).IsEqualTo(root);

			// All elements should still have the same root
			await Assert.That(uf.Find(2)).IsEqualTo(root);
			await Assert.That(uf.Find(3)).IsEqualTo(root);
			await Assert.That(uf.Find(4)).IsEqualTo(root);
			await Assert.That(uf.Find(5)).IsEqualTo(root);
		}
	}

	[Test]
	public async Task UnionFind_Compress()
	{
		var uf = new UnionFind<int>();
		uf.Union(1, 2);
		uf.Union(2, 3);

		uf.Compress();

		// Snapshot first: Find may insert, and it must not run while enumerating the map.
		var entries = uf.Parents.ToList();
		using (Assert.Multiple())
		{
			foreach (var (_, parent) in entries)
			{
				await Assert.That(parent).IsEqualTo(uf.Find(parent));
			}
		}
	}

	[Test]
	public async Task UnionFind_LargeNumberOfElements()
	{
		const int kNumElements = 1000;
		var uf = new UnionFind<int>();
		uf.Reserve(kNumElements);
		// Union many elements
		for (var i = 0; i < kNumElements; ++i)
		{
			uf.Union(i, (i + 1) % kNumElements);
		}

		// All elements should be in the same set
		var root = uf.Find(0);
		using (Assert.Multiple())
		{
			for (var i = 1; i < kNumElements; ++i)
			{
				await Assert.That(uf.Find(i)).IsEqualTo(root);
			}
		}
	}

	[Test]
	public async Task UnionFind_MultipleDisjointSets()
	{
		var uf = new UnionFind<int>();
		// Set 1: {1, 2, 3}
		uf.Union(1, 2);
		uf.Union(2, 3);

		// Set 2: {10, 20, 30}
		uf.Union(10, 20);
		uf.Union(20, 30);

		// Set 3: {100, 200, 300}
		uf.Union(100, 200);
		uf.Union(200, 300);

		using (Assert.Multiple())
		{
			// Verify elements in the same set have the same root
			await Assert.That(uf.Find(1)).IsEqualTo(uf.Find(2));
			await Assert.That(uf.Find(2)).IsEqualTo(uf.Find(3));
			await Assert.That(uf.Find(10)).IsEqualTo(uf.Find(20));
			await Assert.That(uf.Find(20)).IsEqualTo(uf.Find(30));
			await Assert.That(uf.Find(100)).IsEqualTo(uf.Find(200));
			await Assert.That(uf.Find(200)).IsEqualTo(uf.Find(300));

			// Verify elements in different sets have different roots
			await Assert.That(uf.Find(1)).IsNotEqualTo(uf.Find(10));
			await Assert.That(uf.Find(1)).IsNotEqualTo(uf.Find(100));
			await Assert.That(uf.Find(10)).IsNotEqualTo(uf.Find(100));
		}
	}

	[Test]
	public async Task UnionFind_StringType()
	{
		var uf = new UnionFind<string>();
		uf.Union("apple", "apricot");
		uf.Union("banana", "blueberry");
		uf.Union("apricot", "avocado");

		using (Assert.Multiple())
		{
			// Elements in the same set should have the same root
			await Assert.That(uf.Find("apple")).IsEqualTo(uf.Find("apricot"));
			await Assert.That(uf.Find("apple")).IsEqualTo(uf.Find("avocado"));
			await Assert.That(uf.Find("banana")).IsEqualTo(uf.Find("blueberry"));

			// Elements in different sets should have different roots
			await Assert.That(uf.Find("apple")).IsNotEqualTo(uf.Find("banana"));
		}
	}

	[Test]
	public async Task UnionFind_StarTopology()
	{
		var uf = new UnionFind<int>();
		// Create a star topology: center = 0, connected to 1, 2, 3, 4, 5
		for (var i = 1; i <= 5; ++i)
		{
			uf.Union(0, i);
		}

		// All elements should be in the same set
		using (Assert.Multiple())
		{
			for (var i = 1; i <= 5; ++i)
			{
				await Assert.That(uf.Find(0)).IsEqualTo(uf.Find(i));
			}
		}
	}

	[Test]
	public async Task UnionFind_ReverseUnion()
	{
		var uf = new UnionFind<int>();
		uf.Union(5, 4);
		uf.Union(4, 3);
		uf.Union(3, 2);
		uf.Union(2, 1);

		// All should be in the same set
		var root = uf.Find(5);
		using (Assert.Multiple())
		{
			for (var i = 1; i <= 5; ++i)
			{
				await Assert.That(uf.Find(i)).IsEqualTo(root);
			}
		}
	}

	[Test]
	public async Task UnionFind_FindIfExists()
	{
		var uf = new UnionFind<int>();
		var missing = uf.FindIfExists(42, out _);
		await Assert.That(missing).IsFalse();

		await Assert.That(uf.Find(42)).IsEqualTo(42);
		var hasRoot = uf.FindIfExists(42, out var root);
		using (Assert.Multiple())
		{
			await Assert.That(hasRoot).IsTrue();
			await Assert.That(root).IsEqualTo(42);

			await Assert.That(uf.FindIfExists(1, out _)).IsFalse();
			await Assert.That(uf.FindIfExists(2, out _)).IsFalse();
		}

		uf.Union(2, 1);
		using (Assert.Multiple())
		{
			await Assert.That(uf.FindIfExists(1, out _)).IsTrue();
			await Assert.That(uf.FindIfExists(2, out _)).IsTrue();
		}
	}

	[Test]
	public async Task UnionFind_Parents()
	{
		var uf = new UnionFind<int>();
		await Assert.That(uf.Parents.Count).IsEqualTo(0);

		uf.Find(1);
		using (Assert.Multiple())
		{
			await Assert.That(uf.Parents.Count).IsEqualTo(1);
			await Assert.That(uf.Parents.ContainsKey(1)).IsTrue();
		}

		uf.Union(2, 3);
		using (Assert.Multiple())
		{
			await Assert.That(uf.Parents.Count).IsEqualTo(3);
			await Assert.That(uf.Parents.ContainsKey(2)).IsTrue();
			await Assert.That(uf.Parents.ContainsKey(3)).IsTrue();
		}

		uf.Union(1, 2);
		await Assert.That(uf.Parents.Count).IsEqualTo(3);
	}

	[Test]
	public async Task UnionFind_ParentsGroupByRoot()
	{
		var uf = new UnionFind<int>();
		uf.Union(1, 2);
		uf.Union(2, 3);
		uf.Union(10, 20);

		uf.Compress();
		var groups = new Dictionary<int, List<int>>();
		foreach (var (elem, parent) in uf.Parents)
		{
			if (!groups.TryGetValue(parent, out var members))
			{
				members = [];
				groups.Add(parent, members);
			}

			members.Add(elem);
		}

		using (Assert.Multiple())
		{
			await Assert.That(groups.Count).IsEqualTo(2);
			await Assert.That(groups[uf.Find(1)].Count).IsEqualTo(3);
			await Assert.That(groups[uf.Find(10)].Count).IsEqualTo(2);
		}
	}
}
