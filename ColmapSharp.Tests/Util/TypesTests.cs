// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TypesTests: colmap/util/types_test.cc ported 1:1, one method per gtest TEST(Suite, Name)
// named Suite_Name, testing ColmapSharp/Util/Types.cs. FlatHashSet<pair, PairHash> is a
// HashSet of tuples with PairHash.Instance as comparer; h(pair) is PairHash.Hash(pair).
//
// Not ported, because the C++ code they test is not ported (Types.cs header): Span.SizeAndEmpty
// (COLMAP's span<T>; C# uses System.Span<T>) and FilterView.Empty / All / None / Nominal /
// RangeExpression (COLMAP's filter_view; C# uses LINQ Where). Testing the BCL in their place
// would test no ColmapSharp code.

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Util;

public class TypesTests
{
	[Test]
	public async Task ShouldSwapImagePair_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(ShouldSwapImagePair(0, 0)).IsFalse();
			await Assert.That(ShouldSwapImagePair(0, 1)).IsFalse();
			await Assert.That(ShouldSwapImagePair(1, 0)).IsTrue();
			await Assert.That(ShouldSwapImagePair(1, 1)).IsFalse();
		}
	}

	[Test]
	public async Task ImagePairToPairId_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(ImagePairToPairId(0, 0)).IsEqualTo(0UL);
			await Assert.That(ImagePairToPairId(0, 1)).IsEqualTo(1UL);
			await Assert.That(ImagePairToPairId(0, 2)).IsEqualTo(2UL);
			await Assert.That(ImagePairToPairId(0, 3)).IsEqualTo(3UL);
			await Assert.That(ImagePairToPairId(1, 2)).IsEqualTo(MaxNumImages + 2);
			for (uint i = 0; i < 20; ++i)
			{
				for (uint j = 0; j < 20; ++j)
				{
					ulong pairId = ImagePairToPairId(i, j);
					(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
					if (i < j)
					{
						await Assert.That(imageId1).IsEqualTo(i);
						await Assert.That(imageId2).IsEqualTo(j);
					}
					else
					{
						await Assert.That(imageId2).IsEqualTo(i);
						await Assert.That(imageId1).IsEqualTo(j);
					}
				}
			}
		}
	}

	[Test]
	public async Task FeatureMatchHashing_Nominal()
	{
		var set = new HashSet<(uint, uint)>(PairHash.Instance);
		set.Add((1, 2));
		int sizeAfterFirst = set.Count;
		set.Add((1, 2));
		int sizeAfterDuplicate = set.Count;
		bool has00 = set.Contains((0, 0));
		bool has12 = set.Contains((1, 2));
		bool has21 = set.Contains((2, 1));
		set.Add((2, 1));
		using (Assert.Multiple())
		{
			await Assert.That(sizeAfterFirst).IsEqualTo(1);
			await Assert.That(sizeAfterDuplicate).IsEqualTo(1);
			await Assert.That(has00).IsFalse();
			await Assert.That(has12).IsTrue();
			await Assert.That(has21).IsFalse();
			await Assert.That(set.Count).IsEqualTo(2);
			await Assert.That(set.Contains((0, 0))).IsFalse();
			await Assert.That(set.Contains((1, 2))).IsTrue();
			await Assert.That(set.Contains((2, 1))).IsTrue();
		}
	}

	[Test]
	public async Task FeatureMatchHashing_LargeValues()
	{
		const uint hi = uint.MaxValue;
		const uint lo = 1;
		var set = new HashSet<(uint, uint)>(PairHash.Instance) { (hi, lo), (lo, hi), (hi, hi), (lo, lo) };
		using (Assert.Multiple())
		{
			await Assert.That(set.Count).IsEqualTo(4);
			await Assert.That(set.Contains((hi, lo))).IsTrue();
			await Assert.That(set.Contains((lo, hi))).IsTrue();
			await Assert.That(set.Contains((hi, hi))).IsTrue();
			await Assert.That(set.Contains((lo, lo))).IsTrue();
		}
	}

	[Test]
	public async Task FeatureMatchHashing_Deterministic()
	{
		using (Assert.Multiple())
		{
			await Assert.That(PairHash.Hash((42u, 99u))).IsEqualTo(PairHash.Hash((42u, 99u)));
			await Assert.That(PairHash.Hash((42u, 99u))).IsNotEqualTo(PairHash.Hash((99u, 42u)));
		}
	}

	[Test]
	public async Task SignedPairHashing_LargeValues()
	{
		const int hi = int.MaxValue;
		const int lo = int.MinValue;
		var set = new HashSet<(int, int)>(PairHash.Instance) { (hi, lo), (lo, hi), (hi, hi), (lo, lo) };
		using (Assert.Multiple())
		{
			await Assert.That(set.Count).IsEqualTo(4);
			await Assert.That(set.Contains((hi, lo))).IsTrue();
			await Assert.That(set.Contains((lo, hi))).IsTrue();
			await Assert.That(set.Contains((hi, hi))).IsTrue();
			await Assert.That(set.Contains((lo, lo))).IsTrue();
		}
	}

	[Test]
	public async Task SignedPairHashing_Deterministic()
	{
		using (Assert.Multiple())
		{
			await Assert.That(PairHash.Hash((-42, 99))).IsEqualTo(PairHash.Hash((-42, 99)));
			await Assert.That(PairHash.Hash((-42, 99))).IsNotEqualTo(PairHash.Hash((99, -42)));
			// Distinct negatives that share low bits with positives must not collide.
			await Assert.That(PairHash.Hash((-1, 0))).IsNotEqualTo(PairHash.Hash((0, -1)));
		}
	}

	[Test]
	public async Task Point3DPairHashing_Nominal()
	{
		var set = new HashSet<(ulong, ulong)>(PairHash.Instance);
		set.Add((1, 2));
		int sizeAfterFirst = set.Count;
		set.Add((1, 2));
		int sizeAfterDuplicate = set.Count;
		bool has00 = set.Contains((0, 0));
		bool has12 = set.Contains((1, 2));
		bool has21 = set.Contains((2, 1));
		set.Add((2, 1));
		using (Assert.Multiple())
		{
			await Assert.That(sizeAfterFirst).IsEqualTo(1);
			await Assert.That(sizeAfterDuplicate).IsEqualTo(1);
			await Assert.That(has00).IsFalse();
			await Assert.That(has12).IsTrue();
			await Assert.That(has21).IsFalse();
			await Assert.That(set.Count).IsEqualTo(2);
			await Assert.That(set.Contains((0, 0))).IsFalse();
			await Assert.That(set.Contains((1, 2))).IsTrue();
			await Assert.That(set.Contains((2, 1))).IsTrue();
		}
	}

	[Test]
	public async Task Point3DPairHashing_LargeValues()
	{
		const ulong hi = ulong.MaxValue;
		const ulong lo = 1;
		var set = new HashSet<(ulong, ulong)>(PairHash.Instance) { (hi, lo), (lo, hi), (hi, hi), (lo, lo) };
		using (Assert.Multiple())
		{
			await Assert.That(set.Count).IsEqualTo(4);
			await Assert.That(set.Contains((hi, lo))).IsTrue();
			await Assert.That(set.Contains((lo, hi))).IsTrue();
			await Assert.That(set.Contains((hi, hi))).IsTrue();
			await Assert.That(set.Contains((lo, lo))).IsTrue();
		}
	}

	[Test]
	public async Task Point3DPairHashing_Deterministic()
	{
		using (Assert.Multiple())
		{
			await Assert.That(PairHash.Hash((42UL, 99UL))).IsEqualTo(PairHash.Hash((42UL, 99UL)));
			await Assert.That(PairHash.Hash((42UL, 99UL))).IsNotEqualTo(PairHash.Hash((99UL, 42UL)));
		}
	}

	// C#-only: PairHash's .NET hash code must not collapse a pair to a ^ b, which would make
	// swapped pairs and every pair with the same XOR collide in a HashSet.
	[Test]
	public async Task PairHash_GetHashCodeMixesBothHalves()
	{
		var comparer = PairHash.Instance;
		using (Assert.Multiple())
		{
			await Assert.That(comparer.GetHashCode((1u, 2u))).IsNotEqualTo(comparer.GetHashCode((2u, 1u)));
			// 1 ^ 2 == 0 ^ 3 == 4 ^ 7.
			await Assert.That(comparer.GetHashCode((1u, 2u))).IsNotEqualTo(comparer.GetHashCode((0u, 3u)));
			await Assert.That(comparer.GetHashCode((0u, 3u))).IsNotEqualTo(comparer.GetHashCode((4u, 7u)));
			await Assert.That(comparer.GetHashCode((1, 2))).IsNotEqualTo(comparer.GetHashCode((2, 1)));
			await Assert.That(comparer.GetHashCode((1UL, 2UL))).IsNotEqualTo(comparer.GetHashCode((2UL, 1UL)));
		}
	}
}
