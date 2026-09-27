// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// LibcxxUnorderedMapTests (C#-only, not a COLMAP test): LibcxxUnorderedMap's bucket counts and
// iteration order against libc++'s std::unordered_map< int , float >, as recorded by
// oracle/libcxx_unordered_map_harness.cc in TestData/oracle/libcxx_unordered_map.json:
// random insert sequences from 1 to 500 keys (every rehash boundary up to 500), ascending
// keys, colliding multiples of 7 inserted in descending order, and many repeats. Each insert
// is find-then-insert or += 1, as PoissonRecon's SparseMatrix product does. Tier A.

using System.Text.Json;

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

public class LibcxxUnorderedMapTests
{
	[Test]
	public async Task IterationOrder_MatchesLibcxx()
	{
		JsonElement cases = OracleFixture.Load("libcxx_unordered_map.json").GetProperty("cases");
		var names = cases.EnumerateObject().Select(p => p.Name).Where(n => n.EndsWith("/keys", StringComparison.Ordinal)).Select(n => n[..^"/keys".Length]).ToList();
		await Assert.That(names.Count).IsEqualTo(17);
		var map = new LibcxxUnorderedMap<float>();
		foreach (string name in names)
		{
			int[] keys = cases.GetProperty(name + "/keys").EnumerateArray().Select(e => e.GetInt32()).ToArray();
			map.Reset();
			var bucketCounts = new List<long>();
			foreach (int key in keys)
			{
				int node = map.Find(key);
				if (node == -1)
				{
					map.Insert(key, key * 0.5f);
				}
				else
				{
					map.Value(node) += 1.0f;
				}

				bucketCounts.Add(map.BucketCount);
			}

			var order = new List<long>();
			var values = new List<double>();
			for (int node = map.First; node != -1; node = map.Next(node))
			{
				order.Add(map.Key(node));
				values.Add(map.Value(node));
			}

			await Assert.That(string.Join(",", bucketCounts)).IsEqualTo(string.Join(",", cases.GetProperty(name + "/bucketcounts").EnumerateArray().Select(e => e.GetInt64())));
			await Assert.That(string.Join(",", order)).IsEqualTo(string.Join(",", cases.GetProperty(name + "/order").EnumerateArray().Select(e => e.GetInt64())));
			await Assert.That(string.Join(",", values.Select(v => BitConverter.DoubleToInt64Bits(v)))).IsEqualTo(string.Join(",", cases.GetProperty(name + "/values").EnumerateArray().Select(e => BitConverter.DoubleToInt64Bits(e.GetDouble()))));
		}
	}

	[Test]
	public async Task NextPrime_IsTheSmallestPrimeAtLeastN()
	{
		// C#-only: __next_prime's contract on small values and around the libc++ bucket growth.
		await Assert.That(LibcxxUnorderedMap<float>.NextPrime(0)).IsEqualTo(2UL);
		await Assert.That(LibcxxUnorderedMap<float>.NextPrime(2)).IsEqualTo(2UL);
		await Assert.That(LibcxxUnorderedMap<float>.NextPrime(4)).IsEqualTo(5UL);
		await Assert.That(LibcxxUnorderedMap<float>.NextPrime(11)).IsEqualTo(11UL);
		await Assert.That(LibcxxUnorderedMap<float>.NextPrime(24)).IsEqualTo(29UL);
		await Assert.That(LibcxxUnorderedMap<float>.NextPrime(212)).IsEqualTo(223UL);
	}
}
