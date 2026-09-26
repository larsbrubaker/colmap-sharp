// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RandomSamplerTests: colmap/optim/random_sampler_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name. Tests ColmapSharp/Optim/RandomSampler.cs.
//
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does,
// and each test draws everything before its first await (the PRNG is per thread and an
// await may resume elsewhere).

using ColmapSharp.Mathematics;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Optim;

public class RandomSamplerTests
{
	[Test]
	public async Task RandomSampler_LessSamples()
	{
		var sampler = new RandomSampler(2);
		sampler.Initialize(5);
		ulong maxNumSamples = sampler.MaxNumSamples();
		var draws = new List<List<int>>();
		for (int i = 0; i < 100; ++i)
		{
			var samples = new List<int>();
			sampler.Sample(samples);
			draws.Add(samples);
		}

		using (Assert.Multiple())
		{
			await Assert.That(maxNumSamples).IsEqualTo(ulong.MaxValue);
			foreach (List<int> samples in draws)
			{
				await Assert.That(samples.Count).IsEqualTo(2);
				await Assert.That(new HashSet<int>(samples).Count).IsEqualTo(2);
			}
		}
	}

	[Test]
	public async Task RandomSampler_EqualSamples()
	{
		var sampler = new RandomSampler(5);
		sampler.Initialize(5);
		ulong maxNumSamples = sampler.MaxNumSamples();
		var draws = new List<List<int>>();
		for (int i = 0; i < 100; ++i)
		{
			var samples = new List<int>();
			sampler.Sample(samples);
			draws.Add(samples);
		}

		using (Assert.Multiple())
		{
			await Assert.That(maxNumSamples).IsEqualTo(ulong.MaxValue);
			foreach (List<int> samples in draws)
			{
				await Assert.That(samples.Count).IsEqualTo(5);
				await Assert.That(new HashSet<int>(samples).Count).IsEqualTo(5);
			}
		}
	}
}
