// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CombinationSamplerTests: colmap/optim/combination_sampler_test.cc ported 1:1, one method
// per gtest TEST(Suite, Name) named Suite_Name. Tests ColmapSharp/Optim/CombinationSampler.cs.
// The sampler is deterministic, so no PRNG seeding is needed.

using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Optim;

public class CombinationSamplerTests
{
	[Test]
	public async Task CombinationSampler_LessSamples()
	{
		var sampler = new CombinationSampler(2);
		sampler.Initialize(5);
		await Assert.That(sampler.MaxNumSamples()).IsEqualTo(10UL);
		var sampleSets = new List<HashSet<int>>();
		for (int i = 0; i < 10; ++i)
		{
			var samples = new List<int>();
			sampler.Sample(samples);
			await Assert.That(samples.Count).IsEqualTo(2);
			sampleSets.Add(new HashSet<int>(samples));
			await Assert.That(sampleSets[^1].Count).IsEqualTo(2);
			for (int j = 0; j < i; ++j)
			{
				await Assert.That(!sampleSets[j].Contains(samples[0]) || !sampleSets[j].Contains(samples[1]))
					.IsTrue();
			}
		}

		var wrapped = new List<int>();
		sampler.Sample(wrapped);
		await Assert.That(sampleSets[0].Contains(wrapped[0]) && sampleSets[0].Contains(wrapped[1])).IsTrue();
	}

	[Test]
	public async Task CombinationSampler_EqualSamples()
	{
		var sampler = new CombinationSampler(5);
		sampler.Initialize(5);
		await Assert.That(sampler.MaxNumSamples()).IsEqualTo(1UL);
		for (int i = 0; i < 100; ++i)
		{
			var samples = new List<int>();
			sampler.Sample(samples);
			await Assert.That(samples.Count).IsEqualTo(5);
			await Assert.That(new HashSet<int>(samples).Count).IsEqualTo(5);
		}
	}
}
