// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ProgressiveSamplerTests: colmap/optim/progressive_sampler_test.cc ported 1:1, one method
// per gtest TEST(Suite, Name) named Suite_Name. Tests ColmapSharp/Optim/ProgressiveSampler.cs.
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test, so each test starts with
// RandomUtils.SetPRNGSeed(0) and draws everything before its first await (the PRNG is per
// thread and an await may resume elsewhere).

using ColmapSharp.Mathematics;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Optim;

public class ProgressiveSamplerTests
{
	[Test]
	public async Task ProgressiveSampler_LessSamples()
	{
		RandomUtils.SetPRNGSeed(0);
		var sampler = new ProgressiveSampler(2);
		sampler.Initialize(5);
		ulong maxNumSamples = sampler.MaxNumSamples();
		List<List<int>> draws = Draw(sampler, 100);

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
	public async Task ProgressiveSampler_EqualSamples()
	{
		RandomUtils.SetPRNGSeed(0);
		var sampler = new ProgressiveSampler(5);
		sampler.Initialize(5);
		ulong maxNumSamples = sampler.MaxNumSamples();
		List<List<int>> draws = Draw(sampler, 100);

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

	[Test]
	public async Task ProgressiveSampler_Progressive()
	{
		RandomUtils.SetPRNGSeed(0);
		const int NumSamples = 5;
		var sampler = new ProgressiveSampler(NumSamples);
		sampler.Initialize(50);
		List<List<int>> draws = Draw(sampler, 100);

		using (Assert.Multiple())
		{
			int prevLastSample = 5;
			foreach (List<int> samples in draws)
			{
				for (int i = 0; i < samples.Count - 1; ++i)
				{
					await Assert.That(samples[i]).IsLessThan(samples[^1]);
					await Assert.That(samples[^1]).IsGreaterThanOrEqualTo(prevLastSample);
					prevLastSample = samples[^1];
				}
			}
		}
	}

	private static List<List<int>> Draw(ProgressiveSampler sampler, int count)
	{
		var draws = new List<List<int>>(count);
		for (int i = 0; i < count; ++i)
		{
			var samples = new List<int>();
			sampler.Sample(samples);
			draws.Add(samples);
		}

		return draws;
	}
}
