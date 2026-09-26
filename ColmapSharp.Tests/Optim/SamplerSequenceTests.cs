// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SamplerSequenceTests: C#-ONLY tests (no COLMAP counterpart). They pin the Tier A claim of
// RandomSampler.cs and ProgressiveSampler.cs - same seed, same index sequence as COLMAP -
// which the ported *_test.cc files cannot see, since those only check sizes and
// uniqueness. They also cover SamplerExtensions.SampleXY from Sampler.cs.
//
// Expected sequences come from a differential harness: COLMAP 4.2.0's random.cc,
// random_sampler.cc and progressive_sampler.cc compiled unmodified with Apple clang and
// libc++ (macOS arm64), with logging.h/misc.h replaced by minimal shims, calling
// SetPRNGSeed(0), constructing the sampler, Initialize, then Sample repeatedly.

using ColmapSharp.Mathematics;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Optim;

public class SamplerSequenceTests
{
	[Test]
	public async Task RandomSampler_SeededSequenceMatchesCpp()
	{
		int[][] expected =
		{
			new[] { 5, 1, 0 }, new[] { 3, 8, 5 }, new[] { 0, 6, 4 }, new[] { 5, 1, 6 },
			new[] { 4, 9, 8 }, new[] { 9, 7, 1 }, new[] { 4, 8, 3 }, new[] { 2, 7, 1 },
		};

		var sampler = new RandomSampler(3);
		sampler.Initialize(10);
		List<int[]> actual = Draw(sampler, expected.Length);

		await AssertSequence(actual, expected);
	}

	[Test]
	public async Task ProgressiveSampler_SeededSequenceMatchesCpp()
	{
		int[][] expected =
		{
			new[] { 0, 1, 4 }, new[] { 0, 1, 4 }, new[] { 1, 2, 4 }, new[] { 0, 2, 4 },
			new[] { 0, 2, 4 }, new[] { 1, 2, 4 }, new[] { 2, 0, 4 }, new[] { 1, 0, 4 },
		};

		var sampler = new ProgressiveSampler(3);
		sampler.Initialize(20);
		List<int[]> actual = Draw(sampler, expected.Length);

		await AssertSequence(actual, expected);
	}

	[Test]
	public async Task SampleXY_GathersSampledElements()
	{
		// Same seed and sampler as RandomSampler_SeededSequenceMatchesCpp, so the first
		// sample is indices { 5, 1, 0 }.
		int[] x = Enumerable.Range(0, 10).Select(i => i * 10).ToArray();
		string[] y = Enumerable.Range(0, 10).Select(i => $"y{i}").ToArray();
		var xRand = new int[3];
		var yRand = new string[3];

		var sampler = new RandomSampler(3);
		sampler.Initialize(10);
		sampler.SampleXY<RandomSampler, int, string>(x, y, xRand, yRand);

		using (Assert.Multiple())
		{
			await Assert.That(xRand.SequenceEqual(new[] { 50, 10, 0 })).IsTrue();
			await Assert.That(yRand.SequenceEqual(new[] { "y5", "y1", "y0" })).IsTrue();
		}
	}

	private static List<int[]> Draw(ISampler sampler, int count)
	{
		var draws = new List<int[]>(count);
		var samples = new List<int>();
		for (int i = 0; i < count; ++i)
		{
			sampler.Sample(samples);
			draws.Add(samples.ToArray());
		}

		return draws;
	}

	private static async Task AssertSequence(List<int[]> actual, int[][] expected)
	{
		await Assert.That(actual.Count).IsEqualTo(expected.Length);
		using (Assert.Multiple())
		{
			for (int i = 0; i < expected.Length; ++i)
			{
				await Assert.That(string.Join(",", actual[i])).IsEqualTo(string.Join(",", expected[i]));
			}
		}
	}
}
