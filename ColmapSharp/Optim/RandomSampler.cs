// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RandomSampler: colmap/optim/random_sampler.h and random_sampler.cc - plain RANSAC
// sampling, a partial Fisher-Yates shuffle of all indices per sample. Implements the
// ISampler<TSelf> interface of Sampler.cs; siblings are ProgressiveSampler.cs and
// CombinationSampler.cs. Tests: ColmapSharp.Tests/Optim/RandomSamplerTests.cs
// (random_sampler_test.cc 1:1).
//
// Tier A (exact): draws go through RandomUtils.Shuffle, so a seeded thread produces the
// same index sequence as COLMAP. Like COLMAP's, the index permutation carries over from
// one Sample call to the next (only the first num_samples slots are reshuffled), which
// is part of what makes the sequence match.

using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Optim;

/// <summary>
/// Random sampler for RANSAC-based methods. Port of colmap::RandomSampler.
///
/// Note that a separate sampler should be instantiated per thread.
/// </summary>
public sealed class RandomSampler : ISampler<RandomSampler>
{
	private readonly int numSamples;
	private readonly List<int> sampleIdxs = new List<int>();

	/// <summary>Create a sampler that draws <paramref name="numSamples"/> indices per sample.</summary>
	public RandomSampler(int numSamples)
	{
		this.numSamples = numSamples;
	}

	/// <inheritdoc/>
	public static bool IsRandomized => true;

	/// <inheritdoc/>
	public static RandomSampler Create(int numSamples) => new RandomSampler(numSamples);

	/// <inheritdoc/>
	public void Initialize(int totalNumSamples)
	{
		Check.Le(numSamples, totalNumSamples);
		// std::iota over the resized vector.
		sampleIdxs.Clear();
		for (int i = 0; i < totalNumSamples; ++i)
		{
			sampleIdxs.Add(i);
		}
	}

	/// <inheritdoc/>
	public ulong MaxNumSamples()
	{
		return ulong.MaxValue;
	}

	/// <inheritdoc/>
	public void Sample(List<int> sampledIdxs)
	{
		RandomUtils.Shuffle((uint)numSamples, sampleIdxs);

		sampledIdxs.Clear();
		for (int i = 0; i < numSamples; ++i)
		{
			sampledIdxs.Add(sampleIdxs[i]);
		}
	}
}
