// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CombinationSampler: colmap/optim/combination_sampler.h and combination_sampler.cc -
// deterministic sampling that walks every N-choose-K combination once, in lexicographic
// order, and then starts over. Implements ISampler<TSelf> of Sampler.cs; siblings are
// RandomSampler.cs and ProgressiveSampler.cs. Uses MathUtils.NextCombination and NChooseK.
// Tests: ColmapSharp.Tests/Optim/CombinationSamplerTests.cs (combination_sampler_test.cc 1:1).
//
// Tier A (exact): no randomness; the combination order is NextCombination's, which is
// itself a Tier A port.

using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Optim;

/// <summary>
/// Random sampler for RANSAC-based methods that generates unique samples. Port of
/// colmap::CombinationSampler.
///
/// Note that a separate sampler should be instantiated per thread and it assumes that the
/// input data is shuffled in advance.
/// </summary>
public sealed class CombinationSampler : ISampler<CombinationSampler>
{
	private readonly int numSamples;
	private int[] totalSampleIdxs = Array.Empty<int>();

	/// <summary>Create a sampler that draws <paramref name="numSamples"/> indices per sample.</summary>
	public CombinationSampler(int numSamples)
	{
		this.numSamples = numSamples;
	}

	/// <inheritdoc/>
	public static bool IsRandomized => false;

	/// <inheritdoc/>
	public static CombinationSampler Create(int numSamples) => new CombinationSampler(numSamples);

	/// <inheritdoc/>
	public void Initialize(int totalNumSamples)
	{
		Check.Le(numSamples, totalNumSamples);
		if (totalSampleIdxs.Length != totalNumSamples)
		{
			totalSampleIdxs = new int[totalNumSamples];
		}

		// Note that the samples must be in increasing order for `NextCombination`.
		ResetToFirstCombination();
	}

	/// <inheritdoc/>
	public ulong MaxNumSamples()
	{
		return MathUtils.NChooseK((ulong)totalSampleIdxs.Length, (ulong)numSamples);
	}

	/// <inheritdoc/>
	public void Sample(List<int> sampledIdxs)
	{
		sampledIdxs.Clear();
		for (int i = 0; i < numSamples; ++i)
		{
			sampledIdxs.Add(totalSampleIdxs[i]);
		}

		if (!MathUtils.NextCombination(totalSampleIdxs.AsSpan(), numSamples))
		{
			// Reached all possible combinations, so reset to original state.
			// Note that the samples must be in increasing order for `NextCombination`.
			ResetToFirstCombination();
		}
	}

	// std::iota(total_sample_idxs_.begin(), total_sample_idxs_.end(), 0).
	private void ResetToFirstCombination()
	{
		for (int i = 0; i < totalSampleIdxs.Length; ++i)
		{
			totalSampleIdxs[i] = i;
		}
	}
}
