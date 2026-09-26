// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ProgressiveSampler: colmap/optim/progressive_sampler.h and progressive_sampler.cc - the
// PROSAC sampler (Chum and Matas, CVPR 2005), which draws from a growing prefix of
// quality-sorted data. Implements ISampler<TSelf> of Sampler.cs; siblings are
// RandomSampler.cs and CombinationSampler.cs. Tests:
// ColmapSharp.Tests/Optim/ProgressiveSamplerTests.cs (progressive_sampler_test.cc 1:1).
//
// Tier A (exact): the growth schedule is plain double arithmetic in COLMAP's evaluation
// order, and every draw goes through RandomUtils.RandomUniformInteger<uint>, so a seeded
// thread produces COLMAP's index sequence.
//
// Translation notes:
// - t_ (the number of Sample calls) stays a 64-bit unsigned counter and is compared with
//   the double T_n_p_ after conversion to double, as the C++ `t_ == T_n_p_` does.
// - Faithful to COLMAP, not to the paper: in progressive mode the random part is drawn
//   from [0, n - 2] and the mandatory element is index n, so index n - 1 is skipped and
//   index n can equal total_num_samples (for example num_samples == total_num_samples
//   yields index total_num_samples on the first call). Keeping this is what Tier A means;
//   the same happens with fewer samples once n has grown to total_num_samples. COLMAP's
//   RANSAC then reads past the end of the data (undefined behavior); SampleX/SampleXY
//   (Sampler.cs) throw a Check failure instead (docs/CPP_DIVERGENCES.md, entry 16).

using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Optim;

/// <summary>
/// Random sampler for PROSAC (Progressive Sample Consensus), as described in:
///
///    "Matching with PROSAC - Progressive Sample Consensus".
///        Ondrej Chum and Matas, CVPR 2005.
///
/// Port of colmap::ProgressiveSampler. Note that a separate sampler should be instantiated
/// per thread and that the data to be sampled from is assumed to be sorted according to the
/// quality function in descending order, i.e., higher quality data is closer to the front
/// of the list.
/// </summary>
public sealed class ProgressiveSampler : ISampler<ProgressiveSampler>
{
	// Number of iterations before PROSAC behaves like RANSAC. Default value is chosen
	// according to the recommended value in the paper.
	private const int NumProgressiveIterations = 200000;

	private readonly int numSamples;
	private int totalNumSamples;

	// The number of generated samples, i.e. the number of calls to `Sample`.
	private ulong t;
	private int n;

	// Variables defined in equation 3.
	private double tN;
	private double tNP;

	/// <summary>Create a sampler that draws <paramref name="numSamples"/> indices per sample.</summary>
	public ProgressiveSampler(int numSamples)
	{
		this.numSamples = numSamples;
	}

	/// <inheritdoc/>
	public static bool IsRandomized => true;

	/// <inheritdoc/>
	public static ProgressiveSampler Create(int numSamples) => new ProgressiveSampler(numSamples);

	/// <inheritdoc/>
	public void Initialize(int totalNumSamples)
	{
		Check.Le(numSamples, totalNumSamples);
		this.totalNumSamples = totalNumSamples;

		t = 0;
		n = numSamples;

		// Compute T_n using recurrent relation in equation 3 (first part).
		tN = NumProgressiveIterations;
		tNP = 1.0;
		for (int i = 0; i < numSamples; ++i)
		{
			tN *= (double)(numSamples - i) / (double)(totalNumSamples - i);
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
		t += 1;

		sampledIdxs.Clear();
		sampledIdxs.EnsureCapacity(numSamples);

		// Compute T_n_p_ using recurrent relation in equation 3 (second part).
		if (t == tNP && n < totalNumSamples)
		{
			double tNPlus1 = tN * (n + 1.0) / (n + 1.0 - numSamples);
			tNP += Math.Ceiling(tNPlus1 - tN);
			tN = tNPlus1;
			n += 1;
		}

		// Decide how many samples to draw from which part of the data as
		// specified in equation 5.
		int numRandomSamples = numSamples;
		// size_t in COLMAP; it can only go below zero when numRandomSamples is 0 below, and
		// then it is never used.
		int maxRandomSampleIdx = n - 1;
		bool progressive = tNP >= t;
		if (progressive)
		{
			numRandomSamples -= 1;
			maxRandomSampleIdx -= 1;
		}

		// Draw semi-random samples as described in algorithm 1.
		for (int i = 0; i < numRandomSamples; ++i)
		{
			while (true)
			{
				int randomIdx = (int)RandomUtils.RandomUniformInteger(0u, (uint)maxRandomSampleIdx);
				// VectorContainsValue: a linear search, as in COLMAP.
				if (!sampledIdxs.Contains(randomIdx))
				{
					sampledIdxs.Add(randomIdx);
					break;
				}
			}
		}

		// In progressive sampling mode, the last element is mandatory.
		if (progressive)
		{
			sampledIdxs.Add(n);
		}
	}
}
