// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Sampler: colmap/optim/sampler.h - the interface every RANSAC sampler implements
// (Initialize, MaxNumSamples, Sample), COLMAP's is_randomized_sampler trait, and the
// SampleX/SampleXY helpers that gather the sampled elements. The implementations are
// RandomSampler.cs, ProgressiveSampler.cs and CombinationSampler.cs next to this file;
// the consumer is the RANSAC port (optim/ransac.h, loransac.h), which is templated on the
// sampler type. Tests: the three *SamplerTests files in ColmapSharp.Tests/Optim.
//
// Translation notes:
// - COLMAP's abstract base class plus template parameter becomes two interfaces. ISampler
//   carries the instance members and is usable as an ordinary type. ISampler<TSelf> adds
//   what RANSAC needs from the *type*: `Sampler thread_sampler(kMinNumSamples)` becomes
//   TSelf.Create(n), and the `is_randomized_sampler<Sampler>` trait becomes
//   TSelf.IsRandomized. RANSAC is generic over `TSampler : class, ISampler<TSampler>`
//   (Ransac.cs): every sampler is a sealed class, because a sampler carries per-run state
//   (the PROSAC schedule, RandomSampler's running permutation) that a struct copy would
//   silently fork.
// - size_t sample indices become int, since they index C# lists and arrays.
//   MaxNumSamples stays ulong: the randomized samplers report size_t's maximum and
//   CombinationSampler reports NChooseK, which is 64-bit.
// - SampleX/SampleXY's `thread_local std::vector<size_t>` scratch becomes a [ThreadStatic]
//   List<int>, so the hot RANSAC loop does not allocate. X/Y are read-only spans and the
//   outputs are spans, so arrays and lists (via CollectionsMarshal.AsSpan) both work.
// - Each sampled index is checked against the data length. In COLMAP an index past the end
//   reads past the std::vector (undefined behavior); ProgressiveSampler produces one when
//   its progressive growth reaches the last element (see ProgressiveSampler.cs). Here that
//   fails loudly with a Check instead (docs/CPP_DIVERGENCES.md, entry 16).

using ColmapSharp.Util;

namespace ColmapSharp.Optim;

/// <summary>
/// Port of colmap::Sampler: a sampling method for RANSAC-based estimators.
/// </summary>
public interface ISampler
{
	/// <summary>Initialize the sampler, before calling the <see cref="Sample"/> method.</summary>
	void Initialize(int totalNumSamples);

	/// <summary>Maximum number of unique samples that can be generated.</summary>
	ulong MaxNumSamples();

	/// <summary>
	/// Sample <c>num_samples</c> elements from all samples. <paramref name="sampledIdxs"/> is
	/// overwritten with the sampled indices.
	/// </summary>
	void Sample(List<int> sampledIdxs);
}

/// <summary>
/// The static half of colmap::Sampler that generic RANSAC code needs: constructing a sampler
/// of the concrete type and COLMAP's <c>is_randomized_sampler</c> trait.
/// </summary>
public interface ISampler<TSelf> : ISampler
	where TSelf : ISampler<TSelf>
{
	/// <summary>Construct a sampler drawing <paramref name="numSamples"/> elements per sample.</summary>
	static abstract TSelf Create(int numSamples);

	/// <summary>
	/// Port of colmap::is_randomized_sampler: whether the sampler draws from the PRNG, in
	/// which case RANSAC seeds it per thread.
	/// </summary>
	static abstract bool IsRandomized { get; }
}

/// <summary>
/// Port of the Sampler::SampleX and Sampler::SampleXY templates.
/// </summary>
public static class SamplerExtensions
{
	[ThreadStatic]
	private static List<int>? sampledIdxsScratch;

	/// <summary>
	/// Sample elements from <paramref name="x"/> into <paramref name="xRand"/>.
	///
	/// Note that <c>x.Length</c> should equal <c>num_total_samples</c> and
	/// <c>xRand.Length</c> should equal <c>num_samples</c>.
	/// </summary>
	public static void SampleX<TSampler, TX>(this TSampler sampler, ReadOnlySpan<TX> x, Span<TX> xRand)
		where TSampler : ISampler
	{
		List<int> sampledIdxs = sampledIdxsScratch ??= new List<int>();
		sampler.Sample(sampledIdxs);
		for (int i = 0; i < xRand.Length; ++i)
		{
			Check.Lt(sampledIdxs[i], x.Length);
			xRand[i] = x[sampledIdxs[i]];
		}
	}

	/// <summary>
	/// Sample elements from <paramref name="x"/> and <paramref name="y"/> into
	/// <paramref name="xRand"/> and <paramref name="yRand"/>.
	///
	/// Note that <c>x.Length</c> should equal <c>num_total_samples</c> and
	/// <c>xRand.Length</c> should equal <c>num_samples</c>. The same applies for
	/// <paramref name="y"/> and <paramref name="yRand"/>.
	/// </summary>
	public static void SampleXY<TSampler, TX, TY>(
		this TSampler sampler,
		ReadOnlySpan<TX> x,
		ReadOnlySpan<TY> y,
		Span<TX> xRand,
		Span<TY> yRand)
		where TSampler : ISampler
	{
		Check.Eq(x.Length, y.Length);
		Check.Eq(xRand.Length, yRand.Length);
		List<int> sampledIdxs = sampledIdxsScratch ??= new List<int>();
		sampler.Sample(sampledIdxs);
		for (int i = 0; i < xRand.Length; ++i)
		{
			Check.Lt(sampledIdxs[i], x.Length);
			xRand[i] = x[sampledIdxs[i]];
			yRand[i] = y[sampledIdxs[i]];
		}
	}
}
