// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RandomUtils: colmap/math/random.h and random.cc - COLMAP's thread-local PRNG
// (SetPRNGSeed, kDefaultPRNGSeed) and the draws every seeded algorithm goes through:
// RandomUniformInteger, RandomUniformReal, RandomGaussian and the partial Fisher-Yates
// Shuffle. The engine is Mt19937.cs and the distribution algorithms are libc++'s, in
// LibcxxRandom.cs, because the pycolmap oracle is built against libc++ (PORTING_PLAN.md,
// Phase 1). Named RandomUtils (like MathUtils) so it does not shadow System.Random.
// Tests: ColmapSharp.Tests/Mathematics/RandomTests.cs (random_test.cc 1:1, plus the
// C#-only oracle comparison). random_eigen.h's helpers are in RandomEigen.cs.
//
// Tier A (exact): the same seed gives the same draws as COLMAP on macOS, bit for bit
// (see LibcxxRandom.cs for the one libm caveat in RandomGaussian).
//
// Translation notes:
// - `thread_local std::unique_ptr<std::mt19937> PRNG` becomes a [ThreadStatic] field
//   behind Prng, null until first use on each thread, exactly like COLMAP. The usual C#
//   caveat applies: an async method may resume on another thread after an await and then
//   sees that thread's PRNG. Seeded code must draw synchronously between awaits (COLMAP's
//   own threads never migrate). Parallel.For workers each get their own lazily seeded
//   engine, as COLMAP's ThreadPool workers do.
// - SetPRNGSeed also calls srand(seed) in COLMAP. Nothing in COLMAP 4.2.0 calls rand()
//   (random_eigen.h exists precisely to avoid Eigen's rand()-based Random()), and .NET has
//   no C runtime rand() to seed, so that line has no C# counterpart.
// - The templates over integer/floating T become generic math. RandomUniformInteger takes
//   any integer of at most 64 bits; RandomUniformReal and RandomGaussian take float or
//   double, the types libc++ supports (long double has no C# equivalent).
// - RandomGaussian builds a fresh normal_distribution per call as COLMAP does, so the polar
//   method's cached second value is thrown away and every call consumes a new pair.

using System.Numerics;

using ColmapSharp.Util;

namespace ColmapSharp.Mathematics;

/// <summary>
/// Port of colmap/math/random.h: COLMAP's per-thread seeded random number generator.
/// </summary>
public static class RandomUtils
{
	[ThreadStatic]
	private static Mt19937? prng;

	/// <summary>
	/// The calling thread's PRNG (COLMAP's <c>thread_local PRNG</c>); null until
	/// <see cref="SetPRNGSeed(uint)"/> or the first draw on this thread. Setting it to null
	/// mirrors <c>PRNG.reset()</c>.
	/// </summary>
	public static Mt19937? Prng
	{
		get => prng;
		set => prng = value;
	}

	/// <summary>
	/// Seed used by <see cref="SetPRNGSeed()"/> and by a thread's first draw when it was never
	/// seeded. Port of colmap::kDefaultPRNGSeed, which is a mutable global (not per thread).
	/// </summary>
	public static int DefaultPRNGSeed { get; set; }

	/// <summary>Initialize the calling thread's PRNG with <see cref="DefaultPRNGSeed"/>.</summary>
	public static void SetPRNGSeed()
	{
		// COLMAP's default argument converts the int kDefaultPRNGSeed to unsigned.
		SetPRNGSeed(unchecked((uint)DefaultPRNGSeed));
	}

	/// <summary>Initialize the calling thread's PRNG with the given seed.</summary>
	public static void SetPRNGSeed(uint seed)
	{
		prng = new Mt19937(seed);
	}

	/// <summary>
	/// Generate uniformly distributed random integer number in [min, max], both inclusive.
	/// This implementation is unbiased and thread-safe in contrast to <c>rand()</c>.
	/// </summary>
	public static T RandomUniformInteger<T>(T min, T max)
		where T : IBinaryInteger<T>
	{
		return LibcxxRandom.UniformInt(EnsureSeeded(), min, max);
	}

	/// <summary>
	/// Generate uniformly distributed random real number in [min, max).
	/// This implementation is unbiased and thread-safe in contrast to <c>rand()</c>.
	/// </summary>
	public static T RandomUniformReal<T>(T min, T max)
		where T : IBinaryFloatingPointIeee754<T>
	{
		return LibcxxRandom.UniformReal(EnsureSeeded(), min, max);
	}

	/// <summary>
	/// Generate Gaussian distributed random real number.
	/// This implementation is unbiased and thread-safe in contrast to <c>rand()</c>.
	/// </summary>
	public static T RandomGaussian<T>(T mean, T stddev)
		where T : IBinaryFloatingPointIeee754<T>
	{
		Mt19937 generator = EnsureSeeded();
		return new NormalDistribution<T>(mean, stddev).Next(generator);
	}

	/// <summary>
	/// Fisher-Yates shuffling. Only the first <paramref name="numToShuffle"/> elements are
	/// shuffled (each swapped with a uniformly chosen element at or after it); the rest of
	/// the list keeps whatever lands there.
	/// </summary>
	public static void Shuffle<T>(uint numToShuffle, IList<T> elems)
	{
		Check.Le((ulong)numToShuffle, (ulong)elems.Count);
		// Wraps to uint.MaxValue for an empty list, as in COLMAP; the loop never runs then.
		uint lastIdx = unchecked((uint)(elems.Count - 1));
		for (uint i = 0; i < numToShuffle; ++i)
		{
			uint j = RandomUniformInteger(i, lastIdx);
			(elems[(int)i], elems[(int)j]) = (elems[(int)j], elems[(int)i]);
		}
	}

	// COLMAP's `if (PRNG == nullptr) SetPRNGSeed();` at the top of every draw.
	private static Mt19937 EnsureSeeded()
	{
		if (prng == null)
		{
			SetPRNGSeed();
		}

		return prng!;
	}
}
