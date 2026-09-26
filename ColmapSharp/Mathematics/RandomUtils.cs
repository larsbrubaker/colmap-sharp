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
// - Test isolation (C#-only): COLMAP's gtest_main reseeds the PRNG with 0 before every test,
//   and its tests run one at a time on one thread. TUnit runs tests in parallel on pool
//   threads and may start a test body on a different thread than its hooks, so a hook
//   cannot reach "the test's thread". BeginIsolatedScope instead tags the test's execution
//   context (AsyncLocal, which flows through awaits and into new threads, Task.Run and
//   Parallel.For workers) with a scope. A thread that touches the PRNG under a scope its
//   [ThreadStatic] engine was not created for drops that engine and behaves like a fresh
//   COLMAP thread: Prng reads null, and its first draw seeds with the scope's seed (the
//   test seed, 0, which is also DefaultPRNGSeed) rather than whatever DefaultPRNGSeed holds.
//   Production never calls it: the scope slot stays null and the only cost is one static
//   null check per access.
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

	// Test isolation (header, "Test isolation"). isolationScope is null until the test
	// assembly calls BeginIsolatedScope; threadScope is the scope the calling thread's prng
	// was created under. scopeResetSeed is set when a scope change dropped the thread's
	// engine, and seeds the next lazy draw in place of DefaultPRNGSeed; any explicit
	// SetPRNGSeed or Prng assignment clears it.
	private static AsyncLocal<IsolationScope?>? isolationScope;

	[ThreadStatic]
	private static IsolationScope? threadScope;

	[ThreadStatic]
	private static uint? scopeResetSeed;

	/// <summary>
	/// The calling thread's PRNG (COLMAP's <c>thread_local PRNG</c>); null until
	/// <see cref="SetPRNGSeed(uint)"/> or the first draw on this thread. Setting it to null
	/// mirrors <c>PRNG.reset()</c>.
	/// </summary>
	public static Mt19937? Prng
	{
		get
		{
			SyncIsolationScope();
			return prng;
		}
		set
		{
			SyncIsolationScope();
			scopeResetSeed = null;
			prng = value;
		}
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
		SyncIsolationScope();
		scopeResetSeed = null;
		prng = new Mt19937(seed);
	}

	/// <summary>
	/// C#-only, for the test harness (the counterpart of gtest_main's per-test
	/// <c>SetPRNGSeed(0)</c>): starts a new PRNG scope in the current execution context and
	/// seeds the calling thread with <paramref name="seed"/>. Any other thread that later
	/// touches the PRNG under this context (the test body after a thread hop, an async
	/// continuation, a Task.Run, Parallel.For or new Thread worker) first drops the engine
	/// left there by earlier work and starts unseeded, like a fresh COLMAP thread, except that
	/// its first draw seeds with <paramref name="seed"/>. Code that runs outside any scope
	/// after one was begun sees a fresh thread that seeds with DefaultPRNGSeed.
	/// </summary>
	internal static void BeginIsolatedScope(uint seed)
	{
		if (isolationScope == null)
		{
			Interlocked.CompareExchange(ref isolationScope, new AsyncLocal<IsolationScope?>(), null);
		}

		var scope = new IsolationScope(seed);
		isolationScope!.Value = scope;
		threadScope = scope;
		scopeResetSeed = null;
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
		SyncIsolationScope();
		if (prng == null)
		{
			if (scopeResetSeed is uint seed)
			{
				SetPRNGSeed(seed);
			}
			else
			{
				SetPRNGSeed();
			}
		}

		return prng!;
	}

	// Resets the calling thread's engine when it was created under a different isolation
	// scope than the one flowing in the current execution context. A no-op in production.
	private static void SyncIsolationScope()
	{
		AsyncLocal<IsolationScope?>? local = isolationScope;
		if (local == null)
		{
			return;
		}

		IsolationScope? current = local.Value;
		if (!ReferenceEquals(current, threadScope))
		{
			threadScope = current;
			scopeResetSeed = current?.Seed;
			prng = null;
		}
	}

	// One test's scope; compared by reference, so two tests with the same seed still differ.
	private sealed class IsolationScope(uint seed)
	{
		public uint Seed { get; } = seed;
	}
}
