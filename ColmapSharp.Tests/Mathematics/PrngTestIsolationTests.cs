// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PrngTestIsolationTests (C#-only): proves the per-test PRNG seeding in
// ColmapSharp.Tests/PrngTestIsolation.cs, the counterpart of COLMAP's gtest_main, together
// with the isolation scope it opens in ColmapSharp/Mathematics/RandomUtils.cs. No COLMAP test
// corresponds to these; they guard the harness every ported test relies on.

using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class PrngTestIsolationTests
{
	// The first RandomUniformInteger(0, int.MaxValue) draw of a thread seeded with seed.
	private static int FirstDraw(uint seed)
	{
		return LibcxxRandom.UniformInt(new Mt19937(seed), 0, int.MaxValue);
	}

	// An execution context carrying a fresh isolation scope with the given seed, as the hook
	// leaves the test's context. Opened on a helper thread so this test's own scope is kept.
	private static ExecutionContext CaptureScope(uint seed)
	{
		ExecutionContext? captured = null;
		var thread = new Thread(() =>
		{
			RandomUtils.BeginIsolatedScope(seed);
			captured = ExecutionContext.Capture();
		});
		thread.Start();
		thread.Join();
		return captured!;
	}

	// Stands in for a pool thread that ran an earlier test: it seeds 123 and draws outside
	// any test's context, then runs work under the given context. Returns whether Prng read
	// null on entry, the value drawn there, and the value the stale stream would have given.
	private static (bool NullOnEntry, int Drawn, int StaleNext) DrawOnStaleThread(ExecutionContext context)
	{
		bool nullOnEntry = false;
		int drawn = 0;
		int staleNext = 0;
		Thread thread;
		using (ExecutionContext.SuppressFlow())
		{
			thread = new Thread(() =>
			{
				RandomUtils.SetPRNGSeed(123);
				RandomUtils.RandomUniformInteger(0, int.MaxValue);
				var stale = new Mt19937(123);
				LibcxxRandom.UniformInt(stale, 0, int.MaxValue);
				staleNext = LibcxxRandom.UniformInt(stale, 0, int.MaxValue);
				ExecutionContext.Run(context, _ =>
				{
					nullOnEntry = RandomUtils.Prng == null;
					drawn = RandomUtils.RandomUniformInteger(0, int.MaxValue);
				}, null);
			});
			thread.Start();
		}

		thread.Join();
		return (nullOnEntry, drawn, staleNext);
	}

	[Test]
	public async Task CSharpOnly_UnseededTestStartsFromSeedZero()
	{
		// A smoke check of the common case: an unseeded draw in a test body is seed 0's first
		// value. It cannot tell a working hook from a body that happens to run on a thread
		// that never drew (whose lazy default seed is also 0); the stale-thread tests below
		// are what prove the reset.
		int first = RandomUtils.RandomUniformInteger(0, int.MaxValue);

		await Assert.That(first).IsEqualTo(FirstDraw(PrngTestIsolation.DefaultTestPRNGSeed));
	}

	[Test]
	public async Task CSharpOnly_ThreadWithStaleEngineRestartsInsideTest()
	{
		// Under this test's context (opened by the hook) the stale engine must be dropped, so
		// the thread draws seed 0's first value and not the next value of the stale stream.
		(bool nullOnEntry, int drawn, int staleNext) = DrawOnStaleThread(ExecutionContext.Capture()!);

		await Assert.That(nullOnEntry).IsTrue();
		await Assert.That(drawn).IsEqualTo(FirstDraw(PrngTestIsolation.DefaultTestPRNGSeed));
		await Assert.That(drawn).IsNotEqualTo(staleNext);
	}

	[Test]
	public async Task CSharpOnly_ThreadEnteringScopeStartsFromScopeSeed()
	{
		// A reset thread seeds with its scope's seed, not DefaultPRNGSeed (0), while still
		// reading null like a fresh COLMAP thread until it draws.
		const uint ScopeSeed = 7;
		(bool nullOnEntry, int drawn, _) = DrawOnStaleThread(CaptureScope(ScopeSeed));

		await Assert.That(nullOnEntry).IsTrue();
		await Assert.That(drawn).IsEqualTo(FirstDraw(ScopeSeed));
		await Assert.That(drawn).IsNotEqualTo(FirstDraw(0));
	}
}
