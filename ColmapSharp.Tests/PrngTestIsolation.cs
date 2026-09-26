// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PrngTestIsolation: the counterpart of colmap/util/gtest_main.cc's PRNGTestEventListener,
// which calls SetPRNGSeed(kDefaultTestPRNGSeed = 0) at the start of every test. Without it,
// a test that draws before seeding would continue whatever stream an earlier test left on
// its thread, and its result would depend on test order and parallelism.
//
// TUnit may run this hook and the test body on different threads, and RandomUtils' PRNG is
// [ThreadStatic], so seeding "the thread" here would not reliably reach the test. The hook
// therefore opens an isolation scope in the test's execution context
// (RandomUtils.BeginIsolatedScope) and asks TUnit to flow that context into the test body
// (AddAsyncLocalValues); every thread that then draws for the test starts from seed 0.
// Proven by Mathematics/PrngTestIsolationTests.cs.

using ColmapSharp.Mathematics;

using TUnit.Core;

namespace ColmapSharp.Tests;

public static class PrngTestIsolation
{
	// colmap/util/gtest_main.cc: kDefaultTestPRNGSeed.
	public const uint DefaultTestPRNGSeed = 0;

	[BeforeEvery(Test)]
	public static void SeedPrngBeforeEveryTest(TestContext context)
	{
		RandomUtils.BeginIsolatedScope(DefaultTestPRNGSeed);
		context.AddAsyncLocalValues();
	}
}
