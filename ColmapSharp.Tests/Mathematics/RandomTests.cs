// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RandomTests: colmap/math/random_test.cc ported 1:1, one method per gtest TEST(Suite, Name)
// named Suite_Name, same checks and tolerances. Tests ColmapSharp/Mathematics/RandomUtils.cs.
// The bit-exact comparison against libc++ lives in RandomOracleTests (C#-only).
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test (PRNGTestEventListener), so
// each test here starts with RandomUtils.SetPRNGSeed(0). The PRNG is per thread, and an async
// test may resume on another thread after an await, so every test draws all its numbers
// before its first await.

using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class RandomTests
{
	// colmap/util/gtest_main.cc: kDefaultTestPRNGSeed.
	private const uint DefaultTestPRNGSeed = 0;

	[Test]
	public async Task PRNGSeed_Nominal()
	{
		RandomUtils.SetPRNGSeed(DefaultTestPRNGSeed);
		RandomUtils.Prng = null;
		bool nullAfterReset = RandomUtils.Prng == null;
		RandomUtils.SetPRNGSeed();
		bool setAfterDefaultSeed = RandomUtils.Prng != null;
		RandomUtils.SetPRNGSeed(0);
		bool setAfterSeed0 = RandomUtils.Prng != null;

		// Each thread defines their own PRNG instance.
		bool threadNullAtStart = false;
		bool threadSetAfterDefaultSeed = false;
		bool threadSetAfterSeed0 = false;
		var thread = new Thread(() =>
		{
			threadNullAtStart = RandomUtils.Prng == null;
			RandomUtils.SetPRNGSeed();
			threadSetAfterDefaultSeed = RandomUtils.Prng != null;
			RandomUtils.SetPRNGSeed(0);
			threadSetAfterSeed0 = RandomUtils.Prng != null;
		});
		thread.Start();
		thread.Join();

		using (Assert.Multiple())
		{
			await Assert.That(nullAfterReset).IsTrue();
			await Assert.That(setAfterDefaultSeed).IsTrue();
			await Assert.That(setAfterSeed0).IsTrue();
			await Assert.That(threadNullAtStart).IsTrue();
			await Assert.That(threadSetAfterDefaultSeed).IsTrue();
			await Assert.That(threadSetAfterSeed0).IsTrue();
		}
	}

	[Test]
	public async Task Repeatability_Nominal()
	{
		const int NumPoints = 100;

		RandomUtils.SetPRNGSeed(0);
		var numbers1 = new List<int>(NumPoints);
		for (int i = 0; i < NumPoints; ++i)
		{
			numbers1.Add(RandomUtils.RandomUniformInteger(0, 10000));
		}

		RandomUtils.SetPRNGSeed(1);
		var numbers2 = new List<int>(NumPoints);
		for (int i = 0; i < NumPoints; ++i)
		{
			numbers2.Add(RandomUtils.RandomUniformInteger(0, 10000));
		}

		RandomUtils.SetPRNGSeed(0);
		var numbers3 = new List<int>(NumPoints);
		for (int i = 0; i < NumPoints; ++i)
		{
			numbers3.Add(RandomUtils.RandomUniformInteger(0, 10000));
		}

		bool allEqual = true;
		for (int i = 0; i < numbers1.Count; ++i)
		{
			if (numbers1[i] != numbers2[i])
			{
				allEqual = false;
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(numbers1.SequenceEqual(numbers3)).IsTrue();
			await Assert.That(allEqual).IsFalse();
		}
	}

	[Test]
	public async Task RandomUniformInteger_Nominal()
	{
		RandomUtils.SetPRNGSeed(DefaultTestPRNGSeed);
		var lower = new int[1000];
		var upper = new int[1000];
		for (int i = 0; i < 1000; ++i)
		{
			lower[i] = RandomUtils.RandomUniformInteger(-100, 100);
			upper[i] = RandomUtils.RandomUniformInteger(-100, 100);
		}

		using (Assert.Multiple())
		{
			for (int i = 0; i < 1000; ++i)
			{
				await Assert.That(lower[i]).IsGreaterThanOrEqualTo(-100);
				await Assert.That(upper[i]).IsLessThanOrEqualTo(100);
			}
		}
	}

	[Test]
	public async Task RandomUniformReal_Nominal()
	{
		RandomUtils.SetPRNGSeed(DefaultTestPRNGSeed);
		var lower = new double[1000];
		var upper = new double[1000];
		for (int i = 0; i < 1000; ++i)
		{
			lower[i] = RandomUtils.RandomUniformReal(-100.0, 100.0);
			upper[i] = RandomUtils.RandomUniformReal(-100.0, 100.0);
		}

		using (Assert.Multiple())
		{
			for (int i = 0; i < 1000; ++i)
			{
				await Assert.That(lower[i]).IsGreaterThanOrEqualTo(-100.0);
				await Assert.That(upper[i]).IsLessThanOrEqualTo(100.0);
			}
		}
	}

	[Test]
	public async Task RandomGaussian_Nominal()
	{
		RandomUtils.SetPRNGSeed(DefaultTestPRNGSeed);
		const double Mean = 1.0;
		const double Sigma = 1.0;
		const int NumValues = 100000;
		var values = new double[NumValues];
		for (int i = 0; i < NumValues; ++i)
		{
			values[i] = RandomUtils.RandomGaussian(Mean, Sigma);
		}

		double mean = MathUtils.Mean<double>(values);
		double stdDev = MathUtils.StdDev<double>(values);

		using (Assert.Multiple())
		{
			await Assert.That(Math.Abs(mean - Mean)).IsLessThanOrEqualTo(1e-2);
			await Assert.That(Math.Abs(stdDev - Sigma)).IsLessThanOrEqualTo(1e-2);
		}
	}

	[Test]
	public async Task ShuffleNone_Nominal()
	{
		RandomUtils.SetPRNGSeed(DefaultTestPRNGSeed);
		var numbers = new List<int>();
		RandomUtils.Shuffle(0, numbers);
		numbers = [1, 2, 3, 4, 5];
		var shuffledNumbers = new List<int>(numbers);
		RandomUtils.Shuffle(0, shuffledNumbers);

		await Assert.That(shuffledNumbers.SequenceEqual(numbers)).IsTrue();
	}

	[Test]
	public async Task ShuffleAll_Nominal()
	{
		RandomUtils.SetPRNGSeed(DefaultTestPRNGSeed);
		var numbers = Enumerable.Range(0, 1000).ToList();
		var shuffledNumbers = new List<int>(numbers);
		RandomUtils.Shuffle(1000, shuffledNumbers);
		int numShuffled = 0;
		for (int i = 0; i < numbers.Count; ++i)
		{
			if (numbers[i] != shuffledNumbers[i])
			{
				numShuffled += 1;
			}
		}

		await Assert.That(numShuffled).IsGreaterThan(0);
	}
}
