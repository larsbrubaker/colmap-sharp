// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// RandomOracleTests (C#-only, not a COLMAP test): Mt19937, LibcxxRandom and RandomUtils
// against TestData/oracle/random.json, which oracle/fixture_random.py records from libc++'s
// <random> on macOS (the standard library the pycolmap wheel links). Tier A: every draw must
// be bit-identical. random_test.cc (ported 1:1 in RandomTests) only checks ranges and
// moments; this is what pins the actual numbers.
//
// The fixture's "cases" come from a -ffp-contract=off build, the arithmetic ColmapSharp
// does. Its "contracted_cases" (Apple clang's default fused multiply-adds) are not asserted:
// see docs/CPP_DIVERGENCES.md, entry 1.
//
// Each case draws all its numbers synchronously before any await, because the PRNG is per
// thread (see RandomTests).

using System.Globalization;
using System.Text.Json;

using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class RandomOracleTests
{
	private static readonly uint[] Seeds = [0u, 1u, 42u, 4294967295u];

	private enum Kind
	{
		Unsigned,
		Signed,
		Real,
	}

	// Harness case name (without "/seed<n>") -> value kind and the C# draws for one seed.
	// Each producer mirrors the matching block of oracle/random_harness.cc.
	private static readonly (string Name, Kind Kind, Func<uint, string[]> Draw)[] Cases =
	[
		("mt19937", Kind.Unsigned, seed => Draws(seed, 32, () => Format((ulong)RandomUtils.Prng!.Next()))),
		("int_0_10000", Kind.Signed, seed => Draws(seed, 64, () => Format(RandomUtils.RandomUniformInteger(0, 10000)))),
		("int_m100_100", Kind.Signed, seed => Draws(seed, 64, () => Format(RandomUtils.RandomUniformInteger(-100, 100)))),
		("int_full", Kind.Signed, seed => Draws(seed, 16, () => Format(RandomUtils.RandomUniformInteger(int.MinValue, int.MaxValue)))),
		("short_m7_300", Kind.Signed, seed => Draws(seed, 32, () => Format(RandomUtils.RandomUniformInteger<short>(-7, 300)))),
		("uint32_0_999", Kind.Unsigned, seed => Draws(seed, 64, () => Format(RandomUtils.RandomUniformInteger(0u, 999u)))),
		("int64_pm1e12", Kind.Signed, seed => Draws(seed, 32, () => Format(RandomUtils.RandomUniformInteger(-1000000000000L, 1000000000000L)))),
		("uint64_0_2pow40p3", Kind.Unsigned, seed => Draws(seed, 32, () => Format(RandomUtils.RandomUniformInteger(0ul, (1ul << 40) + 3)))),
		("uint64_full", Kind.Unsigned, seed => Draws(seed, 16, () => Format(RandomUtils.RandomUniformInteger(0ul, ulong.MaxValue)))),
		("size_t_0_5", Kind.Unsigned, seed => Draws(seed, 32, () => Format(RandomUtils.RandomUniformInteger<nuint>(0, 5)))),
		("double_m100_100", Kind.Real, seed => Draws(seed, 32, () => Format(RandomUtils.RandomUniformReal(-100.0, 100.0)))),
		("double_0_1", Kind.Real, seed => Draws(seed, 32, () => Format(RandomUtils.RandomUniformReal(0.0, 1.0)))),
		("float_m1_1", Kind.Real, seed => Draws(seed, 32, () => Format(RandomUtils.RandomUniformReal(-1f, 1f)))),
		("float_0_1000", Kind.Real, seed => Draws(seed, 32, () => Format(RandomUtils.RandomUniformReal(0f, 1000f)))),
		("gaussian_double_1_1", Kind.Real, seed => Draws(seed, 32, () => Format(RandomUtils.RandomGaussian(1.0, 1.0)))),
		("gaussian_float_0_2", Kind.Real, seed => Draws(seed, 32, () => Format(RandomUtils.RandomGaussian(0f, 2f)))),
		("normal_reused_0.5_3", Kind.Real, seed =>
		{
			// One distribution reused: exercises libc++'s cached second polar value.
			var distribution = new NormalDistribution<double>(0.5, 3.0);
			return Draws(seed, 32, () => Format(distribution.Next(RandomUtils.Prng!)));
		}),
		("colmap_shuffle_5_of_20", Kind.Signed, seed => ColmapShuffle(seed, 5)),
		("colmap_shuffle_20_of_20", Kind.Signed, seed => ColmapShuffle(seed, 20)),
		("std_shuffle_30", Kind.Signed, seed =>
		{
			RandomUtils.SetPRNGSeed(seed);
			var values = Enumerable.Range(0, 30).Select(i => (long)i).ToList();
			LibcxxRandom.Shuffle(values, RandomUtils.Prng!);
			return values.Select(Format).ToArray();
		}),
	];

	[Test]
	public async Task Mt19937_DefaultSeed10000thDraw()
	{
		var engine = new Mt19937();
		for (int i = 1; i < 10000; ++i)
		{
			engine.Next();
		}

		uint draw = engine.Next();
		JsonElement expected = OracleFixture.Load("random.json").GetProperty("cases").GetProperty("mt19937_default_10000th");

		using (Assert.Multiple())
		{
			// The C++ standard pins this value ([rand.predef]); the fixture confirms libc++ agrees.
			await Assert.That(draw).IsEqualTo(4123659995u);
			await Assert.That((ulong)draw).IsEqualTo(OracleFixture.UInt64s(expected)[0]);
		}
	}

	[Test]
	public async Task AllCases_MatchLibcxxBitForBit()
	{
		JsonElement cases = OracleFixture.Load("random.json").GetProperty("cases");
		var mismatches = new List<string>();
		int compared = 0;
		foreach (var (name, kind, draw) in Cases)
		{
			foreach (uint seed in Seeds)
			{
				string key = $"{name}/seed{seed}";
				string[] expected = Expected(cases.GetProperty(key), kind);
				string[] actual = draw(seed);
				compared++;
				string? mismatch = FirstMismatch(expected, actual);
				if (mismatch != null)
				{
					mismatches.Add($"{key}: {mismatch}");
				}
			}
		}

		int fixtureCases = cases.EnumerateObject().Count();

		using (Assert.Multiple())
		{
			await Assert.That(string.Join("\n", mismatches)).IsEqualTo(string.Empty);
			// Every fixture case except mt19937_default_10000th (its own test) is compared, so a
			// case added to the harness cannot go unchecked.
			await Assert.That(compared).IsEqualTo(fixtureCases - 1);
		}
	}

	private static string[] Draws(uint seed, int count, Func<string> draw)
	{
		RandomUtils.SetPRNGSeed(seed);
		var values = new string[count];
		for (int i = 0; i < count; ++i)
		{
			values[i] = draw();
		}

		return values;
	}

	private static string[] ColmapShuffle(uint seed, uint numToShuffle)
	{
		RandomUtils.SetPRNGSeed(seed);
		var values = Enumerable.Range(0, 20).Select(i => (long)i).ToList();
		RandomUtils.Shuffle(numToShuffle, values);
		return values.Select(Format).ToArray();
	}

	private static string[] Expected(JsonElement array, Kind kind)
	{
		return kind switch
		{
			Kind.Unsigned => OracleFixture.UInt64s(array).Select(Format).ToArray(),
			Kind.Signed => OracleFixture.Int64s(array).Select(Format).ToArray(),
			_ => OracleFixture.Doubles(array).Select(Format).ToArray(),
		};
	}

	private static string? FirstMismatch(string[] expected, string[] actual)
	{
		if (expected.Length != actual.Length)
		{
			return $"expected {expected.Length} values, got {actual.Length}";
		}

		for (int i = 0; i < expected.Length; ++i)
		{
			if (expected[i] != actual[i])
			{
				return $"first difference at [{i}]: expected {expected[i]}, got {actual[i]}";
			}
		}

		return null;
	}

	private static string Format(ulong value) => value.ToString(CultureInfo.InvariantCulture);

	private static string Format(long value) => value.ToString(CultureInfo.InvariantCulture);

	private static string Format(int value) => Format((long)value);

	private static string Format(short value) => Format((long)value);

	private static string Format(uint value) => Format((ulong)value);

	private static string Format(nuint value) => Format((ulong)value);

	// A float widens to double exactly, as the harness's printf("%a") does, so both sides
	// compare as doubles; the bit pattern makes -0.0 vs 0.0 and last-ulp differences visible.
	private static string Format(float value) => Format((double)value);

	private static string Format(double value) =>
		$"{value.ToString("R", CultureInfo.InvariantCulture)} (0x{BitConverter.DoubleToInt64Bits(value):X16})";
}
