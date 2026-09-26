// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// FormatTestValues (C#-only, test input): seeded doubles that stress %g's rounding, for the
// differential between ColmapSharp/Util/CppStreamFormat.cs and its slow oracle
// (SlowCppStreamFormat.cs). Used by CppStreamFormatTests.CSharpOnly_FastMatchesSlowOracle.

using System.Globalization;
using System.Numerics;

namespace ColmapSharp.Tests.Util;

/// <summary>
/// Seeded test doubles for the CppStreamFormat differential, so a failure reproduces.
/// </summary>
public static class FormatTestValues
{
	/// <summary>
	/// Yields <paramref name="count"/> finite doubles, cycling through six kinds: random bit
	/// patterns (every exponent), subnormals, decimal-looking values (k / 10^j, the shape of
	/// real coordinates), odd multiples of 5 (exact ties at precisions up to 12), dyadic
	/// fractions k / 2^j whose exact expansions are short, so they are exact ties at some
	/// precision (the 18-digit ones are ties at precision 17), and short-digit powers of ten
	/// (<see cref="DecimalPower"/>).
	/// </summary>
	public static IEnumerable<double> Generate(int count, int seed)
	{
		var random = new Random(seed);
		for (int i = 0; i < count; i++)
		{
			double value;
			switch (i % 6)
			{
				case 5:
					value = DecimalPower(random);
					break;
				case 0:
					do
					{
						value = BitConverter.Int64BitsToDouble(random.NextInt64(long.MinValue, long.MaxValue));
					}
					while (!double.IsFinite(value));
					break;
				case 1:
					value = BitConverter.Int64BitsToDouble(random.NextInt64(1, 1L << 52));
					break;
				case 2:
					value = random.NextInt64(-10_000_000_000, 10_000_000_000) / Math.Pow(10, random.Next(0, 12));
					break;
				case 3:
					value = (random.NextInt64(0, 100_000_000_000) * 10 + 5) * (random.Next(2) == 0 ? 1.0 : -1.0);
					break;
				default:
					value = Dyadic(random);
					break;
			}

			yield return value;
		}
	}

	/// <summary>
	/// Yields <paramref name="count"/> values of the <see cref="DecimalPower"/> kind only.
	/// </summary>
	public static IEnumerable<double> GenerateDecimalPowers(int count, int seed)
	{
		var random = new Random(seed);
		for (int i = 0; i < count; i++)
		{
			yield return DecimalPower(random);
		}
	}

	/// <summary>
	/// ±d × 10^k (the double nearest it) with 1 to 3 digits d and k in [-308, 308]: after
	/// rounding, most of the digits are zeros, which %g strips, while the decimal exponent
	/// can be far larger than the digit count (e.g. -1e16 at precision 17 prints 17 digits).
	/// </summary>
	public static double DecimalPower(Random random)
	{
		int digits = random.Next(1, 1000);
		int exponent = random.Next(-308, 309);
		double value = double.Parse(
			digits.ToString(CultureInfo.InvariantCulture) + "e" + exponent.ToString(CultureInfo.InvariantCulture),
			CultureInfo.InvariantCulture);
		if (double.IsInfinity(value))
		{
			value = double.MaxValue;
		}

		return random.Next(2) == 0 ? value : -value;
	}

	// An odd k / 2^j (exact) whose decimal expansion has about 2..20 significant digits and
	// always ends in 5, so it is an exact tie one precision below its length.
	private static double Dyadic(Random random)
	{
		int targetDigits = random.Next(2, 21);
		int j = random.Next(1, 30);
		BigInteger five = BigInteger.Pow(5, j);
		// k * 5^j must have targetDigits digits: k in [10^(t-1) / 5^j, 10^t / 5^j).
		BigInteger low = (BigInteger.Pow(10, targetDigits - 1) / five) + 1;
		BigInteger high = BigInteger.Pow(10, targetDigits) / five;
		if (high <= low || high >= (BigInteger.One << 53))
		{
			return (random.NextInt64(1, 1L << 40) | 1) / Math.Pow(2, j);
		}

		long k = (long)low + random.NextInt64(0, (long)(high - low));
		return (k | 1) / Math.Pow(2, j);
	}
}
