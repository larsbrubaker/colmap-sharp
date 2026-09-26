// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// CppStreamFormatTests (C#-only; COLMAP has no test of the C++ standard library):
// ColmapSharp/Util/CppStreamFormat.cs against printf's %g and %.17g. The expected strings
// were produced with Python's '%g' % v and '%.17g' % v, which format like C's printf
// (correctly rounded from the exact binary value, half-even on exact ties). The pycolmap
// comparison of whole Rigid3d/Sim3d strings is GeometryOracleTests.Strings. Tier A.

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

public class CppStreamFormatTests
{
	[Test]
	[Arguments(0.0, "0", "0")]
	[Arguments(-0.0, "-0", "-0")]
	[Arguments(1.0, "1", "1")]
	[Arguments(0.5, "0.5", "0.5")]
	[Arguments(1234565.0, "1.23456e+06", "1234565")] // exact tie, rounds to even
	[Arguments(1234575.0, "1.23458e+06", "1234575")] // exact tie, rounds to even
	[Arguments(123456.0, "123456", "123456")]
	[Arguments(1234567.0, "1.23457e+06", "1234567")]
	[Arguments(0.0001, "0.0001", "0.0001")]
	[Arguments(0.00001, "1e-05", "1.0000000000000001e-05")]
	[Arguments(9.9999996, "10", "9.9999996000000007")] // carry into a new leading digit
	[Arguments(99999.95, "99999.9", "99999.949999999997")] // below the tie in binary
	[Arguments(1e-20, "1e-20", "9.9999999999999995e-21")]
	[Arguments(1.0 / 3, "0.333333", "0.33333333333333331")]
	[Arguments(-2.5e300, "-2.5e+300", "-2.5000000000000001e+300")]
	[Arguments(5e-324, "4.94066e-324", "4.9406564584124654e-324")]
	[Arguments(0.1, "0.1", "0.10000000000000001")]
	public async Task FormatDouble_MatchesPrintf(double value, string g6, string g17)
	{
		using (Assert.Multiple())
		{
			await Assert.That(CppStreamFormat.FormatDouble(value)).IsEqualTo(g6);
			await Assert.That(CppStreamFormat.FormatDouble(value, 17)).IsEqualTo(g17);
		}
	}

	[Test]
	public async Task FormatDouble_NonFinite()
	{
		using (Assert.Multiple())
		{
			await Assert.That(CppStreamFormat.FormatDouble(double.PositiveInfinity)).IsEqualTo("inf");
			await Assert.That(CppStreamFormat.FormatDouble(double.NegativeInfinity)).IsEqualTo("-inf");
			// macOS libc++ prints "nan" whatever the sign bit; .NET's double.NaN has it set.
			await Assert.That(CppStreamFormat.FormatDouble(double.NaN)).IsEqualTo("nan");
			await Assert.That(CppStreamFormat.FormatDouble(BitConverter.Int64BitsToDouble(0x7FF8000000000000))).IsEqualTo("nan");
			await Assert.That(CppStreamFormat.FormatDouble(double.NaN, 17)).IsEqualTo("nan");
		}
	}

	// C#-only: the fast formatter relies on .NET's "E{n}" being correctly rounded with
	// half-even ties, which .NET does not document; this pins it against the exact-expansion
	// oracle. 20k values x 9 precisions keeps the suite fast; the same generator was run
	// with 5M values when the fast path was written.
	[Test]
	public async Task CSharpOnly_FastMatchesSlowOracle()
	{
		string mismatches = DiffAgainstOracle(FormatTestValues.Generate(20_000, seed: 20260926), [0, 1, 2, 6, 12, 15, 16, 17, 20]);
		await Assert.That(mismatches).IsEqualTo(string.Empty);
	}

	// C#-only: short-digit powers of ten at every precision from 1 to 30, where the fixed
	// layout pads far more integer zeros than there are significant digits.
	[Test]
	public async Task CSharpOnly_DecimalPowersMatchSlowOracle()
	{
		int[] precisions = [.. Enumerable.Range(1, 30)];
		string mismatches = DiffAgainstOracle(FormatTestValues.GenerateDecimalPowers(3_000, seed: 20260927), precisions);
		await Assert.That(mismatches).IsEqualTo(string.Empty);
	}

	// C#-only regression: the fixed layout of a value whose digits were stripped to one but
	// whose exponent is large overflowed the output buffer (IndexOutOfRangeException).
	[Test]
	[Arguments(-1e16, 17, "-10000000000000000")]
	[Arguments(1e20, 21, "100000000000000000000")]
	[Arguments(1e22, 30, "10000000000000000000000")]
	public async Task CSharpOnly_LargeExponentFixedLayout(double value, int precision, string expected)
	{
		await Assert.That(CppStreamFormat.FormatDouble(value, precision)).IsEqualTo(expected);
	}

	// The first ten fast/slow mismatches, one per line; empty when they all agree.
	private static string DiffAgainstOracle(IEnumerable<double> values, int[] precisions)
	{
		var mismatches = new List<string>();
		foreach (double value in values)
		{
			foreach (int precision in precisions)
			{
				string fast;
				try
				{
					fast = CppStreamFormat.FormatDouble(value, precision);
				}
				catch (Exception ex)
				{
					fast = ex.GetType().Name;
				}

				string slow = SlowCppStreamFormat.FormatDouble(value, precision);
				if (fast != slow && mismatches.Count < 10)
				{
					mismatches.Add($"{BitConverter.DoubleToInt64Bits(value):X16} p={precision}: fast={fast} slow={slow}");
				}
			}
		}

		return string.Join("\n", mismatches);
	}

	// C#-only: precisions past a double's longest exact expansion (767 significant digits)
	// print the whole expansion. The oracle only accepts up to 801, so the subnormal's
	// 751-digit expansion at precision 1000 is compared against the oracle at 800.
	[Test]
	[Arguments(123.0, 800, "123")]
	[Arguments(0.1, 60, "0.1000000000000000055511151231257827021181583404541015625")]
	public async Task CSharpOnly_LargePrecision(double value, int precision, string expected)
	{
		await Assert.That(CppStreamFormat.FormatDouble(value, precision)).IsEqualTo(expected);
	}

	[Test]
	public async Task CSharpOnly_LargePrecisionSubnormalMatchesSlowOracle()
	{
		string fast = CppStreamFormat.FormatDouble(5e-324, 1000);
		string slow = SlowCppStreamFormat.FormatDouble(5e-324, 800);
		using (Assert.Multiple())
		{
			await Assert.That(fast).IsEqualTo(slow);
			await Assert.That(fast.Length).IsEqualTo(751 + "e-324".Length + 1);
		}
	}
}
