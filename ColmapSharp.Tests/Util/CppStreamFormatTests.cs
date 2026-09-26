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
}
