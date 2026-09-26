// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// CppStreamFormat: how a C++ std::ostream writes a double in its default float field
// (neither std::fixed nor std::scientific), which is printf's %.<precision>g. COLMAP's
// operator<< for Rigid3d/Sim3d (precision 6, Eigen::StreamPrecision) and Sim3d::ToFile
// (precision 17) go through it, and their output is part of COLMAP's tested contract
// (rigid3_test.cc / sim3_test.cc "Print"). Written here from the C standard's definition
// of %g (C11 7.21.6.1); not a port of any library. Users: Geometry/Rigid3d.cs,
// Geometry/Sim3d.cs. Tests: ColmapSharp.Tests/Util/CppStreamFormatTests.cs.
//
// Tier A (exact): the digits are rounded from the double's exact decimal expansion,
// round-half-even on exact ties, which is what glibc's and Apple's printf do in the
// default rounding mode. .NET's "E" format with a large precision returns that exact
// expansion (it is IEEE-compliant since .NET Core 3.0), so only the final rounding is
// done here.

using System.Globalization;
using System.Text;

namespace ColmapSharp.Util;

/// <summary>
/// Formats doubles the way a C++ <c>std::ostream</c> does in its default float field.
/// </summary>
public static class CppStreamFormat
{
	/// <summary>std::ostream's default precision.</summary>
	public const int DefaultPrecision = 6;

	// A double's exact decimal expansion has at most 767 significant digits.
	private const string ExactFormat = "E800";

	/// <summary>
	/// <c>stream &lt;&lt; value</c> with <c>stream.precision(precision)</c> and the default
	/// float field, i.e. printf's <c>%.{precision}g</c>: at most <paramref name="precision"/>
	/// significant digits, trailing zeros removed, scientific notation when the decimal
	/// exponent is below -4 or at least the precision.
	/// </summary>
	public static string FormatDouble(double value, int precision = DefaultPrecision)
	{
		if (double.IsNaN(value))
		{
			// libc++ on macOS (the pycolmap oracle's) prints "nan" for every NaN, whatever
			// its sign bit; .NET's double.NaN has the sign bit set, so it must not decide.
			return "nan";
		}

		if (double.IsInfinity(value))
		{
			return value < 0 ? "-inf" : "inf";
		}

		// %g treats a precision of 0 as 1.
		int significant = precision <= 0 ? 1 : precision;
		string sign = double.IsNegative(value) ? "-" : string.Empty;
		if (value == 0)
		{
			return sign + "0";
		}

		(string digits, int exponent) = RoundSignificant(Math.Abs(value), significant);
		if (exponent < -4 || exponent >= significant)
		{
			string mantissa = StripTrailingZeros(digits[..1] + "." + digits[1..]);
			string exponentSign = exponent < 0 ? "-" : "+";
			return string.Create(
				CultureInfo.InvariantCulture,
				$"{sign}{mantissa}e{exponentSign}{Math.Abs(exponent):00}");
		}

		var builder = new StringBuilder(sign);
		if (exponent < 0)
		{
			builder.Append("0.").Append('0', -exponent - 1).Append(digits);
		}
		else
		{
			builder.Append(digits, 0, exponent + 1).Append('.').Append(digits, exponent + 1, digits.Length - exponent - 1);
		}

		return StripTrailingZeros(builder.ToString());
	}

	// The first `significant` digits of |value| rounded half-even from its exact decimal
	// expansion, and the decimal exponent of the first digit after rounding.
	private static (string Digits, int Exponent) RoundSignificant(double magnitude, int significant)
	{
		string exact = magnitude.ToString(ExactFormat, CultureInfo.InvariantCulture);
		int exponentIndex = exact.IndexOf('E');
		int exponent = int.Parse(exact.AsSpan(exponentIndex + 1), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
		string all = exact[0] + exact[2..exponentIndex];

		char[] kept = all[..significant].ToCharArray();
		bool roundUp = false;
		if (all.Length > significant)
		{
			char next = all[significant];
			if (next > '5')
			{
				roundUp = true;
			}
			else if (next == '5')
			{
				bool anyNonZeroAfter = all.AsSpan(significant + 1).IndexOfAnyExcept('0') >= 0;
				roundUp = anyNonZeroAfter || (kept[^1] - '0') % 2 == 1;
			}
		}

		if (roundUp)
		{
			int i = kept.Length - 1;
			while (i >= 0 && kept[i] == '9')
			{
				kept[i] = '0';
				i--;
			}

			if (i < 0)
			{
				// 9.99... rounded up to 10.0...: one more leading digit, drop the last.
				return ("1" + new string(kept, 0, kept.Length - 1), exponent + 1);
			}

			kept[i]++;
		}

		return (new string(kept), exponent);
	}

	private static string StripTrailingZeros(string number)
	{
		if (!number.Contains('.'))
		{
			return number;
		}

		return number.TrimEnd('0').TrimEnd('.');
	}
}
