// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// CppStreamFormat: how a C++ std::ostream writes a double in its default float field
// (neither std::fixed nor std::scientific), which is printf's %.<precision>g. COLMAP's
// operator<< for Rigid3d/Sim3d (precision 6, Eigen::StreamPrecision), Sim3d::ToFile and the
// text reconstruction writers and exporters (precision 17) go through it, and their output
// is part of COLMAP's tested contract (rigid3_test.cc / sim3_test.cc "Print",
// reconstruction_io_test.cc). Written here from the C standard's definition of %g
// (C11 7.21.6.1); not a port of any library. Tests: ColmapSharp.Tests/Util/CppStreamFormatTests.cs.
//
// Tier A (exact): the digits are rounded from the double's exact decimal expansion,
// round-half-even on exact ties, which is what glibc's and Apple's printf do in the
// default rounding mode. .NET's "E{n}" format produces exactly those n+1 digits: since
// .NET Core 3.0 it is IEEE-compliant (exact Dragon4 when Grisu cannot decide) and it breaks
// exact ties to even (1234565 "E5" -> 1.23456E+006, 1234575 -> 1.23458E+006). That is not
// promised by its documentation, so CppStreamFormatTests diffs this against the old
// exact-expansion formatter (kept in the tests as SlowCppStreamFormat), ties and subnormals
// included (5M values x 9 precisions offline when this was written; a slice in the suite).
// Only the %g layout is done here, on a stack buffer. Formatting 3M floats (a million PLY
// vertices) takes ~0.3 s at precision 6 and ~0.35-0.4 s at 17 in Release on an M-series
// Mac, about 15x faster than formatting "E800" and rounding by hand (~5 s); .NET's "E"
// formatting alone is ~0.2 s of that.

using System.Globalization;

namespace ColmapSharp.Util;

/// <summary>
/// Formats doubles the way a C++ <c>std::ostream</c> does in its default float field.
/// </summary>
public static class CppStreamFormat
{
	/// <summary>std::ostream's default precision.</summary>
	public const int DefaultPrecision = 6;

	// "E0".."E39", so the common precisions do not allocate a format string per call.
	private static readonly string[] ExponentFormats = BuildExponentFormats(40);

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

		bool negative = double.IsNegative(value);
		if (value == 0)
		{
			return negative ? "-0" : "0";
		}

		// %g treats a precision of 0 as 1. Beyond 767 significant digits a double's exact
		// expansion has only zeros left, which %g strips anyway.
		int significant = precision <= 0 ? 1 : Math.Min(precision, 800);
		int bufferLength = significant + 16;
		Span<char> scientific = bufferLength <= 256 ? stackalloc char[bufferLength] : new char[bufferLength];
		string format = significant <= ExponentFormats.Length
			? ExponentFormats[significant - 1]
			: "E" + (significant - 1).ToString(CultureInfo.InvariantCulture);
		bool formatted = Math.Abs(value).TryFormat(scientific, out int written, format, CultureInfo.InvariantCulture);
		Check.That(formatted);

		// "d.ddddE+ddd" (no '.' when there is one digit): gather the digits in place.
		Span<char> digits = scientific;
		int exponentIndex = scientific[..written].IndexOf('E');
		if (significant > 1)
		{
			scientific.Slice(2, significant - 1).CopyTo(digits[1..]);
		}

		int exponent = 0;
		for (int i = exponentIndex + 2; i < written; i++)
		{
			exponent = exponent * 10 + (scientific[i] - '0');
		}

		if (scientific[exponentIndex + 1] == '-')
		{
			exponent = -exponent;
		}

		int length = significant;
		while (length > 1 && digits[length - 1] == '0')
		{
			length--;
		}

		return Layout(negative, digits[..length], exponent, significant);
	}

	// %g's layout of the rounded digits (trailing zeros already removed) whose first digit
	// has decimal exponent `exponent`.
	private static string Layout(bool negative, ReadOnlySpan<char> digits, int exponent, int significant)
	{
		// The fixed layout pads the integer part with zeros up to the exponent, which can be
		// far beyond the stripped digits (-1e16 at precision 17 is 17 characters from one
		// digit). The exponent is below the precision there, so this is at most ~800.
		Span<char> output = stackalloc char[Math.Max(digits.Length, exponent + 1) + 16];
		int n = 0;
		if (negative)
		{
			output[n++] = '-';
		}

		if (exponent < -4 || exponent >= significant)
		{
			output[n++] = digits[0];
			if (digits.Length > 1)
			{
				output[n++] = '.';
				digits[1..].CopyTo(output[n..]);
				n += digits.Length - 1;
			}

			output[n++] = 'e';
			output[n++] = exponent < 0 ? '-' : '+';
			int magnitude = Math.Abs(exponent);
			if (magnitude >= 100)
			{
				output[n++] = (char)('0' + magnitude / 100);
			}

			output[n++] = (char)('0' + magnitude / 10 % 10);
			output[n++] = (char)('0' + magnitude % 10);
		}
		else if (exponent < 0)
		{
			output[n++] = '0';
			output[n++] = '.';
			for (int i = 0; i < -exponent - 1; i++)
			{
				output[n++] = '0';
			}

			digits.CopyTo(output[n..]);
			n += digits.Length;
		}
		else
		{
			// The integer part keeps its zeros; only fraction zeros were stripped.
			for (int i = 0; i <= exponent; i++)
			{
				output[n++] = i < digits.Length ? digits[i] : '0';
			}

			if (digits.Length > exponent + 1)
			{
				output[n++] = '.';
				digits[(exponent + 1)..].CopyTo(output[n..]);
				n += digits.Length - exponent - 1;
			}
		}

		return new string(output[..n]);
	}

	private static string[] BuildExponentFormats(int count)
	{
		var formats = new string[count];
		for (int i = 0; i < count; i++)
		{
			formats[i] = "E" + i.ToString(CultureInfo.InvariantCulture);
		}

		return formats;
	}
}
