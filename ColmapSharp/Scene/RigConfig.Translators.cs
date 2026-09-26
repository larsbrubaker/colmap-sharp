// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// RigConfig's value translators: which property tree texts read as a bool or a double, for
// the JSON reader in RigConfig.cs. Written here, not ported: COLMAP reads values through
// boost::property_tree's stream_translator, i.e. `istringstream >> value` on the node's text,
// and what that accepts is libc++'s num_get (the macOS pycolmap wheel links libc++). The
// rules below were pinned against a libc++ harness (Apple clang); the cases are in
// ColmapSharp.Tests/Scene/RigConfigTests.CSharpOnly.cs.
//
// The translator skips leading white space, reads the value, skips trailing white space, and
// fails unless the whole text was used.
// - bool: num_get reads a decimal long (optional sign, then digits). 0 is false, 1 is true,
//   and anything else (no digits, out of range, another value) fails. On that failure ptree
//   retries from where num_get stopped with boolalpha, which matches the words "true" or
//   "false" exactly. So "01", "+1" and even "2true" read true, while "TRUE" and "-1" fail.
// - double: strtod on the characters num_get collected, failing on ERANGE. That gives decimal
//   and hex floats ("0x1p3"), and rejects inf, nan, overflow, and results that are below
//   DBL_MIN after rounding and inexact (IEEE underflow). An exact subnormal (e.g.
//   "0x1p-1074") reads fine.

using System.Globalization;
using System.Numerics;

namespace ColmapSharp.Scene;

public sealed partial class RigConfig
{
	// The C locale's isspace, which istream's skipws and std::ws use.
	private static bool IsCSpace(char c) => c is ' ' or '\t' or '\n' or '\v' or '\f' or '\r';

	private static int SkipCSpace(string text, int pos)
	{
		while (pos < text.Length && IsCSpace(text[pos]))
		{
			pos++;
		}

		return pos;
	}

	private static bool IsDecDigit(char c) => c is >= '0' and <= '9';

	private static bool IsHexDigit(char c) => IsDecDigit(c) || c is >= 'a' and <= 'f' or >= 'A' and <= 'F';

	private static ArgumentException ConversionFailed(string type, string text) =>
		new($"conversion of data to type \"{type}\" failed ({text})");

	// ptree's bool translator over libc++'s num_get (see the file header).
	private static bool ParseBool(string text)
	{
		int pos = SkipCSpace(text, 0);

		// num_get<bool> without boolalpha: a decimal long, which must be 0 or 1.
		int start = pos;
		if (pos < text.Length && text[pos] is '+' or '-')
		{
			pos++;
		}

		int digitsStart = pos;
		while (pos < text.Length && IsDecDigit(text[pos]))
		{
			pos++;
		}

		bool? value = null;
		if (pos > digitsStart
			&& long.TryParse(text.AsSpan(start, pos - start), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long number)
			&& number is 0 or 1)
		{
			value = number == 1;
		}

		if (value is null)
		{
			// Retry in word form from where the number read stopped.
			pos = SkipCSpace(text, pos);
			if (string.CompareOrdinal(text, pos, "true", 0, 4) == 0 && text.Length - pos >= 4)
			{
				value = true;
				pos += 4;
			}
			else if (string.CompareOrdinal(text, pos, "false", 0, 5) == 0 && text.Length - pos >= 5)
			{
				value = false;
				pos += 5;
			}
			else
			{
				throw ConversionFailed("bool", text);
			}
		}

		return SkipCSpace(text, pos) == text.Length ? value.Value : throw ConversionFailed("bool", text);
	}

	// ptree's double translator over libc++'s num_get and strtod (see the file header).
	private static double ParseDouble(string text)
	{
		int begin = SkipCSpace(text, 0);
		int end = text.Length;
		while (end > begin && IsCSpace(text[end - 1]))
		{
			end--;
		}

		ReadOnlySpan<char> token = text.AsSpan(begin, end - begin);
		double? value = IsHexPrefix(token) ? ParseHexFloat(token) : ParseDecimalFloat(token);
		return value ?? throw ConversionFailed("double", text);
	}

	private static bool IsHexPrefix(ReadOnlySpan<char> token)
	{
		int i = token.Length > 0 && token[0] is '+' or '-' ? 1 : 0;
		return token.Length > i + 1 && token[i] == '0' && token[i + 1] is 'x' or 'X';
	}

	// Scans [sign] mantissa [exponent] where the mantissa is digits with at most one '.' and at
	// least one digit, and the exponent is the marker, an optional sign and decimal digits.
	// Returns false unless the token matches in full.
	private static bool ScanFloat(
		ReadOnlySpan<char> token, int pos, Func<char, bool> isDigit, char exponentMarker,
		out bool negative, out string mantissaDigits, out int fractionDigits, out BigInteger exponent)
	{
		negative = token[0] == '-';
		mantissaDigits = "";
		fractionDigits = 0;
		exponent = BigInteger.Zero;

		var digits = new System.Text.StringBuilder();
		bool seenPoint = false;
		for (; pos < token.Length; pos++)
		{
			char c = token[pos];
			if (isDigit(c))
			{
				digits.Append(c);
				fractionDigits += seenPoint ? 1 : 0;
			}
			else if (c == '.' && !seenPoint)
			{
				seenPoint = true;
			}
			else
			{
				break;
			}
		}

		if (digits.Length == 0)
		{
			return false;
		}

		mantissaDigits = digits.ToString();
		if (pos == token.Length)
		{
			return true;
		}

		if (char.ToLowerInvariant(token[pos]) != exponentMarker)
		{
			return false;
		}

		pos++;
		int exponentStart = pos;
		if (pos < token.Length && token[pos] is '+' or '-')
		{
			pos++;
		}

		int exponentDigits = pos;
		while (pos < token.Length && IsDecDigit(token[pos]))
		{
			pos++;
		}

		if (pos == exponentDigits || pos != token.Length)
		{
			return false;
		}

		exponent = BigInteger.Parse(token[exponentStart..], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
		return true;
	}

	private static double? ParseDecimalFloat(ReadOnlySpan<char> token)
	{
		if (token.Length == 0)
		{
			return null;
		}

		int pos = token[0] is '+' or '-' ? 1 : 0;
		if (!ScanFloat(token, pos, IsDecDigit, 'e', out _, out string mantissaDigits, out int fractionDigits, out BigInteger exponent))
		{
			return null;
		}

		// .NET's parse is correctly rounded (round half to even) like strtod.
		double value = double.Parse(token, NumberStyles.Float, CultureInfo.InvariantCulture);
		if (double.IsInfinity(value))
		{
			return null;
		}

		double magnitude = Math.Abs(value);
		if (magnitude < 2.2250738585072014e-308)
		{
			// Below DBL_MIN: strtod reports ERANGE unless the result is exact.
			var decimalMantissa = BigInteger.Parse(mantissaDigits, CultureInfo.InvariantCulture);
			BigInteger decimalExponent = exponent - fractionDigits;
			if (!IsExactSubnormal(decimalMantissa, decimalExponent, magnitude))
			{
				return null;
			}
		}

		return value;
	}

	// Whether mantissa * 10^exponent equals the subnormal (or zero) magnitude exactly.
	private static bool IsExactSubnormal(BigInteger mantissa, BigInteger exponent, double magnitude)
	{
		if (mantissa.IsZero || magnitude == 0.0)
		{
			return mantissa.IsZero;
		}

		// magnitude = units * 2^-1074 with units < 2^52. An exact match needs a negative
		// decimal exponent (the magnitude is below 1): mantissa * 2^1074 == units * 10^-exponent.
		if (exponent.Sign >= 0 || -exponent > 2000)
		{
			return false;
		}

		var units = new BigInteger(BitConverter.DoubleToInt64Bits(magnitude));
		return mantissa * BigInteger.Pow(2, 1074) == units * BigInteger.Pow(10, (int)-exponent);
	}

	private static double? ParseHexFloat(ReadOnlySpan<char> token)
	{
		int pos = (token[0] is '+' or '-' ? 1 : 0) + 2;
		if (!ScanFloat(token, pos, IsHexDigit, 'p', out bool negative, out string mantissaDigits, out int fractionDigits, out BigInteger exponent))
		{
			return null;
		}

		BigInteger mantissa = BigInteger.Parse("0" + mantissaDigits, NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture);
		if (mantissa.IsZero)
		{
			return negative ? -0.0 : 0.0;
		}

		// value = mantissa * 2^binaryExponent; round it to the double grid (53 bits, or the
		// subnormal grid 2^-1074 below DBL_MIN) with round half to even.
		BigInteger binaryExponent = exponent - 4 * fractionDigits;
		BigInteger topExponent = binaryExponent + (long)mantissa.GetBitLength() - 1;
		if (topExponent > 1023)
		{
			return null;
		}

		if (topExponent < -1100)
		{
			// Far below the smallest subnormal: underflows to zero, inexactly.
			return null;
		}

		int lsbExponent = (int)BigInteger.Max(topExponent - 52, -1074);
		int shift = (int)(lsbExponent - binaryExponent);
		BigInteger units;
		bool inexact = false;
		if (shift <= 0)
		{
			units = mantissa << -shift;
		}
		else
		{
			units = mantissa >> shift;
			BigInteger remainder = mantissa - (units << shift);
			BigInteger half = BigInteger.One << (shift - 1);
			inexact = !remainder.IsZero;
			if (remainder > half || (remainder == half && !units.IsEven))
			{
				units += 1;
			}
		}

		double magnitude = Math.ScaleB((double)units, lsbExponent);
		if (double.IsInfinity(magnitude) || (magnitude < 2.2250738585072014e-308 && inexact))
		{
			return null;
		}

		return negative ? -magnitude : magnitude;
	}
}
