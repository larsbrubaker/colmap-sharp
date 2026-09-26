// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// CppLineTokens: the subset of C++ std::istream text parsing (std::getline, StringTrim and
// `line_stream >> value` in the classic "C" locale) that COLMAP's text readers use, so that
// ReconstructionIOText.cs reads its files token by token the way reconstruction_io_text.cc
// does. Written here, not a port of libc++: a value is one whitespace-separated token, parsed
// with the invariant culture (so, like COLMAP's classic-locale streams, the decimal separator
// is '.' whatever the process culture), and a failed extraction reports false, which is
// where C++ sets failbit. Util/Ply*.cs reads ASCII PLY bodies through it too.
//
// The spellings follow libc++'s num_get as probed with a libc++ harness (see
// ColmapSharp.Tests/Scene/ReconstructionIORobustnessTests.cs):
// - unsigned: an optional '-' negates with wrap-around ("-1" is the maximum), a magnitude
//   above the type's maximum fails;
// - double: decimal and hexadecimal ("0x1p3" = 8) floats; "inf"/"nan" fail; a result that
//   overflows, or underflows inexactly to a subnormal or zero, fails (strtod's ERANGE).
// - float: the same rules in single precision (libc++ reads a float with strtof, rounding
//   the token once to float, not through double).
// A token is taken whole, so "12abc" fails where libc++ would read 12 into an integer and
// leave "abc" for the next extraction (docs/CPP_DIVERGENCES.md, entry 25). Correctly rounded
// parsing on both sides gives the same bits for the same token.

using System.Globalization;
using System.Numerics;
using System.Text;

namespace ColmapSharp.Util;

/// <summary>Whitespace-separated tokens of one line, read the way <c>istream &gt;&gt;</c> reads them.</summary>
internal sealed class CppLineTokens
{
	/// <summary>The characters C's isspace accepts in the "C" locale: what separates tokens.</summary>
	public static readonly char[] CppWhitespace = [' ', '\t', '\n', '\v', '\f', '\r'];

	// COLMAP's StringTrim strips only these (util/string.cc IsNotWhiteSpace), not '\v'/'\f'.
	private static readonly char[] StringTrimWhitespace = [' ', '\n', '\r', '\t'];

	// std::numeric_limits<double>::min(), the smallest normal double.
	private const double MinNormal = 2.2250738585072014e-308;

	// std::numeric_limits<float>::min(), the smallest normal float.
	private const float MinNormalFloat = 1.17549435e-38f;

	private static readonly UTF8Encoding StrictUtf8 = new(false, true);
	private static readonly UTF8Encoding LenientUtf8 = new(false, false);

	private readonly string[] _tokens;
	private int _next;

	/// <summary>Splits <paramref name="line"/> at C isspace characters.</summary>
	public CppLineTokens(string line)
	{
		_tokens = line.Split(CppWhitespace, StringSplitOptions.RemoveEmptyEntries);
	}

	/// <summary>
	/// std::getline over a whole text stream: lines split at '\n' only (a '\r' stays in the
	/// line, for StringTrim to remove), and no empty final line after a trailing '\n'. Each
	/// line is decoded as UTF-8; a line that is not valid UTF-8 comes back decoded with
	/// replacement characters and <c>ValidUtf8 = false</c>, so the caller can say which
	/// record is broken.
	/// </summary>
	public static List<(string Text, bool ValidUtf8)> ReadLines(Stream stream)
	{
		var buffer = new MemoryStream();
		stream.CopyTo(buffer);
		ReadOnlySpan<byte> bytes = buffer.GetBuffer().AsSpan(0, (int)buffer.Length);

		var lines = new List<(string, bool)>();
		while (bytes.Length > 0)
		{
			int end = bytes.IndexOf((byte)'\n');
			ReadOnlySpan<byte> line = end < 0 ? bytes : bytes[..end];
			lines.Add(Decode(line));
			bytes = end < 0 ? [] : bytes[(end + 1)..];
		}

		return lines;
	}

	/// <summary>StringTrim: strips leading and trailing ' ', '\n', '\r' and '\t'.</summary>
	public static string Trim(string line) => line.Trim(StringTrimWhitespace);

	/// <summary><c>stream &gt;&gt; std::string</c>: the next token.</summary>
	public bool TryReadString(out string value)
	{
		if (_next < _tokens.Length)
		{
			value = _tokens[_next++];
			return true;
		}

		value = "";
		return false;
	}

	/// <summary><c>stream &gt;&gt; double</c>.</summary>
	public bool TryReadDouble(out double value)
	{
		value = 0;
		if (_next >= _tokens.Length || !ParseDouble(_tokens[_next], out value))
		{
			return false;
		}

		_next++;
		return true;
	}

	/// <summary><c>stream &gt;&gt; float</c>.</summary>
	public bool TryReadFloat(out float value)
	{
		value = 0;
		if (_next >= _tokens.Length || !ParseFloat(_tokens[_next], out value))
		{
			return false;
		}

		_next++;
		return true;
	}

	/// <summary><c>stream &gt;&gt; uint32_t</c>.</summary>
	public bool TryReadUInt32(out uint value)
	{
		value = 0;
		if (_next >= _tokens.Length || !ParseUnsigned(_tokens[_next], uint.MaxValue, out ulong parsed))
		{
			return false;
		}

		value = (uint)parsed;
		_next++;
		return true;
	}

	/// <summary><c>stream &gt;&gt; uint64_t</c>.</summary>
	public bool TryReadUInt64(out ulong value)
	{
		value = 0;
		if (_next >= _tokens.Length || !ParseUnsigned(_tokens[_next], ulong.MaxValue, out value))
		{
			return false;
		}

		_next++;
		return true;
	}

	/// <summary><c>stream &gt;&gt; int</c>: out of range fails.</summary>
	public bool TryReadInt32(out int value)
	{
		value = 0;
		if (_next >= _tokens.Length ||
			!int.TryParse(_tokens[_next], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out value))
		{
			return false;
		}

		_next++;
		return true;
	}

	/// <summary><c>stream &gt;&gt; int64_t</c>: out of range fails.</summary>
	public bool TryReadInt64(out long value)
	{
		value = 0;
		if (_next >= _tokens.Length ||
			!long.TryParse(_tokens[_next], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out value))
		{
			return false;
		}

		_next++;
		return true;
	}

	private static (string, bool) Decode(ReadOnlySpan<byte> line)
	{
		try
		{
			return (StrictUtf8.GetString(line), true);
		}
		catch (DecoderFallbackException)
		{
			return (LenientUtf8.GetString(line), false);
		}
	}

	// libc++'s __num_get_unsigned_integral: strtoull on the magnitude, fail above max, then
	// negate in the unsigned type when the token starts with '-'.
	private static bool ParseUnsigned(string token, ulong max, out ulong value)
	{
		value = 0;
		bool negate = token.StartsWith('-');
		string magnitude = negate ? token[1..] : token.StartsWith('+') ? token[1..] : token;
		if (magnitude.Length == 0 || !magnitude.All(char.IsAsciiDigit) ||
			!ulong.TryParse(magnitude, NumberStyles.None, CultureInfo.InvariantCulture, out ulong parsed) || parsed > max)
		{
			return false;
		}

		value = negate ? unchecked(0 - parsed) & max : parsed;
		return true;
	}

	/// <summary>
	/// <c>stream &gt;&gt; double</c> on one whitespace-free token that must be consumed whole
	/// (COLMAP's StringToDouble on a column already split out of a line).
	/// </summary>
	public static bool TryParseDoubleToken(string token, out double value) => ParseDouble(token, out value);

	private static bool ParseDouble(string token, out double value)
	{
		value = 0;
		bool negative = token.StartsWith('-');
		string body = negative || token.StartsWith('+') ? token[1..] : token;
		if (body.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
		{
			if (!ParseHexMagnitude(body[2..], 53, -1074, out value, out bool inexact) ||
				double.IsInfinity(value) || (inexact && Math.Abs(value) < MinNormal))
			{
				return false;
			}

			value = negative ? -value : value;
			return true;
		}

		// Digits, '.', and an exponent only: .NET would also take "Infinity" and "NaN".
		if (body.Length == 0 || !body.All(c => char.IsAsciiDigit(c) || c is '.' or 'e' or 'E' or '+' or '-') ||
			!double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
		{
			return false;
		}

		// strtod reports ERANGE on overflow and on an (inexact) underflow below the normal
		// range; libc++ turns that into failbit. A decimal subnormal is practically never exact.
		bool nonZeroDigits = body.TakeWhile(c => c is not ('e' or 'E')).Any(c => c is >= '1' and <= '9');
		return !double.IsInfinity(value) && !(nonZeroDigits && Math.Abs(value) < MinNormal);
	}

	// strtof: ParseDouble's rules with float's range and one rounding to float.
	private static bool ParseFloat(string token, out float value)
	{
		value = 0;
		bool negative = token.StartsWith('-');
		string body = negative || token.StartsWith('+') ? token[1..] : token;
		if (body.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
		{
			if (!ParseHexMagnitude(body[2..], 24, -149, out double magnitude, out bool inexact))
			{
				return false;
			}

			// The magnitude already has at most 24 significant bits, so this cast is exact
			// unless it overflows to infinity.
			value = (float)(negative ? -magnitude : magnitude);
			return !float.IsInfinity(value) && !(inexact && Math.Abs(value) < MinNormalFloat);
		}

		if (body.Length == 0 || !body.All(c => char.IsAsciiDigit(c) || c is '.' or 'e' or 'E' or '+' or '-') ||
			!float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
		{
			return false;
		}

		bool nonZeroDigits = body.TakeWhile(c => c is not ('e' or 'E')).Any(c => c is >= '1' and <= '9');
		return !float.IsInfinity(value) && !(nonZeroDigits && Math.Abs(value) < MinNormalFloat);
	}

	// The magnitude after "0x": hex digits with an optional '.', then an optional binary
	// exponent p[+-]digits, rounded to nearest-even to `significandBits` bits with the lowest
	// subnormal bit at 2^minExponent (53/-1074 for strtod, 24/-149 for strtof). `inexact`
	// reports rounding, for the caller's underflow (ERANGE) check.
	private static bool ParseHexMagnitude(string text, int significandBits, int minExponent, out double value, out bool inexact)
	{
		inexact = false;
		value = 0;
		int pIndex = text.IndexOfAny(['p', 'P']);
		string mantissaText = pIndex < 0 ? text : text[..pIndex];
		long exponent = 0;
		if (pIndex >= 0 && !long.TryParse(text[(pIndex + 1)..], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out exponent))
		{
			return false;
		}

		BigInteger mantissa = BigInteger.Zero;
		int digits = 0;
		bool seenPoint = false;
		foreach (char c in mantissaText)
		{
			if (c == '.' && !seenPoint)
			{
				seenPoint = true;
				continue;
			}

			if (!char.IsAsciiHexDigit(c))
			{
				return false;
			}

			mantissa = (mantissa << 4) + Convert.ToInt32(c.ToString(), 16);
			digits++;
			if (seenPoint)
			{
				exponent -= 4;
			}
		}

		if (digits == 0)
		{
			return false;
		}

		if (mantissa.IsZero)
		{
			return true;
		}

		// value = mantissa * 2^exponent. Keep `significandBits` significant bits, or fewer
		// where the result is subnormal (its lowest bit is 2^minExponent).
		long bitLength = (long)mantissa.GetBitLength();
		long shift = Math.Max(bitLength - significandBits, minExponent - exponent);
		if (shift > bitLength)
		{
			// Below half the smallest subnormal: rounds to zero, an inexact underflow.
			return false;
		}

		inexact = false;
		if (shift > 0)
		{
			BigInteger remainder = mantissa & ((BigInteger.One << (int)shift) - 1);
			mantissa >>= (int)shift;
			inexact = !remainder.IsZero;
			BigInteger half = BigInteger.One << (int)(shift - 1);
			if (remainder > half || (remainder == half && !mantissa.IsEven))
			{
				mantissa += 1;
			}

			exponent += shift;
		}

		value = Math.ScaleB((double)mantissa, (int)Math.Clamp(exponent, int.MinValue, int.MaxValue));
		return true;
	}
}

