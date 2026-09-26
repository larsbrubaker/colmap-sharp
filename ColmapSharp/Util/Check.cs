// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Check: the THROW_CHECK family of colmap/util/logging.h (THROW_CHECK, THROW_CHECK_EQ/NE/
// LT/LE/GT/GE, THROW_CHECK_NOTNULL). Every ported module uses it for argument validation,
// so COLMAP's error messages carry over. The CHECK/LOG(FATAL) abort family is not ported:
// CLAUDE.md maps LOG(FATAL) to an exception at each site.
//
// Messages follow COLMAP's LogMessageFatalThrow exactly in shape:
//   "[<file>:<line>] Check failed: <expr> "                          THROW_CHECK
//   "[<file>:<line>] Check failed: <a> <op> <b> (<va> vs. <vb>) "    THROW_CHECK_OP
//   "[<file>:<line>] '<expr>' Must be non NULL"                      THROW_CHECK_NOTNULL
// followed by the optional message (C++'s `THROW_CHECK(x) << message`). Two things differ
// by construction: <file>:<line> is the C# caller's, and <expr> is the C# argument text
// (CallerArgumentExpression), not the C++ source text. Operand values are formatted
// invariantly, floating point with "G6" to match an ostream's default precision of 6.
//
// COLMAP throws std::invalid_argument for all of these (LogMessageFatalThrowDefault), which
// maps to ArgumentException.
//
// C++'s `THROW_CHECK(x) << a << b` only streams the message when the check fails. A plain
// string argument would be formatted on every call, so Check.That also takes an interpolated
// string through CheckMessageHandler (the same trick as Debug.Assert's handler): an
// interpolated message is only built when the condition is false. Holes are formatted with
// the invariant culture; a double hole prints C#'s shortest round-trip form, not an
// ostream's 6 significant digits, so pass CppStreamFormat.FormatDouble(x) where that matters.

using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace ColmapSharp.Util;

/// <summary>
/// Port of COLMAP's THROW_CHECK macros. Each method throws <see cref="ArgumentException"/>
/// with COLMAP's "Check failed: ..." message when its condition does not hold.
/// </summary>
internal static class Check
{
	/// <summary>Port of THROW_CHECK(condition).</summary>
	public static void That(
		[DoesNotReturnIf(false)] bool condition,
		string? message = null,
		[CallerArgumentExpression(nameof(condition))] string expression = "",
		[CallerFilePath] string file = "",
		[CallerLineNumber] int line = 0)
	{
		if (!condition)
		{
			throw Fail($"Check failed: {expression} ", message, file, line);
		}
	}

	/// <summary>
	/// Port of THROW_CHECK(condition) &lt;&lt; message with an interpolated message, which is only
	/// formatted when <paramref name="condition"/> is false.
	/// </summary>
	public static void That(
		[DoesNotReturnIf(false)] bool condition,
		[InterpolatedStringHandlerArgument(nameof(condition))] ref CheckMessageHandler message,
		[CallerArgumentExpression(nameof(condition))] string expression = "",
		[CallerFilePath] string file = "",
		[CallerLineNumber] int line = 0)
	{
		if (!condition)
		{
			throw Fail($"Check failed: {expression} ", message.ToStringAndClear(), file, line);
		}
	}

	/// <summary>Port of THROW_CHECK_EQ(val1, val2).</summary>
	public static void Eq<T>(
		T val1,
		T val2,
		string? message = null,
		[CallerArgumentExpression(nameof(val1))] string expr1 = "",
		[CallerArgumentExpression(nameof(val2))] string expr2 = "",
		[CallerFilePath] string file = "",
		[CallerLineNumber] int line = 0)
		where T : IEqualityOperators<T, T, bool>
	{
		if (!(val1 == val2))
		{
			throw FailOp("==", val1, val2, expr1, expr2, message, file, line);
		}
	}

	/// <summary>Port of THROW_CHECK_NE(val1, val2).</summary>
	public static void Ne<T>(
		T val1,
		T val2,
		string? message = null,
		[CallerArgumentExpression(nameof(val1))] string expr1 = "",
		[CallerArgumentExpression(nameof(val2))] string expr2 = "",
		[CallerFilePath] string file = "",
		[CallerLineNumber] int line = 0)
		where T : IEqualityOperators<T, T, bool>
	{
		if (!(val1 != val2))
		{
			throw FailOp("!=", val1, val2, expr1, expr2, message, file, line);
		}
	}

	/// <summary>Port of THROW_CHECK_LT(val1, val2).</summary>
	public static void Lt<T>(
		T val1,
		T val2,
		string? message = null,
		[CallerArgumentExpression(nameof(val1))] string expr1 = "",
		[CallerArgumentExpression(nameof(val2))] string expr2 = "",
		[CallerFilePath] string file = "",
		[CallerLineNumber] int line = 0)
		where T : IComparisonOperators<T, T, bool>
	{
		if (!(val1 < val2))
		{
			throw FailOp("<", val1, val2, expr1, expr2, message, file, line);
		}
	}

	/// <summary>Port of THROW_CHECK_LE(val1, val2).</summary>
	public static void Le<T>(
		T val1,
		T val2,
		string? message = null,
		[CallerArgumentExpression(nameof(val1))] string expr1 = "",
		[CallerArgumentExpression(nameof(val2))] string expr2 = "",
		[CallerFilePath] string file = "",
		[CallerLineNumber] int line = 0)
		where T : IComparisonOperators<T, T, bool>
	{
		if (!(val1 <= val2))
		{
			throw FailOp("<=", val1, val2, expr1, expr2, message, file, line);
		}
	}

	/// <summary>Port of THROW_CHECK_GT(val1, val2).</summary>
	public static void Gt<T>(
		T val1,
		T val2,
		string? message = null,
		[CallerArgumentExpression(nameof(val1))] string expr1 = "",
		[CallerArgumentExpression(nameof(val2))] string expr2 = "",
		[CallerFilePath] string file = "",
		[CallerLineNumber] int line = 0)
		where T : IComparisonOperators<T, T, bool>
	{
		if (!(val1 > val2))
		{
			throw FailOp(">", val1, val2, expr1, expr2, message, file, line);
		}
	}

	/// <summary>Port of THROW_CHECK_GE(val1, val2).</summary>
	public static void Ge<T>(
		T val1,
		T val2,
		string? message = null,
		[CallerArgumentExpression(nameof(val1))] string expr1 = "",
		[CallerArgumentExpression(nameof(val2))] string expr2 = "",
		[CallerFilePath] string file = "",
		[CallerLineNumber] int line = 0)
		where T : IComparisonOperators<T, T, bool>
	{
		if (!(val1 >= val2))
		{
			throw FailOp(">=", val1, val2, expr1, expr2, message, file, line);
		}
	}

	/// <summary>
	/// Port of THROW_CHECK_NOTNULL(val): returns the value when it is not null, as the C++
	/// does, so it can be used inline.
	/// </summary>
	public static T NotNull<T>(
		[NotNull] T? value,
		[CallerArgumentExpression(nameof(value))] string expression = "",
		[CallerFilePath] string file = "",
		[CallerLineNumber] int line = 0)
		where T : class
	{
		if (value is null)
		{
			throw Fail($"'{expression}' Must be non NULL", null, file, line);
		}

		return value;
	}

	private static ArgumentException FailOp<T>(
		string op, T val1, T val2, string expr1, string expr2, string? message, string file, int line)
	{
		// glog's CheckOpString is "<a> <op> <b> (<va> vs. <vb>)"; COLMAP prepends
		// "Check failed: " and appends a space.
		return Fail(
			$"Check failed: {expr1} {op} {expr2} ({Format(val1)} vs. {Format(val2)}) ", message, file, line);
	}

	private static ArgumentException Fail(string text, string? message, string file, int line)
	{
		// COLMAP's __MakeExceptionPrefix: "[" + base name of __FILE__ + ":" + line + "] ".
		return new ArgumentException($"[{Path.GetFileName(file)}:{line}] {text}{message}");
	}

	private static string Format<T>(T value)
	{
		return value switch
		{
			double d => d.ToString("G6", CultureInfo.InvariantCulture),
			float f => f.ToString("G6", CultureInfo.InvariantCulture),
			IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
			null => "null",
			_ => value.ToString() ?? string.Empty,
		};
	}
}

/// <summary>
/// Interpolated string handler for <see cref="Check.That(bool, ref CheckMessageHandler, string, string, int)"/>:
/// appends nothing unless the checked condition is false, so a passing check never formats
/// its message.
/// </summary>
[InterpolatedStringHandler]
internal ref struct CheckMessageHandler
{
	private DefaultInterpolatedStringHandler _builder;
	private readonly bool _enabled;

	/// <summary>Called by the compiler with the checked condition.</summary>
	public CheckMessageHandler(int literalLength, int formattedCount, bool condition, out bool shouldAppend)
	{
		_enabled = !condition;
		shouldAppend = _enabled;
		_builder = _enabled
			? new DefaultInterpolatedStringHandler(literalLength, formattedCount, CultureInfo.InvariantCulture)
			: default;
	}

	/// <summary>Appends a literal part.</summary>
	public void AppendLiteral(string value) => _builder.AppendLiteral(value);

	/// <summary>Appends a hole, formatted invariantly.</summary>
	public void AppendFormatted<T>(T value) => _builder.AppendFormatted(value);

	/// <summary>Appends a hole with a format string, formatted invariantly.</summary>
	public void AppendFormatted<T>(T value, string? format) => _builder.AppendFormatted(value, format);

	/// <summary>Appends a string hole.</summary>
	public void AppendFormatted(string? value) => _builder.AppendFormatted(value);

	/// <summary>The built message, or "" when the check passed.</summary>
	internal string ToStringAndClear() => _enabled ? _builder.ToStringAndClear() : string.Empty;
}
