// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ExpectationLog: gtest EXPECT_* semantics for ported tests whose C++ helpers make
// thousands of checks in loops (first user: Sensor/ModelsTests.cs). A failed expectation is
// recorded and the helper keeps going, as EXPECT_* does; the test then asserts the log is
// empty, so one run lists every failure. EXPECT_NEAR is |a - b| <= tolerance, as in gtest.
// C#-only test infrastructure.

using System.Globalization;

namespace ColmapSharp.Tests;

/// <summary>
/// Collects failed gtest-style expectations instead of stopping at the first.
/// </summary>
internal sealed class ExpectationLog
{
	/// <summary>One message per failed expectation, in order.</summary>
	public List<string> Failures { get; } = [];

	/// <summary>EXPECT_TRUE. Returns the condition, so ASSERT_TRUE can return early.</summary>
	public bool True(bool condition, string what)
	{
		if (!condition)
		{
			Failures.Add($"expected true: {what}");
		}

		return condition;
	}

	/// <summary>EXPECT_FALSE.</summary>
	public bool False(bool condition, string what) => True(!condition, $"not {what}");

	/// <summary>EXPECT_EQ on doubles: exact equality (bit-level for non-NaN values).</summary>
	public void Equal(double actual, double expected, string what)
	{
		if (!(actual == expected))
		{
			Failures.Add(string.Create(CultureInfo.InvariantCulture, $"{what}: expected {expected:R}, got {actual:R}"));
		}
	}

	/// <summary>EXPECT_EQ on any equatable value.</summary>
	public void Equal<T>(T actual, T expected, string what)
	{
		if (!EqualityComparer<T>.Default.Equals(actual, expected))
		{
			Failures.Add($"{what}: expected {expected}, got {actual}");
		}
	}

	/// <summary>EXPECT_EQ on sequences (element-wise, same length).</summary>
	public void Equal<T>(T[] actual, T[] expected, string what)
	{
		if (!actual.SequenceEqual(expected))
		{
			Failures.Add($"{what}: expected [{string.Join(", ", expected)}], got [{string.Join(", ", actual)}]");
		}
	}

	/// <summary>EXPECT_NEAR: |actual - expected| &lt;= tolerance.</summary>
	public void Near(double actual, double expected, double tolerance, string what)
	{
		if (!(Math.Abs(actual - expected) <= tolerance))
		{
			Failures.Add(string.Create(CultureInfo.InvariantCulture, $"{what}: expected {expected:R} +- {tolerance}, got {actual:R}"));
		}
	}

	/// <summary>EXPECT_LT: actual &lt; bound.</summary>
	public void Less(double actual, double bound, string what)
	{
		if (!(actual < bound))
		{
			Failures.Add(string.Create(CultureInfo.InvariantCulture, $"{what}: expected < {bound:R}, got {actual:R}"));
		}
	}

	/// <summary>EXPECT_GT: actual &gt; bound.</summary>
	public void Greater(double actual, double bound, string what)
	{
		if (!(actual > bound))
		{
			Failures.Add(string.Create(CultureInfo.InvariantCulture, $"{what}: expected > {bound:R}, got {actual:R}"));
		}
	}

	/// <summary>ADD_FAILURE() &lt;&lt; message.</summary>
	public void Fail(string message) => Failures.Add(message);
}
