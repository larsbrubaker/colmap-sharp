// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from googletest (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GTestDouble: gtest's EXPECT_DOUBLE_EQ comparison, following FloatingPoint<double>::
// AlmostEquals in googletest's gtest-internal.h: NaN equals nothing, and two values are
// equal when at most 4 units in the last place apart, measured on the sign-and-magnitude bits
// mapped to a biased (monotonic) integer, so -0 and +0, and tiny values of opposite sign, are
// close. Shared by the ported tests that use EXPECT_DOUBLE_EQ.
// Tests: ColmapSharp.Tests/GTestDoubleTests.cs.

namespace ColmapSharp.Tests;

/// <summary>gtest's floating-point equality for doubles.</summary>
internal static class GTestDouble
{
	// gtest's kMaxUlps.
	private const ulong MaxUlps = 4;

	private const ulong SignBitMask = 1UL << 63;

	/// <summary>EXPECT_DOUBLE_EQ: AlmostEquals, within 4 units in the last place.</summary>
	public static bool DoubleEq(double a, double b)
	{
		// The IEEE standard says that any comparison operation involving a NaN must return
		// false.
		if (double.IsNaN(a) || double.IsNaN(b))
		{
			return false;
		}

		return DistanceBetweenSignAndMagnitudeNumbers(
			unchecked((ulong)BitConverter.DoubleToInt64Bits(a)),
			unchecked((ulong)BitConverter.DoubleToInt64Bits(b))) <= MaxUlps;
	}

	// gtest's SignAndMagnitudeToBiased: maps the sign-and-magnitude bits to an unsigned
	// integer that increases with the value (negative numbers below positive ones).
	private static ulong SignAndMagnitudeToBiased(ulong sam) =>
		(sam & SignBitMask) != 0 ? unchecked(~sam + 1) : SignBitMask | sam;

	// gtest's DistanceBetweenSignAndMagnitudeNumbers.
	private static ulong DistanceBetweenSignAndMagnitudeNumbers(ulong sam1, ulong sam2)
	{
		ulong biased1 = SignAndMagnitudeToBiased(sam1);
		ulong biased2 = SignAndMagnitudeToBiased(sam2);
		return biased1 >= biased2 ? biased1 - biased2 : biased2 - biased1;
	}
}
