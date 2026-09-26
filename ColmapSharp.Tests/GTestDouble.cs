// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// GTestDouble: gtest's EXPECT_DOUBLE_EQ comparison (equal within 4 units in the last place),
// shared by the ported tests that use it.

namespace ColmapSharp.Tests;

/// <summary>gtest's floating-point equality for doubles.</summary>
internal static class GTestDouble
{
	/// <summary>EXPECT_DOUBLE_EQ: equal within 4 units in the last place.</summary>
	public static bool DoubleEq(double a, double b)
	{
		if (a == b)
		{
			return true;
		}

		long ia = BitConverter.DoubleToInt64Bits(a);
		long ib = BitConverter.DoubleToInt64Bits(b);
		if ((ia < 0) != (ib < 0))
		{
			return false;
		}

		return Math.Abs(ia - ib) <= 4;
	}
}
