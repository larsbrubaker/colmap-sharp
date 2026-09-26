// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// GTestDoubleTests: C#-only tests of the test helper ColmapSharp.Tests/GTestDouble.cs
// (gtest's EXPECT_DOUBLE_EQ), pinning its NaN and near-zero behavior.

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests;

public class GTestDoubleTests
{
	private static double UlpsAbove(double value, long ulps) =>
		BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(value) + ulps);

	[Test]
	public async Task CSharpOnly_DoubleEqMatchesGTestAlmostEquals()
	{
		await Assert.That(GTestDouble.DoubleEq(1.0, 1.0)).IsTrue();
		await Assert.That(GTestDouble.DoubleEq(1.0, UlpsAbove(1.0, 4))).IsTrue();
		await Assert.That(GTestDouble.DoubleEq(1.0, UlpsAbove(1.0, 5))).IsFalse();
		await Assert.That(GTestDouble.DoubleEq(UlpsAbove(1.0, 5), 1.0)).IsFalse();

		// NaN equals nothing, not even itself.
		await Assert.That(GTestDouble.DoubleEq(double.NaN, double.NaN)).IsFalse();
		await Assert.That(GTestDouble.DoubleEq(double.NaN, 1.0)).IsFalse();

		// Near zero the distance runs across the sign: -0 and +0 are 0 apart, the smallest
		// subnormals of opposite sign are 2 apart, and 3 of them against -3 are 6 apart.
		await Assert.That(GTestDouble.DoubleEq(-0.0, 0.0)).IsTrue();
		await Assert.That(GTestDouble.DoubleEq(-double.Epsilon, double.Epsilon)).IsTrue();
		await Assert.That(GTestDouble.DoubleEq(-3 * double.Epsilon, 3 * double.Epsilon)).IsFalse();
	}
}
