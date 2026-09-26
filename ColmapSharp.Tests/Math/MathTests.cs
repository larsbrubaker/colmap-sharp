// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MathTests: colmap/math/math_test.cc ported 1:1, one method per gtest TEST(Suite, Name)
// named Suite_Name, same expected values and tolerances. Tests ColmapSharp/Math/MathUtils.cs.
// Each test runs its checks under Assert.Multiple so a failure reports every mismatch,
// like gtest's EXPECT_* rather than stopping at the first.
//
// Tier A (exact): EXPECT_EQ stays IsEqualTo on doubles, never a tolerance.

using ColmapSharp.Math;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Math;

public class MathTests
{
	// M_PI in math.h is 3.14159265358979323846..., which rounds to the same double.
	private const double MPi = System.Math.PI;

	[Test]
	public async Task SignOfNumber_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(MathUtils.SignOfNumber(0)).IsEqualTo(1);
			await Assert.That(MathUtils.SignOfNumber(-0.1)).IsEqualTo(-1);
			await Assert.That(MathUtils.SignOfNumber(0.1)).IsEqualTo(1);
			await Assert.That(MathUtils.SignOfNumber(float.PositiveInfinity)).IsEqualTo(1);
			await Assert.That(MathUtils.SignOfNumber(float.NegativeInfinity)).IsEqualTo(-1);
		}
	}

	[Test]
	public async Task Clamp_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(MathUtils.Clamp(0, -1, 1)).IsEqualTo(0);
			await Assert.That(MathUtils.Clamp(0, 0, 1)).IsEqualTo(0);
			await Assert.That(MathUtils.Clamp(0, -1, 0)).IsEqualTo(0);
			await Assert.That(MathUtils.Clamp(0, -1, 1)).IsEqualTo(0);
			await Assert.That(MathUtils.Clamp(0, 1, 2)).IsEqualTo(1);
			await Assert.That(MathUtils.Clamp(0, -2, -1)).IsEqualTo(-1);
			await Assert.That(MathUtils.Clamp(0, 0, 0)).IsEqualTo(0);
		}
	}

	[Test]
	public async Task DegToRad_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(MathUtils.DegToRad(0.0f)).IsEqualTo(0.0f);
			await Assert.That(MathUtils.DegToRad(0.0)).IsEqualTo(0.0);
			// float - double promotes to double, as in the C++.
			await Assert.That(double.Abs(MathUtils.DegToRad(180.0f) - MPi)).IsLessThan((double)1e-6f);
			await Assert.That(double.Abs(MathUtils.DegToRad(180.0) - MPi)).IsLessThan(1e-6);
		}
	}

	[Test]
	public async Task RadToDeg_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(MathUtils.RadToDeg(0.0f)).IsEqualTo(0.0f);
			await Assert.That(MathUtils.RadToDeg(0.0)).IsEqualTo(0.0);
			await Assert.That(double.Abs(MathUtils.RadToDeg(MPi) - 180.0f)).IsLessThan((double)1e-6f);
			await Assert.That(double.Abs(MathUtils.RadToDeg(MPi) - 180.0)).IsLessThan(1e-6);
		}
	}

	[Test]
	public async Task RadToDeg_Roundtrip()
	{
		using (Assert.Multiple())
		{
			for (int i = 0; i < 360; ++i)
			{
				double angle = i;
				await Assert.That(double.Abs(angle - MathUtils.RadToDeg(MathUtils.DegToRad(angle)))).IsLessThanOrEqualTo(1e-6);
			}
		}
	}

	[Test]
	public async Task Median_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(MathUtils.Median<int>([1, 2, 3, 4])).IsEqualTo(2.5);
			await Assert.That(MathUtils.Median<int>([4, 1, 3, 2])).IsEqualTo(2.5);
			await Assert.That(MathUtils.Median<int>([1, 2, 3, 100])).IsEqualTo(2.5);
			await Assert.That(MathUtils.Median<int>([1, 2, 3, 4, 100])).IsEqualTo(3);
			await Assert.That(MathUtils.Median<int>([4, 100, 1, 3, 2])).IsEqualTo(3);
			await Assert.That(MathUtils.Median<int>([-100, 1, 2, 3, 4])).IsEqualTo(2);
			await Assert.That(MathUtils.Median<int>([-1, -2, -3, -4])).IsEqualTo(-2.5);
			await Assert.That(MathUtils.Median<int>([-3, -1, -4, -2])).IsEqualTo(-2.5);
			await Assert.That(MathUtils.Median<int>([-1, -2, 3, 4])).IsEqualTo(1);
			// Test integer overflow scenario.
			await Assert.That(MathUtils.Median<sbyte>([100, 115, 119, 127])).IsEqualTo(117);
		}
	}

	[Test]
	public async Task MedianAbsoluteDeviation_Nominal()
	{
		using (Assert.Multiple())
		{
			// {1, 2, 3, 4, 5} -> median=3, deviations={2, 1, 0, 1, 2}, MAD=1
			var (median1, mad1) = MathUtils.MedianAbsoluteDeviation<int>([1, 2, 3, 4, 5]);
			await Assert.That(median1).IsEqualTo(3);
			await Assert.That(mad1).IsEqualTo(1);

			// {1, 2, 3, 4} -> median=2.5, deviations={1.5, 0.5, 0.5, 1.5}, MAD=1
			var (median2, mad2) = MathUtils.MedianAbsoluteDeviation<int>([1, 2, 3, 4]);
			await Assert.That(median2).IsEqualTo(2.5);
			await Assert.That(mad2).IsEqualTo(1);

			// Unsorted input: {5, 1, 3, 2, 4} -> same as {1, 2, 3, 4, 5}
			var (median3, mad3) = MathUtils.MedianAbsoluteDeviation<int>([5, 1, 3, 2, 4]);
			await Assert.That(median3).IsEqualTo(3);
			await Assert.That(mad3).IsEqualTo(1);

			// Single element: {42} -> median=42, MAD=0
			var (median4, mad4) = MathUtils.MedianAbsoluteDeviation<int>([42]);
			await Assert.That(median4).IsEqualTo(42);
			await Assert.That(mad4).IsEqualTo(0);

			// With outlier: {1, 2, 3, 4, 100} -> median=3, deviations={2, 1, 0, 1, 97}
			var (median5, mad5) = MathUtils.MedianAbsoluteDeviation<int>([1, 2, 3, 4, 100]);
			await Assert.That(median5).IsEqualTo(3);
			await Assert.That(mad5).IsEqualTo(1);
		}
	}

	[Test]
	public async Task Percentile_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(MathUtils.Percentile<int>([0], 0)).IsEqualTo(0);
			await Assert.That(MathUtils.Percentile<int>([0], 50)).IsEqualTo(0);
			await Assert.That(MathUtils.Percentile<int>([0], 100)).IsEqualTo(0);
			await Assert.That(MathUtils.Percentile<int>([0, 1], 0)).IsEqualTo(0);
			await Assert.That(MathUtils.Percentile<int>([1, 0], 0)).IsEqualTo(0);
			await Assert.That(MathUtils.Percentile<int>([0, 1], 50)).IsEqualTo(0.5);
			await Assert.That(MathUtils.Percentile<int>([1, 0], 50)).IsEqualTo(0.5);
			await Assert.That(MathUtils.Percentile<int>([0, 1], 100)).IsEqualTo(1);
			await Assert.That(MathUtils.Percentile<int>([1, 0], 100)).IsEqualTo(1);
			await Assert.That(MathUtils.Percentile<int>([0, 1, 2], 0)).IsEqualTo(0);
			await Assert.That(MathUtils.Percentile<int>([0, 1, 2], 50)).IsEqualTo(1);
			await Assert.That(MathUtils.Percentile<int>([0, 1, 2], 100)).IsEqualTo(2);
			await Assert.That(MathUtils.Percentile<int>([0, 1, 1, 2], 0)).IsEqualTo(0);
			await Assert.That(MathUtils.Percentile<int>([0, 1, 1, 2], 100.0 / 3.0)).IsEqualTo(1);
			await Assert.That(MathUtils.Percentile<int>([0, 1, 1, 2], 50)).IsEqualTo(1);
			await Assert.That(MathUtils.Percentile<int>([0, 1, 1, 2], 100 / 3.0 * 2.0)).IsEqualTo(1);
			await Assert.That(MathUtils.Percentile<int>([1, 2, 0, 1], 100 / 3.0 * 2.0)).IsEqualTo(1);
			await Assert.That(MathUtils.Percentile<int>([0, 1, 1, 2], 100)).IsEqualTo(2);
			await Assert.That(MathUtils.Percentile<int>([1, 2, 0, 1], 100)).IsEqualTo(2);
			await Assert.That(MathUtils.Percentile<int>([0, 100], 1)).IsEqualTo(1);
			await Assert.That(MathUtils.Percentile<int>([0, 100], 50)).IsEqualTo(50);
			await Assert.That(MathUtils.Percentile<int>([0, 100], 50.1)).IsEqualTo(50.1);
			await Assert.That(MathUtils.Percentile<int>([0, 100], 99)).IsEqualTo(99);
			await Assert.That(MathUtils.Percentile<int>([0, 1, 2, 3], 1)).IsEqualTo(0.03);
			await Assert.That(MathUtils.Percentile<int>([0, 1, 2, 3], 2)).IsEqualTo(0.06);
			await Assert.That(MathUtils.Percentile<int>([0, 1, 2, 3], 33)).IsEqualTo(0.99);
			await Assert.That(MathUtils.Percentile<int>([0, 1, 2, 3], 34)).IsEqualTo(1.02);
			await Assert.That(MathUtils.Percentile<int>([3, 0, 1, 2], 34)).IsEqualTo(1.02);
		}
	}

	[Test]
	public async Task Mean_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(MathUtils.Mean<int>([1, 2, 3, 4])).IsEqualTo(2.5);
			await Assert.That(MathUtils.Mean<int>([1, 2, 3, 100])).IsEqualTo(26.5);
			await Assert.That(MathUtils.Mean<int>([1, 2, 3, 4, 100])).IsEqualTo(22);
			await Assert.That(MathUtils.Mean<int>([-100, 1, 2, 3, 4])).IsEqualTo(-18);
			await Assert.That(MathUtils.Mean<int>([-1, -2, -3, -4])).IsEqualTo(-2.5);
			await Assert.That(MathUtils.Mean<int>([-1, -2, 3, 4])).IsEqualTo(1);
		}
	}

	[Test]
	public async Task Variance_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(double.Abs(MathUtils.Variance<int>([1, 2, 3, 4]) - 1.66666666)).IsLessThanOrEqualTo(1e-6);
			await Assert.That(double.Abs(MathUtils.Variance<int>([1, 2, 3, 100]) - 2401.66666666)).IsLessThanOrEqualTo(1e-6);
			await Assert.That(double.Abs(MathUtils.Variance<int>([1, 2, 3, 4, 100]) - 1902.5)).IsLessThanOrEqualTo(1e-6);
			await Assert.That(double.Abs(MathUtils.Variance<int>([-100, 1, 2, 3, 4]) - 2102.5)).IsLessThanOrEqualTo(1e-6);
			await Assert.That(double.Abs(MathUtils.Variance<int>([-1, -2, -3, -4]) - 1.66666666)).IsLessThanOrEqualTo(1e-6);
			await Assert.That(double.Abs(MathUtils.Variance<int>([-1, -2, 3, 4]) - 8.66666666)).IsLessThanOrEqualTo(1e-6);
		}
	}

	[Test]
	public async Task StdDev_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(double.Abs(double.Sqrt(MathUtils.Variance<int>([1, 2, 3, 4])) - MathUtils.StdDev<int>([1, 2, 3, 4])))
				.IsLessThanOrEqualTo(1e-6);
			await Assert.That(double.Abs(double.Sqrt(MathUtils.Variance<int>([1, 2, 3, 100])) - MathUtils.StdDev<int>([1, 2, 3, 100])))
				.IsLessThanOrEqualTo(1e-6);
		}
	}

	[Test]
	public async Task NextCombination_Nominal()
	{
		using (Assert.Multiple())
		{
			int[] list = [0];
			await Assert.That(MathUtils.NextCombination<int>(list, 1)).IsFalse();
			list = [0, 1];
			await Assert.That(MathUtils.NextCombination<int>(list, 2)).IsFalse();
			await Assert.That(list[0]).IsEqualTo(0);
			await Assert.That(MathUtils.NextCombination<int>(list, 1)).IsTrue();
			await Assert.That(list[0]).IsEqualTo(1);
			await Assert.That(MathUtils.NextCombination<int>(list, 1)).IsFalse();
			await Assert.That(list[0]).IsEqualTo(0);
			list = [0, 1, 2];
			await Assert.That(list[0]).IsEqualTo(0);
			await Assert.That(list[1]).IsEqualTo(1);
			await Assert.That(list[2]).IsEqualTo(2);
			await Assert.That(MathUtils.NextCombination<int>(list, 2)).IsTrue();
			await Assert.That(list[0]).IsEqualTo(0);
			await Assert.That(list[1]).IsEqualTo(2);
			await Assert.That(list[2]).IsEqualTo(1);
			await Assert.That(MathUtils.NextCombination<int>(list, 2)).IsTrue();
			await Assert.That(list[0]).IsEqualTo(1);
			await Assert.That(list[1]).IsEqualTo(2);
			await Assert.That(list[2]).IsEqualTo(0);
			await Assert.That(MathUtils.NextCombination<int>(list, 2)).IsFalse();
			await Assert.That(list[0]).IsEqualTo(0);
			await Assert.That(list[1]).IsEqualTo(1);
			await Assert.That(list[2]).IsEqualTo(2);
		}
	}

	[Test]
	public async Task Sigmoid_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(MathUtils.Sigmoid(0.0)).IsEqualTo(0.5);
			await Assert.That(double.Abs(MathUtils.Sigmoid(100.0) - 1.0)).IsLessThanOrEqualTo(1e-10);
			await Assert.That(double.Abs(MathUtils.Sigmoid(-100.0) - 0)).IsLessThanOrEqualTo(1e-10);
		}
	}

	[Test]
	public async Task ScaleSigmoid_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(double.Abs(MathUtils.ScaleSigmoid(0.5) - 0.5)).IsLessThanOrEqualTo(1e-10);
			await Assert.That(double.Abs(MathUtils.ScaleSigmoid(1.0) - 1.0)).IsLessThanOrEqualTo(1e-10);
			await Assert.That(double.Abs(MathUtils.ScaleSigmoid(-1.0) - 0)).IsLessThanOrEqualTo(1e-4);
		}
	}

	[Test]
	public async Task NChooseK_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(MathUtils.NChooseK(0, 0)).IsEqualTo(0UL);

			await Assert.That(MathUtils.NChooseK(1, 0)).IsEqualTo(1UL);
			await Assert.That(MathUtils.NChooseK(2, 0)).IsEqualTo(1UL);
			await Assert.That(MathUtils.NChooseK(3, 0)).IsEqualTo(1UL);

			await Assert.That(MathUtils.NChooseK(1, 1)).IsEqualTo(1UL);
			await Assert.That(MathUtils.NChooseK(2, 1)).IsEqualTo(2UL);
			await Assert.That(MathUtils.NChooseK(3, 1)).IsEqualTo(3UL);

			await Assert.That(MathUtils.NChooseK(2, 2)).IsEqualTo(1UL);
			await Assert.That(MathUtils.NChooseK(2, 3)).IsEqualTo(0UL);

			await Assert.That(MathUtils.NChooseK(3, 2)).IsEqualTo(3UL);
			await Assert.That(MathUtils.NChooseK(4, 2)).IsEqualTo(6UL);
			await Assert.That(MathUtils.NChooseK(5, 2)).IsEqualTo(10UL);

			await Assert.That(MathUtils.NChooseK(500, 3)).IsEqualTo(20708500UL);
			await Assert.That(MathUtils.NChooseK(500, 7)).IsEqualTo(1486071034734000UL);
			await Assert.That(MathUtils.NChooseK(10000, 5)).IsEqualTo(832500291625002000UL);
		}
	}

	[Test]
	public async Task TruncateCast_Nominal()
	{
		using (Assert.Multiple())
		{
			await Assert.That(MathUtils.TruncateCast<int, sbyte>(-129)).IsEqualTo((sbyte)-128);
			await Assert.That(MathUtils.TruncateCast<int, sbyte>(128)).IsEqualTo((sbyte)127);
			await Assert.That(MathUtils.TruncateCast<int, byte>(-1)).IsEqualTo((byte)0);
			await Assert.That(MathUtils.TruncateCast<int, byte>(256)).IsEqualTo((byte)255);
			await Assert.That(MathUtils.TruncateCast<int, ushort>(-1)).IsEqualTo((ushort)0);
			await Assert.That(MathUtils.TruncateCast<int, ushort>(65536)).IsEqualTo((ushort)65535);
		}
	}
}
