// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BitmapColorTests: the TEST(BitmapColor, *) cases of colmap/sensor/bitmap_test.cc ported
// 1:1 (Suite_Name). The TEST(Bitmap, *) cases are in BitmapTests.cs. Tests
// ColmapSharp/Sensor/BitmapColor.cs. Printing (operator<<) is ToString().
//
// Tier A: exact comparisons throughout.

using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sensor;

public class BitmapColorTests
{
	[Test]
	public async Task BitmapColor_Empty()
	{
		var color = new BitmapColor<byte>();
		using (Assert.Multiple())
		{
			await Assert.That(color.R).IsEqualTo((byte)0);
			await Assert.That(color.G).IsEqualTo((byte)0);
			await Assert.That(color.B).IsEqualTo((byte)0);
			await Assert.That(color == new BitmapColor<byte>(0)).IsTrue();
			await Assert.That(color == new BitmapColor<byte>(0, 0, 0)).IsTrue();
		}
	}

	[Test]
	public async Task BitmapColor_Gray()
	{
		var color = new BitmapColor<byte>(5);
		using (Assert.Multiple())
		{
			await Assert.That(color.R).IsEqualTo((byte)5);
			await Assert.That(color.G).IsEqualTo((byte)5);
			await Assert.That(color.B).IsEqualTo((byte)5);
		}
	}

	[Test]
	public async Task BitmapColor_RGB()
	{
		var color = new BitmapColor<byte>(1, 2, 3);
		using (Assert.Multiple())
		{
			await Assert.That(color.R).IsEqualTo((byte)1);
			await Assert.That(color.G).IsEqualTo((byte)2);
			await Assert.That(color.B).IsEqualTo((byte)3);
		}
	}

	[Test]
	public async Task BitmapColor_Cast()
	{
		var color1 = new BitmapColor<float>(1.1f, 2.9f, -3.0f);
		BitmapColor<byte> color2 = color1.Cast<byte>();
		using (Assert.Multiple())
		{
			await Assert.That(color2.R).IsEqualTo((byte)1);
			await Assert.That(color2.G).IsEqualTo((byte)3);
			await Assert.That(color2.B).IsEqualTo((byte)0);
		}
	}

	[Test]
	public async Task BitmapColor_PrintUint8()
	{
		var color = new BitmapColor<byte>(1, 2, 3);
		await Assert.That(color.ToString()).IsEqualTo("RGB(1, 2, 3)");
	}

	[Test]
	public async Task BitmapColor_PrintFloat()
	{
		var color = new BitmapColor<float>(1.3f, 2.4f, 3.5f);
		await Assert.That(color.ToString()).IsEqualTo("RGB(1.3, 2.4, 3.5)");
	}
}
