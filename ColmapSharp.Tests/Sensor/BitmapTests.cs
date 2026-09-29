// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BitmapTests: the TEST(Bitmap, *) cases of colmap/sensor/bitmap_test.cc ported 1:1
// (Suite_Name), split in two files: this one covers storage, pixels, interpolation, resizing,
// rotation and cloning; BitmapTests.Exif.cs covers metadata and the EXIF getters. The
// TEST(BitmapColor, *) cases are in BitmapColorTests.cs. Tests ColmapSharp/Sensor/Bitmap*.cs.
//
// Not ported (listed in PORTING_PLAN.md):
// - MoveConstructEmpty, MoveConstruct, MoveAssignEmpty, MoveAssign: C++ move semantics
//   (the moved-from object becomes empty) have no C# counterpart; Bitmap is a reference type.
// - Every case that reads or writes image files through OpenImageIO (ReadWrite*,
//   WriteJpegWithQuality, WriteInvalidFormat, the ParameterizedBitmapFormatTests, ReadNonImageFile,
//   ReadNonExistentFile, ReadUnsupportedChannels): the library does no file decoding; the
//   host does (docs/LICENSE_AUDIT.md). CloneAsRGB and CloneAsGrey end with a PNG write/read
//   round trip; their in-memory checks are ported and the round trip is dropped for the
//   same reason.
//
// Tier A everywhere except the Rescale cases, which (like COLMAP's) pin only dimensions
// and that the two filters differ (divergence 9).
// C++ copy construction (const Bitmap&) is a reference here and copy assignment is Clone().

using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sensor;

public partial class BitmapTests
{
	[Test]
	public async Task Bitmap_Empty()
	{
		var bitmap = new Bitmap();
		using (Assert.Multiple())
		{
			await Assert.That(bitmap.Width).IsEqualTo(0);
			await Assert.That(bitmap.Height).IsEqualTo(0);
			await Assert.That(bitmap.Channels).IsEqualTo(0);
			await Assert.That(bitmap.IsRGB).IsFalse();
			await Assert.That(bitmap.IsGrey).IsFalse();
			await Assert.That(bitmap.IsEmpty).IsTrue();
		}
	}

	[Test]
	public async Task Bitmap_Print()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);
		await Assert.That(bitmap.ToString()).IsEqualTo("Bitmap(width=100, height=80, channels=3)");
	}

	[Test]
	public async Task Bitmap_AllocateRGB()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);
		using (Assert.Multiple())
		{
			await Assert.That(bitmap.Width).IsEqualTo(100);
			await Assert.That(bitmap.Height).IsEqualTo(80);
			await Assert.That(bitmap.Channels).IsEqualTo(3);
			await Assert.That(bitmap.IsRGB).IsTrue();
			await Assert.That(bitmap.IsGrey).IsFalse();
			await Assert.That(bitmap.IsEmpty).IsFalse();
		}
	}

	[Test]
	public async Task Bitmap_AllocateGrey()
	{
		var bitmap = new Bitmap(100, 80, asRgb: false);
		using (Assert.Multiple())
		{
			await Assert.That(bitmap.Width).IsEqualTo(100);
			await Assert.That(bitmap.Height).IsEqualTo(80);
			await Assert.That(bitmap.Channels).IsEqualTo(1);
			await Assert.That(bitmap.IsRGB).IsFalse();
			await Assert.That(bitmap.IsGrey).IsTrue();
			await Assert.That(bitmap.IsEmpty).IsFalse();
		}
	}

	[Test]
	public async Task Bitmap_ConstructCopyEmpty()
	{
		var bitmap = new Bitmap();
		Bitmap copiedBitmap = bitmap;
		using (Assert.Multiple())
		{
			await Assert.That(copiedBitmap.Width).IsEqualTo(0);
			await Assert.That(copiedBitmap.Height).IsEqualTo(0);
			await Assert.That(copiedBitmap.Channels).IsEqualTo(0);
		}
	}

	[Test]
	public async Task Bitmap_ConstructCopy()
	{
		var bitmap = new Bitmap(2, 1, asRgb: true);
		Bitmap copiedBitmap = bitmap;
		using (Assert.Multiple())
		{
			await Assert.That(copiedBitmap.Width).IsEqualTo(2);
			await Assert.That(copiedBitmap.Height).IsEqualTo(1);
			await Assert.That(copiedBitmap.Channels).IsEqualTo(3);
			await Assert.That(bitmap.Width).IsEqualTo(2);
			await Assert.That(bitmap.Height).IsEqualTo(1);
			await Assert.That(bitmap.Channels).IsEqualTo(3);
		}
	}

	[Test]
	public async Task Bitmap_AssignCopyEmpty()
	{
		var bitmap = new Bitmap();
		var copiedBitmap = new Bitmap();
		copiedBitmap = bitmap.Clone();
		using (Assert.Multiple())
		{
			await Assert.That(copiedBitmap.Width).IsEqualTo(0);
			await Assert.That(copiedBitmap.Height).IsEqualTo(0);
			await Assert.That(copiedBitmap.Channels).IsEqualTo(0);
		}
	}

	[Test]
	public async Task Bitmap_AssignCopy()
	{
		var bitmap = new Bitmap(2, 1, asRgb: true);
		var copiedBitmap = new Bitmap();
		copiedBitmap = bitmap.Clone();
		using (Assert.Multiple())
		{
			await Assert.That(copiedBitmap.Width).IsEqualTo(2);
			await Assert.That(copiedBitmap.Height).IsEqualTo(1);
			await Assert.That(copiedBitmap.Channels).IsEqualTo(3);
			await Assert.That(bitmap.Width).IsEqualTo(2);
			await Assert.That(bitmap.Height).IsEqualTo(1);
			await Assert.That(bitmap.Channels).IsEqualTo(3);
		}
	}

	[Test]
	public async Task Bitmap_BitsPerPixel()
	{
		var bitmap = new Bitmap(1, 1, asRgb: true);
		await Assert.That(bitmap.BitsPerPixel).IsEqualTo(24);
		bitmap = new Bitmap(1, 1, asRgb: false);
		await Assert.That(bitmap.BitsPerPixel).IsEqualTo(8);
	}

	[Test]
	public async Task Bitmap_NumBytes()
	{
		var bitmap = new Bitmap();
		await Assert.That(bitmap.NumBytes).IsEqualTo(0);
		bitmap = new Bitmap(100, 80, asRgb: true);
		await Assert.That(bitmap.NumBytes).IsEqualTo(3 * 100 * 80);
		bitmap = new Bitmap(100, 80, asRgb: false);
		await Assert.That(bitmap.NumBytes).IsEqualTo(100 * 80);
	}

	[Test]
	public async Task Bitmap_RowMajorDataRGB()
	{
		var bitmap = new Bitmap(2, 3, asRgb: true);
		bitmap.SetPixel(0, 0, new BitmapColor<byte>(0, 0, 0));
		bitmap.SetPixel(0, 1, new BitmapColor<byte>(1, 0, 0));
		bitmap.SetPixel(0, 2, new BitmapColor<byte>(2, 0, 0));
		bitmap.SetPixel(1, 0, new BitmapColor<byte>(3, 0, 0));
		bitmap.SetPixel(1, 1, new BitmapColor<byte>(4, 0, 0));
		bitmap.SetPixel(1, 2, new BitmapColor<byte>(5, 0, 0));
		byte[] expected = [0, 0, 0, 3, 0, 0, 1, 0, 0, 4, 0, 0, 2, 0, 0, 5, 0, 0];
		await Assert.That(bitmap.RowMajorData.SequenceEqual(expected)).IsTrue();
	}

	[Test]
	public async Task Bitmap_RowMajorDataGrey()
	{
		var bitmap = new Bitmap(2, 3, asRgb: false);
		bitmap.SetPixel(0, 0, new BitmapColor<byte>(0, 0, 0));
		bitmap.SetPixel(0, 1, new BitmapColor<byte>(1, 0, 0));
		bitmap.SetPixel(0, 2, new BitmapColor<byte>(2, 0, 0));
		bitmap.SetPixel(1, 0, new BitmapColor<byte>(3, 0, 0));
		bitmap.SetPixel(1, 1, new BitmapColor<byte>(4, 0, 0));
		bitmap.SetPixel(1, 2, new BitmapColor<byte>(5, 0, 0));
		byte[] expected = [0, 3, 1, 4, 2, 5];
		await Assert.That(bitmap.RowMajorData.SequenceEqual(expected)).IsTrue();
	}

	[Test]
	public async Task Bitmap_GetAndSetPixelRGB()
	{
		var bitmap = new Bitmap(2, 3, asRgb: true);
		bitmap.SetPixel(1, 1, new BitmapColor<byte>(1, 2, 3));
		var color = bitmap.GetPixel(1, 1);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(color!.Value == new BitmapColor<byte>(1, 2, 3)).IsTrue();
	}

	[Test]
	public async Task Bitmap_GetAndSetPixelGrey()
	{
		var bitmap = new Bitmap(2, 3, asRgb: false);
		bitmap.SetPixel(1, 1, new BitmapColor<byte>(0, 2, 3));
		var color = bitmap.GetPixel(1, 1);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(color!.Value == new BitmapColor<byte>(0, 0, 0)).IsTrue();
		bitmap.SetPixel(1, 1, new BitmapColor<byte>(1, 2, 3));
		color = bitmap.GetPixel(1, 1);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(color!.Value == new BitmapColor<byte>(1, 1, 1)).IsTrue();
	}

	[Test]
	public async Task Bitmap_Fill()
	{
		var bitmap = new Bitmap(100, 100, asRgb: true);
		bitmap.Fill(new BitmapColor<byte>(1, 2, 3));
		for (int y = 0; y < bitmap.Height; ++y)
		{
			for (int x = 0; x < bitmap.Width; ++x)
			{
				var color = bitmap.GetPixel(x, y);
				await Assert.That(color.HasValue).IsTrue();
				await Assert.That(color!.Value == new BitmapColor<byte>(1, 2, 3)).IsTrue();
			}
		}
	}

	[Test]
	public async Task Bitmap_FillGrey()
	{
		var bitmap = new Bitmap(10, 10, asRgb: false);
		bitmap.Fill(new BitmapColor<byte>(42));
		for (int y = 0; y < bitmap.Height; ++y)
		{
			for (int x = 0; x < bitmap.Width; ++x)
			{
				var color = bitmap.GetPixel(x, y);
				await Assert.That(color.HasValue).IsTrue();
				await Assert.That(color!.Value.R).IsEqualTo((byte)42);
			}
		}
	}

	[Test]
	public async Task Bitmap_InterpolateNearestNeighbor()
	{
		var bitmap = new Bitmap(11, 10, asRgb: true);
		bitmap.Fill(new BitmapColor<byte>(0, 0, 0));
		bitmap.SetPixel(5, 4, new BitmapColor<byte>(1, 2, 3));
		var color = bitmap.InterpolateNearestNeighbor(5, 4);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(color!.Value == new BitmapColor<byte>(1, 2, 3)).IsTrue();
		color = bitmap.InterpolateNearestNeighbor(5.4999, 4.4999);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(color!.Value == new BitmapColor<byte>(1, 2, 3)).IsTrue();
		color = bitmap.InterpolateNearestNeighbor(5.5, 4.5);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(color!.Value == new BitmapColor<byte>(0, 0, 0)).IsTrue();
		color = bitmap.InterpolateNearestNeighbor(4.5, 4.4999);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(color!.Value == new BitmapColor<byte>(1, 2, 3)).IsTrue();
	}

	[Test]
	public async Task Bitmap_InterpolateBilinear()
	{
		var bitmap = new Bitmap(11, 10, asRgb: true);
		bitmap.Fill(new BitmapColor<byte>(0, 0, 0));
		bitmap.SetPixel(5, 4, new BitmapColor<byte>(1, 2, 3));
		var color = bitmap.InterpolateBilinear(5, 4);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(color!.Value == new BitmapColor<float>(1, 2, 3)).IsTrue();
		color = bitmap.InterpolateBilinear(5.5, 4);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(color!.Value == new BitmapColor<float>(0.5f, 1, 1.5f)).IsTrue();
		color = bitmap.InterpolateBilinear(5.5, 4.5);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(color!.Value == new BitmapColor<float>(0.25f, 0.5f, 0.75f)).IsTrue();
	}

	[Test]
	public async Task Bitmap_InterpolateBilinearGrey()
	{
		var bitmap = new Bitmap(11, 10, asRgb: false);
		bitmap.Fill(new BitmapColor<byte>(0));
		bitmap.SetPixel(5, 4, new BitmapColor<byte>(100));
		var color = bitmap.InterpolateBilinear(5, 4);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(color!.Value.R).IsEqualTo(100.0f);
		color = bitmap.InterpolateBilinear(5.5, 4);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(Math.Abs((double)color!.Value.R - 50.0)).IsLessThanOrEqualTo(1e-5);
		color = bitmap.InterpolateBilinear(5.5, 4.5);
		await Assert.That(color.HasValue).IsTrue();
		await Assert.That(Math.Abs((double)color!.Value.R - 25.0)).IsLessThanOrEqualTo(1e-5);
	}

	[Test]
	public async Task Bitmap_InterpolateBilinearOutOfBounds()
	{
		var bitmap = new Bitmap(11, 10, asRgb: true);
		bitmap.Fill(new BitmapColor<byte>(1, 2, 3));
		using (Assert.Multiple())
		{
			// x at the right boundary: x0=10, x1=11 >= width_=11
			await Assert.That(bitmap.InterpolateBilinear(10.0, 5.0).HasValue).IsFalse();
			// y at the bottom boundary: y0=9, y1=10 >= height_=10
			await Assert.That(bitmap.InterpolateBilinear(5.0, 9.0).HasValue).IsFalse();
			// Negative x: x0=-1 < 0
			await Assert.That(bitmap.InterpolateBilinear(-0.5, 5.0).HasValue).IsFalse();
			// Negative y: y0=-1 < 0
			await Assert.That(bitmap.InterpolateBilinear(5.0, -0.5).HasValue).IsFalse();
		}
	}

	// C#-only (divergence 117): a sample point beyond int range, or NaN,
	// is outside the image. .NET saturates (int)double, so floor(x) = int.MaxValue made
	// x0 + 1 wrap negative, pass the bounds check and index far outside the pixel array.
	[Test]
	public async Task CSharpOnly_InterpolateFarOutsideOrNaN_ReturnsNull()
	{
		foreach (bool asRgb in new[] { true, false })
		{
			var bitmap = new Bitmap(11, 10, asRgb);
			bitmap.Fill(new BitmapColor<byte>(1, 2, 3));
			double[] bad = [3e9, -3e9, int.MaxValue, double.PositiveInfinity, double.NegativeInfinity, double.NaN];
			foreach (double v in bad)
			{
				using (Assert.Multiple())
				{
					await Assert.That(bitmap.InterpolateBilinear(v, 5.0).HasValue).IsFalse();
					await Assert.That(bitmap.InterpolateBilinear(5.0, v).HasValue).IsFalse();
					await Assert.That(bitmap.InterpolateNearestNeighbor(v, 5.0).HasValue).IsFalse();
					await Assert.That(bitmap.InterpolateNearestNeighbor(5.0, v).HasValue).IsFalse();
				}
			}
		}
	}

	[Test]
	public async Task Bitmap_RescaleRGB()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);
		Bitmap bitmap1 = bitmap.Clone();
		bitmap1.Rescale(50, 25);
		Bitmap bitmap2 = bitmap.Clone();
		bitmap2.Rescale(150, 20);
		using (Assert.Multiple())
		{
			await Assert.That(bitmap1.Width).IsEqualTo(50);
			await Assert.That(bitmap1.Height).IsEqualTo(25);
			await Assert.That(bitmap1.Channels).IsEqualTo(3);
			await Assert.That(bitmap2.Width).IsEqualTo(150);
			await Assert.That(bitmap2.Height).IsEqualTo(20);
			await Assert.That(bitmap2.Channels).IsEqualTo(3);
		}
	}

	[Test]
	public async Task Bitmap_RescaleGrey()
	{
		var bitmap = new Bitmap(100, 80, asRgb: false);
		Bitmap bitmap1 = bitmap.Clone();
		bitmap1.Rescale(50, 25);
		Bitmap bitmap2 = bitmap.Clone();
		bitmap2.Rescale(150, 20);
		using (Assert.Multiple())
		{
			await Assert.That(bitmap1.Width).IsEqualTo(50);
			await Assert.That(bitmap1.Height).IsEqualTo(25);
			await Assert.That(bitmap1.Channels).IsEqualTo(1);
			await Assert.That(bitmap2.Width).IsEqualTo(150);
			await Assert.That(bitmap2.Height).IsEqualTo(20);
			await Assert.That(bitmap2.Channels).IsEqualTo(1);
		}
	}

	[Test]
	public async Task Bitmap_RescaleFilters()
	{
		var bitmap = new Bitmap(4, 4, asRgb: false);
		bitmap.Fill(new BitmapColor<byte>(0));
		bitmap.SetPixel(0, 0, new BitmapColor<byte>(255));

		Bitmap bilinear = bitmap.Clone();
		bilinear.Rescale(1, 1, Bitmap.RescaleFilter.Bilinear);
		Bitmap box = bitmap.Clone();
		box.Rescale(1, 1, Bitmap.RescaleFilter.Box);

		await Assert.That(bilinear.GetPixel(0, 0)!.Value.R).IsNotEqualTo(box.GetPixel(0, 0)!.Value.R);
	}

	[Test]
	public async Task Bitmap_Thumbnail()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);

		// Landscape: width is the limiting dimension.
		Bitmap landscape = bitmap.Clone();
		double landscapeScale = landscape.Thumbnail(maxImageSize: 50);

		// Portrait: height is the limiting dimension.
		var portrait = new Bitmap(80, 100, asRgb: true);
		double portraitScale = portrait.Thumbnail(maxImageSize: 50);

		// Non-integer scale: dimensions are rounded to the nearest integer, so the
		// limiting dimension fits exactly into max_image_size (300 * 100/300 rounds
		// to 100 rather than truncating to 99) and the other is rounded (200 *
		// 100/300 = 66.67 rounds to 67).
		var nonInteger = new Bitmap(300, 200, asRgb: true);
		nonInteger.Thumbnail(maxImageSize: 100);

		using (Assert.Multiple())
		{
			await Assert.That(landscapeScale).IsEqualTo(0.5);
			await Assert.That(landscape.Width).IsEqualTo(50);
			await Assert.That(landscape.Height).IsEqualTo(40);
			await Assert.That(portraitScale).IsEqualTo(0.5);
			await Assert.That(portrait.Width).IsEqualTo(40);
			await Assert.That(portrait.Height).IsEqualTo(50);
			await Assert.That(nonInteger.Width).IsEqualTo(100);
			await Assert.That(nonInteger.Height).IsEqualTo(67);
		}
	}

	[Test]
	public async Task Bitmap_ThumbnailNoOp()
	{
		var bitmap = new Bitmap(100, 80, asRgb: false);

		// Bound larger than both dimensions leaves the image unchanged.
		Bitmap larger = bitmap.Clone();
		double largerScale = larger.Thumbnail(maxImageSize: 200);

		// Bound equal to the largest dimension is also a no-op.
		Bitmap equal = bitmap.Clone();
		double equalScale = equal.Thumbnail(maxImageSize: 100);

		using (Assert.Multiple())
		{
			await Assert.That(largerScale).IsEqualTo(1.0);
			await Assert.That(larger.Width).IsEqualTo(100);
			await Assert.That(larger.Height).IsEqualTo(80);
			await Assert.That(equalScale).IsEqualTo(1.0);
			await Assert.That(equal.Width).IsEqualTo(100);
			await Assert.That(equal.Height).IsEqualTo(80);
		}
	}
}
