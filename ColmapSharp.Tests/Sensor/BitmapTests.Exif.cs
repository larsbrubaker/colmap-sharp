// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BitmapTests.Exif: the second half of the TEST(Bitmap, *) cases of bitmap_test.cc - rotation,
// cloning, metadata and the EXIF getters. See BitmapTests.cs for the file's conventions and
// the cases that are not ported.
//
// COLMAP's metadata API takes an OIIO type name ("int", "float", "point"); here the type is
// the C# overload. SetGetMetaData's probes with type "int8" (a type the value does not have)
// become probes with int, which is likewise not the stored type.

using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sensor;

public partial class BitmapTests
{
	[Test]
	public async Task Bitmap_Rot90()
	{
		var bitmap = new Bitmap(10, 5, asRgb: false);
		bitmap.SetPixel(0, 0, new BitmapColor<byte>(255));
		bitmap.SetPixel(9, 0, new BitmapColor<byte>(128));
		bitmap.SetPixel(9, 4, new BitmapColor<byte>(64));

		Bitmap rotated1 = bitmap.Clone();
		rotated1.Rot90(1); // 90 CCW
		Bitmap rotated2 = bitmap.Clone();
		rotated2.Rot90(2); // 180 CCW
		Bitmap rotated3 = bitmap.Clone();
		rotated3.Rot90(3); // 270 CCW

		using (Assert.Multiple())
		{
			await Assert.That(rotated1.Width).IsEqualTo(5);
			await Assert.That(rotated1.Height).IsEqualTo(10);
			// Top-left (0,0) -> Bottom-left (0,9)
			await Assert.That(rotated1.GetPixel(0, 9)!.Value.R).IsEqualTo((byte)255);
			// Top-right (9,0) -> Top-left (0,0)
			await Assert.That(rotated1.GetPixel(0, 0)!.Value.R).IsEqualTo((byte)128);
			// Bottom-right (9,4) -> Top-right (4,0)
			await Assert.That(rotated1.GetPixel(4, 0)!.Value.R).IsEqualTo((byte)64);

			await Assert.That(rotated2.Width).IsEqualTo(10);
			await Assert.That(rotated2.Height).IsEqualTo(5);
			await Assert.That(rotated2.GetPixel(9, 4)!.Value.R).IsEqualTo((byte)255);

			await Assert.That(rotated3.Width).IsEqualTo(5);
			await Assert.That(rotated3.Height).IsEqualTo(10);
			// Top-left (0,0) -> Top-right (4,0)
			await Assert.That(rotated3.GetPixel(4, 0)!.Value.R).IsEqualTo((byte)255);
		}
	}

	[Test]
	public async Task Bitmap_Rot90Empty()
	{
		var bitmap = new Bitmap();
		bitmap.Rot90(1);
		using (Assert.Multiple())
		{
			await Assert.That(bitmap.IsEmpty).IsTrue();
			await Assert.That(bitmap.Width).IsEqualTo(0);
			await Assert.That(bitmap.Height).IsEqualTo(0);
		}
	}

	[Test]
	public async Task Bitmap_Rot90NegativeK()
	{
		var bitmap = new Bitmap(10, 5, asRgb: false);
		bitmap.Fill(new BitmapColor<byte>(0));
		bitmap.SetPixel(0, 0, new BitmapColor<byte>(255));
		bitmap.SetPixel(9, 0, new BitmapColor<byte>(128));

		// Rot90(-1) should be equivalent to Rot90(3) (270 CCW = 90 CW).
		Bitmap rotatedNeg = bitmap.Clone();
		rotatedNeg.Rot90(-1);

		Bitmap rotated3 = bitmap.Clone();
		rotated3.Rot90(3);

		using (Assert.Multiple())
		{
			await Assert.That(rotatedNeg.Width).IsEqualTo(rotated3.Width);
			await Assert.That(rotatedNeg.Height).IsEqualTo(rotated3.Height);
			await Assert.That(rotatedNeg.RowMajorData.SequenceEqual(rotated3.RowMajorData)).IsTrue();
		}
	}

	[Test]
	public async Task Bitmap_Rot90NoOp()
	{
		var bitmap = new Bitmap(10, 5, asRgb: false);
		bitmap.Fill(new BitmapColor<byte>(0));
		bitmap.SetPixel(0, 0, new BitmapColor<byte>(255));
		byte[] originalData = (byte[])bitmap.RowMajorData.Clone();

		Bitmap rotated0 = bitmap.Clone();
		rotated0.Rot90(0);
		Bitmap rotated4 = bitmap.Clone();
		rotated4.Rot90(4);

		using (Assert.Multiple())
		{
			await Assert.That(rotated0.Width).IsEqualTo(10);
			await Assert.That(rotated0.Height).IsEqualTo(5);
			await Assert.That(rotated0.RowMajorData.SequenceEqual(originalData)).IsTrue();
			await Assert.That(rotated4.Width).IsEqualTo(10);
			await Assert.That(rotated4.Height).IsEqualTo(5);
			await Assert.That(rotated4.RowMajorData.SequenceEqual(originalData)).IsTrue();
		}
	}

	[Test]
	public async Task Bitmap_Clone()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);
		bitmap.Fill(new BitmapColor<byte>(0, 0, 0));
		bitmap.SetPixel(0, 0, new BitmapColor<byte>(10, 20, 30));
		Bitmap clonedBitmap = bitmap.Clone();
		using (Assert.Multiple())
		{
			await Assert.That(clonedBitmap.Width).IsEqualTo(100);
			await Assert.That(clonedBitmap.Height).IsEqualTo(80);
			await Assert.That(clonedBitmap.Channels).IsEqualTo(3);
			await Assert.That(clonedBitmap.GetPixel(0, 0)!.Value == new BitmapColor<byte>(10, 20, 30)).IsTrue();
		}
	}

	[Test]
	public async Task Bitmap_CloneAsRGB()
	{
		var bitmap = new Bitmap(100, 80, asRgb: false);
		bitmap.Fill(new BitmapColor<byte>(0, 0, 0));
		bitmap.SetPixel(0, 0, new BitmapColor<byte>(10, 0, 0));
		Bitmap clonedBitmap = bitmap.CloneAsRGB();
		using (Assert.Multiple())
		{
			await Assert.That(clonedBitmap.Width).IsEqualTo(100);
			await Assert.That(clonedBitmap.Height).IsEqualTo(80);
			await Assert.That(clonedBitmap.Channels).IsEqualTo(3);
			await Assert.That(clonedBitmap.GetPixel(0, 0)!.Value == new BitmapColor<byte>(10, 10, 10)).IsTrue();
		}
		// The PNG write/read round trip that follows in bitmap_test.cc is not ported (file I/O).
	}

	[Test]
	public async Task Bitmap_CloneAsRGBFromRGB()
	{
		var bitmap = new Bitmap(4, 3, asRgb: true);
		bitmap.Fill(new BitmapColor<byte>(0, 0, 0));
		bitmap.SetPixel(0, 0, new BitmapColor<byte>(10, 20, 30));
		bitmap.SetPixel(1, 1, new BitmapColor<byte>(40, 50, 60));

		Bitmap cloned = bitmap.CloneAsRGB();
		using (Assert.Multiple())
		{
			await Assert.That(cloned.Width).IsEqualTo(4);
			await Assert.That(cloned.Height).IsEqualTo(3);
			await Assert.That(cloned.Channels).IsEqualTo(3);
			await Assert.That(cloned.IsRGB).IsTrue();
			await Assert.That(cloned.RowMajorData.SequenceEqual(bitmap.RowMajorData)).IsTrue();
		}
	}

	[Test]
	public async Task Bitmap_CloneAsGrey()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);
		bitmap.Fill(new BitmapColor<byte>(0, 0, 0));
		bitmap.SetPixel(0, 0, new BitmapColor<byte>(10, 20, 30));
		bitmap.SetPixel(1, 0, new BitmapColor<byte>(1, 0, 4));
		bitmap.SetPixel(2, 0, new BitmapColor<byte>(0, 0, 6));
		bitmap.SetPixel(3, 0, new BitmapColor<byte>(255, 255, 255));
		Bitmap clonedBitmap = bitmap.CloneAsGrey();
		using (Assert.Multiple())
		{
			await Assert.That(clonedBitmap.Width).IsEqualTo(100);
			await Assert.That(clonedBitmap.Height).IsEqualTo(80);
			await Assert.That(clonedBitmap.Channels).IsEqualTo(1);
			await Assert.That(clonedBitmap.GetPixel(0, 0)!.Value == new BitmapColor<byte>(19, 19, 19)).IsTrue();
			await Assert.That(clonedBitmap.GetPixel(1, 0)!.Value == new BitmapColor<byte>(1, 1, 1)).IsTrue();
			await Assert.That(clonedBitmap.GetPixel(2, 0)!.Value == new BitmapColor<byte>(0, 0, 0)).IsTrue();
			await Assert.That(clonedBitmap.GetPixel(3, 0)!.Value == new BitmapColor<byte>(255, 255, 255)).IsTrue();
		}
		// The PNG write/read round trip that follows in bitmap_test.cc is not ported (file I/O).
	}

	[Test]
	public async Task Bitmap_CloneAsGreyFromGrey()
	{
		var bitmap = new Bitmap(4, 3, asRgb: false);
		bitmap.Fill(new BitmapColor<byte>(0));
		bitmap.SetPixel(0, 0, new BitmapColor<byte>(42));
		bitmap.SetPixel(1, 1, new BitmapColor<byte>(100));

		Bitmap cloned = bitmap.CloneAsGrey();
		using (Assert.Multiple())
		{
			await Assert.That(cloned.Width).IsEqualTo(4);
			await Assert.That(cloned.Height).IsEqualTo(3);
			await Assert.That(cloned.Channels).IsEqualTo(1);
			await Assert.That(cloned.IsGrey).IsTrue();
			await Assert.That(cloned.RowMajorData.SequenceEqual(bitmap.RowMajorData)).IsTrue();
		}
	}

	[Test]
	public async Task Bitmap_SetGetMetaData()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);
		const float kValue = 1f;
		bitmap.SetMetaData("foobar", kValue);
		using (Assert.Multiple())
		{
			await Assert.That(bitmap.GetMetaData("foobar", out float value)).IsTrue();
			await Assert.That(value).IsEqualTo(kValue);
			await Assert.That(bitmap.GetMetaData("does_not_exist", out float _)).IsFalse();
			// Adaptation: COLMAP probes GetMetaData("foobar", "int8", ...). C# selects the type
			// by overload and there is no int8 one, so this probes int, which is likewise not
			// the stored type (float).
			await Assert.That(bitmap.GetMetaData("foobar", out int _)).IsFalse();
			bitmap.SetMetaData("foobar_str", "string");
			await Assert.That(bitmap.GetMetaData("foobar_str")).IsEqualTo("string");
			// Adaptation: "int8" in COLMAP, int here (see above).
			await Assert.That(bitmap.GetMetaData("foobar_str", out int _)).IsFalse();
			await Assert.That(bitmap.GetMetaData("foobar_str", out float _)).IsFalse();
			await Assert.That(bitmap.GetMetaData("does_not_exist")).IsNull();
		}
	}

	/// <summary>
	/// C#-only (no bitmap_test.cc case): metadata names are case-insensitive, as OIIO's
	/// ImageSpec::getattribute is by default (casesensitive = false), and setting a name in
	/// another case replaces the value.
	/// </summary>
	[Test]
	public async Task MetaDataNamesAreCaseInsensitive()
	{
		var bitmap = new Bitmap(10, 8, asRgb: true);
		bitmap.SetMetaData("Exif:FocalLength", 24f);
		using (Assert.Multiple())
		{
			await Assert.That(bitmap.GetMetaData("exif:focallength", out float value)).IsTrue();
			await Assert.That(value).IsEqualTo(24f);
			bitmap.SetMetaData("EXIF:FOCALLENGTH", 35f);
			await Assert.That(bitmap.GetMetaData("Exif:FocalLength", out value)).IsTrue();
			await Assert.That(value).IsEqualTo(35f);
			bitmap.SetMetaData("make", "Canon");
			await Assert.That(bitmap.GetMetaData("Make")).IsEqualTo("Canon");
		}
	}

	[Test]
	public async Task Bitmap_CloneMetaData()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);
		const float kValue = 1f;
		bitmap.SetMetaData("foobar", kValue);

		var bitmap2 = new Bitmap(100, 80, asRgb: true);
		await Assert.That(bitmap2.GetMetaData("foobar", out float _)).IsFalse();
		bitmap.CloneMetadata(bitmap2);
		await Assert.That(bitmap2.GetMetaData("foobar", out float value)).IsTrue();
		await Assert.That(value).IsEqualTo(kValue);
	}

	[Test]
	public async Task Bitmap_ExifOrientation()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);

		await Assert.That(bitmap.ExifOrientation().HasValue).IsFalse();

		int orientation = 6;
		bitmap.SetMetaData("Orientation", orientation);

		int? exifOrientation = bitmap.ExifOrientation();
		await Assert.That(exifOrientation.HasValue).IsTrue();
		await Assert.That(exifOrientation!.Value).IsEqualTo(6);
	}

	[Test]
	public async Task Bitmap_ExifCameraModel()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);

		await Assert.That(bitmap.ExifCameraModel()).IsNull();

		bitmap.SetMetaData("Make", "make");
		bitmap.SetMetaData("Model", "model");
		const float focalLengthIn35mmFilm = 50f;
		bitmap.SetMetaData("Exif:FocalLengthIn35mmFilm", focalLengthIn35mmFilm);

		string? cameraModel = bitmap.ExifCameraModel();
		await Assert.That(cameraModel).IsNotNull();
		await Assert.That(cameraModel).IsEqualTo("make-model-50.000000-100x80");
	}

	[Test]
	public async Task Bitmap_ExifCameraModelNoModel()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);
		bitmap.SetMetaData("Make", "make");
		// Do not set Model.
		const float focalLengthIn35mmFilm = 50f;
		bitmap.SetMetaData("Exif:FocalLengthIn35mmFilm", focalLengthIn35mmFilm);

		await Assert.That(bitmap.ExifCameraModel()).IsNull();
	}

	[Test]
	public async Task Bitmap_ExifCameraModelNoFocalLength()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);
		bitmap.SetMetaData("Make", "make");
		bitmap.SetMetaData("Model", "model");
		// Do not set any focal length metadata.

		await Assert.That(bitmap.ExifCameraModel()).IsNull();
	}

	[Test]
	public async Task Bitmap_ExifFocalLengthIn35mm()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);

		await Assert.That(bitmap.ExifFocalLength().HasValue).IsFalse();

		const float focalLengthIn35mmFilm = 70f;
		bitmap.SetMetaData("Exif:FocalLengthIn35mmFilm", focalLengthIn35mmFilm);

		double? focalLength = bitmap.ExifFocalLength();
		await Assert.That(focalLength.HasValue).IsTrue();
		await Assert.That(Math.Abs(focalLength!.Value - 207.17)).IsLessThanOrEqualTo(0.1);
	}

	[Test]
	public async Task Bitmap_ExifFocalLengthWithPlane()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);

		await Assert.That(bitmap.ExifFocalLength().HasValue).IsFalse();

		const float kFocalLengthVal = 72f;
		bitmap.SetMetaData("Exif:FocalLength", kFocalLengthVal);
		bitmap.SetMetaData("Make", "canon");
		bitmap.SetMetaData("Model", "eos1dsmarkiii");

		double? focalLength = bitmap.ExifFocalLength();
		await Assert.That(focalLength.HasValue).IsTrue();
		await Assert.That(focalLength!.Value).IsEqualTo(200);
	}

	[Test]
	public async Task Bitmap_ExifFocalLengthWithDatabaseLookup()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);

		await Assert.That(bitmap.ExifFocalLength().HasValue).IsFalse();

		const float kFocalLengthVal = 120f;
		bitmap.SetMetaData("Exif:FocalLength", kFocalLengthVal);
		const int kPixelXDim = 100;
		bitmap.SetMetaData("Exif:PixelXDimension", kPixelXDim);
		const float kPlaneXRes = 1f;
		bitmap.SetMetaData("Exif:FocalPlaneXResolution", kPlaneXRes);
		const int kPlanResUnit = 4;
		bitmap.SetMetaData("Exif:FocalPlaneResolutionUnit", kPlanResUnit);

		double? focalLength = bitmap.ExifFocalLength();
		await Assert.That(focalLength.HasValue).IsTrue();
		await Assert.That(focalLength!.Value).IsEqualTo(120);
	}

	[Test]
	public async Task Bitmap_ExifFocalLengthUnits()
	{
		// Initialize a dummy bitmap
		var bitmap = new Bitmap(100, 80, asRgb: true);

		// Set the base focal length and resolution values
		const float focalLengthMm = 50.0f;
		const float focalXRes = 100.0f;
		bitmap.SetMetaData("Exif:FocalLength", focalLengthMm);
		bitmap.SetMetaData("Exif:FocalPlaneXResolution", focalXRes);

		// Case 2: Inches (25.4 mm per inch)
		bitmap.SetMetaData("Exif:FocalPlaneResolutionUnit", 2);
		await Assert.That(Math.Abs(bitmap.ExifFocalLength()!.Value - 50.0 * (100.0 / 25.4))).IsLessThanOrEqualTo(1e-4);

		// Case 3: Centimeters (10 mm per cm)
		bitmap.SetMetaData("Exif:FocalPlaneResolutionUnit", 3);
		await Assert.That(Math.Abs(bitmap.ExifFocalLength()!.Value - 50.0 * (100.0 / 10.0))).IsLessThanOrEqualTo(1e-4);

		// Case 4: Millimeters (1 mm per mm)
		bitmap.SetMetaData("Exif:FocalPlaneResolutionUnit", 4);
		await Assert.That(Math.Abs(bitmap.ExifFocalLength()!.Value - 50.0 * (100.0 * 1.0))).IsLessThanOrEqualTo(1e-4);

		// Case 5: Micrometers (1000 um per mm)
		bitmap.SetMetaData("Exif:FocalPlaneResolutionUnit", 5);
		await Assert.That(Math.Abs(bitmap.ExifFocalLength()!.Value - 50.0 * (100.0 * 1000.0))).IsLessThanOrEqualTo(1e-4);
	}

	[Test]
	public async Task Bitmap_ExifLatitude()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);

		await Assert.That(bitmap.ExifLatitude().HasValue).IsFalse();

		bitmap.SetMetaData("GPS:LatitudeRef", "N");
		float[] kDegMinSec = [46, 30, 900];
		bitmap.SetMetaData("GPS:Latitude", kDegMinSec);

		double? latitude = bitmap.ExifLatitude();
		await Assert.That(latitude.HasValue).IsTrue();
		await Assert.That(latitude!.Value).IsEqualTo(46.75);

		bitmap.SetMetaData("GPS:LatitudeRef", "S");

		latitude = bitmap.ExifLatitude();
		await Assert.That(latitude.HasValue).IsTrue();
		await Assert.That(latitude!.Value).IsEqualTo(-46.75);
	}

	[Test]
	public async Task Bitmap_ExifLongitude()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);

		await Assert.That(bitmap.ExifLongitude().HasValue).IsFalse();

		bitmap.SetMetaData("GPS:LongitudeRef", "W");
		float[] kDegMinSec = [92, 30, 900];
		bitmap.SetMetaData("GPS:Longitude", kDegMinSec);

		double? longitude = bitmap.ExifLongitude();
		await Assert.That(longitude.HasValue).IsTrue();
		await Assert.That(longitude!.Value).IsEqualTo(-92.75);

		bitmap.SetMetaData("GPS:LongitudeRef", "E");

		longitude = bitmap.ExifLongitude();
		await Assert.That(longitude.HasValue).IsTrue();
		await Assert.That(longitude!.Value).IsEqualTo(92.75);
	}

	[Test]
	public async Task Bitmap_ExifAltitude()
	{
		var bitmap = new Bitmap(100, 80, asRgb: true);

		await Assert.That(bitmap.ExifAltitude().HasValue).IsFalse();

		bitmap.SetMetaData("GPS:AltitudeRef", "0");
		const float kAltitudeVal = 123.456f;
		bitmap.SetMetaData("GPS:Altitude", kAltitudeVal);

		double? altitude = bitmap.ExifAltitude();
		await Assert.That(altitude.HasValue).IsTrue();
		await Assert.That(altitude!.Value).IsEqualTo((double)kAltitudeVal);

		bitmap.SetMetaData("GPS:AltitudeRef", "1");

		altitude = bitmap.ExifAltitude();
		await Assert.That(altitude.HasValue).IsTrue();
		await Assert.That(altitude!.Value).IsEqualTo(-(double)kAltitudeVal);
	}
}
