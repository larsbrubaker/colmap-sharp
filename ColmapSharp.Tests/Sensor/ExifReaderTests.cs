// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ExifReaderTests: C#-only tests (COLMAP reads EXIF through OpenImageIO and has no parser
// test) for ColmapSharp/Sensor/ExifReader.cs. Each test hand-builds a minimal JPEG whose
// APP1 segment carries a TIFF/EXIF block laid out per TIFF 6.0 and Exif 2.3, in both byte
// orders, and checks the values land where Bitmap's ported EXIF getters read them.

using System.Buffers.Binary;
using System.Text;

using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sensor;

public class ExifReaderTests
{
	private const ushort Byte = 1;
	private const ushort Ascii = 2;
	private const ushort Short = 3;
	private const ushort Long = 4;
	private const ushort Rational = 5;

	/// <summary>An IFD entry; <see cref="PointsToIfd"/> makes it a LONG pointer to that IFD.</summary>
	private sealed record Entry(ushort Tag, ushort Type, uint Count, byte[] Value, int PointsToIfd = -1);

	[Test]
	[Arguments(true)]
	[Arguments(false)]
	public async Task ReadJpeg_ExtractsCameraFocalAndGps(bool littleEndian)
	{
		var bitmap = new Bitmap(4000, 3000, asRgb: true);
		byte[] jpeg = BuildJpeg(BuildSampleTiff(littleEndian));

		using (Assert.Multiple())
		{
			await Assert.That(ExifReader.ReadJpeg(jpeg, bitmap)).IsTrue();
			await Assert.That(bitmap.GetMetaData("Make")).IsEqualTo("Canon");
			await Assert.That(bitmap.GetMetaData("Model")).IsEqualTo("Canon EOS 5D");
			await Assert.That(bitmap.ExifOrientation()).IsEqualTo(6);
			await Assert.That(bitmap.GetMetaData("Exif:FocalLength", out float focal)).IsTrue();
			await Assert.That(focal).IsEqualTo(24.5f);
			await Assert.That(bitmap.GetMetaData("Exif:FocalLengthIn35mmFilm", out int focal35)).IsTrue();
			await Assert.That(focal35).IsEqualTo(35);
			await Assert.That(bitmap.GetMetaData("Exif:FocalPlaneXResolution", out float xRes)).IsTrue();
			await Assert.That(xRes).IsEqualTo(3000f);
			await Assert.That(bitmap.GetMetaData("Exif:FocalPlaneResolutionUnit", out int unit)).IsTrue();
			await Assert.That(unit).IsEqualTo(2);
			await Assert.That(bitmap.GetMetaData("Exif:PixelXDimension", out int pixelX)).IsTrue();
			await Assert.That(pixelX).IsEqualTo(4000);
			await Assert.That(bitmap.GetMetaData("Exif:PixelYDimension", out int pixelY)).IsTrue();
			await Assert.That(pixelY).IsEqualTo(3000);

			// The getters see what OIIO would have stored: the 35 mm focal length wins.
			await Assert.That(bitmap.ExifFocalLength()).IsEqualTo(35f / 43.27 * 5000.0);
			await Assert.That(bitmap.ExifCameraModel()).IsEqualTo("Canon-Canon EOS 5D-35.000000-4000x3000");
			await Assert.That(bitmap.ExifLatitude()).IsEqualTo(-(46.0 + 30.0 / 60.0 + 900.0 / 3600.0));
			await Assert.That(bitmap.ExifLongitude()).IsEqualTo(92.75);
			await Assert.That(bitmap.ExifAltitude()).IsEqualTo(-(double)12.5f);
		}
	}

	[Test]
	public async Task ReadJpeg_WithoutExifReturnsFalse()
	{
		var bitmap = new Bitmap(2, 2, asRgb: false);
		// SOI, an APP0 (JFIF) segment, then SOS: no APP1.
		byte[] jpeg = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x07, (byte)'J', (byte)'F', (byte)'I', (byte)'F', 0, 0xFF, 0xDA, 0x00, 0x02];
		using (Assert.Multiple())
		{
			await Assert.That(ExifReader.ReadJpeg(jpeg, bitmap)).IsFalse();
			await Assert.That(ExifReader.ReadJpeg([0x89, (byte)'P', (byte)'N', (byte)'G'], bitmap)).IsFalse();
			await Assert.That(bitmap.GetMetaData("Make")).IsNull();
		}
	}

	[Test]
	public async Task ReadJpeg_TruncatedInputNeverThrows()
	{
		byte[] jpeg = BuildJpeg(BuildSampleTiff(littleEndian: true));
		for (int length = 0; length < jpeg.Length; length++)
		{
			var bitmap = new Bitmap(4, 3, asRgb: true);
			// Must not throw for any prefix; the result may be true or false.
			ExifReader.ReadJpeg(jpeg.AsSpan(0, length), bitmap);
		}

		// A TIFF block whose IFD offsets point past its end reads nothing but still parses.
		byte[] tiff = BuildSampleTiff(littleEndian: true);
		var shortBitmap = new Bitmap(4, 3, asRgb: true);
		await Assert.That(ExifReader.ReadTiff(tiff.AsSpan(0, 12), shortBitmap)).IsTrue();
		await Assert.That(shortBitmap.GetMetaData("Make")).IsNull();
	}

	private static byte[] BuildSampleTiff(bool littleEndian)
	{
		var ifd0 = new List<Entry>
		{
			AsciiEntry(0x010F, "Canon"),
			AsciiEntry(0x0110, "Canon EOS 5D"),
			new(0x0112, Short, 1, U16(6, littleEndian)),
			new(0x8769, Long, 1, [], PointsToIfd: 1),
			new(0x8825, Long, 1, [], PointsToIfd: 2),
		};
		var exif = new List<Entry>
		{
			new(0x920A, Rational, 1, Rationals(littleEndian, (49, 2))),
			new(0xA002, Long, 1, U32(4000, littleEndian)),
			new(0xA003, Short, 1, U16(3000, littleEndian)),
			new(0xA20E, Rational, 1, Rationals(littleEndian, (3000, 1))),
			new(0xA210, Short, 1, U16(2, littleEndian)),
			new(0xA405, Short, 1, U16(35, littleEndian)),
		};
		var gps = new List<Entry>
		{
			AsciiEntry(1, "S"),
			new(2, Rational, 3, Rationals(littleEndian, (46, 1), (30, 1), (900, 1))),
			AsciiEntry(3, "E"),
			new(4, Rational, 3, Rationals(littleEndian, (92, 1), (60, 2), (1800, 2))),
			new(5, Byte, 1, [1]),
			new(6, Rational, 1, Rationals(littleEndian, (25, 2))),
		};
		return BuildTiff(littleEndian, [ifd0, exif, gps]);
	}

	/// <summary>
	/// Lays out a TIFF block: 8-byte header, then each IFD followed by the values of its
	/// entries that do not fit in four bytes.
	/// </summary>
	private static byte[] BuildTiff(bool littleEndian, List<Entry>[] ifds)
	{
		var ifdOffsets = new int[ifds.Length];
		int offset = 8;
		for (int i = 0; i < ifds.Length; i++)
		{
			ifdOffsets[i] = offset;
			offset += 2 + 12 * ifds[i].Count + 4;
			foreach (var entry in ifds[i])
			{
				if (entry.Value.Length > 4)
				{
					offset += (entry.Value.Length + 1) & ~1;
				}
			}
		}

		var tiff = new byte[offset];
		tiff[0] = tiff[1] = littleEndian ? (byte)'I' : (byte)'M';
		U16(42, littleEndian).CopyTo(tiff, 2);
		U32((uint)ifdOffsets[0], littleEndian).CopyTo(tiff, 4);
		for (int i = 0; i < ifds.Length; i++)
		{
			int position = ifdOffsets[i];
			int dataPosition = position + 2 + 12 * ifds[i].Count + 4;
			U16((ushort)ifds[i].Count, littleEndian).CopyTo(tiff, position);
			position += 2;
			foreach (var entry in ifds[i])
			{
				byte[] value = entry.PointsToIfd >= 0 ? U32((uint)ifdOffsets[entry.PointsToIfd], littleEndian) : entry.Value;
				U16(entry.Tag, littleEndian).CopyTo(tiff, position);
				U16(entry.Type, littleEndian).CopyTo(tiff, position + 2);
				U32(entry.Count, littleEndian).CopyTo(tiff, position + 4);
				if (value.Length <= 4)
				{
					value.CopyTo(tiff, position + 8);
				}
				else
				{
					U32((uint)dataPosition, littleEndian).CopyTo(tiff, position + 8);
					value.CopyTo(tiff, dataPosition);
					dataPosition += (value.Length + 1) & ~1;
				}
				position += 12;
			}
			// Next-IFD offset stays 0: no chained IFD (thumbnail).
		}
		return tiff;
	}

	private static byte[] BuildJpeg(byte[] tiff)
	{
		var jpeg = new List<byte> { 0xFF, 0xD8 };
		// An APP0 segment first, as most cameras write, so the reader has to skip it.
		jpeg.AddRange([0xFF, 0xE0, 0x00, 0x07, (byte)'J', (byte)'F', (byte)'I', (byte)'F', 0]);
		byte[] payload = [.. "Exif\0\0"u8.ToArray(), .. tiff];
		jpeg.AddRange([0xFF, 0xE1]);
		jpeg.AddRange(U16((ushort)(payload.Length + 2), littleEndian: false));
		jpeg.AddRange(payload);
		jpeg.AddRange([0xFF, 0xD9]);
		return [.. jpeg];
	}

	private static Entry AsciiEntry(ushort tag, string text)
	{
		byte[] value = [.. Encoding.ASCII.GetBytes(text), 0];
		return new Entry(tag, Ascii, (uint)value.Length, value);
	}

	private static byte[] Rationals(bool littleEndian, params (uint Numerator, uint Denominator)[] values)
	{
		var bytes = new byte[8 * values.Length];
		for (int i = 0; i < values.Length; i++)
		{
			U32(values[i].Numerator, littleEndian).CopyTo(bytes, 8 * i);
			U32(values[i].Denominator, littleEndian).CopyTo(bytes, 8 * i + 4);
		}
		return bytes;
	}

	private static byte[] U16(ushort value, bool littleEndian)
	{
		var bytes = new byte[2];
		if (littleEndian)
		{
			BinaryPrimitives.WriteUInt16LittleEndian(bytes, value);
		}
		else
		{
			BinaryPrimitives.WriteUInt16BigEndian(bytes, value);
		}
		return bytes;
	}

	private static byte[] U32(uint value, bool littleEndian)
	{
		var bytes = new byte[4];
		if (littleEndian)
		{
			BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
		}
		else
		{
			BinaryPrimitives.WriteUInt32BigEndian(bytes, value);
		}
		return bytes;
	}
}
