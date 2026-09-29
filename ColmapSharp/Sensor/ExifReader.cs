// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ExifReader: a small managed EXIF parser, written here from the published specifications
// (JEITA CP-3451 "Exif 2.3" and Adobe "TIFF Revision 6.0"), not ported from any library.
// It replaces the EXIF half of OpenImageIO's image reading, which COLMAP's Bitmap::Read
// relies on and which is native and not ported (docs/LICENSE_AUDIT.md). The host decodes
// the pixels itself and passes the file bytes here; the values COLMAP uses land in the
// Bitmap's metadata (Bitmap.Exif.cs) under OpenImageIO's attribute names, so the ported
// EXIF getters read them unchanged.
// Tests: ColmapSharp.Tests/Sensor/ExifReaderTests.cs (C#-only; COLMAP's EXIF tests set
// metadata directly and are ported in BitmapTests.cs).
//
// Extracted (tag, EXIF type -> metadata name, type):
//   IFD0: Make, Model (ASCII -> string), Orientation (SHORT -> "Orientation", int)
//   Exif IFD: FocalLength (RATIONAL -> "Exif:FocalLength", float), FocalPlaneXResolution
//     (RATIONAL -> float), FocalPlaneResolutionUnit (SHORT -> int), FocalLengthIn35mmFilm
//     (SHORT -> int), PixelXDimension/PixelYDimension (SHORT or LONG -> int)
//   GPS IFD: GPSLatitudeRef/GPSLongitudeRef (ASCII -> "GPS:LatitudeRef", string),
//     GPSLatitude/GPSLongitude (3 RATIONAL -> "GPS:Latitude", point), GPSAltitudeRef
//     (BYTE -> int), GPSAltitude (RATIONAL -> float)
// Malformed input never throws: an entry that points outside the block, has an unexpected
// type or count, or a rational with a zero denominator is skipped.
//
// Zero denominators: EXIF allows n/0 (cameras write 0/0 for "unknown"). OIIO most likely
// stores the float quotient (inf or NaN); this reader leaves the attribute unset instead,
// so the getters report "absent" rather than an inf/NaN focal length or GPS coordinate
// (divergence 10).

using System.Buffers.Binary;
using System.Text;

namespace ColmapSharp.Sensor;

/// <summary>
/// Reads the EXIF values COLMAP uses (camera make/model, focal length, focal-plane
/// resolution, orientation, pixel dimensions, GPS position) into a <see cref="Bitmap"/>'s
/// metadata.
/// </summary>
public static class ExifReader
{
	private const ushort TagMake = 0x010F;
	private const ushort TagModel = 0x0110;
	private const ushort TagOrientation = 0x0112;
	private const ushort TagExifIfd = 0x8769;
	private const ushort TagGpsIfd = 0x8825;
	private const ushort TagFocalLength = 0x920A;
	private const ushort TagPixelXDimension = 0xA002;
	private const ushort TagPixelYDimension = 0xA003;
	private const ushort TagFocalPlaneXResolution = 0xA20E;
	private const ushort TagFocalPlaneResolutionUnit = 0xA210;
	private const ushort TagFocalLengthIn35mmFilm = 0xA405;
	private const ushort TagGpsLatitudeRef = 1;
	private const ushort TagGpsLatitude = 2;
	private const ushort TagGpsLongitudeRef = 3;
	private const ushort TagGpsLongitude = 4;
	private const ushort TagGpsAltitudeRef = 5;
	private const ushort TagGpsAltitude = 6;

	private const ushort TypeByte = 1;
	private const ushort TypeAscii = 2;
	private const ushort TypeShort = 3;
	private const ushort TypeLong = 4;
	private const ushort TypeRational = 5;

	/// <summary>
	/// Finds the EXIF APP1 segment of a JPEG file (its raw bytes) and stores its values in
	/// <paramref name="target"/>'s metadata. Returns false when the data is not a JPEG or has
	/// no readable EXIF block.
	/// </summary>
	public static bool ReadJpeg(ReadOnlySpan<byte> jpeg, Bitmap target)
	{
		// SOI marker.
		if (jpeg.Length < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8)
		{
			return false;
		}

		int position = 2;
		while (position + 4 <= jpeg.Length)
		{
			if (jpeg[position] != 0xFF)
			{
				return false;
			}
			byte marker = jpeg[position + 1];
			if (marker == 0xFF)
			{
				// Fill byte before a marker.
				position++;
				continue;
			}
			position += 2;
			if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7))
			{
				// Stand-alone markers (TEM, RSTn) carry no length.
				continue;
			}
			if (marker == 0xD9 || marker == 0xDA)
			{
				// EOI, or SOS after which entropy-coded data follows: APP1 must come before.
				return false;
			}

			// The segment length counts its own two bytes.
			int length = BinaryPrimitives.ReadUInt16BigEndian(jpeg.Slice(position, 2));
			if (length < 2 || position + length > jpeg.Length)
			{
				return false;
			}
			var payload = jpeg.Slice(position + 2, length - 2);
			if (marker == 0xE1 && payload.Length >= 6 && payload[..6].SequenceEqual("Exif\0\0"u8))
			{
				return ReadTiff(payload[6..], target);
			}
			position += length;
		}
		return false;
	}

	/// <summary>
	/// Reads a TIFF-structured EXIF block (the APP1 payload after "Exif\0\0", starting with
	/// "II*\0" or "MM\0*") into <paramref name="target"/>'s metadata. Returns false when the
	/// header is invalid.
	/// </summary>
	public static bool ReadTiff(ReadOnlySpan<byte> tiff, Bitmap target)
	{
		if (tiff.Length < 8)
		{
			return false;
		}
		bool littleEndian;
		if (tiff[0] == (byte)'I' && tiff[1] == (byte)'I')
		{
			littleEndian = true;
		}
		else if (tiff[0] == (byte)'M' && tiff[1] == (byte)'M')
		{
			littleEndian = false;
		}
		else
		{
			return false;
		}

		var reader = new TiffReader(tiff, littleEndian);
		if (reader.UInt16(2) != 42)
		{
			return false;
		}

		long ifd0 = reader.UInt32(4);
		long exifIfd = -1;
		long gpsIfd = -1;
		foreach (var entry in reader.Entries(ifd0))
		{
			switch (entry.Tag)
			{
				case TagMake:
					SetString(target, "Make", reader.Ascii(entry));
					break;
				case TagModel:
					SetString(target, "Model", reader.Ascii(entry));
					break;
				case TagOrientation:
					SetInt(target, "Orientation", reader.Integer(entry));
					break;
				case TagExifIfd:
					exifIfd = reader.Integer(entry) ?? -1;
					break;
				case TagGpsIfd:
					gpsIfd = reader.Integer(entry) ?? -1;
					break;
			}
		}

		if (exifIfd >= 0)
		{
			foreach (var entry in reader.Entries(exifIfd))
			{
				switch (entry.Tag)
				{
					case TagFocalLength:
						SetFloat(target, "Exif:FocalLength", reader.Rational(entry, 0));
						break;
					case TagFocalPlaneXResolution:
						SetFloat(target, "Exif:FocalPlaneXResolution", reader.Rational(entry, 0));
						break;
					case TagFocalPlaneResolutionUnit:
						SetInt(target, "Exif:FocalPlaneResolutionUnit", reader.Integer(entry));
						break;
					case TagFocalLengthIn35mmFilm:
						SetInt(target, "Exif:FocalLengthIn35mmFilm", reader.Integer(entry));
						break;
					case TagPixelXDimension:
						SetInt(target, "Exif:PixelXDimension", reader.Integer(entry));
						break;
					case TagPixelYDimension:
						SetInt(target, "Exif:PixelYDimension", reader.Integer(entry));
						break;
				}
			}
		}

		if (gpsIfd >= 0)
		{
			foreach (var entry in reader.Entries(gpsIfd))
			{
				switch (entry.Tag)
				{
					case TagGpsLatitudeRef:
						SetString(target, "GPS:LatitudeRef", reader.Ascii(entry));
						break;
					case TagGpsLatitude:
						SetPoint(target, "GPS:Latitude", reader, entry);
						break;
					case TagGpsLongitudeRef:
						SetString(target, "GPS:LongitudeRef", reader.Ascii(entry));
						break;
					case TagGpsLongitude:
						SetPoint(target, "GPS:Longitude", reader, entry);
						break;
					case TagGpsAltitudeRef:
						SetInt(target, "GPS:AltitudeRef", reader.Integer(entry));
						break;
					case TagGpsAltitude:
						SetFloat(target, "GPS:Altitude", reader.Rational(entry, 0));
						break;
				}
			}
		}
		return true;
	}

	private static void SetString(Bitmap target, string name, string? value)
	{
		if (value is not null)
		{
			target.SetMetaData(name, value);
		}
	}

	private static void SetInt(Bitmap target, string name, long? value)
	{
		if (value is long v && v >= int.MinValue && v <= int.MaxValue)
		{
			target.SetMetaData(name, (int)v);
		}
	}

	private static void SetFloat(Bitmap target, string name, double? value)
	{
		if (value is double v)
		{
			target.SetMetaData(name, (float)v);
		}
	}

	private static void SetPoint(Bitmap target, string name, TiffReader reader, IfdEntry entry)
	{
		if (entry.Count != 3)
		{
			return;
		}
		double? degrees = reader.Rational(entry, 0);
		double? minutes = reader.Rational(entry, 1);
		double? seconds = reader.Rational(entry, 2);
		if (degrees is double d && minutes is double m && seconds is double s)
		{
			target.SetMetaData(name, [(float)d, (float)m, (float)s]);
		}
	}

	/// <summary>One 12-byte IFD entry: tag, type, count, and where its value bytes are.</summary>
	private readonly record struct IfdEntry(ushort Tag, ushort Type, uint Count, long ValueOffset);

	/// <summary>Bounds-checked, endian-aware reads of a TIFF block.</summary>
	private readonly ref struct TiffReader
	{
		private readonly ReadOnlySpan<byte> data;
		private readonly bool littleEndian;

		public TiffReader(ReadOnlySpan<byte> data, bool littleEndian)
		{
			this.data = data;
			this.littleEndian = littleEndian;
		}

		private bool InRange(long offset, long size) => offset >= 0 && size >= 0 && offset + size <= data.Length;

		public ushort UInt16(long offset)
		{
			var bytes = data.Slice((int)offset, 2);
			return littleEndian ? BinaryPrimitives.ReadUInt16LittleEndian(bytes) : BinaryPrimitives.ReadUInt16BigEndian(bytes);
		}

		public uint UInt32(long offset)
		{
			var bytes = data.Slice((int)offset, 4);
			return littleEndian ? BinaryPrimitives.ReadUInt32LittleEndian(bytes) : BinaryPrimitives.ReadUInt32BigEndian(bytes);
		}

		/// <summary>The entries of the IFD at the given offset (empty if it is out of range).</summary>
		public List<IfdEntry> Entries(long ifdOffset)
		{
			var entries = new List<IfdEntry>();
			if (!InRange(ifdOffset, 2))
			{
				return entries;
			}
			int count = UInt16(ifdOffset);
			for (int i = 0; i < count; i++)
			{
				long entryOffset = ifdOffset + 2 + 12L * i;
				if (!InRange(entryOffset, 12))
				{
					break;
				}
				ushort type = UInt16(entryOffset + 2);
				uint valueCount = UInt32(entryOffset + 4);
				long valueSize = TypeSize(type) * (long)valueCount;
				// Values of up to four bytes are stored in the offset field itself.
				long valueOffset = valueSize <= 4 ? entryOffset + 8 : UInt32(entryOffset + 8);
				entries.Add(new IfdEntry(UInt16(entryOffset), type, valueCount, valueOffset));
			}
			return entries;
		}

		private static long TypeSize(ushort type) => type switch
		{
			TypeByte or TypeAscii or 6 or 7 => 1,
			TypeShort or 8 => 2,
			TypeLong or 9 or 11 => 4,
			TypeRational or 10 or 12 => 8,
			_ => 0,
		};

		/// <summary>An ASCII value without its NUL terminator(s), or null.</summary>
		public string? Ascii(IfdEntry entry)
		{
			if (entry.Type != TypeAscii || !InRange(entry.ValueOffset, entry.Count))
			{
				return null;
			}
			var bytes = data.Slice((int)entry.ValueOffset, (int)entry.Count);
			int end = bytes.IndexOf((byte)0);
			return Encoding.UTF8.GetString(end >= 0 ? bytes[..end] : bytes);
		}

		/// <summary>The first value of a BYTE, SHORT or LONG entry, or null.</summary>
		public long? Integer(IfdEntry entry)
		{
			if (entry.Count < 1 || !InRange(entry.ValueOffset, TypeSize(entry.Type)))
			{
				return null;
			}
			return entry.Type switch
			{
				TypeByte => data[(int)entry.ValueOffset],
				TypeShort => UInt16(entry.ValueOffset),
				TypeLong => UInt32(entry.ValueOffset),
				_ => null,
			};
		}

		/// <summary>The index-th value of a RATIONAL entry, or null (also for a zero denominator).</summary>
		public double? Rational(IfdEntry entry, int index)
		{
			long offset = entry.ValueOffset + 8L * index;
			if (entry.Type != TypeRational || index >= entry.Count || !InRange(offset, 8))
			{
				return null;
			}
			uint numerator = UInt32(offset);
			uint denominator = UInt32(offset + 4);
			if (denominator == 0)
			{
				return null;
			}
			return (double)numerator / denominator;
		}
	}
}
