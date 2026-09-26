// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Bitmap.Exif: the metadata half of colmap/sensor/bitmap.cc - Set/GetMetaData,
// CloneMetadata, SetJpegQuality and the EXIF getters (orientation, camera model string,
// focal length in pixels, GPS latitude/longitude/altitude). Part of Bitmap (Bitmap.cs).
//
// COLMAP keeps metadata in an OpenImageIO ImageSpec's attribute list and names EXIF values
// the way OIIO's EXIF decoder does ("Make", "Exif:FocalLength", "GPS:Latitude", ...).
// BitmapMetaData below is a managed stand-in for that list with the same names and the
// value types COLMAP asks for: int, float, point (3 floats) and string. ExifReader.cs
// fills it from a JPEG's EXIF block. Type conversion on read follows the subset of OIIO's
// convert_type the getters rely on: an int reads as float (EXIF SHORTs such as
// FocalLengthIn35mmFilm are queried as float) and as its decimal string (GPS:AltitudeRef is
// a BYTE queried as "0"/"1"); every other type mismatch reads as absent.
//
// Tier A: the EXIF getters are COLMAP's arithmetic verbatim.

using System.Globalization;

using ColmapSharp.Util;

namespace ColmapSharp.Sensor;

public sealed partial class Bitmap
{
	// Null for a default-constructed bitmap, like COLMAP's meta_data_.
	private BitmapMetaData? metaData;

	/// <summary>Port of Bitmap::SetMetaData(name, "int", value).</summary>
	public void SetMetaData(string name, int value) => Check.NotNull(metaData).Set(name, value);

	/// <summary>Port of Bitmap::SetMetaData(name, "float", value).</summary>
	public void SetMetaData(string name, float value) => Check.NotNull(metaData).Set(name, value);

	/// <summary>
	/// Port of Bitmap::SetMetaData(name, "point", value): three floats, e.g. a GPS
	/// coordinate as degrees, minutes and seconds.
	/// </summary>
	public void SetMetaData(string name, float[] point)
	{
		Check.Eq(point.Length, 3);
		Check.NotNull(metaData).Set(name, (float[])point.Clone());
	}

	/// <summary>Port of Bitmap::SetMetaData(name, value) for a string value.</summary>
	public void SetMetaData(string name, string value) => Check.NotNull(metaData).Set(name, value);

	/// <summary>Port of Bitmap::GetMetaData(name, "int", &amp;value).</summary>
	public bool GetMetaData(string name, out int value)
	{
		if (Check.NotNull(metaData).TryGet(name, out object? stored) && stored is int intValue)
		{
			value = intValue;
			return true;
		}
		value = 0;
		return false;
	}

	/// <summary>
	/// Port of Bitmap::GetMetaData(name, "float", &amp;value). An int attribute converts,
	/// as OIIO's getattribute does.
	/// </summary>
	public bool GetMetaData(string name, out float value)
	{
		Check.NotNull(metaData).TryGet(name, out object? stored);
		switch (stored)
		{
			case float floatValue:
				value = floatValue;
				return true;
			case int intValue:
				value = intValue;
				return true;
			default:
				value = 0;
				return false;
		}
	}

	/// <summary>Port of Bitmap::GetMetaData(name, "point", &amp;value): three floats.</summary>
	public bool GetMetaData(string name, out float[] point)
	{
		if (Check.NotNull(metaData).TryGet(name, out object? stored) && stored is float[] stored3)
		{
			point = (float[])stored3.Clone();
			return true;
		}
		point = [0, 0, 0];
		return false;
	}

	/// <summary>
	/// Port of Bitmap::GetMetaData(name): the string value, or an int attribute in decimal
	/// (OIIO's string conversion); null if absent.
	/// </summary>
	public string? GetMetaData(string name)
	{
		Check.NotNull(metaData).TryGet(name, out object? stored);
		return stored switch
		{
			string text => text,
			int intValue => intValue.ToString(CultureInfo.InvariantCulture),
			_ => null,
		};
	}

	/// <summary>Port of Bitmap::CloneMetadata: copy this bitmap's metadata to the target.</summary>
	public void CloneMetadata(Bitmap target)
	{
		Check.NotNull(target);
		target.metaData = Check.NotNull(metaData).Clone();
	}

	/// <summary>
	/// Port of Bitmap::SetJpegQuality: records the JPEG compression quality in [1, 100] as
	/// "Compression" metadata, for the host's encoder.
	/// </summary>
	public void SetJpegQuality(int quality)
	{
		Check.Gt(quality, 0);
		Check.Le(quality, 100);
		SetMetaData("Compression", "jpeg:" + quality.ToString(CultureInfo.InvariantCulture));
	}

	/// <summary>Port of Bitmap::ExifOrientation: the EXIF orientation tag (1..8), if present.</summary>
	public int? ExifOrientation()
	{
		if (GetMetaData("Orientation", out int orientation))
		{
			return orientation;
		}
		return null;
	}

	/// <summary>
	/// Port of Bitmap::ExifCameraModel: "make-model-focal-WxH", which groups images that can
	/// share intrinsics. Null unless make, model and a focal length are present.
	/// </summary>
	public string? ExifCameraModel()
	{
		// Read camera make and model
		string? make = GetMetaData("Make");
		if (make is null)
		{
			return null;
		}
		string? model = GetMetaData("Model");
		if (model is null)
		{
			return null;
		}
		if (!GetMetaData("Exif:FocalLengthIn35mmFilm", out float focalLength) &&
			!GetMetaData("Exif:FocalLength", out focalLength))
		{
			return null;
		}
		// StringPrintf("%s-%s-%.6f-%dx%d", ...).
		return string.Create(CultureInfo.InvariantCulture,
			$"{make}-{model}-{((double)focalLength).ToString("F6", CultureInfo.InvariantCulture)}-{width}x{height}");
	}

	/// <summary>
	/// Port of Bitmap::ExifFocalLength: the focal length in pixels, from the 35 mm equivalent
	/// focal length, else from the focal length in mm and the focal-plane resolution, else
	/// from the focal length in mm and the sensor width of the camera database.
	/// </summary>
	public double? ExifFocalLength()
	{
		double maxSize = Math.Max(width, height);

		if (GetMetaData("Exif:FocalLengthIn35mmFilm", out float focalLength35mm))
		{
			if (focalLength35mm > 0)
			{
				// Based on https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length
				// According to CIPA guidelines, 35 mm equivalent focal length is to be
				// calculated like this:
				// "focal length in 35 mm camera" =
				//   (Diagonal distance of image area in the 35 mm camera (43.27 mm) /
				//    Diagonal distance of image area on the image sensor of the DSC)
				//    * focal length of the lens of the DSC.
				// C++ computes width_ * width_ + height_ * height_ in int.
				double diagonal = Math.Sqrt(unchecked(width * width + height * height));
				return focalLength35mm / 43.27 * diagonal;
			}
		}

		if (GetMetaData("Exif:FocalLength", out float focalLengthMm))
		{
			if (GetMetaData("Exif:FocalPlaneXResolution", out float focalXRes) &&
				GetMetaData("Exif:FocalPlaneResolutionUnit", out int focalXResUnit))
			{
				if (focalLengthMm > 0 && focalXResUnit > 1 && focalXResUnit <= 5)
				{
					double pixelsPerMm = focalXResUnit switch
					{
						2 => focalXRes / 25.4, // inches
						3 => focalXRes / 10.0, // cm
						4 => focalXRes * 1.0, // mm
						5 => focalXRes * 1000.0, // um
						_ => throw new InvalidOperationException("Unexpected FocalPlaneXResolution value"),
					};
					return focalLengthMm * pixelsPerMm;
				}
			}

			// Lookup sensor width in database.
			string? make = GetMetaData("Make");
			string? model = GetMetaData("Model");
			if (make is not null && model is not null)
			{
				var database = new CameraDatabase();
				double sensorWidthMm = 0;
				if (database.QuerySensorWidth(make, model, ref sensorWidthMm))
				{
					return focalLengthMm / sensorWidthMm * maxSize;
				}
			}
		}

		return null;
	}

	/// <summary>Port of Bitmap::ExifLatitude: decimal degrees, negative for "S".</summary>
	public double? ExifLatitude()
	{
		string? latitudeRef = GetMetaData("GPS:LatitudeRef");
		double sign = 1.0;
		if (latitudeRef is not null)
		{
			if (latitudeRef == "N" || latitudeRef == "n")
			{
				sign = 1.0;
			}
			else if (latitudeRef == "S" || latitudeRef == "s")
			{
				sign = -1.0;
			}
		}
		if (GetMetaData("GPS:Latitude", out float[] degMinSec))
		{
			double latitude = degMinSec[0] + degMinSec[1] / 60.0 + degMinSec[2] / 3600.0;
			if (latitude > 0 && sign < 0)
			{
				latitude *= sign;
			}
			return latitude;
		}
		return null;
	}

	/// <summary>Port of Bitmap::ExifLongitude: decimal degrees, negative for "W".</summary>
	public double? ExifLongitude()
	{
		string? longitudeRef = GetMetaData("GPS:LongitudeRef");
		double sign = 1.0;
		if (longitudeRef is not null)
		{
			if (longitudeRef == "W" || longitudeRef == "w")
			{
				sign = -1.0;
			}
			else if (longitudeRef == "E" || longitudeRef == "e")
			{
				sign = 1.0;
			}
		}
		if (GetMetaData("GPS:Longitude", out float[] degMinSec))
		{
			double longitude = degMinSec[0] + degMinSec[1] / 60.0 + degMinSec[2] / 3600.0;
			if (longitude > 0 && sign < 0)
			{
				longitude *= sign;
			}
			return longitude;
		}
		return null;
	}

	/// <summary>Port of Bitmap::ExifAltitude: meters, negative below sea level (ref "1").</summary>
	public double? ExifAltitude()
	{
		string? altitudeRef = GetMetaData("GPS:AltitudeRef");
		double sign = 1.0;
		if (altitudeRef is not null)
		{
			if (altitudeRef == "0")
			{
				sign = 1.0;
			}
			else if (altitudeRef == "1")
			{
				sign = -1.0;
			}
		}
		if (GetMetaData("GPS:Altitude", out float altitudeFloat))
		{
			double altitude = altitudeFloat;
			if (altitude > 0 && sign < 0)
			{
				altitude *= sign;
			}
			return altitude;
		}
		return null;
	}
}

/// <summary>
/// Managed stand-in for the OIIO ImageSpec attribute list behind COLMAP's Bitmap metadata:
/// named values of type int, float, float[3] (point) or string. Names are case-insensitive,
/// like OIIO's getattribute by default (casesensitive = false). Setting a name replaces its
/// previous value and type, as ImageSpec::attribute does.
/// </summary>
internal sealed class BitmapMetaData
{
	private readonly Dictionary<string, object> values = new(StringComparer.OrdinalIgnoreCase);

	public void Set(string name, object value) => values[name] = value;

	public bool TryGet(string name, out object? value) => values.TryGetValue(name, out value);

	public BitmapMetaData Clone()
	{
		var cloned = new BitmapMetaData();
		foreach (var (name, value) in values)
		{
			cloned.values[name] = value is float[] point ? (float[])point.Clone() : value;
		}
		return cloned;
	}
}
