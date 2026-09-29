// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CameraDatabase: colmap/sensor/database.h/.cc, the fuzzy make/model lookup over the
// sensor-width table of CameraSpecs.cs. Bitmap.ExifFocalLength (Bitmap.Exif.cs) falls back
// to it when the EXIF has a focal length in mm but no focal-plane resolution.
// Tests: ColmapSharp.Tests/Sensor/CameraDatabaseTests.cs (database_test.cc 1:1).
//
// Tier A: same strings in, same answer out, except when the cleaned make matches more than
// one make of the table; then COLMAP's answer depends on its hash map's iteration order and
// ours on specs.cc source order (divergence 8).

namespace ColmapSharp.Sensor;

/// <summary>
/// Port of colmap::CameraDatabase: sensor widths for many cameras, used to derive the focal
/// length in pixels when EXIF information is incomplete.
/// </summary>
public sealed class CameraDatabase
{
	private static readonly IReadOnlyList<CameraMakeSpecs> Specs = CameraSpecs.InitializeCameraSpecs();

	/// <summary>Number of camera makes in the database.</summary>
	public int NumEntries => Specs.Count;

	/// <summary>
	/// Port of CameraDatabase::QuerySensorWidth. Returns true with the sensor width when the
	/// model matches exactly, or exactly one model matches as a substring (either way round).
	/// <paramref name="sensorWidthMm"/> is written on every match, as in COLMAP, so after an
	/// ambiguous query it holds a width even though the result is false.
	/// </summary>
	public bool QuerySensorWidth(string make, string model, ref double sensorWidthMm)
	{
		// Clean the strings from all separators.
		string cleanedMake = ToLowerAscii(make.Replace(" ", "").Replace("-", ""));
		string cleanedModel = ToLowerAscii(model.Replace(" ", "").Replace("-", ""));

		// Make sure that make name is not duplicated. COLMAP's StringReplace is a no-op for
		// an empty search string, while string.Replace throws on one.
		if (cleanedMake.Length > 0)
		{
			cleanedModel = cleanedModel.Replace(cleanedMake, "", StringComparison.Ordinal);
		}

		// Check if cleanedMake exists in database: Test whether EXIF string is substring of
		// database entry and vice versa.
		int specMatches = 0;
		foreach (var makeSpecs in Specs)
		{
			if (cleanedMake.Contains(makeSpecs.Make, StringComparison.Ordinal) ||
				makeSpecs.Make.Contains(cleanedMake, StringComparison.Ordinal))
			{
				foreach (var (modelName, sensorWidth) in makeSpecs.Models)
				{
					if (cleanedModel.Contains(modelName, StringComparison.Ordinal) ||
						modelName.Contains(cleanedModel, StringComparison.Ordinal))
					{
						sensorWidthMm = sensorWidth;
						if (cleanedModel == modelName)
						{
							// Model exactly matches, return immediately.
							return true;
						}
						specMatches += 1;
						if (specMatches > 1)
						{
							break;
						}
					}
				}
			}
		}

		// Only return unique results, if model does not exactly match.
		return specMatches == 1;
	}

	/// <summary>
	/// COLMAP's StringToLower, std::transform with ::tolower in the C locale: only ASCII
	/// A-Z change. ToLowerInvariant would also lower non-ASCII letters.
	/// </summary>
	private static string ToLowerAscii(string value)
	{
		return string.Create(value.Length, value, static (span, source) =>
		{
			for (int i = 0; i < source.Length; i++)
			{
				char c = source[i];
				span[i] = c is >= 'A' and <= 'Z' ? (char)(c + ('a' - 'A')) : c;
			}
		});
	}
}
