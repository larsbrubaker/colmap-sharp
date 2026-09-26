// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CameraSpecs: colmap/sensor/specs.h/.cc, the database of camera sensor widths (mm) keyed
// by make and model. CameraDatabase (CameraDatabase.cs) queries it to turn an EXIF focal
// length in mm into pixels when the EXIF has no focal-plane resolution (Bitmap.Exif.cs).
// The table itself is GENERATED from the C++ reference by scripts/generate-camera-specs.py
// into CameraSpecs.Data1..9.cs; this file holds the types and the entry point.
// COLMAP has no specs_test.cc; the table is tested through database_test.cc
// (ColmapSharp.Tests/Sensor/CameraDatabaseTests.cs).
//
// Translation note: COLMAP's camera_specs_t is a NodeHashMap (hash-ordered). Here it is a
// list in specs.cc source order, so CameraDatabase's query iterates deterministically
// (docs/CPP_DIVERGENCES.md, entry 8).

namespace ColmapSharp.Sensor;

/// <summary>One model of a make and its sensor width in mm (an entry of camera_make_specs_t).</summary>
public readonly record struct CameraModelSpec(string Model, float SensorWidthMm);

/// <summary>All models of one camera make (one key/value pair of camera_specs_t).</summary>
public sealed record CameraMakeSpecs(string Make, IReadOnlyList<CameraModelSpec> Models);

/// <summary>
/// Port of colmap::InitializeCameraSpecs (sensor/specs.cc).
/// </summary>
public static partial class CameraSpecs
{
	/// <summary>
	/// Port of colmap::InitializeCameraSpecs: every make with its models' sensor widths, in
	/// the order specs.cc lists them. Makes and models are lower case without separators,
	/// and makes are unique.
	/// </summary>
	public static IReadOnlyList<CameraMakeSpecs> InitializeCameraSpecs()
	{
		var specs = new List<CameraMakeSpecs>();
		AddMakes1(specs);
		AddMakes2(specs);
		AddMakes3(specs);
		AddMakes4(specs);
		AddMakes5(specs);
		AddMakes6(specs);
		AddMakes7(specs);
		AddMakes8(specs);
		AddMakes9(specs);
		return specs;
	}
}
