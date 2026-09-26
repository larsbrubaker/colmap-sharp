// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CameraModelId: the camera model identifiers of colmap/sensor/models.h
// (MAKE_ENUM_CLASS_OVERLOAD_STREAM(CameraModelId, -1, ...)). The numeric values are part of
// COLMAP's database and reconstruction file formats, so they must never change. The models
// themselves are in the other files of this folder; CameraModels.cs dispatches on this id.

namespace ColmapSharp.Sensor;

/// <summary>
/// Unique camera model identifier; port of colmap::CameraModelId. The C++ enumerators carry
/// a <c>k</c> prefix (kSimplePinhole), dropped here.
/// </summary>
public enum CameraModelId
{
	/// <summary>No model.</summary>
	Invalid = -1,

	/// <summary>SIMPLE_PINHOLE: f, cx, cy.</summary>
	SimplePinhole = 0,

	/// <summary>PINHOLE: fx, fy, cx, cy.</summary>
	Pinhole = 1,

	/// <summary>SIMPLE_RADIAL: f, cx, cy, k.</summary>
	SimpleRadial = 2,

	/// <summary>RADIAL: f, cx, cy, k1, k2.</summary>
	Radial = 3,

	/// <summary>OPENCV: fx, fy, cx, cy, k1, k2, p1, p2.</summary>
	OpenCV = 4,

	/// <summary>OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4.</summary>
	OpenCVFisheye = 5,

	/// <summary>FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6.</summary>
	FullOpenCV = 6,

	/// <summary>FOV: fx, fy, cx, cy, omega.</summary>
	FOV = 7,

	/// <summary>SIMPLE_RADIAL_FISHEYE: f, cx, cy, k.</summary>
	SimpleRadialFisheye = 8,

	/// <summary>RADIAL_FISHEYE: f, cx, cy, k1, k2.</summary>
	RadialFisheye = 9,

	/// <summary>THIN_PRISM_FISHEYE: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1.</summary>
	ThinPrismFisheye = 10,

	/// <summary>RAD_TAN_THIN_PRISM_FISHEYE: fx, fy, cx, cy, k0..k5, p0, p1, s0..s3.</summary>
	RadTanThinPrismFisheye = 11,

	/// <summary>SIMPLE_DIVISION: f, cx, cy, k.</summary>
	SimpleDivision = 12,

	/// <summary>DIVISION: fx, fy, cx, cy, k.</summary>
	Division = 13,

	/// <summary>SIMPLE_FISHEYE: f, cx, cy.</summary>
	SimpleFisheye = 14,

	/// <summary>FISHEYE: fx, fy, cx, cy.</summary>
	Fisheye = 15,

	/// <summary>EUCM: fx, fy, cx, cy, alpha, beta.</summary>
	EUCM = 16,

	/// <summary>EQUIRECTANGULAR: w, h.</summary>
	Equirectangular = 17,
}
