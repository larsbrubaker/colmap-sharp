// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Vector3ub: a 3-vector of bytes, the replacement for Eigen::Vector3ub (colmap/util/types.h),
// which COLMAP uses for RGB colors (Point3D.color, reconstruction file I/O). Written here;
// Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md). Only what callers use so far:
// construction, component access, Zero and value equality.

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Fixed-size 3-vector of bytes. Replacement for Eigen::Vector3ub (an RGB color in COLMAP).
/// </summary>
public readonly record struct Vector3ub(byte X, byte Y, byte Z)
{
	/// <summary>Eigen's Vector3ub::Zero().</summary>
	public static Vector3ub Zero => default;
}
