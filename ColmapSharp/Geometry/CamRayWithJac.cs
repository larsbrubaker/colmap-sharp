// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CamRayWithJac: the struct of that name from colmap/geometry/pose.h (the rest of pose.h is
// a Phase 2 port). Scene/Camera.cs's CamRayFromImgWithJac returns it; the tangent Sampson
// estimators (Phase 6) take a list of them. Tested through
// ColmapSharp.Tests/Scene/CameraTests.cs (camera_test.cc CamRayFromImgWithJac).

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Geometry;

/// <summary>
/// A unit bearing and the Jacobian d(ray)/d(pixel) of its unprojection (see
/// Camera.CamRayFromImgWithJac), bundled so RANSAC subsampling keeps them index-aligned.
/// Port of colmap::CamRayWithJac.
/// </summary>
/// <param name="Ray">Unit bearing vector in the camera frame.</param>
/// <param name="Jacobian">d(u, v, w) / d(x, y) of the bearing with respect to the pixel.</param>
public readonly record struct CamRayWithJac(Vector3d Ray, Matrix3x2d Jacobian)
{
	/// <summary>
	/// Fallback when unprojection fails. The estimators take a dense list of CamRayWithJac,
	/// not optionals, so a failed ray is kept as zero, which the tangent Sampson residual
	/// scores as infinite (rejected).
	/// </summary>
	public static CamRayWithJac Zero => new(Vector3d.Zero, default);
}
