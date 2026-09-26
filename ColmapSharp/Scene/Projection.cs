// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Projection: port of colmap/scene/projection.h and projection.cc, the reprojection error
// and cheirality helpers used by triangulation, the mappers and bundle adjustment filters.
// Built on Scene/Camera.cs and Geometry/Rigid3d.cs. Tests:
// ColmapSharp.Tests/Scene/ProjectionTests.cs (projection_test.cc 1:1).
//
// Tier A for the arithmetic written here (evaluation order kept), on top of the tiers of
// Rigid3d (applying a transform is Tier B, see Rigid3d.cs) and the camera models.
// EIGEN_PI is a long double literal, so `width / (2.0 * EIGEN_PI)` is evaluated in long
// double; on the arm64 macOS build of the oracle long double is double, which is what this
// port computes (an x87 build rounds differently in the last bit).
// HasPointPositiveDepth's row(2).dot(...) walks a strided row, which Eigen does not
// vectorize, so the sum is the sequential left-to-right one.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Scene;

/// <summary>Port of the free functions of colmap/scene/projection.h.</summary>
public static class Projection
{
	/// <summary>
	/// The squared reprojection error: the squared Euclidean distance between the
	/// observation in the image and the projection of the 3D point, or double.MaxValue if
	/// the point is behind the camera. For spherical cameras, the angular error converted to
	/// pixels at the equator (see SquaredSphericalReprojectionError).
	/// </summary>
	public static double CalculateSquaredReprojectionError(Vector2d point2D, Vector3d point3D, Rigid3d camFromWorld, Camera camera)
	{
		if (camera.IsSpherical)
		{
			return SquaredSphericalReprojectionError(
				CalculateAngularReprojectionError(point2D, point3D, camFromWorld, camera), camera);
		}

		if (camera.ImgFromCam(camFromWorld * point3D) is not Vector2d projPoint2D)
		{
			return double.MaxValue;
		}

		return (projPoint2D - point2D).SquaredNorm;
	}

	/// <summary><see cref="CalculateSquaredReprojectionError(Vector2d, Vector3d, Rigid3d, Camera)"/> with a 3x4 cam_from_world matrix.</summary>
	public static double CalculateSquaredReprojectionError(Vector2d point2D, Vector3d point3D, in Matrix3x4d camFromWorld, Camera camera)
	{
		if (camera.IsSpherical)
		{
			return SquaredSphericalReprojectionError(
				CalculateAngularReprojectionError(point2D, point3D, camFromWorld, camera), camera);
		}

		if (camera.ImgFromCam(camFromWorld * point3D.Homogeneous()) is not Vector2d projPoint2D)
		{
			return double.MaxValue;
		}

		return (projPoint2D - point2D).SquaredNorm;
	}

	/// <summary>
	/// The angular reprojection error: the angle between the observed viewing ray and the
	/// ray from the camera center to the 3D point; pi if the pixel cannot be unprojected.
	/// </summary>
	public static double CalculateAngularReprojectionError(Vector2d point2D, Vector3d point3D, Rigid3d camFromWorld, Camera camera)
	{
		// Use the 3D bearing (full sphere) rather than the 2D CamFromImg, which cannot
		// represent back-hemisphere rays of omnidirectional (e.g. EQUIRECTANGULAR) cameras.
		// Identical to the legacy path for perspective cameras.
		if (camera.CamRayFromImg(point2D) is not Vector3d camRay)
		{
			return Math.PI;
		}

		return CalculateAngularReprojectionError(camRay, point3D, camFromWorld);
	}

	/// <summary><see cref="CalculateAngularReprojectionError(Vector2d, Vector3d, Rigid3d, Camera)"/> with a 3x4 cam_from_world matrix.</summary>
	public static double CalculateAngularReprojectionError(Vector2d point2D, Vector3d point3D, in Matrix3x4d camFromWorld, Camera camera)
	{
		if (camera.CamRayFromImg(point2D) is not Vector3d camRay)
		{
			return Math.PI;
		}

		return CalculateAngularReprojectionError(camRay, point3D, camFromWorld);
	}

	/// <summary>The angle between a (unit) camera ray and the ray to the 3D point.</summary>
	public static double CalculateAngularReprojectionError(Vector3d camRay, Vector3d point3D, Rigid3d camFromWorld)
	{
		Vector3d point3DInCam = camFromWorld * point3D;
		double cosAngle = camRay.Dot(point3DInCam.Normalized());
		return Math.Acos(Math.Clamp(cosAngle, -1.0, 1.0));
	}

	/// <summary><see cref="CalculateAngularReprojectionError(Vector3d, Vector3d, Rigid3d)"/> with a 3x4 cam_from_world matrix.</summary>
	public static double CalculateAngularReprojectionError(Vector3d camRay, Vector3d point3D, in Matrix3x4d camFromWorld)
	{
		Vector3d point3DInCam = camFromWorld * point3D.Homogeneous();
		double cosAngle = camRay.Dot(point3DInCam.Normalized());
		return Math.Acos(Math.Clamp(cosAngle, -1.0, 1.0));
	}

	/// <summary>
	/// Whether the 3D point passes the cheirality constraint, i.e. lies in front of the
	/// camera and not in the image plane (depth at least machine epsilon).
	/// </summary>
	public static bool HasPointPositiveDepth(in Matrix3x4d camFromWorld, Vector3d point3D)
	{
		double depth = camFromWorld[2, 0] * point3D.X
			+ camFromWorld[2, 1] * point3D.Y
			+ camFromWorld[2, 2] * point3D.Z
			+ camFromWorld[2, 3] * 1.0;
		return depth >= LinearAlgebraConstants.MachineEpsilon;
	}

	// Pixel reprojection error is ill-defined for equirectangular cameras: the azimuth is
	// discontinuous at the +-pi seam and its pixel scale diverges towards the poles (azimuth
	// pixels grow as 1/cos(elevation)). Instead measure the angular error between the
	// observed bearing and the 3D point and convert it to an equivalent pixel error at the
	// equator, consistent with the model's pixel<->angle scale used by CamFromImgThreshold.
	// This makes the (squared) error continuous across the seam and uniform over the sphere.
	private static double SquaredSphericalReprojectionError(double angularError, Camera camera)
	{
		double pixelsPerRadian = camera.Width / (2.0 * Math.PI);
		double pixelError = angularError * pixelsPerRadian;
		return pixelError * pixelError;
	}
}
