// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AbsolutePose: colmap/estimators/solvers/absolute_pose.h and .cc, except EPNPEstimator,
// which is in Epnp.cs next to this file. The minimal absolute-pose estimators P3PEstimator
// (PoseLib's P3P, PoseLib/P3p.cs) and P4PFEstimator (PoseLib's P4Pf, PoseLib/P4pf.cs), the
// Point2DWithRay input type and ComputeSquaredReprojectionError, which P3P and EPnP score
// with. Run by RANSAC (Optim/Ransac.cs) in the pose estimators of Phase 6.
// Tests: ColmapSharp.Tests/Estimators/Solvers/AbsolutePoseTests.cs (absolute_pose_test.cc).
//
// Tier B (minimal solvers); the RANSAC runs of the tests are Tier C.
//
// Translation notes:
// - ImgFromCamFunc, a std::function returning std::optional<Eigen::Vector2d>, becomes a
//   delegate returning Vector2d?.
// - P4PFEstimator::M_t is the nested struct P4PFModel.

using ColmapSharp.Estimators.Solvers.PoseLib;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>
/// Function mapping 3D point in the camera frame to 2D point in the image.
/// Returns null if the point projection is invalid (e.g., behind the camera).
/// </summary>
public delegate Vector2d? ImgFromCamFunc(Vector3d point3DInCam);

/// <summary>A 2D image observation together with its camera ray. Port of colmap::Point2DWithRay.</summary>
public struct Point2DWithRay
{
	/// <summary>The 2D image point in pixels.</summary>
	public Vector2d ImagePoint;

	/// <summary>The normalized 3D ray direction in the camera frame.</summary>
	public Vector3d CameraRay;

	/// <summary>An observation from its image point and camera ray.</summary>
	public Point2DWithRay(Vector2d imagePoint, Vector3d cameraRay)
	{
		ImagePoint = imagePoint;
		CameraRay = cameraRay;
	}
}

/// <summary>
/// P3P estimator: the pose cam_from_world (3x4) from three 2D-3D correspondences. Port of
/// colmap::P3PEstimator.
/// </summary>
public readonly struct P3PEstimator : IEstimator<Point2DWithRay, Vector3d, Matrix3x4d>
{
	private readonly ImgFromCamFunc _imgFromCamFunc;

	/// <summary>An estimator scoring in pixels through <paramref name="imgFromCamFunc"/>.</summary>
	public P3PEstimator(ImgFromCamFunc imgFromCamFunc)
	{
		_imgFromCamFunc = imgFromCamFunc;
	}

	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 3;

	/// <summary>
	/// Estimate the most probable solution of the P3P problem from a set of three 2D-3D point
	/// correspondences.
	/// </summary>
	public void Estimate(ReadOnlySpan<Point2DWithRay> points2D, ReadOnlySpan<Vector3d> points3D, List<Matrix3x4d> camsFromWorld)
	{
		Check.Eq(points2D.Length, 3);
		Check.Eq(points3D.Length, 3);

		camsFromWorld.Clear();

		Span<Vector3d> rays = stackalloc Vector3d[3];
		for (int i = 0; i < 3; ++i)
		{
			rays[i] = points2D[i].CameraRay;
		}

		var poses = new List<CameraPose>(4);
		int numPoses = P3p.Solve(rays, points3D, poses);

		for (int i = 0; i < numPoses; ++i)
		{
			camsFromWorld.Add(poses[i].Rt());
		}
	}

	/// <summary>
	/// Calculate the squared reprojection error given a set of 2D-3D point correspondences
	/// and a projection matrix.
	/// </summary>
	public void Residuals(ReadOnlySpan<Point2DWithRay> points2D, ReadOnlySpan<Vector3d> points3D, in Matrix3x4d camFromWorld, Span<double> residuals)
	{
		AbsolutePose.ComputeSquaredReprojectionError(points2D, points3D, camFromWorld, _imgFromCamFunc, residuals);
	}
}

/// <summary>The P4PF model: pose and focal lengths. Port of colmap::P4PFEstimator::M_t.</summary>
public struct P4PFModel
{
	/// <summary>The transformation from the world to the camera frame.</summary>
	public Matrix3x4d CamFromWorld;

	/// <summary>
	/// The focal lengths (fx, fy) of the camera. Equal when the focal length is shared
	/// (e.g. single-focal camera models).
	/// </summary>
	public Vector2d FocalLengths;
}

/// <summary>
/// Minimal solver for 6-DOF pose and focal length. The 2D points are expected to be
/// normalized by the principal point. Port of colmap::P4PFEstimator.
/// </summary>
public readonly struct P4PFEstimator : IEstimator<Vector2d, Vector3d, P4PFModel>
{
	private readonly bool _shareFocalLength;

	/// <summary>The default estimator, with a shared focal length.</summary>
	public P4PFEstimator()
		: this(true)
	{
	}

	/// <summary>
	/// If shareFocalLength is true, a single shared focal length is estimated (suitable for
	/// single-focal camera models, e.g. SIMPLE_PINHOLE). Otherwise, separate focal lengths
	/// for x and y are estimated (e.g. PINHOLE, OPENCV).
	/// </summary>
	public P4PFEstimator(bool shareFocalLength)
	{
		_shareFocalLength = shareFocalLength;
	}

	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 4;

	/// <inheritdoc/>
	public void Estimate(ReadOnlySpan<Vector2d> points2D, ReadOnlySpan<Vector3d> points3D, List<P4PFModel> models)
	{
		Check.Eq(points2D.Length, 4);
		Check.Eq(points3D.Length, 4);

		models.Clear();

		var poses = new List<CameraPose>(8);
		var focalsX = new List<double>(8);
		List<double> focalsY;
		int numPoses;
		if (_shareFocalLength)
		{
			// Estimate a single shared focal length. filter_solutions additionally
			// removes solutions whose aspect ratio (fx/fy) is far from 1.
			numPoses = P4pf.Solve(points2D, points3D, poses, focalsX, filterSolutions: true);
			focalsY = focalsX;
		}
		else
		{
			// Estimate separate focal lengths for x and y. filter_solutions only
			// removes solutions with non-positive focal lengths.
			focalsY = new List<double>(8);
			numPoses = P4pf.Solve(points2D, points3D, poses, focalsX, focalsY, filterSolutions: true);
		}

		for (int i = 0; i < numPoses; ++i)
		{
			models.Add(new P4PFModel
			{
				CamFromWorld = poses[i].Rt(),
				FocalLengths = new Vector2d(focalsX[i], focalsY[i]),
			});
		}
	}

	/// <inheritdoc/>
	public void Residuals(ReadOnlySpan<Vector2d> points2D, ReadOnlySpan<Vector3d> points3D, in P4PFModel model, Span<double> residuals)
	{
		int numPoints2D = points2D.Length;
		Check.Eq(numPoints2D, points3D.Length);
		for (int i = 0; i < numPoints2D; ++i)
		{
			Vector3d point3DInCam = model.CamFromWorld * points3D[i].Homogeneous();
			// Check if 3D point is in front of camera.
			if (point3DInCam.Z > LinearAlgebraConstants.MachineEpsilon)
			{
				residuals[i] = (model.FocalLengths.CwiseProduct(point3DInCam.HNormalized()) - points2D[i]).SquaredNorm;
			}
			else
			{
				residuals[i] = double.MaxValue;
			}
		}
	}
}

/// <summary>Free functions of colmap/estimators/solvers/absolute_pose.h.</summary>
public static class AbsolutePose
{
	/// <summary>Compute squared reprojection error in pixels.</summary>
	public static void ComputeSquaredReprojectionError(
		ReadOnlySpan<Point2DWithRay> points2D, ReadOnlySpan<Vector3d> points3D, in Matrix3x4d camFromWorld,
		ImgFromCamFunc imgFromCamFunc, Span<double> residuals)
	{
		int numPoints = points2D.Length;
		Check.Eq(numPoints, points3D.Length);
		for (int i = 0; i < numPoints; ++i)
		{
			Vector3d point3DInCam = camFromWorld * points3D[i].Homogeneous();
			Vector2d? projImagePoint = imgFromCamFunc(point3DInCam);
			if (projImagePoint is Vector2d p)
			{
				residuals[i] = (p - points2D[i].ImagePoint).SquaredNorm;
			}
			else
			{
				residuals[i] = double.MaxValue;
			}
		}
	}
}
