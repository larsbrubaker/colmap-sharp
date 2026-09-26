// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GeneralizedAbsolutePose: colmap/estimators/solvers/generalized_absolute_pose.h and .cc -
// GP3PEstimator, the minimal absolute pose of a generalized (multi-camera) camera, i.e. the
// rig_from_world pose, from three 2D-3D correspondences observed by cameras at known poses in
// the rig. It solves with PoseLib's gp3p (PoseLib/Gp3p.cs), falling back to PoseLib's p3p
// (PoseLib/P3p.cs) when all three rays share one origin (a panoramic rig), where gp3p's
// translation elimination is degenerate. Run by RANSAC (Optim/Ransac.cs) in the generalized
// absolute pose estimation of estimators/pose.cc (a later Phase 6 step).
// Tests: ColmapSharp.Tests/Estimators/Solvers/GeneralizedAbsolutePoseTests.cs
// (generalized_absolute_pose_test.cc 1:1).
//
// Tier B for the solver; the RANSAC runs of the tests are Tier C.
//
// Translation notes:
// - GP3PEstimator::X_t is the top-level struct GP3PObservation, as P4PFEstimator::M_t is
//   P4PFModel in AbsolutePose.cs; ResidualType stays nested, as in C++.
// - LOG(FATAL_THROW) for an invalid residual type becomes InvalidOperationException.
// - Residuals resizes with `resize(n, 0)` in C++, which keeps stale values in a reused
//   vector, but every slot is written below, so the span contract loses nothing.

using ColmapSharp.Estimators.Solvers.PoseLib;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>
/// A generalized image observation: the relative pose of a camera in the generalized camera
/// and a ray in the camera frame. Port of colmap::GP3PEstimator::X_t.
/// </summary>
public struct GP3PObservation
{
	/// <summary>The pose of the observing camera in the rig, stored as a matrix for fast residuals.</summary>
	public Matrix3x4d CamFromRig;

	/// <summary>The ray in the camera frame.</summary>
	public Vector3d RayInCam;

	/// <summary>An observation from its camera's rig pose and its ray.</summary>
	public GP3PObservation(Matrix3x4d camFromRig, Vector3d rayInCam)
	{
		CamFromRig = camFromRig;
		RayInCam = rayInCam;
	}
}

/// <summary>
/// Minimal generalized absolute pose estimator: the rig_from_world pose of a generalized
/// camera from three 2D-3D correspondences. Port of colmap::GP3PEstimator.
/// </summary>
public readonly struct GP3PEstimator : IEstimator<GP3PObservation, Vector3d, Rigid3d>
{
	/// <summary>
	/// Whether to compute the cosine similarity or the reprojection error.
	/// [WARNING] The reprojection error being in normalized coordinates, the unique error
	/// threshold of RANSAC corresponds to different pixel values in the different cameras of
	/// the rig if they have different intrinsics.
	/// </summary>
	public enum ResidualType
	{
		/// <summary>The squared cosine distance between the observed and projected rays.</summary>
		CosineDistance,

		/// <summary>The squared reprojection error in normalized image coordinates.</summary>
		ReprojectionError,
	}

	private readonly ResidualType _residualType;

	/// <summary>An estimator scoring with the cosine distance (COLMAP's default).</summary>
	public GP3PEstimator()
		: this(ResidualType.CosineDistance)
	{
	}

	/// <summary>An estimator scoring with <paramref name="residualType"/>.</summary>
	public GP3PEstimator(ResidualType residualType)
	{
		_residualType = residualType;
	}

	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 3;

	/// <summary>
	/// Estimate the most probable solution of the GP3P problem from a set of three 2D-3D
	/// point correspondences.
	/// </summary>
	public void Estimate(ReadOnlySpan<GP3PObservation> points2D, ReadOnlySpan<Vector3d> points3D, List<Rigid3d> rigsFromWorld)
	{
		Check.Eq(points2D.Length, 3);
		Check.Eq(points3D.Length, 3);

		rigsFromWorld.Clear();

		Span<Vector3d> raysInRig = stackalloc Vector3d[3];
		Span<Vector3d> originsInRig = stackalloc Vector3d[3];
		for (int i = 0; i < 3; ++i)
		{
			Matrix3d rigFromCamRotation = points2D[i].CamFromRig.LeftCols3().Transpose();
			raysInRig[i] = (rigFromCamRotation * points2D[i].RayInCam).Normalized();
			originsInRig[i] = rigFromCamRotation * -points2D[i].CamFromRig.Col(3);
		}

		var poses = new List<CameraPose>(8);
		if (originsInRig[0].IsApprox(originsInRig[1], 1e-6) &&
			originsInRig[0].IsApprox(originsInRig[2], 1e-6))
		{
			// In case of a panoramic camera/rig, fall back to P3P.
			P3p.Solve(raysInRig, points3D, poses);
			for (int i = 0; i < poses.Count; ++i)
			{
				poses[i] = new CameraPose(poses[i].Q, poses[i].T + originsInRig[0]);
			}
		}
		else
		{
			Gp3p.Solve(originsInRig, raysInRig, points3D, poses);
		}

		foreach (CameraPose pose in poses)
		{
			rigsFromWorld.Add(PoseLibUtils.ConvertPoseLibPoseToRigid3d(pose));
		}
	}

	/// <summary>
	/// Calculate the squared cosine distance error between the rays (or the squared
	/// reprojection error, per the residual type) given a set of 2D-3D point correspondences
	/// and the rig pose of the generalized camera. Points behind their camera get
	/// double.MaxValue.
	/// </summary>
	public void Residuals(ReadOnlySpan<GP3PObservation> points2D, ReadOnlySpan<Vector3d> points3D, in Rigid3d rigFromWorld, Span<double> residuals)
	{
		Check.Eq(points2D.Length, points3D.Length);

		// Precompute matrix to avoid repeated quaternion-to-matrix conversion.
		Matrix3x4d rigFromWorldMatrix = rigFromWorld.ToMatrix();

		for (int i = 0; i < points2D.Length; ++i)
		{
			Vector3d point3DInCam = points2D[i].CamFromRig * (rigFromWorldMatrix * points3D[i].Homogeneous()).Homogeneous();

			// Check if 3D point is in front of camera.
			if (point3DInCam.Z > LinearAlgebraConstants.MachineEpsilon)
			{
				Vector3d ray = points2D[i].RayInCam;

				if (_residualType == ResidualType.CosineDistance)
				{
					double cosineDist = 1 - point3DInCam.Normalized().Dot(ray.Normalized());
					residuals[i] = cosineDist * cosineDist;
				}
				else if (_residualType == ResidualType.ReprojectionError)
				{
					Vector2d diff = ray.HNormalized() - point3DInCam.HNormalized();
					residuals[i] = diff.SquaredNorm;
				}
				else
				{
					throw new InvalidOperationException("Invalid residual type");
				}
			}
			else
			{
				residuals[i] = double.MaxValue;
			}
		}
	}
}
