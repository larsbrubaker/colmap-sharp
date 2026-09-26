// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GeneralizedRelativePose: colmap/estimators/solvers/generalized_relative_pose.h and the
// estimator half of its .cc - the relative pose rig2_from_rig1 between two generalized
// (multi-camera) cameras from ray correspondences. GR6PEstimator wraps PoseLib's minimal
// 6-point solver (PoseLib/GenRelpose6pt.cs); GR8PEstimator is Kneip and Li's 8-point
// eigenvalue minimization, whose solver lives in GeneralizedRelativePose.GR8P.cs. Both
// score with the tangent Sampson error in pixels (Geometry/EssentialMatrix.cs), which is
// why an observation carries its ray's unprojection Jacobian (Scene/Camera.cs
// CamRayFromImgWithJac). Consumers: the generalized relative pose estimation of
// estimators/generalized_pose.cc. Tests: ColmapSharp.Tests/Estimators/Solvers/
// GeneralizedRelativePoseTests.cs (generalized_relative_pose_test.cc 1:1).
//
// Tier B for the solvers (LU, QR, a 64x64 eigenvalue problem, a 4x4 eigenproblem) and
// Tier C through RANSAC.

using ColmapSharp.Estimators.Solvers.PoseLib;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>
/// One ray of a generalized camera: the pose of its camera in the rig and the bearing in the
/// camera frame, bundled with its Jacobian d(ray) / d(pixel) so Residuals can score in pixel
/// units with the tangent Sampson error. Only the residual uses the Jacobian; the solvers
/// read the ray alone. Port of colmap::GRNPObservation.
/// </summary>
/// <param name="CamFromRig">The pose of the observing camera in its rig.</param>
/// <param name="RayWithJacInCam">The bearing (and its Jacobian) in the camera frame.</param>
public readonly record struct GrnpObservation(Rigid3d CamFromRig, CamRayWithJac RayWithJacInCam);

/// <summary>
/// Minimal generalized relative pose estimator based on PoseLib's gen_relpose_6pt. The
/// model is rig2_from_rig1. Port of colmap::GR6PEstimator.
/// </summary>
public readonly struct GR6PEstimator : IEstimator<GrnpObservation, GrnpObservation, Rigid3d>
{
	/// <summary>
	/// The minimum number of samples needed to estimate a model. Note that in theory the
	/// minimum required number of samples is 6 but Laurent Kneip showed in his paper that
	/// using 8 samples is more stable.
	/// </summary>
	public static int MinNumSamples => 6;

	/// <summary>
	/// Estimate the most probable solution of the GR6P problem from a set of six 2D-2D point
	/// correspondences.
	/// </summary>
	public void Estimate(ReadOnlySpan<GrnpObservation> points1, ReadOnlySpan<GrnpObservation> points2, List<Rigid3d> rigs2FromRigs1)
	{
		Check.Eq(points1.Length, 6);
		Check.Eq(points2.Length, 6);

		rigs2FromRigs1.Clear();

		Span<Vector3d> originsInRig1 = stackalloc Vector3d[6];
		Span<Vector3d> originsInRig2 = stackalloc Vector3d[6];
		Span<Vector3d> raysInRig1 = stackalloc Vector3d[6];
		Span<Vector3d> raysInRig2 = stackalloc Vector3d[6];
		for (int i = 0; i < 6; ++i)
		{
			originsInRig1[i] = points1[i].CamFromRig.TgtOriginInSrc();
			originsInRig2[i] = points2[i].CamFromRig.TgtOriginInSrc();
			raysInRig1[i] = points1[i].CamFromRig.Rotation.Inverse() * points1[i].RayWithJacInCam.Ray;
			raysInRig2[i] = points2[i].CamFromRig.Rotation.Inverse() * points2[i].RayWithJacInCam.Ray;
		}

		GenRelpose6pt.Solve(originsInRig1, raysInRig1, originsInRig2, raysInRig2, rigs2FromRigs1);
	}

	/// <summary>
	/// Calculate the squared tangent Sampson error (in pixels) between corresponding points.
	/// </summary>
	public void Residuals(ReadOnlySpan<GrnpObservation> points1, ReadOnlySpan<GrnpObservation> points2, in Rigid3d rig2FromRig1, Span<double> residuals)
	{
		ComputeResiduals(points1, points2, rig2FromRig1, residuals);
	}

	/// <summary>The shared GR6P/GR8P residual (GR8PEstimator::Residuals forwards to GR6P's).</summary>
	internal static void ComputeResiduals(ReadOnlySpan<GrnpObservation> points1, ReadOnlySpan<GrnpObservation> points2, in Rigid3d rig2FromRig1, Span<double> residuals)
	{
		Check.Eq(points1.Length, points2.Length);

		// Score in pixel units with the tangent Sampson error. Each correspondence has its
		// own essential matrix, formed from the two cameras' rig poses, and is evaluated in
		// each camera's own frame, where that camera's ray Jacobian lives - so no rig-frame
		// rotation of the rays or Jacobians is needed.
		for (int i = 0; i < points1.Length; ++i)
		{
			Rigid3d cam2FromCam1 = points2[i].CamFromRig * rig2FromRig1 * points1[i].CamFromRig.Inverse();
			Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
			residuals[i] = EssentialMatrix.ComputeSquaredTangentSampsonError(
				points1[i].RayWithJacInCam, points2[i].RayWithJacInCam, e);
		}
	}
}

/// <summary>
/// Solver for the Generalized Relative Pose problem using a minimal of 8 2D-2D
/// correspondences. This implementation is based on: "Efficient Computation of Relative
/// Pose for Multi-Camera Systems", Kneip and Li. CVPR 2014. Note that the solution to this
/// problem is degenerate in the case of pure translation and when all correspondences are
/// observed from the same cameras. The implementation is a modified and improved version of
/// Kneip's original implementation in OpenGV licensed under the BSD license. The model is
/// rig2_from_rig1. Port of colmap::GR8PEstimator.
/// </summary>
public readonly struct GR8PEstimator
	: IEstimator<GrnpObservation, GrnpObservation, Rigid3d>, ILocalEstimator<GrnpObservation, GrnpObservation, Rigid3d>
{
	/// <summary>
	/// The minimum number of samples needed to estimate a model. Note that in theory the
	/// minimum required number of samples is 6 but Laurent Kneip showed in his paper that
	/// using 8 samples is more stable.
	/// </summary>
	public static int MinNumSamples => 8;

	/// <summary>
	/// Estimate the most probable solution of the GR8P problem from a set of eight (at least
	/// six) 2D-2D point correspondences. Always yields the four candidates of the final 4x4
	/// eigenproblem, as COLMAP does.
	/// </summary>
	public void Estimate(ReadOnlySpan<GrnpObservation> points1, ReadOnlySpan<GrnpObservation> points2, List<Rigid3d> rigs2FromRigs1)
	{
		GR8PSolver.Estimate(points1, points2, rigs2FromRigs1);
	}

	/// <summary>
	/// LO-RANSAC's local estimate (the generalized relative pose estimators use GR8P as GR6P's
	/// local optimizer): GR8PEstimator has no Refine, so this re-estimates from the inliers.
	/// </summary>
	public void EstimateLocal(ReadOnlySpan<GrnpObservation> points1, ReadOnlySpan<GrnpObservation> points2, in Rigid3d initialModel, List<Rigid3d> rigs2FromRigs1) =>
		Estimate(points1, points2, rigs2FromRigs1);

	/// <summary>
	/// Calculate the squared tangent Sampson error (in pixels) between corresponding points.
	/// </summary>
	public void Residuals(ReadOnlySpan<GrnpObservation> points1, ReadOnlySpan<GrnpObservation> points2, in Rigid3d rig2FromRig1, Span<double> residuals)
	{
		GR6PEstimator.ComputeResiduals(points1, points2, rig2FromRig1, residuals);
	}
}
