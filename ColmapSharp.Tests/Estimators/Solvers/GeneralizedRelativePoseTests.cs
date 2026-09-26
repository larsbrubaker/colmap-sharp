// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GeneralizedRelativePoseTests: colmap/estimators/solvers/generalized_relative_pose_test.cc
// ported 1:1. The TEST_P suite ParameterizedGRNPEstimatorTests (GR6P, GR8P) is instantiated
// over the same seven (num_cams1, num_cams2, panoramic1, panoramic2) tuples as
// INSTANTIATE_TEST_SUITE_P(GRNPEstimatorTests, ...), with the same checks and tolerances.
// Tests ColmapSharp/Estimators/Solvers/GeneralizedRelativePose.cs (and .GR8P.cs) and
// PoseLib/GenRelpose6pt.cs. Tier C: the models come out of RANSAC over Tier B solvers, and
// COLMAP's Rigid3dNear tolerances are the bar.
//
// Every test seeds the PRNG with 1 as COLMAP's does, and all random draws (problem
// generation, RANSAC sampling, GR8P's random restarts) happen synchronously before the
// first await, because the PRNG is per thread. Failed EXPECTs are collected in an
// ExpectationLog so one run lists them all, as gtest does.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class GeneralizedRelativePoseTests
{
	private sealed record GeneralizedRelativePoseProblem(
		List<GrnpObservation> Points1, List<GrnpObservation> Points2, Rigid3d Rig2FromRig1);

	private static GeneralizedRelativePoseProblem CreateGeneralizedRelativePoseProblem(
		int numPoints,
		int numCameras1,
		int numCameras2,
		bool panoramic1,
		bool panoramic2)
	{
		Rigid3d[] rigsFromWorld =
		[
			new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d().Normalized()),
			new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d().Normalized()),
		];

		Rigid3d rig2FromRig1 = rigsFromWorld[1] * rigsFromWorld[0].Inverse();

		var camsFromRig1 = new Rigid3d[numCameras1];
		for (int i = 0; i < numCameras1; ++i)
		{
			Quaterniond cam1FromRigRotation = RandomEigen.RandomEigenQuaterniond();
			camsFromRig1[i] = new Rigid3d(
				cam1FromRigRotation,
				panoramic1 ? cam1FromRigRotation * new Vector3d(1, 2, 3)
					: RandomEigen.RandomEigenVector3d().Normalized());
		}

		var camsFromRig2 = new Rigid3d[numCameras2];
		for (int i = 0; i < numCameras2; ++i)
		{
			Quaterniond cam2FromRigRotation = RandomEigen.RandomEigenQuaterniond();
			camsFromRig2[i] = new Rigid3d(
				cam2FromRigRotation,
				panoramic2 ? cam2FromRigRotation * new Vector3d(-3, -2, -1)
					: RandomEigen.RandomEigenVector3d().Normalized());
		}

		var points3D = new List<Vector3d>(numPoints);
		for (int i = 0; i < numPoints; ++i)
		{
			points3D.Add(RandomEigen.RandomEigenVector3d());
		}

		var points1 = new List<GrnpObservation>(numPoints);
		var points2 = new List<GrnpObservation>(numPoints);

		// GR6P/GR8P::Residuals score in pixel units with the tangent Sampson error, so each
		// observation carries its ray's unprojection Jacobian. A spherical camera maps every
		// bearing to a pixel, keeping the synthetic points valid in all directions (a pinhole
		// would drop the back hemisphere).
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Equirectangular, focalLength: 0.0, 1000, 500);
		for (int i = 0; i < numPoints; ++i)
		{
			int camIdx1 = i % numCameras1;
			int camIdx2 = i % numCameras2;
			Vector3d point3DInCam1 = camsFromRig1[camIdx1] * (rigsFromWorld[0] * points3D[i]);
			Vector3d point3DInCam2 = camsFromRig2[camIdx2] * (rigsFromWorld[1] * points3D[i]);
			if (point3DInCam1.Norm < 1e-8 || point3DInCam2.Norm < 1e-8)
			{
				continue;
			}

			CamRayWithJac ray1WithJac = camera.CamRayFromImgWithJac(camera.ImgFromCam(point3DInCam1)!.Value)!.Value;
			CamRayWithJac ray2WithJac = camera.CamRayFromImgWithJac(camera.ImgFromCam(point3DInCam2)!.Value)!.Value;

			points1.Add(new GrnpObservation(camsFromRig1[camIdx1], ray1WithJac));
			points2.Add(new GrnpObservation(camsFromRig2[camIdx2], ray2WithJac));
		}

		return new GeneralizedRelativePoseProblem(points1, points2, rig2FromRig1);
	}

	[Test]
	[Arguments(1, 2, false, false)]
	[Arguments(2, 1, false, false)]
	[Arguments(2, 2, false, false)]
	[Arguments(3, 3, false, false)]
	[Arguments(4, 4, false, false)]
	[Arguments(4, 4, false, true)]
	[Arguments(4, 4, true, false)]
	public async Task ParameterizedGRNPEstimatorTests_GR6P(int kNumCams1, int kNumCams2, bool kPanoramic1, bool kPanoramic2)
	{
		RandomUtils.SetPRNGSeed(1);

		// Note that we can estimate the minimal problem from only 6 points but we use an
		// additional points to choose the correct solution.
		int kNumPoints = GR6PEstimator.MinNumSamples + 1;
		const int kNumTrials = 10;

		var log = new ExpectationLog();
		for (int i = 0; i < kNumTrials; ++i)
		{
			GeneralizedRelativePoseProblem problem = CreateGeneralizedRelativePoseProblem(
				kNumPoints, kNumCams1, kNumCams2, kPanoramic1, kPanoramic2);

			var options = new RansacOptions();
			options.MaxError = 1.0; // pixels
			var ransac = new Ransac<GR6PEstimator, GrnpObservation, GrnpObservation, Rigid3d>(options, new GR6PEstimator());
			var report = ransac.Estimate(problem.Points1.ToArray(), problem.Points2.ToArray());

			log.True(report.Success, $"trial {i}: report.success");
			log.True(
				Rigid3dNear(report.Model, problem.Rig2FromRig1, rtol: 1e-4, ttol: 1e-4),
				$"trial {i}: model {report.Model} near {problem.Rig2FromRig1}");

			var residuals = new double[problem.Points1.Count];
			new GR6PEstimator().Residuals(problem.Points1.ToArray(), problem.Points2.ToArray(), report.Model, residuals);

			// Residuals are squared pixels. The RANSAC inlier bound is max_error^2.
			for (int k = 0; k < residuals.Length; ++k)
			{
				log.True(residuals[k] <= options.MaxError * options.MaxError, $"trial {i}: residual {k} = {residuals[k]}");
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	[Arguments(1, 2, false, false)]
	[Arguments(2, 1, false, false)]
	[Arguments(2, 2, false, false)]
	[Arguments(3, 3, false, false)]
	[Arguments(4, 4, false, false)]
	[Arguments(4, 4, false, true)]
	[Arguments(4, 4, true, false)]
	public async Task ParameterizedGRNPEstimatorTests_GR8P(int kNumCams1, int kNumCams2, bool kPanoramic1, bool kPanoramic2)
	{
		RandomUtils.SetPRNGSeed(1);

		// Note that we can estimate the minimal problem from only 8 points but we use the
		// additional points to choose the correct solution. The GR8P estimator is numerically
		// much more sensitive than the GR6P estimator, so we only expect one successful
		// estimation in all trials.
		int kNumPoints = 2 * GR8PEstimator.MinNumSamples;
		const int kNumTrials = 10;

		bool success = false;
		for (int i = 0; i < kNumTrials; ++i)
		{
			GeneralizedRelativePoseProblem problem = CreateGeneralizedRelativePoseProblem(
				kNumPoints, kNumCams1, kNumCams2, kPanoramic1, kPanoramic2);

			var options = new RansacOptions();
			options.MaxNumTrials = 1000;
			options.MaxError = 5.0; // pixels
			var ransac = new Ransac<GR8PEstimator, GrnpObservation, GrnpObservation, Rigid3d>(options, new GR8PEstimator());
			var report = ransac.Estimate(problem.Points1.ToArray(), problem.Points2.ToArray());

			if (!report.Success)
			{
				continue;
			}

			if (!Rigid3dNear(report.Model, problem.Rig2FromRig1, rtol: 1e-2, ttol: 1e-2))
			{
				continue;
			}

			// COLMAP computes the residuals here but its `continue` inside the inner loop
			// never skips the trial, so they do not affect the outcome; the call is kept.
			var residuals = new double[problem.Points1.Count];
			new GR8PEstimator().Residuals(problem.Points1.ToArray(), problem.Points2.ToArray(), report.Model, residuals);

			success = true;
			break;
		}

		await Assert.That(success).IsTrue();
	}
}
