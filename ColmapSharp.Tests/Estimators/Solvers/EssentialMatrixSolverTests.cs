// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// EssentialMatrixSolverTests: colmap/estimators/solvers/essential_matrix_test.cc ported 1:1
// (the Geometry/EssentialMatrixTests.cs file is geometry/essential_matrix_test.cc). Test
// names are Suite_Name; parameterized suites take the gtest values as [Arguments]. Same
// problem generator, loop counts and tolerances. Tests
// ColmapSharp/Estimators/Solvers/EssentialMatrixEstimators.cs and PoseLib/Relpose5pt.cs.
// Tier B for the solvers, Tier C for the refiner and LO-RANSAC.

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

namespace ColmapSharp.Tests.Estimators.Solvers;

public class EssentialMatrixSolverTests
{
	// Rejection thresholds used by RandomEpipolarCorrespondences to condition a minimal
	// sample: minimum depth in front of either camera, and minimum parallax between the two
	// rays as sin^2 of their angle (1 - a^2 in the cheirality form).
	private const double kMinDepth = 0.2;
	private const double kMinParallax = 1e-2;  // ~5.7 degrees.

	[Test]
	[Arguments(5)]
	[Arguments(20)]
	[Arguments(1000)]
	public async Task EssentialMatrixFivePointEstimatorTests_Nominal(int kNumRays)
	{
		// The minimal case has no redundancy, so it conditions its sample to stay well-posed
		// and accepts the solver's numerical accuracy with a looser tolerance.
		bool isMinimal = kNumRays == EssentialMatrixFivePointEstimator.MinNumSamples;
		var log = new ExpectationLog();
		for (int k = 0; k < 100; ++k)
		{
			Rigid3d cam2FromCam1 = TestCam2FromCam1();
			Matrix3d expectedE = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
			var rays1 = new List<Vector3d>();
			var rays2 = new List<Vector3d>();
			RandomEpipolarCorrespondences(cam2FromCam1, kNumRays, rejectDegenerate: isMinimal, rays1, rays2);

			var estimator = new EssentialMatrixFivePointEstimator();
			var models = new List<Matrix3d>();
			estimator.Estimate(rays1.ToArray(), rays2.ToArray(), models);

			ExpectAtLeastOneValidModel(log, $"k={k}", rays1, rays2, expectedE, models, eEps: isMinimal ? 5e-3 : 1e-4);
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	[Arguments(8)]
	[Arguments(64)]
	[Arguments(1024)]
	public async Task EssentialMatrixEightPointEstimatorTests_Nominal(int kNumRays)
	{
		var log = new ExpectationLog();
		for (int k = 0; k < 1; ++k)
		{
			Rigid3d cam2FromCam1 = TestCam2FromCam1();
			Matrix3d expectedE = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
			var rays1 = new List<Vector3d>();
			var rays2 = new List<Vector3d>();
			RandomEpipolarCorrespondences(cam2FromCam1, kNumRays, rejectDegenerate: false, rays1, rays2);

			var estimator = new EssentialMatrixEightPointEstimator();
			var models = new List<Matrix3d>();
			estimator.Estimate(rays1.ToArray(), rays2.ToArray(), models);

			ExpectAtLeastOneValidModel(log, $"k={k}", rays1, rays2, expectedE, models);
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// Refine recovers the true pose from a perturbed initial E on exact rays.
	[Test]
	public async Task EssentialMatrixTangentSampsonEstimator_RefineRecoversPose()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Equirectangular, focalLength: 0.0, 1000, 500);
		var log = new ExpectationLog();
		for (int k = 0; k < 30; ++k)
		{
			Rigid3d cam2FromCam1 = TestCam2FromCam1();
			Matrix3d expected = Normalized(EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1));
			var rays1 = new List<Vector3d>();
			var rays2 = new List<Vector3d>();
			RandomEpipolarCorrespondences(cam2FromCam1, 50, rejectDegenerate: false, rays1, rays2);
			CamRayWithJac[] crj1 = WithJacobians(camera, rays1);
			CamRayWithJac[] crj2 = WithJacobians(camera, rays2);

			// Seed the refinement from a slightly perturbed pose. C++ evaluates the Rigid3d
			// constructor arguments left to right (clang), so the rotation's random axis is
			// drawn before the translation's perturbation.
			Quaterniond perturbation = new AngleAxisd(0.01, RandomEigen.RandomEigenVector3d().Normalized()).ToQuaternion();
			Quaterniond initRotation = cam2FromCam1.Rotation * perturbation;
			Vector3d initTranslation = (cam2FromCam1.Translation + 0.01 * RandomEigen.RandomEigenVector3d()).Normalized();
			var init = new Rigid3d(initRotation, initTranslation);
			Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(init);

			double Dist(Matrix3d m)
			{
				Matrix3d n = Normalized(m);
				return Math.Min((n - expected).Norm(), (n + expected).Norm());
			}

			double initDist = Dist(e);
			if (!log.True(EssentialMatrixTangentSampsonEstimator.Refine(crj1, crj2, ref e), $"k={k}: Refine succeeds"))
			{
				break;
			}

			log.Less(Dist(e), 1e-4, $"k={k}: refined distance");
			log.Less(Dist(e), initDist, $"k={k}: refined distance vs initial");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// The LO-RANSAC estimator recovers the pose despite 30% gross outliers.
	[Test]
	public async Task EssentialMatrixTangentSampsonEstimator_LORANSACWithOutliers()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Equirectangular, focalLength: 0.0, 1000, 500);
		var log = new ExpectationLog();
		for (int k = 0; k < 10; ++k)
		{
			Rigid3d cam2FromCam1 = TestCam2FromCam1();
			Matrix3d expected = Normalized(EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1));
			var rays1 = new List<Vector3d>();
			var rays2 = new List<Vector3d>();
			RandomEpipolarCorrespondences(cam2FromCam1, 200, rejectDegenerate: false, rays1, rays2);
			for (int i = 0; i < 60; ++i)
			{
				// 30% gross outliers.
				rays2[i] = RandomEigen.RandomEigenVector3d().Normalized();
			}

			CamRayWithJac[] crj1 = WithJacobians(camera, rays1);
			CamRayWithJac[] crj2 = WithJacobians(camera, rays2);

			var options = new RansacOptions { MaxError = 2.0 };  // pixels
			var ransac = new LoRansac<EssentialMatrixTangentSampsonEstimator, EssentialMatrixTangentSampsonEstimator,
				CamRayWithJac, CamRayWithJac, Matrix3d>(options, default, default);
			var report = ransac.Estimate(crj1, crj2);

			if (!log.True(report.Success, $"k={k}: RANSAC succeeds"))
			{
				break;
			}

			Matrix3d e = Normalized(report.Model);
			log.Less(Math.Min((e - expected).Norm(), (e + expected).Norm()), 1e-2, $"k={k}: model distance");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// Generates a random relative pose with a unit-norm baseline. Bounding the baseline away
	// from zero avoids the (near) pure-rotation degeneracy, where the essential matrix
	// vanishes and the cheirality of every correspondence becomes ill-defined.
	private static Rigid3d TestCam2FromCam1()
	{
		// Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>().normalized()): clang
		// evaluates the arguments left to right.
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d().Normalized();
		return new Rigid3d(rotation, translation);
	}

	// When rejectDegenerate is set, resamples correspondences that make a minimal 5-point
	// solve ill-conditioned (near-zero depth or near-parallel rays). Only the minimal case
	// needs this; larger samples are robust to such points.
	private static void RandomEpipolarCorrespondences(
		Rigid3d cam2FromCam1, int numRays, bool rejectDegenerate, List<Vector3d> rays1, List<Vector3d> rays2)
	{
		for (int i = 0; i < numRays; ++i)
		{
			Vector3d ray1;
			Vector3d pointInCam2;
			bool degenerate;
			do
			{
				ray1 = RandomEigen.RandomEigenVector3d().Normalized();
				double randomDepth = RandomUtils.RandomUniformReal(kMinDepth, 2.0);
				pointInCam2 = cam2FromCam1 * (randomDepth * ray1);
				Vector3d ray1InCam2 = cam2FromCam1.Rotation * ray1;
				double cosParallax = ray1InCam2.Dot(pointInCam2.Normalized());
				degenerate = pointInCam2.Norm < kMinDepth || 1.0 - cosParallax * cosParallax < kMinParallax;
			}
			while (rejectDegenerate && degenerate);
			rays1.Add(ray1);
			rays2.Add(pointInCam2.Normalized());
		}
	}

	private static void ExpectAtLeastOneValidModel(
		ExpectationLog log,
		string what,
		List<Vector3d> rays1,
		List<Vector3d> rays2,
		Matrix3d expectedE,
		List<Matrix3d> models,
		double eEps = 1e-4,
		double rEps = 1e-5)
	{
		expectedE = Normalized(expectedE);
		for (int i = 0; i < models.Count; ++i)
		{
			Matrix3d e = Normalized(models[i]);
			if (Math.Min((e - expectedE).Norm(), (e + expectedE).Norm()) > eEps)
			{
				continue;
			}

			// The five/eight-point solvers no longer expose Residuals (bearing Sampson was
			// retired). Verify the recovered model directly with the plain Sampson error,
			// which is ~0 for these noiseless, in-front correspondences.
			var residuals = new List<double>();
			EssentialMatrix.ComputeSquaredSampsonError(rays1, rays2, e, residuals);
			for (int j = 0; j < rays1.Count; ++j)
			{
				log.Less(residuals[j], rEps, $"{what}: residual[{j}]");
			}

			return;
		}

		log.Fail($"{what}: No essential matrix is equal up to scale.");
	}

	// Attaches an unprojection Jacobian to each bearing via a spherical camera, which maps
	// every direction to a valid pixel.
	private static CamRayWithJac[] WithJacobians(Camera camera, List<Vector3d> rays)
	{
		var camRaysWithJac = new CamRayWithJac[rays.Count];
		for (int i = 0; i < rays.Count; ++i)
		{
			camRaysWithJac[i] = camera.CamRayFromImgWithJac(camera.ImgFromCam(rays[i])!.Value)!.Value;
		}

		return camRaysWithJac;
	}

	// Eigen's Matrix3d::normalized() (by the Frobenius norm).
	private static Matrix3d Normalized(Matrix3d m) => m / m.Norm();
}
