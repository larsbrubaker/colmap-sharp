// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FundamentalMatrixDegensacTests: colmap/estimators/fundamental_matrix_degensac_test.cc
// ported 1:1. TEST(Suite, Name) becomes Suite_Name; same scene generator, loop counts and
// tolerances. Tests ColmapSharp/Estimators/FundamentalMatrixDegensac.cs and
// FundamentalMatrixDegensacEstimator.cs. Tier B for the epipole and the compatible
// homography, Tier C for the degeneracy test and the robust estimates.
//
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does,
// and each test draws everything before its first await (the PRNG is per thread). Draw
// order follows clang: constructor and comma-initializer arguments are evaluated left to
// right.

using ColmapSharp.Estimators;
using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public class FundamentalMatrixDegensacTests
{
	[Test]
	public async Task EpipoleFromFundamentalMatrix_Nominal()
	{
		var log = new ExpectationLog();
		for (int k = 0; k < 20; ++k)
		{
			Matrix3d kMat = RandomCalibrationMatrix();
			var cam2FromCam1 = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());
			Matrix3d f = EssentialMatrix.FundamentalFromEssentialMatrix(
				kMat, EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), kMat);
			Vector3d epipole2 = FundamentalMatrixDegensac.EpipoleFromFundamentalMatrix(f);
			log.Near(epipole2.Norm, 1.0, 1e-9, $"k={k}: epipole norm");
			log.Less((f.Transpose() * epipole2).Norm, 1e-6, $"k={k}: F^T e2");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task HomographyFromFundamentalAndPoints_Nominal()
	{
		Matrix3d kMat = RandomCalibrationMatrix();
		var cam2FromCam1 = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());
		Matrix3d f = EssentialMatrix.FundamentalFromEssentialMatrix(
			kMat, EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), kMat);
		Vector3d epipole2 = FundamentalMatrixDegensac.EpipoleFromFundamentalMatrix(f);

		Vector3d normal = new Vector3d(0.2, -0.1, 1.0).Normalized();
		const double kDistance = 2.0;
		GenerateDominantPlaneScene(
			cam2FromCam1, kMat, normal, kDistance,
			numPoints: 6, numOnPlane: 6, noise: 0.0,
			out Vector2d[] points1, out Vector2d[] points2, out _);

		Matrix3d? h = FundamentalMatrixDegensac.HomographyFromFundamentalAndPoints(
			f, epipole2, points1.AsSpan(0, 3), points2.AsSpan(0, 3));
		await Assert.That(h.HasValue).IsTrue();

		// All on-plane correspondences must be explained by the recovered homography.
		var errors = new double[points1.Length];
		for (int i = 0; i < points1.Length; ++i)
		{
			errors[i] = HomographyMatrix.ComputeSquaredHomographyError(points1[i], points2[i], h!.Value);
		}

		using (Assert.Multiple())
		{
			foreach (double error in errors)
			{
				await Assert.That(error).IsLessThan(1e-6);
			}
		}
	}

	[Test]
	public async Task HomographyFromFundamentalAndPoints_CollinearReturnsNullopt()
	{
		Matrix3d kMat = RandomCalibrationMatrix();
		var cam2FromCam1 = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());
		Matrix3d f = EssentialMatrix.FundamentalFromEssentialMatrix(
			kMat, EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), kMat);
		Vector3d epipole2 = FundamentalMatrixDegensac.EpipoleFromFundamentalMatrix(f);

		// Three collinear first-image points make M rank-deficient.
		Vector2d[] triPoints1 = [new(100, 201), new(200, 401), new(300, 601)];
		Vector2d[] triPoints2 = [new(110, 210), new(220, 430), new(330, 650)];
		bool hasValue = FundamentalMatrixDegensac.HomographyFromFundamentalAndPoints(
			f, epipole2, triPoints1, triPoints2).HasValue;
		await Assert.That(hasValue).IsFalse();
	}

	[Test]
	public async Task IsSampleHDegenerate_DetectsAndRejects()
	{
		Matrix3d kMat = RandomCalibrationMatrix();
		var cam2FromCam1 = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());
		Matrix3d f = EssentialMatrix.FundamentalFromEssentialMatrix(
			kMat, EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), kMat);
		Vector3d normal = new Vector3d(0.2, -0.1, 1.0).Normalized();

		// A sample with 5 of 7 correspondences on the plane is H-degenerate.
		GenerateDominantPlaneScene(
			cam2FromCam1, kMat, normal, distance: 2.0,
			numPoints: 7, numOnPlane: 5, noise: 0.0,
			out Vector2d[] planarPoints1, out Vector2d[] planarPoints2, out _);
		bool planarDegenerate = FundamentalMatrixDegensac.IsSampleHDegenerate(
			f, planarPoints1, planarPoints2, hMaxResidual: 4.0, minSampleHInlierRatio: 5.0 / 7.0);

		// A general (non-planar) sample is not H-degenerate.
		GenerateDominantPlaneScene(
			cam2FromCam1, kMat, normal, distance: 2.0,
			numPoints: 7, numOnPlane: 0, noise: 0.0,
			out Vector2d[] generalPoints1, out Vector2d[] generalPoints2, out _);
		bool generalDegenerate = FundamentalMatrixDegensac.IsSampleHDegenerate(
			f, generalPoints1, generalPoints2, hMaxResidual: 4.0, minSampleHInlierRatio: 5.0 / 7.0);

		using (Assert.Multiple())
		{
			await Assert.That(planarDegenerate).IsTrue();
			await Assert.That(generalDegenerate).IsFalse();
		}
	}

	// On a general non-planar scene, DEGENSAC recovers the fundamental matrix as well as the
	// plain LO-RANSAC estimator.
	[Test]
	public async Task FundamentalMatrixDegensac_NonPlanarParity()
	{
		Matrix3d kMat = RandomCalibrationMatrix();
		var cam2FromCam1 = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());
		Matrix3d expectedF = EssentialMatrix.FundamentalFromEssentialMatrix(
			kMat, EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), kMat);
		expectedF /= expectedF[2, 2];

		Vector3d normal = new Vector3d(0.2, -0.1, 1.0).Normalized();
		GenerateDominantPlaneScene(
			cam2FromCam1, kMat, normal, distance: 2.0,
			numPoints: 200, numOnPlane: 0, noise: 0.0,
			out Vector2d[] points1, out Vector2d[] points2, out _);

		var options = new FundamentalMatrixDegensacOptions { Ransac = TestRansacOptions() };
		var report = FundamentalMatrixDegensac.EstimateFundamentalMatrixDegensac(points1, points2, options);
		await Assert.That(report.Success).IsTrue();

		Matrix3d f = report.Model / report.Model[2, 2];
		using (Assert.Multiple())
		{
			await Assert.That(f.IsApprox(expectedF, 1e-3) || (-f).IsApprox(expectedF, 1e-3)).IsTrue();
			await Assert.That(report.Support.NumInliers).IsGreaterThanOrEqualTo(points1.Length - 2);
		}
	}

	// On a dominant-plane scene with real off-plane parallax, DEGENSAC recovers the correct
	// fundamental matrix via plane-and-parallax completion, explaining the off-plane points
	// well.
	[Test]
	public async Task FundamentalMatrixDegensac_RecoversFOnDominantPlane()
	{
		Matrix3d kMat = RandomCalibrationMatrix();
		var cam2FromCam1 = new Rigid3d(
			RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d().Normalized());
		Matrix3d expectedF = EssentialMatrix.FundamentalFromEssentialMatrix(
			kMat, EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), kMat);
		expectedF /= expectedF[2, 2];

		Vector3d normal = new Vector3d(0.2, -0.1, 1.0).Normalized();
		const int kNumPoints = 200;
		const int kNumOnPlane = 190;  // 95% dominant plane.
		GenerateDominantPlaneScene(
			cam2FromCam1, kMat, normal, distance: 2.0,
			kNumPoints, kNumOnPlane, noise: 0.1,
			out Vector2d[] points1, out Vector2d[] points2, out bool[] onPlaneMask);

		bool[] offPlaneMask = Array.ConvertAll(onPlaneMask, onPlane => !onPlane);

		var degensacOptions = new FundamentalMatrixDegensacOptions { Ransac = TestRansacOptions() };
		var report = FundamentalMatrixDegensac.EstimateFundamentalMatrixDegensac(points1, points2, degensacOptions);
		await Assert.That(report.Success).IsTrue();

		Matrix3d f = report.Model / report.Model[2, 2];
		double offPlaneError = MeanSampsonErrorOnSubset(points1, points2, report.Model, offPlaneMask);
		using (Assert.Multiple())
		{
			await Assert.That(f.IsApprox(expectedF, 1e-2) || (-f).IsApprox(expectedF, 1e-2)).IsTrue();

			// DEGENSAC must explain the off-plane parallax points well, which a
			// plane-corrupted fundamental matrix cannot.
			await Assert.That(offPlaneError).IsLessThan(1.0);
		}
	}

	// Aggregated over many dominant-plane scenes, DEGENSAC recovers the correct epipolar
	// geometry at least as often as plain LO-RANSAC, which is prone to terminating on a
	// plane-corrupted model.
	[Test]
	public async Task FundamentalMatrixDegensac_OutperformsLoRansacOnDominantPlane()
	{
		const int kNumScenes = 40;
		const int kNumPoints = 200;
		const int kNumOnPlane = 190;  // 95% dominant plane.
		const double kOffPlaneErrorThreshold = 1.0;

		int degensacSuccesses = 0;
		int loransacSuccesses = 0;
		for (int s = 0; s < kNumScenes; ++s)
		{
			RandomUtils.SetPRNGSeed((uint)s);
			Matrix3d kMat = RandomCalibrationMatrix();
			var cam2FromCam1 = new Rigid3d(
				RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d().Normalized());
			Vector3d normal = new Vector3d(0.2, -0.1, 1.0).Normalized();
			GenerateDominantPlaneScene(
				cam2FromCam1, kMat, normal, distance: 2.0,
				kNumPoints, kNumOnPlane, noise: 0.1,
				out Vector2d[] points1, out Vector2d[] points2, out bool[] onPlaneMask);
			bool[] offPlaneMask = Array.ConvertAll(onPlaneMask, onPlane => !onPlane);

			RansacOptions ransacOptions = TestRansacOptions();

			var degensacOptions = new FundamentalMatrixDegensacOptions { Ransac = ransacOptions };
			var degensacReport = FundamentalMatrixDegensac.EstimateFundamentalMatrixDegensac(
				points1, points2, degensacOptions);
			if (degensacReport.Success
				&& MeanSampsonErrorOnSubset(points1, points2, degensacReport.Model, offPlaneMask) < kOffPlaneErrorThreshold)
			{
				++degensacSuccesses;
			}

			var loransac = new LoRansac<FundamentalMatrixSevenPointEstimator, FundamentalMatrixEightPointEstimator,
				Vector2d, Vector2d, Matrix3d>(ransacOptions, default, default);
			var loransacReport = loransac.Estimate(points1, points2);
			if (loransacReport.Success
				&& MeanSampsonErrorOnSubset(points1, points2, loransacReport.Model, offPlaneMask) < kOffPlaneErrorThreshold)
			{
				++loransacSuccesses;
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(degensacSuccesses).IsGreaterThan(0);
			await Assert.That(degensacSuccesses).IsGreaterThanOrEqualTo(loransacSuccesses);
		}
	}

	private static Matrix3d RandomCalibrationMatrix()
	{
		// Comma-initializer order: fx, cx, fy, cy.
		double fx = RandomUtils.RandomUniformReal(800.0, 1200.0);
		double cx = RandomUtils.RandomUniformReal(400.0, 600.0);
		double fy = RandomUtils.RandomUniformReal(800.0, 1200.0);
		double cy = RandomUtils.RandomUniformReal(400.0, 600.0);
		return new Matrix3d(fx, 0, cx, 0, fy, cy, 0, 0, 1);
	}

	// Generates a two-view scene with a dominant plane. The first `numOnPlane`
	// correspondences lie on the plane `normal . X = distance` (in the first camera frame);
	// the remaining ones are at random depths off the plane. All correspondences are
	// consistent with the true epipolar geometry; only the on-plane ones are additionally
	// consistent with the plane homography.
	private static void GenerateDominantPlaneScene(
		Rigid3d cam2FromCam1,
		Matrix3d kMat,
		Vector3d normal,
		double distance,
		int numPoints,
		int numOnPlane,
		double noise,
		out Vector2d[] points1,
		out Vector2d[] points2,
		out bool[] onPlaneMask)
	{
		Matrix3d kInv = kMat.Inverse();
		points1 = new Vector2d[numPoints];
		points2 = new Vector2d[numPoints];
		onPlaneMask = new bool[numPoints];
		for (int i = 0; i < numPoints; ++i)
		{
			// K.topRows<2>() * x.homogeneous().
			Vector2d point1 = (kMat * RandomEigen.RandomEigenVector2d().Homogeneous()).Head2();
			Vector3d ray = kInv * point1.Homogeneous();
			bool onPlane = i < numOnPlane;
			double depth = onPlane
				? distance / normal.Dot(ray)
				: RandomUtils.RandomUniformReal(0.5, 3.0);
			Vector3d point3DInCam1 = depth * ray;
			Vector2d point2 = (kMat * (cam2FromCam1 * point3DInCam1)).HNormalized();
			points1[i] = point1 + noise * RandomEigen.RandomEigenVector2d();
			points2[i] = point2 + noise * RandomEigen.RandomEigenVector2d();
			onPlaneMask[i] = onPlane;
		}
	}

	private static double MeanSampsonErrorOnSubset(Vector2d[] points1, Vector2d[] points2, Matrix3d f, bool[] subsetMask)
	{
		var residuals = new List<double>();
		EssentialMatrix.ComputeSquaredSampsonError(points1, points2, f, residuals);
		double sum = 0;
		int count = 0;
		for (int i = 0; i < residuals.Count; ++i)
		{
			if (subsetMask[i])
			{
				sum += residuals[i];
				++count;
			}
		}

		return count == 0 ? 0.0 : sum / count;
	}

	private static RansacOptions TestRansacOptions()
	{
		return new RansacOptions
		{
			MaxError = 1.0,
			Confidence = 0.9999,
			MinInlierRatio = 0.1,
			MaxNumTrials = 10000,
			RandomSeed = 0,
		};
	}
}
