// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FundamentalMatrixTests: colmap/estimators/solvers/fundamental_matrix_test.cc. TEST(Suite,
// Name) becomes Suite_Name; the TEST_P suite FundamentalMatrixEightPointEstimatorTests,
// instantiated with 8, 64 and 1024 points, becomes Suite_Name methods with the point count
// as [Arguments]. Same checks and tolerances. Tests
// ColmapSharp/Estimators/Solvers/FundamentalMatrixEstimators.cs (Tier B).
//
// The RefineFundamentalMatrixSampson cases test the TinySolver refiner (Tier C).
//
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does,
// and each test draws everything before its first await (the PRNG is per thread). The loops
// record EXPECT_* / ADD_FAILURE failures in an ExpectationLog and the test asserts it is
// empty.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class FundamentalMatrixTests
{
	[Test]
	public async Task FundamentalSevenPointEstimator_Reference()
	{
		double[] points1Raw =
		[
			0.4964, 1.0577, 0.3650, -0.0919, -0.5412, 0.0159, -0.5239,
			0.9467, 0.3467, 0.5301, 0.2797, 0.0012, -0.1986, 0.0460,
		];

		double[] points2Raw =
		[
			0.7570, 2.7340, 0.3961, 0.6981, -0.6014, 0.7110, -0.7385,
			2.2712, 0.4177, 1.2132, 0.3052, 0.4835, -0.2171, 0.5057,
		];

		const int kNumPoints = 7;
		var (points1, points2) = FromRaw(points1Raw, points2Raw, kNumPoints);

		var models = new List<Matrix3d>();
		new FundamentalMatrixSevenPointEstimator().Estimate(points1, points2, models);

		await Assert.That(models.Count).IsEqualTo(1);
		Matrix3d f = models[0] / models[0][2, 2];

		// Reference values obtained from Matlab.
		using (Assert.Multiple())
		{
			await Assert.That(f[0, 0]).IsEqualTo(4.81441976).Within(1e-6);
			await Assert.That(f[0, 1]).IsEqualTo(-8.16978909).Within(1e-6);
			await Assert.That(f[0, 2]).IsEqualTo(6.73133404).Within(1e-6);
			await Assert.That(f[1, 0]).IsEqualTo(5.16247992).Within(1e-6);
			await Assert.That(f[1, 1]).IsEqualTo(0.19325606).Within(1e-6);
			await Assert.That(f[1, 2]).IsEqualTo(-2.87239381).Within(1e-6);
			await Assert.That(f[2, 0]).IsEqualTo(-9.92570126).Within(1e-6);
			await Assert.That(f[2, 1]).IsEqualTo(3.64159554).Within(1e-6);
			await Assert.That(f[2, 2]).IsEqualTo(1.0).Within(1e-6);
		}
	}

	[Test]
	public async Task FundamentalSevenPointEstimator_Nominal()
	{
		const int kNumPoints = 7;
		var log = new ExpectationLog();
		for (int k = 0; k < 100; ++k)
		{
			var (points1, points2, expectedF) = RandomProblem(kNumPoints, coordinateScale: 1);

			var estimator = new FundamentalMatrixSevenPointEstimator();
			var models = new List<Matrix3d>();
			estimator.Estimate(points1, points2, models);

			ExpectAtLeastOneValidModel(log, estimator, points1, points2, expectedF, models, $"k={k}");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task FundamentalMatrixEightPointEstimator_Reference()
	{
		double[] points1Raw =
		[
			1.839035, 1.924743, 0.543582, 0.375221, 0.473240, 0.142522, 0.964910, 0.598376,
			0.102388, 0.140092, 15.994343, 9.622164, 0.285901, 0.430055, 0.091150, 0.254594,
		];

		double[] points2Raw =
		[
			1.002114, 1.129644, 1.521742, 1.846002, 1.084332, 0.275134, 0.293328, 0.588992,
			0.839509, 0.087290, 1.779735, 1.116857, 0.878616, 0.602447, 0.642616, 1.028681,
		];

		const int kNumPoints = 8;
		var (points1, points2) = FromRaw(points1Raw, points2Raw, kNumPoints);

		var models = new List<Matrix3d>();
		new FundamentalMatrixEightPointEstimator().Estimate(points1, points2, models);

		await Assert.That(models.Count).IsEqualTo(1);
		Matrix3d f = models[0] / models[0][2, 2];

		// Reference values obtained from Matlab.
		using (Assert.Multiple())
		{
			await Assert.That(f[0, 0]).IsEqualTo(-9.85701).Within(1e-5);
			await Assert.That(f[0, 1]).IsEqualTo(18.97038).Within(1e-5);
			await Assert.That(f[0, 2]).IsEqualTo(-1.55224).Within(1e-5);
			await Assert.That(f[1, 0]).IsEqualTo(-3.24832).Within(1e-5);
			await Assert.That(f[1, 1]).IsEqualTo(2.04346).Within(1e-5);
			await Assert.That(f[1, 2]).IsEqualTo(0.977619).Within(1e-5);
			await Assert.That(f[2, 0]).IsEqualTo(11.22355).Within(1e-5);
			await Assert.That(f[2, 1]).IsEqualTo(-19.43171).Within(1e-5);
			await Assert.That(f[2, 2]).IsEqualTo(1.0).Within(1e-5);
		}
	}

	[Test]
	[Arguments(8)]
	[Arguments(64)]
	[Arguments(1024)]
	public async Task FundamentalMatrixEightPointEstimatorTests_Nominal(int kNumPoints)
	{
		var log = new ExpectationLog();
		for (int k = 0; k < 100; ++k)
		{
			var (points1, points2, expectedF) = RandomProblem(kNumPoints, coordinateScale: 1);

			var estimator = new FundamentalMatrixEightPointEstimator();
			var models = new List<Matrix3d>();
			estimator.Estimate(points1, points2, models);

			ExpectAtLeastOneValidModel(log, estimator, points1, points2, expectedF, models, $"k={k}");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	[Arguments(8)]
	[Arguments(64)]
	[Arguments(1024)]
	public async Task FundamentalMatrixEightPointEstimatorTests_NumericalStability(int kNumPoints)
	{
		const double kCoordinateScale = 1e3;
		var log = new ExpectationLog();
		for (int k = 0; k < 100; ++k)
		{
			var (points1, points2, expectedF) = RandomProblem(kNumPoints, kCoordinateScale);

			var estimator = new FundamentalMatrixEightPointEstimator();
			var models = new List<Matrix3d>();
			estimator.Estimate(points1, points2, models);

			ExpectAtLeastOneValidModel(log, estimator, points1, points2, expectedF, models, $"k={k}", 1e-4, 1e-4);
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	[Arguments(8)]
	[Arguments(64)]
	[Arguments(1024)]
	public async Task FundamentalMatrixEightPointEstimatorTests_NoiseStability(int kNumPoints)
	{
		const double kNoise = 1e-4;
		var log = new ExpectationLog();
		for (int k = 0; k < 100; ++k)
		{
			var (points1, points2, expectedF) = RandomProblem(kNumPoints, coordinateScale: 1);
			for (int i = 0; i < kNumPoints; ++i)
			{
				points2[i] += RandomEigen.RandomEigenVector2d() * kNoise;
			}

			var estimator = new FundamentalMatrixEightPointEstimator();
			var models = new List<Matrix3d>();
			estimator.Estimate(points1, points2, models);

			ExpectAtLeastOneValidModel(log, estimator, points1, points2, expectedF, models, $"k={k}", 1e-3, 1e-2);
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// The exact model is a fixed point on noise-free correspondences: the Sampson
	// error is already zero there, so the refinement must not move away from it.
	[Test]
	public async Task RefineFundamentalMatrixSampson_IsFixedPointAtOptimum()
	{
		const int kNumPoints = 64;
		var successes = new List<bool>();
		var errors = new List<double>();
		for (int k = 0; k < 20; ++k)
		{
			var (points1, points2, f) = RandomProblem(kNumPoints, coordinateScale: 1);
			bool success = FundamentalMatrix.RefineFundamentalMatrixSampson(points1, points2, ref f);
			successes.Add(success);
			if (!success)
			{
				// ASSERT_TRUE ends the test.
				break;
			}

			errors.Add(MeanSquaredSampsonError(points1, points2, f));
		}

		await Assert.That(successes).DoesNotContain(false);
		using (Assert.Multiple())
		{
			foreach (double error in errors)
			{
				await Assert.That(error).IsLessThan(1e-15);
			}
		}
	}

	// The refined model stays rank 2 by construction, unlike an eight-point fit,
	// which has to truncate its smallest singular value.
	[Test]
	public async Task RefineFundamentalMatrixSampson_PreservesRankTwo()
	{
		const int kNumPoints = 100;
		var (points1, points2, f) = RandomProblem(kNumPoints, coordinateScale: 1);
		AddNoise(0.5, points1, points2);

		bool success = FundamentalMatrix.RefineFundamentalMatrixSampson(points1, points2, ref f);
		await Assert.That(success).IsTrue();

		Vector3d singularValues = Svd3d.Compute(f).SingularValues;
		await Assert.That(singularValues.Z).IsLessThan(1e-12 * singularValues.X);
	}

	// Models that cannot be factorized leave the input untouched, so local
	// optimization falls back to the model RANSAC already had.
	[Test]
	public async Task RefineFundamentalMatrixSampson_RejectsDegenerateModels()
	{
		const int kNumPoints = 32;
		var (points1, points2, _) = RandomProblem(kNumPoints, coordinateScale: 1);

		Matrix3d zeroF = Matrix3d.Zero;
		bool zeroRefined = FundamentalMatrix.RefineFundamentalMatrixSampson(points1, points2, ref zeroF);

		// Rank 1: only one non-zero singular value, so the ratio is undefined.
		var rank1F = new Matrix3d(4, 5, 6, 8, 10, 12, 12, 15, 18);
		Matrix3d expectedRank1F = rank1F;
		bool rank1Refined = FundamentalMatrix.RefineFundamentalMatrixSampson(points1, points2, ref rank1F);

		using (Assert.Multiple())
		{
			await Assert.That(zeroRefined).IsFalse();
			await Assert.That(zeroF == Matrix3d.Zero).IsTrue();
			await Assert.That(rank1Refined).IsFalse();
			await Assert.That(rank1F == expectedRank1F).IsTrue();
		}
	}

	// Adds isotropic Gaussian pixel noise to both point sets. Eigen::Vector2d(a, b) with two
	// RandomGaussian calls: clang evaluates constructor arguments left to right.
	private static void AddNoise(double stddev, Vector2d[] points1, Vector2d[] points2)
	{
		for (int i = 0; i < points1.Length; ++i)
		{
			double x1 = RandomUtils.RandomGaussian(0.0, stddev);
			double y1 = RandomUtils.RandomGaussian(0.0, stddev);
			points1[i] += new Vector2d(x1, y1);
			double x2 = RandomUtils.RandomGaussian(0.0, stddev);
			double y2 = RandomUtils.RandomGaussian(0.0, stddev);
			points2[i] += new Vector2d(x2, y2);
		}
	}

	private static double MeanSquaredSampsonError(Vector2d[] points1, Vector2d[] points2, Matrix3d f)
	{
		var residuals = new List<double>();
		EssentialMatrix.ComputeSquaredSampsonError(points1, points2, f, residuals);
		// std::accumulate: left to right.
		double sum = 0.0;
		foreach (double residual in residuals)
		{
			sum += residual;
		}

		return sum / residuals.Count;
	}

	private static (Vector2d[] Points1, Vector2d[] Points2) FromRaw(double[] points1Raw, double[] points2Raw, int numPoints)
	{
		var points1 = new Vector2d[numPoints];
		var points2 = new Vector2d[numPoints];
		for (int i = 0; i < numPoints; ++i)
		{
			points1[i] = new Vector2d(points1Raw[2 * i], points1Raw[2 * i + 1]);
			points2[i] = new Vector2d(points2Raw[2 * i], points2Raw[2 * i + 1]);
		}

		return (points1, points2);
	}

	/// <summary>
	/// The draws the TEST bodies make, in their order: K (RandomCalibrationMatrix, with fx,
	/// fy, cx, cy scaled by <paramref name="coordinateScale"/> as NumericalStability does),
	/// cam2_from_cam1, expected_F and the correspondences.
	/// </summary>
	private static (Vector2d[] Points1, Vector2d[] Points2, Matrix3d ExpectedF) RandomProblem(int numPoints, double coordinateScale)
	{
		Matrix3d k = RandomCalibrationMatrix();
		if (coordinateScale != 1)
		{
			k = new Matrix3d(
				k[0, 0] * coordinateScale, k[0, 1], k[0, 2] * coordinateScale,
				k[1, 0], k[1, 1] * coordinateScale, k[1, 2] * coordinateScale,
				k[2, 0], k[2, 1], k[2, 2]);
		}

		// Separate statements fix the draw order (rotation, then translation).
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d();
		var cam2FromCam1 = new Rigid3d(rotation, translation);
		Matrix3d expectedF = EssentialMatrix.FundamentalFromEssentialMatrix(
			k, EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), k);
		var (points1, points2) = RandomEpipolarCorrespondences(cam2FromCam1, k, numPoints);
		return (points1, points2, expectedF);
	}

	private static Matrix3d RandomCalibrationMatrix()
	{
		// Eigen's comma initializer evaluates left to right: fx, cx, fy, cy.
		double fx = RandomUtils.RandomUniformReal(800.0, 1200.0);
		double cx = RandomUtils.RandomUniformReal(400.0, 600.0);
		double fy = RandomUtils.RandomUniformReal(800.0, 1200.0);
		double cy = RandomUtils.RandomUniformReal(400.0, 600.0);
		return new Matrix3d(fx, 0, cx, 0, fy, cy, 0, 0, 1);
	}

	private static (Vector2d[] Points1, Vector2d[] Points2) RandomEpipolarCorrespondences(
		Rigid3d cam2FromCam1, Matrix3d k, int numPoints)
	{
		Matrix3d kInverse = k.Inverse();
		var points1 = new Vector2d[numPoints];
		var points2 = new Vector2d[numPoints];
		for (int i = 0; i < numPoints; ++i)
		{
			// K.topRows<2>() * x.homogeneous() is the first two rows of K * x.homogeneous().
			Vector3d projected = k * RandomEigen.RandomEigenVector2d().Homogeneous();
			points1[i] = new Vector2d(projected.X, projected.Y);
			double randomDepth = RandomUtils.RandomUniformReal(0.2, 2.0);
			points2[i] = (k * (cam2FromCam1 * (randomDepth * kInverse * points1[i].Homogeneous()))).HNormalized();
		}

		return (points1, points2);
	}

	/// <summary>
	/// Port of the test's ExpectAtLeastOneValidModel: some model equals expected_F up to
	/// scale (both divided by their (2, 2) entry, isApprox within fEps), and its residuals are
	/// all below rEps; otherwise ADD_FAILURE.
	/// </summary>
	private static void ExpectAtLeastOneValidModel<TEstimator>(
		ExpectationLog log,
		TEstimator estimator,
		Vector2d[] points1,
		Vector2d[] points2,
		Matrix3d expectedF,
		List<Matrix3d> models,
		string what,
		double fEps = 1e-6,
		double rEps = 1e-6)
		where TEstimator : IEstimator<Vector2d, Vector2d, Matrix3d>
	{
		expectedF /= expectedF[2, 2];
		foreach (Matrix3d model in models)
		{
			Matrix3d f = model / model[2, 2];
			if (!f.IsApprox(expectedF, fEps))
			{
				continue;
			}

			var residuals = new double[points1.Length];
			estimator.Residuals(points1, points2, f, residuals);
			for (int j = 0; j < points1.Length; ++j)
			{
				log.Less(residuals[j], rEps, $"{what} residual[{j}]");
			}

			return;
		}

		log.Fail($"{what}: No fundamental matrix is equal up to scale.");
	}
}
