// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// HomographyMatrixSolverTests: colmap/estimators/solvers/homography_matrix_test.cc ported
// 1:1 (the class name differs from the file's only because Geometry/HomographyMatrixTests
// already ports colmap/geometry/homography_matrix_test.cc). TEST(Suite, Name) becomes
// Suite_Name; the TEST_P suites HomographyMatrixTests and HomographyMatrixRayTests,
// instantiated with 4, 8, 64 and 1024 points, become Suite_Name methods with the point
// count as [Arguments]. Same checks and tolerances. Tests
// ColmapSharp/Estimators/Solvers/HomographyMatrixEstimator.cs (Tier B).
//
// Each test seeds the PRNG with 0, as COLMAP's gtest_main does, and draws everything before
// its first await (the PRNG is per thread). The loops record EXPECT_* failures in an
// ExpectationLog and the test asserts it is empty; ASSERT_* in a loop ends the loop.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class HomographyMatrixSolverTests
{
	[Test]
	public async Task HomographyMatrixEstimator_CollinearMinimalSampleTriplets()
	{
		Vector2d[] dst = [new(0, 0), new(1, 0), new(1, 1), new(0, 1)];
		var modelCounts = new List<int>();
		for (int outlierIdx = 0; outlierIdx < 4; ++outlierIdx)
		{
			Vector2d[] src = [new(0, 0), new(1, 0), new(2, 0), new(3, 0)];
			src[outlierIdx] = new Vector2d(0, 1);

			var models = new List<Matrix3d>();
			new HomographyMatrixEstimator().Estimate(src, dst, models);
			modelCounts.Add(models.Count);
		}

		await Assert.That(modelCounts).IsEquivalentTo([0, 0, 0, 0]);
	}

	[Test]
	[Arguments(4)]
	[Arguments(8)]
	[Arguments(64)]
	[Arguments(1024)]
	public async Task HomographyMatrixTests_Nominal(int kNumPoints)
	{
		RandomUtils.SetPRNGSeed(0);
		var log = new ExpectationLog();
		for (int x = 0; x < 10; ++x)
		{
			var expectedH = new Matrix3d(x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1);

			var src = new Vector2d[kNumPoints];
			var dst = new Vector2d[kNumPoints];
			for (int i = 0; i < kNumPoints; ++i)
			{
				src[i] = RandomEigen.RandomEigenVector2d();
				dst[i] = (expectedH * src[i].Homogeneous()).HNormalized();
			}

			if (!EstimateAndCheck(log, src, dst, 1e-6, $"x={x}"))
			{
				break;
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// Test numerical stability with large coordinates. This is to ensure that the
	// homography matrix estimator is numerically stable despite not using
	// coordinate normalization. We can do this because of double precision.
	[Test]
	[Arguments(4)]
	[Arguments(8)]
	[Arguments(64)]
	[Arguments(1024)]
	public async Task HomographyMatrixTests_NumericalStability(int kNumPoints)
	{
		RandomUtils.SetPRNGSeed(0);
		const double kCoordinateScale = 1e6;
		var log = new ExpectationLog();
		for (int x = 1; x < 10; ++x)
		{
			var expectedH = new Matrix3d(x, 0, 0, 0, 1, 0, 0, 0, 1);

			var src = new Vector2d[kNumPoints];
			var dst = new Vector2d[kNumPoints];
			for (int i = 0; i < kNumPoints; ++i)
			{
				src[i] = RandomEigen.RandomEigenVector2d() * kCoordinateScale;
				dst[i] = (expectedH * src[i].Homogeneous()).HNormalized();
			}

			if (!EstimateAndCheck(log, src, dst, 1e-6, $"x={x}"))
			{
				break;
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	[Arguments(4)]
	[Arguments(8)]
	[Arguments(64)]
	[Arguments(1024)]
	public async Task HomographyMatrixTests_NoiseStability(int kNumPoints)
	{
		RandomUtils.SetPRNGSeed(0);
		const double kNoise = 1e-3;
		var log = new ExpectationLog();
		for (int x = 1; x < 10; ++x)
		{
			var expectedH = new Matrix3d(x, 0, 0, 0, 1, 0, 0, 0, 1);

			var src = new Vector2d[kNumPoints];
			var dst = new Vector2d[kNumPoints];
			for (int i = 0; i < kNumPoints; ++i)
			{
				src[i] = RandomEigen.RandomEigenVector2d();
				Vector2d projected = (expectedH * src[i].Homogeneous()).HNormalized();
				dst[i] = projected + RandomEigen.RandomEigenVector2d() * kNoise;
			}

			if (!EstimateAndCheck(log, src, dst, 1e-5, $"x={x}"))
			{
				break;
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	[Arguments(4)]
	[Arguments(8)]
	[Arguments(64)]
	[Arguments(1024)]
	public async Task HomographyMatrixTests_Degenerate(int kNumPoints)
	{
		RandomUtils.SetPRNGSeed(0);
		const double kNoise = 1e-3;
		var modelCounts = new List<int>();
		for (int x = 0; x < 10; ++x)
		{
			var expectedH = new Matrix3d(x, 0.2, 0.3, 30, 0.2, 0.1, 0.3, 20, 1);

			var src = new List<Vector2d> { new(2, 1), new(3, 1), new(10, 30) };
			int numRedundantPoints = kNumPoints - src.Count;
			for (int i = 0; i < numRedundantPoints; ++i)
			{
				src.Add(src[0]);
			}

			var dst = new Vector2d[src.Count];
			for (int i = 0; i < src.Count; ++i)
			{
				Vector3d dsth = expectedH * src[i].Homogeneous();
				dst[i] = dsth.HNormalized() + RandomEigen.RandomEigenVector2d() * kNoise;
			}

			var models = new List<Matrix3d>();
			new HomographyMatrixEstimator().Estimate(src.ToArray(), dst, models);
			modelCounts.Add(models.Count);
		}

		await Assert.That(kNumPoints).IsGreaterThanOrEqualTo(4);
		await Assert.That(modelCounts).IsEquivalentTo(Enumerable.Repeat(0, 10));
	}

	[Test]
	[Arguments(4)]
	[Arguments(8)]
	[Arguments(64)]
	[Arguments(1024)]
	public async Task HomographyMatrixRayTests_Nominal(int kNumPoints)
	{
		RandomUtils.SetPRNGSeed(0);
		Camera camera1 = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1000, 1920, 1080);
		Camera camera2 = Camera.CreateFromModelId(2, CameraModelId.SimplePinhole, 1200, 1920, 1080);

		var log = new ExpectationLog();
		for (int x = 1; x < 10; ++x)
		{
			var hPix = new Matrix3d(1 + 0.1 * x, 0.02, 30, 0.03, 1.1, 20, 1e-5, 2e-5, 1);

			var (_, _, camRays1, camRays2) = SyntheticRayCorrespondences(camera1, camera2, hPix, kNumPoints, offset: 0);

			var estimator = new HomographyMatrixRayEstimator(camera2);
			var models = new List<Matrix3d>();
			estimator.Estimate(camRays1, camRays2, models);

			if (!log.True(models.Count == 1, $"x={x}: models.size() == 1 (got {models.Count})"))
			{
				break;
			}

			// Also guards the global sign: a negated H transfers every ray behind the
			// camera and scores every residual at the maximum.
			var residuals = new double[kNumPoints];
			estimator.Residuals(camRays1, camRays2, models[0], residuals);
			for (int i = 0; i < kNumPoints; ++i)
			{
				log.Less(residuals[i], 1e-6, $"x={x} residual[{i}]");
			}

			// The estimate maps rays to rays, so it must agree with the pixel-space
			// homography only after conjugation by the calibration matrices.
			Matrix3d hFromRays = camera2.CalibrationMatrix() * models[0] * camera1.CalibrationMatrix().Inverse();
			log.True(
				Normalized(hFromRays).IsApprox(Normalized(hPix), 1e-6) ||
				Normalized(hFromRays).IsApprox(-Normalized(hPix), 1e-6),
				$"x={x}: H_from_rays ~ +-H_pix");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// For undistorted pinhole cameras the two residuals are algebraically
	// identical, since projecting K2^-1 H K1 x1 back through K2 dehomogenizes to
	// exactly H p1. Large offsets are included because a first-order relationship
	// would also pass near zero.
	[Test]
	[Arguments(4)]
	[Arguments(8)]
	[Arguments(64)]
	[Arguments(1024)]
	public async Task HomographyMatrixRayTests_PixelEquivalence(int kNumPoints)
	{
		RandomUtils.SetPRNGSeed(0);
		Camera camera1 = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1000, 1920, 1080);
		Camera camera2 = Camera.CreateFromModelId(2, CameraModelId.Pinhole, 1200, 1920, 1080);

		var log = new ExpectationLog();
		foreach (double offset in new[] { 0.0, 1.0, 20.0, 200.0 })
		{
			var hPix = new Matrix3d(1.2, 0.02, 30, 0.03, 1.1, 20, 1e-5, 2e-5, 1);

			var (src, dst, camRays1, camRays2) = SyntheticRayCorrespondences(camera1, camera2, hPix, kNumPoints, offset);

			Matrix3d hRay = camera2.CalibrationMatrix().Inverse() * hPix * camera1.CalibrationMatrix();

			var pixelResiduals = new double[kNumPoints];
			new HomographyMatrixEstimator().Residuals(src, dst, hPix, pixelResiduals);

			var rayResiduals = new double[kNumPoints];
			new HomographyMatrixRayEstimator(camera2).Residuals(camRays1, camRays2, hRay, rayResiduals);

			for (int i = 0; i < kNumPoints; ++i)
			{
				log.Near(
					rayResiduals[i],
					pixelResiduals[i],
					1e-9 * Math.Max(1.0, pixelResiduals[i]),
					$"offset={offset} residual[{i}]");
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// A hypothesis that transfers a ray out of a perspective camera's field has no
	// image point to score against. Cannot happen for a true inlier, whose measured
	// ray faces forward, so this only guards wrong hypotheses.
	[Test]
	public async Task HomographyMatrixRay_BehindCamera()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1000, 1920, 1080);
		Matrix3d h = new AngleAxisd(MathUtils.DegToRad(100.0), Vector3d.UnitY).ToRotationMatrix();

		// The second ray stays in front after the rotation, so the rejection must be
		// selective rather than blanket.
		Vector3d forward = new AngleAxisd(MathUtils.DegToRad(-70.0), Vector3d.UnitY).ToRotationMatrix() * Vector3d.UnitZ;
		Vector3d[] camRays1 = [Vector3d.UnitZ, forward];
		CamRayWithImgPoint[] camRays2 =
		[
			new(Vector3d.UnitZ, new Vector2d(960, 540)),
			new((h * forward).Normalized(), camera.ImgFromCam((h * forward).Normalized())!.Value),
		];

		var residuals = new double[2];
		new HomographyMatrixRayEstimator(camera).Residuals(camRays1, camRays2, h, residuals);

		using (Assert.Multiple())
		{
			await Assert.That(residuals[0]).IsEqualTo(double.MaxValue);
			await Assert.That(residuals[1]).IsLessThan(1e-6);
		}
	}

	// A spherical camera images every direction, so the estimator must work over
	// the whole sphere and reject nothing on cheirality grounds.
	[Test]
	[Arguments(4)]
	[Arguments(8)]
	[Arguments(64)]
	[Arguments(1024)]
	public async Task HomographyMatrixRayTests_Spherical(int kNumPoints)
	{
		RandomUtils.SetPRNGSeed(0);
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Equirectangular, focalLength: 0, 2048, 1024);
		bool isSpherical = camera.IsSpherical;

		var log = new ExpectationLog();
		for (int x = 1; x < 10 && isSpherical; ++x)
		{
			Matrix3d expectedH = new AngleAxisd(0.2 * x, new Vector3d(0.2, 1, 0.3).Normalized()).ToRotationMatrix();

			var camRays1 = new Vector3d[kNumPoints];
			var camRays2 = new CamRayWithImgPoint[kNumPoints];
			for (int i = 0; i < kNumPoints; ++i)
			{
				Vector3d ray1 = RandomEigen.RandomEigenVector3d().Normalized();
				Vector3d ray2 = (expectedH * ray1).Normalized();
				camRays1[i] = ray1;
				camRays2[i] = new CamRayWithImgPoint(ray2, camera.ImgFromCam(ray2)!.Value);
			}

			var estimator = new HomographyMatrixRayEstimator(camera);
			var models = new List<Matrix3d>();
			estimator.Estimate(camRays1, camRays2, models);

			if (!log.True(models.Count == 1, $"x={x}: models.size() == 1 (got {models.Count})"))
			{
				break;
			}

			log.True(Normalized(models[0]).IsApprox(Normalized(expectedH), 1e-6), $"x={x}: H ~ expected_H");

			var residuals = new double[kNumPoints];
			estimator.Residuals(camRays1, camRays2, models[0], residuals);
			for (int i = 0; i < kNumPoints; ++i)
			{
				log.Less(residuals[i], 1e-6, $"x={x} residual[{i}]");
			}
		}

		await Assert.That(isSpherical).IsTrue();
		await Assert.That(log.Failures).IsEmpty();
	}

	// The azimuth wraps, so two nearly identical directions can sit a full image
	// width apart in pixels. Scoring that raw difference would reject every
	// correspondence near the seam.
	[Test]
	public async Task HomographyMatrixRay_SphericalSeam()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Equirectangular, focalLength: 0, 2048, 1024);
		const double kDelta = 0.01;
		var ray1 = new Vector3d(Math.Sin(Math.PI - kDelta), 0, Math.Cos(Math.PI - kDelta));
		var ray2 = new Vector3d(Math.Sin(-Math.PI + kDelta), 0, Math.Cos(-Math.PI + kDelta));
		Vector2d point1 = camera.ImgFromCam(ray1)!.Value;
		Vector2d point2 = camera.ImgFromCam(ray2)!.Value;
		await Assert.That((point1 - point2).Norm).IsGreaterThan(2000);

		var residuals = new double[1];
		new HomographyMatrixRayEstimator(camera).Residuals([ray1], [new CamRayWithImgPoint(ray2, point2)], Matrix3d.Identity, residuals);

		double expected = 2 * kDelta * camera.Width / (2 * Math.PI);
		await Assert.That(residuals[0]).IsEqualTo(expected * expected).Within(1e-6);
	}

	// Why Estimate must resolve the sign of H. A spherical camera projects -H x1 as
	// happily as H x1, only to the antipode, so a negated model is not rejected but
	// silently scored against the wrong half of the sphere.
	[Test]
	public async Task HomographyMatrixRay_SphericalSign()
	{
		RandomUtils.SetPRNGSeed(0);
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Equirectangular, focalLength: 0, 2048, 1024);
		Matrix3d h = new AngleAxisd(0.4, new Vector3d(0.2, 1, 0.3).Normalized()).ToRotationMatrix();

		var camRays1 = new Vector3d[64];
		var camRays2 = new CamRayWithImgPoint[64];
		for (int i = 0; i < 64; ++i)
		{
			Vector3d ray1 = RandomEigen.RandomEigenVector3d().Normalized();
			Vector3d ray2 = (h * ray1).Normalized();
			camRays1[i] = ray1;
			camRays2[i] = new CamRayWithImgPoint(ray2, camera.ImgFromCam(ray2)!.Value);
		}

		var estimator = new HomographyMatrixRayEstimator(camera);

		var log = new ExpectationLog();
		var residuals = new double[64];
		estimator.Residuals(camRays1, camRays2, h, residuals);
		foreach (double residual in residuals)
		{
			log.Less(residual, 1e-6, "residual");
		}

		var negatedResiduals = new double[64];
		estimator.Residuals(camRays1, camRays2, -h, negatedResiduals);
		foreach (double residual in negatedResiduals)
		{
			// Finite, so nothing rejects it, but half the image away in azimuth.
			log.Less(residual, double.MaxValue, "negated residual");
			log.Greater(residual, 1e4, "negated residual");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	/// <summary>
	/// Estimate from (src, dst), then ASSERT_EQ(models.size(), 1) and EXPECT_LT every
	/// residual against <paramref name="maxResidual"/>. False when the ASSERT fails.
	/// </summary>
	private static bool EstimateAndCheck(ExpectationLog log, Vector2d[] src, Vector2d[] dst, double maxResidual, string what)
	{
		var estimator = new HomographyMatrixEstimator();
		var models = new List<Matrix3d>();
		estimator.Estimate(src, dst, models);

		if (!log.True(models.Count == 1, $"{what}: models.size() == 1 (got {models.Count})"))
		{
			return false;
		}

		var residuals = new double[src.Length];
		estimator.Residuals(src, dst, models[0], residuals);
		for (int i = 0; i < src.Length; ++i)
		{
			log.Less(residuals[i], maxResidual, $"{what} residual[{i}]");
		}

		return true;
	}

	/// <summary>
	/// Correspondences for a plane whose pixel-space homography is <paramref name="hPix"/>,
	/// with dst displaced by <paramref name="offset"/> pixels so that residuals are
	/// non-trivial. Port of the test's SyntheticRayCorrespondences.
	/// </summary>
	private static (Vector2d[] Src, Vector2d[] Dst, Vector3d[] CamRays1, CamRayWithImgPoint[] CamRays2) SyntheticRayCorrespondences(
		Camera camera1, Camera camera2, Matrix3d hPix, int numPoints, double offset)
	{
		var src = new Vector2d[numPoints];
		var dst = new Vector2d[numPoints];
		var camRays1 = new Vector3d[numPoints];
		var camRays2 = new CamRayWithImgPoint[numPoints];
		for (int i = 0; i < numPoints; ++i)
		{
			var imageSize = new Vector2d(camera1.Width, camera1.Height);
			Vector2d unit = 0.5 * (RandomEigen.RandomEigenVector2d() + Vector2d.Ones);
			var point1 = new Vector2d(imageSize.X * unit.X, imageSize.Y * unit.Y);
			Vector2d point2 = (hPix * point1.Homogeneous()).HNormalized() + offset * RandomEigen.RandomEigenVector2d();
			src[i] = point1;
			dst[i] = point2;
			camRays1[i] = camera1.CamRayFromImg(point1)!.Value;
			camRays2[i] = new CamRayWithImgPoint(camera2.CamRayFromImg(point2)!.Value, point2);
		}

		return (src, dst, camRays1, camRays2);
	}

	// Eigen's Matrix3d::normalized(): divided by the Frobenius norm (unchanged when zero).
	private static Matrix3d Normalized(Matrix3d m)
	{
		double norm = m.Norm();
		return norm > 0 ? m / norm : m;
	}
}
