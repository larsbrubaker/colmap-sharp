// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RelativePoseOneSidedFocalTests: colmap/estimators/solvers/relpose_one_sided_focal_test.cc
// ported 1:1 (test names are Suite_Name). Same problem generator, loop counts, tolerances
// and failure-rate bound. Tests ColmapSharp/Estimators/Solvers/RelativePoseOneSidedFocal.cs
// and PoseLib/Relpose6ptOnesidedFocal.cs. Tier B for Estimate/Residuals, Tier C for Refine.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using M_t = ColmapSharp.Estimators.Solvers.RelativePoseOneSidedFocalEstimator.Model;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class RelativePoseOneSidedFocalTests
{
	// The unknown focal of the first (uncalibrated) view. The second view is calibrated and
	// enters through a real camera, whose unprojection Jacobian lets its share of the
	// residual be measured in its own pixels.
	private const double kFocal1 = 1000.0;
	private const double kFocal2 = 800.0;
	private const int kWidth2 = 1600;
	private const int kHeight2 = 1200;

	// Rejection thresholds used to condition a minimal sample: minimum depth in front of the
	// second camera and minimum parallax (sin^2 of the ray angle).
	private const double kMinDepth = 0.5;
	private const double kMinParallax = 1e-2;  // ~5.7 degrees.

	// The minimal 6-point solver recovers the pose and the unknown focal on clean samples.
	[Test]
	public async Task RelativePoseOneSidedFocalEstimator_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		Camera camera2 = TestCamera2();
		var estimator = new RelativePoseOneSidedFocalEstimator();
		var log = new ExpectationLog();
		for (int k = 0; k < 100; ++k)
		{
			Rigid3d cam2FromCam1 = TestCam2FromCam1();
			Matrix3d expectedE = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
			var points1 = new List<Vector2d>();
			var camRays2WithJac = new List<CamRayWithJac>();
			RandomOneSidedFocalCorrespondences(
				cam2FromCam1, camera2, kFocal1, RelativePoseOneSidedFocalEstimator.MinNumSamples,
				rejectDegenerate: true, points1, camRays2WithJac);

			var models = new List<M_t>();
			estimator.Estimate(points1.ToArray(), camRays2WithJac.ToArray(), models);

			log.True(HasValidModel(points1, camRays2WithJac, expectedE, kFocal1, models), $"k={k}: HasValidModel");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// Nominal residual behavior: zero on exact correspondences, growing with the error, in
	// first-view pixels, and max on degenerate inputs (non-positive focal or a zero ray). The
	// exact formula is pinned by the cost-function tests.
	[Test]
	public async Task RelativePoseOneSidedFocalEstimator_Residuals()
	{
		RandomUtils.SetPRNGSeed(0);
		Camera camera2 = TestCamera2();
		Rigid3d cam2FromCam1 = TestCam2FromCam1();
		var points1 = new List<Vector2d>();
		var camRays2WithJac = new List<CamRayWithJac>();
		RandomOneSidedFocalCorrespondences(
			cam2FromCam1, camera2, kFocal1, 30, rejectDegenerate: false, points1, camRays2WithJac);

		var estimator = new RelativePoseOneSidedFocalEstimator();
		var log = new ExpectationLog();
		var model = new M_t(EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), kFocal1);
		CamRayWithJac[] rays2 = camRays2WithJac.ToArray();

		// Exact correspondences lie on the epipolar variety.
		double[] residuals = new double[points1.Count];
		estimator.Residuals(points1.ToArray(), rays2, model, residuals);
		foreach (double residual in residuals)
		{
			log.Less(Math.Sqrt(residual), 1e-6, "exact residual");
		}

		// Shift the view-1 points off their epipolar lines along the line normal. The
		// constraint is linear in the first view's pixel, so the residual is at most the
		// offset, and its mean is of pixel order rather than the ~offset/focal of ray units.
		// The mean stays robust where a single point's sensitivity vanishes near an epipole.
		Matrix3d m = model.E * Matrix3d.FromDiagonal(new Vector3d(1.0 / kFocal1, 1.0 / kFocal1, 1.0));
		double prevMean = 0.0;
		foreach (double offsetPixels in new[] { 1.0, 4.0, 16.0 })
		{
			Vector2d[] shiftedPoints1 = points1.ToArray();
			for (int i = 0; i < points1.Count; ++i)
			{
				Vector3d mtRay2 = m.Transpose() * rays2[i].Ray;
				Vector2d lineNormal = new Vector2d(mtRay2.X, mtRay2.Y).Normalized();
				shiftedPoints1[i] += offsetPixels * lineNormal;
			}

			estimator.Residuals(shiftedPoints1, rays2, model, residuals);
			double mean = 0.0;
			foreach (double residual in residuals)
			{
				log.Less(Math.Sqrt(residual), offsetPixels + 1e-6, $"offset {offsetPixels}: residual");
				mean += Math.Sqrt(residual);
			}

			mean /= residuals.Length;
			log.Greater(mean, 0.1 * offsetPixels, $"offset {offsetPixels}: mean");
			log.Greater(mean, prevMean, $"offset {offsetPixels}: mean increases");
			prevMean = mean;
		}

		M_t invalidModel = model with { Focal = 0.0 };
		estimator.Residuals(points1.ToArray(), rays2, invalidModel, residuals);
		foreach (double residual in residuals)
		{
			log.Equal(residual, double.MaxValue, "invalid residual");
		}

		CamRayWithJac[] degenerateRays2 = camRays2WithJac.ToArray();
		degenerateRays2[0] = CamRayWithJac.Zero;
		estimator.Residuals(points1.ToArray(), degenerateRays2, model, residuals);
		log.Equal(residuals[0], double.MaxValue, "degenerate residual[0]");
		log.Less(Math.Sqrt(residuals[1]), 1e-6, "degenerate residual[1]");

		await Assert.That(log.Failures).IsEmpty();
	}

	// Refinement pulls a perturbed pose + focal back to the ground truth.
	[Test]
	public async Task RelativePoseOneSidedFocalEstimator_RefineFromInitialModel()
	{
		RandomUtils.SetPRNGSeed(0);
		Camera camera2 = TestCamera2();
		var log = new ExpectationLog();
		for (int k = 0; k < 50; ++k)
		{
			Rigid3d cam2FromCam1 = TestCam2FromCam1();
			Matrix3d expectedE = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
			var points1 = new List<Vector2d>();
			var camRays2WithJac = new List<CamRayWithJac>();
			RandomOneSidedFocalCorrespondences(
				cam2FromCam1, camera2, kFocal1, 50, rejectDegenerate: false, points1, camRays2WithJac);

			Quaterniond seedRotation = cam2FromCam1.Rotation *
				new AngleAxisd(0.02, RandomEigen.RandomEigenVector3d().Normalized()).ToQuaternion();
			Vector3d seedTranslation = (cam2FromCam1.Translation + 0.02 * RandomEigen.RandomEigenVector3d()).Normalized();
			var model = new M_t(EssentialMatrix.EssentialMatrixFromPose(new Rigid3d(seedRotation, seedTranslation)), 1.1 * kFocal1);

			// ASSERT_TRUE.
			await Assert.That(RelativePoseOneSidedFocalEstimator.Refine(points1.ToArray(), camRays2WithJac.ToArray(), ref model)).IsTrue();

			log.True(
				HasValidModel(points1, camRays2WithJac, expectedE, kFocal1, [model], eEps: 1e-3, focalRelEps: 1e-2, rEps: 1e-2),
				$"k={k}: HasValidModel");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// The calibrated view enters as bearing rays, which span the full sphere rather than a
	// pinhole image plane: rays with z <= 0, which only a camera seeing beyond 180 degrees
	// observes and which no pinhole image plane can represent, must be usable
	// correspondences. An equirectangular second camera supplies both those rays and the
	// Jacobians that put its share of the residual in its own pixels.
	[Test]
	public async Task RelativePoseOneSidedFocalEstimator_FullSphereCalibratedRays()
	{
		RandomUtils.SetPRNGSeed(0);
		Camera camera2 = TestCamera2(CameraModelId.Equirectangular);
		// Without the cheirality filter the configurations are harsher than any pinhole pair,
		// and the minimal solve occasionally loses the true root. Never exceeded 2% over 199
		// measured runs; a real regression exceeds it at once.
		const int kNumTrials = 100;
		const double kMaxFailureRate = 0.03;
		int numFailures = 0;
		var estimator = new RelativePoseOneSidedFocalEstimator();
		for (int k = 0; k < kNumTrials; ++k)
		{
			Rigid3d cam2FromCam1;
			var points1 = new List<Vector2d>();
			var camRays2WithJac = new List<CamRayWithJac>();
			// Resample until the sample really contains such a ray, so that every trial
			// exercises the property rather than a random subset of them. A large rotation puts
			// a good share of the points outside any pinhole frustum.
			do
			{
				points1.Clear();
				camRays2WithJac.Clear();
				cam2FromCam1 = TestCam2FromCam1(maxAngleDeg: 140.0);
				RandomOneSidedFocalCorrespondences(
					cam2FromCam1, camera2, kFocal1, RelativePoseOneSidedFocalEstimator.MinNumSamples,
					rejectDegenerate: true, points1, camRays2WithJac, requireFrontOfCam2: false);
			}
			while (!camRays2WithJac.Any(camRayWithJac => camRayWithJac.Ray.Z <= 0.0));

			var models = new List<M_t>();
			estimator.Estimate(points1.ToArray(), camRays2WithJac.ToArray(), models);

			if (!HasValidModel(points1, camRays2WithJac, EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), kFocal1, models))
			{
				++numFailures;
			}
		}

		await Assert.That((double)numFailures / kNumTrials).IsLessThanOrEqualTo(kMaxFailureRate);
	}

	// The calibrated second camera. A distortion-free model keeps the projection round trip
	// used to build exact correspondences exact.
	private static Camera TestCamera2(CameraModelId modelId = CameraModelId.Pinhole) =>
		Camera.CreateFromModelId(2, modelId, kFocal2, kWidth2, kHeight2);

	// Random relative pose with a unit-norm baseline (away from the pure-rotation degeneracy)
	// and a rotation bounded so the two view frustums overlap, which keeps the sampled points
	// in front of both cameras.
	private static Rigid3d TestCam2FromCam1(double maxAngleDeg = 60.0)
	{
		Vector3d axis = RandomEigen.RandomEigenVector3d().Normalized();
		// clang evaluates the Rigid3d constructor arguments left to right.
		double angle = MathUtils.DegToRad(RandomUtils.RandomUniformReal(0.0, maxAngleDeg));
		Quaterniond rotation = new AngleAxisd(angle, axis).ToQuaternion();
		return new Rigid3d(rotation, RandomEigen.RandomEigenVector3d().Normalized());
	}

	// Generates correspondences from random 3D points: centered image points scaled by
	// `focal1` for the uncalibrated view, and for the calibrated one the bearing and
	// unprojection Jacobian read back from `camera2`. Clearing requireFrontOfCam2 admits rays
	// at any angle, as a fisheye or spherical second camera would observe.
	//
	// The second view is round-tripped through the camera rather than synthesized as a bare
	// bearing, because the residual now needs d(ray2)/d(pixel2), which only the camera model
	// can supply.
	private static void RandomOneSidedFocalCorrespondences(
		Rigid3d cam2FromCam1,
		Camera camera2,
		double focal1,
		int numPoints,
		bool rejectDegenerate,
		List<Vector2d> points1,
		List<CamRayWithJac> camRays2WithJac,
		bool requireFrontOfCam2 = true)
	{
		for (int i = 0; i < numPoints; ++i)
		{
			Vector3d pointInCam1;
			CamRayWithJac? camRay2WithJac;
			bool degenerate;
			do
			{
				// Moderate field of view (|x/z|, |y/z| <= ~0.5): wide-angle points span a
				// magnitude range that degrades the conditioning of the minimal solve.
				Vector3d ray1 = RandomEigen.RandomEigenVector3d();
				ray1 = new Vector3d(ray1.X, ray1.Y, Math.Abs(ray1.Z) + 2.0);
				double depth = RandomUtils.RandomUniformReal(1.0, 3.0);
				pointInCam1 = depth * ray1.Normalized();
				Vector3d pointInCam2 = cam2FromCam1 * pointInCam1;
				Vector3d ray1InCam2 = cam2FromCam1.Rotation * pointInCam1.Normalized();
				double cosParallax = ray1InCam2.Dot(pointInCam2.Normalized());
				// Cheirality in cam1 holds by construction; requiring it in cam2 is only
				// meaningful for a pinhole second view.
				degenerate = 1.0 - cosParallax * cosParallax < kMinParallax ||
					(requireFrontOfCam2 && pointInCam2.Z < kMinDepth);
				camRay2WithJac = null;
				Vector2d? xy = camera2.ImgFromCam(pointInCam2);
				if (xy.HasValue)
				{
					camRay2WithJac = camera2.CamRayFromImgWithJac(xy.Value);
				}

				// An unprojectable or rank-deficient pixel carries no usable Jacobian, so it is
				// always resampled, whatever the caller asked for.
				degenerate = degenerate || !camRay2WithJac.HasValue;
			}
			while (degenerate && (rejectDegenerate || !camRay2WithJac.HasValue));
			points1.Add(new Vector2d(focal1 * pointInCam1.X / pointInCam1.Z, focal1 * pointInCam1.Y / pointInCam1.Z));
			camRays2WithJac.Add(camRay2WithJac!.Value);
		}
	}

	// Whether at least one model recovers the essential matrix (up to scale/sign) and the
	// unknown focal length, with small residuals on the exact correspondences. Returns a bool
	// rather than asserting, so that FullSphereCalibratedRays can tolerate the solver's
	// intrinsic failure rate over many draws.
	private static bool HasValidModel(
		List<Vector2d> points1,
		List<CamRayWithJac> camRays2WithJac,
		Matrix3d expectedE,
		double expectedFocal,
		List<M_t> models,
		double eEps = 5e-3,
		double focalRelEps = 1e-2,
		double rEps = 1e-2)
	{
		Matrix3d expectedEN = expectedE / expectedE.Norm();
		foreach (M_t model in models)
		{
			Matrix3d e = model.E / model.E.Norm();
			if (Math.Min((e - expectedEN).Norm(), (e + expectedEN).Norm()) > eEps)
			{
				continue;
			}

			if (Math.Abs(model.Focal - expectedFocal) / expectedFocal > focalRelEps)
			{
				continue;
			}

			double[] residuals = new double[points1.Count];
			new RelativePoseOneSidedFocalEstimator().Residuals(points1.ToArray(), camRays2WithJac.ToArray(), model, residuals);
			if (residuals.Any(r => r >= rEps))
			{
				continue;
			}

			return true;
		}

		return false;
	}
}
