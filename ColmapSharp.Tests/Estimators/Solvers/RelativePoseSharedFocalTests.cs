// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RelativePoseSharedFocalTests: colmap/estimators/solvers/relpose_shared_focal_test.cc ported
// 1:1 (test names are Suite_Name). Same problem generator, loop counts, tolerances and
// failure-rate bounds. Tests ColmapSharp/Estimators/Solvers/RelativePoseSharedFocal.cs and
// PoseLib/Relpose6ptSharedFocal.cs. Tier B for Estimate/Residuals/IsFocalIdentifiable,
// Tier C for Refine.

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using M_t = ColmapSharp.Estimators.Solvers.RelativePoseSharedFocalEstimator.Model;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class RelativePoseSharedFocalTests
{
	// Rejection thresholds used to condition a minimal sample: minimum depth in front of
	// either camera and minimum parallax (sin^2 of the ray angle).
	private const double kMinDepth = 0.5;
	private const double kMinParallax = 1e-2;  // ~5.7 degrees.

	// Maximum fraction of samples that may fail. The minimal polynomial solve does not
	// succeed on every sample: it loses the true root, or returns it imprecisely, for ~0.15%
	// of samples. A 100-trial run therefore almost always observes a failure rate of 0% or
	// 1%, and never exceeded 3% over 500 measured runs, while a real regression exceeds it
	// immediately.
	private const int kNumTrials = 100;
	private const double kMaxFailureRate = 0.03;

	// The minimal 6-point solver recovers the pose and focal on clean samples.
	[Test]
	public async Task RelativePoseSharedFocalEstimator_Nominal()
	{
		const double kFocal = 1000.0;
		int numFailures = 0;
		var estimator = new RelativePoseSharedFocalEstimator();
		for (int k = 0; k < kNumTrials; ++k)
		{
			Rigid3d cam2FromCam1 = TestCam2FromCam1();
			Matrix3d expectedE = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
			var points1 = new List<Vector2d>();
			var points2 = new List<Vector2d>();
			RandomSharedFocalCorrespondences(
				cam2FromCam1, kFocal, RelativePoseSharedFocalEstimator.MinNumSamples, rejectDegenerate: true, points1, points2);

			var models = new List<M_t>();
			estimator.Estimate(points1.ToArray(), points2.ToArray(), models);

			if (!HasValidModel(points1, points2, expectedE, kFocal, models))
			{
				++numFailures;
			}
		}

		await Assert.That((double)numFailures / kNumTrials).IsLessThanOrEqualTo(kMaxFailureRate);
	}

	// Residuals are near-zero on exact points, grow with a wrong focal, and are infinite for
	// a non-positive focal.
	[Test]
	public async Task RelativePoseSharedFocalEstimator_Residuals()
	{
		const double kFocal = 1000.0;
		Rigid3d cam2FromCam1 = TestCam2FromCam1();
		var points1 = new List<Vector2d>();
		var points2 = new List<Vector2d>();
		RandomSharedFocalCorrespondences(cam2FromCam1, kFocal, 30, rejectDegenerate: false, points1, points2);

		var estimator = new RelativePoseSharedFocalEstimator();
		var log = new ExpectationLog();
		var model = new M_t(EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1), kFocal);
		double[] residuals = new double[points1.Count];
		estimator.Residuals(points1.ToArray(), points2.ToArray(), model, residuals);
		double sumExact = 0.0;
		foreach (double residual in residuals)
		{
			log.Less(residual, 1e-2, "exact residual");
			sumExact += residual;
		}

		M_t wrongModel = model with { Focal = 1.5 * kFocal };
		double[] wrongResiduals = new double[points1.Count];
		estimator.Residuals(points1.ToArray(), points2.ToArray(), wrongModel, wrongResiduals);
		double sumWrong = wrongResiduals.Sum();
		log.Greater(sumWrong, sumExact, "sum_wrong");

		M_t invalidModel = model with { Focal = 0.0 };
		double[] invalidResiduals = new double[points1.Count];
		estimator.Residuals(points1.ToArray(), points2.ToArray(), invalidModel, invalidResiduals);
		foreach (double residual in invalidResiduals)
		{
			log.Equal(residual, double.MaxValue, "invalid residual");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// Refinement pulls a perturbed pose + focal back to the ground truth.
	[Test]
	public async Task RelativePoseSharedFocalEstimator_RefineFromInitialModel()
	{
		const double kFocal = 1000.0;
		var log = new ExpectationLog();
		for (int k = 0; k < 50; ++k)
		{
			Rigid3d cam2FromCam1 = TestCam2FromCam1();
			Matrix3d expectedE = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
			var points1 = new List<Vector2d>();
			var points2 = new List<Vector2d>();
			RandomSharedFocalCorrespondences(cam2FromCam1, kFocal, 50, rejectDegenerate: false, points1, points2);

			Quaterniond seedRotation = cam2FromCam1.Rotation *
				new AngleAxisd(0.02, RandomEigen.RandomEigenVector3d().Normalized()).ToQuaternion();
			Vector3d seedTranslation = (cam2FromCam1.Translation + 0.02 * RandomEigen.RandomEigenVector3d()).Normalized();
			var model = new M_t(EssentialMatrix.EssentialMatrixFromPose(new Rigid3d(seedRotation, seedTranslation)), 1.1 * kFocal);

			// ASSERT_TRUE.
			await Assert.That(RelativePoseSharedFocalEstimator.Refine(points1.ToArray(), points2.ToArray(), ref model)).IsTrue();

			// Refinement is a nonlinear least squares over 50 exact points seeded near the
			// solution, not a minimal polynomial solve, so it is expected to succeed on every
			// draw.
			log.True(
				HasValidModel(points1, points2, expectedE, kFocal, [model], eEps: 1e-3, focalRelEps: 1e-2, rEps: 1e-2),
				$"k={k}: HasValidModel");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// The focal is unidentifiable for parallel axes and for coplanar axes meeting at an
	// isosceles configuration, but identifiable for coplanar axes meeting asymmetrically, and
	// for skew axes.
	[Test]
	public async Task RelativePoseSharedFocalEstimator_IsFocalIdentifiable()
	{
		var fixation = new Vector3d(0, 0, 2.0);
		var log = new ExpectationLog();

		// Both centers 2.0 from the fixation point: isosceles, unidentifiable.
		log.False(
			RelativePoseSharedFocalEstimator.IsFocalIdentifiable(FixatingPose(fixation, dist2: 2.0, angleAtFixationDeg: 40.0)),
			"isosceles");

		// Same axes, but cam2 much closer to the fixation point than cam1. Still coplanar, yet
		// far from isosceles, so the focal is constrained. This is the case a pure coplanarity
		// criterion would wrongly reject.
		log.True(
			RelativePoseSharedFocalEstimator.IsFocalIdentifiable(FixatingPose(fixation, dist2: 1.0, angleAtFixationDeg: 40.0)),
			"asymmetric");

		// Parallel axes with a lateral baseline: coplanar, no intersection.
		var parallel = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0));
		log.False(RelativePoseSharedFocalEstimator.IsFocalIdentifiable(parallel), "parallel");

		// 90 deg about x with a lateral baseline: maximally skew axes.
		var skew = new Rigid3d(
			new AngleAxisd(MathUtils.DegToRad(90.0), new Vector3d(1, 0, 0)).ToQuaternion(), new Vector3d(1, 0, 0));
		log.True(RelativePoseSharedFocalEstimator.IsFocalIdentifiable(skew), "skew");

		// Pure rotation: no baseline at all.
		log.False(RelativePoseSharedFocalEstimator.IsFocalIdentifiable(new Rigid3d(skew.Rotation, Vector3d.Zero)), "pure rotation");

		// Scaling the translation scales the whole configuration, leaving both predicates
		// unchanged.
		foreach (Rigid3d pose in new[] { FixatingPose(fixation, 2.0, 40.0), FixatingPose(fixation, 1.0, 40.0) })
		{
			var scaled = new Rigid3d(pose.Rotation, 1000.0 * pose.Translation);
			log.Equal(
				RelativePoseSharedFocalEstimator.IsFocalIdentifiable(scaled),
				RelativePoseSharedFocalEstimator.IsFocalIdentifiable(pose),
				"scaled");
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// Random relative pose with a unit-norm baseline (away from the pure-rotation degeneracy)
	// and a bounded rotation. The rotation is bounded so the two view frustums overlap: this
	// keeps the sampled points cheirality-consistent (in front of both cameras) without the
	// rejection sampler spinning for near-opposite orientations.
	private static Rigid3d TestCam2FromCam1()
	{
		const double maxAngleDeg = 60.0;
		// Resample until the shared focal is identifiable (the same precondition the estimator
		// enforces via IsFocalIdentifiable). For a singular pose the focal is unrecoverable and
		// the minimal solver returns a meaningless focal, which no downstream assertion can
		// meaningfully check.
		while (true)
		{
			Vector3d axis = RandomEigen.RandomEigenVector3d().Normalized();
			// clang evaluates the Rigid3d constructor arguments left to right.
			double angle = MathUtils.DegToRad(RandomUtils.RandomUniformReal(0.0, maxAngleDeg));
			var rotation = new AngleAxisd(angle, axis).ToQuaternion();
			Vector3d translation = RandomEigen.RandomEigenVector3d().Normalized();
			var cam2FromCam1 = new Rigid3d(rotation, translation);
			if (RelativePoseSharedFocalEstimator.IsFocalIdentifiable(cam2FromCam1))
			{
				return cam2FromCam1;
			}
		}
	}

	// Generates principal-point-centered image point pairs (f * X / Z) for a shared focal
	// length `focal`, from random 3D points in front of both cameras. When rejectDegenerate
	// is set, resamples points that make a minimal 6-point solve ill-conditioned.
	private static void RandomSharedFocalCorrespondences(
		Rigid3d cam2FromCam1, double focal, int numPoints, bool rejectDegenerate, List<Vector2d> points1, List<Vector2d> points2)
	{
		for (int i = 0; i < numPoints; ++i)
		{
			Vector3d pointInCam1;
			Vector3d pointInCam2;
			bool degenerate;
			do
			{
				// Point in front of cam1 with a moderate field of view (|x/z|, |y/z| <= ~0.5):
				// wide-angle points span a large magnitude range that degrades the conditioning
				// of the minimal solve.
				Vector3d ray1 = RandomEigen.RandomEigenVector3d();
				ray1 = new Vector3d(ray1.X, ray1.Y, Math.Abs(ray1.Z) + 2.0);
				double depth = RandomUtils.RandomUniformReal(1.0, 3.0);
				pointInCam1 = depth * ray1.Normalized();
				pointInCam2 = cam2FromCam1 * pointInCam1;
				Vector3d ray1InCam2 = cam2FromCam1.Rotation * pointInCam1.Normalized();
				double cosParallax = ray1InCam2.Dot(pointInCam2.Normalized());
				// The point is always in front of cam1; require it in front of cam2 too, with
				// sufficient parallax.
				degenerate = pointInCam2.Z < kMinDepth || 1.0 - cosParallax * cosParallax < kMinParallax;
			}
			while (rejectDegenerate && degenerate);
			points1.Add(new Vector2d(focal * pointInCam1.X / pointInCam1.Z, focal * pointInCam1.Y / pointInCam1.Z));
			points2.Add(new Vector2d(focal * pointInCam2.X / pointInCam2.Z, focal * pointInCam2.Y / pointInCam2.Z));
		}
	}

	// Whether at least one model recovers the essential matrix (up to scale/sign) and the
	// focal length, with small residuals on the exact points. Returns a bool rather than
	// asserting, so callers can tolerate the solver's intrinsic failure rate over many draws;
	// see kMaxFailureRate.
	private static bool HasValidModel(
		List<Vector2d> points1,
		List<Vector2d> points2,
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
			new RelativePoseSharedFocalEstimator().Residuals(points1.ToArray(), points2.ToArray(), model, residuals);
			if (residuals.Any(r => r >= rEps))
			{
				continue;
			}

			return true;
		}

		return false;
	}

	// A relative pose whose two optical axes intersect at a common fixation point on cam1's
	// +z axis, so the axes are coplanar. cam1 sits at distance |fixation| from that point and
	// cam2 at dist2, so the configuration is isosceles iff dist2 == |fixation|.
	private static Rigid3d FixatingPose(Vector3d fixation, double dist2, double angleAtFixationDeg)
	{
		double angle = MathUtils.DegToRad(angleAtFixationDeg);
		Vector3d center2 = fixation + dist2 * new Vector3d(-Math.Sin(angle), 0, -Math.Cos(angle));
		Quaterniond rotation = Quaterniond.FromTwoVectors((fixation - center2).Normalized(), new Vector3d(0, 0, 1));
		return new Rigid3d(rotation, rotation * -center2);
	}
}
