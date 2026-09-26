// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometryEstimationTests.Focal: the EstimateTwoViewGeometry.* cases of
// two_view_geometry_test.cc that route to the one-sided focal solver or pair a spherical
// camera with a perspective one, 1:1: OneSidedFocal, SphericalAndCalibratedPerspective,
// SphericalAndUncalibratedPerspective. Helpers shared with them live in
// TwoViewGeometryEstimationTests.Estimate.cs.

using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public partial class TwoViewGeometryEstimationTests
{
	// Asserts that F satisfies x2^T F x1 = 0 on the exact correspondences, in raw image
	// coordinates and in the given order. F is built as K2^-T E K1^-1 and the swap path
	// transposes it again, so a flipped transpose or a K1/K2 swap would survive every other
	// assertion.
	private static void ExpectValidF(ExpectationLog log, Matrix3d f, List<Vector2d> pointsFirst, List<Vector2d> pointsSecond, string what)
	{
		for (int i = 0; i < pointsFirst.Count; ++i)
		{
			Vector3d x1 = pointsFirst[i].Homogeneous();
			Vector3d x2 = pointsSecond[i].Homogeneous();
			Vector3d line = f * x1;
			double lineNorm = new Vector2d(line.X, line.Y).Norm;
			if (!log.True(lineNorm > 0, $"{what}: line norm {i}"))
			{
				return;
			}

			// Distance from x2 to its epipolar line, in pixels.
			log.Near(Math.Abs(x2.Dot(line)) / lineNorm, 0, 1e-6, $"{what}: epipolar distance {i}");
		}
	}

	// Two distinct cameras where exactly one has a known focal length route to the one-sided
	// focal solver. The pair is run in both orders, so that the internal canonicalization
	// (uncalibrated view first, inverted otherwise) is covered in both directions.
	[Test]
	public async Task EstimateTwoViewGeometry_OneSidedFocal()
	{
		const double UncalibFocal = 1280.0;
		const double CalibFocal = 900.0;
		var log = new ExpectationLog();
		foreach (CameraModelId modelId in new[] { CameraModelId.SimplePinhole, CameraModelId.Pinhole })
		{
			Camera uncalibCamera = Camera.CreateFromModelId(1, modelId, UncalibFocal, 2048, 2048);
			uncalibCamera.HasPriorFocalLength = false;
			Camera calibCamera = Camera.CreateFromModelId(2, modelId, CalibFocal, 2048, 2048);
			calibCamera.HasPriorFocalLength = true;

			// No rejection sampling is needed: the known focal removes the coplanar-axes
			// singularity.
			Rigid3d calibFromUncalib = RandomBoundedPose();

			var uncalibPoints = new List<Vector2d>();
			var calibPoints = new List<Vector2d>();
			var matches = new List<FeatureMatch>();
			SynthesizeWideFovCorrespondences(uncalibCamera, calibCamera, calibFromUncalib, uncalibPoints, calibPoints, matches);

			var twoViewGeometryOptions = new TwoViewGeometryOptions { ComputeRelativePose = true };

			// Order 1: the uncalibrated view first, the solver's native orientation.
			{
				TwoViewGeometry geometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
					uncalibCamera, uncalibPoints, calibCamera, calibPoints, matches, twoViewGeometryOptions);
				string what = $"{modelId} order 1";
				log.Equal(geometry.Config, TwoViewGeometry.ConfigurationType.Uncalibrated, $"{what}: config");
				log.True(geometry.E.HasValue, $"{what}: E");
				// Only the uncalibrated side carries estimated intrinsics.
				await Assert.That(geometry.Camera1 is not null).IsTrue();
				log.True(geometry.Camera2 is null, $"{what}: no camera2");
				log.True(geometry.InlierMatches.Count >= matches.Count / 2, $"{what}: inliers");
				log.Near(geometry.Camera1!.FocalLengthX(), UncalibFocal, 0.05 * UncalibFocal, $"{what}: focal");
				await Assert.That(geometry.F.HasValue).IsTrue();
				ExpectValidF(log, geometry.F!.Value, uncalibPoints, calibPoints, what);
				await Assert.That(geometry.Cam2FromCam1.HasValue).IsTrue();
				log.True(NormalizedPoseNear(geometry.Cam2FromCam1!.Value, calibFromUncalib, 1e-2, 1e-1), $"{what}: pose");
			}

			// Order 2: the calibrated view first, exercising the swap-and-invert path.
			{
				TwoViewGeometry geometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
					calibCamera, calibPoints, uncalibCamera, uncalibPoints, matches, twoViewGeometryOptions);
				string what = $"{modelId} order 2";
				log.Equal(geometry.Config, TwoViewGeometry.ConfigurationType.Uncalibrated, $"{what}: config");
				log.True(geometry.E.HasValue, $"{what}: E");
				// The estimated intrinsics must follow the uncalibrated view, now the second.
				log.True(geometry.Camera1 is null, $"{what}: no camera1");
				await Assert.That(geometry.Camera2 is not null).IsTrue();
				log.True(geometry.InlierMatches.Count >= matches.Count / 2, $"{what}: inliers");
				log.Near(geometry.Camera2!.FocalLengthX(), UncalibFocal, 0.05 * UncalibFocal, $"{what}: focal");
				await Assert.That(geometry.F.HasValue).IsTrue();
				ExpectValidF(log, geometry.F!.Value, calibPoints, uncalibPoints, what);
				await Assert.That(geometry.Cam2FromCam1.HasValue).IsTrue();
				log.True(NormalizedPoseNear(geometry.Cam2FromCam1!.Value, calibFromUncalib.Inverse(), 1e-2, 1e-1), $"{what}: pose");
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// Re-projects the synthetic scene's points into image 1 through `sphericalCamera` and
	// into image 2 through `perspectiveCamera`, forming a mixed spherical/perspective pair.
	private static void ProjectMixedSphericalPair(
		Reconstruction reconstruction, Camera sphericalCamera, Camera perspectiveCamera,
		List<Vector2d> points1, List<Vector2d> points2, List<FeatureMatch> matches)
	{
		Image image1 = reconstruction.Image(1);
		Image image2 = reconstruction.Image(2);
		foreach (ulong point3DId in reconstruction.Points3D.Keys.OrderBy(id => id))
		{
			Vector3d xyz = reconstruction.Points3D[point3DId].Xyz;
			Vector2d? xy1 = sphericalCamera.ImgFromCam(image1.CamFromWorld() * xyz);
			Vector2d? xy2 = perspectiveCamera.ImgFromCam(image2.CamFromWorld() * xyz);
			if (xy1 is null || xy2 is null)
			{
				continue;
			}

			matches.Add(new FeatureMatch((uint)points1.Count, (uint)points2.Count));
			points1.Add(xy1.Value);
			points2.Add(xy2.Value);
		}
	}

	// Both sides are calibrated here; see SphericalAndUncalibratedPerspective for the case
	// where they are not.
	[Test]
	public async Task EstimateTwoViewGeometry_SphericalAndCalibratedPerspective()
	{
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(TwoRigOptions(200, true), reconstruction);

		Camera perspectiveCamera = reconstruction.Camera(reconstruction.Image(2).CameraId).Clone();
		Camera sphericalCamera = Camera.CreateFromModelId(100, CameraModelId.Equirectangular, 0.0, 1000, 500);
		await Assert.That(sphericalCamera.IsSpherical).IsTrue();
		await Assert.That(perspectiveCamera.IsPerspective).IsTrue();

		var points1 = new List<Vector2d>();
		var points2 = new List<Vector2d>();
		var matches = new List<FeatureMatch>();
		ProjectMixedSphericalPair(reconstruction, sphericalCamera, perspectiveCamera, points1, points2, matches);
		await Assert.That(matches.Count).IsGreaterThanOrEqualTo(50);

		// A spherical pair whose other camera is also calibrated routes to the bearing-based
		// essential matrix path, committing to CALIBRATED without estimating a fundamental
		// matrix.
		TwoViewGeometry geometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
			sphericalCamera, points1, perspectiveCamera, points2, matches, new TwoViewGeometryOptions());
		var log = new ExpectationLog();
		log.Equal(geometry.Config, TwoViewGeometry.ConfigurationType.Calibrated, "config");
		log.True(geometry.E.HasValue, "E");
		log.False(geometry.F.HasValue, "F");
		log.True(geometry.H.HasValue, "H");
		log.True(geometry.InlierMatches.Count >= matches.Count / 2, "inliers");
		await Assert.That(log.Failures).IsEmpty();
	}

	// A spherical camera paired with a perspective camera whose focal is unknown routes to
	// the one-sided focal solver, and the perspective camera's focal is recovered.
	[Test]
	public async Task EstimateTwoViewGeometry_SphericalAndUncalibratedPerspective()
	{
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(TwoRigOptions(200, true), reconstruction);

		Image image1 = reconstruction.Image(1);
		Image image2 = reconstruction.Image(2);
		Camera trueCamera = reconstruction.Camera(image2.CameraId).Clone();
		double trueFocal = trueCamera.FocalLengthX();

		// The estimator is handed a camera carrying a deliberately wrong focal and no prior,
		// so the recovered focal cannot merely echo its input.
		Camera uncalibCamera = trueCamera.Clone();
		uncalibCamera.HasPriorFocalLength = false;
		uncalibCamera.SetFocalLength(0.5 * trueFocal);

		Camera sphericalCamera = Camera.CreateFromModelId(100, CameraModelId.Equirectangular, 0.0, 1000, 500);
		await Assert.That(sphericalCamera.IsSpherical).IsTrue();

		var sphericalPoints = new List<Vector2d>();
		var perspectivePoints = new List<Vector2d>();
		var matches = new List<FeatureMatch>();
		ProjectMixedSphericalPair(reconstruction, sphericalCamera, trueCamera, sphericalPoints, perspectivePoints, matches);
		await Assert.That(matches.Count).IsGreaterThanOrEqualTo(50);

		Rigid3d perspectiveFromSpherical = image2.CamFromWorld() * image1.CamFromWorld().Inverse();

		var twoViewGeometryOptions = new TwoViewGeometryOptions { ComputeRelativePose = true };
		var log = new ExpectationLog();

		// A spherical view has no image plane, so there is no F or H to publish. Both orders
		// are run; the spherical-first case exercises the internal swap.
		{
			TwoViewGeometry geometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
				sphericalCamera, sphericalPoints, uncalibCamera, perspectivePoints, matches, twoViewGeometryOptions);
			log.Equal(geometry.Config, TwoViewGeometry.ConfigurationType.Uncalibrated, "order 1: config");
			log.True(geometry.E.HasValue, "order 1: E");
			log.False(geometry.F.HasValue, "order 1: F");
			log.False(geometry.H.HasValue, "order 1: H");
			// The estimated intrinsics follow the uncalibrated view, here the second.
			log.True(geometry.Camera1 is null, "order 1: no camera1");
			await Assert.That(geometry.Camera2 is not null).IsTrue();
			log.Near(geometry.Camera2!.FocalLengthX(), trueFocal, 0.05 * trueFocal, "order 1: focal");
			log.True(geometry.InlierMatches.Count >= matches.Count / 2, "order 1: inliers");
			await Assert.That(geometry.Cam2FromCam1.HasValue).IsTrue();
			log.True(NormalizedPoseNear(geometry.Cam2FromCam1!.Value, perspectiveFromSpherical, 1e-2, 1e-1), "order 1: pose");
		}

		{
			var swappedMatches = matches.Select(match => new FeatureMatch(match.Point2DIdx2, match.Point2DIdx1)).ToList();
			TwoViewGeometry geometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
				uncalibCamera, perspectivePoints, sphericalCamera, sphericalPoints, swappedMatches, twoViewGeometryOptions);
			log.Equal(geometry.Config, TwoViewGeometry.ConfigurationType.Uncalibrated, "order 2: config");
			log.True(geometry.E.HasValue, "order 2: E");
			log.False(geometry.F.HasValue, "order 2: F");
			log.False(geometry.H.HasValue, "order 2: H");
			await Assert.That(geometry.Camera1 is not null).IsTrue();
			log.True(geometry.Camera2 is null, "order 2: no camera2");
			log.Near(geometry.Camera1!.FocalLengthX(), trueFocal, 0.05 * trueFocal, "order 2: focal");
			await Assert.That(geometry.Cam2FromCam1.HasValue).IsTrue();
			log.True(NormalizedPoseNear(geometry.Cam2FromCam1!.Value, perspectiveFromSpherical.Inverse(), 1e-2, 1e-1), "order 2: pose");
		}

		await Assert.That(log.Failures).IsEmpty();
	}
}
