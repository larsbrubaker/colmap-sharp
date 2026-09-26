// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TwoViewGeometryEstimationTests.Estimate: the EstimateTwoViewGeometry.* cases of
// two_view_geometry_test.cc whose scenes come from SynthesizeDataset or are built from
// projected points, 1:1. TwoViewGeometryEstimationTests.cs holds the file notes and the
// pose cases; TwoViewGeometryEstimationTests.Multiple.cs the forced-homography,
// seed-determinism and multiple-model cases.
//
// Translation note: C++ evaluates Rigid3d(Quaterniond(AngleAxisd(DegToRad(RandomUniformReal
// ...), axis)), RandomEigenVectord<3>().normalized()) left to right under clang (the macOS
// build this port matches), so the angle is drawn before the translation here too.

using ColmapSharp.Estimators;
using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Estimators;

public partial class TwoViewGeometryEstimationTests
{
	private sealed class TwoViewGeometryTestData
	{
		public Camera Camera1 = null!;
		public Camera Camera2 = null!;
		public List<Vector2d> Points1 = [];
		public List<Vector2d> Points2 = [];
		public List<FeatureMatch> Matches = [];

		// Ground-truth relative pose (cam2_from_cam1) of the synthetic dataset.
		public Rigid3d Cam2FromCam1;
	}

	private static TwoViewGeometryTestData CreateTwoViewGeometryTestData(
		SyntheticDatasetOptions syntheticDatasetOptions, SyntheticNoiseOptions? syntheticNoiseOptions = null)
	{
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);
		Synthetic.SynthesizeNoise(syntheticNoiseOptions ?? new SyntheticNoiseOptions(), reconstruction);

		if (reconstruction.NumImages != 2)
		{
			throw new InvalidOperationException("Check failed: reconstruction.NumImages() == 2");
		}

		Image image1 = reconstruction.Image(1);
		Image image2 = reconstruction.Image(2);

		var data = new TwoViewGeometryTestData
		{
			Camera1 = reconstruction.Camera(image1.CameraId).Clone(),
			Camera2 = reconstruction.Camera(image2.CameraId).Clone(),
			Cam2FromCam1 = image2.CamFromWorld() * image1.CamFromWorld().Inverse(),
		};

		ExtractPointsAndMatches(reconstruction, image1, image2, data.Points1, data.Points2, [], data.Matches);
		return data;
	}

	private static SyntheticDatasetOptions TwoRigOptions(int numPoints3D, bool hasPriorFocalLength) => new()
	{
		NumRigs = 2,
		NumCamerasPerRig = 1,
		NumFramesPerRig = 1,
		NumPoints3D = numPoints3D,
		CameraHasPriorFocalLength = hasPriorFocalLength,
	};

	private static bool NormalizedPoseNear(Rigid3d actual, Rigid3d expected, double rtol, double ttol) =>
		Rigid3dNear(
			new Rigid3d(actual.Rotation, actual.Translation.Normalized()),
			new Rigid3d(expected.Rotation, expected.Translation.Normalized()),
			rtol,
			ttol);

	[Test]
	public async Task EstimateTwoViewGeometry_Spherical()
	{
		SyntheticDatasetOptions syntheticDatasetOptions = TwoRigOptions(200, false);
		syntheticDatasetOptions.CameraModelId = CameraModelId.Equirectangular;
		syntheticDatasetOptions.CameraWidth = 1000;
		syntheticDatasetOptions.CameraHeight = 500;
		syntheticDatasetOptions.CameraParams = [1000, 500];
		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(syntheticDatasetOptions);
		await Assert.That(testData.Camera1.IsPerspective).IsFalse();
		await Assert.That(testData.Camera2.IsPerspective).IsFalse();

		var twoViewGeometryOptions = new TwoViewGeometryOptions { ComputeRelativePose = true };

		// Spherical cameras have no pinhole image plane, so the fundamental matrix is not
		// estimated; the pair is classified from the bearing-based essential matrix and a
		// ray-space homography.
		TwoViewGeometry geometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
			testData.Camera1, testData.Points1, testData.Camera2, testData.Points2, testData.Matches, twoViewGeometryOptions);
		var log = new ExpectationLog();
		log.Equal(geometry.Config, TwoViewGeometry.ConfigurationType.Calibrated, "config");
		log.True(geometry.E.HasValue, "E");
		log.False(geometry.F.HasValue, "F");
		log.True(geometry.H.HasValue, "H");
		log.True(geometry.InlierMatches.Count >= testData.Matches.Count / 2, "inliers");

		// The recovered relative pose should match the ground truth: rotation exactly,
		// translation up to scale, so compare with normalized translations.
		await Assert.That(geometry.Cam2FromCam1.HasValue).IsTrue();
		log.True(NormalizedPoseNear(geometry.Cam2FromCam1!.Value, testData.Cam2FromCam1, 1e-3, 1e-2), "pose");

		// EstimateCalibratedTwoViewGeometry delegates to the spherical path rather than
		// estimating a meaningless fundamental matrix.
		TwoViewGeometry calibratedGeometry = TwoViewGeometryEstimation.EstimateCalibratedTwoViewGeometry(
			testData.Camera1, testData.Points1, testData.Camera2, testData.Points2, testData.Matches, twoViewGeometryOptions);
		log.Equal(calibratedGeometry.Config, TwoViewGeometry.ConfigurationType.Calibrated, "calibrated config");
		log.True(calibratedGeometry.E.HasValue, "calibrated E");
		log.False(calibratedGeometry.F.HasValue, "calibrated F");
		log.True(calibratedGeometry.H.HasValue, "calibrated H");
		await Assert.That(log.Failures).IsEmpty();
	}

	private static SyntheticDatasetOptions FisheyeOptions(bool hasPriorFocalLength)
	{
		SyntheticDatasetOptions options = TwoRigOptions(200, hasPriorFocalLength);
		options.CameraModelId = CameraModelId.OpenCVFisheye;
		options.CameraParams = [1280, 1280, 512, 384, 0, 0, 0, 0];
		return options;
	}

	[Test]
	public async Task EstimateTwoViewGeometry_UncalibratedFisheyeIsDegenerate()
	{
		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(FisheyeOptions(false));
		await Assert.That(testData.Camera1.IsPerspectiveFisheye).IsTrue();
		await Assert.That(testData.Camera1.HasPriorFocalLength).IsFalse();

		// Without a focal length prior the only remaining model is the fundamental matrix,
		// which assumes a pinhole projection that a fisheye camera does not have. The pair is
		// therefore rejected rather than fit with a meaningless fundamental matrix.
		TwoViewGeometry geometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
			testData.Camera1, testData.Points1, testData.Camera2, testData.Points2, testData.Matches, new TwoViewGeometryOptions());
		await Assert.That(geometry.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Degenerate);
		await Assert.That(geometry.F.HasValue).IsFalse();
		await Assert.That(geometry.H.HasValue).IsFalse();
	}

	[Test]
	public async Task EstimateTwoViewGeometry_CalibratedFisheyeIsEstimated()
	{
		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(FisheyeOptions(true));
		await Assert.That(testData.Camera1.IsPerspectiveFisheye).IsTrue();
		await Assert.That(testData.Camera1.HasPriorFocalLength).IsTrue();

		// With a known focal length the calibrated path applies, which works on bearing
		// vectors and is therefore valid for fisheye cameras.
		TwoViewGeometry geometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
			testData.Camera1, testData.Points1, testData.Camera2, testData.Points2, testData.Matches, new TwoViewGeometryOptions());
		await Assert.That(geometry.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Calibrated);
		await Assert.That(geometry.E.HasValue).IsTrue();
		await Assert.That(geometry.InlierMatches.Count).IsGreaterThanOrEqualTo(testData.Matches.Count / 2);
	}

	[Test]
	public async Task EstimateTwoViewGeometry_ForceHUseWithFisheyeIsDegenerate()
	{
		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(FisheyeOptions(true));
		await Assert.That(testData.Camera1.IsPerspectiveFisheye).IsTrue();

		// A homography only relates two images of a plane under a pinhole projection, so it
		// is not estimated for fisheye cameras even when the caller explicitly asks for it.
		var twoViewGeometryOptions = new TwoViewGeometryOptions { ForceHUse = true };
		TwoViewGeometry geometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
			testData.Camera1, testData.Points1, testData.Camera2, testData.Points2, testData.Matches, twoViewGeometryOptions);
		await Assert.That(geometry.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Degenerate);
		await Assert.That(geometry.H.HasValue).IsFalse();
	}

	// A pure-rotation pair (one rig, two cameras without sensor-from-rig translation).
	private static async Task RunPanoramicTest(
		CameraModelId modelId, int? width, int? height, double[] cameraParams, bool hasPriorFocalLength, bool expectEAndNoF,
		Func<Camera, bool> cameraPrecondition)
	{
		RandomUtils.SetPRNGSeed(42);
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 2,
			NumFramesPerRig = 1,
			NumPoints3D = 500,
			CameraModelId = modelId,
			CameraParams = cameraParams,
			CameraHasPriorFocalLength = hasPriorFocalLength,
			SensorFromRigTranslationStddev = 0,
		};
		if (width is int w && height is int h)
		{
			syntheticDatasetOptions.CameraWidth = w;
			syntheticDatasetOptions.CameraHeight = h;
		}

		TwoViewGeometryTestData testData = CreateTwoViewGeometryTestData(syntheticDatasetOptions);
		await Assert.That(cameraPrecondition(testData.Camera1)).IsTrue();

		var twoViewGeometryOptions = new TwoViewGeometryOptions { ComputeRelativePose = true };
		twoViewGeometryOptions.RansacOptions.RandomSeed = 42;
		TwoViewGeometry geometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
			testData.Camera1, testData.Points1, testData.Camera2, testData.Points2, testData.Matches, twoViewGeometryOptions);
		var log = new ExpectationLog();
		log.Equal(geometry.Config, TwoViewGeometry.ConfigurationType.Panoramic, "config");
		if (expectEAndNoF)
		{
			log.True(geometry.E.HasValue, "E");
		}

		log.True(geometry.H.HasValue, "H");
		if (expectEAndNoF)
		{
			log.False(geometry.F.HasValue, "F");
		}

		log.True(geometry.InlierMatches.Count >= testData.Matches.Count / 2, "inliers");
		await Assert.That(log.Failures).IsEmpty();
	}

	// A rotating 360 degree camera is the usual capture mode, and the essential matrix
	// degenerates there, so the spherical path must catch it.
	[Test]
	public async Task EstimateTwoViewGeometry_PanoramicWithSphericalCameraIsDetected()
	{
		await RunPanoramicTest(CameraModelId.Equirectangular, 1000, 500, [1000, 500], false, expectEAndNoF: true,
			camera => camera.IsSpherical);
	}

	// With a strongly distorted camera the pixel-space homography cannot fit the
	// correspondences at all; estimating it on bearing rays restores the fit and with it the
	// PLANAR_OR_PANORAMIC label.
	[Test]
	public async Task EstimateTwoViewGeometry_PanoramicWithDistortedCameraIsDetected()
	{
		await RunPanoramicTest(CameraModelId.OpenCV, null, null, [1280, 1280, 512, 384, 0.15, -0.03, 0.001, 0.001], true, expectEAndNoF: false,
			camera => camera.IsPerspectivePinhole && !camera.IsUndistorted());
	}

	// As above for a fisheye camera, which reaches the calibrated path because it
	// short-circuits only spherical models.
	[Test]
	public async Task EstimateTwoViewGeometry_PanoramicWithFisheyeCameraIsDetected()
	{
		await RunPanoramicTest(CameraModelId.OpenCVFisheye, null, null, [1280, 1280, 512, 384, 0.05, -0.01, 0, 0], true, expectEAndNoF: false,
			camera => camera.IsPerspectiveFisheye);
	}

	// Ground-truth relative pose with a unit baseline and a bounded rotation (20-60 deg).
	private static Rigid3d RandomBoundedPose()
	{
		Vector3d axis = RandomEigen.RandomEigenVector3d().Normalized();
		Quaterniond rotation = new AngleAxisd(MathUtils.DegToRad(RandomUtils.RandomUniformReal(20.0, 60.0)), axis).ToQuaternion();
		return new Rigid3d(rotation, RandomEigen.RandomEigenVector3d().Normalized());
	}

	// 200 exact correspondences of points in front of the first view with a moderate field
	// of view (|x/z|, |y/z| <= ~0.5), kept when the second view also sees them.
	private static void SynthesizeWideFovCorrespondences(
		Camera firstCamera, Camera secondCamera, Rigid3d secondFromFirst,
		List<Vector2d> points1, List<Vector2d> points2, List<FeatureMatch> matches)
	{
		while (points1.Count < 200)
		{
			Vector3d dir = RandomEigen.RandomEigenVector3d();
			dir = new Vector3d(dir.X, dir.Y, Math.Abs(dir.Z) + 2.0);
			Vector3d pointInFirst = RandomUtils.RandomUniformReal(2.0, 5.0) * dir.Normalized();
			Vector3d pointInSecond = secondFromFirst * pointInFirst;
			if (pointInSecond.Z < 0.5)
			{
				continue;  // Require cheirality in the second view (the first holds by construction).
			}

			Vector2d? xy1 = firstCamera.ImgFromCam(pointInFirst);
			Vector2d? xy2 = secondCamera.ImgFromCam(pointInSecond);
			if (xy1 is null || xy2 is null)
			{
				continue;
			}

			uint idx = (uint)points1.Count;
			points1.Add(xy1.Value);
			points2.Add(xy2.Value);
			matches.Add(new FeatureMatch(idx, idx));
		}
	}

	// A single shared, uncalibrated, pinhole-projection camera observed from two frames
	// routes to the shared-focal solver, which jointly recovers the relative pose and the
	// unknown focal length, for SIMPLE_PINHOLE and (isotropic) PINHOLE. A dedicated
	// wide-baseline, wide field-of-view scene is built, as focal-from-two-views needs points
	// spread well off the optical axis.
	[Test]
	public async Task EstimateTwoViewGeometry_SharedFocal()
	{
		const double Focal = 1280.0;
		var log = new ExpectationLog();
		foreach (CameraModelId modelId in new[] { CameraModelId.SimplePinhole, CameraModelId.Pinhole })
		{
			Camera camera = Camera.CreateFromModelId(1, modelId, Focal, 2048, 2048);
			camera.HasPriorFocalLength = false;
			await Assert.That(camera.IsPerspective).IsTrue();

			// The focal is unidentifiable for singular optical axis configurations, so
			// resample until the pose is clear of that degeneracy.
			Rigid3d cam2FromCam1;
			do
			{
				cam2FromCam1 = RandomBoundedPose();
			}
			while (!RelativePoseSharedFocalEstimator.IsFocalIdentifiable(cam2FromCam1));

			var points1 = new List<Vector2d>();
			var points2 = new List<Vector2d>();
			var matches = new List<FeatureMatch>();
			SynthesizeWideFovCorrespondences(camera, camera, cam2FromCam1, points1, points2, matches);

			var twoViewGeometryOptions = new TwoViewGeometryOptions { ComputeRelativePose = true };
			TwoViewGeometry geometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
				camera, points1, camera, points2, matches, twoViewGeometryOptions);

			log.Equal(geometry.Config, TwoViewGeometry.ConfigurationType.Uncalibrated, $"{modelId}: config");
			log.True(geometry.E.HasValue, $"{modelId}: E");
			await Assert.That(geometry.Camera1 is not null).IsTrue();
			await Assert.That(geometry.Camera2 is not null).IsTrue();
			log.True(geometry.Camera1 == geometry.Camera2, $"{modelId}: camera1 == camera2");
			log.True(geometry.InlierMatches.Count >= matches.Count / 2, $"{modelId}: inliers");
			log.Near(geometry.Camera1!.FocalLengthX(), Focal, 0.05 * Focal, $"{modelId}: focal");
			await Assert.That(geometry.Cam2FromCam1.HasValue).IsTrue();
			log.True(NormalizedPoseNear(geometry.Cam2FromCam1!.Value, cam2FromCam1, 1e-2, 1e-1), $"{modelId}: pose");
		}

		await Assert.That(log.Failures).IsEmpty();
	}
}
