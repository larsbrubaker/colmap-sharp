// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SiftMatcherGuidedTests: the MatchGuidedSiftFeaturesCPU cases of colmap/feature/sift_test.cc,
// 1:1 - TypeMismatch, Nominal, EssentialMatrix, Spherical, SphericalMixedHemispheres,
// UnprojectableKeypoints, SharedFocal and SharedFocalPerPairFocal, with the shared test bodies
// (TestGuidedMatching*) as private methods taking the matcher factory. Tests
// ColmapSharp/Feature/SiftMatcher.cs and SiftGuidedFilters.cs. The GPU variants are skipped
// (SiftGPU, excluded; see SiftMatcherTests.cs).
//
// The premise checks inside the C++ `project` lambdas (EXPECT_TRUE / EXPECT_FALSE on
// CamFromImg) are collected and asserted after the call, since a lambda cannot await.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class SiftMatcherGuidedTests
{
	private static FeatureKeypoint Kp(Vector2d p) => new((float)p.X, (float)p.Y);

	private static FeatureMatcherImage Image(uint id, Camera camera, FeatureDescriptors descriptors, params FeatureKeypoint[] keypoints) =>
		new() { ImageId = id, Camera = camera, Keypoints = keypoints.ToList(), Descriptors = descriptors };

	// Eigen::Vector2d -> Vector2f (cast<float>), kept as floats in a Vector2d for Kp.
	private static Vector2d ToFloat(Vector2d p) => new((float)p.X, (float)p.Y);

	[Test]
	public async Task MatchGuidedSiftFeaturesCPU_TypeMismatch()
	{
		Camera camera = SiftMatcherTestUtils.SimplePinhole100();

		FeatureDescriptors siftDesc = SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2);
		await Assert.That(siftDesc.Type).IsEqualTo(FeatureExtractorType.Sift);

		FeatureDescriptors undefinedDesc = SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2);
		undefinedDesc.Type = FeatureExtractorType.Undefined;

		FeatureMatcherImage imageSift = Image(1, camera, siftDesc, new FeatureKeypoint(1, 0), new FeatureKeypoint(2, 0));
		FeatureMatcherImage imageUndefined = Image(2, camera, undefinedDesc, new FeatureKeypoint(2, 0), new FeatureKeypoint(1, 0));

		var options = new FeatureMatchingOptions(FeatureMatcherType.SiftBruteForce);
		options.Sift.CpuBruteForceMatcher = true;
		FeatureMatcher matcher = SiftFeatureMatchers.CreateSiftFeatureMatcher(options);

		TwoViewGeometry twoViewGeometry = SiftMatcherTestUtils.CreatePlanarTwoViewGeometry();

		Assert.Throws<ArgumentException>(() => matcher.MatchGuided(1.0, imageSift, imageUndefined, twoViewGeometry));
	}

	[Test]
	public async Task MatchGuidedSiftFeaturesCPU_Nominal()
	{
		Camera camera = SiftMatcherTestUtils.SimplePinhole100();
		FeatureMatcherImage image0 = Image(0, camera, SiftMatcherTestUtils.CreateEmptyDescriptors());
		FeatureMatcherImage image1 = Image(
			1, camera, SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2), new FeatureKeypoint(1, 0), new FeatureKeypoint(2, 0));
		FeatureMatcherImage image2 = Image(
			2, camera, SiftMatcherTestUtils.CreateReversedDescriptors(image1.Descriptors!), new FeatureKeypoint(2, 0), new FeatureKeypoint(1, 0));
		FeatureMatcherImage image3 = Image(
			3, camera, SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2), new FeatureKeypoint(100, 0), new FeatureKeypoint(2, 0));

		TwoViewGeometry twoViewGeometry = SiftMatcherTestUtils.CreatePlanarTwoViewGeometry();

		FeatureMatcher matcher = SiftMatcherTestUtils.CreateCpuMatcherWithIndexCache([image0, image1, image2, image3]);

		const double kMaxError = 1.0;

		matcher.MatchGuided(kMaxError, image1, image2, twoViewGeometry);
		await SiftMatcherTestUtils.ExpectReversedInlierMatches(twoViewGeometry);

		matcher.MatchGuided(kMaxError, image3, image2, twoViewGeometry);
		await Assert.That(twoViewGeometry.InlierMatches.Count).IsEqualTo(1);
		await Assert.That(twoViewGeometry.InlierMatches[0].Point2DIdx1).IsEqualTo(1u);
		await Assert.That(twoViewGeometry.InlierMatches[0].Point2DIdx2).IsEqualTo(0u);

		matcher.MatchGuided(kMaxError, image0, image2, twoViewGeometry);
		await Assert.That(twoViewGeometry.InlierMatches.Count).IsEqualTo(0);
		matcher.MatchGuided(kMaxError, image1, image0, twoViewGeometry);
		await Assert.That(twoViewGeometry.InlierMatches.Count).IsEqualTo(0);
		matcher.MatchGuided(kMaxError, image0, image0, twoViewGeometry);
		await Assert.That(twoViewGeometry.InlierMatches.Count).IsEqualTo(0);
	}

	private static Camera DistortedOpenCv(double focal)
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.OpenCV, focal, 100, 200);
		camera.Params[4] = -0.5; // k1
		camera.Params[5] = 0.5; // k2
		camera.Params[6] = -0.5; // p1
		return camera;
	}

	private static async Task TestGuidedMatchingWithCameraDistortion(Func<FeatureMatcherImage[], FeatureMatcher> matcherFactory)
	{
		// Test guided matching with essential matrix using calibrated cameras. This exercises
		// the code path that uses normalized coordinates. Use the OPENCV model with strong
		// radial and tangential distortion. The distortion is strong enough that the
		// pixel-coordinate fundamental matrix finds no matches, but must stay invertible over
		// the keypoints used below; p2 is left at zero for that reason.
		Camera camera = DistortedOpenCv(100.0);

		// Two points on the epipolar line (v=0 in normalized coordinates).
		Vector2d imgPoint11 = ToFloat(camera.ImgFromCam(new Vector3d(-0.5, 0.1, 1.0))!.Value);
		Vector2d imgPoint12 = ToFloat(camera.ImgFromCam(new Vector3d(0.4, -0.1, 1.0))!.Value);
		Vector2d imgPoint21 = ToFloat(camera.ImgFromCam(new Vector3d(0.3, -0.1, 1.0))!.Value);
		Vector2d imgPoint22 = ToFloat(camera.ImgFromCam(new Vector3d(-0.4, 0.1, 1.0))!.Value);

		FeatureMatcherImage image0 = Image(0, camera, SiftMatcherTestUtils.CreateEmptyDescriptors());
		FeatureMatcherImage image1 = Image(1, camera, SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2), Kp(imgPoint11), Kp(imgPoint12));
		FeatureMatcherImage image2 = Image(
			2, camera, SiftMatcherTestUtils.CreateReversedDescriptors(image1.Descriptors!), Kp(imgPoint21), Kp(imgPoint22));

		FeatureMatcher matcher = matcherFactory([image0, image1, image2]);

		var twoViewGeometry = new TwoViewGeometry();
		twoViewGeometry.E = EssentialMatrix.EssentialMatrixFromPose(new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0)));
		twoViewGeometry.F = EssentialMatrix.FundamentalFromEssentialMatrix(
			camera.CalibrationMatrix(), twoViewGeometry.E.Value, camera.CalibrationMatrix());

		const double kMaxError = 1.0;

		// With uncalibrated cameras, the fundamental matrix is used with pixel coordinates and
		// no matches are expected to be found due to strong distortion.
		twoViewGeometry.Config = TwoViewGeometry.ConfigurationType.Uncalibrated;
		matcher.MatchGuided(kMaxError, image1, image2, twoViewGeometry);
		await Assert.That(twoViewGeometry.InlierMatches.Count).IsEqualTo(0);

		// With calibrated cameras, the essential matrix is used with normalized coordinates
		// and matches are expected to be found.
		twoViewGeometry.Config = TwoViewGeometry.ConfigurationType.Calibrated;
		matcher.MatchGuided(kMaxError, image1, image2, twoViewGeometry);
		await SiftMatcherTestUtils.ExpectReversedInlierMatches(twoViewGeometry);

		twoViewGeometry.Config = TwoViewGeometry.ConfigurationType.Calibrated;
		matcher.MatchGuided(kMaxError, image0, image2, twoViewGeometry);
		await Assert.That(twoViewGeometry.InlierMatches.Count).IsEqualTo(0);
	}

	// Guided matching for a spherical camera, with correspondences deliberately in the back
	// hemisphere. Those pixels have no normalized image plane representation at all -
	// CamFromImg fails for them - so they are only matchable via the full-sphere bearing.
	private static async Task TestGuidedMatchingSpherical(Func<FeatureMatcherImage[], FeatureMatcher> matcherFactory)
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Equirectangular, 0, 512, 256);
		var cam2FromCam1 = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0));

		// Both points are behind both cameras, i.e. in the back hemisphere.
		var point3D1 = new Vector3d(0.3, 0.1, -2.0);
		var point3D2 = new Vector3d(-0.25, -0.15, -2.5);

		var premiseUnprojectable = new List<bool>();
		Vector2d Project(Vector3d point3D)
		{
			Vector2d imagePoint = camera.ImgFromCam(point3D)!.Value;

			// The premise of this test: these pixels are unprojectable through the normalized
			// image plane. If this ever starts failing, the test is no longer exercising the
			// back hemisphere.
			premiseUnprojectable.Add(!camera.CamFromImg(imagePoint).HasValue);
			return ToFloat(imagePoint);
		}

		Vector2d imgPoint11 = Project(point3D1);
		Vector2d imgPoint12 = Project(point3D2);
		Vector2d imgPoint21 = Project(cam2FromCam1 * point3D2);
		Vector2d imgPoint22 = Project(cam2FromCam1 * point3D1);

		FeatureMatcherImage image1 = Image(1, camera, SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2), Kp(imgPoint11), Kp(imgPoint12));
		FeatureMatcherImage image2 = Image(
			2, camera, SiftMatcherTestUtils.CreateReversedDescriptors(image1.Descriptors!), Kp(imgPoint21), Kp(imgPoint22));

		// Same as image2, but with the second correspondence replaced by a decoy far off the
		// epipolar great circle, which must be rejected. This is the load-bearing assertion:
		// only a decoy that the filter must actively reject distinguishes "the epipolar
		// constraint is evaluated correctly for back-hemisphere rays" from "the constraint has
		// stopped constraining anything".
		Vector2d imgPointDecoy = Project(cam2FromCam1 * new Vector3d(2.0, -1.5, -0.5));
		FeatureMatcherImage image3 = Image(3, camera, image2.Descriptors!, Kp(imgPoint21), Kp(imgPointDecoy));

		foreach (bool unprojectable in premiseUnprojectable)
		{
			await Assert.That(unprojectable).IsTrue();
		}

		FeatureMatcher matcher = matcherFactory([image1, image2, image3]);

		var twoViewGeometry = new TwoViewGeometry
		{
			Config = TwoViewGeometry.ConfigurationType.Calibrated,
			E = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1),
		};

		const double kMaxError = 4.0;

		matcher.MatchGuided(kMaxError, image1, image2, twoViewGeometry);
		await SiftMatcherTestUtils.ExpectReversedInlierMatches(twoViewGeometry);

		matcher.MatchGuided(kMaxError, image1, image3, twoViewGeometry);
		await Assert.That(twoViewGeometry.InlierMatches.Count).IsEqualTo(1);
		await Assert.That(twoViewGeometry.InlierMatches[0].Point2DIdx1).IsEqualTo(1u);
		await Assert.That(twoViewGeometry.InlierMatches[0].Point2DIdx2).IsEqualTo(0u);
	}

	// One correspondence in the front hemisphere and one in the back, to verify the two are
	// handled by the same code path rather than being swapped.
	private static async Task TestGuidedMatchingSphericalMixedHemispheres(Func<FeatureMatcherImage[], FeatureMatcher> matcherFactory)
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Equirectangular, 0, 512, 256);
		var cam2FromCam1 = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0));

		var point3DFront = new Vector3d(0.2, 0.1, 2.0);
		var point3DBack = new Vector3d(-0.25, -0.15, -2.5);

		Vector2d Project(Vector3d point3D) => ToFloat(camera.ImgFromCam(point3D)!.Value);

		Vector2d imgPoint11 = Project(point3DFront);
		Vector2d imgPoint12 = Project(point3DBack);
		Vector2d imgPoint21 = Project(cam2FromCam1 * point3DBack);
		Vector2d imgPoint22 = Project(cam2FromCam1 * point3DFront);

		await Assert.That(camera.CamFromImg(imgPoint11).HasValue).IsTrue();
		await Assert.That(camera.CamFromImg(imgPoint12).HasValue).IsFalse();

		FeatureMatcherImage image1 = Image(1, camera, SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2), Kp(imgPoint11), Kp(imgPoint12));
		FeatureMatcherImage image2 = Image(
			2, camera, SiftMatcherTestUtils.CreateReversedDescriptors(image1.Descriptors!), Kp(imgPoint21), Kp(imgPoint22));

		// Replaces the back-hemisphere correspondence with a decoy off the epipolar great
		// circle, so that the filter has to actively reject it.
		Vector2d imgPointDecoy = Project(cam2FromCam1 * new Vector3d(2.0, -1.5, -0.5));
		FeatureMatcherImage image3 = Image(3, camera, image2.Descriptors!, Kp(imgPointDecoy), Kp(imgPoint22));

		FeatureMatcher matcher = matcherFactory([image1, image2, image3]);

		var twoViewGeometry = new TwoViewGeometry
		{
			Config = TwoViewGeometry.ConfigurationType.Calibrated,
			E = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1),
		};

		const double kMaxError = 4.0;

		matcher.MatchGuided(kMaxError, image1, image2, twoViewGeometry);
		await SiftMatcherTestUtils.ExpectReversedInlierMatches(twoViewGeometry);

		// Only the front-hemisphere correspondence survives.
		matcher.MatchGuided(kMaxError, image1, image3, twoViewGeometry);
		await Assert.That(twoViewGeometry.InlierMatches.Count).IsEqualTo(1);
		await Assert.That(twoViewGeometry.InlierMatches[0].Point2DIdx1).IsEqualTo(0u);
		await Assert.That(twoViewGeometry.InlierMatches[0].Point2DIdx2).IsEqualTo(1u);
	}

	// A keypoint the camera cannot unproject must be rejected outright. It used to be
	// relocated to a (1e6, 1e6) sentinel, which does not reject: the Sampson error is a ratio
	// whose numerator and denominator scale together, so the residual converges to the finite
	// squared distance from the partner to the epipolar line of the point at infinity in
	// direction (1, 1, 0). Any partner near that line was therefore silently accepted.
	private static async Task TestGuidedMatchingUnprojectableKeypoints(Func<FeatureMatcherImage[], FeatureMatcher> matcherFactory)
	{
		Camera camera = DistortedOpenCv(100.0);

		// Well inside a region where the iterative undistortion does not converge.
		var unprojectable = new Vector2d(50.0, 150.0);
		await Assert.That(camera.CamFromImg(unprojectable).HasValue).IsFalse();

		var premiseProjectable = new List<bool>();
		Vector2d Project(Vector3d point3D)
		{
			Vector2d imagePoint = camera.ImgFromCam(point3D)!.Value;

			// Everything except the sentinel keypoint must be a normal, usable keypoint, or the
			// test would pass for the wrong reason.
			premiseProjectable.Add(camera.CamFromImg(imagePoint).HasValue);
			return ToFloat(imagePoint);
		}

		// A translation with tx == ty, so that the epipolar line of the sentinel direction
		// (1, 1, 0) passes through the image center and the decoy below can sit on it at a
		// well-behaved location.
		var cam2FromCam1 = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 1, 1));
		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);

		var point3D = new Vector3d(-0.3, -0.2, 2.0);
		Vector2d imgPointGood1 = Project(point3D);
		Vector2d imgPointGood2 = Project(cam2FromCam1 * point3D);

		// The old (1e6, 1e6) sentinel converges to the direction (1, 1, 0); its epipolar line
		// is where spurious matches used to concentrate, so the decoy is placed exactly on it.
		Vector3d sentinelLine = e * new Vector3d(1, 1, 0);
		const double decoyX = 0.2;
		double decoyY = -((sentinelLine.X * decoyX) + sentinelLine.Z) / sentinelLine.Y;
		Vector2d imgPointDecoy = Project(new Vector3d(decoyX, decoyY, 1.0));

		foreach (bool projectable in premiseProjectable)
		{
			await Assert.That(projectable).IsTrue();
		}

		FeatureMatcherImage image1 = Image(
			1, camera, SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2),
			new FeatureKeypoint((float)unprojectable.X, (float)unprojectable.Y), Kp(imgPointGood1));
		FeatureMatcherImage image2 = Image(
			2, camera, SiftMatcherTestUtils.CreateReversedDescriptors(image1.Descriptors!), Kp(imgPointGood2), Kp(imgPointDecoy));

		FeatureMatcher matcher = matcherFactory([image1, image2]);

		var twoViewGeometry = new TwoViewGeometry { Config = TwoViewGeometry.ConfigurationType.Calibrated, E = e };

		matcher.MatchGuided(1.0, image1, image2, twoViewGeometry);

		// Only the good pair survives; the unprojectable keypoint 0 matches nothing.
		await Assert.That(twoViewGeometry.InlierMatches.Count).IsEqualTo(1);
		await Assert.That(twoViewGeometry.InlierMatches[0].Point2DIdx1).IsEqualTo(1u);
		await Assert.That(twoViewGeometry.InlierMatches[0].Point2DIdx2).IsEqualTo(0u);
	}

	[Test]
	public Task MatchGuidedSiftFeaturesCPU_EssentialMatrix() =>
		TestGuidedMatchingWithCameraDistortion(SiftMatcherTestUtils.CreateCpuMatcherWithIndexCache);

	[Test]
	public Task MatchGuidedSiftFeaturesCPU_Spherical() =>
		TestGuidedMatchingSpherical(SiftMatcherTestUtils.CreateCpuMatcherWithIndexCache);

	[Test]
	public Task MatchGuidedSiftFeaturesCPU_SphericalMixedHemispheres() =>
		TestGuidedMatchingSphericalMixedHemispheres(SiftMatcherTestUtils.CreateCpuMatcherWithIndexCache);

	[Test]
	public Task MatchGuidedSiftFeaturesCPU_UnprojectableKeypoints() =>
		TestGuidedMatchingUnprojectableKeypoints(SiftMatcherTestUtils.CreateCpuMatcherWithIndexCache);

	private static async Task TestGuidedMatchingSharedFocal(Func<FeatureMatcherImage[], FeatureMatcher> matcherFactory)
	{
		// An UNCALIBRATED pair carrying solver-estimated intrinsics (camera1/camera2) is
		// guided-matched via the essential matrix, using those estimated intrinsics rather
		// than the images' cameras, whose focal length is only a placeholder. Distortion is
		// strong enough that the pixel-coordinate F path finds nothing, and the placeholder
		// focal is wrong enough that normalizing with it finds nothing either, so the test
		// passes only if the estimated camera is the one used. As elsewhere on the E path, E
		// is taken to relate undistorted rays.
		const double kEstimatedFocal = 100.0;
		const double kPlaceholderFocal = 500.0;
		Camera camera = DistortedOpenCv(kEstimatedFocal);

		// The camera as stored in the database: same model and distortion, but the focal
		// length has not been recovered yet.
		Camera placeholderCamera = camera.Clone();
		placeholderCamera.SetFocalLength(kPlaceholderFocal);

		// Two points on the epipolar line (v=0 in normalized coordinates).
		Vector2d imgPoint11 = ToFloat(camera.ImgFromCam(new Vector3d(-0.5, 0.1, 1.0))!.Value);
		Vector2d imgPoint12 = ToFloat(camera.ImgFromCam(new Vector3d(0.4, -0.1, 1.0))!.Value);
		Vector2d imgPoint21 = ToFloat(camera.ImgFromCam(new Vector3d(0.3, -0.1, 1.0))!.Value);
		Vector2d imgPoint22 = ToFloat(camera.ImgFromCam(new Vector3d(-0.4, 0.1, 1.0))!.Value);

		FeatureMatcherImage image1 = Image(
			1, placeholderCamera, SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2), Kp(imgPoint11), Kp(imgPoint12));
		FeatureMatcherImage image2 = Image(
			2, placeholderCamera, SiftMatcherTestUtils.CreateReversedDescriptors(image1.Descriptors!), Kp(imgPoint21), Kp(imgPoint22));

		FeatureMatcher matcher = matcherFactory([image1, image2]);

		var twoViewGeometry = new TwoViewGeometry
		{
			Config = TwoViewGeometry.ConfigurationType.Uncalibrated,
			E = EssentialMatrix.EssentialMatrixFromPose(new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0))),
			Camera1 = camera,
			Camera2 = camera,
		};

		// F = K^-T E K^-1, as the estimator populates it for this config.
		twoViewGeometry.F = EssentialMatrix.FundamentalFromEssentialMatrix(
			camera.CalibrationMatrix(), twoViewGeometry.E.Value, camera.CalibrationMatrix());

		// Matches are found only by normalizing with the estimated intrinsics.
		matcher.MatchGuided(1.0, image1, image2, twoViewGeometry);
		await SiftMatcherTestUtils.ExpectReversedInlierMatches(twoViewGeometry);
	}

	// The normalizing camera is not a function of the image alone: a shared-focal pair carries
	// a focal length estimated per pair, so the same image matched against different partners
	// must be renormalized.
	private static async Task TestGuidedMatchingSharedFocalPerPairFocal(Func<FeatureMatcherImage[], FeatureMatcher> matcherFactory)
	{
		const double kFocalA = 100.0;
		const double kFocalB = 200.0;
		Camera cameraA = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, kFocalA, 100, 200);
		Camera cameraB = Camera.CreateFromModelId(2, CameraModelId.SimplePinhole, kFocalB, 100, 200);

		// image1's pixels normalize to y = +-0.1 under camera_a, and to half that, y = +-0.05,
		// under camera_b. The relative pose is a pure x-translation, so a match requires the
		// partner's normalized y to agree.
		Vector2d imgPoint11 = ToFloat(cameraA.ImgFromCam(new Vector3d(-0.5, 0.1, 1.0))!.Value);
		Vector2d imgPoint12 = ToFloat(cameraA.ImgFromCam(new Vector3d(0.4, -0.1, 1.0))!.Value);

		// Partner for the camera_a pair.
		Vector2d imgPoint21 = ToFloat(cameraA.ImgFromCam(new Vector3d(0.3, -0.1, 1.0))!.Value);
		Vector2d imgPoint22 = ToFloat(cameraA.ImgFromCam(new Vector3d(-0.4, 0.1, 1.0))!.Value);

		// Partner for the camera_b pair.
		Vector2d imgPoint31 = ToFloat(cameraB.ImgFromCam(new Vector3d(0.3, -0.05, 1.0))!.Value);
		Vector2d imgPoint32 = ToFloat(cameraB.ImgFromCam(new Vector3d(-0.4, 0.05, 1.0))!.Value);

		FeatureMatcherImage image1 = Image(1, cameraA, SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2), Kp(imgPoint11), Kp(imgPoint12));
		FeatureMatcherImage image2 = Image(
			2, cameraA, SiftMatcherTestUtils.CreateReversedDescriptors(image1.Descriptors!), Kp(imgPoint21), Kp(imgPoint22));
		FeatureMatcherImage image3 = Image(
			3, cameraA, SiftMatcherTestUtils.CreateReversedDescriptors(image1.Descriptors!), Kp(imgPoint31), Kp(imgPoint32));

		FeatureMatcher matcher = matcherFactory([image1, image2, image3]);

		var twoViewGeometry = new TwoViewGeometry
		{
			Config = TwoViewGeometry.ConfigurationType.Uncalibrated,
			E = EssentialMatrix.EssentialMatrixFromPose(new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0))),
		};

		const double kMaxError = 1.0;

		twoViewGeometry.Camera1 = cameraA;
		twoViewGeometry.Camera2 = cameraA;
		matcher.MatchGuided(kMaxError, image1, image2, twoViewGeometry);
		await SiftMatcherTestUtils.ExpectReversedInlierMatches(twoViewGeometry);

		// Same image1, different estimated focal: stale normalized keypoints from the previous
		// call would put image1 at y = +-0.1 instead of +-0.05, far outside the ~1/f
		// normalized threshold.
		twoViewGeometry.Camera1 = cameraB;
		twoViewGeometry.Camera2 = cameraB;
		matcher.MatchGuided(kMaxError, image1, image3, twoViewGeometry);
		await SiftMatcherTestUtils.ExpectReversedInlierMatches(twoViewGeometry);
	}

	[Test]
	public Task MatchGuidedSiftFeaturesCPU_SharedFocal() =>
		TestGuidedMatchingSharedFocal(SiftMatcherTestUtils.CreateCpuMatcherWithIndexCache);

	[Test]
	public Task MatchGuidedSiftFeaturesCPU_SharedFocalPerPairFocal() =>
		TestGuidedMatchingSharedFocalPerPairFocal(SiftMatcherTestUtils.CreateCpuMatcherWithIndexCache);
}
