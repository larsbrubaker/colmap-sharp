// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BundleAdjustmentCeresTests: colmap/estimators/bundle_adjustment_ceres_test.cc 1:1 for
// DefaultBundleAdjuster (Estimators/BundleAdjustmentCeres*.cs). Test names are
// <Suite>_<Name>.
//
// Tier A for the problem layout (num_residuals_reduced, num_effective_parameters_reduced are
// exact counts); Tier C for the solved values, checked as COLMAP does (moved / constant).
//
// Skipped: CeresBundleAdjustmentOptions.FallsBackToCpuWithoutCudaDevice (CUDA only; COLMAP
// compiles it out without COLMAP_CUDA_ENABLED).
// DefaultBundleAdjuster.NominalMultiCameraRig and the five PosePriorBundleAdjuster cases
// are in BundleAdjustmentCeresTests.PosePrior.cs.

using ColmapSharp.Estimators;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Solver;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public partial class BundleAdjustmentCeresTests
{
	// Due to pose normalization operations, constant variables may not be perfectly fixed
	// during bundle adjustment.
	private const double ConstantPoseVarEps = 1e-9;

	private static int FocalLengthIdx => SimpleRadialCameraModel.FocalLengthIdxs[0];

	private static int ExtraParamIdx => SimpleRadialCameraModel.ExtraParamsIdxs[0];

	private static async Task CheckVariableCamera(Camera camera, Camera origCamera)
	{
		await Assert.That(camera.Params[FocalLengthIdx]).IsNotEqualTo(origCamera.Params[FocalLengthIdx]);
		await Assert.That(camera.Params[ExtraParamIdx]).IsNotEqualTo(origCamera.Params[ExtraParamIdx]);
	}

	private static async Task CheckConstantCamera(Camera camera, Camera origCamera)
	{
		await Assert.That(camera.Params[FocalLengthIdx]).IsEqualTo(origCamera.Params[FocalLengthIdx]);
		await Assert.That(camera.Params[ExtraParamIdx]).IsEqualTo(origCamera.Params[ExtraParamIdx]);
	}

	private static async Task CheckVariableCamFromWorld(Image image, Image origImage) =>
		await Assert.That(Rigid3dMatchers.Rigid3dEq(image.CamFromWorld(), origImage.CamFromWorld())).IsFalse();

	private static async Task CheckConstantCamFromWorld(Image image, Image origImage) =>
		await Assert.That(Rigid3dMatchers.Rigid3dNear(image.CamFromWorld(), origImage.CamFromWorld(), ConstantPoseVarEps, ConstantPoseVarEps)).IsTrue();

	private static async Task CheckConstantCamFromWorldTranslationCoord(Image image, Image origImage)
	{
		int numConstantCoords = 0;
		for (int i = 0; i < 3; ++i)
		{
			if (Math.Abs(image.CamFromWorld().Translation[i] - origImage.CamFromWorld().Translation[i]) < ConstantPoseVarEps)
			{
				++numConstantCoords;
			}
		}

		await Assert.That(numConstantCoords).IsEqualTo(1);
	}

	private static async Task CheckVariablePoint(Point3D point, Point3D origPoint) =>
		await Assert.That(point.Xyz != origPoint.Xyz).IsTrue();

	private static async Task CheckConstantPoint(Point3D point, Point3D origPoint) =>
		await Assert.That(point.Xyz == origPoint.Xyz).IsTrue();

	private static SolverSummary GetCeresSummary(BundleAdjustmentSummary summary) =>
		Check.NotNull(summary as CeresBundleAdjustmentSummary).CeresSummary;

	private static int NumVariablePoints(Reconstruction reconstruction, Reconstruction origReconstruction) =>
		reconstruction.Points3D.Count(pair => pair.Value != origReconstruction.Point3D(pair.Key));

	private static Reconstruction SynthesizeWithNoise(int numRigs, int numCamerasPerRig, int numFramesPerRig, int numPoints3D)
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = BundleAdjustmentTests.Synthesize(numRigs, numCamerasPerRig, numFramesPerRig, numPoints3D);
		BundleAdjustmentTests.AddPoint2DNoise(reconstruction, 1);
		return reconstruction;
	}

	[Test]
	public async Task DefaultBundleAdjuster_Cancellation()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = BundleAdjustmentTests.Synthesize(1, 1, 3, 20);

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			config.AddImage(imageId);
		}

		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		int numChecks = 0;
		var options = new BundleAdjustmentOptions
		{
			CheckIfStopped = () =>
			{
				++numChecks;
				return true;
			},
		};
		BundleAdjustmentSummary summary =
			CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(options, config, reconstruction).Solve();

		await Assert.That(numChecks).IsEqualTo(1);
		await Assert.That(summary.TerminationType).IsEqualTo(BundleAdjustmentTerminationType.UserSuccess);
	}

	[Test]
	public async Task DefaultBundleAdjuster_ThreeViewSpherical()
	{
		RandomUtils.SetPRNGSeed(0);
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = 3,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 1,
				NumPoints3D = 100,
				CameraModelId = CameraModelId.Equirectangular,
				CameraWidth = 1000,
				CameraHeight = 500,
				CameraParams = [1000, 500],
			},
			reconstruction);
		Check.That(reconstruction.Camera(1).IsSpherical);
		BundleAdjustmentTests.AddPoint2DNoise(reconstruction, 1);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.AddImage(3);
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		CeresBundleAdjuster bundleAdjuster =
			CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(new BundleAdjustmentOptions(), config, reconstruction);
		BundleAdjustmentSummary summary = bundleAdjuster.Solve();
		Check.That(summary.TerminationType != BundleAdjustmentTerminationType.Failure);

		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(bundleAdjuster.Problem.NumResiduals);

		// The spherical model has no focal length; its (w, h) parameters are held constant
		// during bundle adjustment.
		foreach ((uint cameraId, Camera camera) in reconstruction.Cameras)
		{
			await Assert.That(camera.Params.SequenceEqual(origReconstruction.Camera(cameraId).Params)).IsTrue();
		}

		await CheckConstantCamFromWorld(reconstruction.Image(1), origReconstruction.Image(1));
		await CheckConstantCamFromWorldTranslationCoord(reconstruction.Image(2), origReconstruction.Image(2));
		await CheckVariableCamFromWorld(reconstruction.Image(3), origReconstruction.Image(3));
		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			await CheckVariablePoint(point3D, origReconstruction.Point3D(point3DId));
		}
	}

	[Test]
	public async Task DefaultBundleAdjuster_TwoViewRig()
	{
		Reconstruction reconstruction = SynthesizeWithNoise(1, 2, 2, 100);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			config.AddImage(imageId);
		}

		config.FixGauge(BundleAdjustmentGauge.ThreePoints);

		CeresBundleAdjuster bundleAdjuster =
			CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(new BundleAdjustmentOptions(), config, reconstruction);
		BundleAdjustmentSummary summary = bundleAdjuster.Solve();
		Check.That(summary.TerminationType != BundleAdjustmentTerminationType.Failure);

		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(bundleAdjuster.Problem.NumResiduals);

		// 100 points, 4 images, 2 residuals per point per image
		await Assert.That(GetCeresSummary(summary).NumResidualsReduced).IsEqualTo(800);
		// 97 x 3 point parameters (3 fixed for gauge)
		// + 2 x 6 rig_from_world parameters
		// + 1 x 6 sensor_from_rig parameters
		// + 2 x 2 camera parameters
		await Assert.That(GetCeresSummary(summary).NumEffectiveParametersReduced).IsEqualTo(313);

		await CheckVariableCamera(reconstruction.Camera(1), origReconstruction.Camera(1));
		await CheckVariableCamFromWorld(reconstruction.Image(1), origReconstruction.Image(1));

		await CheckVariableCamera(reconstruction.Camera(2), origReconstruction.Camera(2));
		await CheckVariableCamFromWorld(reconstruction.Image(2), origReconstruction.Image(2));

		await Assert.That(NumVariablePoints(reconstruction, origReconstruction)).IsEqualTo(97);
	}

	[Test]
	public async Task DefaultBundleAdjuster_ManyViewRig()
	{
		Reconstruction reconstruction = SynthesizeWithNoise(2, 3, 5, 100);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			config.AddImage(imageId);
		}

		config.FixGauge(BundleAdjustmentGauge.ThreePoints);

		CeresBundleAdjuster bundleAdjuster =
			CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(new BundleAdjustmentOptions(), config, reconstruction);
		BundleAdjustmentSummary summary = bundleAdjuster.Solve();
		Check.That(summary.TerminationType != BundleAdjustmentTerminationType.Failure);

		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(bundleAdjuster.Problem.NumResiduals);

		// 100 points, 30 images, 2 residuals per point per image
		await Assert.That(GetCeresSummary(summary).NumResidualsReduced).IsEqualTo(6000);
		// 97 x 3 point parameters (3 fixed for gauge)
		// + 10 x 6 rig_from_world parameters
		// + 4 x 6 sensor_from_rig parameters
		// + 6 x 2 camera parameters
		await Assert.That(GetCeresSummary(summary).NumEffectiveParametersReduced).IsEqualTo(387);

		foreach ((uint cameraId, Camera camera) in reconstruction.Cameras)
		{
			await CheckVariableCamera(camera, origReconstruction.Camera(cameraId));
		}

		foreach (uint imageId in reconstruction.RegImageIds())
		{
			await CheckVariableCamFromWorld(reconstruction.Image(imageId), origReconstruction.Image(imageId));
		}

		await Assert.That(NumVariablePoints(reconstruction, origReconstruction)).IsEqualTo(97);
	}

	[Test]
	public async Task DefaultBundleAdjuster_ManyViewRigConstantSensorFromRig()
	{
		Reconstruction reconstruction = SynthesizeWithNoise(2, 3, 5, 100);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			config.AddImage(imageId);
		}

		config.SetConstantSensorFromRigPose(reconstruction.Camera(2).SensorId);
		config.FixGauge(BundleAdjustmentGauge.ThreePoints);

		CeresBundleAdjuster bundleAdjuster =
			CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(new BundleAdjustmentOptions(), config, reconstruction);
		BundleAdjustmentSummary summary = bundleAdjuster.Solve();
		Check.That(summary.TerminationType != BundleAdjustmentTerminationType.Failure);

		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(bundleAdjuster.Problem.NumResiduals);

		// 100 points, 30 images, 2 residuals per point per image
		await Assert.That(GetCeresSummary(summary).NumResidualsReduced).IsEqualTo(6000);
		// 97 x 3 point parameters (3 fixed for gauge)
		// + 10 x 6 rig_from_world parameters
		// + 3 x 6 sensor_from_rig parameters
		// + 6 x 2 camera parameters
		await Assert.That(GetCeresSummary(summary).NumEffectiveParametersReduced).IsEqualTo(381);

		foreach ((uint cameraId, Camera camera) in reconstruction.Cameras)
		{
			await CheckVariableCamera(camera, origReconstruction.Camera(cameraId));
		}

		foreach (uint imageId in reconstruction.RegImageIds())
		{
			await CheckVariableCamFromWorld(reconstruction.Image(imageId), origReconstruction.Image(imageId));
		}

		await Assert.That(NumVariablePoints(reconstruction, origReconstruction)).IsEqualTo(97);
	}

	[Test]
	public async Task DefaultBundleAdjuster_ManyViewRigConstantRigFromWorld()
	{
		Reconstruction reconstruction = SynthesizeWithNoise(2, 3, 5, 100);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			config.AddImage(imageId);
		}

		const uint constantFrameId = 1;
		config.SetConstantRigFromWorldPose(constantFrameId);
		config.FixGauge(BundleAdjustmentGauge.ThreePoints);

		CeresBundleAdjuster bundleAdjuster =
			CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(new BundleAdjustmentOptions(), config, reconstruction);
		BundleAdjustmentSummary summary = bundleAdjuster.Solve();
		Check.That(summary.TerminationType != BundleAdjustmentTerminationType.Failure);

		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(bundleAdjuster.Problem.NumResiduals);

		// 100 points, 30 images, 2 residuals per point per image
		await Assert.That(GetCeresSummary(summary).NumResidualsReduced).IsEqualTo(6000);
		// 97 x 3 point parameters (3 fixed for gauge)
		// + 9 x 6 rig_from_world parameters
		// + 4 x 6 sensor_from_rig parameters
		// + 6 x 2 camera parameters
		await Assert.That(GetCeresSummary(summary).NumEffectiveParametersReduced).IsEqualTo(381);

		foreach ((uint cameraId, Camera camera) in reconstruction.Cameras)
		{
			await CheckVariableCamera(camera, origReconstruction.Camera(cameraId));
		}

		foreach (uint imageId in reconstruction.RegImageIds())
		{
			Image image = reconstruction.Image(imageId);
			if (image.FrameId == constantFrameId && image.FramePtr.RigPtr.IsRefSensor(image.CameraPtr.SensorId))
			{
				await CheckConstantCamFromWorld(image, origReconstruction.Image(imageId));
			}
			else
			{
				await CheckVariableCamFromWorld(image, origReconstruction.Image(imageId));
			}
		}

		await Assert.That(NumVariablePoints(reconstruction, origReconstruction)).IsEqualTo(97);
	}

	[Test]
	public async Task DefaultBundleAdjuster_ConstantRigFromWorldRotation()
	{
		Reconstruction reconstruction = SynthesizeWithNoise(3, 1, 1, 100);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.AddImage(3);
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions { ConstantRigFromWorldRotation = true };
		CeresBundleAdjuster bundleAdjuster = CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
		BundleAdjustmentSummary summary = bundleAdjuster.Solve();
		Check.That(summary.TerminationType != BundleAdjustmentTerminationType.Failure);

		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(bundleAdjuster.Problem.NumResiduals);

		// 100 points, 3 images, 2 residuals per point per image
		await Assert.That(GetCeresSummary(summary).NumResidualsReduced).IsEqualTo(600);
		// 100 x 3 point parameters
		// + 2 translation parameters (second image, one coord fixed for gauge)
		// + 3 translation parameters (third image)
		// + 3 x 2 camera parameters
		await Assert.That(GetCeresSummary(summary).NumEffectiveParametersReduced).IsEqualTo(311);

		// Check rotations are constant for all images
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			// Rotation should be nearly unchanged (use angular distance)
			double angularDistance = reconstruction.Image(imageId).CamFromWorld().Rotation
				.AngularDistance(origReconstruction.Image(imageId).CamFromWorld().Rotation);
			await Assert.That(angularDistance).IsLessThanOrEqualTo(ConstantPoseVarEps);
		}

		// Check translations are variable (except for gauge-fixed parts)
		// At least one image should have changed translation
		bool hasVariableTranslation = false;
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			if ((reconstruction.Image(imageId).CamFromWorld().Translation
				- origReconstruction.Image(imageId).CamFromWorld().Translation).Norm > ConstantPoseVarEps)
			{
				hasVariableTranslation = true;
				break;
			}
		}

		await Assert.That(hasVariableTranslation).IsTrue();

		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			await CheckVariablePoint(point3D, origReconstruction.Point3D(point3DId));
		}
	}

	[Test]
	public async Task DefaultBundleAdjuster_PartiallyContainedTracksForceToOptimizePoint()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = BundleAdjustmentTests.Synthesize(3, 1, 1, 100, numPoints2DWithoutPoint3D: 0);
		BundleAdjustmentTests.AddPoint2DNoise(reconstruction, 1);

		ulong variablePoint3DId = reconstruction.Image(3).Points2D[0].Point3DId;
		ulong addVariablePoint3DId = reconstruction.Image(3).Points2D[1].Point3DId;
		ulong addConstantPoint3DId = reconstruction.Image(3).Points2D[2].Point3DId;
		reconstruction.DeleteObservation(3, 0);

		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.SetConstantRigFromWorldPose(1);
		config.SetConstantRigFromWorldPose(2);
		config.AddVariablePoint(addVariablePoint3DId);
		config.AddConstantPoint(addConstantPoint3DId);

		CeresBundleAdjuster bundleAdjuster =
			CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(new BundleAdjustmentOptions(), config, reconstruction);
		BundleAdjustmentSummary summary = bundleAdjuster.Solve();
		Check.That(summary.TerminationType != BundleAdjustmentTerminationType.Failure);

		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(bundleAdjuster.Problem.NumResiduals);

		// 100 points, 2 images, 2 residuals per point per image
		// + 2 residuals in 3rd image for added variable 3D point
		// (added constant point does not add residuals since the image/camera
		// is also constant).
		await Assert.That(GetCeresSummary(summary).NumResidualsReduced).IsEqualTo(402);
		// 2 x 3 point parameters
		// 2 x 2 camera parameters
		await Assert.That(GetCeresSummary(summary).NumEffectiveParametersReduced).IsEqualTo(10);

		await CheckVariableCamera(reconstruction.Camera(1), origReconstruction.Camera(1));
		await CheckConstantCamFromWorld(reconstruction.Image(1), origReconstruction.Image(1));

		await CheckVariableCamera(reconstruction.Camera(2), origReconstruction.Camera(2));
		await CheckConstantCamFromWorld(reconstruction.Image(2), origReconstruction.Image(2));

		await CheckConstantCamera(reconstruction.Camera(3), origReconstruction.Camera(3));
		await CheckConstantCamFromWorld(reconstruction.Image(3), origReconstruction.Image(3));

		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			if (point3DId == variablePoint3DId || point3DId == addVariablePoint3DId)
			{
				await CheckVariablePoint(point3D, origReconstruction.Point3D(point3DId));
			}
			else
			{
				await CheckConstantPoint(point3D, origReconstruction.Point3D(point3DId));
			}
		}
	}

	[Test]
	public async Task DefaultBundleAdjuster_ConstantFocalLength() => await ConstantIntrinsicCase(refineFocalLength: false);

	[Test]
	public async Task DefaultBundleAdjuster_ConstantExtraParam() => await ConstantIntrinsicCase(refineFocalLength: true);

	// ConstantFocalLength and ConstantExtraParam differ only in which parameter group is
	// held constant (the other must move); the expected counts are the same.
	private static async Task ConstantIntrinsicCase(bool refineFocalLength)
	{
		Reconstruction reconstruction = SynthesizeWithNoise(2, 1, 1, 100);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions
		{
			RefineFocalLength = refineFocalLength,
			RefineExtraParams = !refineFocalLength,
		};
		CeresBundleAdjuster bundleAdjuster = CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
		BundleAdjustmentSummary summary = bundleAdjuster.Solve();
		Check.That(summary.TerminationType != BundleAdjustmentTerminationType.Failure);

		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(bundleAdjuster.Problem.NumResiduals);

		// 100 points, 3 images, 2 residuals per point per image
		await Assert.That(GetCeresSummary(summary).NumResidualsReduced).IsEqualTo(400);
		// 100 x 3 point parameters
		// + 5 rig_from_world parameters (pose of second image)
		// + 2 camera parameters
		await Assert.That(GetCeresSummary(summary).NumEffectiveParametersReduced).IsEqualTo(307);

		await CheckConstantCamFromWorld(reconstruction.Image(1), origReconstruction.Image(1));
		await CheckConstantCamFromWorldTranslationCoord(reconstruction.Image(2), origReconstruction.Image(2));

		foreach (uint cameraId in new uint[] { 1, 2 })
		{
			Camera camera = reconstruction.Camera(cameraId);
			Camera origCamera = origReconstruction.Camera(cameraId);
			await Assert.That(camera.Params[FocalLengthIdx] == origCamera.Params[FocalLengthIdx]).IsEqualTo(!refineFocalLength);
			await Assert.That(camera.Params[ExtraParamIdx] != origCamera.Params[ExtraParamIdx]).IsEqualTo(!refineFocalLength);
		}

		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			await CheckVariablePoint(point3D, origReconstruction.Point3D(point3DId));
		}
	}
}
