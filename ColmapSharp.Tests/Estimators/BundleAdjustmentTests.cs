// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BundleAdjustmentTests: colmap/estimators/bundle_adjustment_test.cc 1:1 for the Ceres
// backend (Estimators/BundleAdjustment.cs, BundleAdjustmentCeres*.cs). The parameterized
// BundleAdjusterBackendTest suite runs for CERES only: CASPAR is COLMAP's GPU backend and out
// of scope (PORTING_PLAN.md). Test names are <Suite>_<Name>.
//
// Tier A for the config bookkeeping and the residual counts (the problem layout is exact);
// Tier C for the solved values, which the tests check the way COLMAP's do: which parameters
// moved and which stayed constant.
//
// BundleAdjusterBackendTest.Nominal, .NominalMultiCameraRigConstantSensorFromRig and
// PosePriorBundleAdjusterBackendTest.Nominal (the ReconstructionNear cases) are in
// BundleAdjustmentTests.Nominal.cs.

using ColmapSharp.Estimators;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public partial class BundleAdjustmentTests
{
	private const double ConstantPoseVarEps = 1e-9;

	private static int FocalLengthIdx => SimpleRadialCameraModel.FocalLengthIdxs[0];

	private static int ExtraParamIdx => SimpleRadialCameraModel.ExtraParamsIdxs[0];

	internal static Reconstruction Synthesize(
		int numRigs, int numCamerasPerRig, int numFramesPerRig, int numPoints3D, int numPoints2DWithoutPoint3D = 10)
	{
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = numRigs,
				NumCamerasPerRig = numCamerasPerRig,
				NumFramesPerRig = numFramesPerRig,
				NumPoints3D = numPoints3D,
				NumPoints2DWithoutPoint3D = numPoints2DWithoutPoint3D,
			},
			reconstruction);
		return reconstruction;
	}

	internal static void AddPoint2DNoise(Reconstruction reconstruction, double stddev) =>
		Synthetic.SynthesizeNoise(new SyntheticNoiseOptions { Point2DStddev = stddev }, reconstruction);

	private static async Task ExpectVariableCamera(Camera camera, Camera origCamera)
	{
		await Assert.That(camera.Params[FocalLengthIdx]).IsNotEqualTo(origCamera.Params[FocalLengthIdx]);
		await Assert.That(camera.Params[ExtraParamIdx]).IsNotEqualTo(origCamera.Params[ExtraParamIdx]);
	}

	private static async Task ExpectConstantCamera(Camera camera, Camera origCamera)
	{
		await Assert.That(camera.Params[FocalLengthIdx]).IsEqualTo(origCamera.Params[FocalLengthIdx]);
		await Assert.That(camera.Params[ExtraParamIdx]).IsEqualTo(origCamera.Params[ExtraParamIdx]);
	}

	private static async Task ExpectConstantCamFromWorld(Image image, Image origImage) =>
		await Assert.That(Rigid3dMatchers.Rigid3dNear(image.CamFromWorld(), origImage.CamFromWorld(), ConstantPoseVarEps, ConstantPoseVarEps)).IsTrue();

	private static async Task ExpectVariableCamFromWorld(Image image, Image origImage) =>
		await Assert.That(Rigid3dMatchers.Rigid3dEq(image.CamFromWorld(), origImage.CamFromWorld())).IsFalse();

	private static async Task ExpectVariablePoint(Point3D point, Point3D origPoint) =>
		await Assert.That(point.Xyz != origPoint.Xyz).IsTrue();

	private static async Task ExpectConstantPoint(Point3D point, Point3D origPoint) =>
		await Assert.That(point.Xyz == origPoint.Xyz).IsTrue();

	private static BundleAdjustmentSummary SolveDefault(BundleAdjustmentOptions options, BundleAdjustmentConfig config, Reconstruction reconstruction) =>
		BundleAdjusters.CreateDefaultBundleAdjuster(options, config, reconstruction).Solve();

	[Test]
	public async Task BundleAdjustmentOptions_Copy()
	{
		var options = new BundleAdjustmentOptions
		{
			RefineFocalLength = false,
			RefinePrincipalPoint = true,
			MinTrackLength = 5,
		};
		options.Ceres!.SolverOptions.MaxNumIterations = 42;

		BundleAdjustmentOptions copy = options.Clone();

		// Verify fields are copied
		await Assert.That(copy.RefineFocalLength).IsFalse();
		await Assert.That(copy.RefinePrincipalPoint).IsTrue();
		await Assert.That(copy.MinTrackLength).IsEqualTo(5);
		await Assert.That(copy.Ceres!.SolverOptions.MaxNumIterations).IsEqualTo(42);

		// Verify deep copy of shared_ptr (different pointer instances)
		await Assert.That(ReferenceEquals(options.Ceres, copy.Ceres)).IsFalse();
	}

	[Test]
	public async Task PosePriorBundleAdjustmentOptions_Copy()
	{
		var options = new PosePriorBundleAdjustmentOptions { PriorPositionFallbackStddev = 2.5 };
		ColmapSharp.Optim.RansacOptions ransacOptions = options.AlignmentRansacOptions;
		ransacOptions.MaxError = 1.0;
		options.AlignmentRansacOptions = ransacOptions;
		options.Ceres!.PriorPositionLossScale = 0.42;

		PosePriorBundleAdjustmentOptions copy = options.Clone();

		// Verify fields are copied
		await Assert.That(copy.PriorPositionFallbackStddev).IsEqualTo(2.5);
		await Assert.That(copy.AlignmentRansacOptions.MaxError).IsEqualTo(1.0);
		await Assert.That(copy.Ceres!.PriorPositionLossScale).IsEqualTo(0.42);

		// Verify deep copy of shared_ptr (different pointer instances)
		await Assert.That(ReferenceEquals(options.Ceres, copy.Ceres)).IsFalse();
	}

	[Test]
	public async Task BundleAdjustmentSummary_IsSolutionUsable()
	{
		var summary = new BundleAdjustmentSummary { TerminationType = BundleAdjustmentTerminationType.Convergence };
		await Assert.That(summary.IsSolutionUsable()).IsTrue();

		summary.TerminationType = BundleAdjustmentTerminationType.NoConvergence;
		await Assert.That(summary.IsSolutionUsable()).IsTrue();

		summary.TerminationType = BundleAdjustmentTerminationType.UserSuccess;
		await Assert.That(summary.IsSolutionUsable()).IsTrue();

		summary.TerminationType = BundleAdjustmentTerminationType.Failure;
		await Assert.That(summary.IsSolutionUsable()).IsFalse();

		summary.TerminationType = BundleAdjustmentTerminationType.UserFailure;
		await Assert.That(summary.IsSolutionUsable()).IsFalse();
	}

	[Test]
	public async Task BundleAdjustmentConfig_NumResiduals()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = Synthesize(4, 1, 1, 100);

		List<uint> imageIds = reconstruction.RegImageIds();
		Check.Eq(imageIds.Count, 4);

		var config = new BundleAdjustmentConfig();

		config.AddImage(imageIds[0]);
		config.AddImage(imageIds[1]);
		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(400);

		config.AddVariablePoint(1);
		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(404);

		config.AddConstantPoint(2);
		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(408);

		config.AddImage(imageIds[2]);
		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(604);

		config.AddImage(imageIds[3]);
		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(800);

		config.IgnorePoint(3);
		await Assert.That(config.NumResiduals(reconstruction)).IsEqualTo(792);
	}

	[Test]
	public async Task BundleAdjustmentConfig_AddRemoveImage()
	{
		var config = new BundleAdjustmentConfig();
		await Assert.That(config.NumImages).IsEqualTo(0);

		config.AddImage(1);
		config.AddImage(2);
		config.AddImage(3);
		await Assert.That(config.NumImages).IsEqualTo(3);
		await Assert.That(config.HasImage(1)).IsTrue();
		await Assert.That(config.HasImage(2)).IsTrue();
		await Assert.That(config.HasImage(3)).IsTrue();
		await Assert.That(config.HasImage(4)).IsFalse();

		config.RemoveImage(2);
		await Assert.That(config.NumImages).IsEqualTo(2);
		await Assert.That(config.HasImage(1)).IsTrue();
		await Assert.That(config.HasImage(2)).IsFalse();
		await Assert.That(config.HasImage(3)).IsTrue();

		// Removing non-existent image is a no-op
		config.RemoveImage(99);
		await Assert.That(config.NumImages).IsEqualTo(2);
	}

	[Test]
	public async Task BundleAdjustmentConfig_ConstantVariableCamIntrinsics()
	{
		var config = new BundleAdjustmentConfig();
		await Assert.That(config.NumConstantCamIntrinsics).IsEqualTo(0);

		config.SetConstantCamIntrinsics(1);
		config.SetConstantCamIntrinsics(2);
		await Assert.That(config.NumConstantCamIntrinsics).IsEqualTo(2);
		await Assert.That(config.HasConstantCamIntrinsics(1)).IsTrue();
		await Assert.That(config.HasConstantCamIntrinsics(2)).IsTrue();
		await Assert.That(config.HasConstantCamIntrinsics(3)).IsFalse();

		config.SetVariableCamIntrinsics(1);
		await Assert.That(config.NumConstantCamIntrinsics).IsEqualTo(1);
		await Assert.That(config.HasConstantCamIntrinsics(1)).IsFalse();
		await Assert.That(config.HasConstantCamIntrinsics(2)).IsTrue();

		IReadOnlySet<uint> constantCams = config.ConstantCamIntrinsics;
		await Assert.That(constantCams.Count).IsEqualTo(1);
		await Assert.That(constantCams.Contains(2)).IsTrue();
	}

	[Test]
	public async Task BundleAdjustmentConfig_ConstantVariableSensorFromRigPose()
	{
		var config = new BundleAdjustmentConfig();
		await Assert.That(config.NumConstantSensorFromRigPoses).IsEqualTo(0);

		var sensor1 = new SensorId(SensorType.Camera, 1);
		var sensor2 = new SensorId(SensorType.Camera, 2);

		config.SetConstantSensorFromRigPose(sensor1);
		config.SetConstantSensorFromRigPose(sensor2);
		await Assert.That(config.NumConstantSensorFromRigPoses).IsEqualTo(2);
		await Assert.That(config.HasConstantSensorFromRigPose(sensor1)).IsTrue();
		await Assert.That(config.HasConstantSensorFromRigPose(sensor2)).IsTrue();

		config.SetVariableSensorFromRigPose(sensor1);
		await Assert.That(config.NumConstantSensorFromRigPoses).IsEqualTo(1);
		await Assert.That(config.HasConstantSensorFromRigPose(sensor1)).IsFalse();
		await Assert.That(config.HasConstantSensorFromRigPose(sensor2)).IsTrue();

		IReadOnlySet<SensorId> constantPoses = config.ConstantSensorFromRigPoses;
		await Assert.That(constantPoses.Count).IsEqualTo(1);
		await Assert.That(constantPoses.Contains(sensor2)).IsTrue();
	}

	[Test]
	public async Task BundleAdjustmentConfig_ConstantVariableRigFromWorldPose()
	{
		var config = new BundleAdjustmentConfig();
		await Assert.That(config.NumConstantRigFromWorldPoses).IsEqualTo(0);

		config.SetConstantRigFromWorldPose(1);
		config.SetConstantRigFromWorldPose(2);
		await Assert.That(config.NumConstantRigFromWorldPoses).IsEqualTo(2);
		await Assert.That(config.HasConstantRigFromWorldPose(1)).IsTrue();
		await Assert.That(config.HasConstantRigFromWorldPose(2)).IsTrue();

		config.SetVariableRigFromWorldPose(1);
		await Assert.That(config.NumConstantRigFromWorldPoses).IsEqualTo(1);
		await Assert.That(config.HasConstantRigFromWorldPose(1)).IsFalse();
		await Assert.That(config.HasConstantRigFromWorldPose(2)).IsTrue();

		IReadOnlySet<uint> constantRigPoses = config.ConstantRigFromWorldPoses;
		await Assert.That(constantRigPoses.Count).IsEqualTo(1);
		await Assert.That(constantRigPoses.Contains(2)).IsTrue();
	}

	[Test]
	public async Task BundleAdjustmentConfig_ConstantVariablePoints()
	{
		var config = new BundleAdjustmentConfig();
		await Assert.That(config.NumPoints).IsEqualTo(0);
		await Assert.That(config.NumVariablePoints).IsEqualTo(0);
		await Assert.That(config.NumConstantPoints).IsEqualTo(0);

		config.AddVariablePoint(1);
		config.AddVariablePoint(2);
		await Assert.That(config.NumPoints).IsEqualTo(2);
		await Assert.That(config.NumVariablePoints).IsEqualTo(2);
		await Assert.That(config.NumConstantPoints).IsEqualTo(0);
		await Assert.That(config.HasPoint(1)).IsTrue();
		await Assert.That(config.HasVariablePoint(1)).IsTrue();
		await Assert.That(config.HasConstantPoint(1)).IsFalse();

		config.AddConstantPoint(3);
		await Assert.That(config.NumPoints).IsEqualTo(3);
		await Assert.That(config.NumVariablePoints).IsEqualTo(2);
		await Assert.That(config.NumConstantPoints).IsEqualTo(1);
		await Assert.That(config.HasPoint(3)).IsTrue();
		await Assert.That(config.HasVariablePoint(3)).IsFalse();
		await Assert.That(config.HasConstantPoint(3)).IsTrue();

		config.RemoveVariablePoint(1);
		await Assert.That(config.NumVariablePoints).IsEqualTo(1);
		await Assert.That(config.HasPoint(1)).IsFalse();

		config.RemoveConstantPoint(3);
		await Assert.That(config.NumConstantPoints).IsEqualTo(0);
		await Assert.That(config.HasPoint(3)).IsFalse();

		IReadOnlySet<ulong> varPoints = config.VariablePoints;
		await Assert.That(varPoints.Count).IsEqualTo(1);
		await Assert.That(varPoints.Contains(2UL)).IsTrue();
		await Assert.That(config.ConstantPoints.Count).IsEqualTo(0);
	}

	[Test]
	public async Task BundleAdjustmentConfig_IgnoredPoints()
	{
		var config = new BundleAdjustmentConfig();
		await Assert.That(config.IsIgnoredPoint(1)).IsFalse();

		config.IgnorePoint(1);
		await Assert.That(config.IsIgnoredPoint(1)).IsTrue();
		await Assert.That(config.IsIgnoredPoint(2)).IsFalse();
	}

	[Test]
	public async Task BundleAdjustmentConfig_FixGauge()
	{
		var config = new BundleAdjustmentConfig();
		await Assert.That(config.FixedGauge).IsEqualTo(BundleAdjustmentGauge.Unspecified);

		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);
		await Assert.That(config.FixedGauge).IsEqualTo(BundleAdjustmentGauge.TwoCamsFromWorld);

		config.FixGauge(BundleAdjustmentGauge.ThreePoints);
		await Assert.That(config.FixedGauge).IsEqualTo(BundleAdjustmentGauge.ThreePoints);
	}

	[Test]
	public async Task BundleAdjustmentConfig_Images()
	{
		var config = new BundleAdjustmentConfig();
		config.AddImage(5);
		config.AddImage(10);

		IReadOnlySet<uint> images = config.Images;
		await Assert.That(images.Count).IsEqualTo(2);
		await Assert.That(images.Contains(5)).IsTrue();
		await Assert.That(images.Contains(10)).IsTrue();
	}

	[Test]
	public async Task BundleAdjustmentSummary_BriefReport()
	{
		var summary = new BundleAdjustmentSummary
		{
			TerminationType = BundleAdjustmentTerminationType.Convergence,
			NumResiduals = 42,
		};

		string report = summary.BriefReport();
		await Assert.That(report).Contains("CONVERGENCE");
		await Assert.That(report).Contains("42");
	}

	[Test]
	public async Task BundleAdjusterBackendTest_TwoView()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = Synthesize(2, 1, 1, 100);
		AddPoint2DNoise(reconstruction, 1);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions { Backend = BundleAdjustmentBackend.Ceres };
		BundleAdjustmentSummary summary = SolveDefault(options, config, reconstruction);
		await Assert.That(summary.IsSolutionUsable()).IsTrue();
		await Assert.That(summary.NumResiduals).IsEqualTo(400);

		await ExpectConstantCamFromWorld(reconstruction.Image(1), origReconstruction.Image(1));
		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			await ExpectVariablePoint(point3D, origReconstruction.Point3D(point3DId));
		}
	}

	[Test]
	public async Task BundleAdjusterBackendTest_TwoViewConstantCamera()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = Synthesize(2, 1, 1, 100);
		AddPoint2DNoise(reconstruction, 1);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.SetConstantRigFromWorldPose(1);
		config.SetConstantRigFromWorldPose(2);
		config.SetConstantCamIntrinsics(1);

		var options = new BundleAdjustmentOptions { Backend = BundleAdjustmentBackend.Ceres };
		BundleAdjustmentSummary summary = SolveDefault(options, config, reconstruction);
		await Assert.That(summary.IsSolutionUsable()).IsTrue();
		await Assert.That(summary.NumResiduals).IsEqualTo(400);

		await ExpectConstantCamera(reconstruction.Camera(1), origReconstruction.Camera(1));
		await ExpectVariableCamera(reconstruction.Camera(2), origReconstruction.Camera(2));
		await ExpectConstantCamFromWorld(reconstruction.Image(1), origReconstruction.Image(1));
		await ExpectConstantCamFromWorld(reconstruction.Image(2), origReconstruction.Image(2));
		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			await ExpectVariablePoint(point3D, origReconstruction.Point3D(point3DId));
		}
	}

	[Test]
	public async Task BundleAdjusterBackendTest_PartiallyContainedTracks()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = Synthesize(3, 1, 1, 100, numPoints2DWithoutPoint3D: 0);
		AddPoint2DNoise(reconstruction, 1);
		ulong variablePoint3DId = reconstruction.Image(3).Points2D[0].Point3DId;
		reconstruction.DeleteObservation(3, 0);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.SetConstantRigFromWorldPose(1);
		config.SetConstantRigFromWorldPose(2);

		var options = new BundleAdjustmentOptions { Backend = BundleAdjustmentBackend.Ceres };
		BundleAdjustmentSummary summary = SolveDefault(options, config, reconstruction);
		await Assert.That(summary.IsSolutionUsable()).IsTrue();
		await Assert.That(summary.NumResiduals).IsEqualTo(400);

		await ExpectVariableCamera(reconstruction.Camera(1), origReconstruction.Camera(1));
		await ExpectVariableCamera(reconstruction.Camera(2), origReconstruction.Camera(2));
		await ExpectConstantCamera(reconstruction.Camera(3), origReconstruction.Camera(3));
		await ExpectConstantCamFromWorld(reconstruction.Image(3), origReconstruction.Image(3));
		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			if (point3DId == variablePoint3DId)
			{
				await ExpectVariablePoint(point3D, origReconstruction.Point3D(point3DId));
			}
			else
			{
				await ExpectConstantPoint(point3D, origReconstruction.Point3D(point3DId));
			}
		}
	}

	[Test]
	public async Task BundleAdjusterBackendTest_MinimumTrackLength()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = Synthesize(3, 1, 1, 100, numPoints2DWithoutPoint3D: 0);
		AddPoint2DNoise(reconstruction, 1);

		reconstruction.DeleteObservation(3, 0);

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			config.AddImage(imageId);
		}

		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions { Backend = BundleAdjustmentBackend.Ceres, MinTrackLength = 3 };
		BundleAdjustmentSummary summary = SolveDefault(options, config, reconstruction);
		await Assert.That(summary.IsSolutionUsable()).IsTrue();

		// 99 points x 3 observations x 2 residuals per observation. The point with a
		// two-observation track is excluded.
		await Assert.That(summary.NumResiduals).IsEqualTo(594);
	}

	[Test]
	public async Task BundleAdjusterBackendTest_MinimumTrackLengthWithExternalObservations()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = Synthesize(3, 1, 1, 100, numPoints2DWithoutPoint3D: 0);
		AddPoint2DNoise(reconstruction, 1);

		// Shorten one track from three to two observations. The remaining observations are
		// split between a configured image and an external image.
		reconstruction.DeleteObservation(2, 0);

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		foreach (ulong point3DId in reconstruction.Points3D.Keys)
		{
			config.AddVariablePoint(point3DId);
		}

		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions { Backend = BundleAdjustmentBackend.Ceres, MinTrackLength = 3 };
		BundleAdjustmentSummary summary = SolveDefault(options, config, reconstruction);
		await Assert.That(summary.IsSolutionUsable()).IsTrue();

		// 99 points x 3 observations x 2 residuals per observation. The point with a
		// two-observation track is excluded from both configured and external images.
		await Assert.That(summary.NumResiduals).IsEqualTo(594);
	}

	[Test]
	public async Task BundleAdjusterBackendTest_ConstantPoints()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = Synthesize(2, 1, 1, 100);
		AddPoint2DNoise(reconstruction, 1);
		Reconstruction origReconstruction = reconstruction.Clone();

		const ulong constantPoint3DId1 = 1;
		const ulong constantPoint3DId2 = 2;

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.AddConstantPoint(constantPoint3DId1);
		config.AddConstantPoint(constantPoint3DId2);
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions { Backend = BundleAdjustmentBackend.Ceres };
		BundleAdjustmentSummary summary = SolveDefault(options, config, reconstruction);
		await Assert.That(summary.IsSolutionUsable()).IsTrue();
		await Assert.That(summary.NumResiduals).IsEqualTo(400);

		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			if (point3DId == constantPoint3DId1 || point3DId == constantPoint3DId2)
			{
				await ExpectConstantPoint(point3D, origReconstruction.Point3D(point3DId));
			}
			else
			{
				await ExpectVariablePoint(point3D, origReconstruction.Point3D(point3DId));
			}
		}
	}

	[Test]
	public async Task BundleAdjusterBackendTest_ConstantPoints3D()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = Synthesize(2, 1, 1, 20);
		AddPoint2DNoise(reconstruction, 1);
		Reconstruction originalReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			config.AddImage(imageId);
		}

		var options = new BundleAdjustmentOptions { Backend = BundleAdjustmentBackend.Ceres, RefinePoints3D = false };
		BundleAdjustmentSummary summary = SolveDefault(options, config, reconstruction);
		await Assert.That(summary.IsSolutionUsable()).IsTrue();
		await Assert.That(summary.NumResiduals).IsEqualTo(80);

		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			await ExpectConstantPoint(point3D, originalReconstruction.Point3D(point3DId));
		}
	}

	[Test]
	public async Task BundleAdjusterBackendTest_VariableImage()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = Synthesize(3, 1, 1, 100);
		AddPoint2DNoise(reconstruction, 1);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.AddImage(3);
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions { Backend = BundleAdjustmentBackend.Ceres };
		BundleAdjustmentSummary summary = SolveDefault(options, config, reconstruction);
		await Assert.That(summary.IsSolutionUsable()).IsTrue();
		await Assert.That(summary.NumResiduals).IsEqualTo(600);

		await ExpectConstantCamFromWorld(reconstruction.Image(1), origReconstruction.Image(1));
		await ExpectVariableCamFromWorld(reconstruction.Image(3), origReconstruction.Image(3));
		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			await ExpectVariablePoint(point3D, origReconstruction.Point3D(point3DId));
		}
	}

	[Test]
	public async Task BundleAdjusterBackendTest_ConstantFocalLengthAndExtraParams()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = Synthesize(2, 1, 1, 100);
		AddPoint2DNoise(reconstruction, 1);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions
		{
			Backend = BundleAdjustmentBackend.Ceres,
			RefineFocalLength = false,
			RefineExtraParams = false,
		};
		BundleAdjustmentSummary summary = SolveDefault(options, config, reconstruction);
		await Assert.That(summary.IsSolutionUsable()).IsTrue();
		await Assert.That(summary.NumResiduals).IsEqualTo(400);

		foreach ((uint cameraId, Camera camera) in reconstruction.Cameras)
		{
			Camera origCamera = origReconstruction.Camera(cameraId);
			await Assert.That(camera.Params[FocalLengthIdx]).IsEqualTo(origCamera.Params[FocalLengthIdx]);
			await Assert.That(camera.Params[ExtraParamIdx]).IsEqualTo(origCamera.Params[ExtraParamIdx]);
		}
	}

	[Test]
	public async Task BundleAdjusterBackendTest_VariablePrincipalPoint()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = Synthesize(2, 1, 1, 100);
		AddPoint2DNoise(reconstruction, 1);
		Reconstruction origReconstruction = reconstruction.Clone();

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions { Backend = BundleAdjustmentBackend.Ceres, RefinePrincipalPoint = true };
		BundleAdjustmentSummary summary = SolveDefault(options, config, reconstruction);
		await Assert.That(summary.IsSolutionUsable()).IsTrue();
		await Assert.That(summary.NumResiduals).IsEqualTo(400);

		int principalPointIdxX = SimpleRadialCameraModel.PrincipalPointIdxs[0];
		int principalPointIdxY = SimpleRadialCameraModel.PrincipalPointIdxs[1];
		foreach ((uint cameraId, Camera camera) in reconstruction.Cameras)
		{
			Camera origCamera = origReconstruction.Camera(cameraId);
			await Assert.That(camera.Params[principalPointIdxX]).IsNotEqualTo(origCamera.Params[principalPointIdxX]);
			await Assert.That(camera.Params[principalPointIdxY]).IsNotEqualTo(origCamera.Params[principalPointIdxY]);
		}
	}

	[Test]
	public async Task BundleAdjusterBackendTest_IgnorePoint()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = Synthesize(2, 1, 1, 100);
		AddPoint2DNoise(reconstruction, 1);

		var config = new BundleAdjustmentConfig();
		config.AddImage(1);
		config.AddImage(2);
		config.IgnorePoint(42);
		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions { Backend = BundleAdjustmentBackend.Ceres };
		BundleAdjustmentSummary summary = SolveDefault(options, config, reconstruction);
		await Assert.That(summary.IsSolutionUsable()).IsTrue();

		// 99 points (point 42 ignored), 2 images, 2 residuals per observation.
		await Assert.That(summary.NumResiduals).IsEqualTo(396);
	}
}
