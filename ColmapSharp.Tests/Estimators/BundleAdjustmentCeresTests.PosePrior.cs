// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BundleAdjustmentCeresTests.PosePrior: the cases of
// colmap/estimators/bundle_adjustment_ceres_test.cc that check the solved scene through
// ReconstructionNear (ColmapSharp.Tests/ReconstructionMatchers.cs):
// DefaultBundleAdjuster.NominalMultiCameraRig and the five PosePriorBundleAdjuster cases
// (Estimators/BundleAdjustmentCeres.PosePrior.cs). Tier C: the outcome bounds are COLMAP's.
//
// Translation notes: the SQLite test database is an InMemoryDatabase. PrngTestIsolation
// seeds the PRNG with 0 before every test, as COLMAP's gtest_main does.

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
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
	private static Reconstruction SynthesizeWithPriors(
		InMemoryDatabase database, int numCamerasPerRig, int numFramesPerRig, int numPoints3D)
	{
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = 1,
				NumCamerasPerRig = numCamerasPerRig,
				NumFramesPerRig = numFramesPerRig,
				NumPoints3D = numPoints3D,
				PriorPosition = true,
			},
			reconstruction,
			database);
		return reconstruction;
	}

	// The pose and point noise of AlignmentRobustToOutliers and OptimizationRobustToOutliers.
	private static void AddOutlierCaseNoise(Reconstruction reconstruction) =>
		Synthetic.SynthesizeNoise(
			new SyntheticNoiseOptions
			{
				Point3DStddev = 0.2,
				RigFromWorldRotationStddev = 1.0,
				RigFromWorldTranslationStddev = 0.2,
				PriorPositionStddev = 0.05,
			},
			reconstruction);

	// Every image of every registered frame, as the pose prior cases configure BA.
	private static BundleAdjustmentConfig ConfigOfRegFrames(Reconstruction reconstruction)
	{
		var config = new BundleAdjustmentConfig();
		foreach (uint frameId in reconstruction.RegFrameIds)
		{
			Frame frame = reconstruction.Frame(frameId);
			foreach (DataId dataId in frame.ImageIds())
			{
				config.AddImage((uint)dataId.Id);
			}
		}

		return config;
	}

	private static PosePriorBundleAdjustmentOptions PriorOptionsWithSeed0()
	{
		var priorOptions = new PosePriorBundleAdjustmentOptions();
		RansacOptions alignmentRansacOptions = priorOptions.AlignmentRansacOptions;
		alignmentRansacOptions.RandomSeed = 0;
		priorOptions.AlignmentRansacOptions = alignmentRansacOptions;
		return priorOptions;
	}

	private static string? ExplainNearGroundTruth(Reconstruction gtReconstruction, Reconstruction reconstruction, double numObsTolerance) =>
		ReconstructionMatchers.ExplainReconstructionNear(
			gtReconstruction,
			reconstruction,
			maxRotationErrorDeg: 0.1,
			maxProjCenterError: 0.1,
			maxScaleError: null,
			numObsTolerance: numObsTolerance);

	[Test]
	public async Task DefaultBundleAdjuster_NominalMultiCameraRig()
	{
		var gtReconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 3, NumFramesPerRig = 5, NumPoints3D = 200 },
			gtReconstruction);

		Reconstruction reconstruction = gtReconstruction.Clone();
		Synthetic.SynthesizeNoise(
			new SyntheticNoiseOptions
			{
				Point2DStddev = 0.5,
				Point3DStddev = 0.1,
				RigFromWorldRotationStddev = 0.5,
				RigFromWorldTranslationStddev = 0.1,
			},
			reconstruction);

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			config.AddImage(imageId);
		}

		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions();
		BundleAdjuster bundleAdjuster = CeresBundleAdjusters.CreateDefaultCeresBundleAdjuster(options, config, reconstruction);
		BundleAdjustmentSummary summary = bundleAdjuster.Solve();
		await Assert.That(summary.TerminationType).IsNotEqualTo(BundleAdjustmentTerminationType.Failure);

		await Assert.That(ExplainNearGroundTruth(gtReconstruction, reconstruction, 0.0)).IsNull();
	}

	[Test]
	public async Task PosePriorBundleAdjuster_AlignmentRobustToOutliers()
	{
		var database = new InMemoryDatabase();
		Reconstruction gtReconstruction = SynthesizeWithPriors(database, 1, 7, 50);

		Reconstruction reconstruction = gtReconstruction.Clone();
		AddOutlierCaseNoise(reconstruction);

		List<PosePrior> posePriors = database.ReadAllPosePriors();

		// Add 2 outlier priors with very large covariance
		PosePrior prior0 = posePriors[0];
		prior0.Position += new Vector3d(10, 10, 10);
		prior0.PositionCovariance = Matrix3d.Identity * 1e6;
		posePriors[0] = prior0;
		PosePrior prior1 = posePriors[1];
		prior1.Position += new Vector3d(1, 1, 1);
		prior1.PositionCovariance = Matrix3d.Identity * 1e2;
		posePriors[1] = prior1;

		PosePriorBundleAdjustmentOptions priorBaOptions = PriorOptionsWithSeed0();
		RansacOptions alignmentRansacOptions = priorBaOptions.AlignmentRansacOptions;
		alignmentRansacOptions.MaxError = 0.0;
		priorBaOptions.AlignmentRansacOptions = alignmentRansacOptions;

		var baOptions = new BundleAdjustmentOptions();
		BundleAdjustmentConfig baConfig = ConfigOfRegFrames(reconstruction);

		BundleAdjuster adjuster = BundleAdjusters.CreatePosePriorBundleAdjuster(
			baOptions, priorBaOptions, baConfig, posePriors, reconstruction);
		BundleAdjustmentSummary summary = adjuster.Solve();
		await Assert.That(summary.IsSolutionUsable()).IsTrue();

		await Assert.That(ExplainNearGroundTruth(gtReconstruction, reconstruction, 0.02)).IsNull();
	}

	[Test]
	public async Task PosePriorBundleAdjuster_InsufficientPriorsUseTwoCameraGauge()
	{
		var database = new InMemoryDatabase();
		Reconstruction reconstruction = SynthesizeWithPriors(database, 1, 3, 50);

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			config.AddImage(imageId);
		}

		List<PosePrior> posePriors = database.ReadAllPosePriors();
		posePriors.RemoveRange(2, posePriors.Count - 2);
		BundleAdjuster adjuster = BundleAdjusters.CreatePosePriorBundleAdjuster(
			new BundleAdjustmentOptions(),
			new PosePriorBundleAdjustmentOptions(),
			config,
			posePriors,
			reconstruction);

		await Assert.That(adjuster.Config.FixedGauge).IsEqualTo(BundleAdjustmentGauge.TwoCamsFromWorld);
		await Assert.That(adjuster.Solve().IsSolutionUsable()).IsTrue();
	}

	[Test]
	public async Task PosePriorBundleAdjuster_MissingPositionCov()
	{
		var database = new InMemoryDatabase();
		Reconstruction gtReconstruction = SynthesizeWithPriors(database, 1, 7, 100);

		Reconstruction reconstruction = gtReconstruction.Clone();

		List<PosePrior> posePriors = database.ReadAllPosePriors();
		foreach (PosePrior posePrior in posePriors)
		{
			await Assert.That(posePrior.HasPositionCov()).IsFalse();
		}

		PosePriorBundleAdjustmentOptions priorBaOptions = PriorOptionsWithSeed0();
		priorBaOptions.Ceres!.PriorPositionLossFunctionType = BundleAdjustmentLossFunctionType.Cauchy;

		var baOptions = new BundleAdjustmentOptions();
		BundleAdjustmentConfig baConfig = ConfigOfRegFrames(reconstruction);

		BundleAdjuster adjuster = BundleAdjusters.CreatePosePriorBundleAdjuster(
			baOptions, priorBaOptions, baConfig, posePriors, reconstruction);
		BundleAdjustmentSummary summary = adjuster.Solve();
		await Assert.That(summary.IsSolutionUsable()).IsTrue();

		await Assert.That(ExplainNearGroundTruth(gtReconstruction, reconstruction, 0.02)).IsNull();
	}

	[Test]
	public async Task PosePriorBundleAdjuster_ConstantSensorFromRigWithMissingPositionCov()
	{
		var database = new InMemoryDatabase();
		Reconstruction reconstruction = SynthesizeWithPriors(database, 2, 3, 50);

		List<PosePrior> posePriors = database.ReadAllPosePriors();
		foreach (PosePrior posePrior in posePriors)
		{
			await Assert.That(posePrior.HasPositionCov()).IsFalse();
		}

		var baOptions = new BundleAdjustmentOptions { RefineSensorFromRig = false };
		var baConfig = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			baConfig.AddImage(imageId);
		}

		CeresBundleAdjuster adjuster = (CeresBundleAdjuster)BundleAdjusters.CreatePosePriorBundleAdjuster(
			baOptions,
			new PosePriorBundleAdjustmentOptions(),
			baConfig,
			posePriors,
			reconstruction);
		Problem problem = adjuster.Problem;

		foreach ((uint _, Rig rig) in reconstruction.Rigs)
		{
			foreach ((SensorId sensorId, Rigid3d? sensorFromRig) in rig.NonRefSensors)
			{
				await Assert.That(sensorFromRig.HasValue).IsTrue();
				double[] sensorFromRigParams = rig.SensorFromRigStorage(sensorId).Params;
				await Assert.That(problem.HasParameterBlock(sensorFromRigParams)).IsTrue();
				await Assert.That(problem.IsParameterBlockConstant(sensorFromRigParams)).IsTrue();
			}
		}
	}

	[Test]
	public async Task PosePriorBundleAdjuster_OptimizationRobustToOutliers()
	{
		var database = new InMemoryDatabase();
		Reconstruction gtReconstruction = SynthesizeWithPriors(database, 1, 7, 100);

		Reconstruction reconstruction = gtReconstruction.Clone();
		AddOutlierCaseNoise(reconstruction);

		List<PosePrior> posePriors = database.ReadAllPosePriors();

		// Add 2 confident but wrong priors.
		PosePrior prior0 = posePriors[0];
		prior0.PositionCovariance = Matrix3d.Identity * 0.01;
		prior0.Position += new Vector3d(10, 10, 10);
		posePriors[0] = prior0;
		PosePrior prior1 = posePriors[1];
		prior1.PositionCovariance = Matrix3d.Identity * 1.01;
		prior1.Position += new Vector3d(10, 10, 10);
		posePriors[1] = prior1;

		PosePriorBundleAdjustmentOptions priorBaOptions = PriorOptionsWithSeed0();
		priorBaOptions.Ceres!.PriorPositionLossFunctionType = BundleAdjustmentLossFunctionType.Cauchy;

		var baOptions = new BundleAdjustmentOptions();
		BundleAdjustmentConfig baConfig = ConfigOfRegFrames(reconstruction);

		BundleAdjuster adjuster = BundleAdjusters.CreatePosePriorBundleAdjuster(
			baOptions, priorBaOptions, baConfig, posePriors, reconstruction);
		BundleAdjustmentSummary summary = adjuster.Solve();
		await Assert.That(summary.IsSolutionUsable()).IsTrue();

		await Assert.That(ExplainNearGroundTruth(gtReconstruction, reconstruction, 0.02)).IsNull();
	}
}
