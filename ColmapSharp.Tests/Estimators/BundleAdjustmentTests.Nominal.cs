// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BundleAdjustmentTests.Nominal: the cases of colmap/estimators/bundle_adjustment_test.cc
// that check the solved scene against ground truth through ReconstructionNear
// (ColmapSharp.Tests/ReconstructionMatchers.cs): BundleAdjusterBackendTest.Nominal and
// .NominalMultiCameraRigConstantSensorFromRig (DefaultBundleAdjuster), and
// PosePriorBundleAdjusterBackendTest.Nominal (PosePriorBundleAdjuster,
// Estimators/BundleAdjustmentCeres.PosePrior.cs). CERES backend only, as in
// BundleAdjustmentTests.cs. Tier C: the outcome bounds are COLMAP's.
//
// Translation notes: the SQLite test database is an InMemoryDatabase. COLMAP's gtest_main
// reseeds the PRNG with 0 at every test start; each case does the same first.

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public partial class BundleAdjustmentTests
{
	// The synthetic scene and noise shared by the three cases.
	private static (Reconstruction Gt, Reconstruction Reconstruction) SynthesizeNoisy(
		int numCamerasPerRig, int numFramesPerRig, int numPoints3D, Database? database = null)
	{
		var gtReconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = 1,
				NumCamerasPerRig = numCamerasPerRig,
				NumFramesPerRig = numFramesPerRig,
				NumPoints3D = numPoints3D,
				PriorPosition = database is not null,
			},
			gtReconstruction,
			database);

		Reconstruction reconstruction = gtReconstruction.Clone();
		var noiseOptions = new SyntheticNoiseOptions
		{
			Point2DStddev = 0.5,
			Point3DStddev = 0.1,
			RigFromWorldRotationStddev = 0.5,
			RigFromWorldTranslationStddev = 0.1,
		};
		if (database is not null)
		{
			noiseOptions.PriorPositionStddev = 0.05;
		}

		Synthetic.SynthesizeNoise(noiseOptions, reconstruction);
		return (gtReconstruction, reconstruction);
	}

	[Test]
	public async Task BundleAdjusterBackendTest_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		(Reconstruction gtReconstruction, Reconstruction reconstruction) = SynthesizeNoisy(1, 10, 200);

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			config.AddImage(imageId);
		}

		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions { Backend = BundleAdjustmentBackend.Ceres };

		BundleAdjuster bundleAdjuster = BundleAdjusters.CreateDefaultBundleAdjuster(options, config, reconstruction);

		// Test abstract interface accessors
		await Assert.That(bundleAdjuster.Options.Backend).IsEqualTo(BundleAdjustmentBackend.Ceres);
		await Assert.That(bundleAdjuster.Config.NumImages).IsEqualTo(10);

		// Solve and verify through abstract interface
		BundleAdjustmentSummary summary = bundleAdjuster.Solve();
		await Assert.That(summary.IsSolutionUsable()).IsTrue();
		await Assert.That(summary.NumResiduals).IsGreaterThan(0);

		await Assert.That(ReconstructionMatchers.ExplainReconstructionNear(
			gtReconstruction,
			reconstruction,
			maxRotationErrorDeg: 0.1,
			maxProjCenterError: 0.1,
			maxScaleError: null,
			numObsTolerance: 0.0)).IsNull();
	}

	[Test]
	public async Task BundleAdjusterBackendTest_NominalMultiCameraRigConstantSensorFromRig()
	{
		RandomUtils.SetPRNGSeed(0);
		(Reconstruction gtReconstruction, Reconstruction reconstruction) = SynthesizeNoisy(2, 10, 200);

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			config.AddImage(imageId);
		}

		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);

		var options = new BundleAdjustmentOptions
		{
			Backend = BundleAdjustmentBackend.Ceres,
			RefineSensorFromRig = false,
		};
		BundleAdjustmentSummary summary = BundleAdjusters.CreateDefaultBundleAdjuster(options, config, reconstruction).Solve();
		await Assert.That(summary.IsSolutionUsable()).IsTrue();

		await Assert.That(ReconstructionMatchers.ExplainReconstructionNear(
			gtReconstruction,
			reconstruction,
			maxRotationErrorDeg: 0.1,
			maxProjCenterError: 0.1,
			maxScaleError: null,
			numObsTolerance: 0.0)).IsNull();
	}

	[Test]
	public async Task PosePriorBundleAdjusterBackendTest_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		(Reconstruction gtReconstruction, Reconstruction reconstruction) = SynthesizeNoisy(1, 7, 100, database);

		List<PosePrior> posePriors = database.ReadAllPosePriors();

		var config = new BundleAdjustmentConfig();
		foreach (uint frameId in reconstruction.RegFrameIds)
		{
			Frame frame = reconstruction.Frame(frameId);
			foreach (DataId dataId in frame.ImageIds())
			{
				config.AddImage((uint)dataId.Id);
			}
		}

		var options = new BundleAdjustmentOptions { Backend = BundleAdjustmentBackend.Ceres };

		var priorOptions = new PosePriorBundleAdjustmentOptions();
		ColmapSharp.Optim.RansacOptions alignmentRansacOptions = priorOptions.AlignmentRansacOptions;
		alignmentRansacOptions.RandomSeed = 0;
		priorOptions.AlignmentRansacOptions = alignmentRansacOptions;

		BundleAdjuster bundleAdjuster = BundleAdjusters.CreatePosePriorBundleAdjuster(
			options, priorOptions, config, posePriors, reconstruction);

		// Test abstract interface accessors
		await Assert.That(bundleAdjuster.Options.Backend).IsEqualTo(BundleAdjustmentBackend.Ceres);
		await Assert.That(bundleAdjuster.Config.NumImages).IsEqualTo(7);

		// Solve and verify through abstract interface
		BundleAdjustmentSummary summary = bundleAdjuster.Solve();
		await Assert.That(summary.IsSolutionUsable()).IsTrue();
		await Assert.That(summary.NumResiduals).IsGreaterThan(0);

		await Assert.That(ReconstructionMatchers.ExplainReconstructionNear(
			gtReconstruction,
			reconstruction,
			maxRotationErrorDeg: 0.1,
			maxProjCenterError: 0.1,
			maxScaleError: null,
			numObsTolerance: 0.02)).IsNull();
	}
}
