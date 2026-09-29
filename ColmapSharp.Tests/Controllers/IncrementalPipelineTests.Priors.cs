// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalPipelineTests.Priors: the pose-prior and random-seed-stability
// TEST(IncrementalPipeline, ...) cases of colmap/controllers/incremental_pipeline_test.cc,
// and the TEST(IncrementalPipelineOptions, ...) cases. The rest of the file is
// IncrementalPipelineTests.cs (shared helpers there).
//
// Skipped lines: the `caspar` assertions of PropagatesExplicitMaxNumIterations and
// DefaultMaxNumIterationsUsesBackendDefaults (ASSERT_TRUE(options.caspar) and its
// solver_iter_max checks): ColmapSharp has no Caspar (GPU) backend options
// (divergence 66). The Ceres lines are ported unchanged.

using ColmapSharp.Controllers;
using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class IncrementalPipelineTests
{
	[Test]
	public async Task IncrementalPipeline_PriorBasedSfMWithoutNoise()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 10, NumPoints3D = 100, PriorPosition = true },
			gt,
			new SyntheticNoiseOptions { Point2DStddev = 0.5, PriorPositionStddev = 0.0 });

		var options = new IncrementalPipelineOptions { UsePriorPosition = true };

		// No noise on prior so do not align gt & computed (expected to be aligned from
		// PositionPriorBundleAdjustment)
		string? near = Near(gt, RunPipeline(options, database), 1e-1, 1e-1, null, 0.02, align: false);
		await Assert.That(near).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_PriorBasedSfMWithoutNoiseAndWithNonTrivialFrames()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 2,
				NumFramesPerRig = 7,
				NumPoints3D = 100,
				CameraHasPriorFocalLength = false,
				PriorPosition = true,
			},
			gt);

		// Match the common rig setup where only the reference sensor has absolute positions.
		// Registering two frames then yields many images but only two usable pose priors.
		var refSensorIds = new HashSet<SensorId>();
		foreach (Rig rig in gt.Rigs.Values)
		{
			refSensorIds.Add(rig.RefSensorId);
		}

		List<PosePrior> posePriors = database.ReadAllPosePriors();
		database.ClearPosePriors();
		foreach (PosePrior posePrior in posePriors)
		{
			if (refSensorIds.Contains(posePrior.CorrDataId.SensorId))
			{
				database.WritePosePrior(posePrior);
			}
		}

		var options = new IncrementalPipelineOptions { UsePriorPosition = true, UseRobustLossOnPriorPosition = true };
		string? near = Near(gt, RunPipeline(options, database), 1e-1, 1e-1, null, 0.02, align: false);
		await Assert.That(near).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_PriorBasedSfMWithNoise()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 7, NumPoints3D = 100, PriorPosition = true },
			gt,
			new SyntheticNoiseOptions { Point2DStddev = 0.5, PriorPositionStddev = 1.5 });

		var options = new IncrementalPipelineOptions { UsePriorPosition = true, UseRobustLossOnPriorPosition = true };
		string? near = Near(gt, RunPipeline(options, database), 1e-1, 1e-1, null, 0.02);
		await Assert.That(near).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_GPSPriorBasedSfMWithNoise()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 10,
				NumPoints3D = 100,
				PriorPosition = true,
				PriorPositionCoordinateSystem = PosePriorCoordinateSystem.Wgs84,
			},
			gt,
			new SyntheticNoiseOptions { Point2DStddev = 0.5, PriorPositionStddev = 1.5 });

		var options = new IncrementalPipelineOptions { UsePriorPosition = true, UseRobustLossOnPriorPosition = true };
		string? near = Near(gt, RunPipeline(options, database), 1e-1, 1e-1, null, 0.02);
		await Assert.That(near).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_SfMWithRandomSeedStability()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 3, NumPoints3D = 50, PriorPosition = false },
			gt,
			new SyntheticNoiseOptions { Point2DStddev = 0.1 });

		const int RandomSeed = 42;
		ReconstructionManager RunMapper(int numThreads, int randomSeed) => RunPipeline(
			new IncrementalPipelineOptions { UsePriorPosition = false, NumThreads = numThreads, RandomSeed = randomSeed },
			database);

		ReconstructionManager manager0 = RunMapper(numThreads: 1, randomSeed: RandomSeed);
		ReconstructionManager manager1 = RunMapper(numThreads: 1, randomSeed: RandomSeed);
		string? eq = manager0.Size == 1 && manager1.Size == 1
			? ReconstructionMatchers.ExplainReconstructionEq(manager0.Get(0), manager1.Get(0))
			: null;

		await Assert.That(manager0.Size).IsEqualTo(1);
		await Assert.That(manager1.Size).IsEqualTo(1);
		await Assert.That(eq).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_PriorBasedSfMWithRandomSeedStability()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 5, NumPoints3D = 50, PriorPosition = true },
			gt,
			new SyntheticNoiseOptions { Point2DStddev = 0.1, PriorPositionStddev = 0.1 });

		const int RandomSeed = 42;
		ReconstructionManager RunMapper(int numThreads, int randomSeed) => RunPipeline(
			new IncrementalPipelineOptions { UsePriorPosition = true, NumThreads = numThreads, RandomSeed = randomSeed },
			database);

		ReconstructionManager manager0 = RunMapper(numThreads: 1, randomSeed: RandomSeed);
		ReconstructionManager manager1 = RunMapper(numThreads: 1, randomSeed: RandomSeed);
		string? eq = manager0.Size == 1 && manager1.Size == 1
			? ReconstructionMatchers.ExplainReconstructionEq(manager0.Get(0), manager1.Get(0))
			: null;

		await Assert.That(manager0.Size).IsEqualTo(1);
		await Assert.That(manager1.Size).IsEqualTo(1);
		await Assert.That(eq).IsNull();
	}

	[Test]
	public async Task IncrementalPipelineOptions_PropagatesExplicitMaxNumIterations()
	{
		// Explicitly set iteration bounds must be forwarded to both backends, including values
		// that coincide with the previous compiled-in defaults.
		var options = new IncrementalPipelineOptions { BaLocalMaxNumIterations = 25, BaGlobalMaxNumIterations = 50 };

		BundleAdjustmentOptions localOptions = options.LocalBundleAdjustment();
		await Assert.That(localOptions.Ceres).IsNotNull();
		await Assert.That(localOptions.Ceres!.SolverOptions.MaxNumIterations).IsEqualTo(25);

		BundleAdjustmentOptions globalOptions = options.GlobalBundleAdjustment();
		await Assert.That(globalOptions.Ceres).IsNotNull();
		await Assert.That(globalOptions.Ceres!.SolverOptions.MaxNumIterations).IsEqualTo(50);
	}

	[Test]
	public async Task IncrementalPipelineOptions_DefaultMaxNumIterationsUsesBackendDefaults()
	{
		// With the -1 sentinel default, each backend must keep its own default: Ceres the
		// previous 25/50 mapper defaults.
		var options = new IncrementalPipelineOptions();
		await Assert.That(options.BaLocalMaxNumIterations).IsEqualTo(-1);
		await Assert.That(options.BaGlobalMaxNumIterations).IsEqualTo(-1);

		BundleAdjustmentOptions localOptions = options.LocalBundleAdjustment();
		await Assert.That(localOptions.Ceres).IsNotNull();
		await Assert.That(localOptions.Ceres!.SolverOptions.MaxNumIterations).IsEqualTo(25);

		BundleAdjustmentOptions globalOptions = options.GlobalBundleAdjustment();
		await Assert.That(globalOptions.Ceres).IsNotNull();
		await Assert.That(globalOptions.Ceres!.SolverOptions.MaxNumIterations).IsEqualTo(50);
	}

	[Test]
	public async Task IncrementalPipelineOptions_EffBaMaxNumIterations()
	{
		var options = new IncrementalPipelineOptions();
		// COLMAP's CasparBundleAdjustmentOptions().solver_iter_max.
		const int CasparDefault = 200;

		// Sentinel resolves to the configured backend's default.
		await Assert.That(options.EffBaLocalMaxNumIterations()).IsEqualTo(25);
		await Assert.That(options.EffBaGlobalMaxNumIterations()).IsEqualTo(50);
		options.BaLocalBackend = BundleAdjustmentBackend.Caspar;
		options.BaGlobalBackend = BundleAdjustmentBackend.Caspar;
		await Assert.That(options.EffBaLocalMaxNumIterations()).IsEqualTo(CasparDefault);
		await Assert.That(options.EffBaGlobalMaxNumIterations()).IsEqualTo(CasparDefault);

		// Explicit values win regardless of backend.
		options.BaLocalMaxNumIterations = 7;
		options.BaGlobalMaxNumIterations = 13;
		await Assert.That(options.EffBaLocalMaxNumIterations()).IsEqualTo(7);
		await Assert.That(options.EffBaGlobalMaxNumIterations()).IsEqualTo(13);
	}
}
