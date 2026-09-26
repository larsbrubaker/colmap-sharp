// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalPipelineTests: the TEST(IncrementalPipeline, ...) cases of
// colmap/controllers/incremental_pipeline_test.cc ported 1:1, testing
// ColmapSharp/Controllers/IncrementalPipeline*.cs end to end on synthetic datasets. The
// pose-prior, seed-stability and TEST(IncrementalPipelineOptions, ...) cases are in
// IncrementalPipelineTests.Priors.cs, the C#-only cancellation/progress/color tests in
// IncrementalPipelineTests.Host.cs.
//
// Tier C (outcome): ReconstructionNear against the synthetic ground truth with COLMAP's
// bounds; ReconstructionEq for the seed-stability tests.
//
// Translation notes: the SQLite test database is InMemoryDatabase. PrngTestIsolation seeds
// the PRNG with 0 before every test, as COLMAP's gtest_main does; the PRNG is per thread, so
// each test runs all of its mapping before its first await, collecting values to assert
// afterwards. ASSERT_EQ(size, n) preconditions become Require.

using ColmapSharp.Controllers;
using ColmapSharp.Geometry;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Controllers;

public partial class IncrementalPipelineTests
{
	// ASSERT_* inside the C++ tests: fail the test when a precondition does not hold.
	private static void Require(bool condition, string message = "Test precondition failed")
	{
		if (!condition)
		{
			throw new InvalidOperationException(message);
		}
	}

	// Synthesizes the dataset into a fresh database.
	private static InMemoryDatabase Synthesize(
		SyntheticDatasetOptions options, Reconstruction gtReconstruction, SyntheticNoiseOptions? noise = null)
	{
		var database = new InMemoryDatabase();
		Synthetic.SynthesizeDataset(options, gtReconstruction, database);
		if (noise is not null)
		{
			Synthetic.SynthesizeNoise(noise, gtReconstruction, database);
		}

		return database;
	}

	private static ReconstructionManager RunPipeline(IncrementalPipelineOptions options, Database database)
	{
		var reconstructionManager = new ReconstructionManager();
		var mapper = new IncrementalPipeline(options, database, reconstructionManager);
		mapper.Run();
		return reconstructionManager;
	}

	private static string? Near(
		Reconstruction gt,
		ReconstructionManager manager,
		double maxRotationErrorDeg,
		double maxProjCenterError,
		double? maxScaleError = null,
		double numObsTolerance = 0.0,
		bool align = true)
	{
		Require(manager.Size == 1, $"Expected 1 reconstruction, got {manager.Size}");
		return ReconstructionMatchers.ExplainReconstructionNear(
			gt, manager.Get(0), maxRotationErrorDeg, maxProjCenterError, maxScaleError, numObsTolerance, align);
	}

	[Test]
	public async Task IncrementalPipeline_WithoutNoise()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 7, NumPoints3D = 50, CameraHasPriorFocalLength = false },
			gt);
		string? near = Near(gt, RunPipeline(new IncrementalPipelineOptions(), database), 1e-2, 1e-4);
		await Assert.That(near).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_WithoutNoiseSphericalCameras()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 5,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 1,
				NumPoints3D = 100,
				CameraModelId = CameraModelId.Equirectangular,
				CameraWidth = 1000,
				CameraHeight = 500,
				CameraParams = [1000, 500],
			},
			gt);
		ReconstructionManager manager = RunPipeline(new IncrementalPipelineOptions(), database);
		Require(manager.Size == 1);
		int numRegImages = manager.Get(0).NumRegImages;
		string? near = Near(gt, manager, 1e-2, 1e-4);
		await Assert.That(numRegImages).IsEqualTo(gt.NumImages);
		await Assert.That(near).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_WithoutNoiseAndWithNonTrivialFrames()
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
				SensorFromRigTranslationStddev = 0.05,
				SensorFromRigRotationStddev = 30,
			},
			gt);

		var results = new List<string?>();
		foreach (bool refineSensorFromRig in new[] { true, false })
		{
			var options = new IncrementalPipelineOptions { BaRefineSensorFromRig = refineSensorFromRig };
			results.Add(Near(gt, RunPipeline(options, database), 1e-2, 1e-3, refineSensorFromRig ? 1e-2 : 1e-4));
		}

		await Assert.That(results[0]).IsNull();
		await Assert.That(results[1]).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_UnknownSensorFromRigExitsGracefully()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 3,
				NumFramesPerRig = 5,
				NumPoints3D = 50,
				CameraHasPriorFocalLength = false,
				SensorFromRigTranslationStddev = 0.05,
				SensorFromRigRotationStddev = 30,
			},
			gt);

		// Set one of the sensor from rig to unknown.
		Rig rig = database.ReadAllRigs()[0];
		rig.ResetSensorFromRig(rig.NonRefSensors.First().Key);
		database.UpdateRig(rig);

		ReconstructionManager manager = RunPipeline(new IncrementalPipelineOptions(), database);
		await Assert.That(manager.Size).IsEqualTo(0);
	}

	[Test]
	public async Task IncrementalPipeline_WithNonTrivialFramesAndConstantRigsAndCameras()
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
				SensorFromRigTranslationStddev = 0.05,
				SensorFromRigRotationStddev = 30,
			},
			gt);

		const uint ConstantRigId = 1;
		const uint ConstantCameraId = 1;

		var options = new IncrementalPipelineOptions();
		options.ConstantRigs.Add(ConstantRigId);
		options.ConstantCameras.Add(ConstantCameraId);
		ReconstructionManager manager = RunPipeline(options, database);
		string? near = Near(gt, manager, 1e-2, 1e-3);
		Reconstruction reconstruction = manager.Get(0);

		var sensorChecks = new List<bool>();
		foreach (KeyValuePair<SensorId, Rigid3d?> sensor in reconstruction.Rig(ConstantRigId).NonRefSensors)
		{
			sensorChecks.Add(Rigid3dMatchers.Rigid3dNear(
				sensor.Value!.Value, gt.Rig(ConstantRigId).SensorFromRig(sensor.Key), rtol: 1e-4, ttol: 1e-4));
		}

		double[] params1 = [.. reconstruction.Camera(ConstantCameraId).Params];
		double[] gtParams1 = [.. gt.Camera(ConstantCameraId).Params];

		await Assert.That(near).IsNull();
		await Assert.That(sensorChecks).DoesNotContain(false);
		await Assert.That(params1).IsEquivalentTo(gtParams1);
	}

	[Test]
	public async Task IncrementalPipeline_WithoutNoiseAndWithPanoramicNonTrivialFrames()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 3,
				NumFramesPerRig = 7,
				NumPoints3D = 100,
				CameraHasPriorFocalLength = false,
				SensorFromRigTranslationStddev = 0,
				SensorFromRigRotationStddev = 30,
			},
			gt);

		var results = new List<string?>();
		foreach (bool refineSensorFromRig in new[] { true, false })
		{
			var options = new IncrementalPipelineOptions { BaRefineSensorFromRig = refineSensorFromRig };
			results.Add(Near(gt, RunPipeline(options, database), 1e-2, 1e-3));
		}

		await Assert.That(results[0]).IsNull();
		await Assert.That(results[1]).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_WithPriorFocalLength()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 7, NumPoints3D = 50, CameraHasPriorFocalLength = true },
			gt);
		string? near = Near(gt, RunPipeline(new IncrementalPipelineOptions(), database), 1e-2, 1e-4);
		await Assert.That(near).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_WithNoise()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 7, NumPoints3D = 100 },
			gt,
			new SyntheticNoiseOptions { Point2DStddev = 0.5 });
		ReconstructionManager manager = RunPipeline(new IncrementalPipelineOptions(), database);
		string? near = Near(gt, manager, 1e-1, 1e-1, null, 0.02);

		// After the pipeline runs, point3D.error must be in pixel units, i.e. equal to what
		// UpdatePoint3DErrors would recompute.
		Reconstruction reconstruction = manager.Get(0);
		Require(reconstruction.NumPoints3D > 0);
		double meanAfterRun = reconstruction.ComputeMeanReprojectionError();
		reconstruction.UpdatePoint3DErrors();
		double meanRecomputed = reconstruction.ComputeMeanReprojectionError();

		await Assert.That(near).IsNull();
		// EXPECT_DOUBLE_EQ: within 4 ULPs.
		await Assert.That(GTestDouble.DoubleEq(meanAfterRun, meanRecomputed)).IsTrue();
	}

	[Test]
	public async Task IncrementalPipeline_IgnoreRedundantPoints3D()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 7, NumPoints3D = 50 },
			gt);
		var options = new IncrementalPipelineOptions();
		options.MapperOptions.BaGlobalIgnoreRedundantPoints3D = true;
		options.MapperOptions.BaGlobalIgnoreRedundantPoints3DMinCoverageGain = 0.5;
		string? near = Near(gt, RunPipeline(options, database), 1e-2, 1e-4);
		await Assert.That(near).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_StructureLessRegistrationOnly()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 7, NumPoints3D = 50 },
			gt);
		var options = new IncrementalPipelineOptions { StructureLessRegistrationOnly = true };
		string? near = Near(gt, RunPipeline(options, database), 1e-3, 1e-4);
		await Assert.That(near).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_MultiReconstruction()
	{
		using var database = new InMemoryDatabase();
		var gt1 = new Reconstruction();
		var gt2 = new Reconstruction();
		var syntheticOptions = new SyntheticDatasetOptions { NumRigs = 1, NumCamerasPerRig = 1, NumFramesPerRig = 5, NumPoints3D = 50 };
		Synthetic.SynthesizeDataset(syntheticOptions, gt1, database);
		syntheticOptions.NumFramesPerRig = 4;
		Synthetic.SynthesizeDataset(syntheticOptions, gt2, database);

		var options = new IncrementalPipelineOptions { MinModelSize = 4 };
		ReconstructionManager manager = RunPipeline(options, database);

		Require(manager.Size == 2, $"Expected 2 reconstructions, got {manager.Size}");
		Reconstruction computed1;
		Reconstruction computed2;
		if (manager.Get(0).NumRegImages == 5)
		{
			computed1 = manager.Get(0);
			computed2 = manager.Get(1);
		}
		else
		{
			computed1 = manager.Get(1);
			computed2 = manager.Get(0);
		}

		string? near1 = ReconstructionMatchers.ExplainReconstructionNear(gt1, computed1, 1e-2, 1e-4);
		string? near2 = ReconstructionMatchers.ExplainReconstructionNear(gt2, computed2, 1e-2, 1e-4);
		await Assert.That(near1).IsNull();
		await Assert.That(near2).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_FixExistingFrames()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 1, NumCamerasPerRig = 1, NumFramesPerRig = 7, NumPoints3D = 50, CameraHasPriorFocalLength = false },
			gt);

		var reconstructionManager = new ReconstructionManager();
		var options = new IncrementalPipelineOptions();
		var results = new List<string?>();
		foreach (bool fixExistingFrames in new[] { false, true })
		{
			if (fixExistingFrames)
			{
				Require(reconstructionManager.Size == 1);
				Reconstruction reconstruction = reconstructionManager.Get(0);
				// De-register a frame that expect to be re-registered in the second run.
				reconstruction.DeRegisterFrame(1);
				// Clear all the observations of one image but keep it registered. We do not
				// expect fixed images to be filtered (due to insufficient observations).
				Image image2 = reconstruction.Image(2);
				for (uint point2DIdx = 0; point2DIdx < image2.NumPoints2D; ++point2DIdx)
				{
					if (image2.Points2D[(int)point2DIdx].HasPoint3D)
					{
						reconstruction.DeleteObservation(image2.ImageId, point2DIdx);
					}
				}
			}

			options.FixExistingFrames = fixExistingFrames;
			var mapper = new IncrementalPipeline(options, database, reconstructionManager);
			mapper.Run();

			results.Add(Near(gt, reconstructionManager, 1e-2, 1e-4));
		}

		await Assert.That(results[0]).IsNull();
		await Assert.That(results[1]).IsNull();
	}

	[Test]
	public async Task IncrementalPipeline_ChainedMatches()
	{
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				MatchConfig = SyntheticMatchConfig.Chained,
				NumRigs = 1,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 4,
				NumPoints3D = 100,
			},
			gt);
		var options = new IncrementalPipelineOptions { NumThreads = 1 };
		string? near = Near(gt, RunPipeline(options, database), 1e-2, 1e-4);
		await Assert.That(near).IsNull();
	}
}
