// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RotationAveragingPipelineTests: colmap/controllers/rotation_averaging_test.cc ported 1:1,
// one method per gtest TEST named Suite_Name, testing
// ColmapSharp/Controllers/RotationAveragingPipeline.cs. (The estimator-level
// rotation_averaging_test.cc is Estimators/RotationAveragingTests.cs.)
//
// Tier C (outcome): relative rotations against the ground truth within COLMAP's tolerances;
// the seeded single-threaded runs must agree exactly.
//
// Translation notes: the SQLite database file is InMemoryDatabase. PrngTestIsolation seeds
// the PRNG with 0 before every test, as COLMAP's gtest_main does; the PRNG is per thread, so
// each test runs the controllers before its first await, collecting the largest errors to assert afterwards.

using ColmapSharp.Controllers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public class RotationAveragingPipelineTests
{
	// ExpectEqualRotations: the largest pairwise relative rotation error (radians), to be
	// compared against DegToRad(max_rotation_error_deg).
	private static double MaxRelativeRotationError(Reconstruction gt, Reconstruction computed)
	{
		List<uint> regImageIds = gt.RegImageIds();
		double maxError = 0;
		for (int i = 0; i < regImageIds.Count; i++)
		{
			uint imageId1 = regImageIds[i];
			for (int j = 0; j < i; j++)
			{
				uint imageId2 = regImageIds[j];
				Quaterniond cam2FromCam1 = computed.Image(imageId2).CamFromWorld().Rotation
					* computed.Image(imageId1).CamFromWorld().Rotation.Inverse();
				Quaterniond cam2FromCam1Gt = gt.Image(imageId2).CamFromWorld().Rotation
					* gt.Image(imageId1).CamFromWorld().Rotation.Inverse();
				maxError = Math.Max(maxError, cam2FromCam1.AngularDistance(cam2FromCam1Gt));
			}
		}

		return maxError;
	}

	// ExpectExactEqualRotations: the number of registered images that differ in count or in
	// any rotation coefficient (0 when equal).
	private static int NumRotationMismatches(Reconstruction reconstruction1, Reconstruction reconstruction2)
	{
		List<uint> regImageIds = reconstruction1.RegImageIds();
		if (regImageIds.Count != reconstruction2.RegImageIds().Count)
		{
			return int.MaxValue;
		}

		int mismatches = 0;
		foreach (uint imageId in regImageIds)
		{
			if (reconstruction1.Image(imageId).CamFromWorld().Rotation.Coeffs
				!= reconstruction2.Image(imageId).CamFromWorld().Rotation.Coeffs)
			{
				mismatches++;
			}
		}

		return mismatches;
	}

	private static Reconstruction RunController(Database database, RotationAveragingPipelineOptions options)
	{
		var reconstruction = new Reconstruction();
		var controller = new RotationAveragingPipeline(options, database, reconstruction);
		controller.Run();
		return reconstruction;
	}

	[Test]
	public async Task RotationAveragingPipeline_WithoutNoise()
	{
		using var database = new InMemoryDatabase();
		var gtReconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions { NumRigs = 1, NumCamerasPerRig = 1, NumFramesPerRig = 5, NumPoints3D = 50 },
			gtReconstruction,
			database);

		Reconstruction reconstruction = RunController(database, new RotationAveragingPipelineOptions());

		double maxError = MaxRelativeRotationError(gtReconstruction, reconstruction);
		await Assert.That(maxError).IsLessThan(MathUtils.DegToRad(1e-2));
	}

	[Test]
	public async Task RotationAveragingPipeline_WithNoiseAndOutliers()
	{
		using var database = new InMemoryDatabase();
		var gtReconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 7,
				NumPoints3D = 100,
				InlierMatchRatio = 0.6,
			},
			gtReconstruction,
			database);
		Synthetic.SynthesizeNoise(new SyntheticNoiseOptions { Point2DStddev = 1 }, gtReconstruction, database);

		Reconstruction reconstruction = RunController(database, new RotationAveragingPipelineOptions());

		double maxError = MaxRelativeRotationError(gtReconstruction, reconstruction);
		await Assert.That(maxError).IsLessThan(MathUtils.DegToRad(3.0));
	}

	[Test]
	public async Task RotationAveragingPipeline_WithRandomSeedStability()
	{
		using var database = new InMemoryDatabase();
		var gtReconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 5, NumPoints3D = 100 },
			gtReconstruction,
			database);
		Synthetic.SynthesizeNoise(new SyntheticNoiseOptions { Point2DStddev = 0.5 }, gtReconstruction, database);

		Reconstruction Run(int numThreads, int randomSeed) =>
			RunController(database, new RotationAveragingPipelineOptions { NumThreads = numThreads, RandomSeed = randomSeed });

		const int kRandomSeed = 42;

		// Single-threaded execution.
		int singleThreadMismatches = NumRotationMismatches(
			Run(numThreads: 1, randomSeed: kRandomSeed), Run(numThreads: 1, randomSeed: kRandomSeed));

		// Multi-threaded execution.
		Reconstruction multi0 = Run(numThreads: 3, randomSeed: kRandomSeed);
		Reconstruction multi1 = Run(numThreads: 3, randomSeed: kRandomSeed);
		// Same seed should produce similar results, up to floating-point variations in
		// optimization.
		double multiThreadMaxError = MaxRelativeRotationError(multi0, multi1);

		await Assert.That(singleThreadMismatches).IsEqualTo(0);
		await Assert.That(multiThreadMaxError).IsLessThan(MathUtils.DegToRad(1e-10));
	}

	// C#-only: a cancelled CancellationToken stops the controller before rotation averaging
	// (COLMAP's Run has no stop check), and Progress reports the finished stage otherwise.
	[Test]
	public async Task CSharpOnly_CancellationStopsBeforeRotationAveraging()
	{
		using var database = new InMemoryDatabase();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions { NumRigs = 1, NumCamerasPerRig = 1, NumFramesPerRig = 5, NumPoints3D = 50 },
			new Reconstruction(),
			database);

		var cancelled = new Reconstruction();
		var cancelledController = new RotationAveragingPipeline(new RotationAveragingPipelineOptions(), database, cancelled)
		{
			CancellationToken = new CancellationToken(canceled: true),
		};
		var cancelledStages = new List<string>();
		cancelledController.Progress = new SynchronousProgress(p => cancelledStages.Add(p.Stage));
		cancelledController.Run();
		int numPosedWhenCancelled = cancelled.Images.Values.Count(image => image.HasPose);

		var completed = new Reconstruction();
		var completedStages = new List<string>();
		var controller = new RotationAveragingPipeline(new RotationAveragingPipelineOptions(), database, completed)
		{
			Progress = new SynchronousProgress(p => completedStages.Add(p.Stage)),
		};
		controller.Run();
		int numPosed = completed.Images.Values.Count(image => image.HasPose);

		await Assert.That(numPosedWhenCancelled).IsEqualTo(0);
		await Assert.That(cancelledStages.Count).IsEqualTo(0);
		await Assert.That(numPosed).IsEqualTo(5);
		await Assert.That(completedStages).IsEquivalentTo([RotationAveragingPipeline.RotationAveragingStage]);
	}

	// C#-only: gravity priors seed the rotation of the image their corr_data_id names, not the
	// image whose id equals the prior's pose_prior_id (divergence 102). The
	// priors here have ids 101.. while the images have ids 1.., so a lookup by pose_prior_id
	// finds no image.
	[Test]
	public async Task CSharpOnly_GravityPriorsSeedTheirCorrespondingImage()
	{
		using var database = new InMemoryDatabase();
		var gtReconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = 1,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 5,
				NumPoints3D = 50,
				PriorGravity = true,
			},
			gtReconstruction,
			database);

		List<PosePrior> posePriors = database.ReadAllPosePriors();
		database.ClearPosePriors();
		foreach (PosePrior posePrior in posePriors)
		{
			PosePrior renumbered = posePrior;
			renumbered.PosePriorId = posePrior.PosePriorId + 100;
			database.WritePosePrior(renumbered, usePosePriorId: true);
		}

		var options = new RotationAveragingPipelineOptions();
		options.RotationEstimation.UseGravity = true;
		Reconstruction reconstruction = RunController(database, options);

		int numPosed = reconstruction.Images.Values.Count(image => image.HasPose);
		double maxError = MaxRelativeRotationError(gtReconstruction, reconstruction);

		await Assert.That(numPosed).IsEqualTo(5);
		await Assert.That(maxError).IsLessThan(MathUtils.DegToRad(1e-2));
	}

	// A database with a gravity prior per image, for the stop tests.
	private static InMemoryDatabase GravityPriorDatabase()
	{
		var database = new InMemoryDatabase();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = 1,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 5,
				NumPoints3D = 50,
				PriorGravity = true,
			},
			new Reconstruction(),
			database);
		return database;
	}

	// C#-only (divergence 101): the first stop check runs before gravity
	// seeding. The stop function records whether any frame was already posed when it was
	// first asked; without the check before seeding it would first be asked after seeding.
	[Test]
	public async Task CSharpOnly_StopBeforeGravitySeedingSeedsNothing()
	{
		using InMemoryDatabase database = GravityPriorDatabase();
		var reconstruction = new Reconstruction();
		var controller = new RotationAveragingPipeline(new RotationAveragingPipelineOptions(), database, reconstruction);
		bool? posedAtFirstCheck = null;
		controller.SetCheckIfStoppedFunc(() =>
		{
			posedAtFirstCheck ??= reconstruction.Frames.Values.Any(frame => frame.HasPose);
			return true;
		});
		controller.Run();
		int numPosed = reconstruction.Frames.Values.Count(frame => frame.HasPose);

		// Null (never asked) counts as a failure too.
		await Assert.That(posedAtFirstCheck ?? true).IsFalse();
		await Assert.That(numPosed).IsEqualTo(0);
	}

	// C#-only (divergence 101): a stop that arrives after gravity seeding
	// un-poses the seeded frames instead of leaving gravity rotations with NaN translations.
	[Test]
	public async Task CSharpOnly_StopAfterGravitySeedingUnposesSeededFrames()
	{
		using InMemoryDatabase database = GravityPriorDatabase();
		var reconstruction = new Reconstruction();
		var controller = new RotationAveragingPipeline(new RotationAveragingPipelineOptions(), database, reconstruction);
		int numChecks = 0;
		int numPosedAtStop = 0;
		controller.SetCheckIfStoppedFunc(() =>
		{
			// Let the check before seeding pass, then stop.
			if (++numChecks == 1)
			{
				return false;
			}

			numPosedAtStop = reconstruction.Frames.Values.Count(frame => frame.HasPose);
			return true;
		});
		controller.Run();
		int numPosed = reconstruction.Frames.Values.Count(frame => frame.HasPose);

		await Assert.That(numPosedAtStop).IsEqualTo(5);
		await Assert.That(numPosed).IsEqualTo(0);
	}

	// Progress<T> posts to the thread pool; this one reports inline so the test sees every
	// report before it asserts.
	private sealed class SynchronousProgress(Action<ControllerProgress> report) : IProgress<ControllerProgress>
	{
		public void Report(ControllerProgress value) => report(value);
	}
}
