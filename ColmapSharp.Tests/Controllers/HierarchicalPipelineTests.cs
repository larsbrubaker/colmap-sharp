// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// HierarchicalPipelineTests: colmap/controllers/hierarchical_pipeline_test.cc ported 1:1, one
// method per gtest TEST named Suite_Name, testing ColmapSharp/Controllers/HierarchicalPipeline.cs.
//
// Tier C (outcome): the merged reconstruction against the ground truth after alignment via
// projection centers, with COLMAP's bounds.
//
// Translation notes: the SQLite database file is InMemoryDatabase. COLMAP's gtest_main seeds
// the PRNG with 0 before every test; the PRNG is per thread, so each test seeds it and runs
// the pipeline before its first await. ExpectEqualReconstructions collects its failures as
// messages, asserted once; ASSERT_* preconditions throw through Require.

using ColmapSharp.Controllers;
using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public class HierarchicalPipelineTests
{
	private static void Require(bool condition, string message = "Test precondition failed")
	{
		if (!condition)
		{
			throw new InvalidOperationException(message);
		}
	}

	// ExpectEqualReconstructions: the failed expectations (empty when all hold).
	private static List<string> ExpectEqualReconstructions(
		Reconstruction gt,
		Reconstruction computed,
		double maxRotationErrorDeg,
		double maxProjCenterError,
		double numObsTolerance)
	{
		var failures = new List<string>();
		if (computed.NumCameras != gt.NumCameras)
		{
			failures.Add($"NumCameras {computed.NumCameras} != {gt.NumCameras}");
		}

		if (computed.NumImages != gt.NumImages)
		{
			failures.Add($"NumImages {computed.NumImages} != {gt.NumImages}");
		}

		if (computed.NumRegImages != gt.NumRegImages)
		{
			failures.Add($"NumRegImages {computed.NumRegImages} != {gt.NumRegImages}");
		}

		if (computed.ComputeNumObservations() < (1 - numObsTolerance) * gt.ComputeNumObservations())
		{
			failures.Add($"NumObservations {computed.ComputeNumObservations()} < {gt.ComputeNumObservations()}");
		}

		var gtFromComputed = new Sim3d();
		Require(Alignment.AlignReconstructionsViaProjCenters(computed, gt, 0.1, ref gtFromComputed));

		List<ImageAlignmentError> errors = Alignment.ComputeImageAlignmentError(computed, gt, gtFromComputed);
		if (errors.Count != gt.NumImages)
		{
			failures.Add($"{errors.Count} alignment errors for {gt.NumImages} images");
		}

		foreach (ImageAlignmentError error in errors)
		{
			if (!(error.RotationErrorDeg < maxRotationErrorDeg))
			{
				failures.Add($"Image {error.ImageName} rotation error {error.RotationErrorDeg}");
			}

			if (!(error.ProjCenterError < maxProjCenterError))
			{
				failures.Add($"Image {error.ImageName} projection center error {error.ProjCenterError}");
			}
		}

		return failures;
	}

	// The mean reprojection error after the run and after UpdatePoint3DErrors (must be equal).
	private static (double AfterRun, double Recomputed) MeanErrors(Reconstruction reconstruction)
	{
		Require(reconstruction.NumPoints3D > 0);
		double meanAfterRun = reconstruction.ComputeMeanReprojectionError();
		reconstruction.UpdatePoint3DErrors();
		return (meanAfterRun, reconstruction.ComputeMeanReprojectionError());
	}

	private static ReconstructionManager Run(HierarchicalPipelineOptions options, Database database)
	{
		var reconstructionManager = new ReconstructionManager();
		var mapper = new HierarchicalPipeline(options, database, reconstructionManager);
		mapper.Run();
		return reconstructionManager;
	}

	[Test]
	public async Task HierarchicalPipeline_WithoutNoise()
	{
		RandomUtils.SetPRNGSeed(0);
		using var database = new InMemoryDatabase();
		var gt = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 20, NumPoints3D = 100 },
			gt,
			database);

		var mapperOptions = new HierarchicalPipelineOptions();
		mapperOptions.ClusteringOptions.LeafMaxNumImages = 5;
		mapperOptions.ClusteringOptions.ImageOverlap = 3;
		ReconstructionManager reconstructionManager = Run(mapperOptions, database);

		Require(reconstructionManager.Size == 1, $"Size {reconstructionManager.Size}: {string.Join(",", Enumerable.Range(0, reconstructionManager.Size).Select(i => reconstructionManager.Get(i).NumRegImages))}");
		Reconstruction reconstruction = reconstructionManager.Get(0);
		List<string> failures = ExpectEqualReconstructions(gt, reconstruction, 1e-2, 5e-4, 0);

		// After the pipeline runs, point3D.error must be in pixel units, i.e. equal to what
		// UpdatePoint3DErrors would recompute.
		(double afterRun, double recomputed) = MeanErrors(reconstruction);

		await Assert.That(failures).IsEmpty();
		await Assert.That(GTestDouble.DoubleEq(afterRun, recomputed)).IsTrue();
	}

	[Test]
	public async Task HierarchicalPipeline_WithoutNoiseAndNonTrivialFrames()
	{
		RandomUtils.SetPRNGSeed(0);
		using var database = new InMemoryDatabase();
		var gt = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 2,
				NumFramesPerRig = 10,
				NumPoints3D = 100,
				SensorFromRigTranslationStddev = 0.05,
				SensorFromRigRotationStddev = 30,
			},
			gt,
			database);

		var mapperOptions = new HierarchicalPipelineOptions();
		mapperOptions.ClusteringOptions.LeafMaxNumImages = 10;
		mapperOptions.ClusteringOptions.ImageOverlap = 3;
		// Note that the hierarchical mapper does not work well when the sensor_from_rig poses
		// are inconsistently refined in different clusters, because then the merging does not
		// work well.
		mapperOptions.IncrementalOptions.BaRefineSensorFromRig = false;
		ReconstructionManager reconstructionManager = Run(mapperOptions, database);

		Require(reconstructionManager.Size == 1, $"Size {reconstructionManager.Size}: {string.Join(",", Enumerable.Range(0, reconstructionManager.Size).Select(i => reconstructionManager.Get(i).NumRegImages))}");
		List<string> failures = ExpectEqualReconstructions(gt, reconstructionManager.Get(0), 1e-2, 1e-3, 0);
		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task HierarchicalPipeline_WithoutNoiseAndPanoramicNonTrivialFrames()
	{
		RandomUtils.SetPRNGSeed(0);
		using var database = new InMemoryDatabase();
		var gt = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 3,
				NumFramesPerRig = 10,
				NumPoints3D = 100,
				SensorFromRigTranslationStddev = 0,
				SensorFromRigRotationStddev = 30,
			},
			gt,
			database);

		var mapperOptions = new HierarchicalPipelineOptions();
		mapperOptions.ClusteringOptions.LeafMaxNumImages = 10;
		mapperOptions.ClusteringOptions.ImageOverlap = 3;
		// Note that the hierarchical mapper does not work well when the sensor_from_rig poses
		// are inconsistently refined in different clusters, because then the merging does not
		// work well.
		mapperOptions.IncrementalOptions.BaRefineSensorFromRig = false;
		ReconstructionManager reconstructionManager = Run(mapperOptions, database);

		Require(reconstructionManager.Size == 1, $"Size {reconstructionManager.Size}: {string.Join(",", Enumerable.Range(0, reconstructionManager.Size).Select(i => reconstructionManager.Get(i).NumRegImages))}");
		List<string> failures = ExpectEqualReconstructions(gt, reconstructionManager.Get(0), 1e-2, 1e-3, 0);
		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task HierarchicalPipeline_MultiReconstruction()
	{
		RandomUtils.SetPRNGSeed(0);
		using var database = new InMemoryDatabase();
		var gt1 = new Reconstruction();
		var gt2 = new Reconstruction();
		var syntheticOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 5,
			NumPoints3D = 50,
		};
		Synthetic.SynthesizeDataset(syntheticOptions, gt1, database);
		syntheticOptions.NumFramesPerRig = 4;
		Synthetic.SynthesizeDataset(syntheticOptions, gt2, database);

		var mapperOptions = new HierarchicalPipelineOptions();
		mapperOptions.ClusteringOptions.LeafMaxNumImages = 5;
		mapperOptions.ClusteringOptions.ImageOverlap = 3;
		ReconstructionManager reconstructionManager = Run(mapperOptions, database);

		Require(reconstructionManager.Size == 2);
		(Reconstruction computed1, Reconstruction computed2) = reconstructionManager.Get(0).NumRegImages == 5
			? (reconstructionManager.Get(0), reconstructionManager.Get(1))
			: (reconstructionManager.Get(1), reconstructionManager.Get(0));
		List<string> failures1 = ExpectEqualReconstructions(gt1, computed1, 1e-2, 1e-4, 0);
		List<string> failures2 = ExpectEqualReconstructions(gt2, computed2, 1e-2, 1e-4, 0);

		// After the pipeline runs, point3D.error must be in pixel units for every
		// reconstruction in the manager, i.e. equal to what UpdatePoint3DErrors would
		// recompute.
		(double afterRun1, double recomputed1) = MeanErrors(computed1);
		(double afterRun2, double recomputed2) = MeanErrors(computed2);

		await Assert.That(failures1).IsEmpty();
		await Assert.That(failures2).IsEmpty();
		await Assert.That(GTestDouble.DoubleEq(afterRun1, recomputed1)).IsTrue();
		await Assert.That(GTestDouble.DoubleEq(afterRun2, recomputed2)).IsTrue();
	}

	// C#-only (docs/CPP_DIVERGENCES.md, entry 108): cancelling after the first cluster stops
	// the remaining clusters' incremental pipelines, and the run returns without merging.
	[Test]
	public async Task CSharpOnly_CancellationStopsBeforeMerging()
	{
		RandomUtils.SetPRNGSeed(0);
		using var database = new InMemoryDatabase();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 20, NumPoints3D = 100 },
			new Reconstruction(),
			database);

		var mapperOptions = new HierarchicalPipelineOptions { NumWorkers = 1 };
		mapperOptions.ClusteringOptions.LeafMaxNumImages = 5;
		mapperOptions.ClusteringOptions.ImageOverlap = 3;
		using var cancellation = new CancellationTokenSource();
		var stages = new List<string>();
		var reconstructionManager = new ReconstructionManager();
		var mapper = new HierarchicalPipeline(mapperOptions, database, reconstructionManager)
		{
			CancellationToken = cancellation.Token,
			Progress = new SynchronousProgress(progress =>
			{
				stages.Add(progress.Stage);
				cancellation.Cancel();
			}),
		};
		mapper.Run();

		await Assert.That(stages.Count).IsGreaterThan(1);
		await Assert.That(stages.Contains(HierarchicalPipeline.MergingStage)).IsFalse();
		await Assert.That(reconstructionManager.Size).IsEqualTo(0);
	}

	// C#-only (docs/CPP_DIVERGENCES.md, entry 121). Parallel.ForEach runs clusters on the
	// calling thread and on reused pool threads, whose PRNGs carry whatever earlier work drew,
	// whereas COLMAP's workers are fresh threads. Before the fix a cluster continued that
	// stream: HierarchicalPipeline_WithoutNoise passed alone but missed its 5e-4 bound in the
	// full suite, and one worker (every cluster on the calling thread, which then also merges)
	// gave different poses than eight.
	[Test]
	public async Task CSharpOnly_ResultIgnoresClusterSchedule()
	{
		RandomUtils.SetPRNGSeed(0);
		using var database = new InMemoryDatabase();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 20, NumPoints3D = 100 },
			new Reconstruction(),
			database);

		List<(uint ImageId, Rigid3d CamFromWorld)> Poses(int numWorkers)
		{
			RandomUtils.SetPRNGSeed(0);
			var mapperOptions = new HierarchicalPipelineOptions { NumWorkers = numWorkers, NumThreads = 8 };
			mapperOptions.ClusteringOptions.LeafMaxNumImages = 5;
			mapperOptions.ClusteringOptions.ImageOverlap = 3;
			Reconstruction reconstruction = Run(mapperOptions, database).Get(0);
			return [.. reconstruction.RegImageIds().Order().Select(id => (id, reconstruction.Image(id).CamFromWorld()))];
		}

		List<(uint ImageId, Rigid3d CamFromWorld)> oneWorker = Poses(1);
		List<(uint ImageId, Rigid3d CamFromWorld)> eightWorkers = Poses(8);

		await Assert.That(eightWorkers).IsEquivalentTo(oneWorker, TUnit.Assertions.Enums.CollectionOrdering.Matching);
	}

	// Progress<T> posts to the thread pool; this one reports inline so the cancellation
	// happens before the next cluster starts.
	private sealed class SynchronousProgress(Action<ControllerProgress> report) : IProgress<ControllerProgress>
	{
		public void Report(ControllerProgress value) => report(value);
	}
}
