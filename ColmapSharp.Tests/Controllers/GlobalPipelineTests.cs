// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GlobalPipelineTests: colmap/controllers/global_pipeline_test.cc ported 1:1, one method per
// gtest TEST named Suite_Name, testing ColmapSharp/Controllers/GlobalPipeline.cs. This file
// holds the helpers and the single-model cases; GlobalPipelineTests.Components.cs holds the
// multi-component cases.
//
// Tier C (outcome): reconstructions against the ground truth through ReconstructionNear at
// COLMAP's bounds; ReconstructionEq for the seeded single-threaded runs.
//
// Translation notes: the SQLite database file is InMemoryDatabase. COLMAP's gtest_main seeds
// the PRNG with 0 before every test (several tests reseed it with 1); the PRNG is per thread,
// so each test seeds it and runs everything before its first await. ASSERT_* preconditions
// throw through Require. testing::UnorderedElementsAreArray over sets of image ids is
// SameSets.

using ColmapSharp.Controllers;
using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.Util;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Controllers;

public partial class GlobalPipelineTests
{
	private static void Require(bool condition, string message = "Test precondition failed")
	{
		if (!condition)
		{
			throw new InvalidOperationException(message);
		}
	}

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

	private static ReconstructionManager RunPipeline(GlobalPipelineOptions options, Database database)
	{
		var reconstructionManager = new ReconstructionManager();
		var mapper = new GlobalPipeline(options, database, reconstructionManager);
		mapper.Run();
		return reconstructionManager;
	}

	// RegImageIdSetsPerReconstruction.
	private static List<HashSet<uint>> RegImageIdSetsPerReconstruction(ReconstructionManager reconstructionManager)
	{
		var imageIdSets = new List<HashSet<uint>>();
		for (int i = 0; i < reconstructionManager.Size; ++i)
		{
			imageIdSets.Add([.. reconstructionManager.Get(i).RegImageIds()]);
		}

		return imageIdSets;
	}

	// UnorderedElementsAreArray over sets: each actual set matches a distinct expected set.
	private static bool SameSets(List<HashSet<uint>> actual, List<HashSet<uint>> expected)
	{
		if (actual.Count != expected.Count)
		{
			return false;
		}

		var used = new bool[expected.Count];
		foreach (HashSet<uint> set in actual)
		{
			int match = expected.FindIndex(e => !used[expected.IndexOf(e)] && e.SetEquals(set));
			if (match < 0)
			{
				return false;
			}

			used[match] = true;
		}

		return true;
	}

	// GroupImageIdsByRig: the images of the reconstruction grouped by rig (in rig id order).
	private static List<HashSet<uint>> GroupImageIdsByRig(Reconstruction reconstruction)
	{
		var imagesByRig = new SortedDictionary<uint, HashSet<uint>>();
		foreach ((_, Frame frame) in reconstruction.Frames)
		{
			if (!imagesByRig.TryGetValue(frame.RigId, out HashSet<uint>? imageIds))
			{
				imageIds = [];
				imagesByRig.Add(frame.RigId, imageIds);
			}

			foreach (DataId dataId in frame.ImageIds())
			{
				imageIds.Add((uint)dataId.Id);
			}
		}

		return [.. imagesByRig.Values];
	}

	// ExtractGroundTruthSubset: the ground truth restricted to the group's frames.
	private static Reconstruction ExtractGroundTruthSubset(Reconstruction gtReconstruction, HashSet<uint> groupImageIds)
	{
		Reconstruction subset = gtReconstruction.Clone();
		var framesToDeregister = new List<uint>();
		foreach ((uint frameId, Frame frame) in subset.Frames)
		{
			bool inGroup = frame.ImageIds().Any(dataId => groupImageIds.Contains((uint)dataId.Id));
			if (!inGroup)
			{
				framesToDeregister.Add(frameId);
			}
		}

		foreach (uint frameId in framesToDeregister)
		{
			subset.DeRegisterFrame(frameId);
		}

		subset.TearDown();
		return subset;
	}

	private static Dictionary<uint, int> ImageToGroup(List<HashSet<uint>> groups)
	{
		var imageToGroup = new Dictionary<uint, int>();
		for (int group = 0; group < groups.Count; ++group)
		{
			foreach (uint imageId in groups[group])
			{
				imageToGroup[imageId] = group;
			}
		}

		return imageToGroup;
	}

	// DisconnectDatabaseComponents: deletes every cross-group pair.
	private static void DisconnectDatabaseComponents(List<HashSet<uint>> groups, Database database)
	{
		Dictionary<uint, int> imageToGroup = ImageToGroup(groups);
		foreach ((ulong pairId, _) in database.ReadTwoViewGeometries())
		{
			(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
			if (imageToGroup[imageId1] != imageToGroup[imageId2])
			{
				database.DeleteTwoViewGeometry(imageId1, imageId2);
				database.DeleteInlierMatches(imageId1, imageId2);
				database.DeleteMatches(imageId1, imageId2);
			}
		}
	}

	// BridgeGroupsWithOutlierEdges: keeps numOutlierEdges cross-group pairs with a random
	// relative rotation and deletes the other cross-group pairs.
	private static void BridgeGroupsWithOutlierEdges(List<HashSet<uint>> groups, int numOutlierEdges, Database database)
	{
		Dictionary<uint, int> imageToGroup = ImageToGroup(groups);
		int numKept = 0;
		foreach ((ulong pairId, TwoViewGeometry twoViewGeometry) in database.ReadTwoViewGeometries())
		{
			(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
			if (imageToGroup[imageId1] == imageToGroup[imageId2])
			{
				continue; // Keep intra-group edges untouched.
			}

			if (numKept < numOutlierEdges && twoViewGeometry.Cam2FromCam1 is Rigid3d cam2FromCam1)
			{
				// Corrupt the relative rotation so this bridge edge is an outlier.
				twoViewGeometry.Cam2FromCam1 = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), cam2FromCam1.Translation);
				database.UpdateTwoViewGeometry(imageId1, imageId2, twoViewGeometry);
				++numKept;
			}
			else
			{
				database.DeleteTwoViewGeometry(imageId1, imageId2);
				database.DeleteInlierMatches(imageId1, imageId2);
				database.DeleteMatches(imageId1, imageId2);
			}
		}
	}

	[Test]
	public async Task GlobalPipeline_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 7,
				NumPoints3D = 50,
				CameraHasPriorFocalLength = false,
			},
			gt);

		ViewGraphCalibration.CalibrateViewGraph(new ViewGraphCalibrationOptions(), database);
		ReconstructionManager reconstructionManager = RunPipeline(new GlobalPipelineOptions(), database);

		Require(reconstructionManager.Size == 1);
		Reconstruction reconstruction = reconstructionManager.Get(0);
		string? near = ReconstructionMatchers.ExplainReconstructionNear(gt, reconstruction, 1e-2, 1e-4);

		// After the pipeline runs, point3D.error must be in pixel units, i.e. equal to what
		// UpdatePoint3DErrors would recompute.
		Require(reconstruction.NumPoints3D > 0);
		double meanAfterRun = reconstruction.ComputeMeanReprojectionError();
		reconstruction.UpdatePoint3DErrors();
		double meanRecomputed = reconstruction.ComputeMeanReprojectionError();

		await Assert.That(near).IsNull();
		await Assert.That(GTestDouble.DoubleEq(meanAfterRun, meanRecomputed)).IsTrue();
	}

	[Test]
	public async Task GlobalPipeline_SfMWithRandomSeedStability()
	{
		RandomUtils.SetPRNGSeed(0);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions { NumRigs = 2, NumCamerasPerRig = 1, NumFramesPerRig = 4, NumPoints3D = 100 },
			gt,
			new SyntheticNoiseOptions { Point2DStddev = 0.5 });

		var sizes = new List<int>();
		ReconstructionManager RunMapper(int numThreads, int randomSeed)
		{
			var vgcOptions = new ViewGraphCalibrationOptions { RandomSeed = randomSeed };
			vgcOptions.SolverOptions.NumThreads = numThreads;
			ViewGraphCalibration.CalibrateViewGraph(vgcOptions, database);
			ReconstructionManager manager = RunPipeline(
				new GlobalPipelineOptions { NumThreads = numThreads, RandomSeed = randomSeed }, database);
			sizes.Add(manager.Size);
			return manager;
		}

		const int kRandomSeed = 42;

		// Single-threaded execution.
		ReconstructionManager single0 = RunMapper(numThreads: 1, randomSeed: kRandomSeed);
		ReconstructionManager single1 = RunMapper(numThreads: 1, randomSeed: kRandomSeed);
		string? eq = ReconstructionMatchers.ExplainReconstructionEq(single0.Get(0), single1.Get(0));

		// Multi-threaded execution.
		ReconstructionManager multi0 = RunMapper(numThreads: 3, randomSeed: kRandomSeed);
		ReconstructionManager multi1 = RunMapper(numThreads: 3, randomSeed: kRandomSeed);
		// Same seed should produce similar results, up to floating-point variations in
		// optimization.
		string? near = ReconstructionMatchers.ExplainReconstructionNear(
			multi0.Get(0), multi1.Get(0), 1e-9, 1e-9, maxScaleError: null, numObsTolerance: 0.01, align: false);

		await Assert.That(sizes).IsEquivalentTo([1, 1, 1, 1]);
		await Assert.That(eq).IsNull();
		await Assert.That(near).IsNull();
	}

	[Test]
	public async Task GlobalPipeline_WithExistingRelativePoses()
	{
		RandomUtils.SetPRNGSeed(0);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 7,
				NumPoints3D = 50,
				CameraHasPriorFocalLength = false,
				TwoViewGeometryHasRelativePose = true,
			},
			gt);

		ViewGraphCalibration.CalibrateViewGraph(new ViewGraphCalibrationOptions(), database);
		ReconstructionManager reconstructionManager = RunPipeline(new GlobalPipelineOptions(), database);

		Require(reconstructionManager.Size == 1);
		string? near = ReconstructionMatchers.ExplainReconstructionNear(gt, reconstructionManager.Get(0), 1e-2, 1e-4);
		await Assert.That(near).IsNull();
	}

	// To test relative pose re-estimation from view graph calibration.
	[Test]
	public async Task GlobalPipeline_WithNoisyExistingRelativePoses()
	{
		RandomUtils.SetPRNGSeed(0);
		var gt = new Reconstruction();
		using InMemoryDatabase database = Synthesize(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 7,
				NumPoints3D = 50,
				CameraHasPriorFocalLength = false,
				TwoViewGeometryHasRelativePose = true,
			},
			gt);

		// Replace relative poses with completely random values.
		foreach ((ulong pairId, TwoViewGeometry twoViewGeometry) in database.ReadTwoViewGeometries())
		{
			if (twoViewGeometry.Cam2FromCam1 is null)
			{
				continue;
			}

			Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
			Vector3d translation = RandomEigen.RandomEigenVector3d().Normalized();
			twoViewGeometry.Cam2FromCam1 = new Rigid3d(rotation, translation);

			(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
			database.UpdateTwoViewGeometry(imageId1, imageId2, twoViewGeometry);
		}

		ViewGraphCalibration.CalibrateViewGraph(new ViewGraphCalibrationOptions(), database);
		ReconstructionManager reconstructionManager = RunPipeline(new GlobalPipelineOptions(), database);

		Require(reconstructionManager.Size == 1);
		// Expect slightly worse accuracy due to noisy input poses.
		string? near = ReconstructionMatchers.ExplainReconstructionNear(gt, reconstructionManager.Get(0), 1e-2, 1e-4);
		await Assert.That(near).IsNull();
	}
}
