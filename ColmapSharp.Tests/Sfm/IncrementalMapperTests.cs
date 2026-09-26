// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalMapperTests: colmap/sfm/incremental_mapper_test.cc ported 1:1, one method per
// gtest TEST / TEST_F named Suite_Name, testing ColmapSharp/Sfm/IncrementalMapper*.cs and
// IncrementalMapperImpl.cs (COLMAP 4.2.0 has no incremental_mapper_impl_test.cc).
//
// Tier C (outcome): registration counts, registration statistics and a ReconstructionNear
// comparison with the synthetic ground truth, with COLMAP's bounds.
//
// Translation notes: the gtest fixtures (IncrementalMapperTest and
// IncrementalMapperLargeDatasetTest) are the Fixture class, constructed per test with the
// fixture's number of frames per rig; TearDown is Fixture.TearDown. The in-memory SQLite
// database is InMemoryDatabase. PrngTestIsolation seeds the PRNG with 0 before every test,
// as COLMAP's gtest_main does; the PRNG is per thread, so each test does all of its mapping
// before its first await, collecting values to assert afterwards. ASSERT_* preconditions
// throw through Require.

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sfm;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Sfm;

public class IncrementalMapperTests
{
	private static SyntheticDatasetOptions DefaultSyntheticOptions() => new()
	{
		NumRigs = 2,
		NumCamerasPerRig = 1,
		NumFramesPerRig = 5,
		NumPoints3D = 100,
	};

	// ASSERT_* inside the C++ tests and helpers: fail the test when a precondition does not
	// hold.
	private static void Require(bool condition, string message = "Test precondition failed")
	{
		if (!condition)
		{
			throw new InvalidOperationException(message);
		}
	}

	private sealed class Fixture
	{
		public readonly Reconstruction GtReconstruction = new();
		public readonly DatabaseCache Cache;
		public readonly IncrementalMapper Mapper;
		public Reconstruction Reconstruction;
		public readonly IncrementalMapper.Options Options = new();
		public readonly IncrementalTriangulator.Options TriOptions = new();
		public uint ImageId1 = InvalidImageId;
		public uint ImageId2 = InvalidImageId;
		public Rigid3d Cam2FromCam1 = Rigid3d.Identity;

		// SetUp; numFramesPerRig is 12 for IncrementalMapperLargeDatasetTest.
		// priorPosition is for the C#-only pose-prior test at the bottom.
		public Fixture(int numFramesPerRig = 5, bool priorPosition = false)
		{
			SyntheticDatasetOptions syntheticOptions = DefaultSyntheticOptions();
			syntheticOptions.NumFramesPerRig = numFramesPerRig;
			syntheticOptions.PriorPosition = priorPosition;
			using var database = new InMemoryDatabase();
			Synthetic.SynthesizeDataset(syntheticOptions, GtReconstruction, database);
			Cache = DatabaseCache.Create(database, new DatabaseCache.Options());
			Mapper = new IncrementalMapper(Cache);
			Reconstruction = new Reconstruction();
			Mapper.BeginReconstruction(Reconstruction);
			Options.InitMinNumInliers = 10;
			Options.AbsPoseMinNumInliers = 10;
			Options.AbsPoseMinInlierRatio = 0.1;
		}

		public void TearDown()
		{
			if (Mapper.Reconstruction is not null)
			{
				Mapper.EndReconstruction(discard: false);
			}
		}

		public void FindAndRegisterInitialPair()
		{
			Require(Mapper.FindInitialImagePair(Options, ref ImageId1, ref ImageId2, ref Cam2FromCam1));
			Mapper.RegisterInitialImagePair(Options, ImageId1, ImageId2, Cam2FromCam1);
		}

		public void TriangulateInitialPair()
		{
			Mapper.TriangulateImage(TriOptions, ImageId1);
			Mapper.TriangulateImage(TriOptions, ImageId2);
		}

		public void RegisterAllRemainingImages()
		{
			while (true)
			{
				bool anyImageRegistered = false;
				List<uint> nextImageIds = Mapper.FindNextImages(Options);
				foreach (uint imageId in nextImageIds)
				{
					if (Mapper.RegisterNextImage(Options, imageId))
					{
						Mapper.TriangulateImage(TriOptions, imageId);
						anyImageRegistered = true;
					}
				}

				if (!anyImageRegistered)
				{
					break;
				}
			}
		}

		public void BeginWithSynthesizedReconstruction()
		{
			if (Mapper.Reconstruction is not null)
			{
				Mapper.EndReconstruction(discard: false);
			}

			Reconstruction = GtReconstruction.Clone();
			Mapper.BeginReconstruction(Reconstruction);
		}

		public bool IsFrameRegistered(uint frameId) => Reconstruction.RegFrameIds.Contains(frameId);

		public long CountFramePoints3D(uint frameId)
		{
			long numPoints3D = 0;
			foreach (DataId dataId in Reconstruction.Frame(frameId).ImageIds())
			{
				numPoints3D += Reconstruction.Image((uint)dataId.Id).NumPoints3D;
			}

			return numPoints3D;
		}

		public int CountRegisteredFramesWithZeroPoints3D() =>
			Reconstruction.RegFrameIds.Count(frameId => CountFramePoints3D(frameId) == 0);

		public uint FindRegisteredFrameWithPoints3D()
		{
			foreach (uint frameId in Reconstruction.RegFrameIds)
			{
				if (CountFramePoints3D(frameId) > 0)
				{
					return frameId;
				}
			}

			return InvalidFrameId;
		}

		public void DeleteAllObservationsInFrame(uint frameId)
		{
			var observationsToDelete = new List<(uint ImageId, uint Point2DIdx)>();
			foreach (DataId dataId in Reconstruction.Frame(frameId).ImageIds())
			{
				Image image = Reconstruction.Image((uint)dataId.Id);
				for (uint point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
				{
					if (image.Points2D[(int)point2DIdx].HasPoint3D)
					{
						observationsToDelete.Add(((uint)dataId.Id, point2DIdx));
					}
				}
			}

			foreach ((uint imageId, uint point2DIdx) in observationsToDelete)
			{
				if (Reconstruction.Image(imageId).Points2D[(int)point2DIdx].HasPoint3D)
				{
					Mapper.ObservationManager.DeleteObservation(imageId, point2DIdx);
				}
			}
		}
	}

	// Standalone test: needs to check state before BeginReconstruction.
	[Test]
	public async Task IncrementalMapper_GettersAfterBeginReconstruction()
	{
		var gtReconstruction = new Reconstruction();
		using var database = new InMemoryDatabase();
		Synthetic.SynthesizeDataset(DefaultSyntheticOptions(), gtReconstruction, database);

		DatabaseCache cache = DatabaseCache.Create(database, new DatabaseCache.Options());
		// 2 rigs x 1 camera x 5 frames = 10 images
		await Assert.That(cache.Images.Count).IsEqualTo(10);

		var mapper = new IncrementalMapper(cache);

		// Before BeginReconstruction, Reconstruction() returns nullptr
		await Assert.That(mapper.Reconstruction).IsNull();

		var reconstruction = new Reconstruction();
		mapper.BeginReconstruction(reconstruction);

		await Assert.That(ReferenceEquals(mapper.Reconstruction, reconstruction)).IsTrue();
		await Assert.That(() => mapper.Triangulator).ThrowsNothing();
		await Assert.That(() => mapper.ObservationManager).ThrowsNothing();
		await Assert.That(mapper.FilteredFrames.Count).IsEqualTo(0);
		await Assert.That(mapper.ExistingFrameIds.Count).IsEqualTo(0);
		await Assert.That(mapper.NumRegFramesPerRig.Count).IsEqualTo(0);
		await Assert.That(mapper.NumRegImagesPerCamera.Count).IsEqualTo(0);
		await Assert.That(mapper.NumTotalRegImages).IsEqualTo(0);
		await Assert.That(mapper.NumSharedRegImages).IsEqualTo(0);

		mapper.EndReconstruction(discard: false);
	}

	[Test]
	public async Task IncrementalMapperTest_EndReconstructionDiscard()
	{
		var f = new Fixture();
		f.FindAndRegisterInitialPair();

		// Initial pair registers 2 frames, each with 1 image
		int numTotalRegImages = f.Mapper.NumTotalRegImages;
		int numRegFrames = f.Reconstruction.NumRegFrames;

		// Discard the reconstruction — stats should be rolled back
		f.Mapper.EndReconstruction(discard: true);
		int numTotalRegImagesAfter = f.Mapper.NumTotalRegImages;
		int numSharedRegImagesAfter = f.Mapper.NumSharedRegImages;
		f.TearDown();

		await Assert.That(numTotalRegImages).IsEqualTo(2);
		await Assert.That(numRegFrames).IsEqualTo(2);
		await Assert.That(numTotalRegImagesAfter).IsEqualTo(0);
		await Assert.That(numSharedRegImagesAfter).IsEqualTo(0);
	}

	[Test]
	public async Task IncrementalMapperTest_EstimateInitialTwoViewGeometry()
	{
		var f = new Fixture();

		// Use FindInitialImagePair to select a pair with enough correspondences, then verify
		// EstimateInitialTwoViewGeometry re-estimates valid geometry.
		uint imageId1 = InvalidImageId;
		uint imageId2 = InvalidImageId;
		Rigid3d cam2FromCam1Found = Rigid3d.Identity;
		Require(f.Mapper.FindInitialImagePair(f.Options, ref imageId1, ref imageId2, ref cam2FromCam1Found));

		Rigid3d cam2FromCam1 = Rigid3d.Identity;
		Require(f.Mapper.EstimateInitialTwoViewGeometry(f.Options, imageId1, imageId2, ref cam2FromCam1));
		f.TearDown();

		await Assert.That(imageId1).IsNotEqualTo(imageId2);

		// The estimated pose should have non-trivial translation
		await Assert.That(cam2FromCam1.Translation.Norm).IsGreaterThan(0.0);
	}

	[Test]
	public async Task IncrementalMapperTest_ModifiedPoints3D()
	{
		var f = new Fixture();

		// Initially no modified points
		f.Mapper.ClearModifiedPoints3D();
		int initialModified = f.Mapper.GetModifiedPoints3D().Count;

		f.FindAndRegisterInitialPair();
		int numTotalRegImages = f.Mapper.NumTotalRegImages;
		int numRegFrames = f.Reconstruction.NumRegFrames;

		f.TriangulateInitialPair();
		int numPoints3D = f.Reconstruction.NumPoints3D;

		// After triangulation, modified points should be non-empty
		var modified = f.Mapper.GetModifiedPoints3D().ToList();
		// All modified point IDs should exist in the reconstruction
		bool allExist = modified.All(f.Reconstruction.ExistsPoint3D);

		// After clearing, modified points should be empty again
		f.Mapper.ClearModifiedPoints3D();
		int clearedModified = f.Mapper.GetModifiedPoints3D().Count;
		f.TearDown();

		await Assert.That(initialModified).IsEqualTo(0);
		await Assert.That(numTotalRegImages).IsEqualTo(2);
		await Assert.That(numRegFrames).IsEqualTo(2);
		await Assert.That(numPoints3D).IsGreaterThan(0);
		await Assert.That(modified.Count).IsGreaterThan(0);
		await Assert.That(allExist).IsTrue();
		await Assert.That(clearedModified).IsEqualTo(0);
	}

	[Test]
	public async Task IncrementalMapperTest_FullPipeline()
	{
		var f = new Fixture();

		// Step 1: Find and register initial image pair
		f.FindAndRegisterInitialPair();
		int numRegFramesInit = f.Reconstruction.NumRegFrames;

		// Step 2: Triangulate initial observations
		f.TriangulateInitialPair();
		int numPoints3DInit = f.Reconstruction.NumPoints3D;

		// Step 3: Find and register next images
		f.RegisterAllRemainingImages();
		int numRegFramesAll = f.Reconstruction.NumRegFrames;

		// Step 4: Global bundle adjustment
		bool globalBaOk = f.Mapper.AdjustGlobalBundle(f.Options, new BundleAdjustmentOptions());

		// Step 5: Filtering
		f.Mapper.FilterPoints(f.Options);
		f.Mapper.FilterFrames(f.Options);

		// Step 6: Track completion and merging
		f.Mapper.CompleteAndMergeTracks(f.TriOptions);

		// Verify the reconstruction is reasonable
		string? nearExplanation = ReconstructionMatchers.ExplainReconstructionNear(
			f.GtReconstruction, f.Reconstruction, maxRotationErrorDeg: 1e-1, maxProjCenterError: 1e-1);

		int numRegFramesFinal = f.Reconstruction.NumRegFrames;
		var numPerRig = f.Mapper.NumRegFramesPerRig.ToDictionary();
		var numPerCamera = f.Mapper.NumRegImagesPerCamera.ToDictionary();
		int numTotalRegImages = f.Mapper.NumTotalRegImages;
		int numSharedRegImages = f.Mapper.NumSharedRegImages;
		int numFilteredFrames = f.Mapper.FilteredFrames.Count;

		// Sanity check tracks: with dense visibility, at least one track should be observed
		// by all registered images.
		int maxTrackLength = f.Reconstruction.Points3D.Values.Max(point3D => point3D.Track.Length);
		int numRegImages = f.Reconstruction.NumRegImages;
		f.TearDown();

		await Assert.That(f.ImageId1).IsNotEqualTo(InvalidImageId);
		await Assert.That(f.ImageId2).IsNotEqualTo(InvalidImageId);
		await Assert.That(numRegFramesInit).IsGreaterThanOrEqualTo(2);
		await Assert.That(numPoints3DInit).IsGreaterThan(0);
		await Assert.That(numRegFramesAll).IsEqualTo(10);
		await Assert.That(globalBaOk).IsTrue();
		await Assert.That(nearExplanation).IsNull();

		// Verify registration stats match the reconstruction. The synthetic dataset has 2
		// rigs with 1 camera each and 5 frames per rig, giving 10 total frames/images. All
		// should be registered.
		await Assert.That(numRegFramesFinal).IsEqualTo(10);

		await Assert.That(numPerRig.Count).IsEqualTo(2);
		foreach (int count in numPerRig.Values)
		{
			await Assert.That(count).IsEqualTo(5);
		}

		await Assert.That(numPerCamera.Count).IsEqualTo(2);
		foreach (int count in numPerCamera.Values)
		{
			await Assert.That(count).IsEqualTo(5);
		}

		await Assert.That(numTotalRegImages).IsEqualTo(10);
		await Assert.That(numSharedRegImages).IsEqualTo(0);
		await Assert.That(numFilteredFrames).IsEqualTo(0);
		await Assert.That(maxTrackLength).IsGreaterThanOrEqualTo(numRegImages);
	}

	[Test]
	public async Task IncrementalMapperTest_FindLocalBundle()
	{
		var f = new Fixture();
		f.FindAndRegisterInitialPair();
		f.TriangulateInitialPair();
		f.RegisterAllRemainingImages();

		Require(f.Reconstruction.NumRegFrames >= 3);
		List<uint> localBundle = f.Mapper.FindLocalBundle(f.Options, f.ImageId1);
		int numRegImages = f.Reconstruction.NumRegImages;
		var regImageIdSet = new HashSet<uint>(f.Reconstruction.RegImageIds());
		f.TearDown();

		await Assert.That(localBundle.Count).IsGreaterThan(0);
		// Local bundle should not exceed the total number of registered images
		await Assert.That(localBundle.Count).IsLessThanOrEqualTo(numRegImages);
		// All images in the local bundle should be registered
		foreach (uint imageId in localBundle)
		{
			await Assert.That(regImageIdSet.Contains(imageId)).IsTrue();
		}
	}

	[Test]
	public async Task IncrementalMapperTest_ResetInitializationStats()
	{
		var f = new Fixture();

		// Find an initial pair (updates init stats internally)
		Require(f.Mapper.FindInitialImagePair(f.Options, ref f.ImageId1, ref f.ImageId2, ref f.Cam2FromCam1));

		// Reset stats and find again — should succeed with valid IDs
		f.Mapper.ResetInitializationStats();
		uint imageId3 = InvalidImageId;
		uint imageId4 = InvalidImageId;
		Rigid3d cam2FromCam12 = Rigid3d.Identity;
		Require(f.Mapper.FindInitialImagePair(f.Options, ref imageId3, ref imageId4, ref cam2FromCam12));
		f.TearDown();

		await Assert.That(f.ImageId1).IsNotEqualTo(InvalidImageId);
		await Assert.That(f.ImageId2).IsNotEqualTo(InvalidImageId);
		await Assert.That(imageId3).IsNotEqualTo(InvalidImageId);
		await Assert.That(imageId4).IsNotEqualTo(InvalidImageId);
	}

	// Frame filtering is disabled before the 20-frame mapper threshold.
	[Test]
	public async Task IncrementalMapperTest_FilterFramesNoOpBelowMinFrames()
	{
		var f = new Fixture();
		f.BeginWithSynthesizedReconstruction();

		Require(f.Reconstruction.NumRegFrames < 20);

		uint targetFrameId = f.FindRegisteredFrameWithPoints3D();
		Require(targetFrameId != InvalidFrameId);
		Require(f.IsFrameRegistered(targetFrameId));
		Require(f.CountFramePoints3D(targetFrameId) > 0);

		f.DeleteAllObservationsInFrame(targetFrameId);
		Require(f.CountFramePoints3D(targetFrameId) == 0);

		int numRegFramesBefore = f.Reconstruction.NumRegFrames;
		int numFilteredFrames = f.Mapper.FilterFrames(f.Options);
		int numRegFramesAfter = f.Reconstruction.NumRegFrames;
		bool stillRegistered = f.IsFrameRegistered(targetFrameId);
		bool inFiltered = f.Mapper.FilteredFrames.Contains(targetFrameId);
		f.TearDown();

		await Assert.That(numFilteredFrames).IsEqualTo(0);
		await Assert.That(numRegFramesAfter).IsEqualTo(numRegFramesBefore);
		await Assert.That(stillRegistered).IsTrue();
		await Assert.That(inFiltered).IsFalse();
	}

	// At or above threshold, filtering removes the chosen zero-observation frame.
	[Test]
	public async Task IncrementalMapperLargeDatasetTest_FilterFramesRemovesZeroObservationFrameAfterThreshold()
	{
		var f = new Fixture(numFramesPerRig: 12);
		f.BeginWithSynthesizedReconstruction();

		Require(f.Reconstruction.NumRegFrames >= 20);

		uint targetFrameId = f.FindRegisteredFrameWithPoints3D();
		Require(targetFrameId != InvalidFrameId);
		Require(f.IsFrameRegistered(targetFrameId));
		Require(f.CountFramePoints3D(targetFrameId) > 0);

		f.DeleteAllObservationsInFrame(targetFrameId);
		Require(f.CountFramePoints3D(targetFrameId) == 0);
		Require(f.IsFrameRegistered(targetFrameId));
		Require(f.CountRegisteredFramesWithZeroPoints3D() == 1);

		int numRegFramesBefore = f.Reconstruction.NumRegFrames;
		int numFilteredFrames = f.Mapper.FilterFrames(f.Options);
		int numRegFramesAfter = f.Reconstruction.NumRegFrames;
		bool stillRegistered = f.IsFrameRegistered(targetFrameId);
		bool inFiltered = f.Mapper.FilteredFrames.Contains(targetFrameId);
		f.TearDown();

		await Assert.That(numFilteredFrames).IsEqualTo(1);
		await Assert.That(numRegFramesAfter).IsEqualTo(numRegFramesBefore - 1);
		await Assert.That(stillRegistered).IsFalse();
		await Assert.That(inFiltered).IsTrue();
	}

	// Strict reprojection filtering removes the intentionally corrupted point.
	[Test]
	public async Task IncrementalMapperTest_FilterPointsRemovesCorruptedPoint()
	{
		var f = new Fixture();
		f.BeginWithSynthesizedReconstruction();

		Require(f.Reconstruction.NumPoints3D > 0);

		HashSet<ulong> point3DIds = f.Reconstruction.Point3DIds();
		Require(point3DIds.Count > 0);
		ulong corruptedPoint3DId = point3DIds.Max();
		Require(f.Reconstruction.Point3D(corruptedPoint3DId).Track.Length >= 2);
		Point3D corrupted = f.Reconstruction.Point3D(corruptedPoint3DId);
		corrupted.Xyz += new Vector3d(100, 100, 100);

		IncrementalMapper.Options strictOptions = f.Options.Clone();
		strictOptions.FilterMaxReprojError = 0.1;
		strictOptions.FilterMinTriAngle = 0.0;
		int numFilteredObservations = f.Mapper.FilterPoints(strictOptions);
		bool stillExists = f.Reconstruction.ExistsPoint3D(corruptedPoint3DId);
		f.TearDown();

		await Assert.That(numFilteredObservations).IsGreaterThan(0);
		await Assert.That(stillExists).IsFalse();
	}

	[Test]
	public async Task IncrementalMapperLargeDatasetTest_ObservationBookkeepingAfterInitAndFilter()
	{
		var f = new Fixture(numFramesPerRig: 12);
		f.BeginWithSynthesizedReconstruction();

		int numRegFramesBefore = f.Reconstruction.NumRegFrames;
		int numRegImagesBefore = f.Mapper.NumTotalRegImages;

		// All images are registered, so num_visible_correspondences must exactly equal
		// num_correspondences (every correspondence partner is registered).
		ObservationManager obsManager = f.Mapper.ObservationManager;
		var before = new List<(uint ImageId, long NumObs, long NumPoints3DInImage, long Visible, long NumPoints3D, long VisCorrs, long Corrs)>();
		foreach ((uint imageId, Image image) in f.Reconstruction.Images)
		{
			before.Add((
				imageId,
				obsManager.NumObservations(imageId),
				image.NumPoints3D,
				obsManager.NumVisiblePoints3D(imageId),
				f.Reconstruction.NumPoints3D,
				obsManager.NumVisibleCorrespondences(imageId),
				obsManager.NumCorrespondences(imageId)));
		}

		uint targetFrameId = f.FindRegisteredFrameWithPoints3D();
		Require(targetFrameId != InvalidFrameId);
		Frame targetFrame = f.Reconstruction.Frame(targetFrameId);
		int numImagesInFrame = targetFrame.ImageIds().Count();

		f.DeleteAllObservationsInFrame(targetFrameId);
		Require(f.CountFramePoints3D(targetFrameId) == 0);

		f.Mapper.FilterFrames(f.Options);

		// Verify reconstruction state.
		int numRegFramesAfter = f.Reconstruction.NumRegFrames;

		// Verify reg_stats_ consistency after filtering.
		int numTotalRegImagesAfter = f.Mapper.NumTotalRegImages;
		bool targetFiltered = f.Mapper.FilteredFrames.Contains(targetFrameId);

		var filteredImageIds = new HashSet<uint>(targetFrame.ImageIds().Select(dataId => (uint)dataId.Id));

		// After filtering, verify exact num_visible_correspondences for every image.
		// DeRegisterFrame decrements the correspondence partners' counters, so each remaining
		// image loses exactly the matches it had to the deregistered images. The deregistered
		// images' own counters are unchanged (DeRegisterFrame only touches partners). An
		// underflowed uint32_t would wrap to a very large value, failing these exact checks.
		CorrespondenceGraph corrGraph = f.Cache.CorrespondenceGraph;
		var after = new List<(uint ImageId, uint Visible, uint Expected, uint VisiblePoints, uint NumObs)>();
		foreach (uint imageId in f.Reconstruction.Images.Keys)
		{
			uint expectedVisibleCorrs = obsManager.NumCorrespondences(imageId);
			if (!filteredImageIds.Contains(imageId))
			{
				// Registered image: lost matches to each deregistered image.
				foreach (uint filteredImageId in filteredImageIds)
				{
					expectedVisibleCorrs -= corrGraph.NumMatchesBetweenImages(imageId, filteredImageId);
				}
			}

			after.Add((
				imageId,
				obsManager.NumVisibleCorrespondences(imageId),
				expectedVisibleCorrs,
				obsManager.NumVisiblePoints3D(imageId),
				obsManager.NumObservations(imageId)));
		}

		f.TearDown();

		foreach (var row in before)
		{
			await Assert.That(row.NumObs).IsEqualTo(row.NumPoints3DInImage);
			await Assert.That(row.Visible).IsEqualTo(row.NumPoints3D);
			await Assert.That(row.VisCorrs).IsEqualTo(row.Corrs).Because($"num_visible_correspondences wrong for image {row.ImageId}");
		}

		await Assert.That(numRegFramesAfter).IsEqualTo(numRegFramesBefore - 1);
		await Assert.That(numTotalRegImagesAfter).IsEqualTo(numRegImagesBefore - numImagesInFrame);
		await Assert.That(targetFiltered).IsTrue();

		foreach (var row in after)
		{
			await Assert.That(row.Visible).IsEqualTo(row.Expected).Because($"num_visible_correspondences wrong for image {row.ImageId}");

			// Visibility counts must remain internally consistent.
			await Assert.That(row.VisiblePoints).IsLessThanOrEqualTo(row.NumObs);
		}
	}

	// Re-initializing from a discarded reconstruction must produce correct observation
	// bookkeeping, verifying symmetry of register/deregister paths.
	[Test]
	public async Task IncrementalMapperLargeDatasetTest_ObservationBookkeepingAfterDiscardAndReinit()
	{
		var f = new Fixture(numFramesPerRig: 12);
		f.BeginWithSynthesizedReconstruction();

		int numRegFrames = f.Reconstruction.NumRegFrames;
		Require(numRegFrames >= 20);

		// Discard and re-initialize with the same ground truth reconstruction.
		f.Mapper.EndReconstruction(discard: true);
		f.BeginWithSynthesizedReconstruction();

		int numRegFramesAfter = f.Reconstruction.NumRegFrames;
		int numTotalRegImages = f.Mapper.NumTotalRegImages;
		int numRegImages = f.Reconstruction.NumRegImages;

		// After a full discard + reinit cycle, the observation bookkeeping must be correct
		// again — no state leaks from the previous cycle.
		ObservationManager obsManager = f.Mapper.ObservationManager;
		var rows = f.Reconstruction.Images.Keys
			.Select(imageId => (imageId, obsManager.NumVisibleCorrespondences(imageId), obsManager.NumCorrespondences(imageId)))
			.ToList();
		f.TearDown();

		await Assert.That(numRegFramesAfter).IsEqualTo(numRegFrames);
		await Assert.That(numTotalRegImages).IsEqualTo(numRegImages);
		foreach ((uint imageId, uint visible, uint total) in rows)
		{
			await Assert.That(visible).IsEqualTo(total).Because($"num_visible_correspondences wrong for image {imageId}");
		}
	}

	// Filtering must update reg_stats_ (per-rig and per-camera counts) to match the actual
	// reconstruction state.
	[Test]
	public async Task IncrementalMapperLargeDatasetTest_FilterFramesRegStatsConsistency()
	{
		var f = new Fixture(numFramesPerRig: 12);
		f.BeginWithSynthesizedReconstruction();

		Require(f.Reconstruction.NumRegFrames >= 20);

		uint targetFrameId = f.FindRegisteredFrameWithPoints3D();
		Require(targetFrameId != InvalidFrameId);
		Frame targetFrame = f.Reconstruction.Frame(targetFrameId);
		uint targetRigId = targetFrame.RigId;

		// Record per-rig and per-camera counts before filtering.
		int rigCountBefore = f.Mapper.NumRegFramesPerRig[targetRigId];
		var cameraCountsBefore = new Dictionary<uint, int>();
		foreach (DataId dataId in targetFrame.ImageIds())
		{
			uint cameraId = f.Reconstruction.Image((uint)dataId.Id).CameraId;
			cameraCountsBefore[cameraId] = f.Mapper.NumRegImagesPerCamera[cameraId];
		}

		f.DeleteAllObservationsInFrame(targetFrameId);
		f.Mapper.FilterFrames(f.Options);

		int rigCountAfter = f.Mapper.NumRegFramesPerRig[targetRigId];
		var cameraCountsAfter = new List<(int After, int Before)>();
		foreach (DataId dataId in targetFrame.ImageIds())
		{
			uint cameraId = f.Reconstruction.Image((uint)dataId.Id).CameraId;
			cameraCountsAfter.Add((f.Mapper.NumRegImagesPerCamera[cameraId], cameraCountsBefore[cameraId]));
		}

		f.TearDown();

		// Per-rig count should decrease by exactly 1.
		await Assert.That(rigCountAfter).IsEqualTo(rigCountBefore - 1);

		// Per-camera counts should decrease by exactly the number of images in the filtered
		// frame that used each camera.
		foreach ((int after, int before) in cameraCountsAfter)
		{
			await Assert.That(after).IsEqualTo(before - 1);
		}
	}

	// Reproduces the crash when FilterFrames aggressively deregisters frames, leaving fewer
	// than 2 images for a subsequent AdjustGlobalBundle call.
	[Test]
	public async Task IncrementalMapperLargeDatasetTest_AdjustGlobalBundleReturnsFalseWithInsufficientFrames()
	{
		var f = new Fixture(numFramesPerRig: 12);
		f.BeginWithSynthesizedReconstruction();

		Require(f.Reconstruction.NumRegFrames >= 20);

		foreach (uint frameId in f.Reconstruction.RegFrameIds)
		{
			f.DeleteAllObservationsInFrame(frameId);
		}

		f.Mapper.FilterFrames(f.Options);
		Require(f.Reconstruction.NumRegImages < 2);

		bool result = f.Mapper.AdjustGlobalBundle(f.Options, new BundleAdjustmentOptions());
		f.TearDown();

		await Assert.That(result).IsFalse();
	}

	[Test]
	public async Task IncrementalMapperTest_RegStatsResetBetweenReconstructions()
	{
		var f = new Fixture();
		f.BeginWithSynthesizedReconstruction();

		int numRegImagesFirst = f.Mapper.NumTotalRegImages;
		var rigCountsFirst = f.Mapper.NumRegFramesPerRig.ToDictionary();
		var cameraCountsFirst = f.Mapper.NumRegImagesPerCamera.ToDictionary();
		Require(numRegImagesFirst > 0);
		Require(rigCountsFirst.Count > 0);
		Require(cameraCountsFirst.Count > 0);
		Require(f.Mapper.ExistingFrameIds.Count > 0);

		// End without discard (keeps cross-reconstruction state).
		f.Mapper.EndReconstruction(discard: false);

		// Begin a fresh empty reconstruction.
		f.Reconstruction = new Reconstruction();
		f.Mapper.BeginReconstruction(f.Reconstruction);

		// Per-reconstruction stats must be reset for a fresh reconstruction.
		int freshRigCounts = f.Mapper.NumRegFramesPerRig.Count;
		int freshCameraCounts = f.Mapper.NumRegImagesPerCamera.Count;
		int freshFiltered = f.Mapper.FilteredFrames.Count;
		int freshExisting = f.Mapper.ExistingFrameIds.Count;
		int freshShared = f.Mapper.NumSharedRegImages;

		// Cross-reconstruction stats must persist.
		int freshTotal = f.Mapper.NumTotalRegImages;

		f.Mapper.EndReconstruction(discard: false);

		// Re-add the same reconstruction — images are now shared across two cycles.
		f.BeginWithSynthesizedReconstruction();

		// Per-reconstruction stats must match the first cycle.
		var rigCountsSecond = f.Mapper.NumRegFramesPerRig.ToDictionary();
		var cameraCountsSecond = f.Mapper.NumRegImagesPerCamera.ToDictionary();
		int secondExisting = f.Mapper.ExistingFrameIds.Count;

		// All images were already registered in the first cycle, so they are now shared.
		// num_total_reg_images stays the same (same images, not new ones), and
		// num_shared_reg_images equals the number of re-registered images.
		int secondTotal = f.Mapper.NumTotalRegImages;
		int secondShared = f.Mapper.NumSharedRegImages;

		f.Mapper.EndReconstruction(discard: false);

		await Assert.That(freshRigCounts).IsEqualTo(0);
		await Assert.That(freshCameraCounts).IsEqualTo(0);
		await Assert.That(freshFiltered).IsEqualTo(0);
		await Assert.That(freshExisting).IsEqualTo(0);
		await Assert.That(freshShared).IsEqualTo(0);
		await Assert.That(freshTotal).IsEqualTo(numRegImagesFirst);

		await Assert.That(DictionaryEquals(rigCountsSecond, rigCountsFirst)).IsTrue();
		await Assert.That(DictionaryEquals(cameraCountsSecond, cameraCountsFirst)).IsTrue();
		await Assert.That(secondExisting).IsGreaterThan(0);

		await Assert.That(secondTotal).IsEqualTo(numRegImagesFirst);
		await Assert.That(secondShared).IsEqualTo(numRegImagesFirst);
	}

	// C#-only (no COLMAP counterpart; incremental_mapper_test.cc has no pose-prior case):
	// global bundle adjustment through the pose-prior adjuster on the noise-free synthetic
	// scene with position priors stays usable and keeps the ground-truth poses, and the
	// global refinement loop with priors runs (it skips normalization).
	[Test]
	public async Task IncrementalMapperTest_CSharpOnly_AdjustGlobalBundleWithPriorPositions()
	{
		var f = new Fixture(priorPosition: true);
		f.BeginWithSynthesizedReconstruction();
		Require(f.Cache.NumPosePriors >= 3);

		IncrementalMapper.Options priorOptions = f.Options.Clone();
		priorOptions.UsePriorPosition = true;
		priorOptions.UseRobustLossOnPriorPosition = true;
		bool usable = f.Mapper.AdjustGlobalBundle(priorOptions, new BundleAdjustmentOptions());
		string? nearExplanation = ReconstructionMatchers.ExplainReconstructionNear(
			f.GtReconstruction, f.Reconstruction, maxRotationErrorDeg: 1e-1, maxProjCenterError: 1e-1);

		f.Mapper.IterativeGlobalRefinement(
			2, 0.0005, priorOptions, new BundleAdjustmentOptions(), f.TriOptions);
		int numRegFrames = f.Reconstruction.NumRegFrames;
		f.TearDown();

		await Assert.That(usable).IsTrue();
		await Assert.That(nearExplanation).IsNull();
		await Assert.That(numRegFrames).IsEqualTo(10);
	}

	// C++ compares the maps with operator==: same keys and values, order-free.
	private static bool DictionaryEquals(Dictionary<uint, int> a, Dictionary<uint, int> b) =>
		a.Count == b.Count && a.All(kv => b.TryGetValue(kv.Key, out int value) && value == kv.Value);
}
