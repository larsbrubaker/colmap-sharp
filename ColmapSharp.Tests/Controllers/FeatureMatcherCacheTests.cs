// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureMatcherCacheTests: colmap/controllers/matcher_cache_test.cc ported 1:1, one method
// per gtest case named Suite_Name. Tests ColmapSharp/Controllers/FeatureMatcherCache.cs.
// Database::Open of a test-dir file becomes an InMemoryDatabase (SQLite files are out of
// scope). PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main
// does, and each test synthesizes its dataset before its first await. Tier A (exact).

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Controllers;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Controllers;

public class FeatureMatcherCacheTests
{
	private static InMemoryDatabase CreateTestData(int numImages, bool withPriors = false, int numCamerasPerRig = 1)
	{
		var database = new InMemoryDatabase();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = numImages / numCamerasPerRig,
			NumCamerasPerRig = numCamerasPerRig,
			NumFramesPerRig = 1,
			NumPoints3D = 20,
			NumPoints2DWithoutPoint3D = 3,
			PriorPosition = withPriors,
		};
		Synthetic.SynthesizeDataset(options, new Reconstruction(), database);
		return database;
	}

	[Test]
	public async Task FeatureMatcherCache_GetCamera()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		List<Camera> cameras = database.ReadAllCameras();
		await Assert.That(cameras).IsNotEmpty();

		foreach (Camera camera in cameras)
		{
			await Assert.That(cache.GetCamera(camera.CameraId) == camera).IsTrue();
		}
	}

	[Test]
	public async Task FeatureMatcherCache_GetFrame()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		List<Frame> frames = database.ReadAllFrames();
		await Assert.That(frames).IsNotEmpty();

		foreach (Frame frame in frames)
		{
			await Assert.That(cache.GetFrame(frame.FrameId) == frame).IsTrue();
		}
	}

	[Test]
	public async Task FeatureMatcherCache_GetImage()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		List<Image> images = database.ReadAllImages();
		await Assert.That(images).IsNotEmpty();

		foreach (Image image in images)
		{
			await Assert.That(cache.GetImage(image.ImageId) == image).IsTrue();
		}
	}

	[Test]
	public async Task FeatureMatcherCache_GetImageIds()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		List<uint> expectedIds = database.ReadAllImages().Select(image => image.ImageId).ToList();
		await Assert.That(cache.GetImageIds()).IsEquivalentTo(expectedIds);
	}

	[Test]
	public async Task FeatureMatcherCache_GetFrameIds()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		List<uint> expectedIds = database.ReadAllFrames().Select(frame => frame.FrameId).ToList();
		await Assert.That(cache.GetFrameIds()).IsEquivalentTo(expectedIds);
	}

	[Test]
	public async Task FeatureMatcherCache_FindImagePosePriorOrNullWithPriors()
	{
		using InMemoryDatabase database = CreateTestData(4, withPriors: true);
		var cache = new FeatureMatcherCache(5, database);

		List<Image> images = database.ReadAllImages();
		await Assert.That(images).IsNotEmpty();

		foreach (Image image in images)
		{
			PosePrior? prior = cache.FindImagePosePriorOrNull(image.ImageId);
			await Assert.That(prior.HasValue).IsTrue();
			await Assert.That(prior!.Value.HasPosition()).IsTrue();
		}
	}

	[Test]
	public async Task FeatureMatcherCache_FindImagePosePriorOrNullWithoutPriors()
	{
		using InMemoryDatabase database = CreateTestData(4, withPriors: false);
		var cache = new FeatureMatcherCache(5, database);

		List<Image> images = database.ReadAllImages();
		await Assert.That(images).IsNotEmpty();

		foreach (Image image in images)
		{
			await Assert.That(cache.FindImagePosePriorOrNull(image.ImageId).HasValue).IsFalse();
		}
	}

	[Test]
	public async Task FeatureMatcherCache_Features()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		List<Image> images = database.ReadAllImages();
		await Assert.That(images).IsNotEmpty();

		foreach (Image image in images)
		{
			await Assert.That(cache.ExistsKeypoints(image.ImageId)).IsTrue();
			await Assert.That(cache.ExistsDescriptors(image.ImageId)).IsTrue();
			await Assert.That(cache.GetKeypoints(image.ImageId).SequenceEqual(database.ReadKeypoints(image.ImageId))).IsTrue();
			FeatureDescriptors cachedDescriptors = cache.GetDescriptors(image.ImageId);
			FeatureDescriptors dbDescriptors = database.ReadDescriptors(image.ImageId);
			await Assert.That(cachedDescriptors.Type).IsEqualTo(dbDescriptors.Type);
			await Assert.That(cachedDescriptors.Data == dbDescriptors.Data).IsTrue();
		}
	}

	[Test]
	public async Task FeatureMatcherCache_Matches()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		var allMatches = database.ReadAllMatches();
		await Assert.That(allMatches).IsNotEmpty();

		foreach ((ulong pairId, List<FeatureMatch> matches) in allMatches)
		{
			(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
			await Assert.That(cache.ExistsMatches(imageId1, imageId2)).IsTrue();
			await Assert.That(cache.GetMatches(imageId1, imageId2).SequenceEqual(matches)).IsTrue();
		}
	}

	[Test]
	public async Task FeatureMatcherCache_TwoViewGeometry()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		var allTvg = database.ReadTwoViewGeometries();
		await Assert.That(allTvg).IsNotEmpty();

		foreach ((ulong pairId, TwoViewGeometry tvg) in allTvg)
		{
			(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
			await Assert.That(cache.ExistsTwoViewGeometry(imageId1, imageId2)).IsTrue();
			await Assert.That(cache.ExistsInlierMatches(imageId1, imageId2)).IsEqualTo(tvg.InlierMatches.Count != 0);
			await Assert.That(cache.GetTwoViewGeometry(imageId1, imageId2).InlierMatches.SequenceEqual(tvg.InlierMatches)).IsTrue();
		}
	}

	[Test]
	public async Task FeatureMatcherCache_WriteAndGetMatches()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsGreaterThanOrEqualTo(2);

		uint id1 = images[0].ImageId;
		uint id2 = images[1].ImageId;

		// Delete existing matches first.
		cache.DeleteMatches(id1, id2);
		await Assert.That(cache.ExistsMatches(id1, id2)).IsFalse();

		// Write new matches.
		var matches = new List<FeatureMatch>();
		for (uint i = 0; i < 5; ++i)
		{
			matches.Add(new FeatureMatch(i, i));
		}

		cache.WriteMatches(id1, id2, matches);
		await Assert.That(cache.ExistsMatches(id1, id2)).IsTrue();

		List<FeatureMatch> readMatches = cache.GetMatches(id1, id2);
		await Assert.That(readMatches.Count).IsEqualTo(5);
	}

	[Test]
	public async Task FeatureMatcherCache_WriteAndGetTwoViewGeometry()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsGreaterThanOrEqualTo(2);

		uint id1 = images[0].ImageId;
		uint id2 = images[1].ImageId;

		// Delete existing two-view geometry first.
		cache.DeleteTwoViewGeometry(id1, id2);
		await Assert.That(cache.ExistsTwoViewGeometry(id1, id2)).IsFalse();

		// Write new two-view geometry.
		var tvg = new TwoViewGeometry { Config = TwoViewGeometry.ConfigurationType.Calibrated };
		tvg.InlierMatches.AddRange(Enumerable.Repeat(new FeatureMatch(), 10));
		cache.WriteTwoViewGeometry(id1, id2, tvg);
		await Assert.That(cache.ExistsTwoViewGeometry(id1, id2)).IsTrue();

		TwoViewGeometry readTvg = cache.GetTwoViewGeometry(id1, id2);
		await Assert.That(readTvg.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Calibrated);
		await Assert.That(readTvg.InlierMatches.Count).IsEqualTo(10);
	}

	[Test]
	public async Task FeatureMatcherCache_UpdateTwoViewGeometry()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		var allTvg = database.ReadTwoViewGeometries();
		await Assert.That(allTvg).IsNotEmpty();

		(uint id1, uint id2) = PairIdToImagePair(allTvg[0].PairId);

		var updatedTvg = new TwoViewGeometry { Config = TwoViewGeometry.ConfigurationType.Uncalibrated };
		updatedTvg.InlierMatches.AddRange(Enumerable.Repeat(new FeatureMatch(), 7));
		cache.UpdateTwoViewGeometry(id1, id2, updatedTvg);

		TwoViewGeometry readTvg = cache.GetTwoViewGeometry(id1, id2);
		await Assert.That(readTvg.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Uncalibrated);
		await Assert.That(readTvg.InlierMatches.Count).IsEqualTo(7);
	}

	[Test]
	public async Task FeatureMatcherCache_DeleteMatches()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		var allMatches = database.ReadAllMatches();
		await Assert.That(allMatches).IsNotEmpty();

		(uint id1, uint id2) = PairIdToImagePair(allMatches[0].PairId);

		await Assert.That(cache.ExistsMatches(id1, id2)).IsTrue();
		cache.DeleteMatches(id1, id2);
		await Assert.That(cache.ExistsMatches(id1, id2)).IsFalse();
	}

	[Test]
	public async Task FeatureMatcherCache_DeleteTwoViewGeometry()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		var allTvg = database.ReadTwoViewGeometries();
		await Assert.That(allTvg).IsNotEmpty();

		(uint id1, uint id2) = PairIdToImagePair(allTvg[0].PairId);

		await Assert.That(cache.ExistsTwoViewGeometry(id1, id2)).IsTrue();
		cache.DeleteTwoViewGeometry(id1, id2);
		await Assert.That(cache.ExistsTwoViewGeometry(id1, id2)).IsFalse();
	}

	[Test]
	public async Task FeatureMatcherCache_DeleteInlierMatches()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		var allTvg = database.ReadTwoViewGeometries();
		await Assert.That(allTvg).IsNotEmpty();

		// Find a pair with inlier matches.
		uint imageId1 = 0;
		uint imageId2 = 0;
		bool found = false;
		foreach ((ulong pairId, TwoViewGeometry tvg) in allTvg)
		{
			if (tvg.InlierMatches.Count != 0)
			{
				(imageId1, imageId2) = PairIdToImagePair(pairId);
				found = true;
				break;
			}
		}

		await Assert.That(found).IsTrue();

		await Assert.That(cache.ExistsInlierMatches(imageId1, imageId2)).IsTrue();
		cache.DeleteInlierMatches(imageId1, imageId2);
		await Assert.That(cache.ExistsInlierMatches(imageId1, imageId2)).IsFalse();
		// The two-view geometry entry should still exist.
		await Assert.That(cache.ExistsTwoViewGeometry(imageId1, imageId2)).IsTrue();
	}

	[Test]
	public async Task FeatureMatcherCache_MaxNumKeypoints()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		long maxNumKeypoints = cache.MaxNumKeypoints();
		await Assert.That(maxNumKeypoints).IsGreaterThan(0);

		// Calling again should return the cached value.
		await Assert.That(cache.MaxNumKeypoints()).IsEqualTo(maxNumKeypoints);
	}

	[Test]
	public async Task FeatureMatcherCache_AccessDatabase()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		int numImages = 0;
		cache.AccessDatabase(db => numImages = db.ReadAllImages().Count);
		await Assert.That(numImages).IsEqualTo(4);
	}

	[Test]
	public async Task FeatureMatcherCache_GetFeatureDescriptorIndexCache()
	{
		using InMemoryDatabase database = CreateTestData(4);
		var cache = new FeatureMatcherCache(5, database);

		var indexCache = cache.GetFeatureDescriptorIndexCache();

		List<Image> images = database.ReadAllImages();
		await Assert.That(images).IsNotEmpty();

		// Access descriptor index for the first image to trigger build.
		FeatureDescriptorIndex index = indexCache.Get(images[0].ImageId);
		await Assert.That(index).IsNotNull();
	}
}
