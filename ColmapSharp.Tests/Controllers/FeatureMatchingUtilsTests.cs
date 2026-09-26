// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureMatchingUtilsTests: colmap/controllers/feature_matching_utils_test.cc ported 1:1, one
// method per gtest case named Suite_Name. Tests ColmapSharp/Controllers/FeatureMatchingUtils.cs.
// Database::Open of a test-dir file becomes an InMemoryDatabase; `use_gpu = false` has no
// counterpart. PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main
// does. The C#-only test at the end is labeled as such.

using ColmapSharp.Controllers;
using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public class FeatureMatchingUtilsTests
{
	private sealed class TestData
	{
		public required InMemoryDatabase Database;
		public required FeatureMatcherCache Cache;
		public required List<uint> ImageIds;
	}

	private static TestData CreateTestData(int numImages)
	{
		var database = new InMemoryDatabase();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = numImages,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
			NumPoints3D = 20,
			NumPoints2DWithoutPoint3D = 3,
		};
		Synthetic.SynthesizeDataset(options, new Reconstruction(), database);

		var cache = new FeatureMatcherCache(100, database);
		return new TestData { Database = database, Cache = cache, ImageIds = cache.GetImageIds() };
	}

	private static FeatureMatchingOptions DefaultMatchingOptions() => new() { NumThreads = 1 };

	private static List<(uint, uint)> AllPairs(List<uint> imageIds)
	{
		var pairs = new List<(uint, uint)>();
		for (int i = 0; i < imageIds.Count; ++i)
		{
			for (int j = i + 1; j < imageIds.Count; ++j)
			{
				pairs.Add((imageIds[i], imageIds[j]));
			}
		}

		return pairs;
	}

	// Match pairs without geometric verification, then clear TVGs. Leaves matches in the
	// database ready for a GeometricVerifierController.
	private static async Task MatchPairsWithoutVerification(TestData data, List<(uint, uint)> pairs)
	{
		data.Database.ClearMatches();
		data.Database.ClearTwoViewGeometries();
		FeatureMatchingOptions matchingOptions = DefaultMatchingOptions();
		matchingOptions.SkipGeometricVerification = true;
		var geometryOptions = new TwoViewGeometryOptions();
		var matcher = new FeatureMatcherController(matchingOptions, geometryOptions, data.Cache);
		await Assert.That(matcher.Setup()).IsTrue();
		matcher.Match(pairs);
		data.Database.ClearTwoViewGeometries();
	}

	[Test]
	public async Task FeatureMatcherController_MatchEmptyPairs()
	{
		TestData data = CreateTestData(3);
		data.Database.ClearMatches();
		data.Database.ClearTwoViewGeometries();

		var controller = new FeatureMatcherController(DefaultMatchingOptions(), new TwoViewGeometryOptions(), data.Cache);
		await Assert.That(controller.Setup()).IsTrue();

		// Matching empty pairs should return without error
		controller.Match([]);

		await Assert.That(data.Database.ReadAllMatches().Count).IsEqualTo(0);
	}

	[Test]
	public async Task FeatureMatcherController_MatchSkipsSelfMatches()
	{
		TestData data = CreateTestData(3);
		data.Database.ClearMatches();
		data.Database.ClearTwoViewGeometries();

		var controller = new FeatureMatcherController(DefaultMatchingOptions(), new TwoViewGeometryOptions(), data.Cache);
		await Assert.That(controller.Setup()).IsTrue();

		// Self-match pairs should be skipped
		var pairs = new List<(uint, uint)>();
		foreach (uint id in data.ImageIds)
		{
			pairs.Add((id, id));
		}

		controller.Match(pairs);

		await Assert.That(data.Database.ReadAllMatches().Count).IsEqualTo(0);
	}

	[Test]
	public async Task FeatureMatcherController_MatchSkipsDuplicatePairs()
	{
		TestData data = CreateTestData(3);
		data.Database.ClearMatches();
		data.Database.ClearTwoViewGeometries();

		var controller = new FeatureMatcherController(DefaultMatchingOptions(), new TwoViewGeometryOptions(), data.Cache);
		await Assert.That(controller.Setup()).IsTrue();

		await Assert.That(data.ImageIds.Count).IsGreaterThanOrEqualTo(2);
		uint id1 = data.ImageIds[0];
		uint id2 = data.ImageIds[1];

		// Submit same pair multiple times - should only process once
		controller.Match([(id1, id2), (id1, id2), (id1, id2)]);

		await Assert.That(data.Database.ReadAllMatches().Count).IsEqualTo(1);
	}

	[Test]
	public async Task FeatureMatcherController_MatchSkipsExistingResults()
	{
		TestData data = CreateTestData(3);

		var controller = new FeatureMatcherController(DefaultMatchingOptions(), new TwoViewGeometryOptions(), data.Cache);
		await Assert.That(controller.Setup()).IsTrue();

		await Assert.That(data.ImageIds.Count).IsGreaterThanOrEqualTo(2);
		uint id1 = data.ImageIds[0];
		uint id2 = data.ImageIds[1];

		// Clear and match once
		data.Database.ClearMatches();
		data.Database.ClearTwoViewGeometries();
		controller.Match([(id1, id2)]);

		int matchesBefore = data.Database.ReadAllMatches().Count;
		int tvgBefore = data.Database.ReadTwoViewGeometries().Count;
		await Assert.That(matchesBefore).IsEqualTo(1);
		await Assert.That(tvgBefore).IsEqualTo(1);

		// Match same pair again - should skip since both matches and TVG exist
		controller.Match([(id1, id2)]);

		await Assert.That(data.Database.ReadAllMatches().Count).IsEqualTo(matchesBefore);
		await Assert.That(data.Database.ReadTwoViewGeometries().Count).IsEqualTo(tvgBefore);

		// Match with reversed pair - should also be skipped
		controller.Match([(id2, id1)]);

		await Assert.That(data.Database.ReadAllMatches().Count).IsEqualTo(matchesBefore);
		await Assert.That(data.Database.ReadTwoViewGeometries().Count).IsEqualTo(tvgBefore);
	}

	[Test]
	public async Task FeatureMatcherController_MatchMultiplePairs()
	{
		TestData data = CreateTestData(4);
		data.Database.ClearMatches();
		data.Database.ClearTwoViewGeometries();

		var controller = new FeatureMatcherController(DefaultMatchingOptions(), new TwoViewGeometryOptions(), data.Cache);
		await Assert.That(controller.Setup()).IsTrue();

		// Match all pairs
		controller.Match(AllPairs(data.ImageIds));

		// 4 choose 2 = 6 pairs
		await Assert.That(data.Database.ReadAllMatches().Count).IsEqualTo(6);
		await Assert.That(data.Database.ReadTwoViewGeometries().Count).IsEqualTo(6);
	}

	[Test]
	public async Task FeatureMatcherController_MatchSkipGeometricVerification()
	{
		TestData data = CreateTestData(3);
		data.Database.ClearMatches();
		data.Database.ClearTwoViewGeometries();

		FeatureMatchingOptions matchingOptions = DefaultMatchingOptions();
		matchingOptions.SkipGeometricVerification = true;
		var controller = new FeatureMatcherController(matchingOptions, new TwoViewGeometryOptions(), data.Cache);
		await Assert.That(controller.Setup()).IsTrue();

		await Assert.That(data.ImageIds.Count).IsGreaterThanOrEqualTo(2);
		controller.Match([(data.ImageIds[0], data.ImageIds[1])]);

		// Matches should be written even without geometric verification
		await Assert.That(data.Database.ReadAllMatches().Count).IsEqualTo(1);

		// Verify geometric verification was skipped: TVG should have UNDEFINED config
		TwoViewGeometry tvg = data.Database.ReadTwoViewGeometry(data.ImageIds[0], data.ImageIds[1]);
		await Assert.That(tvg.Config).IsEqualTo(TwoViewGeometry.ConfigurationType.Undefined);
		await Assert.That(tvg.InlierMatches).IsEmpty();
	}

	[Test]
	public async Task GeometricVerifierController_OptionsAccessor()
	{
		TestData data = CreateTestData(3);

		var controller = new GeometricVerifierController(
			new GeometricVerifierOptions { NumThreads = 1 }, new TwoViewGeometryOptions(), data.Cache);

		await Assert.That(controller.Options.NumThreads).IsEqualTo(1);
		controller.Options.NumThreads = 2;
		await Assert.That(controller.Options.NumThreads).IsEqualTo(2);
	}

	[Test]
	public async Task GeometricVerifierController_VerifyEmptyPairs()
	{
		TestData data = CreateTestData(3);
		data.Database.ClearTwoViewGeometries();

		var controller = new GeometricVerifierController(
			new GeometricVerifierOptions { NumThreads = 1 }, new TwoViewGeometryOptions(), data.Cache);
		await Assert.That(controller.Setup()).IsTrue();

		// Verifying empty pairs should return without error
		controller.Verify([]);

		await Assert.That(data.Database.ReadTwoViewGeometries().Count).IsEqualTo(0);
	}

	[Test]
	public async Task GeometricVerifierController_VerifySkipsSelfMatches()
	{
		TestData data = CreateTestData(3);
		data.Database.ClearTwoViewGeometries();

		var controller = new GeometricVerifierController(
			new GeometricVerifierOptions { NumThreads = 1 }, new TwoViewGeometryOptions(), data.Cache);
		await Assert.That(controller.Setup()).IsTrue();

		var pairs = new List<(uint, uint)>();
		foreach (uint id in data.ImageIds)
		{
			pairs.Add((id, id));
		}

		controller.Verify(pairs);

		await Assert.That(data.Database.ReadTwoViewGeometries().Count).IsEqualTo(0);
	}

	[Test]
	public async Task GeometricVerifierController_VerifySkipsDuplicatePairs()
	{
		TestData data = CreateTestData(3);
		await Assert.That(data.ImageIds.Count).IsGreaterThanOrEqualTo(2);
		await MatchPairsWithoutVerification(data, [(data.ImageIds[0], data.ImageIds[1])]);

		var controller = new GeometricVerifierController(
			new GeometricVerifierOptions { NumThreads = 1 }, new TwoViewGeometryOptions(), data.Cache);
		await Assert.That(controller.Setup()).IsTrue();

		uint id1 = data.ImageIds[0];
		uint id2 = data.ImageIds[1];

		// Submit same pair multiple times - should only process once
		controller.Verify([(id1, id2), (id1, id2), (id1, id2)]);

		await Assert.That(data.Database.ReadTwoViewGeometries().Count).IsEqualTo(1);
	}

	[Test]
	public async Task GeometricVerifierController_VerifyWithExistingMatches()
	{
		TestData data = CreateTestData(4);
		List<(uint, uint)> pairs = AllPairs(data.ImageIds);
		await MatchPairsWithoutVerification(data, pairs);

		var controller = new GeometricVerifierController(
			new GeometricVerifierOptions { NumThreads = 1 }, new TwoViewGeometryOptions(), data.Cache);
		await Assert.That(controller.Setup()).IsTrue();

		controller.Verify(pairs);

		// All 6 pairs should now have TVGs
		await Assert.That(data.Database.ReadTwoViewGeometries().Count).IsEqualTo(6);
	}

	// A dataset whose matches include outliers and whose keypoints carry noise, with the
	// two-view geometries cleared, so verification (RANSAC) has real choices to make.
	private static TestData CreateNoisyMatchedData()
	{
		// Reseeded on every call, not only at test start: a test that compares two runs
		// needs both built from the same dataset.
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 6,
			NumPoints3D = 60,
			NumPoints2DWithoutPoint3D = 10,
			InlierMatchRatio = 0.6,
		};
		Synthetic.SynthesizeDataset(options, reconstruction, database);
		Synthetic.SynthesizeNoise(new SyntheticNoiseOptions { Point2DStddev = 0.5 }, reconstruction, database);
		database.ClearTwoViewGeometries();

		var cache = new FeatureMatcherCache(100, database);
		return new TestData { Database = database, Cache = cache, ImageIds = cache.GetImageIds() };
	}

	// Verifies every pair's existing (outlier-laden) matches with the given thread count.
	private static List<(ulong PairId, TwoViewGeometry TwoViewGeometry)> VerifyNoisyMatches(int numThreads)
	{
		TestData data = CreateNoisyMatchedData();
		var controller = new FeatureMatcherController(
			new FeatureMatchingOptions { NumThreads = numThreads }, new TwoViewGeometryOptions(), data.Cache);
		controller.Setup();
		controller.Match(AllPairs(data.ImageIds));
		return data.Database.ReadTwoViewGeometries();
	}

	// C#-only: verifying outlier-laden, noisy matches with 1 and 4 threads writes identical
	// two-view geometries (results are committed in pair order and each verification starts
	// from the same PRNG state, docs/CPP_DIVERGENCES.md entry 71). Without the per-pair reseed
	// the 1-thread run continues one stream across pairs and the geometries differ.
	[Test]
	public async Task CSharpOnly_MatchIsThreadCountIndependent()
	{
		List<(ulong PairId, TwoViewGeometry TwoViewGeometry)> serial = VerifyNoisyMatches(1);
		List<(ulong PairId, TwoViewGeometry TwoViewGeometry)> parallel = VerifyNoisyMatches(4);

		await Assert.That(serial.Count).IsEqualTo(15);
		await Assert.That(parallel.Count).IsEqualTo(serial.Count);
		for (int i = 0; i < serial.Count; ++i)
		{
			await Assert.That(parallel[i].PairId).IsEqualTo(serial[i].PairId);
			await Assert.That(parallel[i].TwoViewGeometry.Config).IsEqualTo(serial[i].TwoViewGeometry.Config);
			await Assert.That(parallel[i].TwoViewGeometry.InlierMatches.SequenceEqual(serial[i].TwoViewGeometry.InlierMatches)).IsTrue();
			await Assert.That(parallel[i].TwoViewGeometry.F == serial[i].TwoViewGeometry.F).IsTrue();
			await Assert.That(parallel[i].TwoViewGeometry.E == serial[i].TwoViewGeometry.E).IsTrue();
		}
	}

	// C#-only: matching and verification leave the calling thread's PRNG as they found it
	// (Parallel.For runs iterations on the calling thread too), whatever the thread count.
	[Test]
	public async Task CSharpOnly_MatchAndVerifyLeaveCallerPrngUntouched()
	{
		RandomUtils.SetPRNGSeed(123);
		int expected = RandomUtils.RandomUniformInteger(0, int.MaxValue);

		foreach (int numThreads in new[] { 1, 4 })
		{
			TestData data = CreateNoisyMatchedData();
			var matcher = new FeatureMatcherController(
				new FeatureMatchingOptions { NumThreads = numThreads }, new TwoViewGeometryOptions(), data.Cache);
			matcher.Setup();
			RandomUtils.SetPRNGSeed(123);
			matcher.Match(AllPairs(data.ImageIds));
			await Assert.That(RandomUtils.RandomUniformInteger(0, int.MaxValue)).IsEqualTo(expected);

			data.Database.ClearTwoViewGeometries();
			var verifier = new GeometricVerifierController(
				new GeometricVerifierOptions { NumThreads = numThreads }, new TwoViewGeometryOptions(), data.Cache);
			verifier.Setup();
			RandomUtils.SetPRNGSeed(123);
			verifier.Verify(AllPairs(data.ImageIds));
			await Assert.That(RandomUtils.RandomUniformInteger(0, int.MaxValue)).IsEqualTo(expected);
		}
	}

	// C#-only: a cancelled batch leaves the database untouched: existing matches of a pair
	// without a two-view geometry, and the two-view geometry of a pair without matches, are
	// only replaced when the batch's results are written.
	[Test]
	public async Task CSharpOnly_CancelledMatchKeepsExistingResults()
	{
		TestData data = CreateTestData(3);
		List<(uint, uint)> pairs = AllPairs(data.ImageIds);
		await MatchPairsWithoutVerification(data, pairs);

		// Pair 0: matches without a two-view geometry (from MatchPairsWithoutVerification).
		// Pair 1: a two-view geometry without matches.
		(uint id1, uint id2) = pairs[1];
		data.Database.DeleteMatches(id1, id2);
		data.Database.WriteTwoViewGeometry(id1, id2, new TwoViewGeometry
		{
			Config = TwoViewGeometry.ConfigurationType.Calibrated,
			InlierMatches = [new FeatureMatch(1, 2)],
		});
		int numMatchesBefore = data.Database.ReadAllMatches().Count;
		int numTwoViewGeometriesBefore = data.Database.ReadTwoViewGeometries().Count;

		var cache = new FeatureMatcherCache(100, data.Database);
		var controller = new FeatureMatcherController(DefaultMatchingOptions(), new TwoViewGeometryOptions(), cache);
		await Assert.That(controller.Setup()).IsTrue();

		using var cancellation = new CancellationTokenSource();
		cancellation.Cancel();
		await Assert.That(() => controller.Match(pairs, cancellation.Token)).Throws<OperationCanceledException>();

		await Assert.That(data.Database.ReadAllMatches().Count).IsEqualTo(numMatchesBefore);
		await Assert.That(data.Database.ReadTwoViewGeometries().Count).IsEqualTo(numTwoViewGeometriesBefore);
		await Assert.That(data.Database.ExistsMatches(pairs[0].Item1, pairs[0].Item2)).IsTrue();
		await Assert.That(data.Database.ReadTwoViewGeometry(id1, id2).InlierMatches.Count).IsEqualTo(1);
	}
}
