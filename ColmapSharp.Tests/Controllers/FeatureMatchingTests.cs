// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureMatchingTests: colmap/controllers/feature_matching_test.cc ported 1:1, one method
// per gtest case named Suite_Name. Tests ColmapSharp/Controllers/FeatureMatching.cs (and
// FeatureMatching.FeaturePairs.cs). Database files become InMemoryDatabases; the match-list
// files are written to a temporary directory; `use_gpu = false` has no counterpart.
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does.
// Skipped: CreateVocabTreeFeatureMatcher.Nominal (vocabulary-tree retrieval is out of scope).
// Tier C (outcome).

using ColmapSharp.Controllers;
using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Controllers;

public class FeatureMatchingTests
{
	private static InMemoryDatabase CreateTestDatabase(int numImages)
	{
		var database = new InMemoryDatabase();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = numImages,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
			NumPoints3D = 20,
			NumPoints2DWithoutPoint3D = 3,
			PriorPosition = true,
		};
		Synthetic.SynthesizeDataset(options, new Reconstruction(), database);
		return database;
	}

	private static FeatureMatchingOptions MatchingOptions() => new() { NumThreads = 1 };

	// CreateTestDir: a fresh temporary directory, removed after the test.
	private static async Task WithTestDir(Func<string, Task> test)
	{
		string testDir = Path.Combine(Path.GetTempPath(), "colmapsharp-matching-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(testDir);
		try
		{
			await test(testDir);
		}
		finally
		{
			Directory.Delete(testDir, recursive: true);
		}
	}

	[Test]
	public async Task CreateExhaustiveFeatureMatcher_Nominal()
	{
		using InMemoryDatabase database = CreateTestDatabase(4);
		database.ClearMatches();
		database.ClearTwoViewGeometries();

		FeatureMatching.MatchExhaustive(
			database, new ExhaustivePairingOptions(), MatchingOptions(), new TwoViewGeometryOptions());

		await Assert.That(database.ReadAllMatches().Count).IsEqualTo(6);
		await Assert.That(database.ReadTwoViewGeometries().Count).IsEqualTo(6);
	}

	[Test]
	public async Task CreateSequentialFeatureMatcher_Nominal()
	{
		using InMemoryDatabase database = CreateTestDatabase(5);
		database.ClearMatches();
		database.ClearTwoViewGeometries();

		var pairingOptions = new SequentialPairingOptions { Overlap = 2, QuadraticOverlap = false };

		FeatureMatching.MatchSequential(database, pairingOptions, MatchingOptions(), new TwoViewGeometryOptions());

		// With 5 images and overlap=2:
		// (0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4)
		await Assert.That(database.ReadAllMatches().Count).IsEqualTo(7);
		await Assert.That(database.ReadTwoViewGeometries().Count).IsEqualTo(7);
	}

	[Test]
	public async Task CreateSpatialFeatureMatcher_Nominal()
	{
		using InMemoryDatabase database = CreateTestDatabase(4);
		database.ClearMatches();
		database.ClearTwoViewGeometries();

		var pairingOptions = new SpatialPairingOptions { MaxNumNeighbors = 2, MaxDistance = 1e6 };

		FeatureMatching.MatchSpatial(database, pairingOptions, MatchingOptions(), new TwoViewGeometryOptions());

		await Assert.That(database.ReadAllMatches().Count).IsGreaterThan(0);
		await Assert.That(database.ReadTwoViewGeometries().Count).IsGreaterThan(0);
	}

	[Test]
	public async Task CreateTransitiveFeatureMatcher_Nominal()
	{
		using InMemoryDatabase database = CreateTestDatabase(4);
		database.ClearMatches();
		database.ClearTwoViewGeometries();

		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsGreaterThanOrEqualTo(3);

		// Create initial matches: 1-2 and 2-3
		var twoViewGeometry = new TwoViewGeometry
		{
			Config = TwoViewGeometry.ConfigurationType.Calibrated,
			InlierMatches = [.. Enumerable.Repeat(new FeatureMatch(), 10)],
		};

		database.WriteTwoViewGeometry(images[0].ImageId, images[1].ImageId, twoViewGeometry);
		database.WriteTwoViewGeometry(images[1].ImageId, images[2].ImageId, twoViewGeometry);

		var pairingOptions = new TransitivePairingOptions { BatchSize = 100, NumIterations = 1 };

		FeatureMatching.MatchTransitive(database, pairingOptions, MatchingOptions(), new TwoViewGeometryOptions());

		// Should create transitive match 1-3
		int finalMatches = database.ReadTwoViewGeometries().Count;
		await Assert.That(finalMatches).IsGreaterThanOrEqualTo(2); // At least the original 2 matches
	}

	[Test]
	public async Task CreateImagePairsFeatureMatcher_Nominal()
	{
		await WithTestDir(async testDir =>
		{
			string matchListPath = Path.Combine(testDir, "match_list.txt");

			using InMemoryDatabase database = CreateTestDatabase(4);
			database.ClearMatches();
			database.ClearTwoViewGeometries();

			List<Image> images = database.ReadAllImages();
			await Assert.That(images.Count).IsGreaterThanOrEqualTo(3);

			// Create match list file with specific image pairs
			File.WriteAllText(
				matchListPath,
				$"{images[0].Name} {images[1].Name}\n{images[1].Name} {images[2].Name}\n{images[2].Name} {images[3].Name}\n");

			var pairingOptions = new ImportedPairingOptions { MatchListPath = matchListPath };

			FeatureMatching.MatchImagePairs(database, pairingOptions, MatchingOptions(), new TwoViewGeometryOptions());

			await Assert.That(database.ReadAllMatches().Count).IsEqualTo(3);
			await Assert.That(database.ReadTwoViewGeometries().Count).IsEqualTo(3);
		});
	}

	[Test]
	public async Task CreateFeaturePairsFeatureMatcher_Nominal()
	{
		await WithTestDir(async testDir =>
		{
			string matchListPath = Path.Combine(testDir, "feature_match_list.txt");

			using InMemoryDatabase database = CreateTestDatabase(3);
			database.ClearMatches();
			database.ClearTwoViewGeometries();

			List<Image> images = database.ReadAllImages();
			await Assert.That(images.Count).IsGreaterThanOrEqualTo(2);

			// Create feature match list file with many matches for better verification
			var file = new System.Text.StringBuilder();
			file.Append($"{images[0].Name} {images[1].Name}\n");
			for (int i = 0; i < 15; ++i)
			{
				file.Append($"{i} {i}\n");
			}

			file.Append('\n'); // Empty line separates pairs
			file.Append($"{images[1].Name} {images[2].Name}\n");
			for (int i = 0; i < 15; ++i)
			{
				file.Append($"{i} {i}\n");
			}

			file.Append('\n');
			File.WriteAllText(matchListPath, file.ToString());

			var pairingOptions = new FeaturePairsMatchingOptions { MatchListPath = matchListPath, VerifyMatches = true };

			var geometryOptions = new TwoViewGeometryOptions { MinNumInliers = 5 }; // Lower threshold for testing

			FeatureMatching.MatchFeaturePairs(database, pairingOptions, MatchingOptions(), geometryOptions);

			// Should have imported and verified the matches
			await Assert.That(database.ReadTwoViewGeometries().Count).IsGreaterThanOrEqualTo(2);
		});
	}

	// C#-only: `>> idx1 >> idx2` on a line with one index reads idx1 and fails at the end of
	// the line without storing anything, so point2D_idx2 keeps kInvalidPoint2DIdx (libc++);
	// an index that does not parse stores 0.
	[Test]
	public async Task CSharpOnly_FeaturePairsSingleIndexLeavesSecondInvalid()
	{
		await WithTestDir(async testDir =>
		{
			string matchListPath = Path.Combine(testDir, "feature_match_list.txt");

			using InMemoryDatabase database = CreateTestDatabase(2);
			database.ClearMatches();
			database.ClearTwoViewGeometries();
			List<Image> images = database.ReadAllImages();

			File.WriteAllText(matchListPath, $"{images[0].Name} {images[1].Name}\n3\n4 x\n5 6\n\n");

			var pairingOptions = new FeaturePairsMatchingOptions { MatchListPath = matchListPath, VerifyMatches = false };
			FeatureMatching.MatchFeaturePairs(database, pairingOptions, MatchingOptions(), new TwoViewGeometryOptions());

			List<FeatureMatch> inlierMatches = database.ReadTwoViewGeometry(images[0].ImageId, images[1].ImageId).InlierMatches;
			await Assert.That(inlierMatches.Count).IsEqualTo(3);
			await Assert.That(inlierMatches[0] == new FeatureMatch(3, InvalidPoint2DIdx)).IsTrue();
			await Assert.That(inlierMatches[1] == new FeatureMatch(4, 0)).IsTrue();
			await Assert.That(inlierMatches[2] == new FeatureMatch(5, 6)).IsTrue();
		});
	}

	// C#-only: where the pair generator knows its block count, progress reports a real Total
	// (one step per block); transitive matching, whose blocks depend on earlier results,
	// reports a Total of 0.
	[Test]
	public async Task CSharpOnly_MatchingReportsBlockTotals()
	{
		using InMemoryDatabase database = CreateTestDatabase(5);
		database.ClearMatches();
		database.ClearTwoViewGeometries();

		var exhaustiveSteps = new List<ControllerProgress>();
		FeatureMatching.MatchExhaustive(
			database, new ExhaustivePairingOptions { BlockSize = 2 }, MatchingOptions(), new TwoViewGeometryOptions(),
			new SynchronousProgress(exhaustiveSteps));
		// 3 blocks of images (2, 2, 1) -> 3 x 3 blocks of pairs.
		await Assert.That(exhaustiveSteps.Select(step => (step.Done, step.Total)).SequenceEqual(
			Enumerable.Range(1, 9).Select(done => (done, 9)))).IsTrue();

		database.ClearMatches();
		database.ClearTwoViewGeometries();
		var sequentialSteps = new List<ControllerProgress>();
		FeatureMatching.MatchSequential(
			database, new SequentialPairingOptions { Overlap = 2 }, MatchingOptions(), new TwoViewGeometryOptions(),
			new SynchronousProgress(sequentialSteps));
		await Assert.That(sequentialSteps.Count).IsEqualTo(5);
		await Assert.That(sequentialSteps.All(step => step.Total == 5)).IsTrue();

		var transitiveSteps = new List<ControllerProgress>();
		FeatureMatching.MatchTransitive(
			database, new TransitivePairingOptions { NumIterations = 1 }, MatchingOptions(), new TwoViewGeometryOptions(),
			new SynchronousProgress(transitiveSteps));
		await Assert.That(transitiveSteps.All(step => step.Total == 0)).IsTrue();
	}

	// Progress<T> posts to the thread pool; this one records in call order.
	private sealed class SynchronousProgress(List<ControllerProgress> steps) : IProgress<ControllerProgress>
	{
		public void Report(ControllerProgress value) => steps.Add(value);
	}

	[Test]
	public async Task CreateGeometricVerifier_Nominal()
	{
		using InMemoryDatabase database = CreateTestDatabase(4);
		database.ClearTwoViewGeometries();

		FeatureMatching.VerifyGeometry(
			database,
			new GeometricVerifierOptions { NumThreads = 1 },
			new ExistingMatchedPairingOptions(),
			new TwoViewGeometryOptions());

		await Assert.That(database.ReadAllMatches().Count).IsGreaterThanOrEqualTo(3);
		await Assert.That(database.ReadTwoViewGeometries().Count).IsGreaterThanOrEqualTo(3);
	}

	private static async Task ExpectRigVerificationResults(
		Database database, int numExpectedMatches, int numExpectedCalibrated, int numExpectedCalibratedRig)
	{
		// Verify that two-view geometries were created.
		int numCalibrated = 0;
		int numCalibratedRig = 0;
		int numOthers = 0;
		foreach ((_, TwoViewGeometry twoViewGeometry) in database.ReadTwoViewGeometries())
		{
			await Assert.That(twoViewGeometry.InlierMatches.Count).IsEqualTo(numExpectedMatches);
			switch (twoViewGeometry.Config)
			{
				case TwoViewGeometry.ConfigurationType.Calibrated:
					++numCalibrated;
					break;
				case TwoViewGeometry.ConfigurationType.CalibratedRig:
					++numCalibratedRig;
					break;
				default:
					++numOthers;
					break;
			}
		}

		// Two calibrated pairs between images in the same frames.
		await Assert.That(numCalibrated).IsEqualTo(numExpectedCalibrated);
		// Four calibrated pairs between images in different frames.
		await Assert.That(numCalibratedRig).IsEqualTo(numExpectedCalibratedRig);
		await Assert.That(numOthers).IsEqualTo(0);
	}

	[Test]
	public async Task CreateGeometricVerifier_RigVerificationWithNonTrivialFrames()
	{
		using var database = new InMemoryDatabase();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 3,
			NumFramesPerRig = 2,
			NumPoints3D = 25,
			MatchConfig = SyntheticMatchConfig.Exhaustive,
			CameraHasPriorFocalLength = true,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, new Reconstruction(), database);

		FeatureMatching.VerifyGeometry(
			database,
			new GeometricVerifierOptions { NumThreads = -1, RigVerification = true },
			new ExistingMatchedPairingOptions(),
			new TwoViewGeometryOptions { MinNumInliers = 5 });

		// All pairs should be overwritten with calibrated rig pairs.
		await ExpectRigVerificationResults(
			database,
			syntheticDatasetOptions.NumPoints3D,
			numExpectedCalibrated: 0,
			numExpectedCalibratedRig: 15);
	}

	[Test]
	public async Task CreateGeometricVerifier_RigVerificationWithTrivialFrames()
	{
		using var database = new InMemoryDatabase();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 2,
			NumPoints3D = 25,
			MatchConfig = SyntheticMatchConfig.Exhaustive,
			CameraHasPriorFocalLength = true,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, new Reconstruction(), database);

		FeatureMatching.VerifyGeometry(
			database,
			new GeometricVerifierOptions { NumThreads = 1, RigVerification = true },
			new ExistingMatchedPairingOptions(),
			new TwoViewGeometryOptions { MinNumInliers = 5 });

		// Trivial frames should be skipped and unmodified.
		await ExpectRigVerificationResults(
			database,
			syntheticDatasetOptions.NumPoints3D,
			numExpectedCalibrated: 1,
			numExpectedCalibratedRig: 0);
	}

	[Test]
	public async Task CreateGeometricVerifier_Guided()
	{
		using var database = new InMemoryDatabase();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 5,
			NumPoints3D = 50,
			InlierMatchRatio = 0.6,
			TwoViewGeometryHasRelativePose = true,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, new Reconstruction(), database);

		// Clear all inlier matches. cam2_from_cam1 is already gt from the synthesized database.
		List<(ulong PairId, TwoViewGeometry TwoViewGeometry)> gtTwoViewGeometries = database.ReadTwoViewGeometries();
		foreach ((ulong pairId, _) in gtTwoViewGeometries)
		{
			(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
			database.DeleteInlierMatches(imageId1, imageId2);
		}

		FeatureMatching.VerifyGeometry(
			database,
			new GeometricVerifierOptions { NumThreads = 1, UseExistingRelativePose = true },
			new ExistingMatchedPairingOptions(),
			new TwoViewGeometryOptions());

		// Check validity after guided geometric verification.
		List<(ulong PairId, TwoViewGeometry TwoViewGeometry)> twoViewGeometries = database.ReadTwoViewGeometries();
		await Assert.That(twoViewGeometries.Count).IsGreaterThanOrEqualTo(gtTwoViewGeometries.Count);
		for (int i = 0; i < twoViewGeometries.Count; ++i)
		{
			await Assert.That(twoViewGeometries[i].PairId).IsEqualTo(gtTwoViewGeometries[i].PairId);
			await Assert.That(twoViewGeometries[i].TwoViewGeometry.Cam2FromCam1 == gtTwoViewGeometries[i].TwoViewGeometry.Cam2FromCam1).IsTrue();
			await Assert.That(gtTwoViewGeometries[i].TwoViewGeometry.E!.Value.IsApprox(
				twoViewGeometries[i].TwoViewGeometry.E!.Value)).IsTrue();
			// Should at least have all the original inliers. Some generated outliers can be
			// accidentally inliers as well.
			await Assert.That(twoViewGeometries[i].TwoViewGeometry.InlierMatches.Count)
				.IsGreaterThanOrEqualTo(gtTwoViewGeometries[i].TwoViewGeometry.InlierMatches.Count);
		}
	}
}
