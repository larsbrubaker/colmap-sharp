// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PairingTests: colmap/controllers/pairing_test.cc ported 1:1, one method per gtest case
// named Suite_Name; the SpatialPairGenerator cases are PairingTests.Spatial.cs. Tests
// ColmapSharp/Controllers/PairGenerator.cs, SequentialPairGenerator.cs,
// SpatialPairGenerator.cs and PairingOptions.cs. Database::Open(kInMemorySqliteDatabasePath)
// becomes an InMemoryDatabase. COLMAP's gtest_main seeds the PRNG with 0 before every test,
// so each test seeds before synthesizing its dataset. Tier A (exact): testing::ElementsAre
// compares the pair lists in order (ElementsAre below), UnorderedElementsAre sorted.
//
// Skipped (vocabulary-tree retrieval, colmap/retrieval, is out of scope - PORTING_PLAN.md):
// VocabTreePairGenerator.Nominal, VocabTreePairGenerator.DoesNotDeadlockOnFailedQuery,
// SequentialPairGenerator.LoopDetectionMinIndexDistance.

using ColmapSharp.Controllers;
using ColmapSharp.Feature;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class PairingTests
{
	private static InMemoryDatabase CreateSyntheticDatabase(int numImages)
	{
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = numImages,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, new Reconstruction(), database);
		return database;
	}

	private static InMemoryDatabase CreateSyntheticRigDatabase()
	{
		RandomUtils.SetPRNGSeed(0);
		var database = new InMemoryDatabase();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 2,
			NumFramesPerRig = 3,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, new Reconstruction(), database);
		return database;
	}

	// testing::ElementsAre: same pairs in the same order.
	private static async Task ElementsAre(List<(uint ImageId1, uint ImageId2)> actual, params (uint, uint)[] expected)
	{
		await Assert.That(string.Join(" ", actual)).IsEqualTo(string.Join(" ", expected));
	}

	// testing::UnorderedElementsAre: same pairs in any order.
	private static async Task UnorderedElementsAre(List<(uint ImageId1, uint ImageId2)> actual, params (uint, uint)[] expected)
	{
		await Assert.That(string.Join(" ", actual.Order())).IsEqualTo(string.Join(" ", expected.Order()));
	}

	private static (uint, uint) Pair(List<Image> images, int index1, int index2) =>
		(images[index1].ImageId, images[index2].ImageId);

	[Test]
	public async Task ExhaustivePairGenerator_Nominal()
	{
		const int NumImages = 34;
		using InMemoryDatabase database = CreateSyntheticDatabase(NumImages);
		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsEqualTo(NumImages);

		var options = new ExhaustivePairingOptions { BlockSize = 10 };
		var generator = new ExhaustivePairGenerator(options, database);
		int numExpectedBlocks =
			(int)(Math.Ceiling((double)NumImages / options.BlockSize) * Math.Ceiling((double)NumImages / options.BlockSize));
		var pairs = new SortedSet<(uint, uint)>();
		for (int i = 0; i < numExpectedBlocks; ++i)
		{
			foreach ((uint, uint) pair in generator.Next())
			{
				pairs.Add(pair);
			}
		}

		await Assert.That(pairs.Count).IsEqualTo(NumImages * (NumImages - 1) / 2);
		await Assert.That(generator.Next()).IsEmpty();
		await Assert.That(generator.HasFinished()).IsTrue();
	}

	[Test]
	public async Task SequentialPairGenerator_Linear()
	{
		const int NumImages = 5;
		using InMemoryDatabase database = CreateSyntheticDatabase(NumImages);
		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsEqualTo(NumImages);

		var options = new SequentialPairingOptions { Overlap = 3, QuadraticOverlap = false };
		var generator = new SequentialPairGenerator(options, database);
		await ElementsAre(generator.Next(), Pair(images, 0, 1), Pair(images, 0, 2), Pair(images, 0, 3));
		await ElementsAre(generator.Next(), Pair(images, 1, 2), Pair(images, 1, 3), Pair(images, 1, 4));
		await ElementsAre(generator.Next(), Pair(images, 2, 3), Pair(images, 2, 4));
		await ElementsAre(generator.Next(), Pair(images, 3, 4));
		await Assert.That(generator.Next()).IsEmpty();
		await Assert.That(generator.HasFinished()).IsTrue();
	}

	[Test]
	public async Task SequentialPairGenerator_LinearRig()
	{
		using InMemoryDatabase database = CreateSyntheticRigDatabase();
		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsEqualTo(2 * 3);

		var options = new SequentialPairingOptions { Overlap = 1, QuadraticOverlap = false };
		var generator = new SequentialPairGenerator(options, database);
		await ElementsAre(generator.Next(), Pair(images, 0, 1), Pair(images, 0, 2), Pair(images, 0, 3));
		await ElementsAre(generator.Next(), Pair(images, 2, 3), Pair(images, 2, 4), Pair(images, 2, 5));
		await ElementsAre(generator.Next(), Pair(images, 4, 5));
		await ElementsAre(generator.Next(), Pair(images, 1, 0), Pair(images, 1, 3), Pair(images, 1, 2));
		await ElementsAre(generator.Next(), Pair(images, 3, 2), Pair(images, 3, 5), Pair(images, 3, 4));
		await ElementsAre(generator.Next(), Pair(images, 5, 4));
		await Assert.That(generator.Next()).IsEmpty();
		await Assert.That(generator.HasFinished()).IsTrue();
	}

	[Test]
	public async Task SequentialPairGenerator_QuadraticRig()
	{
		using InMemoryDatabase database = CreateSyntheticRigDatabase();
		List<Image> images = database.ReadAllImages();

		var options = new SequentialPairingOptions { Overlap = 3, QuadraticOverlap = true };
		var generator = new SequentialPairGenerator(options, database);
		await ElementsAre(
			generator.Next(),
			Pair(images, 0, 1),
			Pair(images, 0, 2),
			Pair(images, 0, 3),
			Pair(images, 0, 4),
			Pair(images, 0, 5));
		await ElementsAre(generator.Next(), Pair(images, 2, 3), Pair(images, 2, 4), Pair(images, 2, 5));
		await ElementsAre(generator.Next(), Pair(images, 4, 5));
		await ElementsAre(
			generator.Next(),
			Pair(images, 1, 0),
			Pair(images, 1, 3),
			Pair(images, 1, 2),
			Pair(images, 1, 5),
			Pair(images, 1, 4));
		await ElementsAre(generator.Next(), Pair(images, 3, 2), Pair(images, 3, 5), Pair(images, 3, 4));
		await ElementsAre(generator.Next(), Pair(images, 5, 4));
		await Assert.That(generator.Next()).IsEmpty();
		await Assert.That(generator.HasFinished()).IsTrue();
	}

	[Test]
	public async Task SequentialPairGenerator_Quadratic()
	{
		const int NumImages = 5;
		using InMemoryDatabase database = CreateSyntheticDatabase(NumImages);
		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsEqualTo(NumImages);

		var options = new SequentialPairingOptions { Overlap = 3, QuadraticOverlap = true };
		var generator = new SequentialPairGenerator(options, database);
		await ElementsAre(generator.Next(), Pair(images, 0, 1), Pair(images, 0, 2), Pair(images, 0, 4));
		await ElementsAre(generator.Next(), Pair(images, 1, 2), Pair(images, 1, 3));
		await ElementsAre(generator.Next(), Pair(images, 2, 3), Pair(images, 2, 4));
		await ElementsAre(generator.Next(), Pair(images, 3, 4));
		await Assert.That(generator.Next()).IsEmpty();
		await Assert.That(generator.HasFinished()).IsTrue();
	}

	/// <summary>
	/// C#-only (no COLMAP counterpart): loop detection needs vocabulary-tree retrieval, which
	/// is not ported, so asking for it fails up front with a message the user can act on.
	/// </summary>
	[Test]
	public async Task SequentialPairGenerator_LoopDetectionIsNotSupported()
	{
		using InMemoryDatabase database = CreateSyntheticDatabase(3);
		var options = new SequentialPairingOptions { LoopDetection = true };
		await Assert.That(() => new SequentialPairGenerator(options, database)).Throws<NotSupportedException>();
	}

	[Test]
	public async Task TransitivePairGenerator_Nominal()
	{
		const int NumImages = 5;
		using InMemoryDatabase database = CreateSyntheticDatabase(NumImages);
		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsEqualTo(NumImages);

		var twoViewGeometry = new TwoViewGeometry();
		twoViewGeometry.InlierMatches.AddRange(Enumerable.Repeat(new FeatureMatch(), 10));

		database.ClearTwoViewGeometries();
		database.WriteTwoViewGeometry(images[0].ImageId, images[1].ImageId, twoViewGeometry);
		database.WriteTwoViewGeometry(images[0].ImageId, images[2].ImageId, twoViewGeometry);
		database.WriteTwoViewGeometry(images[1].ImageId, images[3].ImageId, twoViewGeometry);

		var options = new TransitivePairingOptions();
		var generator = new TransitivePairGenerator(options, database);
		List<(uint ImageId1, uint ImageId2)> pairs1 = generator.Next();
		await UnorderedElementsAre(pairs1, Pair(images, 1, 2), Pair(images, 0, 3));
		foreach ((uint imageId1, uint imageId2) in pairs1)
		{
			database.WriteTwoViewGeometry(imageId1, imageId2, twoViewGeometry);
		}

		await ElementsAre(generator.Next(), Pair(images, 2, 3));
		await Assert.That(generator.Next()).IsEmpty();
		await Assert.That(generator.HasFinished()).IsTrue();
	}

	[Test]
	public async Task ImportedPairGenerator_Nominal()
	{
		const int NumImages = 10;
		using InMemoryDatabase database = CreateSyntheticDatabase(NumImages);
		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsEqualTo(NumImages);

		string testDir = Path.Combine(Path.GetTempPath(), "colmapsharp-pairing-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(testDir);
		try
		{
			var options = new ImportedPairingOptions { MatchListPath = Path.Combine(testDir, "pairs.txt") };

			{
				File.WriteAllText(options.MatchListPath, "");

				var generator = new ImportedPairGenerator(options, database);
				await Assert.That(generator.Next()).IsEmpty();
				await Assert.That(generator.HasFinished()).IsTrue();
			}

			{
				File.WriteAllText(
					options.MatchListPath,
					$"{images[2].Name} {images[4].Name}\n{images[1].Name} {images[3].Name}\n{images[2].Name} {images[9].Name}\n");

				var generator = new ImportedPairGenerator(options, database);
				await ElementsAre(generator.Next(), Pair(images, 2, 4), Pair(images, 1, 3), Pair(images, 2, 9));
				await Assert.That(generator.Next()).IsEmpty();
				await Assert.That(generator.HasFinished()).IsTrue();
			}
		}
		finally
		{
			Directory.Delete(testDir, recursive: true);
		}
	}

	[Test]
	public async Task ExistingMatchedPairGenerator_Nominal()
	{
		const int NumImages = 5;
		using InMemoryDatabase database = CreateSyntheticDatabase(NumImages);
		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsEqualTo(NumImages);

		database.ClearMatches();
		database.WriteMatches(images[0].ImageId, images[1].ImageId, Matches(1));
		database.WriteMatches(images[0].ImageId, images[2].ImageId, Matches(2));
		database.WriteMatches(images[1].ImageId, images[3].ImageId, Matches(3));
		database.WriteMatches(images[2].ImageId, images[3].ImageId, Matches(0));

		var options = new ExistingMatchedPairingOptions { BatchSize = 2 };
		var generator = new ExistingMatchedPairGenerator(options, database);
		await UnorderedElementsAre(generator.Next(), Pair(images, 0, 1), Pair(images, 0, 2));
		await UnorderedElementsAre(generator.Next(), Pair(images, 1, 3));
		await Assert.That(generator.Next()).IsEmpty();
		await Assert.That(generator.HasFinished()).IsTrue();
	}

	// FeatureMatches(n): n default-constructed matches.
	private static List<FeatureMatch> Matches(int count) => Enumerable.Repeat(new FeatureMatch(), count).ToList();
}
