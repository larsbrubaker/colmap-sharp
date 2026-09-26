// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PairingTests.Spatial: the SpatialPairGenerator cases of colmap/controllers/pairing_test.cc,
// ported 1:1 (see PairingTests.cs for the conventions). Tests
// ColmapSharp/Controllers/SpatialPairGenerator.cs. Tier A (exact).

using ColmapSharp.Controllers;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class PairingTests
{
	private static void WritePositionPrior(Database database, Image image, Vector3d position,
		PosePriorCoordinateSystem? coordinateSystem = null)
	{
		var posePrior = new PosePrior { CorrDataId = image.DataId, Position = position };
		if (coordinateSystem is PosePriorCoordinateSystem system)
		{
			posePrior.CoordinateSystem = system;
		}

		database.WritePosePrior(posePrior);
	}

	[Test]
	public async Task SpatialPairGenerator_Nominal()
	{
		const int NumImages = 3;
		using InMemoryDatabase database = CreateSyntheticDatabase(NumImages);
		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsEqualTo(NumImages);

		WritePositionPrior(database, images[0], new Vector3d(1, 2, 3));
		WritePositionPrior(database, images[1], new Vector3d(2, 3, 4));
		WritePositionPrior(database, images[2], new Vector3d(2, 4, 12));

		var options = new SpatialPairingOptions { MaxNumNeighbors = 1, MaxDistance = 1000, IgnoreZ = false };

		{
			var generator = new SpatialPairGenerator(options, database);
			await ElementsAre(generator.Next(), Pair(images, 0, 1));
			await ElementsAre(generator.Next(), Pair(images, 1, 0));
			await ElementsAre(generator.Next(), Pair(images, 2, 1));
			await Assert.That(generator.Next()).IsEmpty();
			await Assert.That(generator.HasFinished()).IsTrue();
		}

		{
			options.IgnoreZ = true;
			var generator = new SpatialPairGenerator(options, database);
			await ElementsAre(generator.Next(), Pair(images, 0, 1));
			await ElementsAre(generator.Next(), Pair(images, 1, 2));
			await ElementsAre(generator.Next(), Pair(images, 2, 1));
			await Assert.That(generator.Next()).IsEmpty();
			await Assert.That(generator.HasFinished()).IsTrue();
		}

		{
			options.IgnoreZ = false;
			options.MaxDistance = 5;
			var generator = new SpatialPairGenerator(options, database);
			await ElementsAre(generator.Next(), Pair(images, 0, 1));
			await ElementsAre(generator.Next(), Pair(images, 1, 0));
			await Assert.That(generator.Next()).IsEmpty();
			await Assert.That(generator.Next()).IsEmpty();
			await Assert.That(generator.HasFinished()).IsTrue();
		}

		{
			options.MaxNumNeighbors = 2;
			options.MaxDistance = 1000;
			var generator = new SpatialPairGenerator(options, database);
			await ElementsAre(generator.Next(), Pair(images, 0, 1), Pair(images, 0, 2));
			await ElementsAre(generator.Next(), Pair(images, 1, 0), Pair(images, 1, 2));
			await ElementsAre(generator.Next(), Pair(images, 2, 1), Pair(images, 2, 0));
			await Assert.That(generator.Next()).IsEmpty();
			await Assert.That(generator.HasFinished()).IsTrue();
		}
	}

	[Test]
	public async Task SpatialPairGenerator_LargeCoordinates()
	{
		const int NumImages = 3;
		using InMemoryDatabase database = CreateSyntheticDatabase(NumImages);
		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsEqualTo(NumImages);

		// Eigen::Vector3d(x, y, z) + Eigen::Vector3d::Constant(1e16).
		WritePositionPrior(database, images[0], new Vector3d(1 + 1e16, 2 + 1e16, 3 + 1e16));
		WritePositionPrior(database, images[1], new Vector3d(2 + 1e16, 3 + 1e16, 4 + 1e16));
		WritePositionPrior(database, images[2], new Vector3d(2 + 1e16, 4 + 1e16, 12 + 1e16));

		var options = new SpatialPairingOptions { MaxNumNeighbors = 1, MaxDistance = 1000, IgnoreZ = false };

		var generator = new SpatialPairGenerator(options, database);
		await ElementsAre(generator.Next(), Pair(images, 0, 1));
		await ElementsAre(generator.Next(), Pair(images, 1, 0));
		await ElementsAre(generator.Next(), Pair(images, 2, 1));
		await Assert.That(generator.Next()).IsEmpty();
		await Assert.That(generator.HasFinished()).IsTrue();
	}

	[Test]
	public async Task SpatialPairGenerator_CentersLargeCoordinatesWithMissingPosePrior()
	{
		// Verifies that images with missing pose priors do not bias the internal offset
		// applied to position priors during spatial matching, i.e. by including rows of
		// zeros in the average position calculation.

		const int NumImages = 4;
		using InMemoryDatabase database = CreateSyntheticDatabase(NumImages);
		List<Image> images = database.ReadAllImages();
		await Assert.That(images.Count).IsEqualTo(NumImages);

		// Add pose priors for 3 of the 4 images, with large coordinate values.
		database.ClearPosePriors();

		var offset = new Vector3d(1_600_000, 5_400_000, 100);
		WritePositionPrior(database, images[0], offset + new Vector3d(-1, -2, -3), PosePriorCoordinateSystem.Cartesian);
		WritePositionPrior(database, images[1], offset + new Vector3d(0, 0, 0), PosePriorCoordinateSystem.Cartesian);
		WritePositionPrior(database, images[3], offset + new Vector3d(1, 2, 3), PosePriorCoordinateSystem.Cartesian);

		// Read the position prior data, with the expectation that positions will be centered
		// automatically around a local origin.
		var options = new SpatialPairingOptions { IgnoreZ = false };

		var cache = new FeatureMatcherCache(options.CacheSize(), database);
		var generator = new SpatialPairGenerator(options, cache);
		RowMajorMatrix<float> positionMatrix = generator.ReadPositionPriorData(cache);

		// Verify that the missing pose prior did not bias the calculated offset.
		var expectedPositionMatrix = new RowMajorMatrix<float>(3, 3, [-1, -2, -3, 0, 0, 0, 1, 2, 3]);
		await Assert.That(EigenMatchers.EigenMatrixNear(positionMatrix, expectedPositionMatrix)).IsTrue();
	}

	[Test]
	public async Task SpatialPairGenerator_MinNumNeighborsControlsMatchingDistance()
	{
		const int NumImages = 4;
		using InMemoryDatabase database = CreateSyntheticDatabase(NumImages);
		List<Image> images = database.ReadAllImages();

		WritePositionPrior(database, images[0], new Vector3d(1, 1, 2));
		WritePositionPrior(database, images[1], new Vector3d(1, 2, 3));
		WritePositionPrior(database, images[2], new Vector3d(2, 3, 4));
		WritePositionPrior(database, images[3], new Vector3d(2, 4, 12));

		var options = new SpatialPairingOptions { IgnoreZ = false, MaxNumNeighbors = NumImages, MaxDistance = 0.0 };

		{
			options.MinNumNeighbors = 0;
			await Assert.That(options.Check()).IsFalse();
		}

		{
			options.MinNumNeighbors = 1;
			var generator = new SpatialPairGenerator(options, database);
			await ElementsAre(generator.Next(), Pair(images, 0, 1));
			await ElementsAre(generator.Next(), Pair(images, 1, 0));
			await ElementsAre(generator.Next(), Pair(images, 2, 1));
			await ElementsAre(generator.Next(), Pair(images, 3, 2));
			await Assert.That(generator.Next()).IsEmpty();
		}

		{
			options.MinNumNeighbors = 2;
			var generator = new SpatialPairGenerator(options, database);
			await ElementsAre(generator.Next(), Pair(images, 0, 1), Pair(images, 0, 2));
			await ElementsAre(generator.Next(), Pair(images, 1, 0), Pair(images, 1, 2));
			await ElementsAre(generator.Next(), Pair(images, 2, 1), Pair(images, 2, 0));
			await ElementsAre(generator.Next(), Pair(images, 3, 2), Pair(images, 3, 1));
			await Assert.That(generator.Next()).IsEmpty();
		}

		{
			options.MinNumNeighbors = 3;
			var generator = new SpatialPairGenerator(options, database);
			await ElementsAre(generator.Next(), Pair(images, 0, 1), Pair(images, 0, 2), Pair(images, 0, 3));
			await ElementsAre(generator.Next(), Pair(images, 1, 0), Pair(images, 1, 2), Pair(images, 1, 3));
			await ElementsAre(generator.Next(), Pair(images, 2, 1), Pair(images, 2, 0), Pair(images, 2, 3));
			await ElementsAre(generator.Next(), Pair(images, 3, 2), Pair(images, 3, 1), Pair(images, 3, 0));
			await Assert.That(generator.Next()).IsEmpty();
		}
	}

	// One block of SpatialPairGenerator_ReadPositionPriorData: the number of position rows
	// read when the first image's prior is at `firstPosition` and the second and last images
	// have finite priors (with four images, the third has none).
	private static int ReadPositionPriorRows(int numImages, Vector3d firstPosition, bool ignoreZ, bool clearFirst = false)
	{
		using InMemoryDatabase database = CreateSyntheticDatabase(numImages);
		List<Image> images = database.ReadAllImages();
		if (images.Count != numImages)
		{
			// CHECK_EQ(images.size(), kNumImages).
			throw new InvalidOperationException($"Expected {numImages} images, got {images.Count}.");
		}

		if (clearFirst)
		{
			database.ClearPosePriors();
		}

		WritePositionPrior(database, images[0], firstPosition);
		WritePositionPrior(database, images[1], new Vector3d(2, 3, 4));
		// With four images the third has no prior (the "some images don't have a pose
		// prior" block).
		WritePositionPrior(database, images[numImages - 1], new Vector3d(2, 4, 12));

		var options = new SpatialPairingOptions { MaxNumNeighbors = 1, MaxDistance = 1000, IgnoreZ = ignoreZ };

		var cache = new FeatureMatcherCache(options.CacheSize(), database);
		var generator = new SpatialPairGenerator(options, cache);

		return generator.ReadPositionPriorData(cache).Rows;
	}

	[Test]
	public async Task SpatialPairGenerator_ReadPositionPriorData()
	{
		await Assert.That(ReadPositionPriorRows(3, new Vector3d(1, 2, 3), ignoreZ: false)).IsEqualTo(3);

		// Test that the position prior data is read correctly when some images don't have a
		// pose prior.
		await Assert.That(ReadPositionPriorRows(4, new Vector3d(1, 2, 3), ignoreZ: false, clearFirst: true)).IsEqualTo(3);

		await Assert.That(ReadPositionPriorRows(3, new Vector3d(0, 0, double.NaN), ignoreZ: false)).IsEqualTo(2);

		await Assert.That(ReadPositionPriorRows(3, new Vector3d(0, 0, double.NaN), ignoreZ: true)).IsEqualTo(3);

		await Assert.That(ReadPositionPriorRows(3, new Vector3d(double.NaN, double.NaN, double.NaN), ignoreZ: false))
			.IsEqualTo(2);

		await Assert.That(ReadPositionPriorRows(3, new Vector3d(double.NaN, double.NaN, double.NaN), ignoreZ: true))
			.IsEqualTo(2);
	}

	/// <summary>
	/// C#-only (no COLMAP counterpart): the bounded-heap neighbor search gives exactly the
	/// tables of a full sort of every candidate by (float squared distance, index) - the
	/// order faiss's IndexFlatL2 returns - on 2000 random positions on a coarse grid, so
	/// that many distances tie.
	/// </summary>
	[Test]
	public async Task SpatialPairGenerator_HeapSearchMatchesFullSort()
	{
		const int NumPositions = 2000;
		const int Knn = 51;
		var random = new Random(42);
		var positions = new RowMajorMatrix<float>(NumPositions, 3);
		for (int i = 0; i < NumPositions; ++i)
		{
			for (int c = 0; c < 3; ++c)
			{
				positions[i, c] = random.Next(-8, 9) * 0.5f;
			}
		}

		var indexMatrix = new int[NumPositions * Knn];
		var distanceMatrix = new float[NumPositions * Knn];
		SpatialPairGenerator.SearchNearestNeighbors(positions, Knn, indexMatrix, distanceMatrix);

		int mismatches = 0;
		var candidates = new (float DistanceSquared, int Index)[NumPositions];
		for (int query = 0; query < NumPositions; ++query)
		{
			for (int other = 0; other < NumPositions; ++other)
			{
				float dx = positions[query, 0] - positions[other, 0];
				float dy = positions[query, 1] - positions[other, 1];
				float dz = positions[query, 2] - positions[other, 2];
				float distanceSquared = (dx * dx) + (dy * dy);
				distanceSquared += dz * dz;
				candidates[other] = (distanceSquared, other);
			}

			Array.Sort(candidates, (a, b) =>
			{
				int byDistance = a.DistanceSquared.CompareTo(b.DistanceSquared);
				return byDistance != 0 ? byDistance : a.Index.CompareTo(b.Index);
			});

			for (int j = 0; j < Knn; ++j)
			{
				if (indexMatrix[(query * Knn) + j] != candidates[j].Index
					|| distanceMatrix[(query * Knn) + j] != candidates[j].DistanceSquared)
				{
					mismatches++;
				}
			}
		}

		await Assert.That(mismatches).IsEqualTo(0);
	}
}
