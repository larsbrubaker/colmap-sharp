// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SpatialPairGenerator: port of colmap::SpatialPairGenerator
// (colmap/controllers/pairing.h/.cc) - pairs each image that has a position prior (GPS or
// Cartesian, Geometry/PosePrior.cs) with its nearest neighbors in space. Part of the pair
// generator family of Controllers/PairGenerator.cs; options in PairingOptions.cs. Tests:
// ColmapSharp.Tests/Controllers/PairingTests.cs (pairing_test.cc 1:1).
//
// Tier A (exact) up to faiss's float summation: COLMAP finds the neighbors with faiss's
// brute-force IndexFlatL2 (MIT, but native), replaced here by a managed brute-force search
// (docs/CPP_DIVERGENCES.md entry 52). Like faiss, it ranks by float squared distance with
// ties broken by the smaller index, and computes the distance from coordinate differences;
// faiss switches to ||x||^2 + ||y||^2 - 2 x.y (a BLAS matrix product) for 20 or more
// positions, which can reorder near-equal neighbors. COLMAP parallelizes the search with
// OpenMP (num_threads); here it is a Parallel.For over queries, each writing its own row,
// and the num_threads option is not ported.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>
/// Port of colmap::SpatialPairGenerator: pairs each image with a position prior with its
/// nearest neighbors, within MaxDistance or up to MinNumNeighbors of them.
/// </summary>
public sealed class SpatialPairGenerator : PairGenerator
{
	private readonly SpatialPairingOptions options;
	private readonly List<uint> imageIds;
	private readonly List<int> positionIdxs = [];

	// Row-major num_positions x knn tables of the neighbors' position indices and float
	// squared distances, nearest first (faiss's labels and distances).
	private readonly int[] indexMatrix = [];
	private readonly float[] distanceSquaredMatrix = [];
	private readonly int knn;
	private int currentIdx;

	/// <summary>A generator over the images and pose priors of <paramref name="cache"/>.</summary>
	public SpatialPairGenerator(SpatialPairingOptions options, FeatureMatcherCache cache)
	{
		this.options = options;
		imageIds = Check.NotNull(cache).GetImageIds();
		Check.That(options.Check());

		RowMajorMatrix<float> positionMatrix = ReadPositionPriorData(cache);
		int numPositions = positionIdxs.Count;
		if (numPositions == 0)
		{
			// No images with location data.
			return;
		}

		knn = Math.Min(options.MaxNumNeighbors + 1, numPositions);
		indexMatrix = new int[numPositions * knn];
		distanceSquaredMatrix = new float[numPositions * knn];
		SearchNearestNeighbors(positionMatrix, knn, indexMatrix, distanceSquaredMatrix);
	}

	/// <summary>A generator over the images and pose priors of <paramref name="database"/>.</summary>
	public SpatialPairGenerator(SpatialPairingOptions options, Database database)
		: this(options, new FeatureMatcherCache(options.CacheSize(), Check.NotNull(database)))
	{
	}

	/// <inheritdoc/>
	public override void Reset() => currentIdx = 0;

	/// <inheritdoc/>
	public override bool HasFinished() => currentIdx >= positionIdxs.Count;

	/// <inheritdoc/>
	public override int NumBatches => positionIdxs.Count; // One block per image with a position.

	/// <inheritdoc/>
	public override List<(uint ImageId1, uint ImageId2)> Next()
	{
		var imagePairs = new List<(uint, uint)>();
		if (HasFinished())
		{
			return imagePairs;
		}

		float maxDistanceSquared = (float)(options.MaxDistance * options.MaxDistance);
		for (int j = 0; j < knn; ++j)
		{
			// Check if query equals result.
			if (indexMatrix[(currentIdx * knn) + j] == currentIdx)
			{
				continue;
			}

			// Since the nearest neighbors are sorted by distance, we can break once the
			// distance is too large and enough neighbors are collected.
			if (distanceSquaredMatrix[(currentIdx * knn) + j] > maxDistanceSquared && j > options.MinNumNeighbors)
			{
				break;
			}

			uint imageId = imageIds[positionIdxs[currentIdx]];
			int nnIdx = positionIdxs[indexMatrix[(currentIdx * knn) + j]];
			imagePairs.Add((imageId, imageIds[nnIdx]));
		}

		++currentIdx;
		return imagePairs;
	}

	/// <summary>
	/// Port of SpatialPairGenerator::ReadPositionPriorData: the positions of the images with
	/// a usable prior (WGS84 converted to ECEF; z zeroed with IgnoreZ), one row each in
	/// image id order, minus their mean and cast to float. Also records which images they
	/// belong to.
	/// </summary>
	public RowMajorMatrix<float> ReadPositionPriorData(FeatureMatcherCache cache)
	{
		var gpsTransform = new GPSTransform();
		var positions = new List<Vector3d>(imageIds.Count);
		positionIdxs.Clear();

		for (int i = 0; i < imageIds.Count; ++i)
		{
			PosePrior? maybePosePrior = cache.FindImagePosePriorOrNull(imageIds[i]);
			if (maybePosePrior is not PosePrior posePrior)
			{
				continue;
			}

			Vector3d position = posePrior.Position;
			if ((!options.IgnoreZ && !posePrior.HasPosition())
				|| (options.IgnoreZ && !(double.IsFinite(position.X) && double.IsFinite(position.Y))))
			{
				continue;
			}

			positionIdxs.Add(i);

			double z = options.IgnoreZ ? 0 : position.Z;
			if (posePrior.CoordinateSystem == PosePriorCoordinateSystem.Wgs84)
			{
				positions.Add(gpsTransform.EllipsoidToECEF([new Vector3d(position.X, position.Y, z)])[0]);
			}
			else
			{
				// UNDEFINED (and anything else) is assumed to be Cartesian, as in COLMAP
				// (which logs a warning).
				positions.Add(new Vector3d(position.X, position.Y, z));
			}
		}

		// Subtract the mean coordinate (of the populated rows only) before casting to float
		// for better numerical precision when dealing with large coordinates (e.g. GPS). This
		// is particularly important for projected Cartesian coordinate systems, which can
		// contain very large values in metres.
		double sumX = 0;
		double sumY = 0;
		double sumZ = 0;
		foreach (Vector3d position in positions)
		{
			sumX += position.X;
			sumY += position.Y;
			sumZ += position.Z;
		}

		// Eigen's colwise().mean() may sum a column in vectorized blocks, so its double mean
		// can differ from this sequential sum in the last ulp before the float cast
		// (docs/CPP_DIVERGENCES.md entry 52).
		double meanX = sumX / positions.Count;
		double meanY = sumY / positions.Count;
		double meanZ = sumZ / positions.Count;

		var positionMatrix = new RowMajorMatrix<float>(positions.Count, 3);
		for (int row = 0; row < positions.Count; ++row)
		{
			positionMatrix[row, 0] = (float)(positions[row].X - meanX);
			positionMatrix[row, 1] = (float)(positions[row].Y - meanY);
			positionMatrix[row, 2] = (float)(positions[row].Z - meanZ);
		}

		return positionMatrix;
	}

	/// <summary>
	/// Brute-force k nearest neighbors of every position among all positions (itself
	/// included), the replacement for faiss::IndexFlatL2::search. Writes row-major
	/// numPositions x knn tables, nearest first; equal float distances rank by the smaller
	/// index, as faiss's result heap does. Each query keeps its knn best in a bounded max-heap
	/// (top = the worst kept by (distance, index)), so a query costs O(n log knn); queries
	/// run in parallel, each writing only its own row, so the result is the same on any
	/// number of threads.
	/// </summary>
	internal static void SearchNearestNeighbors(
		RowMajorMatrix<float> positionMatrix, int knn, int[] indexMatrix, float[] distanceSquaredMatrix)
	{
		int numPositions = positionMatrix.Rows;
		Parallel.For(
			0,
			numPositions,
			() => new (float DistanceSquared, int Index)[knn],
			(query, _, heap) =>
			{
				SearchOne(positionMatrix, query, heap);
				for (int j = knn - 1; j >= 0; --j)
				{
					// Popping the max-heap yields the kept neighbors farthest first.
					indexMatrix[(query * knn) + j] = heap[0].Index;
					distanceSquaredMatrix[(query * knn) + j] = heap[0].DistanceSquared;
					heap[0] = heap[j];
					SiftDown(heap, j);
				}

				return heap;
			},
			_ => { });
	}

	// Fills `heap` (length knn <= number of positions) with the knn nearest positions to
	// `query`, as a max-heap.
	private static void SearchOne(RowMajorMatrix<float> positionMatrix, int query, (float DistanceSquared, int Index)[] heap)
	{
		int knn = heap.Length;
		int size = 0;
		float qx = positionMatrix[query, 0];
		float qy = positionMatrix[query, 1];
		float qz = positionMatrix[query, 2];
		for (int other = 0; other < positionMatrix.Rows; ++other)
		{
			float dx = qx - positionMatrix[other, 0];
			float dy = qy - positionMatrix[other, 1];
			float dz = qz - positionMatrix[other, 2];
			float distanceSquared = (dx * dx) + (dy * dy);
			distanceSquared += dz * dz;
			var candidate = (distanceSquared, other);
			if (size < knn)
			{
				// Sift up.
				int child = size++;
				while (child > 0)
				{
					int parent = (child - 1) / 2;
					if (Compare(heap[parent], candidate) >= 0)
					{
						break;
					}

					heap[child] = heap[parent];
					child = parent;
				}

				heap[child] = candidate;
			}
			else if (Compare(candidate, heap[0]) < 0)
			{
				heap[0] = candidate;
				SiftDown(heap, knn);
			}
		}
	}

	// Restores the max-heap property of heap[0..size) after heap[0] changed.
	private static void SiftDown((float DistanceSquared, int Index)[] heap, int size)
	{
		int parent = 0;
		var item = heap[0];
		while (true)
		{
			int child = (2 * parent) + 1;
			if (child >= size)
			{
				break;
			}

			if (child + 1 < size && Compare(heap[child + 1], heap[child]) > 0)
			{
				child++;
			}

			if (Compare(heap[child], item) <= 0)
			{
				break;
			}

			heap[parent] = heap[child];
			parent = child;
		}

		heap[parent] = item;
	}

	// Neighbor order: by float squared distance, then by index.
	private static int Compare((float DistanceSquared, int Index) a, (float DistanceSquared, int Index) b)
	{
		int byDistance = a.DistanceSquared.CompareTo(b.DistanceSquared);
		return byDistance != 0 ? byDistance : a.Index.CompareTo(b.Index);
	}
}
