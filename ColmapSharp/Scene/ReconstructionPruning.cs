// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionPruning: port of colmap/scene/reconstruction_pruning.h and .cc. Greedily
// selects the 3D points of a Reconstruction (Reconstruction.cs) that add image coverage
// (an 8x8 tile grid per image) and reports the rest as redundant. Tests:
// ColmapSharp.Tests/Scene/ReconstructionPruningTests.cs (reconstruction_pruning_test.cc).
//
// Tier A (exact). The priority queue orders by (gain, point3D_id), a total order over the
// queued entries (a point is re-queued only after it was popped), so the selection does
// not depend on the heap implementation or on the order points are pushed. The result
// lists the redundant ids in Reconstruction.Points3D order, i.e. ascending id
// (divergence 21); COLMAP's order is its hash map's, and its tests
// compare order-insensitively.

using ColmapSharp.Mathematics;

namespace ColmapSharp.Scene;

/// <summary>Port of colmap/scene/reconstruction_pruning.h.</summary>
public static class ReconstructionPruning
{
	private const int NumImageTilesPerDim = 8;
	private const int NumImageTiles = NumImageTilesPerDim * NumImageTilesPerDim;

	/// <summary>
	/// Port of colmap::FindRedundantPoints3D: the ids of the 3D points whose coverage gain,
	/// when greedily selecting points by largest gain, is at most
	/// <paramref name="minCoverageGain"/>.
	/// </summary>
	public static List<ulong> FindRedundantPoints3D(double minCoverageGain, Reconstruction reconstruction)
	{
		Dictionary<uint, int[]> imageTileIdxs = ComputeImageTileIdxs(NumImageTilesPerDim, reconstruction);
		var numSelectedPoints3DPerImageTile = new Dictionary<uint, int[]>(reconstruction.NumImages);
		foreach (uint imageId in reconstruction.Images.Keys)
		{
			numSelectedPoints3DPerImageTile[imageId] = new int[NumImageTiles];
		}

		// std::priority_queue is a max-heap on (gain, point3D_id); PriorityQueue dequeues the
		// smallest priority, so the comparer is reversed.
		var priorityQueue = new PriorityQueue<(ulong Point3DId, Point3D Point3D), (double Gain, ulong Point3DId)>(
			Comparer<(double Gain, ulong Point3DId)>.Create((left, right) => right.CompareTo(left)));
		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			double gain = ComputeCoverageGain(point3D, numSelectedPoints3DPerImageTile, imageTileIdxs);
			priorityQueue.Enqueue((point3DId, point3D), (gain, point3DId));
		}

		var selectedPoint3DIds = new HashSet<ulong>(reconstruction.NumPoints3D);
		while (priorityQueue.TryDequeue(out (ulong Point3DId, Point3D Point3D) point3DInfo, out (double Gain, ulong Point3DId) priority))
		{
			if (priority.Gain <= minCoverageGain)
			{
				break;
			}

			// If another point has been selected that shares an image with the current point,
			// then the gain of the current point might have changed.
			double updatedGain = ComputeCoverageGain(point3DInfo.Point3D, numSelectedPoints3DPerImageTile, imageTileIdxs);
			if (updatedGain < priority.Gain)
			{
				priorityQueue.Enqueue(point3DInfo, (updatedGain, point3DInfo.Point3DId));
				continue;
			}

			foreach (TrackElement trackEl in point3DInfo.Point3D.Track.Elements)
			{
				int tileIdx = imageTileIdxs[trackEl.ImageId][trackEl.Point2DIdx];
				numSelectedPoints3DPerImageTile[trackEl.ImageId][tileIdx]++;
			}

			selectedPoint3DIds.Add(point3DInfo.Point3DId);
		}

		var redundantPoint3DIds = new List<ulong>(reconstruction.NumPoints3D - selectedPoint3DIds.Count);
		foreach (ulong point3DId in reconstruction.Points3D.Keys)
		{
			if (!selectedPoint3DIds.Contains(point3DId))
			{
				redundantPoint3DIds.Add(point3DId);
			}
		}

		return redundantPoint3DIds;
	}

	private static Dictionary<uint, int[]> ComputeImageTileIdxs(int numTilesPerDim, Reconstruction reconstruction)
	{
		var imageTileIdxs = new Dictionary<uint, int[]>(reconstruction.NumImages);
		foreach ((uint imageId, Image image) in reconstruction.Images)
		{
			Camera camera = reconstruction.Camera(image.CameraId);
			int numPoints2D = (int)image.NumPoints2D;
			var tileIdxs = new int[numPoints2D];
			for (int point2DIdx = 0; point2DIdx < numPoints2D; ++point2DIdx)
			{
				Point2D point2D = image.Points2D[point2DIdx];
				// Clamp<int> converts its double argument to int first (truncation toward zero).
				int tileIdxX = MathUtils.Clamp((int)(numTilesPerDim * point2D.Xy.X / camera.Width), 0, numTilesPerDim - 1);
				int tileIdxY = MathUtils.Clamp((int)(numTilesPerDim * point2D.Xy.Y / camera.Height), 0, numTilesPerDim - 1);
				tileIdxs[point2DIdx] = tileIdxX * numTilesPerDim + tileIdxY;
			}

			imageTileIdxs[imageId] = tileIdxs;
		}

		return imageTileIdxs;
	}

	private static double ComputeCoverageGain(
		Point3D point3D,
		Dictionary<uint, int[]> numSelectedPoints3DPerImageTile,
		Dictionary<uint, int[]> imageTileIdxs)
	{
		double gain = 0;
		foreach (TrackElement trackEl in point3D.Track.Elements)
		{
			int tileIdx = imageTileIdxs[trackEl.ImageId][trackEl.Point2DIdx];
			int n = 1 + numSelectedPoints3DPerImageTile[trackEl.ImageId][tileIdx];
			gain += 1.0 / Math.Sqrt(n) - 1.0 / Math.Sqrt(1 + n);
		}

		return gain;
	}
}
