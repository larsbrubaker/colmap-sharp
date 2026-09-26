// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionClustering: port of colmap/scene/reconstruction_clustering.h and .cc. Groups
// the frames of a Reconstruction (Reconstruction.cs) by how many 3D points they share: a
// covisibility graph, an adaptive median-minus-MAD edge threshold (Mathematics/MathUtils.cs)
// and union-find over the strong edges (Mathematics/UnionFind.cs). Tests:
// ColmapSharp.Tests/Scene/ReconstructionClusteringTests.cs (reconstruction_clustering_test.cc
// 1:1).
//
// Tier A (exact) for the memberships and for the ids of clusters of different sizes.
// Translation notes:
// - Hash-container order. COLMAP counts covisibility in NodeHashMaps and walks them; that
//   order only feeds the union-find and the median, neither of which depends on it. The one
//   place it leaks is the id of clusters of equal size: COLMAP sorts the clusters by size
//   with std::sort, so equal-size clusters are numbered in an unspecified (hash-map) order.
//   Here they are numbered by ascending smallest frame id (docs/CPP_DIVERGENCES.md, entry
//   93).
// - The result is a Dictionary; like COLMAP's NodeHashMap its enumeration order carries no
//   meaning (here it is deterministic: union-find order, then the frames outside it).
// - LOG(WARNING) goes to Util/Log.cs; the LOG(INFO) progress lines are dropped.

using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>Port of colmap::ReconstructionClusteringOptions.</summary>
public sealed class ReconstructionClusteringOptions
{
	/// <summary>
	/// Minimum number of shared 3D points between two frames to consider them connected in
	/// the covisibility graph.
	/// </summary>
	public int MinCovisibilityCount { get; set; } = 5;

	/// <summary>
	/// Minimum edge weight threshold for clustering. If the adaptive threshold (median - MAD)
	/// falls below this, this value is used instead.
	/// </summary>
	public double MinEdgeWeightThreshold { get; set; } = 20.0;

	/// <summary>
	/// Minimum number of registered frames required for a cluster to be kept. Clusters with
	/// fewer frames will be discarded.
	/// </summary>
	public int MinNumRegFrames { get; set; } = 3;

	/// <summary>Port of Check (THROW_CHECK_*: throws on a violation).</summary>
	public void Check()
	{
		Util.Check.Ge(MinCovisibilityCount, 1);
		Util.Check.Gt(MinEdgeWeightThreshold, 0.0);
		Util.Check.Ge(MinNumRegFrames, 2);
	}
}

/// <summary>Port of colmap/scene/reconstruction_clustering.h.</summary>
public static class ReconstructionClustering
{
	/// <summary>
	/// Port of colmap::ClusterReconstructionFrames: clusters frames based on 3D point
	/// covisibility and marks weakly connected frames.
	/// <para>
	/// Covisibility is the number of 3D points visible in both frames. Frames with high
	/// covisibility likely have reliable relative pose estimates, while weakly connected
	/// frames may have less reliable geometry. Algorithm: (1) build a covisibility graph
	/// where edges connect frames sharing at least MinCovisibilityCount points; (2) compute
	/// an adaptive edge weight threshold using median minus median absolute deviation (MAD);
	/// (3) cluster frames using union-find, merging strongly connected frames; (4) assign
	/// cluster ids sorted by number of frames in descending order (cluster 0 is the largest).
	/// </para>
	/// </summary>
	/// <returns>
	/// Map from frame id to cluster id for all registered frames (and every frame seen in a
	/// track of length &gt; 2); -1 marks a frame in a cluster smaller than MinNumRegFrames.
	/// Empty when no frame pair passes MinCovisibilityCount.
	/// </returns>
	public static Dictionary<uint, int> ClusterReconstructionFrames(
		ReconstructionClusteringOptions options,
		Reconstruction reconstruction)
	{
		options.Check();

		// Step 1: Compute covisibility counts between all frame pairs. For each 3D point,
		// increment the count for every pair of frames that sees it.
		var frameCovisibilityCount = new Dictionary<ulong, int>();
		var nodes = new HashSet<uint>();
		// Insert all registered frames to the nodes set.
		foreach (var frameId in reconstruction.RegFrameIds)
		{
			nodes.Add(frameId);
		}

		foreach (var point3D in reconstruction.Points3D.Values)
		{
			var track = point3D.Track;
			if (track.Length <= 2)
			{
				continue;
			}

			for (var i = 0; i < track.Length; i++)
			{
				var frameId1 = reconstruction.Image(track.Element(i).ImageId).FrameId;

				nodes.Add(frameId1);
				// COLMAP's TODO: this may over-count frame pairs when multiple images from the
				// same frame appear in a track (e.g., rig cameras).
				for (var j = i + 1; j < track.Length; j++)
				{
					var frameId2 = reconstruction.Image(track.Element(j).ImageId).FrameId;
					if (frameId1 == frameId2)
					{
						continue;
					}

					var pairId = Types.ImagePairToPairId(frameId1, frameId2);
					frameCovisibilityCount[pairId] = frameCovisibilityCount.GetValueOrDefault(pairId) + 1;
				}
			}
		}

		// Filter edges to keep only reliable connections.
		var edgeWeights = new Dictionary<ulong, int>();
		foreach (var (pairId, count) in frameCovisibilityCount)
		{
			if (count >= options.MinCovisibilityCount)
			{
				edgeWeights[pairId] = count;
			}
		}

		if (edgeWeights.Count == 0)
		{
			Log.Warning("No valid frame pairs found for clustering");
			return [];
		}

		// Compute adaptive threshold using median minus median absolute deviation (MAD).
		var weightValues = edgeWeights.Values.ToArray();
		var (median, mad) = MathUtils.MedianAbsoluteDeviation<int>(weightValues);
		var threshold = Math.Max(median - mad, options.MinEdgeWeightThreshold);

		// Cluster frames based on covisibility weights.
		return EstablishStrongClusters(options, nodes, edgeWeights, threshold);
	}

	// Clusters nodes using union-find, merging nodes connected by strong edges
	// (weight >= threshold).
	private static Dictionary<uint, int> EstablishStrongClusters(
		ReconstructionClusteringOptions options,
		HashSet<uint> nodes,
		Dictionary<ulong, int> edgeWeights,
		double edgeWeightThreshold)
	{
		var uf = new UnionFind<uint>();
		uf.Reserve(nodes.Count);

		// Create initial clusters from strong edges. COLMAP's TODO: use different
		// thresholds for different edges based on local statistics.
		foreach (var (pairId, weight) in edgeWeights)
		{
			if (weight >= edgeWeightThreshold)
			{
				var (frameId1, frameId2) = Types.PairIdToImagePair(pairId);
				uf.Union(frameId1, frameId2);
			}
		}

		// Collect nodes by their union-find roots.
		uf.Compress();
		var rootToNodes = new Dictionary<uint, List<uint>>();
		foreach (var (node, root) in uf.Parents)
		{
			if (!rootToNodes.TryGetValue(root, out var clusterNodes))
			{
				clusterNodes = [];
				rootToNodes.Add(root, clusterNodes);
			}

			clusterNodes.Add(node);
		}

		// Sort by number of frames (largest first); equal sizes by ascending smallest frame
		// id, where COLMAP's order is its hash map's (docs/CPP_DIVERGENCES.md, entry 93).
		var sortedClusters = rootToNodes.Values
			.OrderByDescending(clusterNodes => clusterNodes.Count)
			.ThenBy(clusterNodes => clusterNodes.Min())
			.ToList();

		// Assign cluster ids based on sorted order.
		var clusterIds = new Dictionary<uint, int>();
		var numValidClusters = 0;
		foreach (var clusterNodes in sortedClusters)
		{
			if (clusterNodes.Count >= options.MinNumRegFrames)
			{
				foreach (var node in clusterNodes)
				{
					clusterIds[node] = numValidClusters;
				}

				numValidClusters++;
			}
			else
			{
				// Clusters smaller than MinNumRegFrames are discarded.
				foreach (var node in clusterNodes)
				{
					clusterIds[node] = -1;
				}
			}
		}

		// Ensure all nodes are assigned a cluster id.
		foreach (var node in nodes)
		{
			if (!uf.FindIfExists(node, out _))
			{
				clusterIds[node] = -1;
			}
		}

		return clusterIds;
	}
}
