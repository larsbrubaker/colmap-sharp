// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SceneClustering: port of colmap/scene/scene_clustering.h and .cc. Splits the scene graph
// (images linked by their number of inlier matches) into overlapping clusters with the
// normalized graph cut of Mathematics/GraphCut.cs, either hierarchically until every leaf
// is small enough or flat into `Branching` clusters. Tests:
// ColmapSharp.Tests/Scene/SceneClusteringTests.cs (scene_clustering_test.cc 1:1).
//
// Tier C (outcome): the partition itself comes from our multilevel partitioner, not METIS
// (docs/CPP_DIVERGENCES.md, entry 77), so the memberships match COLMAP's tests but the
// labels and the order of the child clusters can differ.
//
// Translation notes:
// - Ties (docs/CPP_DIVERGENCES.md, entry 93). COLMAP sorts the overlap-candidate edges and
//   each image's related images by descending weight with std::sort, which leaves equal
//   weights in an unspecified order. Here equal weights keep their edge order (a stable
//   sort). That is also what libc++ produces for lists under 24 entries (sort3/4/5 and
//   insertion sort); longer lists with tied weights can differ.
// - The flat child-cluster sort (entry 94). COLMAP's comparator ("size >= other size and
//   min id < other min id") is not a strict weak ordering and dereferences min_element of
//   an empty cluster. Here children are ordered by what its comment says it intends:
//   descending size, then ascending smallest image id, with empty clusters last.
// - COLMAP's image_t -> int narrowing of the edge endpoints is kept (unchecked casts), so
//   the graph cut sees the same vertex ids.
// - Create's pair order (entry 120). COLMAP iterates a hash map of image pairs; here the
//   pairs are ordered by a scrambled pair id, so ties do not all break toward low image ids.
// - The other hash containers COLMAP uses here (FlatHashSet of a child's images, NodeHashMap
//   of related images) are only looked up, never iterated in a way that reaches an output.

using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::SceneClustering: scene clustering using normalized cuts on the scene
/// graph. The scene is hierarchically partitioned into overlapping clusters until a maximum
/// number of images is in a leaf node.
/// </summary>
public sealed class SceneClustering
{
	/// <summary>Port of SceneClustering::Options.</summary>
	public sealed class Options
	{
		/// <summary>Flag for hierarchical vs flat clustering.</summary>
		public bool IsHierarchical { get; set; } = true;

		/// <summary>The branching factor of the hierarchical clustering.</summary>
		public int Branching { get; set; } = 2;

		/// <summary>The number of overlapping images between child clusters.</summary>
		public int ImageOverlap { get; set; } = 50;

		/// <summary>The max related images matches to look for in a flat cluster.</summary>
		public int NumImageMatches { get; set; } = 20;

		/// <summary>
		/// The maximum number of images in a leaf node cluster, otherwise the cluster is
		/// further partitioned using the given branching factor. Note that a cluster leaf node
		/// will have at most LeafMaxNumImages + ImageOverlap images to satisfy the overlap
		/// constraint.
		/// </summary>
		public int LeafMaxNumImages { get; set; } = 500;

		/// <summary>Port of Options::Check (CHECK_OPTION_*: false on a violation).</summary>
		public bool Check() => Branching > 0 && ImageOverlap >= 0;

		internal Options Clone() => (Options)MemberwiseClone();
	}

	/// <summary>Port of SceneClustering::Cluster.</summary>
	public sealed class Cluster
	{
		/// <summary>The images of this cluster, including the overlap images.</summary>
		public List<uint> ImageIds { get; } = [];

		/// <summary>The child clusters; empty for a leaf.</summary>
		public List<Cluster> ChildClusters { get; } = [];
	}

	private readonly Options _options;
	private Cluster? _rootCluster;

	/// <summary>Creates an unpartitioned clustering; throws if the options are invalid.</summary>
	public SceneClustering(Options options)
	{
		_options = options.Clone();
		Check.That(_options.Check());
	}

	/// <summary>
	/// Port of SceneClustering::Partition: builds the cluster tree of the scene graph whose
	/// edges are <paramref name="imagePairs"/> weighted by <paramref name="numInliers"/>. May
	/// only be called once.
	/// </summary>
	public void Partition(IReadOnlyList<(uint ImageId1, uint ImageId2)> imagePairs, IReadOnlyList<int> numInliers)
	{
		Check.That(_rootCluster == null);
		Check.Eq(imagePairs.Count, numInliers.Count);

		var imageIds = new SortedSet<uint>();
		var edges = new List<(int, int)>(imagePairs.Count);
		foreach (var (imageId1, imageId2) in imagePairs)
		{
			imageIds.Add(imageId1);
			imageIds.Add(imageId2);
			edges.Add((unchecked((int)imageId1), unchecked((int)imageId2)));
		}

		_rootCluster = new Cluster();
		_rootCluster.ImageIds.AddRange(imageIds);
		if (_options.IsHierarchical)
		{
			PartitionHierarchicalCluster(edges, numInliers, _rootCluster);
		}
		else
		{
			PartitionFlatCluster(edges, numInliers);
		}
	}

	/// <summary>The root of the cluster tree, or null before <see cref="Partition"/>.</summary>
	public Cluster? GetRootCluster() => _rootCluster;

	/// <summary>
	/// Port of SceneClustering::GetLeafClusters: the clusters without children, in COLMAP's
	/// depth-first (last-pushed first) visiting order. The root is its own only leaf when it
	/// has no children.
	/// </summary>
	public List<Cluster> GetLeafClusters()
	{
		Check.NotNull(_rootCluster);

		var leafClusters = new List<Cluster>();
		if (_rootCluster.ChildClusters.Count == 0)
		{
			leafClusters.Add(_rootCluster);
			return leafClusters;
		}

		var nonLeafClusters = new List<Cluster> { _rootCluster };
		while (nonLeafClusters.Count > 0)
		{
			var cluster = nonLeafClusters[^1];
			nonLeafClusters.RemoveAt(nonLeafClusters.Count - 1);

			foreach (var childCluster in cluster.ChildClusters)
			{
				if (childCluster.ChildClusters.Count == 0)
				{
					leafClusters.Add(childCluster);
				}
				else
				{
					nonLeafClusters.Add(childCluster);
				}
			}
		}

		return leafClusters;
	}

	/// <summary>
	/// Port of SceneClustering::Create: partitions the scene graph of
	/// <paramref name="databaseCache"/>'s correspondence graph, with the number of matches
	/// between two images as the edge weight. Pairs are taken in a fixed pseudo-random order
	/// of their pair ids, not the graph's ascending insertion order
	/// (docs/CPP_DIVERGENCES.md, entry 120).
	/// </summary>
	public static SceneClustering Create(Options options, DatabaseCache databaseCache)
	{
		var numMatchesBetweenImages = databaseCache.CorrespondenceGraph.NumMatchesBetweenAllImages();

		// COLMAP walks a hash map here, so the edge order, which breaks every weight tie in
		// the graph cut and the overlap selection, has nothing to do with image ids. The
		// correspondence graph's insertion order is ascending pair id, which breaks all ties
		// toward the same few low image ids. Those are usually the images of one rig frame,
		// so sibling clusters could overlap in a single frame, too few to align
		// zero-baseline (panoramic) rigs. Scrambling the pair id restores the property
		// COLMAP relies on and stays deterministic.
		var orderedPairs = numMatchesBetweenImages
			.OrderBy(pair => ScramblePairId(pair.Key))
			.ThenBy(pair => pair.Key);

		var allImagePairs = new List<(uint, uint)>(numMatchesBetweenImages.Count);
		var allNumInliers = new List<int>(numMatchesBetweenImages.Count);
		foreach (var (pairId, numMatches) in orderedPairs)
		{
			allImagePairs.Add(Types.PairIdToImagePair(pairId));
			allNumInliers.Add(unchecked((int)numMatches));
		}

		var sceneClustering = new SceneClustering(options);
		sceneClustering.Partition(allImagePairs, allNumInliers);
		return sceneClustering;
	}

	// The SplitMix64 finalizer (Steele, Lea and Flood, OOPSLA 2014): a bijection on 64-bit
	// values whose output order is unrelated to the input order.
	private static ulong ScramblePairId(ulong pairId)
	{
		unchecked
		{
			var z = pairId + 0x9E3779B97F4A7C15UL;
			z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
			z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
			return z ^ (z >> 31);
		}
	}

	private void PartitionHierarchicalCluster(IReadOnlyList<(int, int)> edges, IReadOnlyList<int> weights, Cluster cluster)
	{
		Check.Eq(edges.Count, weights.Count);

		// If the cluster is small enough, we return from the recursive clustering. COLMAP
		// adds the two limits as size_t, so a negative LeafMaxNumImages wraps to "never split".
		var maxSize = unchecked((ulong)(long)_options.LeafMaxNumImages + (ulong)(long)_options.ImageOverlap);
		if (edges.Count == 0 || (ulong)cluster.ImageIds.Count <= maxSize)
		{
			return;
		}

		var branching = _options.Branching;

		// Partition the cluster using a normalized cut on the scene graph.
		var labels = GraphCut.ComputeNormalizedMinGraphCut(edges, weights, branching);

		// Assign the images to the clustered child clusters.
		for (var i = 0; i < branching; i++)
		{
			cluster.ChildClusters.Add(new Cluster());
		}

		foreach (var imageId in cluster.ImageIds)
		{
			if (labels.TryGetValue(unchecked((int)imageId), out var label))
			{
				cluster.ChildClusters[label].ImageIds.Add(imageId);
			}
			else
			{
				Log.Warning($"Graph cut failed to assign cluster label to image {imageId}; assigning to cluster 0");
				cluster.ChildClusters[0].ImageIds.Add(imageId);
			}
		}

		// Collect the edges between child clusters as overlap candidates.
		var overlappingEdges = new List<((int, int) Edge, int Weight)>[branching];
		for (var i = 0; i < branching; i++)
		{
			overlappingEdges[i] = [];
		}

		for (var i = 0; i < edges.Count; i++)
		{
			var label1 = labels[edges[i].Item1];
			var label2 = labels[edges[i].Item2];
			if (label1 != label2)
			{
				overlappingEdges[label1].Add((edges[i], weights[i]));
				overlappingEdges[label2].Add((edges[i], weights[i]));
			}
		}

		// Add overlapping images between sibling clusters before partitioning.
		if (_options.ImageOverlap > 0)
		{
			for (var i = 0; i < branching; i++)
			{
				// Descending weight; equal weights keep their edge order (entry 93).
				var sortedEdges = overlappingEdges[i].OrderByDescending(edge => edge.Weight);

				var overlappingImageIds = new SortedSet<uint>();
				foreach (var ((edgeImage1, edgeImage2), _) in sortedEdges)
				{
					var imageId1 = unchecked((uint)edgeImage1);
					var imageId2 = unchecked((uint)edgeImage2);

					overlappingImageIds.Add(labels[edgeImage1] == i ? imageId2 : imageId1);
					if (overlappingImageIds.Count >= _options.ImageOverlap)
					{
						break;
					}
				}

				cluster.ChildClusters[i].ImageIds.AddRange(overlappingImageIds);
			}
		}

		// Retain the full subgraph induced by every expanded child cluster. Edges incident
		// to inherited overlap images allow the graph cut to assign those images based on
		// all their connections at every subsequent level.
		var childEdges = new List<(int, int)>[branching];
		var childWeights = new List<int>[branching];
		for (var i = 0; i < branching; i++)
		{
			childEdges[i] = [];
			childWeights[i] = [];
			var childImageIds = new HashSet<uint>(cluster.ChildClusters[i].ImageIds);
			for (var j = 0; j < edges.Count; j++)
			{
				if (childImageIds.Contains(unchecked((uint)edges[j].Item1))
					&& childImageIds.Contains(unchecked((uint)edges[j].Item2)))
				{
					childEdges[i].Add(edges[j]);
					childWeights[i].Add(weights[j]);
				}
			}
		}

		// Recursively partition all the child clusters.
		for (var i = 0; i < branching; i++)
		{
			// Skip empty clusters or clusters where the current cluster has as many images
			// as its child to avoid infinite loops. This can happen because the normalized
			// cut sometimes decides to put all images into one cluster.
			var childCount = cluster.ChildClusters[i].ImageIds.Count;
			if (childCount == 0 || childCount == cluster.ImageIds.Count)
			{
				continue;
			}

			PartitionHierarchicalCluster(childEdges[i], childWeights[i], cluster.ChildClusters[i]);
		}

		// Remove empty clusters.
		cluster.ChildClusters.RemoveAll(childCluster => childCluster.ImageIds.Count == 0);

		// If the child cluster is the same as the current cluster, it is redundant and we
		// can remove it.
		if (cluster.ChildClusters.Count == 1
			&& cluster.ImageIds.Count == cluster.ChildClusters[0].ImageIds.Count)
		{
			cluster.ChildClusters.Clear();
		}
	}

	private void PartitionFlatCluster(IReadOnlyList<(int, int)> edges, IReadOnlyList<int> weights)
	{
		Check.Eq(edges.Count, weights.Count);
		var rootCluster = _rootCluster!;
		var branching = _options.Branching;

		// Partition the cluster using a normalized cut on the scene graph.
		var labels = GraphCut.ComputeNormalizedMinGraphCut(edges, weights, branching);

		// Assign the images to the clustered child clusters.
		for (var i = 0; i < branching; i++)
		{
			rootCluster.ChildClusters.Add(new Cluster());
		}

		foreach (var imageId in rootCluster.ImageIds)
		{
			if (labels.TryGetValue(unchecked((int)imageId), out var label))
			{
				rootCluster.ChildClusters[label].ImageIds.Add(imageId);
			}
		}

		// Sort child clusters by descending size of images and secondarily by lowest image
		// id; empty clusters last (entry 94). OrderBy is stable, so equal keys keep label order.
		var sortedChildren = rootCluster.ChildClusters
			.OrderByDescending(child => child.ImageIds.Count)
			.ThenBy(child => child.ImageIds.Count == 0 ? uint.MaxValue : child.ImageIds.Min())
			.ToList();
		rootCluster.ChildClusters.Clear();
		rootCluster.ChildClusters.AddRange(sortedChildren);

		// For each image find all related images with their weights.
		var relatedImages = new Dictionary<int, List<(int ImageId, int Weight)>>();
		List<(int, int)> GetRelated(int imageId)
		{
			if (!relatedImages.TryGetValue(imageId, out var related))
			{
				related = [];
				relatedImages.Add(imageId, related);
			}

			return related;
		}

		for (var i = 0; i < edges.Count; i++)
		{
			GetRelated(edges[i].Item1).Add((edges[i].Item2, weights[i]));
			GetRelated(edges[i].Item2).Add((edges[i].Item1, weights[i]));
		}

		// Sort related images by decreasing weights; equal weights keep their edge order
		// (entry 93). Each list is sorted on its own, so the map's order does not matter.
		foreach (var related in relatedImages.Values)
		{
			var sorted = related.OrderByDescending(image => image.Weight).ToList();
			related.Clear();
			related.AddRange(sorted);
		}

		// For each cluster add as many of the needed matching images up to the max image
		// overlap allowance. We do the process sequentially for each image to ensure that at
		// least we get the best matches first.
		for (var i = 0; i < branching; i++)
		{
			var origImageIds = rootCluster.ChildClusters[i].ImageIds;
			var clusterImages = new SortedSet<int>();
			foreach (var imageId in origImageIds)
			{
				clusterImages.Add(unchecked((int)imageId));
			}

			var maxSize = clusterImages.Count + _options.ImageOverlap;
			// Check up to all the desired matches. COLMAP compares as size_t, so a negative
			// NumImageMatches means "no limit".
			var numImageMatches = unchecked((ulong)(long)_options.NumImageMatches);
			for (var j = 0; (ulong)j < numImageMatches && clusterImages.Count < maxSize; j++)
			{
				foreach (var imageId in origImageIds)
				{
					if (!relatedImages.TryGetValue(unchecked((int)imageId), out var images) || j >= images.Count)
					{
						continue;
					}

					// An image not yet in the cluster goes into the overlap set.
					clusterImages.Add(images[j].ImageId);
					if (clusterImages.Count >= maxSize)
					{
						break;
					}
				}
			}

			origImageIds.Clear();
			foreach (var imageId in clusterImages)
			{
				origImageIds.Add(unchecked((uint)imageId));
			}
		}
	}
}
