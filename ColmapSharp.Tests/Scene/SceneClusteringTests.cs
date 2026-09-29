// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SceneClusteringTests: colmap/scene/scene_clustering_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name. Tests ColmapSharp/Scene/SceneClustering.cs.
//
// Tier C: the partition comes from Mathematics/MultilevelPartitioner.cs instead of METIS
// (divergence 77), and every expected membership below is COLMAP's.
// CSharpOnly_FlatChildClustersOrderedBySizeThenSmallestId and
// CSharpOnly_CreateSpreadsTiedOverlapAcrossFrames are extra C#-only checks.
// UnorderedClustersEq compares the clusters as sets, ignoring the order of the clusters and
// of the images in each; here both sides are reduced to a sorted list of "{a,b,...}" strings.

using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class SceneClusteringTests
{
	private sealed record SceneGraph((uint, uint)[] ImagePairs, int[] NumInliers);

	// Image connectivity graph:
	//
	//               100           10
	//        (0) -------- (1) -------- (2)
	//         |                         |
	//        10                        100
	//         |                         |
	//        (5) -------- (4) -------- (3)
	//              100           10
	//
	// Weak cross-cluster connections (1 inlier):
	//        (0) -------- (3)
	//        (2) -------- (5)
	//        (4) -------- (1)
	private static SceneGraph MakeSceneGraphOneLevel() => new(
		[(0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (5, 0), (0, 3), (2, 5), (4, 1)],
		[100, 100, 100, 10, 10, 10, 1, 1, 1]);

	// Image connectivity graph:
	//
	//               100          50          100
	//        (0) -------- (1) -------- (2) -------- (3)
	//         |                                      |
	//        10                                      10
	//         |                                      |
	//        (7) -------- (6) -------- (5) -------- (4)
	//              100          50          100
	//
	// Weak cross-cluster connections (1 inlier):
	//        (0) -------- (4)
	//        (1) -------- (6)
	//        (2) -------- (5)
	//        (3) -------- (7)
	private static SceneGraph MakeSceneGraphTwoLevel() => new(
		[(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (0, 7), (3, 4), (0, 4), (1, 6), (2, 5), (3, 7)],
		[100, 50, 100, 100, 50, 100, 10, 10, 1, 1, 1, 1]);

	// Path graph with three levels of progressively weaker connections.
	private static SceneGraph MakeSceneGraphThreeLevel() => new(
		[
			(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
			(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
		],
		[1000, 100, 1000, 10, 1000, 100, 1000, 1, 1000, 100, 1000, 10, 1000, 100, 1000]);

	// The first six images form one top-level cluster and the last six images form the
	// other. Image 6 overlaps the first cluster. Its strongest individual connection is to
	// image 0, but its combined connection to images 3 and 4 is stronger.
	private static SceneGraph MakeSceneGraphAggregateOverlap() => new(
		[
			(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8),
			(8, 9), (9, 10), (10, 11), (6, 0), (6, 3), (6, 4),
		],
		[10000, 10000, 1, 10000, 10000, 20000, 20000, 20000, 20000, 20000, 100, 90, 90]);

	private static List<string> UnorderedClusters(IEnumerable<IEnumerable<uint>> clusters) =>
		clusters.Select(cluster => "{" + string.Join(",", new SortedSet<uint>(cluster)) + "}")
			.OrderBy(text => text, StringComparer.Ordinal)
			.ToList();

	private static List<string> UnorderedClustersEq(params uint[][] clusters) => UnorderedClusters(clusters);

	private static List<string> GetLeafImageSets(List<SceneClustering.Cluster> leaves) =>
		UnorderedClusters(leaves.Select(leaf => leaf.ImageIds));

	[Test]
	public async Task SceneClustering_Empty()
	{
		var options = new SceneClustering.Options { Branching = 2, ImageOverlap = 0, LeafMaxNumImages = 2 };
		var sceneClustering = new SceneClustering(options);
		await Assert.That(sceneClustering.GetRootCluster()).IsNull();
		sceneClustering.Partition([], []);
		await Assert.That(sceneClustering.GetRootCluster()!.ImageIds.Count).IsEqualTo(0);
		await Assert.That(sceneClustering.GetRootCluster()!.ChildClusters.Count).IsEqualTo(0);
		await Assert.That(sceneClustering.GetLeafClusters().Count).IsEqualTo(1);
	}

	[Test]
	public async Task SceneClustering_OneLevel()
	{
		var options = new SceneClustering.Options { Branching = 2, ImageOverlap = 0, LeafMaxNumImages = 2 };
		var sceneClustering = new SceneClustering(options);
		await Assert.That(sceneClustering.GetRootCluster()).IsNull();
		sceneClustering.Partition([(0, 1)], [10]);
		await Assert.That(sceneClustering.GetRootCluster()!.ImageIds.Count).IsEqualTo(2);
		await Assert.That(sceneClustering.GetRootCluster()!.ImageIds).IsEquivalentTo(new uint[] { 0, 1 });
		await Assert.That(sceneClustering.GetRootCluster()!.ChildClusters.Count).IsEqualTo(0);
		await Assert.That(sceneClustering.GetLeafClusters().Count).IsEqualTo(1);
		await Assert.That(ReferenceEquals(sceneClustering.GetRootCluster(), sceneClustering.GetLeafClusters()[0])).IsTrue();
	}

	[Test]
	public async Task SceneClustering_ThreeFlatClusters()
	{
		var graph = MakeSceneGraphOneLevel();

		var options = new SceneClustering.Options { Branching = 3, ImageOverlap = 0, IsHierarchical = false };
		var sceneClustering = new SceneClustering(options);
		await Assert.That(sceneClustering.GetRootCluster()).IsNull();
		sceneClustering.Partition(graph.ImagePairs, graph.NumInliers);

		await Assert.That(sceneClustering.GetLeafClusters().Count).IsEqualTo(3);
		await Assert.That(GetLeafImageSets(sceneClustering.GetLeafClusters()))
			.IsEquivalentTo(UnorderedClustersEq([0, 1], [2, 3], [4, 5]));
	}

	// C#-only: pins the flat child order of divergence 94 (descending
	// size, then ascending smallest image id), which COLMAP's comparator leaves undefined.
	[Test]
	public async Task CSharpOnly_FlatChildClustersOrderedBySizeThenSmallestId()
	{
		static List<string> FlatChildren((uint, uint)[] imagePairs, int[] numInliers, int branching)
		{
			var options = new SceneClustering.Options { Branching = branching, ImageOverlap = 0, IsHierarchical = false };
			var sceneClustering = new SceneClustering(options);
			sceneClustering.Partition(imagePairs, numInliers);
			return sceneClustering.GetRootCluster()!.ChildClusters
				.Select(child => "{" + string.Join(",", child.ImageIds) + "}")
				.ToList();
		}

		// Equal sizes: ascending smallest image id.
		var graph = MakeSceneGraphOneLevel();
		await Assert.That(FlatChildren(graph.ImagePairs, graph.NumInliers, 3))
			.IsEquivalentTo(new[] { "{0,1}", "{2,3}", "{4,5}" }, CollectionOrdering.Matching);

		// Descending size first, even when the larger child has the larger smallest id (the
		// case where libc++ keeps COLMAP's order unchanged).
		(uint, uint)[] pairAndTriangle = [(0, 1), (2, 3), (3, 4), (2, 4)];
		await Assert.That(FlatChildren(pairAndTriangle, [5, 5, 5, 5], 2))
			.IsEquivalentTo(new[] { "{2,3,4}", "{0,1}" }, CollectionOrdering.Matching);

		// More parts than images: the empty children come last.
		(uint, uint)[] path = [(0, 1), (1, 2)];
		await Assert.That(FlatChildren(path, [1, 1], 5))
			.IsEquivalentTo(new[] { "{0}", "{1}", "{2}", "{}", "{}" }, CollectionOrdering.Matching);
	}

	[Test]
	public async Task SceneClustering_ThreeFlatClustersTwoOverlap()
	{
		var graph = MakeSceneGraphOneLevel();

		var options = new SceneClustering.Options { Branching = 3, ImageOverlap = 2, IsHierarchical = false };
		var sceneClustering = new SceneClustering(options);
		await Assert.That(sceneClustering.GetRootCluster()).IsNull();
		sceneClustering.Partition(graph.ImagePairs, graph.NumInliers);

		await Assert.That(sceneClustering.GetLeafClusters().Count).IsEqualTo(3);
		await Assert.That(GetLeafImageSets(sceneClustering.GetLeafClusters()))
			.IsEquivalentTo(UnorderedClustersEq([0, 1, 2, 5], [1, 2, 3, 4], [0, 3, 4, 5]));
	}

	[Test]
	public async Task SceneClustering_HierarchicalTwoLevelsNoOverlap()
	{
		var graph = MakeSceneGraphTwoLevel();

		var options = new SceneClustering.Options { Branching = 2, ImageOverlap = 0, LeafMaxNumImages = 2 };
		var sceneClustering = new SceneClustering(options);
		await Assert.That(sceneClustering.GetRootCluster()).IsNull();
		sceneClustering.Partition(graph.ImagePairs, graph.NumInliers);

		await Assert.That(sceneClustering.GetLeafClusters().Count).IsEqualTo(4);
		await Assert.That(GetLeafImageSets(sceneClustering.GetLeafClusters()))
			.IsEquivalentTo(UnorderedClustersEq([0, 1], [2, 3], [4, 5], [6, 7]));
	}

	[Test]
	public async Task SceneClustering_HierarchicalTwoLevelsWithOverlap()
	{
		var graph = MakeSceneGraphTwoLevel();

		var options = new SceneClustering.Options { Branching = 2, ImageOverlap = 2, LeafMaxNumImages = 3 };
		var sceneClustering = new SceneClustering(options);
		await Assert.That(sceneClustering.GetRootCluster()).IsNull();
		sceneClustering.Partition(graph.ImagePairs, graph.NumInliers);

		await Assert.That(sceneClustering.GetLeafClusters().Count).IsEqualTo(4);
		await Assert.That(GetLeafImageSets(sceneClustering.GetLeafClusters()))
			.IsEquivalentTo(UnorderedClustersEq([0, 1, 2, 3, 4], [0, 1, 2, 4, 7], [0, 3, 4, 5, 6], [0, 4, 5, 6, 7]));
	}

	[Test]
	public async Task SceneClustering_HierarchicalThreeLevelsWithOverlap()
	{
		var graph = MakeSceneGraphThreeLevel();

		var options = new SceneClustering.Options { Branching = 2, ImageOverlap = 1, LeafMaxNumImages = 3 };
		var sceneClustering = new SceneClustering(options);
		sceneClustering.Partition(graph.ImagePairs, graph.NumInliers);

		// Top-level overlap images are partitioned twice after they are inherited,
		// exercising connectivity beyond the single generation covered above.
		await Assert.That(sceneClustering.GetLeafClusters().Count).IsEqualTo(8);
		await Assert.That(GetLeafImageSets(sceneClustering.GetLeafClusters()))
			.IsEquivalentTo(UnorderedClustersEq(
				[0, 1, 2],
				[1, 2, 3, 4],
				[3, 4, 5, 6],
				[5, 6, 7, 8],
				[7, 8, 9, 10],
				[9, 10, 11, 12],
				[11, 12, 13, 14],
				[13, 14, 15]));
	}

	[Test]
	public async Task SceneClustering_HierarchicalOverlapUsesAllConnections()
	{
		var graph = MakeSceneGraphAggregateOverlap();

		var options = new SceneClustering.Options { Branching = 2, ImageOverlap = 1, LeafMaxNumImages = 4 };
		var sceneClustering = new SceneClustering(options);
		sceneClustering.Partition(graph.ImagePairs, graph.NumInliers);

		SceneClustering.Cluster? targetCluster = null;
		foreach (var child in sceneClustering.GetRootCluster()!.ChildClusters)
		{
			var imageIds = new HashSet<uint>(child.ImageIds);
			if (imageIds.IsSupersetOf(new uint[] { 0, 1, 2, 3, 4, 5 }))
			{
				targetCluster = child;
				break;
			}
		}

		await Assert.That(targetCluster).IsNotNull();
		var childImageSets = targetCluster!.ChildClusters.Select(child => new HashSet<uint>(child.ImageIds)).ToList();
		await Assert.That(childImageSets.Count).IsEqualTo(2);

		// Image 6 does not simply follow its strongest connection to image 0. Its full
		// connectivity to images 3 and 4 participates in the graph cut. The exact partition
		// is implementation-dependent in METIS.
		await Assert.That(childImageSets.Any(imageIds => imageIds.Contains(3) && imageIds.Contains(4) && imageIds.Contains(6))).IsTrue();
	}

	// C#-only (divergence 120). In a synthetic dataset every image pair
	// has the same number of matches, so every clustering tie falls to the order in which
	// Create hands over the pairs. Taken in ascending pair id, the root split's overlap was
	// images 1-3 and 31-33: one frame per side. For a panoramic rig (zero sensor
	// translation) every image of a frame has the same projection center, so the two halves
	// shared only two distinct centers and could never be aligned, and the hierarchical
	// pipeline ended with two reconstructions
	// (HierarchicalPipeline_WithoutNoiseAndPanoramicNonTrivialFrames). A similarity needs
	// at least three distinct centers.
	[Test]
	public async Task CSharpOnly_CreateSpreadsTiedOverlapAcrossFrames()
	{
		using var database = new InMemoryDatabase();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 3,
				NumFramesPerRig = 10,
				NumPoints3D = 100,
				SensorFromRigTranslationStddev = 0,
				SensorFromRigRotationStddev = 30,
			},
			new Reconstruction(),
			database);
		var databaseCache = DatabaseCache.Create(database, new DatabaseCache.Options());

		var options = new SceneClustering.Options { LeafMaxNumImages = 10, ImageOverlap = 3 };
		var sceneClustering = SceneClustering.Create(options, databaseCache);

		var children = sceneClustering.GetRootCluster()!.ChildClusters;
		var sharedFrameIds = children[0].ImageIds
			.Intersect(children[1].ImageIds)
			.Select(imageId => databaseCache.Images[imageId].FrameId)
			.Distinct()
			.Count();

		await Assert.That(children.Count).IsEqualTo(2);
		await Assert.That(sharedFrameIds).IsGreaterThanOrEqualTo(3);
	}
}
