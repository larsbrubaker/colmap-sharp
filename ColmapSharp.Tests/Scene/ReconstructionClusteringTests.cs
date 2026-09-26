// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionClusteringTests: colmap/scene/reconstruction_clustering_test.cc ported 1:1,
// one method per gtest TEST(Suite, Name) named Suite_Name. Tests
// ColmapSharp/Scene/ReconstructionClustering.cs.
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test; each test seeds before it
// synthesizes. PartitionFramesIntoClusters draws one uniform number per cross-cluster
// observation while walking points and clusters in hash-map order; here points are walked in
// ascending id (Reconstruction.Points3D) and clusters in ascending id, so which observations
// survive differs from COLMAP's run, but only the statistics the tests rely on (a keep ratio
// of 0.0, 0.05 or 0.1 leaves the clusters weakly connected) are the same.

using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class ReconstructionClusteringTests
{
	// Partitions frames into clusters based on the assigned cluster ids. Each cluster will
	// only share 3D points among its member frames, and observations connecting frames from
	// different clusters are removed, except a keepRatio fraction of them. This effectively
	// creates isolated connected components in the covisibility graph.
	private static void PartitionFramesIntoClusters(
		Reconstruction reconstruction,
		Dictionary<uint, int> frameToCluster,
		double keepRatio = 0.1)
	{
		// Collect all observations to delete first, because deleting while iterating is unsafe.
		var observationsToDelete = new List<(uint ImageId, uint Point2DIdx)>();

		foreach (var (point3DId, point3D) in reconstruction.Points3D)
		{
			// Determine which clusters observe this 3D point.
			var clusterObservations = new SortedDictionary<int, List<TrackElement>>();
			foreach (var trackEl in point3D.Track.Elements)
			{
				var frameId = reconstruction.Image(trackEl.ImageId).FrameId;
				if (frameToCluster.TryGetValue(frameId, out var clusterId))
				{
					if (!clusterObservations.TryGetValue(clusterId, out var observations))
					{
						observations = [];
						clusterObservations.Add(clusterId, observations);
					}

					observations.Add(trackEl);
				}
			}

			// If multiple clusters observe this point, pick one cluster to keep and remove
			// observations from all other clusters, keeping a keepRatio fraction of them.
			if (clusterObservations.Count > 1)
			{
				var clusterIdsVec = clusterObservations.Keys.ToList();
				var chosenCluster = clusterIdsVec[(int)(point3DId % (ulong)clusterIdsVec.Count)];

				foreach (var (clusterId, observations) in clusterObservations)
				{
					if (clusterId == chosenCluster)
					{
						continue;
					}

					foreach (var observation in observations)
					{
						var randomValue = RandomUtils.RandomUniformReal(0.0, 1.0);
						if (randomValue >= keepRatio)
						{
							observationsToDelete.Add((observation.ImageId, observation.Point2DIdx));
						}
					}
				}
			}
		}

		// Delete the collected observations. An observation may already be gone if its 3D
		// point was removed because its track became too short.
		foreach (var (imageId, point2DIdx) in observationsToDelete)
		{
			var image = reconstruction.Image(imageId);
			if (point2DIdx < image.NumPoints2D && image.Point2DAt(point2DIdx).HasPoint3D)
			{
				reconstruction.DeleteObservation(imageId, point2DIdx);
			}
		}
	}

	// All registered frame ids, sorted for deterministic test behavior.
	private static List<uint> ExtractSortedFrameIds(Reconstruction reconstruction) =>
		reconstruction.Frames.Where(entry => entry.Value.HasPose).Select(entry => entry.Key).Order().ToList();

	// result[clusterId] holds the frame ids in that cluster.
	private static List<HashSet<uint>> BuildClustersFromOutput(Dictionary<uint, int> clusterIds)
	{
		var clusters = new List<HashSet<uint>>();
		foreach (var (frameId, clusterId) in clusterIds)
		{
			if (clusterId == -1)
			{
				continue;
			}

			while (clusters.Count <= clusterId)
			{
				clusters.Add([]);
			}

			clusters[clusterId].Add(frameId);
		}

		return clusters;
	}

	private static Reconstruction Synthesize(int numCamerasPerRig, int numFramesPerRig, int numPoints3D)
	{
		RandomUtils.SetPRNGSeed(0);
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = numCamerasPerRig,
			NumFramesPerRig = numFramesPerRig,
			NumPoints3D = numPoints3D,
			NumPoints2DWithoutPoint3D = 0,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);
		return reconstruction;
	}

	// Assigns the sorted frame ids to consecutive clusters of the given sizes and returns
	// the expected clusters sorted by descending size (stable, as the sizes here differ).
	private static List<HashSet<uint>> AssignClusters(List<uint> allFrameIds, Dictionary<uint, int> frameToCluster, params int[] clusterSizes)
	{
		var expectedClusters = new List<HashSet<uint>>();
		var index = 0;
		for (var clusterId = 0; clusterId < clusterSizes.Length; clusterId++)
		{
			expectedClusters.Add([]);
			for (var i = 0; i < clusterSizes[clusterId]; i++, index++)
			{
				frameToCluster[allFrameIds[index]] = clusterId;
				expectedClusters[clusterId].Add(allFrameIds[index]);
			}
		}

		return expectedClusters.OrderByDescending(cluster => cluster.Count).ToList();
	}

	private static async Task AssertClustersEqual(List<HashSet<uint>> resultClusters, List<HashSet<uint>> expectedClusters)
	{
		await Assert.That(resultClusters.Count).IsEqualTo(3);
		for (var i = 0; i < 3; i++)
		{
			await Assert.That(resultClusters[i].SetEquals(expectedClusters[i])).IsTrue();
		}
	}

	[Test]
	public async Task ClusterReconstructionFrames_Empty()
	{
		var reconstruction = new Reconstruction();
		var options = new ReconstructionClusteringOptions();
		var clusterIds = ReconstructionClustering.ClusterReconstructionFrames(options, reconstruction);
		await Assert.That(clusterIds).IsEmpty();
	}

	[Test]
	public async Task ClusterReconstructionFrames_WellConnectedReconstruction()
	{
		var reconstruction = Synthesize(numCamerasPerRig: 1, numFramesPerRig: 5, numPoints3D: 100);

		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(5);

		var options = new ReconstructionClusteringOptions();
		var clusterIds = ReconstructionClustering.ClusterReconstructionFrames(options, reconstruction);

		// All frames should be assigned cluster 0.
		foreach (var clusterId in clusterIds.Values)
		{
			await Assert.That(clusterId).IsEqualTo(0);
		}
	}

	[Test]
	public async Task ClusterReconstructionFrames_WeaklyConnectedReconstruction()
	{
		// A reconstruction with very few 3D points (10 total). With so few shared
		// observations, the covisibility between frames is weak and no frame should end up
		// in a kept cluster.
		const int NumPoints3D = 10;
		var reconstruction = Synthesize(numCamerasPerRig: 1, numFramesPerRig: 10, numPoints3D: NumPoints3D);

		var numFrames = reconstruction.NumRegFrames;
		await Assert.That(numFrames).IsEqualTo(10);

		var options = new ReconstructionClusteringOptions { MinEdgeWeightThreshold = NumPoints3D + 1 };
		var clusterIds = ReconstructionClustering.ClusterReconstructionFrames(options, reconstruction);

		// All frames should be assigned to clusters.
		await Assert.That(clusterIds.Count).IsEqualTo(numFrames);

		// Each frame should get id -1.
		foreach (var clusterId in clusterIds.Values)
		{
			await Assert.That(clusterId).IsEqualTo(-1);
		}
	}

	[Test]
	public async Task ClusterReconstructionFrames_OneMajorConnectedComponent()
	{
		// 10 frames, all initially well-connected.
		var reconstruction = Synthesize(numCamerasPerRig: 1, numFramesPerRig: 10, numPoints3D: 250);

		var initialNumRegFrames = reconstruction.NumRegFrames;
		await Assert.That(initialNumRegFrames).IsEqualTo(10);

		var allFrameIds = ExtractSortedFrameIds(reconstruction);

		// Partition into one cluster and independent frames: first 8 frames in cluster 0,
		// other frames are independent.
		var frameToCluster = new Dictionary<uint, int>();
		const int LargeClusterSize = 8;
		for (var i = 0; i < LargeClusterSize; i++)
		{
			frameToCluster[allFrameIds[i]] = 0;
		}

		for (var i = LargeClusterSize; i < 10; i++)
		{
			frameToCluster[allFrameIds[i]] = i - LargeClusterSize + 1;
		}

		// Partition the reconstruction to disconnect the clusters. With keepRatio = 0.1,
		// some weak connections may remain.
		PartitionFramesIntoClusters(reconstruction, frameToCluster, 0.1);

		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(initialNumRegFrames);

		var options = new ReconstructionClusteringOptions();
		var clusterIds = ReconstructionClustering.ClusterReconstructionFrames(options, reconstruction);

		// All frames should be assigned to clusters.
		await Assert.That(clusterIds.Count).IsEqualTo(initialNumRegFrames);

		var clusters = BuildClustersFromOutput(clusterIds);

		// Find the largest cluster.
		var largestClusterIdx = 0;
		for (var i = 1; i < clusters.Count; i++)
		{
			if (clusters[i].Count > clusters[largestClusterIdx].Count)
			{
				largestClusterIdx = i;
			}
		}

		// The largest cluster should be cluster 0.
		await Assert.That(largestClusterIdx).IsEqualTo(0);

		// The largest cluster should have exactly LargeClusterSize frames.
		await Assert.That(clusters[largestClusterIdx].Count).IsEqualTo(LargeClusterSize);

		// Other clusters should be single-frame clusters.
		for (var i = 0; i < clusters.Count; i++)
		{
			if (i != largestClusterIdx)
			{
				await Assert.That(clusters[i].Count).IsEqualTo(1);
			}
		}
	}

	[Test]
	public async Task ClusterReconstructionFrames_MultipleWeaklyConnectedClusters()
	{
		// Frames partitioned into weakly connected clusters (some cross-cluster connections
		// remain).
		const int Cluster0Size = 25;
		const int Cluster1Size = 5;
		const int Cluster2Size = 4;
		const int TotalFrames = Cluster0Size + Cluster1Size + Cluster2Size;

		var reconstruction = Synthesize(numCamerasPerRig: 1, numFramesPerRig: TotalFrames, numPoints3D: 400);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(TotalFrames);

		var allFrameIds = ExtractSortedFrameIds(reconstruction);
		var frameToCluster = new Dictionary<uint, int>();
		var expectedClusters = AssignClusters(allFrameIds, frameToCluster, Cluster0Size, Cluster1Size, Cluster2Size);

		// Partition with keepRatio = 0.1 to leave some weak connections.
		PartitionFramesIntoClusters(reconstruction, frameToCluster, 0.1);

		// All frames should still be registered.
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(TotalFrames);

		var options = new ReconstructionClusteringOptions();
		var clusterIds = ReconstructionClustering.ClusterReconstructionFrames(options, reconstruction);

		// All frames should be assigned to clusters.
		await Assert.That(clusterIds.Count).IsEqualTo(TotalFrames);

		// Result clusters are already sorted by size due to the implementation.
		await AssertClustersEqual(BuildClustersFromOutput(clusterIds), expectedClusters);
	}

	[Test]
	public async Task ClusterReconstructionFrames_MultipleDisjointClusters()
	{
		// Completely disjoint clusters (no shared observations between clusters). The
		// clustering should exactly match the original partition.
		const int Cluster0Size = 10;
		const int Cluster1Size = 8;
		const int Cluster2Size = 6;
		const int TotalFrames = Cluster0Size + Cluster1Size + Cluster2Size;

		var reconstruction = Synthesize(numCamerasPerRig: 1, numFramesPerRig: TotalFrames, numPoints3D: 500);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(TotalFrames);

		var allFrameIds = ExtractSortedFrameIds(reconstruction);
		var frameToCluster = new Dictionary<uint, int>();
		var expectedClusters = AssignClusters(allFrameIds, frameToCluster, Cluster0Size, Cluster1Size, Cluster2Size);

		// Use keepRatio = 0.0 to create completely disjoint clusters with no shared
		// observations between them.
		PartitionFramesIntoClusters(reconstruction, frameToCluster, 0.0);

		// All frames should still be registered.
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(TotalFrames);

		var options = new ReconstructionClusteringOptions();
		var clusterIds = ReconstructionClustering.ClusterReconstructionFrames(options, reconstruction);

		// All frames should be assigned to clusters.
		await Assert.That(clusterIds.Count).IsEqualTo(TotalFrames);

		await AssertClustersEqual(BuildClustersFromOutput(clusterIds), expectedClusters);
	}

	// Tests with non-trivial rigs (multiple cameras per frame).

	[Test]
	public async Task ClusterReconstructionFrames_RigOneMajorConnectedComponent()
	{
		// 10 frames from a rig with 3 cameras each. With multi-camera rigs, covisibility
		// between frames is stronger because 3D points are often visible from multiple
		// cameras in the same frame.
		var reconstruction = Synthesize(numCamerasPerRig: 3, numFramesPerRig: 10, numPoints3D: 300);

		var initialNumRegFrames = reconstruction.NumRegFrames;
		await Assert.That(initialNumRegFrames).IsEqualTo(10);
		// Should have 30 images (10 frames * 3 cameras).
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(30);

		var allFrameIds = ExtractSortedFrameIds(reconstruction);

		// Partition into one large cluster and independent frames: first 7 frames in cluster
		// 0, other 3 frames are independent.
		var frameToCluster = new Dictionary<uint, int>();
		var expectedLargeCluster = new HashSet<uint>();
		const int LargeClusterSize = 7;
		for (var i = 0; i < LargeClusterSize; i++)
		{
			frameToCluster[allFrameIds[i]] = 0;
			expectedLargeCluster.Add(allFrameIds[i]);
		}

		for (var i = LargeClusterSize; i < 10; i++)
		{
			frameToCluster[allFrameIds[i]] = i - LargeClusterSize + 1;
		}

		// Use keepRatio = 0.0 to completely disconnect clusters. With multi-camera rigs,
		// weak connections are harder to break.
		PartitionFramesIntoClusters(reconstruction, frameToCluster, 0.0);

		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(initialNumRegFrames);

		var options = new ReconstructionClusteringOptions();
		var clusterIds = ReconstructionClustering.ClusterReconstructionFrames(options, reconstruction);

		// All frames should be assigned to clusters.
		await Assert.That(clusterIds.Count).IsEqualTo(initialNumRegFrames);

		var clusters = BuildClustersFromOutput(clusterIds);

		// Should be only 1 large cluster.
		await Assert.That(clusters.Count).IsEqualTo(1);
		// The largest cluster (cluster 0) should have exactly LargeClusterSize frames.
		await Assert.That(clusters[0].Count).IsEqualTo(LargeClusterSize);
		await Assert.That(clusters[0].SetEquals(expectedLargeCluster)).IsTrue();

		for (var i = LargeClusterSize; i < 10; i++)
		{
			// Other frames should not be in any cluster.
			await Assert.That(clusterIds[allFrameIds[i]]).IsEqualTo(-1);
		}
	}

	[Test]
	public async Task ClusterReconstructionFrames_RigMultipleWeaklyConnectedClusters()
	{
		// Frames from a rig with 2 cameras each, partitioned into 3 clusters. With
		// multi-camera rigs the covisibility between frames is higher, so a very low
		// keepRatio ensures cluster separation while still leaving some weak connections.
		const int Cluster0Size = 30;
		const int Cluster1Size = 4;
		const int Cluster2Size = 3;
		const int TotalFrames = Cluster0Size + Cluster1Size + Cluster2Size;

		var reconstruction = Synthesize(numCamerasPerRig: 2, numFramesPerRig: TotalFrames, numPoints3D: 500);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(TotalFrames);
		// Should have 74 images (37 frames * 2 cameras).
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(TotalFrames * 2);

		var allFrameIds = ExtractSortedFrameIds(reconstruction);
		var frameToCluster = new Dictionary<uint, int>();
		var expectedClusters = AssignClusters(allFrameIds, frameToCluster, Cluster0Size, Cluster1Size, Cluster2Size);

		// Use keepRatio = 0.05 to create nearly disjoint clusters.
		PartitionFramesIntoClusters(reconstruction, frameToCluster, 0.05);

		// All frames should still be registered.
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(TotalFrames);

		var options = new ReconstructionClusteringOptions();
		var clusterIds = ReconstructionClustering.ClusterReconstructionFrames(options, reconstruction);

		// All frames should be assigned to clusters.
		await Assert.That(clusterIds.Count).IsEqualTo(TotalFrames);

		await AssertClustersEqual(BuildClustersFromOutput(clusterIds), expectedClusters);
	}

	[Test]
	public async Task ClusterReconstructionFrames_RigMultipleDisjointClusters()
	{
		// Frames from a rig with 4 cameras each, partitioned into 3 completely disjoint
		// clusters.
		const int Cluster0Size = 8;
		const int Cluster1Size = 6;
		const int Cluster2Size = 4;
		const int TotalFrames = Cluster0Size + Cluster1Size + Cluster2Size;
		const int CamerasPerRig = 4;

		var reconstruction = Synthesize(numCamerasPerRig: CamerasPerRig, numFramesPerRig: TotalFrames, numPoints3D: 600);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(TotalFrames);
		// Should have 72 images (18 frames * 4 cameras).
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(TotalFrames * CamerasPerRig);

		var allFrameIds = ExtractSortedFrameIds(reconstruction);
		var frameToCluster = new Dictionary<uint, int>();
		var expectedClusters = AssignClusters(allFrameIds, frameToCluster, Cluster0Size, Cluster1Size, Cluster2Size);

		// Use keepRatio = 0.0 to create completely disjoint clusters.
		PartitionFramesIntoClusters(reconstruction, frameToCluster, 0.0);

		// All frames should still be registered.
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(TotalFrames);

		var options = new ReconstructionClusteringOptions();
		var clusterIds = ReconstructionClustering.ClusterReconstructionFrames(options, reconstruction);

		// All frames should be assigned to clusters.
		await Assert.That(clusterIds.Count).IsEqualTo(TotalFrames);

		await AssertClustersEqual(BuildClustersFromOutput(clusterIds), expectedClusters);
	}
}
