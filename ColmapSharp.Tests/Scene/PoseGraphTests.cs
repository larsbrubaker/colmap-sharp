// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PoseGraphTests: colmap/scene/pose_graph_test.cc ported 1:1, one method per gtest case
// named Suite_Name, testing ColmapSharp/Scene/PoseGraph.cs. Load runs against
// InMemoryDatabase (the database is not an SQLite file here). std::out_of_range is
// KeyNotFoundException and std::runtime_error InvalidOperationException. PrngTestIsolation
// seeds the PRNG with 0 before every test, as COLMAP's gtest_main does; tests that draw do
// so before their first await (the PRNG is thread-local). CSharpOnly_* cases are labeled
// as such.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Scene;

public class PoseGraphTests
{
	private static PoseGraph.Edge SynthesizeEdge(int numMatches = 50) =>
		new() { Cam2FromCam1 = new Rigid3d(), NumMatches = numMatches };

	private static PoseGraph.Edge TranslatedEdge(double x) =>
		new() { Cam2FromCam1 = new Rigid3d(Quaterniond.Identity, new Vector3d(x, 0, 0)), NumMatches = 50 };

	private static Reconstruction SingleFrameRigs(int numRigs)
	{
		var syntheticOptions = new SyntheticDatasetOptions
		{
			NumRigs = numRigs,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
			NumPoints3D = 10,
		};
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(syntheticOptions, reconstruction);
		return reconstruction;
	}

	[Test]
	public async Task PoseGraph_Nominal()
	{
		var poseGraph = new PoseGraph();

		// Empty view graph.
		await Assert.That(poseGraph.Empty).IsTrue();
		await Assert.That(poseGraph.NumEdges).IsEqualTo(0);

		// Add some pairs.
		poseGraph.AddEdge(1, 2, SynthesizeEdge());
		poseGraph.AddEdge(1, 3, SynthesizeEdge());
		poseGraph.AddEdge(2, 3, SynthesizeEdge());

		await Assert.That(poseGraph.Empty).IsFalse();
		await Assert.That(poseGraph.NumEdges).IsEqualTo(3);

		// Invalidate one pair.
		poseGraph.SetInvalidEdge(ImagePairToPairId(1, 2));
		await Assert.That(poseGraph.NumEdges).IsEqualTo(3);
		await Assert.That(poseGraph.IsValid(ImagePairToPairId(1, 2))).IsFalse();

		// Clear the view graph.
		poseGraph.Clear();
		await Assert.That(poseGraph.Empty).IsTrue();
		await Assert.That(poseGraph.NumEdges).IsEqualTo(0);
	}

	[Test]
	public async Task PoseGraph_AddEdge()
	{
		var poseGraph = new PoseGraph();

		// Normal add.
		poseGraph.AddEdge(1, 2, TranslatedEdge(1));

		await Assert.That(poseGraph.NumEdges).IsEqualTo(1);
		(PoseGraph.Edge stored, bool swapped) = poseGraph.EdgeRef(1, 2);
		await Assert.That(swapped).IsFalse();
		await Assert.That(stored.Cam2FromCam1.Translation.X).IsEqualTo(1);

		// Add with swapped IDs should invert the pair.
		poseGraph.AddEdge(4, 3, TranslatedEdge(2)); // 4 > 3, should swap and invert

		await Assert.That(poseGraph.NumEdges).IsEqualTo(2);
		(PoseGraph.Edge stored2, bool swapped2) = poseGraph.EdgeRef(3, 4);
		await Assert.That(swapped2).IsFalse();
		await Assert.That(stored2.Cam2FromCam1.Translation.X).IsEqualTo(-2);

		// Duplicate should throw.
		await Assert.That(() => poseGraph.AddEdge(1, 2, SynthesizeEdge())).Throws<InvalidOperationException>();
		await Assert.That(() => poseGraph.AddEdge(2, 1, SynthesizeEdge())).Throws<InvalidOperationException>();
	}

	[Test]
	public async Task PoseGraph_HasEdge()
	{
		var poseGraph = new PoseGraph();
		poseGraph.AddEdge(1, 2, SynthesizeEdge());

		await Assert.That(poseGraph.HasEdge(1, 2)).IsTrue();
		await Assert.That(poseGraph.HasEdge(2, 1)).IsTrue(); // Order doesn't matter
		await Assert.That(poseGraph.HasEdge(1, 3)).IsFalse();
	}

	[Test]
	public async Task PoseGraph_EdgeRef()
	{
		var poseGraph = new PoseGraph();
		poseGraph.AddEdge(1, 2, TranslatedEdge(1));

		// Normal order: swapped = false.
		(PoseGraph.Edge ref1, bool swapped1) = poseGraph.EdgeRef(1, 2);
		await Assert.That(swapped1).IsFalse();
		await Assert.That(ref1.Cam2FromCam1.Translation.X).IsEqualTo(1);

		// Reversed order: swapped = true.
		(PoseGraph.Edge ref2, bool swapped2) = poseGraph.EdgeRef(2, 1);
		await Assert.That(swapped2).IsTrue();
		await Assert.That(ref2.Cam2FromCam1.Translation.X).IsEqualTo(1); // Same reference

		// Modify validity through PoseGraph.
		poseGraph.SetInvalidEdge(ImagePairToPairId(1, 2));
		await Assert.That(poseGraph.IsValid(ImagePairToPairId(1, 2))).IsFalse();

		// Non-existent pair should throw.
		await Assert.That(() => poseGraph.EdgeRef(1, 3)).Throws<KeyNotFoundException>();
	}

	[Test]
	public async Task PoseGraph_GetEdge()
	{
		var poseGraph = new PoseGraph();
		poseGraph.AddEdge(1, 2, TranslatedEdge(1));

		// Normal order: returns as-is.
		PoseGraph.Edge copy1 = poseGraph.GetEdge(1, 2);
		await Assert.That(copy1.Cam2FromCam1.Translation.X).IsEqualTo(1);

		// Reversed order: returns inverted copy.
		PoseGraph.Edge copy2 = poseGraph.GetEdge(2, 1);
		await Assert.That(copy2.Cam2FromCam1.Translation.X).IsEqualTo(-1);

		// Original unchanged.
		await Assert.That(poseGraph.EdgeRef(1, 2).Edge.Cam2FromCam1.Translation.X).IsEqualTo(1);

		// Non-existent pair should throw.
		await Assert.That(() => poseGraph.GetEdge(1, 3)).Throws<KeyNotFoundException>();
	}

	[Test]
	public async Task PoseGraph_DeleteEdge()
	{
		var poseGraph = new PoseGraph();
		poseGraph.AddEdge(1, 2, SynthesizeEdge());
		poseGraph.AddEdge(1, 3, SynthesizeEdge());

		await Assert.That(poseGraph.DeleteEdge(1, 2)).IsTrue();
		await Assert.That(poseGraph.HasEdge(1, 2)).IsFalse();
		await Assert.That(poseGraph.NumEdges).IsEqualTo(1);

		// Delete with reversed order.
		await Assert.That(poseGraph.DeleteEdge(3, 1)).IsTrue();
		await Assert.That(poseGraph.NumEdges).IsEqualTo(0);

		// Delete non-existent returns false.
		await Assert.That(poseGraph.DeleteEdge(1, 2)).IsFalse();
	}

	[Test]
	public async Task PoseGraph_UpdateEdge()
	{
		var poseGraph = new PoseGraph();
		poseGraph.AddEdge(1, 2, TranslatedEdge(1));

		// Update with normal order.
		poseGraph.UpdateEdge(1, 2, TranslatedEdge(5));

		await Assert.That(poseGraph.EdgeRef(1, 2).Edge.Cam2FromCam1.Translation.X).IsEqualTo(5);

		// Update with reversed order should invert.
		poseGraph.UpdateEdge(2, 1, TranslatedEdge(3));

		await Assert.That(poseGraph.EdgeRef(1, 2).Edge.Cam2FromCam1.Translation.X).IsEqualTo(-3);

		// Update non-existent should throw.
		await Assert.That(() => poseGraph.UpdateEdge(1, 3, SynthesizeEdge())).Throws<InvalidOperationException>();
	}

	[Test]
	public async Task PoseGraph_ValidEdges()
	{
		var poseGraph = new PoseGraph();

		ulong pairId1 = ImagePairToPairId(1, 2);
		ulong pairId2 = ImagePairToPairId(1, 3);
		ulong pairId3 = ImagePairToPairId(2, 3);
		poseGraph.AddEdge(1, 2, SynthesizeEdge());
		poseGraph.AddEdge(1, 3, SynthesizeEdge());
		poseGraph.AddEdge(2, 3, SynthesizeEdge());

		List<ulong> GetValidPairIds() => [.. poseGraph.ValidEdges().Select(kv => kv.Key)];

		// All pairs start valid.
		await Assert.That(GetValidPairIds()).IsEquivalentTo([pairId1, pairId2, pairId3]);

		// Invalidate one pair.
		poseGraph.SetInvalidEdge(pairId2);
		await Assert.That(GetValidPairIds()).IsEquivalentTo([pairId1, pairId3]);

		// Re-validate the pair.
		poseGraph.SetValidEdge(pairId2);
		await Assert.That(GetValidPairIds()).IsEquivalentTo([pairId1, pairId2, pairId3]);
	}

	[Test]
	public async Task PoseGraph_Load()
	{
		using var database = new InMemoryDatabase();

		Camera camera = Camera.CreateFromModelId(InvalidCameraId, CameraModelId.SimplePinhole, 1, 1, 1);
		uint cameraId = database.WriteCamera(camera);

		// Create images.
		for (int i = 1; i <= 3; ++i)
		{
			var image = new Image { Name = "image" + i };
			image.SetCameraId(cameraId);
			database.WriteImage(image);
		}

		var twoView = new TwoViewGeometry
		{
			Config = TwoViewGeometry.ConfigurationType.Calibrated,
			InlierMatches = [new FeatureMatch(0, 0), new FeatureMatch(1, 1)],
			Cam2FromCam1 = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d().Normalized()),
		};

		// Create pairs (1,2) and (2,3)
		database.WriteMatches(1, 2, Enumerable.Repeat(new FeatureMatch(), 10).ToList());
		database.WriteMatches(2, 3, Enumerable.Repeat(new FeatureMatch(), 10).ToList());
		database.WriteTwoViewGeometry(1, 2, twoView);
		database.WriteTwoViewGeometry(2, 3, twoView);

		// Load into DatabaseCache with relative poses.
		var cache = new DatabaseCache();
		cache.Load(database, new DatabaseCache.Options());

		var poseGraph = new PoseGraph();
		poseGraph.Load(cache.CorrespondenceGraph);

		await Assert.That(poseGraph.NumEdges).IsEqualTo(2);
		await Assert.That(poseGraph.HasEdge(1, 2)).IsTrue();
		await Assert.That(poseGraph.HasEdge(2, 3)).IsTrue();
	}

	[Test]
	public async Task PoseGraph_LargestConnectedFrameComponentEmpty()
	{
		Reconstruction reconstruction = SingleFrameRigs(3);

		// Pose graph with no edges
		var poseGraph = new PoseGraph();
		HashSet<uint> result = poseGraph.LargestConnectedFrameComponent(reconstruction, filterUnregistered: false);
		await Assert.That(result).IsEmpty();
	}

	[Test]
	public async Task PoseGraph_ConnectedFrameComponents()
	{
		// Five single-frame rigs.
		Reconstruction reconstruction = SingleFrameRigs(5);

		List<uint> regImageIds = reconstruction.RegImageIds();
		await Assert.That(regImageIds.Count).IsEqualTo(5);

		// Component A: images 0, 1, 2 ; Component B: images 3, 4.
		var poseGraph = new PoseGraph();
		poseGraph.AddEdge(regImageIds[0], regImageIds[1], SynthesizeEdge());
		poseGraph.AddEdge(regImageIds[1], regImageIds[2], SynthesizeEdge());
		poseGraph.AddEdge(regImageIds[3], regImageIds[4], SynthesizeEdge());

		uint frame0 = reconstruction.Image(regImageIds[0]).FrameId;
		uint frame1 = reconstruction.Image(regImageIds[1]).FrameId;
		uint frame2 = reconstruction.Image(regImageIds[2]).FrameId;
		uint frame3 = reconstruction.Image(regImageIds[3]).FrameId;
		uint frame4 = reconstruction.Image(regImageIds[4]).FrameId;

		List<HashSet<uint>> components = poseGraph.ConnectedFrameComponents(reconstruction);

		await Assert.That(components.Count).IsEqualTo(2);
		// Components are sorted by descending size.
		await Assert.That(components[0].Count).IsEqualTo(3);
		await Assert.That(components[1].Count).IsEqualTo(2);
		await Assert.That(components[0]).IsEquivalentTo([frame0, frame1, frame2]);
		await Assert.That(components[1]).IsEquivalentTo([frame3, frame4]);

		// The largest component must equal the first (largest) entry.
		await Assert.That(poseGraph.LargestConnectedFrameComponent(reconstruction).SetEquals(components[0])).IsTrue();
	}

	[Test]
	public async Task PoseGraph_ConnectedFrameComponentsEmpty()
	{
		Reconstruction reconstruction = SingleFrameRigs(3);

		// Pose graph with no edges yields no components.
		var poseGraph = new PoseGraph();
		await Assert.That(poseGraph.ConnectedFrameComponents(reconstruction, filterUnregistered: false)).IsEmpty();
	}

	[Test]
	public async Task PoseGraph_InvalidatePairsOutsideActiveImageIds()
	{
		var poseGraph = new PoseGraph();
		poseGraph.AddEdge(1, 2, SynthesizeEdge());
		poseGraph.AddEdge(1, 3, SynthesizeEdge());
		poseGraph.AddEdge(2, 3, SynthesizeEdge());

		// Only images 1 and 2 are active; image 3 is excluded
		poseGraph.InvalidatePairsOutsideActiveImageIds(new HashSet<uint> { 1, 2 });

		// Edge (1,2) should remain valid
		await Assert.That(poseGraph.IsValid(ImagePairToPairId(1, 2))).IsTrue();
		// Edges involving image 3 should be invalidated
		await Assert.That(poseGraph.IsValid(ImagePairToPairId(1, 3))).IsFalse();
		await Assert.That(poseGraph.IsValid(ImagePairToPairId(2, 3))).IsFalse();
	}

	// C#-only: equally large components come out by smallest frame id, and
	// LargestConnectedFrameComponent picks the same one (divergence 38).
	[Test]
	public async Task CSharpOnly_EqualSizeComponentsOrderedBySmallestFrameId()
	{
		Reconstruction reconstruction = SingleFrameRigs(4);
		List<uint> regImageIds = reconstruction.RegImageIds();
		var poseGraph = new PoseGraph();
		// Add the pair with the larger ids first, so insertion order would put it first.
		poseGraph.AddEdge(regImageIds[2], regImageIds[3], SynthesizeEdge());
		poseGraph.AddEdge(regImageIds[0], regImageIds[1], SynthesizeEdge());

		List<HashSet<uint>> components = poseGraph.ConnectedFrameComponents(reconstruction);
		uint smallestFrame = regImageIds.Select(id => reconstruction.Image(id).FrameId).Min();

		await Assert.That(components.Count).IsEqualTo(2);
		await Assert.That(components[0].Min()).IsEqualTo(smallestFrame);
		await Assert.That(poseGraph.LargestConnectedFrameComponent(reconstruction).SetEquals(components[0])).IsTrue();
		List<HashSet<uint>> imageComponents = poseGraph.ConnectedImageIdsForFrameComponents(reconstruction);
		await Assert.That(imageComponents.Count).IsEqualTo(2);
		await Assert.That(imageComponents.Sum(c => c.Count)).IsEqualTo(4);
	}
}
