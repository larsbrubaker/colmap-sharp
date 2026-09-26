// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PoseGraph: port of colmap/scene/pose_graph.h/.cc, the graph of relative poses between
// image pairs (one edge per pair, keyed by the pair id with the smaller image id first) that
// global SfM builds from the correspondence graph, plus its connected components in frame
// terms, which take a Reconstruction for the image -> frame mapping. Uses
// Mathematics/ConnectedComponents.cs. Tests: ColmapSharp.Tests/Scene/PoseGraphTests.cs
// (pose_graph_test.cc 1:1).
//
// Tier A. Translation notes:
// - Edge is a class so that EdgeRef hands out the stored edge by reference, as COLMAP's
//   Edge& does; AddEdge, UpdateEdge and GetEdge copy, as COLMAP's by-value Edge does.
// - COLMAP's missing-pair std::out_of_range (map::at) is KeyNotFoundException; its
//   std::runtime_error for a duplicate or missing pair is InvalidOperationException.
// - Edges live in a Dictionary, which enumerates in insertion order until an edge is deleted
//   (COLMAP's NodeHashMap has its own order). The component functions do not depend on it:
//   they walk frames in ascending id order and order equally large components by their
//   smallest frame id (docs/CPP_DIVERGENCES.md, entry 38).
// - Load's LOG(INFO) edge count is dropped, like LOG(INFO) everywhere in the library.

using ColmapSharp.Geometry;
using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>Relative poses between image pairs. Port of colmap::PoseGraph.</summary>
public sealed class PoseGraph
{
	/// <summary>Minimal relative pose data for pose graph edge. Port of PoseGraph::Edge.</summary>
	public sealed class Edge
	{
		/// <summary>An edge with the identity relative pose.</summary>
		public Edge()
		{
		}

		/// <summary>An edge with the given relative pose.</summary>
		public Edge(Rigid3d cam2FromCam1)
		{
			Cam2FromCam1 = cam2FromCam1;
		}

		/// <summary>Relative pose from image 1 to image 2.</summary>
		public Rigid3d Cam2FromCam1 { get; set; } = new();

		/// <summary>Number of two-view matches used to compute the relative pose.</summary>
		public int NumMatches { get; set; }

		/// <summary>Whether this edge is valid for reconstruction.</summary>
		public bool Valid { get; set; } = true;

		/// <summary>Invert the geometry to match swapped image order.</summary>
		public void Invert() => Cam2FromCam1 = Cam2FromCam1.Inverse();

		/// <summary>A copy (COLMAP's Edge is a value type).</summary>
		public Edge Clone() => new(Cam2FromCam1) { NumMatches = NumMatches, Valid = Valid };
	}

	// Map from pair ID to edge data. The pair ID is computed from the
	// two image IDs using ImagePairToPairId, with the smaller ID first.
	private readonly Dictionary<ulong, Edge> edges = [];

	/// <summary>All edges by pair id (smaller image id first).</summary>
	public Dictionary<ulong, Edge> Edges => edges;

	/// <summary>Number of edges, valid or not.</summary>
	public int NumEdges => edges.Count;

	/// <summary>Whether there are no edges.</summary>
	public bool Empty => edges.Count == 0;

	/// <summary>Removes every edge.</summary>
	public void Clear() => edges.Clear();

	/// <summary>
	/// Load edges from the correspondence graph: one edge per image pair with matches whose
	/// two-view geometry has a relative pose, with its number of matches.
	/// </summary>
	public void Load(CorrespondenceGraph corrGraph)
	{
		foreach ((ulong pairId, uint numMatches) in corrGraph.NumMatchesBetweenAllImages())
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			TwoViewGeometry twoViewGeometry = corrGraph.ExtractTwoViewGeometry(imageId1, imageId2, extractInlierMatches: false);
			if (twoViewGeometry.Cam2FromCam1 is Rigid3d cam2FromCam1)
			{
				var edge = new Edge(cam2FromCam1) { NumMatches = (int)numMatches };
				AddEdge(imageId1, imageId2, edge);
			}
		}
	}

	/// <summary>
	/// Adds a copy of <paramref name="edge"/> for the pair, inverted if the ids are given in
	/// swapped order, and returns the stored edge. Throws if the pair already has an edge.
	/// </summary>
	public Edge AddEdge(uint imageId1, uint imageId2, Edge edge)
	{
		Edge stored = edge.Clone();
		if (Types.ShouldSwapImagePair(imageId1, imageId2))
		{
			stored.Invert();
		}

		ulong pairId = Types.ImagePairToPairId(imageId1, imageId2);
		if (!edges.TryAdd(pairId, stored))
		{
			throw new InvalidOperationException($"Image pair already exists: {imageId1}, {imageId2}");
		}

		return stored;
	}

	/// <summary>Whether the pair has an edge, in either order.</summary>
	public bool HasEdge(uint imageId1, uint imageId2) =>
		edges.ContainsKey(Types.ImagePairToPairId(imageId1, imageId2));

	/// <summary>
	/// The stored edge (not inverted) and whether the ids were given in swapped order.
	/// Throws <see cref="KeyNotFoundException"/> for a missing pair.
	/// </summary>
	public (Edge Edge, bool Swapped) EdgeRef(uint imageId1, uint imageId2)
	{
		bool swapped = Types.ShouldSwapImagePair(imageId1, imageId2);
		ulong pairId = Types.ImagePairToPairId(imageId1, imageId2);
		return (edges[pairId], swapped);
	}

	/// <summary>
	/// A copy of the edge oriented from <paramref name="imageId1"/> to
	/// <paramref name="imageId2"/>. Throws <see cref="KeyNotFoundException"/> for a missing pair.
	/// </summary>
	public Edge GetEdge(uint imageId1, uint imageId2)
	{
		bool swapped = Types.ShouldSwapImagePair(imageId1, imageId2);
		ulong pairId = Types.ImagePairToPairId(imageId1, imageId2);
		Edge result = edges[pairId].Clone();
		if (swapped)
		{
			result.Invert();
		}

		return result;
	}

	/// <summary>Removes the pair's edge; false if there was none.</summary>
	public bool DeleteEdge(uint imageId1, uint imageId2) =>
		edges.Remove(Types.ImagePairToPairId(imageId1, imageId2));

	/// <summary>
	/// Replaces the pair's edge with a copy of <paramref name="edge"/> (inverted if the ids
	/// are swapped). Throws if the pair has no edge.
	/// </summary>
	public void UpdateEdge(uint imageId1, uint imageId2, Edge edge)
	{
		Edge stored = edge.Clone();
		if (Types.ShouldSwapImagePair(imageId1, imageId2))
		{
			stored.Invert();
		}

		ulong pairId = Types.ImagePairToPairId(imageId1, imageId2);
		if (!edges.ContainsKey(pairId))
		{
			throw new InvalidOperationException($"Image pair does not exist: {imageId1}, {imageId2}");
		}

		edges[pairId] = stored;
	}

	/// <summary>Whether the pair has an edge and it is valid.</summary>
	public bool IsValid(ulong pairId) => edges.TryGetValue(pairId, out Edge? edge) && edge.Valid;

	/// <summary>Marks an existing edge valid.</summary>
	public void SetValidEdge(ulong pairId)
	{
		Check.That(edges.TryGetValue(pairId, out Edge? edge), "Edge does not exist", "it != edges_.end()");
		edge.Valid = true;
	}

	/// <summary>Marks an existing edge invalid.</summary>
	public void SetInvalidEdge(ulong pairId)
	{
		Check.That(edges.TryGetValue(pairId, out Edge? edge), "Edge does not exist", "it != edges_.end()");
		edge.Valid = false;
	}

	/// <summary>The valid edges only.</summary>
	public IEnumerable<KeyValuePair<ulong, Edge>> ValidEdges() => edges.Where(kv => kv.Value.Valid);

	/// <summary>
	/// Returns connected components as sets of frame IDs, largest first (equally large ones
	/// by smallest frame id). If <paramref name="filterUnregistered"/> is true, only considers
	/// frames with HasPose().
	/// </summary>
	public List<HashSet<uint>> ConnectedFrameComponents(Reconstruction reconstruction, bool filterUnregistered = true)
	{
		(List<uint> nodes, List<(uint, uint)> graphEdges) = BuildFrameGraph(reconstruction, filterUnregistered);
		if (nodes.Count == 0)
		{
			return [];
		}

		// COLMAP's std::sort by size is unstable; the stable OrderByDescending keeps equally
		// large components in the order FindConnectedComponents gives them, which is by
		// smallest frame id since the nodes are sorted.
		return [.. ConnectedComponents.FindConnectedComponents(nodes, graphEdges)
			.OrderByDescending(component => component.Count)
			.Select(component => new HashSet<uint>(component))];
	}

	/// <summary>
	/// Returns image IDs for each connected frame component, in the order of
	/// <see cref="ConnectedFrameComponents"/> (largest first).
	/// </summary>
	public List<HashSet<uint>> ConnectedImageIdsForFrameComponents(Reconstruction reconstruction, bool filterUnregistered = true)
	{
		List<HashSet<uint>> frameComponents = ConnectedFrameComponents(reconstruction, filterUnregistered);

		var frameToComponent = new Dictionary<uint, int>();
		for (int comp = 0; comp < frameComponents.Count; ++comp)
		{
			foreach (uint frameId in frameComponents[comp])
			{
				frameToComponent[frameId] = comp;
			}
		}

		var imageIds = new List<HashSet<uint>>(frameComponents.Count);
		for (int comp = 0; comp < frameComponents.Count; ++comp)
		{
			imageIds.Add([]);
		}

		foreach ((uint imageId, Image image) in reconstruction.Images)
		{
			if (frameToComponent.TryGetValue(image.FrameId, out int comp))
			{
				imageIds[comp].Add(imageId);
			}
		}

		return imageIds;
	}

	/// <summary>
	/// Returns the frame IDs in the largest connected component (of equally large ones, the
	/// one with the smallest frame id). If <paramref name="filterUnregistered"/> is true, only
	/// considers frames with HasPose().
	/// </summary>
	public HashSet<uint> LargestConnectedFrameComponent(Reconstruction reconstruction, bool filterUnregistered = true)
	{
		(List<uint> nodes, List<(uint, uint)> graphEdges) = BuildFrameGraph(reconstruction, filterUnregistered);
		if (nodes.Count == 0)
		{
			return [];
		}

		return [.. ConnectedComponents.FindLargestConnectedComponent(nodes, graphEdges)];
	}

	/// <summary>Mark image pairs as invalid if either image is not in the active set.</summary>
	public void InvalidatePairsOutsideActiveImageIds(IReadOnlySet<uint> activeImageIds)
	{
		foreach (ulong pairId in edges.Keys)
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			if (!activeImageIds.Contains(imageId1) || !activeImageIds.Contains(imageId2))
			{
				// Setting a flag on the edge object does not modify the dictionary, so the
				// enumeration stays valid.
				edges[pairId].Valid = false;
			}
		}
	}

	// BuildFrameGraph (pose_graph.cc's anonymous namespace): the frames of the valid edges'
	// images, skipping edges with an unposed frame when filterUnregistered, and one frame
	// edge per image edge. COLMAP's node set is a FlatHashSet; here the nodes are returned in
	// ascending id order so that the component order is reproducible.
	private (List<uint> Nodes, List<(uint, uint)> Edges) BuildFrameGraph(Reconstruction reconstruction, bool filterUnregistered)
	{
		var nodes = new SortedSet<uint>();
		var graphEdges = new List<(uint, uint)>();
		foreach ((ulong pairId, Edge _) in ValidEdges())
		{
			(uint imageId1, uint imageId2) = Types.PairIdToImagePair(pairId);
			uint frameId1 = reconstruction.Image(imageId1).FrameId;
			uint frameId2 = reconstruction.Image(imageId2).FrameId;

			if (filterUnregistered && (!reconstruction.Frame(frameId1).HasPose || !reconstruction.Frame(frameId2).HasPose))
			{
				continue;
			}

			nodes.Add(frameId1);
			nodes.Add(frameId2);
			graphEdges.Add((frameId1, frameId2));
		}

		return ([.. nodes], graphEdges);
	}
}
