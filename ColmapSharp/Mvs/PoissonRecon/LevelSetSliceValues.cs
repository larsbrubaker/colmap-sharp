// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// LevelSetSliceValues: what the level-set extractor keeps per slice (a plane z = slice at one
// depth): its cell indices (LevelSetCellIndices), the implicit function's value and gradient at
// each distinct corner, each leaf's marching-squares index on the slice, the key of each edge
// that got an iso-vertex, the edge-key to vertex map, the iso-edges of each face and the
// face-key to iso-edge and vertex-pair maps, and the scratch flags and lists the vertex and
// iso-edge passes fill. Ports FEMTree.LevelSet.3D.inl's _LevelSetExtractor< ... , 3 , ...
// >::SliceValues (reset, cornerValues, cornerGradients, mcIndices, edgeKeys, faceEdges,
// edgeVertexMap, faceEdgeMap, vertexPairMap and the three setFromScratch) and
// SliceValues::Scratch (reset, cSet, eSet, fSet, eKeyValues, fKeyValues, vKeyValues);
// LevelSetXSliceValues ports the same for a slab (the edges and faces crossing it):
// XSliceValues and XSliceValues::Scratch. LevelSetSlabValues pairs them as SlabValues does (two
// of each, by parity). Filled by PoissonLevelSetExtractor (the vertices in .IsoVertices.cs and
// .XSliceIsoVertices.cs, the iso-edges and vertex pairs in .IsoEdges.cs). Like upstream, the
// value arrays grow but never shrink or clear on reuse: a corner, edge or face entry is read
// only once its flag is set. The lookups the polygon step makes (addIsoEdges, setVertexPair,
// setEdgeVertex) are in PoissonLevelSetExtractor.Polygons.cs.
//
// Translation notes: the key/vertex lists hold the vertex's index in the output stream rather
// than a copy of the vertex (upstream pairs both); the vertex is the sink's entry at that
// index. Upstream keeps one list per thread and merges them in thread order; the port runs
// single-threaded, so there is one list (docs/CPP_DIVERGENCES.md, entry 123). The maps are only
// looked up, never iterated, so their (unordered) iteration order reaches no output; the order
// of the iso-edges inside one face-edge map entry is the recording order.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Per-slice corner and edge-vertex data of the level-set extractor. Port of the corner and
/// edge parts of PoissonRecon's <c>SliceValues</c> and <c>SliceValues::Scratch</c>.
/// </summary>
public sealed class LevelSetSliceValues
{
	/// <summary>The slice's cell indices. Port of <c>cellIndices</c>.</summary>
	public LevelSetCellIndices CellIndices { get; } = new();

	/// <summary>The slice number (-1 before the first reset). Port of <c>slice()</c>.</summary>
	public int Slice { get; private set; } = -1;

	/// <summary>The value at each corner, by corner index. Port of <c>cornerValues</c>.</summary>
	public float[] CornerValues { get; private set; } = [];

	/// <summary>The gradient at each corner, three floats per corner index, or null. Port of <c>cornerGradients</c>.</summary>
	public float[]? CornerGradients { get; private set; }

	/// <summary>Each node's marching-squares index, by sorted index minus the node offset. Port of <c>mcIndices</c>.</summary>
	public byte[] McIndices { get; private set; } = [];

	/// <summary>Whether each corner has been set. Port of <c>Scratch::cSet</c>.</summary>
	public byte[] CornerSet { get; private set; } = [];

	/// <summary>Whether each edge's iso-vertex has been handled. Port of <c>Scratch::eSet</c>.</summary>
	public byte[] EdgeSet { get; private set; } = [];

	/// <summary>The key of each edge whose iso-vertex this slice made, by edge index. Port of <c>edgeKeys</c>.</summary>
	public LevelSetKey[] EdgeKeys { get; private set; } = [];

	/// <summary>
	/// The edge keys and vertex indices recorded for this slice since the last finalize, in
	/// recording order. Port of <c>Scratch::eKeyValues</c>.
	/// </summary>
	public List<(LevelSetKey Key, int Vertex)> EdgeKeyValues { get; } = [];

	/// <summary>The vertex index of each edge key on the slice. Port of <c>edgeVertexMap</c>.</summary>
	public Dictionary<LevelSetKey, int> EdgeVertexMap { get; } = [];

	/// <summary>Whether each face's iso-edges have been set. Port of <c>Scratch::fSet</c>.</summary>
	public byte[] FaceSet { get; private set; } = [];

	/// <summary>The iso-edges of each face this slice set, by face index. Port of <c>faceEdges</c>.</summary>
	public LevelSetFaceEdges[] FaceEdges { get; private set; } = [];

	/// <summary>
	/// The face keys and iso-edges pushed to this slice from finer leaves since the last
	/// finalize, in recording order. Port of <c>Scratch::fKeyValues</c>.
	/// </summary>
	public List<(LevelSetKey Key, LevelSetIsoEdge[] Edges)> FaceKeyValues { get; } = [];

	/// <summary>
	/// The pairs of edge keys whose vertices a coarser edge joins, recorded since the last
	/// finalize, in recording order. Port of <c>Scratch::vKeyValues</c>.
	/// </summary>
	public List<(LevelSetKey First, LevelSetKey Second)> VertexPairKeyValues { get; } = [];

	/// <summary>The iso-edges of each face key on the slice. Port of <c>faceEdgeMap</c>.</summary>
	public Dictionary<LevelSetKey, List<LevelSetIsoEdge>> FaceEdgeMap { get; } = [];

	/// <summary>Each edge key's partner across a coarser edge (both directions). Port of <c>vertexPairMap</c>.</summary>
	public Dictionary<LevelSetKey, LevelSetKey> VertexPairMap { get; } = [];

	/// <summary>
	/// Starts slice <paramref name="slice"/> after <see cref="CellIndices"/> is set. Port of
	/// <c>SliceValues::reset( slice , computeGradients )</c> followed by
	/// <c>Scratch::reset( cellIndices )</c>'s corner and edge parts.
	/// </summary>
	public void Reset(int slice, bool computeGradients)
	{
		Slice = slice;
		FaceEdgeMap.Clear();
		EdgeVertexMap.Clear();
		VertexPairMap.Clear();
		if (McIndices.Length < CellIndices.Size)
		{
			McIndices = new byte[CellIndices.Size];
		}

		int corners = CellIndices.Count(0);
		if (CornerValues.Length < corners)
		{
			CornerValues = new float[corners];
			CornerGradients = computeGradients ? new float[3 * corners] : null;
		}

		int edges = CellIndices.Count(1);
		if (EdgeKeys.Length < edges)
		{
			// NewPointer< Key >: default-constructed (unset) keys.
			EdgeKeys = new LevelSetKey[edges];
			Array.Fill(EdgeKeys, LevelSetKey.Unset);
		}

		int faces = CellIndices.Count(2);
		if (FaceEdges.Length < faces)
		{
			// NewPointer< FaceEdges >: default-constructed (count -1) entries.
			FaceEdges = new LevelSetFaceEdges[faces];
			Array.Fill(FaceEdges, LevelSetFaceEdges.Unset);
		}

		// Scratch::reset: fresh, zeroed flags per corner, edge and face, and no recorded keys.
		CornerSet = new byte[corners];
		EdgeSet = new byte[edges];
		FaceSet = new byte[faces];
		EdgeKeyValues.Clear();
		FaceKeyValues.Clear();
		VertexPairKeyValues.Clear();
	}

	/// <summary>
	/// Moves the recorded edge keys into <see cref="EdgeVertexMap"/> (a later entry for the same
	/// key wins) and clears them. Port of <c>setFromScratch( Scratch::EKeyValues )</c>.
	/// </summary>
	public void SetEdgesFromScratch()
	{
		foreach ((LevelSetKey key, int vertex) in EdgeKeyValues)
		{
			EdgeVertexMap[key] = vertex;
		}

		EdgeKeyValues.Clear();
	}

	/// <summary>
	/// Moves the recorded vertex pairs into <see cref="VertexPairMap"/>, each in both directions
	/// (a later entry for the same key wins), and clears them. Port of
	/// <c>setFromScratch( Scratch::VKeyValues )</c>.
	/// </summary>
	public void SetVertexPairsFromScratch() => LevelSetScratch.SetVertexPairs(VertexPairKeyValues, VertexPairMap);

	/// <summary>
	/// Appends the recorded face iso-edges to <see cref="FaceEdgeMap"/>'s entry for each key
	/// and clears them. Port of <c>setFromScratch( Scratch::FKeyValues )</c>.
	/// </summary>
	public void SetFacesFromScratch() => LevelSetScratch.SetFaces(FaceKeyValues, FaceEdgeMap);
}

/// <summary>
/// Per-slab edge-vertex data of the level-set extractor: the slab's cell indices (a slab's
/// 2D corners are the edges crossing it) and what the cross-slice vertex pass records. Port of
/// the edge parts of PoissonRecon's <c>XSliceValues</c> and <c>XSliceValues::Scratch</c>.
/// </summary>
public sealed class LevelSetXSliceValues
{
	/// <summary>The slab's cell indices. Port of <c>cellIndices</c>.</summary>
	public LevelSetCellIndices CellIndices { get; } = new();

	/// <summary>The slab number (-1 before the first reset). Port of <c>slab()</c>.</summary>
	public int Slab { get; private set; } = -1;

	/// <summary>Whether each cross edge's iso-vertex has been handled. Port of <c>Scratch::eSet</c>.</summary>
	public byte[] EdgeSet { get; private set; } = [];

	/// <summary>The key of each cross edge whose iso-vertex this slab made, by edge index. Port of <c>edgeKeys</c>.</summary>
	public LevelSetKey[] EdgeKeys { get; private set; } = [];

	/// <summary>
	/// The edge keys and vertex indices recorded for this slab since the last finalize, in
	/// recording order: its own cross-edge vertices and those pushed down from finer slabs and
	/// slices. Port of <c>Scratch::eKeyValues</c>.
	/// </summary>
	public List<(LevelSetKey Key, int Vertex)> EdgeKeyValues { get; } = [];

	/// <summary>The vertex index of each edge key in the slab. Port of <c>edgeVertexMap</c>.</summary>
	public Dictionary<LevelSetKey, int> EdgeVertexMap { get; } = [];

	/// <summary>Whether each cross face's iso-edges have been set. Port of <c>Scratch::fSet</c>.</summary>
	public byte[] FaceSet { get; private set; } = [];

	/// <summary>The iso-edges of each cross face this slab set, by face index. Port of <c>faceEdges</c>.</summary>
	public LevelSetFaceEdges[] FaceEdges { get; private set; } = [];

	/// <summary>
	/// The face keys and iso-edges pushed to this slab from finer leaves since the last
	/// finalize, in recording order. Port of <c>Scratch::fKeyValues</c>.
	/// </summary>
	public List<(LevelSetKey Key, LevelSetIsoEdge[] Edges)> FaceKeyValues { get; } = [];

	/// <summary>
	/// The pairs of edge keys whose vertices a coarser cross edge joins, recorded since the last
	/// finalize, in recording order. Port of <c>Scratch::vKeyValues</c>.
	/// </summary>
	public List<(LevelSetKey First, LevelSetKey Second)> VertexPairKeyValues { get; } = [];

	/// <summary>The iso-edges of each face key in the slab. Port of <c>faceEdgeMap</c>.</summary>
	public Dictionary<LevelSetKey, List<LevelSetIsoEdge>> FaceEdgeMap { get; } = [];

	/// <summary>Each edge key's partner across a coarser cross edge (both directions). Port of <c>vertexPairMap</c>.</summary>
	public Dictionary<LevelSetKey, LevelSetKey> VertexPairMap { get; } = [];

	/// <summary>
	/// Starts slab <paramref name="slab"/> after <see cref="CellIndices"/> is set. Port of
	/// <c>XSliceValues::reset( slab )</c> followed by <c>Scratch::reset( cellIndices )</c>'s
	/// edge parts.
	/// </summary>
	public void Reset(int slab)
	{
		Slab = slab;
		FaceEdgeMap.Clear();
		EdgeVertexMap.Clear();
		VertexPairMap.Clear();
		int edges = CellIndices.Count(0);
		if (EdgeKeys.Length < edges)
		{
			EdgeKeys = new LevelSetKey[edges];
			Array.Fill(EdgeKeys, LevelSetKey.Unset);
		}

		int faces = CellIndices.Count(1);
		if (FaceEdges.Length < faces)
		{
			FaceEdges = new LevelSetFaceEdges[faces];
			Array.Fill(FaceEdges, LevelSetFaceEdges.Unset);
		}

		EdgeSet = new byte[edges];
		FaceSet = new byte[faces];
		EdgeKeyValues.Clear();
		FaceKeyValues.Clear();
		VertexPairKeyValues.Clear();
	}

	/// <summary>
	/// Moves the recorded edge keys into <see cref="EdgeVertexMap"/> (a later entry for the same
	/// key wins) and clears them. Port of <c>XSliceValues::setFromScratch( Scratch::EKeyValues )</c>.
	/// </summary>
	public void SetEdgesFromScratch()
	{
		foreach ((LevelSetKey key, int vertex) in EdgeKeyValues)
		{
			EdgeVertexMap[key] = vertex;
		}

		EdgeKeyValues.Clear();
	}

	/// <summary>
	/// Moves the recorded vertex pairs into <see cref="VertexPairMap"/>, each in both directions
	/// (a later entry for the same key wins), and clears them. Port of
	/// <c>XSliceValues::setFromScratch( Scratch::VKeyValues )</c>.
	/// </summary>
	public void SetVertexPairsFromScratch() => LevelSetScratch.SetVertexPairs(VertexPairKeyValues, VertexPairMap);

	/// <summary>
	/// Appends the recorded face iso-edges to <see cref="FaceEdgeMap"/>'s entry for each key
	/// and clears them. Port of <c>XSliceValues::setFromScratch( Scratch::FKeyValues )</c>.
	/// </summary>
	public void SetFacesFromScratch() => LevelSetScratch.SetFaces(FaceKeyValues, FaceEdgeMap);
}

/// <summary>
/// The extractor's per-depth pair of slices and slabs, addressed by parity. Port of
/// <c>SlabValues</c> (its slice values and scratch, and its x-slice values and scratch).
/// </summary>
public sealed class LevelSetSlabValues
{
	private readonly LevelSetSliceValues[] slices = [new(), new()];
	private readonly LevelSetXSliceValues[] xSlices = [new(), new()];

	/// <summary>The values of slice <paramref name="slice"/> (by parity). Port of <c>sliceValues( idx )</c>.</summary>
	public LevelSetSliceValues SliceValues(int slice) => slices[slice & 1];

	/// <summary>The values of slab <paramref name="slab"/> (by parity). Port of <c>xSliceValues( idx )</c>.</summary>
	public LevelSetXSliceValues XSliceValues(int slab) => xSlices[slab & 1];

	/// <summary>The cell indices of slab <paramref name="slab"/> (by parity). Port of <c>xSliceValues( idx ).cellIndices</c>.</summary>
	public LevelSetCellIndices SlabCellIndices(int slab) => xSlices[slab & 1].CellIndices;

	/// <summary>
	/// The edge keys and vertex indices recorded in slab <paramref name="slab"/> (by parity), in
	/// recording order. Port of <c>xSliceScratch( idx ).eKeyValues</c>.
	/// </summary>
	public List<(LevelSetKey Key, int Vertex)> SlabEdgeKeyValues(int slab) => xSlices[slab & 1].EdgeKeyValues;

	/// <summary>The slab number last reset at this parity. Port of <c>xSliceValues( idx ).slab()</c>.</summary>
	public int Slab(int slab) => xSlices[slab & 1].Slab;

	/// <summary>
	/// Starts slab <paramref name="slab"/> once its cell indices are set. Port of
	/// <c>XSliceValues::reset( slab )</c> and <c>XSliceValues::Scratch::reset</c>.
	/// </summary>
	public void ResetSlab(int slab) => xSlices[slab & 1].Reset(slab);
}

// The setFromScratch bodies shared by SliceValues and XSliceValues (XSliceValues' FKeyValues
// form finds then assigns or appends, which has the same effect as SliceValues' operator[] and
// append).
internal static class LevelSetScratch
{
	public static void SetVertexPairs(List<(LevelSetKey First, LevelSetKey Second)> scratch, Dictionary<LevelSetKey, LevelSetKey> map)
	{
		foreach ((LevelSetKey first, LevelSetKey second) in scratch)
		{
			map[first] = second;
			map[second] = first;
		}

		scratch.Clear();
	}

	public static void SetFaces(List<(LevelSetKey Key, LevelSetIsoEdge[] Edges)> scratch, Dictionary<LevelSetKey, List<LevelSetIsoEdge>> map)
	{
		foreach ((LevelSetKey key, LevelSetIsoEdge[] edges) in scratch)
		{
			if (!map.TryGetValue(key, out List<LevelSetIsoEdge>? faceEdges))
			{
				faceEdges = [];
				map[key] = faceEdges;
			}

			faceEdges.AddRange(edges);
		}

		scratch.Clear();
	}
}
