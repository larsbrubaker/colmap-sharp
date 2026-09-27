// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// LevelSetSliceValues: what the level-set extractor keeps per slice (a plane z = slice at one
// depth): its cell indices (LevelSetCellIndices), the implicit function's value and gradient at
// each distinct corner, each leaf's marching-squares index on the slice, the key of each edge
// that got an iso-vertex, the edge-key to vertex map, and the scratch flags and key/vertex
// lists the vertex pass fills. Ports the corner and edge-vertex parts of
// FEMTree.LevelSet.3D.inl's _LevelSetExtractor< ... , 3 , ... >::SliceValues (reset,
// cornerValues, cornerGradients, mcIndices, edgeKeys, edgeVertexMap, setFromScratch(
// EKeyValues )) and SliceValues::Scratch (reset, cSet, eSet, eKeyValues), and XSliceValues'
// slab cell indices, slab number and scratch eKeyValues. LevelSetSlabValues pairs them as
// SlabValues does (two of each, by parity). Filled by PoissonLevelSetExtractor. Like upstream,
// the value arrays grow but never shrink or clear on reuse: a corner or edge entry is read only
// once its flag is set.
//
// Translation notes: the key/vertex lists hold the vertex's index in the output stream rather
// than a copy of the vertex (upstream pairs both); the vertex is the sink's entry at that
// index. Upstream keeps one list per thread and merges them in thread order; the port runs
// single-threaded, so there is one list (docs/CPP_DIVERGENCES.md, entry 123).

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

	/// <summary>
	/// Starts slice <paramref name="slice"/> after <see cref="CellIndices"/> is set. Port of
	/// <c>SliceValues::reset( slice , computeGradients )</c> followed by
	/// <c>Scratch::reset( cellIndices )</c>'s corner and edge parts.
	/// </summary>
	public void Reset(int slice, bool computeGradients)
	{
		Slice = slice;
		EdgeVertexMap.Clear();
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

		// Scratch::reset: fresh, zeroed flags per corner and edge, and no recorded keys.
		CornerSet = new byte[corners];
		EdgeSet = new byte[edges];
		EdgeKeyValues.Clear();
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
}

/// <summary>
/// The extractor's per-depth pair of slices and slabs, addressed by parity. Port of
/// <c>SlabValues</c> (its slice values, and its x-slice values' cell indices, slab number and
/// scratch edge keys).
/// </summary>
public sealed class LevelSetSlabValues
{
	private readonly LevelSetSliceValues[] slices = [new(), new()];
	private readonly LevelSetCellIndices[] slabCellIndices = [new(), new()];
	private readonly List<(LevelSetKey Key, int Vertex)>[] slabEdgeKeyValues = [[], []];
	private readonly int[] slabs = [-1, -1];

	/// <summary>The values of slice <paramref name="slice"/> (by parity). Port of <c>sliceValues( idx )</c>.</summary>
	public LevelSetSliceValues SliceValues(int slice) => slices[slice & 1];

	/// <summary>The cell indices of slab <paramref name="slab"/> (by parity). Port of <c>xSliceValues( idx ).cellIndices</c>.</summary>
	public LevelSetCellIndices SlabCellIndices(int slab) => slabCellIndices[slab & 1];

	/// <summary>
	/// The edge keys and vertex indices pushed into slab <paramref name="slab"/> (by parity), in
	/// recording order. Port of <c>xSliceScratch( idx ).eKeyValues</c>.
	/// </summary>
	public List<(LevelSetKey Key, int Vertex)> SlabEdgeKeyValues(int slab) => slabEdgeKeyValues[slab & 1];

	/// <summary>The slab number last reset at this parity. Port of <c>xSliceValues( idx ).slab()</c>.</summary>
	public int Slab(int slab) => slabs[slab & 1];

	/// <summary>
	/// Records the slab number and clears its recorded edge keys. Port of
	/// <c>XSliceValues::reset( slab )</c>'s slab and <c>XSliceValues::Scratch::reset</c>'s key lists.
	/// </summary>
	public void ResetSlab(int slab)
	{
		slabs[slab & 1] = slab;
		slabEdgeKeyValues[slab & 1].Clear();
	}
}
