// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// LevelSetIsoEdges: the small value types of the level-set extractor's iso-edge step and the
// marching-squares edge table it reads. Ports FEMTree.LevelSet.inl's
// LevelSetExtraction::IsoEdge< 3 > (a segment of the iso-curve on a face, named by the keys of
// the two edges its ends lie on), FEMTree.LevelSet.3D.inl's _LevelSetExtractor::FaceEdges (the
// up to two iso-edges of one face, count -1 when unset) and MarchingCubes.h's
// HyperCube::MarchingSquares (edges table and AddEdgeIndices). The slice and slab stores that
// hold them are LevelSetSliceValues / LevelSetXSliceValues; the step that fills them is
// PoissonLevelSetExtractor.IsoEdges.cs. Tier A against oracle/poisson_levelset5_harness.cc.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// A segment of the iso-curve on a face, from the vertex on edge <see cref="First"/> to the
/// vertex on edge <see cref="Second"/>. Port of <c>LevelSetExtraction::IsoEdge&lt; 3 &gt;</c>.
/// </summary>
public readonly record struct LevelSetIsoEdge(LevelSetKey First, LevelSetKey Second)
{
	/// <summary>A default-constructed edge: both ends unset keys. Port of <c>IsoEdge()</c>.</summary>
	public static readonly LevelSetIsoEdge Unset = new(LevelSetKey.Unset, LevelSetKey.Unset);

	/// <summary>The end with index <paramref name="idx"/> (0 or 1). Port of <c>operator[]</c>.</summary>
	public LevelSetKey this[int idx] => idx == 0 ? First : Second;
}

/// <summary>
/// The iso-edges of one face (at most two). Port of <c>_LevelSetExtractor::FaceEdges</c>.
/// </summary>
public readonly record struct LevelSetFaceEdges(int Count, LevelSetIsoEdge Edge0, LevelSetIsoEdge Edge1)
{
	/// <summary>A default-constructed entry: no edges set (count -1). Port of <c>FaceEdges()</c>.</summary>
	public static readonly LevelSetFaceEdges Unset = new(-1, LevelSetIsoEdge.Unset, LevelSetIsoEdge.Unset);

	/// <summary>The edge with index <paramref name="idx"/> (0 or 1).</summary>
	public LevelSetIsoEdge this[int idx] => idx == 0 ? Edge0 : Edge1;
}

/// <summary>
/// The marching-squares case table: which square edges the iso-curve joins for each corner sign
/// pattern. Port of <c>HyperCube::MarchingSquares</c>.
/// </summary>
public static class MarchingSquares
{
	/// <summary>The most iso-edges a square can have. Port of <c>MAX_EDGES</c>.</summary>
	public const int MaxEdges = 2;

	// Positive to the right, positive in center. Corners (0,0) (1,0) (0,1) (1,1) are bits 0..3.
	private static readonly int[][] Edges =
	[
		[-1, -1, -1, -1, -1],
		[1, 0, -1, -1, -1],
		[0, 2, -1, -1, -1],
		[1, 2, -1, -1, -1],
		[3, 1, -1, -1, -1],
		[3, 0, -1, -1, -1],
		[0, 1, 3, 2, -1],
		[3, 2, -1, -1, -1],
		[2, 3, -1, -1, -1],
		[1, 3, 2, 0, -1],
		[0, 3, -1, -1, -1],
		[1, 3, -1, -1, -1],
		[2, 1, -1, -1, -1],
		[2, 0, -1, -1, -1],
		[0, 1, -1, -1, -1],
		[-1, -1, -1, -1, -1],
	];

	/// <summary>
	/// Writes the square-edge pairs of <paramref name="mcIndex"/>'s iso-edges into
	/// <paramref name="isoIndices"/> (two entries per edge) and returns the edge count. Port of
	/// <c>MarchingSquares::AddEdgeIndices</c>.
	/// </summary>
	public static int AddEdgeIndices(int mcIndex, Span<int> isoIndices)
	{
		int nEdges = 0;

		/* Square is entirely in/out of the surface */
		if (mcIndex == 0 || mcIndex == 15)
		{
			return 0;
		}

		/* Create the edges */
		int[] edges = Edges[mcIndex];
		for (int i = 0; edges[i] != -1; i += 2)
		{
			for (int j = 0; j < 2; j++)
			{
				isoIndices[i + j] = edges[i + j];
			}

			nEdges++;
		}

		return nEdges;
	}
}
