// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DelaunayTriangulation3: a 3D Delaunay tetrahedralization, the managed replacement for
// CGAL::Delaunay_triangulation_3 that COLMAP's mvs/delaunay_meshing.cc builds on. CGAL is
// GPL and excluded (docs/LICENSE_AUDIT.md); nothing here was read from or transcribed from
// CGAL. What is kept is the *documented* shape of CGAL's triangulation, so the meshing port
// reads like COLMAP's code:
// - the triangulation of the convex hull is closed into a triangulation of the 3-sphere by
//   one infinite vertex; every hull facet has an infinite cell on its outer side;
// - a cell has 4 vertices and 4 neighbors, neighbor i being across the facet opposite
//   vertex i; a facet is (cell, i); its mirror facet is the same triangle seen from the
//   neighbor;
// - every cell is positively oriented (RobustPredicates.Orient3D of its vertices is +1), and
//   an infinite cell is oriented as if its infinite vertex were a point beyond its hull facet.
//
// Algorithm: incremental Bowyer-Watson insertion (A. Bowyer, "Computing Dirichlet
// tessellations", and D. F. Watson, "Computing the n-dimensional Delaunay tessellation...",
// both The Computer Journal 24(2), 1981), with point location by a visibility walk
// (O. Devillers, S. Pion and M. Teillaud, "Walking in a triangulation", IJFCS 13(2), 2002),
// insertion order from SpatialOrder (BRIO + Hilbert), and exact predicates from
// RobustPredicates. The conflict test for infinite cells and the handling of cospherical and
// coplanar configurations follow the standard definition of the Delaunay triangulation of a
// point set closed by a point at infinity (e.g. J.-D. Boissonnat and M. Yvinec, "Algorithmic
// Geometry", 1998, ch. 17).
//
// Handles are ints. Vertex 0 is the infinite vertex; finite vertices are numbered 1, 2, ...
// in the order they were first inserted. Cells are slots in flat arrays; deleted slots are
// reused, so iterate with AllCells() or check IsCellAlive. Per-cell data lives in caller
// arrays of length CellCapacity indexed by cell handle (the "cell info" of CGAL).
//
// This file: storage, accessors, iteration, counts, circumcenters. Insertion and location:
// DelaunayTriangulation3.Insert.cs. Segment traversal: DelaunayTriangulation3.Traverse.cs.
// Tests: ColmapSharp.Tests/Geometry/Delaunay/DelaunayTriangulation3Tests.CSharpOnly.cs.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Geometry.Delaunay;

/// <summary>
/// 3D Delaunay tetrahedralization with an infinite vertex (replacement for CGAL's
/// Delaunay_triangulation_3 as used by colmap/mvs/delaunay_meshing.cc).
/// </summary>
public sealed partial class DelaunayTriangulation3
{
	/// <summary>Handle of the infinite vertex.</summary>
	public const int InfiniteVertex = 0;

	// For facet i of a cell, the three other vertex slots ordered so that the triangle is
	// counterclockwise seen from vertex i (i.e. from inside the cell): Orient3D of
	// (facet vertices, vertex i) is positive. Derived from the positive cell orientation.
	private static readonly int[] FacetVertexSlots = [1, 3, 2, 0, 2, 3, 0, 3, 1, 0, 1, 2];

	private Vector3d[] vertexPoints = new Vector3d[16];
	private int[] vertexCells = new int[16];
	private int vertexCount = 1;

	private int[] cellVertices = new int[64];
	private int[] cellNeighbors = new int[64];
	private int cellSlotCount;
	private int liveCellCount;
	private readonly Stack<int> freeCells = new();

	/// <summary>
	/// Affine dimension of the inserted points: -1 when empty, 0 for one point, 1 for
	/// collinear, 2 for coplanar, 3 once the points span space. Cells exist only in dimension 3.
	/// </summary>
	public int Dimension { get; private set; } = -1;

	/// <summary>Number of finite vertices (distinct inserted points).</summary>
	public int NumberOfVertices => vertexCount - 1;

	/// <summary>Number of live cells, finite and infinite.</summary>
	public int NumberOfCells => liveCellCount;

	/// <summary>One past the largest cell handle; the length of per-cell caller arrays.</summary>
	public int CellCapacity => cellSlotCount;

	/// <summary>One past the largest vertex handle; the length of per-vertex caller arrays.</summary>
	public int VertexCapacity => vertexCount;

	/// <summary>The position of a finite vertex.</summary>
	public Vector3d Point(int vertex)
	{
		Check.Gt(vertex, InfiniteVertex);
		Check.Lt(vertex, vertexCount);
		return vertexPoints[vertex];
	}

	/// <summary>A live cell incident to the vertex, or -1 before the triangulation is 3D.</summary>
	public int IncidentCell(int vertex) => Dimension == 3 ? vertexCells[vertex] : -1;

	/// <summary>Vertex i (0..3) of a cell.</summary>
	public int Vertex(int cell, int i) => cellVertices[4 * cell + i];

	/// <summary>Neighbor i of a cell: the cell across the facet opposite vertex i.</summary>
	public int Neighbor(int cell, int i) => cellNeighbors[4 * cell + i];

	/// <summary>Whether a cell handle refers to a live cell.</summary>
	public bool IsCellAlive(int cell) => cell >= 0 && cell < cellSlotCount && cellVertices[4 * cell] >= 0;

	/// <summary>Whether the cell has the infinite vertex.</summary>
	public bool IsInfinite(int cell) => InfiniteIndex(cell) >= 0;

	/// <summary>The slot of the infinite vertex in the cell, or -1 for a finite cell.</summary>
	public int InfiniteIndex(int cell)
	{
		int b = 4 * cell;
		for (int i = 0; i < 4; ++i)
		{
			if (cellVertices[b + i] == InfiniteVertex)
			{
				return i;
			}
		}

		return -1;
	}

	/// <summary>Whether facet (cell, i) contains the infinite vertex (CGAL's is_infinite(c, i)).</summary>
	public bool IsInfinite(int cell, int i)
	{
		int k = InfiniteIndex(cell);
		return k >= 0 && k != i;
	}

	/// <summary>The slot of vertex in cell, or -1.</summary>
	public int IndexOfVertex(int cell, int vertex)
	{
		int b = 4 * cell;
		for (int i = 0; i < 4; ++i)
		{
			if (cellVertices[b + i] == vertex)
			{
				return i;
			}
		}

		return -1;
	}

	/// <summary>The slot i with Neighbor(cell, i) == neighbor (CGAL's c-&gt;index(n)).</summary>
	public int IndexOfNeighbor(int cell, int neighbor)
	{
		int b = 4 * cell;
		for (int i = 0; i < 4; ++i)
		{
			if (cellNeighbors[b + i] == neighbor)
			{
				return i;
			}
		}

		throw new InvalidOperationException("Check failed: cells are not neighbors.");
	}

	/// <summary>The same facet seen from the neighboring cell (CGAL's mirror_facet).</summary>
	public (int Cell, int Index) MirrorFacet(int cell, int i)
	{
		int n = Neighbor(cell, i);
		return (n, IndexOfNeighbor(n, cell));
	}

	/// <summary>
	/// Slot of the j-th vertex (j in 0..2) of facet i, ordered counterclockwise as seen from
	/// inside the cell (the documented meaning of CGAL's vertex_triple_index).
	/// </summary>
	public static int VertexTripleIndex(int i, int j) => FacetVertexSlots[3 * i + j];

	/// <summary>Vertex handles of facet (cell, i), counterclockwise seen from inside the cell.</summary>
	public (int A, int B, int C) FacetVertices(int cell, int i)
	{
		int b = 4 * cell;
		return (cellVertices[b + FacetVertexSlots[3 * i]],
			cellVertices[b + FacetVertexSlots[3 * i + 1]],
			cellVertices[b + FacetVertexSlots[3 * i + 2]]);
	}

	/// <summary>All live cells, finite and infinite, in handle order.</summary>
	public IEnumerable<int> AllCells()
	{
		for (int c = 0; c < cellSlotCount; ++c)
		{
			if (cellVertices[4 * c] >= 0)
			{
				yield return c;
			}
		}
	}

	/// <summary>All live finite cells in handle order.</summary>
	public IEnumerable<int> FiniteCells()
	{
		for (int c = 0; c < cellSlotCount; ++c)
		{
			if (cellVertices[4 * c] >= 0 && !IsInfinite(c))
			{
				yield return c;
			}
		}
	}

	/// <summary>
	/// Every finite facet once, as (cell, i) from the side with the smaller cell handle,
	/// in handle order.
	/// </summary>
	public IEnumerable<(int Cell, int Index)> FiniteFacets()
	{
		for (int c = 0; c < cellSlotCount; ++c)
		{
			if (cellVertices[4 * c] < 0)
			{
				continue;
			}

			int k = InfiniteIndex(c);
			for (int i = 0; i < 4; ++i)
			{
				if ((k < 0 || k == i) && c < cellNeighbors[4 * c + i])
				{
					yield return (c, i);
				}
			}
		}
	}

	/// <summary>
	/// Every finite edge once as a vertex pair (smaller handle first), in order of first
	/// appearance over cells in handle order.
	/// </summary>
	public IEnumerable<(int V0, int V1)> FiniteEdges()
	{
		var seen = new HashSet<long>();
		for (int c = 0; c < cellSlotCount; ++c)
		{
			if (cellVertices[4 * c] < 0)
			{
				continue;
			}

			for (int i = 0; i < 3; ++i)
			{
				for (int j = i + 1; j < 4; ++j)
				{
					int a = cellVertices[4 * c + i], b = cellVertices[4 * c + j];
					if (a == InfiniteVertex || b == InfiniteVertex)
					{
						continue;
					}

					if (a > b)
					{
						(a, b) = (b, a);
					}

					if (seen.Add(((long)a << 32) | (uint)b))
					{
						yield return (a, b);
					}
				}
			}
		}
	}

	/// <summary>
	/// Circumcenter of a finite cell, computed in double precision (an inexact construction,
	/// like CGAL's Epick circumcenter()).
	/// </summary>
	public Vector3d Circumcenter(int cell)
	{
		Check.That(!IsInfinite(cell), "!IsInfinite(cell)");
		var a = vertexPoints[Vertex(cell, 0)];
		var b = vertexPoints[Vertex(cell, 1)];
		var c = vertexPoints[Vertex(cell, 2)];
		var d = vertexPoints[Vertex(cell, 3)];
		double bx = b.X - a.X, by = b.Y - a.Y, bz = b.Z - a.Z;
		double cx = c.X - a.X, cy = c.Y - a.Y, cz = c.Z - a.Z;
		double dx = d.X - a.X, dy = d.Y - a.Y, dz = d.Z - a.Z;
		double b2 = bx * bx + by * by + bz * bz;
		double c2 = cx * cx + cy * cy + cz * cz;
		double d2 = dx * dx + dy * dy + dz * dz;

		// cross products c x d, d x b, b x c
		double cdx = cy * dz - cz * dy, cdy = cz * dx - cx * dz, cdz = cx * dy - cy * dx;
		double dbx = dy * bz - dz * by, dby = dz * bx - dx * bz, dbz = dx * by - dy * bx;
		double bcx = by * cz - bz * cy, bcy = bz * cx - bx * cz, bcz = bx * cy - by * cx;
		double denominator = 2 * (bx * cdx + by * cdy + bz * cdz);
		return new Vector3d(
			a.X + (b2 * cdx + c2 * dbx + d2 * bcx) / denominator,
			a.Y + (b2 * cdy + c2 * dby + d2 * bcy) / denominator,
			a.Z + (b2 * cdz + c2 * dbz + d2 * bcz) / denominator);
	}

	private int AllocateVertex(in Vector3d point)
	{
		if (vertexCount == vertexPoints.Length)
		{
			Array.Resize(ref vertexPoints, 2 * vertexCount);
			Array.Resize(ref vertexCells, 2 * vertexCount);
		}

		vertexPoints[vertexCount] = point;
		vertexCells[vertexCount] = -1;
		return vertexCount++;
	}

	private int AllocateCell()
	{
		++liveCellCount;
		if (freeCells.Count > 0)
		{
			return freeCells.Pop();
		}

		if (4 * (cellSlotCount + 1) > cellVertices.Length)
		{
			Array.Resize(ref cellVertices, 2 * cellVertices.Length);
			Array.Resize(ref cellNeighbors, 2 * cellNeighbors.Length);
			Array.Resize(ref cellMarks, cellVertices.Length / 4);
		}

		return cellSlotCount++;
	}

	private void FreeCell(int cell)
	{
		--liveCellCount;
		cellVertices[4 * cell] = -1;
		freeCells.Push(cell);
	}
}
