// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DelaunayTriangulation3.Insert: point insertion (Bowyer-Watson) and point location (a
// visibility walk). See DelaunayTriangulation3.cs for the conventions and sources.
//
// Degenerate input:
// - Until the points span 3D, they are kept as a pending list; the first four affinely
//   independent points form the initial tetrahedron and the rest are then inserted in
//   arrival order.
// - An exact duplicate of an existing point returns the existing vertex handle.
// - Cospherical and coplanar points are handled by the strict conflict test: a cell is
//   destroyed only if the new point is strictly inside its circumsphere (for an infinite
//   cell: strictly beyond its hull facet, or on the facet's plane and strictly inside the
//   facet's circumcircle). That region is always star-shaped from the new point, so every
//   new cell is non-degenerate, and the result is a Delaunay triangulation; among the
//   several that exist for cospherical points, which one is built depends on the order.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Geometry.Delaunay;

public sealed partial class DelaunayTriangulation3
{
	// Per-cell visit stamps for the conflict-region search: 2*stamp = visited and not in
	// conflict, 2*stamp+1 = in conflict. A new stamp per insertion avoids clearing.
	private int[] cellMarks = new int[16];
	private int markStamp;

	private List<int>? pendingVertices = [];
	private Dictionary<Vector3d, int>? pendingLookup = [];
	private readonly int[] basis = new int[4];

	private int lastCell = -1;

	/// <summary>Per-caller state of <see cref="Locate(in Vector3d, ref LocateCursor)"/>; start from default.</summary>
	public struct LocateCursor
	{
		internal bool Started;
		internal int Hint;
		internal uint WalkState;

		/// <summary>Makes the next locate start its walk at <paramref name="cell"/> (a known nearby cell).</summary>
		public void StartNear(int cell)
		{
			if (!Started)
			{
				Started = true;
				WalkState = InitialWalkState;
			}

			Hint = cell;
		}
	}
	private const uint InitialWalkState = 0x2545F491u;

	// Walk state of the insertion path only (Insert is not thread-safe anyway).
	private uint walkState = InitialWalkState;

	// Scratch buffers reused across insertions.
	private readonly List<int> conflictCells = [];
	private readonly List<int> boundaryFacets = [];
	private readonly List<int> cavityStack = [];
	private readonly Dictionary<long, int> openEdges = [];
	private int[] newCellVertices = new int[64];

	/// <summary>
	/// Inserts the points in BRIO + Hilbert order (like CGAL's range insertion) and returns,
	/// for each input point, its vertex handle (exact duplicates share one).
	/// </summary>
	public int[] InsertRange(ReadOnlySpan<Vector3d> points, CancellationToken cancellationToken = default)
	{
		var order = SpatialOrder.BrioHilbertOrder(points);
		var handles = new int[points.Length];
		for (int k = 0; k < order.Length; ++k)
		{
			if ((k & 4095) == 0)
			{
				cancellationToken.ThrowIfCancellationRequested();
			}

			int index = order[k];
			handles[index] = Insert(points[index]);
		}

		return handles;
	}

	/// <summary>Inserts a point and returns its vertex handle (the existing one for a duplicate).</summary>
	public int Insert(in Vector3d point)
	{
		if (!double.IsFinite(point.X) || !double.IsFinite(point.Y) || !double.IsFinite(point.Z))
		{
			throw new ArgumentException("Check failed: point coordinates must be finite.");
		}

		if (Dimension < 3)
		{
			return InsertLowDimension(point);
		}

		int cell = Locate(point, lastCell, ref walkState);
		if (!IsInfinite(cell))
		{
			for (int i = 0; i < 4; ++i)
			{
				int v = cellVertices[4 * cell + i];
				var q = vertexPoints[v];
				if (q.X == point.X && q.Y == point.Y && q.Z == point.Z)
				{
					return v;
				}
			}
		}

		int vertex = AllocateVertex(point);
		InsertInCavity(vertex, cell);
		return vertex;
	}

	/// <summary>
	/// Locates a point: returns a finite cell whose closure contains it, or, when the point
	/// is outside the convex hull, an infinite cell whose hull facet has the point strictly
	/// on its outer side. Requires Dimension == 3. A read-only query: the walk starts at the
	/// hint (or the last inserted cell) with a fresh walk state, so the answer depends only on
	/// the triangulation and the arguments, and concurrent calls are safe.
	/// </summary>
	public int Locate(in Vector3d point, int hint = -1)
	{
		uint localWalkState = InitialWalkState;
		return Locate(point, IsCellAlive(hint) ? hint : lastCell, ref localWalkState);
	}

	/// <summary>
	/// Thread-safe location for concurrent readers: starts at the cursor's last result and
	/// keeps the walk's pseudo-random state in the cursor, so the result depends only on the
	/// cursor's own history, never on other threads. Must not run concurrently with Insert.
	/// </summary>
	public int Locate(in Vector3d point, ref LocateCursor cursor)
	{
		if (!cursor.Started)
		{
			cursor = new LocateCursor { Started = true, Hint = lastCell, WalkState = InitialWalkState };
		}

		int cell = Locate(point, IsCellAlive(cursor.Hint) ? cursor.Hint : lastCell, ref cursor.WalkState);
		cursor.Hint = cell;
		return cell;
	}

	private int Locate(in Vector3d point, int startCell, ref uint walkState)
	{
		Check.Eq(Dimension, 3);
		int cell = startCell;
		int previous = -1;
		while (true)
		{
			int k = InfiniteIndex(cell);
			if (k >= 0)
			{
				if (OrientReplaced(cell, k, point) > 0)
				{
					return cell;
				}

				previous = cell;
				cell = cellNeighbors[4 * cell + k];
				continue;
			}

			// Stochastic walk: start the facet scan at a pseudo-random slot so no fixed
			// facet order can cycle. The generator is fixed-seed, so this is deterministic.
			walkState ^= walkState << 13;
			walkState ^= walkState >> 17;
			walkState ^= walkState << 5;
			int start = (int)(walkState & 3);
			int next = -1;
			for (int j = 0; j < 4; ++j)
			{
				int i = (start + j) & 3;
				int neighbor = cellNeighbors[4 * cell + i];
				if (neighbor == previous)
				{
					// The point is on the non-negative side of the facet we came through.
					continue;
				}

				if (OrientReplaced(cell, i, point) < 0)
				{
					next = neighbor;
					break;
				}
			}

			if (next < 0)
			{
				return cell;
			}

			previous = cell;
			cell = next;
		}
	}

	/// <summary>
	/// Whether the point conflicts with the cell: strictly inside its circumsphere, or for an
	/// infinite cell strictly beyond its hull facet or, on the facet's plane, strictly inside
	/// the facet's circumcircle.
	/// </summary>
	public bool InConflict(int cell, in Vector3d point)
	{
		int b = 4 * cell;
		int k = InfiniteIndex(cell);
		if (k < 0)
		{
			return RobustPredicates.InSphere(vertexPoints[cellVertices[b]], vertexPoints[cellVertices[b + 1]],
				vertexPoints[cellVertices[b + 2]], vertexPoints[cellVertices[b + 3]], point) > 0;
		}

		int orientation = OrientReplaced(cell, k, point);
		if (orientation != 0)
		{
			return orientation > 0;
		}

		var (fa, fb, fc) = FacetVertices(cell, k);
		return CoplanarInCircle(vertexPoints[fa], vertexPoints[fb], vertexPoints[fc], point) > 0;
	}

	/// <summary>Orient3D of the cell's vertices with slot i replaced by the point.</summary>
	internal int OrientReplaced(int cell, int i, in Vector3d point)
	{
		int b = 4 * cell;
		var p0 = i == 0 ? point : vertexPoints[cellVertices[b]];
		var p1 = i == 1 ? point : vertexPoints[cellVertices[b + 1]];
		var p2 = i == 2 ? point : vertexPoints[cellVertices[b + 2]];
		var p3 = i == 3 ? point : vertexPoints[cellVertices[b + 3]];
		return RobustPredicates.Orient3D(p0, p1, p2, p3);
	}

	/// <summary>
	/// For p on the plane of the non-collinear a, b, c: +1 strictly inside their circumcircle,
	/// -1 outside, 0 on it. Uses the sphere through a, b, c and a point q off the plane, whose
	/// intersection with the plane is exactly that circle.
	/// </summary>
	internal static int CoplanarInCircle(in Vector3d a, in Vector3d b, in Vector3d c, in Vector3d p)
	{
		double offset = 2 * Math.Max(1.0, Math.Max(Math.Abs(a.X), Math.Max(Math.Abs(a.Y), Math.Abs(a.Z))));
		for (int axis = 2; axis >= 0; --axis)
		{
			// q differs from a in one coordinate only, so it is off the plane exactly when the
			// plane's normal has a nonzero component on that axis.
			var q = axis switch
			{
				2 => new Vector3d(a.X, a.Y, a.Z + offset),
				1 => new Vector3d(a.X, a.Y + offset, a.Z),
				_ => new Vector3d(a.X + offset, a.Y, a.Z),
			};
			int orientation = RobustPredicates.Orient3D(a, b, c, q);
			if (orientation != 0)
			{
				return orientation * RobustPredicates.InSphere(a, b, c, q, p);
			}
		}

		throw new InvalidOperationException("Check failed: CoplanarInCircle needs non-collinear points.");
	}

	private int InsertLowDimension(in Vector3d point)
	{
		// Fold -0.0 into +0.0 so the key compares points geometrically.
		var key = new Vector3d(point.X == 0 ? 0.0 : point.X, point.Y == 0 ? 0.0 : point.Y, point.Z == 0 ? 0.0 : point.Z);
		if (pendingLookup!.TryGetValue(key, out int existing))
		{
			return existing;
		}

		int vertex = AllocateVertex(point);
		pendingLookup.Add(key, vertex);
		pendingVertices!.Add(vertex);
		var p = vertexPoints;
		switch (Dimension)
		{
			case -1:
				basis[0] = vertex;
				Dimension = 0;
				break;
			case 0:
				basis[1] = vertex;
				Dimension = 1;
				break;
			case 1:
				if (!AreCollinear(p[basis[0]], p[basis[1]], point))
				{
					basis[2] = vertex;
					Dimension = 2;
				}

				break;
			default:
				if (RobustPredicates.Orient3D(p[basis[0]], p[basis[1]], p[basis[2]], point) != 0)
				{
					basis[3] = vertex;
					BuildInitialTetrahedron();
				}

				break;
		}

		return vertex;
	}

	private static bool AreCollinear(in Vector3d a, in Vector3d b, in Vector3d c)
	{
		return RobustPredicates.Orient2D(a.X, a.Y, b.X, b.Y, c.X, c.Y) == 0
			&& RobustPredicates.Orient2D(a.Y, a.Z, b.Y, b.Z, c.Y, c.Z) == 0
			&& RobustPredicates.Orient2D(a.Z, a.X, b.Z, b.X, c.Z, c.X) == 0;
	}

	private void BuildInitialTetrahedron()
	{
		int a = basis[0], b = basis[1], c = basis[2], d = basis[3];
		if (RobustPredicates.Orient3D(vertexPoints[a], vertexPoints[b], vertexPoints[c], vertexPoints[d]) < 0)
		{
			(a, b) = (b, a);
		}

		Span<int> tet = [a, b, c, d];
		Span<int> cells = stackalloc int[5];
		cells[0] = AllocateCell();
		for (int i = 0; i < 4; ++i)
		{
			cellVertices[4 * cells[0] + i] = tet[i];
		}

		// Infinite cell i replaces vertex i by the infinite vertex; that flips which side of
		// the facet counts as positive, so swap two finite slots to restore the orientation.
		for (int i = 0; i < 4; ++i)
		{
			int cell = AllocateCell();
			cells[i + 1] = cell;
			for (int j = 0; j < 4; ++j)
			{
				cellVertices[4 * cell + j] = j == i ? InfiniteVertex : tet[j];
			}

			int s0 = (i + 1) & 3, s1 = (i + 2) & 3;
			(cellVertices[4 * cell + s0], cellVertices[4 * cell + s1]) = (cellVertices[4 * cell + s1], cellVertices[4 * cell + s0]);
		}

		// Link the five cells: two cells are neighbors across the facet made of their three
		// shared vertices.
		for (int x = 0; x < 5; ++x)
		{
			for (int y = 0; y < 5; ++y)
			{
				if (x == y)
				{
					continue;
				}

				for (int i = 0; i < 4; ++i)
				{
					int missing = cellVertices[4 * cells[x] + i];
					if (IndexOfVertex(cells[y], missing) < 0)
					{
						cellNeighbors[4 * cells[x] + i] = cells[y];
					}
				}
			}
		}

		for (int i = 0; i < 5; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				vertexCells[cellVertices[4 * cells[i] + j]] = cells[i];
			}
		}

		lastCell = cells[0];
		Dimension = 3;
		hullFacetIndex = null;

		var pending = pendingVertices!;
		pendingVertices = null;
		pendingLookup = null;
		foreach (int vertex in pending)
		{
			if (vertex == a || vertex == b || vertex == c || vertex == d)
			{
				continue;
			}

			InsertInCavity(vertex, Locate(vertexPoints[vertex], lastCell, ref walkState));
		}
	}

	/// <summary>
	/// Bowyer-Watson step: collects the cells in conflict with the new vertex, starting from
	/// the located cell (which always conflicts), removes them and fans the cavity boundary
	/// to the vertex.
	/// </summary>
	private void InsertInCavity(int vertex, int startCell)
	{
		hullFacetIndex = null;
		var point = vertexPoints[vertex];
		++markStamp;
		int conflictMark = 2 * markStamp + 1, visitedMark = 2 * markStamp;
		conflictCells.Clear();
		boundaryFacets.Clear();
		cavityStack.Clear();

		cellMarks[startCell] = conflictMark;
		conflictCells.Add(startCell);
		cavityStack.Add(startCell);
		while (cavityStack.Count > 0)
		{
			int cell = cavityStack[^1];
			cavityStack.RemoveAt(cavityStack.Count - 1);
			for (int i = 0; i < 4; ++i)
			{
				int neighbor = cellNeighbors[4 * cell + i];
				int mark = cellMarks[neighbor];
				if (mark == conflictMark)
				{
					continue;
				}

				if (mark != visitedMark && InConflict(neighbor, point))
				{
					cellMarks[neighbor] = conflictMark;
					conflictCells.Add(neighbor);
					cavityStack.Add(neighbor);
				}
				else
				{
					cellMarks[neighbor] = visitedMark;
					boundaryFacets.Add(4 * cell + i);
				}
			}
		}

		// Copy the new cells' vertices before the conflict cells are freed and reused.
		int count = boundaryFacets.Count;
		if (newCellVertices.Length < 4 * count)
		{
			newCellVertices = new int[8 * count];
		}

		for (int f = 0; f < count; ++f)
		{
			int cell = boundaryFacets[f] >> 2, i = boundaryFacets[f] & 3;
			for (int j = 0; j < 4; ++j)
			{
				newCellVertices[4 * f + j] = j == i ? vertex : cellVertices[4 * cell + j];
			}
		}

		for (int f = 0; f < count; ++f)
		{
			// The outside neighbor, and its slot pointing back into the cavity, are read now:
			// the neighbor is not in conflict, so it survives.
			int cell = boundaryFacets[f] >> 2, i = boundaryFacets[f] & 3;
			int outside = cellNeighbors[4 * cell + i];
			boundaryFacets[f] = 4 * outside + IndexOfNeighbor(outside, cell);
		}

		foreach (int cell in conflictCells)
		{
			FreeCell(cell);
		}

		openEdges.Clear();
		for (int f = 0; f < count; ++f)
		{
			int cell = AllocateCell();
			int b = 4 * cell;
			int vertexSlot = -1;
			for (int j = 0; j < 4; ++j)
			{
				int v = newCellVertices[4 * f + j];
				cellVertices[b + j] = v;
				vertexCells[v] = cell;
				if (v == vertex)
				{
					vertexSlot = j;
				}
			}

			int outside = boundaryFacets[f] >> 2, outsideSlot = boundaryFacets[f] & 3;
			cellNeighbors[b + vertexSlot] = outside;
			cellNeighbors[4 * outside + outsideSlot] = cell;

			// The other three facets contain the new vertex and one edge of the boundary
			// facet; the new cell sharing that edge is the neighbor across it.
			for (int j = 0; j < 4; ++j)
			{
				if (j == vertexSlot)
				{
					continue;
				}

				int u = -1, w = -1;
				for (int m = 0; m < 4; ++m)
				{
					if (m == j || m == vertexSlot)
					{
						continue;
					}

					if (u < 0)
					{
						u = cellVertices[b + m];
					}
					else
					{
						w = cellVertices[b + m];
					}
				}

				long key = u < w ? ((long)u << 32) | (uint)w : ((long)w << 32) | (uint)u;
				if (openEdges.Remove(key, out int other))
				{
					cellNeighbors[b + j] = other >> 2;
					cellNeighbors[other] = cell;
				}
				else
				{
					openEdges.Add(key, b + j);
				}
			}
		}

		Check.Eq(openEdges.Count, 0);
		lastCell = vertexCells[vertex];
	}
}
