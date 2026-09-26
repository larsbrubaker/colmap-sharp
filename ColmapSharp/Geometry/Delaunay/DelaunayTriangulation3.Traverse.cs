// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DelaunayTriangulation3.Traverse: the facets a line segment crosses, walked cell to cell
// (a "straight walk", Devillers, Pion and Teillaud 2002). COLMAP's Delaunay meshing casts a
// viewing ray from each camera to each observed point and accumulates weights on the facets
// it crosses (DelaunayTriangulationRayCaster, ported 1:1 in Mvs/DelaunayMeshingWeights.cs on
// top of Locate, Orient3D and TryIntersectSegmentTriangle below). TraverseSegment is the
// straight-walk equivalent with exact decisions; the tests use it to cross-check the caster.
// TryIntersectRayTriangle serves the meshing's "facet behind the point" step.
//
// All decisions go through RobustPredicates, so for a segment in general position (not
// through a vertex or an edge, endpoints not on a facet plane) the result is exactly the set
// of finite facets the open segment crosses, in order. In degenerate positions the walk
// still terminates and picks the first qualifying facet in slot order.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Geometry.Delaunay;

public sealed partial class DelaunayTriangulation3
{
	// Built lazily for segments that start outside the hull; any insertion clears it.
	private HullFacetIndex? hullFacetIndex;

	/// <summary>
	/// Collects, in order from source to target, the finite facets the segment crosses. Each
	/// entry is (cell, i): the segment leaves that cell through its facet i. When the source is
	/// outside the convex hull the first entry is the hull facet it enters through, seen from
	/// the infinite cell; when the target is outside, the last is the hull facet it leaves by.
	/// Requires Dimension == 3.
	/// </summary>
	public void TraverseSegment(in Vector3d source, in Vector3d target, List<(int Cell, int Index)> crossedFacets)
	{
		crossedFacets.Clear();
		int cell = Locate(source);
		int entry = -1;
		if (IsInfinite(cell))
		{
			// The hull is convex, so the segment enters it through at most one hull facet;
			// HullFacetIndex narrows the search to the facets near the segment.
			int hullCell = FindEnteredHullFacet(source, target);
			if (hullCell < 0)
			{
				return;
			}

			int k = InfiniteIndex(hullCell);
			crossedFacets.Add((hullCell, k));
			cell = Neighbor(hullCell, k);
			entry = IndexOfNeighbor(cell, hullCell);
		}

		while (true)
		{
			int exit = -1;
			for (int i = 0; i < 4; ++i)
			{
				if (i != entry && OrientReplaced(cell, i, target) < 0 && LinePassesThroughFacet(cell, i, source, target))
				{
					exit = i;
					break;
				}
			}

			if (exit < 0)
			{
				// The target is in this cell.
				return;
			}

			crossedFacets.Add((cell, exit));
			int next = Neighbor(cell, exit);
			if (IsInfinite(next))
			{
				// Left the convex hull; a segment cannot come back into a convex set.
				return;
			}

			entry = IndexOfNeighbor(next, cell);
			cell = next;
		}
	}

	/// <summary>
	/// Whether the open segment (a, b) crosses the triangle (u, v, w): a and b strictly on
	/// opposite sides of its plane and the line through them meeting the closed triangle.
	/// </summary>
	public static bool SegmentCrossesTriangle(in Vector3d a, in Vector3d b, in Vector3d u, in Vector3d v, in Vector3d w)
	{
		int sideA = RobustPredicates.Orient3D(u, v, w, a);
		int sideB = RobustPredicates.Orient3D(u, v, w, b);
		return sideA * sideB < 0 && LinePassesThroughTriangle(a, b, u, v, w);
	}

	/// <summary>
	/// Intersection point of the closed segment [a, b] with the closed triangle (u, v, w),
	/// like CGAL::intersection(Segment_3, Triangle_3) assigned to a Point_3: whether they meet
	/// is decided exactly; the point is an inexact double construction (an endpoint exactly
	/// when that endpoint lies on the triangle). A segment in the triangle's plane returns
	/// false: CGAL returns a segment there (which COLMAP's assign rejects), except when the two
	/// only touch at one point, a case we also report as no intersection (entry 103).
	/// </summary>
	public static bool TryIntersectSegmentTriangle(in Vector3d a, in Vector3d b, in Vector3d u, in Vector3d v,
		in Vector3d w, out Vector3d point)
	{
		point = default;
		int sideA = RobustPredicates.Orient3D(u, v, w, a);
		int sideB = RobustPredicates.Orient3D(u, v, w, b);
		if ((sideA == 0 && sideB == 0) || sideA * sideB > 0 || !LinePassesThroughTriangle(a, b, u, v, w))
		{
			return false;
		}

		if (sideA == 0)
		{
			point = a;
			return true;
		}

		if (sideB == 0)
		{
			point = b;
			return true;
		}

		point = LinePlanePoint(a, b, u, v, w);
		return true;
	}

	/// <summary>
	/// Intersection point of the ray from <paramref name="source"/> through
	/// <paramref name="through"/> with the closed triangle (u, v, w), like
	/// CGAL::intersection(Ray_3, Triangle_3) assigned to a Point_3. Whether they meet (and that
	/// the hit is at or after the source) is decided exactly; the point is a double
	/// construction. A ray in the triangle's plane returns false, as for segments.
	/// </summary>
	public static bool TryIntersectRayTriangle(in Vector3d source, in Vector3d through, in Vector3d u, in Vector3d v,
		in Vector3d w, out Vector3d point)
	{
		point = default;
		int sideSource = RobustPredicates.Orient3D(u, v, w, source);
		int direction = RobustPredicates.OrientDirection(u, v, w, source, through);
		if (direction == 0 || (sideSource != 0 && sideSource == direction))
		{
			// Parallel to the plane (in it, or never reaching it), or heading away from it.
			return false;
		}

		if (!LinePassesThroughTriangle(source, through, u, v, w))
		{
			return false;
		}

		if (sideSource == 0)
		{
			point = source;
			return true;
		}

		if (RobustPredicates.Orient3D(u, v, w, through) == 0)
		{
			point = through;
			return true;
		}

		point = LinePlanePoint(source, through, u, v, w);
		return true;
	}

	// The point where the line through a and b meets the plane of (u, v, w), in doubles.
	private static Vector3d LinePlanePoint(in Vector3d a, in Vector3d b, in Vector3d u, in Vector3d v, in Vector3d w)
	{
		// Signed distances (times the normal's length) of a and b from the plane.
		double e1x = v.X - u.X, e1y = v.Y - u.Y, e1z = v.Z - u.Z;
		double e2x = w.X - u.X, e2y = w.Y - u.Y, e2z = w.Z - u.Z;
		double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
		double da = nx * (a.X - u.X) + ny * (a.Y - u.Y) + nz * (a.Z - u.Z);
		double db = nx * (b.X - u.X) + ny * (b.Y - u.Y) + nz * (b.Z - u.Z);
		double t = da / (da - db);
		return new Vector3d(a.X + t * (b.X - a.X), a.Y + t * (b.Y - a.Y), a.Z + t * (b.Z - a.Z));
	}

	private int FindEnteredHullFacet(in Vector3d source, in Vector3d target)
	{
		// Only facets whose box the segment touches can be crossed; test those in hull order.
		var index = GetHullFacetIndex();
		var candidates = new List<int>();
		index.QuerySegment(source, target, candidates);
		foreach (int f in candidates)
		{
			var (c, k) = index.Facets[f];

			// The source must be strictly outside this hull facet and the target not.
			if (OrientReplaced(c, k, source) > 0 && OrientReplaced(c, k, target) < 0
				&& LinePassesThroughFacet(c, k, source, target))
			{
				return c;
			}
		}

		return -1;
	}

	/// <summary>
	/// The hull facets with a segment index, built on first use after the last insertion
	/// (safe to call from concurrent readers; at worst two threads build the same index).
	/// </summary>
	public HullFacetIndex GetHullFacetIndex()
	{
		var index = Volatile.Read(ref hullFacetIndex);
		if (index is null)
		{
			index = new HullFacetIndex(this);
			Volatile.Write(ref hullFacetIndex, index);
		}

		return index;
	}

	private bool LinePassesThroughFacet(int cell, int i, in Vector3d a, in Vector3d b)
	{
		var (u, v, w) = FacetVertices(cell, i);
		return LinePassesThroughTriangle(a, b, vertexPoints[u], vertexPoints[v], vertexPoints[w]);
	}

	// The line through a and b meets the closed triangle (u, v, w) exactly when the three
	// signed volumes it spans with the triangle's edges do not have mixed signs.
	private static bool LinePassesThroughTriangle(in Vector3d a, in Vector3d b, in Vector3d u, in Vector3d v, in Vector3d w)
	{
		int s1 = RobustPredicates.Orient3D(a, b, u, v);
		int s2 = RobustPredicates.Orient3D(a, b, v, w);
		int s3 = RobustPredicates.Orient3D(a, b, w, u);
		return (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0);
	}
}
