// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// DelaunayMeshingWeights: ports of DelaunayMeshingEdgeWeightComputer and
// DelaunayTriangulationRayCaster from colmap/mvs/delaunay_meshing.cc - the visibility and
// distance probabilities that weight the s-t graph, and the ray caster that finds the facets
// a camera-to-point viewing ray crosses. Both run on Geometry/Delaunay's
// DelaunayTriangulation3 (entry 103), whose API mirrors the CGAL calls COLMAP makes:
// triangulation.triangle(c, i) is FacetVertices (oriented so its normal points into c, CGAL's
// documented convention), CGAL::orientation is RobustPredicates.Orient3D, and
// CGAL::intersection(segment, triangle) is TryIntersectSegmentTriangle.
// Neighbors: DelaunayMeshingInput.cs builds the triangulation. Tests:
// ColmapSharp.Tests/Mvs/DelaunayMeshingTests.CSharpOnly.cs.
//
// Tier C (outcome): the facets found are exact for rays in general position; the
// intersection points (and so the distances) are double constructions like CGAL Epick's but
// not necessarily the same formula. Hull facets are looked up through HullFacetIndex and
// tested in COLMAP's scan order, which accepts the same facet as COLMAP's linear scan.

using System.Runtime.InteropServices;

using ColmapSharp.Geometry.Delaunay;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

namespace ColmapSharp.Mvs;

/// <summary>
/// Port of DelaunayMeshingEdgeWeightComputer: Gaussian-shaped visibility and distance
/// probabilities, with the distance sigma taken from the 25th percentile edge length.
/// </summary>
internal sealed class DelaunayMeshingEdgeWeightComputer
{
	private readonly double visibilityThreshold;
	private readonly double visibilityNormalization;
	private readonly double distanceThreshold;
	private readonly double distanceNormalization;

	public DelaunayMeshingEdgeWeightComputer(DelaunayTriangulation3 triangulation, double visibilitySigma,
		double distanceSigmaFactor)
	{
		visibilityThreshold = 5 * visibilitySigma;
		visibilityNormalization = -0.5 / (visibilitySigma * visibilitySigma);

		var edgeLengths = new List<float>();
		foreach (var (v0, v1) in triangulation.FiniteEdges())
		{
			var p0 = triangulation.Point(v0);
			var p1 = triangulation.Point(v1);
			double dx = p0.X - p1.X, dy = p0.Y - p1.Y, dz = p0.Z - p1.Z;
			edgeLengths.Add((float)(dx * dx + dy * dy + dz * dz));
		}

		DistanceSigma = distanceSigmaFactor
			* Math.Max(Math.Sqrt(MathUtils.Percentile<float>(CollectionsMarshal.AsSpan(edgeLengths), 25)), 1e-7);
		distanceThreshold = 5 * DistanceSigma;
		distanceNormalization = -0.5 / (DistanceSigma * DistanceSigma);
	}

	/// <summary>The distance sigma: factor times the 25th percentile edge length.</summary>
	public double DistanceSigma { get; }

	public double ComputeVisibilityProb(double visibilitySquared)
	{
		if (visibilitySquared < visibilityThreshold)
		{
			return Math.Max(0.0, 1.0 - Math.Exp(visibilitySquared * visibilityNormalization));
		}

		return 1.0;
	}

	public double ComputeDistanceProb(double distanceSquared)
	{
		if (distanceSquared < distanceThreshold)
		{
			return Math.Max(0.0, 1.0 - Math.Exp(distanceSquared * distanceNormalization));
		}

		return 1.0;
	}
}

/// <summary>
/// Port of DelaunayTriangulationRayCaster. The tracing locates the cell of the ray origin
/// and then iteratively intersects the ray with all facets of the current cell and advances
/// to the neighboring cell of the intersected facet. Note that the ray can also pass
/// through outside of the hull of the triangulation, i.e. lie within the infinite
/// cells/facets. The ray caster collects the intersected facets along the ray.
/// </summary>
internal sealed class DelaunayTriangulationRayCaster
{
	/// <summary>An intersected facet (cell, index) and the squared distance of the hit to the target.</summary>
	public readonly record struct Intersection(int Cell, int Index, double TargetDistanceSquared);

	private readonly DelaunayTriangulation3 triangulation;
	private readonly HullFacetIndex hullFacetIndex;

	public DelaunayTriangulationRayCaster(DelaunayTriangulation3 triangulation)
	{
		this.triangulation = triangulation;
		hullFacetIndex = triangulation.GetHullFacetIndex();
	}

	/// <summary>COLMAP's hull_facets_: every finite facet of an infinite cell, in cell-handle order.</summary>
	public IReadOnlyList<(int Cell, int Index)> HullFacets => hullFacetIndex.Facets;

	/// <summary>
	/// Collects the facets the segment start-&gt;end crosses. Safe to call from several threads
	/// at once, each with its own locate cursor (the result depends only on that cursor).
	/// </summary>
	public void CastRaySegment(Vector3d start, Vector3d end, List<Intersection> intersections,
		ref DelaunayTriangulation3.LocateCursor cursor)
	{
		intersections.Clear();
		List<int>? candidates = null;

		int nextCell = triangulation.Locate(start, ref cursor);

		bool nextCellFound = true;
		while (nextCellFound)
		{
			nextCellFound = false;

			if (triangulation.IsInfinite(nextCell))
			{
				// COLMAP linearly checks all hull facets for intersection. A facet the segment
				// does not come near cannot pass the intersection test, so only the index's
				// candidates are checked, in the same order: the first accepted facet is the
				// one the full scan would accept.
				candidates ??= [];
				hullFacetIndex.QuerySegment(start, end, candidates);
				foreach (int f in candidates)
				{
					var (cell, index) = hullFacetIndex.Facets[f];
					if (TryCross(cell, index, start, end, intersections))
					{
						nextCell = triangulation.Neighbor(cell, index);
						nextCellFound = true;
						break;
					}
				}
			}
			else
			{
				// Check all neighboring finite facets for intersection.
				int cell = nextCell;
				for (int i = 0; i < 4; ++i)
				{
					if (TryCross(cell, i, start, end, intersections))
					{
						nextCell = triangulation.Neighbor(cell, i);
						nextCellFound = true;
						break;
					}
				}
			}
		}
	}

	// One facet test of CastRaySegment: the origin not behind the facet, the segment hitting
	// it, and the hit closer to the target than the previous one.
	private bool TryCross(int cell, int index, Vector3d start, Vector3d end, List<Intersection> intersections)
	{
		var (a, b, c) = triangulation.FacetVertices(cell, index);
		var t0 = triangulation.Point(a);
		var t1 = triangulation.Point(b);
		var t2 = triangulation.Point(c);

		// Check if the ray origin is infront of the facet.
		if (RobustPredicates.Orient3D(t0, t1, t2, start) < 0)
		{
			return false;
		}

		// Check if the segment intersects the facet.
		if (!DelaunayTriangulation3.TryIntersectSegmentTriangle(start, end, t0, t1, t2, out var intersectionPoint))
		{
			return false;
		}

		// Make sure the next intersection is closer to target than previous.
		double dx = intersectionPoint.X - end.X, dy = intersectionPoint.Y - end.Y, dz = intersectionPoint.Z - end.Z;
		double targetDistanceSquared = dx * dx + dy * dy + dz * dz;
		if (intersections.Count > 0 && intersections[^1].TargetDistanceSquared < targetDistanceSquared)
		{
			return false;
		}

		intersections.Add(new Intersection(cell, index, targetDistanceSquared));
		return true;
	}
}
