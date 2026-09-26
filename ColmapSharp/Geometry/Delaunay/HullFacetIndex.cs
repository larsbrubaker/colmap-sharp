// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// HullFacetIndex: the convex-hull facets of a DelaunayTriangulation3 (the finite facet of
// every infinite cell) in cell-handle order, with a bounding-volume hierarchy over their
// boxes so a segment finds the few facets it can touch without scanning the hull.
// COLMAP's DelaunayTriangulationRayCaster scans every hull facet for each ray that starts
// outside the hull (Mvs/DelaunayMeshingWeights.cs keeps that logic); with this index it only
// tests the candidates, in the same scan order, so the first facet accepted is the one the
// full scan would accept. DelaunayTriangulation3.TraverseSegment uses it the same way.
//
// The hierarchy is a plain median split on the longest axis of the centroid boxes (any
// textbook BVH, e.g. Ericson, "Real-Time Collision Detection", 2005, ch. 6). The box test is
// conservative (boxes are padded, and the slab test accepts touching), so a facet the exact
// predicates would report is never missed; false candidates are rejected by the caller's
// exact tests. Immutable after construction, so safe to share between threads.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Geometry.Delaunay;

/// <summary>
/// Hull facets of a triangulation in cell-handle order, indexed for segment queries.
/// </summary>
public sealed class HullFacetIndex
{
	private const int LeafSize = 4;

	private readonly List<(int Cell, int Index)> facets = [];

	// Per facet: padded box as min x, y, z, max x, y, z.
	private readonly double[] facetBoxes;

	// Nodes: box (6 doubles), then for a leaf: start and count in `order` with count > 0;
	// for an inner node: left and right child with count == 0.
	private readonly List<double> nodeBoxes = [];
	private readonly List<int> nodeFirst = [];
	private readonly List<int> nodeSecond = [];
	private readonly List<int> nodeCount = [];
	private readonly int[] order;

	/// <summary>Collects the hull facets of the triangulation and builds the hierarchy.</summary>
	public HullFacetIndex(DelaunayTriangulation3 triangulation)
	{
		foreach (int cell in triangulation.AllCells())
		{
			int k = triangulation.InfiniteIndex(cell);
			if (k >= 0)
			{
				facets.Add((cell, k));
			}
		}

		double maxAbs = 1;
		facetBoxes = new double[6 * facets.Count];
		for (int f = 0; f < facets.Count; ++f)
		{
			var (a, b, c) = triangulation.FacetVertices(facets[f].Cell, facets[f].Index);
			var pa = triangulation.Point(a);
			var pb = triangulation.Point(b);
			var pc = triangulation.Point(c);
			facetBoxes[6 * f] = Math.Min(pa.X, Math.Min(pb.X, pc.X));
			facetBoxes[6 * f + 1] = Math.Min(pa.Y, Math.Min(pb.Y, pc.Y));
			facetBoxes[6 * f + 2] = Math.Min(pa.Z, Math.Min(pb.Z, pc.Z));
			facetBoxes[6 * f + 3] = Math.Max(pa.X, Math.Max(pb.X, pc.X));
			facetBoxes[6 * f + 4] = Math.Max(pa.Y, Math.Max(pb.Y, pc.Y));
			facetBoxes[6 * f + 5] = Math.Max(pa.Z, Math.Max(pb.Z, pc.Z));
			for (int j = 0; j < 6; ++j)
			{
				maxAbs = Math.Max(maxAbs, Math.Abs(facetBoxes[6 * f + j]));
			}
		}

		// Pad far beyond the rounding of the slab arithmetic.
		double pad = 1e-9 * maxAbs;
		for (int f = 0; f < facets.Count; ++f)
		{
			for (int j = 0; j < 3; ++j)
			{
				facetBoxes[6 * f + j] -= pad;
				facetBoxes[6 * f + 3 + j] += pad;
			}
		}

		order = new int[facets.Count];
		for (int f = 0; f < order.Length; ++f)
		{
			order[f] = f;
		}

		if (facets.Count > 0)
		{
			Build(0, facets.Count);
		}
	}

	/// <summary>The hull facets as (infinite cell, slot of its infinite vertex), in cell-handle order.</summary>
	public IReadOnlyList<(int Cell, int Index)> Facets => facets;

	/// <summary>
	/// Fills <paramref name="candidates"/> with the indices into <see cref="Facets"/> of every
	/// facet whose box the closed segment [a, b] may touch, in ascending order.
	/// </summary>
	public void QuerySegment(in Vector3d a, in Vector3d b, List<int> candidates)
	{
		candidates.Clear();
		if (facets.Count == 0)
		{
			return;
		}

		Span<int> stack = stackalloc int[128];
		int top = 0;
		stack[top++] = 0;
		while (top > 0)
		{
			int node = stack[--top];
			if (!SegmentTouchesBox(a, b, nodeBoxes, 6 * node))
			{
				continue;
			}

			if (nodeCount[node] > 0)
			{
				for (int k = nodeFirst[node]; k < nodeFirst[node] + nodeCount[node]; ++k)
				{
					if (SegmentTouchesBox(a, b, facetBoxes, 6 * order[k]))
					{
						candidates.Add(order[k]);
					}
				}
			}
			else
			{
				stack[top++] = nodeFirst[node];
				stack[top++] = nodeSecond[node];
			}
		}

		candidates.Sort();
	}

	private int Build(int start, int count)
	{
		int node = nodeCount.Count;
		nodeFirst.Add(start);
		nodeSecond.Add(0);
		nodeCount.Add(count);
		double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
		double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;
		for (int k = start; k < start + count; ++k)
		{
			int b = 6 * order[k];
			minX = Math.Min(minX, facetBoxes[b]);
			minY = Math.Min(minY, facetBoxes[b + 1]);
			minZ = Math.Min(minZ, facetBoxes[b + 2]);
			maxX = Math.Max(maxX, facetBoxes[b + 3]);
			maxY = Math.Max(maxY, facetBoxes[b + 4]);
			maxZ = Math.Max(maxZ, facetBoxes[b + 5]);
		}

		nodeBoxes.AddRange([minX, minY, minZ, maxX, maxY, maxZ]);
		if (count <= LeafSize)
		{
			return node;
		}

		// Split at the median centroid along the longest axis; ties by facet index keep the
		// build deterministic.
		int axis = maxX - minX >= maxY - minY ? (maxX - minX >= maxZ - minZ ? 0 : 2) : (maxY - minY >= maxZ - minZ ? 1 : 2);
		Array.Sort(order, start, count, Comparer<int>.Create((f, g) =>
		{
			double cf = facetBoxes[6 * f + axis] + facetBoxes[6 * f + 3 + axis];
			double cg = facetBoxes[6 * g + axis] + facetBoxes[6 * g + 3 + axis];
			int c = cf.CompareTo(cg);
			return c != 0 ? c : f.CompareTo(g);
		}));
		int half = count / 2;
		int left = Build(start, half);
		int right = Build(start + half, count - half);
		nodeFirst[node] = left;
		nodeSecond[node] = right;
		nodeCount[node] = 0;
		return node;
	}

	// Slab test of the closed segment against a box, with a little slack in the parameter so
	// rounding never rejects a touching segment.
	private static bool SegmentTouchesBox(in Vector3d a, in Vector3d b, IReadOnlyList<double> boxes, int offset)
	{
		const double slack = 1e-9;
		double tMin = -slack, tMax = 1 + slack;
		for (int axis = 0; axis < 3; ++axis)
		{
			double origin = axis == 0 ? a.X : axis == 1 ? a.Y : a.Z;
			double end = axis == 0 ? b.X : axis == 1 ? b.Y : b.Z;
			double min = boxes[offset + axis], max = boxes[offset + 3 + axis];
			double delta = end - origin;
			if (delta == 0)
			{
				if (origin < min || origin > max)
				{
					return false;
				}

				continue;
			}

			double t1 = (min - origin) / delta, t2 = (max - origin) / delta;
			if (t1 > t2)
			{
				(t1, t2) = (t2, t1);
			}

			tMin = Math.Max(tMin, t1);
			tMax = Math.Min(tMax, t2);
			if (tMin > tMax + slack)
			{
				return false;
			}
		}

		return true;
	}
}
