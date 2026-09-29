// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// TriangleBvh (continued): the closest point on the mesh to a query point. C#-only, not a
// COLMAP port; the reconstruction benchmark (Mvs/Testing/Benchmark) uses it for Chamfer
// distances and F-scores. The tree and its segment queries are in TriangleBvh.cs.
//
// Written from the published algorithm: the point-triangle closest point by Voronoi regions
// of C. Ericson, "Real-Time Collision Detection", Morgan Kaufmann 2005, section 5.1.5.
// Traversal is depth first, nearer child first, pruning any node whose (padded, outward
// rounded) box is at least the best distance found so far away.
//
// Determinism: the returned distance is the minimum over all triangles of the per-triangle
// distance, so like AnyHit it does not depend on the tree's shape. When two triangles are
// exactly equally close (a shared edge or corner), the one reported is the first in the tree's
// fixed traversal order, which does not depend on the thread count either.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Mvs;

public sealed partial class TriangleBvh
{
	/// <summary>
	/// The closest point to <paramref name="point"/> on any triangle: returns its distance, with
	/// the point in <paramref name="closest"/> and the triangle's id in
	/// <paramref name="triangleId"/>. An empty tree returns +infinity and id -1. Thread-safe;
	/// allocates nothing.
	/// </summary>
	public double ClosestPoint(Vector3d point, out Vector3d closest, out int triangleId)
	{
		closest = point;
		triangleId = -1;
		if (ids.Length == 0)
		{
			return double.PositiveInfinity;
		}

		double px = point.X, py = point.Y, pz = point.Z;
		double best = double.PositiveInfinity;
		double bx = 0, by = 0, bz = 0;
		int bestTri = -1;

		Span<int> stack = stackalloc int[StackSize];
		int top = 0;
		stack[top++] = 0;
		while (top > 0)
		{
			int node = stack[--top];
			if (BoxDistanceSquared(node, px, py, pz) >= best)
			{
				continue;
			}

			int n = nodeCount[node];
			if (n < 0)
			{
				int left = node + 1;
				int right = nodeFirst[node];
				double dl = BoxDistanceSquared(left, px, py, pz);
				double dr = BoxDistanceSquared(right, px, py, pz);

				// Push the farther child first so the nearer is searched first; ties go left.
				if (dl <= dr)
				{
					stack[top++] = right;
					stack[top++] = left;
				}
				else
				{
					stack[top++] = left;
					stack[top++] = right;
				}

				continue;
			}

			int first = nodeFirst[node];
			for (int i = first; i < first + n; i++)
			{
				ClosestOnTriangle(i, px, py, pz, out double cx, out double cy, out double cz);
				double dx = cx - px, dy = cy - py, dz = cz - pz;
				double d2 = dx * dx + dy * dy + dz * dz;
				if (d2 < best)
				{
					best = d2;
					bx = cx;
					by = cy;
					bz = cz;
					bestTri = i;
				}
			}
		}

		closest = new Vector3d(bx, by, bz);
		triangleId = ids[bestTri];
		return Math.Sqrt(best);
	}

	// Squared distance from the point to a node's box (0 inside it).
	private double BoxDistanceSquared(int node, double px, double py, double pz)
	{
		int b = 6 * node;
		double dx = Outside(px, nodeBounds[b], nodeBounds[b + 3]);
		double dy = Outside(py, nodeBounds[b + 1], nodeBounds[b + 4]);
		double dz = Outside(pz, nodeBounds[b + 2], nodeBounds[b + 5]);
		return dx * dx + dy * dy + dz * dz;
	}

	private static double Outside(double p, double lo, double hi) => p < lo ? lo - p : p > hi ? p - hi : 0.0;

	// Ericson's ClosestPtPointTriangle: find the Voronoi region of the triangle (a corner, an
	// edge or the face) the point projects into, and project onto that feature.
	private void ClosestOnTriangle(int tri, double px, double py, double pz, out double cx, out double cy, out double cz)
	{
		int c = TriStride * tri;
		double ax = tris[c], ay = tris[c + 1], az = tris[c + 2];
		double abx = tris[c + 3], aby = tris[c + 4], abz = tris[c + 5];
		double acx = tris[c + 6], acy = tris[c + 7], acz = tris[c + 8];

		// Corner A.
		double apx = px - ax, apy = py - ay, apz = pz - az;
		double d1 = abx * apx + aby * apy + abz * apz;
		double d2 = acx * apx + acy * apy + acz * apz;
		if (d1 <= 0 && d2 <= 0)
		{
			(cx, cy, cz) = (ax, ay, az);
			return;
		}

		// Corner B.
		double bpx = apx - abx, bpy = apy - aby, bpz = apz - abz;
		double d3 = abx * bpx + aby * bpy + abz * bpz;
		double d4 = acx * bpx + acy * bpy + acz * bpz;
		if (d3 >= 0 && d4 <= d3)
		{
			(cx, cy, cz) = (ax + abx, ay + aby, az + abz);
			return;
		}

		// Edge AB.
		double vc = d1 * d4 - d3 * d2;
		if (vc <= 0 && d1 >= 0 && d3 <= 0)
		{
			double v = d1 / (d1 - d3);
			(cx, cy, cz) = (ax + v * abx, ay + v * aby, az + v * abz);
			return;
		}

		// Corner C.
		double cpx = apx - acx, cpy = apy - acy, cpz = apz - acz;
		double d5 = abx * cpx + aby * cpy + abz * cpz;
		double d6 = acx * cpx + acy * cpy + acz * cpz;
		if (d6 >= 0 && d5 <= d6)
		{
			(cx, cy, cz) = (ax + acx, ay + acy, az + acz);
			return;
		}

		// Edge AC.
		double vb = d5 * d2 - d1 * d6;
		if (vb <= 0 && d2 >= 0 && d6 <= 0)
		{
			double w = d2 / (d2 - d6);
			(cx, cy, cz) = (ax + w * acx, ay + w * acy, az + w * acz);
			return;
		}

		// Edge BC.
		double va = d3 * d6 - d5 * d4;
		if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
		{
			double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
			double bx = ax + abx, by = ay + aby, bz = az + abz;
			(cx, cy, cz) = (bx + w * (acx - abx), by + w * (acy - aby), bz + w * (acz - abz));
			return;
		}

		// Inside the face. Rounding can bring a degenerate (zero-area) triangle here with a zero
		// denominator; its corner A is then as good an answer as any point of it.
		double sum = va + vb + vc;
		if (!(sum != 0))
		{
			(cx, cy, cz) = (ax, ay, az);
			return;
		}

		double denom = 1.0 / sum;
		double vf = vb * denom;
		double wf = vc * denom;
		(cx, cy, cz) = (ax + abx * vf + acx * wf, ay + aby * vf + acy * wf, az + abz * vf + acz * wf);
	}
}
