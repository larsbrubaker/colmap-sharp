// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PointKdTree: nearest-neighbor distance to a point cloud, for the reconstruction benchmark's
// completeness when a run produced a fused cloud but no mesh (docs/QUALITY_PLAN.md, stage 0b).
// Not a COLMAP port; written from the textbook algorithm (J. L. Bentley, "Multidimensional
// binary search trees used for associative searching", CACM 18(9), 1975): median splits on the
// widest axis, then a depth-first query that visits the near side first and prunes the far
// side by the splitting plane. The answer (the minimum distance) does not depend on the tree's
// shape. Its mesh counterpart is TriangleBvh.ClosestPoint.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Mvs.Testing.Benchmark;

/// <summary>A static kd-tree over 3D points answering nearest-distance queries.</summary>
public sealed class PointKdTree
{
	private const int LeafSize = 8;

	private readonly Vector3d[] points;

	// Node layout, built recursively over points[start, end): a leaf when end - start is at
	// most LeafSize, else split at mid = (start + end) / 2 on axis[node]; children are implicit
	// by recursion on the same ranges, so only the axis and plane of each range's split are
	// stored, keyed by the range midpoint (unique per inner range). The plane must be stored:
	// building the children re-sorts their ranges, so points[mid] no longer holds it afterwards.
	private readonly sbyte[] splitAxis;
	private readonly double[] splitValue;

	/// <summary>Builds the tree over a copy of <paramref name="cloud"/>.</summary>
	public PointKdTree(IReadOnlyList<Vector3d> cloud)
	{
		points = [.. cloud];
		splitAxis = new sbyte[points.Length];
		splitValue = new double[points.Length];
		Build(0, points.Length);
	}

	/// <summary>Number of points.</summary>
	public int Count => points.Length;

	/// <summary>Distance from <paramref name="query"/> to the nearest point (+infinity when empty).</summary>
	public double NearestDistance(Vector3d query)
	{
		double best = double.PositiveInfinity;
		Search(0, points.Length, query, ref best);
		return Math.Sqrt(best);
	}

	private void Build(int start, int end)
	{
		if (end - start <= LeafSize)
		{
			return;
		}

		Vector3d min = points[start], max = points[start];
		for (int i = start + 1; i < end; i++)
		{
			Vector3d p = points[i];
			min = new Vector3d(Math.Min(min.X, p.X), Math.Min(min.Y, p.Y), Math.Min(min.Z, p.Z));
			max = new Vector3d(Math.Max(max.X, p.X), Math.Max(max.Y, p.Y), Math.Max(max.Z, p.Z));
		}

		Vector3d extent = max - min;
		int axis = extent.X >= extent.Y && extent.X >= extent.Z ? 0 : extent.Y >= extent.Z ? 1 : 2;

		// A full sort of the range, ties broken by the coordinates of the other axes, so the
		// layout is the same on every run (Array.Sort is unstable).
		points.AsSpan(start, end - start).Sort((a, b) =>
		{
			int c = a[axis].CompareTo(b[axis]);
			if (c != 0)
			{
				return c;
			}

			c = a.X.CompareTo(b.X);
			return c != 0 ? c : (c = a.Y.CompareTo(b.Y)) != 0 ? c : a.Z.CompareTo(b.Z);
		});

		int mid = (start + end) / 2;
		splitAxis[mid] = (sbyte)axis;
		splitValue[mid] = points[mid][axis];
		Build(start, mid);
		Build(mid, end);
	}

	private void Search(int start, int end, Vector3d q, ref double best)
	{
		if (end - start <= LeafSize)
		{
			for (int i = start; i < end; i++)
			{
				double d2 = (points[i] - q).SquaredNorm;
				if (d2 < best)
				{
					best = d2;
				}
			}

			return;
		}

		int mid = (start + end) / 2;
		int axis = splitAxis[mid];
		double delta = q[axis] - splitValue[mid];

		// Every point left of mid is at most the plane value on this axis, every point from mid
		// on is at least it, so the far side is at least |delta| away.
		if (delta < 0)
		{
			Search(start, mid, q, ref best);
			if (delta * delta < best)
			{
				Search(mid, end, q, ref best);
			}
		}
		else
		{
			Search(mid, end, q, ref best);
			if (delta * delta < best)
			{
				Search(start, mid, q, ref best);
			}
		}
	}
}
