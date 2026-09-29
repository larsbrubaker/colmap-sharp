// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// TriangleBvh: a bounding volume hierarchy over triangles for segment "is anything in the
// way" queries. It replaces the CGAL AABB_tree that COLMAP's texture mapping
// (colmap/mvs/texture_mapping.cc, OcclusionTester) uses for occlusion tests; CGAL is GPL and
// excluded (docs/LICENSE_AUDIT.md), and no CGAL code was read. TextureMapping.Views.cs is
// the only user. Differences from CGAL's answers are divergence 90.
//
// Written for colmap-sharp from the published algorithms:
// - Binned surface area heuristic build: I. Wald, "On fast Construction of SAH-based
//   Bounding Volume Hierarchies", IEEE Symposium on Interactive Ray Tracing, 2007.
// - Segment/box slab test: T. L. Kay and J. T. Kajiya, "Ray tracing complex scenes",
//   SIGGRAPH 1986.
// - Segment/triangle test: T. Möller and B. Trumbore, "Fast, Minimum Storage Ray-Triangle
//   Intersection", Journal of Graphics Tools 2(1), 1997.
//
// Layout: flat arrays in depth-first order. An inner node's left child is the next node and
// its right child's index is stored; each subtree owns a fixed block of 2n - 1 node slots
// (n = its triangle count), so subtrees can be built in parallel and still land in the same
// slots (unused slots are left over where SAH stops early). The build uses at most the
// caller's thread count and checks its cancellation token at every subtree. Triangles are
// reordered so every leaf covers a contiguous range, and stored as corner plus edges in
// double.
//
// Determinism: a query's answer is a yes/no over all triangles. The node boxes only prune
// (they are padded and rounded outward, so they never drop a triangle the exact test hits),
// so the answer does not depend on the tree's shape, the traversal order or the thread
// count; only the per-triangle Möller-Trumbore test decides it. Queries allocate nothing (a
// fixed stack on the call stack) and are safe to run from many threads at once.

namespace ColmapSharp.Mvs;

/// <summary>
/// Bounding volume hierarchy over triangles answering "does this segment hit a triangle".
/// Replacement for CGAL::AABB_tree in COLMAP's texture mapping (written from the published
/// algorithms, not from CGAL).
/// </summary>
public sealed class TriangleBvh
{
	private const int BinCount = 16;
	private const int MaxLeafSize = 4;

	// Past this depth the build switches from SAH to median splits, which halve the range,
	// so the tree depth stays below SahDepthLimit + 32 and a query stack of StackSize holds.
	private const int SahDepthLimit = 48;
	private const int StackSize = 128;

	// Subtrees with at least this many triangles build their two children in parallel.
	private const int ParallelBuildThreshold = 1 << 15;

	// Doubles per stored triangle: v0 (3), e1 = v1 - v0 (3), e2 = v2 - v0 (3), |e1|², |e2|².
	private const int TriStride = 11;

	private readonly double[] tris;

	// Caller's id of each triangle, in leaf order.
	private readonly int[] ids;

	// Six floats per node: min x, y, z then max x, y, z, rounded outward.
	private readonly float[] nodeBounds;

	// Leaf: index of its first triangle. Inner node: index of its right child (the left
	// child is the next node).
	private readonly int[] nodeFirst;

	// Leaf: number of triangles (> 0). Inner node: -(split axis + 1), which orders traversal.
	private readonly int[] nodeCount;

	/// <summary>
	/// Builds the tree. <paramref name="triangleCorners"/> holds nine floats per triangle
	/// (x, y, z of each corner) and <paramref name="triangleIds"/> the id reported for each.
	/// Large subtrees are built on up to <paramref name="numThreads"/> threads (-1 or 0: all
	/// available, 1: sequential); the tree is the same for any thread count.
	/// <paramref name="cancellationToken"/> is checked between subtrees.
	/// </summary>
	public TriangleBvh(
		ReadOnlySpan<float> triangleCorners,
		ReadOnlySpan<int> triangleIds,
		int numThreads = -1,
		CancellationToken cancellationToken = default)
	{
		int count = triangleIds.Length;
		if (triangleCorners.Length != 9 * count)
		{
			throw new ArgumentException("Expected nine coordinates per triangle.", nameof(triangleCorners));
		}

		var source = triangleCorners.ToArray();

		// Per-triangle bounds and centroids, which is all the build reads.
		var triBounds = new double[6 * count];
		var centroids = new double[3 * count];
		var order = new int[count];
		for (int i = 0; i < count; i++)
		{
			order[i] = i;
			for (int a = 0; a < 3; a++)
			{
				double c0 = source[9 * i + a], c1 = source[9 * i + 3 + a], c2 = source[9 * i + 6 + a];
				triBounds[6 * i + a] = Math.Min(c0, Math.Min(c1, c2));
				triBounds[6 * i + 3 + a] = Math.Max(c0, Math.Max(c1, c2));
				centroids[3 * i + a] = (c0 + c1 + c2) / 3.0;
			}
		}

		int maxNodes = Math.Max(1, 2 * count - 1);
		nodeBounds = new float[6 * maxNodes];
		nodeFirst = new int[maxNodes];
		nodeCount = new int[maxNodes];
		if (count > 0)
		{
			var builder = new Builder(this, triBounds, centroids, order, Util.Threading.GetEffectiveNumThreads(numThreads), cancellationToken);
			builder.Build(0, 0, count, 0);
			ParallelSplits = builder.ParallelSplits;
		}

		tris = new double[TriStride * count];
		ids = new int[count];
		for (int i = 0; i < count; i++)
		{
			int src = order[i];
			ids[i] = triangleIds[src];
			int s = 9 * src;
			int d = TriStride * i;
			double v0x = source[s], v0y = source[s + 1], v0z = source[s + 2];
			double e1x = source[s + 3] - v0x, e1y = source[s + 4] - v0y, e1z = source[s + 5] - v0z;
			double e2x = source[s + 6] - v0x, e2y = source[s + 7] - v0y, e2z = source[s + 8] - v0z;
			tris[d] = v0x;
			tris[d + 1] = v0y;
			tris[d + 2] = v0z;
			tris[d + 3] = e1x;
			tris[d + 4] = e1y;
			tris[d + 5] = e1z;
			tris[d + 6] = e2x;
			tris[d + 7] = e2y;
			tris[d + 8] = e2z;
			tris[d + 9] = e1x * e1x + e1y * e1y + e1z * e1z;
			tris[d + 10] = e2x * e2x + e2y * e2y + e2z * e2z;
		}
	}

	/// <summary>Number of triangles in the tree.</summary>
	public int Count => ids.Length;

	/// <summary>For tests: how many splits built their two subtrees concurrently.</summary>
	internal int ParallelSplits { get; }

	/// <summary>
	/// For tests: the whole layout (node boxes, node links and counts, triangle ids in leaf
	/// order), which must not depend on the thread count.
	/// </summary>
	internal (float[] NodeBounds, int[] NodeFirst, int[] NodeCount, int[] Ids) Layout =>
		(nodeBounds, nodeFirst, nodeCount, ids);

	/// <summary>
	/// Whether the segment from the origin to the end point crosses a triangle whose id is
	/// not <paramref name="excludedId"/>, at a single point whose distance from the origin
	/// (computed in float, as COLMAP does) is below <paramref name="maxHitDistance"/>.
	/// Triangle edges and corners count as hits; a segment parallel to a triangle's plane
	/// never hits it (CGAL reports a coplanar overlap as a segment, which COLMAP's occlusion
	/// test ignores). Thread-safe; allocates nothing.
	/// </summary>
	public bool AnyHit(
		float originX, float originY, float originZ,
		float endX, float endY, float endZ,
		int excludedId,
		float maxHitDistance) =>
		CountHits(originX, originY, originZ, endX, endY, endZ, true, excludedId, 1, maxHitDistance, out _) > 0;

	/// <summary>
	/// Counts the triangles the segment hits under <see cref="AnyHit"/>'s rules (excluding
	/// none), stopping at two: 0, 1 (with its id in <paramref name="onlyId"/>, else -1) or 2
	/// for "two or more". A caller can then answer AnyHit for any excluded id without another
	/// query: yes for 2, yes for 1 unless the id is the excluded one, no for 0 (which needs
	/// the ids to be unique).
	/// Thread-safe; allocates nothing.
	/// </summary>
	public int CountHitsUpToTwo(
		float originX, float originY, float originZ,
		float endX, float endY, float endZ,
		float maxHitDistance,
		out int onlyId)
	{
		int count = CountHits(originX, originY, originZ, endX, endY, endZ, false, 0, 2, maxHitDistance, out int firstId);
		onlyId = count == 1 ? firstId : -1;
		return count;
	}

	// The shared traversal: counts qualifying triangles (skipping excludedId when exclude is
	// set) until limit is reached. firstId is the first one found (-1 if none).
	private int CountHits(
		float originX, float originY, float originZ,
		float endX, float endY, float endZ,
		bool exclude,
		int excludedId,
		int limit,
		float maxHitDistance,
		out int firstId)
	{
		firstId = -1;
		int found = 0;
		if (ids.Length == 0)
		{
			return 0;
		}

		double ox = originX, oy = originY, oz = originZ;
		double dx = (double)endX - ox, dy = (double)endY - oy, dz = (double)endZ - oz;
		double dd = dx * dx + dy * dy + dz * dz;

		// Reciprocals for the slab test (a zero component takes the parallel branch there).
		double ix = 1.0 / dx, iy = 1.0 / dy, iz = 1.0 / dz;

		Span<int> stack = stackalloc int[StackSize];
		int top = 0;
		stack[top++] = 0;
		while (top > 0)
		{
			int node = stack[--top];
			if (!SegmentHitsBox(node, ox, oy, oz, dx, dy, dz, ix, iy, iz))
			{
				continue;
			}

			int n = nodeCount[node];
			if (n < 0)
			{
				// Visit the child nearer the origin along the split axis first, so blockers
				// near the camera end the search early. The answer is the same either way.
				int axis = -n - 1;
				double d = axis == 0 ? dx : axis == 1 ? dy : dz;
				int left = node + 1;
				int right = nodeFirst[node];
				if (d < 0)
				{
					stack[top++] = left;
					stack[top++] = right;
				}
				else
				{
					stack[top++] = right;
					stack[top++] = left;
				}

				continue;
			}

			int first = nodeFirst[node];
			for (int i = first; i < first + n; i++)
			{
				if (exclude && ids[i] == excludedId)
				{
					continue;
				}

				if (!IntersectTriangle(i, ox, oy, oz, dx, dy, dz, dd, out double t))
				{
					continue;
				}

				// The intersection point in float, then its distance from the origin in float,
				// matching COLMAP's use of a float CGAL kernel and Eigen::Vector3f.
				float hx = (float)(ox + t * dx);
				float hy = (float)(oy + t * dy);
				float hz = (float)(oz + t * dz);
				float ex = hx - originX, ey = hy - originY, ez = hz - originZ;
				float hitDistance = MathF.Sqrt(ex * ex + ey * ey + ez * ez);
				if (hitDistance < maxHitDistance)
				{
					if (found++ == 0)
					{
						firstId = ids[i];
					}

					if (found == limit)
					{
						return found;
					}
				}
			}
		}

		return found;
	}

	// Slab test of the segment origin + t * dir, t in [0, 1], against a node's box.
	private bool SegmentHitsBox(int node, double ox, double oy, double oz, double dx, double dy, double dz, double ix, double iy, double iz)
	{
		int b = 6 * node;
		double tMin = 0.0;
		double tMax = 1.0;
		return Slab(ox, dx, ix, nodeBounds[b], nodeBounds[b + 3], ref tMin, ref tMax)
			&& Slab(oy, dy, iy, nodeBounds[b + 1], nodeBounds[b + 4], ref tMin, ref tMax)
			&& Slab(oz, dz, iz, nodeBounds[b + 2], nodeBounds[b + 5], ref tMin, ref tMax);
	}

	private static bool Slab(double o, double d, double inv, double lo, double hi, ref double tMin, ref double tMax)
	{
		if (d == 0.0)
		{
			// Parallel to the slab: inside it for all t or for none.
			return o >= lo && o <= hi;
		}

		double t0 = (lo - o) * inv;
		double t1 = (hi - o) * inv;
		if (t0 > t1)
		{
			(t0, t1) = (t1, t0);
		}

		if (t0 > tMin)
		{
			tMin = t0;
		}

		if (t1 < tMax)
		{
			tMax = t1;
		}

		return tMin <= tMax;
	}

	// Möller–Trumbore in double, restricted to the segment (t in [0, 1]) with inclusive edges.
	// A segment (nearly) parallel to the plane, relative to the lengths involved, is a miss.
	// That check needs a square root, so it runs last, on the few candidates that pass the
	// others; a miss is a miss whichever check rejects it, so the answer is unchanged.
	private bool IntersectTriangle(int tri, double ox, double oy, double oz, double dx, double dy, double dz, double dd, out double t)
	{
		t = 0;
		int c = TriStride * tri;
		double v0x = tris[c], v0y = tris[c + 1], v0z = tris[c + 2];
		double e1x = tris[c + 3], e1y = tris[c + 4], e1z = tris[c + 5];
		double e2x = tris[c + 6], e2y = tris[c + 7], e2z = tris[c + 8];

		double px = dy * e2z - dz * e2y;
		double py = dz * e2x - dx * e2z;
		double pz = dx * e2y - dy * e2x;
		double det = e1x * px + e1y * py + e1z * pz;
		if (det == 0.0)
		{
			return false;
		}

		double inv = 1.0 / det;
		double sx = ox - v0x, sy = oy - v0y, sz = oz - v0z;
		double u = (sx * px + sy * py + sz * pz) * inv;
		if (u < 0.0 || u > 1.0)
		{
			return false;
		}

		double qx = sy * e1z - sz * e1y;
		double qy = sz * e1x - sx * e1z;
		double qz = sx * e1y - sy * e1x;
		double v = (dx * qx + dy * qy + dz * qz) * inv;
		if (v < 0.0 || u + v > 1.0)
		{
			return false;
		}

		t = (e2x * qx + e2y * qy + e2z * qz) * inv;
		if (!(t >= 0.0 && t <= 1.0))
		{
			return false;
		}

		double scale = Math.Sqrt(dd * tris[c + 9] * tris[c + 10]);
		return Math.Abs(det) > 1e-12 * scale;
	}

	// The build state shared by all subtrees. Each subtree touches only its own range of
	// `order` and its own block of node slots, so subtrees may be built concurrently.
	//
	// Parallelism: a split of a large subtree forks only while a spare worker is available
	// (numThreads - 1 of them, shared by the whole build), so at most numThreads threads
	// build at once; otherwise the two halves are built one after the other.
	private sealed class Builder(
		TriangleBvh tree,
		double[] triBounds,
		double[] centroids,
		int[] order,
		int numThreads,
		CancellationToken cancellationToken)
	{
		private int spareWorkers = numThreads - 1;
		private int parallelSplits;

		public int ParallelSplits => parallelSplits;

		// Builds the subtree over order[start, end) into node slots [node, node + 2n - 1).
		public void Build(int node, int start, int end, int depth)
		{
			cancellationToken.ThrowIfCancellationRequested();
			Span<double> box = stackalloc double[6];
			Span<double> cbox = stackalloc double[6];
			ResetBox(box);
			ResetBox(cbox);
			for (int i = start; i < end; i++)
			{
				int tri = order[i];
				GrowBox(box, triBounds.AsSpan(6 * tri, 6));
				for (int a = 0; a < 3; a++)
				{
					double c = centroids[3 * tri + a];
					cbox[a] = Math.Min(cbox[a], c);
					cbox[3 + a] = Math.Max(cbox[3 + a], c);
				}
			}

			StoreBounds(node, box);

			int count = end - start;
			if (count <= MaxLeafSize)
			{
				MakeLeaf(node, start, count);
				return;
			}

			int axis = 0;
			int mid = depth < SahDepthLimit ? SahSplit(start, end, box, cbox, out axis) : -1;
			if (mid == 0)
			{
				// SAH says a leaf is cheaper.
				MakeLeaf(node, start, count);
				return;
			}

			if (mid < 0)
			{
				mid = MedianSplit(start, end, cbox, out axis);
			}

			int left = node + 1;
			int right = left + (2 * (mid - start) - 1);
			tree.nodeFirst[node] = right;
			tree.nodeCount[node] = -(axis + 1);
			if (count >= ParallelBuildThreshold && TryTakeWorker())
			{
				Interlocked.Increment(ref parallelSplits);
				try
				{
					Parallel.Invoke(
						() => Build(left, start, mid, depth + 1),
						() => Build(right, mid, end, depth + 1));
				}
				catch (AggregateException e)
				{
					// Surface cancellation (or the first failure) as itself, not wrapped.
					cancellationToken.ThrowIfCancellationRequested();
					System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
				}
				finally
				{
					Interlocked.Increment(ref spareWorkers);
				}
			}
			else
			{
				Build(left, start, mid, depth + 1);
				Build(right, mid, end, depth + 1);
			}
		}

		private bool TryTakeWorker()
		{
			if (Interlocked.Decrement(ref spareWorkers) >= 0)
			{
				return true;
			}

			Interlocked.Increment(ref spareWorkers);
			return false;
		}

		private void MakeLeaf(int node, int start, int count)
		{
			tree.nodeFirst[node] = start;
			tree.nodeCount[node] = count;
		}

		// Stores the box padded (so the double slab test stays conservative) and rounded
		// outward to float.
		private void StoreBounds(int node, ReadOnlySpan<double> box)
		{
			for (int a = 0; a < 3; a++)
			{
				double pad = 1e-9 * Math.Max(1.0, Math.Max(Math.Abs(box[a]), Math.Abs(box[3 + a])));
				double lo = box[a] - pad;
				double hi = box[3 + a] + pad;
				float loF = (float)lo;
				if (loF > lo)
				{
					loF = MathF.BitDecrement(loF);
				}

				float hiF = (float)hi;
				if (hiF < hi)
				{
					hiF = MathF.BitIncrement(hiF);
				}

				tree.nodeBounds[6 * node + a] = loF;
				tree.nodeBounds[6 * node + 3 + a] = hiF;
			}
		}

		// Returns the split position, 0 to make a leaf, or -1 when binning cannot separate
		// the centroids (all equal along every axis) and a median split should be used.
		// Bins along all three axes are filled in one pass over the triangles.
		private int SahSplit(int start, int end, ReadOnlySpan<double> box, ReadOnlySpan<double> cbox, out int bestAxis)
		{
			Span<int> binCount = stackalloc int[3 * BinCount];
			Span<double> binBox = stackalloc double[3 * 6 * BinCount];
			Span<double> rightArea = stackalloc double[BinCount];
			Span<int> rightCount = stackalloc int[BinCount];
			Span<double> acc = stackalloc double[6];
			Span<double> scale = stackalloc double[3];
			binCount.Clear();
			for (int b = 0; b < 3 * BinCount; b++)
			{
				ResetBox(binBox.Slice(6 * b, 6));
			}

			for (int a = 0; a < 3; a++)
			{
				double extent = cbox[3 + a] - cbox[a];
				scale[a] = extent > 0 ? BinCount / extent : 0;
			}

			for (int i = start; i < end; i++)
			{
				int tri = order[i];
				ReadOnlySpan<double> bounds = triBounds.AsSpan(6 * tri, 6);
				for (int a = 0; a < 3; a++)
				{
					if (scale[a] == 0)
					{
						continue;
					}

					int b = a * BinCount + BinIndex(centroids[3 * tri + a], cbox[a], scale[a]);
					binCount[b]++;
					GrowBox(binBox.Slice(6 * b, 6), bounds);
				}
			}

			int count = end - start;
			double bestCost = double.PositiveInfinity;
			int bestBin = -1;
			bestAxis = -1;
			for (int axis = 0; axis < 3; axis++)
			{
				if (scale[axis] == 0)
				{
					continue;
				}

				int o = axis * BinCount;

				// Sweep from the right to get the area and count right of each split.
				ResetBox(acc);
				int n = 0;
				for (int b = BinCount - 1; b > 0; b--)
				{
					GrowBox(acc, binBox.Slice(6 * (o + b), 6));
					n += binCount[o + b];
					rightArea[b] = HalfArea(acc);
					rightCount[b] = n;
				}

				ResetBox(acc);
				n = 0;
				for (int b = 0; b < BinCount - 1; b++)
				{
					GrowBox(acc, binBox.Slice(6 * (o + b), 6));
					n += binCount[o + b];
					int nRight = rightCount[b + 1];
					if (n == 0 || nRight == 0)
					{
						continue;
					}

					double cost = HalfArea(acc) * n + rightArea[b + 1] * nRight;
					if (cost < bestCost)
					{
						bestCost = cost;
						bestAxis = axis;
						bestBin = b;
					}
				}
			}

			if (bestAxis < 0)
			{
				return -1;
			}

			// Traversal cost 1 against intersecting every triangle of the node.
			double parentArea = HalfArea(box);
			if (count <= 16 && parentArea > 0 && 1.0 + bestCost / parentArea >= count)
			{
				return 0;
			}

			// In-place partition: bins <= bestBin go left.
			double lo = cbox[bestAxis];
			double s = scale[bestAxis];
			int left = start;
			int right = end - 1;
			while (left <= right)
			{
				if (BinIndex(centroids[3 * order[left] + bestAxis], lo, s) <= bestBin)
				{
					left++;
				}
				else
				{
					(order[left], order[right]) = (order[right], order[left]);
					right--;
				}
			}

			return left;
		}

		// Sorts the range by centroid along the widest centroid axis (index as tie-break) and
		// splits it in half. With all centroids equal this is just the middle of the range.
		private int MedianSplit(int start, int end, ReadOnlySpan<double> cbox, out int axis)
		{
			axis = 0;
			double widest = cbox[3] - cbox[0];
			for (int a = 1; a < 3; a++)
			{
				double extent = cbox[3 + a] - cbox[a];
				if (extent > widest)
				{
					widest = extent;
					axis = a;
				}
			}

			if (widest > 0)
			{
				int sortAxis = axis;
				order.AsSpan(start, end - start).Sort((x, y) =>
				{
					int byCoord = centroids[3 * x + sortAxis].CompareTo(centroids[3 * y + sortAxis]);
					return byCoord != 0 ? byCoord : x.CompareTo(y);
				});
			}

			return start + (end - start) / 2;
		}
	}

	private static int BinIndex(double c, double lo, double scale)
	{
		int b = (int)((c - lo) * scale);
		return b < 0 ? 0 : b >= BinCount ? BinCount - 1 : b;
	}

	private static void ResetBox(Span<double> b)
	{
		for (int a = 0; a < 3; a++)
		{
			b[a] = double.PositiveInfinity;
			b[3 + a] = double.NegativeInfinity;
		}
	}

	private static void GrowBox(Span<double> b, ReadOnlySpan<double> other)
	{
		for (int a = 0; a < 3; a++)
		{
			b[a] = Math.Min(b[a], other[a]);
			b[3 + a] = Math.Max(b[3 + a], other[3 + a]);
		}
	}

	private static double HalfArea(ReadOnlySpan<double> b)
	{
		double x = b[3] - b[0], y = b[4] - b[1], z = b[5] - b[2];
		if (!(x >= 0 && y >= 0 && z >= 0))
		{
			return 0;
		}

		return x * y + y * z + z * x;
	}
}
