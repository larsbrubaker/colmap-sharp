// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonSurfaceTrimmer.Islands: the island merge SurfaceTrimmer.cpp's Execute runs because
// --aRatio defaults to 0.001 (COLMAP never overrides it). Ports SetConnectedComponents,
// PolygonArea, the component-graph build (componentBoundaryHalfEdges, componentEdges) and
// ComponentGraph::Node::merge. The kept (above-trim) and trimmed (below-trim) polygons are
// split into edge-connected components; each component is a node, adjacent to the components
// across its boundary. While the smallest component that has a neighbor covers less than
// 0.001 of the total area, it is merged into its first neighbor, and that neighbor absorbs
// the component's other neighbors too. So a small trimmed hole inside the kept surface is
// filled back in, and a small kept patch inside the trimmed part is dropped. Without
// --removeIslands, a small component with no neighbor is kept as it is.
//
// Translation notes:
// - The neighbor order decides which node a component merges into and the order of the
//   output polygons, and upstream builds it by iterating std::unordered_set /
//   std::unordered_map, so those are TrimmerEdgeHashTable (libc++'s order).
// - Nodes are indices; ComponentGraph's pointer lists become List<int>, with PopBack's
//   swap-with-last removal and the same re-evaluated loop bounds, so aliasing behaves alike.
// - Areas: each triangle's area is float (a float 2x2 Gram determinant, float sqrt), summed
//   into double; the threshold is area * (double)0.001f.

namespace ColmapSharp.Mvs.PoissonRecon;

public static partial class PoissonSurfaceTrimmer
{
	// Execute's IslandAreaRatio block: returns the new above-trim polygons.
	private static List<int[]> MergeIslands(TrimVertices vertices, List<int[]> ltPolygons, List<int[]> gtPolygons, CancellationToken cancellationToken, IProgress<double>? progress)
	{
		List<List<int>> ltComponents = SetConnectedComponents(ltPolygons);
		List<List<int>> gtComponents = SetConnectedComponents(gtPolygons);
		int gtComponentStart = ltComponents.Count;
		var polygons = new List<int[]>(ltPolygons.Count + gtPolygons.Count);
		polygons.AddRange(ltPolygons);
		polygons.AddRange(gtPolygons);
		var components = new List<List<int>>(ltComponents.Count + gtComponents.Count);
		components.AddRange(ltComponents);
		foreach (List<int> component in gtComponents)
		{
			for (int j = 0; j < component.Count; j++)
			{
				component[j] += ltPolygons.Count;
			}

			components.Add(component);
		}

		int n = components.Count;
		double[] areas = new double[n];
		var neighbors = new List<int>[n];
		var polygonIndices = new List<int>[n];

		// Set the polygons within each component and compute areas
		for (int i = 0; i < n; i++)
		{
			neighbors[i] = [];
			polygonIndices[i] = [.. components[i]];
			foreach (int p in polygonIndices[i])
			{
				areas[i] += PolygonArea(vertices, polygons[p]);
			}
		}

		cancellationToken.ThrowIfCancellationRequested();
		progress?.Report(0.5);

		// A map identifying half-edges along the boundaries of components and associating them with the component
		var componentBoundaryHalfEdges = new TrimmerEdgeHashTable();
		for (int i = 0; i < n; i++)
		{
			// All the half-edges for a given component
			var componentHalfEdges = new TrimmerEdgeHashTable();
			foreach (int p in components[i])
			{
				int[] poly = polygons[p];
				for (int k = 0; k < poly.Length; k++)
				{
					componentHalfEdges.Insert(poly[k], poly[(k + 1) % poly.Length], 0);
				}
			}

			for (int e = componentHalfEdges.First; e >= 0; e = componentHalfEdges.Next(e))
			{
				int k1 = componentHalfEdges.Key1(e);
				int k2 = componentHalfEdges.Key2(e);
				if (componentHalfEdges.Find(k2, k1) < 0)
				{
					// operator[]: a half-edge already on another component's boundary is reassigned.
					componentBoundaryHalfEdges.Value(componentBoundaryHalfEdges.Insert(k1, k2, i)) = i;
				}
			}
		}

		// A set identify the dual edges of the component graph
		var componentEdges = new TrimmerEdgeHashTable();
		for (int e = componentBoundaryHalfEdges.First; e >= 0; e = componentBoundaryHalfEdges.Next(e))
		{
			int opposite = componentBoundaryHalfEdges.Find(componentBoundaryHalfEdges.Key2(e), componentBoundaryHalfEdges.Key1(e));
			if (opposite >= 0)
			{
				int c1 = componentBoundaryHalfEdges.Value(e);
				int c2 = componentBoundaryHalfEdges.Value(opposite);
				componentEdges.Insert(Math.Min(c1, c2), Math.Max(c1, c2), 0);
			}
		}

		for (int e = componentEdges.First; e >= 0; e = componentEdges.Next(e))
		{
			int k1 = componentEdges.Key1(e);
			int k2 = componentEdges.Key2(e);
			neighbors[k1].Add(k2);
			neighbors[k2].Add(k1);
		}

		double area = 0;
		for (int i = 0; i < n; i++)
		{
			area += areas[i];
		}

		double threshold = area * (double)IslandAreaRatio;
		while (true)
		{
			cancellationToken.ThrowIfCancellationRequested();
			int idx = -1;
			for (int i = 0; i < n; i++)
			{
				if (polygonIndices[i].Count > 0 && neighbors[i].Count > 0 && (idx == -1 || areas[i] < areas[idx]))
				{
					idx = i;
				}
			}

			if (idx == -1 || !(areas[idx] < threshold))
			{
				break;
			}

			Merge(idx, areas, neighbors, polygonIndices);
		}

		progress?.Report(0.75);

		// Without --removeIslands every component keeps its polygons; the trimmed ones are dropped.
		var kept = new List<int[]>();
		for (int i = gtComponentStart; i < n; i++)
		{
			foreach (int p in polygonIndices[i])
			{
				kept.Add(polygons[p]);
			}
		}

		return kept;
	}

	// ComponentGraph< Index >::Node::merge for node `self`.
	private static void Merge(int self, double[] areas, List<int>[] neighbors, List<int>[] polygonIndices)
	{
		List<int> own = neighbors[self];
		if (own.Count == 0)
		{
			throw new InvalidOperationException("No neighbors");
		}

		// Remove the node from the neighbors of the neighbors
		for (int i = 0; i < own.Count; i++)
		{
			List<int> theirs = neighbors[own[i]];
			for (int j = theirs.Count - 1; j >= 0; j--)
			{
				if (theirs[j] == self)
				{
					PopBack(theirs, j);
				}
			}
		}

		// Merge the node into its first neighbor
		int first = own[0];
		areas[first] += areas[self];
		polygonIndices[first].AddRange(polygonIndices[self]);
		polygonIndices[self].Clear();

		// Merge the remaining neighbors into the first neighbor
		for (int i = 1; i < own.Count; i++)
		{
			int other = own[i];
			areas[first] += areas[other];
			polygonIndices[first].AddRange(polygonIndices[other]);
			polygonIndices[other].Clear();
			for (int j = 0; j < neighbors[other].Count; j++)
			{
				int nj = neighbors[other][j];
				bool foundNeighbor = false;
				for (int k = neighbors[nj].Count - 1; k >= 0; k--)
				{
					if (neighbors[nj][k] == other)
					{
						PopBack(neighbors[nj], k);
					}
				}

				// Re-read as upstream does (neighbors[i]->neighbors[j]), in case the removal above
				// changed this very list.
				nj = neighbors[other][j];
				for (int k = 0; k < neighbors[first].Count; k++)
				{
					foundNeighbor |= nj == neighbors[first][k];
				}

				if (!foundNeighbor)
				{
					neighbors[first].Add(nj);
					neighbors[nj].Add(first);
				}
			}

			areas[other] = 0;
			neighbors[other].Clear();
		}

		// Clean up the node (its neighbor list is left as upstream leaves it)
		areas[self] = 0;
	}

	private static void PopBack(List<int> nodes, int idx)
	{
		nodes[idx] = nodes[^1];
		nodes.RemoveAt(nodes.Count - 1);
	}

	// SetConnectedComponents: union-find over polygons sharing an (undirected) edge, each
	// component listing its polygons in index order, the components ordered by their root.
	private static List<List<int>> SetConnectedComponents(List<int[]> polygons)
	{
		int[] roots = new int[polygons.Count];
		for (int i = 0; i < roots.Length; i++)
		{
			roots[i] = i;
		}

		var edgeTable = new Dictionary<(int, int), int>();
		for (int i = 0; i < polygons.Count; i++)
		{
			int[] poly = polygons[i];
			for (int j = 0; j < poly.Length; j++)
			{
				int v1 = poly[j];
				int v2 = poly[(j + 1) % poly.Length];
				(int, int) key = v1 < v2 ? (v1, v2) : (v2, v1);
				if (!edgeTable.TryGetValue(key, out int p))
				{
					edgeTable[key] = i;
					continue;
				}

				while (roots[p] != p)
				{
					int temp = roots[p];
					roots[p] = i;
					p = temp;
				}

				roots[p] = i;
			}
		}

		for (int i = 0; i < roots.Length; i++)
		{
			int p = i;
			while (roots[p] != p)
			{
				p = roots[p];
			}

			int root = p;
			p = i;
			while (roots[p] != p)
			{
				int temp = roots[p];
				roots[p] = root;
				p = temp;
			}
		}

		int[] componentOf = new int[roots.Length];
		var components = new List<List<int>>();
		for (int i = 0; i < roots.Length; i++)
		{
			if (roots[i] == i)
			{
				componentOf[i] = components.Count;
				components.Add([]);
			}
		}

		for (int i = 0; i < roots.Length; i++)
		{
			components[componentOf[roots[i]]].Add(i);
		}

		return components;
	}

	// PolygonArea: a triangle's area, or for a larger polygon the sum of the triangles fanned
	// around its float centroid.
	private static double PolygonArea(TrimVertices vertices, int[] polygon)
	{
		Span<float> p1 = stackalloc float[3];
		Span<float> p2 = stackalloc float[3];
		Span<float> p3 = stackalloc float[3];
		if (polygon.Length < 3)
		{
			return 0.0;
		}

		if (polygon.Length == 3)
		{
			vertices.CopyPosition(polygon[0], p1);
			vertices.CopyPosition(polygon[1], p2);
			vertices.CopyPosition(polygon[2], p3);
			return TriangleArea(p1, p2, p3);
		}

		Span<float> center = stackalloc float[3];
		center.Clear();
		foreach (int v in polygon)
		{
			vertices.CopyPosition(v, p1);
			center[0] += p1[0];
			center[1] += p1[1];
			center[2] += p1[2];
		}

		float count = polygon.Length;
		center[0] /= count;
		center[1] /= count;
		center[2] /= count;
		double area = 0;
		for (int i = 0; i < polygon.Length; i++)
		{
			vertices.CopyPosition(polygon[i], p2);
			vertices.CopyPosition(polygon[(i + 1) % polygon.Length], p3);
			area += TriangleArea(center, p2, p3);
		}

		return area;
	}

	// PolygonArea's Area lambda: sqrt of the 2x2 Gram determinant of the edge vectors, halved,
	// all in float (XForm< float , 2 >::determinant is 0 + m00 * m11 - m10 * m01).
	private static float TriangleArea(ReadOnlySpan<float> p1, ReadOnlySpan<float> p2, ReadOnlySpan<float> p3)
	{
		float ax = p2[0] - p1[0], ay = p2[1] - p1[1], az = p2[2] - p1[2];
		float bx = p3[0] - p1[0], by = p3[1] - p1[1], bz = p3[2] - p1[2];
		float m00 = Dot(ax, ay, az, ax, ay, az);
		float m01 = Dot(ax, ay, az, bx, by, bz);
		float m10 = Dot(bx, by, bz, ax, ay, az);
		float m11 = Dot(bx, by, bz, bx, by, bz);
		float det = 0f;
		det += m00 * m11;
		det -= m10 * m01;
		if (det < 0)
		{
			return 0f;
		}

		return (float)(Math.Sqrt(det) / 2.0);
	}

	// Point< float , 3 >::Dot: a float sum from zero, in coordinate order.
	private static float Dot(float x1, float y1, float z1, float x2, float y2, float z2)
	{
		float dot = 0f;
		dot += x1 * x2;
		dot += y1 * y2;
		dot += z1 * z2;
		return dot;
	}
}
