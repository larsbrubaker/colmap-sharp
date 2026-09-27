// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonLevelSetExtractor.Polygons: the iso-surface's polygons. For each leaf of a slab it
// gathers the iso-edges of its six faces (PoissonLevelSetExtractor.IsoEdges.cs), oriented
// outward, chains them into closed loops of edge keys (walking across a coarser edge through
// its vertex pair where a loop does not close on the leaf's own edges), looks up each key's
// iso-vertex (PoissonLevelSetExtractor.IsoVertices.cs, .XSliceIsoVertices.cs) and triangulates
// each loop. Ports FEMTree.LevelSet.3D.inl's _LevelSetExtractor< ... , 3 , ... >::SetLevelSet,
// AddIsoPolygons, Extract's IsoSurface (with its faceIndexFunctor) and SliceValues'
// addIsoEdges, setVertexPair and setEdgeVertex. Loops of more than three vertices are split by
// MinimalAreaTriangulation (MAT.h), unless two non-adjacent vertices share a coordinate, in
// which case they are fanned around their barycenter, a new vertex written to Vertices. Tier A
// against oracle/poisson_levelset6_harness.cc.
//
// COLMAP's settings, the only branch ported: PoissonRecon.cpp's Execute passes
// forceManifold = !--nonManifold (true, COLMAP never passes it) to Extract's addBarycenter
// argument, polygonMesh = --polygonMesh (false) and flipOrientation false. So loops become
// triangles with the barycenter test on, in the loop's own order. (Upstream's polygonMesh
// branch and flipOrientation are not reachable from COLMAP.)
//
// Translation notes:
// - Upstream runs SetLevelSet in a ThreadPool::ParallelFor over the slab's leaves, writing
//   triangles and barycenter vertices as each thread gets to them; the port runs the leaves in
//   order, which is what a single-threaded run does (docs/CPP_DIVERGENCES.md, entry 123). The
//   loop assembly uses vectors only (the key maps are looked up, never iterated), so no hash
//   order reaches the output.
// - The barycenter starts from upstream's default-constructed vertex times zero; its
//   positions, gradient and colors are zero-initialized Points, but the depth is an
//   uninitialized float, and garbage (or NaN) times zero is not reliably zero. That is undefined
//   behavior upstream (an optimized build gives every barycenter a NaN depth); the port starts
//   the depth at zero, so the barycenter's depth is the mean of the loop's depths
//   (docs/CPP_DIVERGENCES.md, entry 125).
// - A face whose iso-edges are neither set nor in the face-edge map is only warned about
//   upstream ("Invalid face"); here it is counted in InvalidFaceCount. The other failures
//   (MK_THROW) throw InvalidOperationException.
// - IsoSurface's faceIndexFunctor moves a face key onto the slab bounds only for faces outside
//   [slabStart, slabEnd); for the whole tree no in-bounds leaf has one, so it is the plain key.

namespace ColmapSharp.Mvs.PoissonRecon;

public sealed partial class PoissonLevelSetExtractor
{
	private readonly List<LevelSetIsoEdge> loopEdges = [];

	/// <summary>
	/// The triangles in output order, each three indices into <see cref="Vertices"/>. Port of
	/// the polygon stream (as Extract writes it for COLMAP: triangles only).
	/// </summary>
	public List<int[]> Polygons { get; } = [];

	/// <summary>
	/// How many leaf faces had neither iso-edges set nor an entry in the face-edge map. Port of
	/// addIsoEdges' "Invalid face" warning.
	/// </summary>
	public long InvalidFaceCount { get; private set; }

	/// <summary>
	/// The polygons of the leaves in slab <paramref name="slabAtMaxDepth"/> and in each coarser
	/// slab that it ends, finest depth first. Its slices and slabs must be finalized. Port of
	/// Extract's <c>IsoSurface</c> (without a boundary; InteriorSlab holds for every slab of the
	/// whole tree).
	/// </summary>
	public void IsoSurface(int slabAtMaxDepth)
	{
		for (int d = MaxDepth, o = slabAtMaxDepth; d >= FullDepth; d--, o >>= 1)
		{
			SetLevelSet(d, o);
			if ((o & 1) == 0)
			{
				break;
			}
		}
	}

	/// <summary>
	/// The polygons of the leaves in slab <paramref name="offset"/> at <paramref name="depth"/>.
	/// Port of <c>SetLevelSet</c> with COLMAP's settings (see the file header).
	/// </summary>
	public void SetLevelSet(int depth, int offset)
	{
		LevelSetSliceValues bValues = SlabValues[depth].SliceValues(offset);
		LevelSetSliceValues fValues = SlabValues[depth].SliceValues(offset + 1);
		LevelSetXSliceValues xValues = SlabValues[depth].XSliceValues(offset);
		int globalDepth = depth + tree.DepthOffset;
		int res = 1 << depth;
		List<LevelSetIsoEdge> edges = loopEdges;
		var loops = new List<List<LevelSetKey>>();
		int end = sorted.End(globalDepth, offset + tree.LocalInset(depth));
		for (int i = sorted.Begin(globalDepth, offset + tree.LocalInset(depth)); i < end; i++)
		{
			int leaf = sorted.TreeNodes[i];
			if (!PoissonMultigrid.IsValidSpaceNode(tree, leaf))
			{
				continue;
			}

			int ox = tree.LocalOffset(leaf, 0), oy = tree.LocalOffset(leaf, 1), oz = tree.LocalOffset(leaf, 2);
			bool inBounds = ox >= 0 && ox < res && oy >= 0 && oy < res && oz >= 0 && oz < res;
			if (!inBounds || tree.IsActive(tree.FirstChild(leaf)))
			{
				continue;
			}

			edges.Clear();

			// Gather the edges from the faces (with the correct orientation)
			for (int f = 0; f < HyperCube.ElementNum(3, 2); f++)
			{
				bool isOriented = HyperCube.IsOriented(3, f);
				int flip = isOriented ? 0 : 1;
				HyperCube.Factor(3, 2, f, out HyperCubeDirection fDir, out int coIndex);
				if (fDir == HyperCubeDirection.Back)
				{
					AddIsoEdges(bValues, leaf, offset + 0, edges, isOriented);
				}
				else if (fDir == HyperCubeDirection.Front)
				{
					AddIsoEdges(fValues, leaf, offset + 1, edges, isOriented);
				}
				else
				{
					int fIdx = xValues.CellIndices.Index(1, i, coIndex);
					if (xValues.FaceSet[fIdx] != 0)
					{
						LevelSetFaceEdges fe = xValues.FaceEdges[fIdx];
						for (int j = 0; j < fe.Count; j++)
						{
							edges.Add(new LevelSetIsoEdge(fe[j][flip], fe[j][1 - flip]));
						}
					}
					else
					{
						LevelSetKey key = FaceKey(leaf, f);
						if (!xValues.FaceEdgeMap.TryGetValue(key, out List<LevelSetIsoEdge>? mapped))
						{
							throw new InvalidOperationException($"Invalid faces: {i}  cross");
						}

						foreach (LevelSetIsoEdge e in mapped)
						{
							edges.Add(new LevelSetIsoEdge(e[flip], e[1 - flip]));
						}
					}
				}
			}

			// Get the edge loops
			loops.Clear();
			while (edges.Count > 0)
			{
				var loop = new List<LevelSetKey>();
				loops.Add(loop);
				LevelSetIsoEdge edge = edges[^1];
				edges.RemoveAt(edges.Count - 1);
				LevelSetKey start = edge.First, current = edge.Second;
				while (current != start)
				{
					int idx;
					for (idx = 0; idx < edges.Count; idx++)
					{
						if (edges[idx].First == current)
						{
							break;
						}
					}

					if (idx == edges.Count)
					{
						// setVertexPair on the back and front slices, then the slab's vertexPairMap.
						if (bValues.VertexPairMap.TryGetValue(current, out LevelSetKey pair)
							|| fValues.VertexPairMap.TryGetValue(current, out pair)
							|| xValues.VertexPairMap.TryGetValue(current, out pair))
						{
							loop.Add(current);
							current = pair;
						}
						else
						{
							throw new InvalidOperationException($"Failed to close loop for node[{i}]: [{ox} {oy} {oz} @ {depth}] | {current} -- {start}");
						}
					}
					else
					{
						loop.Add(current);
						current = edges[idx].Second;
						edges[idx] = edges[^1];
						edges.RemoveAt(edges.Count - 1);
					}
				}

				loop.Add(start);
			}

			// Add the loops to the mesh
			foreach (List<LevelSetKey> loop in loops)
			{
				var polygon = new int[loop.Count];
				for (int k = 0; k < loop.Count; k++)
				{
					LevelSetKey key = loop[k];

					// setEdgeVertex on the back and front slices, then the slab's edgeVertexMap.
					if (!bValues.EdgeVertexMap.TryGetValue(key, out polygon[k])
						&& !fValues.EdgeVertexMap.TryGetValue(key, out polygon[k])
						&& !xValues.EdgeVertexMap.TryGetValue(key, out polygon[k]))
					{
						throw new InvalidOperationException($"Couldn't find vertex in edge map: {ox} , {oy} , {oz} @ {depth} : {key}");
					}
				}

				AddIsoPolygons(polygon);
			}
		}
	}

	// SliceValues::addIsoEdges: the iso-edges of the leaf's face on slice sliceIndex (set by
	// the leaf's own pass, or pushed up from finer leaves into the face-edge map), oriented.
	private void AddIsoEdges(LevelSetSliceValues values, int leaf, int sliceIndex, List<LevelSetIsoEdge> edges, bool isOriented)
	{
		int flip = isOriented ? 0 : 1;
		int fIdx = values.CellIndices.Index(2, tree.NodeIndex(leaf), 0);
		if (values.FaceSet[fIdx] != 0)
		{
			LevelSetFaceEdges fe = values.FaceEdges[fIdx];
			for (int i = 0; i < fe.Count; i++)
			{
				edges.Add(new LevelSetIsoEdge(fe[i][flip], fe[i][1 - flip]));
			}

			return;
		}

		// _FaceIndex: the leaf's back or front face, whichever lies on the slice.
		int oz = tree.LocalOffset(leaf, 2);
		int f;
		if (oz + 0 == sliceIndex)
		{
			f = HyperCube.Element(3, 2, HyperCubeDirection.Back, 0);
		}
		else if (oz + 1 == sliceIndex)
		{
			f = HyperCube.Element(3, 2, HyperCubeDirection.Front, 0);
		}
		else
		{
			throw new InvalidOperationException($"Node/slice-index mismatch: {oz} <-> {sliceIndex}");
		}

		if (values.FaceEdgeMap.TryGetValue(FaceKey(leaf, f), out List<LevelSetIsoEdge>? mapped))
		{
			foreach (LevelSetIsoEdge e in mapped)
			{
				edges.Add(new LevelSetIsoEdge(e[flip], e[1 - flip]));
			}
		}
		else
		{
			// [WARNING] Is this right? If the face isn't set, wouldn't it inherit?
			InvalidFaceCount++;
		}
	}

	// IsoSurface's faceIndexFunctor for the whole tree (see the header): the key of face f of node.
	private LevelSetKey FaceKey(int node, int f) =>
		KeyGenerator.Key(2, tree.LocalDepth(node), tree.LocalOffset(node, 0), tree.LocalOffset(node, 1), tree.LocalOffset(node, 2), f);

	// AddIsoPolygons( ... , polygon , polygonMesh = false , addBarycenter = true ): each loop
	// (vertex indices) as triangles, winding reversed.
	private void AddIsoPolygons(int[] polygon)
	{
		int n = polygon.Length;
		if (n > 3)
		{
			bool isCoplanar = false;
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < i; j++)
				{
					if ((i + 1) % n != j && (j + 1) % n != i)
					{
						LevelSetVertex v1 = Vertices[polygon[i]], v2 = Vertices[polygon[j]];
						if (v1.X == v2.X || v1.Y == v2.Y || v1.Z == v2.Z)
						{
							isCoplanar = true;
						}
					}
				}
			}

			if (isCoplanar)
			{
				int cIdx = Vertices.Count;
				Vertices.Add(Barycenter(polygon));
				for (int i = 0; i < n; i++)
				{
					Polygons.Add([polygon[i], cIdx, polygon[(i + 1) % n]]);
				}
			}
			else
			{
				float[] positions = new float[3 * n];
				for (int i = 0; i < n; i++)
				{
					LevelSetVertex v = Vertices[polygon[i]];
					positions[(3 * i) + 0] = v.X;
					positions[(3 * i) + 1] = v.Y;
					positions[(3 * i) + 2] = v.Z;
				}

				List<(int A, int B, int C)> triangles = MinimalAreaTriangulation.Triangulate(positions, n);
				if (triangles.Count != n - 2)
				{
					throw new InvalidOperationException($"Minimal area triangulation failed:{triangles.Count} != {n - 2}");
				}

				foreach ((int a, int b, int c) in triangles)
				{
					// triangle[2-j] = polygon[ idx[j] ]: the winding is reversed.
					Polygons.Add([polygon[c], polygon[b], polygon[a]]);
				}
			}
		}
		else if (n == 3)
		{
			Polygons.Add([polygon[2], polygon[1], polygon[0]]);
		}
	}

	// The loop's barycenter: c *= 0, c += each vertex in loop order, c /= (float)count, per
	// component (position, gradient, depth, data), in float.
	private LevelSetVertex Barycenter(int[] polygon)
	{
		// depth = 0f where upstream's is uninitialized (undefined behavior; see the header and
		// docs/CPP_DIVERGENCES.md, entry 125).
		float x = 0f, y = 0f, z = 0f, gx = 0f, gy = 0f, gz = 0f, depth = 0f;
		float[] data = new float[zeroData.Length];
		foreach (int index in polygon)
		{
			LevelSetVertex v = Vertices[index];
			x += v.X;
			y += v.Y;
			z += v.Z;
			gx += v.GradientX;
			gy += v.GradientY;
			gz += v.GradientZ;
			depth += v.Depth;
			for (int k = 0; k < data.Length; k++)
			{
				data[k] += v.Data[k];
			}
		}

		float count = polygon.Length;
		for (int k = 0; k < data.Length; k++)
		{
			data[k] /= count;
		}

		return new LevelSetVertex(x / count, y / count, z / count, gx / count, gy / count, gz / count, depth / count, data);
	}
}
