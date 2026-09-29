// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// HullMarchingCubes: the iso-surface of an OccupancyGrid as a closed, outward-wound triangle mesh
// (docs/QUALITY_PLAN.md, stage 3a). Not a COLMAP port. Written from the papers:
// - W. E. Lorensen and H. E. Cline, "Marching cubes: a high resolution 3D surface construction
//   algorithm", SIGGRAPH 1987 (US patent 4,710,876, expired 2005);
// - G. M. Nielson and B. Hamann, "The asymptotic decider: resolving the ambiguity in marching
//   cubes", IEEE Visualization 1991 - for the faces with two diagonal inside corners.
// No case table is used, public-domain or otherwise. Instead each cube's polygons are built from
// its faces: on every face the edges the surface crosses are paired into segments (the ambiguous
// four-crossing face is resolved by the asymptotic decider, which depends on that face's four
// values only), and the segments chain into closed loops. A three-vertex loop is one triangle;
// a longer one is fanned from an added centroid vertex, since a fan from one of its own vertices
// can lay a triangle flat in a cube face, where the neighbor cube lays the same one reversed. Because a
// face's segments depend on that face alone, the two cubes sharing it agree, so the mesh is
// closed with no cracks; and a vertex is keyed by the grid edge it lies on, so neighbors share it.
//
// Winding: each segment is directed so that, with n the cube face's outward normal,
// cross(n, b - a) points away from the inside corners. The neighbor cube sees the same face with
// the opposite normal and so walks the segment the other way, which makes the winding consistent,
// and a loop directed this way is counter-clockwise seen from outside the hull (outward normals).
// The grid is treated as 0 beyond its bounds, so the surface always closes.
// Vertices are kept a thousandth of a voxel off the grid nodes, so values exactly at the iso
// level give no zero-area triangles. HullMarchingCubesTests checks all of this on random grids.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Silhouette;

/// <summary>Marching cubes over an <see cref="OccupancyGrid"/>; see the file header.</summary>
public static class HullMarchingCubes
{
	// Corner c of a cube is at offset (c & 1, (c >> 1) & 1, (c >> 2) & 1).
	// Faces: four corners in cyclic order, then the outward normal.
	private static readonly int[][] FaceCorners =
	[
		[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6],
	];

	private static readonly Vector3d[] FaceNormals =
	[
		new(-1, 0, 0), new(1, 0, 0), new(0, -1, 0), new(0, 1, 0), new(0, 0, -1), new(0, 0, 1),
	];

	// The 12 cube edges as (lower corner, axis); the upper corner is lower + (1 << axis).
	private static readonly (int Corner, int Axis)[] Edges =
	[
		(0, 0), (2, 0), (4, 0), (6, 0), (0, 1), (1, 1), (4, 1), (5, 1), (0, 2), (1, 2), (2, 2), (3, 2),
	];

	// How far (in voxels) a vertex is kept from a grid node; far above float rounding at the
	// coordinates a hull has, and far below anything the surface's accuracy notices.
	private const double NodeClearance = 1e-3;

	private static readonly int[,] EdgeOfCorners = BuildEdgeOfCorners();

	/// <summary>
	/// The surface where <paramref name="grid"/> crosses <paramref name="isoLevel"/> (values at or
	/// above it are inside), in world coordinates. Deterministic: vertices and faces come out in
	/// cube order.
	/// </summary>
	public static PlyMesh Extract(OccupancyGrid grid, double isoLevel)
	{
		Check.NotNull(grid);
		Check.That(isoLevel > 0, "The iso level must be above the empty value 0 so the surface closes");
		var mesh = new PlyMesh();
		var vertexOfEdge = new Dictionary<long, int>();
		Span<double> values = stackalloc double[8];
		Span<int> crossingVertex = stackalloc int[12];
		Span<int> next = stackalloc int[12];
		Span<int> loop = stackalloc int[12];
		long strideY = grid.NX + 2, strideZ = (long)(grid.NX + 2) * (grid.NY + 2);
		for (int k = -1; k < grid.NZ; k++)
		{
			for (int j = -1; j < grid.NY; j++)
			{
				for (int i = -1; i < grid.NX; i++)
				{
					int insideMask = 0;
					for (int c = 0; c < 8; c++)
					{
						values[c] = grid.Value(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
						if (values[c] >= isoLevel)
						{
							insideMask |= 1 << c;
						}
					}

					if (insideMask == 0 || insideMask == 255)
					{
						continue;
					}

					// Vertices on the crossed edges, shared with the neighbors through the edge key.
					for (int e = 0; e < 12; e++)
					{
						(int a, int axis) = Edges[e];
						int b = a + (1 << axis);
						crossingVertex[e] = -1;
						next[e] = -1;
						if (((insideMask >> a) & 1) == ((insideMask >> b) & 1))
						{
							continue;
						}

						long node = (k + 1 + ((a >> 2) & 1)) * strideZ + (j + 1 + ((a >> 1) & 1)) * strideY + (i + 1 + (a & 1));
						long key = node * 3 + axis;
						if (!vertexOfEdge.TryGetValue(key, out int index))
						{
							// Kept off the grid nodes: a value exactly at the iso level (common after the
							// [1 2 1]/4 smoothing) would put the vertices of several edges on one node
							// and give zero-area triangles.
							double t = Math.Clamp((isoLevel - values[a]) / (values[b] - values[a]), NodeClearance, 1 - NodeClearance);
							Vector3d pa = grid.Center(i + (a & 1), j + ((a >> 1) & 1), k + ((a >> 2) & 1));
							var offset = new Vector3d(axis == 0 ? t : 0, axis == 1 ? t : 0, axis == 2 ? t : 0);
							Vector3d p = pa + offset * grid.VoxelSize;
							index = mesh.Vertices.Count;
							mesh.Vertices.Add(new PlyMeshVertex((float)p.X, (float)p.Y, (float)p.Z));
							vertexOfEdge.Add(key, index);
						}

						crossingVertex[e] = index;
					}

					for (int f = 0; f < 6; f++)
					{
						AddFaceSegments(f, values, insideMask, isoLevel, next);
					}

					EmitLoops(mesh, crossingVertex, next, loop);
				}
			}
		}

		return mesh;
	}

	// Pairs the crossed edges of face f into directed segments, recorded as next[from] = to.
	private static void AddFaceSegments(int f, ReadOnlySpan<double> values, int insideMask, double iso, Span<int> next)
	{
		int[] corners = FaceCorners[f];
		Span<bool> inside = stackalloc bool[4];
		Span<int> crossed = stackalloc int[4];
		int count = 0;
		for (int m = 0; m < 4; m++)
		{
			inside[m] = ((insideMask >> corners[m]) & 1) != 0;
		}

		for (int m = 0; m < 4; m++)
		{
			if (inside[m] != inside[(m + 1) % 4])
			{
				crossed[count++] = m;
			}
		}

		if (count == 2)
		{
			// Adjacent crossed edges cut off the corner between them; opposite ones split the face,
			// and any corner is on a definite side.
			int reference = crossed[1] == crossed[0] + 1 ? crossed[1] : 0;
			AddSegment(f, crossed[0], crossed[1], reference, inside[reference], next);
			return;
		}

		if (count == 4)
		{
			// Two diagonal inside corners. The asymptotic decider: the bilinear interpolant's saddle
			// value tells whether the inside corners connect across the face.
			double v0 = values[corners[0]], v1 = values[corners[1]], v2 = values[corners[2]], v3 = values[corners[3]];
			double denominator = v0 + v2 - v1 - v3;
			bool insideConnected = denominator != 0 && (v0 * v2 - v1 * v3) / denominator >= iso;

			// Cut off the corners that are not connected: the outside ones if the inside connects.
			int first = inside[0] != insideConnected ? 0 : 1;
			for (int c = first; c < 4; c += 2)
			{
				AddSegment(f, (c + 3) % 4, c, c, inside[c], next);
			}
		}
	}

	// Adds the segment between face edges m1 and m2 (edge m joins face corners m and m+1), directed
	// so that cross(n, b - a) points away from the inside side; `reference` is a face corner off
	// the segment and `referenceInside` its state.
	private static void AddSegment(int f, int m1, int m2, int reference, bool referenceInside, Span<int> next)
	{
		int[] corners = FaceCorners[f];
		Vector3d pa = EdgeMidpoint(corners[m1], corners[(m1 + 1) % 4]);
		Vector3d pb = EdgeMidpoint(corners[m2], corners[(m2 + 1) % 4]);
		Vector3d d = FaceNormals[f].Cross(pb - pa);
		Vector3d toReference = CornerPosition(corners[reference]) - (pa + pb) * 0.5;
		bool pointsAtReference = d.Dot(toReference) > 0;
		int ea = EdgeOfCorners[corners[m1], corners[(m1 + 1) % 4]];
		int eb = EdgeOfCorners[corners[m2], corners[(m2 + 1) % 4]];
		if (pointsAtReference == referenceInside)
		{
			(ea, eb) = (eb, ea);
		}

		next[ea] = eb;
	}

	// Follows the segments into closed loops (lowest edge first) and triangulates each.
	private static void EmitLoops(PlyMesh mesh, ReadOnlySpan<int> crossingVertex, Span<int> next, Span<int> loop)
	{
		for (int start = 0; start < 12; start++)
		{
			if (next[start] < 0)
			{
				continue;
			}

			int length = 0;
			int e = start;
			do
			{
				loop[length++] = crossingVertex[e];
				int following = next[e];
				next[e] = -1;
				e = following;
			}
			while (e != start);

			if (length == 3)
			{
				mesh.Faces.Add(new PlyMeshFace(loop[0], loop[1], loop[2]));
				continue;
			}

			// A fan from a loop vertex can lay a triangle flat in a cube face (three of the loop's
			// vertices on one face, as on a face the asymptotic decider joined), and the neighbor
			// cube would lay the same triangle reversed: a non-manifold doubled face. A fan from
			// the loop's centroid, which lies inside the cube, cannot.
			double cx = 0, cy = 0, cz = 0;
			for (int t = 0; t < length; t++)
			{
				PlyMeshVertex v = mesh.Vertices[loop[t]];
				cx += v.X;
				cy += v.Y;
				cz += v.Z;
			}

			int center = mesh.Vertices.Count;
			mesh.Vertices.Add(new PlyMeshVertex((float)(cx / length), (float)(cy / length), (float)(cz / length)));
			for (int t = 0; t < length; t++)
			{
				mesh.Faces.Add(new PlyMeshFace(center, loop[t], loop[(t + 1) % length]));
			}
		}
	}

	private static Vector3d CornerPosition(int c) => new(c & 1, (c >> 1) & 1, (c >> 2) & 1);

	private static Vector3d EdgeMidpoint(int a, int b) => (CornerPosition(a) + CornerPosition(b)) * 0.5;

	private static int[,] BuildEdgeOfCorners()
	{
		var table = new int[8, 8];
		for (int e = 0; e < Edges.Length; e++)
		{
			(int a, int axis) = Edges[e];
			int b = a + (1 << axis);
			table[a, b] = e;
			table[b, a] = e;
		}

		return table;
	}
}
