// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// HullSurfaceFusion: how the visual hull (VisualHull.cs) meets the dense surface in object mode
// (docs/QUALITY_PLAN.md, stage 3b; AutomaticReconstruction.Object.cs runs it). Not a COLMAP
// port. Written from C. Hernández Esteban and F. Schmitt, "Silhouette and stereo fusion for 3D
// object modeling", CVIU 96(3), 2004: stereo gives the accurate surface where it is textured,
// the silhouettes bound it everywhere, so the hull stands in where stereo has nothing.
//
// Two steps:
// - GapSamples, before Poisson: hull surface samples (the hull mesh's vertices with
//   area-weighted normals) where the fused cloud has no point within a distance d. Where stereo
//   saw the surface the hull adds nothing, so its looseness (concavities, k-disagreement
//   inflation) never competes with measured points; where stereo saw nothing (the dark or
//   untextured side) Poisson closes the surface along the hull rather than ballooning.
//   Low weight: PoissonMeshingOptions has no per-point weight (COLMAP's PoissonRecon call passes
//   none, and the ported solver's Confidence mode is not exposed), so low weight is emulated by
//   sampling density, which is what a Poisson sample's weight is: one sample per hull vertex,
//   about one per voxel face, is far sparser than a fused cloud's roughly one point per pixel
//   footprint, and the gap test removes them wherever the two would overlap.
// - After Poisson (which runs untrimmed, so its surface is watertight), CleanUp:
//   RemoveOutsideComponents drops the connected pieces that lie wholly outside the silhouettes,
//   then PullInsideHull moves every remaining vertex outside the hull onto the nearest hull
//   surface point, unless that would flip one of its faces (at concave creases of the hull;
//   those vertices stay outside). Faces may collapse to zero area where vertices snap to one
//   hull point; they are kept rather than cut. Neither cuts a face out of a piece that is kept, so a
//   closed Poisson mesh stays closed. This replaces Poisson's
//   density trim in object mode: the trim threshold is a guess per capture, whereas the
//   silhouettes say directly where the object is not. (Cutting single faces by the silhouettes
//   was tried first; it opened holes along the silhouette rims.)

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Silhouette;

/// <summary>Combines a visual hull with a fused dense cloud and trims meshes by silhouettes.</summary>
public static class HullSurfaceFusion
{
	/// <summary>The grey given to hull samples, which carry no photo colour.</summary>
	public const byte SampleGrey = 128;

	/// <summary>
	/// The vertices of <paramref name="hullMesh"/> (with unit, area-weighted outward normals) that
	/// have no point of <paramref name="cloud"/> within <paramref name="distance"/>, in vertex order.
	/// Vertices on no face, or whose faces are degenerate, are skipped (they have no normal).
	/// </summary>
	public static List<PlyPoint> GapSamples(PlyMesh hullMesh, IReadOnlyList<Vector3d> cloud, double distance)
	{
		Check.NotNull(hullMesh);
		Check.NotNull(cloud);
		Check.That(distance >= 0);
		double[] normals = VertexNormals(hullMesh);
		var tree = new PointKdTree(cloud);
		var samples = new List<PlyPoint>();
		for (int v = 0; v < hullMesh.Vertices.Count; v++)
		{
			double nx = normals[3 * v], ny = normals[3 * v + 1], nz = normals[3 * v + 2];
			double length = Math.Sqrt(nx * nx + ny * ny + nz * nz);
			if (!(length > 0))
			{
				continue;
			}

			PlyMeshVertex vertex = hullMesh.Vertices[v];
			if (tree.NearestDistance(new Vector3d(vertex.X, vertex.Y, vertex.Z)) <= distance)
			{
				continue;
			}

			samples.Add(new PlyPoint
			{
				X = vertex.X,
				Y = vertex.Y,
				Z = vertex.Z,
				Nx = (float)(nx / length),
				Ny = (float)(ny / length),
				Nz = (float)(nz / length),
				R = SampleGrey,
				G = SampleGrey,
				B = SampleGrey,
			});
		}

		return samples;
	}

	/// <summary>
	/// <paramref name="mesh"/> with every vertex that lies outside the hull (where
	/// <paramref name="occupancy"/>, the grid the hull surface was extracted from, samples below
	/// <paramref name="isoLevel"/>) moved to the nearest point of <paramref name="hullMesh"/>.
	/// Only positions change, so the faces, and with them closedness, are untouched. A move is
	/// kept only if no face using the vertex ends up flipped (its normal more than 90 degrees from
	/// the original): where the hull is concave, neighbouring vertices can snap to opposite sides
	/// of a crease and fold the faces between them. Such vertices stay where Poisson put them,
	/// outside the hull, so the pull is a best effort there. Faces can still collapse to zero area
	/// where several vertices snap to the same hull point (a sharp hull edge or corner); they are
	/// kept, since removing them would open the surface, and they cost nothing in the result.
	/// </summary>
	public static PlyMesh PullInsideHull(PlyMesh mesh, PlyMesh hullMesh, OccupancyGrid occupancy, double isoLevel = 0.5)
	{
		Check.NotNull(mesh);
		Check.NotNull(hullMesh);
		Check.NotNull(occupancy);
		var result = new PlyMesh { Faces = [.. mesh.Faces] };
		var vertices = mesh.Vertices.ToArray();
		if (hullMesh.Faces.Count > 0)
		{
			TriangleBvh bvh = Bvh(hullMesh);
			System.Threading.Tasks.Parallel.For(0, vertices.Length, v =>
			{
				PlyMeshVertex vertex = vertices[v];
				var p = new Vector3d(vertex.X, vertex.Y, vertex.Z);
				if (occupancy.Sample(p) >= isoLevel)
				{
					return;
				}

				bvh.ClosestPoint(p, out Vector3d closest, out _);
				vertices[v] = new PlyMeshVertex((float)closest.X, (float)closest.Y, (float)closest.Z, vertex.R, vertex.G, vertex.B);
			});
			RevertFlips(mesh, vertices);
		}

		result.Vertices = [.. vertices];
		return result;
	}

	// Undoes the moves of every vertex of a face that the moves flipped, repeating
	// until no face is (undoing one vertex can flip a neighbouring face). Each pass only undoes
	// moves, and with none left every face is as it was, so it terminates. Sequential, so the
	// result does not depend on thread scheduling.
	private static void RevertFlips(PlyMesh original, PlyMeshVertex[] moved)
	{
		bool changed = true;
		while (changed)
		{
			changed = false;
			foreach (PlyMeshFace face in original.Faces)
			{
				Vector3d before = FaceNormal(original.Vertices[face.VertexIdx1], original.Vertices[face.VertexIdx2], original.Vertices[face.VertexIdx3]);
				Vector3d after = FaceNormal(moved[face.VertexIdx1], moved[face.VertexIdx2], moved[face.VertexIdx3]);
				if (before.SquaredNorm == 0 || before.Dot(after) >= 0)
				{
					continue;
				}

				foreach (int v in (ReadOnlySpan<int>)[face.VertexIdx1, face.VertexIdx2, face.VertexIdx3])
				{
					if (!moved[v].Equals(original.Vertices[v]))
					{
						moved[v] = original.Vertices[v];
						changed = true;
					}
				}
			}
		}
	}

	private static Vector3d FaceNormal(PlyMeshVertex a, PlyMeshVertex b, PlyMeshVertex c)
	{
		var u = new Vector3d((double)b.X - a.X, (double)b.Y - a.Y, (double)b.Z - a.Z);
		var w = new Vector3d((double)c.X - a.X, (double)c.Y - a.Y, (double)c.Z - a.Z);
		return u.Cross(w);
	}

	/// <summary>
	/// Object mode's clean-up of the Poisson mesh: <see cref="RemoveOutsideComponents"/>, then
	/// <see cref="PullInsideHull"/> on what is left. In that order: pulled first, a floater wholly
	/// outside the object would be flattened onto the hull and then kept, since it would touch it.
	/// </summary>
	/// A piece within <paramref name="keepWithin"/> of the hull surface anywhere is part of the
	/// object (a Poisson surface a little outside the silhouettes all round), not a floater.
	public static PlyMesh CleanUp(
		PlyMesh mesh,
		PlyMesh hullMesh,
		OccupancyGrid occupancy,
		double isoLevel,
		IReadOnlyList<VisualHullView> views,
		int tolerance,
		double keepWithin) =>
		PullInsideHull(RemoveOutsideComponents(mesh, views, tolerance, hullMesh, keepWithin), hullMesh, occupancy, isoLevel);

	/// <summary>
	/// <paramref name="mesh"/> without its connected components (faces joined through shared
	/// vertices) that lie wholly outside the silhouettes: every face's centroid is outside the mask
	/// (value below 0.5) in more than <paramref name="tolerance"/> of the <paramref name="views"/>
	/// that see it, and, when <paramref name="hullMesh"/> is given, farther than
	/// <paramref name="keepWithin"/> from its surface. A component that touches the object
	/// anywhere is kept whole, so a closed mesh stays closed. Vertices no kept face uses are
	/// dropped; the rest keep their order and colour.
	/// </summary>
	public static PlyMesh RemoveOutsideComponents(
		PlyMesh mesh, IReadOnlyList<VisualHullView> views, int tolerance, PlyMesh? hullMesh = null, double keepWithin = 0)
	{
		TriangleBvh? hullBvh = hullMesh is { Faces.Count: > 0 } ? Bvh(hullMesh) : null;
		Check.NotNull(mesh);
		Check.NotNull(views);
		Check.That(tolerance >= 0);
		var masks = new HullMaskView[views.Count];
		for (int v = 0; v < masks.Length; v++)
		{
			masks[v] = new HullMaskView(views[v], offImageIsOutside: false);
		}

		// Each face is judged on its own, so the faces are tested in parallel into their own slots.
		var outsideFace = new bool[mesh.Faces.Count];
		System.Threading.Tasks.Parallel.For(0, outsideFace.Length, f =>
		{
			PlyMeshFace face = mesh.Faces[f];
			PlyMeshVertex a = mesh.Vertices[face.VertexIdx1], b = mesh.Vertices[face.VertexIdx2], c = mesh.Vertices[face.VertexIdx3];
			var centroid = new Vector3d(
				((double)a.X + b.X + c.X) / 3,
				((double)a.Y + b.Y + c.Y) / 3,
				((double)a.Z + b.Z + c.Z) / 3);
			int outside = 0;
			foreach (HullMaskView mask in masks)
			{
				if (mask.PointValue(centroid) is double value && value < 0.5 && ++outside > tolerance)
				{
					break;
				}
			}

			outsideFace[f] = outside > tolerance && (hullBvh is null || hullBvh.ClosestPoint(centroid, out _, out _) > keepWithin);
		});

		// Components by union-find over the faces' vertices; a component is kept when any face is not outside.
		var parent = new int[mesh.Vertices.Count];
		for (int v = 0; v < parent.Length; v++)
		{
			parent[v] = v;
		}

		foreach (PlyMeshFace face in mesh.Faces)
		{
			Union(face.VertexIdx1, face.VertexIdx2);
			Union(face.VertexIdx2, face.VertexIdx3);
		}

		var keepRoot = new bool[parent.Length];
		for (int f = 0; f < outsideFace.Length; f++)
		{
			if (!outsideFace[f])
			{
				keepRoot[Find(mesh.Faces[f].VertexIdx1)] = true;
			}
		}

		var remap = new int[parent.Length];
		var result = new PlyMesh();
		for (int v = 0; v < parent.Length; v++)
		{
			remap[v] = keepRoot[Find(v)] ? result.Vertices.Count : -1;
			if (remap[v] >= 0)
			{
				result.Vertices.Add(mesh.Vertices[v]);
			}
		}

		foreach (PlyMeshFace face in mesh.Faces)
		{
			if (remap[face.VertexIdx1] >= 0)
			{
				result.Faces.Add(new PlyMeshFace(remap[face.VertexIdx1], remap[face.VertexIdx2], remap[face.VertexIdx3]));
			}
		}

		return result;

		int Find(int v)
		{
			while (parent[v] != v)
			{
				parent[v] = parent[parent[v]];
				v = parent[v];
			}

			return v;
		}

		void Union(int a, int b)
		{
			int ra = Find(a), rb = Find(b);
			if (ra != rb)
			{
				parent[Math.Max(ra, rb)] = Math.Min(ra, rb);
			}
		}
	}

	/// <summary>
	/// Whether every edge of <paramref name="mesh"/> is shared by exactly two faces (closed and
	/// edge-manifold): the "the user gets a closed shape" check.
	/// </summary>
	public static bool IsClosed(PlyMesh mesh)
	{
		Check.NotNull(mesh);
		if (mesh.Faces.Count == 0)
		{
			return false;
		}

		var edges = new Dictionary<(int, int), int>();
		foreach (PlyMeshFace face in mesh.Faces)
		{
			Count(face.VertexIdx1, face.VertexIdx2);
			Count(face.VertexIdx2, face.VertexIdx3);
			Count(face.VertexIdx3, face.VertexIdx1);
		}

		return edges.Values.All(n => n == 2);

		void Count(int i, int j)
		{
			var key = i < j ? (i, j) : (j, i);
			edges[key] = edges.GetValueOrDefault(key) + 1;
		}
	}

	private static TriangleBvh Bvh(PlyMesh mesh)
	{
		var corners = new float[9 * mesh.Faces.Count];
		var ids = new int[mesh.Faces.Count];
		for (int f = 0; f < ids.Length; f++)
		{
			ids[f] = f;
			PlyMeshFace face = mesh.Faces[f];
			int c = 9 * f;
			foreach (int v in (ReadOnlySpan<int>)[face.VertexIdx1, face.VertexIdx2, face.VertexIdx3])
			{
				corners[c++] = mesh.Vertices[v].X;
				corners[c++] = mesh.Vertices[v].Y;
				corners[c++] = mesh.Vertices[v].Z;
			}
		}

		return new TriangleBvh(corners, ids);
	}

	// Per-vertex sums of the face normals' cross products (length = twice the area), so larger
	// faces count more; the hull is wound outward, so these point out of the object.
	private static double[] VertexNormals(PlyMesh mesh)
	{
		var normals = new double[3 * mesh.Vertices.Count];
		foreach (PlyMeshFace face in mesh.Faces)
		{
			PlyMeshVertex a = mesh.Vertices[face.VertexIdx1], b = mesh.Vertices[face.VertexIdx2], c = mesh.Vertices[face.VertexIdx3];
			double ux = (double)b.X - a.X, uy = (double)b.Y - a.Y, uz = (double)b.Z - a.Z;
			double wx = (double)c.X - a.X, wy = (double)c.Y - a.Y, wz = (double)c.Z - a.Z;
			double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
			foreach (int v in (ReadOnlySpan<int>)[face.VertexIdx1, face.VertexIdx2, face.VertexIdx3])
			{
				normals[3 * v] += nx;
				normals[3 * v + 1] += ny;
				normals[3 * v + 2] += nz;
			}
		}

		return normals;
	}
}
