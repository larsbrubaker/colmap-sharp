// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MeshSimplifier: the body of colmap::mvs::SimplifyMesh (colmap/mvs/mesh_simplification.cc)
// - mesh topology, the initial vertex quadrics (face planes plus boundary constraint planes),
// the lazy-deletion edge-collapse loop and the compaction into a new PlyMesh. The public
// entry point and the options are in MeshSimplification.cs; the per-edge math is in
// MeshSimplifier.Candidates.cs; the queue is CollapseHeap.cs and the adjacency lists
// SortedIndexLists.cs.
//
// Storage is struct-of-arrays (positions, colors, row-major 4x4 quadrics, timestamps) so a
// million-face mesh costs a few flat arrays and the collapse loop allocates nothing beyond
// the amortized growth of the queue and the pooled adjacency arrays.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>The mutable state of one SimplifyMesh run.</summary>
internal sealed partial class MeshSimplifier
{
	// How many popped candidates pass between cancellation checks.
	private const int CancellationCheckInterval = 4096;

	private readonly MeshSimplificationOptions options;
	private readonly bool interpolateColors;
	private readonly int numVertices;
	private readonly int numFaces;
	private readonly int numThreads;

	private readonly Vector3d[] positions;
	private readonly float[] colors;
	private readonly double[] quadrics;
	private readonly uint[] timestamps;
	private readonly bool[] vertexRemoved;
	private readonly SortedIndexLists adjacentFaces;
	private readonly SortedIndexLists adjacentVertices;

	private readonly int[] faceIndices;
	private readonly bool[] faceRemoved;

	public MeshSimplifier(PlyMesh mesh, MeshSimplificationOptions options)
	{
		this.options = options;
		interpolateColors = options.InterpolateColors;
		numVertices = mesh.Vertices.Count;
		numFaces = mesh.Faces.Count;
		numThreads = Threading.GetEffectiveNumThreads(options.NumThreads);

		positions = new Vector3d[numVertices];
		colors = new float[numVertices * 3];
		quadrics = new double[numVertices * 16];
		timestamps = new uint[numVertices];
		vertexRemoved = new bool[numVertices];
		adjacentFaces = new SortedIndexLists(numVertices);
		adjacentVertices = new SortedIndexLists(numVertices);
		faceIndices = new int[numFaces * 3];
		faceRemoved = new bool[numFaces];

		for (int i = 0; i < numVertices; i++)
		{
			PlyMeshVertex v = mesh.Vertices[i];
			positions[i] = new Vector3d(v.X, v.Y, v.Z);
			colors[i * 3] = v.R;
			colors[i * 3 + 1] = v.G;
			colors[i * 3 + 2] = v.B;
		}

		for (int fi = 0; fi < numFaces; fi++)
		{
			PlyMeshFace face = mesh.Faces[fi];
			faceIndices[fi * 3] = face.VertexIdx1;
			faceIndices[fi * 3 + 1] = face.VertexIdx2;
			faceIndices[fi * 3 + 2] = face.VertexIdx3;
		}
	}

	/// <summary>Runs the simplification down to <paramref name="targetFaces"/> faces.</summary>
	public PlyMesh Run(int targetFaces, IProgress<double>? progress, CancellationToken cancellationToken)
	{
		int currentFaces = BuildTopology();
		cancellationToken.ThrowIfCancellationRequested();
		ComputeFaceQuadrics();
		if (options.BoundaryWeight > 0)
		{
			AddBoundaryQuadrics();
		}

		cancellationToken.ThrowIfCancellationRequested();
		var heap = new CollapseHeap(ComputeInitialCandidates());
		CollapseEdges(heap, currentFaces, targetFaces, progress, cancellationToken);
		return Compact();
	}

	// Builds the face and vertex adjacency, marking degenerate faces removed, and returns the
	// number of faces left.
	private int BuildTopology()
	{
		int currentFaces = 0;
		for (int fi = 0; fi < numFaces; fi++)
		{
			int f0 = faceIndices[fi * 3], f1 = faceIndices[fi * 3 + 1], f2 = faceIndices[fi * 3 + 2];

			// Skip degenerate faces.
			if (f0 == f1 || f0 == f2 || f1 == f2)
			{
				faceRemoved[fi] = true;
				continue;
			}

			currentFaces++;
			for (int j = 0; j < 3; j++)
			{
				int va = faceIndices[fi * 3 + j];
				int vb = faceIndices[fi * 3 + (j + 1) % 3];
				adjacentFaces.Insert(va, fi);
				adjacentVertices.Insert(va, vb);
				adjacentVertices.Insert(vb, va);
			}
		}

		return currentFaces;
	}

	// Step 1: the per-face plane quadrics (Section 5) accumulated into their vertices. The
	// planes are computed in parallel, each face into its own slot; the accumulation runs in
	// face order, as in C++, so the sums do not depend on the thread count.
	private void ComputeFaceQuadrics()
	{
		var planes = new double[numFaces * 4];
		var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = numThreads };
		Parallel.For(0, numThreads, parallelOptions, chunk =>
		{
			int begin = (int)((long)numFaces * chunk / numThreads);
			int end = (int)((long)numFaces * (chunk + 1) / numThreads);
			for (int fi = begin; fi < end; fi++)
			{
				if (faceRemoved[fi])
				{
					continue;
				}

				Vector3d p0 = positions[faceIndices[fi * 3]];
				Vector3d p1 = positions[faceIndices[fi * 3 + 1]];
				Vector3d p2 = positions[faceIndices[fi * 3 + 2]];
				Vector3d normal = (p1 - p0).Cross(p2 - p0);
				double len = normal.Norm;
				if (len < Epsilon)
				{
					continue;
				}

				normal /= len;
				planes[fi * 4] = normal.X;
				planes[fi * 4 + 1] = normal.Y;
				planes[fi * 4 + 2] = normal.Z;
				planes[fi * 4 + 3] = -normal.Dot(p0);
			}
		});

		for (int fi = 0; fi < numFaces; fi++)
		{
			if (faceRemoved[fi])
			{
				continue;
			}

			// A face whose normal was too short keeps a zero plane, and so a zero quadric.
			ReadOnlySpan<double> plane = planes.AsSpan(fi * 4, 4);
			for (int j = 0; j < 3; j++)
			{
				Span<double> q = quadrics.AsSpan(faceIndices[fi * 3 + j] * 16, 16);
				for (int r = 0; r < 4; r++)
				{
					for (int c = 0; c < 4; c++)
					{
						q[r * 4 + c] += plane[r] * plane[c];
					}
				}
			}
		}
	}

	// Boundary preservation quadrics (Section 6, geometric discontinuity only): every edge
	// with exactly one face adds a plane through the edge, perpendicular to the face, to both
	// of its vertices. C++ finds the edges through a hash map and visits them in its
	// iteration order; here they are visited by (first, second) vertex index, which decides
	// the order of the floating-point sums (docs/CPP_DIVERGENCES.md, entry 73).
	private void AddBoundaryQuadrics()
	{
		Span<double> qBoundary = stackalloc double[16];
		for (int a = 0; a < numVertices; a++)
		{
			foreach (int b in adjacentVertices[a])
			{
				if (b <= a)
				{
					continue;
				}

				// The faces of edge (a, b) are a's faces that contain b.
				int numEdgeFaces = 0;
				int fi = -1;
				foreach (int candidate in adjacentFaces[a])
				{
					if (faceIndices[candidate * 3] == b || faceIndices[candidate * 3 + 1] == b || faceIndices[candidate * 3 + 2] == b)
					{
						numEdgeFaces++;
						fi = candidate;
					}
				}

				if (numEdgeFaces != 1)
				{
					continue;
				}

				Vector3d p0 = positions[faceIndices[fi * 3]];
				Vector3d p1 = positions[faceIndices[fi * 3 + 1]];
				Vector3d p2 = positions[faceIndices[fi * 3 + 2]];
				Vector3d faceNormal = (p1 - p0).Cross(p2 - p0);

				Vector3d edgeDir = positions[b] - positions[a];
				Vector3d constraintNormal = edgeDir.Cross(faceNormal);
				double clen = constraintNormal.Norm;
				if (clen < Epsilon)
				{
					continue;
				}

				constraintNormal /= clen;
				double cd = -constraintNormal.Dot(positions[a]);
				ReadOnlySpan<double> cplane = [constraintNormal.X, constraintNormal.Y, constraintNormal.Z, cd];

				// boundary_weight * cplane * cplane^T associates as (w * cplane) * cplane^T.
				for (int r = 0; r < 4; r++)
				{
					double weighted = options.BoundaryWeight * cplane[r];
					for (int c = 0; c < 4; c++)
					{
						qBoundary[r * 4 + c] = weighted * cplane[c];
					}
				}

				AddQuadric(a, qBoundary);
				AddQuadric(b, qBoundary);
			}
		}
	}

	private void AddQuadric(int vertex, ReadOnlySpan<double> q)
	{
		Span<double> target = quadrics.AsSpan(vertex * 16, 16);
		for (int i = 0; i < 16; i++)
		{
			target[i] += q[i];
		}
	}

	// Steps 2 and 3: every mesh edge (vi, vj) with vi < vj once (not the close vertex pairs
	// of Section 8), with its collapse cost computed in parallel into its own slot.
	private CollapseKey[] ComputeInitialCandidates()
	{
		int numEdges = 0;
		for (int vi = 0; vi < numVertices; vi++)
		{
			foreach (int vj in adjacentVertices[vi])
			{
				if (vi < vj)
				{
					numEdges++;
				}
			}
		}

		var edgeFirst = new int[numEdges];
		var edgeSecond = new int[numEdges];
		int e = 0;
		for (int vi = 0; vi < numVertices; vi++)
		{
			foreach (int vj in adjacentVertices[vi])
			{
				if (vi < vj)
				{
					edgeFirst[e] = vi;
					edgeSecond[e] = vj;
					e++;
				}
			}
		}

		var candidates = new CollapseKey[numEdges];
		var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = numThreads };
		Parallel.For(0, numThreads, parallelOptions, chunk =>
		{
			int begin = (int)((long)numEdges * chunk / numThreads);
			int end = (int)((long)numEdges * (chunk + 1) / numThreads);
			for (int i = begin; i < end; i++)
			{
				candidates[i] = MakeCandidate(edgeFirst[i], edgeSecond[i]);
			}
		});

		return candidates;
	}

	private CollapseKey MakeCandidate(int v1, int v2)
	{
		double cost = ComputeEdgeCollapse(v1, v2, out _, out _);
		return new CollapseKey(cost, v1, v2, timestamps[v1], timestamps[v2]);
	}
}
