// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MeshSimplifier.Collapse: steps 5 and 6 of colmap::mvs::SimplifyMesh - the iterative
// edge-collapse loop with lazy deletion of stale queue entries, and the compaction of the
// surviving faces and vertices into a new PlyMesh. Setup is in MeshSimplifier.cs.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

internal sealed partial class MeshSimplifier
{
	private void CollapseEdges(CollapseHeap heap, int currentFaces, int targetFaces,
		IProgress<double>? progress, CancellationToken cancellationToken)
	{
		int facesToRemove = currentFaces - targetFaces;
		int lastProgressPercent = -1;
		int iteration = 0;

		while (currentFaces > targetFaces && heap.Count > 0)
		{
			if (++iteration % CancellationCheckInterval == 0)
			{
				cancellationToken.ThrowIfCancellationRequested();
			}

			CollapseKey candidate = heap.Pop();
			int v1 = candidate.V1;
			int v2 = candidate.V2;

			// Skip stale candidates (lazy deletion).
			if (vertexRemoved[v1] || vertexRemoved[v2])
			{
				continue;
			}

			if (timestamps[v1] != candidate.TimestampV1 || timestamps[v2] != candidate.TimestampV2)
			{
				continue;
			}

			// Check max error threshold. COLMAP logs "Stopping early" here.
			if (options.MaxError > 0 && candidate.Cost > options.MaxError)
			{
				break;
			}

			// The queue keeps only the key; both vertices are unchanged since it was computed
			// (their timestamps match), so this recomputes the same position (CollapseHeap.cs).
			ComputeEdgeCollapse(v1, v2, out Vector3d optimalPosition, out bool v1HasLowerError);

			// Check for face flips.
			if (WouldCauseFlip(v1, v2, optimalPosition))
			{
				continue;
			}

			ComputeOptimalColor(v1, v2, optimalPosition, v1HasLowerError, out float r, out float g, out float b);

			// Execute collapse: merge v2 into v1.
			positions[v1] = optimalPosition;
			colors[v1 * 3] = r;
			colors[v1 * 3 + 1] = g;
			colors[v1 * 3 + 2] = b;
			AddQuadric(v1, quadrics.AsSpan(v2 * 16, 16));

			currentFaces -= MoveFaces(v1, v2);
			MoveNeighbors(v1, v2);

			// Mark v2 as removed.
			vertexRemoved[v2] = true;
			adjacentFaces.Clear(v2);
			adjacentVertices.Clear(v2);

			// Increment v1's timestamp for lazy queue deletion.
			timestamps[v1]++;

			// Recompute candidates for all edges incident to v1.
			foreach (int u in adjacentVertices[v1])
			{
				if (vertexRemoved[u])
				{
					continue;
				}

				heap.Push(MakeCandidate(v1, u));
			}

			// Report progress at 10% intervals, where COLMAP logs it. faces_removed counts
			// the degenerate input faces too, as in C++.
			int facesRemoved = numFaces - currentFaces;
			int progressPercent = (int)(100.0 * facesRemoved / facesToRemove);
			if (progressPercent / 10 > lastProgressPercent / 10)
			{
				lastProgressPercent = progressPercent;
				progress?.Report(Math.Min(1.0, progressPercent / 100.0));
			}
		}
	}

	// Moves v2's faces to v1: the faces shared by both are removed, the others are remapped.
	// Returns the number of faces removed.
	private int MoveFaces(int v1, int v2)
	{
		int removed = 0;
		foreach (int fi in adjacentFaces[v2])
		{
			if (faceRemoved[fi])
			{
				continue;
			}

			Span<int> f = faceIndices.AsSpan(fi * 3, 3);

			// Check if this is a shared face (contains both v1 and v2).
			if (f[0] == v1 || f[1] == v1 || f[2] == v1)
			{
				// Shared face: remove it.
				faceRemoved[fi] = true;
				removed++;
				for (int j = 0; j < 3; j++)
				{
					if (f[j] != v1 && f[j] != v2)
					{
						adjacentFaces.Erase(f[j], fi);
					}
				}

				adjacentFaces.Erase(v1, fi);
				continue;
			}

			// Remap v2 -> v1 in this face.
			for (int j = 0; j < 3; j++)
			{
				if (f[j] == v2)
				{
					f[j] = v1;
					break;
				}
			}

			// Check for a degenerate face after remapping.
			if (f[0] == f[1] || f[0] == f[2] || f[1] == f[2])
			{
				faceRemoved[fi] = true;
				removed++;
				for (int j = 0; j < 3; j++)
				{
					adjacentFaces.Erase(f[j], fi);
				}
			}
			else
			{
				adjacentFaces.Insert(v1, fi);
			}
		}

		return removed;
	}

	// Moves v2's neighbors to v1.
	private void MoveNeighbors(int v1, int v2)
	{
		foreach (int u in adjacentVertices[v2])
		{
			if (u == v1)
			{
				continue;
			}

			adjacentVertices.Erase(u, v2);
			adjacentVertices.Insert(u, v1);
			adjacentVertices.Insert(v1, u);
		}

		adjacentVertices.Erase(v1, v2);
	}

	// Compaction: a new mesh with only the surviving faces, and their vertices numbered in
	// order of first use.
	private PlyMesh Compact()
	{
		var result = new PlyMesh();
		var oldToNew = new int[numVertices];
		Array.Fill(oldToNew, -1);

		for (int fi = 0; fi < numFaces; fi++)
		{
			if (faceRemoved[fi])
			{
				continue;
			}

			for (int j = 0; j < 3; j++)
			{
				int v = faceIndices[fi * 3 + j];
				if (oldToNew[v] != -1)
				{
					continue;
				}

				oldToNew[v] = result.Vertices.Count;
				Vector3d p = positions[v];
				result.Vertices.Add(new PlyMeshVertex((float)p.X, (float)p.Y, (float)p.Z,
					ToColorByte(colors[v * 3]), ToColorByte(colors[v * 3 + 1]), ToColorByte(colors[v * 3 + 2])));
			}

			result.Faces.Add(new PlyMeshFace(
				oldToNew[faceIndices[fi * 3]], oldToNew[faceIndices[fi * 3 + 1]], oldToNew[faceIndices[fi * 3 + 2]]));
		}

		return result;
	}

	// color.array().round().max(0).min(255), then static_cast<uint8_t>. std::round rounds
	// halves away from zero; Eigen's max/min are std::max(x, 0) and std::min(x, 255).
	private static byte ToColorByte(float value)
	{
		float rounded = MathF.Round(value, MidpointRounding.AwayFromZero);
		float atLeastZero = rounded < 0.0f ? 0.0f : rounded;
		float clamped = 255.0f < atLeastZero ? 255.0f : atLeastZero;
		return (byte)clamped;
	}
}
