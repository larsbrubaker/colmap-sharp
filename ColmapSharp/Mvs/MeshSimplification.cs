// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MeshSimplification: colmap/mvs/mesh_simplification.h and the entry point of
// mesh_simplification.cc - MeshSimplificationOptions and SimplifyMesh, Quadric Error Metric
// decimation (Garland & Heckbert, "Surface Simplification Using Quadric Error Metrics",
// SIGGRAPH 1997) of a PlyMesh (Util/PlyTypes.cs). The work is done by MeshSimplifier
// (MeshSimplifier*.cs) with its queue in CollapseHeap.cs and adjacency in
// SortedIndexLists.cs. Tests: ColmapSharp.Tests/Mvs/MeshSimplificationTests.cs
// (mesh_simplification_test.cc 1:1).
//
// Tier C (outcome): the collapse order follows COLMAP's (the queue is libc++'s heap with
// COLMAP's cost-only comparator, CollapseHeap.cs), but costs from non-singular 4x4 systems
// and boundary quadric sums can differ in the last bits (docs/CPP_DIVERGENCES.md, entries 72
// and 73), which can reorder near-equal collapses. Flat meshes without boundary weight match
// pycolmap byte for byte; curved ones matched on every oracle case. The result does not
// depend on the thread count.
//
// COLMAP's LOG(INFO) progress lines are not ported; the 10% progress steps are reported
// through IProgress instead, and the collapse loop honors a CancellationToken.

using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>Port of colmap::mvs::MeshSimplificationOptions.</summary>
public sealed class MeshSimplificationOptions
{
	/// <summary>Fraction of faces to retain, in (0, 1].</summary>
	public double TargetFaceRatio { get; set; } = 0.1;

	/// <summary>Maximum quadric error per collapse; 0 = disabled.</summary>
	public double MaxError { get; set; }

	/// <summary>Penalty weight for boundary edges; 0 = disabled.</summary>
	public double BoundaryWeight { get; set; } = 1000.0;

	/// <summary>Blend colors on collapse vs. pick the lower-error vertex's color.</summary>
	public bool InterpolateColors { get; set; } = true;

	/// <summary>The number of threads to use for initialization. Default (-1) is all threads.</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>
	/// Port of MeshSimplificationOptions::Check: false if an option is out of range (COLMAP's
	/// CHECK_OPTION logs and returns false; SimplifyMesh wraps it in a throwing check).
	/// </summary>
	public bool Check()
	{
		return TargetFaceRatio > 0.0 && TargetFaceRatio <= 1.0
			&& MaxError >= 0.0
			&& BoundaryWeight >= 0.0
			&& NumThreads >= -1 && NumThreads != 0;
	}
}

/// <summary>Port of the free function colmap::mvs::SimplifyMesh.</summary>
public static class MeshSimplification
{
	/// <summary>
	/// Simplify a triangle mesh using Quadric Error Metric (QEM) decimation (Garland &amp;
	/// Heckbert, SIGGRAPH 1997): collapse the cheapest edges until
	/// <see cref="MeshSimplificationOptions.TargetFaceRatio"/> of the faces remain, or the next
	/// collapse would exceed <see cref="MeshSimplificationOptions.MaxError"/>. Returns a new
	/// mesh; the input is not modified.
	/// </summary>
	/// <param name="mesh">The mesh to simplify.</param>
	/// <param name="options">The simplification options; they must pass Check.</param>
	/// <param name="progress">Receives the fraction of the faces to remove that are gone, in 10% steps.</param>
	/// <param name="cancellationToken">Stops a long simplification with an OperationCanceledException.</param>
	public static PlyMesh SimplifyMesh(
		PlyMesh mesh,
		MeshSimplificationOptions options,
		IProgress<double>? progress = null,
		CancellationToken cancellationToken = default)
	{
		Util.Check.That(options.Check());

		if (mesh.Faces.Count == 0 || mesh.Vertices.Count == 0)
		{
			return Copy(mesh);
		}

		// Validate that all face vertex indices are within bounds. The unsigned compare also
		// rejects negative indices, which are huge as C++'s size_t.
		uint numVerts = (uint)mesh.Vertices.Count;
		for (int fi = 0; fi < mesh.Faces.Count; fi++)
		{
			PlyMeshFace face = mesh.Faces[fi];
			Util.Check.That((uint)face.VertexIdx1 < numVerts, $"Face {fi} has out-of-bounds vertex index");
			Util.Check.That((uint)face.VertexIdx2 < numVerts, $"Face {fi} has out-of-bounds vertex index");
			Util.Check.That((uint)face.VertexIdx3 < numVerts, $"Face {fi} has out-of-bounds vertex index");
		}

		int numFaces = mesh.Faces.Count;
		int targetFaces = (int)Math.Max(1L, (long)Math.Floor(numFaces * options.TargetFaceRatio));
		if (targetFaces >= numFaces)
		{
			return Copy(mesh);
		}

		return new MeshSimplifier(mesh, options).Run(targetFaces, progress, cancellationToken);
	}

	// C++ returns the input by value.
	private static PlyMesh Copy(PlyMesh mesh)
	{
		return new PlyMesh { Vertices = [.. mesh.Vertices], Faces = [.. mesh.Faces] };
	}
}
