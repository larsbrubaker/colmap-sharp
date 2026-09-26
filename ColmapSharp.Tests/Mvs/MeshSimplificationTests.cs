// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MeshSimplificationTests: colmap/mvs/mesh_simplification_test.cc ported 1:1, one method per
// gtest TEST(Suite, Name) named Suite_Name, testing ColmapSharp/Mvs/MeshSimplification.cs.
// Tier C (outcome): COLMAP's own cases already check bounds, validity and geometry rather
// than exact meshes. C#-only cases live next to it: MeshSimplificationTests.CSharpOnly.cs
// (thread-count determinism, cancellation, progress, the 4x4 solve against Matrix4d) and
// MeshSimplificationOracleTests.cs (exact and Tier C matches against COLMAP).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class MeshSimplificationTests
{
	private static PlyMesh CreateTetrahedronMesh()
	{
		var mesh = new PlyMesh();
		mesh.Vertices.Add(new PlyMeshVertex(0.0f, 0.0f, 0.0f, 255, 0, 0));
		mesh.Vertices.Add(new PlyMeshVertex(1.0f, 0.0f, 0.0f, 0, 255, 0));
		mesh.Vertices.Add(new PlyMeshVertex(0.5f, (float)(Math.Sqrt(3.0) / 2.0), 0.0f, 0, 0, 255));
		mesh.Vertices.Add(new PlyMeshVertex(0.5f, (float)(Math.Sqrt(3.0) / 6.0), (float)(Math.Sqrt(6.0) / 3.0), 255, 255, 0));

		mesh.Faces.Add(new PlyMeshFace(0, 2, 1));
		mesh.Faces.Add(new PlyMeshFace(0, 1, 3));
		mesh.Faces.Add(new PlyMeshFace(1, 2, 3));
		mesh.Faces.Add(new PlyMeshFace(0, 3, 2));
		return mesh;
	}

	private static void AddGridFaces(PlyMesh mesh, int n)
	{
		for (int j = 0; j < n; ++j)
		{
			for (int i = 0; i < n; ++i)
			{
				int v00 = j * (n + 1) + i;
				int v10 = j * (n + 1) + (i + 1);
				int v01 = (j + 1) * (n + 1) + i;
				int v11 = (j + 1) * (n + 1) + (i + 1);
				mesh.Faces.Add(new PlyMeshFace(v00, v10, v11));
				mesh.Faces.Add(new PlyMeshFace(v00, v11, v01));
			}
		}
	}

	internal static PlyMesh CreateGridMesh(int n)
	{
		var mesh = new PlyMesh();
		for (int j = 0; j <= n; ++j)
		{
			for (int i = 0; i <= n; ++i)
			{
				mesh.Vertices.Add(new PlyMeshVertex(i, j, 0.0f,
					(byte)(i * 255 / Math.Max(n, 1)), (byte)(j * 255 / Math.Max(n, 1)), 128));
			}
		}

		AddGridFaces(mesh, n);
		return mesh;
	}

	internal static PlyMesh CreateWavyGridMesh(int n)
	{
		var mesh = new PlyMesh();
		for (int j = 0; j <= n; ++j)
		{
			for (int i = 0; i <= n; ++i)
			{
				float x = i;
				float y = j;
				float z = MathF.Sin(x * 0.5f) * MathF.Cos(y * 0.5f);
				mesh.Vertices.Add(new PlyMeshVertex(x, y, z));
			}
		}

		AddGridFaces(mesh, n);
		return mesh;
	}

	private static async Task ExpectValidFaces(PlyMesh result)
	{
		foreach (PlyMeshFace face in result.Faces)
		{
			await Assert.That(face.VertexIdx1).IsGreaterThanOrEqualTo(0).And.IsLessThan(result.Vertices.Count);
			await Assert.That(face.VertexIdx2).IsGreaterThanOrEqualTo(0).And.IsLessThan(result.Vertices.Count);
			await Assert.That(face.VertexIdx3).IsGreaterThanOrEqualTo(0).And.IsLessThan(result.Vertices.Count);
		}
	}

	[Test]
	public async Task MeshSimplificationOptions_DefaultsAreValid()
	{
		var options = new MeshSimplificationOptions();
		await Assert.That(options.Check()).IsTrue();
	}

	[Test]
	public async Task MeshSimplificationOptions_InvalidRatio()
	{
		var options = new MeshSimplificationOptions();

		options.TargetFaceRatio = 0.0;
		await Assert.That(options.Check()).IsFalse();

		options.TargetFaceRatio = -0.5;
		await Assert.That(options.Check()).IsFalse();

		options.TargetFaceRatio = 1.5;
		await Assert.That(options.Check()).IsFalse();
	}

	[Test]
	public async Task MeshSimplificationOptions_InvalidMaxError()
	{
		var options = new MeshSimplificationOptions { MaxError = -1.0 };
		await Assert.That(options.Check()).IsFalse();
	}

	[Test]
	public async Task SimplifyMesh_IdentityWithRatioOne()
	{
		PlyMesh mesh = CreateTetrahedronMesh();
		var options = new MeshSimplificationOptions { TargetFaceRatio = 1.0 };

		PlyMesh result = MeshSimplification.SimplifyMesh(mesh, options);

		await Assert.That(result.Faces.Count).IsEqualTo(mesh.Faces.Count);
		await Assert.That(result.Vertices.Count).IsEqualTo(mesh.Vertices.Count);
	}

	[Test]
	public async Task SimplifyMesh_ReducesFaceCount()
	{
		PlyMesh mesh = CreateGridMesh(10); // 200 faces
		await Assert.That(mesh.Faces.Count).IsEqualTo(200);

		var options = new MeshSimplificationOptions { TargetFaceRatio = 0.5 };

		PlyMesh result = MeshSimplification.SimplifyMesh(mesh, options);

		// Should be approximately 100 faces, allow some tolerance.
		await Assert.That(result.Faces.Count).IsLessThanOrEqualTo(110);
		await Assert.That(result.Faces.Count).IsGreaterThanOrEqualTo(50);
		await Assert.That(result.Faces.Count).IsLessThan(mesh.Faces.Count);
	}

	[Test]
	public async Task SimplifyMesh_MaxErrorThreshold()
	{
		// Use a wavy surface so collapses have non-zero quadric error.
		PlyMesh mesh = CreateWavyGridMesh(10); // 200 faces
		await Assert.That(mesh.Faces.Count).IsEqualTo(200);

		var options = new MeshSimplificationOptions { TargetFaceRatio = 0.1 };

		// Without error threshold.
		options.MaxError = 0.0;
		PlyMesh resultNoLimit = MeshSimplification.SimplifyMesh(mesh, options);

		// With strict error threshold.
		options.MaxError = 1e-6;
		PlyMesh resultStrict = MeshSimplification.SimplifyMesh(mesh, options);

		// Strict threshold should preserve more faces.
		await Assert.That(resultStrict.Faces.Count).IsGreaterThan(resultNoLimit.Faces.Count);
	}

	[Test]
	public async Task SimplifyMesh_PreservesColorsAtRatioOne()
	{
		PlyMesh mesh = CreateTetrahedronMesh();
		var options = new MeshSimplificationOptions { TargetFaceRatio = 1.0 };

		PlyMesh result = MeshSimplification.SimplifyMesh(mesh, options);

		await Assert.That(result.Vertices.Count).IsEqualTo(mesh.Vertices.Count);
		for (int i = 0; i < mesh.Vertices.Count; ++i)
		{
			await Assert.That(result.Vertices[i].R).IsEqualTo(mesh.Vertices[i].R);
			await Assert.That(result.Vertices[i].G).IsEqualTo(mesh.Vertices[i].G);
			await Assert.That(result.Vertices[i].B).IsEqualTo(mesh.Vertices[i].B);
		}
	}

	[Test]
	public async Task SimplifyMesh_OutputMeshIsValid()
	{
		PlyMesh mesh = CreateGridMesh(10);
		var options = new MeshSimplificationOptions { TargetFaceRatio = 0.5 };

		PlyMesh result = MeshSimplification.SimplifyMesh(mesh, options);

		await Assert.That(result.Faces.Count).IsGreaterThan(0);
		await Assert.That(result.Vertices.Count).IsGreaterThan(0);

		await ExpectValidFaces(result);
		foreach (PlyMeshFace face in result.Faces)
		{
			// No degenerate faces.
			await Assert.That(face.VertexIdx1).IsNotEqualTo(face.VertexIdx2);
			await Assert.That(face.VertexIdx1).IsNotEqualTo(face.VertexIdx3);
			await Assert.That(face.VertexIdx2).IsNotEqualTo(face.VertexIdx3);
		}
	}

	[Test]
	public async Task SimplifyMesh_BoundaryPreservation()
	{
		const int n = 10;
		PlyMesh mesh = CreateGridMesh(n);
		var options = new MeshSimplificationOptions { TargetFaceRatio = 0.3 };

		// With high boundary weight.
		options.BoundaryWeight = 1e6;
		PlyMesh resultHigh = MeshSimplification.SimplifyMesh(mesh, options);

		// Verify that the bounding box is approximately preserved.
		float minX = float.MaxValue;
		float maxX = float.MinValue;
		float minY = float.MaxValue;
		float maxY = float.MinValue;
		foreach (PlyMeshVertex v in resultHigh.Vertices)
		{
			minX = Math.Min(minX, v.X);
			maxX = Math.Max(maxX, v.X);
			minY = Math.Min(minY, v.Y);
			maxY = Math.Max(maxY, v.Y);
		}

		await Assert.That(Math.Abs(minX - 0.0f)).IsLessThanOrEqualTo(1.0f);
		await Assert.That(Math.Abs(maxX - n)).IsLessThanOrEqualTo(1.0f);
		await Assert.That(Math.Abs(minY - 0.0f)).IsLessThanOrEqualTo(1.0f);
		await Assert.That(Math.Abs(maxY - n)).IsLessThanOrEqualTo(1.0f);

		// Compare with no boundary weight: high weight should preserve at least as many faces
		// (within a small tolerance for different collapse orderings due to numerical precision).
		options.BoundaryWeight = 0.0;
		PlyMesh resultNone = MeshSimplification.SimplifyMesh(mesh, options);
		await Assert.That(resultHigh.Faces.Count + 5).IsGreaterThanOrEqualTo(resultNone.Faces.Count);
	}

	[Test]
	public async Task SimplifyMesh_SingleTriangle()
	{
		var mesh = new PlyMesh();
		mesh.Vertices.Add(new PlyMeshVertex(0.0f, 0.0f, 0.0f, 255, 0, 0));
		mesh.Vertices.Add(new PlyMeshVertex(1.0f, 0.0f, 0.0f, 0, 255, 0));
		mesh.Vertices.Add(new PlyMeshVertex(0.0f, 1.0f, 0.0f, 0, 0, 255));
		mesh.Faces.Add(new PlyMeshFace(0, 1, 2));

		var options = new MeshSimplificationOptions { TargetFaceRatio = 0.5 };

		PlyMesh result = MeshSimplification.SimplifyMesh(mesh, options);

		// A single triangle cannot be simplified below 1 face.
		await Assert.That(result.Faces.Count).IsEqualTo(1);
		await Assert.That(result.Vertices.Count).IsEqualTo(3);
	}

	[Test]
	public async Task SimplifyMesh_OutOfBoundsVertexIndex()
	{
		var mesh = new PlyMesh();
		mesh.Vertices.Add(new PlyMeshVertex(0.0f, 0.0f, 0.0f, 255, 0, 0));
		mesh.Vertices.Add(new PlyMeshVertex(1.0f, 0.0f, 0.0f, 0, 255, 0));
		mesh.Vertices.Add(new PlyMeshVertex(0.0f, 1.0f, 0.0f, 0, 0, 255));

		// Face references vertex index 5, which is out of bounds.
		mesh.Faces.Add(new PlyMeshFace(0, 1, 5));

		var options = new MeshSimplificationOptions { TargetFaceRatio = 0.5 };

		await Assert.That(() => MeshSimplification.SimplifyMesh(mesh, options)).Throws<Exception>();
	}

	[Test]
	public async Task SimplifyMesh_LargerMeshStressTest()
	{
		PlyMesh mesh = CreateGridMesh(50); // 5000 faces
		await Assert.That(mesh.Faces.Count).IsEqualTo(5000);

		var options = new MeshSimplificationOptions { TargetFaceRatio = 0.1 };

		PlyMesh result = MeshSimplification.SimplifyMesh(mesh, options);

		await Assert.That(result.Faces.Count).IsLessThanOrEqualTo(600);
		await Assert.That(result.Faces.Count).IsGreaterThanOrEqualTo(100);
		await Assert.That(result.Faces.Count).IsLessThan(mesh.Faces.Count);

		// Verify output validity.
		await ExpectValidFaces(result);
	}

	[Test]
	public async Task SimplifyMesh_OptimalVertexPlacement()
	{
		// Simplify a flat grid mesh and verify that the QEM optimal vertex positions remain on
		// the original surface (z=0 plane) and within the bounding box of the original mesh.
		PlyMesh mesh = CreateGridMesh(4); // 32 faces

		var options = new MeshSimplificationOptions { TargetFaceRatio = 0.25, BoundaryWeight = 0.0 };

		PlyMesh result = MeshSimplification.SimplifyMesh(mesh, options);

		await Assert.That(result.Faces.Count).IsLessThan(mesh.Faces.Count);
		await Assert.That(result.Vertices.Count).IsGreaterThan(0);

		foreach (PlyMeshVertex v in result.Vertices)
		{
			// All vertices should remain on the z=0 plane.
			await Assert.That(Math.Abs(v.Z - 0.0f)).IsLessThanOrEqualTo(1e-5f);

			// All vertices should remain within the original bounding box.
			await Assert.That(v.X).IsGreaterThanOrEqualTo(-0.01f);
			await Assert.That(v.X).IsLessThanOrEqualTo(4.01f);
			await Assert.That(v.Y).IsGreaterThanOrEqualTo(-0.01f);
			await Assert.That(v.Y).IsLessThanOrEqualTo(4.01f);
		}
	}
}
