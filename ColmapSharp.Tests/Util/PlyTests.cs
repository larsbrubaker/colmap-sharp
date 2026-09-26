// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PlyTests: colmap/util/ply_test.cc ported 1:1, one method per gtest TEST(Suite, Name)
// named Suite_Name, testing ColmapSharp/Util/Ply*.cs. The C#-only cases at the end pin the
// byte-exact text format and the malformed-input behavior (PlyTests.Files.cs).
// Tier A (exact): every comparison is EXPECT_EQ, as in C++.

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

public partial class PlyTests
{
	private static string CreateTestDir()
	{
		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-ply-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		return dir;
	}

	[Test]
	public async Task PlyPoint_DefaultConstructor()
	{
		var point = new PlyPoint();
		await Assert.That(point.X).IsEqualTo(0.0f);
		await Assert.That(point.Y).IsEqualTo(0.0f);
		await Assert.That(point.Z).IsEqualTo(0.0f);
		await Assert.That(point.Nx).IsEqualTo(0.0f);
		await Assert.That(point.Ny).IsEqualTo(0.0f);
		await Assert.That(point.Nz).IsEqualTo(0.0f);
		await Assert.That(point.R).IsEqualTo((byte)0);
		await Assert.That(point.G).IsEqualTo((byte)0);
		await Assert.That(point.B).IsEqualTo((byte)0);
	}

	[Test]
	public async Task PlyMeshVertex_DefaultConstructor()
	{
		var vertex = new PlyMeshVertex();
		await Assert.That(vertex.X).IsEqualTo(0.0f);
		await Assert.That(vertex.Y).IsEqualTo(0.0f);
		await Assert.That(vertex.Z).IsEqualTo(0.0f);
		await Assert.That(vertex.R).IsEqualTo((byte)200);
		await Assert.That(vertex.G).IsEqualTo((byte)200);
		await Assert.That(vertex.B).IsEqualTo((byte)200);
	}

	[Test]
	public async Task PlyMeshVertex_ParameterizedConstructor()
	{
		var vertex = new PlyMeshVertex(1.5f, 2.5f, 3.5f);
		await Assert.That(vertex.X).IsEqualTo(1.5f);
		await Assert.That(vertex.Y).IsEqualTo(2.5f);
		await Assert.That(vertex.Z).IsEqualTo(3.5f);
		await Assert.That(vertex.R).IsEqualTo((byte)200);
		await Assert.That(vertex.G).IsEqualTo((byte)200);
		await Assert.That(vertex.B).IsEqualTo((byte)200);
	}

	[Test]
	public async Task PlyMeshVertex_ColorConstructor()
	{
		var vertex = new PlyMeshVertex(1.0f, 2.0f, 3.0f, 10, 20, 30);
		await Assert.That(vertex.X).IsEqualTo(1.0f);
		await Assert.That(vertex.Y).IsEqualTo(2.0f);
		await Assert.That(vertex.Z).IsEqualTo(3.0f);
		await Assert.That(vertex.R).IsEqualTo((byte)10);
		await Assert.That(vertex.G).IsEqualTo((byte)20);
		await Assert.That(vertex.B).IsEqualTo((byte)30);
	}

	[Test]
	public async Task PlyMeshFace_DefaultConstructor()
	{
		var face = new PlyMeshFace();
		await Assert.That(face.VertexIdx1).IsEqualTo(0);
		await Assert.That(face.VertexIdx2).IsEqualTo(0);
		await Assert.That(face.VertexIdx3).IsEqualTo(0);
	}

	[Test]
	public async Task PlyMeshFace_ParameterizedConstructor()
	{
		var face = new PlyMeshFace(10, 20, 30);
		await Assert.That(face.VertexIdx1).IsEqualTo(10);
		await Assert.That(face.VertexIdx2).IsEqualTo(20);
		await Assert.That(face.VertexIdx3).IsEqualTo(30);
	}

	private static async Task ExpectPointsEqual(List<PlyPoint> loaded, List<PlyPoint> original)
	{
		await Assert.That(loaded.Count).IsEqualTo(original.Count);
		for (int i = 0; i < original.Count; ++i)
		{
			await Assert.That(loaded[i].X).IsEqualTo(original[i].X);
			await Assert.That(loaded[i].Y).IsEqualTo(original[i].Y);
			await Assert.That(loaded[i].Z).IsEqualTo(original[i].Z);
			await Assert.That(loaded[i].Nx).IsEqualTo(original[i].Nx);
			await Assert.That(loaded[i].Ny).IsEqualTo(original[i].Ny);
			await Assert.That(loaded[i].Nz).IsEqualTo(original[i].Nz);
			await Assert.That(loaded[i].R).IsEqualTo(original[i].R);
			await Assert.That(loaded[i].G).IsEqualTo(original[i].G);
			await Assert.That(loaded[i].B).IsEqualTo(original[i].B);
		}
	}

	[Test]
	public async Task Ply_RoundTripTextPlyPointsFullData()
	{
		string testFile = Path.Combine(CreateTestDir(), "test.ply");

		// Create test points with full data
		var originalPoints = new List<PlyPoint>();
		for (int i = 0; i < 3; ++i)
		{
			originalPoints.Add(new PlyPoint
			{
				X = i * 1.0f,
				Y = i * 2.0f,
				Z = i * 3.0f,
				Nx = i * 0.1f,
				Ny = i * 0.2f,
				Nz = i * 0.3f,
				R = (byte)(i * 10),
				G = (byte)(i * 20),
				B = (byte)(i * 30),
			});
		}

		Ply.WriteTextPlyPoints(testFile, originalPoints, true, true);
		List<PlyPoint> loadedPoints = Ply.ReadPly(testFile);
		await ExpectPointsEqual(loadedPoints, originalPoints);
	}

	[Test]
	public async Task Ply_RoundTripTextPlyPointsXYZOnly()
	{
		string testFile = Path.Combine(CreateTestDir(), "test.ply");
		var originalPoints = new List<PlyPoint>
		{
			new() { X = 1.5f, Y = 2.5f, Z = 3.5f, Nx = 0.15f, Ny = 0.25f, Nz = 0.35f, R = 15, G = 25, B = 35 },
		};

		Ply.WriteTextPlyPoints(testFile, originalPoints, false, false);
		List<PlyPoint> loadedPoints = Ply.ReadPly(testFile);

		await Assert.That(loadedPoints.Count).IsEqualTo(1);
		await Assert.That(loadedPoints[0].X).IsEqualTo(1.5f);
		await Assert.That(loadedPoints[0].Y).IsEqualTo(2.5f);
		await Assert.That(loadedPoints[0].Z).IsEqualTo(3.5f);

		// Normals and colors should be default (0)
		await ExpectNormalsAndColorsZero(loadedPoints[0]);
	}

	private static async Task ExpectNormalsAndColorsZero(PlyPoint point)
	{
		await Assert.That(point.Nx).IsEqualTo(0.0f);
		await Assert.That(point.Ny).IsEqualTo(0.0f);
		await Assert.That(point.Nz).IsEqualTo(0.0f);
		await Assert.That(point.R).IsEqualTo((byte)0);
		await Assert.That(point.G).IsEqualTo((byte)0);
		await Assert.That(point.B).IsEqualTo((byte)0);
	}

	[Test]
	public async Task Ply_RoundTripBinaryPlyPointsFullData()
	{
		string testFile = Path.Combine(CreateTestDir(), "test.ply");

		// Create test points
		var originalPoints = new List<PlyPoint>();
		for (int i = 0; i < 5; ++i)
		{
			originalPoints.Add(new PlyPoint
			{
				X = i * 1.5f,
				Y = i * 2.5f,
				Z = i * 3.5f,
				Nx = i * 0.15f,
				Ny = i * 0.25f,
				Nz = i * 0.35f,
				R = (byte)(i * 15),
				G = (byte)(i * 25),
				B = (byte)(i * 35),
			});
		}

		Ply.WriteBinaryPlyPoints(testFile, originalPoints, true, true);
		List<PlyPoint> loadedPoints = Ply.ReadPly(testFile);
		await ExpectPointsEqual(loadedPoints, originalPoints);
	}

	[Test]
	public async Task Ply_RoundTripBinaryPlyPointsXYZOnly()
	{
		string testFile = Path.Combine(CreateTestDir(), "test.ply");
		var originalPoints = new List<PlyPoint>
		{
			new() { X = 10.5f, Y = 20.5f, Z = 30.5f, Nx = 0.15f, Ny = 0.25f, Nz = 0.35f, R = 15, G = 25, B = 35 },
		};

		Ply.WriteBinaryPlyPoints(testFile, originalPoints, false, false);
		List<PlyPoint> loadedPoints = Ply.ReadPly(testFile);

		await Assert.That(loadedPoints.Count).IsEqualTo(1);
		await Assert.That(loadedPoints[0].X).IsEqualTo(10.5f);
		await Assert.That(loadedPoints[0].Y).IsEqualTo(20.5f);
		await Assert.That(loadedPoints[0].Z).IsEqualTo(30.5f);

		// Normals and colors should be default (0)
		await ExpectNormalsAndColorsZero(loadedPoints[0]);
	}

	private static PlyMesh CreateQuadMesh()
	{
		var mesh = new PlyMesh();
		mesh.Vertices.Add(new PlyMeshVertex(0.0f, 0.0f, 0.0f));
		mesh.Vertices.Add(new PlyMeshVertex(1.0f, 0.0f, 0.0f));
		mesh.Vertices.Add(new PlyMeshVertex(0.0f, 1.0f, 0.0f));
		mesh.Vertices.Add(new PlyMeshVertex(1.0f, 1.0f, 1.0f));
		mesh.Faces.Add(new PlyMeshFace(0, 1, 2));
		mesh.Faces.Add(new PlyMeshFace(1, 3, 2));
		return mesh;
	}

	private static async Task ExpectPlainMeshEqual(PlyMesh loadedMesh, PlyMesh originalMesh)
	{
		await Assert.That(loadedMesh.Vertices.Count).IsEqualTo(originalMesh.Vertices.Count);
		await Assert.That(loadedMesh.Faces.Count).IsEqualTo(originalMesh.Faces.Count);

		for (int i = 0; i < originalMesh.Vertices.Count; ++i)
		{
			await Assert.That(loadedMesh.Vertices[i].X).IsEqualTo(originalMesh.Vertices[i].X);
			await Assert.That(loadedMesh.Vertices[i].Y).IsEqualTo(originalMesh.Vertices[i].Y);
			await Assert.That(loadedMesh.Vertices[i].Z).IsEqualTo(originalMesh.Vertices[i].Z);

			// No colors in file, should get defaults.
			await Assert.That(loadedMesh.Vertices[i].R).IsEqualTo((byte)200);
			await Assert.That(loadedMesh.Vertices[i].G).IsEqualTo((byte)200);
			await Assert.That(loadedMesh.Vertices[i].B).IsEqualTo((byte)200);
		}

		await ExpectFacesEqual(loadedMesh, originalMesh);
	}

	private static async Task ExpectFacesEqual(PlyMesh loadedMesh, PlyMesh originalMesh)
	{
		for (int i = 0; i < originalMesh.Faces.Count; ++i)
		{
			await Assert.That(loadedMesh.Faces[i].VertexIdx1).IsEqualTo(originalMesh.Faces[i].VertexIdx1);
			await Assert.That(loadedMesh.Faces[i].VertexIdx2).IsEqualTo(originalMesh.Faces[i].VertexIdx2);
			await Assert.That(loadedMesh.Faces[i].VertexIdx3).IsEqualTo(originalMesh.Faces[i].VertexIdx3);
		}
	}

	[Test]
	public async Task Ply_RoundTripTextPlyMesh()
	{
		string testFile = Path.Combine(CreateTestDir(), "mesh.ply");
		PlyMesh originalMesh = CreateQuadMesh();

		Ply.WriteTextPlyMesh(testFile, new PlyTexturedMesh(originalMesh));
		PlyMesh loadedMesh = Ply.ReadPlyMesh(testFile).Mesh;
		await ExpectPlainMeshEqual(loadedMesh, originalMesh);
	}

	[Test]
	public async Task Ply_RoundTripBinaryPlyMesh()
	{
		string testFile = Path.Combine(CreateTestDir(), "mesh.ply");
		PlyMesh originalMesh = CreateQuadMesh();

		Ply.WriteBinaryPlyMesh(testFile, new PlyTexturedMesh(originalMesh));
		PlyMesh loadedMesh = Ply.ReadPlyMesh(testFile).Mesh;
		await ExpectPlainMeshEqual(loadedMesh, originalMesh);
	}

	private static PlyTexturedMesh CreateTestTexturedMesh()
	{
		var texturedMesh = new PlyTexturedMesh(CreateQuadMesh()) { TextureFile = "texture.png" };

		// 6 UV floats per face (u1,v1, u2,v2, u3,v3)
		// Face 0
		texturedMesh.FaceUvs.AddRange([0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f]);

		// Face 1
		texturedMesh.FaceUvs.AddRange([0.5f, 0.5f, 1.0f, 1.0f, 0.25f, 0.75f]);
		return texturedMesh;
	}

	private static async Task VerifyTexturedMesh(PlyTexturedMesh loaded, PlyTexturedMesh original)
	{
		await Assert.That(loaded.TextureFile).IsEqualTo(original.TextureFile);

		await Assert.That(loaded.Mesh.Vertices.Count).IsEqualTo(original.Mesh.Vertices.Count);
		await Assert.That(loaded.Mesh.Faces.Count).IsEqualTo(original.Mesh.Faces.Count);
		await Assert.That(loaded.FaceUvs.Count).IsEqualTo(original.FaceUvs.Count);

		for (int i = 0; i < original.Mesh.Vertices.Count; ++i)
		{
			await Assert.That(loaded.Mesh.Vertices[i].X).IsEqualTo(original.Mesh.Vertices[i].X);
			await Assert.That(loaded.Mesh.Vertices[i].Y).IsEqualTo(original.Mesh.Vertices[i].Y);
			await Assert.That(loaded.Mesh.Vertices[i].Z).IsEqualTo(original.Mesh.Vertices[i].Z);
		}

		await ExpectFacesEqual(loaded.Mesh, original.Mesh);

		for (int i = 0; i < original.FaceUvs.Count; ++i)
		{
			await Assert.That(loaded.FaceUvs[i]).IsEqualTo(original.FaceUvs[i]);
		}
	}

	[Test]
	public async Task Ply_RoundTripTextTexturedPlyMesh()
	{
		string testFile = Path.Combine(CreateTestDir(), "textured_mesh.ply");
		PlyTexturedMesh original = CreateTestTexturedMesh();
		Ply.WriteTextPlyMesh(testFile, original);
		PlyTexturedMesh loaded = Ply.ReadPlyMesh(testFile);
		await VerifyTexturedMesh(loaded, original);
	}

	[Test]
	public async Task Ply_RoundTripBinaryTexturedPlyMesh()
	{
		string testFile = Path.Combine(CreateTestDir(), "textured_mesh.ply");
		PlyTexturedMesh original = CreateTestTexturedMesh();
		Ply.WriteBinaryPlyMesh(testFile, original);
		PlyTexturedMesh loaded = Ply.ReadPlyMesh(testFile);
		await VerifyTexturedMesh(loaded, original);
	}

	[Test]
	public async Task Ply_ReadPlyMeshWithoutTexcoords()
	{
		string testFile = Path.Combine(CreateTestDir(), "mesh.ply");

		// Write a plain mesh (no texcoords) and read it with ReadPlyMesh.
		var plainMesh = new PlyMesh();
		plainMesh.Vertices.Add(new PlyMeshVertex(0.0f, 0.0f, 0.0f));
		plainMesh.Vertices.Add(new PlyMeshVertex(1.0f, 0.0f, 0.0f));
		plainMesh.Vertices.Add(new PlyMeshVertex(0.0f, 1.0f, 0.0f));
		plainMesh.Faces.Add(new PlyMeshFace(0, 1, 2));

		Ply.WriteTextPlyMesh(testFile, new PlyTexturedMesh(plainMesh));
		PlyTexturedMesh loaded = Ply.ReadPlyMesh(testFile);

		await Assert.That(loaded.Mesh.Vertices.Count).IsEqualTo(3);
		await Assert.That(loaded.Mesh.Faces.Count).IsEqualTo(1);
		await Assert.That(loaded.FaceUvs.Count == 0).IsTrue();
		await Assert.That(loaded.TextureFile.Length == 0).IsTrue();
	}

	[Test]
	public async Task Ply_HasPlyMeshFacesWithMesh()
	{
		string testFile = Path.Combine(CreateTestDir(), "mesh.ply");

		var mesh = new PlyMesh();
		mesh.Vertices.Add(new PlyMeshVertex(0.0f, 0.0f, 0.0f));
		mesh.Vertices.Add(new PlyMeshVertex(1.0f, 0.0f, 0.0f));
		mesh.Vertices.Add(new PlyMeshVertex(0.0f, 1.0f, 0.0f));
		mesh.Faces.Add(new PlyMeshFace(0, 1, 2));

		Ply.WriteTextPlyMesh(testFile, new PlyTexturedMesh(mesh));
		await Assert.That(Ply.HasPlyMeshFaces(testFile)).IsTrue();
	}

	[Test]
	public async Task Ply_HasPlyMeshFacesWithPointCloud()
	{
		string testFile = Path.Combine(CreateTestDir(), "points.ply");
		var points = new List<PlyPoint> { new() };

		Ply.WriteTextPlyPoints(testFile, points, false, false);
		await Assert.That(Ply.HasPlyMeshFaces(testFile)).IsFalse();
	}

	[Test]
	public async Task Ply_HasPlyMeshFacesWithZeroFaces()
	{
		string testFile = Path.Combine(CreateTestDir(), "mesh_no_faces.ply");
		Ply.WriteTextPlyMesh(testFile, new PlyTexturedMesh(new PlyMesh()));
		await Assert.That(Ply.HasPlyMeshFaces(testFile)).IsFalse();
	}
}
