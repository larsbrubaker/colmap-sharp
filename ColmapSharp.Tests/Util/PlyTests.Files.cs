// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PlyTests.Files: the ply_test.cc cases that read hand-written PLY files (vertex colors and
// extra vertex properties), then C#-only cases (labeled CSharpOnly_) that pin the exact
// bytes the writers produce and that malformed files fail cleanly instead of allocating
// what the header declares.

using System.Text;

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

public partial class PlyTests
{
	private const string ColorMeshHeaderBody =
		"element vertex 3\n" +
		"property float x\n" +
		"property float y\n" +
		"property float z\n";

	private const string ColorProperties =
		"property uchar red\n" +
		"property uchar green\n" +
		"property uchar blue\n";

	private const string NormalProperties =
		"property float nx\n" +
		"property float ny\n" +
		"property float nz\n";

	private const string FaceHeader =
		"element face 1\n" +
		"property list uchar int vertex_index\n" +
		"end_header\n";

	private static void WriteBinaryFile(string path, string header, params object[] values)
	{
		using var stream = new FileStream(path, FileMode.Create);
		using var writer = new BinaryWriter(stream);
		writer.Write(Encoding.ASCII.GetBytes(header));
		foreach (object value in values)
		{
			switch (value)
			{
				case float f:
					writer.Write(f);
					break;
				case byte b:
					writer.Write(b);
					break;
				case int i:
					writer.Write(i);
					break;
			}
		}
	}

	private static async Task ExpectColor(PlyMeshVertex vertex, byte r, byte g, byte b)
	{
		await Assert.That(vertex.R).IsEqualTo(r);
		await Assert.That(vertex.G).IsEqualTo(g);
		await Assert.That(vertex.B).IsEqualTo(b);
	}

	[Test]
	public async Task Ply_ReadTextPlyMeshWithVertexColors()
	{
		string testFile = Path.Combine(CreateTestDir(), "mesh_colors.ply");

		// Write a PLY mesh with per-vertex colors manually.
		File.WriteAllText(
			testFile,
			"ply\nformat ascii 1.0\n" + ColorMeshHeaderBody + ColorProperties + FaceHeader +
			"0 0 0 255 0 0\n" +
			"1 0 0 0 255 0\n" +
			"0 1 0 0 0 255\n" +
			"3 0 1 2\n");

		PlyMesh mesh = Ply.ReadPlyMesh(testFile).Mesh;

		await Assert.That(mesh.Vertices.Count).IsEqualTo(3);
		await Assert.That(mesh.Faces.Count).IsEqualTo(1);

		await ExpectColor(mesh.Vertices[0], 255, 0, 0);
		await ExpectColor(mesh.Vertices[1], 0, 255, 0);
		await ExpectColor(mesh.Vertices[2], 0, 0, 255);

		await Assert.That(mesh.Faces[0].VertexIdx1).IsEqualTo(0);
		await Assert.That(mesh.Faces[0].VertexIdx2).IsEqualTo(1);
		await Assert.That(mesh.Faces[0].VertexIdx3).IsEqualTo(2);
	}

	[Test]
	public async Task Ply_ReadBinaryPlyMeshWithVertexColors()
	{
		string testFile = Path.Combine(CreateTestDir(), "mesh_colors_bin.ply");

		// Write a binary PLY mesh with per-vertex colors manually.
		WriteBinaryFile(
			testFile,
			"ply\nformat binary_little_endian 1.0\n" + ColorMeshHeaderBody + ColorProperties + FaceHeader,
			1.0f, 2.0f, 3.0f, (byte)100, (byte)150, (byte)200,
			4.0f, 5.0f, 6.0f, (byte)10, (byte)20, (byte)30,
			7.0f, 8.0f, 9.0f, (byte)50, (byte)60, (byte)70,
			(byte)3, 0, 1, 2);

		PlyMesh mesh = Ply.ReadPlyMesh(testFile).Mesh;

		await Assert.That(mesh.Vertices.Count).IsEqualTo(3);
		await Assert.That(mesh.Faces.Count).IsEqualTo(1);

		await Assert.That(mesh.Vertices[0].X).IsEqualTo(1.0f);
		await Assert.That(mesh.Vertices[0].Y).IsEqualTo(2.0f);
		await Assert.That(mesh.Vertices[0].Z).IsEqualTo(3.0f);
		await ExpectColor(mesh.Vertices[0], 100, 150, 200);

		await Assert.That(mesh.Vertices[1].X).IsEqualTo(4.0f);
		await ExpectColor(mesh.Vertices[1], 10, 20, 30);

		await Assert.That(mesh.Vertices[2].X).IsEqualTo(7.0f);
		await ExpectColor(mesh.Vertices[2], 50, 60, 70);
	}

	private static async Task ExpectExtraPropertiesMesh(PlyMesh mesh)
	{
		await Assert.That(mesh.Vertices.Count).IsEqualTo(3);

		// Positions should be read correctly.
		await Assert.That(mesh.Vertices[0].X).IsEqualTo(1.0f);
		await Assert.That(mesh.Vertices[0].Y).IsEqualTo(2.0f);
		await Assert.That(mesh.Vertices[0].Z).IsEqualTo(3.0f);
		await Assert.That(mesh.Vertices[1].X).IsEqualTo(4.0f);
		await Assert.That(mesh.Vertices[2].X).IsEqualTo(7.0f);

		// Colors should be read correctly despite extra normal properties.
		await ExpectColor(mesh.Vertices[0], 255, 128, 64);
		await ExpectColor(mesh.Vertices[1], 32, 64, 96);
		await ExpectColor(mesh.Vertices[2], 10, 20, 30);
	}

	[Test]
	public async Task Ply_ReadTextPlyMeshWithExtraProperties()
	{
		string testFile = Path.Combine(CreateTestDir(), "mesh_extra.ply");

		// Write a PLY mesh with normals and colors (normals should be skipped).
		File.WriteAllText(
			testFile,
			"ply\nformat ascii 1.0\n" + ColorMeshHeaderBody + NormalProperties + ColorProperties + FaceHeader +
			"1 2 3 0.1 0.2 0.3 255 128 64\n" +
			"4 5 6 0.4 0.5 0.6 32 64 96\n" +
			"7 8 9 0.7 0.8 0.9 10 20 30\n" +
			"3 0 1 2\n");

		await ExpectExtraPropertiesMesh(Ply.ReadPlyMesh(testFile).Mesh);
	}

	[Test]
	public async Task Ply_ReadBinaryPlyMeshWithExtraProperties()
	{
		string testFile = Path.Combine(CreateTestDir(), "mesh_extra_bin.ply");

		// Write a binary PLY mesh with normals + colors.
		WriteBinaryFile(
			testFile,
			"ply\nformat binary_little_endian 1.0\n" + ColorMeshHeaderBody + NormalProperties + ColorProperties + FaceHeader,
			1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f, (byte)255, (byte)128, (byte)64,
			4.0f, 5.0f, 6.0f, 0.4f, 0.5f, 0.6f, (byte)32, (byte)64, (byte)96,
			7.0f, 8.0f, 9.0f, 0.7f, 0.8f, 0.9f, (byte)10, (byte)20, (byte)30,
			(byte)3, 0, 1, 2);

		await ExpectExtraPropertiesMesh(Ply.ReadPlyMesh(testFile).Mesh);
	}

	[Test]
	public async Task CSharpOnly_TextWritersAreByteExact()
	{
		// COLMAP streams floats with the default precision 6 (printf %g).
		var points = new List<PlyPoint>
		{
			new() { X = 0.1f, Y = -2.5f, Z = 1234567.0f, Nx = 1e-5f, Ny = 0, Nz = 1, R = 1, G = 22, B = 255 },
		};
		var stream = new MemoryStream();
		Ply.WriteTextPlyPoints(stream, points);
		string expected =
			"ply\nformat ascii 1.0\nelement vertex 1\n" +
			"property float x\nproperty float y\nproperty float z\n" +
			"property float nx\nproperty float ny\nproperty float nz\n" +
			"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" +
			"0.1 -2.5 1.23457e+06 1e-05 0 1 1 22 255\n";
		await Assert.That(Encoding.UTF8.GetString(stream.ToArray())).IsEqualTo(expected);

		var mesh = new PlyTexturedMesh(CreateQuadMesh()) { TextureFile = "tex.png" };
		mesh.FaceUvs.AddRange([0, 0, 1, 0, 0, 1, 0.5f, 0.5f, 1, 1, 0.25f, 0.75f]);
		var meshStream = new MemoryStream();
		Ply.WriteTextPlyMesh(meshStream, mesh);
		string expectedMesh =
			"ply\nformat ascii 1.0\ncomment TextureFile tex.png\nelement vertex 4\n" +
			"property float x\nproperty float y\nproperty float z\nelement face 2\n" +
			"property list uchar int vertex_indices\nproperty list uchar float texcoord\nend_header\n" +
			"0 0 0\n1 0 0\n0 1 0\n1 1 1\n" +
			"3 0 1 2 6 0 0 1 0 0 1\n3 1 3 2 6 0.5 0.5 1 1 0.25 0.75\n";
		await Assert.That(Encoding.UTF8.GetString(meshStream.ToArray())).IsEqualTo(expectedMesh);
	}

	[Test]
	public async Task CSharpOnly_BinaryPointsLayout()
	{
		var points = new List<PlyPoint> { new() { X = 1, Y = 2, Z = 3, R = 4, G = 5, B = 6 } };
		var stream = new MemoryStream();
		Ply.WriteBinaryPlyPoints(stream, points, writeNormal: false, writeRgb: true);
		byte[] bytes = stream.ToArray();
		const string header =
			"ply\nformat binary_little_endian 1.0\nelement vertex 1\n" +
			"property float x\nproperty float y\nproperty float z\n" +
			"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
		await Assert.That(bytes.Length).IsEqualTo(header.Length + 15);
		await Assert.That(Encoding.ASCII.GetString(bytes, 0, header.Length)).IsEqualTo(header);
		await Assert.That(BitConverter.ToSingle(bytes, header.Length + 8)).IsEqualTo(3.0f);
		await Assert.That(bytes[^1]).IsEqualTo((byte)6);
	}

	[Test]
	public async Task CSharpOnly_HugeDeclaredCountFailsOnMissingData()
	{
		// 2^32 vertices are allowed by the header check, but the body is empty: the read
		// must fail at the first vertex rather than reserve 2^32 points.
		const string header = "ply\nformat binary_little_endian 1.0\nelement vertex 4294967296\n" +
			"property float x\nproperty float y\nproperty float z\nend_header\n";
		await Assert.That(() => Ply.ReadPly(new MemoryStream(Encoding.ASCII.GetBytes(header))))
			.Throws<ArgumentException>().WithMessageContaining("Unexpected end of PLY file at vertex 0");

		const string tooMany = "ply\nformat binary_little_endian 1.0\nelement vertex 4294967297\n" +
			"property float x\nproperty float y\nproperty float z\nend_header\n";
		await Assert.That(() => Ply.ReadPly(new MemoryStream(Encoding.ASCII.GetBytes(tooMany))))
			.Throws<ArgumentException>().WithMessageContaining("PLY file declares too many vertices");

		const string negative = "ply\nformat ascii 1.0\nelement vertex 1\n" +
			"property float x\nproperty float y\nproperty float z\nelement face -1\n" +
			"property list uchar int vertex_index\nend_header\n0 0 0\n";
		await Assert.That(() => Ply.ReadPlyMesh(new MemoryStream(Encoding.ASCII.GetBytes(negative))))
			.Throws<ArgumentException>().WithMessageContaining("PLY mesh declares too many faces");
	}

	[Test]
	public async Task CSharpOnly_MalformedMeshFacesFail()
	{
		const string header = "ply\nformat ascii 1.0\nelement vertex 1\n" +
			"property float x\nproperty float y\nproperty float z\nelement face 1\n" +
			"property list uchar int vertex_index\nend_header\n0 0 0\n";
		await Assert.That(() => Ply.ReadPlyMesh(new MemoryStream(Encoding.ASCII.GetBytes(header + "3 0 0 1\n"))))
			.Throws<ArgumentException>().WithMessageContaining("Face vertex index out of bounds at face 0");
		await Assert.That(() => Ply.ReadPlyMesh(new MemoryStream(Encoding.ASCII.GetBytes(header + "3 0 -1 0\n"))))
			.Throws<ArgumentException>().WithMessageContaining("Negative face vertex index at face 0");
		await Assert.That(() => Ply.ReadPlyMesh(new MemoryStream(Encoding.ASCII.GetBytes(header + "4 0 0 0 0\n"))))
			.Throws<ArgumentException>().WithMessageContaining("Only triangular faces are supported");
	}
}
