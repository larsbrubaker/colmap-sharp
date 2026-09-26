// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PlyTests.Formats: C#-only tests (no ply_test.cc case covers them) of ColmapSharp/Util/Ply*.cs
// on inputs COLMAP's writers never produce: big-endian binary files for the point and mesh
// readers, a 1M-point ASCII cloud read in bounded time (the buffered line reader), and ASCII
// texture coordinates parsed with libc++'s `>> float` rules (CppLineTokens.TryReadFloat).

using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.Text;

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

public partial class PlyTests
{
	private static void PutFloatBe(MemoryStream stream, float value)
	{
		Span<byte> bytes = stackalloc byte[4];
		BinaryPrimitives.WriteSingleBigEndian(bytes, value);
		stream.Write(bytes);
	}

	private static void PutDoubleBe(MemoryStream stream, double value)
	{
		Span<byte> bytes = stackalloc byte[8];
		BinaryPrimitives.WriteDoubleBigEndian(bytes, value);
		stream.Write(bytes);
	}

	[Test]
	public async Task CSharpOnly_ReadsBigEndianPoints()
	{
		var stream = new MemoryStream();
		stream.Write(Encoding.ASCII.GetBytes(
			"ply\nformat binary_big_endian 1.0\nelement vertex 2\n" +
			"property double x\nproperty float y\nproperty float z\n" +
			"property float nx\nproperty float ny\nproperty float nz\n" +
			"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"));
		for (int i = 1; i <= 2; i++)
		{
			PutDoubleBe(stream, 1.25 * i);
			PutFloatBe(stream, -2.5f * i);
			PutFloatBe(stream, 3.75f * i);
			PutFloatBe(stream, 0.5f);
			PutFloatBe(stream, 0.25f);
			PutFloatBe(stream, -1);
			stream.Write([(byte)(10 * i), (byte)(20 * i), (byte)(30 * i)]);
		}

		stream.Position = 0;
		List<PlyPoint> points = Ply.ReadPly(stream);

		await Assert.That(points.Count).IsEqualTo(2);
		await Assert.That(points[1].X).IsEqualTo(2.5f);
		await Assert.That(points[1].Y).IsEqualTo(-5.0f);
		await Assert.That(points[1].Z).IsEqualTo(7.5f);
		await Assert.That(points[0].Nx).IsEqualTo(0.5f);
		await Assert.That(points[0].Ny).IsEqualTo(0.25f);
		await Assert.That(points[0].Nz).IsEqualTo(-1.0f);
		await Assert.That(points[1].R).IsEqualTo((byte)20);
		await Assert.That(points[1].G).IsEqualTo((byte)40);
		await Assert.That(points[1].B).IsEqualTo((byte)60);
	}

	[Test]
	public async Task CSharpOnly_ReadsBigEndianMesh()
	{
		var stream = new MemoryStream();
		stream.Write(Encoding.ASCII.GetBytes(
			"ply\nformat binary_big_endian 1.0\ncomment TextureFile tex.png\nelement vertex 3\n" +
			"property float x\nproperty float y\nproperty float z\n" +
			"property uchar red\nproperty uchar green\nproperty uchar blue\nelement face 1\n" +
			"property list uchar ushort vertex_indices\nproperty list uchar float texcoord\nend_header\n"));
		for (int i = 0; i < 3; i++)
		{
			PutFloatBe(stream, i);
			PutFloatBe(stream, i + 0.5f);
			PutFloatBe(stream, -i);
			stream.Write([(byte)i, (byte)(i + 1), (byte)(i + 2)]);
		}

		stream.WriteByte(3);
		Span<byte> index = stackalloc byte[2];
		foreach (short idx in new short[] { 2, 0, 1 })
		{
			BinaryPrimitives.WriteInt16BigEndian(index, idx);
			stream.Write(index);
		}

		stream.WriteByte(6);
		foreach (float uv in new[] { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 0.125f })
		{
			PutFloatBe(stream, uv);
		}

		stream.Position = 0;
		PlyTexturedMesh mesh = Ply.ReadPlyMesh(stream);

		await Assert.That(mesh.TextureFile).IsEqualTo("tex.png");
		await Assert.That(mesh.Mesh.Vertices.Count).IsEqualTo(3);
		await Assert.That(mesh.Mesh.Vertices[2].X).IsEqualTo(2.0f);
		await Assert.That(mesh.Mesh.Vertices[2].Y).IsEqualTo(2.5f);
		await Assert.That(mesh.Mesh.Vertices[2].Z).IsEqualTo(-2.0f);
		await ExpectColor(mesh.Mesh.Vertices[1], 1, 2, 3);
		await Assert.That(mesh.Mesh.Faces.Count).IsEqualTo(1);
		await Assert.That(mesh.Mesh.Faces[0].VertexIdx1).IsEqualTo(2);
		await Assert.That(mesh.Mesh.Faces[0].VertexIdx2).IsEqualTo(0);
		await Assert.That(mesh.Mesh.Faces[0].VertexIdx3).IsEqualTo(1);
		await Assert.That(mesh.FaceUvs.SequenceEqual(new[] { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 0.125f })).IsTrue();
	}

	[Test]
	public async Task CSharpOnly_ReadsMillionPointTextCloudQuickly()
	{
		// The text is built here rather than with WriteTextPlyPoints, whose %g formatting would
		// dominate the test's time; these values print the same either way.
		const int kNumPoints = 1_000_000;
		var text = new StringBuilder(
			"ply\nformat ascii 1.0\nelement vertex 1000000\nproperty float x\nproperty float y\n" +
			"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n");
		for (int i = 0; i < kNumPoints; i++)
		{
			text.Append(CultureInfo.InvariantCulture, $"{i} {i * 0.5} {-i} {(byte)i} 1 2\n");
		}

		var stream = new MemoryStream(Encoding.ASCII.GetBytes(text.ToString()));

		var watch = Stopwatch.StartNew();
		List<PlyPoint> loaded = Ply.ReadPly(stream);
		watch.Stop();

		await Assert.That(loaded.Count).IsEqualTo(kNumPoints);
		await Assert.That(loaded[kNumPoints - 1].Z).IsEqualTo(-999999.0f);
		await Assert.That(loaded[300].R).IsEqualTo((byte)44);

		// Well under this on a laptop even in Debug; the bound only catches a pathological reader.
		await Assert.That(watch.Elapsed.TotalSeconds).IsLessThan(10.0);
	}

	[Test]
	public async Task CSharpOnly_TextTexcoordsUseStreamFloatRules()
	{
		const string header = "ply\nformat ascii 1.0\nelement vertex 3\n" +
			"property float x\nproperty float y\nproperty float z\nelement face 1\n" +
			"property list uchar int vertex_indices\nproperty list uchar float texcoord\nend_header\n" +
			"0 0 0\n1 0 0\n0 1 0\n";

		// Hexadecimal floats are read like strtof.
		PlyTexturedMesh mesh = Ply.ReadPlyMesh(new MemoryStream(Encoding.ASCII.GetBytes(header + "3 0 1 2 6 0x1p-2 0.1 1 0 0 1\n")));
		await Assert.That(mesh.FaceUvs[0]).IsEqualTo(0.25f);
		await Assert.That(mesh.FaceUvs[1]).IsEqualTo(0.1f);

		// Out of float's range (strtof's ERANGE) and "inf" fail like libc++'s failbit.
		foreach (string uv in new[] { "1e39", "inf", "1e-50" })
		{
			string file = header + string.Create(CultureInfo.InvariantCulture, $"3 0 1 2 6 {uv} 0 1 0 0 1\n");
			await Assert.That(() => Ply.ReadPlyMesh(new MemoryStream(Encoding.ASCII.GetBytes(file))))
				.Throws<ArgumentException>().WithMessageContaining("line_stream >> uv");
		}
	}
}
