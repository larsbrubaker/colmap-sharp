// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Ply.Write: the PLY writers of colmap/util/ply.cc - WriteTextPlyPoints,
// WriteBinaryPlyPoints, WriteTextPlyMesh and WriteBinaryPlyMesh. The readers are in Ply.cs
// and Ply.Mesh.cs. Tests: ColmapSharp.Tests/Util/PlyTests.cs.
//
// Tier A (exact): the files are byte-identical to COLMAP's. A float goes through
// `ostream << float` in the classic locale, which is printf's %g with 6 significant digits
// of the float widened to double (Util/CppStreamFormat.cs); binary values are little-endian.
// COLMAP writes a binary file's header with a text stream, closes it and appends the body;
// writing both through one stream gives the same bytes.

using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace ColmapSharp.Util;

public static partial class Ply
{
	private static readonly UTF8Encoding Utf8NoBom = new(false);

	/// <summary>Writes a PLY point cloud as text.</summary>
	public static void WriteTextPlyPoints(string path, IReadOnlyList<PlyPoint> points, bool writeNormal = true, bool writeRgb = true)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		WriteTextPlyPoints(file, points, writeNormal, writeRgb);
	}

	/// <summary>Writes a PLY point cloud as text to a stream.</summary>
	public static void WriteTextPlyPoints(Stream stream, IReadOnlyList<PlyPoint> points, bool writeNormal = true, bool writeRgb = true)
	{
		using var file = new StreamWriter(stream, Utf8NoBom, 1 << 16, leaveOpen: true);
		file.Write(PointsHeader("ascii", points.Count, writeNormal, writeRgb));

		var line = new StringBuilder();
		foreach (PlyPoint point in points)
		{
			line.Clear();
			line.Append(F(point.X)).Append(' ').Append(F(point.Y)).Append(' ').Append(F(point.Z));

			if (writeNormal)
			{
				line.Append(' ').Append(F(point.Nx)).Append(' ').Append(F(point.Ny)).Append(' ').Append(F(point.Nz));
			}

			if (writeRgb)
			{
				line.Append(' ').Append(I(point.R)).Append(' ').Append(I(point.G)).Append(' ').Append(I(point.B));
			}

			line.Append('\n');
			file.Write(line);
		}
	}

	/// <summary>Writes a PLY point cloud as binary little-endian.</summary>
	public static void WriteBinaryPlyPoints(string path, IReadOnlyList<PlyPoint> points, bool writeNormal = true, bool writeRgb = true)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		WriteBinaryPlyPoints(file, points, writeNormal, writeRgb);
	}

	/// <summary>Writes a PLY point cloud as binary little-endian to a stream.</summary>
	public static void WriteBinaryPlyPoints(Stream stream, IReadOnlyList<PlyPoint> points, bool writeNormal = true, bool writeRgb = true)
	{
		var file = new BufferedStream(stream, 1 << 16);
		file.Write(Utf8NoBom.GetBytes(PointsHeader("binary_little_endian", points.Count, writeNormal, writeRgb)));

		Span<byte> record = stackalloc byte[27];
		foreach (PlyPoint point in points)
		{
			int n = 0;
			n = PutFloat(record, n, point.X);
			n = PutFloat(record, n, point.Y);
			n = PutFloat(record, n, point.Z);

			if (writeNormal)
			{
				n = PutFloat(record, n, point.Nx);
				n = PutFloat(record, n, point.Ny);
				n = PutFloat(record, n, point.Nz);
			}

			if (writeRgb)
			{
				record[n++] = point.R;
				record[n++] = point.G;
				record[n++] = point.B;
			}

			file.Write(record[..n]);
		}

		file.Flush();
	}

	/// <summary>
	/// Writes a PLY mesh as text, with the texture coordinates and TextureFile comment when
	/// the mesh has them.
	/// </summary>
	public static void WriteTextPlyMesh(string path, PlyTexturedMesh mesh)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		WriteTextPlyMesh(file, mesh);
	}

	/// <summary>Writes a PLY mesh as text to a stream.</summary>
	public static void WriteTextPlyMesh(Stream stream, PlyTexturedMesh mesh)
	{
		bool hasTexcoords = CheckTexcoords(mesh);

		using var file = new StreamWriter(stream, Utf8NoBom, 1 << 16, leaveOpen: true);
		file.Write(MeshHeader("ascii", mesh, hasTexcoords));

		var line = new StringBuilder();
		foreach (PlyMeshVertex vertex in mesh.Mesh.Vertices)
		{
			line.Clear();
			line.Append(F(vertex.X)).Append(' ').Append(F(vertex.Y)).Append(' ').Append(F(vertex.Z)).Append('\n');
			file.Write(line);
		}

		for (int i = 0; i < mesh.Mesh.Faces.Count; ++i)
		{
			PlyMeshFace face = mesh.Mesh.Faces[i];
			line.Clear();
			line.Append("3 ").Append(I(face.VertexIdx1)).Append(' ').Append(I(face.VertexIdx2)).Append(' ').Append(I(face.VertexIdx3));
			if (hasTexcoords)
			{
				line.Append(" 6");
				for (int j = 0; j < 6; ++j)
				{
					line.Append(' ').Append(F(mesh.FaceUvs[(i * 6) + j]));
				}
			}

			line.Append('\n');
			file.Write(line);
		}
	}

	/// <summary>Writes a PLY mesh as binary little-endian.</summary>
	public static void WriteBinaryPlyMesh(string path, PlyTexturedMesh mesh)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		WriteBinaryPlyMesh(file, mesh);
	}

	/// <summary>Writes a PLY mesh as binary little-endian to a stream.</summary>
	public static void WriteBinaryPlyMesh(Stream stream, PlyTexturedMesh mesh)
	{
		bool hasTexcoords = CheckTexcoords(mesh);

		var file = new BufferedStream(stream, 1 << 16);
		file.Write(Utf8NoBom.GetBytes(MeshHeader("binary_little_endian", mesh, hasTexcoords)));

		Span<byte> record = stackalloc byte[12];
		foreach (PlyMeshVertex vertex in mesh.Mesh.Vertices)
		{
			PutFloat(record, 0, vertex.X);
			PutFloat(record, 4, vertex.Y);
			PutFloat(record, 8, vertex.Z);
			file.Write(record);
		}

		int numVertices = mesh.Mesh.Vertices.Count;
		for (int i = 0; i < mesh.Mesh.Faces.Count; ++i)
		{
			PlyMeshFace face = mesh.Mesh.Faces[i];

			// C++ compares size_t indices, so a negative index fails too.
			Check.Lt((uint)face.VertexIdx1, (uint)numVertices);
			Check.Lt((uint)face.VertexIdx2, (uint)numVertices);
			Check.Lt((uint)face.VertexIdx3, (uint)numVertices);
			file.WriteByte(3);
			BinaryPrimitives.WriteInt32LittleEndian(record, face.VertexIdx1);
			BinaryPrimitives.WriteInt32LittleEndian(record[4..], face.VertexIdx2);
			BinaryPrimitives.WriteInt32LittleEndian(record[8..], face.VertexIdx3);
			file.Write(record);

			if (hasTexcoords)
			{
				file.WriteByte(6);
				for (int j = 0; j < 6; ++j)
				{
					PutFloat(record, 0, mesh.FaceUvs[(i * 6) + j]);
					file.Write(record[..4]);
				}
			}
		}

		file.Flush();
	}

	private static string PointsHeader(string format, int numPoints, bool writeNormal, bool writeRgb)
	{
		var header = new StringBuilder();
		header.Append("ply\n");
		header.Append("format ").Append(format).Append(" 1.0\n");
		header.Append("element vertex ").Append(I(numPoints)).Append('\n');
		header.Append("property float x\n");
		header.Append("property float y\n");
		header.Append("property float z\n");

		if (writeNormal)
		{
			header.Append("property float nx\n");
			header.Append("property float ny\n");
			header.Append("property float nz\n");
		}

		if (writeRgb)
		{
			header.Append("property uchar red\n");
			header.Append("property uchar green\n");
			header.Append("property uchar blue\n");
		}

		header.Append("end_header\n");
		return header.ToString();
	}

	private static string MeshHeader(string format, PlyTexturedMesh mesh, bool hasTexcoords)
	{
		var header = new StringBuilder();
		header.Append("ply\n");
		header.Append("format ").Append(format).Append(" 1.0\n");
		if (mesh.TextureFile.Length > 0)
		{
			header.Append("comment TextureFile ").Append(mesh.TextureFile).Append('\n');
		}

		header.Append("element vertex ").Append(I(mesh.Mesh.Vertices.Count)).Append('\n');
		header.Append("property float x\n");
		header.Append("property float y\n");
		header.Append("property float z\n");
		header.Append("element face ").Append(I(mesh.Mesh.Faces.Count)).Append('\n');
		if (hasTexcoords)
		{
			header.Append("property list uchar int vertex_indices\n");
			header.Append("property list uchar float texcoord\n");
		}
		else
		{
			header.Append("property list uchar int vertex_index\n");
		}

		header.Append("end_header\n");
		return header.ToString();
	}

	private static bool CheckTexcoords(PlyTexturedMesh mesh)
	{
		bool hasTexcoords = mesh.FaceUvs.Count > 0;
		if (hasTexcoords)
		{
			Check.Eq((long)mesh.FaceUvs.Count, (long)mesh.Mesh.Faces.Count * 6, "Expected 6 UV coordinates per face");
		}

		return hasTexcoords;
	}

	private static int PutFloat(Span<byte> record, int offset, float value)
	{
		BinaryPrimitives.WriteSingleLittleEndian(record[offset..], value);
		return offset + 4;
	}

	// `ostream << float` with the default precision.
	private static string F(float value) => CppStreamFormat.FormatDouble(value);

	private static string I(long value) => value.ToString(CultureInfo.InvariantCulture);
}
