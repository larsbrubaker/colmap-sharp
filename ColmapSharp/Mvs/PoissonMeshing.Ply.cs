// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonMeshing.Ply: the PLY side of PoissonMeshing's file wrapper. Reading: PoissonRecon.cpp's
// RunPoissonRecon requires x, y, z and nx, ny, nz on the input's vertex element ("Ply file does
// not contain normals") and carries its other properties through as auxiliary data; COLMAP's
// fused.ply has exactly red, green and blue as uchar, and those become the mesh colors. The
// values are read with COLMAP's own Ply.ReadPly. Other extra properties are not carried
// (divergence 130). Writing: PoissonRecon.cpp's WriteMesh (PLY::Write) and
// SurfaceTrimmer.cpp's PLY::WritePolygons write the same layout for COLMAP's arguments -
// binary_little_endian, no comments, vertex x, y, z (float), then "value" (float) when the
// density is output, then red, green, blue (uchar, the input's type); face
// "property list int int vertex_indices". WriteMeshPly writes that layout from a
// PoissonMeshOutput, whose colors are already PlyUChar-converted.

using System.Buffers.Binary;
using System.Text;

using ColmapSharp.Mvs.PoissonRecon;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

public static partial class PoissonMeshing
{
	/// <summary>
	/// Writes <paramref name="mesh"/> as PoissonRecon writes its output PLY for COLMAP's
	/// arguments (see the file header).
	/// </summary>
	public static void WriteMeshPly(string path, PoissonMeshOutput mesh)
	{
		ArgumentNullException.ThrowIfNull(mesh);
		using FileStream file = FileOpen.OpenWrite(path);
		WriteMeshPly(file, mesh);
	}

	/// <summary>Writes <paramref name="mesh"/> in PoissonRecon's output PLY layout to a stream.</summary>
	public static void WriteMeshPly(Stream stream, PoissonMeshOutput mesh)
	{
		ArgumentNullException.ThrowIfNull(mesh);
		if (mesh.ColorChannels is not (0 or 3))
		{
			throw new ArgumentException($"The mesh has {mesh.ColorChannels} color channels; PLY output supports red, green, blue or none.", nameof(mesh));
		}

		var header = new StringBuilder();
		header.Append("ply\nformat binary_little_endian 1.0\n");
		header.Append("element vertex ").Append(mesh.VertexCount).Append('\n');
		header.Append("property float x\nproperty float y\nproperty float z\n");
		if (mesh.Values != null)
		{
			header.Append("property float value\n");
		}

		if (mesh.Colors != null)
		{
			header.Append("property uchar red\nproperty uchar green\nproperty uchar blue\n");
		}

		int triangleCount = mesh.Triangles.Length / 3;
		header.Append("element face ").Append(triangleCount).Append('\n');
		header.Append("property list int int vertex_indices\nend_header\n");

		var output = new BufferedStream(stream, 1 << 16);
		output.Write(Encoding.ASCII.GetBytes(header.ToString()));
		Span<byte> record = stackalloc byte[16];
		for (int i = 0; i < mesh.VertexCount; i++)
		{
			int n = 0;
			for (int d = 0; d < 3; d++)
			{
				BinaryPrimitives.WriteSingleLittleEndian(record[n..], mesh.Positions[(3 * i) + d]);
				n += 4;
			}

			if (mesh.Values != null)
			{
				BinaryPrimitives.WriteSingleLittleEndian(record[n..], mesh.Values[i]);
				n += 4;
			}

			output.Write(record[..n]);
			if (mesh.Colors != null)
			{
				output.Write(mesh.Colors.AsSpan(3 * i, 3));
			}
		}

		for (int t = 0; t < triangleCount; t++)
		{
			BinaryPrimitives.WriteInt32LittleEndian(record, 3);
			for (int k = 0; k < 3; k++)
			{
				BinaryPrimitives.WriteInt32LittleEndian(record[(4 + (4 * k))..], mesh.Triangles[(3 * t) + k]);
			}

			output.Write(record);
		}

		output.Flush();
	}

	// HasColors comes from the header, so it holds for a PLY without points too.
	private readonly record struct PoissonInputPoints(float[] Positions, float[] Normals, byte[] Colors, bool HasColors);

	// RunPoissonRecon's input: the positions and normals (required), and the colors when the
	// vertex element has red, green and blue uchar properties.
	private static PoissonInputPoints ReadInputPoints(string path)
	{
		HashSet<string> properties = ReadVertexProperties(path);
		bool HasReal(string name) =>
			properties.Contains("float " + name) || properties.Contains("float32 " + name)
			|| properties.Contains("double " + name) || properties.Contains("float64 " + name);
		if (!HasReal("x") || !HasReal("y") || !HasReal("z"))
		{
			throw new InvalidDataException("Ply file does not contain positions");
		}

		if (!HasReal("nx") || !HasReal("ny") || !HasReal("nz"))
		{
			throw new InvalidDataException("Ply file does not contain normals");
		}

		bool hasColors = properties.Contains("uchar red") && properties.Contains("uchar green") && properties.Contains("uchar blue");
		List<PlyPoint> points = Ply.ReadPly(path);
		int n = points.Count;
		float[] positions = new float[3 * n], normals = new float[3 * n];
		byte[] colors = hasColors ? new byte[3 * n] : [];
		for (int i = 0; i < n; i++)
		{
			PlyPoint p = points[i];
			positions[3 * i] = p.X;
			positions[(3 * i) + 1] = p.Y;
			positions[(3 * i) + 2] = p.Z;
			normals[3 * i] = p.Nx;
			normals[(3 * i) + 1] = p.Ny;
			normals[(3 * i) + 2] = p.Nz;
			if (hasColors)
			{
				colors[3 * i] = p.R;
				colors[(3 * i) + 1] = p.G;
				colors[(3 * i) + 2] = p.B;
			}
		}

		return new PoissonInputPoints(positions, normals, colors, hasColors);
	}

	// The "type name" of every scalar property of the vertex element (x, y, z as "float x" etc.).
	private static HashSet<string> ReadVertexProperties(string path)
	{
		using FileStream file = FileOpen.OpenRead(path);
		var reader = new PlyByteReader(file);
		var properties = new HashSet<string>(StringComparer.Ordinal);
		bool inVertex = false;
		string? line;
		while ((line = reader.ReadLine()) != null)
		{
			string[] tokens = line.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
			if (tokens.Length == 0)
			{
				continue;
			}

			if (tokens[0] == "end_header")
			{
				break;
			}

			if (tokens[0] == "element")
			{
				inVertex = tokens.Length >= 2 && tokens[1] == "vertex";
			}
			else if (inVertex && tokens[0] == "property" && tokens.Length == 3)
			{
				properties.Add(tokens[1] + " " + tokens[2]);
			}
		}

		return properties;
	}
}
