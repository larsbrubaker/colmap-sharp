// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Ply.Mesh: ReadPlyMesh of colmap/util/ply.cc, which reads plain and textured triangle
// meshes (per-face "texcoord" lists and a "comment TextureFile" header line). The point
// cloud reader and the shared header helpers are in Ply.cs; the writers in Ply.Write.cs.
// Tests: ColmapSharp.Tests/Util/PlyTests.cs.

using System.Buffers.Binary;

namespace ColmapSharp.Util;

public static partial class Ply
{
	/// <summary>
	/// Reads a PLY mesh from a text or binary file. Supports both plain and textured meshes
	/// (with per-face UV coordinates and a "comment TextureFile" header).
	/// </summary>
	public static PlyTexturedMesh ReadPlyMesh(string path)
	{
		using FileStream file = FileOpen.OpenRead(path);
		return ReadPlyMesh(file);
	}

	/// <summary>Reads a PLY mesh from a text or binary stream.</summary>
	public static PlyTexturedMesh ReadPlyMesh(Stream stream)
	{
		var file = new PlyByteReader(stream);
		var result = new PlyTexturedMesh();

		bool isBinary = false;
		bool isLittleEndian = false;
		ulong numVertices = 0;
		ulong numFaces = 0;
		bool hasTexcoord = false;

		// Track vertex properties for proper parsing.
		bool inVertexSection = false;
		bool inFaceSection = false;
		string faceCountType = "uchar";
		string faceIndexType = "int";
		ulong numBytesPerVertex = 0;
		int numVertexProps = 0;

		// Vertex properties x, y, z, r, g, b: indices (for ASCII) and byte positions (binary).
		var props = new PointProperty[6];
		for (int i = 0; i < props.Length; i++)
		{
			props[i] = new PointProperty();
		}

		string? line;
		while ((line = file.ReadLine()) != null)
		{
			line = CppLineTokens.Trim(line);
			if (line.Length == 0)
			{
				continue;
			}

			if (line == "end_header")
			{
				break;
			}

			ParseFormat(line, ref isBinary, ref isLittleEndian);

			string[] lineElems = StringSplit(line);

			if (line.StartsWith("comment TextureFile", StringComparison.Ordinal) && lineElems.Length >= 3)
			{
				result.TextureFile = lineElems[2];
			}

			if (lineElems.Length >= 3 && lineElems[0] == "element")
			{
				inVertexSection = false;
				inFaceSection = false;
				if (lineElems[1] == "vertex")
				{
					numVertices = unchecked((ulong)Stoll(lineElems[2]));
					inVertexSection = true;
				}
				else if (lineElems[1] == "face")
				{
					numFaces = unchecked((ulong)Stoll(lineElems[2]));
					inFaceSection = true;
				}
			}

			// Parse face property lists.
			if (inFaceSection && lineElems.Length >= 5 && lineElems[0] == "property" && lineElems[1] == "list")
			{
				if (lineElems[4] == "texcoord")
				{
					hasTexcoord = true;
				}
				else
				{
					faceCountType = lineElems[2];
					faceIndexType = lineElems[3];
				}
			}

			if (inVertexSection && lineElems.Length >= 3 && lineElems[0] == "property")
			{
				string dtype = lineElems[1];
				int propIdx = lineElems[2] switch
				{
					"x" => 0,
					"y" => 1,
					"z" => 2,
					"red" or "r" or "diffuse_red" => 3,
					"green" or "g" or "diffuse_green" => 4,
					"blue" or "b" or "diffuse_blue" => 5,
					_ => -1,
				};
				if (propIdx >= 0)
				{
					props[propIdx].Index = numVertexProps;
					props[propIdx].BytePos = (int)numBytesPerVertex;
					props[propIdx].IsDouble = propIdx < 3 && dtype is "double" or "float64";
				}

				numVertexProps += 1;
				numBytesPerVertex += dtype switch
				{
					"float" or "float32" or "int" or "int32" or "uint" or "uint32" => 4UL,
					"double" or "float64" => 8UL,
					"short" or "int16" or "ushort" or "uint16" => 2UL,
					"uchar" or "uint8" or "char" or "int8" => 1UL,
					_ => throw new InvalidOperationException($"Invalid vertex data type: {dtype}"),
				};
			}
		}

		Check.That(
			props[0].Index != -1 && props[1].Index != -1 && props[2].Index != -1,
			"Invalid PLY mesh format: x, y, z properties missing");

		bool hasColors = props[3].Index != -1 && props[4].Index != -1 && props[5].Index != -1;

		// Validate byte positions against buffer size for binary PLY mesh files.
		if (isBinary && numBytesPerVertex > 0)
		{
			string[] names = ["x", "y", "z", "r", "g", "b"];
			for (int i = 0; i < (hasColors ? 6 : 3); i++)
			{
				CheckBytePos(props[i], i < 3 ? (props[i].IsDouble ? 8UL : 4UL) : 1UL, numBytesPerVertex, $"PLY mesh property {names[i]} byte position exceeds vertex buffer size");
			}
		}

		// Sanity-check counts to prevent unbounded memory allocation.
		Check.Le(numVertices, MaxPlyVertices, "PLY mesh declares too many vertices");
		Check.Le(numFaces, MaxPlyFaces, "PLY mesh declares too many faces");

		result.Mesh.Vertices.Capacity = InitialCapacity(numVertices);
		result.Mesh.Faces.Capacity = InitialCapacity(numFaces);
		if (hasTexcoord)
		{
			result.FaceUvs.Capacity = InitialCapacity(numFaces) * 6;
		}

		if (isBinary)
		{
			ReadBinaryMeshBody(file, result, props, hasColors, hasTexcoord, isLittleEndian, numVertices, numFaces, numBytesPerVertex, faceCountType, faceIndexType);
		}
		else
		{
			ReadTextMeshBody(file, result, props, hasColors, hasTexcoord, numVertices, numFaces);
		}

		return result;
	}

	private static void ReadBinaryMeshBody(
		PlyByteReader file,
		PlyTexturedMesh result,
		PointProperty[] props,
		bool hasColors,
		bool hasTexcoord,
		bool isLittleEndian,
		ulong numVertices,
		ulong numFaces,
		ulong numBytesPerVertex,
		string faceCountType,
		string faceIndexType)
	{
		byte[] buffer = new byte[numBytesPerVertex];
		for (ulong i = 0; i < numVertices; ++i)
		{
			Check.That(file.ReadExactly(buffer), $"Unexpected end of PLY file at vertex {i}");

			float x = ReadCoord(buffer, props[0], isLittleEndian);
			float y = ReadCoord(buffer, props[1], isLittleEndian);
			float z = ReadCoord(buffer, props[2], isLittleEndian);
			result.Mesh.Vertices.Add(hasColors
				? new PlyMeshVertex(x, y, z, buffer[props[3].BytePos], buffer[props[4].BytePos], buffer[props[5].BytePos])
				: new PlyMeshVertex(x, y, z));
		}

		// Determine byte sizes of the face count and index types from the header.
		int faceCountBytes = PlyTypeBytes(faceCountType);
		int faceIndexBytes = PlyTypeBytes(faceIndexType);

		byte[] faceBuffer = new byte[Math.Max(faceCountBytes, faceIndexBytes)];
		byte[] uvBuffer = new byte[sizeof(float)];
		int maxIndex = unchecked((int)numVertices);
		Span<int> indices = stackalloc int[3];
		for (ulong i = 0; i < numFaces; ++i)
		{
			Check.That(file.ReadExactly(faceBuffer.AsSpan(0, faceCountBytes)), $"Unexpected end of PLY file at face {i}");
			int numFaceVertices = ReadInt(faceBuffer, faceCountBytes, isLittleEndian);
			Check.Eq(numFaceVertices, 3, "Only triangular faces are supported");

			for (int j = 0; j < 3; ++j)
			{
				Check.That(file.ReadExactly(faceBuffer.AsSpan(0, faceIndexBytes)), $"Unexpected end of PLY file at face {i} index {j}");
				indices[j] = ReadInt(faceBuffer, faceIndexBytes, isLittleEndian);
				Check.Ge(indices[j], 0, $"Negative face vertex index at face {i}");
				Check.Lt(indices[j], maxIndex, $"Face vertex index out of bounds at face {i}");
			}

			result.Mesh.Faces.Add(new PlyMeshFace(indices[0], indices[1], indices[2]));

			if (hasTexcoord)
			{
				// COLMAP reads the count and the UVs unchecked; a truncated file throws here
				// (divergence 113).
				Check.That(file.ReadExactly(faceBuffer.AsSpan(0, 1)), $"Unexpected end of PLY file at face {i}");
				Check.Eq(faceBuffer[0], (byte)6, "Expected 6 texture coordinates per triangular face");

				for (int j = 0; j < 6; ++j)
				{
					Check.That(file.ReadExactly(uvBuffer), $"Unexpected end of PLY file at face {i}");
					result.FaceUvs.Add(isLittleEndian ? BinaryPrimitives.ReadSingleLittleEndian(uvBuffer) : BinaryPrimitives.ReadSingleBigEndian(uvBuffer));
				}
			}
		}
	}

	private static void ReadTextMeshBody(
		PlyByteReader file,
		PlyTexturedMesh result,
		PointProperty[] props,
		bool hasColors,
		bool hasTexcoord,
		ulong numVertices,
		ulong numFaces)
	{
		for (ulong i = 0; i < numVertices; ++i)
		{
			// A missing line reads as empty (std::getline clears it) and fails in At.
			string[] items = SplitWhitespace(CppLineTokens.Trim(file.ReadLine() ?? ""));

			float x = (float)StringToDouble(At(items, props[0].Index));
			float y = (float)StringToDouble(At(items, props[1].Index));
			float z = (float)StringToDouble(At(items, props[2].Index));
			result.Mesh.Vertices.Add(hasColors
				? new PlyMeshVertex(
					x,
					y,
					z,
					unchecked((byte)Stoi(At(items, props[3].Index))),
					unchecked((byte)Stoi(At(items, props[4].Index))),
					unchecked((byte)Stoi(At(items, props[5].Index))))
				: new PlyMeshVertex(x, y, z));
		}

		int maxIndex = unchecked((int)numVertices);
		for (ulong i = 0; i < numFaces; ++i)
		{
			var lineStream = new CppLineTokens(CppLineTokens.Trim(file.ReadLine() ?? ""));

			Check.That(lineStream.TryReadInt32(out int numFaceVertices), null, "line_stream >> num_face_vertices");
			Check.Eq(numFaceVertices, 3, "Only triangular faces are supported");

			int idx1 = 0, idx2 = 0, idx3 = 0;
			Check.That(
				lineStream.TryReadInt32(out idx1) && lineStream.TryReadInt32(out idx2) && lineStream.TryReadInt32(out idx3),
				null,
				"line_stream >> idx1 >> idx2 >> idx3");
			foreach (int idx in (ReadOnlySpan<int>)[idx1, idx2, idx3])
			{
				Check.Ge(idx, 0, $"Negative face vertex index at face {i}");
			}

			foreach (int idx in (ReadOnlySpan<int>)[idx1, idx2, idx3])
			{
				Check.Lt(idx, maxIndex, $"Face vertex index out of bounds at face {i}");
			}

			result.Mesh.Faces.Add(new PlyMeshFace(idx1, idx2, idx3));

			if (hasTexcoord)
			{
				Check.That(lineStream.TryReadInt32(out int numTexcoords), null, "line_stream >> num_texcoords");
				Check.Eq(numTexcoords, 6, "Expected 6 texture coordinates per triangular face");

				for (int j = 0; j < 6; ++j)
				{
					float uv = 0;
					Check.That(lineStream.TryReadFloat(out uv), null, "line_stream >> uv");
					result.FaceUvs.Add(uv);
				}
			}
		}
	}

	private static int PlyTypeBytes(string type) => type switch
	{
		"int" or "int32" or "uint" or "uint32" => 4,
		"short" or "int16" or "ushort" or "uint16" => 2,
		"char" or "int8" or "uchar" or "uint8" => 1,
		_ => throw new InvalidOperationException($"Unsupported PLY face data type: {type}"),
	};

	// The C++ ReadInt: 4 and 2 bytes are signed in the file's byte order, 1 byte is unsigned.
	private static int ReadInt(byte[] buffer, int numBytes, bool isLittleEndian) => numBytes switch
	{
		4 => isLittleEndian ? BinaryPrimitives.ReadInt32LittleEndian(buffer) : BinaryPrimitives.ReadInt32BigEndian(buffer),
		2 => isLittleEndian ? BinaryPrimitives.ReadInt16LittleEndian(buffer) : BinaryPrimitives.ReadInt16BigEndian(buffer),
		_ => buffer[0],
	};
}
