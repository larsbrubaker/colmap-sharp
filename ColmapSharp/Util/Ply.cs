// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Ply: the PLY readers of colmap/util/ply.cc - ReadPly (point clouds), ReadPlyMesh
// (triangle meshes, optionally textured) and HasPlyMeshFaces. The writers are in
// Ply.Write.cs, the data types in PlyTypes.cs, the buffered byte source in PlyByteReader.cs.
// Scene/Reconstruction.Ply.cs (ConvertToPLY / ImportPLY) builds on them; reconstruction_io's
// ExportPLY, which writes ConvertToPLY's points, is not ported yet. Tests:
// ColmapSharp.Tests/Util/PlyTests.cs (ply_test.cc 1:1).
//
// The header is parsed line by line exactly as COLMAP does (std::getline, StringTrim,
// StringSplit on ' ' with token compression, std::stoll for element counts), and the body
// starts at the byte after the "end_header" line. Every value is read with the same types
// and conversions as the C++ (a double coordinate is narrowed to float, an ASCII color goes
// through std::stoi and wraps into a byte).
//
// Malformed input fails with an exception, never with a huge allocation: counts are checked
// against COLMAP's limits (2^32), and the lists grow as data is actually read instead of
// reserving the declared count up front. Where the C++ reads past the end of a binary file
// without checking (a face's texcoord count and UVs, which is undefined behavior there), this
// port throws "Unexpected end of PLY file" instead (divergence 113).

using System.Buffers.Binary;

namespace ColmapSharp.Util;

/// <summary>Port of colmap/util/ply.h: PLY point cloud and mesh files, text and binary.</summary>
public static partial class Ply
{
	private const ulong MaxPlyVertices = 1UL << 32;
	private const ulong MaxPlyFaces = 1UL << 32;

	// Lists start at most this large and grow with the data actually read.
	private const int MaxInitialCapacity = 1 << 16;

	/// <summary>Reads a PLY point cloud from a text or binary file.</summary>
	public static List<PlyPoint> ReadPly(string path)
	{
		using FileStream file = FileOpen.OpenRead(path);
		return ReadPly(file);
	}

	/// <summary>Reads a PLY point cloud from a text or binary stream.</summary>
	public static List<PlyPoint> ReadPly(Stream stream)
	{
		var file = new PlyByteReader(stream);

		// The index of the property for ASCII PLY files, and its position in bytes for
		// binary PLY files, per property x, y, z, nx, ny, nz, r, g, b.
		var props = new PointProperty[9];
		for (int i = 0; i < props.Length; i++)
		{
			props[i] = new PointProperty();
		}

		bool inVertexSection = false;
		bool isBinary = false;
		bool isLittleEndian = false;
		ulong numBytesPerLine = 0;
		ulong numVertices = 0;

		int index = 0;
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

			if (lineElems.Length >= 3 && lineElems[0] == "element")
			{
				inVertexSection = false;
				if (lineElems[1] == "vertex")
				{
					numVertices = unchecked((ulong)Stoll(lineElems[2]));
					inVertexSection = true;
				}
				else if (Stoll(lineElems[2]) > 0)
				{
					Log.Warning($"Only vertex elements supported; ignoring {lineElems[1]}");
				}
			}

			if (!inVertexSection)
			{
				continue;
			}

			// Show diffuse, ambient, specular colors as regular colors.
			if (lineElems.Length >= 3 && lineElems[0] == "property")
			{
				string dtype = lineElems[1];
				bool isDouble = dtype is "double" or "float64";
				Check.That(
					dtype is "float" or "float32" or "double" or "float64" or "uchar",
					"PLY import only supports float, double, and uchar data types");

				int propIdx = PointPropertyIndex(line);
				if (propIdx >= 0)
				{
					props[propIdx].Index = index;
					props[propIdx].BytePos = (int)numBytesPerLine;
					props[propIdx].IsDouble = propIdx < 6 && isDouble;
				}

				index += 1;
				numBytesPerLine += dtype switch
				{
					"float" or "float32" => 4UL,
					"double" or "float64" => 8UL,
					_ => 1UL,
				};
			}
		}

		bool isNormalMissing = props[3].Index == -1 || props[4].Index == -1 || props[5].Index == -1;
		bool isRgbMissing = props[6].Index == -1 || props[7].Index == -1 || props[8].Index == -1;

		Check.That(
			props[0].Index != -1 && props[1].Index != -1 && props[2].Index != -1,
			"Invalid PLY file format: x, y, z properties missing");

		// Validate byte positions against buffer size to prevent out-of-bounds reads from
		// crafted PLY headers.
		if (isBinary && numBytesPerLine > 0)
		{
			string[] names = ["x", "y", "z", "nx", "ny", "nz", "r", "g", "b"];
			for (int i = 0; i < props.Length; i++)
			{
				CheckBytePos(props[i], i < 6 ? (props[i].IsDouble ? 8UL : 4UL) : 1UL, numBytesPerLine, $"PLY property {names[i]} byte position exceeds line buffer size");
			}
		}

		// Sanity-check num_vertices to prevent unbounded memory allocation from a crafted
		// PLY header.
		Check.Le(numVertices, MaxPlyVertices, "PLY file declares too many vertices");

		var points = new List<PlyPoint>(InitialCapacity(numVertices));

		if (isBinary)
		{
			byte[] buffer = new byte[numBytesPerLine];
			for (ulong i = 0; i < numVertices; ++i)
			{
				Check.That(file.ReadExactly(buffer), $"Unexpected end of PLY file at vertex {i}");

				var point = new PlyPoint
				{
					X = ReadCoord(buffer, props[0], isLittleEndian),
					Y = ReadCoord(buffer, props[1], isLittleEndian),
					Z = ReadCoord(buffer, props[2], isLittleEndian),
				};

				if (!isNormalMissing)
				{
					point.Nx = ReadCoord(buffer, props[3], isLittleEndian);
					point.Ny = ReadCoord(buffer, props[4], isLittleEndian);
					point.Nz = ReadCoord(buffer, props[5], isLittleEndian);
				}

				if (!isRgbMissing)
				{
					point.R = buffer[props[6].BytePos];
					point.G = buffer[props[7].BytePos];
					point.B = buffer[props[8].BytePos];
				}

				points.Add(point);
			}
		}
		else
		{
			while ((line = file.ReadLine()) != null)
			{
				string[] items = SplitWhitespace(CppLineTokens.Trim(line));

				var point = new PlyPoint
				{
					X = (float)StringToDouble(At(items, props[0].Index)),
					Y = (float)StringToDouble(At(items, props[1].Index)),
					Z = (float)StringToDouble(At(items, props[2].Index)),
				};

				if (!isNormalMissing)
				{
					point.Nx = (float)StringToDouble(At(items, props[3].Index));
					point.Ny = (float)StringToDouble(At(items, props[4].Index));
					point.Nz = (float)StringToDouble(At(items, props[5].Index));
				}

				if (!isRgbMissing)
				{
					point.R = unchecked((byte)Stoi(At(items, props[6].Index)));
					point.G = unchecked((byte)Stoi(At(items, props[7].Index)));
					point.B = unchecked((byte)Stoi(At(items, props[8].Index)));
				}

				points.Add(point);
			}
		}

		return points;
	}

	/// <summary>Returns true if the PLY file declares a non-empty face element (i.e., is a mesh).</summary>
	public static bool HasPlyMeshFaces(string path)
	{
		using FileStream file = FileOpen.OpenRead(path);
		return HasPlyMeshFaces(file);
	}

	/// <summary>Returns true if the PLY stream declares a non-empty face element (i.e., is a mesh).</summary>
	public static bool HasPlyMeshFaces(Stream stream)
	{
		var file = new PlyByteReader(stream);
		string? line;
		while ((line = file.ReadLine()) != null)
		{
			line = CppLineTokens.Trim(line);
			if (line == "end_header")
			{
				break;
			}

			string[] lineElems = StringSplit(line);
			if (lineElems.Length >= 3 && lineElems[0] == "element" && lineElems[1] == "face")
			{
				try
				{
					if (Stoll(lineElems[2]) > 0)
					{
						return true;
					}
				}
				catch (FormatException)
				{
					// e.what() of libc++'s std::invalid_argument from std::stoll.
					Log.Warning("Malformed face element line in PLY header: stoll: no conversion");
				}
				catch (OverflowException)
				{
					// e.what() of libc++'s std::out_of_range from std::stoll.
					Log.Warning("Malformed face element line in PLY header: stoll: out of range");
				}
			}
		}

		return false;
	}

	// One vertex property of ReadPly/ReadPlyMesh: its column (ASCII), byte offset (binary)
	// and whether it is stored as a double.
	private sealed class PointProperty
	{
		public int Index = -1;
		public int BytePos = -1;
		public bool IsDouble;
	}

	// The property of ReadPly the header line names (0..8 = x, y, z, nx, ny, nz, r, g, b), or
	// -1. The whole line must match, so "property  float x" names nothing.
	private static int PointPropertyIndex(string line)
	{
		string[] coords = ["x", "y", "z", "nx", "ny", "nz"];
		for (int i = 0; i < coords.Length; i++)
		{
			string c = coords[i];
			if (line == "property float " + c || line == "property float32 " + c ||
				line == "property double " + c || line == "property float64 " + c)
			{
				return i;
			}
		}

		string[][] colors =
		[
			["r", "red", "diffuse_red", "ambient_red", "specular_red"],
			["g", "green", "diffuse_green", "ambient_green", "specular_green"],
			["b", "blue", "diffuse_blue", "ambient_blue", "specular_blue"],
		];
		for (int i = 0; i < colors.Length; i++)
		{
			foreach (string name in colors[i])
			{
				if (line == "property uchar " + name)
				{
					return 6 + i;
				}
			}
		}

		return -1;
	}

	private static void ParseFormat(string line, ref bool isBinary, ref bool isLittleEndian)
	{
		if (!line.StartsWith("format", StringComparison.Ordinal))
		{
			return;
		}

		if (line == "format ascii 1.0")
		{
			isBinary = false;
		}
		else if (line == "format binary_little_endian 1.0")
		{
			isBinary = true;
			isLittleEndian = true;
		}
		else if (line == "format binary_big_endian 1.0")
		{
			isBinary = true;
			isLittleEndian = false;
		}
	}

	private static void CheckBytePos(PointProperty prop, ulong typeSize, ulong numBytes, string message)
	{
		if (prop.BytePos >= 0)
		{
			Check.Le((ulong)prop.BytePos + typeSize, numBytes, message);
		}
	}

	private static float ReadCoord(byte[] buffer, PointProperty prop, bool isLittleEndian)
	{
		ReadOnlySpan<byte> bytes = buffer.AsSpan(prop.BytePos);
		if (prop.IsDouble)
		{
			return (float)(isLittleEndian ? BinaryPrimitives.ReadDoubleLittleEndian(bytes) : BinaryPrimitives.ReadDoubleBigEndian(bytes));
		}

		return isLittleEndian ? BinaryPrimitives.ReadSingleLittleEndian(bytes) : BinaryPrimitives.ReadSingleBigEndian(bytes);
	}

	private static int InitialCapacity(ulong declared) => (int)Math.Min(declared, MaxInitialCapacity);

	// COLMAP's StringSplit(line, " ") (boost::split with token_compress_on) on a trimmed line.
	private static string[] StringSplit(string line) =>
		line.Length == 0 ? [""] : line.Split(' ', StringSplitOptions.RemoveEmptyEntries);

	// `while (line_stream >> item) items.push_back(item)`.
	private static string[] SplitWhitespace(string line) =>
		line.Split(CppLineTokens.CppWhitespace, StringSplitOptions.RemoveEmptyEntries);

	// std::vector::at: throws on an index past the end (a line with too few columns).
	private static string At(string[] items, int index)
	{
		if (index < 0 || index >= items.Length)
		{
			throw new ArgumentOutOfRangeException(nameof(index), $"PLY line has no column {index}: vector::at out of range");
		}

		return items[index];
	}

	// COLMAP's StringToDouble: the whole string must be one double.
	// The callers pass one token of a whitespace-split line, so "no trailing characters"
	// holds once the token parses whole.
	private static double StringToDouble(string str)
	{
		Check.That(CppLineTokens.TryParseDoubleToken(str, out double value), $"Failed to parse floating-point value: {str}");
		return value;
	}

	// std::stoll (base 10): leading whitespace, an optional sign and at least one digit; the
	// rest of the string is ignored. No digits throws (std::invalid_argument), a value out of
	// range throws (std::out_of_range).
	private static long Stoll(string str)
	{
		int i = 0;
		while (i < str.Length && Array.IndexOf(CppLineTokens.CppWhitespace, str[i]) >= 0)
		{
			i++;
		}

		int start = i;
		if (i < str.Length && str[i] is '+' or '-')
		{
			i++;
		}

		int digitsStart = i;
		while (i < str.Length && char.IsAsciiDigit(str[i]))
		{
			i++;
		}

		if (i == digitsStart)
		{
			throw new FormatException($"stoll: no conversion: '{str}'");
		}

		if (!long.TryParse(str.AsSpan(start, i - start), System.Globalization.NumberStyles.AllowLeadingSign, System.Globalization.CultureInfo.InvariantCulture, out long value))
		{
			throw new OverflowException($"stoll: out of range: '{str}'");
		}

		return value;
	}

	// std::stoi: std::stoll's parsing with the int range.
	private static int Stoi(string str)
	{
		long value = Stoll(str);
		if (value is < int.MinValue or > int.MaxValue)
		{
			throw new OverflowException($"stoi: out of range: '{str}'");
		}

		return (int)value;
	}
}
