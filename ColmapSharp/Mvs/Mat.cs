// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Mat: colmap/mvs/mat.h and mat.cc, the dense width x height x depth array every MVS map
// is stored in (DepthMap.cs and NormalMap.cs derive from it; PatchMatch writes and fusion
// reads its .bin files). Tests: ColmapSharp.Tests/Mvs/MatTests.cs (mat_test.cc 1:1, plus
// C#-only byte-exact file round trips).
//
// Tier A (exact): plain storage, and the .bin format is byte-identical to COLMAP's.
//
// Layout: slice-major, then row-major - element (row, col, slice) lives at
// slice * width * height + row * width + col, so each channel is a contiguous width x height
// plane. PatchMatch and fusion index Data directly with that formula in their hot loops.
//
// File format (Read/Write): an ASCII header "<width>&<height>&<depth>&" followed directly by
// the width * height * depth elements in little-endian binary, in the layout above.
//
// Translation notes:
// - size_t dimensions become int (a single map never approaches 2^31 elements) and
//   GetNumBytes becomes long.
// - C++ only instantiates Read/Write for float; here they work for any unmanaged T through
//   the same little-endian byte layout.
// - Get/Set bounds-check the flat index like std::vector::at (a row or column past its
//   dimension that still lands inside the buffer is not caught, as in COLMAP).
// - Reading a truncated file throws (EndOfStreamException) where COLMAP's stream silently
//   stops filling the buffer (docs/CPP_DIVERGENCES.md, entry 62).

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>
/// Port of colmap::mvs::Mat: a width x height x depth array stored as depth contiguous
/// width x height planes (see the file header for the layout and the .bin format).
/// </summary>
public partial class Mat<T>
	where T : unmanaged
{
	/// <summary>Width in pixels (columns).</summary>
	protected int width;

	/// <summary>Height in pixels (rows).</summary>
	protected int height;

	/// <summary>Number of channels (slices).</summary>
	protected int depth;

	/// <summary>The elements, slice-major then row-major.</summary>
	protected T[] data;

	/// <summary>An empty 0 x 0 x 0 matrix.</summary>
	public Mat()
		: this(0, 0, 0)
	{
	}

	/// <summary>A zero-filled width x height x depth matrix.</summary>
	public Mat(int width, int height, int depth)
	{
		this.width = width;
		this.height = height;
		this.depth = depth;
		data = new T[checked(width * height * depth)];
	}

	/// <summary>Width in pixels (columns).</summary>
	public int GetWidth() => width;

	/// <summary>Height in pixels (rows).</summary>
	public int GetHeight() => height;

	/// <summary>Number of channels (slices).</summary>
	public int GetDepth() => depth;

	/// <summary>Size of the element buffer in bytes.</summary>
	public long GetNumBytes() => (long)data.Length * Unsafe.SizeOf<T>();

	/// <summary>
	/// The element buffer itself (C++ GetPtr / GetData), slice-major then row-major. Writes
	/// go straight into the matrix; hot loops use this instead of Get/Set.
	/// </summary>
	public T[] Data => data;

	/// <summary>The element at (row, col) of slice 0.</summary>
	public T Get(int row, int col) => Get(row, col, 0);

	/// <summary>The element at (row, col, slice).</summary>
	public T Get(int row, int col, int slice) => data[(slice * height + row) * width + col];

	/// <summary>Copies the depth values of pixel (row, col) into <paramref name="values"/>.</summary>
	public void GetSlice(int row, int col, Span<T> values)
	{
		for (int slice = 0; slice < depth; ++slice)
		{
			values[slice] = Get(row, col, slice);
		}
	}

	/// <summary>Sets the element at (row, col) of slice 0.</summary>
	public void Set(int row, int col, T value) => Set(row, col, 0, value);

	/// <summary>Sets the element at (row, col, slice).</summary>
	public void Set(int row, int col, int slice, T value) => data[(slice * height + row) * width + col] = value;

	/// <summary>Sets every element to <paramref name="value"/>.</summary>
	public void Fill(T value) => Array.Fill(data, value);

	/// <summary>Reads a COLMAP .bin matrix file, replacing the dimensions and data.</summary>
	public void Read(string path)
	{
		using FileStream file = FileOpen.OpenRead(path);
		(width, height, depth) = MatFile.ReadHeader(file);
		Check.Gt(width, 0, path);
		Check.Gt(height, 0, path);
		Check.Gt(depth, 0, path);
		data = new T[checked(width * height * depth)];
		MatFile.ReadLittleEndian<T>(file, data);
	}

	/// <summary>Writes the matrix as a COLMAP .bin file.</summary>
	public void Write(string path)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		MatFile.WriteHeader(file, width, height, depth);
		MatFile.WriteLittleEndian<T>(file, data);
	}
}

/// <summary>
/// The pieces of COLMAP's MVS .bin files shared by <see cref="Mat{T}"/> and
/// <see cref="ConsistencyGraph"/>: the "w&amp;h&amp;d&amp;" text header and the little-endian
/// element payload (util/endian.h's Read/WriteBinaryLittleEndian).
/// </summary>
internal static class MatFile
{
	/// <summary>Writes "width&amp;height&amp;depth&amp;" in ASCII, as `stream &lt;&lt; w &lt;&lt; "&amp;"` does.</summary>
	public static void WriteHeader(Stream stream, long width, long height, long depth)
	{
		string header = string.Create(CultureInfo.InvariantCulture, $"{width}&{height}&{depth}&");
		foreach (char c in header)
		{
			stream.WriteByte((byte)c);
		}
	}

	/// <summary>
	/// Parses the header like `file &gt;&gt; width &gt;&gt; c &gt;&gt; height &gt;&gt; c &gt;&gt; depth &gt;&gt; c`:
	/// each number and separator may be preceded by whitespace. Leaves the stream right after
	/// the third separator. A malformed header yields zero dimensions (the stream's failbit
	/// in C++), which the callers' size checks reject.
	/// </summary>
	public static (int Width, int Height, int Depth) ReadHeader(Stream stream)
	{
		long w = ReadNumber(stream);
		ReadSeparator(stream);
		long h = w > 0 ? ReadNumber(stream) : 0;
		ReadSeparator(stream);
		long d = h > 0 ? ReadNumber(stream) : 0;
		ReadSeparator(stream);
		return (checked((int)w), checked((int)h), checked((int)d));
	}

	private static long ReadNumber(Stream stream)
	{
		int c = SkipWhitespace(stream);
		long value = 0;
		bool any = false;
		while (c >= '0' && c <= '9')
		{
			value = checked(value * 10 + (c - '0'));
			any = true;
			c = stream.ReadByte();
		}

		// operator>> leaves the first non-digit in the stream.
		if (c >= 0)
		{
			stream.Seek(-1, SeekOrigin.Current);
		}

		return any ? value : 0;
	}

	private static void ReadSeparator(Stream stream) => SkipWhitespace(stream);

	private static int SkipWhitespace(Stream stream)
	{
		int c = stream.ReadByte();
		while (c is ' ' or '\t' or '\n' or '\v' or '\f' or '\r')
		{
			c = stream.ReadByte();
		}

		return c;
	}

	/// <summary>Fills <paramref name="values"/> from little-endian bytes; throws on a short stream.</summary>
	public static void ReadLittleEndian<T>(Stream stream, Span<T> values)
		where T : unmanaged
	{
		Span<byte> bytes = MemoryMarshal.AsBytes(values);
		stream.ReadExactly(bytes);
		if (!BitConverter.IsLittleEndian)
		{
			ReverseEach<T>(bytes);
		}
	}

	/// <summary>Writes <paramref name="values"/> as little-endian bytes.</summary>
	public static void WriteLittleEndian<T>(Stream stream, ReadOnlySpan<T> values)
		where T : unmanaged
	{
		if (BitConverter.IsLittleEndian)
		{
			stream.Write(MemoryMarshal.AsBytes(values));
			return;
		}

		byte[] bytes = MemoryMarshal.AsBytes(values).ToArray();
		ReverseEach<T>(bytes);
		stream.Write(bytes);
	}

	private static void ReverseEach<T>(Span<byte> bytes)
		where T : unmanaged
	{
		int size = Unsafe.SizeOf<T>();
		for (int i = 0; i < bytes.Length; i += size)
		{
			bytes.Slice(i, size).Reverse();
		}
	}
}
