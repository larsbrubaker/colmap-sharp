// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ConsistencyGraph: colmap/mvs/consistency_graph.h and consistency_graph.cc - the per-pixel
// lists of geometrically consistent source images PatchMatch writes when filtering is on,
// with a width x height index into the flat list. Its .bin file shares Mat.cs's header and
// little-endian payload (MatFile). Tests: ColmapSharp.Tests/Mvs/ConsistencyGraphTests.cs
// (consistency_graph_test.cc 1:1).
//
// Tier A (exact): bookkeeping, and a byte-identical file format.
//
// Translation notes:
// - Eigen::MatrixXi map_ becomes a row-major int[] (only its size and elements are used).
// - GetImageIdxs returns the image list as a ReadOnlySpan<int> into the data (empty for a
//   pixel without consistent images) instead of a count and a pointer.

using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>
/// Port of colmap::mvs::ConsistencyGraph: the list of geometrically consistent images, in
/// the format <c>r_1, c_1, N_1, i_11, ..., i_1N_1, r_2, c_2, N_2, ...</c> - where r, c are
/// the pixel's image coordinates (stored column first), N is the number of consistent
/// images, followed by the N image indices. Only pixels that were not filtered are listed,
/// and the graph is only filled if filtering is enabled.
/// </summary>
public sealed class ConsistencyGraph
{
	private const int NoConsistentImageIds = -1;

	private int[] data = [];
	private int[] map = [];
	private int mapWidth;
	private int mapHeight;

	/// <summary>An empty graph.</summary>
	public ConsistencyGraph()
	{
	}

	/// <summary>A width x height graph over a copy of <paramref name="data"/>.</summary>
	public ConsistencyGraph(int width, int height, ReadOnlySpan<int> data)
	{
		this.data = data.ToArray();
		InitializeMap(width, height);
	}

	/// <summary>Size of the data and the index map in bytes.</summary>
	public long GetNumBytes() => ((long)data.Length + map.Length) * sizeof(int);

	/// <summary>The indices of the images consistent with pixel (row, col); empty if none.</summary>
	public ReadOnlySpan<int> GetImageIdxs(int row, int col)
	{
		int index = map[row * mapWidth + col];
		if (index == NoConsistentImageIds)
		{
			return ReadOnlySpan<int>.Empty;
		}

		int numImages = data[index];
		return data.AsSpan(index + 1, numImages);
	}

	/// <summary>Reads a COLMAP consistency graph .bin file.</summary>
	public void Read(string path)
	{
		using FileStream file = FileOpen.OpenRead(path);
		(int width, int height, int depth) = MatFile.ReadHeader(file);

		Check.Gt(width, 0);
		Check.Gt(height, 0);
		Check.Gt(depth, 0);

		long numBytes = file.Length - file.Position;
		data = new int[numBytes / sizeof(int)];
		MatFile.ReadLittleEndian<int>(file, data);

		InitializeMap(width, height);
	}

	/// <summary>Writes the graph as a COLMAP .bin file ("width&amp;height&amp;1&amp;" + data).</summary>
	public void Write(string path)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		MatFile.WriteHeader(file, mapWidth, mapHeight, 1);
		MatFile.WriteLittleEndian<int>(file, data);
	}

	private void InitializeMap(int width, int height)
	{
		map = new int[checked(width * height)];
		mapWidth = width;
		mapHeight = height;
		Array.Fill(map, NoConsistentImageIds);
		for (int i = 0; i < data.Length;)
		{
			if (i + 2 >= data.Length)
			{
				// Formatted only on failure: this loop visits every listed pixel.
				Check.Lt(i + 2, data.Length, $"Corrupt consistency graph: insufficient data at offset {i}");
			}

			int col = data[i];
			int row = data[i + 1];
			int numImages = data[i + 2];
			if (numImages < 0)
			{
				Check.Ge(numImages, 0, $"Corrupt consistency graph: negative num_images at offset {i}");
			}

			Check.Ge(col, 0);
			Check.Lt(col, width);
			Check.Ge(row, 0);
			Check.Lt(row, height);
			if (numImages > 0)
			{
				map[row * width + col] = i + 2;
			}

			i += 3 + numImages;
		}
	}
}
