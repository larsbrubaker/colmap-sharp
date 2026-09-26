// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SpatialOrder: the insertion order DelaunayTriangulation3.InsertRange uses - a biased
// randomized insertion order (BRIO) whose rounds are each sorted along a 3D Hilbert curve.
// It stands in for the spatial sort CGAL's range-insert constructor performs, which COLMAP's
// CreateDelaunayTriangulation relies on (CGAL is excluded, docs/LICENSE_AUDIT.md; no CGAL
// source was read).
//
// Sources:
// - N. Amenta, S. Choi and G. Rote, "Incremental constructions con BRIO", SoCG 2003: a
//   random order keeps the expected cost of Bowyer-Watson insertion optimal; grouping the
//   points into rounds of doubling size and sorting each round spatially keeps the walk
//   from the previous insertion short without losing that guarantee.
// - C. Hamilton, "Compact Hilbert Indices", Dalhousie University technical report CS-2006-07,
//   2006: the Gray-code, entry-corner and exit-axis description of the Hilbert curve that
//   HilbertKey follows (the algorithm only; no code listing was used).
//
// Everything is deterministic: the shuffle uses a fixed-seed SplitMix64 generator and
// equal Hilbert keys are ordered by input index.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Geometry.Delaunay;

/// <summary>
/// Computes the BRIO + Hilbert insertion order for a point set.
/// </summary>
internal static class SpatialOrder
{
	private const int BitsPerAxis = 21;

	// Rounds smaller than this are not split further; the first round is at most this big.
	private const int MinRoundSize = 64;

	/// <summary>Returns a permutation of 0..points.Length-1 in BRIO + Hilbert order.</summary>
	public static int[] BrioHilbertOrder(ReadOnlySpan<Vector3d> points)
	{
		int n = points.Length;
		var order = new int[n];
		for (int i = 0; i < n; ++i)
		{
			order[i] = i;
		}

		ulong state = 0x9E3779B97F4A7C15UL;
		for (int i = n - 1; i > 0; --i)
		{
			int j = (int)(NextRandom(ref state) % (ulong)(i + 1));
			(order[i], order[j]) = (order[j], order[i]);
		}

		var keys = ComputeHilbertKeys(points);

		// Round k covers [size/2, size) of the shuffled array, walking size down by halves;
		// what remains at the front is the first round. Each round is sorted by key.
		int end = n;
		while (end > 0)
		{
			int begin = end > MinRoundSize ? end / 2 : 0;
			SortByKey(order, begin, end - begin, keys);
			end = begin;
		}

		return order;
	}

	private static void SortByKey(int[] order, int start, int length, ulong[] keys)
	{
		Array.Sort(order, start, length, Comparer<int>.Create((a, b) =>
		{
			int c = keys[a].CompareTo(keys[b]);
			return c != 0 ? c : a.CompareTo(b);
		}));
	}

	private static ulong[] ComputeHilbertKeys(ReadOnlySpan<Vector3d> points)
	{
		var keys = new ulong[points.Length];
		if (points.Length == 0)
		{
			return keys;
		}

		double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
		double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;
		foreach (var p in points)
		{
			minX = Math.Min(minX, p.X);
			minY = Math.Min(minY, p.Y);
			minZ = Math.Min(minZ, p.Z);
			maxX = Math.Max(maxX, p.X);
			maxY = Math.Max(maxY, p.Y);
			maxZ = Math.Max(maxZ, p.Z);
		}

		// One scale for all axes, so the curve follows the actual geometry.
		double extent = Math.Max(maxX - minX, Math.Max(maxY - minY, maxZ - minZ));
		double scale = extent > 0 ? ((1 << BitsPerAxis) - 1) / extent : 0;
		for (int i = 0; i < points.Length; ++i)
		{
			keys[i] = HilbertKey(
				Quantize((points[i].X - minX) * scale),
				Quantize((points[i].Y - minY) * scale),
				Quantize((points[i].Z - minZ) * scale));
		}

		return keys;
	}

	private static uint Quantize(double value)
	{
		const double max = (1 << BitsPerAxis) - 1;
		return (uint)Math.Clamp(value, 0, max);
	}

	/// <summary>
	/// Position of the cell (x, y, z), each coordinate below 2^BitsPerAxis, along a 3D
	/// Hilbert curve, as a 63-bit key. Follows the curve's definition as nested sub-cube
	/// visits: each level picks one of 8 octants in Gray-code order, and the octant chosen
	/// fixes the corner where the curve enters the next level and the axis along which it
	/// leaves. The entry-corner and exit-axis functions are those derived in C. Hamilton,
	/// "Compact Hilbert Indices", Dalhousie University technical report CS-2006-07.
	/// </summary>
	internal static ulong HilbertKey(uint x, uint y, uint z)
	{
		const int dimensions = 3;
		const uint allAxes = (1u << dimensions) - 1;
		uint entry = 0;
		int direction = 0;
		ulong key = 0;
		for (int level = BitsPerAxis - 1; level >= 0; --level)
		{
			// The octant of this level, one bit per axis (x is the lowest bit).
			uint octant = ((x >> level) & 1u) | (((y >> level) & 1u) << 1) | (((z >> level) & 1u) << 2);

			// Map it into the frame where the curve starts at corner 0 and leaves along axis 0,
			// then read its position along the Gray-code order of octants.
			uint local = RotateRight(octant ^ entry, direction + 1, dimensions, allAxes);
			uint rank = GrayRank(local);
			key = (key << dimensions) | rank;

			// Carry the sub-curve's entry corner and exit direction to the next level.
			entry ^= RotateLeft(EntryCorner(rank), direction + 1, dimensions, allAxes);
			direction = (direction + ExitAxis(rank, dimensions) + 1) % dimensions;
		}

		return key;
	}

	// Inverse of the reflected Gray code g(i) = i ^ (i >> 1).
	private static uint GrayRank(uint gray)
	{
		uint rank = gray;
		for (uint shifted = gray >> 1; shifted != 0; shifted >>= 1)
		{
			rank ^= shifted;
		}

		return rank;
	}

	// Corner where the sub-curve of octant rank i starts: 0 for the first, otherwise the
	// Gray code of the largest even number below i.
	private static uint EntryCorner(uint i)
	{
		if (i == 0)
		{
			return 0;
		}

		uint even = (i - 1) & ~1u;
		return even ^ (even >> 1);
	}

	// Axis along which the sub-curve of octant rank i leaves: the bit that flips between the
	// Gray codes around i, found by counting trailing one bits.
	private static int ExitAxis(uint i, int dimensions)
	{
		if (i == 0)
		{
			return 0;
		}

		uint around = (i & 1u) == 0 ? i - 1 : i;
		return System.Numerics.BitOperations.TrailingZeroCount(~around) % dimensions;
	}

	private static uint RotateRight(uint value, int shift, int width, uint mask)
	{
		shift %= width;
		return ((value >> shift) | (value << (width - shift))) & mask;
	}

	private static uint RotateLeft(uint value, int shift, int width, uint mask)
	{
		shift %= width;
		return ((value << shift) | (value >> (width - shift))) & mask;
	}

	private static ulong NextRandom(ref ulong state)
	{
		// SplitMix64 (Steele, Lea and Flood, OOPSLA 2014).
		ulong z = state += 0x9E3779B97F4A7C15UL;
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
		return z ^ (z >> 31);
	}
}
