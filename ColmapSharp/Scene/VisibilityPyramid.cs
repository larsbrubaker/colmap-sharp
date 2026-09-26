// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// VisibilityPyramid: port of colmap/scene/visibility_pyramid.h and .cc, a multi-level grid
// over an image that scores how evenly its triangulated points are spread (each newly
// occupied cell on level l adds that level's cell count). The incremental mapper uses the
// score to choose the next image to register. Tests:
// ColmapSharp.Tests/Scene/VisibilityPyramidTests.cs (visibility_pyramid_test.cc 1:1).
//
// Tier A (exact): integer bookkeeping; the cell of a point is the same double expression
// as COLMAP's.
//
// Translation notes:
// - Each level is a dim x dim int grid (Eigen::MatrixXi); stored row-major here, which only
//   changes the memory layout, not any value.
// - size_t becomes int for the level count and image size, ulong for the scores.
// - CellForPoint converts max_dim * x / width to size_t. C++ leaves that conversion
//   undefined for negative or NaN input; .NET's conversion saturates (negative and NaN give
//   0), and the clamp then keeps the cell in range, which is what the clamp intends.
// - COLMAP accumulates max_score_ from `int dim` in int arithmetic, which overflows
//   (undefined) from 8 levels on; here it is ulong and stays exact. Identical below that.

using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::VisibilityPyramid: scores the spatial spread of points in an image.
/// </summary>
public sealed class VisibilityPyramid
{
	// Occupancy counts per level; level l is (2^(l+1)) x (2^(l+1)), row-major.
	private readonly int[][] pyramid;

	/// <summary>An empty pyramid with no levels.</summary>
	public VisibilityPyramid()
		: this(0, 0, 0)
	{
	}

	/// <summary>A pyramid with <paramref name="numLevels"/> levels over a width x height image.</summary>
	public VisibilityPyramid(int numLevels, int width, int height)
	{
		Width = width;
		Height = height;
		pyramid = new int[numLevels][];
		for (int level = 0; level < numLevels; ++level)
		{
			int dim = 1 << (level + 1);
			pyramid[level] = new int[dim * dim];
			ulong cells = (ulong)dim * (ulong)dim;
			MaxScore += cells * cells;
		}
	}

	/// <summary>The number of levels.</summary>
	public int NumLevels => pyramid.Length;

	/// <summary>The width of the input point range.</summary>
	public int Width { get; }

	/// <summary>The height of the input point range.</summary>
	public int Height { get; }

	/// <summary>The overall visibility score.</summary>
	public ulong Score { get; private set; }

	/// <summary>The maximum score, reached when every cell of every level is populated.</summary>
	public ulong MaxScore { get; }

	/// <summary>Adds a point at (<paramref name="x"/>, <paramref name="y"/>).</summary>
	public void SetPoint(double x, double y)
	{
		Check.Gt(pyramid.Length, 0);

		(int cx, int cy) = CellForPoint(x, y);

		for (int i = pyramid.Length - 1; i >= 0; --i)
		{
			int[] level = pyramid[i];
			int dim = 1 << (i + 1);
			int index = cy * dim + cx;

			level[index] += 1;
			if (level[index] == 1)
			{
				Score += (ulong)level.Length;
			}

			cx >>= 1;
			cy >>= 1;
		}

		Check.Le(Score, MaxScore);
	}

	/// <summary>Removes a point previously added at (<paramref name="x"/>, <paramref name="y"/>).</summary>
	public void ResetPoint(double x, double y)
	{
		Check.Gt(pyramid.Length, 0);

		(int cx, int cy) = CellForPoint(x, y);

		for (int i = pyramid.Length - 1; i >= 0; --i)
		{
			int[] level = pyramid[i];
			int dim = 1 << (i + 1);
			int index = cy * dim + cx;

			level[index] -= 1;
			if (level[index] == 0)
			{
				// size_t arithmetic in COLMAP, so an unmatched reset wraps the same way here
				// before the check below catches it.
				Score = unchecked(Score - (ulong)level.Length);
			}

			cx >>= 1;
			cy >>= 1;
		}

		Check.Le(Score, MaxScore);
	}

	// The cell of the finest level containing (x, y).
	private (int Cx, int Cy) CellForPoint(double x, double y)
	{
		Check.Gt(Width, 0);
		Check.Gt(Height, 0);
		int maxDim = 1 << pyramid.Length;
		ulong cx = Math.Clamp((ulong)(maxDim * x / Width), 0UL, (ulong)(maxDim - 1));
		ulong cy = Math.Clamp((ulong)(maxDim * y / Height), 0UL, (ulong)(maxDim - 1));
		return ((int)cx, (int)cy);
	}
}
