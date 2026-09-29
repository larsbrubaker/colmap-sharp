// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// OccupancyGrid: the visual hull's occupancy at the finest octree level (docs/QUALITY_PLAN.md,
// stage 3a). Not a COLMAP port. VisualHull.cs fills it; HullMarchingCubes.cs extracts its iso
// surface; later stages (3b trimming, 3c PatchMatch depth bounds) read it back through Value and
// Sample. Values are in [0, 1]: 1 inside the hull, 0 outside, and fractional near the silhouette
// boundary, where the value is the mask's bilinear value at the voxel center.
//
// Layout: cubic voxels of side VoxelSize; voxel (i, j, k) covers
// Origin + [i, i+1) x [j, j+1) x [k, k+1) times VoxelSize, and its value is sampled at the center.
// Values are stored x fastest, then y, then z.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Silhouette;

/// <summary>A dense occupancy grid of cubic voxels, values in [0, 1] sampled at voxel centers.</summary>
public sealed class OccupancyGrid
{
	/// <summary>An all-empty grid of nx x ny x nz voxels of side <paramref name="voxelSize"/> from <paramref name="origin"/>.</summary>
	public OccupancyGrid(Vector3d origin, double voxelSize, int nx, int ny, int nz)
	{
		Check.That(voxelSize > 0);
		Check.That(nx > 0 && ny > 0 && nz > 0);
		Origin = origin;
		VoxelSize = voxelSize;
		NX = nx;
		NY = ny;
		NZ = nz;
		Values = new float[(long)nx * ny * nz];
	}

	/// <summary>The minimum corner of voxel (0, 0, 0).</summary>
	public Vector3d Origin { get; }

	/// <summary>The side of a voxel.</summary>
	public double VoxelSize { get; }

	/// <summary>Voxels along x.</summary>
	public int NX { get; }

	/// <summary>Voxels along y.</summary>
	public int NY { get; }

	/// <summary>Voxels along z.</summary>
	public int NZ { get; }

	/// <summary>The values, x fastest, then y, then z.</summary>
	public float[] Values { get; }

	/// <summary>The index of voxel (i, j, k) in <see cref="Values"/>.</summary>
	public int Index(int i, int j, int k) => (k * NY + j) * NX + i;

	/// <summary>The value of voxel (i, j, k); 0 outside the grid.</summary>
	public float Value(int i, int j, int k) =>
		i < 0 || j < 0 || k < 0 || i >= NX || j >= NY || k >= NZ ? 0f : Values[Index(i, j, k)];

	/// <summary>The center of voxel (i, j, k).</summary>
	public Vector3d Center(int i, int j, int k) =>
		new(Origin.X + (i + 0.5) * VoxelSize, Origin.Y + (j + 0.5) * VoxelSize, Origin.Z + (k + 0.5) * VoxelSize);

	/// <summary>
	/// The trilinear interpolation of the values at <paramref name="world"/> (0 beyond the grid),
	/// so a caller can ask "inside the hull?" as Sample(p) >= 0.5 anywhere.
	/// </summary>
	public double Sample(Vector3d world)
	{
		double fx = (world.X - Origin.X) / VoxelSize - 0.5;
		double fy = (world.Y - Origin.Y) / VoxelSize - 0.5;
		double fz = (world.Z - Origin.Z) / VoxelSize - 0.5;
		int i = (int)Math.Floor(fx), j = (int)Math.Floor(fy), k = (int)Math.Floor(fz);
		double tx = fx - i, ty = fy - j, tz = fz - k;
		double c00 = Value(i, j, k) * (1 - tx) + Value(i + 1, j, k) * tx;
		double c10 = Value(i, j + 1, k) * (1 - tx) + Value(i + 1, j + 1, k) * tx;
		double c01 = Value(i, j, k + 1) * (1 - tx) + Value(i + 1, j, k + 1) * tx;
		double c11 = Value(i, j + 1, k + 1) * (1 - tx) + Value(i + 1, j + 1, k + 1) * tx;
		double c0 = c00 * (1 - ty) + c10 * ty;
		double c1 = c01 * (1 - ty) + c11 * ty;
		return c0 * (1 - tz) + c1 * tz;
	}

	/// <summary>
	/// A copy after <paramref name="passes"/> passes of the separable [1 2 1]/4 filter along each
	/// axis, with zeros beyond the grid (the hull is empty there).
	/// </summary>
	public OccupancyGrid Smoothed(int passes)
	{
		Check.That(passes >= 0);
		var result = new OccupancyGrid(Origin, VoxelSize, NX, NY, NZ);
		Array.Copy(Values, result.Values, Values.Length);
		var scratch = new float[Values.Length];
		for (int pass = 0; pass < passes; pass++)
		{
			for (int axis = 0; axis < 3; axis++)
			{
				result.FilterAxis(axis, scratch);
				Array.Copy(scratch, result.Values, scratch.Length);
			}
		}

		return result;
	}

	private void FilterAxis(int axis, float[] output)
	{
		int step = axis == 0 ? 1 : axis == 1 ? NX : NX * NY;
		int length = axis == 0 ? NX : axis == 1 ? NY : NZ;
		for (int k = 0; k < NZ; k++)
		{
			for (int j = 0; j < NY; j++)
			{
				for (int i = 0; i < NX; i++)
				{
					int index = Index(i, j, k);
					int position = axis == 0 ? i : axis == 1 ? j : k;
					float before = position > 0 ? Values[index - step] : 0f;
					float after = position < length - 1 ? Values[index + step] : 0f;
					output[index] = (before + 2 * Values[index] + after) * 0.25f;
				}
			}
		}
	}
}
