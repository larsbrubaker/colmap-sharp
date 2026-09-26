// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// NormalMap: colmap/mvs/normal_map.h and normal_map.cc - per-pixel unit normals stored as
// a width x height x 3 Mat<float> (one plane per component; zero = no normal). DepthMap.cs
// is its single-channel sibling; Workspace.cs loads both from .bin files.
// Tests: ColmapSharp.Tests/Mvs/NormalMapTests.cs (normal_map_test.cc 1:1).
//
// Tier A (exact): each plane is rescaled with ImageProcessing/Warp.Float.cs
// DownsampleImage, then every normal is re-normalized as Eigen does it for a Vector3f:
// squaredNorm summed (x² + y²) + z², then each component divided by the square root.

using ColmapSharp.ImageProcessing;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>Port of colmap::mvs::NormalMap: per-pixel normals as a width x height x 3 map.</summary>
public sealed class NormalMap : Mat<float>
{
	/// <summary>An empty normal map (depth 3).</summary>
	public NormalMap()
		: base(0, 0, 3)
	{
	}

	/// <summary>A zero-filled width x height normal map.</summary>
	public NormalMap(int width, int height)
		: base(width, height, 3)
	{
	}

	/// <summary>A normal map with a copy of a three-channel matrix's data.</summary>
	public NormalMap(Mat<float> mat)
		: base(mat.GetWidth(), mat.GetHeight(), mat.GetDepth())
	{
		Check.Eq(mat.GetDepth(), 3);
		data = (float[])mat.Data.Clone();
	}

	/// <summary>
	/// Resamples to round(width * factor) x round(height * factor) (factor at most 1) and
	/// re-normalizes the normals; an empty map is left unchanged.
	/// </summary>
	public void Rescale(float factor)
	{
		if (width * height == 0)
		{
			return;
		}

		int newWidth = (int)MathF.Round(width * factor, MidpointRounding.AwayFromZero);
		int newHeight = (int)MathF.Round(height * factor, MidpointRounding.AwayFromZero);
		float[] newData = new float[newWidth * newHeight * 3];

		// Resample the normal map.
		for (int d = 0; d < 3; ++d)
		{
			int offset = d * width * height;
			int newOffset = d * newWidth * newHeight;
			Warp.DownsampleImage(
				data, height, width, newHeight, newWidth, newData, offset, newOffset);
		}

		data = newData;
		width = newWidth;
		height = newHeight;

		// Re-normalize the normal vectors.
		for (int r = 0; r < height; ++r)
		{
			for (int c = 0; c < width; ++c)
			{
				float nx = Get(r, c, 0);
				float ny = Get(r, c, 1);
				float nz = Get(r, c, 2);
				float squaredNorm = nx * nx + ny * ny + nz * nz;
				if (squaredNorm > 0)
				{
					float norm = MathF.Sqrt(squaredNorm);
					nx /= norm;
					ny /= norm;
					nz /= norm;
				}

				Set(r, c, 0, nx);
				Set(r, c, 1, ny);
				Set(r, c, 2, nz);
			}
		}
	}

	/// <summary>Shrinks the map to fit within maxWidth x maxHeight, keeping its aspect ratio.</summary>
	public void Downsize(int maxWidth, int maxHeight)
	{
		if (height <= maxHeight && width <= maxWidth)
		{
			return;
		}

		float factorX = maxWidth / (float)width;
		float factorY = maxHeight / (float)height;
		Rescale(Image.MinF(factorX, factorY));
	}

	/// <summary>
	/// An RGB visualization: (127.5 (1 - nx), 127.5 (1 - ny), -255 nz); pixels without a
	/// normal are black.
	/// </summary>
	public Bitmap ToBitmap()
	{
		Check.Gt(width, 0);
		Check.Gt(height, 0);
		Check.Eq(depth, 3);

		var bitmap = new Bitmap(width, height, asRgb: true);

		Span<float> normal = stackalloc float[3];
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				GetSlice(y, x, normal);
				if (normal[0] != 0 || normal[1] != 0 || normal[2] != 0)
				{
					var color = new BitmapColor<float>(
						127.5f * (-normal[0] + 1),
						127.5f * (-normal[1] + 1),
						-255.0f * normal[2]);
					bitmap.SetPixel(x, y, color.Cast<byte>());
				}
				else
				{
					bitmap.SetPixel(x, y, new BitmapColor<byte>(0));
				}
			}
		}

		return bitmap;
	}
}
