// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// DepthMap: colmap/mvs/depth_map.h and depth_map.cc - a single-channel Mat<float> of
// per-pixel depths (0 or negative = no depth) with the depth range PatchMatch searched.
// NormalMap.cs is its three-channel sibling; Workspace.cs loads both from .bin files.
// Tests: ColmapSharp.Tests/Mvs/DepthMapTests.cs (depth_map_test.cc 1:1).
//
// Tier A (exact): Rescale goes through ImageProcessing/Warp.Float.cs DownsampleImage
// (exact), and ToBitmap through MathUtils.Percentile and JetColormap in COLMAP's float
// arithmetic.

using ColmapSharp.ImageProcessing;
using ColmapSharp.Mathematics;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>Port of colmap::mvs::DepthMap: a width x height x 1 map of depths.</summary>
public sealed class DepthMap : Mat<float>
{
	private readonly float depthMin;
	private readonly float depthMax;

	/// <summary>An empty depth map with depth range (-1, -1).</summary>
	public DepthMap()
		: this(0, 0, -1.0f, -1.0f)
	{
	}

	/// <summary>A zero-filled width x height depth map with the given depth range.</summary>
	public DepthMap(int width, int height, float depthMin, float depthMax)
		: base(width, height, 1)
	{
		this.depthMin = depthMin;
		this.depthMax = depthMax;
	}

	/// <summary>A depth map with a copy of a single-channel matrix's data.</summary>
	public DepthMap(Mat<float> mat, float depthMin, float depthMax)
		: base(mat.GetWidth(), mat.GetHeight(), mat.GetDepth())
	{
		this.depthMin = depthMin;
		this.depthMax = depthMax;
		Check.Eq(mat.GetDepth(), 1);
		data = (float[])mat.Data.Clone();
	}

	/// <summary>The minimum depth of the search range (-1 if unknown).</summary>
	public float GetDepthMin() => depthMin;

	/// <summary>The maximum depth of the search range (-1 if unknown).</summary>
	public float GetDepthMax() => depthMax;

	/// <summary>
	/// Resamples to round(width * factor) x round(height * factor) (factor at most 1) with
	/// Gaussian anti-aliasing; an empty map is left unchanged.
	/// </summary>
	public void Rescale(float factor)
	{
		if (width * height == 0)
		{
			return;
		}

		int newWidth = (int)MathF.Round(width * factor, MidpointRounding.AwayFromZero);
		int newHeight = (int)MathF.Round(height * factor, MidpointRounding.AwayFromZero);
		float[] newData = new float[newWidth * newHeight];
		Warp.DownsampleImage(data, height, width, newHeight, newWidth, newData);

		data = newData;
		width = newWidth;
		height = newHeight;
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
	/// An RGB jet-colormap visualization: depths are clamped to the given percentiles
	/// (0..100) of the valid (positive) depths; pixels without depth are black.
	/// </summary>
	public Bitmap ToBitmap(float minPercentile, float maxPercentile)
	{
		Check.Gt(width, 0);
		Check.Gt(height, 0);

		var bitmap = new Bitmap(width, height, asRgb: true);

		var validDepths = new List<float>(data.Length);
		foreach (float depth in data)
		{
			if (depth > 0)
			{
				validDepths.Add(depth);
			}
		}

		if (validDepths.Count == 0)
		{
			bitmap.Fill(new BitmapColor<byte>(0));
			return bitmap;
		}

		Span<float> valid = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(validDepths);
		float robustDepthMin = (float)MathUtils.Percentile(valid, minPercentile);
		float robustDepthMax = (float)MathUtils.Percentile(valid, maxPercentile);

		float robustDepthRange = robustDepthMax - robustDepthMin;
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				float depth = Get(y, x);
				if (depth > 0)
				{
					// std::max(min, std::min(max, depth)).
					float clampedHigh = depth < robustDepthMax ? depth : robustDepthMax;
					float robustDepth = robustDepthMin < clampedHigh ? clampedHigh : robustDepthMin;
					float gray = (robustDepth - robustDepthMin) / robustDepthRange;
					var color = new BitmapColor<float>(
						255 * JetColormap.Red(gray),
						255 * JetColormap.Green(gray),
						255 * JetColormap.Blue(gray));
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
