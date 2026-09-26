// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// JetColormap: the JetColormap class of colmap/sensor/bitmap.h/.cc, used by the MVS depth
// and normal map visualizations. Next to Bitmap (Bitmap.cs) because COLMAP declares it
// there. bitmap_test.cc has no case for it.
//
// Tier A: single-precision arithmetic in COLMAP's order.

namespace ColmapSharp.Sensor;

/// <summary>
/// Port of colmap::JetColormap: Jet colormap inspired by Matlab. Grayvalues are expected in
/// the range [0, 1] and are converted to RGB values in the same range.
/// </summary>
public static class JetColormap
{
	/// <summary>Red component for a gray value in [0, 1].</summary>
	public static float Red(float gray) => Base(gray - 0.25f);

	/// <summary>Green component for a gray value in [0, 1].</summary>
	public static float Green(float gray) => Base(gray);

	/// <summary>Blue component for a gray value in [0, 1].</summary>
	public static float Blue(float gray) => Base(gray + 0.25f);

	private static float Base(float val)
	{
		if (val <= 0.125f)
		{
			return 0.0f;
		}
		else if (val <= 0.375f)
		{
			return Interpolate(2.0f * val - 1.0f, 0.0f, -0.75f, 1.0f, -0.25f);
		}
		else if (val <= 0.625f)
		{
			return 1.0f;
		}
		else if (val <= 0.87f)
		{
			return Interpolate(2.0f * val - 1.0f, 1.0f, 0.25f, 0.0f, 0.75f);
		}
		else
		{
			return 0.0f;
		}
	}

	private static float Interpolate(float val, float y0, float x0, float y1, float x1)
	{
		return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
	}
}
