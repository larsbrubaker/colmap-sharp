// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// HullMaskView: one silhouette view prepared for carving (docs/QUALITY_PLAN.md, stage 3a). Not a
// COLMAP port. It holds the binary mask, its summed-area table (integral image) and the camera,
// and answers the two questions VisualHull.cs asks: how does a whole axis-aligned cell sit
// against the mask (Szeliski 1993's conservative footprint test), and what is the mask's value at
// one point (for the finest level). See VisualHullOptions.cs for the references.
//
// Pixel convention: COLMAP image coordinates, pixel (x, y) covering [x, x+1) x [y, y+1) with its
// center at (x + 0.5, y + 0.5). The point value is the bilinear interpolation of the binary mask
// between pixel centers (edge-clamped), so the 0.5 crossing of the occupancy falls on the
// silhouette edge to sub-pixel precision rather than on voxel centers.
//
// Footprint: the bounding rectangle of the projected cell corners, grown to the pixels bilinear
// interpolation at any point inside it would read. For a pinhole camera the projection of a
// convex cell lies inside the hull of its projected corners, so the test is conservative; with
// lens distortion the edges bow slightly, which the one-pixel growth covers for the mild
// distortion of real cameras.
//
// Off the image: by default a view does not constrain what it cannot see (Unseen). With
// VisualHullOptions.OffImageIsOutside the area off the image counts as background instead, so a
// cell wholly off the image is Outside, one partly off is Outside when its on-image part is, and
// never Inside.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Silhouette;

/// <summary>How a cell sits against one view's mask.</summary>
internal enum CellFootprint
{
	/// <summary>Everything in the cell is outside the mask, and the view sees all of it.</summary>
	Outside,

	/// <summary>Everything in the cell is inside the mask, and the view sees all of it.</summary>
	Inside,

	/// <summary>
	/// The view sees none of the cell: behind the camera, or off the image unless off-image counts
	/// as outside (VisualHullOptions.OffImageIsOutside).
	/// </summary>
	Unseen,

	/// <summary>Anything else: the cell straddles the silhouette, the image edge or the camera.</summary>
	Mixed,
}

/// <summary>A silhouette view with its binary mask and summed-area table.</summary>
internal sealed class HullMaskView
{
	private readonly Camera camera;
	private readonly Rigid3d camFromWorld;
	private readonly bool offImageIsOutside;
	private readonly int width;
	private readonly int height;
	private readonly byte[] binary;

	// integral[(y) * (width + 1) + x] = number of object pixels in [0, x) x [0, y).
	private readonly int[] integral;

	public HullMaskView(VisualHullView view, bool offImageIsOutside)
	{
		Check.NotNull(view);
		this.offImageIsOutside = offImageIsOutside;
		camera = view.Camera;
		camFromWorld = view.CamFromWorld;
		width = view.Mask.Width;
		height = view.Mask.Height;
		Check.That(width == camera.Width && height == camera.Height, "The mask must be the camera's size");
		byte[] data = view.Mask.RowMajorData;
		int channels = view.Mask.Channels;
		binary = new byte[width * height];
		integral = new int[(width + 1) * (height + 1)];
		for (int y = 0; y < height; y++)
		{
			int rowSum = 0;
			for (int x = 0; x < width; x++)
			{
				byte bit = data[(y * width + x) * channels] >= 128 ? (byte)1 : (byte)0;
				binary[y * width + x] = bit;
				rowSum += bit;
				integral[(y + 1) * (width + 1) + x + 1] = integral[y * (width + 1) + x + 1] + rowSum;
			}
		}
	}

	/// <summary>
	/// Classifies the axis-aligned cell [min, max] against this view's mask. Conservative: Outside
	/// and Inside hold for every point of the cell, and Unseen means <see cref="PointValue"/>
	/// returns null for every point of it.
	/// </summary>
	public CellFootprint Classify(Vector3d min, Vector3d max)
	{
		double minU = double.PositiveInfinity, minV = double.PositiveInfinity;
		double maxU = double.NegativeInfinity, maxV = double.NegativeInfinity;
		int behind = 0;
		for (int corner = 0; corner < 8; corner++)
		{
			var world = new Vector3d(
				(corner & 1) == 0 ? min.X : max.X,
				(corner & 2) == 0 ? min.Y : max.Y,
				(corner & 4) == 0 ? min.Z : max.Z);
			if (Project(world) is not Vector2d uv)
			{
				behind++;
				continue;
			}

			minU = Math.Min(minU, uv.X);
			maxU = Math.Max(maxU, uv.X);
			minV = Math.Min(minV, uv.Y);
			maxV = Math.Max(maxV, uv.Y);
		}

		if (behind == 8)
		{
			return CellFootprint.Unseen;
		}

		if (behind > 0)
		{
			// Straddles the camera plane: its projection is unbounded.
			return CellFootprint.Mixed;
		}

		if (maxU < 0 || maxV < 0 || minU >= width || minV >= height)
		{
			return offImageIsOutside ? CellFootprint.Outside : CellFootprint.Unseen;
		}

		bool partlyOff = minU < 0 || minV < 0 || maxU >= width || maxV >= height;
		if (partlyOff && !offImageIsOutside)
		{
			return CellFootprint.Mixed;
		}

		// The pixels bilinear interpolation reads for any point in the rectangle, grown by one
		// more pixel for lens distortion. Clamped, as the interpolation clamps.
		int x0 = Math.Max(0, (int)Math.Floor(minU - 0.5) - 1);
		int y0 = Math.Max(0, (int)Math.Floor(minV - 0.5) - 1);
		int x1 = Math.Min(width - 1, (int)Math.Floor(maxU - 0.5) + 2);
		int y1 = Math.Min(height - 1, (int)Math.Floor(maxV - 0.5) + 2);
		int stride = width + 1;
		int count = integral[(y1 + 1) * stride + x1 + 1] - integral[y0 * stride + x1 + 1]
			- integral[(y1 + 1) * stride + x0] + integral[y0 * stride + x0];
		if (count == 0)
		{
			return CellFootprint.Outside;
		}

		// Partly off the image, only Outside can be certain: the off-image part counts as outside.
		return !partlyOff && count == (x1 - x0 + 1) * (y1 - y0 + 1) ? CellFootprint.Inside : CellFootprint.Mixed;
	}

	/// <summary>
	/// The bilinear mask value in [0, 1] at <paramref name="world"/>, or null where the view does
	/// not see the point: behind the camera, or off the image unless off-image counts as outside
	/// (then 0).
	/// </summary>
	public double? PointValue(Vector3d world)
	{
		if (Project(world) is not Vector2d uv)
		{
			return null;
		}

		if (uv.X < 0 || uv.Y < 0 || uv.X >= width || uv.Y >= height)
		{
			return offImageIsOutside ? 0 : null;
		}

		double fx = uv.X - 0.5, fy = uv.Y - 0.5;
		int ix = (int)Math.Floor(fx), iy = (int)Math.Floor(fy);
		double tx = fx - ix, ty = fy - iy;
		int xa = Math.Clamp(ix, 0, width - 1), xb = Math.Clamp(ix + 1, 0, width - 1);
		int ya = Math.Clamp(iy, 0, height - 1), yb = Math.Clamp(iy + 1, 0, height - 1);
		double top = binary[ya * width + xa] * (1 - tx) + binary[ya * width + xb] * tx;
		double bottom = binary[yb * width + xa] * (1 - tx) + binary[yb * width + xb] * tx;
		return top * (1 - ty) + bottom * ty;
	}

	private Vector2d? Project(Vector3d world) => camera.ImgFromCam(camFromWorld * world);
}
