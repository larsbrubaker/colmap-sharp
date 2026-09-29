// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TextureMapping.Atlas: the texture atlas steps of colmap/mvs/texture_mapping.cc -
// ExtractFaceRegions, ComputeRegionProjections, ScaleRegionProjections, PackAtlas,
// ComputeFaceUVs, Barycentric, ComputeAtlasVerts, BakeTexture and InpaintAtlas. View
// selection is in TextureMapping.Views.cs, the color correction that runs between baking
// and inpainting in TextureMapping.ColorCorrection.cs.
//
// Baking stays sequential, as in COLMAP: with zero padding, neighboring patches' one-pixel
// seam borders can overlap, and the later face must win. Region projections are computed in
// parallel (each region writes its own slot).

using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

public static partial class TextureMapping
{
	/// <summary>Port of FaceRegion: a connected component of faces sharing the same view.</summary>
	internal sealed class FaceRegion
	{
		public int ViewId { get; init; } = -1;

		public List<int> FaceIds { get; } = [];
	}

	/// <summary>Port of RegionProjection: projected face corners and their integer bounding box.</summary>
	internal sealed class RegionProjection
	{
		// Six floats per face in the region: x0, y0, x1, y1, x2, y2.
		public float[] FaceProjections { get; set; } = [];

		public int BboxX { get; set; }

		public int BboxY { get; set; }

		public int BboxWidth { get; set; }

		public int BboxHeight { get; set; }

		// Recomputes the bounding box from the projections (shared by
		// ComputeRegionProjections and ScaleRegionProjections).
		public void UpdateBbox()
		{
			float minX = float.MaxValue;
			float minY = float.MaxValue;
			float maxX = float.MinValue;
			float maxY = float.MinValue;
			float[] p = FaceProjections;
			for (int i = 0; i < p.Length; i += 2)
			{
				minX = StdMin(minX, p[i]);
				minY = StdMin(minY, p[i + 1]);
				maxX = StdMax(maxX, p[i]);
				maxY = StdMax(maxY, p[i + 1]);
			}

			BboxX = (int)MathF.Floor(minX);
			BboxY = (int)MathF.Floor(minY);
			BboxWidth = (int)MathF.Ceiling(maxX) - BboxX + 1;
			BboxHeight = (int)MathF.Ceiling(maxY) - BboxY + 1;
		}
	}

	/// <summary>Port of PackRect: an atlas rectangle.</summary>
	internal struct PackRect
	{
		public int X;
		public int Y;
		public int Width;
		public int Height;
		public int RegionIdx;
	}

	/// <summary>Port of AtlasLayout: atlas size and one placement per region.</summary>
	internal sealed class AtlasLayout
	{
		public int AtlasWidth { get; set; }

		public int AtlasHeight { get; set; }

		public PackRect[] Placements { get; set; } = [];
	}

	// std::min / std::max semantics ((b < a) ? b : a and (a < b) ? b : a).
	private static float StdMin(float a, float b) => b < a ? b : a;

	private static float StdMax(float a, float b) => a < b ? b : a;

	/// <summary>Port of ExtractFaceRegions: breadth-first connected components per view.</summary>
	internal static List<FaceRegion> ExtractFaceRegions(int[] viewPerFace, int[] adjacency, int numFaces)
	{
		var visited = new bool[numFaces];
		var regions = new List<FaceRegion>();
		var queue = new int[numFaces];

		for (int fi = 0; fi < numFaces; fi++)
		{
			if (visited[fi] || viewPerFace[fi] < 0)
			{
				continue;
			}

			var region = new FaceRegion { ViewId = viewPerFace[fi] };
			int head = 0;
			int tail = 0;
			queue[tail++] = fi;
			visited[fi] = true;

			while (head < tail)
			{
				int current = queue[head++];
				region.FaceIds.Add(current);
				for (int s = 3 * current; s < 3 * current + 3; s++)
				{
					int ni = adjacency[s];
					if (ni >= 0 && !visited[ni] && viewPerFace[ni] == region.ViewId)
					{
						visited[ni] = true;
						queue[tail++] = ni;
					}
				}
			}

			regions.Add(region);
		}

		return regions;
	}

	/// <summary>Port of ComputeRegionProjections: each region's faces projected into its view.</summary>
	internal static RegionProjection[] ComputeRegionProjections(
		MeshGeometry geometry,
		List<FaceRegion> regions,
		IReadOnlyList<Image> images,
		MeshTextureMappingOptions options,
		CancellationToken cancellationToken)
	{
		var projections = new RegionProjection[regions.Count];
		float[] v = geometry.Vertices;
		int[] f = geometry.Faces;
		var parallelOptions = new ParallelOptions
		{
			MaxDegreeOfParallelism = Threading.GetEffectiveNumThreads(options.NumThreads),
			CancellationToken = cancellationToken,
		};
		Parallel.For(0, regions.Count, parallelOptions, ri =>
		{
			FaceRegion region = regions[ri];
			ReadOnlySpan<float> p = images[region.ViewId].GetP();
			var rp = new RegionProjection { FaceProjections = new float[6 * region.FaceIds.Count] };
			for (int i = 0; i < region.FaceIds.Count; i++)
			{
				int fi = region.FaceIds[i];
				for (int k = 0; k < 3; k++)
				{
					int vi = 3 * f[3 * fi + k];
					ProjectPoint(p, v[vi], v[vi + 1], v[vi + 2], out rp.FaceProjections[6 * i + 2 * k], out rp.FaceProjections[6 * i + 2 * k + 1]);
				}
			}

			rp.UpdateBbox();
			projections[ri] = rp;
		});

		return projections;
	}

	/// <summary>Port of ScaleRegionProjections: scales the projections and recomputes the boxes.</summary>
	internal static void ScaleRegionProjections(RegionProjection[] projections, double scale)
	{
		float sf = (float)scale;
		foreach (RegionProjection rp in projections)
		{
			for (int i = 0; i < rp.FaceProjections.Length; i++)
			{
				rp.FaceProjections[i] *= sf;
			}

			rp.UpdateBbox();
		}
	}

	/// <summary>
	/// Port of PackAtlas: shelf packing of the padded region boxes, tallest first, into the
	/// smallest power-of-two square that fits, then the height shrunk to the used extent.
	/// Equal heights keep region order (COLMAP's std::sort leaves that order unspecified;
	/// divergence 91).
	/// </summary>
	internal static AtlasLayout PackAtlas(RegionProjection[] projections, int padding)
	{
		var layout = new AtlasLayout();
		if (projections.Length == 0)
		{
			return layout;
		}

		int n = projections.Length;
		var rectWidth = new int[n];
		var rectHeight = new int[n];
		var rects = new int[n];
		long totalArea = 0;
		int maxRectWidth = 0;
		for (int i = 0; i < n; i++)
		{
			rectWidth[i] = projections[i].BboxWidth + 2 * padding;
			rectHeight[i] = projections[i].BboxHeight + 2 * padding;
			rects[i] = i;
			totalArea += (long)rectWidth[i] * rectHeight[i];
			maxRectWidth = Math.Max(maxRectWidth, rectWidth[i]);
		}

		Array.Sort(rects, (a, b) =>
		{
			int byHeight = rectHeight[b].CompareTo(rectHeight[a]);
			return byHeight != 0 ? byHeight : a.CompareTo(b);
		});

		int atlasSide = Math.Max((int)Math.Ceiling(Math.Sqrt(totalArea * 1.3)), maxRectWidth);
		int atlasWidth = 1;
		while (atlasWidth < atlasSide)
		{
			atlasWidth *= 2;
		}

		int atlasHeight = atlasWidth;

		var placements = new PackRect[n];
		bool TryPack(int aw, int ah)
		{
			int shelfX = 0;
			int shelfY = 0;
			int shelfHeight = 0;
			for (int i = 0; i < n; i++)
			{
				int r = rects[i];
				if (shelfX + rectWidth[r] > aw)
				{
					shelfY += shelfHeight;
					shelfX = 0;
					shelfHeight = 0;
				}

				if (shelfY + rectHeight[r] > ah)
				{
					return false;
				}

				placements[i] = new PackRect
				{
					X = shelfX + padding,
					Y = shelfY + padding,
					Width = projections[r].BboxWidth,
					Height = projections[r].BboxHeight,
					RegionIdx = r,
				};
				shelfX += rectWidth[r];
				shelfHeight = Math.Max(shelfHeight, rectHeight[r]);
			}

			return true;
		}

		const int kMaxAtlasDim = 1 << 16; // 65536
		while (!TryPack(atlasWidth, atlasHeight))
		{
			Check.Le(atlasWidth, kMaxAtlasDim, $"Atlas dimensions exceeded maximum ({kMaxAtlasDim})");
			atlasWidth *= 2;
			atlasHeight *= 2;
		}

		// Shrink height to actual used extent.
		int maxUsedY = 0;
		foreach (PackRect p in placements)
		{
			maxUsedY = Math.Max(maxUsedY, p.Y + p.Height + padding);
		}

		atlasHeight = Math.Max(maxUsedY, 1);

		// Reorder placements by region index.
		layout.Placements = new PackRect[n];
		foreach (PackRect p in placements)
		{
			layout.Placements[p.RegionIdx] = p;
		}

		layout.AtlasWidth = atlasWidth;
		layout.AtlasHeight = atlasHeight;
		return layout;
	}

	/// <summary>Port of ComputeFaceUVs: atlas UVs per face corner, v flipped so (0,0) is bottom-left.</summary>
	internal static float[] ComputeFaceUVs(List<FaceRegion> regions, RegionProjection[] projections, AtlasLayout layout, int numFaces)
	{
		var uvs = new float[numFaces * 6];
		float invAtlasWidth = 1.0f / layout.AtlasWidth;
		float invAtlasHeight = 1.0f / layout.AtlasHeight;
		for (int ri = 0; ri < regions.Count; ri++)
		{
			FaceRegion region = regions[ri];
			RegionProjection rp = projections[ri];
			PackRect placement = layout.Placements[ri];
			for (int i = 0; i < region.FaceIds.Count; i++)
			{
				int fi = region.FaceIds[i];
				for (int k = 0; k < 3; k++)
				{
					float atlasX = rp.FaceProjections[6 * i + 2 * k] - rp.BboxX + placement.X;
					float atlasY = rp.FaceProjections[6 * i + 2 * k + 1] - rp.BboxY + placement.Y;
					uvs[fi * 6 + k * 2] = atlasX * invAtlasWidth;
					uvs[fi * 6 + k * 2 + 1] = 1.0f - atlasY * invAtlasHeight;
				}
			}
		}

		return uvs;
	}

	/// <summary>Port of Barycentric: (u, v, w) with P = u A + v B + w C, or (-1, -1, -1) when degenerate.</summary>
	internal static (float U, float V, float W) Barycentric(float px, float py, float ax, float ay, float bx, float by, float cx, float cy)
	{
		float v0x = bx - ax, v0y = by - ay;
		float v1x = cx - ax, v1y = cy - ay;
		float v2x = px - ax, v2y = py - ay;
		float d00 = v0x * v0x + v0y * v0y;
		float d01 = v0x * v1x + v0y * v1y;
		float d11 = v1x * v1x + v1y * v1y;
		float d20 = v2x * v0x + v2y * v0y;
		float d21 = v2x * v1x + v2y * v1y;
		float denom = d00 * d11 - d01 * d01;
		if (MathF.Abs(denom) < 1e-10f)
		{
			return (-1, -1, -1);
		}

		float v = (d11 * d20 - d01 * d21) / denom;
		float w = (d00 * d21 - d01 * d20) / denom;
		float u = 1.0f - v - w;
		return (u, v, w);
	}

	// Port of ComputeAtlasVerts: the face's corners in atlas pixels (six floats).
	private static void ComputeAtlasVerts(RegionProjection rp, PackRect placement, int faceInRegion, Span<float> atlasVerts)
	{
		for (int k = 0; k < 3; k++)
		{
			atlasVerts[2 * k] = rp.FaceProjections[6 * faceInRegion + 2 * k] - rp.BboxX + placement.X;
			atlasVerts[2 * k + 1] = rp.FaceProjections[6 * faceInRegion + 2 * k + 1] - rp.BboxY + placement.Y;
		}
	}

	// The atlas pixel box of a face's corners, grown by border pixels and clipped to the atlas.
	private static void AtlasPixelBox(ReadOnlySpan<float> a, int border, int aw, int ah, out int minPx, out int minPy, out int maxPx, out int maxPy)
	{
		minPx = Math.Max(0, (int)MathF.Floor(StdMin(StdMin(a[0], a[2]), a[4])) - border);
		minPy = Math.Max(0, (int)MathF.Floor(StdMin(StdMin(a[1], a[3]), a[5])) - border);
		maxPx = Math.Min(aw - 1, (int)MathF.Ceiling(StdMax(StdMax(a[0], a[2]), a[4])) + border);
		maxPy = Math.Min(ah - 1, (int)MathF.Ceiling(StdMax(StdMax(a[1], a[3]), a[5])) + border);
	}

	/// <summary>
	/// Port of BakeTexture: fills each face's atlas pixels (plus a one-pixel seam border) by
	/// bilinear lookup in its view's bitmap. Returns the baked mask (row-major, atlas size).
	/// </summary>
	internal static bool[] BakeTexture(
		Bitmap atlas,
		List<FaceRegion> regions,
		RegionProjection[] projections,
		AtlasLayout layout,
		IReadOnlyList<Image> images,
		MeshTextureMappingOptions options,
		CancellationToken cancellationToken)
	{
		int aw = layout.AtlasWidth;
		int ah = layout.AtlasHeight;
		var bakedMask = new bool[(long)aw * ah];
		float textureInvScaleFactor = (float)(1.0 / options.TextureScaleFactor);
		Span<float> atlasVerts = stackalloc float[6];

		for (int ri = 0; ri < regions.Count; ri++)
		{
			cancellationToken.ThrowIfCancellationRequested();
			FaceRegion region = regions[ri];
			RegionProjection rp = projections[ri];
			PackRect placement = layout.Placements[ri];
			Bitmap srcBmp = images[region.ViewId].GetBitmap();
			float[] fp = rp.FaceProjections;

			for (int i = 0; i < region.FaceIds.Count; i++)
			{
				ComputeAtlasVerts(rp, placement, i, atlasVerts);

				// Bounding box with 1-pixel border for seam coverage.
				AtlasPixelBox(atlasVerts, 1, aw, ah, out int minPx, out int minPy, out int maxPx, out int maxPy);
				for (int py = minPy; py <= maxPy; py++)
				{
					for (int px = minPx; px <= maxPx; px++)
					{
						(float bu, float bv, float bw) = Barycentric(
							px + 0.5f, py + 0.5f, atlasVerts[0], atlasVerts[1], atlasVerts[2], atlasVerts[3], atlasVerts[4], atlasVerts[5]);
						float minBary = StdMin(StdMin(bu, bv), bw);
						if (minBary < -1e-4f)
						{
							continue;
						}

						int o = 6 * i;
						float imgX = (bu * fp[o] + bv * fp[o + 2] + bw * fp[o + 4]) * textureInvScaleFactor;
						float imgY = (bu * fp[o + 1] + bv * fp[o + 3] + bw * fp[o + 5]) * textureInvScaleFactor;
						BitmapColor<float>? color = srcBmp.InterpolateBilinear(imgX, imgY);
						if (color is null)
						{
							continue;
						}

						atlas.SetPixel(px, py, color.Value.Cast<byte>());
						bakedMask[(long)py * aw + px] = true;
					}
				}
			}
		}

		return bakedMask;
	}

	/// <summary>
	/// Port of InpaintAtlas: breadth-first dilation of the baked pixels' colors into unbaked
	/// pixels up to <paramref name="inpaintRadius"/> steps (4-neighborhood) away.
	/// </summary>
	internal static void InpaintAtlas(Bitmap atlas, bool[] bakedMask, int inpaintRadius)
	{
		int aw = atlas.Width;
		int ah = atlas.Height;
		if (inpaintRadius <= 0 || aw == 0 || ah == 0)
		{
			return;
		}

		int numPixels = checked(aw * ah);
		var dist = new int[numPixels];
		Array.Fill(dist, int.MaxValue);

		// The baked pixel whose color each pixel takes; baked colors do not change here, so
		// storing the source index is the same as COLMAP's copied fill color.
		var fillSource = new int[numPixels];
		var queue = new int[numPixels];
		int head = 0;
		int tail = 0;
		for (int idx = 0; idx < numPixels; idx++)
		{
			if (bakedMask[idx])
			{
				dist[idx] = 0;
				fillSource[idx] = idx;
				queue[tail++] = idx;
			}
		}

		ReadOnlySpan<int> kDx = [-1, 1, 0, 0];
		ReadOnlySpan<int> kDy = [0, 0, -1, 1];
		while (head < tail)
		{
			int cidx = queue[head++];
			int cx = cidx % aw;
			int cy = cidx / aw;
			int cdist = dist[cidx];
			if (cdist >= inpaintRadius)
			{
				continue;
			}

			for (int d = 0; d < 4; d++)
			{
				int nx = cx + kDx[d];
				int ny = cy + kDy[d];
				if (nx < 0 || nx >= aw || ny < 0 || ny >= ah)
				{
					continue;
				}

				int nidx = ny * aw + nx;
				if (dist[nidx] <= cdist + 1)
				{
					continue;
				}

				// Each pixel is reached first at its smallest distance and never again, so the
				// queue holds every pixel at most once.
				dist[nidx] = cdist + 1;
				fillSource[nidx] = fillSource[cidx];
				queue[tail++] = nidx;
			}
		}

		for (int idx = 0; idx < numPixels; idx++)
		{
			if (!bakedMask[idx] && dist[idx] <= inpaintRadius)
			{
				int src = fillSource[idx];
				BitmapColor<byte> color = atlas.GetPixel(src % aw, src / aw) ?? new BitmapColor<byte>(0);
				atlas.SetPixel(idx % aw, idx / aw, color);
			}
		}
	}
}
