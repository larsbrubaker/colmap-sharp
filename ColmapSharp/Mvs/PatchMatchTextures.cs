// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatchTextures: the source-image inputs of the CPU PatchMatch kernel, replacing the
// layered CUDA textures PatchMatchCuda::InitSourceImages builds (patch_match_cuda.cu,
// cuda_texture.h). Every source image (or depth map) is copied into a layer of
// maxWidth x maxHeight, padded with 0, as COLMAP does; PatchMatchTransforms.cs holds the
// matching pose tables, and PatchMatchRefImage.cs the reference image.
// Tests: ColmapSharp.Tests/Mvs/PatchMatchInputsTests.cs (C#-only).
//
// Sampling follows CUDA's texture rules for unnormalized coordinates with border
// addressing: texel (i, j) covers [i, i + 1) x [j, j + 1), so its centre is at +0.5, and
// every texel outside the layer reads 0.
// - Source images: linear filtering, read as normalized float (byte / 255). NVIDIA hardware
//   interpolates with 9-bit fixed-point weights (8 fractional bits); the CPU port
//   interpolates with exact float weights, the formula of COLMAP's own gfx9 emulation,
//   SampleLayeredBilinear (docs/CPP_DIVERGENCES.md, entry 95).
// - Source depth maps: point filtering, read as the stored float.
// Both are read-only after construction, so any number of threads can sample them.
// Performance: byte / 255 comes from a 256-entry table (the same float values), samples
// whose four texels are all inside the layer skip the per-texel border checks, and Sample4
// blends four samples as Vector128 lanes, each bit-identical to Sample.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace ColmapSharp.Mvs;

/// <summary>The source images of a PatchMatch problem as bilinearly sampled grey layers.</summary>
public sealed class PatchMatchSourceImages
{
	// byte / 255.0f for every byte: the same values as the division, looked up.
	private static readonly float[] ByteToUnit = CreateByteToUnit();

	private readonly byte[] data;

	private static float[] CreateByteToUnit()
	{
		var table = new float[256];
		for (int i = 0; i < 256; ++i)
		{
			table[i] = i / 255.0f;
		}

		return table;
	}

	/// <summary>
	/// Layers of the given images' bitmaps (grey, row-major <see cref="Sensor.Bitmap"/>
	/// data), each padded with 0 to the largest width and height. Pass only the problem's
	/// source images, in the problem's source order: layer i is source i, the row the
	/// kernel's pose tables and cost map use for it.
	/// </summary>
	public PatchMatchSourceImages(IReadOnlyList<Image> images)
	{
		foreach (Image image in images)
		{
			MaxWidth = Math.Max(MaxWidth, image.GetWidth());
			MaxHeight = Math.Max(MaxHeight, image.GetHeight());
		}

		NumLayers = images.Count;
		data = new byte[checked(MaxWidth * MaxHeight * NumLayers)];
		for (int i = 0; i < images.Count; ++i)
		{
			Image image = images[i];
			Util.Check.That(image.GetBitmap().IsGrey);
			byte[] src = image.GetBitmap().RowMajorData;
			int width = image.GetWidth();
			for (int r = 0; r < image.GetHeight(); ++r)
			{
				Array.Copy(src, r * width, data, (i * MaxHeight + r) * MaxWidth, width);
			}
		}
	}

	/// <summary>
	/// Layers already packed: <paramref name="numLayers"/> layers of
	/// <paramref name="maxWidth"/> x <paramref name="maxHeight"/> bytes, laid out as the
	/// public constructor lays them out (the GPU's source image buffer). Takes the array.
	/// </summary>
	internal PatchMatchSourceImages(byte[] data, int maxWidth, int maxHeight, int numLayers)
	{
		Util.Check.Eq(data.Length, checked(maxWidth * maxHeight * numLayers));
		this.data = data;
		MaxWidth = maxWidth;
		MaxHeight = maxHeight;
		NumLayers = numLayers;
	}

	/// <summary>byte / 255.0f for every byte: the table the GPU reads as a uniform.</summary>
	internal static ReadOnlySpan<float> ByteToUnitTable => ByteToUnit;

	/// <summary>The packed layers (see the internal constructor); not a copy.</summary>
	internal ReadOnlySpan<byte> Data => data;

	/// <summary>Layer width (the largest source image width).</summary>
	public int MaxWidth { get; }

	/// <summary>Layer height (the largest source image height).</summary>
	public int MaxHeight { get; }

	/// <summary>Number of layers (source images).</summary>
	public int NumLayers { get; }

	/// <summary>
	/// The layers' bytes, layer l row r column c at (l MaxHeight + r) MaxWidth + c: what the GPU
	/// path (PatchMatchGpu.cs) uploads as the source image buffer.
	/// </summary>
	internal ReadOnlySpan<byte> RawData => data;

	/// <summary>byte / 255.0f for every byte (256 floats), the GPU path's byte table uniform.</summary>
	internal static ReadOnlySpan<float> ByteToUnitTable => ByteToUnit;

	/// <summary>
	/// The bilinearly interpolated value in [0, 1] at unnormalized texture coordinates
	/// (<paramref name="x"/>, <paramref name="y"/>) of <paramref name="layer"/>, as
	/// tex2DLayered with linear filtering, border addressing and normalized-float reads.
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public float Sample(float x, float y, int layer)
	{
		float px = x - 0.5f;
		float py = y - 0.5f;
		float fx = MathF.Floor(px);
		float fy = MathF.Floor(py);

		// Far outside the layer (or NaN) every texel is border: 0. This also keeps the int
		// conversions below in range.
		if (!(fx >= -1.0f && fx < MaxWidth && fy >= -1.0f && fy < MaxHeight))
		{
			return 0.0f;
		}

		float wx = px - fx;
		float wy = py - fy;
		int ix = (int)fx;
		int iy = (int)fy;
		int layerOffset = layer * MaxHeight;
		float c00, c10, c01, c11;
		if (ix >= 0 && ix + 1 < MaxWidth && iy >= 0 && iy + 1 < MaxHeight && (uint)layer < (uint)NumLayers)
		{
			// All four texels inside the layer (the common case): read them without the
			// per-texel border and bounds checks. The indices were just range-checked.
			ref byte texel = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(data), (layerOffset + iy) * MaxWidth + ix);
			ref float unit = ref MemoryMarshal.GetArrayDataReference(ByteToUnit);
			c00 = Unsafe.Add(ref unit, texel);
			c10 = Unsafe.Add(ref unit, Unsafe.Add(ref texel, 1));
			c01 = Unsafe.Add(ref unit, Unsafe.Add(ref texel, MaxWidth));
			c11 = Unsafe.Add(ref unit, Unsafe.Add(ref texel, MaxWidth + 1));
		}
		else
		{
			return SampleAtBorder(ix, iy, wx, wy, layerOffset);
		}

		return (c00 * (1.0f - wx) + c10 * wx) * (1.0f - wy) + (c01 * (1.0f - wx) + c11 * wx) * wy;
	}

	/// <summary>
	/// Four <see cref="Sample"/>s at once, one per lane, each bit-identical to the scalar
	/// call: when all four fall fully inside the layer, the texel-centre offset, floor,
	/// weights and blend run as Vector128 lanes in Sample's operation order (the texels are
	/// fetched per lane); otherwise each lane calls Sample.
	/// </summary>
	internal Vector128<float> Sample4(Vector128<float> x, Vector128<float> y, int layer)
	{
		Vector128<float> half = Vector128.Create(0.5f);
		Vector128<float> px = x - half;
		Vector128<float> py = y - half;
		Vector128<float> fx = Vector128.Floor(px);
		Vector128<float> fy = Vector128.Floor(py);

		// ix >= 0 && ix + 1 < MaxWidth (and the same for y) in every lane; NaN fails.
		bool inside = (uint)layer < (uint)NumLayers
			&& Vector128.GreaterThanOrEqualAll(fx, Vector128<float>.Zero)
			&& Vector128.LessThanAll(fx, Vector128.Create((float)(MaxWidth - 1)))
			&& Vector128.GreaterThanOrEqualAll(fy, Vector128<float>.Zero)
			&& Vector128.LessThanAll(fy, Vector128.Create((float)(MaxHeight - 1)));
		if (!inside)
		{
			return Vector128.Create(
				Sample(x.GetElement(0), y.GetElement(0), layer),
				Sample(x.GetElement(1), y.GetElement(1), layer),
				Sample(x.GetElement(2), y.GetElement(2), layer),
				Sample(x.GetElement(3), y.GetElement(3), layer));
		}

		CountVectorBlend();
		Vector128<float> wx = px - fx;
		Vector128<float> wy = py - fy;
		Vector128<int> ix = Vector128.ConvertToInt32(fx);
		Vector128<int> iy = Vector128.ConvertToInt32(fy);
		int layerOffset = layer * MaxHeight;
		ref byte data0 = ref MemoryMarshal.GetArrayDataReference(data);
		ref float unit = ref MemoryMarshal.GetArrayDataReference(ByteToUnit);
		int w = MaxWidth;
		ref byte t0 = ref Unsafe.Add(ref data0, (layerOffset + iy.GetElement(0)) * w + ix.GetElement(0));
		ref byte t1 = ref Unsafe.Add(ref data0, (layerOffset + iy.GetElement(1)) * w + ix.GetElement(1));
		ref byte t2 = ref Unsafe.Add(ref data0, (layerOffset + iy.GetElement(2)) * w + ix.GetElement(2));
		ref byte t3 = ref Unsafe.Add(ref data0, (layerOffset + iy.GetElement(3)) * w + ix.GetElement(3));
		Vector128<float> c00 = Vector128.Create(
			Unsafe.Add(ref unit, t0), Unsafe.Add(ref unit, t1), Unsafe.Add(ref unit, t2), Unsafe.Add(ref unit, t3));
		Vector128<float> c10 = Vector128.Create(
			Unsafe.Add(ref unit, Unsafe.Add(ref t0, 1)), Unsafe.Add(ref unit, Unsafe.Add(ref t1, 1)),
			Unsafe.Add(ref unit, Unsafe.Add(ref t2, 1)), Unsafe.Add(ref unit, Unsafe.Add(ref t3, 1)));
		Vector128<float> c01 = Vector128.Create(
			Unsafe.Add(ref unit, Unsafe.Add(ref t0, w)), Unsafe.Add(ref unit, Unsafe.Add(ref t1, w)),
			Unsafe.Add(ref unit, Unsafe.Add(ref t2, w)), Unsafe.Add(ref unit, Unsafe.Add(ref t3, w)));
		Vector128<float> c11 = Vector128.Create(
			Unsafe.Add(ref unit, Unsafe.Add(ref t0, w + 1)), Unsafe.Add(ref unit, Unsafe.Add(ref t1, w + 1)),
			Unsafe.Add(ref unit, Unsafe.Add(ref t2, w + 1)), Unsafe.Add(ref unit, Unsafe.Add(ref t3, w + 1)));
		Vector128<float> one = Vector128.Create(1.0f);
		return (c00 * (one - wx) + c10 * wx) * (one - wy) + (c01 * (one - wx) + c11 * wx) * wy;
	}

	/// <summary>
	/// Debug builds only, and only while <see cref="TrackVectorBlends"/> is set: how many
	/// Sample4 calls took the all-lanes-interior vector blend, so the bit-identity test can
	/// show it exercised that path. Not thread safe; only read by single-threaded tests.
	/// </summary>
	internal long VectorBlendCount { get; private set; }

	/// <summary>
	/// Turns on <see cref="VectorBlendCount"/>. Off by default because every sweep thread
	/// shares this instance: unconditional increments from all of them fight over one cache
	/// line and made a multi-threaded Debug PatchMatch run 3-4x slower than Release.
	/// </summary>
	internal bool TrackVectorBlends { get; set; }

	[System.Diagnostics.Conditional("DEBUG")]
	private void CountVectorBlend()
	{
		if (TrackVectorBlends)
		{
			VectorBlendCount++;
		}
	}

	// The bilinear blend when some of the four texels are outside the layer (read as 0);
	// kept out of line so the common path above stays small enough to inline.
	[MethodImpl(MethodImplOptions.NoInlining)]
	private float SampleAtBorder(int ix, int iy, float wx, float wy, int layerOffset)
	{
		float c00 = Texel(ix, iy, layerOffset);
		float c10 = Texel(ix + 1, iy, layerOffset);
		float c01 = Texel(ix, iy + 1, layerOffset);
		float c11 = Texel(ix + 1, iy + 1, layerOffset);
		return (c00 * (1.0f - wx) + c10 * wx) * (1.0f - wy) + (c01 * (1.0f - wx) + c11 * wx) * wy;
	}

	private float Texel(int col, int row, int layerOffset)
	{
		if ((uint)col >= (uint)MaxWidth || (uint)row >= (uint)MaxHeight)
		{
			return 0.0f;
		}

		return ByteToUnit[data[(layerOffset + row) * MaxWidth + col]];
	}
}

/// <summary>The source depth maps of a PatchMatch problem as point-sampled layers.</summary>
public sealed class PatchMatchSourceDepthMaps
{
	private readonly float[] data;

	/// <summary>
	/// Layers of the given depth maps, each padded with 0 to
	/// <paramref name="maxWidth"/> x <paramref name="maxHeight"/> (the source images' largest
	/// size, as in COLMAP).
	/// </summary>
	public PatchMatchSourceDepthMaps(IReadOnlyList<DepthMap> depthMaps, int maxWidth, int maxHeight)
	{
		MaxWidth = maxWidth;
		MaxHeight = maxHeight;
		NumLayers = depthMaps.Count;
		data = new float[checked(maxWidth * maxHeight * NumLayers)];
		for (int i = 0; i < depthMaps.Count; ++i)
		{
			DepthMap depthMap = depthMaps[i];
			int width = depthMap.GetWidth();
			Util.Check.Le(width, maxWidth);
			Util.Check.Le(depthMap.GetHeight(), maxHeight);
			for (int r = 0; r < depthMap.GetHeight(); ++r)
			{
				Array.Copy(depthMap.Data, r * width, data, (i * maxHeight + r) * maxWidth, width);
			}
		}
	}

	/// <summary>
	/// Layers already packed: <paramref name="numLayers"/> layers of
	/// <paramref name="maxWidth"/> x <paramref name="maxHeight"/> floats (the GPU's source
	/// depth buffer). Takes the array.
	/// </summary>
	internal PatchMatchSourceDepthMaps(float[] data, int maxWidth, int maxHeight, int numLayers)
	{
		Util.Check.Eq(data.Length, checked(maxWidth * maxHeight * numLayers));
		this.data = data;
		MaxWidth = maxWidth;
		MaxHeight = maxHeight;
		NumLayers = numLayers;
	}

	/// <summary>The packed layers (see the internal constructor); not a copy.</summary>
	internal ReadOnlySpan<float> Data => data;

	/// <summary>Layer width.</summary>
	public int MaxWidth { get; }

	/// <summary>Layer height.</summary>
	public int MaxHeight { get; }

	/// <summary>Number of layers (source depth maps).</summary>
	public int NumLayers { get; }

	/// <summary>
	/// The layers' depths in the same layout as <see cref="PatchMatchSourceImages.RawData"/>: what
	/// the GPU path (PatchMatchGpu.cs) uploads as the source depth buffer.
	/// </summary>
	internal ReadOnlySpan<float> RawData => data;

	/// <summary>
	/// The depth of the texel containing unnormalized coordinates (<paramref name="x"/>,
	/// <paramref name="y"/>) of <paramref name="layer"/>, or 0 outside it, as tex2DLayered
	/// with point filtering and border addressing.
	/// </summary>
	public float Sample(float x, float y, int layer)
	{
		float fx = MathF.Floor(x);
		float fy = MathF.Floor(y);
		if (!(fx >= 0.0f && fx < MaxWidth && fy >= 0.0f && fy < MaxHeight))
		{
			return 0.0f;
		}

		return data[(layer * MaxHeight + (int)fy) * MaxWidth + (int)fx];
	}
}
