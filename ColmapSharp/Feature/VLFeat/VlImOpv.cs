// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from VLFeat (BSD-2-Clause, see THIRD_PARTY_NOTICES.md) as vendored by COLMAP.
//
// VlImOpv: vl_imconvcol_vf from thirdparty/VLFeat/imopv.c - the separable column
// convolution SIFT's Gaussian smoothing runs twice per level (VlSiftFilter.cs, Smooth).
// Only the float instantiation is ported; it is the only one COLMAP calls. VlImOpv.Smooth.cs
// has vl_imsmooth_f and the image gradients the covariant detector (VlCovDet*.cs) uses.
//
// Tier A (exact). This is the scalar (non-SSE2) path, which is what COLMAP compiles on arm64.
// Each output column is one sequential float accumulation in VLFeat's order, so running the
// columns in parallel (below) gives bit-identical results to running them in order.

namespace ColmapSharp.Feature.VLFeat;

/// <summary>Port of the float image operations of VLFeat's imopv.c that SIFT uses.</summary>
public static partial class VlImOpv
{
	/// <summary>VL_PAD_BY_ZERO.</summary>
	public const int PadByZero = 0x0;

	/// <summary>VL_PAD_BY_CONTINUITY.</summary>
	public const int PadByContinuity = 0x1;

	/// <summary>VL_PAD_MASK.</summary>
	public const int PadMask = 0x3;

	/// <summary>VL_TRANSPOSE.</summary>
	public const int Transpose = 0x1 << 2;

	// Below this many multiply-adds a convolution runs on the calling thread; the task
	// overhead would dominate. The result does not depend on it.
	private const long ParallelWorkThreshold = 1 << 18;

	/// <summary>
	/// Port of vl_imconvcol_vf: convolves each column of <paramref name="src"/> with the filter
	/// <c>filt[filtOffset .. filtOffset + filtEnd - filtBegin]</c> (support
	/// [filtBegin, filtEnd]), subsampling rows by <paramref name="step"/>. With
	/// <see cref="Transpose"/> column x of the input becomes row x of the output.
	/// </summary>
	public static void ConvColVF(
		float[] dst,
		int dstOffset,
		int dstStride,
		float[] src,
		int srcOffset,
		int srcWidth,
		int srcHeight,
		int srcStride,
		float[] filt,
		int filtOffset,
		int filtBegin,
		int filtEnd,
		int step,
		int flags)
	{
		int dheight = ((srcHeight - 1) / step) + 1;
		bool transp = (flags & Transpose) != 0;
		bool zeropad = (flags & PadMask) == PadByZero;

		// Where column x starts in dst: VLFeat walks dst with a pointer and rewinds it after
		// each column; the closed form lets the columns run independently.
		int dstColumnStep = transp ? dstStride : 1;
		int dstRowStep = transp ? 1 : dstStride;

		void Column(int x)
		{
			// Let filt point to the last sample of the filter.
			int filtLast = filtOffset + (filtEnd - filtBegin);
			int d = dstOffset + (x * dstColumnStep);
			for (int y = 0; y < srcHeight; y += step)
			{
				float acc = 0;
				float v = 0;
				float c;
				int filti = filtLast;
				int stop = filtEnd - y;
				int srci = srcOffset + x - (stop * srcStride);

				// CHUNK_A: samples above the image, padded.
				if (stop > 0)
				{
					v = zeropad ? 0 : src[srcOffset + x];
					while (filti > filtLast - stop)
					{
						c = filt[filti--];
						acc += v * c;
						srci += srcStride;
					}
				}

				// CHUNK_B: samples inside the image.
				stop = filtEnd - Math.Max(filtBegin, y - srcHeight + 1) + 1;
				while (filti > filtLast - stop)
				{
					v = src[srci];
					c = filt[filti--];
					acc += v * c;
					srci += srcStride;
				}

				// CHUNK_C: samples below the image, padded with the last value (or zero).
				if (zeropad)
				{
					v = 0;
				}

				stop = filtEnd - filtBegin + 1;
				while (filti > filtLast - stop)
				{
					c = filt[filti--];
					acc += v * c;
				}

				dst[d] = acc;
				d += dstRowStep;
			}
		}

		long work = (long)srcWidth * dheight * (filtEnd - filtBegin + 1);
		if (work < ParallelWorkThreshold)
		{
			for (int x = 0; x < srcWidth; ++x)
			{
				Column(x);
			}
		}
		else
		{
			Parallel.For(0, srcWidth, Column);
		}
	}
}
