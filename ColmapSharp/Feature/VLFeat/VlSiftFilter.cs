// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from VLFeat (BSD-2-Clause, see THIRD_PARTY_NOTICES.md) as vendored by COLMAP.
//
// VlSiftFilter: VLFeat's SIFT filter (thirdparty/VLFeat/sift.h and sift.c), the CPU SIFT
// COLMAP runs. This part holds the filter state and the Gaussian scale space
// (vl_sift_new, vl_sift_process_first_octave / _next_octave and their helpers).
// VlSiftFilter.Detect.cs has the DoG extremum detector, VlSiftFilter.Descriptor.cs the
// gradients, orientations and descriptors. Feature/Sift.cs (COLMAP's SiftCPUFeatureExtractor)
// drives it. Tests: ColmapSharp.Tests/Feature/SiftTests.cs and SiftOracleTests.cs.
//
// Tier A (exact) by construction: pixels are float (vl_sift_pix), filter parameters double,
// and every mixed expression keeps C's promotion points. The oracle comparison and the
// arm64 FMA-contraction caveat are in SiftOracleTests.cs and divergence 41.
//
// Translation notes:
// - VLFeat's pointers into the octave/temp buffers become (array, offset) pairs; the
//   buffers keep VLFeat's sizes and layouts, including the in-place smoothing and the
//   downsample that writes the next octave's base over the current octave's first level.
// - Errors: VLFeat returns VL_ERR_EOF when there is no further octave; here the process
//   methods return false in that case.
// - Cancellation (not in VLFeat): the process methods and Detect take a CancellationToken
//   and check it between scale levels, so a cancelled extraction stops within one level's
//   smoothing of the current octave. The checks never touch the numbers.
// - Deliberate fix (divergence 43): ProcessFirstOctave resets the
//   gradient cache's octave (grad_o), which VLFeat resets only in vl_sift_new, so a filter
//   reused for a new image never computes it from the previous image's gradient.

namespace ColmapSharp.Feature.VLFeat;

/// <summary>Port of VlSiftKeypoint: a detected SIFT keypoint.</summary>
public struct VlSiftKeypoint
{
	/// <summary>Octave index.</summary>
	public int O;

	/// <summary>Integer x in the octave.</summary>
	public int IX;

	/// <summary>Integer y in the octave.</summary>
	public int IY;

	/// <summary>Integer scale level.</summary>
	public int IS;

	/// <summary>x in image coordinates (pixel centers at integers).</summary>
	public float X;

	/// <summary>y in image coordinates.</summary>
	public float Y;

	/// <summary>Fractional scale level.</summary>
	public float S;

	/// <summary>Scale (sigma) in image pixels.</summary>
	public float Sigma;
}

/// <summary>Port of VlSiftFilt and the vl_sift_* functions.</summary>
public sealed partial class VlSiftFilter
{
	private const int Nbo = 8;
	private const int Nbp = 4;

	private readonly double sigman;
	private readonly double sigma0;
	private readonly double sigmak;
	private readonly double dsigma0;

	private readonly float[] temp;
	private readonly float[] octave;
	private readonly float[] dog;
	private readonly float[] grad;

	private float[]? gaussFilter;
	private double gaussFilterSigma;
	private int gaussFilterWidth;

	private VlSiftKeypoint[] keys = [];
	private int gradO;

	/// <summary>
	/// Test diagnostic: called at every cancellation check, just before the token is read, so
	/// a test can cancel at an exact point and count how finely the work is divided.
	/// </summary>
	internal Action? CancellationCheckpoint { get; set; }

	/// <summary>
	/// Port of vl_sift_new. A negative <paramref name="noctaves"/> selects the most octaves
	/// the image size allows.
	/// </summary>
	public VlSiftFilter(int width, int height, int noctaves, int nlevels, int oMin)
	{
		int w = ShiftLeft(width, -oMin);
		int h = ShiftLeft(height, -oMin);
		int nel = w * h;

		if (noctaves < 0)
		{
			noctaves = (int)Math.Max(Math.Floor(Log2(Math.Min(width, height))) - oMin - 3, 1);
		}

		Width = width;
		Height = height;
		NumOctaves = noctaves;
		NumLevels = nlevels;
		OctaveMin = oMin;
		SMin = -1;
		SMax = nlevels + 1;
		CurrentOctave = oMin;

		temp = new float[nel];
		octave = new float[nel * (SMax - SMin + 1)];
		dog = new float[nel * (SMax - SMin)];
		// VLFeat allocates 2 * nel * (s_max - s_min) = 2 * nel * (S + 2) floats, but
		// update_gradient only fills the levels s_min + 1 .. s_max - 2, which is S of them.
		// Sizing for those saves 4 * nel floats (about 490 MB at a 6400 x 4800 first
		// octave) without changing any value.
		grad = new float[nel * 2 * (SMax - 2 - SMin)];

		sigman = 0.5;
		sigmak = Math.Pow(2.0, 1.0 / nlevels);
		sigma0 = 1.6 * sigmak;
		dsigma0 = sigma0 * Math.Sqrt(1.0 - (1.0 / (sigmak * sigmak)));

		gaussFilterSigma = 0;
		gaussFilterWidth = 0;

		PeakThreshold = 0.0;
		EdgeThreshold = 10.0;
		NormThreshold = 0.0;
		Magnif = 3.0;
		WindowSize = Nbp / 2;

		gradO = oMin - 1;
	}

	/// <summary>Image width.</summary>
	public int Width { get; }

	/// <summary>Image height.</summary>
	public int Height { get; }

	/// <summary>Number of octaves (O).</summary>
	public int NumOctaves { get; }

	/// <summary>Number of levels per octave (S).</summary>
	public int NumLevels { get; }

	/// <summary>First octave index (o_min).</summary>
	public int OctaveMin { get; }

	/// <summary>Lowest scale level stored per octave (s_min, always -1).</summary>
	public int SMin { get; }

	/// <summary>Highest scale level stored per octave (s_max = S + 1).</summary>
	public int SMax { get; }

	/// <summary>The octave the filter currently holds (o_cur).</summary>
	public int CurrentOctave { get; private set; }

	/// <summary>Width of the current octave.</summary>
	public int OctaveWidth { get; private set; }

	/// <summary>Height of the current octave.</summary>
	public int OctaveHeight { get; private set; }

	/// <summary>Smoothing of the pyramid base (sigma0).</summary>
	public double Sigma0 => sigma0;

	/// <summary>DoG peak threshold (vl_sift_set_peak_thresh).</summary>
	public double PeakThreshold { get; set; }

	/// <summary>Edge threshold (vl_sift_set_edge_thresh).</summary>
	public double EdgeThreshold { get; set; }

	/// <summary>Descriptor norm threshold (vl_sift_set_norm_thresh); zero disables it.</summary>
	public double NormThreshold { get; set; }

	/// <summary>Descriptor magnification factor (vl_sift_set_magnif).</summary>
	public double Magnif { get; set; }

	/// <summary>Gaussian window size of the descriptor, in spatial bins (vl_sift_set_window_size).</summary>
	public double WindowSize { get; set; }

	/// <summary>Number of keypoints the last <see cref="Detect"/> found (vl_sift_get_nkeypoints).</summary>
	public int NumKeypoints { get; private set; }

	/// <summary>The keypoints the last <see cref="Detect"/> found (vl_sift_get_keypoints).</summary>
	public ReadOnlySpan<VlSiftKeypoint> Keypoints => keys.AsSpan(0, NumKeypoints);

	/// <summary>
	/// Port of vl_sift_process_first_octave: builds the Gaussian scale space of the first
	/// octave of <paramref name="im"/> (row-major, <see cref="Width"/> x <see cref="Height"/>).
	/// Returns false (VL_ERR_EOF) when there are no octaves.
	/// </summary>
	public bool ProcessFirstOctave(ReadOnlySpan<float> im, CancellationToken cancellationToken = default)
	{
		ThrowIfCancellationRequested(cancellationToken);

		int width = Width;
		int height = Height;
		int oMin = OctaveMin;

		CurrentOctave = oMin;
		NumKeypoints = 0;

		// Not in VLFeat, which resets grad_o only in vl_sift_new: a filter reused for a new
		// image would otherwise keep the previous image's gradient when that image's last
		// octave with keypoints is this one's first, making results depend on image order
		// (divergence 43).
		gradO = oMin - 1;

		int w = OctaveWidth = ShiftLeft(Width, -CurrentOctave);
		int h = OctaveHeight = ShiftLeft(Height, -CurrentOctave);

		if (NumOctaves == 0)
		{
			return false;
		}

		// If the first octave has negative index, we upscale the image; if the first octave
		// has positive index, we downscale the image; if the first octave has index zero, we
		// just copy the image.
		int octaveBase = OctaveOffset(SMin);
		if (oMin < 0)
		{
			// Double once.
			CopyAndUpsampleRows(temp, 0, im, width, height);
			CopyAndUpsampleRows(octave, octaveBase, temp.AsSpan(0, 2 * width * height), height, 2 * width);

			// Double more.
			for (int o = -1; o > oMin; --o)
			{
				int n = (width << -o) * (height << -o);
				CopyAndUpsampleRows(temp, 0, octave.AsSpan(octaveBase, n), width << -o, height << -o);
				CopyAndUpsampleRows(octave, octaveBase, temp.AsSpan(0, 2 * n), width << -o, 2 * (height << -o));
				ThrowIfCancellationRequested(cancellationToken);
			}
		}
		else if (oMin > 0)
		{
			CopyAndDownsample(octave, octaveBase, im, 0, width, height, oMin);
		}
		else
		{
			im.Slice(0, width * height).CopyTo(octave.AsSpan(octaveBase));
		}

		// Here we adjust the smoothing of the first level of the octave. The input image is
		// assumed to have nominal smoothing equal to sigman.
		double sa = sigma0 * Math.Pow(sigmak, SMin);
		double sb = sigman * Math.Pow(2.0, -oMin);

		if (sa > sb)
		{
			double sd = Math.Sqrt((sa * sa) - (sb * sb));
			Smooth(octave, octaveBase, octave, octaveBase, w, h, sd);
		}

		FillOctave(w, h, cancellationToken);
		return true;
	}

	/// <summary>
	/// Port of vl_sift_process_next_octave: downsamples to the next octave and builds its
	/// scale space. Clears the keypoints of the previous octave. Returns false (VL_ERR_EOF)
	/// after the last octave.
	/// </summary>
	public bool ProcessNextOctave(CancellationToken cancellationToken = default)
	{
		if (CurrentOctave == OctaveMin + NumOctaves - 1)
		{
			return false;
		}

		// Retrieve base.
		int sBest = Math.Min(SMin + NumLevels, SMax);
		int w = OctaveWidth;
		int h = OctaveHeight;
		int pt = OctaveOffset(sBest);
		int octaveBase = OctaveOffset(SMin);

		ThrowIfCancellationRequested(cancellationToken);

		// Next octave.
		CopyAndDownsample(octave, octaveBase, octave, pt, w, h, 1);

		CurrentOctave += 1;
		NumKeypoints = 0;
		w = OctaveWidth = ShiftLeft(Width, -CurrentOctave);
		h = OctaveHeight = ShiftLeft(Height, -CurrentOctave);

		// VLFeat uses powf here (unlike the first octave): sigmak and the level are rounded
		// to float and the power is computed in single precision.
		double sa = sigma0 * MathF.Pow((float)sigmak, SMin);
		double sb = sigma0 * MathF.Pow((float)sigmak, sBest - NumLevels);

		if (sa > sb)
		{
			double sd = Math.Sqrt((sa * sa) - (sb * sb));
			Smooth(octave, octaveBase, octave, octaveBase, w, h, sd);
		}

		FillOctave(w, h, cancellationToken);
		return true;
	}

	// Start of scale level s of the current octave (vl_sift_get_octave).
	private int OctaveOffset(int s) => OctaveWidth * OctaveHeight * (s - SMin);

	// Smooths each level from the one below it (the "Fill octave" loop of both process calls).
	private void FillOctave(int w, int h, CancellationToken cancellationToken)
	{
		for (int s = SMin + 1; s <= SMax; ++s)
		{
			ThrowIfCancellationRequested(cancellationToken);
			double sd = dsigma0 * Math.Pow(sigmak, s);
			Smooth(octave, OctaveOffset(s), octave, OctaveOffset(s - 1), w, h, sd);
		}
	}

	/// <summary>
	/// The cancellation check between units of work (a scale level, an octave step), shared
	/// with the SIFT extractor that drives this filter so one diagnostic sees every check.
	/// </summary>
	internal void ThrowIfCancellationRequested(CancellationToken cancellationToken)
	{
		CancellationCheckpoint?.Invoke();
		cancellationToken.ThrowIfCancellationRequested();
	}

	// VL_SHIFT_LEFT: a left shift for n >= 0, an arithmetic right shift otherwise.
	private static int ShiftLeft(int x, int n) => n >= 0 ? x << n : x >> -n;

	// sift.c defines log2(x) as log(x) / VL_LOG_OF_2.
	private static double Log2(double x) => Math.Log(x) / VlMathOp.LogOf2;

	// Port of _vl_sift_smooth: separable Gaussian smoothing through the temp buffer. The
	// output may alias the input (the two passes go through temp).
	private void Smooth(float[] output, int outputOffset, float[] input, int inputOffset, int width, int height, double sigma)
	{
		// Prepare Gaussian filter.
		if (gaussFilterSigma != sigma)
		{
			float acc = 0;
			gaussFilterWidth = (int)Math.Max(Math.Ceiling(4.0 * sigma), 1);
			gaussFilterSigma = sigma;
			gaussFilter = new float[(2 * gaussFilterWidth) + 1];

			for (int j = 0; j < (2 * gaussFilterWidth) + 1; ++j)
			{
				float d = (float)(j - gaussFilterWidth) / (float)sigma;
				gaussFilter[j] = (float)Math.Exp(-0.5 * (double)(d * d));
				acc += gaussFilter[j];
			}

			for (int j = 0; j < (2 * gaussFilterWidth) + 1; ++j)
			{
				gaussFilter[j] /= acc;
			}
		}

		if (gaussFilterWidth == 0)
		{
			Array.Copy(input, inputOffset, output, outputOffset, width * height);
			return;
		}

		float[] filter = gaussFilter!;
		VlImOpv.ConvColVF(
			temp, 0, height, input, inputOffset, width, height, width,
			filter, 0, -gaussFilterWidth, gaussFilterWidth,
			1, VlImOpv.PadByContinuity | VlImOpv.Transpose);

		VlImOpv.ConvColVF(
			output, outputOffset, width, temp, 0, height, width, height,
			filter, 0, -gaussFilterWidth, gaussFilterWidth,
			1, VlImOpv.PadByContinuity | VlImOpv.Transpose);
	}

	// Port of copy_and_upsample_rows: upsamples the rows by linear interpolation and
	// transposes, so the output is height x (2 * width) stored as 2 * width rows of height.
	private static void CopyAndUpsampleRows(float[] dst, int dstOffset, ReadOnlySpan<float> src, int width, int height)
	{
		int s = 0;
		int d = dstOffset;
		for (int y = 0; y < height; ++y)
		{
			float a, b;
			b = a = src[s++];
			for (int x = 0; x < width - 1; ++x)
			{
				b = src[s++];
				dst[d] = a;
				d += height;

				// 0.5 * (a + b) promotes the float sum to double; halving is exact, so the
				// float product is the same value.
				dst[d] = (float)(0.5 * (double)(a + b));
				d += height;
				a = b;
			}

			dst[d] = b;
			d += height;
			dst[d] = b;
			d += height;
			d += 1 - (width * 2 * height);
		}
	}

	// Port of copy_and_downsample: keeps every 2^d-th pixel of every 2^d-th row.
	private static void CopyAndDownsample(float[] dst, int dstOffset, ReadOnlySpan<float> src, int srcOffset, int width, int height, int d)
	{
		int step = 1 << d;
		int di = dstOffset;
		for (int y = 0; y < height; y += step)
		{
			int srcrow = srcOffset + (y * width);
			for (int x = 0; x < width - (step - 1); x += step)
			{
				dst[di++] = src[srcrow];
				srcrow += step;
			}
		}
	}
}
