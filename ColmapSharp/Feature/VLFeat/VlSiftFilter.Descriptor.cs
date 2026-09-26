// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from VLFeat (BSD-2-Clause, see THIRD_PARTY_NOTICES.md) as vendored by COLMAP.
//
// VlSiftFilter.Descriptor: update_gradient, vl_sift_calc_keypoint_orientations,
// normalize_histogram, vl_sift_calc_keypoint_descriptor and vl_sift_calc_raw_descriptor from
// thirdparty/VLFeat/sift.c. Part of VlSiftFilter (scale space in VlSiftFilter.cs, detection in
// VlSiftFilter.Detect.cs). The raw descriptor is what COLMAP's covariant extractor
// (Feature/CovariantSift.cs) computes on its warped patches.
//
// Tier A (exact). Gradients and descriptor bins are float, orientation histograms double;
// every conversion between them is written out where the C converts implicitly (a float
// local assigned a double expression rounds once, at the assignment).
//
// vl_sift_keypoint_init is not ported: nothing in COLMAP calls it.

namespace ColmapSharp.Feature.VLFeat;

/// <content>Orientation assignment and descriptors.</content>
public sealed partial class VlSiftFilter
{
	/// <summary>Length of a SIFT descriptor (NBO * NBP * NBP).</summary>
	public const int DescriptorLength = Nbo * Nbp * Nbp;

	/// <summary>
	/// Port of vl_sift_calc_keypoint_orientations: up to four dominant gradient orientations
	/// of <paramref name="k"/>, written to <paramref name="angles"/>. Returns how many were
	/// found; zero when the keypoint is not in the current octave or out of bounds.
	/// </summary>
	public int CalcKeypointOrientations(Span<double> angles, in VlSiftKeypoint k)
	{
		const double Winf = 1.5;
		double xper = Math.Pow(2.0, CurrentOctave);

		int w = OctaveWidth;
		int h = OctaveHeight;
		const int Xo = 2;
		int yo = 2 * w;
		int so = 2 * w * h;
		double x = k.X / xper;
		double y = k.Y / xper;
		double sigma = k.Sigma / xper;

		int xi = (int)(x + 0.5);
		int yi = (int)(y + 0.5);
		int si = k.IS;

		double sigmaw = Winf * sigma;
		int bigW = (int)Math.Max(Math.Floor(3.0 * sigmaw), 1);

		const int Nbins = 36;
		Span<double> hist = stackalloc double[Nbins];

		// Skip if the keypoint octave is not current.
		if (k.O != CurrentOctave)
		{
			return 0;
		}

		// Skip the keypoint if it is out of bounds.
		if (xi < 0 || xi > w - 1 || yi < 0 || yi > h - 1 || si < SMin + 1 || si > SMax - 2)
		{
			return 0;
		}

		UpdateGradient();

		hist.Clear();

		// Compute orientation histogram.
		int pt = (Xo * xi) + (yo * yi) + (so * (si - SMin - 1));

		for (int ys = Math.Max(-bigW, -yi); ys <= Math.Min(+bigW, h - 1 - yi); ++ys)
		{
			for (int xs = Math.Max(-bigW, -xi); xs <= Math.Min(+bigW, w - 1 - xi); ++xs)
			{
				double dx = (double)(xi + xs) - x;
				double dy = (double)(yi + ys) - y;
				double r2 = (dx * dx) + (dy * dy);

				// Limit to a circular window.
				if (r2 >= (bigW * bigW) + 0.6)
				{
					continue;
				}

				double wgt = VlMathOp.FastExpN(r2 / (2 * sigmaw * sigmaw));
				double mod = grad[pt + (xs * Xo) + (ys * yo)];
				double ang = grad[pt + (xs * Xo) + (ys * yo) + 1];
				double fbin = Nbins * ang / (2 * VlMathOp.Pi);

				// VL_SIFT_BILINEAR_ORIENTATIONS is defined: split the vote between two bins.
				int bin = (int)VlMathOp.FloorD(fbin - 0.5);
				double rbin = fbin - bin - 0.5;
				hist[(bin + Nbins) % Nbins] += (1 - rbin) * mod * wgt;
				hist[(bin + 1) % Nbins] += rbin * mod * wgt;
			}
		}

		// Smooth histogram.
		for (int iter = 0; iter < 6; iter++)
		{
			double prev = hist[Nbins - 1];
			double first = hist[0];
			int i;
			for (i = 0; i < Nbins - 1; i++)
			{
				double newh = (prev + hist[i] + hist[(i + 1) % Nbins]) / 3.0;
				prev = hist[i];
				hist[i] = newh;
			}

			hist[i] = (prev + hist[i] + first) / 3.0;
		}

		// Find the histogram maximum.
		double maxh = 0;
		for (int i = 0; i < Nbins; ++i)
		{
			maxh = Math.Max(maxh, hist[i]);
		}

		// Find peaks within 80% from max.
		int nangles = 0;
		for (int i = 0; i < Nbins; ++i)
		{
			double h0 = hist[i];
			double hm = hist[(i - 1 + Nbins) % Nbins];
			double hp = hist[(i + 1 + Nbins) % Nbins];

			// Is this a peak?
			if (h0 > 0.8 * maxh && h0 > hm && h0 > hp)
			{
				// Quadratic interpolation.
				double di = -0.5 * (hp - hm) / (hp + hm - (2 * h0));
				double th = 2 * VlMathOp.Pi * (i + di + 0.5) / Nbins;
				angles[nangles++] = th;
				if (nangles == 4)
				{
					break;
				}
			}
		}

		return nangles;
	}

	/// <summary>
	/// Port of vl_sift_calc_keypoint_descriptor: the 128-bin SIFT descriptor of
	/// <paramref name="k"/> at orientation <paramref name="angle0"/>, in VLFeat's bin order.
	/// Leaves <paramref name="descr"/> untouched when the keypoint is not usable, like VLFeat.
	/// </summary>
	public void CalcKeypointDescriptor(Span<float> descr, in VlSiftKeypoint k, double angle0)
	{
		double magnif = Magnif;
		double xper = Math.Pow(2.0, CurrentOctave);

		int w = OctaveWidth;
		int h = OctaveHeight;
		const int Xo = 2;
		int yo = 2 * w;
		int so = 2 * w * h;
		double x = k.X / xper;
		double y = k.Y / xper;
		double sigma = k.Sigma / xper;

		int xi = (int)(x + 0.5);
		int yi = (int)(y + 0.5);
		int si = k.IS;

		double st0 = Math.Sin(angle0);
		double ct0 = Math.Cos(angle0);
		double sbp = (magnif * sigma) + VlMathOp.EpsilonD;
		int bigW = (int)Math.Floor((Math.Sqrt(2.0) * sbp * (Nbp + 1) / 2.0) + 0.5);

		const int Binto = 1;
		const int Binyo = Nbo * Nbp;
		const int Binxo = Nbo;

		// Check bounds.
		if (k.O != CurrentOctave || xi < 0 || xi >= w || yi < 0 || yi >= h - 1 || si < SMin + 1 || si > SMax - 2)
		{
			return;
		}

		UpdateGradient();

		descr.Slice(0, DescriptorLength).Clear();

		// Center the scale space and the descriptor on the current keypoint. Note that dpt is
		// pointing to the bin of center (SBP/2, SBP/2, 0).
		int pt = (xi * Xo) + (yi * yo) + ((si - SMin - 1) * so);
		int dpt = ((Nbp / 2) * Binyo) + ((Nbp / 2) * Binxo);

		float wsigma = (float)WindowSize;

		// Process pixels in the intersection of the image rectangle (1,1)-(M-1,N-1) and the
		// keypoint bounding box.
		for (int dyi = Math.Max(-bigW, 1 - yi); dyi <= Math.Min(+bigW, h - yi - 2); ++dyi)
		{
			for (int dxi = Math.Max(-bigW, 1 - xi); dxi <= Math.Min(+bigW, w - xi - 2); ++dxi)
			{
				float mod = grad[pt + (dxi * Xo) + (dyi * yo) + 0];
				float angle = grad[pt + (dxi * Xo) + (dyi * yo) + 1];
				float theta = VlMathOp.Mod2PiF((float)(angle - angle0));

				// Fractional displacement.
				float dx = (float)(xi + dxi - x);
				float dy = (float)(yi + dyi - y);

				// Get the displacement normalized w.r.t. the keypoint orientation and extension.
				float nx = (float)(((ct0 * dx) + (st0 * dy)) / sbp);
				float ny = (float)(((-st0 * dx) + (ct0 * dy)) / sbp);
				float nt = (float)((Nbo * theta) / (2 * VlMathOp.Pi));

				// Get the Gaussian weight of the sample. The Gaussian window has a standard
				// deviation equal to NBP/2. Note that dx and dy are in the normalized frame,
				// so that -NBP/2 <= dx <= NBP/2.
				float win = (float)VlMathOp.FastExpN(((nx * nx) + (ny * ny)) / (2.0 * wsigma * wsigma));

				// The sample will be distributed in 8 adjacent bins. We start from the
				// "lower-left" bin. nx - 0.5 is a double rounded to float for vl_floor_f.
				int binx = (int)VlMathOp.FloorF((float)(nx - 0.5));
				int biny = (int)VlMathOp.FloorF((float)(ny - 0.5));
				int bint = (int)VlMathOp.FloorF(nt);
				float rbinx = (float)(nx - (binx + 0.5));
				float rbiny = (float)(ny - (biny + 0.5));
				float rbint = nt - bint;

				// Distribute the current sample into the 8 adjacent bins.
				for (int dbinx = 0; dbinx < 2; ++dbinx)
				{
					for (int dbiny = 0; dbiny < 2; ++dbiny)
					{
						for (int dbint = 0; dbint < 2; ++dbint)
						{
							if (binx + dbinx >= -(Nbp / 2) &&
								binx + dbinx < (Nbp / 2) &&
								biny + dbiny >= -(Nbp / 2) &&
								biny + dbiny < (Nbp / 2))
							{
								float weight = win
									* mod
									* MathF.Abs(1 - dbinx - rbinx)
									* MathF.Abs(1 - dbiny - rbiny)
									* MathF.Abs(1 - dbint - rbint);

								descr[dpt + (((bint + dbint) % Nbo) * Binto) + ((biny + dbiny) * Binyo) + ((binx + dbinx) * Binxo)] += weight;
							}
						}
					}
				}
			}
		}

		// Standard SIFT descriptors are normalized, truncated and normalized again.
		Span<float> d = descr.Slice(0, DescriptorLength);

		// Normalize the histogram to L2 unit length.
		float norm = NormalizeHistogram(d);

		// Set the descriptor to zero if it is lower than our norm_threshold.
		if (NormThreshold != 0 && norm < NormThreshold)
		{
			d.Clear();
		}
		else
		{
			// Truncate at 0.2.
			for (int bin = 0; bin < DescriptorLength; ++bin)
			{
				if ((double)d[bin] > 0.2)
				{
					d[bin] = (float)0.2;
				}
			}

			// Normalize again.
			NormalizeHistogram(d);
		}
	}

	/// <summary>
	/// Port of vl_sift_calc_raw_descriptor: the SIFT descriptor of the keypoint at
	/// (<paramref name="x"/>, <paramref name="y"/>) with scale <paramref name="sigma"/> and
	/// orientation <paramref name="angle0"/>, computed from a caller-supplied gradient image
	/// (interleaved modulus and angle, as vl_imgradient_polar_f writes with a horizontal
	/// stride of 2) of size <paramref name="width"/> x <paramref name="height"/>. Uses only
	/// the filter's magnif, window size and norm threshold. Leaves <paramref name="descr"/>
	/// untouched when the keypoint is outside the image, like VLFeat.
	/// </summary>
	public void CalcRawDescriptor(
		ReadOnlySpan<float> grad, Span<float> descr, int width, int height, double x, double y, double sigma, double angle0)
	{
		double magnif = Magnif;
		int w = width;
		int h = height;
		const int Xo = 2;
		int yo = 2 * w;

		int xi = (int)(x + 0.5);
		int yi = (int)(y + 0.5);

		double st0 = Math.Sin(angle0);
		double ct0 = Math.Cos(angle0);
		double sbp = (magnif * sigma) + VlMathOp.EpsilonD;
		int bigW = (int)Math.Floor((Math.Sqrt(2.0) * sbp * (Nbp + 1) / 2.0) + 0.5);

		const int Binto = 1;
		const int Binyo = Nbo * Nbp;
		const int Binxo = Nbo;

		// Check bounds.
		if (xi < 0 || xi >= w || yi < 0 || yi >= h - 1)
		{
			return;
		}

		Span<float> d = descr.Slice(0, DescriptorLength);
		d.Clear();

		// Center the scale space and the descriptor on the current keypoint. Note that dpt is
		// pointing to the bin of center (SBP/2, SBP/2, 0).
		int pt = (xi * Xo) + (yi * yo);
		int dpt = ((Nbp / 2) * Binyo) + ((Nbp / 2) * Binxo);
		float wsigma = (float)WindowSize;

		// Process pixels in the intersection of the image rectangle and the keypoint bounding
		// box (unlike the scale-space descriptor, the border pixels are included).
		for (int dyi = Math.Max(-bigW, -yi); dyi <= Math.Min(+bigW, h - yi - 1); ++dyi)
		{
			for (int dxi = Math.Max(-bigW, -xi); dxi <= Math.Min(+bigW, w - xi - 1); ++dxi)
			{
				float mod = grad[pt + (dxi * Xo) + (dyi * yo) + 0];
				float angle = grad[pt + (dxi * Xo) + (dyi * yo) + 1];
				float theta = VlMathOp.Mod2PiF((float)(angle - angle0));

				// Fractional displacement.
				float dx = (float)(xi + dxi - x);
				float dy = (float)(yi + dyi - y);

				// Get the displacement normalized w.r.t. the keypoint orientation and extension.
				float nx = (float)(((ct0 * dx) + (st0 * dy)) / sbp);
				float ny = (float)(((-st0 * dx) + (ct0 * dy)) / sbp);
				float nt = (float)((Nbo * theta) / (2 * VlMathOp.Pi));

				// The Gaussian window has a standard deviation equal to NBP/2 in the normalized
				// frame.
				float win = (float)VlMathOp.FastExpN(((nx * nx) + (ny * ny)) / (2.0 * wsigma * wsigma));

				// The sample will be distributed in 8 adjacent bins, from the "lower-left" one.
				int binx = (int)VlMathOp.FloorF((float)(nx - 0.5));
				int biny = (int)VlMathOp.FloorF((float)(ny - 0.5));
				int bint = (int)VlMathOp.FloorF(nt);
				float rbinx = (float)(nx - (binx + 0.5));
				float rbiny = (float)(ny - (biny + 0.5));
				float rbint = nt - bint;

				for (int dbinx = 0; dbinx < 2; ++dbinx)
				{
					for (int dbiny = 0; dbiny < 2; ++dbiny)
					{
						for (int dbint = 0; dbint < 2; ++dbint)
						{
							if (binx + dbinx >= -(Nbp / 2) &&
								binx + dbinx < (Nbp / 2) &&
								biny + dbiny >= -(Nbp / 2) &&
								biny + dbiny < (Nbp / 2))
							{
								float weight = win
									* mod
									* MathF.Abs(1 - dbinx - rbinx)
									* MathF.Abs(1 - dbiny - rbiny)
									* MathF.Abs(1 - dbint - rbint);

								d[dpt + (((bint + dbint) % Nbo) * Binto) + ((biny + dbiny) * Binyo) + ((binx + dbinx) * Binxo)] += weight;
							}
						}
					}
				}
			}
		}

		// Standard SIFT descriptors are normalized, truncated and normalized again.
		float norm = NormalizeHistogram(d);

		// Zero the descriptor if its norm is below norm_thresh times the number of samples (the
		// Gaussian window of the descriptor is not normalized).
		int numSamples =
			(Math.Min(bigW, w - xi - 1) - Math.Max(-bigW, -xi) + 1) *
			(Math.Min(bigW, h - yi - 1) - Math.Max(-bigW, -yi) + 1);

		if (NormThreshold != 0 && norm < NormThreshold * numSamples)
		{
			d.Clear();
		}
		else
		{
			// Truncate at 0.2.
			for (int bin = 0; bin < DescriptorLength; ++bin)
			{
				if ((double)d[bin] > 0.2)
				{
					d[bin] = (float)0.2;
				}
			}

			// Normalize again.
			NormalizeHistogram(d);
		}
	}

	// Port of normalize_histogram: divides by the fast L2 norm (+ VL_EPSILON_F), returns it.
	private static float NormalizeHistogram(Span<float> hist)
	{
		float norm = 0.0f;
		for (int i = 0; i < hist.Length; ++i)
		{
			norm += hist[i] * hist[i];
		}

		norm = VlMathOp.FastSqrtF(norm) + VlMathOp.EpsilonF;

		for (int i = 0; i < hist.Length; ++i)
		{
			hist[i] /= norm;
		}

		return norm;
	}

	// Port of update_gradient: polar gradients (modulus, angle in [0, 2*pi]) of the levels
	// s_min + 1 .. s_max - 2 of the current octave, computed once per octave.
	private void UpdateGradient()
	{
		int w = OctaveWidth;
		int h = OctaveHeight;
		int so = h * w;

		if (gradO == CurrentOctave)
		{
			return;
		}

		int levels = SMax - 2 - SMin;
		void Level(int level)
		{
			int s = SMin + 1 + level;
			GradientLevel(octave, OctaveOffset(s), grad, 2 * so * (s - SMin - 1), w, h);
		}

		// Each level writes its own slice of the gradient buffer.
		if ((long)w * h < 1 << 16)
		{
			for (int level = 0; level < levels; ++level)
			{
				Level(level);
			}
		}
		else
		{
			Parallel.For(0, levels, Level);
		}

		gradO = CurrentOctave;
	}

	// One level of update_gradient: one-sided differences on the border, central inside.
	private static void GradientLevel(float[] octave, int srcStart, float[] grad, int gradStart, int w, int h)
	{
		const int Xo = 1;
		int yo = w;
		int src = srcStart;
		int g = gradStart;
		float gx, gy;

		void SaveBack(ref int src, ref int g, float gx, float gy)
		{
			grad[g++] = VlMathOp.FastSqrtF((gx * gx) + (gy * gy));
			grad[g++] = VlMathOp.Mod2PiF((float)(VlMathOp.FastAtan2F(gy, gx) + (2 * VlMathOp.Pi)));
			++src;
		}

		// "0.5 * (a - b)" promotes the float difference to double; halving is exact, so the
		// value is the same as the float product.
		static float Half(float v) => (float)(0.5 * v);

		// First pixel of the first row.
		gx = octave[src + Xo] - octave[src];
		gy = octave[src + yo] - octave[src];
		SaveBack(ref src, ref g, gx, gy);

		// Middle pixels of the first row.
		int end = (src - 1) + w - 1;
		while (src < end)
		{
			gx = Half(octave[src + Xo] - octave[src - Xo]);
			gy = octave[src + yo] - octave[src];
			SaveBack(ref src, ref g, gx, gy);
		}

		// Last pixel of the first row.
		gx = octave[src] - octave[src - Xo];
		gy = octave[src + yo] - octave[src];
		SaveBack(ref src, ref g, gx, gy);

		for (int y = 1; y < h - 1; ++y)
		{
			// First pixel of the middle rows.
			gx = octave[src + Xo] - octave[src];
			gy = Half(octave[src + yo] - octave[src - yo]);
			SaveBack(ref src, ref g, gx, gy);

			// Middle pixels of the middle rows.
			end = (src - 1) + w - 1;
			while (src < end)
			{
				gx = Half(octave[src + Xo] - octave[src - Xo]);
				gy = Half(octave[src + yo] - octave[src - yo]);
				SaveBack(ref src, ref g, gx, gy);
			}

			// Last pixel of the middle row.
			gx = octave[src] - octave[src - Xo];
			gy = Half(octave[src + yo] - octave[src - yo]);
			SaveBack(ref src, ref g, gx, gy);
		}

		// First pixel of the last row.
		gx = octave[src + Xo] - octave[src];
		gy = octave[src] - octave[src - yo];
		SaveBack(ref src, ref g, gx, gy);

		// Middle pixels of the last row.
		end = (src - 1) + w - 1;
		while (src < end)
		{
			gx = Half(octave[src + Xo] - octave[src - Xo]);
			gy = octave[src] - octave[src - yo];
			SaveBack(ref src, ref g, gx, gy);
		}

		// Last pixel of the last row.
		gx = octave[src] - octave[src - Xo];
		gy = octave[src] - octave[src - yo];
		SaveBack(ref src, ref g, gx, gy);
	}
}
