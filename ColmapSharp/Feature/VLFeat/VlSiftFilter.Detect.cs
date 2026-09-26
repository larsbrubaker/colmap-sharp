// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from VLFeat (BSD-2-Clause, see THIRD_PARTY_NOTICES.md) as vendored by COLMAP.
//
// VlSiftFilter.Detect: vl_sift_detect from thirdparty/VLFeat/sift.c - the difference of
// Gaussians, the 3x3x3 extremum search and the quadratic refinement with the peak and edge
// tests. Part of VlSiftFilter (scale space in VlSiftFilter.cs, descriptors in
// VlSiftFilter.Descriptor.cs).
//
// Tier A (exact). The DoG is float; the refinement is double, with the float samples
// promoted where the C promotes them (float sums and differences first, then the double
// scaling), and the 3x3 solve is VLFeat's own Gauss elimination with partial pivoting.

namespace ColmapSharp.Feature.VLFeat;

/// <content>Keypoint detection.</content>
public sealed partial class VlSiftFilter
{
	/// <summary>
	/// Port of vl_sift_detect: finds the DoG extrema of the current octave and refines them.
	/// Results are in <see cref="Keypoints"/>.
	/// </summary>
	public void Detect()
	{
		int w = OctaveWidth;
		int h = OctaveHeight;
		double te = EdgeThreshold;
		double tp = PeakThreshold;

		const int Xo = 1;
		int yo = w;
		int so = w * h;

		double xper = Math.Pow(2.0, CurrentOctave);

		NumKeypoints = 0;
		int nkeys = 0;

		// Compute difference of Gaussians (DoG).
		int p = 0;
		for (int s = SMin; s <= SMax - 1; ++s)
		{
			int srcA = OctaveOffset(s);
			int srcB = OctaveOffset(s + 1);
			int endA = srcA + (w * h);
			while (srcA != endA)
			{
				dog[p++] = octave[srcB++] - octave[srcA++];
			}
		}

		// Find local maxima of DoG, starting from dog[1, 1, s_min + 1].
		float[] dg = dog;
		p = Xo + yo + so;
		double peakFloor = 0.8 * tp;
		for (int s = SMin + 1; s <= SMax - 2; ++s)
		{
			for (int y = 1; y < h - 1; ++y)
			{
				for (int x = 1; x < w - 1; ++x)
				{
					float v = dg[p];
					if (IsExtremum(dg, p, v, peakFloor, yo, so))
					{
						if (nkeys >= keys.Length)
						{
							Array.Resize(ref keys, keys.Length + 500);
						}

						ref VlSiftKeypoint key = ref keys[nkeys++];
						key.IX = x;
						key.IY = y;
						key.IS = s;
					}

					p += 1;
				}

				p += 2;
			}

			p += 2 * yo;
		}

		// Refine local maxima; k is the write-back index for the survivors.
		int k = 0;
		Span<double> a = stackalloc double[9];
		Span<double> b = stackalloc double[3];
		for (int i = 0; i < nkeys; ++i)
		{
			int x = keys[i].IX;
			int y = keys[i].IY;
			int s = keys[i].IS;

			double dxv = 0, dyv = 0, dsv = 0, dxx = 0, dyy = 0, dss = 0, dxy = 0, dxs = 0, dys = 0;
			int dx = 0;
			int dy = 0;
			int pt = 0;

			for (int iter = 0; iter < 5; ++iter)
			{
				x += dx;
				y += dy;

				pt = (Xo * x) + (yo * y) + (so * (s - SMin));

				// Compute the gradient. The float differences are formed in float and then
				// promoted, like the C.
				dxv = 0.5 * (double)(At(pt, 1, 0, 0) - At(pt, -1, 0, 0));
				dyv = 0.5 * (double)(At(pt, 0, 1, 0) - At(pt, 0, -1, 0));
				dsv = 0.5 * (double)(At(pt, 0, 0, 1) - At(pt, 0, 0, -1));

				// Compute the Hessian.
				dxx = (double)(At(pt, 1, 0, 0) + At(pt, -1, 0, 0)) - (2.0 * At(pt, 0, 0, 0));
				dyy = (double)(At(pt, 0, 1, 0) + At(pt, 0, -1, 0)) - (2.0 * At(pt, 0, 0, 0));
				dss = (double)(At(pt, 0, 0, 1) + At(pt, 0, 0, -1)) - (2.0 * At(pt, 0, 0, 0));

				dxy = 0.25 * (double)(At(pt, 1, 1, 0) + At(pt, -1, -1, 0) - At(pt, -1, 1, 0) - At(pt, 1, -1, 0));
				dxs = 0.25 * (double)(At(pt, 1, 0, 1) + At(pt, -1, 0, -1) - At(pt, -1, 0, 1) - At(pt, 1, 0, -1));
				dys = 0.25 * (double)(At(pt, 0, 1, 1) + At(pt, 0, -1, -1) - At(pt, 0, -1, 1) - At(pt, 0, 1, -1));

				// Solve the linear system; A is column-major like VLFeat's Aat(i, j).
				a[0] = dxx;
				a[4] = dyy;
				a[8] = dss;
				a[3] = a[1] = dxy;
				a[6] = a[2] = dxs;
				a[7] = a[5] = dys;

				b[0] = -dxv;
				b[1] = -dyv;
				b[2] = -dsv;

				SolveGaussElimination(a, b);

				// If the translation of the keypoint is big, move the keypoint and re-iterate
				// the computation. Otherwise we are all set.
				dx = ((b[0] > 0.6 && x < w - 2) ? 1 : 0) + ((b[0] < -0.6 && x > 1) ? -1 : 0);
				dy = ((b[1] > 0.6 && y < h - 2) ? 1 : 0) + ((b[1] < -0.6 && y > 1) ? -1 : 0);

				if (dx == 0 && dy == 0)
				{
					break;
				}
			}

			// Check threshold and other conditions.
			double val = At(pt, 0, 0, 0) + (0.5 * ((dxv * b[0]) + (dyv * b[1]) + (dsv * b[2])));
			double score = (dxx + dyy) * (dxx + dyy) / ((dxx * dyy) - (dxy * dxy));
			double xn = x + b[0];
			double yn = y + b[1];
			double sn = s + b[2];

			bool good =
				Math.Abs(val) > tp &&
				score < (te + 1) * (te + 1) / te &&
				score >= 0 &&
				Math.Abs(b[0]) < 1.5 &&
				Math.Abs(b[1]) < 1.5 &&
				Math.Abs(b[2]) < 1.5 &&
				xn >= 0 &&
				xn <= w - 1 &&
				yn >= 0 &&
				yn <= h - 1 &&
				sn >= SMin &&
				sn <= SMax;

			if (good)
			{
				ref VlSiftKeypoint key = ref keys[k];
				key.O = CurrentOctave;
				key.IX = x;
				key.IY = y;
				key.IS = s;
				key.S = (float)sn;
				key.X = (float)(xn * xper);
				key.Y = (float)(yn * xper);
				key.Sigma = (float)(sigma0 * Math.Pow(2.0, sn / NumLevels) * xper);
				++k;
			}
		}

		NumKeypoints = k;

		float At(int ptIndex, int ddx, int ddy, int dds) => dg[ptIndex + (ddx * Xo) + (ddy * yo) + (dds * so)];
	}

	// CHECK_NEIGHBORS(>, +) || CHECK_NEIGHBORS(<, -): v beats the peak floor and all 26
	// neighbors in space and scale. The float sample is promoted for the threshold test only.
	private static bool IsExtremum(float[] d, int p, float v, double peakFloor, int yo, int so)
	{
		const int Xo = 1;
		return ((double)v >= peakFloor && IsGreaterThanNeighbors(d, p, v, Xo, yo, so)) ||
			((double)v <= -peakFloor && IsLessThanNeighbors(d, p, v, Xo, yo, so));
	}

	private static bool IsGreaterThanNeighbors(float[] d, int p, float v, int Xo, int yo, int so)
	{
		return
			v > d[p + Xo] && v > d[p - Xo] && v > d[p + so] && v > d[p - so] &&
			v > d[p + yo] && v > d[p - yo] &&
			v > d[p + yo + Xo] && v > d[p + yo - Xo] && v > d[p - yo + Xo] && v > d[p - yo - Xo] &&
			v > d[p + Xo + so] && v > d[p - Xo + so] && v > d[p + yo + so] && v > d[p - yo + so] &&
			v > d[p + yo + Xo + so] && v > d[p + yo - Xo + so] && v > d[p - yo + Xo + so] && v > d[p - yo - Xo + so] &&
			v > d[p + Xo - so] && v > d[p - Xo - so] && v > d[p + yo - so] && v > d[p - yo - so] &&
			v > d[p + yo + Xo - so] && v > d[p + yo - Xo - so] && v > d[p - yo + Xo - so] && v > d[p - yo - Xo - so];
	}

	private static bool IsLessThanNeighbors(float[] d, int p, float v, int Xo, int yo, int so)
	{
		return
			v < d[p + Xo] && v < d[p - Xo] && v < d[p + so] && v < d[p - so] &&
			v < d[p + yo] && v < d[p - yo] &&
			v < d[p + yo + Xo] && v < d[p + yo - Xo] && v < d[p - yo + Xo] && v < d[p - yo - Xo] &&
			v < d[p + Xo + so] && v < d[p - Xo + so] && v < d[p + yo + so] && v < d[p - yo + so] &&
			v < d[p + yo + Xo + so] && v < d[p + yo - Xo + so] && v < d[p - yo + Xo + so] && v < d[p - yo - Xo + so] &&
			v < d[p + Xo - so] && v < d[p - Xo - so] && v < d[p + yo - so] && v < d[p - yo - so] &&
			v < d[p + yo + Xo - so] && v < d[p + yo - Xo - so] && v < d[p - yo + Xo - so] && v < d[p - yo - Xo - so];
	}

	// VLFeat's 3x3 Gauss elimination with partial pivoting and back substitution, in place.
	// A is column-major (Aat(i, j) = A[i + 3 * j]). A singular system gives b = 0.
	private static void SolveGaussElimination(Span<double> a, Span<double> b)
	{
		for (int j = 0; j < 3; ++j)
		{
			double maxa = 0;
			double maxabsa = 0;
			int maxi = -1;

			// Look for the maximally stable pivot.
			for (int i = j; i < 3; ++i)
			{
				double v = a[i + (3 * j)];
				double absa = Math.Abs(v);
				if (absa > maxabsa)
				{
					maxa = v;
					maxabsa = absa;
					maxi = i;
				}
			}

			// If singular give up (the threshold is the float literal 1e-10f).
			if (maxabsa < (double)1e-10f)
			{
				b[0] = 0;
				b[1] = 0;
				b[2] = 0;
				break;
			}

			int r = maxi;

			// Swap j-th row with r-th row and normalize j-th row.
			double tmp;
			for (int jj = j; jj < 3; ++jj)
			{
				tmp = a[r + (3 * jj)];
				a[r + (3 * jj)] = a[j + (3 * jj)];
				a[j + (3 * jj)] = tmp;
				a[j + (3 * jj)] /= maxa;
			}

			tmp = b[j];
			b[j] = b[r];
			b[r] = tmp;
			b[j] /= maxa;

			// Elimination.
			for (int ii = j + 1; ii < 3; ++ii)
			{
				double x = a[ii + (3 * j)];
				for (int jj = j; jj < 3; ++jj)
				{
					a[ii + (3 * jj)] -= x * a[j + (3 * jj)];
				}

				b[ii] -= x * b[j];
			}
		}

		// Backward substitution.
		for (int i = 2; i > 0; --i)
		{
			double x = b[i];
			for (int ii = i - 1; ii >= 0; --ii)
			{
				b[ii] -= x * a[ii + (3 * i)];
			}
		}
	}
}
