// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from VLFeat (BSD-2-Clause, see THIRD_PARTY_NOTICES.md) as vendored by COLMAP.
//
// VlCovDet.Shape: the frame-dependent parts of thirdparty/VLFeat/covdet.c -
// vl_covdet_extract_patch_helper, vl_covdet_extract_patch_for_frame,
// vl_covdet_extract_affine_shape(_for_frame) and vl_covdet_extract_orientations(_for_frame).
// Part of VlCovDet (detection in VlCovDet.cs).
//
// Tier A (exact); see VlCovDet.cs. Kept VLFeat quirks, since they decide the numbers:
// - The patch helper picks its scale-space level with floor(log2(sigma / (factor * base)) - o),
//   which ignores the octave resolution.
// - When the warped patch leaves the image, the padded copy stops copying two pixels early on
//   each row and repeats the next source pixel instead (`patchWidth - padx1 - 2`).
// - A feature whose orientation histogram has no peak keeps its frame unrotated (it is not
//   dropped), and affine adaptation never drops a feature (its only failure is out of memory).
// - The orientations are sorted by descending score with qsort; macOS's qsort insertion-sorts
//   arrays this short (at most 4), so ties keep peak order. Here: the same insertion sort.

namespace ColmapSharp.Feature.VLFeat;

/// <content>Patch warping, affine adaptation and orientation assignment.</content>
public sealed partial class VlCovDet
{
	/// <summary>
	/// Port of vl_covdet_extract_patch_for_frame: samples a (2 * resolution + 1)^2 patch covering
	/// [-extent, extent]^2 in the frame's coordinates from the scale-space level whose smoothing
	/// best approximates <paramref name="sigma"/> (in patch units) from below.
	/// </summary>
	public void ExtractPatchForFrame(float[] patchOut, int resolution, double extent, double sigma, in VlFrameOrientedEllipse frame)
	{
		Span<double> a = [frame.A11, frame.A21, frame.A12, frame.A22];
		Span<double> t = [frame.X, frame.Y];
		Span<double> d = stackalloc double[4];

		VlMathOp.Svd2(d, Span<double>.Empty, Span<double>.Empty, a);
		PatchHelper(out _, out _, patchOut, resolution, extent, sigma, a, t, d[0], d[3]);
	}

	/// <summary>
	/// Port of vl_covdet_extract_affine_shape: replaces each feature's frame by its
	/// shape-adapted, upright frame (Baumberg / Mikolajczyk-Schmid iteration on the windowed
	/// second-moment matrix).
	/// </summary>
	public void ExtractAffineShape(CancellationToken cancellationToken = default)
	{
		int j = 0;
		int numFeatures = NumFeatures;
		for (int i = 0; i < numFeatures; ++i)
		{
			if ((i & 63) == 0)
			{
				cancellationToken.ThrowIfCancellationRequested();
			}

			ExtractAffineShapeForFrame(out VlFrameOrientedEllipse adapted, features[i].Frame);
			features[j] = features[i];
			features[j].Frame = adapted;
			++j;
		}

		NumFeatures = j;
	}

	/// <summary>
	/// Port of vl_covdet_extract_orientations: rotates each feature to its dominant gradient
	/// orientation, appending a copy of the feature for each further orientation (up to four).
	/// </summary>
	public void ExtractOrientations(CancellationToken cancellationToken = default)
	{
		int numFeatures = NumFeatures;
		for (int i = 0; i < numFeatures; ++i)
		{
			if ((i & 63) == 0)
			{
				cancellationToken.ThrowIfCancellationRequested();
			}

			VlCovDetFeature feature = features[i];
			int numOrientations = ExtractOrientationsForFrame(feature.Frame);

			for (int j = 0; j < numOrientations; ++j)
			{
				double a0 = feature.Frame.A11;
				double a1 = feature.Frame.A21;
				double a2 = feature.Frame.A12;
				double a3 = feature.Frame.A22;
				double r1 = Math.Cos(orientationAngles[j]);
				double r2 = Math.Sin(orientationAngles[j]);

				int oriented;
				if (j == 0)
				{
					oriented = i;
				}
				else
				{
					AppendFeature(feature);
					oriented = NumFeatures - 1;
				}

				features[oriented].OrientationScore = (float)orientationScores[j];
				features[oriented].Frame.A11 = (float)((+a0 * r1) + (a2 * r2));
				features[oriented].Frame.A21 = (float)((+a1 * r1) + (a3 * r2));
				features[oriented].Frame.A12 = (float)((-a0 * r2) + (a2 * r1));
				features[oriented].Frame.A22 = (float)((-a1 * r2) + (a3 * r1));
			}
		}
	}

	// Port of vl_covdet_extract_affine_shape_for_frame. Always succeeds (VL_ERR_OK).
	private void ExtractAffineShapeForFrame(out VlFrameOrientedEllipse adapted, VlFrameOrientedEllipse frame)
	{
		int iter = 0;
		Span<double> a = [frame.A11, frame.A21, frame.A12, frame.A22];
		Span<double> t = [frame.X, frame.Y];
		Span<double> u = stackalloc double[4];
		Span<double> v = stackalloc double[4];
		Span<double> d = stackalloc double[4];
		Span<double> m = stackalloc double[4];
		Span<double> p = stackalloc double[4];
		Span<double> p_ = stackalloc double[4];
		Span<double> q = stackalloc double[4];
		double sigmaD = AaRelativeDerivativeSigma;
		double referenceScale = 0;
		const int Resolution = AaPatchResolution;
		const int Side = AaPatchSide;
		const double Extent = AaPatchExtent;

		adapted = frame;

		while (true)
		{
			double lxx = 0, lxy = 0, lyy = 0;

			// A = U D V'
			VlMathOp.Svd2(d, u, v, a);
			double anisotropy = VlMax(d[0] / d[3], d[3] / d[0]);

			if (anisotropy > AaMaxAnisotropy)
			{
				// Diverged, give up with the current solution.
				break;
			}

			// Make sure that the smallest singular value stays fixed after the first iteration.
			double factor;
			if (iter == 0)
			{
				referenceScale = VlMin(d[0], d[3]);
				factor = 1.0;
			}
			else
			{
				factor = referenceScale / VlMin(d[0], d[3]);
			}

			d[0] *= factor;
			d[3] *= factor;

			a[0] = u[0] * d[0];
			a[1] = u[1] * d[0];
			a[2] = u[2] * d[3];
			a[3] = u[3] * d[3];

			adapted.A11 = (float)a[0];
			adapted.A21 = (float)a[1];
			adapted.A12 = (float)a[2];
			adapted.A22 = (float)a[3];

			if (++iter >= AaMaxNumIterations)
			{
				break;
			}

			PatchHelper(out _, out _, aaPatch, Resolution, Extent, sigmaD, a, t, d[0], d[3]);

			// VL_COVDET_AA_ACCURATE_SMOOTHING is off: no extra smoothing of the patch here.

			// Compute the second moment matrix.
			VlImOpv.ImGradientF(aaPatchX, 0, aaPatchY, 0, 1, Side, aaPatch, 0, Side, Side, Side);

			for (int k = 0; k < Side * Side; ++k)
			{
				double lx = aaPatchX[k];
				double ly = aaPatchY[k];
				lxx += lx * lx * aaMask[k];
				lyy += ly * ly * aaMask[k];
				lxy += lx * ly * aaMask[k];
			}

			m[0] = lxx;
			m[1] = lxy;
			m[2] = lxy;
			m[3] = lyy;

			if (lxx == 0 || lyy == 0)
			{
				adapted = frame;
				break;
			}

			// Decompose M = P * Q * P'.
			VlMathOp.Svd2(q, p, p_, m);

			// Setting A <- A * dA changes M approximately to dA' M dA = dA' P Q P' dA. To make
			// this proportional to the identity, dA ~= P Q^-1/2, keeping the smallest singular
			// value of A unchanged.
			if (q[3] / q[0] < AaConvergenceThreshold && q[0] / q[3] < AaConvergenceThreshold)
			{
				break;
			}

			double q0 = Math.Sqrt(q[0]);
			double q1 = Math.Sqrt(q[3]);
			double ap0 = ((a[0] * p[0]) + (a[2] * p[1])) / q0;
			double ap1 = ((a[1] * p[0]) + (a[3] * p[1])) / q0;
			double ap2 = ((a[0] * p[2]) + (a[2] * p[3])) / q1;
			double ap3 = ((a[1] * p[2]) + (a[3] * p[3])) / q1;
			a[0] = ap0;
			a[1] = ap1;
			a[2] = ap2;
			a[3] = ap3;
		}

		// Make upright. Shape adaptation does not estimate rotation, so the frame is rotated
		// such that the vertical axis (the non-transposed convention) is not rotated at all.
		Span<double> au = [adapted.A11, adapted.A21, adapted.A12, adapted.A22];
		Span<double> refv = [0, 1];
		Span<double> ref_ = stackalloc double[2];

		VlMathOp.SolveLinearSystem2(ref_, au, refv);
		double angle = Math.Atan2(refv[1], refv[0]);
		double angle_ = Math.Atan2(ref_[1], ref_[0]);
		double dangle = angle_ - angle;
		double r1 = Math.Cos(dangle);
		double r2 = Math.Sin(dangle);
		adapted.A11 = (float)((+au[0] * r1) + (au[2] * r2));
		adapted.A21 = (float)((+au[1] * r1) + (au[3] * r2));
		adapted.A12 = (float)((-au[0] * r2) + (au[2] * r1));
		adapted.A22 = (float)((-au[1] * r2) + (au[3] * r1));
	}

	// Port of vl_covdet_extract_orientations_for_frame: fills orientationAngles/Scores (sorted
	// by decreasing score) and returns how many there are.
	private int ExtractOrientationsForFrame(in VlFrameOrientedEllipse frame)
	{
		const double Extent = AaPatchExtent;
		const int Resolution = AaPatchResolution;
		const int Side = AaPatchSide;
		const int NumBins = OrNumOrientationHistogramBins;
		Span<double> hist = stackalloc double[NumBins];
		const double BinExtent = 2 * VlMathOp.Pi / OrNumOrientationHistogramBins;
		const double PeakRelativeSize = OrAdditionalPeaksRelativeSize;

		Span<double> a = [frame.A11, frame.A21, frame.A12, frame.A22];
		Span<double> t = [frame.X, frame.Y];
		Span<double> u = stackalloc double[4];
		Span<double> v = stackalloc double[4];
		Span<double> d = stackalloc double[4];
		double sigmaD = 1.0;

		// The goal is a rotation R(theta) such that the patch given by A R(theta) has the
		// strongest average gradient pointing right. To compensate for the anisotropic
		// smoothing due to warping, A = U D V' is warped by U D only and R_(theta) is estimated
		// instead, where A R(theta) = U D R_(theta): an extra rotation of theta0.
		VlMathOp.Svd2(d, u, v, a);

		a[0] = u[0] * d[0];
		a[1] = u[1] * d[0];
		a[2] = u[2] * d[3];
		a[3] = u[3] * d[3];

		double theta0 = Math.Atan2(v[1], v[0]);

		PatchHelper(out double sigma1, out double sigma2, aaPatch, Resolution, Extent, sigmaD, a, t, d[0], d[3]);

		double deltaSigma1 = Math.Sqrt(VlMax((sigmaD * sigmaD) - (sigma1 * sigma1), 0));
		double deltaSigma2 = Math.Sqrt(VlMax((sigmaD * sigmaD) - (sigma2 * sigma2), 0));
		double stephat = Extent / Resolution;
		VlImOpv.ImSmoothF(aaPatch, 0, Side, aaPatch, 0, Side, Side, Side, deltaSigma1 / stephat, deltaSigma2 / stephat);

		// Histogram of oriented gradients.
		VlImOpv.ImGradientPolarF(aaPatchX, 0, aaPatchY, 0, 1, Side, aaPatch, 0, Side, Side, Side);

		hist.Clear();
		for (int k = 0; k < Side * Side; ++k)
		{
			double modulus = aaPatchX[k];
			double angle = aaPatchY[k];
			double weight = aaMask[k];

			double x = angle / BinExtent;
			long bin = VlMathOp.FloorD(x);
			double w2 = x - bin;
			double w1 = 1.0 - w2;

			hist[(int)((bin + NumBins) % NumBins)] += w1 * (modulus * weight);
			hist[(int)((bin + NumBins + 1) % NumBins)] += w2 * (modulus * weight);
		}

		// Smooth the histogram.
		for (int iter = 0; iter < 6; iter++)
		{
			double prev = hist[NumBins - 1];
			double first = hist[0];
			int i;
			for (i = 0; i < NumBins - 1; ++i)
			{
				double curr = (prev + hist[i] + hist[(i + 1) % NumBins]) / 3.0;
				prev = hist[i];
				hist[i] = curr;
			}

			hist[i] = (prev + hist[i] + first) / 3.0;
		}

		// Find the histogram maximum.
		double maxPeakValue = 0;
		for (int i = 0; i < NumBins; ++i)
		{
			maxPeakValue = VlMax(maxPeakValue, hist[i]);
		}

		// Find the peaks within 80% of the maximum.
		int numOrientations = 0;
		for (int i = 0; i < NumBins; ++i)
		{
			double h0 = hist[i];
			double hm = hist[(i - 1 + NumBins) % NumBins];
			double hp = hist[(i + 1 + NumBins) % NumBins];

			// Is this a peak?
			if (h0 > PeakRelativeSize * maxPeakValue && h0 > hm && h0 > hp)
			{
				// Quadratic interpolation.
				double di = -0.5 * (hp - hm) / (hp + hm - (2 * h0));
				double th = (BinExtent * (i + di)) + theta0;
				orientationAngles[numOrientations] = th;
				orientationScores[numOrientations] = h0;
				numOrientations += 1;

				if (numOrientations >= MaxNumOrientations)
				{
					break;
				}
			}
		}

		// Sort the orientations by decreasing scores (qsort: an insertion sort at this size).
		for (int i = 1; i < numOrientations; ++i)
		{
			for (int k = i; k > 0 && orientationScores[k - 1] < orientationScores[k]; --k)
			{
				(orientationScores[k - 1], orientationScores[k]) = (orientationScores[k], orientationScores[k - 1]);
				(orientationAngles[k - 1], orientationAngles[k]) = (orientationAngles[k], orientationAngles[k - 1]);
			}
		}

		return numOrientations;
	}

	// Port of vl_covdet_extract_patch_helper: warps [-extent, extent]^2 through x -> A x + T
	// from the scale-space level (o, s) whose smoothing sigma_ satisfies
	// sigma_ * factor <= sigma as closely as possible, factor = 1 / min(d1, d2); sigma1/sigma2
	// return the resulting smoothing along the two axes. Starting from a level smoothed by
	// sigma_, the mapping A = U D V' makes the warped smoothing sigma_^2 V D^-2 V', so with
	// A rotated to U D it is sigma_^2 D^-2, axis-aligned. Always succeeds (VL_ERR_OK).
	private void PatchHelper(
		out double sigma1,
		out double sigma2,
		float[] patchOut,
		int resolution,
		double extent,
		double sigma,
		ReadOnlySpan<double> a_,
		ReadOnlySpan<double> t_,
		double d1,
		double d2)
	{
		VlScaleSpace g = Util.Check.NotNull(gss);
		Span<double> a = [a_[0], a_[1], a_[2], a_[3]];
		Span<double> t = [t_[0], t_[1]];
		VlScaleSpaceGeometry geom = g.Geometry;

		// Determine the best level (o, s) such that sigma_(o, s) factor <= sigma, scanning
		// octaves from the smallest and stopping when no level in the octave satisfies it.
		double factor = 1.0 / VlMin(d1, d2);
		double sigma_;
		long o;
		long s;
		for (o = geom.FirstOctave + 1; o <= geom.LastOctave; ++o)
		{
			s = VlMathOp.FloorD(VlMathOp.Log2D(sigma / (factor * geom.BaseScale)) - o);
			s = Math.Max(s, geom.OctaveFirstSubdivision);
			s = Math.Min(s, geom.OctaveLastSubdivision);
			sigma_ = geom.BaseScale * Math.Pow(2.0, o + ((double)s / geom.OctaveResolution));
			if (factor * sigma_ > sigma)
			{
				o--;
				break;
			}
		}

		o = Math.Min(o, geom.LastOctave);
		s = VlMathOp.FloorD(VlMathOp.Log2D(sigma / (factor * geom.BaseScale)) - o);
		s = Math.Max(s, geom.OctaveFirstSubdivision);
		s = Math.Min(s, geom.OctaveLastSubdivision);
		sigma_ = geom.BaseScale * Math.Pow(2.0, o + ((double)s / geom.OctaveResolution));
		sigma1 = sigma_ / d1;
		sigma2 = sigma_ / d2;

		// Now the level to warp from is known. If the patch is partially or completely out of
		// the image, create a padded copy of the required region first.
		(float[] level, int levelOffset) = g.GetLevel((int)o, (int)s);
		VlScaleSpaceOctaveGeometry oct = g.GetOctaveGeometry((int)o);
		long width = oct.Width;
		long height = oct.Height;
		double step = oct.Step;

		a[0] /= step;
		a[1] /= step;
		a[2] /= step;
		a[3] /= step;
		t[0] /= step;
		t[1] /= step;

		// Warp the patch domain to the image domain, take the enclosing box, and grow it by one
		// pixel on every side for the bilinear interpolation.
		double x0 = double.PositiveInfinity;
		double x1 = double.NegativeInfinity;
		double y0 = double.PositiveInfinity;
		double y1 = double.NegativeInfinity;
		ReadOnlySpan<double> boxx = [extent, extent, -extent, -extent];
		ReadOnlySpan<double> boxy = [-extent, extent, extent, -extent];
		for (int i = 0; i < 4; ++i)
		{
			double x = (a[0] * boxx[i]) + (a[2] * boxy[i]) + t[0];
			double y = (a[1] * boxx[i]) + (a[3] * boxy[i]) + t[1];
			x0 = VlMin(x0, x);
			x1 = VlMax(x1, x);
			y0 = VlMin(y0, y);
			y1 = VlMax(y1, y);
		}

		long x0i = (long)Math.Floor(x0) - 1;
		long y0i = (long)Math.Floor(y0) - 1;
		long x1i = (long)Math.Ceiling(x1) + 1;
		long y1i = (long)Math.Ceiling(y1) + 1;

		// If the box is not inside the image domain, copy the region, extending the image by
		// continuity.
		if (x0i < 0 || x1i > width - 1 || y0i < 0 || y1i > height - 1)
		{
			// The amount of left, right, top and bottom padding needed.
			long padx0 = Math.Max(0, -x0i);
			long pady0 = Math.Max(0, -y0i);
			long padx1 = Math.Max(0, x1i - (width - 1));
			long pady1 = Math.Max(0, y1i - (height - 1));

			// Make enough room for the patch (the buffer only grows, like VLFeat's).
			long patchWidth = x1i - x0i + 1;
			long patchHeight = y1i - y0i + 1;
			long patchSize = patchWidth * patchHeight;
			if (patchSize > patch.Length)
			{
				Array.Resize(ref patch, checked((int)patchSize));
			}

			if (pady0 < patchHeight - pady1)
			{
				// Start by filling the central horizontal band. VLFeat stops the copy two pixels
				// early and repeats the next source pixel for the rest; kept as is.
				for (long yi = y0i + pady0; yi < y0i + patchHeight - pady1; ++yi)
				{
					long dst = (yi - y0i) * patchWidth;
					long src = levelOffset + (yi * width) + Math.Min(Math.Max(0, x0i), width - 1);
					long xi;
					for (xi = x0i; xi < x0i + padx0; ++xi)
					{
						patch[dst++] = level[src];
					}

					for (; xi < x0i + patchWidth - padx1 - 2; ++xi)
					{
						patch[dst++] = level[src++];
					}

					for (; xi < x0i + patchWidth; ++xi)
					{
						patch[dst++] = level[src];
					}
				}

				// Now extend the central band up and down.
				for (long yi = 0; yi < pady0; ++yi)
				{
					Array.Copy(patch, pady0 * patchWidth, patch, yi * patchWidth, patchWidth);
				}

				for (long yi = patchHeight - pady1; yi < patchHeight; ++yi)
				{
					Array.Copy(patch, (patchHeight - pady1 - 1) * patchWidth, patch, yi * patchWidth, patchWidth);
				}
			}
			else
			{
				// "should be handled better!"
				Array.Clear(patch);
			}

			level = patch;
			levelOffset = 0;
			width = patchWidth;
			height = patchHeight;
			t[0] -= x0i;
			t[1] -= y0i;
		}

		// Resample by bilinear interpolation.
		int pt = 0;
		double yhat = -extent;
		double stephat = extent / resolution;
		for (int yyi = 0; yyi < (2 * resolution) + 1; ++yyi)
		{
			double xhat = -extent;
			double rx = (a[2] * yhat) + t[0];
			double ry = (a[3] * yhat) + t[1];
			for (int xxi = 0; xxi < (2 * resolution) + 1; ++xxi)
			{
				double x = (a[0] * xhat) + rx;
				double y = (a[1] * xhat) + ry;
				long xi = VlMathOp.FloorD(x);
				long yi = VlMathOp.FloorD(y);
				double i00 = level[levelOffset + (yi * width) + xi];
				double i10 = level[levelOffset + (yi * width) + xi + 1];
				double i01 = level[levelOffset + ((yi + 1) * width) + xi];
				double i11 = level[levelOffset + ((yi + 1) * width) + xi + 1];
				double wx = x - xi;
				double wy = y - yi;

				patchOut[pt++] = (float)(((1.0 - wy) * (((1.0 - wx) * i00) + (wx * i10))) +
					(wy * (((1.0 - wx) * i01) + (wx * i11))));

				xhat += stephat;
			}

			yhat += stephat;
		}
	}
}
