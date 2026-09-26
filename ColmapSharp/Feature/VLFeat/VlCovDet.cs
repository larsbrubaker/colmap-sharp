// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from VLFeat (BSD-2-Clause, see THIRD_PARTY_NOTICES.md) as vendored by COLMAP.
//
// VlCovDet: VLFeat's covariant feature detector (thirdparty/VLFeat/covdet.h and covdet.c) with
// the DoG method, the one COLMAP's CovariantSiftCPUFeatureExtractor creates
// (Feature/CovariantSift.cs). This part holds the detector state, vl_covdet_put_image, the
// DoG response, vl_find_local_extrema_3, vl_refine_local_extreum_3 and vl_covdet_detect with
// its non-extrema suppression. VlCovDet.Shape.cs has patch extraction, affine adaptation and
// orientation assignment. The Gaussian scale space is VlScaleSpace.cs.
//
// Not ported: the Hessian, Harris and Laplacian methods (and the Laplacian filter bank
// vl_covdet_new precomputes for them), the 2D extremum search they use, feature dropping by
// margin, and the transposed convention. COLMAP uses none of them.
//
// Tier A (exact). Feature frames and scores are float (VlFrameOrientedEllipse,
// VlCovDetFeature, VlCovDetExtremum3 all store float), computation is double; every point
// where the C rounds to float is an explicit cast here. VL_MAX / VL_MIN are ternaries, not
// Math.Max / Math.Min, because they differ on NaN. Checked against VLFeat compiled without
// FMA contraction by VlCovDetTests (oracle/sift_harness.c).
//
// Cancellation (not in VLFeat): PutImage checks its token before each octave, Detect before
// each octave's extremum scan and every 256 features of the suppression pass. The checks
// never touch the numbers.

namespace ColmapSharp.Feature.VLFeat;

/// <summary>Port of VlFrameOrientedEllipse: an affine frame mapping the unit circle to the feature.</summary>
public struct VlFrameOrientedEllipse
{
	/// <summary>Center x-coordinate.</summary>
	public float X;

	/// <summary>Center y-coordinate.</summary>
	public float Y;

	/// <summary>A(1,1).</summary>
	public float A11;

	/// <summary>A(1,2).</summary>
	public float A12;

	/// <summary>A(2,1).</summary>
	public float A21;

	/// <summary>A(2,2).</summary>
	public float A22;
}

/// <summary>Port of VlCovDetFeature.</summary>
public struct VlCovDetFeature
{
	/// <summary>Feature frame.</summary>
	public VlFrameOrientedEllipse Frame;

	/// <summary>Detected octave.</summary>
	public int O;

	/// <summary>Octave subdivision (the rounded level index within the DoG octave).</summary>
	public int S;

	/// <summary>Peak score.</summary>
	public float PeakScore;

	/// <summary>Edge score.</summary>
	public float EdgeScore;

	/// <summary>Orientation score.</summary>
	public float OrientationScore;

	/// <summary>Laplacian scale score (unused by the DoG method).</summary>
	public float LaplacianScaleScore;
}

/// <summary>Port of VlCovDet with VL_COVDET_METHOD_DOG.</summary>
public sealed partial class VlCovDet
{
	// VL_COVDET_* constants of covdet.c.
	private const int MaxNumOrientations = 4;
	private const int AaPatchResolution = 20;
	private const int AaMaxNumIterations = 15;
	private const int OrNumOrientationHistogramBins = 36;
	private const double AaRelativeIntegrationSigma = 3;
	private const double AaRelativeDerivativeSigma = 1;
	private const double AaMaxAnisotropy = 5;
	private const double AaConvergenceThreshold = 1.001;
	private const double AaPatchExtent = 3 * AaRelativeIntegrationSigma;
	private const double OrAdditionalPeaksRelativeSize = 0.8;
	private const int AaPatchSide = (2 * AaPatchResolution) + 1;

	private readonly float[] aaPatch = new float[AaPatchSide * AaPatchSide];
	private readonly float[] aaPatchX = new float[AaPatchSide * AaPatchSide];
	private readonly float[] aaPatchY = new float[AaPatchSide * AaPatchSide];
	private readonly float[] aaMask = new float[AaPatchSide * AaPatchSide];
	private readonly double[] orientationAngles = new double[MaxNumOrientations];
	private readonly double[] orientationScores = new double[MaxNumOrientations];

	private VlScaleSpace? gss;
	private VlScaleSpace? css;
	private VlCovDetFeature[] features = [];
	private float[] patch = [];

	/// <summary>Port of vl_covdet_new(VL_COVDET_METHOD_DOG).</summary>
	public VlCovDet()
	{
		OctaveResolution = 3;
		FirstOctave = -1;
		PeakThreshold = 0.01; // VL_COVDET_DOG_DEF_PEAK_THRESHOLD
		EdgeThreshold = 10.0; // VL_COVDET_DOG_DEF_EDGE_THRESHOLD
		NonExtremaSuppression = 0.5;

		const int W = AaPatchResolution;
		double step = (2.0 * AaPatchExtent) / ((2 * W) + 1);
		double sigma = AaRelativeIntegrationSigma;
		for (int j = -W; j <= W; ++j)
		{
			for (int i = -W; i <= W; ++i)
			{
				double dx = i * step / sigma;
				double dy = j * step / sigma;
				aaMask[(i + W) + (((2 * W) + 1) * (j + W))] = (float)Math.Exp(-0.5 * ((dx * dx) + (dy * dy)));
			}
		}
	}

	/// <summary>Index of the first octave (vl_covdet_set_first_octave).</summary>
	public int FirstOctave { get; set; }

	/// <summary>Number of levels per octave (vl_covdet_set_octave_resolution).</summary>
	public int OctaveResolution { get; set; }

	/// <summary>Peak threshold (vl_covdet_set_peak_threshold).</summary>
	public double PeakThreshold { get; set; }

	/// <summary>Edge threshold (vl_covdet_set_edge_threshold).</summary>
	public double EdgeThreshold { get; set; }

	/// <summary>Non-extrema suppression tolerance; zero disables it.</summary>
	public double NonExtremaSuppression { get; set; }

	/// <summary>Number of features suppressed by the last Detect.</summary>
	public int NumNonExtremaSuppressed { get; private set; }

	/// <summary>Port of vl_covdet_get_num_features.</summary>
	public int NumFeatures { get; private set; }

	/// <summary>Port of vl_covdet_get_features: the stored features, which may be edited in place.</summary>
	public Span<VlCovDetFeature> Features => features.AsSpan(0, NumFeatures);

	/// <summary>Port of vl_covdet_get_gss: the Gaussian scale space of the last image.</summary>
	public VlScaleSpace? Gss => gss;

	/// <summary>Port of vl_covdet_append_feature.</summary>
	public void AppendFeature(in VlCovDetFeature feature)
	{
		NumFeatures++;
		if (NumFeatures > features.Length)
		{
			Array.Resize(ref features, NumFeatures + 1000);
		}

		features[NumFeatures - 1] = feature;
	}

	/// <summary>
	/// Port of vl_covdet_put_image: builds the Gaussian scale space of a row-major float image,
	/// down to octaves of at least 16 pixels on the shorter side.
	/// </summary>
	public void PutImage(ReadOnlySpan<float> image, int width, int height, CancellationToken cancellationToken = default)
	{
		const int MinOctaveSize = 16;
		Util.Check.That(width >= 1);
		Util.Check.That(height >= 1);
		VlScaleSpaceGeometry geom = VlScaleSpaceGeometry.Default(width, height);

		// (minOctaveSize - 1) 2^lastOctave <= min(width,height) - 1
		int lastOctave = (int)VlMathOp.FloorD(
			VlMathOp.Log2D(VlMin((double)width - 1, (double)height - 1) / (MinOctaveSize - 1)));

		geom.Width = width;
		geom.Height = height;
		geom.FirstOctave = FirstOctave;
		geom.LastOctave = lastOctave;
		geom.OctaveResolution = OctaveResolution;
		geom.OctaveFirstSubdivision = -1;
		geom.OctaveLastSubdivision = OctaveResolution + 1;

		gss = new VlScaleSpace(geom);
		gss.PutImage(image, cancellationToken);
	}

	/// <summary>
	/// Port of vl_covdet_detect: DoG scale-space extrema, refined and thresholded, from the
	/// coarsest octave down, stopping after the octave in which the count reaches
	/// <paramref name="maxNumFeatures"/>; then non-extrema suppression.
	/// </summary>
	public void Detect(long maxNumFeatures, CancellationToken cancellationToken = default)
	{
		VlScaleSpace g = Util.Check.NotNull(gss);
		VlScaleSpaceGeometry geom = g.Geometry;

		// Clear previous detections if any.
		NumFeatures = 0;

		// Prepare buffers.
		VlScaleSpaceGeometry cgeom = geom;
		cgeom.OctaveLastSubdivision -= 1;
		css = new VlScaleSpace(cgeom);

		// Compute cornerness: the DoG, level s minus level s + 1.
		for (int o = cgeom.FirstOctave; o <= cgeom.LastOctave; ++o)
		{
			VlScaleSpaceOctaveGeometry oct = css.GetOctaveGeometry(o);
			for (int s = cgeom.OctaveFirstSubdivision; s <= cgeom.OctaveLastSubdivision; ++s)
			{
				(float[] level, int levelOffset) = g.GetLevel(o, s);
				(float[] next, int nextOffset) = g.GetLevel(o, s + 1);
				(float[] clevel, int clevelOffset) = css.GetLevel(o, s);
				int n = oct.Width * oct.Height;
				for (int k = 0; k < n; ++k)
				{
					clevel[clevelOffset + k] = level[levelOffset + k] - next[nextOffset + k];
				}
			}
		}

		// Find and refine local maxima.
		var extrema = new List<int>();
		for (int o = cgeom.LastOctave; o >= cgeom.FirstOctave; --o)
		{
			cancellationToken.ThrowIfCancellationRequested();
			VlScaleSpaceOctaveGeometry octgeom = css.GetOctaveGeometry(o);
			double step = octgeom.Step;
			int width = octgeom.Width;
			int height = octgeom.Height;
			int depth = cgeom.OctaveLastSubdivision - cgeom.OctaveFirstSubdivision + 1;

			(float[] octave, int octaveOffset) = css.GetLevel(o, cgeom.OctaveFirstSubdivision);
			FindLocalExtrema3(extrema, octave, octaveOffset, width, height, depth, 0.8 * PeakThreshold);
			for (int index = 0; index < extrema.Count; index += 3)
			{
				bool ok = RefineLocalExtremum3(
					out Extremum3 refined, octave, octaveOffset, width, height, depth,
					extrema[index], extrema[index + 1], extrema[index + 2]);
				ok &= Math.Abs((double)refined.PeakScore) > PeakThreshold;
				ok &= refined.EdgeScore < EdgeThreshold;
				if (ok)
				{
					// refined.z + octaveFirstSubdivision is float, and so is its quotient by
					// the (unsigned) resolution and the sum with o: only pow is double.
					float exponent = (float)o + ((refined.Z + (float)cgeom.OctaveFirstSubdivision) / (float)cgeom.OctaveResolution);
					double sigma = cgeom.BaseScale * Math.Pow(2.0, exponent);
					var feature = new VlCovDetFeature
					{
						Frame = new VlFrameOrientedEllipse
						{
							X = (float)(refined.X * step),
							Y = (float)(refined.Y * step),
							A11 = (float)sigma,
							A12 = 0.0f,
							A21 = 0.0f,
							A22 = (float)sigma,
						},
						O = o,
						S = (int)Math.Round((double)refined.Z, MidpointRounding.AwayFromZero),
						PeakScore = refined.PeakScore,
						EdgeScore = refined.EdgeScore,
					};
					AppendFeature(feature);
				}
			}

			if (NumFeatures >= maxNumFeatures)
			{
				break;
			}
		}

		if (NonExtremaSuppression != 0)
		{
			SuppressNonExtrema(cancellationToken);
		}
	}

	// VL_MIN / VL_MAX: plain ternaries (NaN handling differs from Math.Min / Math.Max).
	private static double VlMin(double a, double b) => a < b ? a : b;

	private static double VlMax(double a, double b) => a > b ? a : b;

	// The non-extrema suppression of vl_covdet_detect: a feature is dropped when a stronger one
	// sits within tol * sigma of it at a scale within a factor 1 + tol.
	private void SuppressNonExtrema(CancellationToken cancellationToken)
	{
		double tol = NonExtremaSuppression;
		NumNonExtremaSuppressed = 0;
		for (int i = 0; i < NumFeatures; ++i)
		{
			if ((i & 255) == 0)
			{
				cancellationToken.ThrowIfCancellationRequested();
			}

			double x = features[i].Frame.X;
			double y = features[i].Frame.Y;
			double sigma = features[i].Frame.A11;
			double score = features[i].PeakScore;
			if (score == 0)
			{
				continue;
			}

			for (int j = 0; j < NumFeatures; ++j)
			{
				double dx_ = features[j].Frame.X - x;
				double dy_ = features[j].Frame.Y - y;
				double sigma_ = features[j].Frame.A11;
				double score_ = features[j].PeakScore;
				if (score_ == 0)
				{
					continue;
				}

				if (sigma < (1 + tol) * sigma_ &&
					sigma_ < (1 + tol) * sigma &&
					Math.Abs(dx_) < tol * sigma &&
					Math.Abs(dy_) < tol * sigma &&
					Math.Abs(score) > Math.Abs(score_))
				{
					features[j].PeakScore = 0;
					NumNonExtremaSuppressed++;
				}
			}
		}

		int kept = 0;
		for (int i = 0; i < NumFeatures; ++i)
		{
			VlCovDetFeature feature = features[i];
			if (features[i].PeakScore != 0)
			{
				features[kept++] = feature;
			}
		}

		NumFeatures = kept;
	}

	// Port of VlCovDetExtremum3 (all float but the integer location).
	private struct Extremum3
	{
		public int Xi;
		public int Yi;
		public int Zi;
		public float X;
		public float Y;
		public float Z;
		public float PeakScore;
		public float EdgeScore;
	}

	// Port of vl_find_local_extrema_3: (x, y, z) triples of the interior samples that are
	// strictly above all 26 neighbours and >= threshold, or strictly below and <= -threshold,
	// in x-fastest order.
	private static void FindLocalExtrema3(List<int> extrema, float[] map, int offset, int width, int height, int depth, double threshold)
	{
		extrema.Clear();
		int yo = width;
		int zo = width * height;
		for (int z = 1; z < depth - 1; ++z)
		{
			for (int y = 1; y < height - 1; ++y)
			{
				for (int x = 1; x < width - 1; ++x)
				{
					int pt = offset + x + (y * yo) + (z * zo);
					float value = map[pt];
					if ((value >= threshold && IsGreaterThanNeighbors3(map, pt, value, yo, zo)) ||
						(value <= -threshold && IsLessThanNeighbors3(map, pt, value, yo, zo)))
					{
						extrema.Add(x);
						extrema.Add(y);
						extrema.Add(z);
					}
				}
			}
		}
	}

	private static bool IsGreaterThanNeighbors3(float[] m, int p, float v, int yo, int zo)
	{
		for (int dz = -1; dz <= 1; ++dz)
		{
			for (int dy = -1; dy <= 1; ++dy)
			{
				for (int dx = -1; dx <= 1; ++dx)
				{
					if ((dx != 0 || dy != 0 || dz != 0) && !(v > m[p + dx + (dy * yo) + (dz * zo)]))
					{
						return false;
					}
				}
			}
		}

		return true;
	}

	private static bool IsLessThanNeighbors3(float[] m, int p, float v, int yo, int zo)
	{
		for (int dz = -1; dz <= 1; ++dz)
		{
			for (int dy = -1; dy <= 1; ++dy)
			{
				for (int dx = -1; dx <= 1; ++dx)
				{
					if ((dx != 0 || dy != 0 || dz != 0) && !(v < m[p + dx + (dy * yo) + (dz * zo)]))
					{
						return false;
					}
				}
			}
		}

		return true;
	}

	// Port of vl_refine_local_extreum_3: up to five Newton steps on the 3D quadratic fit,
	// moving the integer x/y location while the offset exceeds 0.6; returns whether the
	// refinement was stable and inside the map.
	private static bool RefineLocalExtremum3(
		out Extremum3 refined, float[] map, int offset, int width, int height, int depth, int x, int y, int z)
	{
		int yo = width;
		int zo = width * height;
		double dX = 0, dY = 0, dZ = 0, dXX = 0, dYY = 0, dZZ = 0, dXY = 0;
		Span<double> a = stackalloc double[9];
		Span<double> b = stackalloc double[3];
		int dx = 0;
		int dy = 0;
		int err = VlMathOp.ErrOk;
		int pt = 0;

		float At(int ddx, int ddy, int ddz) => map[pt + ddx + (ddy * yo) + (ddz * zo)];

		for (int iter = 0; iter < 5; ++iter)
		{
			x += dx;
			y += dy;
			pt = offset + x + (y * yo) + (z * zo);

			// Compute the gradient.
			dX = 0.5 * (At(+1, 0, 0) - At(-1, 0, 0));
			dY = 0.5 * (At(0, +1, 0) - At(0, -1, 0));
			dZ = 0.5 * (At(0, 0, +1) - At(0, 0, -1));

			// Compute the Hessian. The first two samples add in float.
			dXX = (At(+1, 0, 0) + At(-1, 0, 0)) - (2.0 * At(0, 0, 0));
			dYY = (At(0, +1, 0) + At(0, -1, 0)) - (2.0 * At(0, 0, 0));
			dZZ = (At(0, 0, +1) + At(0, 0, -1)) - (2.0 * At(0, 0, 0));

			dXY = 0.25 * (At(+1, +1, 0) + At(-1, -1, 0) - At(-1, +1, 0) - At(+1, -1, 0));
			double dXZ = 0.25 * (At(+1, 0, +1) + At(-1, 0, -1) - At(-1, 0, +1) - At(+1, 0, -1));
			double dYZ = 0.25 * (At(0, +1, +1) + At(0, -1, -1) - At(0, -1, +1) - At(0, +1, -1));

			// Solve the linear system (column-major A).
			a[0] = dXX;
			a[4] = dYY;
			a[8] = dZZ;
			a[3] = a[1] = dXY;
			a[6] = a[2] = dXZ;
			a[7] = a[5] = dYZ;

			b[0] = -dX;
			b[1] = -dY;
			b[2] = -dZ;

			err = VlMathOp.SolveLinearSystem3(b, a, b);

			if (err != VlMathOp.ErrOk)
			{
				b.Clear();
				break;
			}

			// Keep going if there is sufficient translation.
			dx = (b[0] > 0.6 && x < width - 2 ? 1 : 0) + (b[0] < -0.6 && x > 1 ? -1 : 0);
			dy = (b[1] > 0.6 && y < height - 2 ? 1 : 0) + (b[1] < -0.6 && y > 1 ? -1 : 0);

			if (dx == 0 && dy == 0)
			{
				break;
			}
		}

		// Check threshold and other conditions.
		double peakScore = At(0, 0, 0) + (0.5 * ((dX * b[0]) + (dY * b[1]) + (dZ * b[2])));
		double alpha = (dXX + dYY) * (dXX + dYY) / ((dXX * dYY) - (dXY * dXY));
		double edgeScore = alpha < 0
			? double.PositiveInfinity // not an extremum
			: ((0.5 * alpha) - 1) + Math.Sqrt(VlMax((0.25 * alpha) - 1, 0) * alpha);

		refined = new Extremum3
		{
			Xi = x,
			Yi = y,
			Zi = z,
			X = (float)(x + b[0]),
			Y = (float)(y + b[1]),
			Z = (float)(z + b[2]),
			PeakScore = (float)peakScore,
			EdgeScore = (float)edgeScore,
		};

		return err == VlMathOp.ErrOk &&
			Math.Abs(b[0]) < 1.5 &&
			Math.Abs(b[1]) < 1.5 &&
			Math.Abs(b[2]) < 1.5 &&
			refined.X >= 0 && refined.X <= width - 1 &&
			refined.Y >= 0 && refined.Y <= height - 1 &&
			refined.Z >= 0 && refined.Z <= depth - 1;
	}
}
