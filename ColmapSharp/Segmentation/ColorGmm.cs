// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColorGmm: a full-covariance Gaussian mixture over Lab colours, one for the foreground and
// one for the background of GrabCutRefiner. Not a COLMAP port. Written from C. Rother,
// V. Kolmogorov and A. Blake, "GrabCut - Interactive Foreground Extraction using Iterated Graph
// Cuts", SIGGRAPH 2004 (section 3: K components, each pixel assigned its most likely component,
// then the parameters re-estimated from those assignments). The seed is k-means (S. Lloyd,
// "Least squares quantization in PCM", IEEE Trans. IT 28(2), 1982) started from farthest-point
// centres, which needs no random numbers, so a run is deterministic.

namespace ColmapSharp.Segmentation;

internal sealed class ColorGmm
{
	private const int MaxKMeansSamples = 20000;

	private readonly int components;
	private readonly double regularization;
	private readonly double[] logWeightNorm;
	private readonly double[] mean;
	private readonly double[] inverseCovariance;

	private ColorGmm(int components, double regularization)
	{
		this.components = components;
		this.regularization = regularization;
		logWeightNorm = new double[components];
		mean = new double[components * 3];
		inverseCovariance = new double[components * 9];
	}

	public int Components => components;

	/// <summary>
	/// Seeds a mixture on the pixels <paramref name="indices"/> of <paramref name="image"/> with
	/// k-means, then estimates it from the resulting assignment. Needs at least one pixel.
	/// </summary>
	public static ColorGmm Fit(LabImage image, int[] indices, int maxComponents, int kMeansIterations, double regularization)
	{
		int k = Math.Max(1, Math.Min(maxComponents, indices.Length));
		// K-means runs on an evenly strided sample; the final assignment covers every pixel.
		int stride = Math.Max(1, indices.Length / MaxKMeansSamples);
		int sampleCount = (indices.Length + stride - 1) / stride;
		var samples = new double[sampleCount * 3];
		for (int s = 0; s < sampleCount; ++s)
		{
			int p = indices[s * stride];
			samples[s * 3] = image.L[p];
			samples[s * 3 + 1] = image.A[p];
			samples[s * 3 + 2] = image.B[p];
		}

		double[] centres = FarthestPointCentres(samples, sampleCount, k);
		var sampleLabel = new int[sampleCount];
		for (int iteration = 0; iteration < kMeansIterations; ++iteration)
		{
			for (int s = 0; s < sampleCount; ++s)
			{
				sampleLabel[s] = Nearest(centres, k, samples[s * 3], samples[s * 3 + 1], samples[s * 3 + 2]);
			}

			var sums = new double[k * 3];
			var counts = new int[k];
			for (int s = 0; s < sampleCount; ++s)
			{
				int c = sampleLabel[s];
				counts[c]++;
				sums[c * 3] += samples[s * 3];
				sums[c * 3 + 1] += samples[s * 3 + 1];
				sums[c * 3 + 2] += samples[s * 3 + 2];
			}

			for (int c = 0; c < k; ++c)
			{
				// An emptied cluster keeps its old centre.
				if (counts[c] > 0)
				{
					centres[c * 3] = sums[c * 3] / counts[c];
					centres[c * 3 + 1] = sums[c * 3 + 1] / counts[c];
					centres[c * 3 + 2] = sums[c * 3 + 2] / counts[c];
				}
			}
		}

		var assignment = new int[indices.Length];
		for (int i = 0; i < indices.Length; ++i)
		{
			int p = indices[i];
			assignment[i] = Nearest(centres, k, image.L[p], image.A[p], image.B[p]);
		}

		var gmm = new ColorGmm(k, regularization);
		gmm.Estimate(image, indices, assignment);
		return gmm;
	}

	/// <summary>The component under which the colour is most likely (GrabCut step 1).</summary>
	public int MostLikelyComponent(double l, double a, double b)
	{
		int best = 0;
		double bestLog = double.NegativeInfinity;
		for (int c = 0; c < components; ++c)
		{
			double value = ComponentLog(c, l, a, b);
			if (value > bestLog)
			{
				bestLog = value;
				best = c;
			}
		}

		return best;
	}

	/// <summary>-log of the mixture density at the colour (the GrabCut data term).</summary>
	public double NegativeLogLikelihood(double l, double a, double b)
	{
		double max = double.NegativeInfinity;
		Span<double> logs = stackalloc double[components];
		for (int c = 0; c < components; ++c)
		{
			logs[c] = ComponentLog(c, l, a, b);
			if (logs[c] > max)
			{
				max = logs[c];
			}
		}

		if (double.IsNegativeInfinity(max))
		{
			return 1e10;
		}

		double sum = 0.0;
		for (int c = 0; c < components; ++c)
		{
			sum += Math.Exp(logs[c] - max);
		}

		return -(max + Math.Log(sum));
	}

	/// <summary>
	/// Re-estimates weights, means and covariances from the component of each pixel (GrabCut
	/// step 2). A component that got no pixel gets zero weight.
	/// </summary>
	public void Estimate(LabImage image, int[] indices, int[] assignment)
	{
		var counts = new int[components];
		var sums = new double[components * 3];
		for (int i = 0; i < indices.Length; ++i)
		{
			int c = assignment[i];
			int p = indices[i];
			counts[c]++;
			sums[c * 3] += image.L[p];
			sums[c * 3 + 1] += image.A[p];
			sums[c * 3 + 2] += image.B[p];
		}

		for (int c = 0; c < components; ++c)
		{
			if (counts[c] > 0)
			{
				mean[c * 3] = sums[c * 3] / counts[c];
				mean[c * 3 + 1] = sums[c * 3 + 1] / counts[c];
				mean[c * 3 + 2] = sums[c * 3 + 2] / counts[c];
			}
		}

		var covariance = new double[components * 9];
		for (int i = 0; i < indices.Length; ++i)
		{
			int c = assignment[i];
			int p = indices[i];
			double dl = image.L[p] - mean[c * 3];
			double da = image.A[p] - mean[c * 3 + 1];
			double db = image.B[p] - mean[c * 3 + 2];
			int o = c * 9;
			covariance[o] += dl * dl;
			covariance[o + 1] += dl * da;
			covariance[o + 2] += dl * db;
			covariance[o + 4] += da * da;
			covariance[o + 5] += da * db;
			covariance[o + 8] += db * db;
		}

		for (int c = 0; c < components; ++c)
		{
			if (counts[c] == 0)
			{
				logWeightNorm[c] = double.NegativeInfinity;
				continue;
			}

			int o = c * 9;
			double n = counts[c];
			double s00 = covariance[o] / n + regularization;
			double s01 = covariance[o + 1] / n;
			double s02 = covariance[o + 2] / n;
			double s11 = covariance[o + 4] / n + regularization;
			double s12 = covariance[o + 5] / n;
			double s22 = covariance[o + 8] / n + regularization;

			// Inverse of the symmetric 3x3 by cofactors.
			double c00 = s11 * s22 - s12 * s12;
			double c01 = s02 * s12 - s01 * s22;
			double c02 = s01 * s12 - s02 * s11;
			double c11 = s00 * s22 - s02 * s02;
			double c12 = s01 * s02 - s00 * s12;
			double c22 = s00 * s11 - s01 * s01;
			double det = s00 * c00 + s01 * c01 + s02 * c02;
			inverseCovariance[o] = c00 / det;
			inverseCovariance[o + 1] = c01 / det;
			inverseCovariance[o + 2] = c02 / det;
			inverseCovariance[o + 3] = c01 / det;
			inverseCovariance[o + 4] = c11 / det;
			inverseCovariance[o + 5] = c12 / det;
			inverseCovariance[o + 6] = c02 / det;
			inverseCovariance[o + 7] = c12 / det;
			inverseCovariance[o + 8] = c22 / det;

			double weight = n / indices.Length;
			logWeightNorm[c] = Math.Log(weight) - 0.5 * Math.Log(det) - 1.5 * Math.Log(2.0 * Math.PI);
		}
	}

	// log(weight_c * N(x | mean_c, cov_c)).
	private double ComponentLog(int c, double l, double a, double b)
	{
		if (double.IsNegativeInfinity(logWeightNorm[c]))
		{
			return double.NegativeInfinity;
		}

		double dl = l - mean[c * 3];
		double da = a - mean[c * 3 + 1];
		double db = b - mean[c * 3 + 2];
		int o = c * 9;
		double m = dl * (inverseCovariance[o] * dl + inverseCovariance[o + 1] * da + inverseCovariance[o + 2] * db)
			+ da * (inverseCovariance[o + 3] * dl + inverseCovariance[o + 4] * da + inverseCovariance[o + 5] * db)
			+ db * (inverseCovariance[o + 6] * dl + inverseCovariance[o + 7] * da + inverseCovariance[o + 8] * db);
		return logWeightNorm[c] - 0.5 * m;
	}

	// First centre: the sample nearest the mean; each next one: the sample farthest from the
	// centres so far. Ties go to the lower sample index.
	private static double[] FarthestPointCentres(double[] samples, int count, int k)
	{
		double ml = 0.0, ma = 0.0, mb = 0.0;
		for (int s = 0; s < count; ++s)
		{
			ml += samples[s * 3];
			ma += samples[s * 3 + 1];
			mb += samples[s * 3 + 2];
		}

		ml /= count;
		ma /= count;
		mb /= count;
		var centres = new double[k * 3];
		var nearest = new double[count];
		int first = 0;
		double firstDistance = double.PositiveInfinity;
		for (int s = 0; s < count; ++s)
		{
			double d = Distance2(samples, s, ml, ma, mb);
			if (d < firstDistance)
			{
				firstDistance = d;
				first = s;
			}
		}

		int chosen = first;
		for (int c = 0; c < k; ++c)
		{
			centres[c * 3] = samples[chosen * 3];
			centres[c * 3 + 1] = samples[chosen * 3 + 1];
			centres[c * 3 + 2] = samples[chosen * 3 + 2];
			double farthest = -1.0;
			int next = 0;
			for (int s = 0; s < count; ++s)
			{
				double d = Distance2(samples, s, centres[c * 3], centres[c * 3 + 1], centres[c * 3 + 2]);
				nearest[s] = c == 0 ? d : Math.Min(nearest[s], d);
				if (nearest[s] > farthest)
				{
					farthest = nearest[s];
					next = s;
				}
			}

			chosen = next;
		}

		return centres;
	}

	private static int Nearest(double[] centres, int k, double l, double a, double b)
	{
		int best = 0;
		double bestDistance = double.PositiveInfinity;
		for (int c = 0; c < k; ++c)
		{
			double dl = l - centres[c * 3], da = a - centres[c * 3 + 1], db = b - centres[c * 3 + 2];
			double d = dl * dl + da * da + db * db;
			if (d < bestDistance)
			{
				bestDistance = d;
				best = c;
			}
		}

		return best;
	}

	private static double Distance2(double[] samples, int s, double l, double a, double b)
	{
		double dl = samples[s * 3] - l, da = samples[s * 3 + 1] - a, db = samples[s * 3 + 2] - b;
		return dl * dl + da * da + db * db;
	}
}
