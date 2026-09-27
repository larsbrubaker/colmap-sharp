// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatchLikelihood: LikelihoodComputer, FindMinCost and TransformPDFToCDF from
// patch_match_cuda.cu - the hidden-Markov model over "source image sees this pixel" of
// Schönberger et al., "Pixelwise View Selection for Unstructured Multi-View Stereo" (ECCV
// 2016): forward/backward messages along a column, the selection probability, and the
// NCC, triangulation-angle, incident-angle and resolution priors. The sweep (a later
// slice) combines them; PatchMatchKernel.Geometry.cs and PatchMatchKernel.Photometric.cs
// hold the rest of the kernel math. Tests: ColmapSharp.Tests/Mvs/PatchMatchKernelTests.cs
// (C#-only).
//
// Float math in COLMAP's order with MathF, and CUDA's NaN-ignoring float min/max as
// PatchMatchKernel.CudaMin/CudaMax (docs/CPP_DIVERGENCES.md, entry 96). The NCC
// normalization factor mixes float and double exactly as the CUDA expression does (M_PI is
// a double), and erff, which .NET lacks, is computed here in double from the series
// erf(x) = 2/sqrt(pi) exp(-x²) sum_n 2^n x^(2n+1) / (1·3·...·(2n+1)) (Abramowitz and
// Stegun, Handbook of Mathematical Functions, 7.1.6) and rounded to float.

namespace ColmapSharp.Mvs;

/// <summary>Port of LikelihoodComputer (patch_match_cuda.cu).</summary>
public readonly struct PatchMatchLikelihood
{
	private readonly float cosMinTriangulationAngle;
	private readonly float invIncidentAngleSigmaSquare;
	private readonly float invNccSigmaSquare;
	private readonly float nccNormFactor;

	/// <summary>
	/// The model for NCC spread <paramref name="nccSigma"/>, minimum triangulation angle
	/// <paramref name="minTriangulationAngle"/> in radians, and incident angle spread
	/// <paramref name="incidentAngleSigma"/>.
	/// </summary>
	public PatchMatchLikelihood(float nccSigma, float minTriangulationAngle, float incidentAngleSigma)
	{
		cosMinTriangulationAngle = MathF.Cos(minTriangulationAngle);
		invIncidentAngleSigmaSquare = -0.5f / (incidentAngleSigma * incidentAngleSigma);
		invNccSigmaSquare = -0.5f / (nccSigma * nccSigma);
		nccNormFactor = ComputeNCCCostNormFactor(nccSigma);
	}

	// The four constants, which the GPU path (PatchMatchGpu.cs) packs into its PmLikelihood
	// uniform so the shaders use exactly these floats.
	internal float CosMinTriangulationAngle => cosMinTriangulationAngle;

	internal float InvIncidentAngleSigmaSquare => invIncidentAngleSigmaSquare;

	internal float InvNccSigmaSquare => invNccSigmaSquare;

	internal float NccNormFactor => nccNormFactor;

	/// <summary>
	/// Compute forward message from current cost and forward message of previous /
	/// neighboring pixel.
	/// </summary>
	public float ComputeForwardMessage(float cost, float prev) => ComputeMessage(true, cost, prev);

	/// <summary>
	/// Compute backward message from current cost and backward message of previous /
	/// neighboring pixel.
	/// </summary>
	public float ComputeBackwardMessage(float cost, float prev) => ComputeMessage(false, cost, prev);

	/// <summary>
	/// The selection probability from the forward and backward messages, blended with the
	/// previous sweep's probability by <paramref name="prevWeight"/>.
	/// </summary>
	public float ComputeSelProb(float alpha, float beta, float prev, float prevWeight)
	{
		float zn0 = (1.0f - alpha) * (1.0f - beta);
		float zn1 = alpha * beta;
		float curr = zn1 / (zn0 + zn1);
		return prevWeight * prev + (1.0f - prevWeight) * curr;
	}

	/// <summary>Compute NCC probability. Note that cost = 1 - NCC.</summary>
	public float ComputeNCCProb(float cost) => MathF.Exp(cost * cost * invNccSigmaSquare) * nccNormFactor;

	/// <summary>
	/// The triangulation angle prior: 1 at or above the minimum angle, falling to 0 at a
	/// zero angle.
	/// </summary>
	public float ComputeTriProb(float cosTriangulationAngle)
	{
		if (cosTriangulationAngle > cosMinTriangulationAngle)
		{
			float scaled = 1.0f - (1.0f - cosTriangulationAngle) / (1.0f - cosMinTriangulationAngle);
			float likelihood = 1.0f - scaled * scaled;
			return PatchMatchKernel.CudaMin(1.0f, PatchMatchKernel.CudaMax(0.0f, likelihood));
		}

		return 1.0f;
	}

	/// <summary>The incident angle prior: 1 head-on, decreasing towards grazing views.</summary>
	public float ComputeIncProb(float cosIncidentAngle)
	{
		float x = 1.0f - PatchMatchKernel.CudaMax(0.0f, cosIncidentAngle);
		return MathF.Exp(x * x * invIncidentAngleSigmaSquare);
	}

	/// <summary>
	/// The warping/resolution prior: the ratio (at most 1) of the areas of the
	/// (2 windowRadius + 1)² reference patch around (row, col) and its image under
	/// <paramref name="h"/>.
	/// </summary>
	public float ComputeResolutionProb(ReadOnlySpan<float> h, float row, float col, int windowRadius)
	{
		int windowSize = 2 * windowRadius + 1;

		// Warp corners of patch in reference image to source image.
		(float src1X, float src1Y) = PatchMatchKernel.Mat33DotVec3Homogeneous(h, col - windowRadius, row - windowRadius);
		(float src2X, float src2Y) = PatchMatchKernel.Mat33DotVec3Homogeneous(h, col - windowRadius, row + windowRadius);
		(float src3X, float src3Y) = PatchMatchKernel.Mat33DotVec3Homogeneous(h, col + windowRadius, row + windowRadius);
		(float src4X, float src4Y) = PatchMatchKernel.Mat33DotVec3Homogeneous(h, col + windowRadius, row - windowRadius);

		// Compute area of patches in reference and source image.
		float refArea = windowSize * windowSize;
		float srcArea = MathF.Abs(0.5f * (src1X * src2Y - src2X * src1Y - src1X * src4Y + src2X * src3Y
			- src3X * src2Y + src4X * src1Y + src3X * src4Y - src4X * src3Y));

		return refArea > srcArea ? srcArea / refArea : refArea / srcArea;
	}

	/// <summary>
	/// The normalization for the likelihood function, i.e. the normalization for the prior
	/// on the matching cost: 2 / (sqrt(2 pi) sigma erf(2 / (sigma sqrt 2))), so the NCC
	/// probability integrates to 1 over costs in [0, 2].
	/// </summary>
	internal static float ComputeNCCCostNormFactor(float nccSigma)
	{
		// sqrt(2.0f * M_PI) is a double expression, so the product and quotient are double.
		float erf = ErfF(2.0f / (nccSigma * 1.414213562f));
		return (float)(2.0f / (Math.Sqrt(2.0f * Math.PI) * nccSigma * erf));
	}

	/// <summary>erff: the error function, rounded to float.</summary>
	internal static float ErfF(float x)
	{
		double ax = Math.Abs((double)x);
		if (double.IsNaN(ax))
		{
			return float.NaN;
		}

		if (ax == 0.0)
		{
			return x;
		}

		// Beyond 6, erf is 1 to far better than float precision.
		if (ax >= 6.0)
		{
			return x < 0 ? -1.0f : 1.0f;
		}

		// All terms are positive, so the sum has no cancellation.
		double x2 = ax * ax;
		double term = ax;
		double sum = ax;
		for (int n = 1; n < 200; ++n)
		{
			term *= 2.0 * x2 / (2 * n + 1);
			sum += term;
			if (term < sum * 1e-17)
			{
				break;
			}
		}

		double erf = 2.0 / Math.Sqrt(Math.PI) * Math.Exp(-x2) * sum;
		return (float)(x < 0 ? -erf : erf);
	}

	// Compute the forward or backward message.
	private float ComputeMessage(bool forward, float cost, float prev)
	{
		const float UniformProb = 0.5f;
		const float NoChangeProb = 0.99999f;
		const float ChangeProb = 1.0f - NoChangeProb;
		float emission = ComputeNCCProb(cost);

		float zn0; // Message for selection probability = 0.
		float zn1; // Message for selection probability = 1.
		if (forward)
		{
			zn0 = (prev * ChangeProb + (1.0f - prev) * NoChangeProb) * UniformProb;
			zn1 = (prev * NoChangeProb + (1.0f - prev) * ChangeProb) * emission;
		}
		else
		{
			zn0 = prev * emission * ChangeProb + (1.0f - prev) * UniformProb * NoChangeProb;
			zn1 = prev * emission * NoChangeProb + (1.0f - prev) * UniformProb * ChangeProb;
		}

		return zn1 / (zn0 + zn1);
	}
}

/// <summary>The small array helpers of the PatchMatch sweep.</summary>
public static partial class PatchMatchKernel
{
	/// <summary>Index of the minimum cost; the last one among equal minima (FindMinCost).</summary>
	public static int FindMinCost(ReadOnlySpan<float> costs)
	{
		float minCost = costs[0];
		int minCostIdx = 0;
		for (int idx = 1; idx < costs.Length; ++idx)
		{
			if (costs[idx] <= minCost)
			{
				minCost = costs[idx];
				minCostIdx = idx;
			}
		}

		return minCostIdx;
	}

	/// <summary>Normalizes <paramref name="probs"/> and replaces it with its cumulative sums (TransformPDFToCDF).</summary>
	public static void TransformPDFToCDF(Span<float> probs)
	{
		float probSum = 0.0f;
		for (int i = 0; i < probs.Length; ++i)
		{
			probSum += probs[i];
		}

		float invProbSum = 1.0f / probSum;

		float cumProb = 0.0f;
		for (int i = 0; i < probs.Length; ++i)
		{
			float prob = probs[i] * invProbSum;
			cumProb += prob;
			probs[i] = cumProb;
		}
	}
}
