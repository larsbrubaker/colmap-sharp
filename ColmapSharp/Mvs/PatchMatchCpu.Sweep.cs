// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatchCpu.Sweep: the SweepFromTopToBottom kernel of patch_match_cuda.cu on the CPU -
// one top-to-bottom pass over a column of the (rotated) reference image that computes the
// backward messages, then per row proposes five depth/normal hypotheses (current,
// propagated from the row above, random, and the two mixes), scores them against source
// images drawn by Monte Carlo sampling from the selection probabilities, keeps the best,
// updates costs, forward messages and selection probabilities, and on the last sweep
// optionally filters the pixel and records its consistent source images.
// PatchMatchCpu.cs owns the maps and the schedule; the helpers are in
// PatchMatchKernel.Geometry.cs, PatchMatchKernel.Photometric.cs and PatchMatchLikelihood.cs.
//
// Columns are independent: each reads and writes only its own pixels (the photo-consistency
// window reads the read-only reference image), so Parallel.For over columns gives the same
// result for any thread count. Random draws come from PatchMatchRandom keyed on the pixel of
// the original reference image and the sweep (docs/CPP_DIVERGENCES.md, entry 86).
// Hypotheses 1-4 of each Monte Carlo sample are scored together by
// PatchMatchPhotoConsistency.ComputeFour (SIMD lanes, bit-identical to four scalar calls).

namespace ColmapSharp.Mvs;

internal sealed partial class PatchMatchCpu
{
	/// <summary>Port of SweepOptions (patch_match_cuda.cu), the per-sweep kernel settings.</summary>
	private struct SweepOptions
	{
		public float Perturbation;
		public float PrevSelProbWeight;
		public int Phase;
		public bool GeomConsistencyTerm;
		public bool FilterPhotoConsistency;
		public bool FilterGeomConsistency;
	}

	// Number of hypotheses per pixel (kNumCosts).
	private const int NumCosts = 5;

	// Probability for boundary pixels.
	private const float UniformProb = 0.5f;

	// FLT_EPSILON.
	private const float FloatEpsilon = 1.1920929e-07f;

	private void SweepColumn(int col, in SweepOptions sweep, PatchMatchPhotoConsistency pcc, float[] scratch)
	{
		int width = costMap.GetWidth();
		int height = costMap.GetHeight();
		int numImages = costMap.GetDepth();
		int planeSize = width * height;
		float[] costs = costMap.Data;
		float[] depths = depthMap.Data;
		float[] normals = normalMap.Data;
		float[] selProbs = selProbMap.Data;
		float[] prevSelProbs = prevSelProbMap.Data;
		float[] poses = transforms.Poses(rotation);
		PatchMatchFrame frame = pcc.Frame;

		Span<float> forwardMessage = scratch.AsSpan(0, numImages);
		Span<float> samplingProbs = scratch.AsSpan(numImages, numImages);
		int windowCount = pcc.WindowCount;
		Span<float> refColors = scratch.AsSpan(2 * numImages, windowCount);
		Span<float> windowWeights = scratch.AsSpan(2 * numImages + windowCount, windowCount);

		// Compute backward message for all rows. Note that the backward messages are
		// temporarily stored in the sel_prob_map and replaced row by row as the updated
		// forward messages are computed further below.
		for (int imageIdx = 0; imageIdx < numImages; ++imageIdx)
		{
			float beta = UniformProb;
			for (int row = height - 1; row >= 0; --row)
			{
				int idx = imageIdx * planeSize + row * width + col;
				beta = likelihood.ComputeBackwardMessage(costs[idx], beta);
				selProbs[idx] = beta;
			}

			// Initialize forward message.
			forwardMessage[imageIdx] = UniformProb;
		}

		// Parameters of previous, current and randomly sampled pixel states.
		Span<float> prevNormal = stackalloc float[3];
		Span<float> currNormal = stackalloc float[3];
		Span<float> randNormal = stackalloc float[3];
		Span<float> point = stackalloc float[3];
		Span<float> h = stackalloc float[9];
		Span<float> hypothesisCosts = stackalloc float[NumCosts];
		Span<float> hypothesisDepths = stackalloc float[NumCosts];
		Span<float> bestNormal = stackalloc float[3];

		// Hypotheses 1-4 (depths and normals) for the lockstep photo-consistency.
		Span<float> fourNormals = stackalloc float[12];
		Span<float> fourCosts = stackalloc float[4];

		// Parameters for first row in column.
		float prevDepth = depths[col];
		ReadNormal(normals, planeSize, col, prevNormal);

		for (int row = 0; row < height; ++row)
		{
			int pixel = row * width + col;
			(int origRow, int origCol) = PatchMatchTransforms.ToOriginalPixel(rotation, row, col, refWidth, refHeight);
			var random = new PatchMatchRandom(seed, origRow, origCol, sweep.Phase);

			// Propagate the depth at which the current ray intersects with the plane of the
			// normal of the previous ray. This helps to better estimate the depth of very
			// oblique structures, i.e. pixels whose normal direction is significantly
			// different from their viewing direction.
			prevDepth = PatchMatchKernel.PropagateDepth(frame, prevDepth, prevNormal, row - 1, row);

			// Read parameters for current pixel from previous sweep.
			float currDepth = depths[pixel];
			ReadNormal(normals, planeSize, pixel, currNormal);

			// Generate random parameters. perturbation * M_PI is a double product passed as
			// a float argument.
			float randDepth = PatchMatchKernel.PerturbDepth(sweep.Perturbation, currDepth, ref random);
			PatchMatchKernel.PerturbNormal(
				frame, row, col, (float)(sweep.Perturbation * Math.PI), currNormal, ref random, randNormal);

			// Read in the backward message, compute selection probabilities and modulate
			// selection probabilities with priors.
			PatchMatchKernel.ComputePointAtDepth(frame, row, col, currDepth, point);
			for (int imageIdx = 0; imageIdx < numImages; ++imageIdx)
			{
				int idx = imageIdx * planeSize + pixel;
				float alpha = likelihood.ComputeForwardMessage(costs[idx], forwardMessage[imageIdx]);
				float selProb = likelihood.ComputeSelProb(alpha, selProbs[idx], prevSelProbs[idx], sweep.PrevSelProbWeight);

				(float cosTriangulationAngle, float cosIncidentAngle) =
					PatchMatchKernel.ComputeViewingAngles(poses, point, currNormal, imageIdx);
				float triProb = likelihood.ComputeTriProb(cosTriangulationAngle);
				float incProb = likelihood.ComputeIncProb(cosIncidentAngle);

				PatchMatchKernel.ComposeHomography(poses, frame, imageIdx, row, col, currDepth, currNormal, h);
				float resProb = likelihood.ComputeResolutionProb(h, row, col, windowRadius);

				samplingProbs[imageIdx] = selProb * triProb * incProb * resProb;
			}

			PatchMatchKernel.TransformPDFToCDF(samplingProbs);

			// The reference half of the NCC is the same for every hypothesis and source image
			// at this pixel.
			float windowWeightSum = pcc.PrepareWindow(row, col, refColors, windowWeights);

			// Compute matching cost using Monte Carlo sampling of source images. Images with
			// higher selection probability are more likely to be sampled. Hence, if only very
			// few source images see the reference image pixel, the same source image is likely
			// to be sampled many times. Instead of taking the best K probabilities, this
			// sampling scheme has the advantage of being adaptive to any distribution of
			// selection probabilities.
			hypothesisCosts.Clear();
			hypothesisDepths[0] = currDepth;
			hypothesisDepths[1] = prevDepth;
			hypothesisDepths[2] = randDepth;
			hypothesisDepths[3] = currDepth;
			hypothesisDepths[4] = randDepth;

			for (int i = 1; i < NumCosts; ++i)
			{
				HypothesisNormal(i, currNormal, prevNormal, randNormal).CopyTo(fourNormals.Slice(3 * (i - 1), 3));
			}

			ReadOnlySpan<float> fourDepths = hypothesisDepths[1..];

			for (int sample = 0; sample < numSamples; ++sample)
			{
				float randProb = random.NextUniform() - FloatEpsilon;

				int srcImageIdx = -1;
				for (int imageIdx = 0; imageIdx < numImages; ++imageIdx)
				{
					if (samplingProbs[imageIdx] > randProb)
					{
						srcImageIdx = imageIdx;
						break;
					}
				}

				if (srcImageIdx == -1)
				{
					continue;
				}

				hypothesisCosts[0] += costs[srcImageIdx * planeSize + pixel];
				if (sweep.GeomConsistencyTerm)
				{
					hypothesisCosts[0] += geomConsistencyRegularizer * GeomCost(poses, frame, row, col, hypothesisDepths[0], srcImageIdx);
				}

				// The four evaluations are independent; ComputeFour runs them in SIMD lockstep,
				// each bit-identical to the scalar Compute.
				pcc.ComputeFour(row, col, fourDepths, fourNormals, srcImageIdx, refColors, windowWeights, windowWeightSum, fourCosts);
				for (int i = 1; i < NumCosts; ++i)
				{
					hypothesisCosts[i] += fourCosts[i - 1];
					if (sweep.GeomConsistencyTerm)
					{
						hypothesisCosts[i] += geomConsistencyRegularizer * GeomCost(poses, frame, row, col, hypothesisDepths[i], srcImageIdx);
					}
				}
			}

			// Find the parameters of the minimum cost.
			int minCostIdx = PatchMatchKernel.FindMinCost(hypothesisCosts);
			float bestDepth = hypothesisDepths[minCostIdx];
			HypothesisNormal(minCostIdx, currNormal, prevNormal, randNormal).CopyTo(bestNormal);

			// Save best new parameters.
			depths[pixel] = bestDepth;
			WriteNormal(normals, planeSize, pixel, bestNormal);

			// Use the new cost to recompute the updated forward message and the selection
			// probability.
			for (int imageIdx = 0; imageIdx < numImages; ++imageIdx)
			{
				int idx = imageIdx * planeSize + pixel;

				// Determine the cost for best depth.
				float cost;
				if (minCostIdx == 0)
				{
					cost = costs[idx];
				}
				else
				{
					cost = pcc.Compute(row, col, bestDepth, bestNormal, imageIdx, refColors, windowWeights, windowWeightSum);
					costs[idx] = cost;
				}

				float alpha = likelihood.ComputeForwardMessage(cost, forwardMessage[imageIdx]);
				float prob = likelihood.ComputeSelProb(alpha, selProbs[idx], prevSelProbs[idx], sweep.PrevSelProbWeight);
				forwardMessage[imageIdx] = alpha;
				selProbs[idx] = prob;
			}

			if (sweep.FilterPhotoConsistency || sweep.FilterGeomConsistency)
			{
				FilterPixel(sweep, poses, frame, row, col, bestDepth, bestNormal);
			}

			// Update previous depth for next row.
			prevDepth = bestDepth;
			bestNormal.CopyTo(prevNormal);
		}
	}

	/// <summary>
	/// Marks the source images the pixel's best depth and normal are consistent with, and
	/// clears the pixel when fewer than filter_min_num_consistent are.
	/// </summary>
	private void FilterPixel(in SweepOptions sweep, float[] poses, in PatchMatchFrame frame, int row, int col, float bestDepth, ReadOnlySpan<float> bestNormal)
	{
		int width = costMap.GetWidth();
		int planeSize = width * costMap.GetHeight();
		int pixel = row * width + col;
		int numImages = costMap.GetDepth();
		byte[] mask = consistencyMask.Data;
		float[] selProbs = selProbMap.Data;

		int numConsistent = 0;

		Span<float> bestPoint = stackalloc float[3];
		PatchMatchKernel.ComputePointAtDepth(frame, row, col, bestDepth, bestPoint);

		float minNccProb = likelihood.ComputeNCCProb(1.0f - filterMinNcc);
		float cosMinTriangulationAngle = MathF.Cos(filterMinTriangulationAngle);

		for (int imageIdx = 0; imageIdx < numImages; ++imageIdx)
		{
			int idx = imageIdx * planeSize + pixel;
			(float cosTriangulationAngle, float cosIncidentAngle) =
				PatchMatchKernel.ComputeViewingAngles(poses, bestPoint, bestNormal, imageIdx);
			if (cosTriangulationAngle > cosMinTriangulationAngle || cosIncidentAngle <= 0.0f)
			{
				continue;
			}

			bool consistent;
			if (!sweep.FilterGeomConsistency)
			{
				consistent = selProbs[idx] >= minNccProb;
			}
			else if (!sweep.FilterPhotoConsistency)
			{
				consistent = GeomCost(poses, frame, row, col, bestDepth, imageIdx) <= filterGeomConsistencyMaxCost;
			}
			else
			{
				consistent = selProbs[idx] >= minNccProb
					&& GeomCost(poses, frame, row, col, bestDepth, imageIdx) <= filterGeomConsistencyMaxCost;
			}

			if (consistent)
			{
				mask[idx] = 1;
				numConsistent += 1;
			}
		}

		if (numConsistent < filterMinNumConsistent)
		{
			depthMap.Data[pixel] = 0.0f;
			normalMap.Data[pixel] = 0.0f;
			normalMap.Data[planeSize + pixel] = 0.0f;
			normalMap.Data[2 * planeSize + pixel] = 0.0f;
			for (int imageIdx = 0; imageIdx < numImages; ++imageIdx)
			{
				mask[imageIdx * planeSize + pixel] = 0;
			}
		}
	}

	private float GeomCost(float[] poses, in PatchMatchFrame frame, int row, int col, float depth, int imageIdx) =>
		PatchMatchKernel.ComputeGeomConsistencyCost(
			poses, srcDepthMaps!, frame, row, col, depth, imageIdx, geomConsistencyMaxCost);

	// The normal of hypothesis i: {curr, prev, rand, rand, curr}.
	private static ReadOnlySpan<float> HypothesisNormal(int i, Span<float> curr, Span<float> prev, Span<float> rand) => i switch
	{
		0 or 4 => curr,
		1 => prev,
		_ => rand,
	};

	private static void ReadNormal(float[] normals, int planeSize, int pixel, Span<float> normal)
	{
		normal[0] = normals[pixel];
		normal[1] = normals[planeSize + pixel];
		normal[2] = normals[2 * planeSize + pixel];
	}

	private static void WriteNormal(float[] normals, int planeSize, int pixel, ReadOnlySpan<float> normal)
	{
		normals[pixel] = normal[0];
		normals[planeSize + pixel] = normal[1];
		normals[2 * planeSize + pixel] = normal[2];
	}
}
