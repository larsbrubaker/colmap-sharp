// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatchCpu.Sweep: the SweepFromTopToBottom kernel of patch_match_cuda.cu on the CPU -
// one top-to-bottom pass over a column of the (rotated) reference image. BackwardMessages
// computes the backward messages bottom to top and initializes the column's carried state;
// SweepRows then, per row of a band, proposes five depth/normal hypotheses (current,
// propagated from the row above, random, and the two mixes), scores them against source
// images drawn by Monte Carlo sampling from the selection probabilities, keeps the best, and
// updates costs, forward messages and selection probabilities. On the last sweep FilterPixels
// then filters every pixel and records its consistent source images.
// PatchMatchCpu.cs owns the maps and the schedule; the helpers are in
// PatchMatchKernel.Geometry.cs, PatchMatchKernel.Photometric.cs and PatchMatchLikelihood.cs.
//
// Columns are independent: each reads and writes only its own pixels (the photo-consistency
// window reads the read-only reference image), so Parallel.For over columns gives the same
// result for any thread count. Random draws come from PatchMatchRandom keyed on the pixel of
// the original reference image and the sweep (divergence 86).
// Hypotheses 1-4 of each Monte Carlo sample are scored together by
// PatchMatchPhotoConsistency.ComputeFour (SIMD lanes, bit-identical to four scalar calls).
//
// Why the sweep is split into BackwardMessages + SweepRows bands + FilterPixels (the CUDA
// kernel does all three in one per-column loop): the GPU port (PORTING_PLAN.md Phase 13) runs
// each piece as its own dispatch, bands of rows across all columns, so no dispatch runs long.
// Everything a column carries from row to row - the forward message per source image, the
// previous row's best depth and normal - lives in an explicit column state
// (ColumnStateSize floats), so a band can stop and the next one resume bit-identically.
// Filtering moved out of the row loop because on the last sweep it reads only the pixel's
// final selection probabilities and best depth/normal, and the next row propagates from the
// carried best depth and normal rather than the (possibly zeroed) maps, so filtering after
// the sweep gives the same result (pinned by PatchMatchSweepBandTests).

namespace ColmapSharp.Mvs;

internal sealed partial class PatchMatchCpu
{
	/// <summary>Port of SweepOptions (patch_match_cuda.cu), the per-sweep kernel settings.</summary>
	internal struct SweepOptions
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

	// Layout of a column's carried state: the forward message per source image, then the
	// previous row's best depth and normal.
	internal static int ColumnStateSize(int numImages) => numImages + 4;

	// Per-row scratch: sampling probabilities per source image, then the reference window's
	// colors and weights.
	internal static int RowScratchSize(int numImages, int windowCount) => numImages + 2 * windowCount;

	/// <summary>
	/// The backward message pass of SweepFromTopToBottom for column <paramref name="col"/>:
	/// runs bottom to top, storing each message in the selection probability map, and
	/// initializes <paramref name="columnState"/> for the first row (forward messages 0.5, the
	/// top pixel's depth and normal as the previous ones).
	/// </summary>
	private void BackwardMessages(int col, Span<float> columnState)
	{
		for (int imageIdx = 0; imageIdx < costMap.GetDepth(); ++imageIdx)
		{
			BackwardMessage(col, imageIdx, columnState);
		}

		InitColumnPrevious(col, columnState);
	}

	/// <summary>
	/// BackwardMessages for one source image of the column: its messages and its forward
	/// message's start. The GPU's backward_messages runs one per (source image, column).
	/// </summary>
	internal void BackwardMessage(int col, int imageIdx, Span<float> columnState)
	{
		int width = costMap.GetWidth();
		int height = costMap.GetHeight();
		int planeSize = width * height;
		float[] costs = costMap.Data;
		float[] selProbs = selProbMap.Data;

		// Compute backward message for all rows. Note that the backward messages are
		// temporarily stored in the sel_prob_map and replaced row by row as the updated
		// forward messages are computed further below.
		float beta = UniformProb;
		for (int row = height - 1; row >= 0; --row)
		{
			int idx = imageIdx * planeSize + row * width + col;
			beta = likelihood.ComputeBackwardMessage(costs[idx], beta);
			selProbs[idx] = beta;
		}

		// Initialize forward message.
		columnState[imageIdx] = UniformProb;
	}

	/// <summary>
	/// The previous depth and normal of the column state for the first row: the top pixel's.
	/// </summary>
	internal void InitColumnPrevious(int col, Span<float> columnState)
	{
		int numImages = costMap.GetDepth();
		int planeSize = costMap.GetWidth() * costMap.GetHeight();

		// Parameters for first row in column.
		columnState[numImages] = depthMap.Data[col];
		ReadNormal(normalMap.Data, planeSize, col, columnState.Slice(numImages + 1, 3));
	}

	/// <summary>
	/// The row loop of SweepFromTopToBottom for rows [<paramref name="rowStart"/>,
	/// <paramref name="rowEnd"/>) of column <paramref name="col"/>, resuming from and updating
	/// <paramref name="columnState"/> (see ColumnStateSize). BackwardMessages must have run for
	/// the column in this sweep, and the column's bands must run in order.
	/// </summary>
	internal void SweepRows(
		int col, int rowStart, int rowEnd, in SweepOptions sweep, PatchMatchPhotoConsistency pcc,
		Span<float> columnState, Span<float> rowScratch)
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

		Span<float> forwardMessage = columnState[..numImages];
		Span<float> prevNormal = columnState.Slice(numImages + 1, 3);
		Span<float> samplingProbs = rowScratch[..numImages];
		int windowCount = pcc.WindowCount;
		Span<float> refColors = rowScratch.Slice(numImages, windowCount);
		Span<float> windowWeights = rowScratch.Slice(numImages + windowCount, windowCount);

		// Parameters of previous, current and randomly sampled pixel states.
		float prevDepth = columnState[numImages];
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

		for (int row = rowStart; row < rowEnd; ++row)
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

			// Update previous depth for next row.
			prevDepth = bestDepth;
			bestNormal.CopyTo(prevNormal);
		}

		columnState[numImages] = prevDepth;
	}

	/// <summary>
	/// The filtering step of the last sweep, run over every pixel after all rows are swept
	/// (see the file header for why that equals filtering inside the row loop).
	/// </summary>
	private void FilterPixels(SweepOptions sweep, CancellationToken cancellationToken)
	{
		int width = costMap.GetWidth();
		int planeSize = width * costMap.GetHeight();
		float[] depths = depthMap.Data;
		float[] normals = normalMap.Data;
		float[] poses = transforms.Poses(rotation);
		var frame = new PatchMatchFrame(transforms, rotation);

		// The same for every pixel (the CUDA kernel computes them per pixel).
		float minNccProb = likelihood.ComputeNCCProb(1.0f - filterMinNcc);
		float cosMinTriangulationAngle = MathF.Cos(filterMinTriangulationAngle);
		ParallelOptions parallelOptions = Mat<float>.ParallelOptionsFor(options.NumThreads);
		parallelOptions.CancellationToken = cancellationToken;
		Parallel.For(0, costMap.GetHeight(), parallelOptions, row =>
		{
			Span<float> bestNormal = stackalloc float[3];
			for (int col = 0; col < width; ++col)
			{
				int pixel = row * width + col;
				ReadNormal(normals, planeSize, pixel, bestNormal);
				FilterPixel(sweep, poses, frame, row, col, depths[pixel], bestNormal, minNccProb, cosMinTriangulationAngle);
			}
		});
	}

	/// <summary>
	/// Marks the source images the pixel's best depth and normal are consistent with, and
	/// clears the pixel when fewer than filter_min_num_consistent are. Writes only the 1s of
	/// the pixel's mask entries (the mask starts zeroed), and 0s when it clears the pixel.
	/// <paramref name="minNccProb"/> is ComputeNCCProb(1 - filter_min_ncc) and
	/// <paramref name="cosMinTriangulationAngle"/> cos(filter_min_triangulation_angle).
	/// </summary>
	internal void FilterPixel(
		in SweepOptions sweep, float[] poses, in PatchMatchFrame frame, int row, int col, float bestDepth, ReadOnlySpan<float> bestNormal,
		float minNccProb, float cosMinTriangulationAngle)
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
