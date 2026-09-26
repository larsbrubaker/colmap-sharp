// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatchKernel.PhotometricLockstep: four photo-consistency costs at once. The sweep
// (PatchMatchCpu.Sweep.cs) scores hypotheses 1-4 of a pixel against the same sampled source
// image; here they run in lockstep as the four lanes of Vector128<float>, one hypothesis per
// lane. Each lane repeats exactly the scalar operations of
// PatchMatchPhotoConsistency.Compute (PatchMatchKernel.Photometric.cs), in the same order:
// the homography setup, the per-row accumulated source coordinates, the division, the
// bilinear sample (PatchMatchSourceImages.Sample4, lane-wise Sample), and the three weighted sums,
// each lane keeping its own sums. There is no fused multiply-add and no horizontal
// reduction, so every lane's result is bit-identical to the scalar path (CLAUDE.md, "No
// FMA"); PatchMatchKernelTests.LockstepMatchesScalarBitForBit pins that. Without hardware
// SIMD (e.g. wasm without SIMD) the sweep calls the scalar path four times instead.

using System.Runtime.Intrinsics;

namespace ColmapSharp.Mvs;

public sealed partial class PatchMatchPhotoConsistency
{
	/// <summary>
	/// The costs of four hypotheses (<paramref name="depths"/>[h], normal
	/// <paramref name="normals"/>[3h..3h+3]) at pixel (row, col) against source image
	/// <paramref name="srcImageIdx"/>, into <paramref name="costs"/>, each bit-identical to
	/// <see cref="Compute(int, int, float, ReadOnlySpan{float}, int, ReadOnlySpan{float}, ReadOnlySpan{float}, float)"/>.
	/// </summary>
	public void ComputeFour(
		int row,
		int col,
		ReadOnlySpan<float> depths,
		ReadOnlySpan<float> normals,
		int srcImageIdx,
		ReadOnlySpan<float> refColors,
		ReadOnlySpan<float> weights,
		float bilateralWeightSum,
		Span<float> costs)
	{
		if (!Vector128.IsHardwareAccelerated)
		{
			for (int h = 0; h < 4; ++h)
			{
				costs[h] = Compute(row, col, depths[h], normals.Slice(3 * h, 3), srcImageIdx, refColors, weights, bilateralWeightSum);
			}

			return;
		}

		ComputeFourLockstep(row, col, depths, normals, srcImageIdx, refColors, weights, bilateralWeightSum, costs);
	}

	/// <summary>The Vector128 lockstep path of <see cref="ComputeFour"/> (internal for the bit-identity test).</summary>
	internal void ComputeFourLockstep(
		int row,
		int col,
		ReadOnlySpan<float> depths,
		ReadOnlySpan<float> normals,
		int srcImageIdx,
		ReadOnlySpan<float> refColors,
		ReadOnlySpan<float> weights,
		float bilateralWeightSum,
		Span<float> costs)
	{
		// One homography per lane, laid out as tforms[9 h + i].
		Span<float> tforms = stackalloc float[36];
		for (int h = 0; h < 4; ++h)
		{
			PatchMatchKernel.ComposeHomography(poses, frame, srcImageIdx, row, col, depths[h], normals.Slice(3 * h, 3), tforms.Slice(9 * h, 9));
		}

		Vector128<float> t0 = Lanes(tforms, 0), t1 = Lanes(tforms, 1), t2 = Lanes(tforms, 2);
		Vector128<float> t3 = Lanes(tforms, 3), t4 = Lanes(tforms, 4), t5 = Lanes(tforms, 5);
		Vector128<float> t6 = Lanes(tforms, 6), t7 = Lanes(tforms, 7), t8 = Lanes(tforms, 8);
		Vector128<float> stepFactor = Vector128.Create((float)windowStep);
		Vector128<float> step0 = stepFactor * t0;
		Vector128<float> step1 = stepFactor * t1;
		Vector128<float> step3 = stepFactor * t3;
		Vector128<float> step4 = stepFactor * t4;
		Vector128<float> step6 = stepFactor * t6;
		Vector128<float> step7 = stepFactor * t7;

		Vector128<float> colStart = Vector128.Create((float)(col - windowRadius));
		Vector128<float> rowStart = Vector128.Create((float)(row - windowRadius));

		Vector128<float> colSrc = t0 * colStart + t1 * rowStart + t2;
		Vector128<float> rowSrc = t3 * colStart + t4 * rowStart + t5;
		Vector128<float> z = t6 * colStart + t7 * rowStart + t8;
		Vector128<float> baseColSrc = colSrc;
		Vector128<float> baseRowSrc = rowSrc;
		Vector128<float> baseZ = z;

		Vector128<float> one = Vector128.Create(1.0f);
		Vector128<float> half = Vector128.Create(0.5f);
		Vector128<float> srcColorSum = Vector128<float>.Zero;
		Vector128<float> srcColorSquaredSum = Vector128<float>.Zero;
		Vector128<float> srcRefColorSum = Vector128<float>.Zero;

		int perAxis = (2 * windowRadius) / windowStep + 1;
		int i = 0;
		for (int windowRow = 0; windowRow < perAxis; ++windowRow)
		{
			for (int windowCol = 0; windowCol < perAxis; ++windowCol, ++i)
			{
				Vector128<float> invZ = one / z;
				Vector128<float> normColSrc = invZ * colSrc + half;
				Vector128<float> normRowSrc = invZ * rowSrc + half;
				Vector128<float> srcColor = srcImages.Sample4(normColSrc, normRowSrc, srcImageIdx);

				Vector128<float> bilateralWeightSrc = Vector128.Create(weights[i]) * srcColor;

				srcColorSum += bilateralWeightSrc;
				srcColorSquaredSum += bilateralWeightSrc * srcColor;
				srcRefColorSum += bilateralWeightSrc * Vector128.Create(refColors[i]);

				colSrc += step0;
				rowSrc += step3;
				z += step6;
			}

			baseColSrc += step1;
			baseRowSrc += step4;
			baseZ += step7;

			colSrc = baseColSrc;
			rowSrc = baseRowSrc;
			z = baseZ;
		}

		// The rest is per lane and cheap: finish each lane exactly as the scalar path does.
		int width = refImage.Image.GetWidth();
		float refColorSum = refImage.SumImage.Data[row * width + col];
		float refColorSquaredSum = refImage.SquaredSumImage.Data[row * width + col];
		for (int h = 0; h < 4; ++h)
		{
			costs[h] = FinishNcc(
				srcColorSum.GetElement(h), srcColorSquaredSum.GetElement(h), srcRefColorSum.GetElement(h),
				bilateralWeightSum, refColorSum, refColorSquaredSum);
		}
	}

	// Entry i of the four homographies, one per lane.
	private static Vector128<float> Lanes(ReadOnlySpan<float> tforms, int i) =>
		Vector128.Create(tforms[i], tforms[9 + i], tforms[18 + i], tforms[27 + i]);
}
