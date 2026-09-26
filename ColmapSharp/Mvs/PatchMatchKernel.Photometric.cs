// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatchKernel.Photometric: PhotoConsistencyCostComputer and the ComputeInitialCost
// kernel of patch_match_cuda.cu - the bilaterally weighted NCC between a reference patch and
// its plane-induced warp into a source image, returned as 1 - NCC in [0, 2]. Reads the
// reference image and its prefiltered window sums (PatchMatchRefImage.cs), the source layers
// (PatchMatchTextures.cs) and the pose table of the current rotation
// (PatchMatchTransforms.cs); PatchMatchKernel.Geometry.cs builds the homography. Tests:
// ColmapSharp.Tests/Mvs/PatchMatchKernelTests.cs (C#-only).
//
// Translation notes:
// - COLMAP stages the reference image rows of a 32-column thread block in shared memory
//   (LocalRefImage) and reads the window from there; the values are the reference texture's
//   (point sampled, byte / 255, 0 outside the image), which the CPU reads directly.
// - The window is a runtime radius and step instead of template parameters (COLMAP
//   instantiates radii 1-20 and steps 1-2; PatchMatchOptions.Check allows up to 32, where
//   COLMAP logs "not supported" and computes nothing for radii above 20; divergence 97).
// - The reference half of the window (colors and bilateral weights, and their sum) does not
//   depend on the hypothesis or source image, so PrepareWindow computes it once per pixel
//   and the sweep reuses it for all ~60 NCC evaluations there. The values and the order of
//   the weight-sum additions are exactly COLMAP's, so the result is bit-identical to
//   recomputing them inside every evaluation, as the CUDA kernel does.
// - Source coordinates are accumulated per row by the homography's column steps and
//   restarted per row from the accumulated row steps, in COLMAP's float order ("to reduce
//   numerical errors").
// - Float math with MathF and no fused multiply-add (docs/CPP_DIVERGENCES.md, entry 96);
//   source samples are exact float bilinear (entry 95).

namespace ColmapSharp.Mvs;

/// <summary>
/// Port of PhotoConsistencyCostComputer: 1 - bilaterally weighted NCC of a reference
/// patch and its warp into a source image, for one rotation of the reference image.
/// Read-only after construction, so any number of threads can call its methods.
/// </summary>
public sealed partial class PatchMatchPhotoConsistency
{
	/// <summary>Maximum photo consistency cost as 1 - min(NCC).</summary>
	public const float MaxCost = 2.0f;

	private readonly PatchMatchRefImage refImage;
	private readonly PatchMatchSourceImages srcImages;
	private readonly float[] poses;
	private readonly PatchMatchFrame frame;
	private readonly BilateralWeightComputer bilateralWeightComputer;
	private readonly int windowRadius;
	private readonly int windowStep;

	/// <summary>
	/// The cost for the (rotated) reference image <paramref name="refImage"/>, whose frame
	/// and pose table are <paramref name="frame"/> and <paramref name="poses"/>.
	/// </summary>
	public PatchMatchPhotoConsistency(
		PatchMatchRefImage refImage,
		PatchMatchSourceImages srcImages,
		float[] poses,
		in PatchMatchFrame frame,
		int windowRadius,
		int windowStep,
		float sigmaSpatial,
		float sigmaColor)
	{
		this.refImage = Util.Check.NotNull(refImage);
		this.srcImages = Util.Check.NotNull(srcImages);
		this.poses = Util.Check.NotNull(poses);
		this.frame = frame;
		this.windowRadius = windowRadius;
		this.windowStep = windowStep;
		bilateralWeightComputer = new BilateralWeightComputer(sigmaSpatial, sigmaColor);
	}

	/// <summary>Width of the (rotated) reference image.</summary>
	public int RefWidth => refImage.Image.GetWidth();

	/// <summary>Height of the (rotated) reference image.</summary>
	public int RefHeight => refImage.Image.GetHeight();

	/// <summary>The reference frame (calibration of the current rotation).</summary>
	public PatchMatchFrame Frame => frame;

	/// <summary>Number of samples in the (strided) window: the size of a <see cref="PrepareWindow"/> buffer.</summary>
	public int WindowCount
	{
		get
		{
			int perAxis = (2 * windowRadius) / windowStep + 1;
			return perAxis * perAxis;
		}
	}

	/// <summary>
	/// The reference half of the NCC at pixel (row, col), which depends on neither the
	/// hypothesis nor the source image: the window's reference colors and bilateral weights
	/// in COLMAP's loop order, and their weight sum accumulated in that order. The sweep
	/// prepares it once per pixel and reuses it for every hypothesis and sample; the values
	/// are exactly those Compute would recompute each time.
	/// </summary>
	public float PrepareWindow(int row, int col, Span<float> refColors, Span<float> weights)
	{
		byte[] refData = refImage.Image.Data;
		int width = refImage.Image.GetWidth();
		int height = refImage.Image.GetHeight();
		float refCenterColor = PatchMatchRefImage.Texel(refData, width, height, row, col);
		float bilateralWeightSum = 0.0f;
		int i = 0;
		for (int windowRow = -windowRadius; windowRow <= windowRadius; windowRow += windowStep)
		{
			for (int windowCol = -windowRadius; windowCol <= windowRadius; windowCol += windowStep)
			{
				float refColor = PatchMatchRefImage.Texel(refData, width, height, row + windowRow, col + windowCol);
				float bilateralWeight = bilateralWeightComputer.Compute(windowRow, windowCol, refCenterColor, refColor);
				refColors[i] = refColor;
				weights[i] = bilateralWeight;
				bilateralWeightSum += bilateralWeight;
				++i;
			}
		}

		return bilateralWeightSum;
	}

	/// <summary>
	/// 1 - NCC in [0, 2] of the patch around pixel (row, col) warped into source image
	/// <paramref name="srcImageIdx"/> by the plane at <paramref name="depth"/> with
	/// <paramref name="normal"/>; 2 when either patch has (nearly) no variance.
	/// </summary>
	public float Compute(int row, int col, float depth, ReadOnlySpan<float> normal, int srcImageIdx)
	{
		int count = WindowCount;
		float[]? heapBuffer = count > 1024 ? new float[2 * count] : null;
		Span<float> buffer = heapBuffer ?? stackalloc float[2 * count];
		Span<float> refColors = buffer[..count];
		Span<float> weights = buffer[count..];
		float weightSum = PrepareWindow(row, col, refColors, weights);
		return Compute(row, col, depth, normal, srcImageIdx, refColors, weights, weightSum);
	}

	/// <summary>
	/// <see cref="Compute(int, int, float, ReadOnlySpan{float}, int)"/> with the reference
	/// window already prepared by <see cref="PrepareWindow"/> for the same pixel.
	/// </summary>
	public float Compute(
		int row,
		int col,
		float depth,
		ReadOnlySpan<float> normal,
		int srcImageIdx,
		ReadOnlySpan<float> refColors,
		ReadOnlySpan<float> weights,
		float bilateralWeightSum)
	{
		Span<float> tform = stackalloc float[9];
		PatchMatchKernel.ComposeHomography(poses, frame, srcImageIdx, row, col, depth, normal, tform);

		float step0 = windowStep * tform[0];
		float step1 = windowStep * tform[1];
		float step3 = windowStep * tform[3];
		float step4 = windowStep * tform[4];
		float step6 = windowStep * tform[6];
		float step7 = windowStep * tform[7];

		int rowStart = row - windowRadius;
		int colStart = col - windowRadius;

		float colSrc = tform[0] * colStart + tform[1] * rowStart + tform[2];
		float rowSrc = tform[3] * colStart + tform[4] * rowStart + tform[5];
		float z = tform[6] * colStart + tform[7] * rowStart + tform[8];
		float baseColSrc = colSrc;
		float baseRowSrc = rowSrc;
		float baseZ = z;

		int width = refImage.Image.GetWidth();
		float refColorSum = refImage.SumImage.Data[row * width + col];
		float refColorSquaredSum = refImage.SquaredSumImage.Data[row * width + col];
		float srcColorSum = 0.0f;
		float srcColorSquaredSum = 0.0f;
		float srcRefColorSum = 0.0f;

		int perAxis = (2 * windowRadius) / windowStep + 1;
		int i = 0;
		for (int windowRow = 0; windowRow < perAxis; ++windowRow)
		{
			for (int windowCol = 0; windowCol < perAxis; ++windowCol, ++i)
			{
				float invZ = 1.0f / z;
				float normColSrc = invZ * colSrc + 0.5f;
				float normRowSrc = invZ * rowSrc + 0.5f;
				float refColor = refColors[i];
				float srcColor = srcImages.Sample(normColSrc, normRowSrc, srcImageIdx);

				float bilateralWeightSrc = weights[i] * srcColor;

				srcColorSum += bilateralWeightSrc;
				srcColorSquaredSum += bilateralWeightSrc * srcColor;
				srcRefColorSum += bilateralWeightSrc * refColor;

				// Accumulate warped source coordinates per row to reduce numerical errors.
				// Note that this is necessary since coordinates usually are in the order of
				// 1000s as opposed to the color values which are normalized to the range
				// [0, 1].
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

		return FinishNcc(srcColorSum, srcColorSquaredSum, srcRefColorSum, bilateralWeightSum, refColorSum, refColorSquaredSum);
	}

	/// <summary>
	/// 1 - NCC from the weighted source sums and the reference window's statistics: the
	/// tail of COLMAP's PhotoConsistencyCostComputer::Compute, shared by the scalar and
	/// lockstep paths.
	/// </summary>
	private static float FinishNcc(
		float srcColorSum,
		float srcColorSquaredSum,
		float srcRefColorSum,
		float bilateralWeightSum,
		float refColorSum,
		float refColorSquaredSum)
	{
		float invBilateralWeightSum = 1.0f / bilateralWeightSum;
		srcColorSum *= invBilateralWeightSum;
		srcColorSquaredSum *= invBilateralWeightSum;
		srcRefColorSum *= invBilateralWeightSum;

		float refColorVar = refColorSquaredSum - refColorSum * refColorSum;
		float srcColorVar = srcColorSquaredSum - srcColorSum * srcColorSum;

		// Based on Jensen's Inequality for convex functions, the variance should always be
		// larger than 0. Do not make this threshold smaller.
		const float MinVar = 1e-5f;
		if (refColorVar < MinVar || srcColorVar < MinVar)
		{
			return MaxCost;
		}

		float srcRefColorCovar = srcRefColorSum - refColorSum * srcColorSum;
		float srcRefColorVar = MathF.Sqrt(refColorVar * srcColorVar);
		return PatchMatchKernel.CudaMax(0.0f, PatchMatchKernel.CudaMin(MaxCost, 1.0f - srcRefColorCovar / srcRefColorVar));
	}
}

/// <summary>The ComputeInitialCost kernel.</summary>
public static partial class PatchMatchKernel
{
	/// <summary>
	/// Fills <paramref name="costMap"/> (width x height x number of source images) with the
	/// photo-consistency cost of every pixel's depth and normal against every source image.
	/// Parallel over rows on at most <paramref name="numThreads"/> threads (-1: all cores);
	/// each pixel writes only its own costs, so the result does not depend on the count.
	/// </summary>
	public static void ComputeInitialCost(
		Mat<float> costMap,
		Mat<float> depthMap,
		Mat<float> normalMap,
		PatchMatchPhotoConsistency photoConsistency,
		int numThreads = -1,
		CancellationToken cancellationToken = default)
	{
		int width = costMap.GetWidth();
		int height = costMap.GetHeight();
		int numImages = costMap.GetDepth();
		Util.Check.Eq(depthMap.GetWidth(), width);
		Util.Check.Eq(depthMap.GetHeight(), height);
		Util.Check.Eq(normalMap.GetWidth(), width);
		Util.Check.Eq(normalMap.GetHeight(), height);
		Util.Check.Eq(normalMap.GetDepth(), 3);
		Util.Check.Eq(photoConsistency.RefWidth, width);
		Util.Check.Eq(photoConsistency.RefHeight, height);

		float[] costs = costMap.Data;
		float[] depths = depthMap.Data;
		float[] normals = normalMap.Data;
		int planeSize = width * height;
		ParallelOptions parallelOptions = Mat<float>.ParallelOptionsFor(numThreads);
		parallelOptions.CancellationToken = cancellationToken;
		Parallel.For(0, height, parallelOptions, row =>
		{
			Span<float> normal = stackalloc float[3];
			for (int col = 0; col < width; ++col)
			{
				int idx = row * width + col;
				float depth = depths[idx];
				normal[0] = normals[idx];
				normal[1] = normals[planeSize + idx];
				normal[2] = normals[2 * planeSize + idx];
				for (int imageIdx = 0; imageIdx < numImages; ++imageIdx)
				{
					costs[imageIdx * planeSize + idx] = photoConsistency.Compute(row, col, depth, normal, imageIdx);
				}
			}
		});
	}
}
