// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Mat.Transforms: the whole-matrix operations of colmap/mvs/gpu_mat.h (FillWithVector,
// Transpose, FlipHorizontal, Rotate) and the kernels behind them (cuda_transpose.h,
// cuda_flip.h, cuda_rotate.h), run on the CPU over Mat.cs's flat storage. PatchMatch
// (patch_match_cuda.cu) rotates its reference image and every per-pixel map by 90 degrees
// after each sweep, so it can always sweep top to bottom. Tests:
// ColmapSharp.Tests/Mvs/GpuMatTests.cs (gpu_mat_test.cu 1:1).
//
// Tier A (exact): pure element moves. Each output element is written by exactly one input
// element, so the parallel loops give the same result for any thread count; numThreads
// bounds them as COLMAP's num_threads does (-1: all cores). Rotate and Transpose move
// 32 x 32 tiles so their scattered writes stay in cache.
//
// Translation notes:
// - COLMAP's GpuMat is a pitched device buffer; the CPU port has no separate type, since
//   Mat<T> already has the same slice-major, row-major layout without row padding.
// - The operations write into a caller-supplied output, as in COLMAP, so PatchMatch can
//   double-buffer its maps instead of allocating per sweep.

namespace ColmapSharp.Mvs;

public partial class Mat<T>
{
	/// <summary>
	/// Sets every pixel's slices to <paramref name="values"/> (one value per slice).
	/// Port of GpuMat::FillWithVector.
	/// </summary>
	public void FillWithVector(ReadOnlySpan<T> values)
	{
		Util.Check.Eq(values.Length, depth);
		int planeSize = width * height;
		for (int slice = 0; slice < depth; ++slice)
		{
			Array.Fill(data, values[slice], slice * planeSize, planeSize);
		}
	}

	/// <summary>
	/// Writes the transpose of every slice into <paramref name="output"/> (height x width x
	/// depth): output(col, row) = this(row, col). Port of GpuMat::Transpose.
	/// </summary>
	public void Transpose(Mat<T> output, int numThreads = -1)
	{
		CheckOutput(output, height, width);

		// Output row col, output column row.
		MoveTiled(output, numThreads, mirrorColumns: false);
	}

	/// <summary>
	/// Writes every slice mirrored left to right into <paramref name="output"/> (same size):
	/// output(row, width - 1 - col) = this(row, col). Port of GpuMat::FlipHorizontal.
	/// </summary>
	public void FlipHorizontal(Mat<T> output, int numThreads = -1)
	{
		CheckOutput(output, width, height);
		T[] input = data;
		T[] result = output.data;
		int w = width;

		// Rows stay rows, so plain row loops are already cache-friendly; one loop over
		// every (slice, row).
		Parallel.For(0, depth * height, ParallelOptionsFor(numThreads), sliceRow =>
		{
			int rowStart = sliceRow * w;
			for (int col = 0; col < w; ++col)
			{
				result[rowStart + w - 1 - col] = input[rowStart + col];
			}
		});
	}

	/// <summary>
	/// Writes every slice rotated by 90 degrees counter-clockwise into
	/// <paramref name="output"/> (height x width x depth):
	/// output(width - 1 - col, row) = this(row, col). Port of GpuMat::Rotate
	/// (CudaRotateKernel).
	/// </summary>
	public void Rotate(Mat<T> output, int numThreads = -1)
	{
		CheckOutput(output, height, width);

		// The output is height wide: output row width - 1 - col, output column row.
		MoveTiled(output, numThreads, mirrorColumns: true);
	}

	/// <summary>
	/// Copies every element (row, col) of every slice to the same slice of
	/// <paramref name="output"/> (height wide) at output row c and column row, where c is
	/// col, or width - 1 - col when <paramref name="mirrorColumns"/> (a transpose, or a
	/// transpose of the mirrored image, which is the rotation), in
	/// TileSize x TileSize tiles so both the reads and the scattered writes stay in cache.
	/// One parallel loop runs over every (slice, tile row); each element has exactly one
	/// writer, so the result does not depend on the thread count.
	/// </summary>
	private void MoveTiled(Mat<T> output, int numThreads, bool mirrorColumns)
	{
		const int TileSize = 32;
		T[] input = data;
		T[] result = output.data;
		int w = width;
		int h = height;
		int planeSize = w * h;
		int numTileRows = (h + TileSize - 1) / TileSize;
		Parallel.For(0, depth * numTileRows, ParallelOptionsFor(numThreads), sliceTileRow =>
		{
			int offset = sliceTileRow / numTileRows * planeSize;
			int rowBegin = sliceTileRow % numTileRows * TileSize;
			int rowEnd = Math.Min(rowBegin + TileSize, h);
			for (int colBegin = 0; colBegin < w; colBegin += TileSize)
			{
				int colEnd = Math.Min(colBegin + TileSize, w);
				for (int row = rowBegin; row < rowEnd; ++row)
				{
					int inputRow = offset + row * w;
					for (int col = colBegin; col < colEnd; ++col)
					{
						int outputRow = mirrorColumns ? w - 1 - col : col;
						result[offset + outputRow * h + row] = input[inputRow + col];
					}
				}
			}
		});
	}

	/// <summary>Parallel options bounded by COLMAP's num_threads (-1: all cores).</summary>
	internal static ParallelOptions ParallelOptionsFor(int numThreads) =>
		new() { MaxDegreeOfParallelism = Util.Threading.GetEffectiveNumThreads(numThreads) };

	private void CheckOutput(Mat<T> output, int outputWidth, int outputHeight)
	{
		Util.Check.NotNull(output);
		Util.Check.That(!ReferenceEquals(output, this), "the output must be a different matrix");
		Util.Check.Eq(output.width, outputWidth);
		Util.Check.Eq(output.height, outputHeight);
		Util.Check.Eq(output.depth, depth);
	}
}
