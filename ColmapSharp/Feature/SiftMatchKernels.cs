// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SiftMatchKernels: the descriptor arithmetic behind the SIFT CPU matcher of
// colmap/feature/sift.cc - ComputeSiftDistanceMatrix (uint8 descriptors as ints, dot product
// or squared L2) fused with the best / second-best scan of FindBestMatchesOneWayBruteForce
// and FindBestMatchesOneWayIndex, in both directions at once. Neighbors: SiftMatcher.cs (the
// ratio / distance tests and cross-check on the scan results) and FeatureDescriptorIndex.cs
// (exact k-NN search, which shares DotProduct). Tests: SiftMatcherTests.cs (sift_test.cc)
// and SiftMatchKernelsTests.cs (C#-only: the fused scan against the plain matrix scan).
//
// Tier A (exact). All distances are integers: a SIFT dot product is at most
// 128 * 255 * 255 < 2^24, so COLMAP's conversion to float is exact and monotone, and
// comparing the ints decides every comparison exactly as COLMAP's floats do.
//
// SIMD: CLAUDE.md bans System.Numerics.Vector<T> in math paths because floating-point
// SIMD may fuse or reorder. These kernels only use integer SIMD (Arm Dp.DotProduct, AVX2
// multiply-add of int16, or the portable Vector128 widening multiply) whose results are
// exact and independent of the lane order, so every path returns the same integer.
//
// Why fused instead of COLMAP's full N x M matrix: an 8k x 8k float matrix is 256 MB, too
// much for wasm. COLMAP's scan keeps, per row (and per column for the reverse direction),
// the best value, the first index reaching it, and the second best of the multiset seen so
// far (a tie with the best becomes the second best). That state is order-exact under
// merging in ascending row order (MergeColumns), so rows are processed in parallel chunks,
// each writing its own rows and its own column states, and the result is identical to the
// sequential scan for any chunking or thread count.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Feature;

/// <summary>
/// Best / second-best state of one row or column of a distance matrix, scanned with COLMAP's
/// strict "greater than" updates on a key (the dot product, or the negated squared L2
/// distance). <see cref="BestIdx"/> is -1 when no value beat the initial sentinel.
/// </summary>
internal struct Top2
{
	public int Best;
	public int Second;
	public int BestIdx;

	public static Top2 Init(int sentinel) => new() { Best = sentinel, Second = sentinel, BestIdx = -1 };

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Update(int key, int idx)
	{
		if (key > Best)
		{
			BestIdx = idx;
			Second = Best;
			Best = key;
		}
		else if (key > Second)
		{
			Second = key;
		}
	}

	// This state followed by later (higher-index) state b, as if scanned in one pass.
	public void MergeLater(in Top2 b)
	{
		if (b.Best > Best)
		{
			Second = Math.Max(Best, b.Second);
			Best = b.Best;
			BestIdx = b.BestIdx;
		}
		else
		{
			Second = Math.Max(Second, b.Best);
		}
	}
}

/// <summary>Integer descriptor kernels shared by the SIFT matcher and the descriptor index.</summary>
internal static class SiftMatchKernels
{
	/// <summary>kSiftDescriptorDim.</summary>
	public const int Dim = 128;

	// Query rows scanned together (DotProduct4).
	private const int RowBlock = 4;

	/// <summary>kSqSiftDescriptorNorm: SIFT descriptors are normalized to length 512 (w/ quantization errors).</summary>
	public const int SqSiftDescriptorNorm = 512 * 512;

	/// <summary>Exact integer dot product of two 128-byte descriptors.</summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static int DotProduct(ref byte a, ref byte b)
	{
		if (Dp.IsSupported)
		{
			// SDOT/UDOT: 16 byte products summed into 4 uint lanes per instruction.
			Vector128<uint> acc0 = Vector128<uint>.Zero;
			Vector128<uint> acc1 = Vector128<uint>.Zero;
			for (int k = 0; k < Dim; k += 32)
			{
				acc0 = Dp.DotProduct(acc0, Vector128.LoadUnsafe(ref a, (nuint)k), Vector128.LoadUnsafe(ref b, (nuint)k));
				acc1 = Dp.DotProduct(acc1, Vector128.LoadUnsafe(ref a, (nuint)(k + 16)), Vector128.LoadUnsafe(ref b, (nuint)(k + 16)));
			}

			return (int)Vector128.Sum(acc0 + acc1);
		}

		if (Avx2.IsSupported)
		{
			Vector256<int> acc = Vector256<int>.Zero;
			for (int k = 0; k < Dim; k += 16)
			{
				Vector256<short> va = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref a, (nuint)k));
				Vector256<short> vb = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref b, (nuint)k));
				acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(va, vb));
			}

			return Vector256.Sum(acc);
		}

		if (Vector128.IsHardwareAccelerated)
		{
			// 255 * 255 fits a ushort, so the widened products are exact before the uint sum.
			Vector128<uint> acc = Vector128<uint>.Zero;
			for (int k = 0; k < Dim; k += 16)
			{
				Vector128<byte> va = Vector128.LoadUnsafe(ref a, (nuint)k);
				Vector128<byte> vb = Vector128.LoadUnsafe(ref b, (nuint)k);
				Vector128<ushort> lo = Vector128.WidenLower(va) * Vector128.WidenLower(vb);
				Vector128<ushort> hi = Vector128.WidenUpper(va) * Vector128.WidenUpper(vb);
				acc += Vector128.WidenLower(lo) + Vector128.WidenUpper(lo) + Vector128.WidenLower(hi) + Vector128.WidenUpper(hi);
			}

			return (int)Vector128.Sum(acc);
		}

		int sum = 0;
		for (int k = 0; k < Dim; ++k)
		{
			sum += Unsafe.Add(ref a, k) * Unsafe.Add(ref b, k);
		}

		return sum;
	}

	/// <summary>
	/// Dot products of four 128-byte descriptors with one: the four share each load of
	/// <paramref name="b"/>, which is what bounds the scan's throughput.
	/// </summary>
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void DotProduct4(ref byte a0, ref byte a1, ref byte a2, ref byte a3, ref byte b, Span<int> result)
	{
		if (Dp.IsSupported)
		{
			Vector128<uint> s0 = Vector128<uint>.Zero;
			Vector128<uint> s1 = Vector128<uint>.Zero;
			Vector128<uint> s2 = Vector128<uint>.Zero;
			Vector128<uint> s3 = Vector128<uint>.Zero;
			for (nuint k = 0; k < Dim; k += 16)
			{
				Vector128<byte> vb = Vector128.LoadUnsafe(ref b, k);
				s0 = Dp.DotProduct(s0, Vector128.LoadUnsafe(ref a0, k), vb);
				s1 = Dp.DotProduct(s1, Vector128.LoadUnsafe(ref a1, k), vb);
				s2 = Dp.DotProduct(s2, Vector128.LoadUnsafe(ref a2, k), vb);
				s3 = Dp.DotProduct(s3, Vector128.LoadUnsafe(ref a3, k), vb);
			}

			result[0] = (int)Vector128.Sum(s0);
			result[1] = (int)Vector128.Sum(s1);
			result[2] = (int)Vector128.Sum(s2);
			result[3] = (int)Vector128.Sum(s3);
			return;
		}

		result[0] = DotProduct(ref a0, ref b);
		result[1] = DotProduct(ref a1, ref b);
		result[2] = DotProduct(ref a2, ref b);
		result[3] = DotProduct(ref a3, ref b);
	}

	/// <summary>Squared L2 norm of each 128-byte row.</summary>
	public static int[] SquaredNorms(RowMajorMatrix<byte> descriptors)
	{
		var norms = new int[descriptors.Rows];
		for (int i = 0; i < norms.Length; ++i)
		{
			ref byte row = ref descriptors.Data[i * Dim];
			norms[i] = DotProduct(ref row, ref row);
		}

		return norms;
	}

	/// <summary>
	/// COLMAP's non-positive thread count means "all cores" (GetEffectiveNumThreads).
	/// </summary>
	public static int EffectiveNumThreads(int numThreads) => numThreads <= 0 ? Environment.ProcessorCount : numThreads;

	/// <summary>
	/// The fused scan of the N x M distance matrix between <paramref name="d1"/> and
	/// <paramref name="d2"/>. <paramref name="useL2"/> false scans dot products (key = dot,
	/// sentinel 0, as FindBestMatchesOneWayBruteForce); true scans squared L2 distances
	/// (key = -l2, sentinel standing for FLT_MAX, as FindBestMatchesOneWayIndex).
	/// <paramref name="guidedFilter"/> (i1, i2) true rejects a pair, which then gets the
	/// worst distance (dot 0, L2 kSqSiftDescriptorNorm) like ComputeSiftDistanceMatrix.
	/// Columns are only scanned when <paramref name="cols"/> is non-null.
	/// </summary>
	public static void ScanDistances(
		RowMajorMatrix<byte> d1,
		RowMajorMatrix<byte> d2,
		bool useL2,
		Func<int, int, bool>? guidedFilter,
		int numThreads,
		Top2[] rows,
		Top2[]? cols)
	{
		int n1 = d1.Rows;
		int n2 = d2.Rows;
		int[]? norms1 = useL2 ? SquaredNorms(d1) : null;
		int[]? norms2 = useL2 ? SquaredNorms(d2) : null;

		// FLT_MAX for L2: every real distance (at most 128 * 255^2) is smaller.
		int sentinel = useL2 ? -int.MaxValue : 0;
		int rejectedKey = useL2 ? -SqSiftDescriptorNorm : 0;

		// Several chunks per thread balance fast and slow cores; results do not depend on the
		// chunking (header).
		int threads = EffectiveNumThreads(numThreads);
		int numChunks = Math.Max(1, Math.Min(threads == 1 ? 1 : 4 * threads, (n1 + 63) / 64));
		var chunkCols = cols is null ? null : new Top2[numChunks][];
		byte[] data1 = d1.Data;
		byte[] data2 = d2.Data;

		void ScanChunk(int chunk)
		{
			int begin = (int)((long)n1 * chunk / numChunks);
			int end = (int)((long)n1 * (chunk + 1) / numChunks);
			Top2[]? colState = null;
			if (chunkCols is not null)
			{
				colState = new Top2[n2];
				Array.Fill(colState, Top2.Init(sentinel));
				chunkCols[chunk] = colState;
			}

			ref byte base1 = ref MemoryMarshal.GetArrayDataReference(data1);
			ref byte base2 = ref MemoryMarshal.GetArrayDataReference(data2);
			Span<int> keys = stackalloc int[RowBlock];
			Span<Top2> rowStates = stackalloc Top2[RowBlock];

			// Blocks of RowBlock query rows share each target row's loads. Within a block the
			// column states are updated in ascending row order, as in the sequential scan.
			for (int blockBegin = begin; blockBegin < end; blockBegin += RowBlock)
			{
				int blockSize = Math.Min(RowBlock, end - blockBegin);
				rowStates.Fill(Top2.Init(sentinel));

				// Missing rows of a short last block alias the first row; their keys are unused.
				ref byte a0 = ref Unsafe.Add(ref base1, blockBegin * Dim);
				ref byte a1 = ref Unsafe.Add(ref base1, (blockBegin + (blockSize > 1 ? 1 : 0)) * Dim);
				ref byte a2 = ref Unsafe.Add(ref base1, (blockBegin + (blockSize > 2 ? 2 : 0)) * Dim);
				ref byte a3 = ref Unsafe.Add(ref base1, (blockBegin + (blockSize > 3 ? 3 : 0)) * Dim);
				for (int i2 = 0; i2 < n2; ++i2)
				{
					DotProduct4(ref a0, ref a1, ref a2, ref a3, ref Unsafe.Add(ref base2, i2 * Dim), keys);
					for (int r = 0; r < blockSize; ++r)
					{
						int i1 = blockBegin + r;
						int key;
						if (guidedFilter is not null && guidedFilter(i1, i2))
						{
							key = rejectedKey;
						}
						else
						{
							key = useL2 ? -(norms1![i1] + norms2![i2] - (2 * keys[r])) : keys[r];
						}

						rowStates[r].Update(key, i2);
						if (colState is not null)
						{
							colState[i2].Update(key, i1);
						}
					}
				}

				for (int r = 0; r < blockSize; ++r)
				{
					rows[blockBegin + r] = rowStates[r];
				}
			}
		}

		if (numChunks == 1)
		{
			ScanChunk(0);
		}
		else
		{
			Parallel.For(0, numChunks, new ParallelOptions { MaxDegreeOfParallelism = threads }, ScanChunk);
		}

		if (cols is not null)
		{
			for (int i2 = 0; i2 < n2; ++i2)
			{
				Top2 state = chunkCols![0][i2];
				for (int chunk = 1; chunk < numChunks; ++chunk)
				{
					state.MergeLater(chunkCols[chunk][i2]);
				}

				cols[i2] = state;
			}
		}
	}
}
