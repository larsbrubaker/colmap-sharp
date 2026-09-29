// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureDescriptorIndex: colmap/feature/index.h and index.cc - the nearest-neighbor index
// the SIFT CPU matcher searches when cpu_brute_force_matcher is off (the default). Neighbors:
// SiftMatchKernels.cs (the integer dot product), SiftMatcher.cs (the caller), and
// Util/Cache.cs (the per-image index cache). Tests: FeatureDescriptorIndexTests.cs
// (index_test.cc 1:1).
//
// COLMAP's only implementation wraps faiss (native, excluded). Here the index is an exact
// k-nearest-neighbor search written from the definition, so it returns exactly what
// COLMAP's IndexFlatL2 returns for fewer than 512 descriptors, but for 512 or more it
// returns the true nearest neighbours where faiss's IVF index (nprobe = 8) is approximate.
// See divergence 42.
//
// Tier A below 512 descriptors, and exact nearest neighbours in general:
// - SIFT descriptors whose float values are integers in [0, 255] (what ToFloat produces)
//   are searched as bytes with integer squared distances, which are exact in float.
// - Anything else (learned descriptors) sums (a - b)^2 in float, left to right.
// - Neighbours are ordered by (distance, index). faiss leaves the order of equal distances
//   unspecified; COLMAP's ratio test rejects a tie for the best anyway.
// Searching is parallel over query rows, each writing its own output row.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Feature;

/// <summary>Port of colmap::FeatureDescriptorIndex: a k-nearest-neighbor descriptor index.</summary>
public abstract class FeatureDescriptorIndex
{
	/// <summary>Port of FeatureDescriptorIndex::Type.</summary>
	public enum IndexType
	{
		/// <summary>DEFAULT: exact search (COLMAP: faiss).</summary>
		Default = 1,
	}

	/// <summary>Port of FeatureDescriptorIndex::Create.</summary>
	public static FeatureDescriptorIndex Create(IndexType type = IndexType.Default, int numThreads = 1)
	{
		return type switch
		{
			IndexType.Default => new ExactFeatureDescriptorIndex(numThreads),
			_ => throw new InvalidOperationException("Feature descriptor index not implemented"),
		};
	}

	/// <summary>Builds the index over <paramref name="descriptors"/> (one per row).</summary>
	public abstract void Build(FeatureDescriptorsFloat descriptors);

	/// <summary>
	/// The <paramref name="numNeighbors"/> nearest indexed descriptors of each query row
	/// (fewer when the index is smaller), nearest first, with their squared L2 distances.
	/// Both outputs are 0 x 0 when <paramref name="numNeighbors"/> is not positive or the
	/// index is empty, and left untouched when there are no queries, like COLMAP.
	/// </summary>
	public abstract void Search(
		int numNeighbors,
		FeatureDescriptorsFloat queryDescriptors,
		ref RowMajorMatrix<int> indices,
		ref RowMajorMatrix<float> l2Dists);
}

/// <summary>
/// Exact k-NN replacement of COLMAP's FaissFeatureDescriptorIndex (header).
/// </summary>
internal sealed class ExactFeatureDescriptorIndex(int numThreads) : FeatureDescriptorIndex
{
	private FeatureExtractorType type = FeatureExtractorType.Undefined;
	private int numIndexed;
	private int dim;

	// Byte path (SIFT with integral values), else the float rows.
	private RowMajorMatrix<byte>? bytes;
	private int[]? byteNorms;
	private RowMajorMatrix<float>? floats;

	/// <inheritdoc/>
	public override void Build(FeatureDescriptorsFloat descriptors)
	{
		type = descriptors.Type;
		numIndexed = descriptors.Data.Rows;
		dim = descriptors.Data.Cols;
		bytes = null;
		byteNorms = null;
		floats = null;
		if (numIndexed == 0)
		{
			return;
		}

		if (TryToBytes(descriptors, out RowMajorMatrix<byte>? asBytes))
		{
			bytes = asBytes;
			byteNorms = SiftMatchKernels.SquaredNorms(asBytes);
		}
		else
		{
			floats = descriptors.Data.Clone();
		}
	}

	/// <inheritdoc/>
	public override void Search(
		int numNeighbors,
		FeatureDescriptorsFloat queryDescriptors,
		ref RowMajorMatrix<int> indices,
		ref RowMajorMatrix<float> l2Dists)
	{
		Check.That(queryDescriptors.Type == type);

		if (numNeighbors <= 0 || numIndexed == 0)
		{
			indices = new RowMajorMatrix<int>(0, 0);
			l2Dists = new RowMajorMatrix<float>(0, 0);
			return;
		}

		Check.Eq(queryDescriptors.Data.Cols, dim);
		int numQueries = queryDescriptors.Data.Rows;
		if (numQueries == 0)
		{
			return;
		}

		int k = Math.Min(numNeighbors, numIndexed);
		var outIndices = new RowMajorMatrix<int>(numQueries, k);
		var outDists = new RowMajorMatrix<float>(numQueries, k);

		RowMajorMatrix<byte>? queryBytes = null;
		int[]? queryNorms = null;
		if (bytes is not null)
		{
			if (TryToBytes(queryDescriptors, out queryBytes))
			{
				queryNorms = SiftMatchKernels.SquaredNorms(queryBytes);
			}
		}

		// The float path needs float index rows even when the index itself is bytes.
		RowMajorMatrix<float>? indexFloats = floats;
		if (queryBytes is null && indexFloats is null)
		{
			indexFloats = new RowMajorMatrix<float>(numIndexed, dim);
			for (int i = 0; i < bytes!.Size; ++i)
			{
				indexFloats.Data[i] = bytes.Data[i];
			}
		}

		// Byte path with few neighbours: blocks of four queries share each indexed row's loads
		// (SiftMatchKernels.DotProduct4) and keep their neighbours by insertion.
		void SearchByteBlock(int block)
		{
			int q0 = block * 4;
			int count = Math.Min(4, numQueries - q0);
			Span<int> dots = stackalloc int[4];
			Span<int> counts = stackalloc int[4];
			counts.Clear();
			ref byte a0 = ref queryBytes!.Data[q0 * SiftMatchKernels.Dim];
			ref byte a1 = ref queryBytes.Data[(q0 + (count > 1 ? 1 : 0)) * SiftMatchKernels.Dim];
			ref byte a2 = ref queryBytes.Data[(q0 + (count > 2 ? 2 : 0)) * SiftMatchKernels.Dim];
			ref byte a3 = ref queryBytes.Data[(q0 + (count > 3 ? 3 : 0)) * SiftMatchKernels.Dim];
			for (int i = 0; i < numIndexed; ++i)
			{
				SiftMatchKernels.DotProduct4(ref a0, ref a1, ref a2, ref a3, ref bytes!.Data[i * SiftMatchKernels.Dim], dots);
				for (int r = 0; r < count; ++r)
				{
					int q = q0 + r;
					float dist = queryNorms![q] + byteNorms![i] - (2 * dots[r]);
					InsertNearest(dist, i, k, ref counts[r], outIndices.Row(q), outDists.Row(q));
				}
			}
		}

		void SearchRow(int q)
		{
			var dists = new float[numIndexed];
			if (queryBytes is not null)
			{
				ref byte a = ref queryBytes.Data[q * SiftMatchKernels.Dim];
				for (int i = 0; i < numIndexed; ++i)
				{
					int dot = SiftMatchKernels.DotProduct(ref a, ref bytes!.Data[i * SiftMatchKernels.Dim]);
					dists[i] = queryNorms![q] + byteNorms![i] - (2 * dot);
				}
			}
			else
			{
				ReadOnlySpan<float> a = queryDescriptors.Data.Row(q);
				for (int i = 0; i < numIndexed; ++i)
				{
					ReadOnlySpan<float> b = indexFloats!.Row(i);
					float sum = 0;
					for (int c = 0; c < dim; ++c)
					{
						float d = a[c] - b[c];
						sum += d * d;
					}

					dists[i] = sum;
				}
			}

			SelectNearest(dists, k, outIndices.Row(q), outDists.Row(q));
		}

		bool byBlocks = queryBytes is not null && k <= MaxInsertionNeighbors;
		int numTasks = byBlocks ? (numQueries + 3) / 4 : numQueries;
		Action<int> task = byBlocks ? SearchByteBlock : SearchRow;
		int threads = SiftMatchKernels.EffectiveNumThreads(numThreads);
		if (threads == 1 || numTasks == 1)
		{
			for (int t = 0; t < numTasks; ++t)
			{
				task(t);
			}
		}
		else
		{
			Parallel.For(0, numTasks, new ParallelOptions { MaxDegreeOfParallelism = threads }, task);
		}

		indices = outIndices;
		l2Dists = outDists;
	}

	// Neighbour counts up to which a sorted insertion beats sorting all distances.
	private const int MaxInsertionNeighbors = 8;

	// Offers (dist, index) to a query's sorted list of its count (at most k) nearest. Indices
	// arrive in ascending order and a later equal distance goes after, so ties keep
	// ascending index order.
	private static void InsertNearest(float d, int index, int k, ref int count, Span<int> outIndices, Span<float> outDists)
	{
		if (count == k && !(d < outDists[k - 1]))
		{
			return;
		}

		int pos = count == k ? k - 1 : count++;
		while (pos > 0 && d < outDists[pos - 1])
		{
			outDists[pos] = outDists[pos - 1];
			outIndices[pos] = outIndices[pos - 1];
			--pos;
		}

		outDists[pos] = d;
		outIndices[pos] = index;
	}

	// The k smallest distances ordered by (distance, index).
	private static void SelectNearest(float[] dists, int k, Span<int> outIndices, Span<float> outDists)
	{
		if (k <= MaxInsertionNeighbors)
		{
			int count = 0;
			for (int i = 0; i < dists.Length; ++i)
			{
				InsertNearest(dists[i], i, k, ref count, outIndices, outDists);
			}

			return;
		}

		var order = new int[dists.Length];
		for (int i = 0; i < order.Length; ++i)
		{
			order[i] = i;
		}

		Array.Sort(order, (x, y) => dists[x] != dists[y] ? dists[x].CompareTo(dists[y]) : x.CompareTo(y));
		for (int j = 0; j < k; ++j)
		{
			outIndices[j] = order[j];
			outDists[j] = dists[order[j]];
		}
	}

	// SIFT rows of integral values in [0, 255] (what FeatureDescriptors.ToFloat produces),
	// 128 wide, as bytes; false for anything else.
	private static bool TryToBytes(FeatureDescriptorsFloat descriptors, out RowMajorMatrix<byte> result)
	{
		result = null!;
		if (descriptors.Type != FeatureExtractorType.Sift || descriptors.Data.Cols != SiftMatchKernels.Dim)
		{
			return false;
		}

		float[] data = descriptors.Data.Data;
		var converted = new byte[data.Length];
		for (int i = 0; i < data.Length; ++i)
		{
			float v = data[i];
			if (!(v >= 0 && v <= 255) || v != MathF.Floor(v))
			{
				return false;
			}

			converted[i] = (byte)v;
		}

		result = new RowMajorMatrix<byte>(descriptors.Data.Rows, descriptors.Data.Cols, converted);
		return true;
	}
}
