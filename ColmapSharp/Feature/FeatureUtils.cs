// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureUtils: colmap/feature/utils.h and utils.cc - keypoints to points, L2 and L1-root
// descriptor normalization, float to uint8 descriptor quantization and top-scale feature
// selection. Neighbors: FeatureKeypoint.cs / FeatureDescriptors.cs (the types) and Sift.cs
// (the SIFT extractor, which normalizes and quantizes every descriptor through here).
// Tests: ColmapSharp.Tests/Feature/FeatureUtilsTests.cs (feature/utils_test.cc 1:1).
//
// Tier A (exact) where the SIFT extractor depends on it. Eigen's row norms are vectorized
// reductions: COLMAP's arm64 build sums a contiguous float row with two 4-lane NEON
// accumulators (lanes i mod 8 < 4 and >= 4), adds them, and folds the packet as
// (l0 + l2) + (l1 + l3), then adds the tail left over after whole packets. EigenRowSum
// reproduces that order, so the norms - and the quantized descriptors - match bit for bit.
//
// ExtractTopScaleFeatures: COLMAP uses std::partial_sort, which leaves ties (keypoints of
// equal scale, e.g. two orientations of one SIFT keypoint) in an unspecified order. Here
// ties keep their input order (index tie-break), see divergence 40.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Feature;

/// <summary>Port of the free functions in colmap/feature/utils.h.</summary>
public static class FeatureUtils
{
	/// <summary>Port of FeatureKeypointsToPointsVector.</summary>
	public static Vector2d[] FeatureKeypointsToPointsVector(IReadOnlyList<FeatureKeypoint> keypoints)
	{
		var points = new Vector2d[keypoints.Count];
		for (int i = 0; i < keypoints.Count; ++i)
		{
			points[i] = new Vector2d(keypoints[i].X, keypoints[i].Y);
		}

		return points;
	}

	/// <summary>Port of L2NormalizeFeatureDescriptors: each row divided by its L2 norm.</summary>
	public static void L2NormalizeFeatureDescriptors(RowMajorMatrix<float> descriptors)
	{
		for (int r = 0; r < descriptors.Rows; ++r)
		{
			L2NormalizeRow(descriptors.Row(r));
		}
	}

	/// <summary>
	/// Port of L1RootNormalizeFeatureDescriptors: each row scaled by the reciprocal of its L1
	/// norm, then square-rooted element-wise. See "Three things everyone should know to
	/// improve object retrieval", Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
	/// </summary>
	public static void L1RootNormalizeFeatureDescriptors(RowMajorMatrix<float> descriptors)
	{
		for (int r = 0; r < descriptors.Rows; ++r)
		{
			L1RootNormalizeRow(descriptors.Row(r));
		}
	}

	/// <summary>
	/// Port of FeatureDescriptorsToUnsignedByte: linear scaling from [0, 0.5] to [0, 255],
	/// rounded, with truncation to 255 (see utils.h for why 0.5).
	/// </summary>
	public static RowMajorMatrix<byte> FeatureDescriptorsToUnsignedByte(RowMajorMatrix<float> descriptors)
	{
		var result = new RowMajorMatrix<byte>(descriptors.Rows, descriptors.Cols);
		ToUnsignedByte(descriptors.Data, result.Data);
		return result;
	}

	/// <summary>
	/// Port of ExtractTopScaleFeatures: keeps the <paramref name="numFeatures"/> keypoints of
	/// largest scale (and their descriptor rows), largest first. No-op when there are no more
	/// than that many.
	/// </summary>
	public static void ExtractTopScaleFeatures(
		ref List<FeatureKeypoint> keypoints,
		FeatureDescriptors descriptors,
		int numFeatures)
	{
		Check.Eq(keypoints.Count, descriptors.Data.Rows);
		Check.Gt(numFeatures, 0);

		if (descriptors.Data.Rows <= numFeatures)
		{
			return;
		}

		var scales = new (int Index, float Scale)[keypoints.Count];
		for (int i = 0; i < keypoints.Count; ++i)
		{
			scales[i] = (i, keypoints[i].ComputeScale());
		}

		// Largest scale first; equal scales keep their input order (entry 40).
		Array.Sort(scales, (a, b) =>
		{
			int byScale = b.Scale.CompareTo(a.Scale);
			return byScale != 0 ? byScale : a.Index.CompareTo(b.Index);
		});

		int cols = descriptors.Data.Cols;
		var topKeypoints = new List<FeatureKeypoint>(numFeatures);
		var topData = new RowMajorMatrix<byte>(numFeatures, cols);
		for (int i = 0; i < numFeatures; ++i)
		{
			topKeypoints.Add(keypoints[scales[i].Index]);
			descriptors.Data.Row(scales[i].Index).CopyTo(topData.Row(i));
		}

		keypoints = topKeypoints;
		descriptors.Data = topData;
	}

	// rowwise().normalize() of one row. Unlike Matrix::normalize(), Eigen's VectorwiseOp
	// version divides by the norm unconditionally, so a zero row becomes NaN like COLMAP's.
	internal static void L2NormalizeRow(Span<float> row)
	{
		float norm = MathF.Sqrt(EigenRowSum(row, squared: true));
		for (int c = 0; c < row.Length; ++c)
		{
			row[c] /= norm;
		}
	}

	// row *= 1 / lpNorm<1>(); row = sqrt(row).
	internal static void L1RootNormalizeRow(Span<float> row)
	{
		float scale = 1 / EigenRowSum(row, squared: false);
		for (int c = 0; c < row.Length; ++c)
		{
			row[c] = MathF.Sqrt(row[c] * scale);
		}
	}

	// Quantization of FeatureDescriptorsToUnsignedByte, element-wise.
	internal static void ToUnsignedByte(ReadOnlySpan<float> values, Span<byte> bytes)
	{
		for (int i = 0; i < values.Length; ++i)
		{
			// std::round rounds halfway cases away from zero.
			float scaled = MathF.Round(512.0f * values[i], MidpointRounding.AwayFromZero);
			bytes[i] = MathUtils.TruncateCast<float, byte>(scaled);
		}
	}

	private static float Term(ReadOnlySpan<float> row, bool squared, int i) => squared ? row[i] * row[i] : MathF.Abs(row[i]);

	// Sum of |x| (or x^2) over a contiguous float row in the order Eigen's vectorized
	// reduction uses with 4-lane packets (see the file header).
	internal static float EigenRowSum(ReadOnlySpan<float> row, bool squared)
	{
		const int PacketSize = 4;
		int size = row.Length;
		int alignedSize2 = size / (2 * PacketSize) * (2 * PacketSize);
		int alignedSize = size / PacketSize * PacketSize;

		if (alignedSize == 0)
		{
			// No whole packet: a plain left-to-right reduction.
			float plain = Term(row, squared, 0);
			for (int i = 1; i < size; ++i)
			{
				plain += Term(row, squared, i);
			}

			return plain;
		}

		Span<float> p0 = stackalloc float[PacketSize];
		for (int l = 0; l < PacketSize; ++l)
		{
			p0[l] = Term(row, squared, l);
		}

		if (alignedSize > PacketSize)
		{
			Span<float> p1 = stackalloc float[PacketSize];
			for (int l = 0; l < PacketSize; ++l)
			{
				p1[l] = Term(row, squared, PacketSize + l);
			}

			for (int index = 2 * PacketSize; index < alignedSize2; index += 2 * PacketSize)
			{
				for (int l = 0; l < PacketSize; ++l)
				{
					p0[l] += Term(row, squared, index + l);
					p1[l] += Term(row, squared, index + PacketSize + l);
				}
			}

			for (int l = 0; l < PacketSize; ++l)
			{
				p0[l] += p1[l];
			}

			if (alignedSize > alignedSize2)
			{
				for (int l = 0; l < PacketSize; ++l)
				{
					p0[l] += Term(row, squared, alignedSize2 + l);
				}
			}
		}

		// NEON predux<Packet4f>: (l0 + l2) + (l1 + l3).
		float res = (p0[0] + p0[2]) + (p0[1] + p0[3]);
		for (int i = alignedSize; i < size; ++i)
		{
			res += Term(row, squared, i);
		}

		return res;
	}
}
