// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureUtilsTests: colmap/feature/utils_test.cc ported 1:1 (TEST(Suite, Nominal) becomes
// Suite_Nominal, same checks and tolerances). Tests ColmapSharp/Feature/FeatureUtils.cs.
//
// RandomEigenMatrixXf is ColmapSharp's port (float[,], column-major fill); it is copied into
// the row-major descriptor matrix. FeatureDescriptorsData::Random draws Eigen's std::rand
// bytes; ExtractTopScaleFeatures only moves rows, so FeatureTypesTests.RandomBytes (seeded
// System.Random) stands in. EXPECT_NEAR on floats compares the floats promoted to double.

using ColmapSharp.Feature;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class FeatureUtilsTests
{
	private static RowMajorMatrix<float> RandomDescriptorsPlusOne(int rows, int cols)
	{
		float[,] random = RandomEigen.RandomEigenMatrixXf(rows, cols);
		var descriptors = new RowMajorMatrix<float>(rows, cols);
		for (int r = 0; r < rows; ++r)
		{
			for (int c = 0; c < cols; ++c)
			{
				descriptors[r, c] = random[r, c] + 1.0f;
			}
		}

		return descriptors;
	}

	private static double RowNorm(RowMajorMatrix<float> m, int r)
	{
		double sum = 0;
		foreach (float v in m.Row(r))
		{
			sum += (double)v * v;
		}

		return Math.Sqrt(sum);
	}

	// A 1x128 row with one large value (index 0) among ones, chosen so the three plausible
	// float summation orders give three different sums:
	// - left to right: every "+ 1" onto 2^24 rounds away, so the sum stays 16777216;
	// - Eigen's NEON reduction (FeatureUtils.cs header): lane 0 of the first accumulator
	//   also absorbs its 15 ones (16777216), lanes 1-3 of it and all lanes of the second
	//   collect 16 ones each; adding the accumulators gives lanes (16777232, 32, 32, 32),
	//   and (l0 + l2) + (l1 + l3) = 16777264 + 64 = 16777328, exact in float;
	// - the true sum, 16777343, is not representable at all.
	private static RowMajorMatrix<float> LargeValueAmongOnes(float large)
	{
		var descriptors = new RowMajorMatrix<float>(1, 128);
		descriptors.Row(0).Fill(1.0f);
		descriptors[0, 0] = large;
		return descriptors;
	}

	// C#-only: pins the order the L1 norm is summed in, independently of any FMA question
	// (a sum of absolute values has no multiply-add to fuse).
	[Test]
	public async Task CSharpOnly_L1RootNormalizeSumsInEigenNeonOrder()
	{
		RowMajorMatrix<float> descriptors = LargeValueAmongOnes(16777216.0f);
		FeatureUtils.L1RootNormalizeFeatureDescriptors(descriptors);

		float scale = 1 / 16777328.0f;
		await Assert.That(descriptors[0, 0]).IsEqualTo(MathF.Sqrt(16777216.0f * scale));
		await Assert.That(descriptors[0, 1]).IsEqualTo(MathF.Sqrt(scale));
		await Assert.That(descriptors[0, 127]).IsEqualTo(MathF.Sqrt(scale));
	}

	// C#-only: the same for the L2 norm. The squares here are exact (4096^2 = 2^24, 1^2 = 1),
	// so a fused square-and-add would round the same way and only the order is pinned.
	[Test]
	public async Task CSharpOnly_L2NormalizeSumsInEigenNeonOrder()
	{
		RowMajorMatrix<float> descriptors = LargeValueAmongOnes(4096.0f);
		FeatureUtils.L2NormalizeFeatureDescriptors(descriptors);

		float norm = MathF.Sqrt(16777328.0f);
		await Assert.That(descriptors[0, 0]).IsEqualTo(4096.0f / norm);
		await Assert.That(descriptors[0, 1]).IsEqualTo(1.0f / norm);
		await Assert.That(descriptors[0, 127]).IsEqualTo(1.0f / norm);
	}

	[Test]
	public async Task FeatureKeypointsToPointsVector_Nominal()
	{
		List<FeatureKeypoint> keypoints = FeatureKeypoints.Create(2);
		FeatureKeypoint k1 = keypoints[1];
		k1.X = 0.1f;
		k1.Y = 0.2f;
		keypoints[1] = k1;
		Vector2d[] points = FeatureUtils.FeatureKeypointsToPointsVector(keypoints);
		await Assert.That(points[0]).IsEqualTo(new Vector2d(0, 0));
		await Assert.That((float)points[1].X).IsEqualTo(0.1f);
		await Assert.That((float)points[1].Y).IsEqualTo(0.2f);
	}

	[Test]
	public async Task L2NormalizeFeatureDescriptors_Nominal()
	{
		RowMajorMatrix<float> descriptors = RandomDescriptorsPlusOne(100, 128);
		FeatureUtils.L2NormalizeFeatureDescriptors(descriptors);
		using (Assert.Multiple())
		{
			for (int r = 0; r < descriptors.Rows; ++r)
			{
				await Assert.That(RowNorm(descriptors, r)).IsEqualTo(1).Within(1e-6);
			}
		}
	}

	[Test]
	public async Task L1RootNormalizeFeatureDescriptors_Nominal()
	{
		RowMajorMatrix<float> descriptors = RandomDescriptorsPlusOne(100, 128);
		FeatureUtils.L1RootNormalizeFeatureDescriptors(descriptors);
		using (Assert.Multiple())
		{
			for (int r = 0; r < descriptors.Rows; ++r)
			{
				await Assert.That(RowNorm(descriptors, r)).IsEqualTo(1).Within(1e-6);
			}
		}
	}

	[Test]
	public async Task FeatureDescriptorsToUnsignedByte_Nominal()
	{
		RowMajorMatrix<float> descriptors = RandomDescriptorsPlusOne(100, 128);
		RowMajorMatrix<byte> descriptorsUint8 = FeatureUtils.FeatureDescriptorsToUnsignedByte(descriptors);
		using (Assert.Multiple())
		{
			for (int r = 0; r < descriptors.Rows; ++r)
			{
				for (int c = 0; c < descriptors.Cols; ++c)
				{
					byte expected = (byte)Math.Min(255.0f, MathF.Round(512.0f * descriptors[r, c], MidpointRounding.AwayFromZero));
					await Assert.That(descriptorsUint8[r, c]).IsEqualTo(expected);
				}
			}
		}
	}

	[Test]
	public async Task ExtractTopScaleFeatures_Nominal()
	{
		List<FeatureKeypoint> keypoints = FeatureKeypoints.Create(5);
		float[] rescale = [3, 4, 1, 5, 2];
		for (int i = 0; i < 5; ++i)
		{
			FeatureKeypoint k = keypoints[i];
			k.Rescale(rescale[i]);
			keypoints[i] = k;
		}

		var descriptors = new FeatureDescriptors { Data = FeatureTypesTests.RandomBytes(5, 128) };

		List<FeatureKeypoint> topKeypoints2 = [.. keypoints];
		FeatureDescriptors topDescriptors2 = descriptors.Clone();
		FeatureUtils.ExtractTopScaleFeatures(ref topKeypoints2, topDescriptors2, 2);
		await Assert.That(topKeypoints2.Count).IsEqualTo(2);
		await Assert.That(topKeypoints2[0].ComputeScale()).IsEqualTo(keypoints[3].ComputeScale());
		await Assert.That(topKeypoints2[1].ComputeScale()).IsEqualTo(keypoints[1].ComputeScale());
		await Assert.That(topDescriptors2.Data.Rows).IsEqualTo(2);
		await Assert.That(topDescriptors2.Data.Row(0).SequenceEqual(descriptors.Data.Row(3))).IsTrue();
		await Assert.That(topDescriptors2.Data.Row(1).SequenceEqual(descriptors.Data.Row(1))).IsTrue();

		List<FeatureKeypoint> topKeypoints5 = [.. keypoints];
		FeatureDescriptors topDescriptors5 = descriptors.Clone();
		FeatureUtils.ExtractTopScaleFeatures(ref topKeypoints5, topDescriptors5, 5);
		await Assert.That(topKeypoints5.Count).IsEqualTo(5);
		await Assert.That(topDescriptors5.Data.Rows).IsEqualTo(5);
		await Assert.That(topDescriptors5.Data).IsEqualTo(descriptors.Data);

		List<FeatureKeypoint> topKeypoints6 = [.. keypoints];
		FeatureDescriptors topDescriptors6 = descriptors.Clone();
		FeatureUtils.ExtractTopScaleFeatures(ref topKeypoints6, topDescriptors6, 6);
		await Assert.That(topKeypoints5.Count).IsEqualTo(5);
		await Assert.That(topDescriptors6.Data.Rows).IsEqualTo(5);
		await Assert.That(topDescriptors6.Data).IsEqualTo(descriptors.Data);
	}
}
