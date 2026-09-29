// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureDescriptorIndexTests: colmap/feature/index_test.cc 1:1. The parameterized
// ParameterizedFeatureDescriptorIndexTests.Nominal runs over the same four rows (SIFT and
// ALIKED_N16ROT, 100 and 1000 descriptors); COLMAP's FAISS index type is the exact index here
// (FeatureDescriptorIndex.IndexType.Default, divergence 42). Tests
// ColmapSharp/Feature/FeatureDescriptorIndex.cs. EXPECT_NEAR on floats compares the floats
// promoted to double.

using ColmapSharp.Feature;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class FeatureDescriptorIndexTests
{
	private static FeatureDescriptorsFloat CreateRandomFeatureDescriptors(FeatureExtractorType type, int numFeatures)
	{
		var data = new RowMajorMatrix<float>(numFeatures, 128);
		for (int i = 0; i < numFeatures; ++i)
		{
			for (int j = 0; j < 128; ++j)
			{
				// std::pow(float, 2) computes in double; the float-rounded result equals x * x.
				float x = RandomUtils.RandomUniformReal(0.0f, 1.0f);
				data[i, j] = (float)((double)x * x);
			}
		}

		FeatureUtils.L2NormalizeFeatureDescriptors(data);
		if (type == FeatureExtractorType.Sift)
		{
			// Mimic the real SIFT pipeline: convert to uint8 [0,255] then back to float. This
			// ensures values are integer-valued floats in [0, 255], which is required for
			// QT_8bit_direct quantization.
			RowMajorMatrix<byte> bytes = FeatureUtils.FeatureDescriptorsToUnsignedByte(data);
			for (int i = 0; i < bytes.Size; ++i)
			{
				data.Data[i] = bytes.Data[i];
			}
		}

		return new FeatureDescriptorsFloat(type, data);
	}

	private static float SquaredDistance(RowMajorMatrix<float> a, int ra, RowMajorMatrix<float> b, int rb)
	{
		float sum = 0;
		for (int c = 0; c < a.Cols; ++c)
		{
			float d = a[ra, c] - b[rb, c];
			sum += d * d;
		}

		return sum;
	}

	[Test]
	[Arguments(FeatureExtractorType.Sift, 100)]
	[Arguments(FeatureExtractorType.Sift, 1000)]
	[Arguments(FeatureExtractorType.AlikedN16Rot, 100)]
	[Arguments(FeatureExtractorType.AlikedN16Rot, 1000)]
	public async Task ParameterizedFeatureDescriptorIndexTests_Nominal(FeatureExtractorType extractorType, int numDescriptors)
	{
		FeatureDescriptorIndex index = FeatureDescriptorIndex.Create(FeatureDescriptorIndex.IndexType.Default);
		await Assert.That(index).IsNotNull();

		FeatureDescriptorsFloat indexDescriptors = CreateRandomFeatureDescriptors(extractorType, numDescriptors);
		FeatureDescriptorsFloat queryDescriptors = indexDescriptors;
		index.Build(indexDescriptors);

		var indices = new RowMajorMatrix<int>(0, 0);
		var distances = new RowMajorMatrix<float>(0, 0);
		index.Search(1, queryDescriptors, ref indices, ref distances);

		await Assert.That(indices.Rows).IsEqualTo(queryDescriptors.Data.Rows);
		await Assert.That(indices.Cols).IsEqualTo(1);
		await Assert.That(distances.Rows).IsEqualTo(queryDescriptors.Data.Rows);
		await Assert.That(distances.Cols).IsEqualTo(1);

		for (int i = 0; i < queryDescriptors.Data.Rows; ++i)
		{
			await Assert.That(indices[i, 0]).IsEqualTo(i);
			await Assert.That((double)distances[i, 0]).IsEqualTo(0).Within(1e-6);
		}

		index.Search(2, queryDescriptors, ref indices, ref distances);
		await Assert.That(indices.Rows).IsEqualTo(queryDescriptors.Data.Rows);
		await Assert.That(indices.Cols).IsEqualTo(2);
		await Assert.That(distances.Rows).IsEqualTo(queryDescriptors.Data.Rows);
		await Assert.That(distances.Cols).IsEqualTo(2);

		for (int i = 0; i < queryDescriptors.Data.Rows; ++i)
		{
			await Assert.That(indices[i, 0]).IsEqualTo(i);
			await Assert.That((double)distances[i, 0]).IsEqualTo(0).Within(1e-6);
			await Assert.That(indices[i, 1]).IsNotEqualTo(i);
			double expected = SquaredDistance(queryDescriptors.Data, i, indexDescriptors.Data, indices[i, 1]);
			await Assert.That((double)distances[i, 1]).IsEqualTo(expected).Within(1e-6);
		}

		index.Search(indexDescriptors.Data.Rows + 1, queryDescriptors, ref indices, ref distances);
		await Assert.That(indices.Rows).IsEqualTo(queryDescriptors.Data.Rows);
		await Assert.That(indices.Cols).IsEqualTo(indexDescriptors.Data.Rows);
		await Assert.That(distances.Rows).IsEqualTo(queryDescriptors.Data.Rows);
		await Assert.That(distances.Cols).IsEqualTo(indexDescriptors.Data.Rows);
	}

	[Test]
	public async Task FeatureDescriptorIndexTests_TypeMismatch()
	{
		const int kNumDescriptors = 100;

		FeatureDescriptorIndex index = FeatureDescriptorIndex.Create(FeatureDescriptorIndex.IndexType.Default);
		await Assert.That(index).IsNotNull();

		// Prepare SIFT descriptors for index build.
		FeatureDescriptorsFloat siftDesc = CreateRandomFeatureDescriptors(FeatureExtractorType.Sift, kNumDescriptors);

		// Prepare descriptors with a different type.
		FeatureDescriptorsFloat alikedDesc = CreateRandomFeatureDescriptors(FeatureExtractorType.AlikedN16Rot, kNumDescriptors);

		// Build should throw when descriptor types are inconsistent.
		index.Build(alikedDesc);

		// Build correctly with SIFT so we can test Query mismatch.
		index.Build(siftDesc);

		// Query should throw when descriptor types are inconsistent.
		var indices = new RowMajorMatrix<int>(0, 0);
		var distances = new RowMajorMatrix<float>(0, 0);
		Assert.Throws<ArgumentException>(() => index.Search(1, alikedDesc, ref indices, ref distances));
	}
}
