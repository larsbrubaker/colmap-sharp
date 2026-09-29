// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SiftMatcherTests: the unguided CPU matcher cases of colmap/feature/sift_test.cc, 1:1 -
// SiftCPUFeatureMatcher.Nominal, SiftCPUFeatureMatcher.TypeMismatch and
// SiftCPUFeatureMatcherFaissVsBruteForce.Nominal (the index path against brute force; the
// index is exact here, divergence 42). The guided cases are in
// SiftMatcherGuidedTests.cs; the helpers of sift_test.cc are in SiftMatcherTestUtils.
// Tests ColmapSharp/Feature/SiftMatcher.cs.
//
// Skipped (SiftGPU, excluded): CreateSiftGPUMatcherOpenGL.Nominal,
// CreateSiftGPUMatcherCUDA.Nominal, MatchSiftFeaturesGPU.{Nominal, TypeMismatch},
// MatchSiftFeaturesCPUvsGPU.Nominal, and every MatchGuidedSiftFeaturesGPU /
// MatchGuidedSiftFeaturesCPUvsGPUGuided case.
// std::invalid_argument (THROW_CHECK) is ArgumentException.

using ColmapSharp.Feature;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

/// <summary>The helpers of sift_test.cc shared by the matcher test classes.</summary>
internal static class SiftMatcherTestUtils
{
	// Helper to create empty descriptors for testing.
	public static FeatureDescriptors CreateEmptyDescriptors() => new(FeatureExtractorType.Sift, new RowMajorMatrix<byte>(0, 128));

	// Helper to create reversed descriptors for testing matcher symmetry
	// (data.colwise().reverse(): the row order reversed).
	public static FeatureDescriptors CreateReversedDescriptors(FeatureDescriptors src)
	{
		int rows = src.Data.Rows;
		var data = new RowMajorMatrix<byte>(rows, src.Data.Cols);
		for (int r = 0; r < rows; ++r)
		{
			src.Data.Row(rows - 1 - r).CopyTo(data.Row(r));
		}

		return new FeatureDescriptors(src.Type, data);
	}

	public static FeatureDescriptors CreateRandomFeatureDescriptors(int numFeatures)
	{
		RandomUtils.SetPRNGSeed(0);
		var descriptorsFloat = new RowMajorMatrix<float>(numFeatures, 128);
		var dims = new List<int>(128);
		for (int d = 0; d < 128; ++d)
		{
			dims.Add(d);
		}

		for (int i = 0; i < numFeatures; ++i)
		{
			LibcxxRandom.Shuffle(dims, RandomUtils.Prng!);
			for (int j = 0; j < 10; ++j)
			{
				descriptorsFloat[i, dims[j]] = 1.0f;
			}
		}

		FeatureUtils.L2NormalizeFeatureDescriptors(descriptorsFloat);
		return new FeatureDescriptors(FeatureExtractorType.Sift, FeatureUtils.FeatureDescriptorsToUnsignedByte(descriptorsFloat));
	}

	public static async Task CheckEqualMatches(List<FeatureMatch> matches1, List<FeatureMatch> matches2)
	{
		await Assert.That(matches1.Count).IsEqualTo(matches2.Count);
		for (int i = 0; i < matches1.Count; ++i)
		{
			await Assert.That(matches1[i].Point2DIdx1).IsEqualTo(matches2[i].Point2DIdx1);
			await Assert.That(matches1[i].Point2DIdx2).IsEqualTo(matches2[i].Point2DIdx2);
		}
	}

	public static async Task ExpectReversedMatches(List<FeatureMatch> matches)
	{
		await Assert.That(matches.Count).IsEqualTo(2);
		await Assert.That(matches[0].Point2DIdx1).IsEqualTo(0u);
		await Assert.That(matches[0].Point2DIdx2).IsEqualTo(1u);
		await Assert.That(matches[1].Point2DIdx1).IsEqualTo(1u);
		await Assert.That(matches[1].Point2DIdx2).IsEqualTo(0u);
	}

	public static Task ExpectReversedInlierMatches(TwoViewGeometry twoViewGeometry) =>
		ExpectReversedMatches(twoViewGeometry.InlierMatches);

	public static TwoViewGeometry CreatePlanarTwoViewGeometry() => new()
	{
		Config = TwoViewGeometry.ConfigurationType.PlanarOrPanoramic,
		H = Matrix3d.Identity,
	};

	// FeatureDescriptorIndexCacheHelper: builds each image's index on first use.
	public static ThreadSafeLRUCache<uint, FeatureDescriptorIndex> CreateIndexCache(params FeatureMatcherImage[] images)
	{
		var imageDescriptors = new Dictionary<uint, FeatureDescriptors>();
		foreach (FeatureMatcherImage image in images)
		{
			imageDescriptors[image.ImageId] = image.Descriptors!;
		}

		return new ThreadSafeLRUCache<uint, FeatureDescriptorIndex>(100, imageId =>
		{
			FeatureDescriptorIndex index = FeatureDescriptorIndex.Create();
			index.Build(imageDescriptors[imageId].ToFloat());
			return index;
		});
	}

	// The matcher factory every MatchGuidedSiftFeaturesCPU case passes to its test body.
	public static FeatureMatcher CreateCpuMatcherWithIndexCache(FeatureMatcherImage[] images)
	{
		var options = new FeatureMatchingOptions(FeatureMatcherType.SiftBruteForce);
		options.Sift.CpuDescriptorIndexCache = CreateIndexCache(images);
		return SiftFeatureMatchers.CreateSiftFeatureMatcher(options);
	}

	public static Camera SimplePinhole100() => Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 100.0, 100, 200);
}

public class SiftMatcherTests
{
	[Test]
	public async Task SiftCPUFeatureMatcher_Nominal()
	{
		Camera camera = SiftMatcherTestUtils.SimplePinhole100();
		var image0 = new FeatureMatcherImage { ImageId = 0, Camera = camera, Descriptors = SiftMatcherTestUtils.CreateEmptyDescriptors() };
		var image1 = new FeatureMatcherImage { ImageId = 1, Camera = camera, Descriptors = SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2) };
		var image2 = new FeatureMatcherImage
		{
			ImageId = 2,
			Camera = camera,
			Descriptors = SiftMatcherTestUtils.CreateReversedDescriptors(image1.Descriptors),
		};

		var options = new FeatureMatchingOptions(FeatureMatcherType.SiftBruteForce);
		options.Sift.CpuBruteForceMatcher = false;
		options.Sift.CpuDescriptorIndexCache = SiftMatcherTestUtils.CreateIndexCache(image0, image1, image2);
		FeatureMatcher matcher = SiftFeatureMatchers.CreateSiftFeatureMatcher(options);

		var matches = new List<FeatureMatch>();
		matcher.Match(image1, image2, matches);
		await SiftMatcherTestUtils.ExpectReversedMatches(matches);

		matcher.Match(image1, image2, matches);
		await SiftMatcherTestUtils.ExpectReversedMatches(matches);

		matcher.Match(image0, image2, matches);
		await Assert.That(matches.Count).IsEqualTo(0);
		matcher.Match(image1, image0, matches);
		await Assert.That(matches.Count).IsEqualTo(0);
		matcher.Match(image0, image0, matches);
		await Assert.That(matches.Count).IsEqualTo(0);
	}

	[Test]
	public async Task SiftCPUFeatureMatcher_TypeMismatch()
	{
		Camera camera = SiftMatcherTestUtils.SimplePinhole100();

		FeatureDescriptors siftDesc = SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2);
		await Assert.That(siftDesc.Type).IsEqualTo(FeatureExtractorType.Sift);

		FeatureDescriptors undefinedDesc = SiftMatcherTestUtils.CreateRandomFeatureDescriptors(2);
		undefinedDesc.Type = FeatureExtractorType.Undefined;

		var imageSift = new FeatureMatcherImage { ImageId = 1, Camera = camera, Descriptors = siftDesc };
		var imageUndefined = new FeatureMatcherImage { ImageId = 2, Camera = camera, Descriptors = undefinedDesc };

		var options = new FeatureMatchingOptions(FeatureMatcherType.SiftBruteForce);
		options.Sift.CpuBruteForceMatcher = true;
		FeatureMatcher matcher = SiftFeatureMatchers.CreateSiftFeatureMatcher(options);

		var matches = new List<FeatureMatch>();
		Assert.Throws<ArgumentException>(() => matcher.Match(imageSift, imageUndefined, matches));
	}

	// The lambda TestFaissVsBruteForce of SiftCPUFeatureMatcherFaissVsBruteForce.Nominal.
	private static async Task<int> TestFaissVsBruteForce(
		FeatureMatchingOptions options, FeatureDescriptors descriptors1, FeatureDescriptors descriptors2)
	{
		Camera camera = SiftMatcherTestUtils.SimplePinhole100();
		var image0 = new FeatureMatcherImage { ImageId = 0, Camera = camera, Descriptors = SiftMatcherTestUtils.CreateEmptyDescriptors() };
		var image1 = new FeatureMatcherImage { ImageId = 1, Camera = camera, Descriptors = descriptors1 };
		var image2 = new FeatureMatcherImage { ImageId = 2, Camera = camera, Descriptors = descriptors2 };

		var matchesBf = new List<FeatureMatch>();
		var matchesFaiss = new List<FeatureMatch>();

		FeatureMatchingOptions customOptions = options.Clone();
		customOptions.Sift.CpuBruteForceMatcher = true;
		FeatureMatcher bfMatcher = SiftFeatureMatchers.CreateSiftFeatureMatcher(customOptions);
		customOptions.Sift.CpuBruteForceMatcher = false;
		customOptions.Sift.CpuDescriptorIndexCache = SiftMatcherTestUtils.CreateIndexCache(image0, image1, image2);
		FeatureMatcher faissMatcher = SiftFeatureMatchers.CreateSiftFeatureMatcher(customOptions);

		bfMatcher.Match(image1, image2, matchesBf);
		faissMatcher.Match(image1, image2, matchesFaiss);
		await SiftMatcherTestUtils.CheckEqualMatches(matchesBf, matchesFaiss);

		int numMatches = matchesBf.Count;

		bfMatcher.Match(image0, image2, matchesBf);
		faissMatcher.Match(image0, image2, matchesFaiss);
		await SiftMatcherTestUtils.CheckEqualMatches(matchesBf, matchesFaiss);

		bfMatcher.Match(image1, image0, matchesBf);
		faissMatcher.Match(image1, image0, matchesFaiss);
		await SiftMatcherTestUtils.CheckEqualMatches(matchesBf, matchesFaiss);

		bfMatcher.Match(image0, image0, matchesBf);
		faissMatcher.Match(image0, image0, matchesFaiss);
		await SiftMatcherTestUtils.CheckEqualMatches(matchesBf, matchesFaiss);

		return numMatches;
	}

	// descriptors.data.row(r).cast<float>().normalized() quantized back to uint8.
	private static void RenormalizeRow(FeatureDescriptors descriptors, int r)
	{
		var row = new RowMajorMatrix<float>(1, 128);
		for (int c = 0; c < 128; ++c)
		{
			row[0, c] = descriptors.Data[r, c];
		}

		FeatureUtils.L2NormalizeFeatureDescriptors(row);
		FeatureUtils.FeatureDescriptorsToUnsignedByte(row).Row(0).CopyTo(descriptors.Data.Row(r));
	}

	private static FeatureDescriptors TopRows(FeatureDescriptors descriptors, int n)
	{
		var data = new RowMajorMatrix<byte>(n, descriptors.Data.Cols);
		descriptors.Data.Data.AsSpan(0, data.Size).CopyTo(data.Data);
		return new FeatureDescriptors(descriptors.Type, data);
	}

	[Test]
	public async Task SiftCPUFeatureMatcherFaissVsBruteForce_Nominal()
	{
		{
			FeatureDescriptors descriptors1 = SiftMatcherTestUtils.CreateRandomFeatureDescriptors(50);
			FeatureDescriptors descriptors2 = SiftMatcherTestUtils.CreateRandomFeatureDescriptors(50);
			await TestFaissVsBruteForce(new FeatureMatchingOptions(), descriptors1, descriptors2);
		}

		{
			FeatureDescriptors descriptors1 = SiftMatcherTestUtils.CreateRandomFeatureDescriptors(50);
			FeatureDescriptors descriptors2 = SiftMatcherTestUtils.CreateReversedDescriptors(descriptors1);
			int numMatches = await TestFaissVsBruteForce(new FeatureMatchingOptions(), descriptors1, descriptors2);
			await Assert.That(numMatches).IsEqualTo(50);
		}

		// Check the ratio test.
		{
			FeatureDescriptors descriptors1 = SiftMatcherTestUtils.CreateRandomFeatureDescriptors(50);
			FeatureDescriptors descriptors2 = descriptors1.Clone();

			var matchOptions = new FeatureMatchingOptions();
			int numMatches1 = await TestFaissVsBruteForce(matchOptions, descriptors1, descriptors2);
			await Assert.That(numMatches1).IsEqualTo(50);

			descriptors2.Data.Row(0).CopyTo(descriptors2.Data.Row(49));

			// uint8 += int wraps modulo 256, like the C++ implicit conversion back to uint8_t.
			descriptors2.Data[0, 0] = unchecked((byte)(descriptors2.Data[0, 0] + 50));
			RenormalizeRow(descriptors2, 0);
			descriptors2.Data[49, 0] = unchecked((byte)(descriptors2.Data[49, 0] + 100));
			RenormalizeRow(descriptors2, 49);

			matchOptions.Sift.MaxRatio = 0.4;
			FeatureDescriptors descriptors1Top49 = TopRows(descriptors1, 49);
			int numMatches2 = await TestFaissVsBruteForce(matchOptions, descriptors1Top49, descriptors2);
			await Assert.That(numMatches2).IsEqualTo(48);

			matchOptions.Sift.MaxRatio = 0.6;
			int numMatches3 = await TestFaissVsBruteForce(matchOptions, descriptors1, descriptors2);
			await Assert.That(numMatches3).IsEqualTo(49);
		}

		// Check the cross check.
		{
			FeatureDescriptors descriptors1 = SiftMatcherTestUtils.CreateRandomFeatureDescriptors(50);
			FeatureDescriptors descriptors2 = descriptors1.Clone();
			descriptors1.Data.Row(1).CopyTo(descriptors1.Data.Row(0));

			var matchOptions = new FeatureMatchingOptions();

			matchOptions.Sift.CrossCheck = false;
			int numMatches1 = await TestFaissVsBruteForce(matchOptions, descriptors1, descriptors2);
			await Assert.That(numMatches1).IsEqualTo(50);

			matchOptions.Sift.CrossCheck = true;
			int numMatches2 = await TestFaissVsBruteForce(matchOptions, descriptors1, descriptors2);
			await Assert.That(numMatches2).IsEqualTo(48);
		}
	}
}
