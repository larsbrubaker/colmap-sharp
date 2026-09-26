// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SiftTests: the extraction cases of colmap/feature/sift_test.cc. TEST_P(SiftCpuExtractionTest,
// Nominal) is SiftCpuExtraction_Nominal over the same parameter rows, with the same checks
// (keypoint count, ValidateKeypoints, ValidateDescriptorNorms). Tests ColmapSharp/Feature/Sift.cs.
//
// All six rows are ported; the five Covariant* rows run CovariantSift.cs (VLFeat covdet).
// Skipped: ExtractSiftFeaturesGPU.Nominal (SiftGPU, excluded). The matcher tests of
// sift_test.cc are in SiftMatcherTests.cs and SiftMatcherGuidedTests.cs.
// The cancellation tests (C#-only) are in SiftCancellationTests.cs.

using ColmapSharp.Feature;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class SiftTests
{
	internal static Bitmap CreateImageWithSquare(int size)
	{
		var bitmap = new Bitmap(size, size, asRgb: false);
		bitmap.Fill(new BitmapColor<byte>(0, 0, 0));
		for (int r = (size / 2) - (size / 8); r < (size / 2) + (size / 8); ++r)
		{
			for (int c = (size / 2) - (size / 8); c < (size / 2) + (size / 8); ++c)
			{
				bitmap.SetPixel(r, c, new BitmapColor<byte>(255));
			}
		}

		return bitmap;
	}

	private static async Task ValidateKeypoints(List<FeatureKeypoint> keypoints, Bitmap bitmap)
	{
		foreach (FeatureKeypoint k in keypoints)
		{
			await Assert.That(k.X).IsGreaterThanOrEqualTo(0);
			await Assert.That(k.Y).IsGreaterThanOrEqualTo(0);
			await Assert.That(k.X).IsLessThanOrEqualTo(bitmap.Width);
			await Assert.That(k.Y).IsLessThanOrEqualTo(bitmap.Height);
			await Assert.That(k.ComputeScale()).IsGreaterThan(0);
			await Assert.That((double)k.ComputeOrientation()).IsGreaterThan(-Math.PI);
			await Assert.That((double)k.ComputeOrientation()).IsLessThan(Math.PI);
		}
	}

	private static async Task ValidateDescriptorNorms(FeatureDescriptors descriptors)
	{
		await Assert.That(descriptors.Type).IsEqualTo(FeatureExtractorType.Sift);
		for (int i = 0; i < descriptors.Data.Rows; ++i)
		{
			// descriptors.data.row(i).cast<float>().norm(): a float norm.
			float sum = 0;
			foreach (byte v in descriptors.Data.Row(i))
			{
				sum += (float)v * v;
			}

			await Assert.That(Math.Abs(MathF.Sqrt(sum) - 512)).IsLessThan(1);
		}
	}

	[Test]
	[Arguments("Sift", false, false, false, false, 22)]
	[Arguments("CovariantSift", false, false, true, false, 22)]
	[Arguments("CovariantAffineSift", true, false, false, false, 22)]
	[Arguments("CovariantAffineSiftUpright", true, false, false, true, 10)]
	[Arguments("CovariantDSPSift", false, true, false, false, 22)]
	[Arguments("CovariantAffineDSPSift", true, true, false, false, 22)]
	public async Task SiftCpuExtraction_Nominal(
		string name,
		bool estimateAffineShape,
		bool domainSizePooling,
		bool forceCovariantExtractor,
		bool upright,
		int expectedKeypoints)
	{
		_ = name;
		Bitmap bitmap = CreateImageWithSquare(256);

		var options = new FeatureExtractionOptions(FeatureExtractorType.Sift);
		options.Sift.EstimateAffineShape = estimateAffineShape;
		options.Sift.DomainSizePooling = domainSizePooling;
		options.Sift.ForceCovariantExtractor = forceCovariantExtractor;
		options.Sift.Upright = upright;
		FeatureExtractor extractor = SiftFeatureExtractors.CreateSiftFeatureExtractor(options);

		var keypoints = new List<FeatureKeypoint>();
		var descriptors = new FeatureDescriptors();
		await Assert.That(extractor.Extract(bitmap, keypoints, descriptors)).IsTrue();

		await Assert.That(keypoints.Count).IsEqualTo(expectedKeypoints);
		await ValidateKeypoints(keypoints, bitmap);
		await Assert.That(descriptors.Data.Rows).IsEqualTo(expectedKeypoints);
		await ValidateDescriptorNorms(descriptors);
	}
}
