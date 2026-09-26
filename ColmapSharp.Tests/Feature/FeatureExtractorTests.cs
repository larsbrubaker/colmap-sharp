// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureExtractorTests: colmap/feature/extractor_test.cc. Tests ColmapSharp/Feature/FeatureExtractor.cs.
//
// The ONNX extractors (ALIKED, LoMa) are excluded, so FeatureExtractionOptions has no aliked /
// loma sub-options, and the GPU extractor is excluded, so it has no use_gpu:
// - Copy and CopyAssignment are ported without their aliked / loma lines (skipped: ONNX
//   options).
// - Move and MoveAssignment are skipped whole: they only check that moving keeps the loma
//   sub-options' shared_ptr (ONNX options; C# references have no move semantics either).
// - EffMaxImageSize and RequiresRGB are ported over all five types (they need no extractor).
// - CheckAndRequiresOpenGLWithNoGpu is ported with RequiresOpenGL over all five types and
//   Check for SIFT only; Check of the four ALIKED / LoMa types is skipped: they need ONNX
//   sub-options, and Check rejects them here.

using ColmapSharp.Feature;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class FeatureExtractorTests
{
	private static readonly FeatureExtractorType[] AllTypes =
	[
		FeatureExtractorType.Sift,
		FeatureExtractorType.AlikedN16Rot,
		FeatureExtractorType.AlikedN32,
		FeatureExtractorType.LomaB,
		FeatureExtractorType.LomaB128,
	];

	[Test]
	public async Task FeatureExtractionOptions_Copy()
	{
		var options = new FeatureExtractionOptions();
		options.MaxImageSize += 100;
		options.Sift.MaxNumFeatures += 100;

		FeatureExtractionOptions copy = options.Clone();

		// Verify fields are copied
		await Assert.That(copy.MaxImageSize).IsEqualTo(options.MaxImageSize);
		await Assert.That(copy.Sift.MaxNumFeatures).IsEqualTo(options.Sift.MaxNumFeatures);

		// Verify deep copy of shared_ptr (different pointer instances)
		await Assert.That(ReferenceEquals(options.Sift, copy.Sift)).IsFalse();
	}

	[Test]
	public async Task FeatureExtractionOptions_EffMaxImageSize()
	{
		var options = new FeatureExtractionOptions();

		// When max_image_size is explicitly set, use that value.
		options.MaxImageSize = 2000;
		foreach (FeatureExtractorType type in AllTypes)
		{
			options.Type = type;
			await Assert.That(options.EffMaxImageSize()).IsEqualTo(2000);
		}

		// When max_image_size is non-positive, use type-specific defaults.
		foreach (int maxImageSize in new[] { -1, 0 })
		{
			options.MaxImageSize = maxImageSize;
			options.Type = FeatureExtractorType.Sift;
			await Assert.That(options.EffMaxImageSize()).IsEqualTo(3200);
			options.Type = FeatureExtractorType.AlikedN16Rot;
			await Assert.That(options.EffMaxImageSize()).IsEqualTo(1600);
			options.Type = FeatureExtractorType.AlikedN32;
			await Assert.That(options.EffMaxImageSize()).IsEqualTo(1600);
			options.Type = FeatureExtractorType.LomaB;
			await Assert.That(options.EffMaxImageSize()).IsEqualTo(1600);
			options.Type = FeatureExtractorType.LomaB128;
			await Assert.That(options.EffMaxImageSize()).IsEqualTo(1600);
		}
	}

	[Test]
	public async Task FeatureExtractionOptions_CopyAssignment()
	{
		var options = new FeatureExtractionOptions();
		options.MaxImageSize = 999;
		options.Sift.MaxNumFeatures += 200;

		// Test copy assignment into a default-constructed instance.
		var assigned = new FeatureExtractionOptions();
		assigned = options.Clone();

		await Assert.That(assigned.MaxImageSize).IsEqualTo(999);
		await Assert.That(assigned.Sift.MaxNumFeatures).IsEqualTo(options.Sift.MaxNumFeatures);

		// Verify deep copy (different pointer instances).
		await Assert.That(ReferenceEquals(assigned.Sift, options.Sift)).IsFalse();

		// Mutating the copy must not affect the original.
		assigned.Sift.MaxNumFeatures += 1;
		await Assert.That(assigned.Sift.MaxNumFeatures).IsNotEqualTo(options.Sift.MaxNumFeatures);

		// Test self-assignment: a C# reference assignment to itself keeps the same instances.
		SiftExtractionOptions siftPtrBefore = options.Sift;
		int maxImageSizeBefore = options.MaxImageSize;
		FeatureExtractionOptions selfRef = options;
		options = selfRef;
		await Assert.That(ReferenceEquals(options.Sift, siftPtrBefore)).IsTrue();
		await Assert.That(options.MaxImageSize).IsEqualTo(maxImageSizeBefore);
	}

	[Test]
	public async Task FeatureExtractionOptions_RequiresRGB()
	{
		var options = new FeatureExtractionOptions();

		(FeatureExtractorType Type, bool Expected)[] testCases =
		[
			(FeatureExtractorType.Sift, false),
			(FeatureExtractorType.AlikedN16Rot, true),
			(FeatureExtractorType.AlikedN32, true),
			(FeatureExtractorType.LomaB, true),
			(FeatureExtractorType.LomaB128, true),
		];

		foreach ((FeatureExtractorType type, bool expected) in testCases)
		{
			options.Type = type;
			await Assert.That(options.RequiresRGB()).IsEqualTo(expected);
		}
	}

	[Test]
	public async Task FeatureExtractionOptions_CheckAndRequiresOpenGLWithNoGpu()
	{
		var options = new FeatureExtractionOptions();

		foreach (FeatureExtractorType type in AllTypes)
		{
			options.Type = type;
			if (type == FeatureExtractorType.Sift)
			{
				await Assert.That(options.Check()).IsTrue();
			}

			await Assert.That(options.RequiresOpenGL()).IsFalse();
		}
	}
}
