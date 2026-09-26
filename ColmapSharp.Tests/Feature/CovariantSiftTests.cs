// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// CovariantSiftTests (C#-only): behavior of ColmapSharp/Feature/CovariantSift.cs that COLMAP
// does not have - a clear error for images too small for the detector's octaves (VLFeat reads
// out of bounds there), and cancellation. The sift_test.cc rows are in SiftTests.cs.

using ColmapSharp.Feature;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class CovariantSiftTests
{
	private static CovariantSiftCpuFeatureExtractor NewExtractor(int firstOctave = -1)
	{
		var options = new FeatureExtractionOptions(FeatureExtractorType.Sift);
		options.Sift.EstimateAffineShape = true;
		options.Sift.FirstOctave = firstOctave;
		return (CovariantSiftCpuFeatureExtractor)FeatureExtractor.Create(options);
	}

	[Test]
	[Arguments(-1, 16)]
	[Arguments(0, 16)]
	[Arguments(1, 31)]
	[Arguments(2, 61)]
	public async Task CSharpOnly_TooSmallImageThrowsClearMessage(int firstOctave, int minimumSide)
	{
		await Assert.That(CovariantSiftCpuFeatureExtractor.MinimumImageSide(firstOctave)).IsEqualTo(minimumSide);

		// One pixel short on the shorter side: a clear, user-facing error.
		Bitmap small = SiftTests.CreateImageWithSquare(minimumSide - 1);
		var keypoints = new List<FeatureKeypoint>();
		ArgumentException? error = null;
		try
		{
			NewExtractor(firstOctave).Extract(small, keypoints, new FeatureDescriptors());
		}
		catch (ArgumentException e)
		{
			error = e;
		}

		await Assert.That(error).IsNotNull();
		await Assert.That(error!.Message).Contains($"at least {minimumSide} pixels");

		// The minimum size itself extracts.
		Bitmap enough = SiftTests.CreateImageWithSquare(minimumSide);
		await Assert.That(NewExtractor(firstOctave).Extract(enough, keypoints, new FeatureDescriptors())).IsTrue();
	}

	[Test]
	public async Task CSharpOnly_CancelMidExtractionThrowsAndLeavesOutputs()
	{
		Bitmap bitmap = SiftTests.CreateImageWithSquare(256);
		CovariantSiftCpuFeatureExtractor extractor = NewExtractor();

		var reference = new List<FeatureKeypoint>();
		var referenceDescriptors = new FeatureDescriptors();
		int checks = 0;
		extractor.CancellationCheckpoint = () => ++checks;
		await Assert.That(extractor.Extract(bitmap, reference, referenceDescriptors)).IsTrue();
		await Assert.That(checks).IsGreaterThan(10);

		// Cancel at a checkpoint in the descriptor loop (the last few checks).
		using var cts = new CancellationTokenSource();
		int seen = 0;
		int cancelAt = checks - 5;
		extractor.CancellationCheckpoint = () =>
		{
			if (++seen == cancelAt)
			{
				cts.Cancel();
			}
		};
		var sentinel = new FeatureKeypoint(1, 2);
		var keypoints = new List<FeatureKeypoint> { sentinel };
		var descriptors = new FeatureDescriptors();
		OperationCanceledException? cancelled = null;
		try
		{
			extractor.Extract(bitmap, keypoints, descriptors, cts.Token);
		}
		catch (OperationCanceledException e)
		{
			cancelled = e;
		}

		await Assert.That(cancelled).IsNotNull();
		await Assert.That(seen).IsEqualTo(cancelAt);
		await Assert.That(keypoints.Count).IsEqualTo(1);
		await Assert.That(keypoints[0]).IsEqualTo(sentinel);
		await Assert.That(descriptors.Data.Rows).IsEqualTo(0);

		// The same extractor then gives exactly the uncancelled result.
		extractor.CancellationCheckpoint = null;
		var again = new List<FeatureKeypoint>();
		var againDescriptors = new FeatureDescriptors();
		await Assert.That(extractor.Extract(bitmap, again, againDescriptors, CancellationToken.None)).IsTrue();
		await Assert.That(again.SequenceEqual(reference)).IsTrue();
		await Assert.That(againDescriptors.Data.Data.SequenceEqual(referenceDescriptors.Data.Data)).IsTrue();
	}
}
