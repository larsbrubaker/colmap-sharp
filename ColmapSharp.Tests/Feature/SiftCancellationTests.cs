// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SiftCancellationTests (C#-only; COLMAP's extractor has no cancellation): the
// CancellationToken of SiftCpuFeatureExtractor.Extract (ColmapSharp/Feature/Sift.cs and
// VLFeat/VlSiftFilter*.cs). MatterCAD must be able to stop a multi-second 12 MP extraction in
// the middle of an image, so cancellation is checked between scale levels, not per image.
//
// Promptness is counted, not timed (a wall-clock bound fails under machine load): the
// extractor's internal CancellationCheckpoint diagnostic fires at every check, so a test can
// cancel at an exact check, see that the very next thing is the OperationCanceledException,
// and count how finely an extraction is divided. Every check sits between single-level
// passes over one octave (a Gaussian smoothing, a DoG extremum scan, one DoG level's
// descriptors), which bounds the work done after a cancel to about one such pass.

using ColmapSharp.Feature;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class SiftCancellationTests
{
	private static SiftCpuFeatureExtractor NewExtractor() =>
		(SiftCpuFeatureExtractor)FeatureExtractor.Create(new FeatureExtractionOptions(FeatureExtractorType.Sift));

	private static (List<FeatureKeypoint> Keypoints, FeatureDescriptors Descriptors, int Checks) ExtractCounting(
		SiftCpuFeatureExtractor extractor, Bitmap bitmap, CancellationToken cancellationToken)
	{
		int checks = 0;
		extractor.CancellationCheckpoint = () => ++checks;
		var keypoints = new List<FeatureKeypoint>();
		var descriptors = new FeatureDescriptors();
		Check(extractor.Extract(bitmap, keypoints, descriptors, cancellationToken));
		extractor.CancellationCheckpoint = null;
		return (keypoints, descriptors, checks);
	}

	private static void Check(bool ok)
	{
		if (!ok)
		{
			throw new InvalidOperationException("extraction failed");
		}
	}

	private static async Task AssertSameFeatures(
		List<FeatureKeypoint> expectedKeypoints,
		FeatureDescriptors expectedDescriptors,
		List<FeatureKeypoint> keypoints,
		FeatureDescriptors descriptors)
	{
		await Assert.That(keypoints.Count).IsEqualTo(expectedKeypoints.Count);
		for (int i = 0; i < keypoints.Count; ++i)
		{
			await Assert.That(keypoints[i]).IsEqualTo(expectedKeypoints[i]);
		}

		await Assert.That(descriptors.Data.Rows).IsEqualTo(expectedDescriptors.Data.Rows);
		for (int r = 0; r < descriptors.Data.Rows; ++r)
		{
			await Assert.That(descriptors.Data.Row(r).SequenceEqual(expectedDescriptors.Data.Row(r))).IsTrue();
		}
	}

	[Test]
	public async Task CSharpOnly_CancelledTokenThrowsBeforeAnyWork()
	{
		Bitmap bitmap = SiftTests.CreateImageWithSquare(256);
		SiftCpuFeatureExtractor extractor = NewExtractor();
		int checks = 0;
		extractor.CancellationCheckpoint = () => ++checks;

		var sentinel = new FeatureKeypoint(1, 2);
		var keypoints = new List<FeatureKeypoint> { sentinel };
		var descriptors = new FeatureDescriptors();
		using var cts = new CancellationTokenSource();
		cts.Cancel();

		Assert.Throws<OperationCanceledException>(() => extractor.Extract(bitmap, keypoints, descriptors, cts.Token));

		// The first check, before the scale space is even allocated, threw; the outputs are
		// untouched.
		await Assert.That(checks).IsEqualTo(1);
		await Assert.That(keypoints.Count).IsEqualTo(1);
		await Assert.That(keypoints[0]).IsEqualTo(sentinel);
		await Assert.That(descriptors.Data.Rows).IsEqualTo(0);
	}

	[Test]
	public async Task CSharpOnly_CancelMidImageStopsAtTheNextCheckWithoutChangingResults()
	{
		Bitmap bitmap = SiftTests.CreateImageWithSquare(256);

		// Reference: an uncancelled run with a live token gives the same features as a run
		// without a token, and is divided into many checked units.
		using var liveCts = new CancellationTokenSource();
		var reference = ExtractCounting(NewExtractor(), bitmap, liveCts.Token);
		var noToken = ExtractCounting(NewExtractor(), bitmap, CancellationToken.None);
		await AssertSameFeatures(reference.Keypoints, reference.Descriptors, noToken.Keypoints, noToken.Descriptors);
		await Assert.That(noToken.Checks).IsEqualTo(reference.Checks);

		// Four octaves (first_octave -1 on a 256 image), each checked before every one of its
		// five smoothed levels and its three DoG extremum scans, plus the per-image, per-octave
		// and per-DoG-level checks: at least 4 * (5 + 3) = 32 units.
		await Assert.That(reference.Checks).IsGreaterThanOrEqualTo(32);

		// Cancel at a check in the middle of the image: the extractor must throw right there.
		SiftCpuFeatureExtractor extractor = NewExtractor();
		int cancelAt = reference.Checks / 2;
		int checks = 0;
		using var cts = new CancellationTokenSource();
		extractor.CancellationCheckpoint = () =>
		{
			if (++checks == cancelAt)
			{
				cts.Cancel();
			}
		};

		var sentinel = new FeatureKeypoint(1, 2);
		var keypoints = new List<FeatureKeypoint> { sentinel };
		var descriptors = new FeatureDescriptors();
		Assert.Throws<OperationCanceledException>(() => extractor.Extract(bitmap, keypoints, descriptors, cts.Token));
		await Assert.That(checks).IsEqualTo(cancelAt);
		await Assert.That(keypoints.Count).IsEqualTo(1);
		await Assert.That(keypoints[0]).IsEqualTo(sentinel);
		await Assert.That(descriptors.Data.Rows).IsEqualTo(0);

		// The same extractor then extracts exactly what an uncancelled one does.
		extractor.CancellationCheckpoint = null;
		var after = ExtractCounting(extractor, bitmap, CancellationToken.None);
		await AssertSameFeatures(reference.Keypoints, reference.Descriptors, after.Keypoints, after.Descriptors);
	}
}
