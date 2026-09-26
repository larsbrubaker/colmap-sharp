// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SiftExtractorReuseTests (C#-only): one SiftCpuFeatureExtractor reused across images of the
// same size must give exactly what a fresh extractor gives (ColmapSharp/Feature/Sift.cs and
// VLFeat/VlSiftFilter*.cs). Upstream it does not: VLFeat resets its cached-gradient octave
// (grad_o) only in vl_sift_new, so when image A's last octave with keypoints is image B's
// first octave, B's orientations and descriptors there come from A's gradient, and COLMAP's
// output depends on the order in which a worker thread sees its images. The port resets the
// cache per image (docs/CPP_DIVERGENCES.md entry 43).
//
// Fixture: oracle/fixture_sift_reuse.py writes sift_reuse.json - a low-contrast fine-noise A
// whose keypoints all sit in octave -1, the seeded texture B, and pycolmap's features of B from
// a fresh extractor and from one reused after A, under the default options and num_octaves = 1.
// The port is compared with pycolmap's FRESH output within the FMA-contraction tolerance of
// SiftOracleTests (entry 41); the fixture's reused output differs from its fresh output, which
// records the upstream behavior.

using System.Text.Json;

using ColmapSharp.Feature;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class SiftExtractorReuseTests
{
	public static IEnumerable<string> CaseNames() => ["Defaults", "NumOctaves1"];

	private static JsonElement LoadCase(string caseName, out Bitmap a, out Bitmap b)
	{
		JsonElement root = OracleFixture.Load("sift_reuse.json");
		a = SiftOracleTests.LoadImage(root.GetProperty("images").GetProperty("a"));
		b = SiftOracleTests.LoadImage(root.GetProperty("images").GetProperty("b"));
		return root.GetProperty("cases").EnumerateArray().First(e => e.GetProperty("name").GetString() == caseName);
	}

	private static FeatureExtractor Create(string caseName)
	{
		var options = new FeatureExtractionOptions(FeatureExtractorType.Sift);
		if (caseName == "NumOctaves1")
		{
			options.Sift.NumOctaves = 1;
		}

		return FeatureExtractor.Create(options);
	}

	private static (List<FeatureKeypoint> Keypoints, FeatureDescriptors Descriptors) Extract(FeatureExtractor extractor, Bitmap bitmap)
	{
		var keypoints = new List<FeatureKeypoint>();
		var descriptors = new FeatureDescriptors();
		if (!extractor.Extract(bitmap, keypoints, descriptors))
		{
			throw new InvalidOperationException("extraction failed");
		}

		return (keypoints, descriptors);
	}

	[Test]
	[MethodDataSource(nameof(CaseNames))]
	public async Task CSharpOnly_ReusedExtractorEqualsFreshExtractor(string caseName)
	{
		LoadCase(caseName, out Bitmap a, out Bitmap b);

		FeatureExtractor reused = Create(caseName);
		var fromA = Extract(reused, a);
		var reusedB = Extract(reused, b);
		var freshB = Extract(Create(caseName), b);

		// The precondition that reaches the upstream bug under the default options: every
		// keypoint of A is in octave -1 (scales below the octave-0 base of 2.02), so A leaves the
		// gradient of octave -1 cached, and octave -1 is where B starts.
		await Assert.That(fromA.Keypoints.Count).IsGreaterThan(0);
		foreach (FeatureKeypoint k in fromA.Keypoints)
		{
			await Assert.That(k.ComputeScale()).IsLessThan(2.0f);
		}

		await Assert.That(reusedB.Keypoints.Count).IsEqualTo(freshB.Keypoints.Count);
		for (int i = 0; i < freshB.Keypoints.Count; ++i)
		{
			await Assert.That(reusedB.Keypoints[i]).IsEqualTo(freshB.Keypoints[i]);
		}

		await Assert.That(reusedB.Descriptors.Data.Data.SequenceEqual(freshB.Descriptors.Data.Data)).IsTrue();
	}

	[Test]
	[MethodDataSource(nameof(CaseNames))]
	public async Task CSharpOnly_ReusedExtractorMatchesPycolmapFreshExtractor(string caseName)
	{
		JsonElement c = LoadCase(caseName, out Bitmap a, out Bitmap b);
		double[][] fresh = c.GetProperty("fresh").GetProperty("keypoints").EnumerateArray().Select(OracleFixture.Doubles).ToArray();
		byte[] freshDescriptors = Convert.FromBase64String(c.GetProperty("fresh").GetProperty("descriptors").GetString()!);
		int reusedCount = c.GetProperty("reused").GetProperty("keypoints").GetArrayLength();

		// Upstream, reuse changes the result: B gets one more keypoint orientation here.
		await Assert.That(reusedCount).IsNotEqualTo(fresh.Length);

		FeatureExtractor extractor = Create(caseName);
		var fromA = Extract(extractor, a);
		await Assert.That(fromA.Keypoints.Count).IsEqualTo(c.GetProperty("a").GetProperty("keypoints").GetArrayLength());
		var got = Extract(extractor, b);

		await Assert.That(got.Keypoints.Count).IsEqualTo(fresh.Length);
		double maxKeypointDiff = 0;
		for (int i = 0; i < fresh.Length; ++i)
		{
			FeatureKeypoint k = got.Keypoints[i];
			float[] values = [k.X, k.Y, k.A11, k.A12, k.A21, k.A22];
			for (int j = 0; j < 6; ++j)
			{
				maxKeypointDiff = Math.Max(maxKeypointDiff, Math.Abs(values[j] - fresh[i][j]));
			}
		}

		int maxByteDiff = 0;
		int mismatchedBytes = 0;
		for (int i = 0; i < freshDescriptors.Length; ++i)
		{
			int diff = Math.Abs(got.Descriptors.Data.Data[i] - freshDescriptors[i]);
			maxByteDiff = Math.Max(maxByteDiff, diff);
			mismatchedBytes += diff != 0 ? 1 : 0;
		}

		// SiftOracleTests' FMA-contraction tolerance (docs/CPP_DIVERGENCES.md entry 41).
		await Assert.That(got.Descriptors.Data.Data.Length).IsEqualTo(freshDescriptors.Length);
		await Assert.That(maxKeypointDiff).IsLessThanOrEqualTo(2e-3);
		await Assert.That(maxByteDiff).IsLessThanOrEqualTo(1);
		await Assert.That(mismatchedBytes).IsLessThanOrEqualTo(freshDescriptors.Length / 1000);
	}
}
