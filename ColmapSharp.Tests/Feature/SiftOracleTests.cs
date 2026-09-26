// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SiftOracleTests (C#-only): ColmapSharp/Feature/Sift.cs and VLFeat/VlSiftFilter*.cs against
// pycolmap's CPU SIFT on the fixture written by oracle/fixture_sift.py (a white square and a
// seeded texture, under several option sets that reach every scale-space path).
//
// Not bit-exact, on purpose: the macOS arm64 wheel's VLFeat is compiled with Apple clang's
// default -ffp-contract=on, which fuses a*b + c (the Gaussian convolution's accumulation among
// others) into FMAs that round once; the port never fuses (CLAUDE.md "No FMA"). The bit-exact
// Tier A check is VlSiftFilterTests, against the same VLFeat compiled unfused, where every
// value agrees. Here the same keypoints must come out (same count and order, so the same
// detections and DoG-level truncation), with values within the drift fusing causes
// (docs/CPP_DIVERGENCES.md entry 41).

using System.Text.Json;

using ColmapSharp.Feature;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class SiftOracleTests
{
	public static IEnumerable<string> CaseNames()
	{
		JsonElement root = OracleFixture.Load("sift.json");
		return root.GetProperty("cases").EnumerateArray().Select(c => c.GetProperty("name").GetString()!).ToArray();
	}

	[Test]
	[MethodDataSource(nameof(CaseNames))]
	public async Task CSharpOnly_MatchesPycolmapWithinFmaTolerance(string caseName)
	{
		JsonElement root = OracleFixture.Load("sift.json");
		JsonElement c = root.GetProperty("cases").EnumerateArray().First(e => e.GetProperty("name").GetString() == caseName);
		Bitmap bitmap = LoadImage(root.GetProperty("images").GetProperty(c.GetProperty("image").GetString()!));
		SiftExtractionOptions options = ParseOptions(c.GetProperty("options"));

		var extractor = new SiftCpuFeatureExtractor(options);
		var keypoints = new List<FeatureKeypoint>();
		var descriptors = new FeatureDescriptors();
		await Assert.That(extractor.Extract(bitmap, keypoints, descriptors)).IsTrue();

		double[][] expectedKeypoints = c.GetProperty("keypoints").EnumerateArray().Select(OracleFixture.Doubles).ToArray();
		byte[] expectedDescriptors = Convert.FromBase64String(c.GetProperty("descriptors").GetString()!);

		await Assert.That(keypoints.Count).IsEqualTo(expectedKeypoints.Length);
		await Assert.That(descriptors.Type).IsEqualTo(FeatureExtractorType.Sift);
		await Assert.That(descriptors.Data.Rows).IsEqualTo(expectedKeypoints.Length);
		await Assert.That(descriptors.Data.Cols).IsEqualTo(128);

		double maxKeypointDiff = 0;
		for (int i = 0; i < keypoints.Count; ++i)
		{
			FeatureKeypoint k = keypoints[i];
			float[] got = [k.X, k.Y, k.A11, k.A12, k.A21, k.A22];
			for (int j = 0; j < 6; ++j)
			{
				maxKeypointDiff = Math.Max(maxKeypointDiff, Math.Abs(got[j] - expectedKeypoints[i][j]));
			}
		}

		int maxByteDiff = 0;
		int mismatchedBytes = 0;
		for (int i = 0; i < expectedDescriptors.Length; ++i)
		{
			int diff = Math.Abs(descriptors.Data.Data[i] - expectedDescriptors[i]);
			maxByteDiff = Math.Max(maxByteDiff, diff);
			mismatchedBytes += diff != 0 ? 1 : 0;
		}

		// Observed on these fixtures: at most 1.03e-3 in a keypoint value, 1 gray level in a
		// descriptor byte, and at most 0.05% of the bytes off.
		await Assert.That(maxKeypointDiff).IsLessThanOrEqualTo(2e-3);
		await Assert.That(maxByteDiff).IsLessThanOrEqualTo(1);
		await Assert.That(mismatchedBytes).IsLessThanOrEqualTo(expectedDescriptors.Length / 1000);
	}

	internal static Bitmap LoadImage(JsonElement image)
	{
		int width = image.GetProperty("width").GetInt32();
		int height = image.GetProperty("height").GetInt32();
		byte[] pixels = Convert.FromBase64String(image.GetProperty("pixels").GetString()!);
		var bitmap = new Bitmap(width, height, asRgb: false);
		pixels.CopyTo(bitmap.RowMajorData, 0);
		return bitmap;
	}

	private static SiftExtractionOptions ParseOptions(JsonElement overrides)
	{
		var options = new SiftExtractionOptions();
		foreach (JsonProperty p in overrides.EnumerateObject())
		{
			switch (p.Name)
			{
				case "normalization": options.Normalization = (SiftNormalization)p.Value.GetInt32(); break;
				case "upright": options.Upright = p.Value.GetBoolean(); break;
				case "max_num_features": options.MaxNumFeatures = p.Value.GetInt32(); break;
				case "first_octave": options.FirstOctave = p.Value.GetInt32(); break;
				case "num_octaves": options.NumOctaves = p.Value.GetInt32(); break;
				case "max_num_orientations": options.MaxNumOrientations = p.Value.GetInt32(); break;
				case "peak_threshold": options.PeakThreshold = p.Value.GetDouble(); break;
				default: throw new InvalidOperationException($"Unknown SIFT option in fixture: {p.Name}");
			}
		}

		return options;
	}
}
