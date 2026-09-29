// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// CovariantSiftOracleTests (C#-only): ColmapSharp/Feature/CovariantSift.cs against pycolmap's
// covariant SIFT extractor on covariant_sift.json (oracle/fixture_covariant_sift.py): affine
// shape, upright, domain-size pooling, L2 normalization and max_num_features truncation on
// the sift_test.cc square and a seeded texture.
//
// Not bit-exact, for the reason SiftOracleTests gives: the macOS arm64 wheel's VLFeat fuses
// multiply-adds (divergence 41). The bit-exact Tier A check is
// VlCovDetTests, against the same VLFeat compiled unfused. Here every keypoint must pair with
// a distinct pycolmap keypoint within the FMA drift, with descriptor bytes within 1 on at most
// 0.1% of the bytes. Pairing, not index order, because the order legitimately differs: on the
// symmetric square two orientations of one feature score equally, so fusing flips which one
// comes first; and COLMAP's std::sort orders equal (octave, level) keys differently from the
// port's stable sort (entry 57).
//
// Two cases differ by more than that, both pinned in CSharpOnly_KnownDivergences:
// TextureAffineMaxFeatures keeps a different last keypoint (the truncation keeps the first
// keypoint of the next (octave, level) group, and which one is first is the std::sort tie
// order, entry 57); TextureDsp has one keypoint whose frame agrees but whose DSP descriptor
// differs, because a pooled scale's scale-space level choice (a floor of a log2) flips under
// the wheel's fused arithmetic (entry 41).

using System.Text.Json;

using ColmapSharp.Feature;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class CovariantSiftOracleTests
{
	private static readonly string[] KnownDivergentCases = ["TextureAffineMaxFeatures", "TextureDsp"];

	public static IEnumerable<string> CaseNames()
	{
		JsonElement root = OracleFixture.Load("covariant_sift.json");
		return root.GetProperty("cases").EnumerateArray().Select(c => c.GetProperty("name").GetString()!)
			.Where(n => !KnownDivergentCases.Contains(n)).ToArray();
	}

	[Test]
	[MethodDataSource(nameof(CaseNames))]
	public async Task CSharpOnly_MatchesPycolmapWithinFmaTolerance(string caseName)
	{
		Comparison r = Compare(caseName);
		await Assert.That(r.Count).IsEqualTo(r.ExpectedCount);
		await Assert.That(r.RowKeypointDiff.Max()).IsLessThanOrEqualTo(2e-3);
		await Assert.That(r.RowMaxByteDiff.Max()).IsLessThanOrEqualTo(1);
		await Assert.That(r.RowMismatchedBytes.Sum()).IsLessThanOrEqualTo(r.ExpectedCount * 128 / 1000);
	}

	[Test]
	public async Task CSharpOnly_KnownDivergences()
	{
		// Entry 57: same count, all but the last keypoint pair up; the last one does not.
		Comparison truncated = Compare("TextureAffineMaxFeatures");
		await Assert.That(truncated.Count).IsEqualTo(truncated.ExpectedCount);
		await Assert.That(truncated.RowKeypointDiff.Take(truncated.Count - 1).Max()).IsLessThanOrEqualTo(2e-3);
		await Assert.That(truncated.RowKeypointDiff[^1]).IsGreaterThan(1.0);

		// Entry 41: every frame pairs up; exactly one DSP descriptor is off by more than 1.
		Comparison dsp = Compare("TextureDsp");
		await Assert.That(dsp.Count).IsEqualTo(dsp.ExpectedCount);
		await Assert.That(dsp.RowKeypointDiff.Max()).IsLessThanOrEqualTo(2e-3);
		await Assert.That(dsp.RowMaxByteDiff.Count(d => d > 1)).IsEqualTo(1);
		int[] others = dsp.RowMismatchedBytes.Where((_, i) => dsp.RowMaxByteDiff[i] <= 1).ToArray();
		await Assert.That(others.Sum()).IsLessThanOrEqualTo(dsp.ExpectedCount * 128 / 1000);
	}

	private static Comparison Compare(string caseName)
	{
		JsonElement c = OracleFixture.Load("covariant_sift.json").GetProperty("cases").EnumerateArray()
			.First(e => e.GetProperty("name").GetString() == caseName);
		JsonElement image = OracleFixture.Load("sift.json").GetProperty("images").GetProperty(c.GetProperty("image").GetString()!);
		Bitmap bitmap = SiftOracleTests.LoadImage(image);
		SiftExtractionOptions options = ParseOptions(c.GetProperty("options"));

		FeatureExtractor extractor = FeatureExtractor.Create(new FeatureExtractionOptions { Sift = options });
		if (extractor is not CovariantSiftCpuFeatureExtractor)
		{
			throw new InvalidOperationException("The fixture options must select the covariant extractor.");
		}

		var keypoints = new List<FeatureKeypoint>();
		var descriptors = new FeatureDescriptors();
		if (!extractor.Extract(bitmap, keypoints, descriptors) || descriptors.Data.Rows != keypoints.Count ||
			descriptors.Type != FeatureExtractorType.Sift)
		{
			throw new InvalidOperationException("Extraction failed.");
		}

		double[][] expectedKeypoints = c.GetProperty("keypoints").EnumerateArray().Select(OracleFixture.Doubles).ToArray();
		byte[] expectedDescriptors = Convert.FromBase64String(c.GetProperty("descriptors").GetString()!);

		// Pair each keypoint with the closest unused expected one.
		var r = new Comparison(keypoints.Count, expectedKeypoints.Length);
		var used = new bool[expectedKeypoints.Length];
		for (int i = 0; i < keypoints.Count; ++i)
		{
			FeatureKeypoint k = keypoints[i];
			float[] got = [k.X, k.Y, k.A11, k.A12, k.A21, k.A22];
			int best = -1;
			double bestDiff = double.PositiveInfinity;
			for (int e = 0; e < expectedKeypoints.Length; ++e)
			{
				if (used[e])
				{
					continue;
				}

				double diff = 0;
				for (int j = 0; j < 6; ++j)
				{
					diff = Math.Max(diff, Math.Abs(got[j] - expectedKeypoints[e][j]));
				}

				if (diff < bestDiff)
				{
					bestDiff = diff;
					best = e;
				}
			}

			if (best < 0)
			{
				r.RowKeypointDiff[i] = double.PositiveInfinity;
				continue;
			}

			used[best] = true;
			r.RowKeypointDiff[i] = bestDiff;
			for (int d = 0; d < 128; ++d)
			{
				int diff = Math.Abs(descriptors.Data[i, d] - expectedDescriptors[(best * 128) + d]);
				r.RowMaxByteDiff[i] = Math.Max(r.RowMaxByteDiff[i], diff);
				r.RowMismatchedBytes[i] += diff != 0 ? 1 : 0;
			}
		}

		return r;
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
				case "estimate_affine_shape": options.EstimateAffineShape = p.Value.GetBoolean(); break;
				case "domain_size_pooling": options.DomainSizePooling = p.Value.GetBoolean(); break;
				default: throw new InvalidOperationException($"Unknown SIFT option in fixture: {p.Name}");
			}
		}

		return options;
	}

	// Per extracted keypoint: distance to its paired expected keypoint, and the descriptor
	// byte differences against the pair's row.
	private sealed class Comparison(int count, int expectedCount)
	{
		public int Count { get; } = count;

		public int ExpectedCount { get; } = expectedCount;

		public double[] RowKeypointDiff { get; } = new double[count];

		public int[] RowMaxByteDiff { get; } = new int[count];

		public int[] RowMismatchedBytes { get; } = new int[count];
	}
}
