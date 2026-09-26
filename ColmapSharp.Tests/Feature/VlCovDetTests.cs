// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VlCovDetTests (C#-only): ColmapSharp/Feature/VLFeat/VlCovDet*.cs, VlScaleSpace.cs,
// VlImOpv.Smooth.cs, VlMathOp.Linear.cs and VlSiftFilter.CalcRawDescriptor, driven by
// CovariantSift.cs's per-scale descriptor loop, against VLFeat itself -
// covariant_sift_vlfeat.json, written by oracle/fixture_covariant_sift.py from
// oracle/sift_harness.c's covdet mode (cpp-reference's VLFeat compiled with
// -ffp-contract=off, driven the way COLMAP's CovariantSiftCPUFeatureExtractor does). Tier A:
// every feature (order, octave, level, frame, peak / edge / orientation scores) and every raw
// float descriptor of every DSP scale must be bit-identical, over plain DoG, affine
// adaptation, upright, domain-size pooling and a first octave of 0.

using System.Runtime.InteropServices;
using System.Text.Json;

using ColmapSharp.Feature;
using ColmapSharp.Feature.VLFeat;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class VlCovDetTests
{
	public static IEnumerable<string> CaseNames()
	{
		JsonElement root = OracleFixture.Load("covariant_sift_vlfeat.json");
		return root.GetProperty("cases").EnumerateArray().Select(c => c.GetProperty("name").GetString()!).ToArray();
	}

	[Test]
	[MethodDataSource(nameof(CaseNames))]
	public async Task CSharpOnly_MatchesUnfusedVLFeatExactly(string caseName)
	{
		JsonElement c = OracleFixture.Load("covariant_sift_vlfeat.json").GetProperty("cases").EnumerateArray()
			.First(e => e.GetProperty("name").GetString() == caseName);
		JsonElement image = OracleFixture.Load("sift.json").GetProperty("images").GetProperty(c.GetProperty("image").GetString()!);
		Bitmap bitmap = SiftOracleTests.LoadImage(image);
		bool dsp = c.GetProperty("dsp").GetBoolean();

		// COLMAP's setup with the default SiftExtractionOptions.
		var covdet = new VlCovDet
		{
			FirstOctave = c.GetProperty("first_octave").GetInt32(),
			OctaveResolution = 3,
			PeakThreshold = 0.02 / 3,
			EdgeThreshold = 10.0,
		};
		float[] image01 = bitmap.RowMajorData.Select(v => (float)v / 255.0f).ToArray();
		covdet.PutImage(image01, bitmap.Width, bitmap.Height);
		covdet.Detect(8192);
		if (c.GetProperty("affine").GetBoolean())
		{
			covdet.ExtractAffineShape();
		}

		if (!c.GetProperty("upright").GetBoolean())
		{
			covdet.ExtractOrientations();
		}

		float dspMinScale = dsp ? (float)(1.0 / 6.0) : 1;
		float dspScaleStep = dsp ? (float)((3.0 - (1.0 / 6.0)) / 10) : 0;
		int dspNumScales = dsp ? 10 : 1;

		JsonElement[] expected = c.GetProperty("features").EnumerateArray().ToArray();
		await Assert.That(covdet.NumFeatures).IsEqualTo(expected.Length);

		VlSiftFilter sift = CovariantSiftCpuFeatureExtractor.CreateDescriptorFilter();
		var patch = new float[CovariantSiftCpuFeatureExtractor.PatchSide * CovariantSiftCpuFeatureExtractor.PatchSide];
		var patchXY = new float[2 * patch.Length];
		var scaled = new float[dspNumScales * VlSiftFilter.DescriptorLength];

		string mismatch = string.Empty;
		for (int i = 0; i < expected.Length && mismatch.Length == 0; ++i)
		{
			VlCovDetFeature f = covdet.Features[i];
			int[] os = expected[i].GetProperty("os").EnumerateArray().Select(e => e.GetInt32()).ToArray();
			double[] values = OracleFixture.Doubles(expected[i].GetProperty("values"));
			float[] got =
			[
				f.Frame.X, f.Frame.Y, f.Frame.A11, f.Frame.A12, f.Frame.A21, f.Frame.A22,
				f.PeakScore, f.EdgeScore, f.OrientationScore,
			];
			if (f.O != os[0] || f.S != os[1])
			{
				mismatch = $"feature {i}: (o, s) = ({f.O}, {f.S}), expected ({os[0]}, {os[1]})";
				break;
			}

			for (int k = 0; k < got.Length; ++k)
			{
				if (got[k] != (float)values[k])
				{
					mismatch = $"feature {i} value {k}: {got[k]:R}, expected {(float)values[k]:R}";
					break;
				}
			}

			CovariantSiftCpuFeatureExtractor.ComputeRawDescriptors(
				covdet, sift, f.Frame, dspMinScale, dspScaleStep, dspNumScales, patch, patchXY, scaled);
			JsonElement[] descriptors = expected[i].GetProperty("d").EnumerateArray().ToArray();
			for (int s = 0; s < dspNumScales && mismatch.Length == 0; ++s)
			{
				float[] want = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String(descriptors[s].GetString()!)).ToArray();
				for (int d = 0; d < want.Length; ++d)
				{
					if (BitConverter.SingleToInt32Bits(scaled[(s * VlSiftFilter.DescriptorLength) + d]) != BitConverter.SingleToInt32Bits(want[d]))
					{
						mismatch = $"feature {i} scale {s} bin {d}: {scaled[(s * VlSiftFilter.DescriptorLength) + d]:R}, expected {want[d]:R}";
						break;
					}
				}
			}
		}

		await Assert.That(mismatch).IsEqualTo(string.Empty);
	}

	[Test]
	public async Task CSharpOnly_PooledMeanAddsRowsInBlocksOfFour()
	{
		// Ten rows whose float sum depends on the grouping: 1 + ((e + e) + (e + e)) + ...
		// differs from a left-to-right sum when e is below half an ulp of 1.
		const float E = 3e-8f;
		float[] rows = [1, E, E, E, E, E, E, E, E, E];
		var mean = new float[1];
		CovariantSiftCpuFeatureExtractor.PooledMean(rows, 10, 1, mean);

		float blocked = 1f + ((E + E) + (E + E));
		blocked += (E + E) + (E + E);
		blocked += E;
		await Assert.That(mean[0]).IsEqualTo(blocked / 10f);
	}
}
