// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VlSiftFilterTests (C#-only): ColmapSharp/Feature/VLFeat/VlSiftFilter*.cs against VLFeat
// itself - sift_vlfeat.json, written by oracle/fixture_sift.py from oracle/sift_harness.c, which
// compiles cpp-reference's VLFeat with -ffp-contract=off and drives it the way COLMAP's
// SiftCPUFeatureExtractor does. Tier A: every keypoint field, orientation and raw float
// descriptor value must be bit-identical, over the upsample (o_min -1, -2), copy (0) and
// downsample (1) scale-space paths and upright mode. VLFeat has no test suite of its own in
// COLMAP's tree, so this fixture is the specification for the port.

using System.Runtime.InteropServices;
using System.Text.Json;

using ColmapSharp.Feature.VLFeat;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class VlSiftFilterTests
{
	public static IEnumerable<string> CaseNames()
	{
		JsonElement root = OracleFixture.Load("sift_vlfeat.json");
		return root.GetProperty("cases").EnumerateArray().Select(c => c.GetProperty("name").GetString()!).ToArray();
	}

	[Test]
	[MethodDataSource(nameof(CaseNames))]
	public async Task CSharpOnly_MatchesUnfusedVLFeatExactly(string caseName)
	{
		JsonElement c = OracleFixture.Load("sift_vlfeat.json").GetProperty("cases").EnumerateArray()
			.First(e => e.GetProperty("name").GetString() == caseName);
		JsonElement image = OracleFixture.Load("sift.json").GetProperty("images").GetProperty(c.GetProperty("image").GetString()!);
		Bitmap bitmap = SiftOracleTests.LoadImage(image);
		bool upright = c.GetProperty("upright").GetBoolean();

		var sift = new VlSiftFilter(
			bitmap.Width, bitmap.Height, c.GetProperty("noctaves").GetInt32(), 3, c.GetProperty("o_min").GetInt32())
		{
			PeakThreshold = 0.02 / 3,
			EdgeThreshold = 10.0,
		};

		var image01 = bitmap.RowMajorData.Select(v => (float)v / 255.0f).ToArray();
		JsonElement[] expected = c.GetProperty("keypoints").EnumerateArray().ToArray();

		int next = 0;
		string mismatch = string.Empty;
		Span<double> angles = stackalloc double[4];
		var desc = new float[VlSiftFilter.DescriptorLength];
		bool first = true;
		while (mismatch.Length == 0)
		{
			if (first)
			{
				if (!sift.ProcessFirstOctave(image01))
				{
					break;
				}

				first = false;
			}
			else if (!sift.ProcessNextOctave())
			{
				break;
			}

			sift.Detect();
			foreach (VlSiftKeypoint k in sift.Keypoints.ToArray())
			{
				if (next >= expected.Length)
				{
					mismatch = $"extra keypoint {next}";
					break;
				}

				double[] ek = OracleFixture.Doubles(expected[next].GetProperty("k"));
				double[] got = [k.O, k.IX, k.IY, k.IS, k.X, k.Y, k.S, k.Sigma];
				for (int j = 0; j < 8 && mismatch.Length == 0; ++j)
				{
					if ((j < 4 ? got[j] : (float)got[j]) != (j < 4 ? ek[j] : (float)ek[j]))
					{
						mismatch = $"keypoint {next} field {j}: got {got[j]:R}, expected {ek[j]:R}";
					}
				}

				int numOrientations;
				if (upright)
				{
					numOrientations = 1;
					angles[0] = 0.0;
				}
				else
				{
					numOrientations = sift.CalcKeypointOrientations(angles, k);
				}

				JsonElement[] ea = expected[next].GetProperty("a").EnumerateArray().ToArray();
				if (mismatch.Length == 0 && ea.Length != numOrientations)
				{
					mismatch = $"keypoint {next}: {numOrientations} orientations, expected {ea.Length}";
				}

				for (int o = 0; o < numOrientations && mismatch.Length == 0; ++o)
				{
					sift.CalcKeypointDescriptor(desc, k, angles[o]);
					double expectedAngle = ea[o].GetProperty("angle").GetDouble();
					float[] expectedDesc = MemoryMarshal.Cast<byte, float>(
						Convert.FromBase64String(ea[o].GetProperty("descriptor").GetString()!)).ToArray();
					if (angles[o] != expectedAngle)
					{
						mismatch = $"keypoint {next} angle {o}: got {angles[o]:R}, expected {expectedAngle:R}";
					}

					for (int d = 0; d < desc.Length && mismatch.Length == 0; ++d)
					{
						// Bit comparison, so a NaN or signed-zero difference would show too.
						if (BitConverter.SingleToInt32Bits(desc[d]) != BitConverter.SingleToInt32Bits(expectedDesc[d]))
						{
							mismatch = $"keypoint {next} angle {o} bin {d}: got {desc[d]:R}, expected {expectedDesc[d]:R}";
						}
					}
				}

				if (mismatch.Length > 0)
				{
					break;
				}

				next++;
			}
		}

		await Assert.That(mismatch).IsEqualTo(string.Empty);
		await Assert.That(next).IsEqualTo(expected.Length);
	}
}
