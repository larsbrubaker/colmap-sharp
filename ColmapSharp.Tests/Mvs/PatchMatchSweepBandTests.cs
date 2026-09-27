// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchSweepBandTests: C#-only golden tests for the sweep schedule of
// ColmapSharp/Mvs/PatchMatchCpu.cs and PatchMatchCpu.Sweep.cs. COLMAP has no test for this;
// the GPU port (PORTING_PLAN.md Phase 13) runs each column sweep as a backward-message pass
// followed by row bands, and filters in a separate pass after the last sweep, so these tests
// pin that the banded schedule gives exactly the output the whole-column sweep gave. The
// golden hashes were recorded from the unbanded sweep before it was split: a SHA-256 over the
// depth map, normal map, selection probability map (float bits) and the consistent image
// list, on the synthetic scene (Mvs/Testing/PatchMatchSyntheticScene.cs). Any change to PatchMatch arithmetic
// changes these hashes; re-record them only for a deliberate, reviewed change of results.

using System.Security.Cryptography;

using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Testing;

using Image = ColmapSharp.Mvs.Image;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchSweepBandTests
{
	private const int Width = 26;
	private const int Height = 19;

	// Recorded from the whole-column sweep before the band split (see the file header).
	private const string PhotometricHash = "917B622875DD2552E76325D6972EE1F5B732A29B967BE97346FA04FA675583DB";
	private const string GeometricHash = "4099C752C6CC5F6AFD4AAA4FE7156A5926BF6CCA0DDFAFE938D185119C571133";
	private const string FilterHash = "532D76791BAA6C608FC834F05E18C764576CB32300754F867AC991CB6048C469";

	public static IEnumerable<(string Config, int BandHeight, int NumThreads)> Cases()
	{
		foreach (string config in new[] { "photometric", "geometric", "filter" })
		{
			// 0 is the default (whole height); 1 and 7 split the 19- and 26-row sweeps into
			// many bands, 7 with a ragged last band.
			foreach (int bandHeight in new[] { 0, 1, 7 })
			{
				foreach (int numThreads in new[] { 1, -1 })
				{
					yield return (config, bandHeight, numThreads);
				}
			}
		}
	}

	[Test]
	[MethodDataSource(nameof(Cases))]
	public async Task BandedSweep_MatchesGoldenHash(string config, int bandHeight, int numThreads)
	{
		string expected = config switch
		{
			"photometric" => PhotometricHash,
			"geometric" => GeometricHash,
			_ => FilterHash,
		};

		await Assert.That(RunHash(config, bandHeight, numThreads)).IsEqualTo(expected);
	}

	private static string RunHash(string config, int bandHeight, int numThreads)
	{
		(List<Image> images, List<DepthMap> truth, List<NormalMap> normals) = PatchMatchSyntheticScene.Scene(Width, Height);
		PatchMatchOptions options;
		PatchMatch.Problem problem;
		switch (config)
		{
			case "photometric":
				options = PatchMatchSyntheticScene.Options(2, geomConsistency: false, filter: false, numThreads);
				problem = PatchMatchSyntheticScene.Problem(images);
				break;
			case "geometric":
				// Filtering on, so the last sweep also runs the geometric consistency filter.
				options = PatchMatchSyntheticScene.Options(2, geomConsistency: true, filter: true, numThreads);
				problem = PatchMatchSyntheticScene.Problem(images, truth, normals);
				break;
			default:
				options = PatchMatchSyntheticScene.Options(2, geomConsistency: false, filter: true, numThreads);
				problem = PatchMatchSyntheticScene.Problem(images);
				break;
		}

		var run = new PatchMatchCpu(options, problem) { SweepBandHeight = bandHeight };
		run.Run();

		using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
		AppendFloats(hash, run.GetDepthMap().Data);
		AppendFloats(hash, run.GetNormalMap().Data);
		AppendFloats(hash, run.GetSelProbMap().Data);
		foreach (int value in run.GetConsistentImageIdxs())
		{
			hash.AppendData(BitConverter.GetBytes(value));
		}

		return Convert.ToHexString(hash.GetHashAndReset());
	}

	private static void AppendFloats(IncrementalHash hash, float[] values)
	{
		foreach (float value in values)
		{
			hash.AppendData(BitConverter.GetBytes(BitConverter.SingleToInt32Bits(value)));
		}
	}
}
