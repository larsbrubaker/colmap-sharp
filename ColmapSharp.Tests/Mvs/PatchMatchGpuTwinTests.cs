// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchGpuTwinTests: C#-only end-to-end check of the GPU PatchMatch pipeline. COLMAP has no
// counterpart (its PatchMatch is CUDA). The orchestrator (ColmapSharp/Mvs/PatchMatchGpu*.cs)
// records its whole schedule - init, every sweep's bands, the last sweep's filter, the
// rotations and buffer roles - on ReferenceComputeDevice (ColmapSharp/Mvs/Testing/), the CPU
// twin that runs each WGSL kernel through the production CPU code over the GPU's buffers. The
// results must be bit-identical to PatchMatchCpu.Run on the same problem and seed (Tier A:
// same arithmetic, so any difference is a packing, schedule or buffer-role bug in the
// orchestrator or the twin). PatchMatchGpuTests pins the recorded schedule;
// ReferenceComputeDeviceTests pins each kernel twin on its own; this pins the two together.
// The scene and options are PatchMatchSweepBandTests' (PatchMatchRunTests.Scene(26, 19), two
// iterations), run with the planned bands and with a forced 4-row band so every sweep of both
// orientations runs several sweep_band dispatches.

using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Testing;

using Image = ColmapSharp.Mvs.Image;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchGpuTwinTests
{
	private const int Width = 26;
	private const int Height = 19;

	public static IEnumerable<(string Config, int BandRows)> Cases()
	{
		foreach (string config in new[] { "photometric", "geometric", "filter" })
		{
			// 0: the plan's bands (one per sweep at this size); 4: five bands across the
			// 19 rows, seven across the 26.
			yield return (config, 0);
			yield return (config, 4);
		}
	}

	[Test]
	[MethodDataSource(nameof(Cases))]
	public async Task GpuRunOnTwin_IsBitIdenticalToCpuRun(string config, int bandRows)
	{
		(PatchMatchOptions options, PatchMatch.Problem problem) = Setup(config);
		const ulong Seed = 0x5EED_1234_ABCDUL;

		var cpu = new PatchMatchCpu(options, problem, Seed);
		cpu.Run();

		var gpu = new PatchMatchGpu(options, problem, Seed) { SweepBandRows = bandRows };
		var device = new ReferenceComputeDevice();
		PatchMatchGpuPlan.TryCreate(gpu.ProblemShape, options, device.Limits, null, out PatchMatchGpuPlan? plan, out _, bandRows);
		int[] expectedBands = bandRows == 0 ? [1, 1] : [5, 7];
		await Assert.That(plan!.BandCount(0)).IsEqualTo(expectedBands[0]);
		await Assert.That(plan.BandCount(1)).IsEqualTo(expectedBands[1]);
		await gpu.RunAsync(device);

		await AssertBits("depth", gpu.GetDepthMap().Data, cpu.GetDepthMap().Data);
		await AssertBits("normals", gpu.GetNormalMap().Data, cpu.GetNormalMap().Data);
		await AssertBits("selection probabilities", gpu.GetSelProbMap().Data, cpu.GetSelProbMap().Data);
		List<int> cpuConsistent = cpu.GetConsistentImageIdxs();
		await Assert.That(string.Join(",", gpu.GetConsistentImageIdxs())).IsEqualTo(string.Join(",", cpuConsistent));
		if (options.Filter)
		{
			// Guard against a vacuous pass: the filter must keep some pixels.
			await Assert.That(cpuConsistent.Count).IsGreaterThan(0);
		}
	}

	private static (PatchMatchOptions Options, PatchMatch.Problem Problem) Setup(string config)
	{
		(List<Image> images, List<DepthMap> truth, List<NormalMap> normals) = PatchMatchRunTests.Scene(Width, Height);
		return config switch
		{
			"photometric" => (PatchMatchRunTests.Options(2, geomConsistency: false, filter: false), PatchMatchRunTests.Problem(images)),

			// Filtering on, so the last sweep also runs the geometric consistency filter.
			"geometric" => (PatchMatchRunTests.Options(2, geomConsistency: true, filter: true), PatchMatchRunTests.Problem(images, truth, normals)),
			_ => (PatchMatchRunTests.Options(2, geomConsistency: false, filter: true), PatchMatchRunTests.Problem(images)),
		};
	}

	// Bitwise, so NaN payloads and signed zeros count; reports the first differing index.
	private static async Task AssertBits(string what, float[] actual, float[] expected)
	{
		await Assert.That(actual.Length).IsEqualTo(expected.Length);
		int firstDiff = -1;
		int diffCount = 0;
		for (int i = 0; i < actual.Length; ++i)
		{
			if (BitConverter.SingleToUInt32Bits(actual[i]) != BitConverter.SingleToUInt32Bits(expected[i]))
			{
				diffCount++;
				if (firstDiff < 0)
				{
					firstDiff = i;
				}
			}
		}

		string detail = firstDiff < 0 ? "" : $"{what}: {diffCount} of {actual.Length} differ, first at {firstDiff}: {actual[firstDiff]:R} vs {expected[firstDiff]:R}";
		await Assert.That(detail).IsEqualTo("");
	}
}
