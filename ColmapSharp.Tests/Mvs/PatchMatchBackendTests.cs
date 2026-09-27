// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchBackendTests: C#-only tests of PatchMatch's public GPU entry point
// (ColmapSharp/Mvs/PatchMatch.cs: RunAsync with an IComputeDevice, Backend,
// FallbackReason; docs/CPP_DIVERGENCES.md entry 136). COLMAP has no counterpart: its PatchMatch
// is CUDA-only and has no fallback. The device is ReferenceComputeDevice
// (ColmapSharp/Mvs/Testing/), the CPU twin that runs the WGSL kernels through the production
// CPU code, so a GPU run is bit-identical to the CPU run (Tier A for this pairing;
// PatchMatchGpuTwinTests pins the orchestrator itself). The scene and options are
// PatchMatchGpuTwinTests': PatchMatchSyntheticScene.Scene(26, 19), two iterations, with filtering.

using ColmapSharp.Compute;
using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Testing;

using Image = ColmapSharp.Mvs.Image;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchBackendTests
{
	private const int Width = 26;
	private const int Height = 19;

	[Test]
	public async Task RunAsync_OnTwinDevice_RunsOnGpuBitIdenticalToCpuRun()
	{
		(PatchMatchOptions options, PatchMatch.Problem problem) = Setup();
		var cpu = new PatchMatch(options, problem);
		cpu.Run();

		var gpu = new PatchMatch(options, problem);
		var device = new ReferenceComputeDevice();
		await gpu.RunAsync(device);

		await Assert.That(gpu.Backend).IsEqualTo(PatchMatchBackend.Gpu);
		await Assert.That(gpu.FallbackReason).IsNull();
		await Assert.That(device.ExecutedDispatchCount).IsGreaterThan(0);
		await AssertSameResults(gpu, cpu);
	}

	[Test]
	public async Task RunAsync_DeviceTooSmall_RunsOnCpuWithPlannersReason()
	{
		(PatchMatchOptions options, PatchMatch.Problem problem) = Setup();
		var cpu = new PatchMatch(options, problem);
		cpu.Run();

		// 1 KiB storage bindings: no map of this problem fits.
		ComputeDeviceLimits limits = ComputeDeviceLimits.Defaults with { MaxStorageBufferBindingSize = 1024 };
		PatchMatchGpuPlan.TryCreate(PatchMatchGpu.ShapeOf(problem), options, limits, null, out _, out string? expectedReason);
		await Assert.That(expectedReason).IsNotNull();

		var fallback = new PatchMatch(options, problem);
		var device = new ReferenceComputeDevice(limits);
		await fallback.RunAsync(device);

		await Assert.That(fallback.Backend).IsEqualTo(PatchMatchBackend.Cpu);
		await Assert.That(fallback.FallbackReason).IsEqualTo(expectedReason);
		await Assert.That(fallback.FallbackReason!.EndsWith(" Using the CPU.", StringComparison.Ordinal)).IsTrue();
		await Assert.That(device.Commands.Count).IsEqualTo(0);
		await AssertSameResults(fallback, cpu);
	}

	[Test]
	public async Task RunAsync_WithoutDevice_RunsOnCpu()
	{
		(PatchMatchOptions options, PatchMatch.Problem problem) = Setup();
		var cpu = new PatchMatch(options, problem);
		cpu.Run();

		var run = new PatchMatch(options, problem);
		await run.RunAsync(null);

		await Assert.That(run.Backend).IsEqualTo(PatchMatchBackend.Cpu);
		await Assert.That(run.FallbackReason).IsNull();
		await AssertSameResults(run, cpu);
	}

	[Test]
	public async Task RunAsync_OnNonBlockingDevice_RunsOnGpu()
	{
		// A browser-like device: every flush and read completes asynchronously.
		(PatchMatchOptions options, PatchMatch.Problem problem) = Setup();
		var cpu = new PatchMatch(options, problem);
		cpu.Run();

		var gpu = new PatchMatch(options, problem);
		await gpu.RunAsync(new ReferenceComputeDevice { SupportsBlockingWait = false });

		await Assert.That(gpu.Backend).IsEqualTo(PatchMatchBackend.Gpu);
		await AssertSameResults(gpu, cpu);
	}

	[Test]
	public async Task ShapeOf_MatchesTheConstructedRunsShape()
	{
		// Source images of different sizes, so the layer size is a maximum over sources.
		(List<Image> images, _, _) = PatchMatchSyntheticScene.Scene(Width, Height);
		(List<Image> small, _, _) = PatchMatchSyntheticScene.Scene(Width - 5, Height + 3);
		images[1] = small[1];
		PatchMatch.Problem problem = PatchMatchSyntheticScene.Problem(images);
		PatchMatchOptions options = PatchMatchSyntheticScene.Options(1, geomConsistency: false, filter: false);

		PatchMatchGpuProblemShape shape = PatchMatchGpu.ShapeOf(problem);
		await Assert.That(shape).IsEqualTo(new PatchMatchGpu(options, problem).ProblemShape);
		await Assert.That(shape.SourceLayerHeight).IsEqualTo(Height + 3);
	}

	private static (PatchMatchOptions Options, PatchMatch.Problem Problem) Setup()
	{
		(List<Image> images, _, _) = PatchMatchSyntheticScene.Scene(Width, Height);
		return (PatchMatchSyntheticScene.Options(2, geomConsistency: false, filter: true), PatchMatchSyntheticScene.Problem(images));
	}

	private static async Task AssertSameResults(PatchMatch actual, PatchMatch expected)
	{
		await AssertBits("depth", actual.GetDepthMap().Data, expected.GetDepthMap().Data);
		await AssertBits("normals", actual.GetNormalMap().Data, expected.GetNormalMap().Data);
		await AssertBits("selection probabilities", actual.GetSelProbMap().Data, expected.GetSelProbMap().Data);

		ConsistencyGraph actualGraph = actual.GetConsistencyGraph();
		ConsistencyGraph expectedGraph = expected.GetConsistencyGraph();
		int consistentPixels = 0;
		var differing = new List<string>();
		for (int row = 0; row < Height; ++row)
		{
			for (int col = 0; col < Width; ++col)
			{
				int[] a = actualGraph.GetImageIdxs(row, col).ToArray();
				int[] e = expectedGraph.GetImageIdxs(row, col).ToArray();
				consistentPixels += e.Length > 0 ? 1 : 0;
				if (!a.AsSpan().SequenceEqual(e))
				{
					differing.Add($"({row}, {col})");
				}
			}
		}

		await Assert.That(string.Join(" ", differing)).IsEqualTo("");

		// Guard against a vacuous pass: the filter must keep some pixels.
		await Assert.That(consistentPixels).IsGreaterThan(0);
	}

	// Bitwise, so NaN payloads and signed zeros count.
	private static async Task AssertBits(string what, float[] actual, float[] expected)
	{
		await Assert.That(actual.Length).IsEqualTo(expected.Length);
		int diffCount = 0;
		for (int i = 0; i < actual.Length; ++i)
		{
			diffCount += BitConverter.SingleToUInt32Bits(actual[i]) != BitConverter.SingleToUInt32Bits(expected[i]) ? 1 : 0;
		}

		await Assert.That($"{what}: {diffCount} differ").IsEqualTo($"{what}: 0 differ");
	}
}
