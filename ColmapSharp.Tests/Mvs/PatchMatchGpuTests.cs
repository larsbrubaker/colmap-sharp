// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchGpuTests: C#-only tests of the GPU PatchMatch orchestrator
// (ColmapSharp/Mvs/PatchMatchGpu.cs and PatchMatchGpu.Run.cs) on RecordingComputeDevice, which
// records every call and throws on any broken seam rule (a write racing recorded dispatches, a
// read_write buffer bound twice in one dispatch, a layout mismatch, a limit). COLMAP has no
// counterpart (its PatchMatch is CUDA). These pin the recorded schedule against
// PatchMatchCpu.Run's (kernel order, bands, the last sweep's filter, the rotations and the
// buffer roles that rotate with them), one flush per sweep, cancellation between sweeps, the
// progress fractions, the uniform bytes, and the decoding of the read-back buffers. Whether the
// kernels compute the CPU's numbers is the reference twin's job (ReferenceComputeDevice).

using System.Buffers.Binary;
using System.Runtime.InteropServices;

using ColmapSharp.Compute;
using ColmapSharp.Compute.Testing;
using ColmapSharp.Mvs;

using Image = ColmapSharp.Mvs.Image;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchGpuTests
{
	private const int Width = 26;
	private const int Height = 19;

	private static (PatchMatchGpu Gpu, PatchMatchOptions Options, PatchMatch.Problem Problem) Setup(
		string config, int width = Width, int height = Height, int numIterations = 2, int numSamples = 15)
	{
		(List<Image> images, List<DepthMap> truth, List<NormalMap> normals) = PatchMatchRunTests.Scene(width, height);
		bool geometric = config == "geometric";
		PatchMatchOptions options = PatchMatchRunTests.Options(numIterations, geomConsistency: geometric, filter: config != "photometric");
		options.NumSamples = numSamples;
		PatchMatch.Problem problem = geometric
			? PatchMatchRunTests.Problem(images, truth, normals)
			: PatchMatchRunTests.Problem(images);
		return (new PatchMatchGpu(options, problem), options, problem);
	}

	private static PatchMatchGpuPlan PlanOf(PatchMatchGpu gpu, PatchMatchOptions options)
	{
		PatchMatchGpuPlan.TryCreate(gpu.ProblemShape, options, ComputeDeviceLimits.Defaults, null, out PatchMatchGpuPlan? plan, out _);
		return plan!;
	}

	// The dispatches between flushes: one list of entry points per flush.
	private static List<List<string>> DispatchesPerFlush(RecordingComputeDevice device)
	{
		var result = new List<List<string>>();
		var current = new List<string>();
		foreach (ComputeCommand command in device.Commands)
		{
			if (command is DispatchCommand dispatch)
			{
				current.Add(dispatch.Kernel.Descriptor.EntryPoint);
			}
			else if (command is FlushCommand)
			{
				result.Add(current);
				current = [];
			}
		}

		return result;
	}

	// The label of the buffer a dispatch binds at (group, binding).
	private static string? Bound(DispatchCommand dispatch, int group, int binding) =>
		dispatch.Groups[group].Entries.SingleOrDefault(e => e.Binding == binding).Buffer?.Label;

	private static RecordingComputeBuffer Created(RecordingComputeDevice device, string label) =>
		device.CommandsOf<CreateBufferCommand>().Single(c => c.Buffer.Label == label).Buffer;

	public static IEnumerable<string> Configs() => ["photometric", "geometric", "filter"];

	[Test]
	[MethodDataSource(nameof(Configs))]
	public async Task Schedule_MatchesCpuSweepOrder(string config)
	{
		// 64 x 64 with 100 samples: two bands per sweep in both orientations.
		var (gpu, options, _) = Setup(config, 64, 64, numSamples: 100);
		PatchMatchGpuPlan plan = PlanOf(gpu, options);
		await Assert.That(plan.BandCount(0)).IsEqualTo(2);
		await Assert.That(plan.BandCount(1)).IsEqualTo(2);

		var device = new RecordingComputeDevice();
		await gpu.RunAsync(device);

		bool geometric = config == "geometric";
		bool filter = config != "photometric";
		int sweeps = 4 * options.NumIterations;
		var expected = new List<List<string>>();
		for (int k = 0; k < sweeps; ++k)
		{
			var list = new List<string>();
			if (k == 0)
			{
				if (!geometric)
				{
					list.Add("init_random");
				}

				list.Add("initial_cost");
			}

			list.Add("backward_messages");
			list.AddRange(Enumerable.Repeat("sweep_band", plan.BandCount(k % 2)));
			bool last = k == sweeps - 1;
			if (last && filter)
			{
				list.Add("filter_pixels");
			}

			list.AddRange(["rotate_planes", "rotate_normals", "rotate_planes", "rotate_planes"]);
			if (!last || filter)
			{
				list.Add("rotate_planes");
			}

			expected.Add(list);
		}

		List<List<string>> actual = DispatchesPerFlush(device);
		await Assert.That(actual.Count).IsEqualTo(sweeps);
		for (int k = 0; k < sweeps; ++k)
		{
			await Assert.That(actual[k]).IsEquivalentTo(expected[k], TUnit.Assertions.Enums.CollectionOrdering.Matching);
		}

		// The readbacks run after the last flush and submit nothing more.
		var reads = device.CommandsOf<ReadBufferCommand>();
		await Assert.That(reads.Count).IsEqualTo(filter ? 3 : 2);
		await Assert.That(reads.All(r => r.DispatchCount == 0)).IsTrue();
		await Assert.That(device.CommandsOf<WriteBufferCommand>().Count).IsEqualTo(0);
	}

	[Test]
	public async Task Schedule_BandUniformsCoverEveryRowInOrder()
	{
		var (gpu, options, _) = Setup("photometric", 64, 64, numIterations: 1, numSamples: 100);
		PatchMatchGpuPlan plan = PlanOf(gpu, options);
		var device = new RecordingComputeDevice();
		await gpu.RunAsync(device);

		foreach (int orientation in new[] { 0, 1 })
		{
			// The frame is square, so both orientations sweep 64 rows.
			const int rows = 64;
			int expectedStart = 0;
			for (int b = 0; b < plan.BandCount(orientation); ++b)
			{
				byte[] band = Created(device, $"pm_band_{orientation}_{b}").GetContents();
				int start = BinaryPrimitives.ReadInt32LittleEndian(band);
				int end = BinaryPrimitives.ReadInt32LittleEndian(band.AsSpan(4));
				await Assert.That(start).IsEqualTo(expectedStart);
				await Assert.That(end).IsEqualTo(Math.Min(rows, start + plan.BandRows(orientation)));
				expectedStart = end;
			}

			await Assert.That(expectedStart).IsEqualTo(rows);
		}

		// Sweep k's band dispatches bind orientation k % 2's band uniforms in order, and sweep k's
		// sweep uniform.
		var dispatches = device.CommandsOf<DispatchCommand>().Where(d => d.Kernel.Descriptor.EntryPoint == "sweep_band").ToList();
		int index = 0;
		for (int k = 0; k < 4; ++k)
		{
			for (int b = 0; b < plan.BandCount(k % 2); ++b)
			{
				DispatchCommand d = dispatches[index++];
				await Assert.That(Bound(d, 2, 0)).IsEqualTo($"pm_band_{k % 2}_{b}");
				await Assert.That(Bound(d, 1, 0)).IsEqualTo($"pm_sweep_{k}");
			}
		}
	}

	[Test]
	[MethodDataSource(nameof(Configs))]
	public async Task Schedule_BufferRolesRotateWithTheMaps(string config)
	{
		var (gpu, options, _) = Setup(config);
		var device = new RecordingComputeDevice();
		await gpu.RunAsync(device);

		var dispatches = device.CommandsOf<DispatchCommand>().ToList();
		var backward = dispatches.Where(d => d.Kernel.Descriptor.EntryPoint == "backward_messages").ToList();
		var sweepBands = dispatches.Where(d => d.Kernel.Descriptor.EntryPoint == "sweep_band").ToList();
		await Assert.That(backward.Count).IsEqualTo(4 * options.NumIterations);
		for (int k = 0; k < backward.Count; ++k)
		{
			int p = k % 2;
			string cost = p == 0 ? "pm_source_maps_a" : "pm_source_maps_b";
			string sel = p == 0 ? "pm_source_maps_b" : "pm_source_maps_a";
			await Assert.That(Bound(backward[k], 0, PatchMatchGpuKernels.StateBinding)).IsEqualTo($"pm_state_{p}");
			await Assert.That(Bound(backward[k], 0, PatchMatchGpuKernels.CostsBinding)).IsEqualTo(cost);
			await Assert.That(Bound(backward[k], 0, PatchMatchGpuKernels.SelProbsBinding)).IsEqualTo(sel);
			await Assert.That(Bound(sweepBands[k], 0, PatchMatchGpuKernels.ReferenceBinding)).IsEqualTo($"pm_reference_{p}");
			await Assert.That(Bound(sweepBands[k], 0, PatchMatchGpuKernels.CostsBinding)).IsEqualTo(cost);
			await Assert.That(Bound(sweepBands[k], 0, PatchMatchGpuKernels.SelProbsBinding)).IsEqualTo(sel);
			await Assert.That(Bound(sweepBands[k], 0, PatchMatchGpuKernels.PrevSelProbsBinding)).IsEqualTo("pm_source_maps_c");
		}

		// The rotation copies of each sweep, in PatchMatchCpu.Rotate's order: (source, destination).
		var copies = dispatches.Where(d => d.Kernel.Descriptor.EntryPoint.StartsWith("rotate_", StringComparison.Ordinal)).ToList();
		int index = 0;
		bool filter = config != "photometric";
		for (int k = 0; k < backward.Count; ++k)
		{
			int p = k % 2;
			string cost = p == 0 ? "pm_source_maps_a" : "pm_source_maps_b";
			string sel = p == 0 ? "pm_source_maps_b" : "pm_source_maps_a";
			var expected = new List<(string, string)>
			{
				($"pm_state_{p}", $"pm_state_{1 - p}"),
				($"pm_state_{p}", $"pm_state_{1 - p}"),
				($"pm_reference_{p}", $"pm_reference_{1 - p}"),
				(sel, "pm_source_maps_c"),
			};
			if (k != backward.Count - 1 || filter)
			{
				expected.Add((cost, sel));
			}

			foreach ((string source, string destination) in expected)
			{
				DispatchCommand copy = copies[index++];
				await Assert.That(Bound(copy, 0, PatchMatchGpuKernels.RotateSourceBinding)).IsEqualTo(source);
				await Assert.That(Bound(copy, 0, PatchMatchGpuKernels.RotateDestinationBinding)).IsEqualTo(destination);
			}
		}

		await Assert.That(index).IsEqualTo(copies.Count);

		// The last (odd) sweep filters into its cost buffer, B; the rotation moves the mask to A.
		var filters = dispatches.Where(d => d.Kernel.Descriptor.EntryPoint == "filter_pixels").ToList();
		await Assert.That(filters.Count).IsEqualTo(filter ? 1 : 0);
		if (filter)
		{
			await Assert.That(Bound(filters[0], 0, PatchMatchGpuKernels.CostsBinding)).IsEqualTo("pm_source_maps_b");
			await Assert.That(Bound(filters[0], 0, PatchMatchGpuKernels.SelProbsBinding)).IsEqualTo("pm_source_maps_a");
			await Assert.That(Bound(filters[0], 0, PatchMatchGpuKernels.StateBinding)).IsEqualTo("pm_state_1");
		}

		var reads = device.CommandsOf<ReadBufferCommand>().Select(r => r.Buffer.Label).ToList();
		List<string?> expectedReads = filter
			? ["pm_state_0", "pm_source_maps_c", "pm_source_maps_a"]
			: ["pm_state_0", "pm_source_maps_c"];
		await Assert.That(reads).IsEquivalentTo(expectedReads, TUnit.Assertions.Enums.CollectionOrdering.Matching);
	}

	[Test]
	public async Task Dispatch_ShapesFollowElementCounts()
	{
		var (gpu, _, _) = Setup("filter");
		var device = new RecordingComputeDevice();
		await gpu.RunAsync(device);

		int p = Width * Height;
		foreach (DispatchCommand d in device.CommandsOf<DispatchCommand>())
		{
			await Assert.That(d.Y).IsEqualTo(1u);
		}

		// initial_cost: P S invocations; backward_messages alternates W S and H S.
		var byName = device.CommandsOf<DispatchCommand>().GroupBy(d => d.Kernel.Descriptor.EntryPoint).ToDictionary(g => g.Key, g => g.ToList());
		await Assert.That(byName["initial_cost"][0].X).IsEqualTo((uint)((p * 2 + 63) / 64));
		await Assert.That(byName["backward_messages"][0].X).IsEqualTo((uint)((Width * 2 + 63) / 64));
		await Assert.That(byName["backward_messages"][1].X).IsEqualTo((uint)((Height * 2 + 63) / 64));
		await Assert.That(byName["filter_pixels"][0].X).IsEqualTo((uint)((p + 63) / 64));
	}

	[Test]
	public async Task Dispatch_FoldsIntoTwoDimensionsPastTheLimit()
	{
		// A per-dimension limit of 6 workgroups: 494 pixels x 2 sources need 16 workgroups, 6 x 3.
		// (The widest copy, 3 planes of 494, needs 24 workgroups, which 6 x 6 still holds.)
		var (gpu, _, _) = Setup("photometric");
		var device = new RecordingComputeDevice { Limits = ComputeDeviceLimits.Defaults with { MaxComputeWorkgroupsPerDimension = 6 } };
		await gpu.RunAsync(device);
		DispatchCommand initialCost = device.CommandsOf<DispatchCommand>().Single(d => d.Kernel.Descriptor.EntryPoint == "initial_cost");
		await Assert.That((initialCost.X, initialCost.Y)).IsEqualTo((6u, 3u));
		await Assert.That(initialCost.Kernel.Descriptor.Source).Contains("PM_GROUPS_X");
	}

	[Test]
	public async Task Progress_EqualsCpuFractions()
	{
		var (gpu, options, problem) = Setup("photometric", numIterations: 1);
		var cpuFractions = new List<double>();
		new PatchMatchCpu(options, problem).Run(progress: new SyncProgress(cpuFractions));

		var gpuFractions = new List<double>();
		var device = new RecordingComputeDevice();
		await gpu.RunAsync(device, progress: new SyncProgress(gpuFractions));
		await Assert.That(gpuFractions).IsEquivalentTo(cpuFractions, TUnit.Assertions.Enums.CollectionOrdering.Matching);
	}

	[Test]
	public async Task Cancellation_BetweenSweepsLeavesNothingPending()
	{
		var (gpu, _, _) = Setup("filter");
		var device = new RecordingComputeDevice();
		using var cts = new CancellationTokenSource();
		var fractions = new List<double>();
		var progress = new SyncProgress(fractions, onReport: () =>
		{
			if (fractions.Count == 3)
			{
				cts.Cancel();
			}
		});

		await Assert.That(async () => await gpu.RunAsync(device, cts.Token, progress)).Throws<OperationCanceledException>();
		await Assert.That(device.CommandsOf<FlushCommand>().Count).IsEqualTo(3);
		await Assert.That(device.PendingDispatchCount).IsEqualTo(0);
		await Assert.That(device.CommandsOf<ReadBufferCommand>().Count).IsEqualTo(0);
		await Assert.That(() => gpu.GetDepthMap()).Throws<InvalidOperationException>();

		// Every handle the run created was released.
		await Assert.That(device.CommandsOf<CreateBufferCommand>().All(c => c.Buffer.IsDisposed)).IsTrue();
	}

	[Test]
	public async Task Run_AwaitsOnADeviceThatCannotBlock()
	{
		var (gpu, options, _) = Setup("photometric", numIterations: 1);
		var device = new RecordingComputeDevice { SupportsBlockingWait = false };
		await gpu.RunAsync(device);
		await Assert.That(device.CommandsOf<FlushCommand>().Count).IsEqualTo(4 * options.NumIterations);
	}

	[Test]
	public async Task Run_ThrowsWithTheReasonWhenTheDeviceCannotRunIt()
	{
		var (gpu, _, _) = Setup("photometric");
		var device = new RecordingComputeDevice { Limits = ComputeDeviceLimits.Defaults with { MaxStorageBuffersPerShaderStage = 4 } };
		await Assert.That(async () => await gpu.RunAsync(device)).Throws<InvalidOperationException>().WithMessageContaining("Using the CPU.");
		await Assert.That(device.Commands.Count).IsEqualTo(0);
	}

	[Test]
	public async Task Uniforms_ProblemAndSweepFields()
	{
		var (gpu, options, _) = Setup("geometric");
		var device = new RecordingComputeDevice();
		await gpu.RunAsync(device);

		float[] problem = Floats(Created(device, "pm_problem").GetContents());
		uint[] problemWords = Words(Created(device, "pm_problem").GetContents());
		var likelihood = new PatchMatchLikelihood(
			(float)options.NccSigma, (float)(options.MinTriangulationAngle * 0.0174532925199432), (float)options.IncidentAngleSigma);
		await Assert.That(problem[0]).IsEqualTo(MathF.Cos((float)(options.MinTriangulationAngle * 0.0174532925199432)));
		await Assert.That(problem[1]).IsEqualTo(-0.5f / ((float)options.IncidentAngleSigma * (float)options.IncidentAngleSigma));
		await Assert.That(problem[2]).IsEqualTo(-0.5f / ((float)options.NccSigma * (float)options.NccSigma));
		await Assert.That(problem[3]).IsEqualTo(PatchMatchLikelihood.ComputeNCCCostNormFactor((float)options.NccSigma));
		await Assert.That((problemWords[4], problemWords[5])).IsEqualTo((0u, 0u));
		await Assert.That((problem[6], problem[7])).IsEqualTo(((float)options.DepthMin, (float)options.DepthMax));
		await Assert.That(problem[8]).IsEqualTo((float)options.GeomConsistencyRegularizer);
		await Assert.That(problem[9]).IsEqualTo((float)options.GeomConsistencyMaxCost);
		await Assert.That(problem[10]).IsEqualTo(likelihood.ComputeNCCProb(1.0f - (float)options.FilterMinNcc));
		await Assert.That(problem[11]).IsEqualTo(MathF.Cos((float)(options.FilterMinTriangulationAngle * 0.0174532925199432)));
		await Assert.That(problem[12]).IsEqualTo((float)options.FilterGeomConsistencyMaxCost);
		await Assert.That((int)problemWords[13]).IsEqualTo(options.FilterMinNumConsistent);
		float sigmaSpatial = (float)options.SigmaSpatial;
		float sigmaColor = (float)options.SigmaColor;
		await Assert.That(problem[14]).IsEqualTo(1.0f / (2.0f * sigmaSpatial * sigmaSpatial));
		await Assert.That(problem[15]).IsEqualTo(1.0f / (2.0f * sigmaColor * sigmaColor));

		float total = options.NumIterations * 4;
		for (int k = 0; k < 4 * options.NumIterations; ++k)
		{
			int iter = k / 4;
			int sweep = k % 4;
			byte[] bytes = Created(device, $"pm_sweep_{k}").GetContents();
			float[] f = Floats(bytes);
			uint[] w = Words(bytes);
			float perturbation = 1.0f / MathF.Pow(2.0f, iter + sweep / 4.0f);
			bool last = k == 4 * options.NumIterations - 1;
			await Assert.That((int)w[0]).IsEqualTo(sweep);
			await Assert.That((int)w[1]).IsEqualTo(PatchMatchRandom.SweepPhase(iter, sweep));
			await Assert.That(f[2]).IsEqualTo(perturbation);
			await Assert.That(f[3]).IsEqualTo((float)(perturbation * Math.PI));
			await Assert.That(f[4]).IsEqualTo(k / total);
			await Assert.That((w[5], w[6])).IsEqualTo(last ? (1u, 1u) : (0u, 0u));
		}
	}

	[Test]
	public async Task Uniforms_PoseTableByteTableAndRotations()
	{
		var (gpu, _, problem) = Setup("photometric");
		var device = new RecordingComputeDevice();
		await gpu.RunAsync(device);

		var transforms = new PatchMatchTransforms(problem.Images!, problem.RefImageIdx, problem.SrcImageIdxs);
		float[] table = Floats(Created(device, "pm_poses").GetContents());
		int s = problem.SrcImageIdxs.Count;
		await Assert.That(table.Length).IsEqualTo(4 * (8 + 43 * s));
		for (int r = 0; r < 4; ++r)
		{
			await Assert.That(table.AsSpan(8 * r, 4).SequenceEqual(transforms.RefK(r))).IsTrue();
			await Assert.That(table.AsSpan(8 * r + 4, 4).SequenceEqual(transforms.RefInvK(r))).IsTrue();
			for (int src = 0; src < s; ++src)
			{
				ReadOnlySpan<float> row = transforms.Poses(r).AsSpan(43 * src, 43);
				await Assert.That(table.AsSpan(32 + 43 * (r * s + src), 43).SequenceEqual(row)).IsTrue();
			}
		}

		float[] byteTable = Floats(Created(device, "pm_byte_to_unit").GetContents());
		await Assert.That(byteTable.Length).IsEqualTo(256);
		for (int i = 0; i < 256; ++i)
		{
			await Assert.That(byteTable[i]).IsEqualTo(i / 255.0f);
		}

		// Rotation copies of an even sweep read the Width x Height frame; odd ones Height x Width.
		uint[] normals0 = Words(Created(device, "pm_rotate_normals_0").GetContents());
		await Assert.That(normals0[..5]).IsEquivalentTo(new uint[] { Width, Height, 3, Width * Height, Width * Height }, TUnit.Assertions.Enums.CollectionOrdering.Matching);
		uint[] maps1 = Words(Created(device, "pm_rotate_sel_1").GetContents());
		await Assert.That(maps1[..5]).IsEquivalentTo(new uint[] { Height, Width, (uint)s, 0, 0 }, TUnit.Assertions.Enums.CollectionOrdering.Matching);
	}

	[Test]
	public async Task Buffers_InitialContentsMatchCpuInputs()
	{
		var (gpu, _, problem) = Setup("geometric");
		var device = new RecordingComputeDevice();
		await gpu.RunAsync(device);
		int p = Width * Height;

		float[] state = Floats(Created(device, "pm_state_0").GetContents());
		await Assert.That(state.AsSpan(0, p).SequenceEqual(problem.DepthMaps![0].Data)).IsTrue();
		await Assert.That(state.AsSpan(p, 3 * p).SequenceEqual(problem.NormalMaps![0].Data)).IsTrue();
		await Assert.That(Floats(Created(device, "pm_source_maps_c").GetContents()).All(v => v == 0.5f)).IsTrue();

		var reference = new PatchMatchRefImage(Width, Height);
		reference.Filter(problem.Images![0].GetBitmap().RowMajorData, 3, 1, 3, (float)new PatchMatchOptions().SigmaColor);
		float[] planes = Floats(Created(device, "pm_reference_0").GetContents());
		await Assert.That(planes.AsSpan(0, p).SequenceEqual(reference.Image.Data.Select(b => b / 255.0f).ToArray())).IsTrue();
		await Assert.That(planes.AsSpan(p, p).SequenceEqual(reference.SumImage.Data)).IsTrue();
		await Assert.That(planes.AsSpan(2 * p, p).SequenceEqual(reference.SquaredSumImage.Data)).IsTrue();

		// Source layers: source i's row r at (i H + r) W; depth layers likewise.
		byte[] images = Created(device, "pm_source_images").GetContents();
		float[] depths = Floats(Created(device, "pm_source_depths").GetContents());
		for (int i = 0; i < 2; ++i)
		{
			int imageIdx = problem.SrcImageIdxs[i];
			await Assert.That(images.AsSpan(i * p, p).SequenceEqual(problem.Images[imageIdx].GetBitmap().RowMajorData)).IsTrue();
			await Assert.That(depths.AsSpan(i * p, p).SequenceEqual(problem.DepthMaps[imageIdx].Data)).IsTrue();
		}
	}

	[Test]
	public async Task Readback_DecodesIntoCpuResultTypes()
	{
		// The recording device runs no kernel, so what comes back is the buffers' initial
		// contents: the geometric run's input maps, 0.5 selection probabilities, a zero mask.
		var (gpu, _, problem) = Setup("geometric");
		var device = new RecordingComputeDevice();
		await gpu.RunAsync(device);

		DepthMap depth = gpu.GetDepthMap();
		await Assert.That(depth.Data.SequenceEqual(problem.DepthMaps![0].Data)).IsTrue();
		await Assert.That((depth.GetWidth(), depth.GetHeight())).IsEqualTo((Width, Height));
		NormalMap normal = gpu.GetNormalMap();
		await Assert.That(normal.Data.SequenceEqual(problem.NormalMaps![0].Data)).IsTrue();
		Mat<float> sel = gpu.GetSelProbMap();
		await Assert.That((sel.GetWidth(), sel.GetHeight(), sel.GetDepth())).IsEqualTo((Width, Height, 2));
		await Assert.That(sel.Data.All(v => v == 0.5f)).IsTrue();
		await Assert.That(gpu.GetConsistentImageIdxs().Count).IsEqualTo(0);
	}

	private static float[] Floats(byte[] bytes) => MemoryMarshal.Cast<byte, float>(bytes).ToArray();

	private static uint[] Words(byte[] bytes) => MemoryMarshal.Cast<byte, uint>(bytes).ToArray();

	// Progress<T> posts to a synchronization context; this records on the reporting thread.
	private sealed class SyncProgress(List<double> fractions, Action? onReport = null) : IProgress<double>
	{
		public void Report(double value)
		{
			fractions.Add(value);
			onReport?.Invoke();
		}
	}
}
