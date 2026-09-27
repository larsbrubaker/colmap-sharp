// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// The probe stages of PatchMatchGpuConformance (PatchMatchGpuConformance.cs): small dispatches
// that localize a disagreement before the full runs. Not a COLMAP port.
// - The random-number probe runs the shipped patch_match_common.wgsl (Mvs/Shaders/) with a
//   tiny entry point that draws from pm_random_init / pm_random_next_uniform, and compares the
//   bits with PatchMatchRandom on the CPU (Tier A: must be identical).
// - The conversion probe feeds chosen u32s (rounding ties at several exponents, extremes, and
//   random values) to u32_to_f32_rne and to the uniform mapping pm_random_next_uniform applies
//   to its draw bits, against C#'s (float)uint and PatchMatchRandom.NextUniform's mapping.
// - The kernel probe runs init_random then initial_cost, wired as PatchMatchGpu.Run wires the
//   first sweep, on the device and on the CPU twin (ReferenceComputeDevice) with the device's
//   limits, so both get the same kernel text, buffers and dispatch shapes; only the arithmetic
//   differs. A real GPU's sqrt, exp and division round differently, hence the bounds.

using System.Runtime.InteropServices;

using ColmapSharp.Compute;

namespace ColmapSharp.Mvs.Testing;

public static partial class PatchMatchGpuConformance
{
	private const int DrawsPerKey = 8;

	// Five u32s per key: seed low word, seed high word, row, col, phase (the last three as i32).
	private const int WordsPerKey = 5;

	private const int KernelProbeWidth = 48;
	private const int KernelProbeHeight = 36;

	private const string RandomProbeKernel = @"
@group(0) @binding(0) var<storage, read> keys : array<u32>;
@group(0) @binding(1) var<storage, read_write> draws : array<u32>;

const DRAWS_PER_KEY : u32 = 8u;

@compute @workgroup_size(64)
fn pm_conformance_random(@builtin(global_invocation_id) id : vec3<u32>)
{
	let k = id.x;
	if (k >= arrayLength(&keys) / 5u)
	{
		return;
	}

	let seed = vec2<u32>(keys[k * 5u], keys[k * 5u + 1u]);
	let row = bitcast<i32>(keys[k * 5u + 2u]);
	let col = bitcast<i32>(keys[k * 5u + 3u]);
	let phase = bitcast<i32>(keys[k * 5u + 4u]);
	var state = pm_random_init(seed, row, col, phase);
	for (var d = 0u; d < DRAWS_PER_KEY; d = d + 1u)
	{
		draws[k * DRAWS_PER_KEY + d] = bitcast<u32>(pm_random_next_uniform(&state));
	}
}
";

	// For each input u32: u32_to_f32_rne(v), then the uniform mapping pm_random_next_uniform
	// applies to its draw bits (copied from patch_match_common.wgsl: that function takes a
	// stream, not bits, so the probe cannot feed it chosen values through the function itself).
	private const string ConversionProbeKernel = @"
@group(0) @binding(0) var<storage, read> values : array<u32>;
@group(0) @binding(1) var<storage, read_write> results : array<u32>;

@compute @workgroup_size(64)
fn pm_conformance_conversion(@builtin(global_invocation_id) id : vec3<u32>)
{
	let i = id.x;
	if (i >= arrayLength(&values))
	{
		return;
	}

	let bits = values[i];
	results[i * 2u] = bitcast<u32>(u32_to_f32_rne(bits));
	results[i * 2u + 1u] = bitcast<u32>(u32_to_f32_rne(bits) * bitcast<f32>(0x2F800000u) + bitcast<f32>(0x2F000000u));
}
";

	private static async Task RandomProbeAsync(IComputeDevice device, bool isTwin, List<PatchMatchGpuConformanceCheck> checks, CancellationToken ct)
	{
		const string Name = "probe.random.differing_draws";
		if (isTwin)
		{
			checks.Add(NotApplicable(Name, TwinProbeReason));
			return;
		}

		var keys = new List<(ulong Seed, int Row, int Col, int Phase)>();
		foreach (ulong seed in new ulong[] { PatchMatchRandom.DefaultSeed, 1, 0x123456789ABCDEF0UL, ulong.MaxValue })
		{
			foreach (int row in new[] { 0, 1, 17, 2047, int.MaxValue })
			{
				foreach (int col in new[] { 0, 3, 640, 4095, int.MaxValue })
				{
					foreach (int phase in new[]
					{
						PatchMatchRandom.InitDepthPhase, PatchMatchRandom.InitNormalPhase, PatchMatchRandom.SweepPhase(0, 0),
						PatchMatchRandom.SweepPhase(0, 3), PatchMatchRandom.SweepPhase(4, 2), PatchMatchRandom.SweepPhase(99, 3),
					})
					{
						keys.Add((seed, row, col, phase));
					}
				}
			}
		}

		var keyWords = new uint[keys.Count * WordsPerKey];
		for (int k = 0; k < keys.Count; k++)
		{
			keyWords[k * WordsPerKey] = (uint)keys[k].Seed;
			keyWords[k * WordsPerKey + 1] = (uint)(keys[k].Seed >> 32);
			keyWords[k * WordsPerKey + 2] = unchecked((uint)keys[k].Row);
			keyWords[k * WordsPerKey + 3] = unchecked((uint)keys[k].Col);
			keyWords[k * WordsPerKey + 4] = unchecked((uint)keys[k].Phase);
		}

		uint[] gpu;
		try
		{
			gpu = await RunProbeAsync(device, "pm_conformance_random", RandomProbeKernel, keyWords, keys.Count * DrawsPerKey, keys.Count, ct).ConfigureAwait(false);
		}
		catch (Exception e) when (e is not OperationCanceledException)
		{
			checks.Add(StageFailed(Name, e));
			return;
		}

		int mismatches = 0;
		string detail = "";
		for (int k = 0; k < keys.Count; k++)
		{
			var random = new PatchMatchRandom(keys[k].Seed, keys[k].Row, keys[k].Col, keys[k].Phase);
			for (int d = 0; d < DrawsPerKey; d++)
			{
				uint expected = BitConverter.SingleToUInt32Bits(random.NextUniform());
				uint actual = gpu[k * DrawsPerKey + d];
				if (actual != expected)
				{
					if (mismatches++ == 0)
					{
						detail = $"first: key {keys[k]} draw {d}: device 0x{actual:X8}, CPU 0x{expected:X8}";
					}
				}
			}
		}

		checks.Add(Compare(Name, mismatches, "==", 0, $"{keys.Count * DrawsPerKey} draws compared" + (detail.Length > 0 ? "; " + detail : "")));
	}

	private static async Task ConversionProbeAsync(IComputeDevice device, bool isTwin, List<PatchMatchGpuConformanceCheck> checks, CancellationToken ct)
	{
		const string Name = "probe.conversion.differing_values";
		if (isTwin)
		{
			checks.Add(NotApplicable(Name, TwinProbeReason));
			return;
		}

		var values = new List<uint> { 0, 1, 2, (1u << 24) - 1, 1u << 24, (1u << 24) + 1, (1u << 24) + 2, (1u << 24) + 3 };

		// Exact ties and their neighbours at several exponents. With the top bit at msb, the low
		// shift = msb - 23 bits are rounded away; a tie rounds down to an even significand and up
		// from an odd one.
		foreach (int shift in new[] { 1, 2, 5, 8 })
		{
			uint half = 1u << (shift - 1);
			foreach (uint significand in new uint[] { 0x800000, 0x800001, 0xABCDEE, 0xABCDEF, 0xFFFFFE, 0xFFFFFF })
			{
				uint tie = (significand << shift) | half;
				values.Add(tie);
				values.Add(tie - 1);
				values.Add(tie + 1);
			}
		}

		values.AddRange(new uint[] { 0x7FFFFFFF, 0x80000000, 0xFFFFFF7F, 0xFFFFFF80, 0xFFFFFFFF });

		// A fixed seed, so every host probes the same values.
		var random = new Random(12345);
		for (int i = 0; i < 10000; i++)
		{
			values.Add((uint)random.NextInt64(0, 1L << 32));
		}

		uint[] input = values.ToArray();
		uint[] gpu;
		try
		{
			gpu = await RunProbeAsync(device, "pm_conformance_conversion", ConversionProbeKernel, input, input.Length * 2, input.Length, ct).ConfigureAwait(false);
		}
		catch (Exception e) when (e is not OperationCanceledException)
		{
			checks.Add(StageFailed(Name, e));
			return;
		}

		// PatchMatchRandom.NextUniform's mapping, bits * 2^-32 + 2^-33; 2.3283064e-10f is exactly 2^-32.
		const float TwoPow32Inv = 2.3283064e-10f;
		int mismatches = 0;
		string detail = "";
		for (int i = 0; i < input.Length; i++)
		{
			uint bits = input[i];
			uint converted = BitConverter.SingleToUInt32Bits((float)bits);
			uint mapped = BitConverter.SingleToUInt32Bits(bits * TwoPow32Inv + TwoPow32Inv / 2.0f);
			if (gpu[i * 2] != converted || gpu[i * 2 + 1] != mapped)
			{
				if (mismatches++ == 0)
				{
					detail = $"first: 0x{bits:X8}: device (0x{gpu[i * 2]:X8}, 0x{gpu[i * 2 + 1]:X8}), C# (0x{converted:X8}, 0x{mapped:X8})";
				}
			}
		}

		checks.Add(Compare(Name, mismatches, "==", 0, $"{input.Length} values compared" + (detail.Length > 0 ? "; " + detail : "")));
	}

	/// <summary>
	/// Compiles patch_match_common.wgsl + <paramref name="kernel"/>, runs it over
	/// <paramref name="input"/> (binding 0) into <paramref name="outputWords"/> u32s (binding 1),
	/// one invocation per item, and reads the output back.
	/// </summary>
	private static async Task<uint[]> RunProbeAsync(IComputeDevice device, string entryPoint, string kernel, uint[] input, int outputWords, int items, CancellationToken ct)
	{
		var descriptor = new ComputeKernelDescriptor(
			entryPoint,
			PatchMatchShaders.Load(PatchMatchShaders.Common) + kernel,
			entryPoint,
			[new ComputeKernelBinding(0, 0, ComputeBindingType.ReadOnlyStorage), new ComputeKernelBinding(0, 1, ComputeBindingType.Storage)]);
		var output = new byte[outputWords * 4L];
		using IComputeKernel compiled = device.CreateKernel(descriptor);
		using IComputeBuffer inputBuffer = device.CreateBuffer(ComputeBufferKind.Storage, input.Length * 4L, MemoryMarshal.AsBytes(input.AsSpan()), entryPoint + "_input");
		using IComputeBuffer outputBuffer = device.CreateBuffer(ComputeBufferKind.Storage, output.Length, label: entryPoint + "_output");
		using IComputeBindGroup group = device.CreateBindGroup(compiled, 0,
		[
			new ComputeBufferBinding(0, inputBuffer, 0, inputBuffer.Size),
			new ComputeBufferBinding(1, outputBuffer, 0, outputBuffer.Size),
		]);
		device.Dispatch(compiled, [group], (uint)((items + PatchMatchGpuKernels.WorkgroupSize - 1) / PatchMatchGpuKernels.WorkgroupSize));
		await device.ReadBufferAsync(outputBuffer, 0, output, ct).ConfigureAwait(false);
		return MemoryMarshal.Cast<byte, uint>(output).ToArray();
	}

	private static async Task KernelProbeAsync(IComputeDevice device, List<PatchMatchGpuConformanceCheck> checks, CancellationToken ct)
	{
		const string DepthName = "kernel.init_random.depth_max_ulps";
		const string CostName = "kernel.initial_cost.agreement";
		float[] deviceState, deviceCosts, twinState, twinCosts;
		try
		{
			(deviceState, deviceCosts) = await RunInitAsync(device, ct).ConfigureAwait(false);

			// The twin gets the device's limits so the plan, and so the kernel text and dispatch shapes, match.
			(twinState, twinCosts) = await RunInitAsync(new ReferenceComputeDevice(device.Limits), ct).ConfigureAwait(false);
		}
		catch (Exception e) when (e is not OperationCanceledException)
		{
			checks.Add(StageFailed(DepthName, e));
			checks.Add(StageFailed(CostName, e));
			return;
		}

		int planeSize = KernelProbeWidth * KernelProbeHeight;
		long maxDepthUlps = 0;
		for (int i = 0; i < planeSize; ++i)
		{
			maxDepthUlps = Math.Max(maxDepthUlps, Ulps(deviceState[i], twinState[i]));
		}

		long maxNormalUlps = 0;
		for (int i = planeSize; i < 4 * planeSize; ++i)
		{
			maxNormalUlps = Math.Max(maxNormalUlps, Ulps(deviceState[i], twinState[i]));
		}

		int costAgree = 0;
		float maxCostDifference = 0;
		for (int i = 0; i < deviceCosts.Length; ++i)
		{
			float difference = MathF.Abs(deviceCosts[i] - twinCosts[i]);
			maxCostDifference = MathF.Max(maxCostDifference, difference);
			costAgree += difference <= MaxInitialCostDifference ? 1 : 0;
		}

		checks.Add(Compare(DepthName, maxDepthUlps, "<=", MaxInitDepthUlps, $"max normal difference {maxNormalUlps} ULP (not bounded)"));
		checks.Add(Compare(CostName, costAgree / (double)deviceCosts.Length, ">=", MinInitialCostAgreement,
			$"fraction of {deviceCosts.Length} costs within {MaxInitialCostDifference}; max difference {maxCostDifference:E3}"));
	}

	/// <summary>
	/// Runs init_random then initial_cost on <paramref name="device"/> over a photometric problem
	/// of the synthetic scene, wired as PatchMatchGpu.Run wires the first sweep, and reads back
	/// the state buffer's depth and normal planes and the cost map.
	/// </summary>
	private static async Task<(float[] State, float[] Costs)> RunInitAsync(IComputeDevice device, CancellationToken ct)
	{
		var (images, _, _) = PatchMatchSyntheticScene.Scene(KernelProbeWidth, KernelProbeHeight);
		PatchMatchOptions options = PatchMatchSyntheticScene.Options(1, geomConsistency: false, filter: false);
		var pm = new PatchMatchGpu(options, PatchMatchSyntheticScene.Problem(images), Seed);
		if (!PatchMatchGpuPlan.TryCreate(pm.ProblemShape, options, device.Limits, null, out PatchMatchGpuPlan? plan, out string? reason))
		{
			throw new InvalidOperationException("The GPU cannot run the probe problem: " + PatchMatchGpuPlan.ReasonWithoutFallback(reason!));
		}

		var owned = new List<IDisposable>();
		T Own<T>(T handle)
			where T : IDisposable
		{
			owned.Add(handle);
			return handle;
		}

		long Size(PatchMatchGpuBufferRole role) => plan!.Buffers.First(b => b.Role == role).Size;
		try
		{
			long imagesSize = Size(PatchMatchGpuBufferRole.SourceImages);
			var buffers = new Dictionary<int, IComputeBuffer>
			{
				[PatchMatchGpuKernels.ByteToUnitBinding] = Own(device.CreateBuffer(ComputeBufferKind.Uniform, Size(PatchMatchGpuBufferRole.ByteTable), PatchMatchGpu.PackByteTable(), "byte_table")),
				[PatchMatchGpuKernels.PosesBinding] = Own(device.CreateBuffer(ComputeBufferKind.Uniform, Size(PatchMatchGpuBufferRole.PoseTable), pm.PackPoseTable(), "poses")),
				[PatchMatchGpuKernels.ProblemBinding] = Own(device.CreateBuffer(ComputeBufferKind.Uniform, Size(PatchMatchGpuBufferRole.Problem), pm.PackProblem(), "problem")),
				[PatchMatchGpuKernels.ReferenceBinding] = Own(device.CreateBuffer(ComputeBufferKind.Storage, Size(PatchMatchGpuBufferRole.ReferencePlanes), pm.PackReference(), "reference")),
				[PatchMatchGpuKernels.SourceImagesBinding] = Own(device.CreateBuffer(ComputeBufferKind.Storage, imagesSize, pm.PackSourceImages(imagesSize), "source_images")),
				[PatchMatchGpuKernels.SourceDepthsBinding] = Own(device.CreateBuffer(ComputeBufferKind.Storage, Size(PatchMatchGpuBufferRole.SourceDepths), pm.PackSourceDepths(), "source_depths")),
				[PatchMatchGpuKernels.StateBinding] = Own(device.CreateBuffer(ComputeBufferKind.Storage, Size(PatchMatchGpuBufferRole.State), default, "state")),
				[PatchMatchGpuKernels.CostsBinding] = Own(device.CreateBuffer(ComputeBufferKind.Storage, Size(PatchMatchGpuBufferRole.SourceMaps), default, "costs")),
			};

			int planeSize = KernelProbeWidth * KernelProbeHeight;
			Dispatch(PatchMatchGpuKernel.InitRandom, planeSize);
			Dispatch(PatchMatchGpuKernel.InitialCost, (long)planeSize * pm.ProblemShape.NumSources);

			var state = new byte[4L * 4 * planeSize];
			await device.ReadBufferAsync(buffers[PatchMatchGpuKernels.StateBinding], 0, state, ct).ConfigureAwait(false);
			var costs = new byte[4L * pm.ProblemShape.NumSources * planeSize];
			await device.ReadBufferAsync(buffers[PatchMatchGpuKernels.CostsBinding], 0, costs, ct).ConfigureAwait(false);
			return (MemoryMarshal.Cast<byte, float>(state).ToArray(), MemoryMarshal.Cast<byte, float>(costs).ToArray());

			// As PatchMatchGpu.Run dispatches: the plan's workgroups along x, the rest along y.
			void Dispatch(PatchMatchGpuKernel kernel, long elements)
			{
				string entryPoint = PatchMatchGpuKernels.EntryPoint(kernel);
				uint maxX = plan!.Dispatches.Single(d => d.Kernel == entryPoint).X;
				IComputeKernel compiled = Own(device.CreateKernel(PatchMatchGpuKernels.Descriptor(kernel, pm.ShaderShape, maxX)));
				ComputeBufferBinding[] entries = PatchMatchGpuKernels.Bindings(kernel)
					.Where(b => b.Group == 0)
					.Select(b => new ComputeBufferBinding(b.Binding, buffers[b.Binding], 0, buffers[b.Binding].Size))
					.ToArray();
				IComputeBindGroup group = Own(device.CreateBindGroup(compiled, 0, entries));
				long workgroups = (elements + PatchMatchGpuKernels.WorkgroupSize - 1) / PatchMatchGpuKernels.WorkgroupSize;
				long x = Math.Min(workgroups, maxX);
				device.Dispatch(compiled, [group], (uint)x, (uint)((workgroups + x - 1) / x));
			}
		}
		finally
		{
			for (int i = owned.Count - 1; i >= 0; --i)
			{
				owned[i].Dispose();
			}
		}
	}

	// Units in the last place between two floats of the same sign; int.MaxValue across a sign
	// change or when one of them is NaN.
	private static long Ulps(float a, float b)
	{
		if (float.IsNaN(a) || float.IsNaN(b))
		{
			return float.IsNaN(a) && float.IsNaN(b) ? 0 : int.MaxValue;
		}

		if (a == b)
		{
			return 0;
		}

		if ((a < 0) != (b < 0))
		{
			return int.MaxValue;
		}

		return Math.Min(Math.Abs((long)BitConverter.SingleToInt32Bits(a) - BitConverter.SingleToInt32Bits(b)), int.MaxValue);
	}
}
