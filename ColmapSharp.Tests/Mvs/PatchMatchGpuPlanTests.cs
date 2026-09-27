// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchGpuPlanTests: C#-only tests (COLMAP has no GPU planner; its PatchMatch is CUDA)
// of ColmapSharp/Mvs/PatchMatchGpuPlan.cs: the buffer sizes of the GPU PatchMatch memory
// table (the problem sizes of the LOW, MEDIUM and HIGH dense quality presets), which of them
// fit the WebGPU default limits and which fit an Apple M5's raised limits, the text of every
// fallback reason, the band sizes, and determinism. Expected sizes are the layout formula
// worked by hand (P pixels, S sources, L = P source layer pixels, C = max(W, H)):
//   2 x 12P + 2 x (16P + 4C(S + 4)) + 3 x 4SP + SL + (geometric ? 4SL : 4)
//   + 1024 + 704S + 256 + 20 x 64 + 16 x bands.

using ColmapSharp.Compute;
using ColmapSharp.Mvs;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchGpuPlanTests
{
	// An Apple M5 through wgpu with every limit raised to the adapter's: only these differ
	// from the defaults.
	private static readonly ComputeDeviceLimits M5Limits = ComputeDeviceLimits.Defaults with
	{
		MaxBufferSize = 14_302_248_960,
		MaxStorageBufferBindingSize = 4_294_967_292,
		MaxStorageBuffersPerShaderStage = 31,
		MaxComputeInvocationsPerWorkgroup = 1024,
	};

	private static PatchMatchGpuProblemShape Shape(int width, int height, int numSources)
		=> new(width, height, numSources, width, height);

	private static PatchMatchOptions Options(bool geometric) => new() { GeomConsistency = geometric, Filter = true };

	private static (bool Fits, PatchMatchGpuPlan? Plan, string? Reason) Plan(
		PatchMatchGpuProblemShape shape, bool geometric, ComputeDeviceLimits limits, long? budget = null)
	{
		bool fits = PatchMatchGpuPlan.TryCreate(shape, Options(geometric), limits, budget, out PatchMatchGpuPlan? plan, out string? reason);
		return (fits, plan, reason);
	}

	[Test]
	[Arguments(1000, 750, 4, false, 81_072_340L, 12_032_000L, 8, 11)]
	[Arguments(1000, 750, 4, true, 93_072_336L, 12_032_000L, 8, 11)]
	[Arguments(1000, 750, 20, false, 237_212_436L, 60_000_000L, 6, 9)]
	[Arguments(1000, 750, 20, true, 297_212_432L, 60_000_000L, 6, 9)]
	public async Task LowPresetFitsDefaultLimits(
		int width, int height, int numSources, bool geometric, long totalBytes, long largestBinding, int rows0, int rows1)
	{
		var (fits, plan, reason) = Plan(Shape(width, height, numSources), geometric, ComputeDeviceLimits.Defaults);
		await Assert.That(fits).IsTrue();
		await Assert.That(reason).IsNull();
		await Assert.That(plan!.TotalBytes).IsEqualTo(totalBytes);
		await Assert.That(plan.LargestStorageBinding).IsEqualTo(largestBinding);
		await Assert.That(plan.BandRows(0)).IsEqualTo(rows0);
		await Assert.That(plan.BandRows(1)).IsEqualTo(rows1);
	}

	[Test]
	[Arguments(1000, 750, 4, false, 81_072_340L)]
	[Arguments(1000, 750, 20, true, 297_212_432L)]
	[Arguments(1600, 1200, 20, false, 607_053_764L)]
	[Arguments(1600, 1200, 20, true, 760_653_760L)]
	[Arguments(2400, 1800, 20, false, 1_365_624_644L)]
	[Arguments(2400, 1800, 20, true, 1_711_224_640L)]
	public async Task EveryPresetFitsM5Limits(int width, int height, int numSources, bool geometric, long totalBytes)
	{
		var (fits, plan, reason) = Plan(Shape(width, height, numSources), geometric, M5Limits);
		await Assert.That(reason).IsNull();
		await Assert.That(fits).IsTrue();
		await Assert.That(plan!.TotalBytes).IsEqualTo(totalBytes);
	}

	[Test]
	public async Task LowPresetBufferList()
	{
		var (_, plan, _) = Plan(Shape(1000, 750, 4), false, ComputeDeviceLimits.Defaults);
		(PatchMatchGpuBufferRole, ComputeBufferKind, long, int)[] expected =
		[
			(PatchMatchGpuBufferRole.ReferencePlanes, ComputeBufferKind.Storage, 9_000_000, 2),
			(PatchMatchGpuBufferRole.State, ComputeBufferKind.Storage, 12_032_000, 2),
			(PatchMatchGpuBufferRole.SourceMaps, ComputeBufferKind.Storage, 12_000_000, 3),
			(PatchMatchGpuBufferRole.SourceImages, ComputeBufferKind.Storage, 3_000_000, 1),
			(PatchMatchGpuBufferRole.SourceDepths, ComputeBufferKind.Storage, 4, 1),
			(PatchMatchGpuBufferRole.ByteTable, ComputeBufferKind.Uniform, 1024, 1),
			(PatchMatchGpuBufferRole.PoseTable, ComputeBufferKind.Uniform, 2816, 1),
			(PatchMatchGpuBufferRole.Problem, ComputeBufferKind.Uniform, 256, 1),
			(PatchMatchGpuBufferRole.SweepUniforms, ComputeBufferKind.Uniform, 64, 20),
			(PatchMatchGpuBufferRole.BandUniforms, ComputeBufferKind.Uniform, 16, 94 + 91),
		];
		var actual = plan!.Buffers.Select(b => (b.Role, b.Kind, b.Size, b.Count)).ToArray();
		await Assert.That(actual).IsEquivalentTo(expected);
		await Assert.That(plan.SweepCount).IsEqualTo(20);
		await Assert.That(plan.BandCount(0)).IsEqualTo(94);
		await Assert.That(plan.BandCount(1)).IsEqualTo(91);
		await Assert.That(plan.WindowCount).IsEqualTo(121);
		await Assert.That(plan.WindowSamplesPerPixel).IsEqualTo(121L * (4 * 15 + 4));
		foreach (PatchMatchGpuBuffer buffer in plan.Buffers)
		{
			await Assert.That(buffer.Size % (buffer.Kind == ComputeBufferKind.Uniform ? 16 : 4)).IsEqualTo(0L);
		}
	}

	[Test]
	public async Task GeometricRunBindsSourceDepths()
	{
		var (_, plan, _) = Plan(Shape(1000, 750, 4), true, ComputeDeviceLimits.Defaults);
		PatchMatchGpuBuffer depths = plan!.Buffers.Single(b => b.Role == PatchMatchGpuBufferRole.SourceDepths);
		await Assert.That(depths.Size).IsEqualTo(4L * 4 * 1000 * 750);
		// A geometric run starts from the given maps, so there is no random init.
		await Assert.That(plan.Dispatches.Any(d => d.Kernel == "init_random")).IsFalse();
	}

	[Test]
	public async Task MediumPresetExceedsDefaultBindingSize()
	{
		var (fits, plan, reason) = Plan(Shape(1600, 1200, 20), false, ComputeDeviceLimits.Defaults);
		await Assert.That(fits).IsFalse();
		await Assert.That(plan).IsNull();
		await Assert.That(reason).IsEqualTo(
			"The GPU can bind at most 128 MiB per buffer; this image's matching costs of all source images need 147 MiB. Using the CPU.");
	}

	[Test]
	public async Task HighPresetExceedsDefaultBufferSize()
	{
		var (fits, _, reason) = Plan(Shape(2400, 1800, 20), true, ComputeDeviceLimits.Defaults);
		await Assert.That(fits).IsFalse();
		await Assert.That(reason).IsEqualTo(
			"The GPU can hold at most 256 MiB in one buffer; this image's matching costs of all source images need 330 MiB. Using the CPU.");
	}

	[Test]
	public async Task MemoryBudgetIsEnforced()
	{
		var (fits, _, reason) = Plan(Shape(2400, 1800, 20), true, M5Limits, budget: 1L << 30);
		await Assert.That(fits).IsFalse();
		await Assert.That(reason).IsEqualTo(
			"This image needs 1,632 MiB of GPU memory, more than the 1,024 MiB available. Using the CPU.");

		// A budget of exactly the total fits.
		var (fitsExactly, _, _) = Plan(Shape(2400, 1800, 20), true, M5Limits, budget: 1_711_224_640);
		await Assert.That(fitsExactly).IsTrue();
	}

	[Test]
	public async Task DeviceCapabilityReasons()
	{
		PatchMatchGpuProblemShape shape = Shape(100, 80, 4);
		ComputeDeviceLimits d = ComputeDeviceLimits.Defaults;
		(ComputeDeviceLimits Limits, string Reason)[] cases =
		[
			(d with { MaxComputeInvocationsPerWorkgroup = 32 },
				"The GPU runs at most 32 threads per workgroup; PatchMatch needs 64. Using the CPU."),
			(d with { MaxBindGroups = 2 },
				"The GPU allows 2 bind groups per shader; PatchMatch needs 3. Using the CPU."),
			(d with { MaxStorageBuffersPerShaderStage = 6 },
				"The GPU allows 6 storage buffers per shader; PatchMatch needs 7. Using the CPU."),
			(d with { MaxUniformBuffersPerShaderStage = 4 },
				"The GPU allows 4 uniform buffers per shader; PatchMatch needs 5. Using the CPU."),
		];
		foreach (var (limits, expected) in cases)
		{
			var (fits, _, reason) = Plan(shape, false, limits);
			await Assert.That(fits).IsFalse();
			await Assert.That(reason).IsEqualTo(expected);
		}

		// Exactly the required counts fit.
		var (fitsAtMinimum, _, _) = Plan(shape, false, d with
		{
			MaxComputeInvocationsPerWorkgroup = 64,
			MaxBindGroups = 3,
			MaxStorageBuffersPerShaderStage = 7,
			MaxUniformBuffersPerShaderStage = 5,
		});
		await Assert.That(fitsAtMinimum).IsTrue();
	}

	[Test]
	public async Task PoseTableExceedsUniformBindingSize()
	{
		// 4 rotations x 100 sources x 176 bytes = 70,400 bytes, past 64 KiB.
		var (fits, _, reason) = Plan(Shape(100, 80, 100), false, ComputeDeviceLimits.Defaults);
		await Assert.That(fits).IsFalse();
		await Assert.That(reason).IsEqualTo(
			"The GPU can bind at most 64 KiB of uniform data per buffer; the camera poses of the source images need 69 KiB. Using the CPU.");

		// 93 sources (65,472 bytes) still fit.
		var (fits93, _, _) = Plan(Shape(100, 80, 93), false, ComputeDeviceLimits.Defaults);
		await Assert.That(fits93).IsTrue();
	}

	[Test]
	public async Task DispatchBeyondWorkgroupLimit()
	{
		// init_random: 750,000 pixels = 11,719 workgroups; 16 x 16 cannot hold them.
		var (fits, _, reason) = Plan(Shape(1000, 750, 4), false, ComputeDeviceLimits.Defaults with { MaxComputeWorkgroupsPerDimension = 16 });
		await Assert.That(fits).IsFalse();
		await Assert.That(reason).IsEqualTo(
			"The GPU can run at most 16 x 16 workgroups at once; this image's init_random step needs 11,719. Using the CPU.");
	}

	[Test]
	public async Task LargeDispatchesFoldIntoTwoDimensions()
	{
		var (_, plan, _) = Plan(Shape(2400, 1800, 20), true, M5Limits);
		PatchMatchGpuDispatch initialCost = plan!.Dispatches.Single(d => d.Kernel == "initial_cost");

		// 86,400,000 invocations = 1,350,000 workgroups = 65,535 x 21 (folded).
		await Assert.That(initialCost.Elements).IsEqualTo(86_400_000L);
		await Assert.That(initialCost.X).IsEqualTo(65535u);
		await Assert.That(initialCost.Y).IsEqualTo(21u);
		foreach (PatchMatchGpuDispatch d in plan.Dispatches)
		{
			await Assert.That((long)d.X * d.Y * PatchMatchGpuPlan.WorkgroupSize).IsGreaterThanOrEqualTo(d.Elements);
		}

		PatchMatchGpuDispatch sweep = plan.Dispatches.Single(d => d.Kernel == "sweep_band");
		await Assert.That((sweep.X, sweep.Y)).IsEqualTo((38u, 1u));
	}

	[Test]
	public async Task SourceImagesPastIndexRange()
	{
		// 20 layers of 20,000 x 6,000 bytes = 2.4e9, past i32 byte indexing, while M5's
		// buffer and binding limits would take it.
		var shape = new PatchMatchGpuProblemShape(100, 80, 20, 20000, 6000);
		var (fits, _, reason) = Plan(shape, false, M5Limits);
		await Assert.That(fits).IsFalse();
		await Assert.That(reason).IsEqualTo(
			"This image is too large for the GPU: its source images need more than 2,147,483,647 values in one buffer. Using the CPU.");
	}

	[Test]
	[Arguments(100, 80, 4)]
	[Arguments(1000, 750, 20)]
	[Arguments(2400, 1800, 20)]
	[Arguments(4000, 300, 200)]
	public async Task BandSizesCoverEveryRowWithinTheTarget(int width, int height, int numSources)
	{
		var (_, plan, _) = Plan(Shape(width, height, numSources), true, M5Limits with { MaxUniformBufferBindingSize = 1 << 20 });
		int[] columns = [width, height];
		int[] rows = [height, width];
		for (int orientation = 0; orientation < 2; ++orientation)
		{
			int r = plan!.BandRows(orientation);
			int bands = plan.BandCount(orientation);
			await Assert.That(r).IsGreaterThanOrEqualTo(1);
			await Assert.That(r).IsLessThanOrEqualTo(rows[orientation]);
			await Assert.That((long)r * bands).IsGreaterThanOrEqualTo(rows[orientation]);
			await Assert.That((long)r * (bands - 1)).IsLessThan(rows[orientation]);

			// One band's work stays within the target unless a single row already exceeds it.
			long bandWork = (long)r * columns[orientation] * plan.WindowSamplesPerPixel;
			if (r > 1)
			{
				await Assert.That(bandWork).IsLessThanOrEqualTo(PatchMatchGpuPlan.TargetWindowSamplesPerDispatch);
			}
		}

		// A tiny image is one band per sweep.
		if (width == 100)
		{
			await Assert.That(plan!.BandCount(0)).IsEqualTo(1);
			await Assert.That(plan.BandCount(1)).IsEqualTo(1);
		}

		// Hundreds of sources: a single row exceeds the target, so a band is one row.
		if (numSources == 200)
		{
			await Assert.That(plan!.BandRows(0)).IsEqualTo(1);
			await Assert.That(plan.BandCount(0)).IsEqualTo(height);
		}
	}

	[Test]
	public async Task PlanIsDeterministic()
	{
		var (_, a, _) = Plan(Shape(1600, 1200, 20), true, M5Limits);
		var (_, b, _) = Plan(Shape(1600, 1200, 20), true, M5Limits);
		await Assert.That(b!.Buffers.SequenceEqual(a!.Buffers)).IsTrue();
		await Assert.That(b.Dispatches.SequenceEqual(a.Dispatches)).IsTrue();
		await Assert.That((b.BandRows(0), b.BandRows(1), b.BandCount(0), b.BandCount(1)))
			.IsEqualTo((a.BandRows(0), a.BandRows(1), a.BandCount(0), a.BandCount(1)));

		var (_, _, reasonA) = Plan(Shape(1600, 1200, 20), true, ComputeDeviceLimits.Defaults);
		var (_, _, reasonB) = Plan(Shape(1600, 1200, 20), true, ComputeDeviceLimits.Defaults);
		await Assert.That(reasonB).IsEqualTo(reasonA);
	}
}
