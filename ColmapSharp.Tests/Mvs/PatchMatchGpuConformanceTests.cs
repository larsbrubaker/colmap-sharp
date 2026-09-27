// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchGpuConformanceTests: C#-only tests (COLMAP has no counterpart; its PatchMatch is
// CUDA) of ColmapSharp/Mvs/Testing/PatchMatchGpuConformance*.cs, the check a host runs on its
// IComputeDevice. On the CPU twin (ReferenceComputeDevice) every check passes and the Tier C
// agreement measures are perfect, since the twin computes through the CPU code; the two WGSL
// probes the twin cannot compile report "not applicable". A device that corrupts the depth
// plane of every state readback must fail exactly the depth checks, and a failure is a report
// entry, not an exception.

using System.Runtime.InteropServices;
using System.Text.Json;

using ColmapSharp.Compute;
using ColmapSharp.Mvs.Testing;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class PatchMatchGpuConformanceTests
{
	[Test]
	public async Task ReferenceDevice_PassesEveryCheckWithPerfectAgreement()
	{
		PatchMatchGpuConformanceReport report = await PatchMatchGpuConformance.RunAsync(new ReferenceComputeDevice());
		string text = report.ToString();
		await Assert.That(report.AllPassed).IsTrue().Because(text);

		string[] notApplicable = report.Checks.Where(c => c.Outcome == PatchMatchGpuConformanceOutcome.NotApplicable).Select(c => c.Name).ToArray();
		await Assert.That(string.Join(",", notApplicable)).IsEqualTo("probe.random.differing_draws,probe.conversion.differing_values");

		// The twin is bit-identical to the CPU, so the bounded measures are at their ideal.
		foreach (string config in new[] { "photometric", "geometric", "filter" })
		{
			await Assert.That(Measured(report, $"run.{config}.cpu.depth_agreement")).IsEqualTo(1.0);
			await Assert.That(Measured(report, $"run.{config}.cpu.median_normal_angle_degrees")).IsEqualTo(0.0);
			await Assert.That(Measured(report, $"run.{config}.cpu.valid_count_difference")).IsEqualTo(0.0);
			await Assert.That(Measured(report, $"run.{config}.cpu.graph_agreement")).IsEqualTo(1.0);
		}

		await Assert.That(Measured(report, "kernel.init_random.depth_max_ulps")).IsEqualTo(0.0);
		await Assert.That(Measured(report, "kernel.initial_cost.agreement")).IsEqualTo(1.0);
		await Assert.That(report.Timings.Select(t => t.Config).ToArray()).IsEquivalentTo(new[] { "photometric", "geometric", "filter" });

		// The JSON carries the same verdict and one entry per check; NaN measures become null.
		using JsonDocument json = JsonDocument.Parse(report.ToJson());
		await Assert.That(json.RootElement.GetProperty("allPassed").GetBoolean()).IsTrue();
		await Assert.That(json.RootElement.GetProperty("checks").GetArrayLength()).IsEqualTo(report.Checks.Count);
		await Assert.That(json.RootElement.GetProperty("checks")[0].GetProperty("measured").ValueKind).IsEqualTo(JsonValueKind.Null);
		await Assert.That(json.RootElement.GetProperty("timings").GetArrayLength()).IsEqualTo(3);
	}

	[Test]
	public async Task DeviceCorruptingDepthReadbacks_FailsTheDepthChecks()
	{
		PatchMatchGpuConformanceReport report = await PatchMatchGpuConformance.RunAsync(new DepthCorruptingDevice(new ReferenceComputeDevice()));
		await Assert.That(report.AllPassed).IsFalse();

		// The probes fail because this device is not the twin and cannot compile them; the rest
		// fail only where the corrupted depth reaches: init_random's depths, the depth truth
		// checks and the depth agreement. Normals, costs, the graph, the valid count (0 * 1.05
		// is still 0) and the repeat (the corruption is deterministic) all still pass.
		string[] failed = report.Failures.Select(c => c.Name).ToArray();
		await Assert.That(string.Join("\n", failed)).IsEqualTo(string.Join("\n", new[]
		{
			"probe.random.differing_draws",
			"probe.conversion.differing_values",
			"kernel.init_random.depth_max_ulps",
			"run.photometric.truth.depth_within_tolerance",
			"run.photometric.cpu.depth_agreement",
			"run.geometric.truth.depth_within_tolerance",
			"run.geometric.cpu.depth_agreement",
			"run.filter.cpu.depth_agreement",
		})).Because(report.ToString());
		await Assert.That(report.Checks.First(c => c.Name == "probe.random.differing_draws").Detail).Contains("ArgumentException");

		using JsonDocument json = JsonDocument.Parse(report.ToJson());
		await Assert.That(json.RootElement.GetProperty("allPassed").GetBoolean()).IsFalse();
	}

	private static double Measured(PatchMatchGpuConformanceReport report, string name) => report.Checks.Single(c => c.Name == name).Measured;

	/// <summary>
	/// Forwards to an inner device, but scales the depth plane (the first quarter) of every
	/// readback of a buffer labeled as PatchMatch state by 1.05: a device whose arithmetic is
	/// wrong in exactly one output.
	/// </summary>
	private sealed class DepthCorruptingDevice(IComputeDevice inner) : IComputeDevice
	{
		public ComputeDeviceLimits Limits => inner.Limits;

		public bool SupportsBlockingWait => inner.SupportsBlockingWait;

		public IComputeBuffer CreateBuffer(ComputeBufferKind kind, long size, ReadOnlySpan<byte> initial = default, string? label = null)
			=> inner.CreateBuffer(kind, size, initial, label);

		public void WriteBuffer(IComputeBuffer buffer, long offset, ReadOnlySpan<byte> data) => inner.WriteBuffer(buffer, offset, data);

		public IComputeKernel CreateKernel(in ComputeKernelDescriptor descriptor) => inner.CreateKernel(descriptor);

		public IComputeBindGroup CreateBindGroup(IComputeKernel kernel, int group, ReadOnlySpan<ComputeBufferBinding> entries)
			=> inner.CreateBindGroup(kernel, group, entries);

		public void Dispatch(IComputeKernel kernel, ReadOnlySpan<IComputeBindGroup> groups, uint x, uint y = 1, uint z = 1)
			=> inner.Dispatch(kernel, groups, x, y, z);

		public ValueTask FlushAsync(CancellationToken cancellationToken = default) => inner.FlushAsync(cancellationToken);

		public async ValueTask ReadBufferAsync(IComputeBuffer buffer, long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
		{
			await inner.ReadBufferAsync(buffer, offset, destination, cancellationToken);
			if (offset == 0 && buffer.Label is { } label && label.Contains("state", StringComparison.Ordinal))
			{
				Span<float> depths = MemoryMarshal.Cast<byte, float>(destination.Span);
				depths = depths[..(depths.Length / 4)];
				for (int i = 0; i < depths.Length; ++i)
				{
					depths[i] *= 1.05f;
				}
			}
		}
	}
}
