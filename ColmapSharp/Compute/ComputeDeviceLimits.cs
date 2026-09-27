// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ComputeDeviceLimits: the limits of an IComputeDevice (IComputeDevice.cs). Not a COLMAP port;
// part of the compute seam that lets a host run PatchMatch on its GPU (PORTING_PLAN.md
// Phase 13). The fields are the WebGPU limits (WGPULimits) that PatchMatch's buffer and
// dispatch plan can actually reach; Defaults is what a WebGPU device grants when it asks for
// nothing more, which is what a browser adapter reports unless the host raised them.

namespace ColmapSharp.Compute;

/// <summary>
/// The limits an <see cref="IComputeDevice"/> was created with. Every call that could cross
/// one is refused by the device (see <see cref="IComputeDevice"/>), so a caller plans its
/// buffers and dispatches against these values up front rather than discovering them.
/// </summary>
/// <param name="MaxBufferSize">The largest buffer <see cref="IComputeDevice.CreateBuffer"/>
/// will create, in bytes (<c>maxBufferSize</c>).</param>
/// <param name="MaxStorageBufferBindingSize">The largest byte range one
/// <see cref="ComputeBindingType.Storage"/> or <see cref="ComputeBindingType.ReadOnlyStorage"/>
/// binding may expose (<c>maxStorageBufferBindingSize</c>). Smaller than
/// <paramref name="MaxBufferSize"/> by default, so a buffer can be too big to bind whole.</param>
/// <param name="MaxStorageBuffersPerShaderStage">The most storage bindings (read-only or
/// read-write, across all groups) one kernel may declare
/// (<c>maxStorageBuffersPerShaderStage</c>).</param>
/// <param name="MaxUniformBufferBindingSize">The largest byte range one
/// <see cref="ComputeBindingType.Uniform"/> binding may expose
/// (<c>maxUniformBufferBindingSize</c>).</param>
/// <param name="MaxComputeWorkgroupsPerDimension">The most workgroups along any one dimension
/// of <see cref="IComputeDevice.Dispatch"/> (<c>maxComputeWorkgroupsPerDimension</c>).</param>
/// <param name="MaxComputeInvocationsPerWorkgroup">The most invocations one workgroup may have,
/// the product of a kernel's <c>@workgroup_size</c> (<c>maxComputeInvocationsPerWorkgroup</c>).
/// The device cannot see the WGSL's workgroup size, so this one is the kernel author's to
/// honor.</param>
/// <param name="MinStorageBufferOffsetAlignment">The alignment, in bytes, of every
/// <see cref="ComputeBufferBinding.Offset"/> (<c>minStorageBufferOffsetAlignment</c>). The seam
/// applies it to uniform bindings too, so a host adapter reports the larger of WebGPU's storage
/// and uniform offset alignments here (both are 256 by default).</param>
public readonly record struct ComputeDeviceLimits(
	long MaxBufferSize,
	long MaxStorageBufferBindingSize,
	int MaxStorageBuffersPerShaderStage,
	long MaxUniformBufferBindingSize,
	uint MaxComputeWorkgroupsPerDimension,
	int MaxComputeInvocationsPerWorkgroup,
	int MinStorageBufferOffsetAlignment)
{
	/// <summary>
	/// The WebGPU default limits: 256 MiB buffers, 128 MiB storage bindings, 8 storage buffers
	/// per stage, 64 KiB uniform bindings, 65535 workgroups per dimension, 256 invocations per
	/// workgroup, 256-byte binding offsets. Every WebGPU device supports at least these.
	/// </summary>
	public static ComputeDeviceLimits Defaults { get; } = new(
		MaxBufferSize: 268435456,
		MaxStorageBufferBindingSize: 134217728,
		MaxStorageBuffersPerShaderStage: 8,
		MaxUniformBufferBindingSize: 65536,
		MaxComputeWorkgroupsPerDimension: 65535,
		MaxComputeInvocationsPerWorkgroup: 256,
		MinStorageBufferOffsetAlignment: 256);
}
