// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// The value types of the compute seam (IComputeDevice.cs): buffer kinds, binding types, the
// kernel descriptor and bind group entries, and the handle interfaces the device returns. Not
// a COLMAP port (PORTING_PLAN.md Phase 13). Each maps one-to-one onto a WebGPU concept so a
// host adapter is a thin pass-through: a kernel is a compute pipeline with an explicit
// pipeline layout, a bind group is a bind group against one of that layout's groups.

namespace ColmapSharp.Compute;

/// <summary>What a buffer can be bound as.</summary>
public enum ComputeBufferKind
{
	/// <summary>
	/// A storage buffer: bindable as <see cref="ComputeBindingType.Storage"/> or
	/// <see cref="ComputeBindingType.ReadOnlyStorage"/>.
	/// </summary>
	Storage,

	/// <summary>A uniform buffer: bindable only as <see cref="ComputeBindingType.Uniform"/>.</summary>
	Uniform,
}

/// <summary>How a kernel declares one buffer binding (the WGSL address space and access).</summary>
public enum ComputeBindingType
{
	/// <summary><c>var&lt;uniform&gt;</c>: read-only, small, bound from a <see cref="ComputeBufferKind.Uniform"/> buffer.</summary>
	Uniform,

	/// <summary><c>var&lt;storage, read&gt;</c>: bound from a <see cref="ComputeBufferKind.Storage"/> buffer.</summary>
	ReadOnlyStorage,

	/// <summary><c>var&lt;storage, read_write&gt;</c>: bound from a <see cref="ComputeBufferKind.Storage"/> buffer.</summary>
	Storage,
}

/// <summary>One buffer binding a kernel declares: WGSL's <c>@group(Group) @binding(Binding)</c>.</summary>
/// <param name="Group">The bind group index; a kernel's groups run 0, 1, ... with no gap.</param>
/// <param name="Binding">The binding index within the group.</param>
/// <param name="Type">How the shader declares the variable.</param>
public readonly record struct ComputeKernelBinding(int Group, int Binding, ComputeBindingType Type);

/// <summary>
/// Everything needed to compile a kernel: WGSL text, the entry point to run, and the bind
/// group layout, stated explicitly so bind groups made for one kernel can be used with any
/// other kernel whose layout for that group is identical.
/// </summary>
/// <param name="Label">A debug name, shown in errors and command dumps.</param>
/// <param name="Source">The complete WGSL module text.</param>
/// <param name="EntryPoint">The <c>@compute</c> function to run.</param>
/// <param name="Bindings">Every binding the kernel's WGSL declares, in any order. It must agree
/// with the WGSL exactly; the device does not parse the source to check.</param>
public readonly record struct ComputeKernelDescriptor(
	string Label,
	string Source,
	string EntryPoint,
	IReadOnlyList<ComputeKernelBinding> Bindings);

/// <summary>One entry of a bind group: a byte range of a buffer exposed at one binding.</summary>
/// <param name="Binding">The binding index within the group.</param>
/// <param name="Buffer">The buffer to bind.</param>
/// <param name="Offset">Start of the range, a multiple of
/// <see cref="ComputeDeviceLimits.MinStorageBufferOffsetAlignment"/>.</param>
/// <param name="Size">Length of the range, a positive multiple of 4, within the binding-size
/// limit of its type.</param>
public readonly record struct ComputeBufferBinding(int Binding, IComputeBuffer Buffer, long Offset, long Size);

/// <summary>
/// A GPU buffer. Its contents start as the initial data given at creation, zero after it.
/// Disposing it releases the GPU memory; it must not be disposed while a dispatch that binds it
/// is recorded but not yet flushed.
/// </summary>
public interface IComputeBuffer : IDisposable
{
	/// <summary>Size in bytes.</summary>
	long Size { get; }

	/// <summary>What the buffer can be bound as.</summary>
	ComputeBufferKind Kind { get; }

	/// <summary>The debug name given at creation, or null.</summary>
	string? Label { get; }
}

/// <summary>A compiled kernel (a WebGPU compute pipeline with its explicit layout).</summary>
public interface IComputeKernel : IDisposable
{
	/// <summary>The descriptor's debug name.</summary>
	string Label { get; }
}

/// <summary>
/// A set of buffer ranges bound to one group index of a kernel's layout. Holds its buffers; a
/// bind group whose buffer has been disposed can no longer be dispatched.
/// </summary>
public interface IComputeBindGroup : IDisposable
{
	/// <summary>The group index it was created for.</summary>
	int Group { get; }
}
