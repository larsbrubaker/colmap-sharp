// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// The handles and command records of RecordingComputeDevice (RecordingComputeDevice.cs), the
// headless IComputeDevice double. Not a COLMAP port (PORTING_PLAN.md Phase 13). Public and
// shipped, like agg-sharp's RenderCore/Testing, so a host can test its own use of the seam.
// The commands are what a test asserts on: the stream of creates, writes, dispatches, flushes
// and reads, in the order they were received.

using System.Globalization;

namespace ColmapSharp.Compute.Testing;

/// <summary>A buffer of <see cref="RecordingComputeDevice"/>, backed by managed bytes.</summary>
public sealed class RecordingComputeBuffer : IComputeBuffer
{
	// Allocated on the first write, so a test can plan buffers of realistic (hundreds of
	// megabytes) size without paying for them; until then the buffer reads as zeros.
	private byte[]? contents;

	internal RecordingComputeBuffer(RecordingComputeDevice device, int id, ComputeBufferKind kind, long size, string? label)
	{
		this.Device = device;
		this.Id = id;
		this.Kind = kind;
		this.Size = size;
		this.Label = label;
	}

	/// <inheritdoc/>
	public long Size { get; }

	/// <inheritdoc/>
	public ComputeBufferKind Kind { get; }

	/// <inheritdoc/>
	public string? Label { get; }

	/// <summary>Creation order among this device's buffers, from 0; names the buffer in dumps.</summary>
	public int Id { get; }

	/// <summary>True once disposed.</summary>
	public bool IsDisposed { get; private set; }

	internal RecordingComputeDevice Device { get; }

	/// <summary>
	/// A copy of the buffer's current bytes: initial data and every write so far. The double
	/// runs no kernels, so dispatches never change them.
	/// </summary>
	public byte[] GetContents()
	{
		var copy = new byte[this.Size];
		this.contents?.CopyTo(copy, 0);
		return copy;
	}

	/// <inheritdoc/>
	public void Dispose() => this.IsDisposed = true;

	/// <inheritdoc/>
	public override string ToString() => $"buffer#{this.Id.ToString(CultureInfo.InvariantCulture)}"
		+ (this.Label == null ? string.Empty : $" '{this.Label}'");

	internal void Write(long offset, ReadOnlySpan<byte> data)
	{
		if (data.IsEmpty)
		{
			return;
		}

		this.contents ??= new byte[this.Size];
		data.CopyTo(this.contents.AsSpan(checked((int)offset)));
	}

	internal void Read(long offset, Span<byte> destination)
	{
		if (this.contents == null)
		{
			destination.Clear();
			return;
		}

		this.contents.AsSpan(checked((int)offset), destination.Length).CopyTo(destination);
	}
}

/// <summary>A kernel of <see cref="RecordingComputeDevice"/>. Nothing is compiled.</summary>
public sealed class RecordingComputeKernel : IComputeKernel
{
	internal RecordingComputeKernel(RecordingComputeDevice device, ComputeKernelDescriptor descriptor, IReadOnlyList<ComputeKernelBinding>[] groupLayouts)
	{
		this.Device = device;
		this.Descriptor = descriptor;
		this.GroupLayouts = groupLayouts;
	}

	/// <inheritdoc/>
	public string Label => this.Descriptor.Label;

	/// <summary>The descriptor the kernel was created from.</summary>
	public ComputeKernelDescriptor Descriptor { get; }

	/// <summary>The declared bindings of each group, indexed by group, each sorted by binding.</summary>
	public IReadOnlyList<IReadOnlyList<ComputeKernelBinding>> GroupLayouts { get; }

	/// <summary>True once disposed.</summary>
	public bool IsDisposed { get; private set; }

	internal RecordingComputeDevice Device { get; }

	/// <inheritdoc/>
	public void Dispose() => this.IsDisposed = true;

	/// <inheritdoc/>
	public override string ToString() => $"kernel '{this.Label}' ({this.Descriptor.EntryPoint})";
}

/// <summary>A bind group of <see cref="RecordingComputeDevice"/>.</summary>
public sealed class RecordingComputeBindGroup : IComputeBindGroup
{
	internal RecordingComputeBindGroup(RecordingComputeDevice device, RecordingComputeKernel kernel, int group, IReadOnlyList<ComputeBufferBinding> entries)
	{
		this.Device = device;
		this.Kernel = kernel;
		this.Group = group;
		this.Entries = entries;
	}

	/// <inheritdoc/>
	public int Group { get; }

	/// <summary>The kernel whose layout the group was created against.</summary>
	public RecordingComputeKernel Kernel { get; }

	/// <summary>The bound buffer ranges, sorted by binding.</summary>
	public IReadOnlyList<ComputeBufferBinding> Entries { get; }

	/// <summary>The layout the group follows: its kernel's bindings for <see cref="Group"/>.</summary>
	public IReadOnlyList<ComputeKernelBinding> Layout => this.Kernel.GroupLayouts[this.Group];

	/// <summary>True once disposed.</summary>
	public bool IsDisposed { get; private set; }

	internal RecordingComputeDevice Device { get; }

	/// <inheritdoc/>
	public void Dispose() => this.IsDisposed = true;

	/// <inheritdoc/>
	public override string ToString() => $"group {this.Group.ToString(CultureInfo.InvariantCulture)} ["
		+ string.Join(", ", this.Entries.Select(e => FormattableString.Invariant($"{e.Binding}: {e.Buffer} +{e.Offset} x{e.Size}")))
		+ "]";
}

/// <summary>One call a <see cref="RecordingComputeDevice"/> received.</summary>
public abstract record ComputeCommand;

/// <summary>A <see cref="IComputeDevice.CreateBuffer"/> call.</summary>
/// <param name="Buffer">The buffer created.</param>
/// <param name="InitialLength">Bytes of initial data given.</param>
public sealed record CreateBufferCommand(RecordingComputeBuffer Buffer, int InitialLength) : ComputeCommand
{
	/// <inheritdoc/>
	public override string ToString() => FormattableString.Invariant(
		$"CreateBuffer {this.Buffer} {this.Buffer.Kind} {this.Buffer.Size} bytes, {this.InitialLength} initial");
}

/// <summary>A <see cref="IComputeDevice.WriteBuffer"/> call.</summary>
/// <param name="Buffer">The destination.</param>
/// <param name="Offset">Byte offset written at.</param>
/// <param name="Length">Bytes written.</param>
public sealed record WriteBufferCommand(RecordingComputeBuffer Buffer, long Offset, int Length) : ComputeCommand
{
	/// <inheritdoc/>
	public override string ToString() => FormattableString.Invariant($"WriteBuffer {this.Buffer} +{this.Offset} x{this.Length}");
}

/// <summary>A <see cref="IComputeDevice.CreateKernel"/> call.</summary>
/// <param name="Kernel">The kernel created.</param>
public sealed record CreateKernelCommand(RecordingComputeKernel Kernel) : ComputeCommand
{
	/// <inheritdoc/>
	public override string ToString() => $"CreateKernel {this.Kernel}";
}

/// <summary>A <see cref="IComputeDevice.CreateBindGroup"/> call.</summary>
/// <param name="BindGroup">The bind group created.</param>
public sealed record CreateBindGroupCommand(RecordingComputeBindGroup BindGroup) : ComputeCommand
{
	/// <inheritdoc/>
	public override string ToString() => $"CreateBindGroup for {this.BindGroup.Kernel} {this.BindGroup}";
}

/// <summary>A <see cref="IComputeDevice.Dispatch"/> call.</summary>
/// <param name="Kernel">The kernel dispatched.</param>
/// <param name="Groups">The bind groups, indexed by group.</param>
/// <param name="X">Workgroups along X.</param>
/// <param name="Y">Workgroups along Y.</param>
/// <param name="Z">Workgroups along Z.</param>
public sealed record DispatchCommand(RecordingComputeKernel Kernel, IReadOnlyList<RecordingComputeBindGroup> Groups, uint X, uint Y, uint Z) : ComputeCommand
{
	/// <inheritdoc/>
	public override string ToString() => FormattableString.Invariant($"  Dispatch {this.Kernel} {this.X}x{this.Y}x{this.Z}");
}

/// <summary>A <see cref="IComputeDevice.FlushAsync"/> call that submitted.</summary>
/// <param name="DispatchCount">Dispatches it carried.</param>
public sealed record FlushCommand(int DispatchCount) : ComputeCommand
{
	/// <inheritdoc/>
	public override string ToString() => FormattableString.Invariant($"Flush ({this.DispatchCount} dispatches)");
}

/// <summary>A <see cref="IComputeDevice.ReadBufferAsync"/> call that submitted.</summary>
/// <param name="Buffer">The buffer read.</param>
/// <param name="Offset">Byte offset read from.</param>
/// <param name="Length">Bytes read.</param>
/// <param name="DispatchCount">Recorded dispatches the read flushed ahead of itself.</param>
public sealed record ReadBufferCommand(RecordingComputeBuffer Buffer, long Offset, int Length, int DispatchCount) : ComputeCommand
{
	/// <inheritdoc/>
	public override string ToString() => FormattableString.Invariant(
		$"ReadBuffer {this.Buffer} +{this.Offset} x{this.Length} ({this.DispatchCount} dispatches flushed)");
}
