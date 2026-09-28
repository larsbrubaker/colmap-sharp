// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// RecordingComputeDevice: an IComputeDevice (Compute/IComputeDevice.cs) that talks to no GPU.
// Not a COLMAP port (PORTING_PLAN.md Phase 13). It records every call as a ComputeCommand
// (RecordingComputeHandles.cs) and backs buffers with managed bytes, so the GPU PatchMatch
// orchestration can be tested headlessly by asserting on the command stream: which kernels
// run, in what order, how many flushes. It runs no kernels; a read returns what was uploaded.
//
// It is deliberately strict about every rule IComputeDevice states - limits, alignment,
// layouts, disposed handles, writable-binding aliasing, and the write-after-dispatch hazard -
// because a real WebGPU
// device reports most of those out of band and a readback then returns stale or zero bytes.
// A test that never sees a rule enforced proves nothing about a caller that keeps it.

using System.Numerics;

namespace ColmapSharp.Compute.Testing;

/// <summary>
/// A headless <see cref="IComputeDevice"/> that records every call and enforces every rule of
/// the seam. Buffers hold real bytes (initial data and writes); dispatches execute nothing.
/// </summary>
public sealed class RecordingComputeDevice : IComputeDevice
{
	private readonly List<ComputeCommand> commands = new();

	// Dispatches recorded since the last flush, and every buffer they bind: the buffers a
	// WriteBuffer must not touch until the next flush submits those dispatches.
	private readonly List<DispatchCommand> pendingDispatches = new();
	private readonly HashSet<RecordingComputeBuffer> pendingBoundBuffers = new();
	private readonly ComputeDeviceLimits limits = ComputeDeviceLimits.Defaults;
	private int bufferCount;

	// Set by FaultNextSubmit: the GPU-side failure (a pipeline that failed to compile, a lost
	// device) the next submission reports through its task.
	private Exception? nextSubmitFault;

	/// <summary>
	/// The limits this double reports and enforces, fixed at construction (an object
	/// initializer) so a test can pin small limits and prove a caller that plans against them
	/// really stays inside them.
	/// </summary>
	/// <exception cref="ArgumentException">The offset alignment is not a positive power of two.</exception>
	public ComputeDeviceLimits Limits
	{
		get => this.limits;
		init
		{
			var alignment = value.MinStorageBufferOffsetAlignment;
			if (alignment <= 0 || !BitOperations.IsPow2(alignment))
			{
				throw new ArgumentException($"MinStorageBufferOffsetAlignment ({alignment}) must be a positive power of two.", nameof(value));
			}

			this.limits = value;
		}
	}

	/// <summary>
	/// What the double reports. When false (a browser device) <see cref="FlushAsync"/> and
	/// <see cref="ReadBufferAsync"/> stay pending until the caller awaits them (like a map that
	/// completes only once control returns to the JS event loop), so a caller that blocks on
	/// them instead of awaiting is caught (DeferredCompletion.cs); when true they complete
	/// before returning.
	/// </summary>
	public bool SupportsBlockingWait { get; set; } = true;

	/// <summary>Every call received, oldest first.</summary>
	public IReadOnlyList<ComputeCommand> Commands => this.commands;

	/// <summary>Dispatches recorded since the last flush or read.</summary>
	public int PendingDispatchCount => this.pendingDispatches.Count;

	/// <summary>Every recorded command of one kind, in order.</summary>
	/// <typeparam name="T">The command type wanted.</typeparam>
	public IReadOnlyList<T> CommandsOf<T>()
		where T : ComputeCommand
		=> this.commands.OfType<T>().ToList();

	/// <summary>
	/// Makes the next submission (the next <see cref="FlushAsync"/> or
	/// <see cref="ReadBufferAsync"/>) fail on the "GPU": it still submits everything pending,
	/// then its task faults with <paramref name="fault"/>, as a real device reports a WGSL
	/// compile error or a lost device. Lets a test prove its caller handles such a fault.
	/// </summary>
	/// <param name="fault">The exception the task faults with.</param>
	public void FaultNextSubmit(Exception fault)
	{
		ArgumentNullException.ThrowIfNull(fault);
		this.nextSubmitFault = fault;
	}

	/// <summary>Drops the recorded commands so a test can measure only what follows.</summary>
	public void ClearRecording() => this.commands.Clear();

	/// <summary>The command stream as one line per command; a readable assertion message.</summary>
	public string Dump() => string.Join(Environment.NewLine, this.commands.Select(command => command.ToString()));

	/// <inheritdoc/>
	public IComputeBuffer CreateBuffer(ComputeBufferKind kind, long size, ReadOnlySpan<byte> initial = default, string? label = null)
	{
		if (size <= 0 || size % 4 != 0)
		{
			throw new ArgumentException($"A buffer's size ({size}) must be a positive multiple of 4.", nameof(size));
		}

		if (initial.Length > size)
		{
			throw new ArgumentException($"{initial.Length} bytes of initial data do not fit a {size} byte buffer.", nameof(initial));
		}

		if (size > this.Limits.MaxBufferSize)
		{
			throw new InvalidOperationException(
				$"A {size:N0} byte buffer exceeds this device's maxBufferSize of {this.Limits.MaxBufferSize:N0} bytes.");
		}

		var buffer = new RecordingComputeBuffer(this, this.bufferCount++, kind, size, label);
		buffer.Write(0, initial);
		this.commands.Add(new CreateBufferCommand(buffer, initial.Length));
		return buffer;
	}

	/// <inheritdoc/>
	public void WriteBuffer(IComputeBuffer buffer, long offset, ReadOnlySpan<byte> data)
	{
		var target = this.Own(buffer, nameof(buffer));
		ValidateRange(target, offset, data.Length, "write");
		if (this.pendingBoundBuffers.Contains(target))
		{
			throw new InvalidOperationException(
				$"WriteBuffer to {target}, which a dispatch recorded since the last flush binds. A write is a queue"
				+ " write that lands before the flush runs any dispatch, so that dispatch would read the new bytes;"
				+ " give per-dispatch data its own buffer, or flush before writing.");
		}

		target.Write(offset, data);
		this.commands.Add(new WriteBufferCommand(target, offset, data.Length));
	}

	/// <inheritdoc/>
	public IComputeKernel CreateKernel(in ComputeKernelDescriptor descriptor)
	{
		RequireText(descriptor.Label, "label");
		RequireText(descriptor.Source, "WGSL source");
		RequireText(descriptor.EntryPoint, "entry point");
		if (descriptor.Bindings == null)
		{
			throw new ArgumentException($"Kernel '{descriptor.Label}' has no bindings list.", nameof(descriptor));
		}

		var groupCount = 0;
		var storageCount = 0;
		var uniformCount = 0;
		var seen = new HashSet<(int Group, int Binding)>();
		foreach (var binding in descriptor.Bindings)
		{
			if (binding.Group < 0 || binding.Binding < 0)
			{
				throw new ArgumentException($"Kernel '{descriptor.Label}' declares a negative group or binding index ({binding}).", nameof(descriptor));
			}

			if (!seen.Add((binding.Group, binding.Binding)))
			{
				throw new ArgumentException(
					$"Kernel '{descriptor.Label}' declares @group({binding.Group}) @binding({binding.Binding}) twice.", nameof(descriptor));
			}

			if (binding.Binding >= this.Limits.MaxBindingsPerBindGroup)
			{
				throw new ArgumentException(
					$"Kernel '{descriptor.Label}' declares @binding({binding.Binding}); this device allows binding indices below"
					+ $" {this.Limits.MaxBindingsPerBindGroup}.",
					nameof(descriptor));
			}

			groupCount = Math.Max(groupCount, binding.Group + 1);
			if (binding.Type == ComputeBindingType.Uniform)
			{
				uniformCount++;
			}
			else
			{
				storageCount++;
			}
		}

		if (groupCount > this.Limits.MaxBindGroups)
		{
			throw new ArgumentException(
				$"Kernel '{descriptor.Label}' uses {groupCount} bind groups; this device allows at most {this.Limits.MaxBindGroups}.",
				nameof(descriptor));
		}

		var layouts = new IReadOnlyList<ComputeKernelBinding>[groupCount];
		for (var group = 0; group < groupCount; group++)
		{
			layouts[group] = descriptor.Bindings.Where(b => b.Group == group).OrderBy(b => b.Binding).ToArray();
			if (layouts[group].Count == 0)
			{
				// A pipeline layout has a bind group layout at every index below its highest, so a
				// gap would need an empty bind group bound at every dispatch.
				throw new ArgumentException(
					$"Kernel '{descriptor.Label}' declares no binding in group {group} but uses group {groupCount - 1};"
					+ " groups must run 0, 1, ... with no gap.",
					nameof(descriptor));
			}
		}

		if (storageCount > this.Limits.MaxStorageBuffersPerShaderStage)
		{
			throw new ArgumentException(
				$"Kernel '{descriptor.Label}' declares {storageCount} storage buffers; this device allows at most"
				+ $" {this.Limits.MaxStorageBuffersPerShaderStage} per shader stage.",
				nameof(descriptor));
		}

		if (uniformCount > this.Limits.MaxUniformBuffersPerShaderStage)
		{
			throw new ArgumentException(
				$"Kernel '{descriptor.Label}' declares {uniformCount} uniform buffers; this device allows at most"
				+ $" {this.Limits.MaxUniformBuffersPerShaderStage} per shader stage.",
				nameof(descriptor));
		}

		var kernel = new RecordingComputeKernel(this, descriptor, layouts);
		this.commands.Add(new CreateKernelCommand(kernel));
		return kernel;
	}

	/// <inheritdoc/>
	public IComputeBindGroup CreateBindGroup(IComputeKernel kernel, int group, ReadOnlySpan<ComputeBufferBinding> entries)
	{
		var owner = this.Own(kernel, nameof(kernel));
		if (group < 0 || group >= owner.GroupLayouts.Count)
		{
			throw new ArgumentException($"{owner} declares no group {group}.", nameof(group));
		}

		var layout = owner.GroupLayouts[group];
		if (entries.Length != layout.Count)
		{
			throw new ArgumentException(
				$"{owner} declares {layout.Count} bindings in group {group}, but {entries.Length} entries were given.", nameof(entries));
		}

		var sorted = entries.ToArray().OrderBy(e => e.Binding).ToArray();
		for (var i = 0; i < sorted.Length; i++)
		{
			var entry = sorted[i];
			var declared = layout[i];
			if (entry.Binding != declared.Binding)
			{
				throw new ArgumentException(
					$"{owner} group {group} has no binding {entry.Binding}, or it was given twice; the layout declares"
					+ $" bindings {string.Join(", ", layout.Select(b => b.Binding))}.",
					nameof(entries));
			}

			this.ValidateEntry(owner, group, entry, declared.Type);
		}

		var bindGroup = new RecordingComputeBindGroup(this, owner, group, sorted);
		this.commands.Add(new CreateBindGroupCommand(bindGroup));
		return bindGroup;
	}

	/// <inheritdoc/>
	public void Dispatch(IComputeKernel kernel, ReadOnlySpan<IComputeBindGroup> groups, uint x, uint y = 1, uint z = 1)
	{
		var owner = this.Own(kernel, nameof(kernel));
		if (groups.Length != owner.GroupLayouts.Count)
		{
			throw new ArgumentException(
				$"{owner} declares {owner.GroupLayouts.Count} bind groups, but {groups.Length} were given.", nameof(groups));
		}

		var bound = new RecordingComputeBindGroup[groups.Length];
		for (var i = 0; i < groups.Length; i++)
		{
			var bindGroup = this.Own(groups[i], nameof(groups));
			if (bindGroup.Group != i || !bindGroup.Layout.SequenceEqual(owner.GroupLayouts[i]))
			{
				throw new ArgumentException(
					$"The bind group at index {i} ({bindGroup}, made for {bindGroup.Kernel}'s group {bindGroup.Group})"
					+ $" does not match {owner}'s layout for group {i}.",
					nameof(groups));
			}

			foreach (var entry in bindGroup.Entries)
			{
				if (((RecordingComputeBuffer)entry.Buffer).IsDisposed)
				{
					throw new ObjectDisposedException(entry.Buffer.ToString(), $"The bind group at index {i} binds {entry.Buffer}, which has been disposed.");
				}
			}

			bound[i] = bindGroup;
		}

		RejectWritableAliasing(owner, bound);
		ValidateWorkgroups(this.Limits, x, nameof(x));
		ValidateWorkgroups(this.Limits, y, nameof(y));
		ValidateWorkgroups(this.Limits, z, nameof(z));

		var dispatch = new DispatchCommand(owner, bound, x, y, z);
		this.commands.Add(dispatch);
		this.pendingDispatches.Add(dispatch);
		foreach (var bindGroup in bound)
		{
			foreach (var entry in bindGroup.Entries)
			{
				this.pendingBoundBuffers.Add((RecordingComputeBuffer)entry.Buffer);
			}
		}
	}

	/// <inheritdoc/>
	public ValueTask FlushAsync(CancellationToken cancellationToken = default)
	{
		// Cancellation is observed only here, before submitting, and even then everything
		// pending is submitted so the device is left with nothing pending.
		var canceled = cancellationToken.IsCancellationRequested;
		var (dispatchCount, fault) = this.Submit();
		this.commands.Add(new FlushCommand(dispatchCount));
		return this.Complete(fault, canceled, cancellationToken, onSuccess: null);
	}

	/// <inheritdoc/>
	public ValueTask ReadBufferAsync(IComputeBuffer buffer, long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
	{
		var source = this.Own(buffer, nameof(buffer));
		ValidateRange(source, offset, destination.Length, "read");
		var canceled = cancellationToken.IsCancellationRequested;
		var (dispatchCount, fault) = this.Submit();
		if (canceled)
		{
			// The pending work went out, but the copy to the readback buffer did not.
			this.commands.Add(new FlushCommand(dispatchCount));
			return this.Complete(fault, canceled, cancellationToken, onSuccess: null);
		}

		// The GPU copies the bytes as of submission; the caller sees them only when the map
		// completes, so a browser caller cannot mistake an unfinished read for a finished one.
		var snapshot = new byte[destination.Length];
		source.Read(offset, snapshot);
		this.commands.Add(new ReadBufferCommand(source, offset, destination.Length, dispatchCount));
		return this.Complete(fault, canceled: false, cancellationToken, () => snapshot.CopyTo(destination.Span));
	}

	// Rejects a dispatch that binds a buffer read_write and also binds it anywhere else: WebGPU
	// tracks usage per whole buffer per dispatch, so even disjoint ranges fail validation.
	private static void RejectWritableAliasing(RecordingComputeKernel kernel, RecordingComputeBindGroup[] bound)
	{
		var bindingCounts = new Dictionary<RecordingComputeBuffer, int>();
		var writable = new List<RecordingComputeBuffer>();
		for (var i = 0; i < bound.Length; i++)
		{
			var layout = kernel.GroupLayouts[i];
			var entries = bound[i].Entries;
			for (var e = 0; e < entries.Count; e++)
			{
				var buffer = (RecordingComputeBuffer)entries[e].Buffer;
				bindingCounts[buffer] = bindingCounts.GetValueOrDefault(buffer) + 1;
				if (layout[e].Type == ComputeBindingType.Storage)
				{
					writable.Add(buffer);
				}
			}
		}

		foreach (var buffer in writable)
		{
			if (bindingCounts[buffer] > 1)
			{
				throw new ArgumentException(
					$"{kernel} binds {buffer} read_write and binds it again in the same dispatch. WebGPU forbids any other"
					+ " binding of a buffer bound read_write, even of a disjoint range; split it into separate buffers.",
					"groups");
			}
		}
	}

	/// <summary>
	/// Hands the pending dispatches to the (absent) GPU. Always leaves nothing pending, even when
	/// the submission faults. Returns how many dispatches went and the fault to report, if any.
	/// </summary>
	private (int DispatchCount, Exception? Fault) Submit()
	{
		var count = this.pendingDispatches.Count;
		var fault = this.nextSubmitFault;
		this.nextSubmitFault = null;
		this.pendingDispatches.Clear();
		this.pendingBoundBuffers.Clear();
		return (count, fault);
	}

	// A GPU fault wins over cancellation: it is the more important thing for the caller to see.
	private ValueTask Complete(Exception? fault, bool canceled, CancellationToken cancellationToken, Action? onSuccess)
	{
		if (!this.SupportsBlockingWait)
		{
			var deferred = new DeferredCompletion(fault, canceled, cancellationToken, onSuccess);
			return new ValueTask(deferred, 0);
		}

		if (fault != null)
		{
			return ValueTask.FromException(fault);
		}

		if (canceled)
		{
			return ValueTask.FromCanceled(cancellationToken);
		}

		onSuccess?.Invoke();
		return default;
	}

	private void ValidateEntry(RecordingComputeKernel kernel, int group, in ComputeBufferBinding entry, ComputeBindingType type)
	{
		var buffer = this.Own(entry.Buffer, "entries");
		var where = $"{kernel} group {group} binding {entry.Binding}";
		var wantKind = type == ComputeBindingType.Uniform ? ComputeBufferKind.Uniform : ComputeBufferKind.Storage;
		if (buffer.Kind != wantKind)
		{
			throw new ArgumentException($"{where} is declared {type} and needs a {wantKind} buffer; {buffer} is {buffer.Kind}.", "entries");
		}

		if (entry.Offset < 0 || entry.Offset % this.Limits.MinStorageBufferOffsetAlignment != 0)
		{
			throw new ArgumentException(
				$"{where}: offset {entry.Offset} is not a multiple of minStorageBufferOffsetAlignment ({this.Limits.MinStorageBufferOffsetAlignment}).", "entries");
		}

		if (entry.Size <= 0 || entry.Size % 4 != 0)
		{
			throw new ArgumentException($"{where}: size {entry.Size} is not a positive multiple of 4.", "entries");
		}

		if (entry.Offset > buffer.Size || entry.Size > buffer.Size - entry.Offset)
		{
			throw new ArgumentException($"{where}: {entry.Size} bytes at offset {entry.Offset} run past the end of {buffer} ({buffer.Size} bytes).", "entries");
		}

		var limit = type == ComputeBindingType.Uniform ? this.Limits.MaxUniformBufferBindingSize : this.Limits.MaxStorageBufferBindingSize;
		if (entry.Size > limit)
		{
			throw new ArgumentException(
				$"{where}: a {entry.Size:N0} byte {type} binding exceeds this device's limit of {limit:N0} bytes.", "entries");
		}
	}

	internal static void ValidateRange(RecordingComputeBuffer buffer, long offset, int length, string verb)
	{
		if (offset < 0 || offset % 4 != 0 || length % 4 != 0)
		{
			throw new ArgumentException($"A buffer {verb}'s offset ({offset}) and length ({length}) must be multiples of 4.", nameof(offset));
		}

		if (offset > buffer.Size || length > buffer.Size - offset)
		{
			throw new ArgumentException($"A {verb} of {length} bytes at offset {offset} runs past the end of {buffer} ({buffer.Size} bytes).", nameof(offset));
		}
	}

	private static void ValidateWorkgroups(in ComputeDeviceLimits limits, uint count, string parameterName)
	{
		if (count > limits.MaxComputeWorkgroupsPerDimension)
		{
			throw new ArgumentOutOfRangeException(
				parameterName,
				count,
				$"A dispatch may have at most {limits.MaxComputeWorkgroupsPerDimension:N0} workgroups along one dimension.");
		}
	}

	private static void RequireText(string? value, string what)
	{
		if (string.IsNullOrEmpty(value))
		{
			throw new ArgumentException($"A kernel needs a {what}.", "descriptor");
		}
	}

	private RecordingComputeBuffer Own(IComputeBuffer? handle, string parameterName)
	{
		if (handle is not RecordingComputeBuffer buffer || buffer.Device != this)
		{
			throw new ArgumentException("The buffer was not created by this device.", parameterName);
		}

		ObjectDisposedException.ThrowIf(buffer.IsDisposed, buffer);
		return buffer;
	}

	private RecordingComputeKernel Own(IComputeKernel? handle, string parameterName)
	{
		if (handle is not RecordingComputeKernel kernel || kernel.Device != this)
		{
			throw new ArgumentException("The kernel was not created by this device.", parameterName);
		}

		ObjectDisposedException.ThrowIf(kernel.IsDisposed, kernel);
		return kernel;
	}

	private RecordingComputeBindGroup Own(IComputeBindGroup? handle, string parameterName)
	{
		if (handle is not RecordingComputeBindGroup bindGroup || bindGroup.Device != this)
		{
			throw new ArgumentException("The bind group was not created by this device.", parameterName);
		}

		ObjectDisposedException.ThrowIf(bindGroup.IsDisposed, bindGroup);
		return bindGroup;
	}
}
