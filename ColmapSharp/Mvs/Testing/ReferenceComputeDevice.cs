// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ReferenceComputeDevice: an IComputeDevice (Compute/IComputeDevice.cs) that runs the GPU
// PatchMatch kernels (Mvs/Shaders/patch_match_*.wgsl) on the CPU. Not a COLMAP port
// (PORTING_PLAN.md Phase 13). It is the "CPU twin" the GPU orchestrator is tested on: each
// kernel, picked by its entry point, reads and writes the same byte buffers with the same
// layouts, uniforms, constants and folded dispatch index as its WGSL, and computes through the
// production CPU code (PatchMatchCpu's kernel-level methods, PatchMatchPhotoConsistency,
// Mat.Rotate, PatchMatchKernel.RotateNormalMap) rather than a re-implementation, so the GPU
// orchestration run on it must reproduce PatchMatchCpu bit for bit.
//
// Every call is first handed to an inner RecordingComputeDevice, so this device enforces
// exactly the seam's rules (limits, alignment, layouts, writable aliasing, the
// write-after-dispatch hazard) and exposes the same command recording. Dispatches execute at
// the next flush or read, in recording order, which is when a GPU would run them; the
// write-after-dispatch rule makes that equivalent to running them at record time. A kernel
// that fails (a binding too small, a uniform the CPU code cannot honor) faults the flush's
// task, as a GPU error would. The kernels themselves are in ReferenceComputeDevice.Kernels.cs.
//
// Deliberately unlike a real device: a buffer created without initial data is filled with
// the garbage word GarbageWord instead of zeros, so a caller that relies on zero-filled
// buffers, or a kernel output that is never written (the filter's mask is written over the
// cost buffer), shows up as NaNs and wrong mask words instead of passing by luck. A buffer
// created with some initial data is zero after it, as the seam specifies.

using ColmapSharp.Compute;
using ColmapSharp.Compute.Testing;

namespace ColmapSharp.Mvs.Testing;

/// <summary>
/// A headless <see cref="IComputeDevice"/> that executes the GPU PatchMatch kernels on the CPU
/// through the production CPU PatchMatch code, over the same buffers the GPU would use.
/// </summary>
public sealed partial class ReferenceComputeDevice : IComputeDevice
{
	/// <summary>
	/// The word every buffer created without initial data starts filled with: a NaN as f32
	/// (payload 0x3ADBAD), and neither 0 nor 1 as a u32 mask entry.
	/// </summary>
	public const uint GarbageWord = 0x7FBADBADu;

	private readonly RecordingComputeDevice inner;
	private readonly Dictionary<RecordingComputeKernel, TwinKernel> kernels = new();
	private readonly List<DispatchCommand> pending = new();

	/// <summary>A device with the default WebGPU limits.</summary>
	public ReferenceComputeDevice()
		: this(ComputeDeviceLimits.Defaults)
	{
	}

	/// <summary>A device that reports and enforces <paramref name="limits"/>.</summary>
	/// <param name="limits">The limits; the offset alignment must be a positive power of two.</param>
	public ReferenceComputeDevice(ComputeDeviceLimits limits)
	{
		this.inner = new RecordingComputeDevice { Limits = limits };
	}

	/// <inheritdoc/>
	public ComputeDeviceLimits Limits => this.inner.Limits;

	/// <summary>
	/// What the device reports; when false, flushes and reads complete asynchronously (as in
	/// the browser) so a caller that blocks on them is caught.
	/// </summary>
	public bool SupportsBlockingWait
	{
		get => this.inner.SupportsBlockingWait;
		set => this.inner.SupportsBlockingWait = value;
	}

	/// <summary>Every call received, oldest first (the inner recording device's stream).</summary>
	public IReadOnlyList<ComputeCommand> Commands => this.inner.Commands;

	/// <summary>Dispatches executed so far, over the device's lifetime.</summary>
	public int ExecutedDispatchCount { get; private set; }

	/// <summary>
	/// A copy of <paramref name="buffer"/>'s current bytes, without flushing: only what has
	/// executed so far. A test's way to look inside a buffer the caller has not read back.
	/// </summary>
	/// <param name="buffer">A buffer of this device.</param>
	public byte[] GetContents(IComputeBuffer buffer) => this.Own(buffer).GetContents();

	/// <inheritdoc/>
	public IComputeBuffer CreateBuffer(ComputeBufferKind kind, long size, ReadOnlySpan<byte> initial = default, string? label = null)
	{
		var buffer = (RecordingComputeBuffer)this.inner.CreateBuffer(kind, size, initial, label);
		if (initial.IsEmpty)
		{
			System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(buffer.Memory).Fill(GarbageWord);
		}

		return buffer;
	}

	/// <inheritdoc/>
	public void WriteBuffer(IComputeBuffer buffer, long offset, ReadOnlySpan<byte> data) => this.inner.WriteBuffer(buffer, offset, data);

	/// <inheritdoc/>
	/// <exception cref="ArgumentException">Also when the entry point is not a GPU PatchMatch
	/// kernel, the declared bindings are not that kernel's, or the constants header lacks one of
	/// its constants: the CPU twin could not run it.</exception>
	public IComputeKernel CreateKernel(in ComputeKernelDescriptor descriptor)
	{
		// Parsed first: an unknown kernel is rejected before anything is recorded.
		TwinKernel twin = TwinKernel.Parse(descriptor);
		var kernel = (RecordingComputeKernel)this.inner.CreateKernel(descriptor);
		this.kernels.Add(kernel, twin);
		return kernel;
	}

	/// <inheritdoc/>
	public IComputeBindGroup CreateBindGroup(IComputeKernel kernel, int group, ReadOnlySpan<ComputeBufferBinding> entries)
		=> this.inner.CreateBindGroup(kernel, group, entries);

	/// <inheritdoc/>
	/// <exception cref="ArgumentOutOfRangeException">Also when <paramref name="x"/> exceeds the
	/// kernel's PM_GROUPS_X or <paramref name="z"/> exceeds 1: the folded index would give some
	/// elements to two invocations, a data race on a GPU.</exception>
	public void Dispatch(IComputeKernel kernel, ReadOnlySpan<IComputeBindGroup> groups, uint x, uint y = 1, uint z = 1)
	{
		if (kernel is RecordingComputeKernel recording && this.kernels.TryGetValue(recording, out TwinKernel? twin))
		{
			DispatchCoverage.Validate(twin.GroupsX, x, z);
		}

		this.inner.Dispatch(kernel, groups, x, y, z);
		this.pending.Add((DispatchCommand)this.inner.Commands[^1]);
	}

	/// <inheritdoc/>
	public ValueTask FlushAsync(CancellationToken cancellationToken = default)
	{
		this.ExecutePending();
		return this.inner.FlushAsync(cancellationToken);
	}

	/// <inheritdoc/>
	public ValueTask ReadBufferAsync(IComputeBuffer buffer, long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
	{
		// Validate the read before running anything, as the inner device does before submitting.
		RecordingComputeDevice.ValidateRange(this.Own(buffer), offset, destination.Length, "read");
		this.ExecutePending();
		return this.inner.ReadBufferAsync(buffer, offset, destination, cancellationToken);
	}

	/// <summary>
	/// Runs every pending dispatch in order. A failure stops the rest (a GPU would report the
	/// first error of the submission) and faults the submission's task, not this call.
	/// </summary>
	private void ExecutePending()
	{
		try
		{
			foreach (DispatchCommand dispatch in this.pending)
			{
				this.Execute(dispatch);
				this.ExecutedDispatchCount++;
			}
		}
		catch (Exception fault)
		{
			// Whatever the CPU code threw is what a GPU would have reported as an execution
			// error of this submission.
			this.inner.FaultNextSubmit(fault);
		}
		finally
		{
			this.pending.Clear();
		}
	}

	private RecordingComputeBuffer Own(IComputeBuffer buffer)
	{
		if (buffer is not RecordingComputeBuffer recording || recording.Device != this.inner)
		{
			throw new ArgumentException("The buffer was not created by this device.", nameof(buffer));
		}

		ObjectDisposedException.ThrowIf(recording.IsDisposed, recording);
		return recording;
	}
}
