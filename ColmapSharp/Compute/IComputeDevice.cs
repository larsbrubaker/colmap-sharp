// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// IComputeDevice: the GPU compute seam PatchMatch stereo runs through when a host supplies a
// device (PORTING_PLAN.md Phase 13). Not a COLMAP port: COLMAP's GPU path is CUDA, which this
// library cannot use. colmap-sharp stays pure managed and never references a graphics
// library, so it states the small slice of WebGPU compute it needs here and the host (for
// MatterCAD, an adapter over agg-sharp's IRenderDevice) implements it. The value types are in
// ComputeDescriptors.cs and ComputeDeviceLimits.cs; Testing/RecordingComputeDevice.cs is the
// headless double that enforces every rule stated below.

namespace ColmapSharp.Compute;

/// <summary>
/// A WebGPU-shaped compute device: storage and uniform buffers, WGSL kernels, bind groups,
/// recorded dispatches, and asynchronous flush and readback. Implemented by the host.
/// <para>
/// <b>Recording and ordering.</b> <see cref="Dispatch"/> only records. Recorded dispatches reach
/// the GPU at the next <see cref="FlushAsync"/> (or <see cref="ReadBufferAsync"/>, which flushes
/// first) and run in recording order: every dispatch sees the storage writes of every dispatch
/// recorded before it, with no barrier to spell out. Workgroups within one dispatch run in any
/// order.
/// </para>
/// <para>
/// <b>Writes are not ordered against dispatches.</b> <see cref="WriteBuffer"/> is a queue write
/// (<c>wgpuQueueWriteBuffer</c>): it lands before the next flush executes <i>any</i> of its
/// dispatches, including dispatches recorded before the write. So never write a buffer that a
/// dispatch recorded since the last flush binds; every such dispatch would see the new bytes.
/// Per-dispatch parameters get their own buffer (created with initial data) and bind group. The
/// device throws on this hazard rather than letting it corrupt results silently.
/// </para>
/// <para>
/// <b>Writable bindings are exclusive.</b> WebGPU checks buffer usage per whole buffer for the
/// whole dispatch, so a buffer bound as <see cref="ComputeBindingType.Storage"/> (read_write) in
/// any of a dispatch's groups may not be bound again anywhere in that dispatch: not by another
/// binding in any group, not as read-only, not over a disjoint range. <see cref="Dispatch"/>
/// throws when it is. Read-only and uniform bindings may share a buffer freely.
/// </para>
/// <para>
/// <b>Buffers.</b> Both buffer kinds can be written and read back: an adapter creates every
/// buffer with <c>CopySrc | CopyDst</c> usage in addition to <c>Storage</c> or <c>Uniform</c>.
/// </para>
/// <para>
/// <b>Alignment.</b> Buffer sizes, write and read offsets and lengths, and binding sizes are
/// multiples of 4. Binding offsets are multiples of
/// <see cref="ComputeDeviceLimits.MinStorageBufferOffsetAlignment"/>.
/// </para>
/// <para>
/// <b>Limits.</b> Every call that would cross one of <see cref="Limits"/> throws instead of
/// failing on the GPU, where WebGPU would only report it out of band and a readback would
/// return stale or zero bytes.
/// </para>
/// <para>
/// <b>Errors.</b> Argument and state errors (a broken rule above, a disposed or foreign handle)
/// throw synchronously from the call. GPU execution and map failures - including a WGSL compile
/// or pipeline error the device only learns of asynchronously - fault the task returned by
/// <see cref="FlushAsync"/> or <see cref="ReadBufferAsync"/>.
/// </para>
/// <para>
/// <b>Async.</b> <see cref="FlushAsync"/> and <see cref="ReadBufferAsync"/> complete when the GPU
/// has finished. On a desktop device they may complete synchronously; in the browser they
/// genuinely wait for the JS event loop, which a blocked thread would never return to. So when
/// <see cref="SupportsBlockingWait"/> is false a caller must <c>await</c> them and never block on
/// them. Cancellation is observed only on entry, before anything is submitted: a call whose
/// token is already canceled still submits every pending write and dispatch, then completes as
/// canceled (a canceled read copies nothing into its destination); a token canceled after the
/// call has submitted does not cut the wait short. So after any flush or read completes -
/// successfully, canceled or faulted - nothing is pending and the device is reusable.
/// </para>
/// <para>
/// <b>Threading.</b> Not thread-safe, and not safe against the host's own use of the underlying
/// GPU device either. Every call must be serialized with any other use of that device: one
/// caller at a time, and the host must not interleave its own rendering or other work on the
/// same device while compute work is being recorded (between the first <see cref="Dispatch"/>
/// and the flush or read that submits it). A host should give compute its own device.
/// Calls are never concurrent, but they may arrive from different threads one after another:
/// callers await <see cref="FlushAsync"/> and <see cref="ReadBufferAsync"/> with
/// <c>ConfigureAwait(false)</c>, so the calls after an <c>await</c> can come from whichever
/// thread-pool thread completed it. So an implementation must not assume thread affinity: a
/// host whose GPU API only accepts calls from one thread marshals each call to that thread
/// itself.
/// </para>
/// <para>
/// <b>Ownership.</b> The host owns the device and disposes it; callers dispose only the handles
/// they created. Disposing a handle that recorded or in-flight work binds is allowed: WebGPU
/// releases are reference counted, so that work keeps it alive until it has run. A disposed
/// handle cannot be used in any later call.
/// </para>
/// </summary>
public interface IComputeDevice
{
	/// <summary>
	/// The limits this device enforces. Fixed when the device is created and constant for its
	/// lifetime; <see cref="ComputeDeviceLimits.MinStorageBufferOffsetAlignment"/> is a positive
	/// power of two.
	/// </summary>
	ComputeDeviceLimits Limits { get; }

	/// <summary>
	/// True when a caller may block a thread until <see cref="FlushAsync"/> or
	/// <see cref="ReadBufferAsync"/> completes (a desktop device), on ANY thread - including a
	/// thread the host marshals device calls to, since a synchronous caller may be running on
	/// it. A host that marshals device calls to a thread it may block (a UI or render thread
	/// whose loop must run for the work to complete) must report false, as must the browser,
	/// where blocking would deadlock. False sends callers to their async entry points.
	/// </summary>
	bool SupportsBlockingWait { get; }

	/// <summary>Creates a buffer, filled with <paramref name="initial"/> and zero after it.</summary>
	/// <param name="kind">What the buffer can be bound as.</param>
	/// <param name="size">Size in bytes: a positive multiple of 4, at most
	/// <see cref="ComputeDeviceLimits.MaxBufferSize"/>.</param>
	/// <param name="initial">Initial contents, no longer than <paramref name="size"/>. The way to
	/// give a buffer data that no recorded dispatch can race.</param>
	/// <param name="label">Optional debug name.</param>
	/// <exception cref="ArgumentException">The size is not a positive multiple of 4, or the
	/// initial data is longer than the buffer.</exception>
	/// <exception cref="InvalidOperationException">The size exceeds the device's limit.</exception>
	IComputeBuffer CreateBuffer(ComputeBufferKind kind, long size, ReadOnlySpan<byte> initial = default, string? label = null);

	/// <summary>
	/// Writes bytes into a buffer as a queue write: it lands before the next flush's dispatches
	/// run. See the ordering rule on <see cref="IComputeDevice"/>.
	/// </summary>
	/// <param name="buffer">Destination buffer.</param>
	/// <param name="offset">Byte offset, a multiple of 4.</param>
	/// <param name="data">Bytes to write; the length is a multiple of 4 and fits in the buffer.</param>
	/// <exception cref="ArgumentException">The range is misaligned or runs past the buffer.</exception>
	/// <exception cref="InvalidOperationException">A dispatch recorded since the last flush binds
	/// <paramref name="buffer"/>.</exception>
	/// <exception cref="ObjectDisposedException">The buffer has been disposed.</exception>
	void WriteBuffer(IComputeBuffer buffer, long offset, ReadOnlySpan<byte> data);

	/// <summary>Compiles a kernel from WGSL with an explicit bind group layout.</summary>
	/// <remarks>
	/// A WGSL compile or pipeline creation error may throw from this call, or - where the device
	/// reports it asynchronously, as browser WebGPU does - fault the task of a later
	/// <see cref="FlushAsync"/> or <see cref="ReadBufferAsync"/> instead. A caller handles both.
	/// </remarks>
	/// <param name="descriptor">Source, entry point and bindings. Groups run 0, 1, ... with no gap
	/// and number at most <see cref="ComputeDeviceLimits.MaxBindGroups"/>; no (group, binding)
	/// repeats; binding indices are below <see cref="ComputeDeviceLimits.MaxBindingsPerBindGroup"/>;
	/// at most <see cref="ComputeDeviceLimits.MaxStorageBuffersPerShaderStage"/> storage and
	/// <see cref="ComputeDeviceLimits.MaxUniformBuffersPerShaderStage"/> uniform bindings.</param>
	/// <exception cref="ArgumentException">The descriptor breaks one of those rules, or the device
	/// reports synchronously that the WGSL does not compile.</exception>
	IComputeKernel CreateKernel(in ComputeKernelDescriptor descriptor);

	/// <summary>
	/// Creates a bind group for group <paramref name="group"/> of a kernel's layout. It supplies
	/// exactly the bindings the kernel declares for that group, each from a buffer of the matching
	/// kind, and may be used with any kernel whose layout for that group is identical.
	/// </summary>
	/// <param name="kernel">The kernel whose layout the group follows.</param>
	/// <param name="group">The group index.</param>
	/// <param name="entries">One entry per declared binding, in any order.</param>
	/// <exception cref="ArgumentException">The entries do not match the layout, a range is
	/// misaligned, out of bounds, or larger than its binding-size limit.</exception>
	/// <exception cref="ObjectDisposedException">The kernel or a buffer has been disposed.</exception>
	IComputeBindGroup CreateBindGroup(IComputeKernel kernel, int group, ReadOnlySpan<ComputeBufferBinding> entries);

	/// <summary>
	/// Records a dispatch of <paramref name="x"/> × <paramref name="y"/> × <paramref name="z"/>
	/// workgroups (not invocations; the WGSL's <c>@workgroup_size</c> multiplies them). Runs at the
	/// next flush, after every dispatch recorded before it.
	/// </summary>
	/// <param name="kernel">The kernel to run.</param>
	/// <param name="groups">The bind groups, <c>groups[i]</c> bound at group <c>i</c>; one per group
	/// the kernel declares, each with a layout identical to the kernel's for that group.</param>
	/// <param name="x">Workgroups along X.</param>
	/// <param name="y">Workgroups along Y.</param>
	/// <param name="z">Workgroups along Z.</param>
	/// <exception cref="ArgumentException">The groups do not match the kernel's layout, or a buffer
	/// bound read_write is bound again in the same dispatch.</exception>
	/// <exception cref="ArgumentOutOfRangeException">A count exceeds
	/// <see cref="ComputeDeviceLimits.MaxComputeWorkgroupsPerDimension"/>.</exception>
	/// <exception cref="ObjectDisposedException">The kernel, a bind group, or a buffer one of them
	/// binds has been disposed.</exception>
	void Dispatch(IComputeKernel kernel, ReadOnlySpan<IComputeBindGroup> groups, uint x, uint y = 1, uint z = 1);

	/// <summary>
	/// Submits every pending write and recorded dispatch, and completes when the GPU has finished
	/// them. Must be awaited, not blocked on, when <see cref="SupportsBlockingWait"/> is false.
	/// </summary>
	/// <param name="cancellationToken">Checked on entry only; see the async rule on <see cref="IComputeDevice"/>.</param>
	/// <returns>A task that is canceled if the token was canceled on entry (the work is still
	/// submitted), and faults if the GPU reports an execution or pipeline error.</returns>
	ValueTask FlushAsync(CancellationToken cancellationToken = default);

	/// <summary>
	/// Flushes everything recorded (as <see cref="FlushAsync"/>), then copies
	/// <c>destination.Length</c> bytes of <paramref name="buffer"/> starting at
	/// <paramref name="offset"/> into <paramref name="destination"/>.
	/// </summary>
	/// <param name="buffer">The buffer to read, of either kind.</param>
	/// <param name="offset">Byte offset, a multiple of 4.</param>
	/// <param name="destination">Where the bytes go; its length, a multiple of 4, is how many are read.</param>
	/// <param name="cancellationToken">Checked on entry only; see the async rule on <see cref="IComputeDevice"/>.</param>
	/// <returns>A task that completes once <paramref name="destination"/> holds the bytes; nothing
	/// is written to it before then. Canceled (with the pending work still submitted and nothing
	/// copied) if the token was canceled on entry; faulted on a GPU execution or map failure.</returns>
	/// <exception cref="ArgumentException">The range is misaligned or runs past the buffer.</exception>
	/// <exception cref="ObjectDisposedException">The buffer has been disposed.</exception>
	ValueTask ReadBufferAsync(IComputeBuffer buffer, long offset, Memory<byte> destination, CancellationToken cancellationToken = default);
}
