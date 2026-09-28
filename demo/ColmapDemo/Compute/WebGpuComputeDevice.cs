/*
Copyright (c) 2026, Lars Brubaker
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#nullable enable

// WebGpuComputeDevice: colmap-sharp's GPU compute seam (ColmapSharp.Compute.IComputeDevice) implemented
// over agg-sharp's WebGPU device (MatterHackers.WebGpuRender.WebGpuRenderDevice, through the RenderCore
// IRenderDevice compute members). The handles it returns are in WebGpuComputeHandles.cs; the tests that
// pin it on a real GPU are WebGpuComputeDeviceTests.cs and PatchMatchGpuConformanceTests.cs.
//
// Copied from MatterCAD's Tests/ColmapGpuTests/WebGpuComputeDevice.cs (same author and license as above),
// where the real-GPU tests that pin it still live; only the namespace changed (and the explicit usings /
// #nullable, since the demo projects turn implicit usings and nullable off). Keep the two copies in step
// until MatterCAD takes the adapter from here or from agg-sharp.
//
// Every rule the seam states is checked here, before agg-sharp or wgpu sees the call, because WebGPU
// reports most broken rules only out of band and a readback then returns stale or zero bytes. The checks
// mirror ColmapSharp.Compute.Testing.RecordingComputeDevice, the seam's headless double, so a caller that
// passes against the double passes here.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using ColmapSharp.Compute;
using MatterHackers.RenderCore;
using MatterHackers.WebGpuRender;

namespace ColmapDemo.Compute
{
	/// <summary>
	/// An <see cref="IComputeDevice"/> over an agg-sharp <see cref="WebGpuRenderDevice"/>.
	/// <para>
	/// <b>One compute pass per dispatch.</b> Each <see cref="Dispatch"/> opens a compute pass, records the
	/// dispatch and ends the pass. The seam would allow one pass held open across dispatches, but a held
	/// pass makes every readback, submit and host call on the device throw until something closes it, and
	/// buys nothing: WebGPU orders dispatches across passes exactly as within one, and beginning a pass is
	/// a CPU-side encoder call.
	/// </para>
	/// <para>
	/// <b>Flush is a fence.</b> <see cref="FlushAsync"/> reads four bytes of a private buffer back through
	/// agg-sharp's <see cref="IRenderDevice.ReadBufferAsync"/>, which submits everything recorded and
	/// completes only once the GPU has run it, so the flush completes when the GPU has finished.
	/// </para>
	/// <para>
	/// <b>GPU errors.</b> wgpu reports validation and WGSL errors through the device's uncaptured-error
	/// callback rather than failing the call. This device collects them: one raised while a kernel is being
	/// created fails that call with <see cref="ArgumentException"/>; any other faults the next flush or read
	/// with <see cref="InvalidOperationException"/>, as does a lost device. The callback is per WebGPU device
	/// and cannot say which call caused an error, so the attribution is only right because this adapter
	/// always owns a dedicated device that nothing else records on.
	/// </para>
	/// <para>
	/// <b>A device of its own.</b> Created only through <see cref="Create"/> or <see cref="CreateAsync"/>,
	/// which open an offscreen WebGPU device that this owns and disposes. The seam's threading rule (no
	/// host work interleaved with compute recording) and the error attribution above both hold by
	/// construction that way.
	/// </para>
	/// </summary>
	public sealed class WebGpuComputeDevice : IComputeDevice, IDisposable
	{
		private const int FenceBytes = 4;

		private readonly WebGpuRenderDevice gpu;
		private readonly KernelSources kernelSources = new KernelSources();
		private readonly List<string> gpuErrors = new List<string>();
		private readonly IGpuBuffer fence;
		private readonly byte[] fenceDestination = new byte[FenceBytes];

		// Buffers bound by dispatches recorded since the last flush: the ones WriteBuffer must refuse.
		private readonly HashSet<WebGpuComputeBuffer> pendingBoundBuffers = new HashSet<WebGpuComputeBuffer>();
		private int kernelCount;
		private bool isDisposed;

		/// <summary>
		/// Takes ownership of <paramref name="device"/>. If anything here throws, the device is disposed
		/// before the exception leaves, so a failed construction leaks nothing.
		/// </summary>
		private WebGpuComputeDevice(WebGpuRenderDevice device, bool supportsBlockingWait)
		{
			this.gpu = device;
			this.SupportsBlockingWait = supportsBlockingWait;
			try
			{
				this.Limits = MapLimits(device.Limits);
				this.gpu.RegisterShaderSources(this.kernelSources);
				this.gpu.UncapturedError += this.OnUncapturedError;
				this.fence = this.gpu.CreateBuffer(BufferUsage.Storage | BufferUsage.CopySrc, FenceBytes);
			}
			catch
			{
				// The registered provider goes with the device; detaching the handler first keeps a
				// disposal-time error from landing on a half-built object.
				this.gpu.UncapturedError -= this.OnUncapturedError;
				this.gpu.Dispose();
				throw;
			}
		}

		/// <inheritdoc/>
		public ComputeDeviceLimits Limits { get; }

		/// <inheritdoc/>
		public bool SupportsBlockingWait { get; }

		/// <summary>The agg-sharp device underneath, for diagnostics (adapter name, backend, limits).</summary>
		public WebGpuRenderDevice RenderDevice => this.gpu;

		/// <summary>
		/// Opens a dedicated offscreen WebGPU device, synchronously, on the desktop. In the browser use
		/// <see cref="CreateAsync"/>.
		/// </summary>
		/// <param name="raiseComputeLimits">True to ask the adapter for its maximum buffer, storage-binding,
		/// storage-buffer-count and workgroup-size limits instead of the WebGPU defaults, which is what
		/// PatchMatch on full-size photos needs. What was granted is in <see cref="Limits"/>.</param>
		/// <param name="label">Optional debug name carried into wgpu's messages.</param>
		/// <exception cref="InvalidOperationException">No WebGPU adapter or device could be opened.</exception>
		/// <exception cref="PlatformNotSupportedException">Called in the browser.</exception>
		public static WebGpuComputeDevice Create(bool raiseComputeLimits, string? label = null)
		{
			var device = new WebGpuRenderDevice(label: label ?? "colmap compute", raiseComputeLimits: raiseComputeLimits);
			return new WebGpuComputeDevice(device, supportsBlockingWait: true);
		}

		/// <summary>
		/// Opens a dedicated offscreen WebGPU device anywhere, the browser included, where the adapter and
		/// device requests are Promises that settle only once control returns to the JS event loop. On the
		/// desktop the task is already complete when it is returned.
		/// </summary>
		/// <param name="raiseComputeLimits">As for <see cref="Create"/>. A browser that refuses the raised
		/// limits yields a device at the defaults.</param>
		/// <param name="isBrowser">Whether this runs in the browser, where a thread must never block on a GPU
		/// wait (<see cref="SupportsBlockingWait"/> is its negation). Passed in rather than read here, so a
		/// desktop test can build the browser answer.</param>
		/// <param name="label">Optional debug name carried into wgpu's messages.</param>
		/// <exception cref="InvalidOperationException">No WebGPU adapter or device could be opened.</exception>
		public static async ValueTask<WebGpuComputeDevice> CreateAsync(bool raiseComputeLimits, Func<bool> isBrowser, string? label = null)
		{
			ArgumentNullException.ThrowIfNull(isBrowser);
			bool supportsBlockingWait = !isBrowser();
			var device = await WebGpuRenderDevice.CreateAsync(null, label ?? "colmap compute", raiseComputeLimits);
			return new WebGpuComputeDevice(device, supportsBlockingWait);
		}

		/// <summary>
		/// The seam's limits from agg-sharp's. Four limits agg-sharp does not read from the device
		/// (<c>maxBindGroups</c>, <c>maxUniformBuffersPerShaderStage</c>, <c>maxBindingsPerBindGroup</c>,
		/// <c>maxComputeWorkgroupStorageSize</c>) keep their WebGPU defaults: agg-sharp never raises them (it
		/// requests the workgroup storage size as undefined, i.e. the 16 KiB default), and every device grants
		/// at least the defaults.
		/// The binding offset alignment is the larger of the storage and uniform ones, because the seam
		/// applies one alignment to both kinds of binding.
		/// </summary>
		/// <param name="limits">The agg-sharp device's limits.</param>
		/// <returns>The limits this device enforces.</returns>
		public static ComputeDeviceLimits MapLimits(DeviceLimits limits)
			=> new ComputeDeviceLimits(
				MaxBufferSize: ToLong(limits.MaxBufferSize),
				MaxStorageBufferBindingSize: ToLong(limits.MaxStorageBufferBindingSize),
				MaxStorageBuffersPerShaderStage: ToInt(limits.MaxStorageBuffersPerShaderStage),
				MaxUniformBufferBindingSize: ToLong(limits.MaxUniformBufferBindingSize),
				MaxComputeWorkgroupsPerDimension: limits.MaxComputeWorkgroupsPerDimension,
				MaxComputeInvocationsPerWorkgroup: ToInt(limits.MaxComputeInvocationsPerWorkgroup),
				MinStorageBufferOffsetAlignment: ToInt(Math.Max(limits.MinStorageBufferOffsetAlignment, limits.MinUniformBufferOffsetAlignment)));

		/// <inheritdoc/>
		public IComputeBuffer CreateBuffer(ComputeBufferKind kind, long size, ReadOnlySpan<byte> initial = default, string? label = null)
		{
			this.ThrowIfDisposed();
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

			// Every buffer can be written and read back, whatever it is bound as (the seam's buffer rule).
			var usage = (kind == ComputeBufferKind.Uniform ? BufferUsage.Uniform : BufferUsage.Storage)
				| BufferUsage.CopySrc | BufferUsage.CopyDst;
			var buffer = this.gpu.CreateBuffer(usage, (ulong)size, initial);
			return new WebGpuComputeBuffer(this, buffer, kind, size, label);
		}

		/// <inheritdoc/>
		public void WriteBuffer(IComputeBuffer buffer, long offset, ReadOnlySpan<byte> data)
		{
			this.ThrowIfDisposed();
			var target = this.Own(buffer, nameof(buffer));
			ValidateRange(target, offset, data.Length, "write");
			if (this.pendingBoundBuffers.Contains(target))
			{
				throw new InvalidOperationException(
					$"WriteBuffer to {target}, which a dispatch recorded since the last flush binds. A write is a queue"
					+ " write that lands before the flush runs any dispatch, so that dispatch would read the new bytes;"
					+ " give per-dispatch data its own buffer, or flush before writing.");
			}

			this.gpu.WriteBuffer(target.GpuBuffer, (ulong)offset, data);
		}

		/// <inheritdoc/>
		public IComputeKernel CreateKernel(in ComputeKernelDescriptor descriptor)
		{
			this.ThrowIfDisposed();
			var layouts = this.ValidateKernel(descriptor);
			var layoutEntries = descriptor.Bindings
				.Select(b => new BindGroupLayoutEntry((uint)b.Group, (uint)b.Binding, ShaderStage.Compute, ToAgg(b.Type)))
				.ToArray();

			// agg-sharp compiles shaders by key through its registered providers; each kernel gets a key of its
			// own, dropped again once the module exists.
			var key = $"colmap.compute.{this.kernelCount++}.{descriptor.Label}";
			this.kernelSources.Add(key, descriptor.Source);
			int errorsBefore = this.gpuErrors.Count;
			IComputePipeline pipeline;
			try
			{
				using var module = this.gpu.CreateShaderModule(key);
				pipeline = this.gpu.CreateComputePipeline(new ComputePipelineDescriptor(module, descriptor.EntryPoint, layoutEntries, descriptor.Label));
			}
			finally
			{
				this.kernelSources.Remove(key);
			}

			// wgpu-native reports a WGSL or pipeline error through the uncaptured-error callback, during the
			// create call, and hands back a handle that is invalid. Report it here, where the caller can still
			// tell which kernel it was, rather than as a fault of some later flush.
			if (this.gpuErrors.Count > errorsBefore)
			{
				var messages = string.Join(Environment.NewLine, this.gpuErrors.Skip(errorsBefore));
				this.gpuErrors.RemoveRange(errorsBefore, this.gpuErrors.Count - errorsBefore);
				pipeline.Dispose();
				throw new ArgumentException($"Kernel '{descriptor.Label}' does not compile: {messages}", nameof(descriptor));
			}

			return new WebGpuComputeKernel(this, pipeline, descriptor.Label, layouts);
		}

		/// <inheritdoc/>
		public IComputeBindGroup CreateBindGroup(IComputeKernel kernel, int group, ReadOnlySpan<ComputeBufferBinding> entries)
		{
			this.ThrowIfDisposed();
			var owner = this.Own(kernel, nameof(kernel));
			if (group < 0 || group >= owner.GroupLayouts.Length)
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
			var aggEntries = new BindGroupEntry[sorted.Length];
			for (int i = 0; i < sorted.Length; i++)
			{
				var entry = sorted[i];
				if (entry.Binding != layout[i].Binding)
				{
					throw new ArgumentException(
						$"{owner} group {group} has no binding {entry.Binding}, or it was given twice; the layout declares"
						+ $" bindings {string.Join(", ", layout.Select(b => b.Binding))}.",
						nameof(entries));
				}

				var buffer = this.ValidateEntry(owner, group, entry, layout[i].Type);
				aggEntries[i] = BindGroupEntry.ForBuffer((uint)entry.Binding, buffer.GpuBuffer, (ulong)entry.Offset, (ulong)entry.Size);
			}

			var bindGroup = this.gpu.CreateBindGroup(new BindGroupDescriptor(owner.Pipeline, (uint)group, aggEntries, $"{owner.Label} group {group}"));
			return new WebGpuComputeBindGroup(this, bindGroup, group, layout, sorted);
		}

		/// <inheritdoc/>
		public void Dispatch(IComputeKernel kernel, ReadOnlySpan<IComputeBindGroup> groups, uint x, uint y = 1, uint z = 1)
		{
			this.ThrowIfDisposed();
			var owner = this.Own(kernel, nameof(kernel));
			if (groups.Length != owner.GroupLayouts.Length)
			{
				throw new ArgumentException(
					$"{owner} declares {owner.GroupLayouts.Length} bind groups, but {groups.Length} were given.", nameof(groups));
			}

			var bound = new WebGpuComputeBindGroup[groups.Length];
			for (int i = 0; i < groups.Length; i++)
			{
				var bindGroup = this.Own(groups[i], nameof(groups));
				if (bindGroup.Group != i || !bindGroup.Layout.SequenceEqual(owner.GroupLayouts[i]))
				{
					throw new ArgumentException(
						$"The bind group at index {i} ({bindGroup}) does not match {owner}'s layout for group {i}.", nameof(groups));
				}

				foreach (var entry in bindGroup.Entries)
				{
					var buffer = (WebGpuComputeBuffer)entry.Buffer;
					ObjectDisposedException.ThrowIf(buffer.IsDisposed, buffer);
				}

				bound[i] = bindGroup;
			}

			RejectWritableAliasing(owner, bound);
			ValidateWorkgroups(this.Limits, x, nameof(x));
			ValidateWorkgroups(this.Limits, y, nameof(y));
			ValidateWorkgroups(this.Limits, z, nameof(z));

			using (var pass = this.gpu.BeginComputePass(owner.Label))
			{
				pass.SetPipeline(owner.Pipeline);
				for (int i = 0; i < bound.Length; i++)
				{
					pass.SetBindGroup(i, bound[i].GpuBindGroup);
				}

				pass.Dispatch(x, y, z);
			}

			foreach (var bindGroup in bound)
			{
				foreach (var entry in bindGroup.Entries)
				{
					this.pendingBoundBuffers.Add((WebGpuComputeBuffer)entry.Buffer);
				}
			}
		}

		/// <inheritdoc/>
		public ValueTask FlushAsync(CancellationToken cancellationToken = default)
		{
			this.ThrowIfDisposed();
			if (cancellationToken.IsCancellationRequested)
			{
				return this.SubmitCanceled(cancellationToken);
			}

			this.pendingBoundBuffers.Clear();
			return this.SubmitAndWaitAsync(this.fence, 0, this.fenceDestination);
		}

		/// <inheritdoc/>
		public ValueTask ReadBufferAsync(IComputeBuffer buffer, long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
		{
			this.ThrowIfDisposed();
			var source = this.Own(buffer, nameof(buffer));
			ValidateRange(source, offset, destination.Length, "read");
			if (cancellationToken.IsCancellationRequested)
			{
				return this.SubmitCanceled(cancellationToken);
			}

			this.pendingBoundBuffers.Clear();

			// agg-sharp's empty read submits without waiting; the fence waits.
			return destination.Length == 0
				? this.SubmitAndWaitAsync(this.fence, 0, this.fenceDestination)
				: this.SubmitAndWaitAsync(source.GpuBuffer, offset, destination);
		}

		/// <summary>
		/// Releases the fence and the WebGPU device (which waits for submitted work).
		/// Handles created from this device must not be used afterwards.
		/// </summary>
		public void Dispose()
		{
			if (this.isDisposed)
			{
				return;
			}

			this.isDisposed = true;
			this.gpu.UncapturedError -= this.OnUncapturedError;
			this.fence.Dispose();
			this.gpu.Dispose();
		}

		// Everything recorded still goes to the GPU (so nothing is left pending), but the call does not wait
		// for it. A GPU error already known wins over the cancellation, as in the recording double.
		private ValueTask SubmitCanceled(CancellationToken cancellationToken)
		{
			this.pendingBoundBuffers.Clear();
			try
			{
				this.gpu.Submit();
				this.ThrowIfGpuFaulted();
			}
			catch (Exception fault)
			{
				return ValueTask.FromException(fault);
			}

			return ValueTask.FromCanceled(cancellationToken);
		}

		// Async so that a map failure or a GPU error faults the returned task instead of throwing from the
		// call, as the seam's error rule requires; the argument checks have all run before this.
		private async ValueTask SubmitAndWaitAsync(IGpuBuffer source, long offset, Memory<byte> destination)
		{
			await this.gpu.ReadBufferAsync(source, (ulong)offset, destination);
			this.ThrowIfGpuFaulted();
		}

		private void ThrowIfGpuFaulted()
		{
			if (this.gpu.IsDeviceLost)
			{
				throw new InvalidOperationException($"The GPU device was lost: {this.gpu.DeviceLostMessage}");
			}

			if (this.gpuErrors.Count > 0)
			{
				var messages = string.Join(Environment.NewLine, this.gpuErrors);
				this.gpuErrors.Clear();
				throw new InvalidOperationException($"The GPU reported an error running compute work: {messages}");
			}
		}

		private void OnUncapturedError(object? sender, string message) => this.gpuErrors.Add(message);

		private IReadOnlyList<ComputeKernelBinding>[] ValidateKernel(in ComputeKernelDescriptor descriptor)
		{
			RequireText(descriptor.Label, "label");
			RequireText(descriptor.Source, "WGSL source");
			RequireText(descriptor.EntryPoint, "entry point");
			if (descriptor.Bindings == null)
			{
				throw new ArgumentException($"Kernel '{descriptor.Label}' has no bindings list.", nameof(descriptor));
			}

			int groupCount = 0;
			int storageCount = 0;
			int uniformCount = 0;
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
			for (int group = 0; group < groupCount; group++)
			{
				layouts[group] = descriptor.Bindings.Where(b => b.Group == group).OrderBy(b => b.Binding).ToArray();
				if (layouts[group].Count == 0)
				{
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

			return layouts;
		}

		private WebGpuComputeBuffer ValidateEntry(WebGpuComputeKernel kernel, int group, in ComputeBufferBinding entry, ComputeBindingType type)
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
				throw new ArgumentException($"{where}: {entry.Size} bytes at offset {entry.Offset} run past the end of {buffer}.", "entries");
			}

			var limit = type == ComputeBindingType.Uniform ? this.Limits.MaxUniformBufferBindingSize : this.Limits.MaxStorageBufferBindingSize;
			if (entry.Size > limit)
			{
				throw new ArgumentException(
					$"{where}: a {entry.Size:N0} byte {type} binding exceeds this device's limit of {limit:N0} bytes.", "entries");
			}

			return buffer;
		}

		// WebGPU tracks usage per whole buffer per dispatch, so a buffer bound read_write may not be bound
		// anywhere else in the same dispatch, even over a disjoint range.
		private static void RejectWritableAliasing(WebGpuComputeKernel kernel, WebGpuComputeBindGroup[] bound)
		{
			var bindingCounts = new Dictionary<WebGpuComputeBuffer, int>();
			var writable = new List<WebGpuComputeBuffer>();
			for (int i = 0; i < bound.Length; i++)
			{
				var layout = kernel.GroupLayouts[i];
				var entries = bound[i].Entries;
				for (int e = 0; e < entries.Length; e++)
				{
					var buffer = (WebGpuComputeBuffer)entries[e].Buffer;
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

		private static void ValidateRange(WebGpuComputeBuffer buffer, long offset, int length, string verb)
		{
			if (offset < 0 || offset % 4 != 0 || length % 4 != 0)
			{
				throw new ArgumentException($"A buffer {verb}'s offset ({offset}) and length ({length}) must be multiples of 4.", nameof(offset));
			}

			if (offset > buffer.Size || length > buffer.Size - offset)
			{
				throw new ArgumentException($"A {verb} of {length} bytes at offset {offset} runs past the end of {buffer}.", nameof(offset));
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

		private static BindingType ToAgg(ComputeBindingType type) => type switch
		{
			ComputeBindingType.Uniform => BindingType.UniformBuffer,
			ComputeBindingType.ReadOnlyStorage => BindingType.ReadOnlyStorageBuffer,
			_ => BindingType.StorageBuffer,
		};

		private static long ToLong(ulong value) => value > long.MaxValue ? long.MaxValue : (long)value;

		private static int ToInt(uint value) => value > int.MaxValue ? int.MaxValue : (int)value;

		private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(this.isDisposed, this);

		private WebGpuComputeBuffer Own(IComputeBuffer? handle, string parameterName)
		{
			if (handle is not WebGpuComputeBuffer buffer || buffer.Device != this)
			{
				throw new ArgumentException("The buffer was not created by this device.", parameterName);
			}

			ObjectDisposedException.ThrowIf(buffer.IsDisposed, buffer);
			return buffer;
		}

		private WebGpuComputeKernel Own(IComputeKernel? handle, string parameterName)
		{
			if (handle is not WebGpuComputeKernel kernel || kernel.Device != this)
			{
				throw new ArgumentException("The kernel was not created by this device.", parameterName);
			}

			ObjectDisposedException.ThrowIf(kernel.IsDisposed, kernel);
			return kernel;
		}

		private WebGpuComputeBindGroup Own(IComputeBindGroup? handle, string parameterName)
		{
			if (handle is not WebGpuComputeBindGroup bindGroup || bindGroup.Device != this)
			{
				throw new ArgumentException("The bind group was not created by this device.", parameterName);
			}

			ObjectDisposedException.ThrowIf(bindGroup.IsDisposed, bindGroup);
			return bindGroup;
		}

		/// <summary>The WGSL of kernels being created, by the key their shader module is compiled under.</summary>
		private sealed class KernelSources : IShaderSourceProvider
		{
			private readonly Dictionary<string, string> sources = new Dictionary<string, string>(StringComparer.Ordinal);

			public void Add(string key, string source) => this.sources.Add(key, source);

			public void Remove(string key) => this.sources.Remove(key);

			public string? TryGetSource(string sourceKey) => this.sources.GetValueOrDefault(sourceKey);
		}
	}
}
