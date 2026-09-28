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

// The handle types WebGpuComputeDevice (WebGpuComputeDevice.cs) hands out for colmap-sharp's compute
// seam: each wraps the agg-sharp RenderCore object it stands for and remembers what the seam's rules need
// that the agg object does not carry (the owning device, the buffer kind, the kernel's per-group layout,
// the bind group's entries). Copied from MatterCAD's Tests/ColmapGpuTests/WebGpuComputeHandles.cs with
// the namespace changed; see WebGpuComputeDevice.cs.

using System;
using System.Collections.Generic;
using ColmapSharp.Compute;
using MatterHackers.RenderCore;

namespace ColmapDemo.Compute
{
	/// <summary>An <see cref="IComputeBuffer"/> over an agg-sharp storage or uniform buffer.</summary>
	internal sealed class WebGpuComputeBuffer : IComputeBuffer
	{
		internal WebGpuComputeBuffer(WebGpuComputeDevice device, IGpuBuffer buffer, ComputeBufferKind kind, long size, string? label)
		{
			this.Device = device;
			this.GpuBuffer = buffer;
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

		internal WebGpuComputeDevice Device { get; }

		internal IGpuBuffer GpuBuffer { get; }

		internal bool IsDisposed { get; private set; }

		/// <summary>
		/// Releases the caller's reference (<c>wgpuBufferRelease</c>, not destroy): a command buffer or bind
		/// group that holds the buffer keeps it alive until its work has run, as the seam promises.
		/// </summary>
		public void Dispose()
		{
			if (!this.IsDisposed)
			{
				this.IsDisposed = true;
				this.GpuBuffer.Dispose();
			}
		}

		/// <inheritdoc/>
		public override string ToString() => $"buffer '{this.Label ?? "unlabeled"}' ({this.Kind}, {this.Size} bytes)";
	}

	/// <summary>An <see cref="IComputeKernel"/> over an agg-sharp compute pipeline.</summary>
	internal sealed class WebGpuComputeKernel : IComputeKernel
	{
		internal WebGpuComputeKernel(WebGpuComputeDevice device, IComputePipeline pipeline, string label, IReadOnlyList<ComputeKernelBinding>[] groupLayouts)
		{
			this.Device = device;
			this.Pipeline = pipeline;
			this.Label = label;
			this.GroupLayouts = groupLayouts;
		}

		/// <inheritdoc/>
		public string Label { get; }

		internal WebGpuComputeDevice Device { get; }

		internal IComputePipeline Pipeline { get; }

		/// <summary>Each group's bindings, sorted by binding index.</summary>
		internal IReadOnlyList<ComputeKernelBinding>[] GroupLayouts { get; }

		internal bool IsDisposed { get; private set; }

		/// <summary>Releases the pipeline; dispatches already recorded with it keep it alive until they run.</summary>
		public void Dispose()
		{
			if (!this.IsDisposed)
			{
				this.IsDisposed = true;
				this.Pipeline.Dispose();
			}
		}

		/// <inheritdoc/>
		public override string ToString() => $"kernel '{this.Label}'";
	}

	/// <summary>An <see cref="IComputeBindGroup"/> over an agg-sharp bind group.</summary>
	internal sealed class WebGpuComputeBindGroup : IComputeBindGroup
	{
		internal WebGpuComputeBindGroup(WebGpuComputeDevice device, IBindGroup bindGroup, int group, IReadOnlyList<ComputeKernelBinding> layout, ComputeBufferBinding[] entries)
		{
			this.Device = device;
			this.GpuBindGroup = bindGroup;
			this.Group = group;
			this.Layout = layout;
			this.Entries = entries;
		}

		/// <inheritdoc/>
		public int Group { get; }

		internal WebGpuComputeDevice Device { get; }

		internal IBindGroup GpuBindGroup { get; }

		/// <summary>The layout the group was made for, sorted by binding; compared by value at dispatch.</summary>
		internal IReadOnlyList<ComputeKernelBinding> Layout { get; }

		/// <summary>The entries, sorted by binding so entry i matches <see cref="Layout"/>[i].</summary>
		internal ComputeBufferBinding[] Entries { get; }

		internal bool IsDisposed { get; private set; }

		/// <summary>Releases the bind group; dispatches already recorded with it keep it alive until they run.</summary>
		public void Dispose()
		{
			if (!this.IsDisposed)
			{
				this.IsDisposed = true;
				this.GpuBindGroup.Dispose();
			}
		}

		/// <inheritdoc/>
		public override string ToString() => $"bind group {this.Group}";
	}
}
