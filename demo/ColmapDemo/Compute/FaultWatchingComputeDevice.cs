// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// FaultWatchingComputeDevice: passes every call through to the host's GPU and remembers the first
// one that failed. ReconstructionSession wraps the device in it so that, when a run throws, it can
// tell a GPU failure (retry the dense stage on the CPU) from any other error (report it). The
// library sees an ordinary IComputeDevice; cancellation is not a fault.

using System;
using System.Threading;
using System.Threading.Tasks;
using ColmapSharp.Compute;

namespace ColmapDemo.Compute
{
	/// <summary>An <see cref="IComputeDevice"/> that records the first exception its inner device threw.</summary>
	public sealed class FaultWatchingComputeDevice : IComputeDevice
	{
		private readonly IComputeDevice inner;

		public FaultWatchingComputeDevice(IComputeDevice inner)
		{
			this.inner = inner ?? throw new ArgumentNullException(nameof(inner));
		}

		/// <summary>The first exception the device threw or faulted with, or null.</summary>
		public Exception Fault { get; private set; }

		public ComputeDeviceLimits Limits => this.inner.Limits;

		public bool SupportsBlockingWait => this.inner.SupportsBlockingWait;

		public IComputeBuffer CreateBuffer(ComputeBufferKind kind, long size, ReadOnlySpan<byte> initial = default, string label = null)
		{
			try
			{
				return this.inner.CreateBuffer(kind, size, initial, label);
			}
			catch (Exception e) when (this.Record(e))
			{
				throw;
			}
		}

		public void WriteBuffer(IComputeBuffer buffer, long offset, ReadOnlySpan<byte> data)
		{
			try
			{
				this.inner.WriteBuffer(buffer, offset, data);
			}
			catch (Exception e) when (this.Record(e))
			{
				throw;
			}
		}

		public IComputeKernel CreateKernel(in ComputeKernelDescriptor descriptor)
		{
			try
			{
				return this.inner.CreateKernel(descriptor);
			}
			catch (Exception e) when (this.Record(e))
			{
				throw;
			}
		}

		public IComputeBindGroup CreateBindGroup(IComputeKernel kernel, int group, ReadOnlySpan<ComputeBufferBinding> entries)
		{
			try
			{
				return this.inner.CreateBindGroup(kernel, group, entries);
			}
			catch (Exception e) when (this.Record(e))
			{
				throw;
			}
		}

		public void Dispatch(IComputeKernel kernel, ReadOnlySpan<IComputeBindGroup> groups, uint x, uint y = 1, uint z = 1)
		{
			try
			{
				this.inner.Dispatch(kernel, groups, x, y, z);
			}
			catch (Exception e) when (this.Record(e))
			{
				throw;
			}
		}

		public async ValueTask FlushAsync(CancellationToken cancellationToken = default)
		{
			try
			{
				await this.inner.FlushAsync(cancellationToken).ConfigureAwait(false);
			}
			catch (Exception e) when (this.Record(e))
			{
				throw;
			}
		}

		public async ValueTask ReadBufferAsync(IComputeBuffer buffer, long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
		{
			try
			{
				await this.inner.ReadBufferAsync(buffer, offset, destination, cancellationToken).ConfigureAwait(false);
			}
			catch (Exception e) when (this.Record(e))
			{
				throw;
			}
		}

		// An exception filter, so the stack is not unwound to record it; always false (never catches).
		private bool Record(Exception e)
		{
			if (this.Fault == null && e is not OperationCanceledException)
			{
				this.Fault = e;
			}

			return false;
		}
	}
}
