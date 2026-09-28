// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// RecordingComputeDeviceFlushTests: C#-only tests (COLMAP has no compute seam) of the
// submission side of ColmapSharp/Compute/Testing/RecordingComputeDevice.cs: the command
// stream's order, what a flush and a readback carry, the write-after-dispatch hazard the
// IComputeDevice contract forbids, disposing handles a pending dispatch binds, the browser
// (no blocking wait) completion, and how cancellation and GPU faults leave the device. Argument rules are in
// RecordingComputeDeviceTests.cs, whose kernel helpers these reuse.

using ColmapSharp.Compute;
using ColmapSharp.Compute.Testing;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Compute.RecordingComputeDeviceTests;

namespace ColmapSharp.Tests.Compute;

public class RecordingComputeDeviceFlushTests
{
	private sealed record Setup(
		RecordingComputeDevice Device, IComputeKernel Kernel, IComputeBuffer Storage, IComputeBuffer Uniform, IComputeBindGroup Group);

	private static Setup Create(bool supportsBlockingWait = true)
	{
		var device = new RecordingComputeDevice { SupportsBlockingWait = supportsBlockingWait };
		var kernel = StorageAndUniformKernel(device);
		var storage = device.CreateBuffer(ComputeBufferKind.Storage, 16, new byte[] { 1, 2, 3, 4 }, "storage");
		var uniform = device.CreateBuffer(ComputeBufferKind.Uniform, 16, label: "uniform");
		return new Setup(device, kernel, storage, uniform, BindWhole(device, kernel, storage, uniform));
	}

	[Test]
	public async Task Commands_AreRecordedInCallOrder()
	{
		var s = Create();
		s.Device.Dispatch(s.Kernel, new[] { s.Group }, 1);
		s.Device.Dispatch(s.Kernel, new[] { s.Group }, 2);
		await s.Device.FlushAsync();
		s.Device.WriteBuffer(s.Uniform, 0, new byte[16]);
		await s.Device.ReadBufferAsync(s.Storage, 0, new byte[4]);

		var kinds = s.Device.Commands.Select(c => c.GetType().Name).ToArray();
		await Assert.That(kinds).IsEquivalentTo(new[]
		{
			nameof(CreateKernelCommand), nameof(CreateBufferCommand), nameof(CreateBufferCommand), nameof(CreateBindGroupCommand),
			nameof(DispatchCommand), nameof(DispatchCommand), nameof(FlushCommand), nameof(WriteBufferCommand), nameof(ReadBufferCommand),
		}, CollectionOrdering.Matching).Because(s.Device.Dump());
		await Assert.That(s.Device.CommandsOf<DispatchCommand>().Select(d => d.X)).IsEquivalentTo(new[] { 1u, 2u }, CollectionOrdering.Matching);

		s.Device.ClearRecording();
		await Assert.That(s.Device.Commands.Count).IsEqualTo(0);
	}

	[Test]
	public async Task Flush_CarriesThePendingDispatchesOnce()
	{
		var s = Create();
		s.Device.Dispatch(s.Kernel, new[] { s.Group }, 1);
		s.Device.Dispatch(s.Kernel, new[] { s.Group }, 1);
		await Assert.That(s.Device.PendingDispatchCount).IsEqualTo(2);
		await s.Device.FlushAsync();
		await Assert.That(s.Device.PendingDispatchCount).IsEqualTo(0);
		await s.Device.FlushAsync();
		await Assert.That(s.Device.CommandsOf<FlushCommand>().Select(f => f.DispatchCount)).IsEquivalentTo(new[] { 2, 0 }, CollectionOrdering.Matching);
	}

	[Test]
	public async Task WriteBuffer_ToABufferAPendingDispatchBinds_Throws()
	{
		var s = Create();
		var unrelated = s.Device.CreateBuffer(ComputeBufferKind.Uniform, 16);
		s.Device.Dispatch(s.Kernel, new[] { s.Group }, 1);

		// The write would land before the dispatch runs, so the dispatch would see it.
		var exception = Assert.Throws<InvalidOperationException>(() => s.Device.WriteBuffer(s.Uniform, 0, new byte[16]));
		await Assert.That(exception.Message).Contains("since the last flush");
		Assert.Throws<InvalidOperationException>(() => s.Device.WriteBuffer(s.Storage, 0, new byte[4]));
		await Assert.That(s.Device.CommandsOf<WriteBufferCommand>().Count).IsEqualTo(0);

		// A buffer no pending dispatch binds is free to write.
		s.Device.WriteBuffer(unrelated, 0, new byte[16]);

		// Once flushed, the dispatch has run and the buffer may be written for the next one.
		await s.Device.FlushAsync();
		s.Device.WriteBuffer(s.Uniform, 0, new byte[16]);
		await Assert.That(s.Device.CommandsOf<WriteBufferCommand>().Count).IsEqualTo(2);
	}

	[Test]
	public async Task WriteBuffer_AfterAReadFlushedTheDispatch_IsAllowed()
	{
		var s = Create();
		s.Device.Dispatch(s.Kernel, new[] { s.Group }, 1);
		await s.Device.ReadBufferAsync(s.Storage, 0, new byte[4]);
		s.Device.WriteBuffer(s.Uniform, 0, new byte[16]);
		await Assert.That(s.Device.CommandsOf<ReadBufferCommand>().Single().DispatchCount).IsEqualTo(1);
		await Assert.That(s.Device.PendingDispatchCount).IsEqualTo(0);
	}

	[Test]
	public async Task ReadBuffer_ReturnsTheStoredBytes()
	{
		var s = Create();
		s.Device.WriteBuffer(s.Storage, 8, new byte[] { 5, 6, 7, 8 });
		var destination = new byte[12];
		await s.Device.ReadBufferAsync(s.Storage, 4, destination);
		await Assert.That(destination).IsEquivalentTo(new byte[] { 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0 }, CollectionOrdering.Matching);
		await Assert.That(s.Device.CommandsOf<ReadBufferCommand>().Single())
			.IsEqualTo(new ReadBufferCommand((RecordingComputeBuffer)s.Storage, 4, 12, 0));

		// Never-written buffers read as zeros; uniform buffers can be read too.
		var fresh = new byte[] { 9, 9, 9, 9 };
		await s.Device.ReadBufferAsync(s.Uniform, 12, fresh);
		await Assert.That(fresh).IsEquivalentTo(new byte[4], CollectionOrdering.Matching);
	}

	[Test]
	public async Task ReadBuffer_ChecksTheRange()
	{
		var s = Create();
		Assert.Throws<ArgumentException>(() => s.Device.ReadBufferAsync(s.Storage, 2, new byte[4]));
		Assert.Throws<ArgumentException>(() => s.Device.ReadBufferAsync(s.Storage, 0, new byte[6]));
		Assert.Throws<ArgumentException>(() => s.Device.ReadBufferAsync(s.Storage, 12, new byte[8]));
		s.Storage.Dispose();
		Assert.Throws<ObjectDisposedException>(() => s.Device.ReadBufferAsync(s.Storage, 0, new byte[4]));
		await Assert.That(s.Device.CommandsOf<ReadBufferCommand>().Count).IsEqualTo(0);
	}

	[Test]
	public async Task DisposingHandlesAPendingDispatchBinds_IsAllowed()
	{
		// WebGPU releases are reference counted: work already recorded keeps its buffers,
		// bind groups and pipeline alive, so a caller may drop them as soon as it has dispatched.
		var s = Create();
		s.Device.Dispatch(s.Kernel, new[] { s.Group }, 1);
		s.Storage.Dispose();
		s.Uniform.Dispose();
		s.Group.Dispose();
		s.Kernel.Dispose();
		await s.Device.FlushAsync();
		await Assert.That(s.Device.CommandsOf<FlushCommand>().Single().DispatchCount).IsEqualTo(1);
	}

	[Test]
	public async Task CanceledFlush_SubmitsThePendingWorkAndLeavesTheDeviceReusable()
	{
		var s = Create();
		s.Device.Dispatch(s.Kernel, new[] { s.Group }, 1);
		using var cancel = new CancellationTokenSource();
		cancel.Cancel();
		await Assert.ThrowsAsync<OperationCanceledException>(async () => await s.Device.FlushAsync(cancel.Token));
		await Assert.That(s.Device.CommandsOf<FlushCommand>().Single().DispatchCount).IsEqualTo(1);
		await Assert.That(s.Device.PendingDispatchCount).IsEqualTo(0);

		// Nothing is pending, so the buffers the dispatch bound may be written again.
		s.Device.WriteBuffer(s.Uniform, 0, new byte[16]);
	}

	[Test]
	public async Task CanceledRead_SubmitsThePendingWorkAndLeavesTheDestinationUntouched()
	{
		var s = Create(supportsBlockingWait: false);
		s.Device.Dispatch(s.Kernel, new[] { s.Group }, 1);
		using var cancel = new CancellationTokenSource();
		cancel.Cancel();
		var destination = new byte[] { 9, 9, 9, 9 };
		await Assert.ThrowsAsync<OperationCanceledException>(async () => await s.Device.ReadBufferAsync(s.Storage, 0, destination, cancel.Token));
		await Assert.That(destination).IsEquivalentTo(new byte[] { 9, 9, 9, 9 }, CollectionOrdering.Matching);
		await Assert.That(s.Device.CommandsOf<ReadBufferCommand>().Count).IsEqualTo(0);
		await Assert.That(s.Device.CommandsOf<FlushCommand>().Single().DispatchCount).IsEqualTo(1);
		await Assert.That(s.Device.PendingDispatchCount).IsEqualTo(0);
		s.Device.WriteBuffer(s.Storage, 0, new byte[4]);
	}

	[Test]
	public async Task CancelingAfterSubmission_DoesNotCancelTheWait()
	{
		// Cancellation is observed only before anything is submitted.
		var s = Create(supportsBlockingWait: false);
		using var cancel = new CancellationTokenSource();
		var destination = new byte[4];
		var read = s.Device.ReadBufferAsync(s.Storage, 0, destination, cancel.Token);
		cancel.Cancel();
		await read;
		await Assert.That(destination).IsEquivalentTo(new byte[] { 1, 2, 3, 4 }, CollectionOrdering.Matching);
	}

	[Test]
	public async Task FaultedSubmit_FaultsTheTaskAndLeavesTheDeviceReusable()
	{
		foreach (var supportsBlockingWait in new[] { true, false })
		{
			var s = Create(supportsBlockingWait);
			s.Device.Dispatch(s.Kernel, new[] { s.Group }, 1);
			s.Device.FaultNextSubmit(new InvalidOperationException("pipeline compile failed"));

			// A GPU-side failure faults the returned task rather than throwing from the call.
			var flush = s.Device.FlushAsync();
			var exception = await Assert.ThrowsAsync<InvalidOperationException>(async () => await flush);
			await Assert.That(exception!.Message).IsEqualTo("pipeline compile failed");
			await Assert.That(s.Device.PendingDispatchCount).IsEqualTo(0);

			// The fault is spent; the next flush and read succeed.
			s.Device.WriteBuffer(s.Uniform, 0, new byte[16]);
			await s.Device.FlushAsync();
			s.Device.Dispatch(s.Kernel, new[] { s.Group }, 1);
			s.Device.FaultNextSubmit(new InvalidOperationException("device lost"));
			var destination = new byte[] { 9, 9, 9, 9 };
			var read = s.Device.ReadBufferAsync(s.Storage, 0, destination);
			await Assert.ThrowsAsync<InvalidOperationException>(async () => await read);
			await Assert.That(destination).IsEquivalentTo(new byte[] { 9, 9, 9, 9 }, CollectionOrdering.Matching);
			await Assert.That(s.Device.PendingDispatchCount).IsEqualTo(0);
		}
	}

	[Test]
	public async Task WithoutBlockingWait_ReadCopiesTheSubmittedBytesOnlyOnCompletion()
	{
		var s = Create(supportsBlockingWait: false);
		var destination = new byte[4];
		var read = s.Device.ReadBufferAsync(s.Storage, 0, destination);

		// Like the browser's event loop, nothing completes while the caller keeps running
		// without yielding, however long that takes. (The double once completed on a pool
		// thread after Task.Yield, which raced this check and failed it about one run in five.)
		SpinWait.SpinUntil(() => read.IsCompleted, TimeSpan.FromMilliseconds(100));
		await Assert.That(read.IsCompleted).IsFalse();

		// Nothing lands in the destination before the task completes, and a write queued after
		// the read was submitted does not reach it.
		await Assert.That(destination).IsEquivalentTo(new byte[4], CollectionOrdering.Matching);
		s.Device.WriteBuffer(s.Storage, 0, new byte[] { 7, 7, 7, 7 });
		await read;
		await Assert.That(destination).IsEquivalentTo(new byte[] { 1, 2, 3, 4 }, CollectionOrdering.Matching);
	}

	[Test]
	public async Task WithoutBlockingWait_FlushAndReadCompleteAsynchronously()
	{
		var s = Create(supportsBlockingWait: false);
		await Assert.That(s.Device.SupportsBlockingWait).IsFalse();
		s.Device.Dispatch(s.Kernel, new[] { s.Group }, 1);
		var flush = s.Device.FlushAsync();
		await Assert.That(flush.IsCompleted).IsFalse();
		await flush;

		var destination = new byte[4];
		var read = s.Device.ReadBufferAsync(s.Storage, 0, destination);
		await Assert.That(read.IsCompleted).IsFalse();
		await read;
		await Assert.That(destination).IsEquivalentTo(new byte[] { 1, 2, 3, 4 }, CollectionOrdering.Matching);
	}

	[Test]
	public async Task WithBlockingWait_FlushAndReadCompleteBeforeReturning()
	{
		var s = Create();
		await Assert.That(s.Device.FlushAsync().IsCompletedSuccessfully).IsTrue();
		await Assert.That(s.Device.ReadBufferAsync(s.Storage, 0, new byte[4]).IsCompletedSuccessfully).IsTrue();
	}
}
