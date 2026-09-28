// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DeferredCompletion: how RecordingComputeDevice.cs (and so ReferenceComputeDevice, which wraps
// it) completes FlushAsync and ReadBufferAsync when SupportsBlockingWait is false, i.e. when it
// stands in for a browser WebGPU device. Not a COLMAP port (PORTING_PLAN.md Phase 13).
//
// In the browser a map completes only after the caller returns control to the JS event loop.
// The double models that by completing when the caller awaits (the awaiter registers its
// continuation) and never before, however long the caller keeps running. It used to complete
// on a pool thread after Task.Yield, which could finish while the caller was still between the
// call and its await, so "nothing has completed yet" was a race rather than a guarantee
// (RecordingComputeDeviceFlushTests.WithoutBlockingWait_ReadCopiesTheSubmittedBytesOnlyOnCompletion
// failed about one full-suite run in five).
//
// Consequences: a caller that blocks with GetAwaiter().GetResult() on a pending operation gets
// an InvalidOperationException instead of silently succeeding, and an operation nobody awaits
// never copies its bytes.

using System.Threading.Tasks.Sources;

namespace ColmapSharp.Compute.Testing;

/// <summary>
/// A single-use <see cref="IValueTaskSource"/> that stays pending until its continuation is
/// registered, then reports the fault, the cancellation, or runs <c>onSuccess</c> and succeeds.
/// </summary>
internal sealed class DeferredCompletion(
	Exception? fault, bool canceled, CancellationToken cancellationToken, Action? onSuccess) : IValueTaskSource
{
	// Continuations always run asynchronously, as they do off the browser's event loop.
	private ManualResetValueTaskSourceCore<bool> core = new() { RunContinuationsAsynchronously = true };
	private int finished;

	/// <inheritdoc/>
	public ValueTaskSourceStatus GetStatus(short token) => this.core.GetStatus(token);

	/// <inheritdoc/>
	public void OnCompleted(Action<object?> continuation, object? state, short token, ValueTaskSourceOnCompletedFlags flags)
	{
		this.core.OnCompleted(continuation, state, token, flags);
		this.Finish();
	}

	/// <inheritdoc/>
	public void GetResult(short token) => this.core.GetResult(token);

	// A GPU fault wins over cancellation: it is the more important thing for the caller to see.
	private void Finish()
	{
		if (Interlocked.Exchange(ref this.finished, 1) != 0)
		{
			return;
		}

		if (fault != null)
		{
			this.core.SetException(fault);
		}
		else if (canceled)
		{
			this.core.SetException(new OperationCanceledException(cancellationToken));
		}
		else
		{
			onSuccess?.Invoke();
			this.core.SetResult(true);
		}
	}
}
