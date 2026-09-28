// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// LogCapture: test-only helper that records the Log messages one test causes. Log.Sink is a
// process-wide static, and TUnit runs the rest of the suite in parallel with a test that
// holds the [NotInParallel(nameof(Log))] key (the key only excludes other holders of it). A
// sink that recorded everything would therefore also collect warnings from whatever else
// happened to be running (e.g. the Recon3D/NVM export warnings of ReconstructionIO tests),
// which made "no warnings" assertions flaky. The capture tags the creating test's execution
// context with an AsyncLocal marker and keeps only messages logged from a context that
// carries it; tasks, Parallel.For workers and new threads all inherit the marker, so the
// test's own worker-thread warnings are still seen. Proven by Util/LogTests.cs.

using System.Collections.Concurrent;

using ColmapSharp.Util;

namespace ColmapSharp.Tests;

/// <summary>
/// Installs a <see cref="Log.Sink"/> that records only the messages logged from the creating
/// execution context (and contexts that flow from it), and restores the previous sink on
/// dispose. Tests using it still take [NotInParallel(nameof(Log))], because two captures
/// swapping the global sink at once could restore it out of order.
/// </summary>
public sealed class LogCapture : IDisposable
{
	private static readonly AsyncLocal<LogCapture?> Owner = new();

	private readonly Action<LogLevel, string>? previousSink;
	private readonly ConcurrentQueue<(LogLevel Level, string Message)> messages = new();

	/// <summary>Start capturing. Must be created in the test body so the marker flows into it.</summary>
	public LogCapture()
	{
		// Set from a synchronous call, so the change is visible to the calling (async) test
		// method and everything it starts afterwards.
		Owner.Value = this;
		previousSink = Log.Sink;
		Log.Sink = Record;
	}

	/// <summary>The captured messages, in arrival order.</summary>
	public IReadOnlyList<(LogLevel Level, string Message)> Messages => messages.ToArray();

	/// <summary>Restore the previous sink.</summary>
	public void Dispose()
	{
		Log.Sink = previousSink;
		Owner.Value = null;
	}

	private void Record(LogLevel level, string message)
	{
		if (ReferenceEquals(Owner.Value, this))
		{
			messages.Enqueue((level, message));
		}
		else
		{
			// Someone else's message: pass it on rather than swallow it.
			previousSink?.Invoke(level, message);
		}
	}
}
