// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// LogTests: C#-only tests (COLMAP has no test for its LOG(WARNING) output) checking that
// COLMAP's warnings reach ColmapSharp/Util/Log.cs's sink with COLMAP's text, through one
// representative site: CorrespondenceGraph.AddTwoViewGeometry, which warns for self-matches,
// duplicate matches and out-of-range point indices.
//
// Log.Sink is process-wide and other tests run concurrently, so captures go through
// LogCapture, which keeps only the messages logged from this test's own execution context;
// the second test pins that isolation.

using System.Collections.Concurrent;

using ColmapSharp.Feature;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

[NotInParallel(nameof(Log))]
public class LogTests
{
	[Test]
	public async Task CSharpOnly_CorrespondenceGraphWarningsReachTheSink()
	{
		string[] expected =
		[
			"Cannot use self-matches for image_id=7",
			"Duplicate correspondence between point2D_idx=1 in image_id=7 and point2D_idx=2 in image_id=8",
			"point2D_idx=20 in image_id=7 does not exist",
			"point2D_idx=30 in image_id=8 does not exist",
		];

		IReadOnlyList<(LogLevel Level, string Message)> captured;
		using (var capture = new LogCapture())
		{
			var correspondenceGraph = new CorrespondenceGraph();
			correspondenceGraph.AddImage(7, 10);
			correspondenceGraph.AddImage(8, 10);
			correspondenceGraph.AddTwoViewGeometry(7, 7, new TwoViewGeometry());
			correspondenceGraph.AddTwoViewGeometry(7, 8, new TwoViewGeometry
			{
				InlierMatches = [new FeatureMatch(1, 2), new FeatureMatch(1, 2), new FeatureMatch(20, 30)],
			});

			// The counts are COLMAP's: only the first (1, 2) match survives.
			await Assert.That(correspondenceGraph.NumMatchesBetweenImages(7, 8)).IsEqualTo(1u);
			captured = capture.Messages;
		}

		await Assert.That(captured.Select(entry => entry.Message).ToArray()).IsEquivalentTo(expected);
		await Assert.That(captured.All(entry => entry.Level == LogLevel.Warning)).IsTrue();
	}

	// A warning logged by code running concurrently outside the test (here a task started with
	// execution-context flow suppressed, as another test's work would be) is not captured and
	// still reaches the previous sink, while the test's own warnings are captured whether
	// logged on the test's thread or on Parallel.For workers. Without this isolation,
	// CSharpOnly_RunAsyncGivesTheSameResultsAsRunOnAnyDevice's "no warnings" check failed
	// whenever a parallel test's export or texturing warning landed in its window.
	[Test]
	public async Task CSharpOnly_CaptureKeepsOnlyThisTestsMessages()
	{
		var forwarded = new ConcurrentQueue<string>();
		Action<LogLevel, string>? previousSink = Log.Sink;
		Log.Sink = (_, message) => forwarded.Enqueue(message);
		IReadOnlyList<(LogLevel Level, string Message)> captured;
		try
		{
			using var capture = new LogCapture();
			Log.Warning("own thread");
			Parallel.For(0, 4, i => Log.Error("own worker " + i));
			Task foreign;
			using (ExecutionContext.SuppressFlow())
			{
				foreign = Task.Run(() => Log.Warning("other test"));
			}

			await foreign;
			captured = capture.Messages;
		}
		finally
		{
			Log.Sink = previousSink;
		}

		await Assert.That(captured.Select(entry => entry.Message).ToArray())
			.IsEquivalentTo(["own thread", "own worker 0", "own worker 1", "own worker 2", "own worker 3"]);
		await Assert.That(forwarded.ToArray()).IsEquivalentTo(["other test"]);
	}
}
