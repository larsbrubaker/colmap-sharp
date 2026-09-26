// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// LogTests: C#-only tests (COLMAP has no test for its LOG(WARNING) output) checking that
// COLMAP's warnings reach ColmapSharp/Util/Log.cs's sink with COLMAP's text, through one
// representative site: CorrespondenceGraph.AddTwoViewGeometry, which warns for self-matches,
// duplicate matches and out-of-range point indices.
//
// Log.Sink is process-wide and other tests run concurrently, so the capture keeps only the
// messages this test expects to see, and the previous sink is restored afterwards.

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

		var captured = new ConcurrentQueue<(LogLevel Level, string Message)>();
		Action<LogLevel, string>? previousSink = Log.Sink;
		Log.Sink = (level, message) =>
		{
			if (expected.Contains(message))
			{
				captured.Enqueue((level, message));
			}
		};

		try
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
		}
		finally
		{
			Log.Sink = previousSink;
		}

		await Assert.That(captured.Select(entry => entry.Message).ToArray()).IsEquivalentTo(expected);
		await Assert.That(captured.All(entry => entry.Level == LogLevel.Warning)).IsTrue();
	}
}
