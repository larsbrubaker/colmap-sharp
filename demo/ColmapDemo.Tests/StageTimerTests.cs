// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of StageTimer (demo/ColmapDemo/StageTimer.cs): each stage gets the time between its first
// report and the next stage's first report, by the reports' own timestamps.

namespace ColmapDemo.Tests;

public class StageTimerTests
{
	[Test]
	public async Task EachStageGetsTheTimeUntilTheNextStageBegins()
	{
		var timer = new StageTimer();

		// Loading photos 0-2 s (three reports), extraction 2-7 s, matching 7-8 s, run ends at 10 s.
		timer.Observe("Loading photos", TimeSpan.FromSeconds(0));
		timer.Observe("Loading photos", TimeSpan.FromSeconds(1));
		timer.Observe("Loading photos", TimeSpan.FromSeconds(1.5));
		timer.Observe("Feature extraction", TimeSpan.FromSeconds(2));
		timer.Observe("Feature extraction", TimeSpan.FromSeconds(6));
		timer.Observe("Feature matching", TimeSpan.FromSeconds(7));
		timer.Observe("Sparse reconstruction", TimeSpan.FromSeconds(8));
		timer.Finish(TimeSpan.FromSeconds(10));

		string times = string.Join("; ", timer.Times.Select(t => $"{t.Stage}={t.Seconds}"));
		await Assert.That(times).IsEqualTo("Loading photos=2; Feature extraction=5; Feature matching=1; Sparse reconstruction=2");
	}

	[Test]
	public async Task ResetForgetsTheLastRun()
	{
		var timer = new StageTimer();
		timer.Observe("Feature extraction", TimeSpan.FromSeconds(0));
		timer.Finish(TimeSpan.FromSeconds(3));

		timer.Reset();
		timer.Finish(TimeSpan.FromSeconds(9));

		await Assert.That(timer.Times.Count).IsEqualTo(0);
	}
}
