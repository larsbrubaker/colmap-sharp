// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// StageTimer: the wall time of each stage of a run, from the progress reports' own timestamps.
// ColmapDemoApp.Run.cs stamps each report on the run's thread, as it is made, and feeds it here
// later on the UI thread; timing on arrival instead would shift every stage by however long the
// UI thread took to get to the report (a whole stage, in the browser, where the run holds the
// thread until it yields).

using System;
using System.Collections.Generic;

namespace ColmapDemo
{
	/// <summary>Turns a sequence of (stage, time) progress reports into per-stage durations.</summary>
	public sealed class StageTimer
	{
		private readonly List<(string Stage, double Seconds)> times = new List<(string Stage, double Seconds)>();

		private string stage;

		private TimeSpan stageStart;

		/// <summary>Each finished stage and its wall time in seconds, in the order they ran.</summary>
		public IReadOnlyList<(string Stage, double Seconds)> Times => this.times;

		/// <summary>Raised with each stage as it finishes.</summary>
		public event Action<string, double> StageFinished;

		/// <summary>Forgets the last run.</summary>
		public void Reset()
		{
			this.times.Clear();
			this.stage = null;
		}

		/// <summary>
		/// A progress report of <paramref name="reportStage"/> made at <paramref name="at"/>. A report of
		/// a new stage ends the one before at that same moment.
		/// </summary>
		public void Observe(string reportStage, TimeSpan at)
		{
			if (reportStage == this.stage)
			{
				return;
			}

			this.Finish(at);
			this.stage = reportStage;
			this.stageStart = at;
		}

		/// <summary>Ends the stage in progress, if any, at <paramref name="at"/>.</summary>
		public void Finish(TimeSpan at)
		{
			if (this.stage == null)
			{
				return;
			}

			double seconds = (at - this.stageStart).TotalSeconds;
			this.times.Add((this.stage, seconds));
			this.StageFinished?.Invoke(this.stage, seconds);
			this.stage = null;
		}
	}
}
