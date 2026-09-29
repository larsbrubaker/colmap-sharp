// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// StageTimer: wall-clock time per stage of AutomaticReconstructionController, for the
// reconstruction benchmark (docs/QUALITY_PLAN.md, stage 0b). Not a COLMAP port. The controller
// reports a heading at the start of every step (ControllerProgress with the step's Stage), and a
// step runs until the next step's heading, so the timer attributes the time between two changes
// of Stage to the earlier one; a stage that comes back (dense, fusion and meshing repeat per
// model) accumulates. Reports arrive from worker threads too, hence the lock. It forwards every
// report to an optional inner progress.

using System.Diagnostics;

using ColmapSharp.Controllers;

namespace ColmapSharp.Mvs.Testing.Benchmark;

/// <summary>An <see cref="IProgress{T}"/> that times the controller's stages.</summary>
public sealed class StageTimer(IProgress<ControllerProgress>? inner = null) : IProgress<ControllerProgress>
{
	private readonly Stopwatch clock = Stopwatch.StartNew();
	private readonly List<(string Stage, double Seconds)> totals = [];
	private string? current;
	private double currentStart;

	/// <inheritdoc/>
	public void Report(ControllerProgress value)
	{
		lock (totals)
		{
			if (value.Stage != current)
			{
				Close();
				current = value.Stage;
				currentStart = clock.Elapsed.TotalSeconds;
			}
		}

		inner?.Report(value);
	}

	/// <summary>
	/// Seconds per stage so far, in first-seen order; the running stage counts up to now. Call
	/// it after the run for the final numbers.
	/// </summary>
	public IReadOnlyList<(string Stage, double Seconds)> Stages()
	{
		lock (totals)
		{
			Close();
			if (current is not null)
			{
				currentStart = clock.Elapsed.TotalSeconds;
			}

			return [.. totals];
		}
	}

	// Adds the running stage's time since currentStart to its total.
	private void Close()
	{
		if (current is null)
		{
			return;
		}

		double seconds = clock.Elapsed.TotalSeconds - currentStart;
		int i = totals.FindIndex(t => t.Stage == current);
		if (i < 0)
		{
			totals.Add((current, seconds));
		}
		else
		{
			totals[i] = (current, totals[i].Seconds + seconds);
		}
	}
}
