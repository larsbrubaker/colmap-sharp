// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouettePoseRegistration.Turntable: the turntable-prior pass of silhouette pose registration
// (docs/QUALITY_PLAN.md, stage 5). Not a COLMAP port. SilhouettePoseRegistration.cs runs it in
// place of the interpolate-and-refine pass when SilhouetteTurntable.Fit finds the registered
// cameras on a circle.
//
// Order: the unplaced frame nearest in time to a posed one goes first (earlier on a tie), and a
// frame placed here is a posed neighbour for the next pick, so a run of frames grows from its
// registered ends. Each frame is tried once per pass.
//
// Per frame: candidates are the turntable search's best peaks (SilhouetteTurntable.SearchStarts,
// turned from the nearest posed frame) plus the interpolated start when there is one. Each is
// refined (SilhouettePoseRefiner). A silhouette and its 180-degree twin can both fit, so the
// choice is by continuity among the near-best: of the refined candidates within
// ContinuityIouMargin of the best IoU, the one closest in rotation to the nearest posed frame
// comes first (see AttemptTurntable for what "closest" is measured against). Candidates are then tried in that order against MinIou and the consistency gate;
// the first to pass is placed.

using ColmapSharp.Geometry;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Scene;

namespace ColmapSharp.Sfm.Silhouette;

public static partial class SilhouettePoseRegistration
{
	// Refined candidates this close to the best IoU count as ties, decided by continuity.
	private const double ContinuityIouMargin = 0.01;

	private static int TurntablePass(
		Reconstruction reconstruction,
		IReadOnlyList<SilhouetteFrame> frames,
		SilhouettePoseFrameReport?[] reports,
		SilhouetteHullModel model,
		TurntableFit fit,
		SilhouetteConsistencyGate? gate,
		int pass,
		SilhouettePoseOptions options,
		CancellationToken cancellationToken)
	{
		int n = frames.Count;
		var tried = new bool[n];
		int placed = 0;
		while (true)
		{
			cancellationToken.ThrowIfCancellationRequested();
			var poses = new Rigid3d?[n];
			for (int i = 0; i < n; i++)
			{
				poses[i] = PoseOf(reconstruction, frames[i].Name);
			}

			int next = -1, nextDistance = int.MaxValue;
			for (int i = 0; i < n; i++)
			{
				if (poses[i] is not null || tried[i])
				{
					continue;
				}

				int distance = NearestPosed(poses, i) is int j ? Math.Abs(j - i) : int.MaxValue;
				if (distance < nextDistance)
				{
					next = i;
					nextDistance = distance;
				}
			}

			if (next < 0 || nextDistance == int.MaxValue)
			{
				return placed;
			}

			tried[next] = true;
			(SilhouettePoseFrameReport report, Rigid3d pose, uint cameraId) = AttemptTurntable(
				reconstruction, frames, poses, next, model, fit, gate, pass, options, cancellationToken);
			reports[next] = report;
			if (report.Placed)
			{
				Place(reconstruction, frames[next].Name, cameraId, pose);
				placed++;
			}
		}
	}

	private static (SilhouettePoseFrameReport Report, Rigid3d Pose, uint CameraId) AttemptTurntable(
		Reconstruction reconstruction,
		IReadOnlyList<SilhouetteFrame> frames,
		Rigid3d?[] poses,
		int index,
		SilhouetteHullModel model,
		TurntableFit fit,
		SilhouetteConsistencyGate? gate,
		int pass,
		SilhouettePoseOptions options,
		CancellationToken cancellationToken)
	{
		string name = frames[index].Name;
		Camera? camera = CameraFor(reconstruction, frames, poses, index);
		Rigid3d reference = poses[NearestPosed(poses, index)!.Value]!.Value;
		if (camera is null)
		{
			return (new SilhouettePoseFrameReport(name, pass, false, double.NaN, double.NaN, 0, double.NaN, "no camera"), default, 0);
		}

		var starts = SilhouetteTurntable.SearchStarts(
			fit, reference, camera, frames[index].Mask, model, options.TurntableStepDegrees,
			options.PyramidLevels - 1, options.TurntableCandidates).Select(s => s.Pose).ToList();
		// Continuity is measured against the interpolated start when there is one (the motion's
		// prediction), else against the nearest posed frame. Measured on the DarkObject with
		// every other frame registered: against the nearest frame alone the near-tied search
		// peaks won over the interpolation and the median error rose from 0.30 to 0.74 degrees.
		Rigid3d predicted = reference;
		bool hasPrediction = false;
		if (InitializePose(poses, index, options.MaxExtrapolationRatio, options.MinExtrapolationSpanDegrees) is Rigid3d interpolated)
		{
			starts.Add(interpolated);
			predicted = interpolated;
			hasPrediction = true;
		}

		double turnLimit = TurnLimitDegrees(poses, index, hasPrediction, options);

		double priorScale = PriorScale(poses, index);
		var refined = new List<(SilhouettePoseRefinement Refinement, Rigid3d Start, double Turn)>();
		foreach (Rigid3d start in starts)
		{
			SilhouettePoseRefinement r = SilhouettePoseRefiner.Refine(
				camera, frames[index].Mask, start, model, options, priorScale, cancellationToken);
			refined.Add((r, start, r.CamFromWorld.Rotation.AngularDistance(predicted.Rotation) * 180 / Math.PI));
		}

		if (refined.Count == 0)
		{
			return (new SilhouettePoseFrameReport(name, pass, false, double.NaN, double.NaN, 0, double.NaN, "no candidate start"), default, 0);
		}

		double best = refined.Max(r => r.Refinement.Iou);
		var ordered = refined
			.Select((r, k) => (r.Refinement, r.Start, r.Turn, Index: k, Near: r.Refinement.Iou >= best - ContinuityIouMargin))
			.OrderByDescending(r => r.Near)
			.ThenBy(r => r.Near ? r.Turn : -r.Refinement.Iou)
			.ThenBy(r => r.Index)
			.ToList();
		// The first candidate that is continuous and clears MinIou is the one that would be
		// placed; only it goes to the gate (each gate check is a hull carve).
		string reason = "";
		foreach (var candidate in ordered)
		{
			SilhouettePoseRefinement r = candidate.Refinement;
			if (candidate.Turn > turnLimit)
			{
				reason = reason.Length > 0 ? reason : $"turned {candidate.Turn:F1} degrees from its neighbours' motion (limit {turnLimit:F1})";
				continue;
			}

			if (r.Iou < options.MinIou)
			{
				reason = reason.Length > 0 ? reason : $"silhouette IoU {r.Iou:F3} below {options.MinIou:F3}";
				continue;
			}

			bool placed = true;
			if (gate is not null)
			{
				var view = new VisualHullView(camera, r.CamFromWorld, frames[index].Mask);
				if (!gate.TryAccept(view, out double medianDrop, out double maxDrop, cancellationToken))
				{
					reason = $"carving with it lowers the registered frames' hull IoU (median {medianDrop:F4}, worst {maxDrop:F4})";
					placed = false;
				}
			}

			return (new SilhouettePoseFrameReport(
				name, pass, placed, r.InitialIou, r.Iou, r.Iterations, candidate.Turn, placed ? "" : reason, candidate.Start, r.CamFromWorld),
				placed ? r.CamFromWorld : default, placed ? camera.CameraId : 0);
		}

		var first = ordered[0];
		return (new SilhouettePoseFrameReport(
			name, pass, false, first.Refinement.InitialIou, first.Refinement.Iou, first.Refinement.Iterations, first.Turn,
			reason, first.Start, first.Refinement.CamFromWorld), default, 0);
	}

	/// <summary>
	/// How far (degrees) a placed pose may turn from the motion's prediction for frame
	/// <paramref name="index"/>. With an interpolated or extrapolated start it is
	/// MaxRotationFromInitDegrees times the prior scale (the start's uncertainty). Without one,
	/// the prediction is the nearest posed frame, and the bound is the motion's rate (median
	/// rotation per frame between time-consecutive posed frames) times the time distance, plus
	/// MaxRotationFromInitDegrees as margin. Either way a 180-degree silhouette twin is out, which
	/// on the mouse was placed 131.7 degrees from its prediction before this bound existed.
	/// </summary>
	internal static double TurnLimitDegrees(Rigid3d?[] poses, int index, bool hasPrediction, SilhouettePoseOptions options)
	{
		if (hasPrediction)
		{
			return options.MaxRotationFromInitDegrees * PriorScale(poses, index);
		}

		var rates = new List<double>();
		int previous = -1;
		for (int i = 0; i < poses.Length; i++)
		{
			if (poses[i] is not Rigid3d pose)
			{
				continue;
			}

			if (previous >= 0)
			{
				double turn = pose.Rotation.AngularDistance(poses[previous]!.Value.Rotation) * 180 / Math.PI;
				rates.Add(turn / (i - previous));
			}

			previous = i;
		}

		rates.Sort();
		double rate = rates.Count == 0 ? 0 : rates[rates.Count / 2];
		int distance = NearestPosed(poses, index) is int j ? Math.Abs(j - index) : 0;
		return rate * distance + options.MaxRotationFromInitDegrees;
	}

	// The posed frame nearest in time to `index` (earlier on a tie), or null.
	private static int? NearestPosed(Rigid3d?[] poses, int index)
	{
		for (int d = 1; d < poses.Length; d++)
		{
			if (index - d >= 0 && poses[index - d] is not null)
			{
				return index - d;
			}

			if (index + d < poses.Length && poses[index + d] is not null)
			{
				return index + d;
			}
		}

		return null;
	}
}
