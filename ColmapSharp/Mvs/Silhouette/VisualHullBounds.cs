// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VisualHullBounds: the box VisualHull.cs carves, derived from the sparse points when no box is
// supplied (docs/QUALITY_PLAN.md, stage 3a). Not a COLMAP port. The points' bounds after dropping
// a small fraction of outliers on each side of each axis (sparse clouds carry a few stray
// triangulations), grown by a margin, because a sparse cloud rarely reaches the object's
// extremities (the dark mouse's top has almost no points). The carving itself removes everything
// the silhouettes rule out, so a loose box costs resolution, never correctness; a tight one would
// cut the object off.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs.Silhouette;

/// <summary>Bounding boxes for <see cref="VisualHull.Build"/>.</summary>
public static class VisualHullBounds
{
	/// <summary>
	/// The bounds of <paramref name="points"/> with <paramref name="trimFraction"/> of them dropped
	/// from each end of each axis, grown on every side by <paramref name="marginFraction"/> times
	/// the trimmed box's longest side.
	/// </summary>
	public static AlignedBox3d FromPoints(IReadOnlyList<Vector3d> points, double marginFraction = 0.25, double trimFraction = 0.01)
	{
		Check.NotNull(points);
		Check.That(points.Count > 0, "No points to bound");
		Check.That(trimFraction >= 0 && trimFraction < 0.5);
		Check.That(marginFraction >= 0);
		var low = new double[3];
		var high = new double[3];
		var coordinates = new double[points.Count];
		int drop = (int)Math.Floor(trimFraction * points.Count);
		for (int axis = 0; axis < 3; axis++)
		{
			for (int i = 0; i < points.Count; i++)
			{
				coordinates[i] = points[i][axis];
			}

			Array.Sort(coordinates);
			low[axis] = coordinates[drop];
			high[axis] = coordinates[points.Count - 1 - drop];
		}

		double side = Math.Max(high[0] - low[0], Math.Max(high[1] - low[1], high[2] - low[2]));
		double margin = Math.Max(marginFraction * side, 1e-9);
		return new AlignedBox3d(
			new Vector3d(low[0] - margin, low[1] - margin, low[2] - margin),
			new Vector3d(high[0] + margin, high[1] + margin, high[2] + margin));
	}
}
