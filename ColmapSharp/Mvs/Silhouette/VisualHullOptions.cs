// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VisualHullOptions and VisualHullView: the inputs of the silhouette visual hull
// (docs/QUALITY_PLAN.md, stage 3a). Not a COLMAP port; COLMAP has no visual hull. The carving
// that uses them is VisualHull.cs (per-view mask lookups in HullMaskView.cs), the output grid is
// OccupancyGrid.cs, the surface extraction is HullMarchingCubes.cs and the bounding-box helper is
// VisualHullBounds.cs.
//
// References (written from the papers only):
// - A. Laurentini, "The visual hull concept for silhouette-based image understanding", IEEE
//   PAMI 16(2), 1994 - the visual hull as the intersection of the silhouette cones.
// - R. Szeliski, "Rapid octree construction from image sequences", CVGIP: Image Understanding
//   58(1), 1993 - hierarchical octree carving with a conservative per-cell footprint test.
// - K. N. Kutulakos and S. M. Seitz, "A theory of shape by space carving", IJCV 38(3), 2000 -
//   the carving framework the k-disagreement tolerance leans on.

using ColmapSharp.Geometry;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

namespace ColmapSharp.Mvs.Silhouette;

/// <summary>
/// One silhouette view: a camera (any COLMAP model), its pose and a foreground mask the camera's
/// size (the first channel is read; a value of 128 or more is the object).
/// </summary>
public sealed record VisualHullView(Camera Camera, Rigid3d CamFromWorld, Bitmap Mask);

/// <summary>Options of <see cref="VisualHull.Build"/>.</summary>
public sealed class VisualHullOptions
{
	/// <summary>
	/// Voxels along the longest side of the bounding box. The other sides get as many voxels of
	/// the same (cubic) size as they need.
	/// </summary>
	public int Resolution { get; set; } = 128;

	/// <summary>
	/// How many views may disagree (k). A point is carved only where more than k of the views
	/// that see it put it outside their mask, so one bad mask or pose cannot bite the hull when
	/// k = 1. The price is a looser hull where few views constrain a region. 0 is the classic
	/// visual hull. The cost: a point that only k views see is carved even where they all say
	/// inside (too few votes to outvote a bad view), so k = 1 can lose parts of the object seen
	/// from a single view.
	/// </summary>
	public int DisagreementTolerance { get; set; }

	/// <summary>
	/// Whether a point in front of a camera but off its image counts as outside that view's mask.
	/// Off (the default), a view does not constrain what it cannot see, which is right when views
	/// crop the object, but the parts of the box that few views see stay filled and inflate the
	/// hull toward the box. On suits a capture that keeps the whole object in frame (a static
	/// camera, a turntable): the box outside the frusta is carved, but a view that crops the
	/// object cuts off the cropped part.
	/// </summary>
	public bool OffImageIsOutside { get; set; }

	/// <summary>
	/// Passes of the separable [1 2 1]/4 filter over the occupancy before marching cubes. One pass
	/// takes off the voxel staircase; each pass also rounds convex corners in by a fraction of a
	/// voxel.
	/// </summary>
	public int SmoothingPasses { get; set; } = 1;

	/// <summary>Iso level of the smoothed occupancy that becomes the surface.</summary>
	public double IsoLevel { get; set; } = 0.5;

	/// <summary>
	/// Carve octants in parallel. Each worker writes only its own block of the grid, so the result
	/// is the same either way.
	/// </summary>
	public bool Parallel { get; set; } = true;
}
