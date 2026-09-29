// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SilhouettePoseOptions and the records around silhouette pose registration
// (docs/QUALITY_PLAN.md, stages 4a and 4b). Not a COLMAP port. SilhouettePoseRegistration.cs
// drives the passes, SilhouettePoseRefiner.cs refines one pose, SilhouetteHullModel.cs holds the
// hull surface it projects.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs.Silhouette;
using ColmapSharp.Sensor;

namespace ColmapSharp.Sfm.Silhouette;

/// <summary>A video frame by image name with its foreground mask (nonzero, &gt;= 128, is object).</summary>
public sealed record SilhouetteFrame(string Name, Bitmap Mask);

/// <summary>Settings for <see cref="SilhouettePoseRegistration"/> and <see cref="SilhouettePoseRefiner"/>.</summary>
public sealed class SilhouettePoseOptions
{
	/// <summary>
	/// Pyramid levels, coarse to fine; level L is 1/2^L of full size. More levels widen the basin
	/// of convergence (a coarse contour point sees further); 3 gives a quarter-size first level.
	/// </summary>
	public int PyramidLevels { get; set; } = 3;

	/// <summary>Correspondence-and-step rounds per pyramid level.</summary>
	public int MaxIterationsPerLevel { get; set; } = 30;

	/// <summary>
	/// Contour points per silhouette used by the cost, each side (mask and hull), sampled evenly
	/// along the contour list. More points cost time, not accuracy, once the contour is covered.
	/// </summary>
	public int MaxContourPoints { get; set; } = 800;

	/// <summary>
	/// Huber threshold, in pixels of the level being refined: residuals beyond it count
	/// linearly, so a bitten or bloated part of a mask pulls less than it would squared.
	/// </summary>
	public double HuberPixels { get; set; } = 2.0;

	/// <summary>
	/// The motion prior's rotation sigma (degrees): refinement pays (angle / sigma)^2 for
	/// turning away from the time-interpolated start. A single silhouette barely changes when
	/// the camera swings a degree or two around a smooth object while shifting to keep it in
	/// place, so without the prior the pose wanders along that valley; the silhouette's pixel
	/// residuals, normalized to a mean, are far stronger than the prior anywhere else.
	/// The sigma scales with the span the start was interpolated across (1 for a frame between
	/// two registered neighbours), since an interpolation across a long gap is that much less
	/// certain. Infinity turns it off.
	/// </summary>
	public double PriorRotationDegrees { get; set; } = 5.0;

	/// <summary>
	/// The motion prior's camera-centre sigma as a fraction of the start's distance to the hull
	/// (see <see cref="PriorRotationDegrees"/>). Infinity turns it off.
	/// </summary>
	public double PriorCenterFraction { get; set; } = 0.05;

	/// <summary>Accept a refined frame only if its full-size silhouette IoU with the hull reaches this.</summary>
	public double MinIou { get; set; } = 0.9;

	/// <summary>
	/// Accept a refined frame only if its rotation moved at most this far (degrees) from the
	/// time-interpolated start. A pose that swings further has left its neighbours' motion.
	/// </summary>
	public double MaxRotationFromInitDegrees { get; set; } = 20.0;

	/// <summary>
	/// One-sided extrapolation (only registered frames before, or only after) continues the
	/// motion of the two nearest by at most this many times their time spacing.
	/// </summary>
	public double MaxExtrapolationRatio { get; set; } = 2.0;

	/// <summary>
	/// One-sided extrapolation needs the two frames it continues to differ by at least this
	/// rotation (degrees). Two frames a degree apart give a motion direction that is mostly
	/// noise, which extrapolated over several frames was the mouse's worst start (IoU 0 at
	/// frames 1-8 extrapolated from 10-12). Below it, only the frame right next to the known
	/// one starts (from that frame's pose); the rest wait for a later pass.
	/// </summary>
	public double MinExtrapolationSpanDegrees { get; set; } = 5.0;

	/// <summary>
	/// Check each placed frame with <see cref="SilhouetteConsistencyGate"/> against the frames
	/// that were registered before silhouettes were used, in time order.
	/// </summary>
	public bool ConsistencyGate { get; set; } = true;

	/// <summary>The gate's allowed drop of the median registered-frame hull IoU.</summary>
	public double ConsistencyMedianTolerance { get; set; } = 0.005;

	/// <summary>The gate's allowed drop of any one registered frame's hull IoU.</summary>
	public double ConsistencyMaxTolerance { get; set; } = 0.02;

	/// <summary>
	/// Use the turntable prior (SilhouetteTurntable, stage 5) when the registered frames fit a
	/// circle: each unplaced frame, nearest to a posed frame first, gets starts from a search
	/// round the whole turn plus its interpolated start, and the best refined candidate that
	/// the consistency gate accepts is placed at once, so it is a neighbour for the next.
	/// </summary>
	public bool Turntable { get; set; } = true;

	/// <summary>The turntable search's angle step (degrees).</summary>
	public double TurntableStepDegrees { get; set; } = 3.0;

	/// <summary>How many search peaks are refined per frame.</summary>
	public int TurntableCandidates { get; set; } = 3;

	/// <summary>
	/// Passes over the unplaced frames. After a pass that placed frames, the hull is rebuilt
	/// from every placed frame and the rejected frames are tried again from interpolations
	/// that now include the new neighbours; frames seen from new angles tighten the hull.
	/// </summary>
	public int MaxPasses { get; set; } = 2;

	/// <summary>
	/// The box the hull is carved in. Null takes it from the reconstruction's 3D points
	/// (<see cref="VisualHullBounds.FromPoints"/>).
	/// </summary>
	public AlignedBox3d? HullBox { get; set; }

	/// <summary>How the hull is carved from the registered frames' masks.</summary>
	public VisualHullOptions HullOptions { get; set; } = new();

	/// <summary>
	/// Refine the frames of a pass in parallel. Each frame's start comes from the frames
	/// registered at the start of the pass, so parallel and sequential runs agree exactly.
	/// </summary>
	public bool Parallel { get; set; } = true;
}

/// <summary>The outcome of refining one pose.</summary>
/// <param name="CamFromWorld">The refined pose.</param>
/// <param name="InitialIou">Silhouette IoU (full size) of the hull at the starting pose.</param>
/// <param name="Iou">Silhouette IoU (full size) of the hull at the refined pose.</param>
/// <param name="Iterations">Accepted Levenberg-Marquardt steps over all levels.</param>
public sealed record SilhouettePoseRefinement(Rigid3d CamFromWorld, double InitialIou, double Iou, int Iterations);

/// <summary>What happened to one frame that feature matching had not placed.</summary>
/// <param name="Name">The frame's image name.</param>
/// <param name="Pass">The pass (0-based) of this, its last attempt.</param>
/// <param name="Placed">Whether it was registered into the reconstruction.</param>
/// <param name="InitialIou">IoU at the interpolated start (NaN when there was no start).</param>
/// <param name="Iou">IoU at the refined pose (NaN when there was no start).</param>
/// <param name="Iterations">Accepted refinement steps.</param>
/// <param name="RotationFromInitDegrees">How far refinement rotated the start.</param>
/// <param name="Reason">Why it was rejected; empty when placed.</param>
/// <param name="Start">The time-interpolated starting pose (null when there was none).</param>
/// <param name="Refined">The refined pose, placed or not (null when there was no start).</param>
public sealed record SilhouettePoseFrameReport(
	string Name,
	int Pass,
	bool Placed,
	double InitialIou,
	double Iou,
	int Iterations,
	double RotationFromInitDegrees,
	string Reason,
	Rigid3d? Start = null,
	Rigid3d? Refined = null);

/// <summary>The result of <see cref="SilhouettePoseRegistration.Register"/>.</summary>
/// <param name="Frames">One report per frame that was unplaced at the start, in time order.</param>
/// <param name="PassesRun">Passes that ran.</param>
/// <param name="Hull">The last hull built (from every frame placed before the last pass).</param>
/// <param name="Turntable">The turntable fit when <see cref="SilhouettePoseOptions.Turntable"/> is on.</param>
public sealed record SilhouettePoseResult(
	IReadOnlyList<SilhouettePoseFrameReport> Frames, int PassesRun, VisualHull? Hull, TurntableFit? Turntable = null)
{
	/// <summary>How many frames were placed.</summary>
	public int PlacedCount => Frames.Count(f => f.Placed);
}
