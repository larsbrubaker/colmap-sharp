// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RotationEstimatorOptions: the options half of colmap/estimators/rotation_averaging.h
// (RotationAveragingReweighting, RotationEstimatorOptions and its WeightType). COLMAP adapts
// its rotation averaging from Theia's RobustRotationEstimator (http://www.theia-sfm.org/);
// for gravity aligned rotation averaging see the paper "Gravity Aligned Rotation Averaging".
// The options drive RotationAveragingProblem.cs (the linear system) and
// RotationAveragingSolver.cs (L1 + IRLS).

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Estimators;

/// <summary>
/// Reweighting scheme applied to the relative-rotation constraints. Port of
/// colmap::RotationAveragingReweighting.
/// </summary>
public enum RotationAveragingReweighting
{
	/// <summary>All constraints are weighted equally.</summary>
	Uniform = 0,

	/// <summary>
	/// Weight each constraint by the number of inlier two-view matches
	/// (PoseGraph.Edge.NumMatches) of the corresponding edge, normalized to (0, 1].
	/// </summary>
	InlierMatchCount = 1,
}

/// <summary>Robust IRLS weight function. Port of RotationEstimatorOptions::WeightType.</summary>
public enum RotationWeightType
{
	/// <summary>
	/// Geman-McClure weight from "Efficient and robust large-scale rotation averaging"
	/// (Chatterjee et al., 2013).
	/// </summary>
	GemanMcClure,

	/// <summary>Half norm from "Robust Relative Rotation Averaging" (Chatterjee et al., 2017).</summary>
	HalfNorm,
}

/// <summary>Port of colmap::RotationEstimatorOptions.</summary>
public sealed class RotationEstimatorOptions
{
	/// <summary>
	/// PRNG seed for stochastic methods during rotation averaging. If -1 (default), the seed
	/// is not reset (non-deterministic across runs). If &gt;= 0, rotation averaging is
	/// deterministic with the given seed.
	/// </summary>
	public int RandomSeed { get; set; } = -1;

	/// <summary>Maximum number of times to run L1 minimization.</summary>
	public int MaxNumL1Iterations { get; set; } = 5;

	/// <summary>Average step size threshold to terminate the L1 minimization.</summary>
	public double L1StepConvergenceThreshold { get; set; } = 0.001;

	/// <summary>The number of iterative reweighted least squares iterations to perform.</summary>
	public int MaxNumIrlsIterations { get; set; } = 100;

	/// <summary>Average step size threshold to terminate the IRLS minimization.</summary>
	public double IrlsStepConvergenceThreshold { get; set; } = 0.001;

	/// <summary>Gravity direction.</summary>
	public Vector3d GravityDir { get; set; } = Vector3d.UnitY;

	/// <summary>The point (in degrees) where the Huber-like cost function switches from L1 to L2.</summary>
	public double IrlsLossParameterSigma { get; set; } = 5.0;

	/// <summary>
	/// Tikhonov ridge added to the diagonal of the normal equations A^T (W) A before each
	/// Cholesky factorization in the L1 and IRLS phases. The theoretical normal equations of a
	/// connected pose graph plus gauge fix are positive definite, but Cholesky may still report
	/// "matrix not positive definite" on poorly conditioned graphs (e.g., long sequential video
	/// chains). A small positive value (e.g., 1e-9) stabilizes such systems; zero disables it.
	/// </summary>
	public double RidgeRegularization { get; set; } = 1e-9;

	/// <summary>Robust weight function of the IRLS phase.</summary>
	public RotationWeightType WeightType { get; set; } = RotationWeightType.GemanMcClure;

	/// <summary>Flag to skip maximum spanning tree initialization.</summary>
	public bool SkipInitialization { get; set; }

	/// <summary>Flag to use gravity priors for rotation averaging.</summary>
	public bool UseGravity { get; set; }

	/// <summary>
	/// Flag to use stratified solving for mixed gravity systems. If true and UseGravity is
	/// true, first solves the 1-DOF system with gravity-only pairs, then the full 3-DOF system.
	/// </summary>
	public bool UseStratified { get; set; } = true;

	/// <summary>
	/// If true, only consider frames with existing poses when computing connected components.
	/// Set to true for refinement passes.
	/// </summary>
	public bool FilterUnregistered { get; set; }

	/// <summary>
	/// If &gt; 0, filter image pairs with rotation error exceeding this threshold after
	/// solving, then recompute the active set.
	/// </summary>
	public double MaxRotationErrorDeg { get; set; } = 10.0;

	/// <summary>
	/// When false, treat each non-ref sensor's cam_from_rig rotation as a pre-calibrated
	/// constant.
	/// </summary>
	public bool RefineSensorFromRig { get; set; } = true;

	/// <summary>
	/// Reweighting scheme for the relative-rotation constraints. Weights are applied as a
	/// block-diagonal scaling of the linear system, so a noise-free (consistent) system yields
	/// an identical solution regardless of reweighting.
	/// </summary>
	public RotationAveragingReweighting Reweighting { get; set; } = RotationAveragingReweighting.Uniform;

	/// <summary>
	/// A copy of these options (COLMAP's problem and solver hold their options by value).
	/// </summary>
	public RotationEstimatorOptions Clone() => (RotationEstimatorOptions)MemberwiseClone();
}
