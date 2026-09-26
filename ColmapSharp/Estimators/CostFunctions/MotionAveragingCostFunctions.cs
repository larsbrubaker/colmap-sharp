// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MotionAveragingCostFunctions: colmap/estimators/cost_functions/motion_averaging.h - the
// BATA translation-averaging functors global positioning (estimators/global_positioning.cc)
// minimizes: BATAPairwiseDirectionCostFunctor and its two rig variants. Tests:
// ColmapSharp.Tests/Estimators/CostFunctions/MotionAveragingTests.cs. Tier B.
//
// Reference: Zhuang et al., "Baseline Desensitizing In Translation Averaging", CVPR 2018.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>
/// Computes the error between a translation direction and the direction formed from two
/// positions such that t_ij - scale * (p_j - p_i) is minimized. The positions can either be
/// two camera centers or one camera center and one 3D point. Blocks: pos1 (3), pos2 (3),
/// scale (1). Port of colmap::BATAPairwiseDirectionCostFunctor.
/// </summary>
public readonly struct BATAPairwiseDirectionCostFunctor(Vector3d pos2FromPos1Dir) : ISizedAutoDiffFunctor
{
	private static readonly int[] Sizes = [3, 3, 1];

	private readonly Vector3d _pos2FromPos1Dir = pos2FromPos1Dir;

	/// <inheritdoc/>
	public int NumResiduals => 3;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Vector3d pos2FromPos1Dir) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new BATAPairwiseDirectionCostFunctor(pos2FromPos1Dir));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		T scale = parameters[6];
		Vector3T<T> r = _pos2FromPos1Dir - scale * (Vector3T<T>.Map(parameters[3..]) - Vector3T<T>.Map(parameters));
		r.CopyTo(residuals);
		return true;
	}
}

/// <summary>
/// Computes the error between a translation direction and the direction formed from a
/// camera (c) and 3D point (p) with constant rig extrinsics, such that
/// t_ij - scale * (p - c + t_rig) is minimized. Blocks: point3D (3), rig_in_world (3),
/// scale (1). Port of colmap::RigBATAPairwiseDirectionConstantRigCostFunctor.
/// </summary>
public readonly struct RigBATAPairwiseDirectionConstantRigCostFunctor(Vector3d camFromPoint3DDir, Vector3d camFromRigTranslation)
	: ISizedAutoDiffFunctor
{
	private static readonly int[] Sizes = [3, 3, 1];

	private readonly Vector3d _camFromPoint3DDir = camFromPoint3DDir;
	private readonly Vector3d _camFromRigTranslation = camFromRigTranslation;

	/// <inheritdoc/>
	public int NumResiduals => 3;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Vector3d camFromPoint3DDir, Vector3d camFromRigTranslation) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(
			new RigBATAPairwiseDirectionConstantRigCostFunctor(camFromPoint3DDir, camFromRigTranslation));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		T scale = parameters[6];
		Vector3T<T> r = _camFromPoint3DDir
			- scale * (Vector3T<T>.Map(parameters) - Vector3T<T>.Map(parameters[3..]) + _camFromRigTranslation);
		r.CopyTo(residuals);
		return true;
	}
}

/// <summary>
/// Computes the error between a translation direction and the direction formed from a
/// camera (c) and 3D point (p) with variable rig extrinsics, such that
/// t_ij - scale * (p - c + t_rig) is minimized. Blocks: point3D (3), rig_in_world (3),
/// cam_in_rig (3), scale (1). Port of colmap::RigBATAPairwiseDirectionCostFunctor.
/// </summary>
public readonly struct RigBATAPairwiseDirectionCostFunctor(Vector3d camFromPoint3DDir, Quaterniond rigFromWorldRot)
	: ISizedAutoDiffFunctor
{
	private static readonly int[] Sizes = [3, 3, 3, 1];

	private readonly Vector3d _camFromPoint3DDir = camFromPoint3DDir;
	private readonly Quaterniond _worldFromRigRot = rigFromWorldRot.Inverse();

	/// <inheritdoc/>
	public int NumResiduals => 3;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Vector3d camFromPoint3DDir, Quaterniond rigFromWorldRot) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new RigBATAPairwiseDirectionCostFunctor(camFromPoint3DDir, rigFromWorldRot));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		Vector3T<T> camFromRigTranslation = QuaternionT<T>.From(_worldFromRigRot) * Vector3T<T>.Map(parameters[6..]);
		T scale = parameters[9];
		Vector3T<T> r = _camFromPoint3DDir
			- scale * (Vector3T<T>.Map(parameters) - Vector3T<T>.Map(parameters[3..]) - camFromRigTranslation);
		r.CopyTo(residuals);
		return true;
	}
}
