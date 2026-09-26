// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PosePriorCostFunctions: colmap/estimators/cost_functions/pose_prior.h - the autodiff
// functors that tie poses to priors: AbsolutePosePriorCostFunctor (6-DoF),
// AbsolutePosePositionPriorCostFunctor and AbsoluteRigPosePositionPriorCostFunctor
// (3-DoF position), and RelativePosePriorCostFunctor (6-DoF relative). Bundle adjustment
// and pose refinement wrap the position priors in CovarianceWeightedCostFunctor
// (CostFunctionUtils.cs). Pose blocks use the Rigid3d params layout
// [qx, qy, qz, qw, tx, ty, tz]; the quaternion algebra is QuaternionT.cs and the log map
// QuaternionUtils.AngleAxisFromEigenQuaternion. Tests:
// ColmapSharp.Tests/Estimators/CostFunctions/PosePriorTests.cs. Tier B.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>
/// 6-DoF error on the absolute sensor pose. The residual is the log of the error pose,
/// splitting SE(3) into SO(3) x R^3. The residual is computed in the sensor frame. Its first
/// and last three components correspond to the rotation and translation errors,
/// respectively. Block: sensor_from_world (7). Port of colmap::AbsolutePosePriorCostFunctor.
/// </summary>
public readonly struct AbsolutePosePriorCostFunctor(Rigid3d sensorFromWorldPrior) : ISizedAutoDiffFunctor
{
	private static readonly int[] Sizes = [7];

	private readonly Rigid3d _worldFromSensorPrior = sensorFromWorldPrior.Inverse();

	/// <inheritdoc/>
	public int NumResiduals => 6;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Rigid3d sensorFromWorldPrior) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new AbsolutePosePriorCostFunctor(sensorFromWorldPrior));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		QuaternionT<T> sensorFromWorldRotation = QuaternionT<T>.Map(parameters);
		QuaternionT<T> paramFromPriorRotation = sensorFromWorldRotation * _worldFromSensorPrior.Rotation;
		ReadOnlySpan<T> coeffs = [paramFromPriorRotation.X, paramFromPriorRotation.Y, paramFromPriorRotation.Z, paramFromPriorRotation.W];
		QuaternionUtils.AngleAxisFromEigenQuaternion(coeffs, residuals);

		Vector3T<T> paramFromPriorTranslation =
			Vector3T<T>.Map(parameters[4..]) + sensorFromWorldRotation * _worldFromSensorPrior.Translation;
		paramFromPriorTranslation.CopyTo(residuals[3..]);
		return true;
	}
}

/// <summary>
/// 3-DoF error on the sensor position in the world coordinate frame. Block:
/// sensor_from_world (7). Port of colmap::AbsolutePosePositionPriorCostFunctor.
/// </summary>
public readonly struct AbsolutePosePositionPriorCostFunctor(Vector3d positionInWorldPrior) : ISizedAutoDiffFunctor
{
	private static readonly int[] Sizes = [7];

	private readonly Vector3d _positionInWorldPrior = positionInWorldPrior;

	/// <inheritdoc/>
	public int NumResiduals => 3;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Vector3d positionInWorldPrior) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new AbsolutePosePositionPriorCostFunctor(positionInWorldPrior));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		Vector3T<T> r = _positionInWorldPrior
			+ QuaternionT<T>.Map(parameters).Inverse() * Vector3T<T>.Map(parameters[4..]);
		r.CopyTo(residuals);
		return true;
	}
}

/// <summary>
/// 3-DoF error on the rig sensor position in the world coordinate frame. Blocks:
/// sensor_from_rig (7), rig_from_world (7).
/// Port of colmap::AbsoluteRigPosePositionPriorCostFunctor.
/// </summary>
public readonly struct AbsoluteRigPosePositionPriorCostFunctor(Vector3d positionInWorldPrior) : ISizedAutoDiffFunctor
{
	private static readonly int[] Sizes = [7, 7];

	private readonly Vector3d _positionInWorldPrior = positionInWorldPrior;

	/// <inheritdoc/>
	public int NumResiduals => 3;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Vector3d positionInWorldPrior) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new AbsoluteRigPosePositionPriorCostFunctor(positionInWorldPrior));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		QuaternionT<T> sensorFromRigRotation = QuaternionT<T>.Map(parameters);
		QuaternionT<T> sensorFromWorldRotation = sensorFromRigRotation * QuaternionT<T>.Map(parameters[7..]);
		Vector3T<T> sensorFromWorldTranslation =
			Vector3T<T>.Map(parameters[4..]) + sensorFromRigRotation * Vector3T<T>.Map(parameters[11..]);
		Vector3T<T> r = _positionInWorldPrior + sensorFromWorldRotation.Inverse() * sensorFromWorldTranslation;
		r.CopyTo(residuals);
		return true;
	}
}

/// <summary>
/// 6-DoF error between two absolute camera poses based on a prior on their relative pose,
/// with identical scale for the translation. The residual is computed in the frame of
/// camera i. Its first and last three components correspond to the rotation and
/// translation errors, respectively.
/// <para>
/// Derivation: i_T_w = ΔT_i·i_T_j·j_T_w, where ΔT_i = exp(η_i) is the residual in SE(3) and
/// η_i in tangent space. Thus η_i = log(i_T_w·j_T_w⁻¹·j_T_i). Rotation term:
/// ΔR = log(i_R_w·j_R_w⁻¹·j_R_i). Translation term: Δt = i_t_w + i_R_w·j_R_w⁻¹·(j_t_i - j_t_w).
/// </para>
/// Blocks: i_from_world (7), j_from_world (7). Port of colmap::RelativePosePriorCostFunctor.
/// </summary>
public readonly struct RelativePosePriorCostFunctor(Rigid3d iFromJPrior) : ISizedAutoDiffFunctor
{
	private static readonly int[] Sizes = [7, 7];

	private readonly Rigid3d _jFromIPrior = iFromJPrior.Inverse();

	/// <inheritdoc/>
	public int NumResiduals => 6;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Rigid3d iFromJPrior) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new RelativePosePriorCostFunctor(iFromJPrior));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		QuaternionT<T> iFromJRotation = QuaternionT<T>.Map(parameters) * QuaternionT<T>.Map(parameters[7..]).Inverse();
		QuaternionT<T> paramFromPriorRotation = iFromJRotation * _jFromIPrior.Rotation;
		ReadOnlySpan<T> coeffs = [paramFromPriorRotation.X, paramFromPriorRotation.Y, paramFromPriorRotation.Z, paramFromPriorRotation.W];
		QuaternionUtils.AngleAxisFromEigenQuaternion(coeffs, residuals);

		Vector3T<T> jFromIPriorTranslation = _jFromIPrior.Translation - Vector3T<T>.Map(parameters[11..]);
		Vector3T<T> paramFromPriorTranslation = Vector3T<T>.Map(parameters[4..]) + iFromJRotation * jFromIPriorTranslation;
		paramFromPriorTranslation.CopyTo(residuals[3..]);
		return true;
	}
}
