// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AlignmentCostFunctions: colmap/estimators/cost_functions/alignment.h -
// Point3DAlignmentCostFunctor, the Sim3 point-alignment residual that reconstruction
// alignment (estimators/alignment.cc) minimizes. The Sim3d block uses COLMAP's Sim3d params
// layout [qx, qy, qz, qw, tx, ty, tz, scale]. Tests:
// ColmapSharp.Tests/Estimators/CostFunctions/AlignmentTests.cs. Tier B.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>
/// Cost function for aligning one 3D point with a reference 3D point. The residual is
/// computed in frame b. Coordinate transformation convention is equivalent to Sim3d.
/// With <c>useLogScale</c> the scale parameter is the log of the scale. Blocks:
/// point_in_a (3), b_from_a (8). Port of colmap::Point3DAlignmentCostFunctor.
/// </summary>
public readonly struct Point3DAlignmentCostFunctor(Vector3d pointInBPrior, bool useLogScale = true) : ISizedAutoDiffFunctor
{
	private static readonly int[] Sizes = [3, 8];

	private readonly Vector3d _pointInBPrior = pointInBPrior;
	private readonly bool _useLogScale = useLogScale;

	/// <inheritdoc/>
	public int NumResiduals => 3;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(Vector3d pointInBPrior, bool useLogScale = true) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new Point3DAlignmentCostFunctor(pointInBPrior, useLogScale));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		ReadOnlySpan<T> bFromA = parameters[3..];

		// Select whether to exponentiate.
		T bFromAScale = _useLogScale ? T.Exp(bFromA[7]) : bFromA[7];

		Vector3T<T> pointInB =
			QuaternionT<T>.Map(bFromA) * Vector3T<T>.Map(parameters) * bFromAScale + Vector3T<T>.Map(bFromA[4..]);
		(pointInB - _pointInBPrior).CopyTo(residuals);
		return true;
	}
}
