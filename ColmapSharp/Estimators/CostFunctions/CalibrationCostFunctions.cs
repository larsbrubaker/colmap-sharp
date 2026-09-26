// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CalibrationCostFunctions: colmap/estimators/cost_functions/calibration.h - the Fetzer
// focal-length residuals view-graph calibration (estimators/view_graph_calibration.cc)
// minimizes: ComputeFetzerPolynomialCoefficients, DecomposeFundamentalMatrixForFetzer,
// ComputeFetzerResidual1/2, FetzerFocalLengthCostFunctor and
// FetzerFocalLengthSameCameraCostFunctor. The SVD is LinearAlgebra/SvdFixed.cs (Eigen's
// JacobiSVD semantics). Tests: ColmapSharp.Tests/Estimators/CostFunctions/CalibrationTests.cs.
// Tier B (the SVD).
//
// See: "Stable Intrinsic Auto-Calibration from Fundamental Matrices of Devices with
// Uncorrelated Camera Parameters", Fetzer et al., WACV 2020.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>The free functions of calibration.h.</summary>
public static class FetzerCalibration
{
	/// <summary>
	/// Polynomial coefficients from cross-products of SVD-derived vectors for the Fetzer
	/// focal length estimation method. The coefficients encode the relationship between the
	/// two focal lengths derived from the fundamental matrix constraint.
	/// Port of colmap::ComputeFetzerPolynomialCoefficients.
	/// </summary>
	public static Vector4d ComputeFetzerPolynomialCoefficients(Vector3d ai, Vector3d bi, Vector3d aj, Vector3d bj, int u, int v) =>
		new(
			ai[u] * aj[v] - ai[v] * aj[u],
			ai[u] * bj[v] - ai[v] * bj[u],
			bi[u] * aj[v] - bi[v] * aj[u],
			bi[u] * bj[v] - bi[v] * bj[u]);

	/// <summary>
	/// Decomposes the fundamental matrix (adjusted by principal points) via SVD and computes
	/// the two coefficient vectors used to estimate the two focal lengths.
	/// Port of colmap::DecomposeFundamentalMatrixForFetzer.
	/// </summary>
	public static (Vector4d D01, Vector4d D12) DecomposeFundamentalMatrixForFetzer(
		in Matrix3d i1FI0, Vector2d principalPoint0, Vector2d principalPoint1)
	{
		var k0 = new Matrix3d(1, 0, principalPoint0.X, 0, 1, principalPoint0.Y, 0, 0, 1);
		var k1 = new Matrix3d(1, 0, principalPoint1.X, 0, 1, principalPoint1.Y, 0, 0, 1);

		// Factoring out the principal points before the SVD appears to be numerically more
		// stable than the method described in the paper.
		Matrix3d i1GI0 = k1.Transpose() * i1FI0 * k0;

		Svd3d svd = Svd3d.Compute(i1GI0);
		Vector3d s = svd.SingularValues;

		Vector3d v0 = svd.MatrixV.Col(0);
		Vector3d v1 = svd.MatrixV.Col(1);

		Vector3d u0 = svd.MatrixU.Col(0);
		Vector3d u1 = svd.MatrixU.Col(1);

		// Equation 11. Notice there is a sign error in the paper. Equation 8 shows the sign
		// in aj(1) and bj(1) correctly.
		var ai = new Vector3d(
			s[0] * s[0] * (v0[0] * v0[0] + v0[1] * v0[1]),
			s[0] * s[1] * (v0[0] * v1[0] + v0[1] * v1[1]),
			s[1] * s[1] * (v1[0] * v1[0] + v1[1] * v1[1]));

		var aj = new Vector3d(
			u1[0] * u1[0] + u1[1] * u1[1],
			-(u0[0] * u1[0] + u0[1] * u1[1]),
			u0[0] * u0[0] + u0[1] * u0[1]);

		var bi = new Vector3d(
			s[0] * s[0] * v0[2] * v0[2],
			s[0] * s[1] * v0[2] * v1[2],
			s[1] * s[1] * v1[2] * v1[2]);

		var bj = new Vector3d(u1[2] * u1[2], -(u0[2] * u1[2]), u0[2] * u0[2]);

		// Equation 12. Experiments showed that the d02 term is not useful. The d10, d21 and
		// d20 are redundant to d01, d12, d02.
		Vector4d d01 = ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 1, 0);
		Vector4d d12 = ComputeFetzerPolynomialCoefficients(ai, bi, aj, bj, 2, 1);
		return (d01, d12);
	}

	/// <summary>Equation 13. Port of colmap::ComputeFetzerResidual1.</summary>
	public static T ComputeFetzerResidual1<T>(Vector4d d, in T fiSq, in T fjSq)
		where T : struct, IScalar<T>
	{
		T denom = fjSq * d[0] + d[1];
		denom = T.ScalarPart(denom) == 0 ? T.FromDouble(1e-6) : denom;
		T k1 = -(fjSq * d[2] + d[3]) / denom;
		return (fiSq - k1) / fiSq;
	}

	/// <summary>Equation 14. Port of colmap::ComputeFetzerResidual2.</summary>
	public static T ComputeFetzerResidual2<T>(Vector4d d, in T fiSq, in T fjSq)
		where T : struct, IScalar<T>
	{
		T denom = fiSq * d[0] + d[2];
		denom = T.ScalarPart(denom) == 0 ? T.FromDouble(1e-6) : denom;
		T k2 = -(fiSq * d[1] + d[3]) / denom;
		return (fjSq - k2) / fjSq;
	}
}

/// <summary>
/// Cost functor for estimating focal lengths from the fundamental matrix using the Fetzer
/// method. Used when two images have different cameras (different focal lengths). The
/// residual measures the relative error between the estimated and expected focal lengths
/// based on the fundamental matrix constraint. Blocks: focal_length_i (1),
/// focal_length_j (1). Port of colmap::FetzerFocalLengthCostFunctor.
/// </summary>
public readonly struct FetzerFocalLengthCostFunctor : ISizedAutoDiffFunctor
{
	private static readonly int[] Sizes = [1, 1];

	private readonly Vector4d _d01;
	private readonly Vector4d _d12;

	/// <summary>Creates the functor from j_F_i and the two principal points.</summary>
	public FetzerFocalLengthCostFunctor(in Matrix3d jFI, Vector2d principalPointI, Vector2d principalPointJ)
	{
		(_d01, _d12) = FetzerCalibration.DecomposeFundamentalMatrixForFetzer(jFI, principalPointI, principalPointJ);
	}

	/// <inheritdoc/>
	public int NumResiduals => 2;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(in Matrix3d jFI, Vector2d principalPointI, Vector2d principalPointJ) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new FetzerFocalLengthCostFunctor(jFI, principalPointI, principalPointJ));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		T fiSq = parameters[0] * parameters[0];
		T fjSq = parameters[1] * parameters[1];
		residuals[0] = FetzerCalibration.ComputeFetzerResidual1(_d01, fiSq, fjSq);
		residuals[1] = FetzerCalibration.ComputeFetzerResidual2(_d12, fiSq, fjSq);
		return true;
	}
}

/// <summary>
/// Cost functor for estimating the focal length from the fundamental matrix using the
/// Fetzer method. Used when two images share the same camera (same focal length). Block:
/// focal_length (1). Port of colmap::FetzerFocalLengthSameCameraCostFunctor.
/// </summary>
public readonly struct FetzerFocalLengthSameCameraCostFunctor : ISizedAutoDiffFunctor
{
	private static readonly int[] Sizes = [1];

	private readonly Vector4d _d01;
	private readonly Vector4d _d12;

	/// <summary>Creates the functor from j_F_i and the shared principal point.</summary>
	public FetzerFocalLengthSameCameraCostFunctor(in Matrix3d jFI, Vector2d principalPoint)
	{
		(_d01, _d12) = FetzerCalibration.DecomposeFundamentalMatrixForFetzer(jFI, principalPoint, principalPoint);
	}

	/// <inheritdoc/>
	public int NumResiduals => 2;

	/// <inheritdoc/>
	public int[] ParameterBlockSizes => Sizes;

	/// <summary>The autodiff cost function.</summary>
	public static CostFunction Create(in Matrix3d jFI, Vector2d principalPoint) =>
		CostFunctionUtils.CreateAutoDiffCostFunction(new FetzerFocalLengthSameCameraCostFunctor(jFI, principalPoint));

	/// <inheritdoc/>
	public bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		T fSq = parameters[0] * parameters[0];
		residuals[0] = FetzerCalibration.ComputeFetzerResidual1(_d01, fSq, fSq);
		residuals[1] = FetzerCalibration.ComputeFetzerResidual2(_d12, fSq, fSq);
		return true;
	}
}
