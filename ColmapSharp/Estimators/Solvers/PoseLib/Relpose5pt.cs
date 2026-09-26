// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08.
//
// Relpose5pt: PoseLib/solvers/relpose_5pt.cc (poselib::relpose_5pt, the essential-matrix
// overload COLMAP calls) - the minimal five-point relative pose solver of D. Nister, "An
// Efficient Solution to the Five-Point Relative Pose Problem", PAMI 2004. The minimal case of
// COLMAP's EssentialMatrixFivePointEstimator (Estimators/Solvers/EssentialMatrixEstimators.cs).
// The CameraPose overload (which decomposes each E with misc/essential.cc's
// motion_from_essential, in Essential.cs) is not ported: COLMAP does its own decomposition.
//
// Steps: the 4D null space of the five epipolar constraints; the ten cubic constraints
// (determinant plus the trace constraint) in its coefficients (compute_trace_constraints);
// Gauss-Jordan elimination with a 10 x 10 LU solve; the degree-10 determinant polynomial of
// the resulting 3 x 3 polynomial matrix, whose real roots Sturm.cs brackets; back
// substitution for each root.
//
// The null space comes from an unpivoted Householder QR instead of PoseLib's
// fullPivHouseholderQr (docs/CPP_DIVERGENCES.md, entry 27): same space, different basis, same
// solution set. Tier B. Allocation-free apart from the output list (and the rare
// three-row fallback solve of the back substitution).

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Estimators.Solvers.PoseLib;

/// <summary>The five-point essential matrix solver. Port of poselib::relpose_5pt.</summary>
public static class Relpose5pt
{
	/// <summary>
	/// Computes the essential matrices from five point correspondences (bearings
	/// <paramref name="x1"/>, <paramref name="x2"/>, with x2^T E x1 = 0), appends them to
	/// <paramref name="essentialMatrices"/> (not cleared, as in PoseLib) and returns their
	/// count. Every E has unit Frobenius norm.
	/// </summary>
	public static int Solve(ReadOnlySpan<Vector3d> x1, ReadOnlySpan<Vector3d> x2, List<Matrix3d> essentialMatrices)
	{
		// Compute nullspace to epipolar constraints. Column i holds x1[i](k) * x2[i](r) at
		// 3k + r, the coefficient of E(r, k) in column-major order.
		Span<double> epipolarConstraints = stackalloc double[9 * 5];
		for (int i = 0; i < 5; ++i)
		{
			Span<double> col = epipolarConstraints.Slice(9 * i, 9);
			for (int k = 0; k < 3; ++k)
			{
				for (int r = 0; r < 3; ++r)
				{
					col[3 * k + r] = x1[i][k] * x2[i][r];
				}
			}
		}

		// The last four columns of the full Q of the 9 x 5 system are orthogonal to every
		// constraint. N (4 x 9, column-major) holds them as rows: N[4k + j] is basis vector j's
		// coefficient of E element k.
		Span<double> tau = stackalloc double[5];
		Householder.FactorInPlace(epipolarConstraints, 9, 5, tau);
		Span<double> n = stackalloc double[4 * 9];
		Span<double> q = stackalloc double[9];
		for (int j = 0; j < 4; ++j)
		{
			q.Clear();
			q[5 + j] = 1;
			Householder.ApplyQ(epipolarConstraints, 9, tau, q);
			for (int k = 0; k < 9; ++k)
			{
				n[4 * k + j] = q[k];
			}
		}

		// Compute equation coefficients for the trace constraints + determinant.
		Span<double> coeffs = stackalloc double[10 * 20];
		ComputeTraceConstraints(n, coeffs);

		// coeffs.block<10, 10>(0, 10) = coeffs.block<10, 10>(0, 0).partialPivLu().solve(
		//     coeffs.block<10, 10>(0, 10)); both blocks are contiguous column-major spans.
		Span<double> lu = stackalloc double[10 * 10];
		coeffs[..100].CopyTo(lu);
		Span<int> permutation = stackalloc int[10];
		PartialPivLU.FactorInPlace(lu, 10, permutation);
		Span<double> rhs = stackalloc double[10];
		for (int col = 10; col < 20; ++col)
		{
			Span<double> column = coeffs.Slice(10 * col, 10);
			column.CopyTo(rhs);
			PartialPivLU.SolveInPlace(lu, 10, permutation, rhs, column);
		}

		// Perform eliminations using the 6 bottom rows.
		var A = new Matrix3xNView(stackalloc double[3 * 13]);
		for (int i = 0; i < 3; ++i)
		{
			A[i, 0] = 0.0;
			A[i, 4] = 0.0;
			A[i, 8] = 0.0;
			for (int k = 0; k < 3; ++k)
			{
				A[i, 1 + k] = Coeff(coeffs, 4 + 2 * i, 10 + k);
				A[i, 5 + k] = Coeff(coeffs, 4 + 2 * i, 13 + k);
			}

			for (int k = 0; k < 4; ++k)
			{
				A[i, 9 + k] = Coeff(coeffs, 4 + 2 * i, 16 + k);
			}

			for (int k = 0; k < 3; ++k)
			{
				A[i, k] -= Coeff(coeffs, 5 + 2 * i, 10 + k);
				A[i, 4 + k] -= Coeff(coeffs, 5 + 2 * i, 13 + k);
			}

			for (int k = 0; k < 4; ++k)
			{
				A[i, 8 + k] -= Coeff(coeffs, 5 + 2 * i, 16 + k);
			}
		}

		// Compute degree 10 poly representing determinant (equation 14 in the paper).
		Span<double> c = stackalloc double[11];
		c[0] = A[0, 12] * A[1, 3] * A[2, 7] - A[0, 12] * A[1, 7] * A[2, 3] - A[0, 3] * A[2, 7] * A[1, 12] +
			A[0, 7] * A[2, 3] * A[1, 12] + A[0, 3] * A[1, 7] * A[2, 12] - A[0, 7] * A[1, 3] * A[2, 12];
		c[1] = A[0, 11] * A[1, 3] * A[2, 7] - A[0, 11] * A[1, 7] * A[2, 3] + A[0, 12] * A[1, 2] * A[2, 7] +
			A[0, 12] * A[1, 3] * A[2, 6] - A[0, 12] * A[1, 6] * A[2, 3] - A[0, 12] * A[1, 7] * A[2, 2] -
			A[0, 2] * A[2, 7] * A[1, 12] - A[0, 3] * A[2, 6] * A[1, 12] - A[0, 3] * A[2, 7] * A[1, 11] +
			A[0, 6] * A[2, 3] * A[1, 12] + A[0, 7] * A[2, 2] * A[1, 12] + A[0, 7] * A[2, 3] * A[1, 11] +
			A[0, 2] * A[1, 7] * A[2, 12] + A[0, 3] * A[1, 6] * A[2, 12] + A[0, 3] * A[1, 7] * A[2, 11] -
			A[0, 6] * A[1, 3] * A[2, 12] - A[0, 7] * A[1, 2] * A[2, 12] - A[0, 7] * A[1, 3] * A[2, 11];
		c[2] = A[0, 10] * A[1, 3] * A[2, 7] - A[0, 10] * A[1, 7] * A[2, 3] + A[0, 11] * A[1, 2] * A[2, 7] +
			A[0, 11] * A[1, 3] * A[2, 6] - A[0, 11] * A[1, 6] * A[2, 3] - A[0, 11] * A[1, 7] * A[2, 2] +
			A[1, 1] * A[0, 12] * A[2, 7] + A[0, 12] * A[1, 2] * A[2, 6] + A[0, 12] * A[1, 3] * A[2, 5] -
			A[0, 12] * A[1, 5] * A[2, 3] - A[0, 12] * A[1, 6] * A[2, 2] - A[0, 12] * A[1, 7] * A[2, 1] -
			A[0, 1] * A[2, 7] * A[1, 12] - A[0, 2] * A[2, 6] * A[1, 12] - A[0, 2] * A[2, 7] * A[1, 11] -
			A[0, 3] * A[2, 5] * A[1, 12] - A[0, 3] * A[2, 6] * A[1, 11] - A[0, 3] * A[2, 7] * A[1, 10] +
			A[0, 5] * A[2, 3] * A[1, 12] + A[0, 6] * A[2, 2] * A[1, 12] + A[0, 6] * A[2, 3] * A[1, 11] +
			A[0, 7] * A[2, 1] * A[1, 12] + A[0, 7] * A[2, 2] * A[1, 11] + A[0, 7] * A[2, 3] * A[1, 10] +
			A[0, 1] * A[1, 7] * A[2, 12] + A[0, 2] * A[1, 6] * A[2, 12] + A[0, 2] * A[1, 7] * A[2, 11] +
			A[0, 3] * A[1, 5] * A[2, 12] + A[0, 3] * A[1, 6] * A[2, 11] + A[0, 3] * A[1, 7] * A[2, 10] -
			A[0, 5] * A[1, 3] * A[2, 12] - A[0, 6] * A[1, 2] * A[2, 12] - A[0, 6] * A[1, 3] * A[2, 11] -
			A[0, 7] * A[1, 1] * A[2, 12] - A[0, 7] * A[1, 2] * A[2, 11] - A[0, 7] * A[1, 3] * A[2, 10];
		c[3] = A[0, 3] * A[1, 7] * A[2, 9] - A[0, 3] * A[1, 9] * A[2, 7] - A[0, 7] * A[1, 3] * A[2, 9] +
			A[0, 7] * A[1, 9] * A[2, 3] + A[0, 9] * A[1, 3] * A[2, 7] - A[0, 9] * A[1, 7] * A[2, 3] +
			A[0, 10] * A[1, 2] * A[2, 7] + A[0, 10] * A[1, 3] * A[2, 6] - A[0, 10] * A[1, 6] * A[2, 3] -
			A[0, 10] * A[1, 7] * A[2, 2] + A[1, 0] * A[0, 12] * A[2, 7] + A[0, 11] * A[1, 1] * A[2, 7] +
			A[0, 11] * A[1, 2] * A[2, 6] + A[0, 11] * A[1, 3] * A[2, 5] - A[0, 11] * A[1, 5] * A[2, 3] -
			A[0, 11] * A[1, 6] * A[2, 2] - A[0, 11] * A[1, 7] * A[2, 1] + A[1, 1] * A[0, 12] * A[2, 6] +
			A[0, 12] * A[1, 2] * A[2, 5] + A[0, 12] * A[1, 3] * A[2, 4] - A[0, 12] * A[1, 4] * A[2, 3] -
			A[0, 12] * A[1, 5] * A[2, 2] - A[0, 12] * A[1, 6] * A[2, 1] - A[0, 12] * A[1, 7] * A[2, 0] -
			A[0, 0] * A[2, 7] * A[1, 12] - A[0, 1] * A[2, 6] * A[1, 12] - A[0, 1] * A[2, 7] * A[1, 11] -
			A[0, 2] * A[2, 5] * A[1, 12] - A[0, 2] * A[2, 6] * A[1, 11] - A[0, 2] * A[2, 7] * A[1, 10] -
			A[0, 3] * A[2, 4] * A[1, 12] - A[0, 3] * A[2, 5] * A[1, 11] - A[0, 3] * A[2, 6] * A[1, 10] +
			A[0, 4] * A[2, 3] * A[1, 12] + A[0, 5] * A[2, 2] * A[1, 12] + A[0, 5] * A[2, 3] * A[1, 11] +
			A[0, 6] * A[2, 1] * A[1, 12] + A[0, 6] * A[2, 2] * A[1, 11] + A[0, 6] * A[2, 3] * A[1, 10] +
			A[0, 7] * A[2, 0] * A[1, 12] + A[0, 7] * A[2, 1] * A[1, 11] + A[0, 7] * A[2, 2] * A[1, 10] +
			A[0, 0] * A[1, 7] * A[2, 12] + A[0, 1] * A[1, 6] * A[2, 12] + A[0, 1] * A[1, 7] * A[2, 11] +
			A[0, 2] * A[1, 5] * A[2, 12] + A[0, 2] * A[1, 6] * A[2, 11] + A[0, 2] * A[1, 7] * A[2, 10] +
			A[0, 3] * A[1, 4] * A[2, 12] + A[0, 3] * A[1, 5] * A[2, 11] + A[0, 3] * A[1, 6] * A[2, 10] -
			A[0, 4] * A[1, 3] * A[2, 12] - A[0, 5] * A[1, 2] * A[2, 12] - A[0, 5] * A[1, 3] * A[2, 11] -
			A[0, 6] * A[1, 1] * A[2, 12] - A[0, 6] * A[1, 2] * A[2, 11] - A[0, 6] * A[1, 3] * A[2, 10] -
			A[0, 7] * A[1, 0] * A[2, 12] - A[0, 7] * A[1, 1] * A[2, 11] - A[0, 7] * A[1, 2] * A[2, 10];
		c[4] = A[0, 2] * A[1, 7] * A[2, 9] - A[0, 2] * A[1, 9] * A[2, 7] + A[0, 3] * A[1, 6] * A[2, 9] +
			A[0, 3] * A[1, 7] * A[2, 8] - A[0, 3] * A[1, 8] * A[2, 7] - A[0, 3] * A[1, 9] * A[2, 6] -
			A[0, 6] * A[1, 3] * A[2, 9] + A[0, 6] * A[1, 9] * A[2, 3] - A[0, 7] * A[1, 2] * A[2, 9] -
			A[0, 7] * A[1, 3] * A[2, 8] + A[0, 7] * A[1, 8] * A[2, 3] + A[0, 7] * A[1, 9] * A[2, 2] +
			A[0, 8] * A[1, 3] * A[2, 7] - A[0, 8] * A[1, 7] * A[2, 3] + A[0, 9] * A[1, 2] * A[2, 7] +
			A[0, 9] * A[1, 3] * A[2, 6] - A[0, 9] * A[1, 6] * A[2, 3] - A[0, 9] * A[1, 7] * A[2, 2] +
			A[0, 10] * A[1, 1] * A[2, 7] + A[0, 10] * A[1, 2] * A[2, 6] + A[0, 10] * A[1, 3] * A[2, 5] -
			A[0, 10] * A[1, 5] * A[2, 3] - A[0, 10] * A[1, 6] * A[2, 2] - A[0, 10] * A[1, 7] * A[2, 1] +
			A[1, 0] * A[0, 11] * A[2, 7] + A[1, 0] * A[0, 12] * A[2, 6] + A[0, 11] * A[1, 1] * A[2, 6] +
			A[0, 11] * A[1, 2] * A[2, 5] + A[0, 11] * A[1, 3] * A[2, 4] - A[0, 11] * A[1, 4] * A[2, 3] -
			A[0, 11] * A[1, 5] * A[2, 2] - A[0, 11] * A[1, 6] * A[2, 1] - A[0, 11] * A[1, 7] * A[2, 0] +
			A[1, 1] * A[0, 12] * A[2, 5] + A[0, 12] * A[1, 2] * A[2, 4] - A[0, 12] * A[1, 4] * A[2, 2] -
			A[0, 12] * A[1, 5] * A[2, 1] - A[0, 12] * A[1, 6] * A[2, 0] - A[0, 0] * A[2, 6] * A[1, 12] -
			A[0, 0] * A[2, 7] * A[1, 11] - A[0, 1] * A[2, 5] * A[1, 12] - A[0, 1] * A[2, 6] * A[1, 11] -
			A[0, 1] * A[2, 7] * A[1, 10] - A[0, 2] * A[2, 4] * A[1, 12] - A[0, 2] * A[2, 5] * A[1, 11] -
			A[0, 2] * A[2, 6] * A[1, 10] - A[0, 3] * A[2, 4] * A[1, 11] - A[0, 3] * A[2, 5] * A[1, 10] +
			A[0, 4] * A[2, 2] * A[1, 12] + A[0, 4] * A[2, 3] * A[1, 11] + A[0, 5] * A[2, 1] * A[1, 12] +
			A[0, 5] * A[2, 2] * A[1, 11] + A[0, 5] * A[2, 3] * A[1, 10] + A[0, 6] * A[2, 0] * A[1, 12] +
			A[0, 6] * A[2, 1] * A[1, 11] + A[0, 6] * A[2, 2] * A[1, 10] + A[0, 7] * A[2, 0] * A[1, 11] +
			A[0, 7] * A[2, 1] * A[1, 10] + A[0, 0] * A[1, 6] * A[2, 12] + A[0, 0] * A[1, 7] * A[2, 11] +
			A[0, 1] * A[1, 5] * A[2, 12] + A[0, 1] * A[1, 6] * A[2, 11] + A[0, 1] * A[1, 7] * A[2, 10] +
			A[0, 2] * A[1, 4] * A[2, 12] + A[0, 2] * A[1, 5] * A[2, 11] + A[0, 2] * A[1, 6] * A[2, 10] +
			A[0, 3] * A[1, 4] * A[2, 11] + A[0, 3] * A[1, 5] * A[2, 10] - A[0, 4] * A[1, 2] * A[2, 12] -
			A[0, 4] * A[1, 3] * A[2, 11] - A[0, 5] * A[1, 1] * A[2, 12] - A[0, 5] * A[1, 2] * A[2, 11] -
			A[0, 5] * A[1, 3] * A[2, 10] - A[0, 6] * A[1, 0] * A[2, 12] - A[0, 6] * A[1, 1] * A[2, 11] -
			A[0, 6] * A[1, 2] * A[2, 10] - A[0, 7] * A[1, 0] * A[2, 11] - A[0, 7] * A[1, 1] * A[2, 10];
		c[5] = A[0, 1] * A[1, 7] * A[2, 9] - A[0, 1] * A[1, 9] * A[2, 7] + A[0, 2] * A[1, 6] * A[2, 9] +
			A[0, 2] * A[1, 7] * A[2, 8] - A[0, 2] * A[1, 8] * A[2, 7] - A[0, 2] * A[1, 9] * A[2, 6] +
			A[0, 3] * A[1, 5] * A[2, 9] + A[0, 3] * A[1, 6] * A[2, 8] - A[0, 3] * A[1, 8] * A[2, 6] -
			A[0, 3] * A[1, 9] * A[2, 5] - A[0, 5] * A[1, 3] * A[2, 9] + A[0, 5] * A[1, 9] * A[2, 3] -
			A[0, 6] * A[1, 2] * A[2, 9] - A[0, 6] * A[1, 3] * A[2, 8] + A[0, 6] * A[1, 8] * A[2, 3] +
			A[0, 6] * A[1, 9] * A[2, 2] - A[0, 7] * A[1, 1] * A[2, 9] - A[0, 7] * A[1, 2] * A[2, 8] +
			A[0, 7] * A[1, 8] * A[2, 2] + A[0, 7] * A[1, 9] * A[2, 1] + A[0, 8] * A[1, 2] * A[2, 7] +
			A[0, 8] * A[1, 3] * A[2, 6] - A[0, 8] * A[1, 6] * A[2, 3] - A[0, 8] * A[1, 7] * A[2, 2] +
			A[0, 9] * A[1, 1] * A[2, 7] + A[0, 9] * A[1, 2] * A[2, 6] + A[0, 9] * A[1, 3] * A[2, 5] -
			A[0, 9] * A[1, 5] * A[2, 3] - A[0, 9] * A[1, 6] * A[2, 2] - A[0, 9] * A[1, 7] * A[2, 1] +
			A[0, 10] * A[1, 0] * A[2, 7] + A[0, 10] * A[1, 1] * A[2, 6] + A[0, 10] * A[1, 2] * A[2, 5] +
			A[0, 10] * A[1, 3] * A[2, 4] - A[0, 10] * A[1, 4] * A[2, 3] - A[0, 10] * A[1, 5] * A[2, 2] -
			A[0, 10] * A[1, 6] * A[2, 1] - A[0, 10] * A[1, 7] * A[2, 0] + A[1, 0] * A[0, 11] * A[2, 6] +
			A[1, 0] * A[0, 12] * A[2, 5] + A[0, 11] * A[1, 1] * A[2, 5] + A[0, 11] * A[1, 2] * A[2, 4] -
			A[0, 11] * A[1, 4] * A[2, 2] - A[0, 11] * A[1, 5] * A[2, 1] - A[0, 11] * A[1, 6] * A[2, 0] +
			A[1, 1] * A[0, 12] * A[2, 4] - A[0, 12] * A[1, 4] * A[2, 1] - A[0, 12] * A[1, 5] * A[2, 0] -
			A[0, 0] * A[2, 5] * A[1, 12] - A[0, 0] * A[2, 6] * A[1, 11] - A[0, 0] * A[2, 7] * A[1, 10] -
			A[0, 1] * A[2, 4] * A[1, 12] - A[0, 1] * A[2, 5] * A[1, 11] - A[0, 1] * A[2, 6] * A[1, 10] -
			A[0, 2] * A[2, 4] * A[1, 11] - A[0, 2] * A[2, 5] * A[1, 10] - A[0, 3] * A[2, 4] * A[1, 10] +
			A[0, 4] * A[2, 1] * A[1, 12] + A[0, 4] * A[2, 2] * A[1, 11] + A[0, 4] * A[2, 3] * A[1, 10] +
			A[0, 5] * A[2, 0] * A[1, 12] + A[0, 5] * A[2, 1] * A[1, 11] + A[0, 5] * A[2, 2] * A[1, 10] +
			A[0, 6] * A[2, 0] * A[1, 11] + A[0, 6] * A[2, 1] * A[1, 10] + A[0, 7] * A[2, 0] * A[1, 10] +
			A[0, 0] * A[1, 5] * A[2, 12] + A[0, 0] * A[1, 6] * A[2, 11] + A[0, 0] * A[1, 7] * A[2, 10] +
			A[0, 1] * A[1, 4] * A[2, 12] + A[0, 1] * A[1, 5] * A[2, 11] + A[0, 1] * A[1, 6] * A[2, 10] +
			A[0, 2] * A[1, 4] * A[2, 11] + A[0, 2] * A[1, 5] * A[2, 10] + A[0, 3] * A[1, 4] * A[2, 10] -
			A[0, 4] * A[1, 1] * A[2, 12] - A[0, 4] * A[1, 2] * A[2, 11] - A[0, 4] * A[1, 3] * A[2, 10] -
			A[0, 5] * A[1, 0] * A[2, 12] - A[0, 5] * A[1, 1] * A[2, 11] - A[0, 5] * A[1, 2] * A[2, 10] -
			A[0, 6] * A[1, 0] * A[2, 11] - A[0, 6] * A[1, 1] * A[2, 10] - A[0, 7] * A[1, 0] * A[2, 10];
		c[6] = A[0, 0] * A[1, 7] * A[2, 9] - A[0, 0] * A[1, 9] * A[2, 7] + A[0, 1] * A[1, 6] * A[2, 9] +
			A[0, 1] * A[1, 7] * A[2, 8] - A[0, 1] * A[1, 8] * A[2, 7] - A[0, 1] * A[1, 9] * A[2, 6] +
			A[0, 2] * A[1, 5] * A[2, 9] + A[0, 2] * A[1, 6] * A[2, 8] - A[0, 2] * A[1, 8] * A[2, 6] -
			A[0, 2] * A[1, 9] * A[2, 5] + A[0, 3] * A[1, 4] * A[2, 9] + A[0, 3] * A[1, 5] * A[2, 8] -
			A[0, 3] * A[1, 8] * A[2, 5] - A[0, 3] * A[1, 9] * A[2, 4] - A[0, 4] * A[1, 3] * A[2, 9] +
			A[0, 4] * A[1, 9] * A[2, 3] - A[0, 5] * A[1, 2] * A[2, 9] - A[0, 5] * A[1, 3] * A[2, 8] +
			A[0, 5] * A[1, 8] * A[2, 3] + A[0, 5] * A[1, 9] * A[2, 2] - A[0, 6] * A[1, 1] * A[2, 9] -
			A[0, 6] * A[1, 2] * A[2, 8] + A[0, 6] * A[1, 8] * A[2, 2] + A[0, 6] * A[1, 9] * A[2, 1] -
			A[0, 7] * A[1, 0] * A[2, 9] - A[0, 7] * A[1, 1] * A[2, 8] + A[0, 7] * A[1, 8] * A[2, 1] +
			A[0, 7] * A[1, 9] * A[2, 0] + A[0, 8] * A[1, 1] * A[2, 7] + A[0, 8] * A[1, 2] * A[2, 6] +
			A[0, 8] * A[1, 3] * A[2, 5] - A[0, 8] * A[1, 5] * A[2, 3] - A[0, 8] * A[1, 6] * A[2, 2] -
			A[0, 8] * A[1, 7] * A[2, 1] + A[0, 9] * A[1, 0] * A[2, 7] + A[0, 9] * A[1, 1] * A[2, 6] +
			A[0, 9] * A[1, 2] * A[2, 5] + A[0, 9] * A[1, 3] * A[2, 4] - A[0, 9] * A[1, 4] * A[2, 3] -
			A[0, 9] * A[1, 5] * A[2, 2] - A[0, 9] * A[1, 6] * A[2, 1] - A[0, 9] * A[1, 7] * A[2, 0] +
			A[0, 10] * A[1, 0] * A[2, 6] + A[0, 10] * A[1, 1] * A[2, 5] + A[0, 10] * A[1, 2] * A[2, 4] -
			A[0, 10] * A[1, 4] * A[2, 2] - A[0, 10] * A[1, 5] * A[2, 1] - A[0, 10] * A[1, 6] * A[2, 0] +
			A[1, 0] * A[0, 11] * A[2, 5] + A[1, 0] * A[0, 12] * A[2, 4] + A[0, 11] * A[1, 1] * A[2, 4] -
			A[0, 11] * A[1, 4] * A[2, 1] - A[0, 11] * A[1, 5] * A[2, 0] - A[0, 12] * A[1, 4] * A[2, 0] -
			A[0, 0] * A[2, 4] * A[1, 12] - A[0, 0] * A[2, 5] * A[1, 11] - A[0, 0] * A[2, 6] * A[1, 10] -
			A[0, 1] * A[2, 4] * A[1, 11] - A[0, 1] * A[2, 5] * A[1, 10] - A[0, 2] * A[2, 4] * A[1, 10] +
			A[0, 4] * A[2, 0] * A[1, 12] + A[0, 4] * A[2, 1] * A[1, 11] + A[0, 4] * A[2, 2] * A[1, 10] +
			A[0, 5] * A[2, 0] * A[1, 11] + A[0, 5] * A[2, 1] * A[1, 10] + A[0, 6] * A[2, 0] * A[1, 10] +
			A[0, 0] * A[1, 4] * A[2, 12] + A[0, 0] * A[1, 5] * A[2, 11] + A[0, 0] * A[1, 6] * A[2, 10] +
			A[0, 1] * A[1, 4] * A[2, 11] + A[0, 1] * A[1, 5] * A[2, 10] + A[0, 2] * A[1, 4] * A[2, 10] -
			A[0, 4] * A[1, 0] * A[2, 12] - A[0, 4] * A[1, 1] * A[2, 11] - A[0, 4] * A[1, 2] * A[2, 10] -
			A[0, 5] * A[1, 0] * A[2, 11] - A[0, 5] * A[1, 1] * A[2, 10] - A[0, 6] * A[1, 0] * A[2, 10];
		c[7] = A[0, 0] * A[1, 6] * A[2, 9] + A[0, 0] * A[1, 7] * A[2, 8] - A[0, 0] * A[1, 8] * A[2, 7] -
			A[0, 0] * A[1, 9] * A[2, 6] + A[0, 1] * A[1, 5] * A[2, 9] + A[0, 1] * A[1, 6] * A[2, 8] -
			A[0, 1] * A[1, 8] * A[2, 6] - A[0, 1] * A[1, 9] * A[2, 5] + A[0, 2] * A[1, 4] * A[2, 9] +
			A[0, 2] * A[1, 5] * A[2, 8] - A[0, 2] * A[1, 8] * A[2, 5] - A[0, 2] * A[1, 9] * A[2, 4] +
			A[0, 3] * A[1, 4] * A[2, 8] - A[0, 3] * A[1, 8] * A[2, 4] - A[0, 4] * A[1, 2] * A[2, 9] -
			A[0, 4] * A[1, 3] * A[2, 8] + A[0, 4] * A[1, 8] * A[2, 3] + A[0, 4] * A[1, 9] * A[2, 2] -
			A[0, 5] * A[1, 1] * A[2, 9] - A[0, 5] * A[1, 2] * A[2, 8] + A[0, 5] * A[1, 8] * A[2, 2] +
			A[0, 5] * A[1, 9] * A[2, 1] - A[0, 6] * A[1, 0] * A[2, 9] - A[0, 6] * A[1, 1] * A[2, 8] +
			A[0, 6] * A[1, 8] * A[2, 1] + A[0, 6] * A[1, 9] * A[2, 0] - A[0, 7] * A[1, 0] * A[2, 8] +
			A[0, 7] * A[1, 8] * A[2, 0] + A[0, 8] * A[1, 0] * A[2, 7] + A[0, 8] * A[1, 1] * A[2, 6] +
			A[0, 8] * A[1, 2] * A[2, 5] + A[0, 8] * A[1, 3] * A[2, 4] - A[0, 8] * A[1, 4] * A[2, 3] -
			A[0, 8] * A[1, 5] * A[2, 2] - A[0, 8] * A[1, 6] * A[2, 1] - A[0, 8] * A[1, 7] * A[2, 0] +
			A[0, 9] * A[1, 0] * A[2, 6] + A[0, 9] * A[1, 1] * A[2, 5] + A[0, 9] * A[1, 2] * A[2, 4] -
			A[0, 9] * A[1, 4] * A[2, 2] - A[0, 9] * A[1, 5] * A[2, 1] - A[0, 9] * A[1, 6] * A[2, 0] +
			A[0, 10] * A[1, 0] * A[2, 5] + A[0, 10] * A[1, 1] * A[2, 4] - A[0, 10] * A[1, 4] * A[2, 1] -
			A[0, 10] * A[1, 5] * A[2, 0] + A[1, 0] * A[0, 11] * A[2, 4] - A[0, 11] * A[1, 4] * A[2, 0] -
			A[0, 0] * A[2, 4] * A[1, 11] - A[0, 0] * A[2, 5] * A[1, 10] - A[0, 1] * A[2, 4] * A[1, 10] +
			A[0, 4] * A[2, 0] * A[1, 11] + A[0, 4] * A[2, 1] * A[1, 10] + A[0, 5] * A[2, 0] * A[1, 10] +
			A[0, 0] * A[1, 4] * A[2, 11] + A[0, 0] * A[1, 5] * A[2, 10] + A[0, 1] * A[1, 4] * A[2, 10] -
			A[0, 4] * A[1, 0] * A[2, 11] - A[0, 4] * A[1, 1] * A[2, 10] - A[0, 5] * A[1, 0] * A[2, 10];
		c[8] = A[0, 0] * A[1, 5] * A[2, 9] + A[0, 0] * A[1, 6] * A[2, 8] - A[0, 0] * A[1, 8] * A[2, 6] -
			A[0, 0] * A[1, 9] * A[2, 5] + A[0, 1] * A[1, 4] * A[2, 9] + A[0, 1] * A[1, 5] * A[2, 8] -
			A[0, 1] * A[1, 8] * A[2, 5] - A[0, 1] * A[1, 9] * A[2, 4] + A[0, 2] * A[1, 4] * A[2, 8] -
			A[0, 2] * A[1, 8] * A[2, 4] - A[0, 4] * A[1, 1] * A[2, 9] - A[0, 4] * A[1, 2] * A[2, 8] +
			A[0, 4] * A[1, 8] * A[2, 2] + A[0, 4] * A[1, 9] * A[2, 1] - A[0, 5] * A[1, 0] * A[2, 9] -
			A[0, 5] * A[1, 1] * A[2, 8] + A[0, 5] * A[1, 8] * A[2, 1] + A[0, 5] * A[1, 9] * A[2, 0] -
			A[0, 6] * A[1, 0] * A[2, 8] + A[0, 6] * A[1, 8] * A[2, 0] + A[0, 8] * A[1, 0] * A[2, 6] +
			A[0, 8] * A[1, 1] * A[2, 5] + A[0, 8] * A[1, 2] * A[2, 4] - A[0, 8] * A[1, 4] * A[2, 2] -
			A[0, 8] * A[1, 5] * A[2, 1] - A[0, 8] * A[1, 6] * A[2, 0] + A[0, 9] * A[1, 0] * A[2, 5] +
			A[0, 9] * A[1, 1] * A[2, 4] - A[0, 9] * A[1, 4] * A[2, 1] - A[0, 9] * A[1, 5] * A[2, 0] +
			A[0, 10] * A[1, 0] * A[2, 4] - A[0, 10] * A[1, 4] * A[2, 0] - A[0, 0] * A[2, 4] * A[1, 10] +
			A[0, 4] * A[2, 0] * A[1, 10] + A[0, 0] * A[1, 4] * A[2, 10] - A[0, 4] * A[1, 0] * A[2, 10];
		c[9] = A[0, 0] * A[1, 4] * A[2, 9] + A[0, 0] * A[1, 5] * A[2, 8] - A[0, 0] * A[1, 8] * A[2, 5] -
			A[0, 0] * A[1, 9] * A[2, 4] + A[0, 1] * A[1, 4] * A[2, 8] - A[0, 1] * A[1, 8] * A[2, 4] -
			A[0, 4] * A[1, 0] * A[2, 9] - A[0, 4] * A[1, 1] * A[2, 8] + A[0, 4] * A[1, 8] * A[2, 1] +
			A[0, 4] * A[1, 9] * A[2, 0] - A[0, 5] * A[1, 0] * A[2, 8] + A[0, 5] * A[1, 8] * A[2, 0] +
			A[0, 8] * A[1, 0] * A[2, 5] + A[0, 8] * A[1, 1] * A[2, 4] - A[0, 8] * A[1, 4] * A[2, 1] -
			A[0, 8] * A[1, 5] * A[2, 0] + A[0, 9] * A[1, 0] * A[2, 4] - A[0, 9] * A[1, 4] * A[2, 0];
		c[10] = A[0, 0] * A[1, 4] * A[2, 8] - A[0, 0] * A[1, 8] * A[2, 4] - A[0, 4] * A[1, 0] * A[2, 8] +
			A[0, 4] * A[1, 8] * A[2, 0] + A[0, 8] * A[1, 0] * A[2, 4] - A[0, 8] * A[1, 4] * A[2, 0];

		// Solve for the roots using sturm bracketing.
		Span<double> roots = stackalloc double[10];
		int nSols = Sturm.BisectSturm(10, c, roots);

		// Back substitution to recover essential matrices.
		Span<double> e = stackalloc double[9];
		Span<double> bCol0 = stackalloc double[3];
		Span<double> bCol1 = stackalloc double[3];
		Span<double> b = stackalloc double[3];
		for (int i = 0; i < nSols; ++i)
		{
			double z = roots[i];
			double z2 = z * z;
			double z3 = z2 * z;
			double z4 = z2 * z2;

			for (int r = 0; r < 3; ++r)
			{
				bCol0[r] = A[r, 0] * z3 + A[r, 1] * z2 + A[r, 2] * z + A[r, 3];
				bCol1[r] = A[r, 4] * z3 + A[r, 5] * z2 + A[r, 6] * z + A[r, 7];
				b[r] = A[r, 8] * z4 + A[r, 9] * z3 + A[r, 10] * z2 + A[r, 11] * z + A[r, 12];
			}

			// We try to solve using top two rows.
			Matrix2d top = new Matrix2d(bCol0[0], bCol1[0], bCol0[1], bCol1[1]).Inverse();
			double xz0 = top[0, 0] * b[0] + top[0, 1] * b[1];
			double xz1 = top[1, 0] * b[0] + top[1, 1] * b[1];

			// If this fails we revert to more expensive QR solver using all three rows.
			if (Math.Abs(bCol0[2] * xz0 + bCol1[2] * xz1 - b[2]) > 1e-6)
			{
				var bigB = new MatrixXd(3, 2);
				var rhsB = new VectorXd(3);
				for (int r = 0; r < 3; ++r)
				{
					bigB[r, 0] = bCol0[r];
					bigB[r, 1] = bCol1[r];
					rhsB[r] = b[r];
				}

				VectorXd solution = new ColPivHouseholderQR(bigB).Solve(rhsB);
				xz0 = solution[0];
				xz1 = solution[1];
			}

			double x = -xz0, y = -xz1;

			// Since the rows of N are orthogonal unit vectors, we can normalize the
			// coefficients instead.
			double invNorm = 1.0 / Math.Sqrt(x * x + y * y + z * z + 1.0);
			for (int k = 0; k < 9; ++k)
			{
				e[k] = (n[4 * k] * x + n[4 * k + 1] * y + n[4 * k + 2] * z + n[4 * k + 3]) * invNorm;
			}

			// Eigen::Map<Matrix<double, 1, 9>>(E.data()): e is E column-major.
			essentialMatrices.Add(Matrix3d.FromColumnMajor(e));
		}

		return nSols;
	}

	// Entry (row, col) of the column-major 10 x 20 coefficient matrix.
	private static double Coeff(ReadOnlySpan<double> coeffs, int row, int col) => coeffs[row + 10 * col];

	/// <summary>
	/// The ten cubic constraints on the null-space coefficients [x, y, z, 1]: rows 0-8 the
	/// trace constraint 2 E E^T E - tr(E E^T) E = 0 (equation 22 of the paper, with the
	/// factor 1/2 folded in), row 9 det(E) = 0. Each row holds the 20 cubic monomials in
	/// Nister's order. Port of compute_trace_constraints.
	/// </summary>
	private static void ComputeTraceConstraints(ReadOnlySpan<double> n, Span<double> coeffs)
	{
		Span<double> d = stackalloc double[60];

		// Determinant constraint.
		Span<double> row = stackalloc double[20];

		O1(Ee(n, 0, 1), Ee(n, 1, 2), d);
		O1m(Ee(n, 0, 2), Ee(n, 1, 1), d);
		O2(d, Ee(n, 2, 0), row);

		O1(Ee(n, 0, 2), Ee(n, 1, 0), d);
		O1m(Ee(n, 0, 0), Ee(n, 1, 2), d);
		O2p(d, Ee(n, 2, 1), row);

		O1(Ee(n, 0, 0), Ee(n, 1, 1), d);
		O1m(Ee(n, 0, 1), Ee(n, 1, 0), d);
		O2p(d, Ee(n, 2, 2), row);

		SetRow(coeffs, 9, row);

		// EET[i][j] is the offset into d of the (symmetric) entry (i, j) of E E^T:
		// {{d, d + 10, d + 20}, {d + 10, d + 40, d + 30}, {d + 20, d + 30, d + 50}}.
		ReadOnlySpan<int> eet = [0, 10, 20, 10, 40, 30, 20, 30, 50];

		// Compute EE^T (equation 20 in paper).
		for (int i = 0; i < 3; ++i)
		{
			for (int j = i; j < 3; ++j)
			{
				Span<double> target = d.Slice(eet[3 * i + j], 10);
				O1(Ee(n, i, 0), Ee(n, j, 0), target);
				O1p(Ee(n, i, 1), Ee(n, j, 1), target);
				O1p(Ee(n, i, 2), Ee(n, j, 2), target);
			}
		}

		// Subtract trace (equation 22 in paper).
		for (int i = 0; i < 10; ++i)
		{
			double t = 0.5 * (d[eet[0] + i] + d[eet[4] + i] + d[eet[8] + i]);
			d[eet[0] + i] -= t;
			d[eet[4] + i] -= t;
			d[eet[8] + i] -= t;
		}

		int cnt = 0;
		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				O2(d.Slice(eet[3 * i], 10), Ee(n, 0, j), row);
				O2p(d.Slice(eet[3 * i + 1], 10), Ee(n, 1, j), row);
				O2p(d.Slice(eet[3 * i + 2], 10), Ee(n, 2, j), row);
				SetRow(coeffs, cnt++, row);
			}
		}
	}

	// PoseLib's EE(i, j) macro: the four null-space coefficients [x, y, z, 1] of E(i, j).
	private static ReadOnlySpan<double> Ee(ReadOnlySpan<double> n, int i, int j) => n.Slice(4 * (3 * j + i), 4);

	private static void SetRow(Span<double> coeffs, int r, ReadOnlySpan<double> row)
	{
		for (int col = 0; col < 20; ++col)
		{
			coeffs[r + 10 * col] = row[col];
		}
	}

	// a, b are first order polys [x, y, z, 1]; c is degree 2 poly with order
	// [x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1].
	private static void O1(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c)
	{
		c[0] = a[0] * b[0];
		c[1] = a[0] * b[1] + a[1] * b[0];
		c[2] = a[0] * b[2] + a[2] * b[0];
		c[3] = a[0] * b[3] + a[3] * b[0];
		c[4] = a[1] * b[1];
		c[5] = a[1] * b[2] + a[2] * b[1];
		c[6] = a[1] * b[3] + a[3] * b[1];
		c[7] = a[2] * b[2];
		c[8] = a[2] * b[3] + a[3] * b[2];
		c[9] = a[3] * b[3];
	}

	private static void O1p(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c)
	{
		c[0] += a[0] * b[0];
		c[1] += a[0] * b[1] + a[1] * b[0];
		c[2] += a[0] * b[2] + a[2] * b[0];
		c[3] += a[0] * b[3] + a[3] * b[0];
		c[4] += a[1] * b[1];
		c[5] += a[1] * b[2] + a[2] * b[1];
		c[6] += a[1] * b[3] + a[3] * b[1];
		c[7] += a[2] * b[2];
		c[8] += a[2] * b[3] + a[3] * b[2];
		c[9] += a[3] * b[3];
	}

	private static void O1m(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c)
	{
		c[0] -= a[0] * b[0];
		c[1] -= a[0] * b[1] + a[1] * b[0];
		c[2] -= a[0] * b[2] + a[2] * b[0];
		c[3] -= a[0] * b[3] + a[3] * b[0];
		c[4] -= a[1] * b[1];
		c[5] -= a[1] * b[2] + a[2] * b[1];
		c[6] -= a[1] * b[3] + a[3] * b[1];
		c[7] -= a[2] * b[2];
		c[8] -= a[2] * b[3] + a[3] * b[2];
		c[9] -= a[3] * b[3];
	}

	// a is second degree poly with order [x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1]; b is
	// first degree with order [x y z 1]; c is third degree with order (same as Nister's
	// paper) [x^3, y^3, x^2*y, x*y^2, x^2*z, x^2, y^2*z, y^2, x*y*z, x*y, x*z^2, x*z, x,
	// y*z^2, y*z, y, z^3, z^2, z, 1].
	private static void O2(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c)
	{
		c[0] = a[0] * b[0];
		c[1] = a[4] * b[1];
		c[2] = a[0] * b[1] + a[1] * b[0];
		c[3] = a[1] * b[1] + a[4] * b[0];
		c[4] = a[0] * b[2] + a[2] * b[0];
		c[5] = a[0] * b[3] + a[3] * b[0];
		c[6] = a[4] * b[2] + a[5] * b[1];
		c[7] = a[4] * b[3] + a[6] * b[1];
		c[8] = a[1] * b[2] + a[2] * b[1] + a[5] * b[0];
		c[9] = a[1] * b[3] + a[3] * b[1] + a[6] * b[0];
		c[10] = a[2] * b[2] + a[7] * b[0];
		c[11] = a[2] * b[3] + a[3] * b[2] + a[8] * b[0];
		c[12] = a[3] * b[3] + a[9] * b[0];
		c[13] = a[5] * b[2] + a[7] * b[1];
		c[14] = a[5] * b[3] + a[6] * b[2] + a[8] * b[1];
		c[15] = a[6] * b[3] + a[9] * b[1];
		c[16] = a[7] * b[2];
		c[17] = a[7] * b[3] + a[8] * b[2];
		c[18] = a[8] * b[3] + a[9] * b[2];
		c[19] = a[9] * b[3];
	}

	private static void O2p(ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c)
	{
		c[0] += a[0] * b[0];
		c[1] += a[4] * b[1];
		c[2] += a[0] * b[1] + a[1] * b[0];
		c[3] += a[1] * b[1] + a[4] * b[0];
		c[4] += a[0] * b[2] + a[2] * b[0];
		c[5] += a[0] * b[3] + a[3] * b[0];
		c[6] += a[4] * b[2] + a[5] * b[1];
		c[7] += a[4] * b[3] + a[6] * b[1];
		c[8] += a[1] * b[2] + a[2] * b[1] + a[5] * b[0];
		c[9] += a[1] * b[3] + a[3] * b[1] + a[6] * b[0];
		c[10] += a[2] * b[2] + a[7] * b[0];
		c[11] += a[2] * b[3] + a[3] * b[2] + a[8] * b[0];
		c[12] += a[3] * b[3] + a[9] * b[0];
		c[13] += a[5] * b[2] + a[7] * b[1];
		c[14] += a[5] * b[3] + a[6] * b[2] + a[8] * b[1];
		c[15] += a[6] * b[3] + a[9] * b[1];
		c[16] += a[7] * b[2];
		c[17] += a[7] * b[3] + a[8] * b[2];
		c[18] += a[8] * b[3] + a[9] * b[2];
		c[19] += a[9] * b[3];
	}

	/// <summary>A column-major 3 x N matrix over a span, indexed like Eigen's A(r, c).</summary>
	private readonly ref struct Matrix3xNView(Span<double> data)
	{
		private readonly Span<double> _data = data;

		public ref double this[int r, int c] => ref _data[r + 3 * c];
	}
}
