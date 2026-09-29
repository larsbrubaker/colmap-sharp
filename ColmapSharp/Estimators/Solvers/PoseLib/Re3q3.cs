// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08.
//
// Re3q3: PoseLib/misc/re3q3.cc (re3q3 and refine_3q3) - solves three quadratic equations in
// three unknowns, PoseLib's re-implementation of Kukelova's E3Q3 (adapted from Jan Heller's
// implementation) with the elimination-variable choice of Zhou et al., "A Stable Algebraic
// Camera Pose Estimation for Minimal Configurations of 2D/3D Point and Line
// Correspondences", ACCV 2018. Used by P4pf.cs and Gp3p.cs; the univariate step goes
// through Sturm.cs.
//
// Translation notes:
// - The 3x10 coefficient matrix and the 3x8 solution matrix are column-major spans
//   (coeffs(k, j) = coeffs[k + 3 j]), so a call allocates nothing on the heap.
// - When all three elimination matrices are near-singular, PoseLib retries after a random
//   affine change of variables drawn from Eigen's Quaternion::UnitRandom and setRandom,
//   which read the C library's global std::rand(). Here the draw comes from a fresh
//   mt19937 with a fixed seed on every call (Shoemake's uniform rotation, Graphics Gems III,
//   for the rotation), so the result depends only on the input and is thread-safe. See
//   divergence 26.
// - Upstream behavior, kept: a system with a purely linear equation (for example x^2 = 1,
//   y^2 = 4, z = x + y, which has four finite solutions) returns 0 solutions. An affine change
//   of variables cannot give a linear equation quadratic terms, so every elimination matrix
//   keeps a zero row, the retry runs with det = 0, and the inverse yields non-finite
//   coefficients that Sturm bisection rejects. PoseLib fa7280f built with clang++ and Eigen
//   3.4 returns 0 for this system for every std::srand seed 1..20 (and with the retry off);
//   Re3q3Tests.CSharpOnly_LinearEquationReturnsNoSolutionsLikePoseLib pins it. Callers
//   (P4pf) never pass a linear row.
// - The inhomogeneous (3 x 10) rotation_to_3q3 and re3q3_rotation overloads, and
//   quat_multiply from PoseLib/misc/quaternion.h, are ported for Gp3p.cs. The homogeneous
//   (3 x 9) overloads and cayley_param are not: no solver COLMAP calls uses them.
//
// Tier B.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

namespace ColmapSharp.Estimators.Solvers.PoseLib;

/// <summary>Three quadratics in three unknowns. Port of poselib::re3q3.</summary>
public static class Re3q3
{
	/// <summary>The fixed seed of the random change of variables (see the file header).</summary>
	private const uint RandomVarChangeSeed = 0;

	/// <summary>
	/// The fixed seed of re3q3_rotation's random pre-rotation. Distinct from
	/// <see cref="RandomVarChangeSeed"/> so the two draws are not the same rotation.
	/// </summary>
	private const uint RotationPreconditionSeed = 1;

	/// <summary>
	/// Solve the system coeffs * [x^2, xy, xz, y^2, yz, z^2, x, y, z, 1]^T = 0 (3 x 10,
	/// column-major). Writes up to eight solutions as the columns of
	/// <paramref name="solutions"/> (3 x 8, column-major) and returns their count.
	/// </summary>
	public static int Solve(ReadOnlySpan<double> coeffs, Span<double> solutions, bool tryRandomVarChange = true)
	{
		Matrix3d ax = Matrix3d.FromColumns(Col(coeffs, 3), Col(coeffs, 5), Col(coeffs, 4)); // y^2, z^2, yz
		Matrix3d ay = Matrix3d.FromColumns(Col(coeffs, 0), Col(coeffs, 5), Col(coeffs, 2)); // x^2, z^2, xz
		Matrix3d az = Matrix3d.FromColumns(Col(coeffs, 3), Col(coeffs, 0), Col(coeffs, 1)); // y^2, x^2, yx

		// We check det(A) as a cheaper proxy for condition number
		int elimVar = 0;
		double detx = Math.Abs(ax.Determinant());
		double dety = Math.Abs(ay.Determinant());
		double detz = Math.Abs(az.Determinant());
		double det = detx;
		if (det < dety)
		{
			det = dety;
			elimVar = 1;
		}

		if (det < detz)
		{
			det = detz;
			elimVar = 2;
		}

		if (tryRandomVarChange && det < 1e-10)
		{
			return SolveWithRandomVarChange(coeffs, solutions);
		}

		// P is 3 x 7, column-major: P(r, c) = p[r + 3 c].
		Span<double> p = stackalloc double[21];
		ReadOnlySpan<int> order = elimVar switch
		{
			// re-order columns to eliminate x (target: y^2 z^2 yz x^2 xy xz x y z 1)
			0 => [0, 1, 2, 6, 7, 8, 9],
			// re-order columns to eliminate y (target: x^2 z^2 xz y^2 xy yz y x z 1)
			1 => [3, 1, 4, 7, 6, 8, 9],
			// re-order columns to eliminate z (target: y^2 x^2 yx z^2 zy z y x 1)
			_ => [5, 4, 2, 8, 7, 6, 9],
		};
		Matrix3d elim = elimVar switch { 0 => ax, 1 => ay, _ => az };
		Matrix3d negInv = -elim.Inverse();
		for (int k = 0; k < 7; ++k)
		{
			Vector3d col = negInv * Col(coeffs, order[k]);
			p[3 * k] = col.X;
			p[3 * k + 1] = col.Y;
			p[3 * k + 2] = col.Z;
		}

		double a11 = p[3] * p[5] + p[6] * p[4] - p[5] * p[3] - p[8] * p[5] - p[2];
		double a12 = p[3] * p[14] + p[12] * p[5] + p[6] * p[13] + p[15] * p[4] - p[5] * p[12] -
		             p[14] * p[3] - p[8] * p[14] - p[17] * p[5] - p[11];
		double a13 = p[12] * p[14] + p[15] * p[13] - p[14] * p[12] - p[17] * p[14] - p[20];
		double a14 = p[3] * p[8] + p[6] * p[7] - p[5] * p[6] - p[8] * p[8] + p[0];
		double a15 = p[3] * p[17] + p[12] * p[8] + p[6] * p[16] + p[15] * p[7] - p[5] * p[15] -
		             p[14] * p[6] - p[8] * p[17] - p[17] * p[8] + p[9];
		double a16 = p[12] * p[17] + p[15] * p[16] - p[14] * p[15] - p[17] * p[17] + p[18];
		double a17 = p[3] * p[2] + p[6] * p[1] - p[5] * p[0] - p[8] * p[2];
		double a18 = p[3] * p[11] + p[12] * p[2] + p[6] * p[10] + p[15] * p[1] - p[5] * p[9] -
		             p[14] * p[0] - p[8] * p[11] - p[17] * p[2];
		double a19 = p[3] * p[20] + p[12] * p[11] + p[6] * p[19] + p[15] * p[10] - p[5] * p[18] -
		             p[14] * p[9] - p[8] * p[20] - p[17] * p[11];
		double a110 = p[12] * p[20] + p[15] * p[19] - p[14] * p[18] - p[17] * p[20];

		double a21 = p[5] * p[5] + p[8] * p[4] - p[4] * p[3] - p[7] * p[5] - p[1];
		double a22 = p[5] * p[14] + p[14] * p[5] + p[8] * p[13] + p[17] * p[4] - p[4] * p[12] -
		             p[13] * p[3] - p[7] * p[14] - p[16] * p[5] - p[10];
		double a23 = p[14] * p[14] + p[17] * p[13] - p[13] * p[12] - p[16] * p[14] - p[19];
		double a24 = p[5] * p[8] + p[8] * p[7] - p[4] * p[6] - p[7] * p[8] + p[2];
		double a25 = p[5] * p[17] + p[14] * p[8] + p[8] * p[16] + p[17] * p[7] - p[4] * p[15] -
		             p[13] * p[6] - p[7] * p[17] - p[16] * p[8] + p[11];
		double a26 = p[14] * p[17] + p[17] * p[16] - p[13] * p[15] - p[16] * p[17] + p[20];
		double a27 = p[5] * p[2] + p[8] * p[1] - p[4] * p[0] - p[7] * p[2];
		double a28 = p[5] * p[11] + p[14] * p[2] + p[8] * p[10] + p[17] * p[1] - p[4] * p[9] -
		             p[13] * p[0] - p[7] * p[11] - p[16] * p[2];
		double a29 = p[5] * p[20] + p[14] * p[11] + p[8] * p[19] + p[17] * p[10] - p[4] * p[18] -
		             p[13] * p[9] - p[7] * p[20] - p[16] * p[11];
		double a210 = p[14] * p[20] + p[17] * p[19] - p[13] * p[18] - p[16] * p[20];

		double t2 = p[5] * p[5];
		double t3 = p[8] * p[8];
		double t4 = p[3] * p[13];
		double t5 = p[12] * p[4];
		double t6 = t4 + t5;
		double t7 = p[6] * p[16];
		double t8 = p[15] * p[7];
		double t9 = t7 + t8;
		double t10 = p[3] * p[16];
		double t11 = p[12] * p[7];
		double t12 = t10 + t11;
		double t13 = p[6] * p[13];
		double t14 = p[15] * p[4];
		double t15 = t13 + t14;
		double t16 = p[5] * p[17];
		double t17 = p[8] * p[14];
		double t18 = t16 + t17;
		double t19 = p[14] * p[14];
		double t20 = p[17] * p[17];
		double a31 = p[0] * p[4] + p[3] * p[1] - p[2] * p[5] * 2.0 - p[3] * t2 - p[4] * t3 -
		             p[8] * t2 * 2.0 + (p[3] * p[3]) * p[4] + p[6] * p[4] * p[7] +
		             p[3] * p[7] * p[5] + p[6] * p[4] * p[5];
		double a32 = p[0] * p[13] + p[3] * p[10] + p[9] * p[4] + p[12] * p[1] -
		             p[2] * p[14] * 2.0 - p[5] * p[11] * 2.0 - p[12] * t2 + p[3] * t6 - p[13] * t3 +
		             p[4] * t9 + p[5] * t12 + p[5] * t15 - p[5] * t18 * 2.0 + p[3] * p[12] * p[4] +
		             p[6] * p[7] * p[13] + p[3] * p[7] * p[14] + p[6] * p[4] * p[14] -
		             p[3] * p[5] * p[14] * 2.0 - p[4] * p[8] * p[17] * 2.0 -
		             p[5] * p[8] * p[14] * 2.0;
		double a33 = p[3] * p[19] + p[9] * p[13] + p[12] * p[10] + p[18] * p[4] -
		             p[5] * p[20] * 2.0 - p[11] * p[14] * 2.0 + p[12] * t6 - p[3] * t19 + p[13] * t9 -
		             p[4] * t20 + p[14] * t12 + p[14] * t15 - p[14] * t18 * 2.0 + p[3] * p[12] * p[13] +
		             p[15] * p[4] * p[16] + p[12] * p[16] * p[5] + p[15] * p[13] * p[5] -
		             p[12] * p[5] * p[14] * 2.0 - p[13] * p[8] * p[17] * 2.0 -
		             p[5] * p[14] * p[17] * 2.0;
		double a34 = p[12] * p[19] + p[18] * p[13] - p[14] * p[20] * 2.0 - p[12] * t19 - p[13] * t20 -
		             p[17] * t19 * 2.0 + (p[12] * p[12]) * p[13] + p[15] * p[13] * p[16] +
		             p[12] * p[16] * p[14] + p[15] * p[13] * p[14];
		double a35 = p[0] * p[7] + p[6] * p[1] - p[2] * p[8] * 2.0 - p[6] * t2 - p[7] * t3 -
		             p[5] * t3 * 2.0 + p[6] * (p[7] * p[7]) + p[3] * p[6] * p[4] +
		             p[3] * p[7] * p[8] + p[6] * p[4] * p[8];
		double a36 = p[0] * p[16] + p[6] * p[10] + p[9] * p[7] + p[15] * p[1] -
		             p[2] * p[17] * 2.0 - p[8] * p[11] * 2.0 - p[15] * t2 + p[6] * t6 - p[16] * t3 +
		             p[7] * t9 + p[8] * t12 + p[8] * t15 - p[8] * t18 * 2.0 + p[3] * p[15] * p[4] +
		             p[6] * p[7] * p[16] + p[3] * p[7] * p[17] + p[6] * p[4] * p[17] -
		             p[6] * p[5] * p[14] * 2.0 - p[7] * p[8] * p[17] * 2.0 -
		             p[5] * p[8] * p[17] * 2.0;
		double a37 = p[6] * p[19] + p[9] * p[16] + p[15] * p[10] + p[18] * p[7] -
		             p[8] * p[20] * 2.0 - p[11] * p[17] * 2.0 + p[15] * t6 - p[6] * t19 + p[16] * t9 -
		             p[7] * t20 + p[17] * t12 + p[17] * t15 - p[17] * t18 * 2.0 + p[6] * p[12] * p[13] +
		             p[15] * p[7] * p[16] + p[12] * p[16] * p[8] + p[15] * p[13] * p[8] -
		             p[15] * p[5] * p[14] * 2.0 - p[16] * p[8] * p[17] * 2.0 -
		             p[8] * p[14] * p[17] * 2.0;
		double a38 = p[15] * p[19] + p[18] * p[16] - p[17] * p[20] * 2.0 - p[15] * t19 - p[16] * t20 -
		             p[14] * t20 * 2.0 + p[15] * (p[16] * p[16]) + p[12] * p[15] * p[13] +
		             p[12] * p[16] * p[17] + p[15] * p[13] * p[17];
		double a39 = p[0] * p[1] - p[0] * t2 - p[1] * t3 - p[2] * p[2] + p[0] * p[3] * p[4] +
		             p[6] * p[1] * p[7] + p[3] * p[7] * p[2] + p[6] * p[4] * p[2] -
		             p[2] * p[5] * p[8] * 2.0;
		double a310 = p[0] * p[10] + p[9] * p[1] - p[2] * p[11] * 2.0 - p[9] * t2 + p[0] * t6 -
		              p[10] * t3 + p[1] * t9 + p[2] * t12 + p[2] * t15 - p[2] * t18 * 2.0 +
		              p[3] * p[9] * p[4] + p[6] * p[7] * p[10] + p[3] * p[7] * p[11] +
		              p[6] * p[4] * p[11] - p[0] * p[5] * p[14] * 2.0 - p[1] * p[8] * p[17] * 2.0 -
		              p[5] * p[8] * p[11] * 2.0;
		double a311 = p[0] * p[19] + p[9] * p[10] + p[18] * p[1] - p[2] * p[20] * 2.0 - p[18] * t2 +
		              p[9] * t6 - p[0] * t19 - p[19] * t3 + p[10] * t9 - p[1] * t20 + p[11] * t12 +
		              p[11] * t15 - p[11] * t18 * 2.0 - p[11] * p[11] + p[0] * p[12] * p[13] +
		              p[3] * p[18] * p[4] + p[6] * p[7] * p[19] + p[15] * p[1] * p[16] +
		              p[3] * p[7] * p[20] + p[6] * p[4] * p[20] + p[12] * p[16] * p[2] +
		              p[15] * p[13] * p[2] - p[9] * p[5] * p[14] * 2.0 - p[10] * p[8] * p[17] * 2.0 -
		              p[2] * p[14] * p[17] * 2.0 - p[5] * p[8] * p[20] * 2.0;
		double a312 = p[9] * p[19] + p[18] * p[10] - p[11] * p[20] * 2.0 + p[18] * t6 - p[9] * t19 +
		              p[19] * t9 - p[10] * t20 + p[20] * t12 + p[20] * t15 - p[20] * t18 * 2.0 +
		              p[9] * p[12] * p[13] + p[15] * p[10] * p[16] + p[12] * p[16] * p[11] +
		              p[15] * p[13] * p[11] - p[18] * p[5] * p[14] * 2.0 - p[19] * p[8] * p[17] * 2.0 -
		              p[11] * p[14] * p[17] * 2.0;
		double a313 = p[18] * p[19] - p[18] * t19 - p[19] * t20 - p[20] * p[20] + p[12] * p[18] * p[13] +
		              p[15] * p[16] * p[19] + p[12] * p[16] * p[20] + p[15] * p[13] * p[20] -
		              p[14] * p[17] * p[20] * 2.0;

		// det(M(x))
		Span<double> c = stackalloc double[9];
		c[8] = a14 * a27 * a31 - a17 * a24 * a31 - a11 * a27 * a35 + a17 * a21 * a35 + a11 * a24 * a39 - a14 * a21 * a39;
		c[7] = a14 * a27 * a32 + a14 * a28 * a31 + a15 * a27 * a31 - a17 * a24 * a32 - a17 * a25 * a31 - a18 * a24 * a31 -
		       a11 * a27 * a36 - a11 * a28 * a35 - a12 * a27 * a35 + a17 * a21 * a36 + a17 * a22 * a35 + a18 * a21 * a35 +
		       a11 * a25 * a39 + a12 * a24 * a39 - a14 * a22 * a39 - a15 * a21 * a39 + a11 * a24 * a310 - a14 * a21 * a310;
		c[6] = a14 * a27 * a33 + a14 * a28 * a32 + a14 * a29 * a31 + a15 * a27 * a32 + a15 * a28 * a31 + a16 * a27 * a31 -
		       a17 * a24 * a33 - a17 * a25 * a32 - a17 * a26 * a31 - a18 * a24 * a32 - a18 * a25 * a31 - a19 * a24 * a31 -
		       a11 * a27 * a37 - a11 * a28 * a36 - a11 * a29 * a35 - a12 * a27 * a36 - a12 * a28 * a35 - a13 * a27 * a35 +
		       a17 * a21 * a37 + a17 * a22 * a36 + a17 * a23 * a35 + a18 * a21 * a36 + a18 * a22 * a35 + a19 * a21 * a35 +
		       a11 * a26 * a39 + a12 * a25 * a39 + a13 * a24 * a39 - a14 * a23 * a39 - a15 * a22 * a39 - a16 * a21 * a39 +
		       a11 * a24 * a311 + a11 * a25 * a310 + a12 * a24 * a310 - a14 * a21 * a311 - a14 * a22 * a310 -
		       a15 * a21 * a310;
		c[5] = a14 * a27 * a34 + a14 * a28 * a33 + a14 * a29 * a32 + a15 * a27 * a33 + a15 * a28 * a32 + a15 * a29 * a31 +
		       a16 * a27 * a32 + a16 * a28 * a31 - a17 * a24 * a34 - a17 * a25 * a33 - a17 * a26 * a32 - a18 * a24 * a33 -
		       a18 * a25 * a32 - a18 * a26 * a31 - a19 * a24 * a32 - a19 * a25 * a31 - a11 * a27 * a38 - a11 * a28 * a37 -
		       a11 * a29 * a36 - a12 * a27 * a37 - a12 * a28 * a36 - a12 * a29 * a35 - a13 * a27 * a36 - a13 * a28 * a35 +
		       a17 * a21 * a38 + a17 * a22 * a37 + a17 * a23 * a36 + a18 * a21 * a37 + a18 * a22 * a36 + a18 * a23 * a35 +
		       a19 * a21 * a36 + a19 * a22 * a35 + a12 * a26 * a39 + a13 * a25 * a39 - a15 * a23 * a39 - a16 * a22 * a39 -
		       a24 * a31 * a110 + a21 * a35 * a110 + a14 * a31 * a210 - a11 * a35 * a210 + a11 * a24 * a312 +
		       a11 * a25 * a311 + a11 * a26 * a310 + a12 * a24 * a311 + a12 * a25 * a310 + a13 * a24 * a310 -
		       a14 * a21 * a312 - a14 * a22 * a311 - a14 * a23 * a310 - a15 * a21 * a311 - a15 * a22 * a310 -
		       a16 * a21 * a310;
		c[4] = a14 * a28 * a34 + a14 * a29 * a33 + a15 * a27 * a34 + a15 * a28 * a33 + a15 * a29 * a32 + a16 * a27 * a33 +
		       a16 * a28 * a32 + a16 * a29 * a31 - a17 * a25 * a34 - a17 * a26 * a33 - a18 * a24 * a34 - a18 * a25 * a33 -
		       a18 * a26 * a32 - a19 * a24 * a33 - a19 * a25 * a32 - a19 * a26 * a31 - a11 * a28 * a38 - a11 * a29 * a37 -
		       a12 * a27 * a38 - a12 * a28 * a37 - a12 * a29 * a36 - a13 * a27 * a37 - a13 * a28 * a36 - a13 * a29 * a35 +
		       a17 * a22 * a38 + a17 * a23 * a37 + a18 * a21 * a38 + a18 * a22 * a37 + a18 * a23 * a36 + a19 * a21 * a37 +
		       a19 * a22 * a36 + a19 * a23 * a35 + a13 * a26 * a39 - a16 * a23 * a39 - a24 * a32 * a110 - a25 * a31 * a110 +
		       a21 * a36 * a110 + a22 * a35 * a110 + a14 * a32 * a210 + a15 * a31 * a210 - a11 * a36 * a210 -
		       a12 * a35 * a210 + a11 * a24 * a313 + a11 * a25 * a312 + a11 * a26 * a311 + a12 * a24 * a312 +
		       a12 * a25 * a311 + a12 * a26 * a310 + a13 * a24 * a311 + a13 * a25 * a310 - a14 * a21 * a313 -
		       a14 * a22 * a312 - a14 * a23 * a311 - a15 * a21 * a312 - a15 * a22 * a311 - a15 * a23 * a310 -
		       a16 * a21 * a311 - a16 * a22 * a310;
		c[3] = a14 * a29 * a34 + a15 * a28 * a34 + a15 * a29 * a33 + a16 * a27 * a34 + a16 * a28 * a33 + a16 * a29 * a32 -
		       a17 * a26 * a34 - a18 * a25 * a34 - a18 * a26 * a33 - a19 * a24 * a34 - a19 * a25 * a33 - a19 * a26 * a32 -
		       a11 * a29 * a38 - a12 * a28 * a38 - a12 * a29 * a37 - a13 * a27 * a38 - a13 * a28 * a37 - a13 * a29 * a36 +
		       a17 * a23 * a38 + a18 * a22 * a38 + a18 * a23 * a37 + a19 * a21 * a38 + a19 * a22 * a37 + a19 * a23 * a36 -
		       a24 * a33 * a110 - a25 * a32 * a110 - a26 * a31 * a110 + a21 * a37 * a110 + a22 * a36 * a110 +
		       a23 * a35 * a110 + a14 * a33 * a210 + a15 * a32 * a210 + a16 * a31 * a210 - a11 * a37 * a210 -
		       a12 * a36 * a210 - a13 * a35 * a210 + a11 * a25 * a313 + a11 * a26 * a312 + a12 * a24 * a313 +
		       a12 * a25 * a312 + a12 * a26 * a311 + a13 * a24 * a312 + a13 * a25 * a311 + a13 * a26 * a310 -
		       a14 * a22 * a313 - a14 * a23 * a312 - a15 * a21 * a313 - a15 * a22 * a312 - a15 * a23 * a311 -
		       a16 * a21 * a312 - a16 * a22 * a311 - a16 * a23 * a310;
		c[2] = a15 * a29 * a34 + a16 * a28 * a34 + a16 * a29 * a33 - a18 * a26 * a34 - a19 * a25 * a34 - a19 * a26 * a33 -
		       a12 * a29 * a38 - a13 * a28 * a38 - a13 * a29 * a37 + a18 * a23 * a38 + a19 * a22 * a38 + a19 * a23 * a37 -
		       a24 * a34 * a110 - a25 * a33 * a110 - a26 * a32 * a110 + a21 * a38 * a110 + a22 * a37 * a110 +
		       a23 * a36 * a110 + a14 * a34 * a210 + a15 * a33 * a210 + a16 * a32 * a210 - a11 * a38 * a210 -
		       a12 * a37 * a210 - a13 * a36 * a210 + a11 * a26 * a313 + a12 * a25 * a313 + a12 * a26 * a312 +
		       a13 * a24 * a313 + a13 * a25 * a312 + a13 * a26 * a311 - a14 * a23 * a313 - a15 * a22 * a313 -
		       a15 * a23 * a312 - a16 * a21 * a313 - a16 * a22 * a312 - a16 * a23 * a311;
		c[1] = a16 * a29 * a34 - a19 * a26 * a34 - a13 * a29 * a38 + a19 * a23 * a38 - a25 * a34 * a110 - a26 * a33 * a110 +
		       a22 * a38 * a110 + a23 * a37 * a110 + a15 * a34 * a210 + a16 * a33 * a210 - a12 * a38 * a210 -
		       a13 * a37 * a210 + a12 * a26 * a313 + a13 * a25 * a313 + a13 * a26 * a312 - a15 * a23 * a313 -
		       a16 * a22 * a313 - a16 * a23 * a312;
		c[0] = -a26 * a34 * a110 + a23 * a38 * a110 + a16 * a34 * a210 - a13 * a38 * a210 + a13 * a26 * a313 -
		       a16 * a23 * a313;

		Span<double> roots = stackalloc double[8];

		int nRoots = Sturm.BisectSturm(8, c, roots);

		for (int i = 0; i < nRoots; ++i)
		{
			double xs1 = roots[i];
			double xs2 = xs1 * xs1;
			double xs3 = xs1 * xs2;

			double a00 = a11 * xs2 + a12 * xs1 + a13;
			double a01 = a14 * xs2 + a15 * xs1 + a16;
			double a02 = a17 * xs3 + a18 * xs2 + a19 * xs1 + a110;
			double a10 = a21 * xs2 + a22 * xs1 + a23;
			double a11v = a24 * xs2 + a25 * xs1 + a26;
			double a12v = a27 * xs3 + a28 * xs2 + a29 * xs1 + a210;
			// The third row of PoseLib's A is filled but never read.

			solutions[3 * i] = xs1;
			solutions[3 * i + 1] = (a12v * a01 - a02 * a11v) / (a00 * a11v - a10 * a01);
			solutions[3 * i + 2] = (a12v * a00 - a02 * a10) / (a01 * a10 - a11v * a00);
		}

		if (elimVar == 1)
		{
			SwapRows(solutions, 0, 1);
		}
		else if (elimVar == 2)
		{
			SwapRows(solutions, 0, 2);
		}

		Refine3q3(coeffs, solutions, nRoots);

		return nRoots;
	}

	/// <summary>
	/// Inhomogeneous linear constraints on a rotation matrix, Rcoeffs * [R(:); 1] = 0 (3 x 10,
	/// column-major, R(:) column-major), converted into a 3q3 problem in the Cayley
	/// parameters. Port of the 3 x 10 overload of poselib::re3q3::rotation_to_3q3.
	/// </summary>
	public static void RotationTo3q3(ReadOnlySpan<double> rcoeffs, Span<double> coeffs)
	{
		for (int k = 0; k < 3; k++)
		{
			double r0 = rcoeffs[k], r1 = rcoeffs[k + 3], r2 = rcoeffs[k + 6], r3 = rcoeffs[k + 9], r4 = rcoeffs[k + 12];
			double r5 = rcoeffs[k + 15], r6 = rcoeffs[k + 18], r7 = rcoeffs[k + 21], r8 = rcoeffs[k + 24], r9 = rcoeffs[k + 27];
			coeffs[k] = r0 - r4 - r8 + r9;
			coeffs[k + 3] = 2 * r1 + 2 * r3;
			coeffs[k + 6] = 2 * r2 + 2 * r6;
			coeffs[k + 9] = r4 - r0 - r8 + r9;
			coeffs[k + 12] = 2 * r5 + 2 * r7;
			coeffs[k + 15] = r8 - r4 - r0 + r9;
			coeffs[k + 18] = 2 * r5 - 2 * r7;
			coeffs[k + 21] = 2 * r6 - 2 * r2;
			coeffs[k + 24] = 2 * r1 - 2 * r3;
			coeffs[k + 27] = r0 + r4 + r8 + r9;
		}
	}

	/// <summary>
	/// Solve Rcoeffs * [R(:); 1] = 0 (3 x 10, column-major) for rotations R. Writes up to
	/// eight unit quaternions (PoseLib order, real part first) to
	/// <paramref name="solutions"/> and returns their count. PoseLib first rotates the
	/// problem by a random rotation R0 so that no solution sits at the Cayley transform's
	/// singularity (a rotation by pi), solves in Cayley parameters, and composes R0 back. Port
	/// of the 3 x 10 overload of poselib::re3q3::re3q3_rotation (re3q3_rotation_impl).
	/// </summary>
	/// <remarks>
	/// PoseLib draws R0 from Eigen's Quaternion::UnitRandom, i.e. from std::rand; here it
	/// comes from a fresh mt19937 with a fixed seed on every call, so the result depends only
	/// on the input (divergence 29).
	/// </remarks>
	public static int SolveRotation(ReadOnlySpan<double> rcoeffs, Span<Vector4d> solutions, bool tryRandomVarChange = true)
	{
		// PoseLib reads UnitRandom().coeffs(), which is Eigen's (x, y, z, w) memory order,
		// as a (w, x, y, z) vector. The draw is uniform on the unit sphere either way, so the
		// component order does not change the distribution; the port keeps PoseLib's reading.
		Quaterniond draw = UnitRandomQuaternion(new Mt19937(RotationPreconditionSeed));
		var q0 = new Vector4d(draw.X, draw.Y, draw.Z, draw.W);
		Matrix3d r0 = CameraPose.QuatToRotmat(q0);

		// Rcoeffs.block<3, 3>(0, 3 b) = Rcoeffs.block<3, 3>(0, 3 b) * R0 for b = 0, 1, 2.
		Span<double> rotated = stackalloc double[30];
		rcoeffs[..30].CopyTo(rotated);
		for (int b = 0; b < 3; ++b)
		{
			for (int row = 0; row < 3; ++row)
			{
				for (int col = 0; col < 3; ++col)
				{
					rotated[row + 3 * (3 * b + col)] =
						rcoeffs[row + 3 * (3 * b)] * r0[0, col] +
						rcoeffs[row + 3 * (3 * b + 1)] * r0[1, col] +
						rcoeffs[row + 3 * (3 * b + 2)] * r0[2, col];
				}
			}
		}

		Span<double> coeffs = stackalloc double[30];
		RotationTo3q3(rotated, coeffs);

		Span<double> solutionsCayley = stackalloc double[24];
		int nSols = Solve(coeffs, solutionsCayley, tryRandomVarChange);

		for (int i = 0; i < nSols; ++i)
		{
			Vector4d q = new Vector4d(1.0, solutionsCayley[3 * i], solutionsCayley[3 * i + 1], solutionsCayley[3 * i + 2]).Normalized();
			solutions[i] = QuatMultiply(q0, q);
		}

		return nSols;
	}

	/// <summary>Port of poselib::quat_multiply on (w, x, y, z) vectors.</summary>
	internal static Vector4d QuatMultiply(Vector4d qa, Vector4d qb)
	{
		double qa1 = qa.X, qa2 = qa.Y, qa3 = qa.Z, qa4 = qa.W;
		double qb1 = qb.X, qb2 = qb.Y, qb3 = qb.Z, qb4 = qb.W;

		return new Vector4d(
			qa1 * qb1 - qa2 * qb2 - qa3 * qb3 - qa4 * qb4,
			qa1 * qb2 + qa2 * qb1 + qa3 * qb4 - qa4 * qb3,
			qa1 * qb3 + qa3 * qb1 - qa2 * qb4 + qa4 * qb2,
			qa1 * qb4 + qa2 * qb3 - qa3 * qb2 + qa4 * qb1);
	}

	private static Vector3d Col(ReadOnlySpan<double> m3xN, int j) => new(m3xN[3 * j], m3xN[3 * j + 1], m3xN[3 * j + 2]);

	// Eigen's row(a).swap(row(b)) over all 8 columns of the 3 x 8 solution matrix.
	private static void SwapRows(Span<double> solutions, int r0, int r1)
	{
		for (int j = 0; j < 8; ++j)
		{
			(solutions[r0 + 3 * j], solutions[r1 + 3 * j]) = (solutions[r1 + 3 * j], solutions[r0 + 3 * j]);
		}
	}

	private static int SolveWithRandomVarChange(ReadOnlySpan<double> coeffs, Span<double> solutions)
	{
		var rng = new Mt19937(RandomVarChangeSeed);
		Matrix3d rot = UnitRandomQuaternion(rng).ToRotationMatrix();
		Vector3d shift = new Vector3d(
			LibcxxRandom.UniformReal(rng, -1.0, 1.0),
			LibcxxRandom.UniformReal(rng, -1.0, 1.0),
			LibcxxRandom.UniformReal(rng, -1.0, 1.0)).Normalized();

		// A is 3 x 4, column-major: A(r, c) = a[r + 3 c].
		Span<double> a = stackalloc double[12];
		rot.CopyToColumnMajor(a);
		a[9] = shift.X;
		a[10] = shift.Y;
		a[11] = shift.Z;

		// B is 10 x 10, column-major: B(r, c) = bm[r + 10 c].
		Span<double> bm = stackalloc double[100];
		{
			bm[0] = a[0] * a[0];
			bm[10] = 2 * a[0] * a[3];
			bm[20] = 2 * a[0] * a[6];
			bm[30] = a[3] * a[3];
			bm[40] = 2 * a[3] * a[6];
			bm[50] = a[6] * a[6];
			bm[60] = 2 * a[0] * a[9];
			bm[70] = 2 * a[3] * a[9];
			bm[80] = 2 * a[6] * a[9];
			bm[90] = a[9] * a[9];
			bm[1] = a[0] * a[1];
			bm[11] = a[0] * a[4] + a[3] * a[1];
			bm[21] = a[0] * a[7] + a[6] * a[1];
			bm[31] = a[3] * a[4];
			bm[41] = a[3] * a[7] + a[6] * a[4];
			bm[51] = a[6] * a[7];
			bm[61] = a[0] * a[10] + a[9] * a[1];
			bm[71] = a[3] * a[10] + a[9] * a[4];
			bm[81] = a[6] * a[10] + a[9] * a[7];
			bm[91] = a[9] * a[10];
			bm[2] = a[0] * a[2];
			bm[12] = a[0] * a[5] + a[3] * a[2];
			bm[22] = a[0] * a[8] + a[6] * a[2];
			bm[32] = a[3] * a[5];
			bm[42] = a[3] * a[8] + a[6] * a[5];
			bm[52] = a[6] * a[8];
			bm[62] = a[0] * a[11] + a[9] * a[2];
			bm[72] = a[3] * a[11] + a[9] * a[5];
			bm[82] = a[6] * a[11] + a[9] * a[8];
			bm[92] = a[9] * a[11];
			bm[3] = a[1] * a[1];
			bm[13] = 2 * a[1] * a[4];
			bm[23] = 2 * a[1] * a[7];
			bm[33] = a[4] * a[4];
			bm[43] = 2 * a[4] * a[7];
			bm[53] = a[7] * a[7];
			bm[63] = 2 * a[1] * a[10];
			bm[73] = 2 * a[4] * a[10];
			bm[83] = 2 * a[7] * a[10];
			bm[93] = a[10] * a[10];
			bm[4] = a[1] * a[2];
			bm[14] = a[1] * a[5] + a[4] * a[2];
			bm[24] = a[1] * a[8] + a[7] * a[2];
			bm[34] = a[4] * a[5];
			bm[44] = a[4] * a[8] + a[7] * a[5];
			bm[54] = a[7] * a[8];
			bm[64] = a[1] * a[11] + a[10] * a[2];
			bm[74] = a[4] * a[11] + a[10] * a[5];
			bm[84] = a[7] * a[11] + a[10] * a[8];
			bm[94] = a[10] * a[11];
			bm[5] = a[2] * a[2];
			bm[15] = 2 * a[2] * a[5];
			bm[25] = 2 * a[2] * a[8];
			bm[35] = a[5] * a[5];
			bm[45] = 2 * a[5] * a[8];
			bm[55] = a[8] * a[8];
			bm[65] = 2 * a[2] * a[11];
			bm[75] = 2 * a[5] * a[11];
			bm[85] = 2 * a[8] * a[11];
			bm[95] = a[11] * a[11];
			bm[6] = 0;
			bm[16] = 0;
			bm[26] = 0;
			bm[36] = 0;
			bm[46] = 0;
			bm[56] = 0;
			bm[66] = a[0];
			bm[76] = a[3];
			bm[86] = a[6];
			bm[96] = a[9];
			bm[7] = 0;
			bm[17] = 0;
			bm[27] = 0;
			bm[37] = 0;
			bm[47] = 0;
			bm[57] = 0;
			bm[67] = a[1];
			bm[77] = a[4];
			bm[87] = a[7];
			bm[97] = a[10];
			bm[8] = 0;
			bm[18] = 0;
			bm[28] = 0;
			bm[38] = 0;
			bm[48] = 0;
			bm[58] = 0;
			bm[68] = a[2];
			bm[78] = a[5];
			bm[88] = a[8];
			bm[98] = a[11];
			bm[9] = 0;
			bm[19] = 0;
			bm[29] = 0;
			bm[39] = 0;
			bm[49] = 0;
			bm[59] = 0;
			bm[69] = 0;
			bm[79] = 0;
			bm[89] = 0;
			bm[99] = 1;
		}

		Span<double> coeffsB = stackalloc double[30];
		for (int k = 0; k < 3; ++k)
		{
			for (int j = 0; j < 10; ++j)
			{
				double sum = 0.0;
				for (int l = 0; l < 10; ++l)
				{
					sum += coeffs[k + 3 * l] * bm[l + 10 * j];
				}

				coeffsB[k + 3 * j] = sum;
			}
		}

		int nSols = Solve(coeffsB, solutions, false);

		// Revert change of variables
		for (int k = 0; k < nSols; k++)
		{
			Vector3d s = rot * new Vector3d(solutions[3 * k], solutions[3 * k + 1], solutions[3 * k + 2]) + shift;
			solutions[3 * k] = s.X;
			solutions[3 * k + 1] = s.Y;
			solutions[3 * k + 2] = s.Z;
		}

		// In some cases the numerics are quite poor after the change of variables, so we do some newton steps with the
		// original coefficients.
		Refine3q3(coeffs, solutions, nSols);

		return nSols;
	}

	// A uniformly distributed unit quaternion: K. Shoemake, "Uniform random rotations",
	// Graphics Gems III (1992), from three uniform draws in [0, 1).
	private static Quaterniond UnitRandomQuaternion(Mt19937 rng)
	{
		double u1 = LibcxxRandom.UniformReal(rng, 0.0, 1.0);
		double u2 = 2.0 * Math.PI * LibcxxRandom.UniformReal(rng, 0.0, 1.0);
		double u3 = 2.0 * Math.PI * LibcxxRandom.UniformReal(rng, 0.0, 1.0);
		double a = Math.Sqrt(1.0 - u1);
		double b = Math.Sqrt(u1);
		return new Quaterniond(b * Math.Cos(u3), a * Math.Sin(u2), a * Math.Cos(u2), b * Math.Sin(u3));
	}

	// Port of poselib::re3q3::refine_3q3: up to five Newton steps per solution.
	private static void Refine3q3(ReadOnlySpan<double> coeffs, Span<double> solutions, int nSols)
	{
		for (int i = 0; i < nSols; ++i)
		{
			double x = solutions[3 * i];
			double y = solutions[3 * i + 1];
			double z = solutions[3 * i + 2];

			// [x^2, x*y, x*z, y^2, y*z, z^2, x, y, z, 1.0]
			for (int iter = 0; iter < 5; ++iter)
			{
				Vector3d r = Col(coeffs, 0) * x * x + Col(coeffs, 1) * x * y + Col(coeffs, 2) * x * z + Col(coeffs, 3) * y * y +
					Col(coeffs, 4) * y * z + Col(coeffs, 5) * z * z + Col(coeffs, 6) * x + Col(coeffs, 7) * y +
					Col(coeffs, 8) * z + Col(coeffs, 9);

				Vector3d rAbs = r.CwiseAbs();
				if (Math.Max(Math.Max(rAbs.X, rAbs.Y), rAbs.Z) < 1e-8)
				{
					break;
				}

				Matrix3d j = Matrix3d.FromColumns(
					2.0 * Col(coeffs, 0) * x + Col(coeffs, 1) * y + Col(coeffs, 2) * z + Col(coeffs, 6),
					Col(coeffs, 1) * x + 2.0 * Col(coeffs, 3) * y + Col(coeffs, 4) * z + Col(coeffs, 7),
					Col(coeffs, 2) * x + Col(coeffs, 4) * y + 2.0 * Col(coeffs, 5) * z + Col(coeffs, 8));

				Vector3d dx = j.Inverse() * r;

				x -= dx.X;
				y -= dx.Y;
				z -= dx.Z;
			}

			solutions[3 * i] = x;
			solutions[3 * i + 1] = y;
			solutions[3 * i + 2] = z;
		}
	}
}
