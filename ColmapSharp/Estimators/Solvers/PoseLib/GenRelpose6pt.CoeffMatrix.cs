// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08 as pinned by
// COLMAP's src/thirdparty/CMakeLists.txt.
//
// GenRelpose6pt.CoeffMatrix: setup_coeff_matrix and pt_index of PoseLib's
// gen_relpose_6pt.cc. Each of the 15 equations takes 4 of the 6 correspondences, eliminates
// the translation through a 3x3 matrix whose entries are quadratic in the Cayley parameters
// (F1, F2, F3 are its columns, 10 monomials each), and expands its determinant into a
// quartic with 84 coefficients. The F expressions are generated code transcribed verbatim
// by a script (operand order kept). The coefficients feed GenRelpose6pt.cs.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Estimators.Solvers.PoseLib;

internal static partial class GenRelpose6pt
{
	/// <summary>Which combinations of 4 points are used to generate the 15 equations.</summary>
	private static ReadOnlySpan<int> PointIndex => new int[]
	{
		0, 1, 2, 3,
		0, 1, 2, 4,
		0, 1, 2, 5,
		0, 1, 3, 4,
		0, 1, 3, 5,
		0, 1, 4, 5,
		0, 2, 3, 4,
		0, 2, 3, 5,
		0, 2, 4, 5,
		0, 3, 4, 5,
		1, 2, 3, 4,
		1, 2, 3, 5,
		1, 2, 4, 5,
		1, 3, 4, 5,
		2, 3, 4, 5,
	};

	/// <summary>
	/// Computes the 84 x 15 matrix of coefficients for the 15 equations (column-major into
	/// <paramref name="m"/>, one column of 84 monomial coefficients per equation). Port of
	/// PoseLib's setup_coeff_matrix.
	/// </summary>
	private static void SetupCoeffMatrix(
		ReadOnlySpan<Vector3d> pp1,
		ReadOnlySpan<Vector3d> xx1,
		ReadOnlySpan<Vector3d> pp2,
		ReadOnlySpan<Vector3d> xx2,
		Span<double> m)
	{
		// F1, F2, F3 are Eigen 10x3 column-major matrices, so column i starts at 10 * i.
		Span<double> f1 = stackalloc double[30];
		Span<double> f2 = stackalloc double[30];
		Span<double> f3 = stackalloc double[30];
		Span<double> p4 = stackalloc double[35];

		Span<Vector3d> qq1 = stackalloc Vector3d[6];
		Span<Vector3d> qq2 = stackalloc Vector3d[6];
		for (int k = 0; k < 6; ++k)
		{
			qq1[k] = xx1[k].Cross(pp1[k]);
			qq2[k] = xx2[k].Cross(pp2[k]);
		}

		m[..(84 * 15)].Clear();
		for (int eqK = 0; eqK < 15; ++eqK)
		{
			int i0 = PointIndex[4 * eqK];

			Vector3d x1 = xx1[i0];
			Vector3d p1 = pp1[i0];
			Vector3d x2 = xx2[i0];
			Vector3d p2 = pp2[i0];

			// Compute 3x3 matrix where each element is quadratic in cayley parameters
			// F1 is the first column of the matrix, etc..
			// This is for eliminating the translation.
			for (int i = 0; i < 3; ++i)
			{
				int i1 = PointIndex[4 * eqK + i + 1];

				Vector3d xp1 = xx1[i1];
				Vector3d qp1 = qq1[i1];
				Vector3d xp2 = xx2[i1];
				Vector3d qp2 = qq2[i1];

				f1[0 + 10 * i] = qp1.X * xp2.X + qp2.X * xp1.X - qp1.Y * xp2.Y - qp2.Y * xp1.Y - qp1.Z * xp2.Z - qp2.Z * xp1.Z + xp1.X * (xp2.Z * (p1.Y + p2.Y) - xp2.Y * (p1.Z + p2.Z)) + xp1.Z * (xp2.X * (p1.Y + p2.Y) + xp2.Y * (p1.X - p2.X)) - xp1.Y * (xp2.X * (p1.Z + p2.Z) + xp2.Z * (p1.X - p2.X));
				f1[1 + 10 * i] = 2 * qp1.X * xp2.Y + 2 * qp1.Y * xp2.X + 2 * qp2.X * xp1.Y + 2 * qp2.Y * xp1.X - xp1.X * (2 * p2.X * xp2.Z - xp2.X * (2 * p1.Z + 2 * p2.Z)) + xp1.Y * (2 * p2.Y * xp2.Z - xp2.Y * (2 * p1.Z + 2 * p2.Z)) - xp1.Z * (2 * p1.X * xp2.X - 2 * p1.Y * xp2.Y);
				f1[2 + 10 * i] = qp1.Y * xp2.Y - qp2.X * xp1.X - qp1.X * xp2.X + qp2.Y * xp1.Y - qp1.Z * xp2.Z - qp2.Z * xp1.Z - xp1.Y * (xp2.Z * (p1.X + p2.X) - xp2.X * (p1.Z + p2.Z)) - xp1.Z * (xp2.Y * (p1.X + p2.X) + xp2.X * (p1.Y - p2.Y)) + xp1.X * (xp2.Y * (p1.Z + p2.Z) + xp2.Z * (p1.Y - p2.Y));
				f1[3 + 10 * i] = 2 * qp1.X * xp2.Z + 2 * qp1.Z * xp2.X + 2 * qp2.X * xp1.Z + 2 * qp2.Z * xp1.X + xp1.X * (2 * p2.X * xp2.Y - xp2.X * (2 * p1.Y + 2 * p2.Y)) - xp1.Z * (2 * p2.Z * xp2.Y - xp2.Z * (2 * p1.Y + 2 * p2.Y)) + xp1.Y * (2 * p1.X * xp2.X - 2 * p1.Z * xp2.Z);
				f1[4 + 10 * i] = 2 * qp1.Y * xp2.Z + 2 * qp1.Z * xp2.Y + 2 * qp2.Y * xp1.Z + 2 * qp2.Z * xp1.Y - xp1.Y * (2 * p2.Y * xp2.X - xp2.Y * (2 * p1.X + 2 * p2.X)) + xp1.Z * (2 * p2.Z * xp2.X - xp2.Z * (2 * p1.X + 2 * p2.X)) - xp1.X * (2 * p1.Y * xp2.Y - 2 * p1.Z * xp2.Z);
				f1[5 + 10 * i] = qp1.Z * xp2.Z - qp2.X * xp1.X - qp1.Y * xp2.Y - qp2.Y * xp1.Y - qp1.X * xp2.X + qp2.Z * xp1.Z + xp1.Z * (xp2.Y * (p1.X + p2.X) - xp2.X * (p1.Y + p2.Y)) + xp1.Y * (xp2.Z * (p1.X + p2.X) + xp2.X * (p1.Z - p2.Z)) - xp1.X * (xp2.Z * (p1.Y + p2.Y) + xp2.Y * (p1.Z - p2.Z));
				f1[6 + 10 * i] = 2 * qp1.Y * xp2.Z - 2 * qp1.Z * xp2.Y - 2 * qp2.Y * xp1.Z + 2 * qp2.Z * xp1.Y - xp1.Y * (2 * p2.Y * xp2.X + xp2.Y * (2 * p1.X - 2 * p2.X)) - xp1.Z * (2 * p2.Z * xp2.X + xp2.Z * (2 * p1.X - 2 * p2.X)) + xp1.X * (2 * p1.Y * xp2.Y + 2 * p1.Z * xp2.Z);
				f1[7 + 10 * i] = 2 * qp1.Z * xp2.X - 2 * qp1.X * xp2.Z + 2 * qp2.X * xp1.Z - 2 * qp2.Z * xp1.X - xp1.X * (2 * p2.X * xp2.Y + xp2.X * (2 * p1.Y - 2 * p2.Y)) - xp1.Z * (2 * p2.Z * xp2.Y + xp2.Z * (2 * p1.Y - 2 * p2.Y)) + xp1.Y * (2 * p1.X * xp2.X + 2 * p1.Z * xp2.Z);
				f1[8 + 10 * i] = 2 * qp1.X * xp2.Y - 2 * qp1.Y * xp2.X - 2 * qp2.X * xp1.Y + 2 * qp2.Y * xp1.X - xp1.X * (2 * p2.X * xp2.Z + xp2.X * (2 * p1.Z - 2 * p2.Z)) - xp1.Y * (2 * p2.Y * xp2.Z + xp2.Y * (2 * p1.Z - 2 * p2.Z)) + xp1.Z * (2 * p1.X * xp2.X + 2 * p1.Y * xp2.Y);
				f1[9 + 10 * i] = xp1.Y * (xp2.Z * (p1.X - p2.X) - xp2.X * (p1.Z - p2.Z)) - xp1.Z * (xp2.Y * (p1.X - p2.X) - xp2.X * (p1.Y - p2.Y)) - xp1.X * (xp2.Z * (p1.Y - p2.Y) - xp2.Y * (p1.Z - p2.Z)) + qp1.X * xp2.X + qp2.X * xp1.X + qp1.Y * xp2.Y + qp2.Y * xp1.Y + qp1.Z * xp2.Z + qp2.Z * xp1.Z;
				f2[0 + 10 * i] = xp1.Z * (x1.X * xp2.Y + x1.Y * xp2.X) - xp1.Y * (x1.X * xp2.Z + x1.Z * xp2.X) + xp1.X * (x1.Y * xp2.Z - x1.Z * xp2.Y);
				f2[1 + 10 * i] = 2 * x1.Z * xp1.X * xp2.X - xp1.Z * (2 * x1.X * xp2.X - 2 * x1.Y * xp2.Y) - 2 * x1.Z * xp1.Y * xp2.Y;
				f2[2 + 10 * i] = xp1.X * (x1.Y * xp2.Z + x1.Z * xp2.Y) - xp1.Z * (x1.X * xp2.Y + x1.Y * xp2.X) - xp1.Y * (x1.X * xp2.Z - x1.Z * xp2.X);
				f2[3 + 10 * i] = xp1.Y * (2 * x1.X * xp2.X - 2 * x1.Z * xp2.Z) - 2 * x1.Y * xp1.X * xp2.X + 2 * x1.Y * xp1.Z * xp2.Z;
				f2[4 + 10 * i] = 2 * x1.X * xp1.Y * xp2.Y - xp1.X * (2 * x1.Y * xp2.Y - 2 * x1.Z * xp2.Z) - 2 * x1.X * xp1.Z * xp2.Z;
				f2[5 + 10 * i] = xp1.Y * (x1.X * xp2.Z + x1.Z * xp2.X) + xp1.Z * (x1.X * xp2.Y - x1.Y * xp2.X) - xp1.X * (x1.Y * xp2.Z + x1.Z * xp2.Y);
				f2[6 + 10 * i] = xp1.X * (2 * x1.Y * xp2.Y + 2 * x1.Z * xp2.Z) - 2 * x1.X * xp1.Y * xp2.Y - 2 * x1.X * xp1.Z * xp2.Z;
				f2[7 + 10 * i] = xp1.Y * (2 * x1.X * xp2.X + 2 * x1.Z * xp2.Z) - 2 * x1.Y * xp1.X * xp2.X - 2 * x1.Y * xp1.Z * xp2.Z;
				f2[8 + 10 * i] = xp1.Z * (2 * x1.X * xp2.X + 2 * x1.Y * xp2.Y) - 2 * x1.Z * xp1.X * xp2.X - 2 * x1.Z * xp1.Y * xp2.Y;
				f2[9 + 10 * i] = xp1.Y * (x1.X * xp2.Z - x1.Z * xp2.X) - xp1.Z * (x1.X * xp2.Y - x1.Y * xp2.X) - xp1.X * (x1.Y * xp2.Z - x1.Z * xp2.Y);
				f3[0 + 10 * i] = xp1.Y * (x2.X * xp2.Z - x2.Z * xp2.X) - xp1.Z * (x2.X * xp2.Y - x2.Y * xp2.X) + xp1.X * (x2.Y * xp2.Z - x2.Z * xp2.Y);
				f3[1 + 10 * i] = xp1.Y * (2 * x2.Y * xp2.Z - 2 * x2.Z * xp2.Y) - xp1.X * (2 * x2.X * xp2.Z - 2 * x2.Z * xp2.X);
				f3[2 + 10 * i] = -xp1.Z * (x2.X * xp2.Y - x2.Y * xp2.X) - xp1.Y * (x2.X * xp2.Z - x2.Z * xp2.X) - xp1.X * (x2.Y * xp2.Z - x2.Z * xp2.Y);
				f3[3 + 10 * i] = xp1.X * (2 * x2.X * xp2.Y - 2 * x2.Y * xp2.X) + xp1.Z * (2 * x2.Y * xp2.Z - 2 * x2.Z * xp2.Y);
				f3[4 + 10 * i] = xp1.Y * (2 * x2.X * xp2.Y - 2 * x2.Y * xp2.X) - xp1.Z * (2 * x2.X * xp2.Z - 2 * x2.Z * xp2.X);
				f3[5 + 10 * i] = xp1.Z * (x2.X * xp2.Y - x2.Y * xp2.X) + xp1.Y * (x2.X * xp2.Z - x2.Z * xp2.X) - xp1.X * (x2.Y * xp2.Z - x2.Z * xp2.Y);
				f3[6 + 10 * i] = xp1.Y * (2 * x2.X * xp2.Y - 2 * x2.Y * xp2.X) + xp1.Z * (2 * x2.X * xp2.Z - 2 * x2.Z * xp2.X);
				f3[7 + 10 * i] = xp1.Z * (2 * x2.Y * xp2.Z - 2 * x2.Z * xp2.Y) - xp1.X * (2 * x2.X * xp2.Y - 2 * x2.Y * xp2.X);
				f3[8 + 10 * i] = -xp1.X * (2 * x2.X * xp2.Z - 2 * x2.Z * xp2.X) - xp1.Y * (2 * x2.Y * xp2.Z - 2 * x2.Z * xp2.Y);
				f3[9 + 10 * i] = xp1.Z * (x2.X * xp2.Y - x2.Y * xp2.X) - xp1.Y * (x2.X * xp2.Z - x2.Z * xp2.X) + xp1.X * (x2.Y * xp2.Z - x2.Z * xp2.Y);
			}

			Span<double> c = m.Slice(84 * eqK, 84);

			// Compute the determinant by expansion along first column
			Mul2x2(f2[10..], f3[20..], p4);
			Mul2x2Subtract(f2[20..], f3[10..], p4);
			Mul2x4Add(f1, p4, c);

			Mul2x2(f2[20..], f3, p4);
			Mul2x2Subtract(f2, f3[20..], p4);
			Mul2x4Add(f1[10..], p4, c);

			Mul2x2(f2, f3[10..], p4);
			Mul2x2Subtract(f2[10..], f3, p4);
			Mul2x4Add(f1[20..], p4, c);
		}
	}
}
