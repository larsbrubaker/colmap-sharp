// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MatrixUtils: port of colmap/math/matrix.h, DecomposeMatrixRQ (RQ decomposition of a
// square matrix through a Householder QR of its flipped transpose). Used by
// geometry/pose.cc DecomposeProjectionMatrix with a 3x3 matrix. Builds on
// ColmapSharp.LinearAlgebra.HouseholderQR; neighbors are MathUtils (math.h) and the other
// Mathematics files. Tests: ColmapSharp.Tests/Mathematics/MatrixUtilsTests.cs
// (matrix_test.cc 1:1).
//
// Tier B: the result goes through a Householder QR, so it matches COLMAP within the test's
// 1e-6 tolerance, not bit for bit. COLMAP's template takes any square Eigen matrix type;
// here there is a MatrixXd version and a Matrix3d overload (the only fixed shape COLMAP
// instantiates outside tests).

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Mathematics;

/// <summary>
/// Port of the free functions in colmap/math/matrix.h.
/// </summary>
public static class MatrixUtils
{
	/// <summary>
	/// Port of colmap::DecomposeMatrixRQ. Decomposes a square A into A = R * Q with R upper
	/// triangular and Q orthogonal, made unique by requiring det(Q) &gt; 0.
	/// </summary>
	public static void DecomposeMatrixRQ(MatrixXd a, out MatrixXd r, out MatrixXd q)
	{
		if (a.Rows != a.Cols)
		{
			throw new ArgumentException($"DecomposeMatrixRQ needs a square matrix, got {a.Rows}x{a.Cols}.", nameof(a));
		}

		// A.transpose().rowwise().reverse()
		MatrixXd aFlipudTranspose = a.Transpose().ReverseCols();

		var qr = new HouseholderQR(aFlipudTranspose);
		MatrixXd q0 = qr.HouseholderQ();

		// COLMAP reads matrixQR() whole, essentials below the diagonal included, and relies
		// on the zeroing loop below to clear them.
		MatrixXd r0 = qr.MatrixQR();

		// R0.transpose().colwise().reverse(), then rowwise().reverse().
		r = r0.Transpose().ReverseRows().ReverseCols();
		for (int i = 0; i < r.Rows; i++)
		{
			for (int j = 0; j < r.Cols && (r.Cols - j) > (r.Rows - i); j++)
			{
				r[i, j] = 0;
			}
		}

		// Q0.transpose().colwise().reverse()
		q = q0.Transpose().ReverseRows();

		// Make the decomposition unique by requiring that det(Q) > 0.
		if (q.Determinant() < 0)
		{
			for (int c = 0; c < q.Cols; c++)
			{
				q[1, c] *= -1.0;
			}

			for (int row = 0; row < r.Rows; row++)
			{
				r[row, 1] *= -1.0;
			}
		}
	}

	/// <summary>
	/// Port of colmap::DecomposeMatrixRQ for Eigen::Matrix3d (the shape
	/// DecomposeProjectionMatrix uses).
	/// </summary>
	public static void DecomposeMatrixRQ(in Matrix3d a, out Matrix3d r, out Matrix3d q)
	{
		DecomposeMatrixRQ(MatrixXd.From(a), out MatrixXd rx, out MatrixXd qx);
		r = rx.ToMatrix3d();
		q = qx.ToMatrix3d();
	}
}
