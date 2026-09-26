// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// MatrixTests (C#-only; COLMAP has no test for Eigen itself): Matrix2d, Matrix3d,
// Matrix3x4d and Matrix4d in ColmapSharp/LinearAlgebra. Covers the layout contract
// (row-major constructor like Eigen's comma initializer, column-major FromColumnMajor like
// Eigen's memory), products, transpose, determinant and inverse (inverse * M = I within
// isApprox), and the block helpers COLMAP's Rigid3d/Sim3d use.

using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.LinearAlgebra;

public class MatrixTests
{
	private static readonly Matrix3d A3 = new(2, -1, 0.5, 3, 4, -2, 1, 0.25, 5);

	private static readonly Matrix4d A4 = new(
		4, 1, -2, 0.5,
		1, 3, 0, -1,
		-2, 0.75, 5, 2,
		0.5, -1, 2, 6);

	[Test]
	public async Task Layout_RowMajorConstructorAndColumnMajorStorage()
	{
		var fromColumns = Matrix3d.FromColumnMajor([1, 4, 7, 2, 5, 8, 3, 6, 9]);
		var m = new Matrix3d(1, 2, 3, 4, 5, 6, 7, 8, 9);
		var data = new double[9];
		m.CopyToColumnMajor(data);
		using (Assert.Multiple())
		{
			await Assert.That(fromColumns).IsEqualTo(m);
			await Assert.That(m[0, 2]).IsEqualTo(3.0);
			await Assert.That(m[2, 0]).IsEqualTo(7.0);
			await Assert.That(data).IsEquivalentTo(new double[] { 1, 4, 7, 2, 5, 8, 3, 6, 9 });
			await Assert.That(m.Row(1)).IsEqualTo(new Vector3d(4, 5, 6));
			await Assert.That(m.Col(1)).IsEqualTo(new Vector3d(2, 5, 8));
			await Assert.That(Matrix3d.FromRows(m.Row(0), m.Row(1), m.Row(2))).IsEqualTo(m);
			await Assert.That(Matrix3d.FromColumns(m.Col(0), m.Col(1), m.Col(2))).IsEqualTo(m);
			await Assert.That(m.Transpose()).IsEqualTo(Matrix3d.FromColumnMajor([1, 2, 3, 4, 5, 6, 7, 8, 9]));
			await Assert.That(() => m[3, 0]).Throws<ArgumentOutOfRangeException>();
		}
	}

	[Test]
	public async Task Matrix3d_ProductsDeterminantInverse()
	{
		var m = new Matrix3d(1, 2, 3, 4, 5, 6, 7, 8, 10);
		using (Assert.Multiple())
		{
			await Assert.That(m * new Vector3d(1, 0, -1)).IsEqualTo(new Vector3d(-2, -2, -3));
			await Assert.That(m * Matrix3d.Identity).IsEqualTo(m);
			await Assert.That(Matrix3d.Identity * m).IsEqualTo(m);
			await Assert.That(m.Determinant()).IsEqualTo(-3.0);
			await Assert.That(m.Trace()).IsEqualTo(16.0);
			await Assert.That((m.Inverse() * m).IsApprox(Matrix3d.Identity)).IsTrue();
			await Assert.That((A3 * A3.Inverse()).IsApprox(Matrix3d.Identity)).IsTrue();
			await Assert.That((A3 * m).Transpose().IsApprox(m.Transpose() * A3.Transpose())).IsTrue();
			await Assert.That(A3.Determinant() * A3.Inverse().Determinant()).IsEqualTo(1.0).Within(1e-14);
			await Assert.That(Matrix3d.FromDiagonal(new Vector3d(1, 2, 3)).Diagonal()).IsEqualTo(new Vector3d(1, 2, 3));
			await Assert.That(-m + m).IsEqualTo(Matrix3d.Zero);
			await Assert.That(m * 2 / 2).IsEqualTo(m);
		}
	}

	[Test]
	public async Task Matrix2d_ProductsDeterminantInverse()
	{
		var m = new Matrix2d(4, 7, 2, 6);
		using (Assert.Multiple())
		{
			await Assert.That(m.Determinant()).IsEqualTo(10.0);
			await Assert.That(m.Inverse()).IsEqualTo(new Matrix2d(0.6, -0.7, -0.2, 0.4));
			await Assert.That((m * m.Inverse()).IsApprox(Matrix2d.Identity)).IsTrue();
			await Assert.That(m * new Vector2d(1, 1)).IsEqualTo(new Vector2d(11, 8));
			await Assert.That(m.Transpose()).IsEqualTo(Matrix2d.FromColumnMajor([4, 7, 2, 6]));
			await Assert.That(Matrix2d.FromColumns(m.Col(0), m.Col(1))).IsEqualTo(m);
		}
	}

	[Test]
	public async Task Matrix4d_DeterminantAndInverse()
	{
		// A diagonal-plus-permutation matrix with a known determinant: swapping rows of
		// diag(1, 2, 3, 4) once gives -24.
		var permuted = new Matrix4d(0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4);
		using (Assert.Multiple())
		{
			await Assert.That(permuted.Determinant()).IsEqualTo(-24.0);
			await Assert.That(Matrix4d.Identity.Determinant()).IsEqualTo(1.0);
			await Assert.That((A4.Inverse() * A4).IsApprox(Matrix4d.Identity)).IsTrue();
			await Assert.That((A4 * A4.Inverse()).IsApprox(Matrix4d.Identity)).IsTrue();
			await Assert.That((A4 * A4.Transpose()).IsApprox((A4 * A4.Transpose()).Transpose())).IsTrue();
			await Assert.That(A4 * new Vector4d(1, 0, 0, 0)).IsEqualTo(A4.Col(0));
			await Assert.That(A4.Transpose().Row(2)).IsEqualTo(A4.Col(2));
		}
	}

	[Test]
	public async Task Matrix4d_DeterminantMatchesCofactorExpansion()
	{
		// Independent check: expand along the first row with Matrix3d minors.
		double expected = 0;
		for (int j = 0; j < 4; j++)
		{
			var minor = new double[9];
			int n = 0;
			for (int c = 0; c < 4; c++)
			{
				if (c == j)
				{
					continue;
				}

				for (int r = 1; r < 4; r++)
				{
					minor[n++] = A4[r, c];
				}
			}

			double sign = j % 2 == 0 ? 1 : -1;
			expected += sign * A4[0, j] * Matrix3d.FromColumnMajor(minor).Determinant();
		}

		await Assert.That(A4.Determinant()).IsEqualTo(expected).Within(1e-12);
	}

	[Test]
	public async Task Matrix3x4d_Blocks()
	{
		var t = new Vector3d(1, 2, 3);
		Matrix3x4d p = Matrix3x4d.FromBlocks(A3, t);
		var x = new Vector3d(-1, 0.5, 2);
		using (Assert.Multiple())
		{
			await Assert.That(p.LeftCols3()).IsEqualTo(A3);
			await Assert.That(p.Col(3)).IsEqualTo(t);
			await Assert.That(p.Row(0)).IsEqualTo(new Vector4d(2, -1, 0.5, 1));
			await Assert.That(p * x.Homogeneous()).IsEqualTo(A3 * x + t);
			await Assert.That(Matrix3x4d.Identity.LeftCols3()).IsEqualTo(Matrix3d.Identity);
			await Assert.That(Matrix3x4d.Identity.Col(3)).IsEqualTo(Vector3d.Zero);
			await Assert.That(Matrix4d.FromTopRows(p).TopRows3()).IsEqualTo(p);
			await Assert.That(Matrix4d.FromTopRows(p).Row(3)).IsEqualTo(new Vector4d(0, 0, 0, 1));
			await Assert.That(Matrix4d.FromTopRows(p).TopLeft3x3()).IsEqualTo(A3);
			await Assert.That((p * Matrix4d.Identity)).IsEqualTo(p);
			await Assert.That((Matrix3d.Identity * p)).IsEqualTo(p);
			await Assert.That((A3 * p).IsApprox(Matrix3x4d.FromBlocks(A3 * A3, A3 * t))).IsTrue();
			await Assert.That((p * Matrix4d.FromTopRows(p)).IsApprox(Matrix3x4d.FromBlocks(A3 * A3, A3 * t + t))).IsTrue();
		}
	}

	[Test]
	public async Task IsApprox_UsesFrobeniusNormRelativeRule()
	{
		Matrix3d scaled = A3 * (1 + 1e-13);
		using (Assert.Multiple())
		{
			await Assert.That(A3.IsApprox(scaled)).IsTrue();
			await Assert.That(A3.IsApprox(A3 * (1 + 1e-10))).IsFalse();
			await Assert.That(Matrix3d.Zero.IsApprox(Matrix3d.Zero)).IsTrue();
			await Assert.That(Matrix3d.Zero.IsApprox(Matrix3d.Identity * 1e-100)).IsFalse();
		}
	}
}
