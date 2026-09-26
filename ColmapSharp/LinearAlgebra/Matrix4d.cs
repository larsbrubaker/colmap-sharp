// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Matrix4d: fixed-size 4x4 matrix of doubles, the replacement for Eigen::Matrix4d
// (homogeneous transforms, quaternion averaging accumulators). Written here to Eigen's
// documented semantics; Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md). Siblings:
// Matrix3d and Matrix3x4d (its top three rows). Tests:
// ColmapSharp.Tests/LinearAlgebra/MatrixTests.cs.
//
// Layout and arithmetic follow Matrix3d.cs: column-major storage, a row-major
// 16-argument constructor like Eigen's comma initializer, left-to-right product sums, no
// FMA. Determinant and Inverse use the Laplace expansion by complementary 2x2 minors of
// the top two and bottom two rows (D. Eberly, "The Laplace Expansion Theorem: Computing
// the Determinants and Inverses of Matrices", Geometric Tools, 2008), written here from
// the published formulas. Tier B: Eigen's closed form may order the terms differently.
// No singularity check, like Eigen.

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Fixed-size 4x4 matrix of doubles. Replacement for Eigen::Matrix4d.
/// </summary>
public readonly struct Matrix4d : IEquatable<Matrix4d>
{
	private const int Size = 4;

	// Column-major: element (row, col) is at col * 4 + row.
	private readonly Buffer16 _m;

	private Matrix4d(in Buffer16 m)
	{
		_m = m;
	}

	/// <summary>
	/// Creates the matrix from its coefficients in row-major reading order, like Eigen's
	/// comma initializer.
	/// </summary>
	public Matrix4d(
		double m00, double m01, double m02, double m03,
		double m10, double m11, double m12, double m13,
		double m20, double m21, double m22, double m23,
		double m30, double m31, double m32, double m33)
	{
		_m[0] = m00;
		_m[1] = m10;
		_m[2] = m20;
		_m[3] = m30;
		_m[4] = m01;
		_m[5] = m11;
		_m[6] = m21;
		_m[7] = m31;
		_m[8] = m02;
		_m[9] = m12;
		_m[10] = m22;
		_m[11] = m32;
		_m[12] = m03;
		_m[13] = m13;
		_m[14] = m23;
		_m[15] = m33;
	}

	/// <summary>The zero matrix.</summary>
	public static Matrix4d Zero => default;

	/// <summary>The identity matrix.</summary>
	public static Matrix4d Identity => new(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

	/// <summary>Coefficient at (row, col).</summary>
	public double this[int row, int col]
	{
		get
		{
			if ((uint)row >= Size || (uint)col >= Size)
			{
				throw new ArgumentOutOfRangeException((uint)row < Size ? nameof(col) : nameof(row));
			}

			return _m[col * Size + row];
		}
	}

	/// <summary>Coefficient by its column-major index (col * 4 + row).</summary>
	internal double AtColumnMajor(int index) => _m[index];

	/// <summary>Builds a matrix from 16 values in column-major order (Eigen's memory layout).</summary>
	public static Matrix4d FromColumnMajor(ReadOnlySpan<double> values)
	{
		if (values.Length != 16)
		{
			throw new ArgumentException("Expected 16 values.", nameof(values));
		}

		var m = default(Buffer16);
		values.CopyTo(m);
		return new Matrix4d(m);
	}

	/// <summary>
	/// The homogeneous transform with top rows P and bottom row (0, 0, 0, 1), as COLMAP
	/// builds it with Matrix4d::Identity() then topRows&lt;3&gt;() = P.
	/// </summary>
	public static Matrix4d FromTopRows(Matrix3x4d topRows)
	{
		return new Matrix4d(
			topRows[0, 0], topRows[0, 1], topRows[0, 2], topRows[0, 3],
			topRows[1, 0], topRows[1, 1], topRows[1, 2], topRows[1, 3],
			topRows[2, 0], topRows[2, 1], topRows[2, 2], topRows[2, 3],
			0, 0, 0, 1);
	}

	/// <summary>Builds a matrix whose rows are the given vectors.</summary>
	public static Matrix4d FromRows(Vector4d r0, Vector4d r1, Vector4d r2, Vector4d r3)
	{
		return new Matrix4d(
			r0.X, r0.Y, r0.Z, r0.W,
			r1.X, r1.Y, r1.Z, r1.W,
			r2.X, r2.Y, r2.Z, r2.W,
			r3.X, r3.Y, r3.Z, r3.W);
	}

	/// <summary>Copies the 16 coefficients out in column-major order.</summary>
	public void CopyToColumnMajor(Span<double> destination)
	{
		((ReadOnlySpan<double>)_m).CopyTo(destination);
	}

	/// <summary>Row i as a vector.</summary>
	public Vector4d Row(int i) => new(this[i, 0], this[i, 1], this[i, 2], this[i, 3]);

	/// <summary>Column j as a vector.</summary>
	public Vector4d Col(int j) => new(this[0, j], this[1, j], this[2, j], this[3, j]);

	/// <summary>The top three rows, Eigen's topRows&lt;3&gt;().</summary>
	public Matrix3x4d TopRows3() => Matrix3x4d.FromColumns(Col(0).Head3(), Col(1).Head3(), Col(2).Head3(), Col(3).Head3());

	/// <summary>The top-left 3x3 block, Eigen's topLeftCorner&lt;3, 3&gt;().</summary>
	public Matrix3d TopLeft3x3() => Matrix3d.FromColumns(Col(0).Head3(), Col(1).Head3(), Col(2).Head3());

	/// <summary>
	/// Sum of the diagonal, left to right. Unverified: Matrix3d.Trace showed Eigen groups a
	/// diagonal sum its own way, so this grouping may not match Eigen bit for bit.
	/// </summary>
	public double Trace() => _m[0] + _m[5] + _m[10] + _m[15];

	/// <summary>The transpose.</summary>
	public Matrix4d Transpose()
	{
		var m = default(Buffer16);
		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 4; i++)
			{
				m[j * 4 + i] = _m[i * 4 + j];
			}
		}

		return new Matrix4d(m);
	}

	/// <summary>Frobenius norm.</summary>
	public double Norm()
	{
		double sum = _m[0] * _m[0];
		for (int i = 1; i < 16; i++)
		{
			sum += _m[i] * _m[i];
		}

		return Math.Sqrt(sum);
	}

	/// <summary>Determinant by Laplace expansion over 2x2 minors of rows 0-1 and rows 2-3.</summary>
	public double Determinant()
	{
		Minors(out double s0, out double s1, out double s2, out double s3, out double s4, out double s5,
			out double c0, out double c1, out double c2, out double c3, out double c4, out double c5);
		return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
	}

	/// <summary>
	/// The inverse, as the adjugate divided by the determinant, both from the Laplace
	/// expansion over 2x2 minors. No singularity check, like Eigen's inverse().
	/// </summary>
	public Matrix4d Inverse()
	{
		Minors(out double s0, out double s1, out double s2, out double s3, out double s4, out double s5,
			out double c0, out double c1, out double c2, out double c3, out double c4, out double c5);
		double det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;

		double a00 = _m[0], a01 = _m[4], a02 = _m[8], a03 = _m[12];
		double a10 = _m[1], a11 = _m[5], a12 = _m[9], a13 = _m[13];
		double a20 = _m[2], a21 = _m[6], a22 = _m[10], a23 = _m[14];
		double a30 = _m[3], a31 = _m[7], a32 = _m[11], a33 = _m[15];

		return new Matrix4d(
			(a11 * c5 - a12 * c4 + a13 * c3) / det,
			(-a01 * c5 + a02 * c4 - a03 * c3) / det,
			(a31 * s5 - a32 * s4 + a33 * s3) / det,
			(-a21 * s5 + a22 * s4 - a23 * s3) / det,
			(-a10 * c5 + a12 * c2 - a13 * c1) / det,
			(a00 * c5 - a02 * c2 + a03 * c1) / det,
			(-a30 * s5 + a32 * s2 - a33 * s1) / det,
			(a20 * s5 - a22 * s2 + a23 * s1) / det,
			(a10 * c4 - a11 * c2 + a13 * c0) / det,
			(-a00 * c4 + a01 * c2 - a03 * c0) / det,
			(a30 * s4 - a31 * s2 + a33 * s0) / det,
			(-a20 * s4 + a21 * s2 - a23 * s0) / det,
			(-a10 * c3 + a11 * c1 - a12 * c0) / det,
			(a00 * c3 - a01 * c1 + a02 * c0) / det,
			(-a30 * s3 + a31 * s1 - a32 * s0) / det,
			(a20 * s3 - a21 * s1 + a22 * s0) / det);
	}

	// The six 2x2 minors of rows 0-1 (s) and of rows 2-3 (c), indexed as in Eberly's paper:
	// s0 = cols 01, s1 = 02, s2 = 03, s3 = 12, s4 = 13, s5 = 23; c5 pairs with s0, etc.
	private void Minors(out double s0, out double s1, out double s2, out double s3, out double s4, out double s5,
		out double c0, out double c1, out double c2, out double c3, out double c4, out double c5)
	{
		double a00 = _m[0], a01 = _m[4], a02 = _m[8], a03 = _m[12];
		double a10 = _m[1], a11 = _m[5], a12 = _m[9], a13 = _m[13];
		double a20 = _m[2], a21 = _m[6], a22 = _m[10], a23 = _m[14];
		double a30 = _m[3], a31 = _m[7], a32 = _m[11], a33 = _m[15];
		s0 = a00 * a11 - a10 * a01;
		s1 = a00 * a12 - a10 * a02;
		s2 = a00 * a13 - a10 * a03;
		s3 = a01 * a12 - a11 * a02;
		s4 = a01 * a13 - a11 * a03;
		s5 = a02 * a13 - a12 * a03;
		c5 = a22 * a33 - a32 * a23;
		c4 = a21 * a33 - a31 * a23;
		c3 = a21 * a32 - a31 * a22;
		c2 = a20 * a33 - a30 * a23;
		c1 = a20 * a32 - a30 * a22;
		c0 = a20 * a31 - a30 * a21;
	}

	/// <summary>
	/// Eigen's isApprox with the Frobenius norm:
	/// ||a - b|| &lt;= precision * min(||a||, ||b||).
	/// </summary>
	public bool IsApprox(Matrix4d other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		return LinearAlgebraConstants.IsApprox(_m, other._m, precision);
	}

	/// <summary>Matrix sum.</summary>
	public static Matrix4d operator +(in Matrix4d a, in Matrix4d b)
	{
		var m = default(Buffer16);
		for (int i = 0; i < 16; i++)
		{
			m[i] = a._m[i] + b._m[i];
		}

		return new Matrix4d(m);
	}

	/// <summary>Matrix difference.</summary>
	public static Matrix4d operator -(in Matrix4d a, in Matrix4d b)
	{
		var m = default(Buffer16);
		for (int i = 0; i < 16; i++)
		{
			m[i] = a._m[i] - b._m[i];
		}

		return new Matrix4d(m);
	}

	/// <summary>Negation.</summary>
	public static Matrix4d operator -(in Matrix4d a)
	{
		var m = default(Buffer16);
		for (int i = 0; i < 16; i++)
		{
			m[i] = -a._m[i];
		}

		return new Matrix4d(m);
	}

	/// <summary>Scalar product.</summary>
	public static Matrix4d operator *(in Matrix4d a, double s)
	{
		var m = default(Buffer16);
		for (int i = 0; i < 16; i++)
		{
			m[i] = a._m[i] * s;
		}

		return new Matrix4d(m);
	}

	/// <summary>Scalar product.</summary>
	public static Matrix4d operator *(double s, in Matrix4d a) => a * s;

	/// <summary>Scalar division (divides each coefficient, as Eigen does).</summary>
	public static Matrix4d operator /(in Matrix4d a, double s)
	{
		var m = default(Buffer16);
		for (int i = 0; i < 16; i++)
		{
			m[i] = a._m[i] / s;
		}

		return new Matrix4d(m);
	}

	/// <summary>Matrix-vector product, each row a left-to-right four-term sum.</summary>
	public static Vector4d operator *(in Matrix4d a, Vector4d v)
	{
		return new Vector4d(
			a._m[0] * v.X + a._m[4] * v.Y + a._m[8] * v.Z + a._m[12] * v.W,
			a._m[1] * v.X + a._m[5] * v.Y + a._m[9] * v.Z + a._m[13] * v.W,
			a._m[2] * v.X + a._m[6] * v.Y + a._m[10] * v.Z + a._m[14] * v.W,
			a._m[3] * v.X + a._m[7] * v.Y + a._m[11] * v.Z + a._m[15] * v.W);
	}

	/// <summary>Matrix product, left-to-right sums.</summary>
	public static Matrix4d operator *(in Matrix4d a, in Matrix4d b)
	{
		var m = default(Buffer16);
		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 4; i++)
			{
				m[j * 4 + i] = a._m[i] * b._m[j * 4] + a._m[4 + i] * b._m[j * 4 + 1]
					+ a._m[8 + i] * b._m[j * 4 + 2] + a._m[12 + i] * b._m[j * 4 + 3];
			}
		}

		return new Matrix4d(m);
	}

	/// <summary>Eigen's operator==: every coefficient compares equal with double ==.</summary>
	public static bool operator ==(in Matrix4d a, in Matrix4d b)
	{
		for (int i = 0; i < 16; i++)
		{
			if (a._m[i] != b._m[i])
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(in Matrix4d a, in Matrix4d b) => !(a == b);

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Matrix4d other) => ((ReadOnlySpan<double>)_m).SequenceEqual(other._m);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Matrix4d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode()
	{
		var hash = default(HashCode);
		for (int i = 0; i < 16; i++)
		{
			hash.Add(_m[i]);
		}

		return hash.ToHashCode();
	}

	/// <inheritdoc/>
	public override string ToString()
	{
		return string.Create(CultureInfo.InvariantCulture, $"[{Row(0)}, {Row(1)}, {Row(2)}, {Row(3)}]");
	}
}
