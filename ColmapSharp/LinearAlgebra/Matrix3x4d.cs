// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Matrix3x4d: fixed-size 3x4 matrix of doubles, the replacement for COLMAP's
// Eigen::Matrix3x4d (camera projection matrices [R | t], Rigid3d/Sim3d ToMatrix). Written
// here to Eigen's documented semantics; Eigen (MPL-2.0) is not ported
// (docs/LICENSE_AUDIT.md). Siblings: Matrix3d (its left 3x3 block) and Matrix4d. Tests:
// ColmapSharp.Tests/LinearAlgebra/MatrixTests.cs.
//
// Layout and arithmetic follow Matrix3d.cs: column-major storage, a row-major
// 12-argument constructor like Eigen's comma initializer, left-to-right product sums, no
// FMA. COLMAP's P * X.homogeneous() is written here as P * X.Homogeneous(), a full
// four-term sum per row whose last term multiplies by exactly 1.

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Fixed-size 3x4 matrix of doubles. Replacement for Eigen::Matrix3x4d.
/// </summary>
public readonly struct Matrix3x4d : IEquatable<Matrix3x4d>
{
	private const int Rows = 3;
	private const int Cols = 4;

	// Column-major: element (row, col) is at col * 3 + row.
	private readonly Buffer12 _m;

	private Matrix3x4d(in Buffer12 m)
	{
		_m = m;
	}

	/// <summary>
	/// Creates the matrix from its coefficients in row-major reading order, like Eigen's
	/// comma initializer.
	/// </summary>
	public Matrix3x4d(
		double m00, double m01, double m02, double m03,
		double m10, double m11, double m12, double m13,
		double m20, double m21, double m22, double m23)
	{
		_m[0] = m00;
		_m[1] = m10;
		_m[2] = m20;
		_m[3] = m01;
		_m[4] = m11;
		_m[5] = m21;
		_m[6] = m02;
		_m[7] = m12;
		_m[8] = m22;
		_m[9] = m03;
		_m[10] = m13;
		_m[11] = m23;
	}

	/// <summary>The zero matrix.</summary>
	public static Matrix3x4d Zero => default;

	/// <summary>Eigen's Identity() for a 3x4 matrix: [I | 0].</summary>
	public static Matrix3x4d Identity => new(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);

	/// <summary>Coefficient at (row, col).</summary>
	public double this[int row, int col]
	{
		get
		{
			if ((uint)row >= Rows || (uint)col >= Cols)
			{
				throw new ArgumentOutOfRangeException((uint)row < Rows ? nameof(col) : nameof(row));
			}

			return _m[col * Rows + row];
		}
	}

	/// <summary>Builds a matrix from 12 values in column-major order (Eigen's memory layout).</summary>
	public static Matrix3x4d FromColumnMajor(ReadOnlySpan<double> values)
	{
		if (values.Length != 12)
		{
			throw new ArgumentException("Expected 12 values.", nameof(values));
		}

		var m = default(Buffer12);
		values.CopyTo(m);
		return new Matrix3x4d(m);
	}

	/// <summary>
	/// [left | col3]: the left 3x3 block and the last column, as COLMAP builds
	/// matrix.leftCols&lt;3&gt;() = R; matrix.col(3) = t.
	/// </summary>
	public static Matrix3x4d FromBlocks(Matrix3d left, Vector3d col3)
	{
		var m = default(Buffer12);
		left.CopyToColumnMajor(((Span<double>)m)[..9]);
		m[9] = col3.X;
		m[10] = col3.Y;
		m[11] = col3.Z;
		return new Matrix3x4d(m);
	}

	/// <summary>Builds a matrix whose columns are the given vectors.</summary>
	public static Matrix3x4d FromColumns(Vector3d c0, Vector3d c1, Vector3d c2, Vector3d c3)
	{
		return new Matrix3x4d(
			c0.X, c1.X, c2.X, c3.X,
			c0.Y, c1.Y, c2.Y, c3.Y,
			c0.Z, c1.Z, c2.Z, c3.Z);
	}

	/// <summary>Copies the 12 coefficients out in column-major order.</summary>
	public void CopyToColumnMajor(Span<double> destination)
	{
		((ReadOnlySpan<double>)_m).CopyTo(destination);
	}

	/// <summary>Row i as a vector.</summary>
	public Vector4d Row(int i) => new(this[i, 0], this[i, 1], this[i, 2], this[i, 3]);

	/// <summary>Column j as a vector (Col(3) is Eigen's rightCols&lt;1&gt;()).</summary>
	public Vector3d Col(int j) => new(this[0, j], this[1, j], this[2, j]);

	/// <summary>The left 3x3 block, Eigen's leftCols&lt;3&gt;().</summary>
	public Matrix3d LeftCols3() => Matrix3d.FromColumnMajor(((ReadOnlySpan<double>)_m)[..9]);

	/// <summary>Frobenius norm.</summary>
	public double Norm()
	{
		double sum = _m[0] * _m[0];
		for (int i = 1; i < 12; i++)
		{
			sum += _m[i] * _m[i];
		}

		return Math.Sqrt(sum);
	}

	/// <summary>
	/// Eigen's isApprox with the Frobenius norm:
	/// ||a - b|| &lt;= precision * min(||a||, ||b||).
	/// </summary>
	public bool IsApprox(Matrix3x4d other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		return LinearAlgebraConstants.IsApprox(_m, other._m, precision);
	}

	/// <summary>Matrix sum.</summary>
	public static Matrix3x4d operator +(in Matrix3x4d a, in Matrix3x4d b)
	{
		var m = default(Buffer12);
		for (int i = 0; i < 12; i++)
		{
			m[i] = a._m[i] + b._m[i];
		}

		return new Matrix3x4d(m);
	}

	/// <summary>Matrix difference.</summary>
	public static Matrix3x4d operator -(in Matrix3x4d a, in Matrix3x4d b)
	{
		var m = default(Buffer12);
		for (int i = 0; i < 12; i++)
		{
			m[i] = a._m[i] - b._m[i];
		}

		return new Matrix3x4d(m);
	}

	/// <summary>Negation.</summary>
	public static Matrix3x4d operator -(in Matrix3x4d a)
	{
		var m = default(Buffer12);
		for (int i = 0; i < 12; i++)
		{
			m[i] = -a._m[i];
		}

		return new Matrix3x4d(m);
	}

	/// <summary>Scalar product.</summary>
	public static Matrix3x4d operator *(in Matrix3x4d a, double s)
	{
		var m = default(Buffer12);
		for (int i = 0; i < 12; i++)
		{
			m[i] = a._m[i] * s;
		}

		return new Matrix3x4d(m);
	}

	/// <summary>Scalar product.</summary>
	public static Matrix3x4d operator *(double s, in Matrix3x4d a) => a * s;

	/// <summary>Scalar division (divides each coefficient, as Eigen does).</summary>
	public static Matrix3x4d operator /(in Matrix3x4d a, double s)
	{
		var m = default(Buffer12);
		for (int i = 0; i < 12; i++)
		{
			m[i] = a._m[i] / s;
		}

		return new Matrix3x4d(m);
	}

	/// <summary>Matrix-vector product, each row a left-to-right four-term sum.</summary>
	public static Vector3d operator *(in Matrix3x4d a, Vector4d v)
	{
		return new Vector3d(
			a._m[0] * v.X + a._m[3] * v.Y + a._m[6] * v.Z + a._m[9] * v.W,
			a._m[1] * v.X + a._m[4] * v.Y + a._m[7] * v.Z + a._m[10] * v.W,
			a._m[2] * v.X + a._m[5] * v.Y + a._m[8] * v.Z + a._m[11] * v.W);
	}

	/// <summary>3x3 times 3x4, left-to-right sums.</summary>
	public static Matrix3x4d operator *(in Matrix3d a, in Matrix3x4d b)
	{
		var m = default(Buffer12);
		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				m[j * 3 + i] = a.AtColumnMajor(i) * b._m[j * 3] + a.AtColumnMajor(3 + i) * b._m[j * 3 + 1] + a.AtColumnMajor(6 + i) * b._m[j * 3 + 2];
			}
		}

		return new Matrix3x4d(m);
	}

	/// <summary>3x4 times 4x4, left-to-right sums.</summary>
	public static Matrix3x4d operator *(in Matrix3x4d a, in Matrix4d b)
	{
		var m = default(Buffer12);
		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				m[j * 3 + i] = a._m[i] * b.AtColumnMajor(j * 4) + a._m[3 + i] * b.AtColumnMajor(j * 4 + 1)
					+ a._m[6 + i] * b.AtColumnMajor(j * 4 + 2) + a._m[9 + i] * b.AtColumnMajor(j * 4 + 3);
			}
		}

		return new Matrix3x4d(m);
	}

	/// <summary>Eigen's operator==: every coefficient compares equal with double ==.</summary>
	public static bool operator ==(in Matrix3x4d a, in Matrix3x4d b)
	{
		for (int i = 0; i < 12; i++)
		{
			if (a._m[i] != b._m[i])
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(in Matrix3x4d a, in Matrix3x4d b) => !(a == b);

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Matrix3x4d other) => ((ReadOnlySpan<double>)_m).SequenceEqual(other._m);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Matrix3x4d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode()
	{
		var hash = default(HashCode);
		for (int i = 0; i < 12; i++)
		{
			hash.Add(_m[i]);
		}

		return hash.ToHashCode();
	}

	/// <inheritdoc/>
	public override string ToString()
	{
		return string.Create(CultureInfo.InvariantCulture, $"[{Row(0)}, {Row(1)}, {Row(2)}]");
	}
}
