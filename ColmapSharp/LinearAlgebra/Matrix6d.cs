// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Matrix6d: fixed-size 6x6 matrix of doubles, the replacement for COLMAP's
// Eigen::Matrix6d (the Rigid3d adjoint and 6-DoF pose covariances). Written here to Eigen's
// documented semantics; Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md). Minimal on
// purpose: only what Geometry/Rigid3d.cs and its tests use. Siblings: Matrix3d (its 3x3
// blocks). Tests: ColmapSharp.Tests/LinearAlgebra/MatrixTests.cs and, through the adjoint,
// ColmapSharp.Tests/Geometry/Rigid3dTests.cs.
//
// Layout and arithmetic follow Matrix3d.cs: column-major storage, left-to-right product
// sums starting from the first term, no FMA. Tier B for products: Eigen evaluates a 6x6
// product with its vectorized coefficient-based kernel, which the pycolmap wheel may
// contract into FMAs, and no pycolmap binding exposes a 6x6 product to pin it.

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Fixed-size 6x6 matrix of doubles. Replacement for Eigen::Matrix6d.
/// </summary>
public readonly struct Matrix6d : IEquatable<Matrix6d>
{
	private const int Size = 6;

	// Column-major: element (row, col) is at col * 6 + row.
	private readonly Buffer36 _m;

	private Matrix6d(in Buffer36 m)
	{
		_m = m;
	}

	/// <summary>The zero matrix.</summary>
	public static Matrix6d Zero => default;

	/// <summary>The identity matrix.</summary>
	public static Matrix6d Identity
	{
		get
		{
			var m = default(Buffer36);
			for (int i = 0; i < Size; i++)
			{
				m[i * Size + i] = 1;
			}

			return new Matrix6d(m);
		}
	}

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

	/// <summary>Builds a matrix from 36 values in column-major order (Eigen's memory layout).</summary>
	public static Matrix6d FromColumnMajor(ReadOnlySpan<double> values)
	{
		if (values.Length != Size * Size)
		{
			throw new ArgumentException("Expected 36 values.", nameof(values));
		}

		var m = default(Buffer36);
		values.CopyTo(m);
		return new Matrix6d(m);
	}

	/// <summary>
	/// [topLeft, topRight; bottomLeft, bottomRight], Eigen's block&lt;3, 3&gt;(r, c)
	/// assignments.
	/// </summary>
	public static Matrix6d FromBlocks(Matrix3d topLeft, Matrix3d topRight, Matrix3d bottomLeft, Matrix3d bottomRight)
	{
		var m = default(Buffer36);
		for (int col = 0; col < 3; col++)
		{
			for (int row = 0; row < 3; row++)
			{
				m[col * Size + row] = topLeft[row, col];
				m[(col + 3) * Size + row] = topRight[row, col];
				m[col * Size + row + 3] = bottomLeft[row, col];
				m[(col + 3) * Size + row + 3] = bottomRight[row, col];
			}
		}

		return new Matrix6d(m);
	}

	/// <summary>The 3x3 block starting at (row, col), Eigen's block&lt;3, 3&gt;(row, col).</summary>
	public Matrix3d Block3(int row, int col)
	{
		return new Matrix3d(
			this[row, col], this[row, col + 1], this[row, col + 2],
			this[row + 1, col], this[row + 1, col + 1], this[row + 1, col + 2],
			this[row + 2, col], this[row + 2, col + 1], this[row + 2, col + 2]);
	}

	/// <summary>Copies the 36 coefficients out in column-major order.</summary>
	public void CopyToColumnMajor(Span<double> destination)
	{
		((ReadOnlySpan<double>)_m).CopyTo(destination);
	}

	/// <summary>The transpose.</summary>
	public Matrix6d Transpose()
	{
		var m = default(Buffer36);
		for (int col = 0; col < Size; col++)
		{
			for (int row = 0; row < Size; row++)
			{
				m[row * Size + col] = _m[col * Size + row];
			}
		}

		return new Matrix6d(m);
	}

	/// <summary>Frobenius norm, Eigen's norm() on a matrix (left-to-right sum of squares).</summary>
	public double Norm()
	{
		double sum = _m[0] * _m[0];
		for (int i = 1; i < Size * Size; i++)
		{
			sum += _m[i] * _m[i];
		}

		return Math.Sqrt(sum);
	}

	/// <summary>
	/// Eigen's isApprox with the Frobenius norm:
	/// ||a - b|| &lt;= precision * min(||a||, ||b||).
	/// </summary>
	public bool IsApprox(Matrix6d other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		return LinearAlgebraConstants.IsApprox(_m, other._m, precision);
	}

	/// <summary>Matrix sum.</summary>
	public static Matrix6d operator +(in Matrix6d a, in Matrix6d b)
	{
		var m = default(Buffer36);
		for (int i = 0; i < Size * Size; i++)
		{
			m[i] = a._m[i] + b._m[i];
		}

		return new Matrix6d(m);
	}

	/// <summary>Matrix difference.</summary>
	public static Matrix6d operator -(in Matrix6d a, in Matrix6d b)
	{
		var m = default(Buffer36);
		for (int i = 0; i < Size * Size; i++)
		{
			m[i] = a._m[i] - b._m[i];
		}

		return new Matrix6d(m);
	}

	/// <summary>Negation.</summary>
	public static Matrix6d operator -(in Matrix6d a)
	{
		var m = default(Buffer36);
		for (int i = 0; i < Size * Size; i++)
		{
			m[i] = -a._m[i];
		}

		return new Matrix6d(m);
	}

	/// <summary>Matrix product; (i, j) = a(i,0)*b(0,j) + ... + a(i,5)*b(5,j), left to right.</summary>
	public static Matrix6d operator *(in Matrix6d a, in Matrix6d b)
	{
		var m = default(Buffer36);
		for (int j = 0; j < Size; j++)
		{
			for (int i = 0; i < Size; i++)
			{
				double sum = a._m[i] * b._m[j * Size];
				for (int k = 1; k < Size; k++)
				{
					sum += a._m[k * Size + i] * b._m[j * Size + k];
				}

				m[j * Size + i] = sum;
			}
		}

		return new Matrix6d(m);
	}

	/// <summary>Eigen's operator==: every coefficient compares equal with double ==.</summary>
	public static bool operator ==(in Matrix6d a, in Matrix6d b)
	{
		for (int i = 0; i < Size * Size; i++)
		{
			if (a._m[i] != b._m[i])
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(in Matrix6d a, in Matrix6d b) => !(a == b);

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Matrix6d other) => ((ReadOnlySpan<double>)_m).SequenceEqual(other._m);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Matrix6d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode()
	{
		var hash = default(HashCode);
		for (int i = 0; i < Size * Size; i++)
		{
			hash.Add(_m[i]);
		}

		return hash.ToHashCode();
	}

	/// <inheritdoc/>
	public override string ToString()
	{
		var rows = new string[Size];
		for (int i = 0; i < Size; i++)
		{
			rows[i] = string.Create(
				CultureInfo.InvariantCulture,
				$"({this[i, 0]:R}, {this[i, 1]:R}, {this[i, 2]:R}, {this[i, 3]:R}, {this[i, 4]:R}, {this[i, 5]:R})");
		}

		return "[" + string.Join(", ", rows) + "]";
	}
}
