// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// MatrixXd arithmetic: coefficient-wise operators, matrix-matrix and matrix-vector
// products, and the transposed products A^T * B, A^T * v and A^T * A that COLMAP's
// estimators form on tall constraint matrices without materializing A^T. Part of MatrixXd
// (storage and the arithmetic-order contract are in MatrixXd.cs). Every product coefficient
// is a left-to-right sum over k seeded with the k = 0 term, no FMA.

namespace ColmapSharp.LinearAlgebra;

public sealed partial class MatrixXd
{
	/// <summary>Coefficient-wise sum.</summary>
	public static MatrixXd operator +(MatrixXd a, MatrixXd b)
	{
		RequireSameShape(a, b);
		var r = new MatrixXd(a.Rows, a.Cols);
		for (int i = 0; i < r._data.Length; i++)
		{
			r._data[i] = a._data[i] + b._data[i];
		}

		return r;
	}

	/// <summary>Coefficient-wise difference.</summary>
	public static MatrixXd operator -(MatrixXd a, MatrixXd b)
	{
		RequireSameShape(a, b);
		var r = new MatrixXd(a.Rows, a.Cols);
		for (int i = 0; i < r._data.Length; i++)
		{
			r._data[i] = a._data[i] - b._data[i];
		}

		return r;
	}

	/// <summary>Negation.</summary>
	public static MatrixXd operator -(MatrixXd a) => a * -1.0;

	/// <summary>Scalar product.</summary>
	public static MatrixXd operator *(MatrixXd a, double s)
	{
		var r = new MatrixXd(a.Rows, a.Cols);
		for (int i = 0; i < r._data.Length; i++)
		{
			r._data[i] = a._data[i] * s;
		}

		return r;
	}

	/// <summary>Scalar product.</summary>
	public static MatrixXd operator *(double s, MatrixXd a) => a * s;

	/// <summary>Scalar division (divides each coefficient, like Eigen).</summary>
	public static MatrixXd operator /(MatrixXd a, double s)
	{
		var r = new MatrixXd(a.Rows, a.Cols);
		for (int i = 0; i < r._data.Length; i++)
		{
			r._data[i] = a._data[i] / s;
		}

		return r;
	}

	/// <summary>Matrix product a * b.</summary>
	public static MatrixXd operator *(MatrixXd a, MatrixXd b)
	{
		RequireSameLength(a.Cols, b.Rows);
		var r = new MatrixXd(a.Rows, b.Cols);
		if (a.Cols == 0)
		{
			return r;
		}

		for (int j = 0; j < b.Cols; j++)
		{
			for (int i = 0; i < a.Rows; i++)
			{
				double sum = a._data[i] * b._data[j * b.Rows];
				for (int k = 1; k < a.Cols; k++)
				{
					sum += a._data[k * a.Rows + i] * b._data[j * b.Rows + k];
				}

				r._data[j * r.Rows + i] = sum;
			}
		}

		return r;
	}

	/// <summary>Matrix-vector product a * v.</summary>
	public static VectorXd operator *(MatrixXd a, VectorXd v)
	{
		RequireSameLength(a.Cols, v.Length);
		var r = new VectorXd(a.Rows);
		if (a.Cols == 0)
		{
			return r;
		}

		for (int i = 0; i < a.Rows; i++)
		{
			double sum = a._data[i] * v[0];
			for (int k = 1; k < a.Cols; k++)
			{
				sum += a._data[k * a.Rows + i] * v[k];
			}

			r[i] = sum;
		}

		return r;
	}

	/// <summary>
	/// this^T * b without forming the transpose; both operands are walked down contiguous
	/// columns, so this is the fast way to build normal equations.
	/// </summary>
	public MatrixXd TransposeTimes(MatrixXd b)
	{
		RequireSameLength(Rows, b.Rows);
		var r = new MatrixXd(Cols, b.Cols);
		for (int j = 0; j < b.Cols; j++)
		{
			ReadOnlySpan<double> bj = b.ColumnSpan(j);
			for (int i = 0; i < Cols; i++)
			{
				r._data[j * Cols + i] = VectorXd.Dot(ColumnSpan(i), bj);
			}
		}

		return r;
	}

	/// <summary>this^T * v without forming the transpose.</summary>
	public VectorXd TransposeTimes(VectorXd v)
	{
		RequireSameLength(Rows, v.Length);
		var r = new VectorXd(Cols);
		for (int i = 0; i < Cols; i++)
		{
			r[i] = VectorXd.Dot(ColumnSpan(i), v.AsSpan());
		}

		return r;
	}

	/// <summary>
	/// The Gram matrix this^T * this (Cols x Cols, symmetric). Each coefficient is computed
	/// once and mirrored, so the result is exactly symmetric.
	/// </summary>
	public MatrixXd TransposeTimesSelf()
	{
		var r = new MatrixXd(Cols, Cols);
		for (int j = 0; j < Cols; j++)
		{
			ReadOnlySpan<double> cj = ColumnSpan(j);
			for (int i = j; i < Cols; i++)
			{
				double value = VectorXd.Dot(ColumnSpan(i), cj);
				r._data[j * Cols + i] = value;
				r._data[i * Cols + j] = value;
			}
		}

		return r;
	}

	private static void RequireSameShape(MatrixXd a, MatrixXd b)
	{
		if (a.Rows != b.Rows || a.Cols != b.Cols)
		{
			throw new ArgumentException($"Shape mismatch: {a.Rows}x{a.Cols} vs {b.Rows}x{b.Cols}.");
		}
	}
}
