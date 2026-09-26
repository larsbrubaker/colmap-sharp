// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RandomEigen: colmap/math/random_eigen.h (double precision) - random vectors,
// matrices and unit quaternions drawn from COLMAP's seeded PRNG (RandomUtils.cs) instead of
// Eigen's rand()-based Random(). The geometry tests build their random poses with these
// (rigid3_test.cc, sim3_test.cc). The templates RandomEigenMatrixd<Rows, Cols> and
// RandomEigenVectord<N> become one method per fixed-size type in LinearAlgebra/, plus
// RandomEigenMatrixXd / RandomEigenVectorXd for the dynamic (and larger fixed, e.g. 12x12)
// shapes. There are no float matrix types in LinearAlgebra/, so the float variants
// (RandomEigenVectorf, RandomEigenMatrixXf) return plain arrays; the draw order is the same.
//
// Tier A (exact): each coefficient is one RandomUniformReal(-1, 1) draw, taken in Eigen's
// linear index order (column-major for matrices), so the same seed gives the same values
// as COLMAP. RandomEigenQuaterniond goes through sqrt, sin and cos; .NET calls the platform
// libm for sin and cos, which can round differently from the C++ build's (see
// docs/CPP_DIVERGENCES.md, entry 7).

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Mathematics;

/// <summary>
/// Port of the double-precision functions in colmap/math/random_eigen.h.
/// </summary>
public static class RandomEigen
{
	/// <summary>
	/// Random 3-vector with each entry uniformly distributed in [-1, 1).
	/// Port of colmap::RandomEigenVectord&lt;3&gt;.
	/// </summary>
	public static Vector3d RandomEigenVector3d()
	{
		// Separate statements: C# evaluates arguments left to right anyway, but this keeps
		// the draw order obvious (x, then y, then z, as Eigen's linear index).
		double x = RandomUtils.RandomUniformReal(-1.0, 1.0);
		double y = RandomUtils.RandomUniformReal(-1.0, 1.0);
		double z = RandomUtils.RandomUniformReal(-1.0, 1.0);
		return new Vector3d(x, y, z);
	}

	/// <summary>
	/// Random 4-vector with each entry uniformly distributed in [-1, 1).
	/// Port of colmap::RandomEigenVectord&lt;4&gt;.
	/// </summary>
	public static Vector4d RandomEigenVector4d()
	{
		double x = RandomUtils.RandomUniformReal(-1.0, 1.0);
		double y = RandomUtils.RandomUniformReal(-1.0, 1.0);
		double z = RandomUtils.RandomUniformReal(-1.0, 1.0);
		double w = RandomUtils.RandomUniformReal(-1.0, 1.0);
		return new Vector4d(x, y, z, w);
	}

	/// <summary>
	/// Random dynamic-size vector with each entry uniformly distributed in [-1, 1).
	/// Port of colmap::RandomEigenVectorXd.
	/// </summary>
	public static VectorXd RandomEigenVectorXd(int size)
	{
		var vector = new VectorXd(size);
		Span<double> values = vector.AsSpan();
		for (int i = 0; i < values.Length; ++i)
		{
			values[i] = RandomUtils.RandomUniformReal(-1.0, 1.0);
		}

		return vector;
	}

	/// <summary>
	/// Random single-precision vector with each entry uniformly distributed in [-1, 1).
	/// Port of colmap::RandomEigenVectorf&lt;N&gt; and RandomEigenVectorXf.
	/// </summary>
	public static float[] RandomEigenVectorf(int size)
	{
		var vector = new float[size];
		for (int i = 0; i < vector.Length; ++i)
		{
			vector[i] = RandomUtils.RandomUniformReal(-1.0f, 1.0f);
		}

		return vector;
	}

	/// <summary>
	/// Random single-precision rows x cols matrix with each entry uniformly distributed in
	/// [-1, 1), filled in column-major order (Eigen's linear index).
	/// Port of colmap::RandomEigenMatrixXf and RandomEigenMatrixf&lt;Rows, Cols&gt;.
	/// </summary>
	public static float[,] RandomEigenMatrixXf(int rows, int cols)
	{
		var matrix = new float[rows, cols];
		for (int col = 0; col < cols; ++col)
		{
			for (int row = 0; row < rows; ++row)
			{
				matrix[row, col] = RandomUtils.RandomUniformReal(-1.0f, 1.0f);
			}
		}

		return matrix;
	}

	/// <summary>
	/// Random 3x3 matrix with each entry uniformly distributed in [-1, 1), filled in
	/// column-major order. Port of colmap::RandomEigenMatrixd&lt;3, 3&gt;.
	/// </summary>
	public static Matrix3d RandomEigenMatrix3d()
	{
		Span<double> values = stackalloc double[9];
		for (int i = 0; i < values.Length; ++i)
		{
			values[i] = RandomUtils.RandomUniformReal(-1.0, 1.0);
		}

		return Matrix3d.FromColumnMajor(values);
	}

	/// <summary>
	/// Random 3x4 matrix with each entry uniformly distributed in [-1, 1), filled in
	/// column-major order. Port of colmap::RandomEigenMatrixd&lt;3, 4&gt;.
	/// </summary>
	public static Matrix3x4d RandomEigenMatrix3x4d()
	{
		Span<double> values = stackalloc double[12];
		for (int i = 0; i < values.Length; ++i)
		{
			values[i] = RandomUtils.RandomUniformReal(-1.0, 1.0);
		}

		return Matrix3x4d.FromColumnMajor(values);
	}

	/// <summary>
	/// Random 6x6 matrix with each entry uniformly distributed in [-1, 1), filled in
	/// column-major order. Port of colmap::RandomEigenMatrixd&lt;6, 6&gt;.
	/// </summary>
	public static Matrix6d RandomEigenMatrix6d()
	{
		Span<double> values = stackalloc double[36];
		for (int i = 0; i < values.Length; ++i)
		{
			values[i] = RandomUtils.RandomUniformReal(-1.0, 1.0);
		}

		return Matrix6d.FromColumnMajor(values);
	}

	/// <summary>
	/// Random rows x cols matrix with each entry uniformly distributed in [-1, 1), filled
	/// in column-major order. Port of colmap::RandomEigenMatrixXd; also stands in for
	/// RandomEigenMatrixd&lt;Rows, Cols&gt; with shapes that have no fixed-size type here
	/// (the draw order is the same linear index either way).
	/// </summary>
	public static MatrixXd RandomEigenMatrixXd(int rows, int cols)
	{
		var matrix = new MatrixXd(rows, cols);
		Span<double> values = matrix.AsSpan();
		for (int i = 0; i < values.Length; ++i)
		{
			values[i] = RandomUtils.RandomUniformReal(-1.0, 1.0);
		}

		return matrix;
	}

	/// <summary>
	/// Uniformly distributed random unit quaternion, matching
	/// <c>Eigen::Quaterniond::UnitRandom()</c> (Shoemake's method).
	/// Port of colmap::RandomEigenQuaterniond.
	/// </summary>
	public static Quaterniond RandomEigenQuaterniond()
	{
		double u1 = RandomUtils.RandomUniformReal(0.0, 1.0);
		double u2 = RandomUtils.RandomUniformReal(0.0, 2.0 * Math.PI);
		double u3 = RandomUtils.RandomUniformReal(0.0, 2.0 * Math.PI);
		double a = Math.Sqrt(1.0 - u1);
		double b = Math.Sqrt(u1);
		return new Quaterniond(a * Math.Sin(u2), a * Math.Cos(u2), b * Math.Sin(u3), b * Math.Cos(u3));
	}
}
