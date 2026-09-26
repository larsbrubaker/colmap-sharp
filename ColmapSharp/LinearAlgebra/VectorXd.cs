// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VectorXd: heap-backed, dynamically sized column vector of doubles, the replacement for
// Eigen::VectorXd (and Eigen::Matrix<double, Dynamic, 1>). Written here to Eigen's
// documented semantics; Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md). Sibling of
// MatrixXd, which documents the storage and mutability choice for both; the fixed-size
// Vector2d/3d/4d convert to and from it. Tests: ColmapSharp.Tests/LinearAlgebra/
// DynamicMatrixTests.cs.
//
// Arithmetic order follows the folder contract in Vector3d.cs: reductions are plain
// left-to-right sums seeded with the first term, no FMA.

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Dynamically sized column vector of doubles. Replacement for Eigen::VectorXd. A mutable
/// reference type, like MatrixXd: copies are explicit (<see cref="Clone"/>).
/// </summary>
public sealed class VectorXd
{
	private readonly double[] _data;

	/// <summary>A zero vector of the given length, Eigen's VectorXd::Zero(n).</summary>
	public VectorXd(int length)
	{
		ArgumentOutOfRangeException.ThrowIfNegative(length);
		_data = new double[length];
	}

	/// <summary>A vector holding a copy of the given values.</summary>
	public VectorXd(ReadOnlySpan<double> values)
	{
		_data = values.ToArray();
	}

	/// <summary>Number of coefficients, Eigen's size().</summary>
	public int Length => _data.Length;

	/// <summary>Coefficient i, readable and writable.</summary>
	public double this[int i]
	{
		get => _data[i];
		set => _data[i] = value;
	}

	/// <summary>The coefficients as a writable span, the analogue of Eigen's data().</summary>
	public Span<double> AsSpan() => _data;

	/// <summary>Eigen's VectorXd::Zero(n).</summary>
	public static VectorXd Zero(int length) => new(length);

	/// <summary>Eigen's VectorXd::Ones(n).</summary>
	public static VectorXd Ones(int length) => Constant(length, 1.0);

	/// <summary>Eigen's VectorXd::Constant(n, value).</summary>
	public static VectorXd Constant(int length, double value)
	{
		var v = new VectorXd(length);
		v._data.AsSpan().Fill(value);
		return v;
	}

	/// <summary>Eigen's VectorXd::Unit(n, i): zero except a 1 at index i.</summary>
	public static VectorXd Unit(int length, int i)
	{
		var v = new VectorXd(length);
		v[i] = 1;
		return v;
	}

	/// <summary>A deep copy.</summary>
	public VectorXd Clone() => new(_data);

	/// <summary>Converts a Vector2d.</summary>
	public static VectorXd From(Vector2d v) => new([v.X, v.Y]);

	/// <summary>Converts a Vector3d.</summary>
	public static VectorXd From(Vector3d v) => new([v.X, v.Y, v.Z]);

	/// <summary>Converts a Vector4d.</summary>
	public static VectorXd From(Vector4d v) => new([v.X, v.Y, v.Z, v.W]);

	/// <summary>Converts to a Vector2d; the length must be 2.</summary>
	public Vector2d ToVector2d()
	{
		RequireLength(2);
		return new Vector2d(_data[0], _data[1]);
	}

	/// <summary>Converts to a Vector3d; the length must be 3.</summary>
	public Vector3d ToVector3d()
	{
		RequireLength(3);
		return new Vector3d(_data[0], _data[1], _data[2]);
	}

	/// <summary>Converts to a Vector4d; the length must be 4.</summary>
	public Vector4d ToVector4d()
	{
		RequireLength(4);
		return new Vector4d(_data[0], _data[1], _data[2], _data[3]);
	}

	/// <summary>A copy of coefficients [start, start + length), Eigen's segment(start, length).</summary>
	public VectorXd Segment(int start, int length) => new(_data.AsSpan(start, length));

	/// <summary>A copy of the first n coefficients, Eigen's head(n).</summary>
	public VectorXd Head(int n) => Segment(0, n);

	/// <summary>A copy of the last n coefficients, Eigen's tail(n).</summary>
	public VectorXd Tail(int n) => Segment(Length - n, n);

	/// <summary>Squared Euclidean norm, a left-to-right sum of squares.</summary>
	public double SquaredNorm() => Dot(_data, _data);

	/// <summary>Euclidean norm, sqrt(SquaredNorm()).</summary>
	public double Norm() => Math.Sqrt(SquaredNorm());

	/// <summary>Dot product with another vector of the same length.</summary>
	public double Dot(VectorXd other)
	{
		RequireLength(other.Length);
		return Dot(_data, other._data);
	}

	/// <summary>
	/// This vector divided by its norm. Like Eigen's normalized(), a zero vector stays zero.
	/// </summary>
	public VectorXd Normalized()
	{
		double norm = Norm();
		var result = Clone();
		if (norm > 0)
		{
			for (int i = 0; i < result.Length; i++)
			{
				result._data[i] /= norm;
			}
		}

		return result;
	}

	/// <summary>Largest absolute coefficient, Eigen's lpNorm&lt;Infinity&gt;() (0 when empty).</summary>
	public double MaxAbs()
	{
		double max = 0;
		foreach (double v in _data)
		{
			max = Math.Max(max, Math.Abs(v));
		}

		return max;
	}

	/// <summary>Eigen's isApprox: ||a - b|| &lt;= precision * min(||a||, ||b||).</summary>
	public bool IsApprox(VectorXd other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		return Length == other.Length && LinearAlgebraConstants.IsApprox(_data, other._data, precision);
	}

	/// <summary>Coefficient-wise sum.</summary>
	public static VectorXd operator +(VectorXd a, VectorXd b)
	{
		a.RequireLength(b.Length);
		var r = new VectorXd(a.Length);
		for (int i = 0; i < r.Length; i++)
		{
			r._data[i] = a._data[i] + b._data[i];
		}

		return r;
	}

	/// <summary>Coefficient-wise difference.</summary>
	public static VectorXd operator -(VectorXd a, VectorXd b)
	{
		a.RequireLength(b.Length);
		var r = new VectorXd(a.Length);
		for (int i = 0; i < r.Length; i++)
		{
			r._data[i] = a._data[i] - b._data[i];
		}

		return r;
	}

	/// <summary>Negation.</summary>
	public static VectorXd operator -(VectorXd a) => a * -1.0;

	/// <summary>Scalar product.</summary>
	public static VectorXd operator *(VectorXd a, double s)
	{
		var r = new VectorXd(a.Length);
		for (int i = 0; i < r.Length; i++)
		{
			r._data[i] = a._data[i] * s;
		}

		return r;
	}

	/// <summary>Scalar product.</summary>
	public static VectorXd operator *(double s, VectorXd a) => a * s;

	/// <summary>Scalar division (divides each coefficient, like Eigen).</summary>
	public static VectorXd operator /(VectorXd a, double s)
	{
		var r = new VectorXd(a.Length);
		for (int i = 0; i < r.Length; i++)
		{
			r._data[i] = a._data[i] / s;
		}

		return r;
	}

	/// <inheritdoc/>
	public override string ToString()
	{
		return "[" + string.Join(", ", _data.Select(v => v.ToString("R", CultureInfo.InvariantCulture))) + "]";
	}

	/// <summary>Left-to-right dot product seeded with the first term (the folder contract).</summary>
	internal static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
	{
		if (a.Length == 0)
		{
			return 0;
		}

		double sum = a[0] * b[0];
		for (int i = 1; i < a.Length; i++)
		{
			sum += a[i] * b[i];
		}

		return sum;
	}

	private void RequireLength(int length)
	{
		if (Length != length)
		{
			throw new ArgumentException($"Vector length mismatch: {Length} vs {length}.");
		}
	}
}
