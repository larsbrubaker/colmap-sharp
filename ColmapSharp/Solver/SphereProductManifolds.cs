// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 (BSD-3-Clause, see THIRD_PARTY_NOTICES.md):
// include/ceres/sphere_manifold.h, include/ceres/internal/sphere_manifold_functions.h,
// include/ceres/internal/householder_vector.h and include/ceres/product_manifold.h.
//
// The two composite manifolds, split from Manifolds.cs (the base class and the simple ones)
// by responsibility. SphereManifold steps on the sphere of radius |x| through the
// Householder reflection that maps x to the last axis, so the tangent space is the first
// n - 1 coordinates there (Hartley and Zisserman's "homogeneous vector" update); COLMAP uses
// it with n = 3 for a unit-length relative translation. ProductManifold concatenates
// manifolds block by block.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>
/// ceres::SphereManifold: the vectors of R^n with the norm of x, n &gt; 1, stepped in an
/// (n - 1)-dimensional tangent space.
/// </summary>
public sealed class SphereManifold : Manifold
{
	private readonly int size;

	/// <summary>Creates the sphere in R^size.</summary>
	public SphereManifold(int size)
	{
		Check.That(size > 1, "The size of the manifold needs to be greater than 1.");
		this.size = size;
	}

	/// <inheritdoc/>
	public override int AmbientSize => size;

	/// <inheritdoc/>
	public override int TangentSize => size - 1;

	/// <inheritdoc/>
	public override bool Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta)
	{
		double normDelta = Norm(delta[..(size - 1)]);
		if (normDelta == 0.0)
		{
			x[..size].CopyTo(xPlusDelta);
			return true;
		}

		Span<double> v = stackalloc double[size];
		double beta = ComputeHouseholderVector(x[..size], v);

		// y = [sin(|delta|) / |delta| * delta, cos(|delta|)], reflected back and scaled to |x|.
		double sinDeltaByDelta = Math.Sin(normDelta) / normDelta;
		Span<double> y = stackalloc double[size];
		for (int i = 0; i < size - 1; i++)
		{
			y[i] = sinDeltaByDelta * delta[i];
		}

		y[size - 1] = Math.Cos(normDelta);
		ApplyHouseholderVector(y, v, beta);
		double xNorm = Norm(x[..size]);
		for (int i = 0; i < size; i++)
		{
			xPlusDelta[i] = xNorm * y[i];
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		// Row-major size x (size - 1): column i is -beta v_i v + e_i, all scaled by |x|.
		int tangentSize = size - 1;
		Span<double> v = stackalloc double[size];
		double beta = ComputeHouseholderVector(x[..size], v);
		for (int i = 0; i < tangentSize; i++)
		{
			double scale = -beta * v[i];
			for (int r = 0; r < size; r++)
			{
				jacobian[r * tangentSize + i] = scale * v[r];
			}

			jacobian[i * tangentSize + i] += 1.0;
		}

		double xNorm = Norm(x[..size]);
		for (int k = 0; k < size * tangentSize; k++)
		{
			jacobian[k] *= xNorm;
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool Minus(ReadOnlySpan<double> y, ReadOnlySpan<double> x, Span<double> yMinusX)
	{
		int tangentSize = size - 1;
		Span<double> v = stackalloc double[size];
		double beta = ComputeHouseholderVector(x[..size], v);

		Span<double> hy = stackalloc double[size];
		y[..size].CopyTo(hy);
		ApplyHouseholderVector(hy, v, beta);
		double xNorm = Norm(x[..size]);
		for (int i = 0; i < size; i++)
		{
			hy[i] /= xNorm;
		}

		double yLast = hy[tangentSize];
		double hyNorm = Norm(hy[..tangentSize]);
		if (hyNorm == 0.0)
		{
			yMinusX[..tangentSize].Clear();
			yMinusX[tangentSize - 1] = yLast >= 0 ? 0.0 : Math.PI;
		}
		else
		{
			double scale = Math.Atan2(hyNorm, yLast) / hyNorm;
			for (int i = 0; i < tangentSize; i++)
			{
				yMinusX[i] = scale * hy[i];
			}
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool MinusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		// Row-major (size - 1) x size: row i is -beta v_i v^T + e_i^T, all divided by |x|.
		int tangentSize = size - 1;
		Span<double> v = stackalloc double[size];
		double beta = ComputeHouseholderVector(x[..size], v);
		for (int i = 0; i < tangentSize; i++)
		{
			double scale = -beta * v[i];
			for (int c = 0; c < size; c++)
			{
				jacobian[i * size + c] = scale * v[c];
			}

			jacobian[i * size + i] += 1.0;
		}

		double xNorm = Norm(x[..size]);
		for (int k = 0; k < size * tangentSize; k++)
		{
			jacobian[k] /= xNorm;
		}

		return true;
	}

	private static double Norm(ReadOnlySpan<double> values)
	{
		double sum = 0.0;
		foreach (double value in values)
		{
			sum += value * value;
		}

		return Math.Sqrt(sum);
	}

	/// <summary>
	/// ceres::internal::ComputeHouseholderVector (Golub and Van Loan, Algorithm 5.1.1, with
	/// the pivot at the last coordinate): v and beta with (I - beta v v^T) x = |x| e_n and
	/// v_n = 1. Returns beta.
	/// </summary>
	private static double ComputeHouseholderVector(ReadOnlySpan<double> x, Span<double> v)
	{
		int n = x.Length;
		double sigma = 0.0;
		for (int i = 0; i < n - 1; i++)
		{
			sigma += x[i] * x[i];
		}

		x.CopyTo(v);
		v[n - 1] = 1.0;

		double xPivot = x[n - 1];
		if (sigma <= LinearAlgebraConstants.MachineEpsilon)
		{
			return xPivot < 0.0 ? 2.0 : 0.0;
		}

		double mu = Math.Sqrt(xPivot * xPivot + sigma);
		double vPivot = xPivot <= 0.0 ? xPivot - mu : -sigma / (xPivot + mu);
		double beta = 2.0 * vPivot * vPivot / (sigma + vPivot * vPivot);
		for (int i = 0; i < n - 1; i++)
		{
			v[i] /= vPivot;
		}

		return beta;
	}

	// y <- y - v (beta (v^T y)), ceres::internal::ApplyHouseholderVector in place.
	private static void ApplyHouseholderVector(Span<double> y, ReadOnlySpan<double> v, double beta)
	{
		double dot = 0.0;
		for (int i = 0; i < y.Length; i++)
		{
			dot += v[i] * y[i];
		}

		double scale = beta * dot;
		for (int i = 0; i < y.Length; i++)
		{
			y[i] -= v[i] * scale;
		}
	}
}

/// <summary>
/// ceres::ProductManifold: the Cartesian product of manifolds, laid out block after block
/// in both the ambient and the tangent space, with block-diagonal Jacobians.
/// </summary>
public sealed class ProductManifold : Manifold
{
	private readonly Manifold[] manifolds;
	private readonly int[] ambientOffsets;
	private readonly int[] tangentOffsets;
	private readonly int ambientSize;
	private readonly int tangentSize;

	// The largest component Jacobian, Ceres' buffer_size_.
	private readonly int bufferSize;

	/// <summary>Creates the product of at least two manifolds.</summary>
	public ProductManifold(params Manifold[] manifolds)
	{
		Check.Ge(manifolds.Length, 2);
		this.manifolds = manifolds;
		ambientOffsets = new int[manifolds.Length];
		tangentOffsets = new int[manifolds.Length];
		for (int i = 0; i < manifolds.Length; i++)
		{
			ambientOffsets[i] = ambientSize;
			tangentOffsets[i] = tangentSize;
			ambientSize += manifolds[i].AmbientSize;
			tangentSize += manifolds[i].TangentSize;
			bufferSize = Math.Max(bufferSize, manifolds[i].AmbientSize * manifolds[i].TangentSize);
		}
	}

	/// <inheritdoc/>
	public override int AmbientSize => ambientSize;

	/// <inheritdoc/>
	public override int TangentSize => tangentSize;

	/// <inheritdoc/>
	public override bool Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta)
	{
		for (int i = 0; i < manifolds.Length; i++)
		{
			if (!manifolds[i].Plus(x[ambientOffsets[i]..], delta[tangentOffsets[i]..], xPlusDelta[ambientOffsets[i]..]))
			{
				return false;
			}
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		jacobian[..(ambientSize * tangentSize)].Clear();
		Span<double> buffer = stackalloc double[bufferSize];
		for (int i = 0; i < manifolds.Length; i++)
		{
			int rows = manifolds[i].AmbientSize;
			int cols = manifolds[i].TangentSize;
			if (!manifolds[i].PlusJacobian(x[ambientOffsets[i]..], buffer))
			{
				return false;
			}

			CopyBlock(buffer, rows, cols, jacobian, tangentSize, ambientOffsets[i], tangentOffsets[i]);
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool Minus(ReadOnlySpan<double> y, ReadOnlySpan<double> x, Span<double> yMinusX)
	{
		for (int i = 0; i < manifolds.Length; i++)
		{
			if (!manifolds[i].Minus(y[ambientOffsets[i]..], x[ambientOffsets[i]..], yMinusX[tangentOffsets[i]..]))
			{
				return false;
			}
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool MinusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		jacobian[..(tangentSize * ambientSize)].Clear();
		Span<double> buffer = stackalloc double[bufferSize];
		for (int i = 0; i < manifolds.Length; i++)
		{
			int rows = manifolds[i].TangentSize;
			int cols = manifolds[i].AmbientSize;
			if (!manifolds[i].MinusJacobian(x[ambientOffsets[i]..], buffer))
			{
				return false;
			}

			CopyBlock(buffer, rows, cols, jacobian, ambientSize, tangentOffsets[i], ambientOffsets[i]);
		}

		return true;
	}

	private static void CopyBlock(
		ReadOnlySpan<double> block, int rows, int cols, Span<double> matrix, int matrixCols, int row0, int col0)
	{
		for (int r = 0; r < rows; r++)
		{
			block.Slice(r * cols, cols).CopyTo(matrix.Slice((row0 + r) * matrixCols + col0, cols));
		}
	}
}
