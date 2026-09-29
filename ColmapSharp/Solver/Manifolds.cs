// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 (BSD-3-Clause, see THIRD_PARTY_NOTICES.md):
// include/ceres/manifold.h, internal/ceres/manifold.cc, include/ceres/sphere_manifold.h,
// include/ceres/internal/sphere_manifold_functions.h,
// include/ceres/internal/householder_vector.h and include/ceres/product_manifold.h.
//
// Manifolds let the solver optimize a parameter block that lives on a curved space (a unit
// quaternion, a unit direction) or has frozen coordinates, through a tangent-space step:
// x' = Plus(x, delta) with delta in the tangent space. COLMAP uses exactly the ones here,
// through colmap/estimators/cost_functions/manifold.h: EuclideanManifold<3> for positions,
// EigenQuaternionManifold for rotations (stored x, y, z, w), SubsetManifold for constant
// camera parameters, SphereManifold<3> for a unit baseline, and ProductManifold to glue a
// quaternion to a translation or a sphere. QuaternionManifold (Ceres' w, x, y, z order)
// shares the code and is kept for Rotation.cs's order.
// Jacobians are row-major, as Ceres': PlusJacobian is AmbientSize x TangentSize,
// MinusJacobian is TangentSize x AmbientSize. Vector norms sum squares left to right; Eigen
// may reassociate a fixed-size norm under vectorization, so these are Tier C (outcome)
// like the solver that consumes them (divergence 115).

using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>ceres::Manifold: a parameter block's space and its tangent-space step.</summary>
public abstract class Manifold
{
	/// <summary>Dimension of the ambient space the parameter block is stored in.</summary>
	public abstract int AmbientSize { get; }

	/// <summary>Dimension of the tangent space the solver steps in.</summary>
	public abstract int TangentSize { get; }

	/// <summary>x_plus_delta = Plus(x, delta); false if the step is invalid.</summary>
	public abstract bool Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta);

	/// <summary>The row-major AmbientSize x TangentSize Jacobian of Plus(x, delta) at delta = 0.</summary>
	public abstract bool PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian);

	/// <summary>y_minus_x = Minus(y, x), the inverse of Plus: Plus(x, Minus(y, x)) = y.</summary>
	public abstract bool Minus(ReadOnlySpan<double> y, ReadOnlySpan<double> x, Span<double> yMinusX);

	/// <summary>The row-major TangentSize x AmbientSize Jacobian of Minus(y, x) at y = x.</summary>
	public abstract bool MinusJacobian(ReadOnlySpan<double> x, Span<double> jacobian);

	/// <summary>
	/// tangent_matrix = ambient_matrix * PlusJacobian(x), both row-major with
	/// <paramref name="numRows"/> rows: how the solver turns a cost function's ambient
	/// Jacobian into a tangent one.
	/// </summary>
	public virtual bool RightMultiplyByPlusJacobian(
		ReadOnlySpan<double> x, int numRows, ReadOnlySpan<double> ambientMatrix, Span<double> tangentMatrix)
	{
		int tangentSize = TangentSize;
		if (tangentSize == 0)
		{
			return true;
		}

		int ambientSize = AmbientSize;
		// Stack for every manifold COLMAP builds (the largest, quaternion x sphere, is 7 x 5);
		// the solver calls this once per parameter block per iteration, so no heap garbage.
		const int MaxStackDoubles = 256;
		int count = ambientSize * tangentSize;
		Span<double> plusJacobian = count <= MaxStackDoubles ? stackalloc double[count] : new double[count];
		if (!PlusJacobian(x, plusJacobian))
		{
			return false;
		}

		for (int r = 0; r < numRows; r++)
		{
			for (int c = 0; c < tangentSize; c++)
			{
				double sum = 0.0;
				for (int k = 0; k < ambientSize; k++)
				{
					sum += ambientMatrix[r * ambientSize + k] * plusJacobian[k * tangentSize + c];
				}

				tangentMatrix[r * tangentSize + c] = sum;
			}
		}

		return true;
	}
}

/// <summary>ceres::EuclideanManifold: R^n with Plus = +, Minus = -.</summary>
public sealed class EuclideanManifold : Manifold
{
	private readonly int size;

	/// <summary>Creates R^size.</summary>
	public EuclideanManifold(int size)
	{
		Check.Ge(size, 0);
		this.size = size;
	}

	/// <inheritdoc/>
	public override int AmbientSize => size;

	/// <inheritdoc/>
	public override int TangentSize => size;

	/// <inheritdoc/>
	public override bool Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta)
	{
		for (int i = 0; i < size; i++)
		{
			xPlusDelta[i] = x[i] + delta[i];
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		SetIdentity(jacobian[..(size * size)], size);
		return true;
	}

	/// <inheritdoc/>
	public override bool Minus(ReadOnlySpan<double> y, ReadOnlySpan<double> x, Span<double> yMinusX)
	{
		for (int i = 0; i < size; i++)
		{
			yMinusX[i] = y[i] - x[i];
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool MinusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		SetIdentity(jacobian[..(size * size)], size);
		return true;
	}

	/// <inheritdoc/>
	public override bool RightMultiplyByPlusJacobian(
		ReadOnlySpan<double> x, int numRows, ReadOnlySpan<double> ambientMatrix, Span<double> tangentMatrix)
	{
		ambientMatrix[..(numRows * size)].CopyTo(tangentMatrix);
		return true;
	}

	private static void SetIdentity(Span<double> matrix, int n)
	{
		matrix.Clear();
		for (int i = 0; i < n; i++)
		{
			matrix[i * n + i] = 1.0;
		}
	}
}

/// <summary>
/// ceres::SubsetManifold: R^n with some coordinates held constant; the tangent space is
/// the remaining coordinates, in order.
/// </summary>
public sealed class SubsetManifold : Manifold
{
	private readonly int tangentSize;
	private readonly bool[] constancyMask;

	/// <summary>
	/// Creates the manifold of size <paramref name="size"/> whose
	/// <paramref name="constantParameters"/> (distinct indices in [0, size)) never move.
	/// </summary>
	public SubsetManifold(int size, IReadOnlyList<int> constantParameters)
	{
		tangentSize = size - constantParameters.Count;
		constancyMask = new bool[size];
		if (constantParameters.Count == 0)
		{
			return;
		}

		var constant = constantParameters.ToArray();
		Array.Sort(constant);
		Check.That(constant[0] >= 0, "Indices indicating constant parameter must be greater than equal to zero.");
		Check.That(constant[^1] < size, "Indices indicating constant parameter must be less than the size of the parameter block.");
		for (int i = 1; i < constant.Length; i++)
		{
			Check.That(constant[i] != constant[i - 1], "The set of constant parameters cannot contain duplicates");
		}

		foreach (int index in constantParameters)
		{
			constancyMask[index] = true;
		}
	}

	/// <inheritdoc/>
	public override int AmbientSize => constancyMask.Length;

	/// <inheritdoc/>
	public override int TangentSize => tangentSize;

	/// <inheritdoc/>
	public override bool Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta)
	{
		for (int i = 0, j = 0; i < constancyMask.Length; i++)
		{
			xPlusDelta[i] = constancyMask[i] ? x[i] : x[i] + delta[j++];
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		if (tangentSize == 0)
		{
			return true;
		}

		jacobian[..(constancyMask.Length * tangentSize)].Clear();
		for (int r = 0, c = 0; r < constancyMask.Length; r++)
		{
			if (!constancyMask[r])
			{
				jacobian[r * tangentSize + c++] = 1.0;
			}
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool RightMultiplyByPlusJacobian(
		ReadOnlySpan<double> x, int numRows, ReadOnlySpan<double> ambientMatrix, Span<double> tangentMatrix)
	{
		if (tangentSize == 0)
		{
			return true;
		}

		int ambientSize = constancyMask.Length;
		for (int r = 0; r < numRows; r++)
		{
			for (int idx = 0, c = 0; idx < ambientSize; idx++)
			{
				if (!constancyMask[idx])
				{
					tangentMatrix[r * tangentSize + c++] = ambientMatrix[r * ambientSize + idx];
				}
			}
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool Minus(ReadOnlySpan<double> y, ReadOnlySpan<double> x, Span<double> yMinusX)
	{
		if (tangentSize == 0)
		{
			return true;
		}

		for (int i = 0, j = 0; i < constancyMask.Length; i++)
		{
			if (!constancyMask[i])
			{
				yMinusX[j++] = y[i] - x[i];
			}
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool MinusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		int ambientSize = constancyMask.Length;
		jacobian[..(tangentSize * ambientSize)].Clear();
		for (int c = 0, r = 0; c < ambientSize; c++)
		{
			if (!constancyMask[c])
			{
				jacobian[r++ * ambientSize + c] = 1.0;
			}
		}

		return true;
	}
}

/// <summary>
/// ceres::QuaternionManifold and ceres::EigenQuaternionManifold: unit quaternions with the
/// left-multiplicative step Plus(x, delta) = [cos|delta|, sin|delta| delta/|delta|] * x.
/// The two differ only in storage order.
/// </summary>
public sealed class QuaternionManifold : Manifold
{
	private readonly int kW;
	private readonly int kX;
	private readonly int kY;
	private readonly int kZ;

	private QuaternionManifold(int w, int x, int y, int z)
	{
		kW = w;
		kX = x;
		kY = y;
		kZ = z;
	}

	/// <summary>ceres::EigenQuaternionManifold: stored [x, y, z, w], as COLMAP's poses are.</summary>
	public static QuaternionManifold EigenOrder() => new(3, 0, 1, 2);

	/// <summary>ceres::QuaternionManifold: stored [w, x, y, z], as ceres/rotation.h expects.</summary>
	public static QuaternionManifold CeresOrder() => new(0, 1, 2, 3);

	/// <inheritdoc/>
	public override int AmbientSize => 4;

	/// <inheritdoc/>
	public override int TangentSize => 3;

	/// <inheritdoc/>
	public override bool Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta)
	{
		// x_plus_delta = QuaternionProduct(q_delta, x), where q_delta is the quaternion
		// constructed from delta.
		double normDelta = ScalarMath.Hypot(delta[0], delta[1], delta[2]);
		if (normDelta == 0.0)
		{
			// No change in rotation: return the quaternion as is.
			x[..4].CopyTo(xPlusDelta);
			return true;
		}

		double sinDeltaByDelta = Math.Sin(normDelta) / normDelta;
		double qW = Math.Cos(normDelta);
		double qX = sinDeltaByDelta * delta[0];
		double qY = sinDeltaByDelta * delta[1];
		double qZ = sinDeltaByDelta * delta[2];
		double xW = x[kW];
		double xX = x[kX];
		double xY = x[kY];
		double xZ = x[kZ];

		xPlusDelta[kW] = qW * xW - qX * xX - qY * xY - qZ * xZ;
		xPlusDelta[kX] = qW * xX + qX * xW + qY * xZ - qZ * xY;
		xPlusDelta[kY] = qW * xY - qX * xZ + qY * xW + qZ * xX;
		xPlusDelta[kZ] = qW * xZ + qX * xY - qY * xX + qZ * xW;
		return true;
	}

	/// <inheritdoc/>
	public override bool PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		// Row-major 4 x 3.
		jacobian[kW * 3 + 0] = -x[kX];
		jacobian[kW * 3 + 1] = -x[kY];
		jacobian[kW * 3 + 2] = -x[kZ];
		jacobian[kX * 3 + 0] = x[kW];
		jacobian[kX * 3 + 1] = x[kZ];
		jacobian[kX * 3 + 2] = -x[kY];
		jacobian[kY * 3 + 0] = -x[kZ];
		jacobian[kY * 3 + 1] = x[kW];
		jacobian[kY * 3 + 2] = x[kX];
		jacobian[kZ * 3 + 0] = x[kY];
		jacobian[kZ * 3 + 1] = -x[kX];
		jacobian[kZ * 3 + 2] = x[kW];
		return true;
	}

	/// <inheritdoc/>
	public override bool Minus(ReadOnlySpan<double> y, ReadOnlySpan<double> x, Span<double> yMinusX)
	{
		// ambient_y_minus_x = QuaternionProduct(y, -x) where -x is the conjugate of x.
		double yW = y[kW], yX = y[kX], yY = y[kY], yZ = y[kZ];
		double xW = x[kW], xX = x[kX], xY = x[kY], xZ = x[kZ];
		double aW = yW * xW + yX * xX + yY * xY + yZ * xZ;
		double aX = -yW * xX + yX * xW - yY * xZ + yZ * xY;
		double aY = -yW * xY + yX * xZ + yY * xW - yZ * xX;
		double aZ = -yW * xZ - yX * xY + yY * xX + yZ * xW;

		double uNorm = ScalarMath.Hypot(aX, aY, aZ);
		if (uNorm != 0.0)
		{
			double theta = Math.Atan2(uNorm, aW);
			yMinusX[0] = theta * aX / uNorm;
			yMinusX[1] = theta * aY / uNorm;
			yMinusX[2] = theta * aZ / uNorm;
		}
		else
		{
			yMinusX[..3].Clear();
		}

		return true;
	}

	/// <inheritdoc/>
	public override bool MinusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		// Row-major 3 x 4.
		jacobian[0 * 4 + kW] = -x[kX];
		jacobian[0 * 4 + kX] = x[kW];
		jacobian[0 * 4 + kY] = -x[kZ];
		jacobian[0 * 4 + kZ] = x[kY];
		jacobian[1 * 4 + kW] = -x[kY];
		jacobian[1 * 4 + kX] = x[kZ];
		jacobian[1 * 4 + kY] = x[kW];
		jacobian[1 * 4 + kZ] = -x[kX];
		jacobian[2 * 4 + kW] = -x[kZ];
		jacobian[2 * 4 + kX] = -x[kY];
		jacobian[2 * 4 + kY] = x[kX];
		jacobian[2 * 4 + kZ] = x[kW];
		return true;
	}
}
