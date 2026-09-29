// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TinyManifold: colmap/estimators/cost_functions/tiny_manifold.h, the fixed-size manifold
// policies for Optim/TinySolver.cs, mirroring the Ceres manifolds used elsewhere
// (Solver/Manifolds.cs). Each is a stateless struct implementing ITinyManifold (declared in
// TinySolver.cs next to TinyEuclideanManifold, as tiny_solver.h declares EuclideanManifold).
// The solver evaluates them on doubles only; it never differentiates through a manifold.
//
// Translation notes:
// - Every C++ name gains a "Tiny" prefix (colmap::SphereManifold -> TinySphereManifold, ...)
//   so they do not collide with the Ceres manifolds of Solver/Manifolds.cs, which have the
//   same names in C++ but live in namespace ceres there.
// - C++'s variadic ProductManifold<Manifolds...> becomes TinyProductManifold<THead, TRest>
//   (the recursive case) and TinyProductManifold<TA, TB, TC> (three blocks, which delegates
//   to TinyProductManifold<TA, TinyProductManifold<TB, TC>>), the arities COLMAP uses. A
//   single-block product is just the manifold itself.
// - TinySphereManifold is C++'s SphereManifold<3>, the only size COLMAP implements.
// - TinySphereManifold's tangent basis uses our own unit-orthogonal vector (Hughes & Moller),
//   not Eigen's unitOrthogonal(); see divergence 19.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>
/// Port of colmap::EigenQuaternionManifold: SO(3) for an Eigen (x, y, z, w) quaternion.
/// Retraction q_plus = normalize(q * ExpMap(delta)) (right multiplication).
/// </summary>
public readonly struct TinyEigenQuaternionManifold : ITinyManifold
{
	/// <inheritdoc/>
	public int AmbientSize => 4;

	/// <inheritdoc/>
	public int TangentSize => 3;

	/// <inheritdoc/>
	public bool IsEuclidean => false;

	/// <inheritdoc/>
	public void Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta)
	{
		Span<double> deltaQ = stackalloc double[4];
		QuaternionUtils.EigenQuaternionFromAngleAxis(Real.Cast(delta[..3]), Real.CastWritable(deltaQ));
		var q = new Quaterniond(x[3], x[0], x[1], x[2]);
		var dq = new Quaterniond(deltaQ[3], deltaQ[0], deltaQ[1], deltaQ[2]);
		Quaterniond result = (q * dq).Normalized();
		xPlusDelta[0] = result.X;
		xPlusDelta[1] = result.Y;
		xPlusDelta[2] = result.Z;
		xPlusDelta[3] = result.W;
	}

	/// <summary>
	/// For the right-multiplication retraction q * ExpMap(delta), the Jacobian at delta = 0
	/// is 0.5 * L(q) restricted to the imaginary columns, where L(q) is the quaternion
	/// left-multiplication matrix. Row-major 4x3.
	/// </summary>
	public void PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		Matrix4d left = QuaternionUtils.QuaternionLeftMultMatrix(new Quaterniond(x[3], x[0], x[1], x[2]));
		for (int r = 0; r < 4; ++r)
		{
			for (int c = 0; c < 3; ++c)
			{
				jacobian[r * 3 + c] = 0.5 * left[r, c];
			}
		}
	}
}

/// <summary>
/// Port of colmap::SphereManifold&lt;3&gt;: the unit 2-sphere in R^3. Retraction
/// x_plus = normalize(x + B * delta), where B is an orthonormal basis of the tangent plane
/// at x / ||x||.
/// </summary>
public readonly struct TinySphereManifold : ITinyManifold
{
	/// <inheritdoc/>
	public int AmbientSize => 3;

	/// <inheritdoc/>
	public int TangentSize => 2;

	/// <inheritdoc/>
	public bool IsEuclidean => false;

	/// <inheritdoc/>
	public void Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta)
	{
		var xv = new Vector3d(x[0], x[1], x[2]);
		(Vector3d b1, Vector3d b2) = TangentBasis(xv);
		Vector3d result = new Vector3d(
			xv.X + (b1.X * delta[0] + b2.X * delta[1]),
			xv.Y + (b1.Y * delta[0] + b2.Y * delta[1]),
			xv.Z + (b1.Z * delta[0] + b2.Z * delta[1])).Normalized();
		xPlusDelta[0] = result.X;
		xPlusDelta[1] = result.Y;
		xPlusDelta[2] = result.Z;
	}

	/// <summary>
	/// The tangent basis B as a row-major 3x2 matrix. Assumes x is (approximately) unit
	/// norm, as maintained by the sphere retraction; B equals the Plus Jacobian only at
	/// unit x.
	/// </summary>
	public void PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		(Vector3d b1, Vector3d b2) = TangentBasis(new Vector3d(x[0], x[1], x[2]));
		for (int r = 0; r < 3; ++r)
		{
			jacobian[r * 2] = b1[r];
			jacobian[r * 2 + 1] = b2[r];
		}
	}

	// Orthonormal basis (the two columns of a 3 x 2 matrix) of the tangent plane at x / ||x||:
	// b1 a unit vector orthogonal to x_hat, b2 = x_hat x b1.
	private static (Vector3d B1, Vector3d B2) TangentBasis(Vector3d x)
	{
		Vector3d xHat = x.Normalized();
		Vector3d b1 = UnitOrthogonal(xHat);
		return (b1, xHat.Cross(b1));
	}

	// A unit vector orthogonal to v, by Hughes & Moller, "Building an Orthonormal Basis from
	// a Unit Vector", Journal of Graphics Tools 4(4), 1999: zero the component of smallest
	// magnitude (the first one on a tie), swap the other two and negate one. The other two
	// cannot both be zero for a nonzero v, so the result is well defined. Stands in for
	// Eigen's unitOrthogonal(), whose choice for 3-vectors Eigen does not document.
	private static Vector3d UnitOrthogonal(Vector3d v)
	{
		double ax = Math.Abs(v.X);
		double ay = Math.Abs(v.Y);
		double az = Math.Abs(v.Z);
		Vector3d perpendicular;
		if (ax <= ay && ax <= az)
		{
			perpendicular = new Vector3d(0, -v.Z, v.Y);
		}
		else if (ay <= az)
		{
			perpendicular = new Vector3d(-v.Z, 0, v.X);
		}
		else
		{
			perpendicular = new Vector3d(-v.Y, v.X, 0);
		}

		return perpendicular.Normalized();
	}
}

/// <summary>
/// Port of colmap::ProductManifold&lt;Head, Tail...&gt;: two manifolds on adjacent parameter
/// blocks, laid out in argument order, with a block-diagonal Plus Jacobian. Nest it (or use
/// the three-argument form) for longer products.
/// </summary>
public readonly struct TinyProductManifold<THead, TRest> : ITinyManifold
	where THead : struct, ITinyManifold
	where TRest : struct, ITinyManifold
{
	private readonly THead head;
	private readonly TRest rest;

	/// <summary>Combines two manifolds (needed when a block, like a sized TinyEuclideanManifold, has state; TinyEuclideanManifold1 needs none).</summary>
	public TinyProductManifold(THead head, TRest rest)
	{
		this.head = head;
		this.rest = rest;
	}

	/// <inheritdoc/>
	public int AmbientSize => head.AmbientSize + rest.AmbientSize;

	/// <inheritdoc/>
	public int TangentSize => head.TangentSize + rest.TangentSize;

	/// <inheritdoc/>
	public bool IsEuclidean => head.IsEuclidean && rest.IsEuclidean;

	/// <inheritdoc/>
	public void Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta)
	{
		int headAmbient = head.AmbientSize;
		int headTangent = head.TangentSize;
		head.Plus(x, delta, xPlusDelta);
		rest.Plus(x[headAmbient..], delta[headTangent..], xPlusDelta[headAmbient..]);
	}

	/// <summary>
	/// Row-major AmbientSize x TangentSize, block-diagonal: the head block top-left, the rest
	/// block bottom-right. (C++ stores single-column blocks column-major, the same layout.)
	/// </summary>
	public void PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian)
	{
		int ambient = AmbientSize;
		int tangent = TangentSize;
		int headAmbient = head.AmbientSize;
		int headTangent = head.TangentSize;
		int restAmbient = rest.AmbientSize;
		int restTangent = rest.TangentSize;
		jacobian[..(ambient * tangent)].Clear();

		Span<double> headJacobian = stackalloc double[headAmbient * headTangent];
		head.PlusJacobian(x, headJacobian);
		for (int r = 0; r < headAmbient; ++r)
		{
			for (int c = 0; c < headTangent; ++c)
			{
				jacobian[r * tangent + c] = headJacobian[r * headTangent + c];
			}
		}

		Span<double> restJacobian = stackalloc double[restAmbient * restTangent];
		rest.PlusJacobian(x[headAmbient..], restJacobian);
		for (int r = 0; r < restAmbient; ++r)
		{
			for (int c = 0; c < restTangent; ++c)
			{
				jacobian[(headAmbient + r) * tangent + headTangent + c] = restJacobian[r * restTangent + c];
			}
		}
	}
}

/// <summary>
/// Port of colmap::ProductManifold&lt;A, B, C&gt;: three manifolds on adjacent parameter
/// blocks, e.g. a relative pose plus a focal length. Equivalent to
/// TinyProductManifold&lt;TA, TinyProductManifold&lt;TB, TC&gt;&gt;, as C++'s recursion unrolls.
/// </summary>
public readonly struct TinyProductManifold<TA, TB, TC> : ITinyManifold
	where TA : struct, ITinyManifold
	where TB : struct, ITinyManifold
	where TC : struct, ITinyManifold
{
	private readonly TinyProductManifold<TA, TinyProductManifold<TB, TC>> product;

	/// <summary>Combines three manifolds (needed when a block, like a sized TinyEuclideanManifold, has state; TinyEuclideanManifold1 needs none).</summary>
	public TinyProductManifold(TA a, TB b, TC c)
	{
		product = new TinyProductManifold<TA, TinyProductManifold<TB, TC>>(a, new TinyProductManifold<TB, TC>(b, c));
	}

	/// <inheritdoc/>
	public int AmbientSize => product.AmbientSize;

	/// <inheritdoc/>
	public int TangentSize => product.TangentSize;

	/// <inheritdoc/>
	public bool IsEuclidean => product.IsEuclidean;

	/// <inheritdoc/>
	public void Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta) =>
		product.Plus(x, delta, xPlusDelta);

	/// <inheritdoc/>
	public void PlusJacobian(ReadOnlySpan<double> x, Span<double> jacobian) => product.PlusJacobian(x, jacobian);
}
