// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Quaterniond: rotation quaternion of doubles, the replacement for Eigen::Quaterniond, the
// rotation half of COLMAP's Rigid3d and Sim3d. Written here from the published algorithms
// cited below to Eigen's documented semantics; Eigen (MPL-2.0) is not ported
// (docs/LICENSE_AUDIT.md). Siblings: AngleAxisd (the other rotation representation) and
// Matrix3d (ToRotationMatrix / FromRotationMatrix). Tests:
// ColmapSharp.Tests/LinearAlgebra/QuaternionTests.cs.
//
// Conventions, as Eigen documents them:
// - The constructor takes (w, x, y, z); the coefficients are stored and exposed by Coeffs
//   as (x, y, z, w), Eigen's memory order, which is how COLMAP's Rigid3d packs its params.
//   COLMAP's file formats write qw qx qy qz; that ordering belongs to the I/O code.
// - Hamilton product, rotating a vector v as q v q*. The product composes rotations:
//   (a * b) * v == a * (b * v).
// - Norms and dot products run over Coeffs (x, y, z, w) with Vector4d's paired reduction.
//
// Algorithms:
// - Rotating a vector uses the well-known 15-multiply form of q v q* for a unit q:
//   t = 2 (u x v), v' = v + w t + u x t, with u = (x, y, z) (see e.g. F. Giesen, "Rotating
//   a vector by a unit quaternion", 2015). Like Eigen it assumes q is normalized.
// - ToRotationMatrix is the standard unit-quaternion rotation matrix (K. Shoemake,
//   "Animating rotation with quaternion curves", SIGGRAPH 1985).
// - FromRotationMatrix is Shoemake's branch algorithm ("Quaternion Calculus and Fast
//   Animation", SIGGRAPH 1987 course notes): if the trace is positive, w comes from the
//   trace and is positive; otherwise the largest diagonal element picks the component that
//   is computed first, and that component is positive. This is the sign convention Eigen
//   documents for its Quaternion(Matrix3) constructor.
// Arithmetic order: no FMA (the contract is in Vector3d.cs). Rigid3d/Sim3d algebra
// downstream is Tier A, so every operation here was checked bit for bit against Eigen
// through pycolmap (oracle/linear_algebra_rotations.py, RotationOracleTests):
// - Tier A, bit-identical: the product (with Eigen's SIMD pairing, see operator *), Norm
//   and Inverse (Vector4d's paired reduction), ToRotationMatrix, FromRotationMatrix
//   (Matrix3d.Trace's grouping), AngularDistance, and AngleAxisd.FromQuaternion's angle.
// - Tier B: q * v, which the macOS wheel computes with FMA-contracted cross products
//   (docs/CPP_DIVERGENCES.md, entry 6), and FromAngleAxis, whose sin can differ from the
//   wheel's by 1 ulp (docs/CPP_DIVERGENCES.md, entry 7).

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Rotation quaternion of doubles. Replacement for Eigen::Quaterniond.
/// </summary>
public readonly struct Quaterniond : IEquatable<Quaterniond>
{
	/// <summary>The x (i) coefficient.</summary>
	public readonly double X;

	/// <summary>The y (j) coefficient.</summary>
	public readonly double Y;

	/// <summary>The z (k) coefficient.</summary>
	public readonly double Z;

	/// <summary>The scalar (real) coefficient.</summary>
	public readonly double W;

	/// <summary>Creates the quaternion w + xi + yj + zk. Argument order (w, x, y, z) like Eigen's constructor.</summary>
	public Quaterniond(double w, double x, double y, double z)
	{
		W = w;
		X = x;
		Y = y;
		Z = z;
	}

	/// <summary>The identity rotation (1, 0, 0, 0).</summary>
	public static Quaterniond Identity => new(1, 0, 0, 0);

	/// <summary>Coefficients in Eigen's memory order (x, y, z, w), Eigen's coeffs().</summary>
	public Vector4d Coeffs => new(X, Y, Z, W);

	/// <summary>The vector part (x, y, z), Eigen's vec().</summary>
	public Vector3d Vec => new(X, Y, Z);

	/// <summary>Builds a quaternion from coefficients in Eigen's memory order (x, y, z, w).</summary>
	public static Quaterniond FromCoeffs(Vector4d xyzw) => new(xyzw.W, xyzw.X, xyzw.Y, xyzw.Z);

	/// <summary>Squared norm over the coefficients, x*x + y*y + z*z + w*w.</summary>
	public double SquaredNorm => Coeffs.SquaredNorm;

	/// <summary>Norm, sqrt(SquaredNorm).</summary>
	public double Norm => Coeffs.Norm;

	/// <summary>Dot product of the coefficients, in memory order.</summary>
	public double Dot(Quaterniond other) => Coeffs.Dot(other.Coeffs);

	/// <summary>The unit quaternion; a zero quaternion is returned unchanged, like Eigen.</summary>
	public Quaterniond Normalized() => FromCoeffs(Coeffs.Normalized());

	/// <summary>The conjugate (w, -x, -y, -z); the inverse rotation for a unit quaternion.</summary>
	public Quaterniond Conjugate() => new(W, -X, -Y, -Z);

	/// <summary>
	/// The multiplicative inverse, conjugate / squared norm. A zero quaternion has none and
	/// yields the zero quaternion. For unit quaternions Conjugate is cheaper and equal.
	/// </summary>
	public Quaterniond Inverse()
	{
		double squaredNorm = SquaredNorm;
		if (squaredNorm > 0)
		{
			return new Quaterniond(W / squaredNorm, -X / squaredNorm, -Y / squaredNorm, -Z / squaredNorm);
		}

		return default;
	}

	/// <summary>
	/// The angle in radians of the rotation taking other to this,
	/// 2 atan2(|d.vec|, |d.w|) with d = this * other.Conjugate(); in [0, pi].
	/// Eigen's angularDistance.
	/// </summary>
	public double AngularDistance(Quaterniond other)
	{
		Quaterniond d = this * other.Conjugate();
		return 2 * Math.Atan2(d.Vec.Norm, Math.Abs(d.W));
	}

	/// <summary>
	/// The 3x3 rotation matrix of this (unit) quaternion, in Shoemake's (1985) form
	/// R = I + 2 w [v]x + 2 [v]x^2, written out per element: diagonal 1 - 2(b^2 + c^2),
	/// off-diagonals 2(ab -/+ wc).
	/// </summary>
	public Matrix3d ToRotationMatrix()
	{
		// Grouping (the oracle pins it bit for bit): each off-diagonal is a sum or difference
		// of two doubled products, and each diagonal subtracts a sum of two doubled squares
		// from 1. The 2 must multiply a coordinate before the product is formed, so each
		// doubled product rounds once: 2 * (x * y) rounds x * y first, which differs in the
		// subnormal range (fixture case q = (3e-162, 4e-162, 0, 0.5): 2.5e-323 vs 2e-323).
		// Which factor carries the 2 does not matter; (2a) * b is round(2ab) either way.
		double x2 = 2 * X, y2 = 2 * Y, z2 = 2 * Z;
		double xx = x2 * X, yy = y2 * Y, zz = z2 * Z;
		double xy = x2 * Y, xz = x2 * Z, yz = y2 * Z;
		double wx = x2 * W, wy = y2 * W, wz = z2 * W;

		return new Matrix3d(
			1 - (yy + zz), xy - wz, xz + wy,
			xy + wz, 1 - (xx + zz), yz - wx,
			xz - wy, yz + wx, 1 - (xx + yy));
	}

	/// <summary>
	/// The quaternion of a rotation matrix by Shoemake's branch algorithm (see the file
	/// header for the sign convention). Eigen's Quaternion(const Matrix3&amp;). The input
	/// is not orthogonalized; COLMAP normalizes the result where it needs a unit quaternion.
	/// </summary>
	public static Quaterniond FromRotationMatrix(Matrix3d m)
	{
		// Shoemake: 4 w^2 = 1 + trace. The test is strictly positive because the oracle
		// requires it: on the exactly-zero-trace matrices in the fixture (trace_zero_cases,
		// oracle/linear_algebra_rotations.py) Eigen takes the diagonal branch, and >= 0
		// gives different bits (RotationOracleTests.TraceZeroMatricesTakeTheDiagonalBranch).
		double trace = m.Trace();
		if (trace > 0)
		{
			double root = Math.Sqrt(trace + 1.0); // 2|w|
			double scale = 0.5 / root;            // 1 / (4w)
			return new Quaterniond(
				root * 0.5,
				(m[2, 1] - m[1, 2]) * scale,
				(m[0, 2] - m[2, 0]) * scale,
				(m[1, 0] - m[0, 1]) * scale);
		}

		// Otherwise solve first for the component whose diagonal entry is largest (ties go
		// to the earlier axis), which keeps the square root well away from zero. For that
		// axis a with cyclic successors b, c: 4 a^2 = R_aa - R_bb - R_cc + 1, summed in
		// exactly that order (the oracle distinguishes the orders).
		if (m[0, 0] >= m[1, 1] && m[0, 0] >= m[2, 2])
		{
			double root = Math.Sqrt(m[0, 0] - m[1, 1] - m[2, 2] + 1.0); // 2|x|
			double scale = 0.5 / root;
			return new Quaterniond(
				(m[2, 1] - m[1, 2]) * scale,
				root * 0.5,
				(m[1, 0] + m[0, 1]) * scale,
				(m[2, 0] + m[0, 2]) * scale);
		}

		if (m[1, 1] >= m[2, 2])
		{
			double root = Math.Sqrt(m[1, 1] - m[2, 2] - m[0, 0] + 1.0); // 2|y|
			double scale = 0.5 / root;
			return new Quaterniond(
				(m[0, 2] - m[2, 0]) * scale,
				(m[0, 1] + m[1, 0]) * scale,
				root * 0.5,
				(m[2, 1] + m[1, 2]) * scale);
		}

		double zRoot = Math.Sqrt(m[2, 2] - m[0, 0] - m[1, 1] + 1.0); // 2|z|
		double zScale = 0.5 / zRoot;
		return new Quaterniond(
			(m[1, 0] - m[0, 1]) * zScale,
			(m[0, 2] + m[2, 0]) * zScale,
			(m[1, 2] + m[2, 1]) * zScale,
			zRoot * 0.5);
	}

	/// <summary>The quaternion of an angle-axis rotation: (cos(a/2), sin(a/2) axis).</summary>
	public static Quaterniond FromAngleAxis(AngleAxisd angleAxis)
	{
		double halfAngle = 0.5 * angleAxis.Angle;
		double sine = Math.Sin(halfAngle);
		Vector3d axis = angleAxis.Axis;
		return new Quaterniond(Math.Cos(halfAngle), sine * axis.X, sine * axis.Y, sine * axis.Z);
	}

	/// <summary>
	/// Eigen's isApprox on the coefficients:
	/// ||a - b|| &lt;= precision * min(||a||, ||b||). Note q and -q are the same rotation
	/// but are not approximately equal here, as in Eigen.
	/// </summary>
	public bool IsApprox(Quaterniond other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		return Coeffs.IsApprox(other.Coeffs, precision);
	}

	/// <summary>
	/// Hamilton product: the rotation b followed by a. Each coefficient is the textbook
	/// four-term sum, but grouped in two pairs the way Eigen's packet (SIMD) quaternion
	/// product groups them. The grouping was found with oracle/linear_algebra_rotations.py:
	/// of all 24 left-to-right orders, 3 pairings and their FMA variants, only this one is
	/// bit-identical on all fixture cases, for every coefficient.
	/// </summary>
	public static Quaterniond operator *(Quaterniond a, Quaterniond b)
	{
		return new Quaterniond(
			(a.W * b.W - a.Y * b.Y) + (-(a.X * b.X) - a.Z * b.Z),
			(a.W * b.X + a.Y * b.Z) + (a.X * b.W - a.Z * b.Y),
			(a.W * b.Y + a.Y * b.W) + (a.Z * b.X - a.X * b.Z),
			(a.W * b.Z - a.Y * b.X) + (a.Z * b.W + a.X * b.Y));
	}

	/// <summary>
	/// Rotates v by the unit quaternion q (q v q*), as t = 2 (u x v), v + w t + u x t.
	/// Like Eigen, q is assumed normalized.
	/// </summary>
	public static Vector3d operator *(Quaterniond q, Vector3d v)
	{
		Vector3d u = q.Vec;
		Vector3d uv = u.Cross(v);
		uv += uv;
		return v + q.W * uv + u.Cross(uv);
	}

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Quaterniond other) => Coeffs.Equals(other.Coeffs);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Quaterniond other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => Coeffs.GetHashCode();

	/// <inheritdoc/>
	public override string ToString()
	{
		return string.Create(CultureInfo.InvariantCulture, $"(w={W:R}, x={X:R}, y={Y:R}, z={Z:R})");
	}
}
