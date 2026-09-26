// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// RobustPredicates: exact-sign orientation and in-sphere tests for the 3D Delaunay
// tetrahedralization (DelaunayTriangulation3). This is part of the replacement for CGAL's
// Exact_predicates_inexact_constructions_kernel, which COLMAP's mvs/delaunay_meshing.cc uses
// and which is GPL (excluded, docs/LICENSE_AUDIT.md). No CGAL source was read.
//
// Algorithm: J. R. Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust
// Geometric Predicates", Discrete & Computational Geometry 18:305-363, 1997 (the paper and
// its predicates.c are public domain). Each predicate first evaluates the determinant in
// plain double arithmetic and compares it against Shewchuk's a-priori forward error bound
// (errboundA: ccwerrboundA, o3derrboundA, isperrboundA). Only when the result is within
// that bound does it fall back to exact arithmetic. The exact stage here is not Shewchuk's
// adaptive expansion arithmetic: it converts every input double to an exact scaled integer
// (mantissa * 2^(exponent - minimum exponent)) and evaluates the same polynomial with
// System.Numerics.BigInteger. That is simpler to get right and only runs in (near-)degenerate
// configurations, so its cost does not matter for ordinary input.
//
// Conventions follow CGAL's documented ones (not Shewchuk's, whose orient3d has the opposite
// sign), so the ported meshing code reads the same as COLMAP's:
// - Orient3D(p, q, r, s) > 0 when s lies on the side of plane (p, q, r) from which p, q, r
//   appear counterclockwise, i.e. sign(det[q - p; r - p; s - p]).
// - InSphere(p, q, r, s, t) > 0 when t lies strictly inside the sphere through p, q, r, s
//   and Orient3D(p, q, r, s) > 0 (the sign flips with the orientation of p, q, r, s).
// - Orient2D(a, b, c) > 0 when a, b, c turn counterclockwise.
//
// Inputs must be finite. Differences outside [1e-60, 1e60] skip the filter (see
// InFilterRange), so extreme scales stay exact, just slower. Results are exact signs, so they are
// platform-independent and deterministic. Tests: ColmapSharp.Tests/Geometry/Delaunay.

using System.Numerics;

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Geometry.Delaunay;

/// <summary>
/// Exact-sign geometric predicates with a floating-point filter (Shewchuk 1997).
/// </summary>
public static class RobustPredicates
{
	// Shewchuk's epsilon: half an ulp of 1, 2^-53.
	private const double Epsilon = 1.1102230246251565e-16;
	private const double CcwErrBoundA = (3.0 + 16.0 * Epsilon) * Epsilon;
	private const double O3dErrBoundA = (7.0 + 56.0 * Epsilon) * Epsilon;
	private const double IspErrBoundA = (16.0 + 224.0 * Epsilon) * Epsilon;

	// Below this magnitude the permanent (the bound's scale) may have lost accuracy to
	// underflow, which the a-priori bound does not model; go exact instead.
	private const double MinReliablePermanent = 1e-250;

	// Coordinate differences are only trusted to the filter within [1e-60, 1e60] (or exactly
	// zero): outside it, products and lifted squares can underflow to subnormals or overflow,
	// which the a-priori bound does not model. Such input goes straight to the exact path.
	private const double MinFilterMagnitude = 1e-60;
	private const double MaxFilterMagnitude = 1e60;

	private static bool InFilterRange(params ReadOnlySpan<double> differences)
	{
		foreach (double d in differences)
		{
			double magnitude = Math.Abs(d);
			if (magnitude != 0 && (magnitude < MinFilterMagnitude || magnitude > MaxFilterMagnitude))
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>Sign of the 2D orientation of (a, b, c): +1 counterclockwise, -1 clockwise, 0 collinear.</summary>
	public static int Orient2D(double ax, double ay, double bx, double by, double cx, double cy)
	{
		double detLeft = (ax - cx) * (by - cy);
		double detRight = (ay - cy) * (bx - cx);
		double det = detLeft - detRight;
		double permanent = Math.Abs(detLeft) + Math.Abs(detRight);
		if (permanent > MinReliablePermanent && InFilterRange(ax - cx, ay - cy, bx - cx, by - cy))
		{
			double errBound = CcwErrBoundA * permanent;
			if (det > errBound)
			{
				return 1;
			}

			if (-det > errBound)
			{
				return -1;
			}
		}

		return Orient2DExact(ax, ay, bx, by, cx, cy);
	}

	/// <summary>
	/// Sign of det[q - p; r - p; s - p]: +1 when s is on the side of plane (p, q, r) from
	/// which p, q, r appear counterclockwise (CGAL's orientation), 0 when coplanar.
	/// </summary>
	public static int Orient3D(in Vector3d p, in Vector3d q, in Vector3d r, in Vector3d s)
	{
		// Shewchuk's orient3d(a, b, c, d) = det[a - d; b - d; c - d] is the negation of the
		// CGAL orientation of (a, b, c, d), so the filter evaluates his formula and flips.
		double adx = p.X - s.X, bdx = q.X - s.X, cdx = r.X - s.X;
		double ady = p.Y - s.Y, bdy = q.Y - s.Y, cdy = r.Y - s.Y;
		double adz = p.Z - s.Z, bdz = q.Z - s.Z, cdz = r.Z - s.Z;

		double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
		double cdxady = cdx * ady, adxcdy = adx * cdy;
		double adxbdy = adx * bdy, bdxady = bdx * ady;

		double det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);
		double permanent = (Math.Abs(bdxcdy) + Math.Abs(cdxbdy)) * Math.Abs(adz)
			+ (Math.Abs(cdxady) + Math.Abs(adxcdy)) * Math.Abs(bdz)
			+ (Math.Abs(adxbdy) + Math.Abs(bdxady)) * Math.Abs(cdz);
		if (permanent > MinReliablePermanent
			&& InFilterRange(adx, bdx, cdx, ady, bdy, cdy) && InFilterRange(adz, bdz, cdz))
		{
			double errBound = O3dErrBoundA * permanent;
			if (det > errBound)
			{
				return -1;
			}

			if (-det > errBound)
			{
				return 1;
			}
		}

		return Orient3DExact(p, q, r, s);
	}

	/// <summary>
	/// +1 when t is strictly inside the sphere through p, q, r, s, -1 when strictly outside,
	/// 0 when on it, provided Orient3D(p, q, r, s) &gt; 0; the sign flips for a negatively
	/// oriented p, q, r, s (CGAL's side_of_oriented_sphere).
	/// </summary>
	public static int InSphere(in Vector3d p, in Vector3d q, in Vector3d r, in Vector3d s, in Vector3d t)
	{
		// Shewchuk's insphere(a, b, c, d, e) is positive for e inside when *his* orient3d is
		// positive, which is the CGAL-negative orientation; so again evaluate and flip.
		double aex = p.X - t.X, bex = q.X - t.X, cex = r.X - t.X, dex = s.X - t.X;
		double aey = p.Y - t.Y, bey = q.Y - t.Y, cey = r.Y - t.Y, dey = s.Y - t.Y;
		double aez = p.Z - t.Z, bez = q.Z - t.Z, cez = r.Z - t.Z, dez = s.Z - t.Z;

		double aexbey = aex * bey, bexaey = bex * aey;
		double bexcey = bex * cey, cexbey = cex * bey;
		double cexdey = cex * dey, dexcey = dex * cey;
		double dexaey = dex * aey, aexdey = aex * dey;
		double aexcey = aex * cey, cexaey = cex * aey;
		double bexdey = bex * dey, dexbey = dex * bey;

		double ab = aexbey - bexaey;
		double bc = bexcey - cexbey;
		double cd = cexdey - dexcey;
		double da = dexaey - aexdey;
		double ac = aexcey - cexaey;
		double bd = bexdey - dexbey;

		double abc = aez * bc - bez * ac + cez * ab;
		double bcd = bez * cd - cez * bd + dez * bc;
		double cda = cez * da + dez * ac + aez * cd;
		double dab = dez * ab + aez * bd + bez * da;

		double alift = aex * aex + aey * aey + aez * aez;
		double blift = bex * bex + bey * bey + bez * bez;
		double clift = cex * cex + cey * cey + cez * cez;
		double dlift = dex * dex + dey * dey + dez * dez;

		double det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);

		double aezp = Math.Abs(aez), bezp = Math.Abs(bez), cezp = Math.Abs(cez), dezp = Math.Abs(dez);
		double aexbeyp = Math.Abs(aexbey), bexaeyp = Math.Abs(bexaey);
		double bexceyp = Math.Abs(bexcey), cexbeyp = Math.Abs(cexbey);
		double cexdeyp = Math.Abs(cexdey), dexceyp = Math.Abs(dexcey);
		double dexaeyp = Math.Abs(dexaey), aexdeyp = Math.Abs(aexdey);
		double aexceyp = Math.Abs(aexcey), cexaeyp = Math.Abs(cexaey);
		double bexdeyp = Math.Abs(bexdey), dexbeyp = Math.Abs(dexbey);
		double permanent = ((cexdeyp + dexceyp) * bezp + (dexbeyp + bexdeyp) * cezp + (bexceyp + cexbeyp) * dezp) * alift
			+ ((dexaeyp + aexdeyp) * cezp + (aexceyp + cexaeyp) * dezp + (cexdeyp + dexceyp) * aezp) * blift
			+ ((aexbeyp + bexaeyp) * dezp + (bexdeyp + dexbeyp) * aezp + (dexaeyp + aexdeyp) * bezp) * clift
			+ ((bexceyp + cexbeyp) * aezp + (cexaeyp + aexceyp) * bezp + (aexbeyp + bexaeyp) * cezp) * dlift;
		if (permanent > MinReliablePermanent
			&& InFilterRange(aex, bex, cex, dex, aey, bey, cey, dey) && InFilterRange(aez, bez, cez, dez))
		{
			double errBound = IspErrBoundA * permanent;
			if (det > errBound)
			{
				return -1;
			}

			if (-det > errBound)
			{
				return 1;
			}
		}

		return InSphereExact(p, q, r, s, t);
	}

	/// <summary>
	/// Sign of det[q - p; r - p; to - from]: +1 when the direction from -&gt; to points to the
	/// positive side of the oriented plane (p, q, r) (the side Orient3D(p, q, r, s) &gt; 0 calls
	/// positive), -1 toward the negative side, 0 when parallel to the plane.
	/// </summary>
	public static int OrientDirection(in Vector3d p, in Vector3d q, in Vector3d r, in Vector3d from, in Vector3d to)
	{
		// Three rows of single rounded differences, the shape Shewchuk's orient3d bound covers.
		double adx = q.X - p.X, ady = q.Y - p.Y, adz = q.Z - p.Z;
		double bdx = r.X - p.X, bdy = r.Y - p.Y, bdz = r.Z - p.Z;
		double cdx = to.X - from.X, cdy = to.Y - from.Y, cdz = to.Z - from.Z;

		double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
		double cdxady = cdx * ady, adxcdy = adx * cdy;
		double adxbdy = adx * bdy, bdxady = bdx * ady;

		double det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);
		double permanent = (Math.Abs(bdxcdy) + Math.Abs(cdxbdy)) * Math.Abs(adz)
			+ (Math.Abs(cdxady) + Math.Abs(adxcdy)) * Math.Abs(bdz)
			+ (Math.Abs(adxbdy) + Math.Abs(bdxady)) * Math.Abs(cdz);
		if (permanent > MinReliablePermanent
			&& InFilterRange(adx, bdx, cdx, ady, bdy, cdy) && InFilterRange(adz, bdz, cdz))
		{
			double errBound = O3dErrBoundA * permanent;
			if (det > errBound)
			{
				return 1;
			}

			if (-det > errBound)
			{
				return -1;
			}
		}

		return OrientDirectionExact(p, q, r, from, to);
	}

	/// <summary>Exact OrientDirection without the filter.</summary>
	internal static int OrientDirectionExact(in Vector3d p, in Vector3d q, in Vector3d r, in Vector3d from, in Vector3d to)
	{
		Span<double> values = [p.X, p.Y, p.Z, q.X, q.Y, q.Z, r.X, r.Y, r.Z, from.X, from.Y, from.Z, to.X, to.Y, to.Z];
		Span<BigInteger> v = new BigInteger[15];
		ToScaledIntegers(values, v);
		BigInteger ax = v[3] - v[0], ay = v[4] - v[1], az = v[5] - v[2];
		BigInteger bx = v[6] - v[0], by = v[7] - v[1], bz = v[8] - v[2];
		BigInteger cx = v[12] - v[9], cy = v[13] - v[10], cz = v[14] - v[11];
		BigInteger det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
		return det.Sign;
	}

	/// <summary>Exact Orient2D without the filter; internal so tests can check the filter against it.</summary>
	internal static int Orient2DExact(double ax, double ay, double bx, double by, double cx, double cy)
	{
		Span<double> values = [ax, ay, bx, by, cx, cy];
		Span<BigInteger> v = new BigInteger[6];
		ToScaledIntegers(values, v);
		BigInteger acx = v[0] - v[4], bcx = v[2] - v[4];
		BigInteger acy = v[1] - v[5], bcy = v[3] - v[5];
		return (acx * bcy - acy * bcx).Sign;
	}

	/// <summary>Exact Orient3D without the filter.</summary>
	internal static int Orient3DExact(in Vector3d p, in Vector3d q, in Vector3d r, in Vector3d s)
	{
		Span<double> values = [p.X, p.Y, p.Z, q.X, q.Y, q.Z, r.X, r.Y, r.Z, s.X, s.Y, s.Z];
		Span<BigInteger> v = new BigInteger[12];
		ToScaledIntegers(values, v);
		BigInteger qx = v[3] - v[0], qy = v[4] - v[1], qz = v[5] - v[2];
		BigInteger rx = v[6] - v[0], ry = v[7] - v[1], rz = v[8] - v[2];
		BigInteger sx = v[9] - v[0], sy = v[10] - v[1], sz = v[11] - v[2];
		BigInteger det = qx * (ry * sz - rz * sy) - qy * (rx * sz - rz * sx) + qz * (rx * sy - ry * sx);
		return det.Sign;
	}

	/// <summary>Exact InSphere without the filter.</summary>
	internal static int InSphereExact(in Vector3d p, in Vector3d q, in Vector3d r, in Vector3d s, in Vector3d t)
	{
		Span<double> values = [p.X, p.Y, p.Z, q.X, q.Y, q.Z, r.X, r.Y, r.Z, s.X, s.Y, s.Z, t.X, t.Y, t.Z];
		Span<BigInteger> v = new BigInteger[15];
		ToScaledIntegers(values, v);

		// Rows (x - t, y - t, z - t, |x - t|^2) for x in p, q, r, s. The sign convention is
		// fixed by the unit tests (a point at the centroid of a positive tetrahedron is inside).
		Span<BigInteger> m = new BigInteger[16];
		for (int row = 0; row < 4; ++row)
		{
			BigInteger dx = v[3 * row] - v[12];
			BigInteger dy = v[3 * row + 1] - v[13];
			BigInteger dz = v[3 * row + 2] - v[14];
			m[4 * row] = dx;
			m[4 * row + 1] = dy;
			m[4 * row + 2] = dz;
			m[4 * row + 3] = dx * dx + dy * dy + dz * dz;
		}

		// For p, q, r, s positively oriented and t inside, det[rows] is negative (lifting
		// map: t below the plane of the lifted points), so the in-sphere sign is -det.
		return -Determinant4(m).Sign;
	}

	private static BigInteger Determinant4(ReadOnlySpan<BigInteger> m)
	{
		// Laplace expansion along the last column, with 2x2 minors of the first two rows.
		BigInteger m01 = m[0] * m[5] - m[1] * m[4];
		BigInteger m02 = m[0] * m[9] - m[1] * m[8];
		BigInteger m03 = m[0] * m[13] - m[1] * m[12];
		BigInteger m12 = m[4] * m[9] - m[5] * m[8];
		BigInteger m13 = m[4] * m[13] - m[5] * m[12];
		BigInteger m23 = m[8] * m[13] - m[9] * m[12];

		// 3x3 minors over columns (0, 1, 2), rows excluding one each.
		BigInteger c012 = m01 * m[10] - m02 * m[6] + m12 * m[2];
		BigInteger c013 = m01 * m[14] - m03 * m[6] + m13 * m[2];
		BigInteger c023 = m02 * m[14] - m03 * m[10] + m23 * m[2];
		BigInteger c123 = m12 * m[14] - m13 * m[10] + m23 * m[6];

		return -m[3] * c123 + m[7] * c023 - m[11] * c013 + m[15] * c012;
	}

	/// <summary>
	/// Writes each value as an exact integer multiple of 2^e, e being the smallest exponent
	/// among the nonzero inputs. A common positive scale leaves every predicate sign unchanged.
	/// </summary>
	private static void ToScaledIntegers(ReadOnlySpan<double> values, Span<BigInteger> result)
	{
		Span<long> mantissas = stackalloc long[values.Length];
		Span<int> exponents = stackalloc int[values.Length];
		int minExponent = int.MaxValue;
		for (int i = 0; i < values.Length; ++i)
		{
			double value = values[i];
			if (!double.IsFinite(value))
			{
				throw new ArgumentException("Check failed: coordinates must be finite.");
			}

			long bits = BitConverter.DoubleToInt64Bits(value);
			int biasedExponent = (int)((bits >> 52) & 0x7FF);
			long fraction = bits & 0xFFFFFFFFFFFFFL;
			long mantissa;
			int exponent;
			if (biasedExponent == 0)
			{
				mantissa = fraction;
				exponent = -1074;
			}
			else
			{
				mantissa = fraction | (1L << 52);
				exponent = biasedExponent - 1075;
			}

			if (bits < 0)
			{
				mantissa = -mantissa;
			}

			mantissas[i] = mantissa;
			exponents[i] = exponent;
			if (mantissa != 0 && exponent < minExponent)
			{
				minExponent = exponent;
			}
		}

		for (int i = 0; i < values.Length; ++i)
		{
			result[i] = mantissas[i] == 0 ? BigInteger.Zero : new BigInteger(mantissas[i]) << (exponents[i] - minExponent);
		}
	}
}
