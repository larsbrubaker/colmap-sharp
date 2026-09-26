// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// RobustPredicatesTests (C#-only; COLMAP has no test for CGAL's kernel): sign conventions of
// ColmapSharp/Geometry/Delaunay/RobustPredicates.cs and agreement of the filtered predicates
// with their exact (BigInteger) evaluation on near-degenerate input, where the filter must
// either be right or defer. Exact signs, so Tier A.

using ColmapSharp.Geometry.Delaunay;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Geometry.Delaunay;

public class RobustPredicatesTests
{
	private static readonly Vector3d O = new(0, 0, 0);
	private static readonly Vector3d X = new(1, 0, 0);
	private static readonly Vector3d Y = new(0, 1, 0);
	private static readonly Vector3d Z = new(0, 0, 1);

	[Test]
	public async Task CSharpOnly_SignConventions()
	{
		using (Assert.Multiple())
		{
			// CGAL convention: (0, x, y) is counterclockwise seen from +z, so +z is positive.
			await Assert.That(RobustPredicates.Orient3D(O, X, Y, Z)).IsEqualTo(1);
			await Assert.That(RobustPredicates.Orient3D(X, O, Y, Z)).IsEqualTo(-1);
			await Assert.That(RobustPredicates.Orient3D(O, X, Y, new Vector3d(3, 7, 0))).IsEqualTo(0);

			var centroid = new Vector3d(0.25, 0.25, 0.25);
			await Assert.That(RobustPredicates.InSphere(O, X, Y, Z, centroid)).IsEqualTo(1);
			await Assert.That(RobustPredicates.InSphere(X, O, Y, Z, centroid)).IsEqualTo(-1);
			await Assert.That(RobustPredicates.InSphere(O, X, Y, Z, new Vector3d(5, 5, 5))).IsEqualTo(-1);
			await Assert.That(RobustPredicates.InSphere(O, X, Y, Z, new Vector3d(1, 1, 1))).IsEqualTo(0);

			await Assert.That(RobustPredicates.Orient2D(0, 0, 1, 0, 0, 1)).IsEqualTo(1);
			await Assert.That(RobustPredicates.Orient2D(0, 0, 0, 1, 1, 0)).IsEqualTo(-1);
			await Assert.That(RobustPredicates.Orient2D(0, 0, 1, 1, 3, 3)).IsEqualTo(0);

			// The exact evaluations follow the same conventions.
			await Assert.That(RobustPredicates.Orient3DExact(O, X, Y, Z)).IsEqualTo(1);
			await Assert.That(RobustPredicates.InSphereExact(O, X, Y, Z, centroid)).IsEqualTo(1);
			await Assert.That(RobustPredicates.Orient2DExact(0, 0, 1, 0, 0, 1)).IsEqualTo(1);
		}
	}

	[Test]
	public async Task CSharpOnly_FilteredMatchesExactNearDegenerate()
	{
		var random = new Random(12345);
		int mismatches = 0;
		int zeros = 0;
		for (int trial = 0; trial < 4000; ++trial)
		{
			// Points on a plane / sphere, then nudged by a few ulps or not at all: the filter
			// cannot decide these, so they exercise the exact fallback.
			// Dyadic coordinates keep a + s (b - a) + t (c - a) exact, so un-nudged trials are
			// exactly coplanar.
			double scale = Math.ScaleB(1.0, random.Next(-10, 11));
			var a = DyadicPoint(random, scale);
			var b = DyadicPoint(random, scale);
			var c = DyadicPoint(random, scale);
			double s = random.Next(0, 65) / 64.0, t = random.Next(0, 65) / 64.0;
			var onPlane = new Vector3d(
				a.X + s * (b.X - a.X) + t * (c.X - a.X),
				a.Y + s * (b.Y - a.Y) + t * (c.Y - a.Y),
				a.Z + s * (b.Z - a.Z) + t * (c.Z - a.Z));
			onPlane = Nudge(random, onPlane);
			int orient = RobustPredicates.Orient3D(a, b, c, onPlane);
			if (orient != RobustPredicates.Orient3DExact(a, b, c, onPlane))
			{
				++mismatches;
			}

			if (orient == 0)
			{
				++zeros;
			}

			// Cospherical: points on a sphere around the origin, nudged.
			double radius = scale;
			var p = OnSphere(random, radius);
			var q = OnSphere(random, radius);
			var r = OnSphere(random, radius);
			var u = OnSphere(random, radius);
			var e = Nudge(random, OnSphere(random, radius));
			if (RobustPredicates.InSphere(p, q, r, u, e) != RobustPredicates.InSphereExact(p, q, r, u, e))
			{
				++mismatches;
			}

			// Collinear 2D.
			double x0 = random.NextDouble() * scale, y0 = random.NextDouble() * scale;
			double x1 = random.NextDouble() * scale, y1 = random.NextDouble() * scale;
			double x2 = x0 + s * (x1 - x0), y2 = y0 + s * (y1 - y0);
			if (RobustPredicates.Orient2D(x0, y0, x1, y1, x2, y2) != RobustPredicates.Orient2DExact(x0, y0, x1, y1, x2, y2))
			{
				++mismatches;
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(mismatches).IsEqualTo(0);
			// Some trials must actually be degenerate, or the test proves little.
			await Assert.That(zeros).IsGreaterThan(0);
		}
	}

	[Test]
	public async Task CSharpOnly_OrientDirectionMatchesOrient3DAndItsExactForm()
	{
		var random = new Random(99);
		int mismatches = 0, zeros = 0;
		for (int trial = 0; trial < 2000; ++trial)
		{
			var p = Dyadic(random);
			var q = Dyadic(random);
			var r = Dyadic(random);
			var s = Dyadic(random);

			// The direction p -> s against plane (p, q, r) is Orient3D(p, q, r, s).
			if (RobustPredicates.OrientDirection(p, q, r, p, s) != RobustPredicates.Orient3D(p, q, r, s))
			{
				++mismatches;
			}

			// A direction inside the plane (q - r, applied anywhere) is exactly parallel.
			var from = Dyadic(random);
			var to = new Vector3d(from.X + (q.X - r.X), from.Y + (q.Y - r.Y), from.Z + (q.Z - r.Z));
			int parallel = RobustPredicates.OrientDirection(p, q, r, from, to);
			if (parallel != RobustPredicates.OrientDirectionExact(p, q, r, from, to))
			{
				++mismatches;
			}

			if (parallel == 0)
			{
				++zeros;
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(RobustPredicates.OrientDirection(O, X, Y, new Vector3d(5, 5, 5), new Vector3d(5, 5, 6))).IsEqualTo(1);
			await Assert.That(RobustPredicates.OrientDirection(O, X, Y, new Vector3d(5, 5, 5), new Vector3d(5, 5, 4))).IsEqualTo(-1);
			await Assert.That(mismatches).IsEqualTo(0);
			await Assert.That(zeros).IsGreaterThan(1000);
		}
	}

	[Test]
	public async Task CSharpOnly_FilteredMatchesExactAcrossExtremeScales()
	{
		// Coordinate differences spanning 1e-200 .. 1e200: products can underflow to zero or
		// overflow, which the a-priori error bound does not model, so the filter must defer.
		var random = new Random(2024);
		int mismatches = 0;
		for (int trial = 0; trial < 3000; ++trial)
		{
			var p = ExtremePoint(random);
			var q = ExtremePoint(random);
			var r = ExtremePoint(random);
			var s = ExtremePoint(random);
			var t = ExtremePoint(random);
			if (RobustPredicates.Orient3D(p, q, r, s) != RobustPredicates.Orient3DExact(p, q, r, s))
			{
				++mismatches;
			}

			if (RobustPredicates.InSphere(p, q, r, s, t) != RobustPredicates.InSphereExact(p, q, r, s, t))
			{
				++mismatches;
			}

			if (RobustPredicates.OrientDirection(p, q, r, s, t) != RobustPredicates.OrientDirectionExact(p, q, r, s, t))
			{
				++mismatches;
			}

			if (RobustPredicates.Orient2D(p.X, p.Y, q.X, q.Y, r.X, r.Y) != RobustPredicates.Orient2DExact(p.X, p.Y, q.X, q.Y, r.X, r.Y))
			{
				++mismatches;
			}
		}

		await Assert.That(mismatches).IsEqualTo(0);
	}

	[Test]
	public async Task CSharpOnly_UnderflowingProductsDoNotFoolTheFilter()
	{
		// The z extent (1e-170) squared underflows in the double products, so the plain
		// determinant loses the term that decides the sign while the permanent stays large.
		var p = new Vector3d(0, 0, 0);
		var q = new Vector3d(1, 0, 1e-170);
		var r = new Vector3d(0, 1, -1e-170);
		var s = new Vector3d(1e-170, 1e-170, 0);
		using (Assert.Multiple())
		{
			await Assert.That(RobustPredicates.Orient3D(p, q, r, s)).IsEqualTo(RobustPredicates.Orient3DExact(p, q, r, s));
			await Assert.That(RobustPredicates.OrientDirection(p, q, r, p, s)).IsEqualTo(RobustPredicates.Orient3DExact(p, q, r, s));
		}
	}

	[Test]
	public async Task CSharpOnly_ReviewerUnderflowCounterexample()
	{
		// From the Delaunay review: bdx * cdy underflows while the permanent (about 5e-101) is
		// far above the old 1e-250 cutoff, so the unguarded filter trusted a wrong sign.
		var p = new Vector3d(0, 0.5, 1e300);
		var q = new Vector3d(1e-200, 0, 0);
		var r = new Vector3d(0, 1e-200, 1e100);
		var s = new Vector3d(0, 0, 0);
		int exact = RobustPredicates.Orient3DExact(p, q, r, s);
		using (Assert.Multiple())
		{
			await Assert.That(exact).IsEqualTo(-1);
			await Assert.That(RobustPredicates.Orient3D(p, q, r, s)).IsEqualTo(exact);
			await Assert.That(RobustPredicates.OrientDirection(p, q, r, p, s)).IsEqualTo(exact);
		}
	}

	private static Vector3d ExtremePoint(Random random)
	{
		double Coordinate() => (random.NextDouble() - 0.5) * Math.Pow(10, random.Next(-200, 201));
		return new Vector3d(Coordinate(), Coordinate(), Coordinate());
	}

	private static Vector3d Dyadic(Random random) => DyadicPoint(random, 1.0);

	[Test]
	public async Task CSharpOnly_ExactHandlesWideExponentRange()
	{
		// Coordinates 40 orders of magnitude apart: the scaled-integer exact path must keep
		// every bit (b is one ulp above a).
		var a = new Vector3d(1e20, 0, 0);
		var b = new Vector3d(Math.BitIncrement(1e20), 0, 0);
		var c = new Vector3d(1e20, 1, 0);
		var d = new Vector3d(1e20, 0, 1e-20);
		using (Assert.Multiple())
		{
			await Assert.That(RobustPredicates.Orient3D(a, b, c, d)).IsEqualTo(1);
			await Assert.That(RobustPredicates.Orient3DExact(a, b, c, d)).IsEqualTo(1);
			await Assert.That(RobustPredicates.Orient3DExact(b, a, c, d)).IsEqualTo(-1);
		}
	}

	private static Vector3d DyadicPoint(Random random, double scale) =>
		new(random.Next(-1024, 1025) * scale / 1024, random.Next(-1024, 1025) * scale / 1024, random.Next(-1024, 1025) * scale / 1024);

	private static Vector3d OnSphere(Random random, double radius)
	{
		double theta = random.NextDouble() * 2 * Math.PI, z = 2 * random.NextDouble() - 1;
		double rho = Math.Sqrt(1 - z * z);
		return new Vector3d(radius * rho * Math.Cos(theta), radius * rho * Math.Sin(theta), radius * z);
	}

	private static Vector3d Nudge(Random random, Vector3d p)
	{
		int ulps = random.Next(-2, 3);
		double x = p.X;
		for (int i = 0; i < Math.Abs(ulps); ++i)
		{
			x = ulps > 0 ? Math.BitIncrement(x) : Math.BitDecrement(x);
		}

		return new Vector3d(x, p.Y, p.Z);
	}
}
