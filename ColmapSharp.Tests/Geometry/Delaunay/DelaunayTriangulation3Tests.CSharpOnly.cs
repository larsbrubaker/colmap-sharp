// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DelaunayTriangulation3Tests (C#-only): the tetrahedralization that replaces CGAL's
// Delaunay_triangulation_3 for mvs/delaunay_meshing (ColmapSharp/Geometry/Delaunay). COLMAP
// has no test of CGAL itself; its delaunay_meshing_test.cc integration tests are ported with
// the meshing slice. These pin the properties the meshing relies on: a valid, positively
// oriented, closed cell complex; the empty-circumsphere property; the Euler characteristic;
// robustness on coplanar, cospherical and duplicate input; determinism; and segment traversal
// against a brute-force scan of all facets. Decisions are exact, so these are Tier A
// properties (the specific triangulation of cospherical points is ours, divergence 103).

using ColmapSharp.Geometry.Delaunay;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Geometry.Delaunay;

public class DelaunayTriangulation3Tests
{
	[Test]
	public async Task CSharpOnly_RandomPointsHaveEmptyCircumspheres()
	{
		var points = RandomPoints(new Random(1), 400, 10.0);
		var dt = new DelaunayTriangulation3();
		dt.InsertRange(points);

		using (Assert.Multiple())
		{
			await Assert.That(dt.NumberOfVertices).IsEqualTo(400);
			await Assert.That(DelaunayValidator.StructureErrors(dt)).IsEmpty();
			await Assert.That(DelaunayValidator.GlobalDelaunayViolations(dt)).IsEqualTo(0);
		}
	}

	[Test]
	public async Task CSharpOnly_EulerCharacteristic()
	{
		foreach (int n in new[] { 5, 60, 700 })
		{
			var dt = new DelaunayTriangulation3();
			dt.InsertRange(RandomPoints(new Random(n), n, 1.0));
			long v = dt.NumberOfVertices;
			long e = dt.FiniteEdges().Count();
			long f = dt.FiniteFacets().Count();
			long c = dt.FiniteCells().Count();

			// The finite complex triangulates a ball: V - E + F - C = 1. Adding the infinite
			// vertex closes it into a 3-sphere, whose Euler characteristic is 0.
			long hullFacets = dt.AllCells().Count(dt.IsInfinite);
			long hullEdges = hullFacets * 3 / 2;
			long hullVertices = hullEdges - hullFacets + 2;
			using (Assert.Multiple())
			{
				await Assert.That(v - e + f - c).IsEqualTo(1);
				await Assert.That((v + 1) - (e + hullVertices) + (f + hullEdges) - (c + hullFacets)).IsEqualTo(0);
				await Assert.That((long)dt.NumberOfCells).IsEqualTo(c + hullFacets);
			}
		}
	}

	[Test]
	public async Task CSharpOnly_GridPointsAreCoplanarAndCospherical()
	{
		// A cubic grid: every unit cube's 8 corners are cospherical and every grid plane is
		// full of coplanar points, the worst case for the predicates.
		var points = new List<Vector3d>();
		for (int x = 0; x < 6; ++x)
		{
			for (int y = 0; y < 6; ++y)
			{
				for (int z = 0; z < 6; ++z)
				{
					points.Add(new Vector3d(x, y, z));
				}
			}
		}

		var dt = new DelaunayTriangulation3();
		dt.InsertRange(points.ToArray());
		using (Assert.Multiple())
		{
			await Assert.That(dt.NumberOfVertices).IsEqualTo(216);
			await Assert.That(DelaunayValidator.StructureErrors(dt)).IsEmpty();
			await Assert.That(DelaunayValidator.GlobalDelaunayViolations(dt)).IsEqualTo(0);
			await Assert.That(DelaunayValidator.FiniteVolume(dt)).IsEqualTo(125.0).Within(1e-9);
		}
	}

	[Test]
	public async Task CSharpOnly_ExactlyCosphericalPoints()
	{
		// All integer points with x^2 + y^2 + z^2 = 325: exactly on one sphere.
		var points = new List<Vector3d>();
		for (int x = -18; x <= 18; ++x)
		{
			for (int y = -18; y <= 18; ++y)
			{
				for (int z = -18; z <= 18; ++z)
				{
					if (x * x + y * y + z * z == 325)
					{
						points.Add(new Vector3d(x, y, z));
					}
				}
			}
		}

		var dt = new DelaunayTriangulation3();
		dt.InsertRange(points.ToArray());
		using (Assert.Multiple())
		{
			await Assert.That(points.Count).IsGreaterThan(50);
			await Assert.That(dt.NumberOfVertices).IsEqualTo(points.Count);
			await Assert.That(DelaunayValidator.StructureErrors(dt)).IsEmpty();
			await Assert.That(DelaunayValidator.GlobalDelaunayViolations(dt)).IsEqualTo(0);
		}
	}

	[Test]
	public async Task CSharpOnly_DuplicatePointsShareOneVertex()
	{
		var random = new Random(7);
		var unique = RandomPoints(random, 100, 1.0);
		unique[0] = new Vector3d(0.0, 0.0, 0.0);
		var points = new Vector3d[300];
		for (int i = 0; i < points.Length; ++i)
		{
			points[i] = unique[i % 100];
		}

		// Negative zero is the same point as positive zero.
		points[100] = new Vector3d(-0.0, 0.0, -0.0);

		var dt = new DelaunayTriangulation3();
		var handles = dt.InsertRange(points);
		int mismatched = 0;
		for (int i = 0; i < points.Length; ++i)
		{
			if (handles[i] != handles[i % 100])
			{
				++mismatched;
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(dt.NumberOfVertices).IsEqualTo(100);
			await Assert.That(mismatched).IsEqualTo(0);
			await Assert.That(dt.Insert(unique[42])).IsEqualTo(handles[42]);
			await Assert.That(dt.NumberOfVertices).IsEqualTo(100);
			await Assert.That(DelaunayValidator.StructureErrors(dt)).IsEmpty();
		}
	}

	[Test]
	public async Task CSharpOnly_LowDimensionalInputWaitsForAFourthIndependentPoint()
	{
		var dt = new DelaunayTriangulation3();
		await Assert.That(dt.Dimension).IsEqualTo(-1);
		dt.Insert(new Vector3d(0, 0, 0));
		dt.Insert(new Vector3d(0, 0, 0));
		await Assert.That(dt.Dimension).IsEqualTo(0);

		// Collinear, then coplanar points: no cells yet.
		for (int i = 1; i <= 5; ++i)
		{
			dt.Insert(new Vector3d(i, 0, 0));
		}

		await Assert.That(dt.Dimension).IsEqualTo(1);
		for (int i = 0; i < 30; ++i)
		{
			dt.Insert(new Vector3d(i % 6, i / 6, 0));
		}

		using (Assert.Multiple())
		{
			await Assert.That(dt.Dimension).IsEqualTo(2);
			await Assert.That(dt.NumberOfCells).IsEqualTo(0);
		}

		// One point off the plane makes it 3D and brings in every pending point.
		int before = dt.NumberOfVertices;
		dt.Insert(new Vector3d(2.5, 2.5, 1));
		using (Assert.Multiple())
		{
			await Assert.That(dt.Dimension).IsEqualTo(3);
			await Assert.That(dt.NumberOfVertices).IsEqualTo(before + 1);
			await Assert.That(DelaunayValidator.StructureErrors(dt)).IsEmpty();
			await Assert.That(DelaunayValidator.GlobalDelaunayViolations(dt)).IsEqualTo(0);
			await Assert.That(DelaunayValidator.UsedVertexCount(dt)).IsEqualTo(before + 1);
		}
	}

	[Test]
	public async Task CSharpOnly_IncrementalInsertMatchesDelaunayAndIsDeterministic()
	{
		var points = RandomPoints(new Random(3), 300, 5.0);
		var first = new DelaunayTriangulation3();
		foreach (var p in points)
		{
			first.Insert(p);
		}

		var a = new DelaunayTriangulation3();
		var b = new DelaunayTriangulation3();
		a.InsertRange(points);
		b.InsertRange(points);
		using (Assert.Multiple())
		{
			await Assert.That(DelaunayValidator.StructureErrors(first)).IsEmpty();
			await Assert.That(DelaunayValidator.GlobalDelaunayViolations(first)).IsEqualTo(0);
			await Assert.That(DelaunayValidator.Fingerprint(a)).IsEqualTo(DelaunayValidator.Fingerprint(b));

			// Points in general position have one Delaunay triangulation, whatever the order.
			await Assert.That(DelaunayValidator.CellSet(first)).IsEquivalentTo(DelaunayValidator.CellSet(a));
		}
	}

	[Test]
	public async Task CSharpOnly_LargeInputIsLocallyDelaunay()
	{
		var points = RandomPoints(new Random(11), 20000, 100.0);
		var dt = new DelaunayTriangulation3();
		dt.InsertRange(points);
		using (Assert.Multiple())
		{
			await Assert.That(dt.NumberOfVertices).IsEqualTo(20000);
			await Assert.That(DelaunayValidator.StructureErrors(dt)).IsEmpty();
			// Locally Delaunay everywhere plus a convex hull implies globally Delaunay.
			await Assert.That(DelaunayValidator.LocalDelaunayViolations(dt)).IsEqualTo(0);
		}
	}

	[Test]
	public async Task CSharpOnly_InsertRangeHonorsCancellation()
	{
		using var cancellation = new CancellationTokenSource();
		cancellation.Cancel();
		var dt = new DelaunayTriangulation3();
		await Assert.That(() => dt.InsertRange(RandomPoints(new Random(1), 10, 1.0), cancellation.Token))
			.Throws<OperationCanceledException>();
	}

	[Test]
	public async Task CSharpOnly_CircumcenterIsEquidistant()
	{
		var dt = new DelaunayTriangulation3();
		dt.InsertRange(RandomPoints(new Random(5), 50, 1.0));
		double worst = 0;
		foreach (int cell in dt.FiniteCells())
		{
			var center = dt.Circumcenter(cell);
			double r0 = Distance(center, dt.Point(dt.Vertex(cell, 0)));
			for (int i = 1; i < 4; ++i)
			{
				worst = Math.Max(worst, Math.Abs(Distance(center, dt.Point(dt.Vertex(cell, i))) - r0) / r0);
			}
		}

		await Assert.That(worst).IsLessThan(1e-9);
	}

	[Test]
	public async Task CSharpOnly_SegmentTraversalMatchesBruteForce()
	{
		var random = new Random(21);
		var dt = new DelaunayTriangulation3();
		dt.InsertRange(RandomPoints(random, 250, 2.0));
		var crossed = new List<(int Cell, int Index)>();
		int checkedSegments = 0, mismatches = 0, outsideStarts = 0;
		for (int trial = 0; trial < 200; ++trial)
		{
			// Endpoints inside and outside the hull (points span [-1, 1]^3).
			double sourceScale = trial % 2 == 0 ? 1.6 : 0.8;
			double targetScale = trial % 3 == 0 ? 1.6 : 0.8;
			var source = RandomPoint(random, 2 * sourceScale);
			var target = RandomPoint(random, 2 * targetScale);
			if (dt.IsInfinite(dt.Locate(source)))
			{
				++outsideStarts;
			}

			dt.TraverseSegment(source, target, crossed);
			var expected = new HashSet<(int, int, int)>();
			foreach (var (cell, i) in dt.FiniteFacets())
			{
				var (u, v, w) = dt.FacetVertices(cell, i);
				if (DelaunayTriangulation3.SegmentCrossesTriangle(source, target, dt.Point(u), dt.Point(v), dt.Point(w)))
				{
					expected.Add(SortedTriple(u, v, w));
				}
			}

			var actual = crossed.Select(f => dt.FacetVertices(f.Cell, f.Index)).Select(t => SortedTriple(t.A, t.B, t.C)).ToList();
			bool chained = true;
			for (int k = 1; k < crossed.Count; ++k)
			{
				// Each crossing leaves the cell the previous one entered.
				chained &= dt.Neighbor(crossed[k - 1].Cell, crossed[k - 1].Index) == crossed[k].Cell;
			}

			if (!chained || actual.Count != expected.Count || !expected.SetEquals(actual) || !OrderedAlong(dt, crossed, source, target))
			{
				++mismatches;
			}

			++checkedSegments;
		}

		using (Assert.Multiple())
		{
			await Assert.That(checkedSegments).IsEqualTo(200);
			await Assert.That(outsideStarts).IsGreaterThan(20);
			await Assert.That(mismatches).IsEqualTo(0);
		}
	}

	[Test]
	public async Task CSharpOnly_RayTriangleIntersection()
	{
		var u = new Vector3d(0, 0, 0);
		var v = new Vector3d(1, 0, 0);
		var w = new Vector3d(0, 1, 0);
		var above = new Vector3d(0.25, 0.25, 1);
		using (Assert.Multiple())
		{
			// Toward the plane, beyond the through point: hit.
			await Assert.That(DelaunayTriangulation3.TryIntersectRayTriangle(above, new Vector3d(0.25, 0.25, 0.5), u, v, w, out var hit)).IsTrue();
			await Assert.That(hit).IsEqualTo(new Vector3d(0.25, 0.25, 0));

			// Away from the plane: no hit, even though the line crosses the triangle.
			await Assert.That(DelaunayTriangulation3.TryIntersectRayTriangle(above, new Vector3d(0.25, 0.25, 2), u, v, w, out _)).IsFalse();

			// Toward the plane but missing the triangle.
			await Assert.That(DelaunayTriangulation3.TryIntersectRayTriangle(above, new Vector3d(1, 1, 0.5), u, v, w, out _)).IsFalse();

			// The closed triangle: an edge hit counts.
			await Assert.That(DelaunayTriangulation3.TryIntersectRayTriangle(new Vector3d(0.5, 0, 1), new Vector3d(0.5, 0, 0.5), u, v, w, out var edgeHit)).IsTrue();
			await Assert.That(edgeHit).IsEqualTo(new Vector3d(0.5, 0, 0));

			// Endpoints on the triangle are returned exactly.
			await Assert.That(DelaunayTriangulation3.TryIntersectRayTriangle(new Vector3d(0.2, 0.3, 0), above, u, v, w, out var atSource)).IsTrue();
			await Assert.That(atSource).IsEqualTo(new Vector3d(0.2, 0.3, 0));
			await Assert.That(DelaunayTriangulation3.TryIntersectRayTriangle(above, new Vector3d(0.1, 0.2, 0), u, v, w, out var atThrough)).IsTrue();
			await Assert.That(atThrough).IsEqualTo(new Vector3d(0.1, 0.2, 0));

			// In the plane (CGAL would return a segment) or parallel above it: no point.
			await Assert.That(DelaunayTriangulation3.TryIntersectRayTriangle(new Vector3d(-1, 0.2, 0), new Vector3d(2, 0.2, 0), u, v, w, out _)).IsFalse();
			await Assert.That(DelaunayTriangulation3.TryIntersectRayTriangle(above, new Vector3d(2, 0.25, 1), u, v, w, out _)).IsFalse();

			// The segment version stops at its end point.
			await Assert.That(DelaunayTriangulation3.TryIntersectSegmentTriangle(above, new Vector3d(0.25, 0.25, 0.5), u, v, w, out _)).IsFalse();
			await Assert.That(DelaunayTriangulation3.TryIntersectSegmentTriangle(above, new Vector3d(0.25, 0.25, -1), u, v, w, out var segmentHit)).IsTrue();
			await Assert.That(segmentHit).IsEqualTo(new Vector3d(0.25, 0.25, 0));
		}
	}

	// Crossing points are at increasing parameter along the segment.
	private static bool OrderedAlong(DelaunayTriangulation3 dt, List<(int Cell, int Index)> crossed, Vector3d a, Vector3d b)
	{
		double previous = double.NegativeInfinity;
		foreach (var (cell, i) in crossed)
		{
			var (u, v, w) = dt.FacetVertices(cell, i);
			double t = SegmentPlaneParameter(a, b, dt.Point(u), dt.Point(v), dt.Point(w));
			if (t < previous - 1e-12)
			{
				return false;
			}

			previous = t;
		}

		return true;
	}

	private static double SegmentPlaneParameter(Vector3d a, Vector3d b, Vector3d u, Vector3d v, Vector3d w)
	{
		double e1x = v.X - u.X, e1y = v.Y - u.Y, e1z = v.Z - u.Z;
		double e2x = w.X - u.X, e2y = w.Y - u.Y, e2z = w.Z - u.Z;
		double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
		double da = nx * (a.X - u.X) + ny * (a.Y - u.Y) + nz * (a.Z - u.Z);
		double db = nx * (b.X - u.X) + ny * (b.Y - u.Y) + nz * (b.Z - u.Z);
		return da / (da - db);
	}

	private static (int, int, int) SortedTriple(int a, int b, int c)
	{
		Span<int> s = [a, b, c];
		s.Sort();
		return (s[0], s[1], s[2]);
	}

	private static double Distance(Vector3d a, Vector3d b)
	{
		double dx = a.X - b.X, dy = a.Y - b.Y, dz = a.Z - b.Z;
		return Math.Sqrt(dx * dx + dy * dy + dz * dz);
	}

	private static Vector3d RandomPoint(Random random, double extent) =>
		new((random.NextDouble() - 0.5) * extent, (random.NextDouble() - 0.5) * extent, (random.NextDouble() - 0.5) * extent);

	private static Vector3d[] RandomPoints(Random random, int count, double extent)
	{
		var points = new Vector3d[count];
		for (int i = 0; i < count; ++i)
		{
			points[i] = RandomPoint(random, extent);
		}

		return points;
	}
}
