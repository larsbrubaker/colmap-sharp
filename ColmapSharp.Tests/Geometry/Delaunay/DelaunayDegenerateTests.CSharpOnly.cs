// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DelaunayDegenerateTests (C#-only): the degenerate-input harness for
// ColmapSharp/Geometry/Delaunay - integer grids, coplanar, cospherical and collinear sets and
// duplicates must give valid Delaunay triangulations; grid-aligned segments whose endpoints
// sit on vertices, edges and facets must traverse without looping and only through facets
// they touch; and, for segments in general position, TraverseSegment and HullFacetIndex are
// checked against an oracle that shares no code with them (Moller-Trumbore ray/triangle
// intersection in plain doubles, used only where it is far from ambiguous).

using ColmapSharp.Geometry.Delaunay;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Geometry.Delaunay;

public class DelaunayDegenerateTests
{
	[Test]
	public async Task CSharpOnly_DegenerateInputSetsAreDelaunay()
	{
		var sets = new Dictionary<string, List<Vector3d>>
		{
			["grid"] = Grid(4),
			["grid with duplicates"] = [.. Grid(3), .. Grid(3), new Vector3d(-0.0, 0, 0)],
			["plane plus apex"] = [.. Plane(6), new Vector3d(2.5, 2.5, 3)],
			["collinear first"] = [.. Enumerable.Range(0, 8).Select(i => new Vector3d(i, 0, 0)), .. Grid(3)],
			["cospherical"] = SpherePoints(50),
			["two cospherical shells"] = [.. SpherePoints(50), .. SpherePoints(200)],
		};

		var failures = new List<string>();
		foreach (var (name, points) in sets)
		{
			var dt = new DelaunayTriangulation3();
			dt.InsertRange(points.ToArray());
			int distinct = points.Select(p => (p.X + 0.0, p.Y + 0.0, p.Z + 0.0)).Distinct().Count();
			if (dt.NumberOfVertices != distinct)
			{
				failures.Add($"{name}: {dt.NumberOfVertices} vertices, expected {distinct}");
			}

			failures.AddRange(DelaunayValidator.StructureErrors(dt).Select(e => $"{name}: {e}"));
			int violations = DelaunayValidator.GlobalDelaunayViolations(dt);
			if (violations != 0)
			{
				failures.Add($"{name}: {violations} Delaunay violations");
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task CSharpOnly_GridAlignedSegmentsTraverseThroughTouchedFacets()
	{
		var dt = new DelaunayTriangulation3();
		dt.InsertRange(Grid(4).ToArray());

		// Endpoints on vertices, edge midpoints, face centers and cube centers, inside and
		// just outside the grid.
		var anchors = new List<Vector3d>();
		foreach (double x in new[] { 0.0, 0.5, 1.0, 1.5, 3.0, 4.0 })
		{
			foreach (double y in new[] { 0.0, 0.5, 2.0, 3.0 })
			{
				foreach (double z in new[] { 0.0, 0.5, 1.5, 3.0, -1.0 })
				{
					anchors.Add(new Vector3d(x, y, z));
				}
			}
		}

		var crossed = new List<(int Cell, int Index)>();
		var failures = new List<string>();
		int segments = 0;
		for (int i = 0; i < anchors.Count; i += 3)
		{
			for (int j = 1; j < anchors.Count; j += 7)
			{
				var a = anchors[i];
				var b = anchors[j];
				if (a.Equals(b))
				{
					continue;
				}

				++segments;
				dt.TraverseSegment(a, b, crossed);
				for (int k = 0; k < crossed.Count; ++k)
				{
					var (u, v, w) = dt.FacetVertices(crossed[k].Cell, crossed[k].Index);
					if (!SegmentTouchesClosedTriangle(a, b, dt.Point(u), dt.Point(v), dt.Point(w)))
					{
						failures.Add($"{a}->{b}: facet {k} not touched");
					}

					if (k > 0 && dt.Neighbor(crossed[k - 1].Cell, crossed[k - 1].Index) != crossed[k].Cell)
					{
						failures.Add($"{a}->{b}: chain broken at {k}");
					}
				}

				if (crossed.Count > 4 * dt.NumberOfCells)
				{
					failures.Add($"{a}->{b}: {crossed.Count} crossings");
				}
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(segments).IsGreaterThan(100);
			await Assert.That(failures).IsEmpty();
		}
	}

	[Test]
	public async Task CSharpOnly_TraversalMatchesAnIndependentOracle()
	{
		var random = new Random(77);
		var points = new Vector3d[400];
		for (int i = 0; i < points.Length; ++i)
		{
			points[i] = new Vector3d(random.NextDouble() * 2 - 1, random.NextDouble() * 2 - 1, random.NextDouble() * 2 - 1);
		}

		var dt = new DelaunayTriangulation3();
		dt.InsertRange(points);
		var index = dt.GetHullFacetIndex();
		var crossed = new List<(int Cell, int Index)>();
		var candidates = new List<int>();
		int compared = 0, mismatches = 0, missedCandidates = 0;
		for (int trial = 0; trial < 150; ++trial)
		{
			double scale = trial % 2 == 0 ? 5 : 1.5;
			var a = new Vector3d((random.NextDouble() - 0.5) * scale, (random.NextDouble() - 0.5) * scale, (random.NextDouble() - 0.5) * scale);
			var b = new Vector3d(random.NextDouble() - 0.5, random.NextDouble() - 0.5, random.NextDouble() - 0.5);

			var expected = new HashSet<(int, int, int)>();
			bool ambiguous = false;
			foreach (var (cell, i) in dt.FiniteFacets())
			{
				var (u, v, w) = dt.FacetVertices(cell, i);
				var hit = MollerTrumbore(a, b, dt.Point(u), dt.Point(v), dt.Point(w));
				ambiguous |= hit == Hit.Ambiguous;
				if (hit == Hit.Yes)
				{
					expected.Add(Sorted(u, v, w));
				}
			}

			index.QuerySegment(a, b, candidates);
			for (int f = 0; f < index.Facets.Count; ++f)
			{
				var (u, v, w) = dt.FacetVertices(index.Facets[f].Cell, index.Facets[f].Index);
				if (MollerTrumbore(a, b, dt.Point(u), dt.Point(v), dt.Point(w)) == Hit.Yes && candidates.BinarySearch(f) < 0)
				{
					++missedCandidates;
				}
			}

			if (ambiguous)
			{
				continue;
			}

			++compared;
			dt.TraverseSegment(a, b, crossed);
			var actual = crossed.Select(x => dt.FacetVertices(x.Cell, x.Index)).Select(t => Sorted(t.A, t.B, t.C)).ToList();
			if (actual.Count != expected.Count || !expected.SetEquals(actual))
			{
				++mismatches;
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(compared).IsGreaterThan(120);
			await Assert.That(mismatches).IsEqualTo(0);
			await Assert.That(missedCandidates).IsEqualTo(0);
		}
	}

	private enum Hit
	{
		No,
		Yes,
		Ambiguous,
	}

	// Moller and Trumbore, "Fast, minimum storage ray-triangle intersection", 1997, for the
	// open segment a->b; values within a margin of an edge or an endpoint are "ambiguous".
	private static Hit MollerTrumbore(Vector3d a, Vector3d b, Vector3d v0, Vector3d v1, Vector3d v2)
	{
		const double margin = 1e-9;
		double dx = b.X - a.X, dy = b.Y - a.Y, dz = b.Z - a.Z;
		double e1x = v1.X - v0.X, e1y = v1.Y - v0.Y, e1z = v1.Z - v0.Z;
		double e2x = v2.X - v0.X, e2y = v2.Y - v0.Y, e2z = v2.Z - v0.Z;
		double px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
		double det = e1x * px + e1y * py + e1z * pz;
		if (Math.Abs(det) < margin)
		{
			return Hit.No;
		}

		double tx = a.X - v0.X, ty = a.Y - v0.Y, tz = a.Z - v0.Z;
		double u = (tx * px + ty * py + tz * pz) / det;
		double qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
		double v = (dx * qx + dy * qy + dz * qz) / det;
		double t = (e2x * qx + e2y * qy + e2z * qz) / det;
		bool inside = u > margin && v > margin && u + v < 1 - margin && t > margin && t < 1 - margin;
		bool outside = u < -margin || v < -margin || u + v > 1 + margin || t < -margin || t > 1 + margin;
		return inside ? Hit.Yes : outside ? Hit.No : Hit.Ambiguous;
	}

	// Exact: the closed segment meets the closed triangle.
	private static bool SegmentTouchesClosedTriangle(Vector3d a, Vector3d b, Vector3d u, Vector3d v, Vector3d w)
	{
		int sa = RobustPredicates.Orient3D(u, v, w, a);
		int sb = RobustPredicates.Orient3D(u, v, w, b);
		if (sa * sb > 0)
		{
			return false;
		}

		if (sa == 0 && sb == 0)
		{
			// Coplanar: accept (the walk may slide along a facet plane in a grid).
			return true;
		}

		int s1 = RobustPredicates.Orient3D(a, b, u, v);
		int s2 = RobustPredicates.Orient3D(a, b, v, w);
		int s3 = RobustPredicates.Orient3D(a, b, w, u);
		return (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0);
	}

	private static (int, int, int) Sorted(int a, int b, int c)
	{
		Span<int> s = [a, b, c];
		s.Sort();
		return (s[0], s[1], s[2]);
	}

	private static List<Vector3d> Grid(int size)
	{
		var points = new List<Vector3d>();
		for (int x = 0; x < size; ++x)
		{
			for (int y = 0; y < size; ++y)
			{
				for (int z = 0; z < size; ++z)
				{
					points.Add(new Vector3d(x, y, z));
				}
			}
		}

		return points;
	}

	private static List<Vector3d> Plane(int size)
	{
		var points = new List<Vector3d>();
		for (int x = 0; x < size; ++x)
		{
			for (int y = 0; y < size; ++y)
			{
				points.Add(new Vector3d(x, y, 0));
			}
		}

		return points;
	}

	private static List<Vector3d> SpherePoints(int radiusSquared)
	{
		var points = new List<Vector3d>();
		int r = (int)Math.Ceiling(Math.Sqrt(radiusSquared));
		for (int x = -r; x <= r; ++x)
		{
			for (int y = -r; y <= r; ++y)
			{
				for (int z = -r; z <= r; ++z)
				{
					if (x * x + y * y + z * z == radiusSquared)
					{
						points.Add(new Vector3d(x, y, z));
					}
				}
			}
		}

		return points;
	}
}
