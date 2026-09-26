// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DelaunayValidator: test-side checks for DelaunayTriangulation3 (C#-only helper for
// DelaunayTriangulation3Tests). They inspect the production triangulation through its
// public API and decide with the production RobustPredicates, so a pass is exact.

using ColmapSharp.Geometry.Delaunay;
using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Tests.Geometry.Delaunay;

internal static class DelaunayValidator
{
	/// <summary>
	/// Combinatorial and orientation errors: neighbor symmetry, shared facets, one infinite
	/// vertex per infinite cell, strictly positive finite cells, a convex hull, and incident
	/// cells that really contain their vertex.
	/// </summary>
	public static List<string> StructureErrors(DelaunayTriangulation3 dt)
	{
		var errors = new List<string>();
		foreach (int c in dt.AllCells())
		{
			int infinite = 0;
			for (int i = 0; i < 4; ++i)
			{
				if (dt.Vertex(c, i) == DelaunayTriangulation3.InfiniteVertex)
				{
					++infinite;
				}

				int n = dt.Neighbor(c, i);
				if (!dt.IsCellAlive(n))
				{
					errors.Add($"cell {c} neighbor {i} is dead");
					continue;
				}

				int back = Array.IndexOf(Enumerable.Range(0, 4).Select(j => dt.Neighbor(n, j)).ToArray(), c);
				if (back < 0)
				{
					errors.Add($"cell {c} neighbor {i} does not point back");
					continue;
				}

				// The two cells share exactly the three vertices of the facet.
				for (int j = 0; j < 4; ++j)
				{
					if (j != i && dt.IndexOfVertex(n, dt.Vertex(c, j)) < 0)
					{
						errors.Add($"cell {c} facet {i} not shared with {n}");
					}
				}

				if (dt.IndexOfVertex(c, dt.Vertex(n, back)) >= 0)
				{
					errors.Add($"cells {c} and {n} share all vertices");
				}
			}

			if (infinite > 1)
			{
				errors.Add($"cell {c} has {infinite} infinite vertices");
			}

			int k = dt.InfiniteIndex(c);
			if (k < 0)
			{
				if (RobustPredicates.Orient3D(P(dt, c, 0), P(dt, c, 1), P(dt, c, 2), P(dt, c, 3)) <= 0)
				{
					errors.Add($"finite cell {c} is not positively oriented");
				}
			}
			else
			{
				// Convex hull: the finite neighbor's opposite vertex is strictly inside, and no
				// neighboring hull facet's far vertex is strictly outside this one.
				for (int i = 0; i < 4; ++i)
				{
					int n = dt.Neighbor(c, i);
					int far = dt.Vertex(n, dt.IndexOfNeighbor(n, c));
					if (far == DelaunayTriangulation3.InfiniteVertex)
					{
						continue;
					}

					if (dt.OrientReplaced(c, k, dt.Point(far)) >= 0 && i == k)
					{
						errors.Add($"hull facet of {c} is not strictly visible from inside");
					}
					else if (i != k && dt.OrientReplaced(c, k, dt.Point(far)) > 0)
					{
						errors.Add($"hull is not convex at cell {c}");
					}
				}
			}
		}

		for (int v = 1; v < dt.VertexCapacity; ++v)
		{
			int cell = dt.IncidentCell(v);
			if (!dt.IsCellAlive(cell) || dt.IndexOfVertex(cell, v) < 0)
			{
				errors.Add($"vertex {v} has a bad incident cell");
			}
		}

		return errors;
	}

	/// <summary>Brute force: (cell, vertex) pairs with a vertex strictly inside a finite cell's circumsphere.</summary>
	public static int GlobalDelaunayViolations(DelaunayTriangulation3 dt)
	{
		int violations = 0;
		foreach (int c in dt.FiniteCells())
		{
			for (int v = 1; v < dt.VertexCapacity; ++v)
			{
				if (RobustPredicates.InSphere(P(dt, c, 0), P(dt, c, 1), P(dt, c, 2), P(dt, c, 3), dt.Point(v)) > 0)
				{
					++violations;
				}
			}
		}

		return violations;
	}

	/// <summary>Finite facets whose far vertex across is strictly inside the cell's circumsphere.</summary>
	public static int LocalDelaunayViolations(DelaunayTriangulation3 dt)
	{
		int violations = 0;
		foreach (int c in dt.FiniteCells())
		{
			for (int i = 0; i < 4; ++i)
			{
				int n = dt.Neighbor(c, i);
				int far = dt.Vertex(n, dt.IndexOfNeighbor(n, c));
				if (far != DelaunayTriangulation3.InfiniteVertex
					&& RobustPredicates.InSphere(P(dt, c, 0), P(dt, c, 1), P(dt, c, 2), P(dt, c, 3), dt.Point(far)) > 0)
				{
					++violations;
				}
			}
		}

		return violations;
	}

	/// <summary>Total volume of the finite cells.</summary>
	public static double FiniteVolume(DelaunayTriangulation3 dt)
	{
		double volume = 0;
		foreach (int c in dt.FiniteCells())
		{
			Vector3d a = P(dt, c, 0), b = P(dt, c, 1), cc = P(dt, c, 2), d = P(dt, c, 3);
			double bx = b.X - a.X, by = b.Y - a.Y, bz = b.Z - a.Z;
			double cx = cc.X - a.X, cy = cc.Y - a.Y, cz = cc.Z - a.Z;
			double dx = d.X - a.X, dy = d.Y - a.Y, dz = d.Z - a.Z;
			volume += (bx * (cy * dz - cz * dy) - by * (cx * dz - cz * dx) + bz * (cx * dy - cy * dx)) / 6;
		}

		return volume;
	}

	/// <summary>Number of distinct finite vertices referenced by live cells.</summary>
	public static int UsedVertexCount(DelaunayTriangulation3 dt)
	{
		var used = new HashSet<int>();
		foreach (int c in dt.AllCells())
		{
			for (int i = 0; i < 4; ++i)
			{
				if (dt.Vertex(c, i) != DelaunayTriangulation3.InfiniteVertex)
				{
					used.Add(dt.Vertex(c, i));
				}
			}
		}

		return used.Count;
	}

	/// <summary>The exact cell arrays, for determinism checks.</summary>
	public static string Fingerprint(DelaunayTriangulation3 dt)
	{
		var text = new System.Text.StringBuilder();
		foreach (int c in dt.AllCells())
		{
			text.Append(c).Append(':');
			for (int i = 0; i < 4; ++i)
			{
				text.Append(dt.Vertex(c, i)).Append(',').Append(dt.Neighbor(c, i)).Append(';');
			}
		}

		return text.ToString();
	}

	/// <summary>Finite cells as sorted vertex positions, independent of handles.</summary>
	public static List<string> CellSet(DelaunayTriangulation3 dt)
	{
		return dt.FiniteCells()
			.Select(c => string.Join("|", Enumerable.Range(0, 4).Select(i => dt.Point(dt.Vertex(c, i)).ToString()).Order(StringComparer.Ordinal)))
			.ToList();
	}

	private static Vector3d P(DelaunayTriangulation3 dt, int cell, int i) => dt.Point(dt.Vertex(cell, i));
}
