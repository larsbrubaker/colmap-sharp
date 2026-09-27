// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// MinimalAreaTriangulation: splits a closed polygon (a loop of 3D points) into triangles whose
// total area is smallest, by dynamic programming over the sub-polygons [v_i, ..., v_j]. Ports
// MAT.h / MAT.inl (_MinimalAreaTriangulation: getTriangulation, _set, _subPolygonArea,
// _addTriangles, _subPolygonIndex) and Geometry.h's SquareArea. The level-set extractor's
// polygon step (PoissonLevelSetExtractor.Polygons.cs, AddIsoPolygons) calls it for the loops
// with more than three vertices that it does not fan around a barycenter. Tier A against
// oracle/poisson_levelset6_harness.cc (through the extractor's triangles).
//
// Translation notes:
// - Real is float, as upstream instantiates it; the areas are float sums in upstream's order,
//   and sqrt of a float is correctly rounded on both sides.
// - The four-vertex case keeps upstream's comparison: its loop assigns (not adds) each
//   triangle's squared area, so each split is judged by its second triangle alone.
// - SquareArea's p2 - p1 is upstream's p2 + (-p1), the same IEEE result.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Minimal-area triangulation of a polygon. Port of PoissonRecon's
/// <c>MinimalAreaTriangulation&lt; Index , Real , Dim &gt;</c> (MAT.h) with Real = float, Dim = 3.
/// </summary>
public sealed class MinimalAreaTriangulation
{
	private readonly float[] vertices;
	private readonly int vCount;
	private float[] bestTriangulation = [];
	private int[] midpoint = [];

	private MinimalAreaTriangulation(float[] vertices, int vCount)
	{
		this.vertices = vertices;
		this.vCount = vCount;
	}

	/// <summary>
	/// The triangles (vertex index triples into the polygon) of the minimal-area triangulation
	/// of the polygon whose <paramref name="vCount"/> vertices are
	/// <paramref name="vertices"/> (x, y, z per vertex), in upstream's order. Port of
	/// <c>MinimalAreaTriangulation( vertices , vCount )</c>.
	/// </summary>
	public static List<(int A, int B, int C)> Triangulate(float[] vertices, int vCount) =>
		new MinimalAreaTriangulation(vertices, vCount).GetTriangulation();

	/// <summary>
	/// The squared area of the triangle (p1, p2, p3), each an index into the x, y, z array
	/// <paramref name="xyz"/>. Port of Geometry.h's <c>SquareArea</c> for float points in 3D.
	/// </summary>
	public static float SquareArea(float[] xyz, int p1, int p2, int p3)
	{
		// Area^2 = ( |v1|^2 * |v2|^2 - < v1 , v2 >^2 ) / 4
		float v1x = xyz[(3 * p2) + 0] - xyz[(3 * p1) + 0];
		float v1y = xyz[(3 * p2) + 1] - xyz[(3 * p1) + 1];
		float v1z = xyz[(3 * p2) + 2] - xyz[(3 * p1) + 2];
		float v2x = xyz[(3 * p3) + 0] - xyz[(3 * p1) + 0];
		float v2y = xyz[(3 * p3) + 1] - xyz[(3 * p1) + 1];
		float v2z = xyz[(3 * p3) + 2] - xyz[(3 * p1) + 2];
		float dot = 0f;
		dot += v1x * v2x;
		dot += v1y * v2y;
		dot += v1z * v2z;
		float l1 = 0f;
		l1 += v1x * v1x;
		l1 += v1y * v1y;
		l1 += v1z * v1z;
		float l2 = 0f;
		l2 += v2x * v2x;
		l2 += v2y * v2y;
		l2 += v2z * v2z;
		return ((l1 * l2) - (dot * dot)) / 4;
	}

	// getTriangulation.
	private List<(int A, int B, int C)> GetTriangulation()
	{
		var triangles = new List<(int A, int B, int C)>();
		if (vCount == 3)
		{
			triangles.Add((0, 1, 2));
			return triangles;
		}

		if (vCount == 4)
		{
			(int, int, int)[][] tIndex =
			[
				[(0, 1, 2), (2, 3, 0)],
				[(0, 1, 3), (3, 1, 2)],
			];
			float[] area = [0, 0];

			// Upstream assigns rather than accumulates here (see the header).
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					area[i] = SquareArea(vertices, tIndex[i][j].Item1, tIndex[i][j].Item2, tIndex[i][j].Item3);
				}
			}

			if (area[0] > area[1])
			{
				triangles.Add(tIndex[1][0]);
				triangles.Add(tIndex[1][1]);
			}
			else
			{
				triangles.Add(tIndex[0][0]);
				triangles.Add(tIndex[0][1]);
			}

			return triangles;
		}

		Set();
		AddTriangles(1, 0, triangles);
		return triangles;
	}

	// _set.
	private void Set()
	{
		bestTriangulation = new float[vCount * vCount];
		midpoint = new int[vCount * vCount];
		Array.Fill(bestTriangulation, -1f);
		Array.Fill(midpoint, -1);
		SubPolygonArea(1, 0);
	}

	// _subPolygonIndex.
	private int SubPolygonIndex(int i, int j) => (i * vCount) + j;

	// _addTriangles: the triangles of the best triangulation of sub-polygon [ v_i , ... , v_j ].
	private void AddTriangles(int i, int j, List<(int A, int B, int C)> triangles)
	{
		if (j < i)
		{
			j += vCount;
		}

		if (i == j || i + 1 == j)
		{
			return;
		}

		int mid = midpoint[SubPolygonIndex(i, j % vCount)];
		if (mid != -1)
		{
			triangles.Add((i, mid, j % vCount));
			AddTriangles(i, mid, triangles);
			AddTriangles(mid, j, triangles);
		}
	}

	// _subPolygonArea: the minimal area of the sub-polygon [ v_i , ... , v_j ].
	private float SubPolygonArea(int i, int j)
	{
		int idx = SubPolygonIndex(i, j);
		if (midpoint[idx] != -1)
		{
			return bestTriangulation[idx];
		}

		float a = float.MaxValue;
		float temp;
		if (j < i)
		{
			j += vCount;
		}

		// If either i==j or i+1=j, the polygon has trivial area
		if (i == j || i + 1 == j)
		{
			bestTriangulation[idx] = 0;
			return 0;
		}

		int mid = -1;

		// For each vertex r \in( i , j ):
		// -- Construct the triangle ( j , r , i )
		// -- Compute the Area(j,r,i) + Area( j , ... , r ) + Area( r , ... , i )
		for (int r = i + 1; r < j; r++)
		{
			int idx1 = SubPolygonIndex(i, r % vCount); // SubPolygon( r , ... , i )
			temp = SquareArea(vertices, i, r % vCount, j % vCount);
			temp = temp < 0 ? 0 : MathF.Sqrt(temp);

			// If we have already computed Area( r , ... , i ), use that.
			if (bestTriangulation[idx1] >= 0)
			{
				temp += bestTriangulation[idx1];

				// If the partial area is already too large, terminate
				if (temp > a)
				{
					continue;
				}

				// Otherwise, compute the total area
				temp += SubPolygonArea(r % vCount, j % vCount);
			}
			else
			{
				// Otherwise, compute it now
				temp += SubPolygonArea(r % vCount, j % vCount);

				// If the partial area is already too large, terminate
				if (temp > a)
				{
					continue;
				}

				// Otherwise, compute the total area
				temp += SubPolygonArea(i, r % vCount);
			}

			if (temp < a)
			{
				a = temp;
				mid = r % vCount;
			}
		}

		bestTriangulation[idx] = a;
		midpoint[idx] = mid;
		return a;
	}
}
