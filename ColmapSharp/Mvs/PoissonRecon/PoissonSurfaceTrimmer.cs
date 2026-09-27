// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonSurfaceTrimmer: COLMAP's mesh trimming step. Ports SurfaceTrimmer.cpp's Execute
// (SplitPolygon, InterpolateVertices, Triangulate, RemoveHangingVertices) for exactly the
// arguments src/colmap/mvs/poisson_meshing.cc passes to RunSurfaceTrimmer: --in, --out and
// --trim std::to_string( options.trim ), nothing else. So Real is float and Index int, --aRatio
// keeps its 0.001 default (the island merge always runs; PoissonSurfaceTrimmer.Islands.cs),
// --removeIslands is off (small islands with no neighbor are kept), and --polygonMesh is off
// (the kept polygons are triangulated with MinimalAreaTriangulation). Its input is the
// PoissonRecon output (PoissonMeshOutput.FromLevelSet); upstream reads it back from the PLY
// PoissonRecon wrote, so the float positions and density values arrive unchanged and the
// uchar colors arrive as floats (DynamicFactory< float >), and it writes the kept vertices
// back with the colors converted to uchar again (PoissonMeshOutput.PlyUChar). Tier A against
// oracle/poisson_trim_harness.cc, which runs upstream's RunSurfaceTrimmer through PLY files.
//
// Translation notes:
// - The vertex table SplitPolygon shares across polygons, SetConnectedComponents' edge table
//   and RemoveHangingVertices' map are only looked up, never iterated, so .NET dictionaries
//   stand in for them; the containers the island merge iterates are TrimmerEdgeHashTable.
// - Upstream triangulates the below-trim polygons too and then discards them; that has no
//   effect on the output, so it is skipped here.
// - COLMAP's trim is formatted by std::to_string (six decimals) and parsed by atof into a
//   float; callers pass the float.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Removes the low-density part of a Poisson surface as COLMAP's poisson_meshing does
/// (PoissonRecon's SurfaceTrimmer with <c>--trim</c> only).
/// </summary>
public static partial class PoissonSurfaceTrimmer
{
	/// <summary>SurfaceTrimmer's <c>--aRatio</c> default, which COLMAP does not override.</summary>
	public const float IslandAreaRatio = 0.001f;

	/// <summary>
	/// The part of <paramref name="mesh"/> whose density value exceeds
	/// <paramref name="trim"/>: polygons are split where the value crosses the trim, small
	/// components merge into their neighbors (<see cref="IslandAreaRatio"/>), the kept
	/// polygons are triangulated and unused vertices dropped. The output keeps the density
	/// values and colors. Port of <c>RunSurfaceTrimmer( "--in" , ... , "--trim" , trim )</c>.
	/// </summary>
	public static PoissonMeshOutput Trim(PoissonMeshOutput mesh, float trim, CancellationToken cancellationToken = default, IProgress<double>? progress = null)
	{
		ArgumentNullException.ThrowIfNull(mesh);
		if (mesh.Values == null)
		{
			// Upstream: MK_THROW( "Ply file does not contain values" ).
			throw new ArgumentException("The mesh has no density values to trim by (extract it with a density estimator).", nameof(mesh));
		}

		// Upstream first seeds a --verbose-only value range with vertices[0], which on a mesh with
		// no vertices reads through the empty vector's null data pointer and crashes. The range is
		// not ported (it only feeds a log line), so an empty mesh trims to an empty mesh here
		// (docs/CPP_DIVERGENCES.md, entry 132).
		var vertices = new TrimVertices(mesh);
		var ltPolygons = new List<int[]>();
		var gtPolygons = new List<int[]>();
		var vertexTable = new Dictionary<(int, int), int>();
		int[] triangle = new int[3];
		for (int t = 0; t < mesh.Triangles.Length / 3; t++)
		{
			if ((t & 4095) == 0)
			{
				cancellationToken.ThrowIfCancellationRequested();
			}

			Array.Copy(mesh.Triangles, 3 * t, triangle, 0, 3);
			SplitPolygon(triangle, vertices, ltPolygons, gtPolygons, vertexTable, trim);
		}

		progress?.Report(0.25);
		gtPolygons = MergeIslands(vertices, ltPolygons, gtPolygons, cancellationToken, progress);
		cancellationToken.ThrowIfCancellationRequested();
		List<int[]> triangles = Triangulate(vertices, gtPolygons);
		PoissonMeshOutput output = RemoveHangingVertices(vertices, triangles);
		progress?.Report(1.0);
		return output;
	}

	// SplitPolygon: the polygon goes whole to one side when all its values are on it;
	// otherwise it is cut at every edge whose values straddle the trim, the new vertex on each
	// such edge shared (vertexTable) with the polygon across it.
	private static void SplitPolygon(int[] polygon, TrimVertices vertices, List<int[]> ltPolygons, List<int[]> gtPolygons, Dictionary<(int, int), int> vertexTable, float trimValue)
	{
		int sz = polygon.Length;
		Span<bool> gt = stackalloc bool[sz];
		int gtCount = 0;
		for (int j = 0; j < sz; j++)
		{
			gt[j] = vertices.Value(polygon[j]) > trimValue;
			if (gt[j])
			{
				gtCount++;
			}
		}

		if (gtCount == sz)
		{
			gtPolygons.Add((int[])polygon.Clone());
			return;
		}

		if (gtCount == 0)
		{
			ltPolygons.Add((int[])polygon.Clone());
			return;
		}

		int start;
		for (start = 0; start < sz; start++)
		{
			if (gt[start] && !gt[(start + sz - 1) % sz])
			{
				break;
			}
		}

		bool gtFlag = true;
		var poly = new List<int>();

		// Add the initial vertex
		poly.Add(EdgeVertex(polygon[(start + sz - 1) % sz], polygon[start], vertices, vertexTable, trimValue));
		for (int j = 0; j <= sz; j++)
		{
			int v1 = polygon[(j + start + sz - 1) % sz];
			int j2 = (j + start) % sz;
			int v2 = polygon[j2];
			if (gt[j2] == gtFlag)
			{
				poly.Add(v2);
			}
			else
			{
				int vIdx = EdgeVertex(v1, v2, vertices, vertexTable, trimValue);
				poly.Add(vIdx);
				(gtFlag ? gtPolygons : ltPolygons).Add([.. poly]);
				poly.Clear();
				poly.Add(vIdx);
				poly.Add(v2);
				gtFlag = !gtFlag;
			}
		}
	}

	// The vertex where the trim crosses edge (v1, v2), made on first use. Port of the
	// EdgeKey lookup and InterpolateVertices call in SplitPolygon.
	private static int EdgeVertex(int v1, int v2, TrimVertices vertices, Dictionary<(int, int), int> vertexTable, float trimValue)
	{
		(int, int) key = v1 < v2 ? (v1, v2) : (v2, v1);
		if (vertexTable.TryGetValue(key, out int vIdx))
		{
			return vIdx;
		}

		vIdx = vertices.Count;
		vertexTable[key] = vIdx;
		vertices.AddInterpolated(v1, v2, trimValue);
		return vIdx;
	}

	// Triangulate: triangles stay, larger polygons get their minimal-area triangulation and
	// smaller ones are dropped.
	private static List<int[]> Triangulate(TrimVertices vertices, List<int[]> polygons)
	{
		var triangles = new List<int[]>();
		foreach (int[] polygon in polygons)
		{
			if (polygon.Length > 3)
			{
				float[] points = new float[3 * polygon.Length];
				for (int j = 0; j < polygon.Length; j++)
				{
					vertices.CopyPosition(polygon[j], points.AsSpan(3 * j, 3));
				}

				foreach ((int a, int b, int c) in MinimalAreaTriangulation.Triangulate(points, polygon.Length))
				{
					triangles.Add([polygon[a], polygon[b], polygon[c]]);
				}
			}
			else if (polygon.Length == 3)
			{
				triangles.Add(polygon);
			}
		}

		return triangles;
	}

	// RemoveHangingVertices, then the PLY write: the vertices some triangle uses, in index
	// order, with the colors back in uchar.
	private static PoissonMeshOutput RemoveHangingVertices(TrimVertices vertices, List<int[]> triangles)
	{
		int[] map = new int[vertices.Count];
		foreach (int[] t in triangles)
		{
			foreach (int v in t)
			{
				map[v] = 1;
			}
		}

		int vCount = 0;
		for (int i = 0; i < map.Length; i++)
		{
			map[i] = map[i] != 0 ? vCount++ : -1;
		}

		int channels = vertices.Channels;
		float[] positions = new float[3 * vCount];
		float[] values = new float[vCount];
		byte[]? colors = channels > 0 ? new byte[channels * vCount] : null;
		for (int i = 0; i < map.Length; i++)
		{
			int o = map[i];
			if (o < 0)
			{
				continue;
			}

			vertices.CopyPosition(i, positions.AsSpan(3 * o, 3));
			values[o] = vertices.Value(i);
			for (int c = 0; c < channels; c++)
			{
				colors![(channels * o) + c] = PoissonMeshOutput.PlyUChar(vertices.Color(i, c));
			}
		}

		int[] indices = new int[3 * triangles.Count];
		for (int t = 0; t < triangles.Count; t++)
		{
			for (int k = 0; k < 3; k++)
			{
				indices[(3 * t) + k] = map[triangles[t][k]];
			}
		}

		return new PoissonMeshOutput(vCount, channels, positions, values, colors, indices);
	}

	/// <summary>
	/// The trimmer's vertices as upstream holds them: DirectSum&lt; float , Point3 , float ,
	/// Point&lt; float &gt; &gt; (position, density value, the input's extra properties as floats),
	/// growing as SplitPolygon adds edge vertices.
	/// </summary>
	private sealed class TrimVertices
	{
		private readonly List<float> positions;
		private readonly List<float> values;
		private readonly List<float> colors;

		public TrimVertices(PoissonMeshOutput mesh)
		{
			Channels = mesh.ColorChannels;
			positions = new List<float>(mesh.Positions);
			values = new List<float>(mesh.Values!);
			colors = new List<float>(mesh.VertexCount * Channels);
			if (mesh.Colors != null)
			{
				foreach (byte b in mesh.Colors)
				{
					colors.Add(b);
				}
			}
		}

		public int Channels { get; }

		public int Count => values.Count;

		public float Value(int v) => values[v];

		public float Color(int v, int c) => colors[(Channels * v) + c];

		public void CopyPosition(int v, Span<float> xyz)
		{
			xyz[0] = positions[3 * v];
			xyz[1] = positions[(3 * v) + 1];
			xyz[2] = positions[(3 * v) + 2];
		}

		// InterpolateVertices( v1 , v2 , value ): the midpoint when the two values are equal,
		// else v1 * (float)( 1. - dx ) + v2 * dx with dx the float fraction along the edge,
		// every channel (the value itself included) in float.
		public void AddInterpolated(int v1, int v2, float value)
		{
			float a = values[v1];
			float b = values[v2];
			if (a == b)
			{
				for (int d = 0; d < 3; d++)
				{
					positions.Add((positions[(3 * v1) + d] + positions[(3 * v2) + d]) / 2f);
				}

				values.Add((a + b) / 2f);
				for (int c = 0; c < Channels; c++)
				{
					colors.Add((colors[(Channels * v1) + c] + colors[(Channels * v2) + c]) / 2f);
				}

				return;
			}

			float dx = (a - value) / (a - b);
			float s = (float)(1.0 - dx);
			for (int d = 0; d < 3; d++)
			{
				positions.Add((positions[(3 * v1) + d] * s) + (positions[(3 * v2) + d] * dx));
			}

			values.Add((a * s) + (b * dx));
			for (int c = 0; c < Channels; c++)
			{
				colors.Add((colors[(Channels * v1) + c] * s) + (colors[(Channels * v2) + c] * dx));
			}
		}
	}
}
