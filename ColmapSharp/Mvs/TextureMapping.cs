// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TextureMapping: texturing a triangle mesh from calibrated images (Waechter, Moehrle and
// Goesele, "Let there be color! Large-scale texturing of 3D reconstructions", ECCV 2014).
// Ports colmap/mvs/texture_mapping.h/.cc. This file holds the options, the result, the
// pipeline entry point, face normals, face adjacency and view selection; the atlas steps
// (regions, projections, packing, UVs, baking, inpainting) are in TextureMapping.Atlas.cs
// and the global color correction in TextureMapping.ColorCorrection.cs. Occlusion tests go
// through TriangleBvh.cs, which replaces COLMAP's CGAL AABB tree.
//
// Images are mvs::Image (Image.cs) with their bitmaps already set, as in COLMAP. The result
// is in memory: the atlas Bitmap, per-corner UVs and per-face view ids. COLMAP writes nothing
// to disk here either; its callers encode the atlas, which this library leaves to the host.
//
// Tier C for the pipeline as a whole, B for its float arithmetic: projections are float
// products whose last bits can differ from an Eigen/FMA build (divergence 63).
// Occlusion and tie orders are entries 90-92. Results are identical for any thread
// count: every parallel loop writes only its own slots.

using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>Port of colmap::mvs::MeshTextureMappingOptions.</summary>
public sealed class MeshTextureMappingOptions
{
	/// <summary>
	/// Minimum cosine of angle between face normal and view direction. Faces viewed at more
	/// grazing angles are rejected.
	/// </summary>
	public double MinCosNormalAngle { get; set; } = 0.1;

	/// <summary>
	/// Minimum number of face vertices that must project inside the image for a view to be
	/// considered a candidate (1, 2, or 3).
	/// </summary>
	public int MinVisibleVertices { get; set; } = 3;

	/// <summary>
	/// Number of neighbor-smoothing iterations for view selection. Reduces fragmentation by
	/// swapping face labels to match neighbors.
	/// </summary>
	public int ViewSelectionSmoothingIterations { get; set; } = 3;

	/// <summary>Padding in pixels between atlas patches.</summary>
	public int AtlasPatchPadding { get; set; } = 2;

	/// <summary>Number of pixels to dilate/inpaint around baked regions.</summary>
	public int InpaintRadius { get; set; } = 5;

	/// <summary>Whether to apply global color correction (Waechter et al. 2014).</summary>
	public bool ApplyColorCorrection { get; set; } = true;

	/// <summary>Regularization weight for the color correction system.</summary>
	public double ColorCorrectionRegularization { get; set; } = 0.1;

	/// <summary>Number of threads (-1 or 0: all available).</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>
	/// Scale factor for the texture atlas resolution. 1.0 = native source-image resolution
	/// (default); &lt; 1.0 = lower (e.g. 0.5 = half); &gt; 1.0 = higher (e.g. 2.0 = double).
	/// </summary>
	public double TextureScaleFactor { get; set; } = 1.0;

	/// <summary>Port of MeshTextureMappingOptions::Check (CHECK_OPTION_*: false on a violation).</summary>
	public bool Check() =>
		MinCosNormalAngle > 0.0
		&& MinCosNormalAngle <= 1.0
		&& MinVisibleVertices >= 1
		&& MinVisibleVertices <= 3
		&& ViewSelectionSmoothingIterations >= 0
		&& AtlasPatchPadding >= 0
		&& InpaintRadius >= 0
		&& ColorCorrectionRegularization > 0.0
		&& TextureScaleFactor > 0.0;
}

/// <summary>Port of colmap::mvs::MeshTextureMappingResult.</summary>
public sealed class MeshTextureMappingResult
{
	/// <summary>The texture atlas image (RGB); empty when nothing was textured.</summary>
	public Bitmap TextureAtlas { get; set; } = new();

	/// <summary>
	/// Per-face UV coordinates: 3 UVs per face (per-corner UVs).
	/// FaceUvs[face * 6 + corner * 2 + 0] = u, FaceUvs[face * 6 + corner * 2 + 1] = v.
	/// UV range [0, 1], (0,0) = bottom-left, (1,1) = top-right.
	/// </summary>
	public float[] FaceUvs { get; set; } = [];

	/// <summary>Per-face view assignment: index into the images list; -1 means not textured.</summary>
	public int[] FaceViewIds { get; set; } = [];

	/// <summary>Atlas width in pixels.</summary>
	public int AtlasWidth { get; set; }

	/// <summary>Atlas height in pixels.</summary>
	public int AtlasHeight { get; set; }
}

/// <summary>Port of colmap::mvs::MeshTextureMapping and its helpers (texture_mapping.cc).</summary>
public static partial class TextureMapping
{
	// Progress fractions reached at the end of each stage.
	private const double ProgressPrepared = 0.05;
	private const double ProgressViewsSelected = 0.7;
	private const double ProgressPacked = 0.75;
	private const double ProgressBaked = 0.9;
	private const double ProgressCorrected = 0.97;

	/// <summary>
	/// Produces a texture atlas with UV coordinates for a triangle mesh given calibrated
	/// multi-view images whose bitmaps are set. Port of colmap::mvs::MeshTextureMapping.
	/// <paramref name="progress"/> receives the completed fraction in [0, 1].
	/// </summary>
	public static MeshTextureMappingResult MeshTextureMapping(
		PlyMesh mesh,
		IReadOnlyList<Image> images,
		MeshTextureMappingOptions options,
		IProgress<double>? progress = null,
		CancellationToken cancellationToken = default)
	{
		Check.That(options.Check());

		var result = new MeshTextureMappingResult();
		int numFaces = mesh.Faces.Count;
		if (numFaces == 0 || mesh.Vertices.Count == 0)
		{
			result.FaceViewIds = new int[numFaces];
			Array.Fill(result.FaceViewIds, -1);
			result.FaceUvs = new float[numFaces * 6];
			return result;
		}

		var geometry = new MeshGeometry(mesh);
		float[] faceNormals = ComputeFaceNormals(geometry);
		int[] adjacency = BuildFaceAdjacency(geometry);
		progress?.Report(ProgressPrepared);
		cancellationToken.ThrowIfCancellationRequested();

		int[] viewPerFace = SelectViews(geometry, faceNormals, images, adjacency, options, progress, cancellationToken);
		result.FaceViewIds = viewPerFace;

		if (!viewPerFace.Any(v => v >= 0))
		{
			Log.Warning("No faces were assigned to any view");
			result.FaceUvs = new float[numFaces * 6];
			return result;
		}

		List<FaceRegion> regions = ExtractFaceRegions(viewPerFace, adjacency, numFaces);
		RegionProjection[] projections = ComputeRegionProjections(geometry, regions, images, options, cancellationToken);
		if (options.TextureScaleFactor != 1.0)
		{
			ScaleRegionProjections(projections, options.TextureScaleFactor);
		}

		AtlasLayout layout = PackAtlas(projections, options.AtlasPatchPadding);
		result.AtlasWidth = layout.AtlasWidth;
		result.AtlasHeight = layout.AtlasHeight;
		result.FaceUvs = ComputeFaceUVs(regions, projections, layout, numFaces);
		progress?.Report(ProgressPacked);
		cancellationToken.ThrowIfCancellationRequested();

		var atlas = new Bitmap(layout.AtlasWidth, layout.AtlasHeight, asRgb: true);
		atlas.Fill(new BitmapColor<byte>(0));
		result.TextureAtlas = atlas;
		bool[] bakedMask = BakeTexture(atlas, regions, projections, layout, images, options, cancellationToken);
		progress?.Report(ProgressBaked);

		if (options.ApplyColorCorrection)
		{
			ApplyGlobalColorCorrection(atlas, geometry, regions, projections, layout, images, adjacency, viewPerFace, bakedMask, options, cancellationToken);
			progress?.Report(ProgressCorrected);
		}

		if (options.InpaintRadius > 0)
		{
			cancellationToken.ThrowIfCancellationRequested();
			InpaintAtlas(atlas, bakedMask, options.InpaintRadius);
		}

		progress?.Report(1.0);
		return result;
	}

	/// <summary>
	/// The mesh as flat arrays: three floats per vertex and three vertex indices per face
	/// (COLMAP's GetVertex / GetFaceIndices), so the hot loops index arrays, not lists.
	/// </summary>
	internal sealed class MeshGeometry
	{
		public MeshGeometry(PlyMesh mesh)
		{
			Vertices = new float[3 * mesh.Vertices.Count];
			for (int i = 0; i < mesh.Vertices.Count; i++)
			{
				PlyMeshVertex v = mesh.Vertices[i];
				Vertices[3 * i] = v.X;
				Vertices[3 * i + 1] = v.Y;
				Vertices[3 * i + 2] = v.Z;
			}

			Faces = new int[3 * mesh.Faces.Count];
			for (int i = 0; i < mesh.Faces.Count; i++)
			{
				PlyMeshFace f = mesh.Faces[i];
				Faces[3 * i] = f.VertexIdx1;
				Faces[3 * i + 1] = f.VertexIdx2;
				Faces[3 * i + 2] = f.VertexIdx3;
			}
		}

		public float[] Vertices { get; }

		public int[] Faces { get; }

		public int NumFaces => Faces.Length / 3;
	}

	// Port of ProjectPoint: P (row-major 3x4) times (x, y, z, 1), dehomogenized.
	private static void ProjectPoint(ReadOnlySpan<float> p, float x, float y, float z, out float u, out float v)
	{
		float px = p[0] * x + p[1] * y + p[2] * z + p[3] * 1.0f;
		float py = p[4] * x + p[5] * y + p[6] * z + p[7] * 1.0f;
		float pz = p[8] * x + p[9] * y + p[10] * z + p[11] * 1.0f;
		u = px / pz;
		v = py / pz;
	}

	// Port of ProjectPointDepth: the third row of P times (x, y, z, 1).
	private static float ProjectPointDepth(ReadOnlySpan<float> p, float x, float y, float z) =>
		p[8] * x + p[9] * y + p[10] * z + p[11] * 1.0f;

	// Port of EdgeKey.
	private static ulong EdgeKey(int a, int b)
	{
		if (a > b)
		{
			(a, b) = (b, a);
		}

		return ((ulong)(uint)a << 32) | (uint)b;
	}

	/// <summary>Port of ComputeFaceNormals: unit normals, three floats per face (zero when degenerate).</summary>
	internal static float[] ComputeFaceNormals(MeshGeometry geometry)
	{
		float[] v = geometry.Vertices;
		int[] f = geometry.Faces;
		var normals = new float[f.Length];
		for (int i = 0; i < geometry.NumFaces; i++)
		{
			int a = 3 * f[3 * i], b = 3 * f[3 * i + 1], c = 3 * f[3 * i + 2];
			float e1x = v[b] - v[a], e1y = v[b + 1] - v[a + 1], e1z = v[b + 2] - v[a + 2];
			float e2x = v[c] - v[a], e2y = v[c + 1] - v[a + 1], e2z = v[c + 2] - v[a + 2];
			float nx = e1y * e2z - e1z * e2y;
			float ny = e1z * e2x - e1x * e2z;
			float nz = e1x * e2y - e1y * e2x;
			float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
			if (len > 1e-10f)
			{
				normals[3 * i] = nx / len;
				normals[3 * i + 1] = ny / len;
				normals[3 * i + 2] = nz / len;
			}
		}

		return normals;
	}

	/// <summary>
	/// Port of BuildFaceAdjacency: faces sharing an edge that exactly two faces use. Three
	/// slots per face, sorted ascending, -1 for unused slots (a face has at most three such
	/// neighbors). COLMAP groups edges in a hash map; here the (edge, face) pairs are sorted,
	/// which groups them the same way, and the per-face lists are sorted in both, so the
	/// result is identical.
	/// </summary>
	internal static int[] BuildFaceAdjacency(MeshGeometry geometry)
	{
		int numFaces = geometry.NumFaces;
		int[] f = geometry.Faces;
		var keys = new ulong[3 * numFaces];
		var owners = new int[3 * numFaces];
		for (int fi = 0; fi < numFaces; fi++)
		{
			for (int e = 0; e < 3; e++)
			{
				keys[3 * fi + e] = EdgeKey(f[3 * fi + e], f[3 * fi + (e + 1) % 3]);
				owners[3 * fi + e] = fi;
			}
		}

		// Owners are ascending before the sort; Array.Sort is not stable, so the face index
		// keeps each group's first entry the lower face, as COLMAP's push_back order does.
		var order = new int[keys.Length];
		for (int i = 0; i < order.Length; i++)
		{
			order[i] = i;
		}

		Array.Sort(order, (x, y) =>
		{
			int byKey = keys[x].CompareTo(keys[y]);
			return byKey != 0 ? byKey : x.CompareTo(y);
		});

		var adjacency = new int[3 * numFaces];
		Array.Fill(adjacency, -1);
		for (int start = 0; start < order.Length;)
		{
			int end = start + 1;
			while (end < order.Length && keys[order[end]] == keys[order[start]])
			{
				end++;
			}

			if (end - start == 2)
			{
				int fa = owners[order[start]];
				int fb = owners[order[start + 1]];
				AddNeighbor(adjacency, fa, fb);
				AddNeighbor(adjacency, fb, fa);
			}

			start = end;
		}

		for (int fi = 0; fi < numFaces; fi++)
		{
			SortSlots(adjacency.AsSpan(3 * fi, 3));
		}

		return adjacency;
	}

	// Adds a neighbor unless already present (COLMAP's sort + unique removes duplicates).
	private static void AddNeighbor(int[] adjacency, int face, int neighbor)
	{
		for (int s = 3 * face; s < 3 * face + 3; s++)
		{
			if (adjacency[s] == neighbor)
			{
				return;
			}

			if (adjacency[s] < 0)
			{
				adjacency[s] = neighbor;
				return;
			}
		}
	}

	// Sorts the used (non-negative) slots ascending, leaving -1 at the end.
	private static void SortSlots(Span<int> slots)
	{
		for (int i = 1; i < 3; i++)
		{
			int x = slots[i];
			int j = i - 1;
			while (j >= 0 && Later(slots[j], x))
			{
				slots[j + 1] = slots[j];
				j--;
			}

			slots[j + 1] = x;
		}

		static bool Later(int a, int b) => (a < 0 ? int.MaxValue : a) > (b < 0 ? int.MaxValue : b);
	}
}
