// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchSyntheticScene: the synthetic PatchMatch problem the PatchMatch tests share - a
// textured slanted plane seen by a reference and two source cameras, rendered analytically, so
// the true depth and normal of every pixel are known. Not a COLMAP port (COLMAP tests PatchMatch
// only through its CUDA pipeline). It lives in the library, not the test assembly, so that
// PatchMatchGpuConformance.cs (a host's check of its IComputeDevice) and the test suite
// (PatchMatchRunTests, PatchMatchGpuTwinTests, PatchMatchSweepBandTests, ...) run one copy of
// the scene. The truth measures (FractionWithin, MedianTrueNormalCosine) and the consistency
// list decoder are here for the same reason.

using ColmapSharp.Sensor;

namespace ColmapSharp.Mvs.Testing;

/// <summary>
/// A slanted textured plane, z = <see cref="PlaneZ"/> + <see cref="Slope"/> * x, seen by three
/// cameras looking down +z, with the options and problem the PatchMatch tests run on it.
/// </summary>
public static class PatchMatchSyntheticScene
{
	/// <summary>The focal length of every camera, in pixels.</summary>
	public const float Fx = 40;

	/// <summary>The plane's slope along x.</summary>
	public const float Slope = 0.3f;

	/// <summary>The plane's depth at x = 0.</summary>
	public const float PlaneZ = 4.0f;

	private static readonly float[] Identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];

	// Camera centres; every camera looks down +z with R = I, so T = -C.
	private static readonly float[][] Centers = [[0, 0, 0], [0.5f, 0, 0], [-0.4f, 0.3f, 0]];

	// The plane z = PlaneZ + Slope * X.
	private static double RayDepth(double cx, double cy, double cz, double dx, double dy) =>
		(PlaneZ + Slope * cx - cz) / (1 - Slope * dx);

	private static double Texture(double x, double y) =>
		0.5 + 0.2 * Math.Sin(9 * x + 1) * Math.Cos(8 * y) + 0.15 * Math.Sin(23 * x - 17 * y) + 0.1 * Math.Cos(31 * y + 5 * x);

	/// <summary>
	/// The three images of the scene at <paramref name="width"/> x <paramref name="height"/>
	/// (image 0 is the reference), with each image's true depth and normal map.
	/// </summary>
	public static (List<Image> Images, List<DepthMap> DepthMaps, List<NormalMap> NormalMaps) Scene(int width, int height)
	{
		float cx = width / 2.0f - 0.5f;
		float cy = height / 2.0f - 0.5f;
		float[] k = [Fx, 0, cx, 0, Fx, cy, 0, 0, 1];
		var images = new List<Image>();
		var depthMaps = new List<DepthMap>();
		var normalMaps = new List<NormalMap>();
		float normalNorm = MathF.Sqrt(Slope * Slope + 1);
		foreach (float[] c in Centers)
		{
			var image = new Image("img", width, height, k, Identity, [-c[0], -c[1], -c[2]]);
			var bitmap = new Bitmap(width, height, asRgb: false);
			var depthMap = new DepthMap(width, height, 0, 10);
			var normalMap = new NormalMap(width, height);
			for (int v = 0; v < height; ++v)
			{
				for (int u = 0; u < width; ++u)
				{
					double dx = (u - cx) / Fx;
					double dy = (v - cy) / Fx;
					double t = RayDepth(c[0], c[1], c[2], dx, dy);
					double x = c[0] + t * dx;
					double y = c[1] + t * dy;
					bitmap.RowMajorData[v * width + u] = (byte)Math.Round(20 + 200 * Texture(x, y));
					depthMap.Set(v, u, (float)t);
					normalMap.Set(v, u, 0, Slope / normalNorm);
					normalMap.Set(v, u, 1, 0);
					normalMap.Set(v, u, 2, -1 / normalNorm);
				}
			}

			image.SetBitmap(bitmap);
			images.Add(image);
			depthMaps.Add(depthMap);
			normalMaps.Add(normalMap);
		}

		return (images, depthMaps, normalMaps);
	}

	/// <summary>The PatchMatch options the tests run the scene with (depth range 2..8, a 7x7 window).</summary>
	public static PatchMatchOptions Options(int numIterations, bool geomConsistency, bool filter, int numThreads = -1) => new()
	{
		DepthMin = 2,
		DepthMax = 8,
		WindowRadius = 3,
		SigmaSpatial = 3,
		NumIterations = numIterations,
		GeomConsistency = geomConsistency,
		Filter = filter,
		FilterMinNumConsistent = 1,
		NumThreads = numThreads,
	};

	/// <summary>
	/// Image 0 against sources 1 and 2. A geometric problem needs every image's depth and normal
	/// map (the truth from <see cref="Scene"/>); a photometric one passes none.
	/// </summary>
	public static PatchMatch.Problem Problem(List<Image> images, List<DepthMap>? depthMaps = null, List<NormalMap>? normalMaps = null) => new()
	{
		RefImageIdx = 0,
		SrcImageIdxs = [1, 2],
		Images = images,
		DepthMaps = depthMaps ?? [],
		NormalMaps = normalMaps ?? [],
	};

	/// <summary>
	/// The fraction of pixels at least <paramref name="border"/> from the edge whose depth is
	/// within <paramref name="tolerance"/> (relative) of the truth.
	/// </summary>
	public static double FractionWithin(DepthMap estimate, DepthMap truth, float tolerance, int border)
	{
		int good = 0;
		int total = 0;
		for (int row = border; row < truth.GetHeight() - border; ++row)
		{
			for (int col = border; col < truth.GetWidth() - border; ++col)
			{
				total++;
				if (MathF.Abs(estimate.Get(row, col) - truth.Get(row, col)) <= tolerance * truth.Get(row, col))
				{
					good++;
				}
			}
		}

		return good / (double)total;
	}

	/// <summary>
	/// The median, over pixels at least <paramref name="border"/> from the edge, of the cosine
	/// between the estimated normal and the plane's true normal.
	/// </summary>
	public static float MedianTrueNormalCosine(NormalMap normalMap, int border)
	{
		float normalNorm = MathF.Sqrt(Slope * Slope + 1);
		var cosines = new List<float>();
		for (int row = border; row < normalMap.GetHeight() - border; ++row)
		{
			for (int col = border; col < normalMap.GetWidth() - border; ++col)
			{
				cosines.Add((normalMap.Get(row, col, 0) * Slope - normalMap.Get(row, col, 2)) / normalNorm);
			}
		}

		cosines.Sort();
		return cosines[cosines.Count / 2];
	}

	/// <summary>
	/// The (row, col, image index) memberships of a flat consistency list as
	/// GetConsistentImageIdxs returns it: per pixel, col, row, count, then count image indices.
	/// </summary>
	public static IEnumerable<(int Row, int Col, int ImageIdx)> Memberships(IReadOnlyList<int> flat)
	{
		for (int i = 0; i < flat.Count;)
		{
			int col = flat[i];
			int row = flat[i + 1];
			int count = flat[i + 2];
			for (int j = 0; j < count; ++j)
			{
				yield return (row, col, flat[i + 3 + j]);
			}

			i += 3 + count;
		}
	}
}
