// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Model: colmap/mvs/model.h and model.cc - the sparse model MVS runs on: the registered
// images as Mvs.Image (float K, R, T) and the 3D points with their tracks as image indices,
// plus the overlap, depth-range and triangulation-angle queries PatchMatch uses to pick
// source images. This file holds the COLMAP reader and the queries; Model.Pmvs.cs holds the
// PMVS readers. Workspace.cs owns a Model. Tests: ColmapSharp.Tests/Mvs/ModelTests.cs
// (model_test.cc 1:1).
//
// Tier A (exact) with two deliberate differences:
// - Points come from the reconstruction in ascending point3D id order (Util/IdMap.cs,
//   docs/CPP_DIVERGENCES.md entry 21) where COLMAP iterates its hash map, so a point's
//   index can differ from COLMAP's; images are in registration order in both.
// - GetMaxOverlappingImages breaks ties in the shared-point count by ascending image
//   index (docs/CPP_DIVERGENCES.md, entry 64).
//
// Additions for MatterCAD: ReadFromCOLMAP also takes a Reconstruction in memory (the path
// overload reads one with Reconstruction.Read and delegates to it).

using System.Runtime.InteropServices;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>Port of colmap::mvs::Model: a simple sparse model for MVS.</summary>
public sealed partial class Model
{
	private readonly List<string> imageNames = new();
	private readonly Dictionary<string, int> imageNameToIdx = new();

	/// <summary>A 3D point and the indices (into <see cref="Images"/>) of the images observing it.</summary>
	public sealed class Point
	{
		/// <summary>X coordinate.</summary>
		public float X;

		/// <summary>Y coordinate.</summary>
		public float Y;

		/// <summary>Z coordinate.</summary>
		public float Z;

		/// <summary>Indices of the images that observe the point.</summary>
		public List<int> Track = new();
	}

	/// <summary>
	/// The images. When read from a COLMAP reconstruction, an image's index is its position
	/// in registration order, not its image id; likewise for <see cref="Points"/>. This is
	/// mainly done for more efficient access to the data, which is required during the
	/// stereo fusion stage.
	/// </summary>
	public List<Image> Images { get; } = new();

	/// <summary>The 3D points.</summary>
	public List<Point> Points { get; } = new();

	/// <summary>
	/// A deep copy (images, points, names and PMVS visibility), so a Workspace can downsize
	/// its images without touching the caller's model (C++ copies the Model by value).
	/// </summary>
	public Model Clone()
	{
		var copy = new Model();
		foreach (Image image in Images)
		{
			copy.Images.Add(image.Clone());
		}

		foreach (Point point in Points)
		{
			copy.Points.Add(new Point { X = point.X, Y = point.Y, Z = point.Z, Track = new List<int>(point.Track) });
		}

		copy.imageNames.AddRange(imageNames);
		foreach ((string name, int idx) in imageNameToIdx)
		{
			copy.imageNameToIdx.Add(name, idx);
		}

		foreach (List<int> visible in pmvsVisDat)
		{
			copy.pmvsVisDat.Add(new List<int>(visible));
		}

		return copy;
	}

	/// <summary>
	/// Reads the model in the given format ("COLMAP" or "PMVS", case-insensitive).
	/// <paramref name="bitmaps"/> is only needed for PMVS, whose image sizes come from the
	/// image files.
	/// </summary>
	public void Read(string path, string format, IBitmapSource? bitmaps = null)
	{
		string formatLowerCase = format.ToLowerInvariant();
		if (formatLowerCase == "colmap")
		{
			ReadFromCOLMAP(path);
		}
		else if (formatLowerCase == "pmvs")
		{
			ReadFromPMVS(path, Check.NotNull(bitmaps, "Reading a PMVS model needs a bitmap source"));
		}
		else
		{
			throw new InvalidOperationException("Invalid input format");
		}
	}

	/// <summary>
	/// Reads the reconstruction in <c>path/sparsePath</c>; image paths are
	/// <c>path/imagesPath/&lt;name&gt;</c>.
	/// </summary>
	public void ReadFromCOLMAP(string path, string sparsePath = "sparse", string imagesPath = "images")
	{
		var reconstruction = new Reconstruction();
		reconstruction.Read(System.IO.Path.Combine(path, sparsePath));
		ReadFromCOLMAP(reconstruction, System.IO.Path.Combine(path, imagesPath));
	}

	/// <summary>
	/// Builds the model from a reconstruction in memory; image paths are
	/// <c>imagesDir/&lt;name&gt;</c> (keys for the host's bitmap source).
	/// </summary>
	public void ReadFromCOLMAP(Reconstruction reconstruction, string imagesDir)
	{
		List<uint> regImageIds = reconstruction.RegImageIds();
		Images.Capacity = Math.Max(Images.Capacity, regImageIds.Count);
		var imageIdToIdx = new Dictionary<uint, int>();
		int imageIdx = 0;
		Span<float> k = stackalloc float[9];
		Span<float> r = stackalloc float[9];
		Span<float> t = stackalloc float[3];
		foreach (uint imageId in regImageIds)
		{
			Scene.Image image = reconstruction.Image(imageId);
			Camera camera = image.CameraPtr;

			string imagePath = System.IO.Path.Combine(imagesDir, image.Name);
			Matrix3d kd = camera.CalibrationMatrix();
			Rigid3d camFromWorld = image.CamFromWorld();
			Matrix3d rd = camFromWorld.Rotation.ToRotationMatrix();
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 3; col++)
				{
					k[row * 3 + col] = (float)kd[row, col];
					r[row * 3 + col] = (float)rd[row, col];
				}

				t[row] = (float)camFromWorld.Translation[row];
			}

			Images.Add(new Image(imagePath, camera.Width, camera.Height, k, r, t));
			// emplace keeps the first entry for a repeated key.
			imageIdToIdx.TryAdd(imageId, imageIdx);
			imageNames.Add(image.Name);
			imageNameToIdx.TryAdd(image.Name, imageIdx);
			++imageIdx;
		}

		Points.Capacity = Math.Max(Points.Capacity, Points.Count + reconstruction.NumPoints3D);
		foreach (Point3D point3D in reconstruction.Points3D.Values)
		{
			Vector3d xyz = point3D.Xyz;
			var point = new Point { X = (float)xyz.X, Y = (float)xyz.Y, Z = (float)xyz.Z };
			point.Track.Capacity = point3D.Track.Length;
			foreach (TrackElement trackEl in point3D.Track.Elements)
			{
				point.Track.Add(imageIdToIdx[trackEl.ImageId]);
			}

			Points.Add(point);
		}
	}

	/// <summary>The image index for the given image name.</summary>
	public int GetImageIdx(string name)
	{
		if (!imageNameToIdx.TryGetValue(name, out int idx))
		{
			Check.That(false, $"Image with name `{name}` does not exist");
		}

		return idx;
	}

	/// <summary>The name of the image with the given index.</summary>
	public string GetImageName(int imageIdx)
	{
		Check.Ge(imageIdx, 0);
		Check.Lt(imageIdx, imageNames.Count);
		return imageNames[imageIdx];
	}

	/// <summary>
	/// For each image, the at most <paramref name="numImages"/> images sharing the most
	/// points with it, among those whose 75th-percentile triangulation angle is at least
	/// <paramref name="minTriangulationAngle"/> degrees; most shared points first.
	/// </summary>
	public List<List<int>> GetMaxOverlappingImages(int numImages, double minTriangulationAngle)
	{
		var overlappingImages = new List<List<int>>(Images.Count);

		float minTriangulationAngleRad = (float)MathUtils.DegToRad(minTriangulationAngle);

		List<SortedDictionary<int, int>> sharedNumPoints = ComputeSharedPoints();

		const float TriangulationAnglePercentile = 75;
		List<SortedDictionary<int, float>> triangulationAngles = ComputeTriangulationAngles(TriangulationAnglePercentile);

		for (int imageIdx = 0; imageIdx < Images.Count; ++imageIdx)
		{
			SortedDictionary<int, int> sharedImages = sharedNumPoints[imageIdx];
			SortedDictionary<int, float> overlappingTriangulationAngles = triangulationAngles[imageIdx];

			var orderedImages = new List<(int ImageIdx, int Count)>(sharedImages.Count);
			foreach ((int otherImageIdx, int count) in sharedImages)
			{
				if (overlappingTriangulationAngles[otherImageIdx] >= minTriangulationAngleRad)
				{
					orderedImages.Add((otherImageIdx, count));
				}
			}

			int effNumImages = Math.Min(orderedImages.Count, numImages);

			// COLMAP partial_sorts (or sorts) by count only, leaving ties in libc++'s
			// unspecified order; ties go to the lower image index here (divergence 64). That
			// makes the order total, so a full sort gives the same prefix as partial_sort.
			orderedImages.Sort(static (a, b) => a.Count != b.Count ? b.Count.CompareTo(a.Count) : a.ImageIdx.CompareTo(b.ImageIdx));

			var overlapping = new List<int>(effNumImages);
			for (int i = 0; i < effNumImages; ++i)
			{
				overlapping.Add(orderedImages[i].ImageIdx);
			}

			overlappingImages.Add(overlapping);
		}

		return overlappingImages;
	}

	/// <summary>
	/// The robust (1st to 99th percentile, stretched by 25%) minimum and maximum depths of
	/// each image's points; (-1, -1) for an image without points in front of it.
	/// </summary>
	public List<(float Min, float Max)> ComputeDepthRanges()
	{
		var depths = new List<float>[Images.Count];
		for (int i = 0; i < depths.Length; i++)
		{
			depths[i] = new List<float>();
		}

		foreach (Point point in Points)
		{
			foreach (int imageIdx in point.Track)
			{
				Image image = Images[imageIdx];
				ReadOnlySpan<float> r = image.GetR();
				float depth = r[6] * point.X + r[7] * point.Y + r[8] * point.Z + image.GetT()[2];
				if (depth > 0)
				{
					depths[imageIdx].Add(depth);
				}
			}
		}

		var depthRanges = new List<(float Min, float Max)>(depths.Length);
		for (int imageIdx = 0; imageIdx < depths.Length; ++imageIdx)
		{
			List<float> imageDepths = depths[imageIdx];
			if (imageDepths.Count == 0)
			{
				depthRanges.Add((-1.0f, -1.0f));
				continue;
			}

			imageDepths.Sort();

			const float MinPercentile = 0.01f;
			const float MaxPercentile = 0.99f;
			float min = imageDepths[(int)(imageDepths.Count * MinPercentile)];
			float max = imageDepths[(int)(imageDepths.Count * MaxPercentile)];

			const float StretchRatio = 0.25f;
			min *= 1.0f - StretchRatio;
			max *= 1.0f + StretchRatio;
			depthRanges.Add((min, max));
		}

		return depthRanges;
	}

	/// <summary>For each image, the number of points it shares with each overlapping image.</summary>
	public List<SortedDictionary<int, int>> ComputeSharedPoints()
	{
		var sharedPoints = new List<SortedDictionary<int, int>>(Images.Count);
		for (int i = 0; i < Images.Count; i++)
		{
			sharedPoints.Add(new SortedDictionary<int, int>());
		}

		foreach (Point point in Points)
		{
			for (int i = 0; i < point.Track.Count; ++i)
			{
				int imageIdx1 = point.Track[i];
				for (int j = 0; j < i; ++j)
				{
					int imageIdx2 = point.Track[j];
					if (imageIdx1 != imageIdx2)
					{
						Increment(sharedPoints[imageIdx1], imageIdx2);
						Increment(sharedPoints[imageIdx2], imageIdx1);
					}
				}
			}
		}

		return sharedPoints;
	}

	/// <summary>
	/// For each image, the given percentile (default the median) of the triangulation
	/// angles, in radians, of the points it shares with each overlapping image.
	/// </summary>
	public List<SortedDictionary<int, float>> ComputeTriangulationAngles(float percentile = 50)
	{
		var projCenters = new Vector3d[Images.Count];
		Span<float> c = stackalloc float[3];
		for (int imageIdx = 0; imageIdx < Images.Count; ++imageIdx)
		{
			Image image = Images[imageIdx];
			MvsGeometry.ComputeProjectionCenter(image.GetR(), image.GetT(), c);
			projCenters[imageIdx] = new Vector3d(c[0], c[1], c[2]);
		}

		var allTriangulationAngles = new List<SortedDictionary<int, List<float>>>(Images.Count);
		for (int i = 0; i < Images.Count; i++)
		{
			allTriangulationAngles.Add(new SortedDictionary<int, List<float>>());
		}

		foreach (Point point in Points)
		{
			var xyz = new Vector3d(point.X, point.Y, point.Z);
			for (int i = 0; i < point.Track.Count; ++i)
			{
				int imageIdx1 = point.Track[i];
				for (int j = 0; j < i; ++j)
				{
					int imageIdx2 = point.Track[j];
					if (imageIdx1 != imageIdx2)
					{
						float angle = (float)Triangulation.CalculateTriangulationAngle(
							projCenters[imageIdx1], projCenters[imageIdx2], xyz);
						AnglesOf(allTriangulationAngles[imageIdx1], imageIdx2).Add(angle);
						AnglesOf(allTriangulationAngles[imageIdx2], imageIdx1).Add(angle);
					}
				}
			}
		}

		var triangulationAngles = new List<SortedDictionary<int, float>>(Images.Count);
		foreach (SortedDictionary<int, List<float>> overlappingImages in allTriangulationAngles)
		{
			var angles = new SortedDictionary<int, float>();
			foreach ((int otherImageIdx, List<float> otherAngles) in overlappingImages)
			{
				angles.Add(otherImageIdx, (float)MathUtils.Percentile(CollectionsMarshal.AsSpan(otherAngles), percentile));
			}

			triangulationAngles.Add(angles);
		}

		return triangulationAngles;
	}

	private static void Increment(SortedDictionary<int, int> counts, int key)
	{
		counts.TryGetValue(key, out int count);
		counts[key] = count + 1;
	}

	private static List<float> AnglesOf(SortedDictionary<int, List<float>> angles, int key)
	{
		if (!angles.TryGetValue(key, out List<float>? list))
		{
			list = new List<float>();
			angles.Add(key, list);
		}

		return list;
	}

	private void AddImage(Image image, string imageName, int imageIdx)
	{
		Images.Add(image);
		imageNames.Add(imageName);
		imageNameToIdx.TryAdd(imageName, imageIdx);
	}
}
