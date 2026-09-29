// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// DelaunayMeshingInput: port of the DelaunayMeshingInput class in colmap/mvs/delaunay_meshing.cc
// - the cameras, registered images (float pose, center, observed point indices) and points
// (float position, number of images that see it) that Delaunay meshing works on, plus the
// two ways COLMAP triangulates them (CreateDelaunayTriangulation and
// CreateSubSampledDelaunayTriangulation). The triangulation is Geometry/Delaunay's
// DelaunayTriangulation3, which replaces CGAL (divergence 103).
// Neighbors: DelaunayMeshingOptions.cs, DelaunayMeshingWeights.cs (edge weights and the ray
// caster). Tests: ColmapSharp.Tests/Mvs/DelaunayMeshingTests.CSharpOnly.cs.
//
// Input is in memory, not read from files: FromSparseReconstruction is COLMAP's
// CopyFromSparseReconstruction (ReadSparseReconstruction minus the file read), and FromDense
// is ReadDenseReconstruction given the undistorted sparse model, the fused points and their
// visibility (StereoFusion.GetFusedPoints / GetFusedPointsVisibility) instead of the
// workspace folder.
//
// Tier C (outcome). Deviations:
// - Sparse points are numbered in ascending point3D id; COLMAP walks an unordered_map
//   (entry 105).
// - While the triangulation is not yet 3D, subsampling always inserts (entry 104).
// - The float products cam_from_world * X.homogeneous() sum pairwise like Fusion.cs's Sum4
//   (Eigen's vectorized reduction on arm64, inferred there, not certain).

using ColmapSharp.Geometry.Delaunay;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>
/// The images and points Delaunay meshing integrates (COLMAP's DelaunayMeshingInput).
/// </summary>
public sealed class DelaunayMeshingInput
{
	/// <summary>A registered image: its camera, float pose and center, and the points it sees.</summary>
	public sealed class InputImage
	{
		/// <summary>The camera id.</summary>
		public uint CameraId { get; init; } = Types.InvalidCameraId;

		/// <summary>cam_from_world as a row-major 3x4 float matrix.</summary>
		public float[] CamFromWorld { get; init; } = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

		/// <summary>The projection center in world coordinates.</summary>
		public Vector3f CamInWorld { get; init; }

		/// <summary>Indices into <see cref="Points"/> of the points the image sees.</summary>
		public List<int> PointIdxs { get; } = [];
	}

	/// <summary>An input point: float position and how many images see it.</summary>
	public readonly record struct InputPoint(Vector3f Position, uint NumVisibleImages);

	/// <summary>Cameras by id.</summary>
	public Dictionary<uint, Camera> Cameras { get; } = [];

	/// <summary>Registered images, in the reconstruction's registration order.</summary>
	public List<InputImage> Images { get; } = [];

	/// <summary>The points to mesh.</summary>
	public List<InputPoint> Points { get; } = [];

	/// <summary>Port of CopyFromSparseReconstruction: the reconstruction's 3D points and registered images.</summary>
	public static DelaunayMeshingInput FromSparseReconstruction(Reconstruction reconstruction)
	{
		var input = new DelaunayMeshingInput();
		input.CopyCameras(reconstruction);

		// COLMAP numbers points in unordered_map order; ascending id is the deterministic
		// equivalent (entry 105).
		var pointIds = reconstruction.Points3D.Keys.ToList();
		pointIds.Sort();
		var pointIdToIdx = new Dictionary<ulong, int>(pointIds.Count);
		foreach (ulong pointId in pointIds)
		{
			var point3D = reconstruction.Points3D[pointId];
			pointIdToIdx.Add(pointId, input.Points.Count);
			input.Points.Add(new InputPoint(ToFloat(point3D.Xyz), (uint)point3D.Track.Length));
		}

		foreach (uint imageId in reconstruction.RegImageIds())
		{
			var image = reconstruction.Image(imageId);
			var inputImage = CreateImage(image);
			foreach (var point2D in image.Points2D)
			{
				if (point2D.HasPoint3D)
				{
					inputImage.PointIdxs.Add(pointIdToIdx[point2D.Point3DId]);
				}
			}

			input.Images.Add(inputImage);
		}

		return input;
	}

	/// <summary>
	/// Port of ReadDenseReconstruction with the data in memory: the (undistorted) sparse model
	/// supplies cameras and registered images, and visibility[i] lists the indices, into the
	/// registered images, of the images that see fused point i.
	/// </summary>
	public static DelaunayMeshingInput FromDense(Reconstruction reconstruction,
		IReadOnlyList<PlyPoint> fusedPoints, IReadOnlyList<IReadOnlyList<int>> visibility)
	{
		Check.Eq(visibility.Count, fusedPoints.Count);
		var input = new DelaunayMeshingInput();
		input.CopyCameras(reconstruction);
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			input.Images.Add(CreateImage(reconstruction.Image(imageId)));
		}

		for (int pointIdx = 0; pointIdx < fusedPoints.Count; ++pointIdx)
		{
			var plyPoint = fusedPoints[pointIdx];
			foreach (int imageIdx in visibility[pointIdx])
			{
				// images.at(image_idx): out-of-range indices throw.
				if (imageIdx < 0 || imageIdx >= input.Images.Count)
				{
					throw new ArgumentOutOfRangeException(nameof(visibility), $"Image index {imageIdx} out of range.");
				}

				input.Images[imageIdx].PointIdxs.Add(pointIdx);
			}

			input.Points.Add(new InputPoint(new Vector3f(plyPoint.X, plyPoint.Y, plyPoint.Z), (uint)visibility[pointIdx].Count));
		}

		return input;
	}

	/// <summary>Port of CreateDelaunayTriangulation: all points, range-inserted.</summary>
	public DelaunayTriangulation3 CreateDelaunayTriangulation(CancellationToken cancellationToken = default)
	{
		var positions = new Vector3d[Points.Count];
		for (int i = 0; i < positions.Length; ++i)
		{
			positions[i] = ToDouble(Points[i].Position);
		}

		var triangulation = new DelaunayTriangulation3();
		triangulation.InsertRange(positions, cancellationToken);
		return triangulation;
	}

	/// <summary>
	/// Port of CreateSubSampledDelaunayTriangulation: inserts the points in a random order
	/// (COLMAP's global PRNG), skipping a point when, in every image that sees it, it
	/// reprojects within max_proj_dist pixels and max_depth_dist relative depth of all four
	/// vertices of the cell it falls in. max_proj_dist == 0 inserts all points.
	/// </summary>
	public DelaunayTriangulation3 CreateSubSampledDelaunayTriangulation(float maxProjDist, float maxDepthDist,
		CancellationToken cancellationToken = default)
	{
		Check.Ge(maxProjDist, 0f);
		if (maxProjDist == 0)
		{
			return CreateDelaunayTriangulation(cancellationToken);
		}

		var pointsVisibleImageIdxs = new List<int>[Points.Count];
		for (int i = 0; i < pointsVisibleImageIdxs.Length; ++i)
		{
			pointsVisibleImageIdxs[i] = [];
		}

		for (int imageIdx = 0; imageIdx < Images.Count; ++imageIdx)
		{
			foreach (int pointIdx in Images[imageIdx].PointIdxs)
			{
				pointsVisibleImageIdxs[pointIdx].Add(imageIdx);
			}
		}

		var pointIdxs = new List<int>(Points.Count);
		for (int i = 0; i < Points.Count; ++i)
		{
			pointIdxs.Add(i);
		}

		RandomUtils.Shuffle((uint)pointIdxs.Count, pointIdxs);

		var triangulation = new DelaunayTriangulation3();
		float maxSquaredProjDist = maxProjDist * maxProjDist;
		float minDepthRatio = 1.0f - maxDepthDist;
		float maxDepthRatio = 1.0f + maxDepthDist;

		for (int k = 0; k < pointIdxs.Count; ++k)
		{
			if ((k & 4095) == 0)
			{
				cancellationToken.ThrowIfCancellationRequested();
			}

			int pointIdx = pointIdxs[k];
			var point = Points[pointIdx];
			var pointPosition = ToDouble(point.Position);

			// Insert point into triangulation until there is one cell. COLMAP tests
			// number_of_vertices() < 4; CGAL can then locate in a flat (2D) triangulation,
			// which ours does not have, so we keep inserting until it is 3D (entry 104).
			if (triangulation.Dimension < 3)
			{
				triangulation.Insert(pointPosition);
				continue;
			}

			int cell = triangulation.Locate(pointPosition);

			// If the point is outside the current hull, then extend the hull.
			if (triangulation.IsInfinite(cell))
			{
				triangulation.Insert(pointPosition);
				continue;
			}

			if (ShouldInsert(triangulation, cell, point, pointsVisibleImageIdxs[pointIdx], maxSquaredProjDist,
				minDepthRatio, maxDepthRatio))
			{
				triangulation.Insert(pointPosition);
			}
		}

		return triangulation;
	}

	// Projects the point and the located cell's vertices into all visible images and decides
	// whether the point adds detail (COLMAP's inner loops, in the same order).
	private bool ShouldInsert(DelaunayTriangulation3 triangulation, int cell, InputPoint point,
		List<int> visibleImageIdxs, float maxSquaredProjDist, float minDepthRatio, float maxDepthRatio)
	{
		foreach (int imageIdx in visibleImageIdxs)
		{
			var image = Images[imageIdx];
			var camera = Cameras[image.CameraId];

			for (int i = 0; i < 4; ++i)
			{
				var cellPoint = ToFloat(triangulation.Point(triangulation.Vertex(cell, i)));
				var pointLocal = Transform(image.CamFromWorld, point.Position);
				var cellPointLocal = Transform(image.CamFromWorld, cellPoint);

				// Ensure that both points are infront of camera.
				if (pointLocal.Z <= 0 || cellPointLocal.Z <= 0)
				{
					return true;
				}

				// Check depth ratio between the two points.
				float depthRatio = pointLocal.Z / cellPointLocal.Z;
				if (depthRatio < minDepthRatio || depthRatio > maxDepthRatio)
				{
					return true;
				}

				// Check reprojection error between the two points.
				var pointProj = camera.ImgFromCam(ToDouble(pointLocal));
				var cellPointProj = camera.ImgFromCam(ToDouble(cellPointLocal));
				if (pointProj is null || cellPointProj is null)
				{
					continue;
				}

				double dx = pointProj.Value.X - cellPointProj.Value.X;
				double dy = pointProj.Value.Y - cellPointProj.Value.Y;
				float squaredProjDist = (float)(dx * dx + dy * dy);
				if (squaredProjDist > maxSquaredProjDist)
				{
					return true;
				}
			}
		}

		return false;
	}

	private void CopyCameras(Reconstruction reconstruction)
	{
		foreach (var (cameraId, camera) in reconstruction.Cameras)
		{
			Cameras.Add(cameraId, camera.Clone());
		}
	}

	private static InputImage CreateImage(Scene.Image image)
	{
		var matrix = image.CamFromWorld().ToMatrix();
		var camFromWorld = new float[12];
		for (int row = 0; row < 3; ++row)
		{
			for (int col = 0; col < 4; ++col)
			{
				camFromWorld[4 * row + col] = (float)matrix[row, col];
			}
		}

		return new InputImage
		{
			CameraId = image.CameraId,
			CamFromWorld = camFromWorld,
			CamInWorld = ToFloat(image.ProjectionCenter()),
		};
	}

	// Matrix3x4f * Vector3f.homogeneous(), summed pairwise (see the file header).
	internal static Vector3f Transform(float[] m, Vector3f p) => new(
		(m[0] * p.X + m[1] * p.Y) + (m[2] * p.Z + m[3] * 1.0f),
		(m[4] * p.X + m[5] * p.Y) + (m[6] * p.Z + m[7] * 1.0f),
		(m[8] * p.X + m[9] * p.Y) + (m[10] * p.Z + m[11] * 1.0f));

	internal static Vector3f ToFloat(Vector3d v) => new((float)v.X, (float)v.Y, (float)v.Z);

	internal static Vector3d ToDouble(Vector3f v) => new(v.X, v.Y, v.Z);
}
