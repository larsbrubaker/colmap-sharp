// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalMapperImpl: port of colmap/sfm/incremental_mapper_impl.h and .cc, the
// algorithm half of the incremental mapper (Sfm/IncrementalMapper*.cs): ranking seed images,
// finding and estimating the initial image pair, ranking the next images to register and
// choosing the local bundle. The mapper owns the state and forwards to these static
// functions. COLMAP 4.2.0 has no incremental_mapper_impl_test.cc; the functions are tested
// through ColmapSharp.Tests/Sfm/IncrementalMapperTests.cs (incremental_mapper_test.cc).
//
// Tier C (outcome) for the initial pair (it runs RANSAC); Tier A for the rankings and the
// local bundle given the same reconstruction.
//
// Translation notes:
// - Sort ties. COLMAP ranks with std::sort, whose order among equal keys is unspecified,
//   over inputs in hash-map order. Every ranking here breaks ties by ascending image id
//   (divergence 59).
// - FindInitialImagePair runs its per-seed tasks sequentially in seed order, which is what
//   COLMAP's thread pool reproduces when num_threads is 1 (entry 58).
// - InitInfo's std::optional<Camera> members are nullable Camera references.

using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Sfm;

/// <summary>
/// Port of colmap::IncrementalMapperImpl: the algorithms behind
/// <see cref="IncrementalMapper"/>, kept separate to make them easier to extend.
/// </summary>
public static class IncrementalMapperImpl
{
	/// <summary>
	/// Port of IncrementalMapperImpl::InitInfo: the result of selecting and/or estimating the
	/// initial image pair. <see cref="Camera1"/>/<see cref="Camera2"/> carry the intrinsics
	/// estimated for the chosen pair by two-view solvers that recover them (e.g. the
	/// shared-focal solver, which sets both to the same camera), and are null otherwise.
	/// </summary>
	public sealed class InitInfo
	{
		/// <summary>The first image of the pair.</summary>
		public uint ImageId1 { get; set; } = InvalidImageId;

		/// <summary>The second image of the pair.</summary>
		public uint ImageId2 { get; set; } = InvalidImageId;

		/// <summary>The relative pose of the second camera.</summary>
		public Rigid3d Cam2FromCam1 { get; set; } = Rigid3d.Identity;

		/// <summary>Estimated intrinsics of the first camera, if the solver recovered them.</summary>
		public Camera? Camera1 { get; set; }

		/// <summary>Estimated intrinsics of the second camera, if the solver recovered them.</summary>
		public Camera? Camera2 { get; set; }
	}

	// Meta-data for ranking images during initialization. Used by both FindFirstInitialImage
	// and FindSecondInitialImage.
	private readonly record struct InitImageInfo(uint ImageId, bool PriorFocalLength, long NumCorrespondences);

	// Prefer images with prior focal length, then by number of correspondences (descending).
	// Ties go to the smaller image id (file header).
	private static int CompareImageInfo(InitImageInfo a, InitImageInfo b)
	{
		if (a.PriorFocalLength != b.PriorFocalLength)
		{
			return a.PriorFocalLength ? -1 : 1;
		}

		int byCorrs = b.NumCorrespondences.CompareTo(a.NumCorrespondences);
		return byCorrs != 0 ? byCorrs : a.ImageId.CompareTo(b.ImageId);
	}

	private static List<uint> ExtractSortedImageIds(List<InitImageInfo> imageInfos)
	{
		imageInfos.Sort(CompareImageInfo);
		return imageInfos.ConvertAll(info => info.ImageId);
	}

	// Sorts by descending rank (ties by ascending image id) and appends the ids.
	private static void SortAndAppendNextImages(List<(uint ImageId, float Rank)> imageRanks, List<uint> sortedImageIds)
	{
		imageRanks.Sort((image1, image2) =>
		{
			int byRank = image2.Rank.CompareTo(image1.Rank);
			return byRank != 0 ? byRank : image1.ImageId.CompareTo(image2.ImageId);
		});

		foreach ((uint imageId, float _) in imageRanks)
		{
			sortedImageIds.Add(imageId);
		}
	}

	/// <summary>
	/// Finds seed images for incremental reconstruction. Suitable seed images have a large
	/// number of correspondences and have camera calibration priors. The returned list is
	/// ordered such that most suitable images are in the front.
	/// </summary>
	public static List<uint> FindFirstInitialImage(
		IncrementalMapper.Options options,
		CorrespondenceGraph correspondenceGraph,
		Reconstruction reconstruction,
		IReadOnlyDictionary<uint, int> initNumRegTrials,
		IReadOnlyDictionary<uint, int> numRegistrations)
	{
		int initMaxRegTrials = options.InitMaxRegTrials;

		// Collect information of all not yet registered images with correspondences.
		var imageInfos = new List<InitImageInfo>(reconstruction.NumImages);
		foreach ((uint imageId, Image image) in reconstruction.Images)
		{
			// Only images with correspondences can be registered.
			if (correspondenceGraph.NumCorrespondencesForImage(imageId) == 0)
			{
				continue;
			}

			// Only use images for initialization a maximum number of times.
			if (initNumRegTrials.TryGetValue(imageId, out int trials) && trials >= initMaxRegTrials)
			{
				continue;
			}

			// Only use images for initialization that are not registered in any of the other
			// reconstructions.
			if (numRegistrations.TryGetValue(imageId, out int registrations) && registrations > 0)
			{
				continue;
			}

			imageInfos.Add(new InitImageInfo(
				imageId,
				image.CameraPtr.HasPriorFocalLength,
				correspondenceGraph.NumCorrespondencesForImage(imageId)));
		}

		return ExtractSortedImageIds(imageInfos);
	}

	/// <summary>
	/// For a given first seed image, finds other images that are connected to the first
	/// image. Suitable second images have a large number of correspondences to the first
	/// image and have camera calibration priors. The returned list is ordered such that most
	/// suitable images are in the front.
	/// </summary>
	public static List<uint> FindSecondInitialImage(
		IncrementalMapper.Options options,
		uint imageId1,
		CorrespondenceGraph correspondenceGraph,
		Reconstruction reconstruction,
		IReadOnlyDictionary<uint, int> numRegistrations)
	{
		// Collect images that are connected to the first seed image and have not been
		// registered before in other reconstructions.
		Image image1 = reconstruction.Image(imageId1);
		var numCorrespondences = new Dictionary<uint, long>();
		for (uint point2DIdx = 0; point2DIdx < image1.NumPoints2D; ++point2DIdx)
		{
			foreach (CorrespondenceGraph.Correspondence corr in correspondenceGraph.FindCorrespondences(imageId1, point2DIdx))
			{
				if (!numRegistrations.TryGetValue(corr.ImageId, out int registrations) || registrations == 0)
				{
					numCorrespondences[corr.ImageId] = numCorrespondences.GetValueOrDefault(corr.ImageId) + 1;
				}
			}
		}

		long initMinNumInliers = options.InitMinNumInliers;

		// Compose image information in a compact form for sorting.
		var imageInfos = new List<InitImageInfo>(numCorrespondences.Count);
		foreach ((uint imageId, long numCorrs) in numCorrespondences)
		{
			if (numCorrs >= initMinNumInliers)
			{
				Image image = reconstruction.Image(imageId);
				imageInfos.Add(new InitImageInfo(imageId, image.CameraPtr.HasPriorFocalLength, numCorrs));
			}
		}

		return ExtractSortedImageIds(imageInfos);
	}

	/// <summary>
	/// Implements <see cref="IncrementalMapper.FindInitialImagePair"/>. Returns the selected
	/// pair, or null if no suitable pair was found. <paramref name="imageId1"/> /
	/// <paramref name="imageId2"/> optionally constrain the search to a specific first and/or
	/// second image (<see cref="InvalidImageId"/> leaves the respective image unconstrained).
	/// </summary>
	public static InitInfo? FindInitialImagePair(
		IncrementalMapper.Options options,
		DatabaseCache databaseCache,
		Reconstruction reconstruction,
		IReadOnlyDictionary<uint, int> initNumRegTrials,
		IReadOnlyDictionary<uint, int> numRegistrations,
		HashSet<ulong> initImagePairs,
		uint imageId1,
		uint imageId2)
	{
		Check.That(options.Check());

		CorrespondenceGraph correspondenceGraph = databaseCache.CorrespondenceGraph;

		List<uint> imageIds1;
		if (imageId1 != InvalidImageId && imageId2 == InvalidImageId)
		{
			// Only image_id1 provided.
			if (!databaseCache.ExistsImage(imageId1))
			{
				return null;
			}

			imageIds1 = [imageId1];
		}
		else if (imageId1 == InvalidImageId && imageId2 != InvalidImageId)
		{
			// Only image_id2 provided.
			if (!databaseCache.ExistsImage(imageId2))
			{
				return null;
			}

			imageIds1 = [imageId2];
		}
		else
		{
			// No initial seed image provided.
			imageIds1 = FindFirstInitialImage(
				options, correspondenceGraph, reconstruction, initNumRegTrials, numRegistrations);
		}

		// Try to find good initial pair. COLMAP runs one task per seed image on a thread pool
		// and returns the first successful result in seed order; running the seeds in order
		// and stopping at the first success is its num_threads = 1 behavior (file header).
		foreach (uint seedImageId1 in imageIds1)
		{
			List<uint> imageIds2 = FindSecondInitialImage(
				options, seedImageId1, correspondenceGraph, reconstruction, numRegistrations);

			foreach (uint seedImageId2 in imageIds2)
			{
				ulong pairId = ImagePairToPairId(seedImageId1, seedImageId2);

				// Try every pair only once.
				if (!initImagePairs.Add(pairId))
				{
					continue;
				}

				InitInfo? pairInitInfo = EstimateInitialTwoViewGeometry(
					options, databaseCache, seedImageId1, seedImageId2);
				if (pairInitInfo is not null)
				{
					return pairInitInfo;
				}
			}
		}

		// No suitable pair found in entire dataset.
		return null;
	}

	/// <summary>
	/// Implements <see cref="IncrementalMapper.FindNextImages"/>: the unregistered images
	/// worth trying next, best first, with images that were filtered or failed before at the
	/// back.
	/// </summary>
	public static List<uint> FindNextImages(
		IncrementalMapper.Options options,
		ObservationManager obsManager,
		IReadOnlySet<uint> filteredFrames,
		Dictionary<uint, int> numRegTrials,
		bool structureLess = false)
	{
		Check.That(options.Check());
		Reconstruction reconstruction = obsManager.Reconstruction;

		Func<uint, ObservationManager, float> rankImageFunc;
		if (structureLess)
		{
			rankImageFunc = (imageId, manager) => manager.NumVisibleCorrespondences(imageId);
		}
		else
		{
			rankImageFunc = options.ImageSelectionMethod switch
			{
				IncrementalMapper.ImageSelectionMethod.MaxVisiblePointsNum =>
					(imageId, manager) => manager.NumVisiblePoints3D(imageId),
				IncrementalMapper.ImageSelectionMethod.MaxVisiblePointsRatio =>
					(imageId, manager) => (float)manager.NumVisiblePoints3D(imageId) / manager.NumObservations(imageId),
				_ => (imageId, manager) => manager.Point3DVisibilityScore(imageId),
			};
		}

		var imageRanks = new List<(uint, float)>();
		var otherImageRanks = new List<(uint, float)>();

		// Append images that have not failed to register before.
		foreach ((uint imageId, Image image) in reconstruction.Images)
		{
			// Skip images that are already registered.
			if (image.HasPose)
			{
				continue;
			}

			// Only consider images with a sufficient number of visible points.
			if (obsManager.NumVisiblePoints3D(imageId) < options.AbsPoseMinNumInliers)
			{
				continue;
			}

			// Only try registration for a certain maximum number of times. C++'s operator[]
			// inserts a zero count, as here.
			int imageNumRegTrials = IncrementalMapper.CountAt(numRegTrials, imageId);
			if (imageNumRegTrials >= options.MaxRegTrials)
			{
				continue;
			}

			// If image has been filtered or failed to register, place it in the second bucket
			// and prefer images that have not been tried before.
			float rank = rankImageFunc(imageId, obsManager);
			if (!filteredFrames.Contains(image.FrameId) && imageNumRegTrials == 0)
			{
				imageRanks.Add((imageId, rank));
			}
			else
			{
				otherImageRanks.Add((imageId, rank));
			}
		}

		var rankedImageIds = new List<uint>(imageRanks.Count + otherImageRanks.Count);
		SortAndAppendNextImages(imageRanks, rankedImageIds);
		SortAndAppendNextImages(otherImageRanks, rankedImageIds);

		return rankedImageIds;
	}

	/// <summary>
	/// Implements <see cref="IncrementalMapper.FindLocalBundle"/>: the registered images
	/// most connected to the given image (the most shared 3D points), preferring those with
	/// a sufficient triangulation angle.
	/// </summary>
	public static List<uint> FindLocalBundle(IncrementalMapper.Options options, uint imageId, Reconstruction reconstruction)
	{
		Check.That(options.Check());

		Image image = reconstruction.Image(imageId);
		Check.That(image.HasPose);

		// Extract all images that have at least one 3D point with the query image in
		// common, and simultaneously count the number of common 3D points.
		var sharedObservations = new Dictionary<uint, int>();
		var point3DIds = new HashSet<ulong>((int)image.NumPoints3D);

		foreach (Point2D point2D in image.Points2D)
		{
			if (point2D.HasPoint3D)
			{
				point3DIds.Add(point2D.Point3DId);
				Point3D point3D = reconstruction.Point3D(point2D.Point3DId);
				foreach (TrackElement trackEl in point3D.Track.Elements)
				{
					if (trackEl.ImageId != imageId)
					{
						sharedObservations[trackEl.ImageId] = sharedObservations.GetValueOrDefault(trackEl.ImageId) + 1;
					}
				}
			}
		}

		// Sort overlapping images according to number of shared observations (ties by
		// ascending image id, file header).
		var overlappingImages = sharedObservations.Select(kv => (ImageId: kv.Key, Count: kv.Value)).ToList();
		overlappingImages.Sort((image1, image2) =>
		{
			int byCount = image2.Count.CompareTo(image1.Count);
			return byCount != 0 ? byCount : image1.ImageId.CompareTo(image2.ImageId);
		});

		// The local bundle is composed of the given image and its most connected neighbor
		// images, hence the subtraction of 1.
		int numImages = options.BaLocalNumImages - 1;
		int numEffImages = Math.Min(numImages, overlappingImages.Count);

		// Extract most connected images and ensure sufficient triangulation angle.
		var localBundleImageIds = new List<uint>(numEffImages);

		// If the number of overlapping images equals the number of desired images in the
		// local bundle, then simply copy over the image identifiers.
		if (overlappingImages.Count == numEffImages)
		{
			foreach ((uint overlappingImageId, int _) in overlappingImages)
			{
				localBundleImageIds.Add(overlappingImageId);
			}

			return localBundleImageIds;
		}

		// In the following iteration, we start with the most overlapping images and check
		// whether it has sufficient triangulation angle. If none of the overlapping images
		// has sufficient triangulation angle, we relax the triangulation angle threshold and
		// start from the most overlapping image again. In the end, if we still haven't found
		// enough images, we simply use the most overlapping images.

		double minTriAngleRad = MathUtils.DegToRad(options.BaLocalMinTriAngle);

		// The selection thresholds (minimum triangulation angle, minimum number of shared
		// observations), which are successively relaxed.
		double numPoints3D = image.NumPoints3D;
		(double MinTriAngleRad, double MinNumSharedObs)[] selectionThresholds =
		[
			(minTriAngleRad / 1.0, 0.6 * numPoints3D),
			(minTriAngleRad / 1.5, 0.6 * numPoints3D),
			(minTriAngleRad / 2.0, 0.5 * numPoints3D),
			(minTriAngleRad / 2.5, 0.4 * numPoints3D),
			(minTriAngleRad / 3.0, 0.3 * numPoints3D),
			(minTriAngleRad / 4.0, 0.2 * numPoints3D),
			(minTriAngleRad / 5.0, 0.1 * numPoints3D),
			(minTriAngleRad / 6.0, 0.1 * numPoints3D),
		];

		Vector3d projCenter = image.ProjectionCenter();
		var sharedPoints3D = new List<Vector3d>((int)image.NumPoints3D);
		var triAngles = new double[overlappingImages.Count];
		Array.Fill(triAngles, -1.0);
		var usedOverlappingImages = new bool[overlappingImages.Count];

		foreach ((double thresholdTriAngleRad, double minNumSharedObs) in selectionThresholds)
		{
			for (int overlappingImageIdx = 0; overlappingImageIdx < overlappingImages.Count; ++overlappingImageIdx)
			{
				// Check if the image has sufficient overlap. Since the images are ordered
				// based on the overlap, we can just skip the remaining ones.
				if (overlappingImages[overlappingImageIdx].Count < minNumSharedObs)
				{
					break;
				}

				// Check if the image is already in the local bundle.
				if (usedOverlappingImages[overlappingImageIdx])
				{
					continue;
				}

				Image overlappingImage = reconstruction.Image(overlappingImages[overlappingImageIdx].ImageId);
				Vector3d overlappingProjCenter = overlappingImage.ProjectionCenter();

				// In the first iteration, compute the triangulation angle. In later
				// iterations, reuse the previously computed value.
				if (triAngles[overlappingImageIdx] < 0.0)
				{
					// Collect the commonly observed 3D points.
					sharedPoints3D.Clear();
					foreach (Point2D point2D in overlappingImage.Points2D)
					{
						if (point2D.HasPoint3D && point3DIds.Contains(point2D.Point3DId))
						{
							sharedPoints3D.Add(reconstruction.Point3D(point2D.Point3DId).Xyz);
						}
					}

					// Calculate the triangulation angle at a certain percentile.
					const double kTriangulationAnglePercentile = 75;
					triAngles[overlappingImageIdx] = MathUtils.Percentile(
						Triangulation.CalculateTriangulationAngles(projCenter, overlappingProjCenter, sharedPoints3D).AsSpan(),
						kTriangulationAnglePercentile);
				}

				// Check that the image has sufficient triangulation angle.
				if (triAngles[overlappingImageIdx] >= thresholdTriAngleRad)
				{
					localBundleImageIds.Add(overlappingImage.ImageId);
					usedOverlappingImages[overlappingImageIdx] = true;
					// Check if we already collected enough images.
					if (localBundleImageIds.Count >= numEffImages)
					{
						break;
					}
				}
			}

			// Check if we already collected enough images.
			if (localBundleImageIds.Count >= numEffImages)
			{
				break;
			}
		}

		// In case there are not enough images with sufficient triangulation angle, simply
		// fill up the rest with the most overlapping images.
		if (localBundleImageIds.Count < numEffImages)
		{
			for (int overlappingImageIdx = 0; overlappingImageIdx < overlappingImages.Count; ++overlappingImageIdx)
			{
				// Collect image if it is not yet in the local bundle.
				if (!usedOverlappingImages[overlappingImageIdx])
				{
					localBundleImageIds.Add(overlappingImages[overlappingImageIdx].ImageId);
					usedOverlappingImages[overlappingImageIdx] = true;

					// Check if we already collected enough images.
					if (localBundleImageIds.Count >= numEffImages)
					{
						break;
					}
				}
			}
		}

		return localBundleImageIds;
	}

	/// <summary>
	/// Implements <see cref="IncrementalMapper.EstimateInitialTwoViewGeometry"/>. Returns the
	/// estimated two-view geometry, or null if the pair is unsuitable for initialization.
	/// </summary>
	public static InitInfo? EstimateInitialTwoViewGeometry(
		IncrementalMapper.Options options, DatabaseCache databaseCache, uint imageId1, uint imageId2)
	{
		Image image1 = databaseCache.Image(imageId1);
		Image image2 = databaseCache.Image(imageId2);
		Camera camera1 = databaseCache.Camera(image1.CameraId);
		Camera camera2 = databaseCache.Camera(image2.CameraId);

		var matches = new List<FeatureMatch>();
		databaseCache.CorrespondenceGraph.ExtractMatchesBetweenImages(imageId1, imageId2, matches);

		var points1 = new List<Vector2d>((int)image1.NumPoints2D);
		foreach (Point2D point in image1.Points2D)
		{
			points1.Add(point.Xy);
		}

		var points2 = new List<Vector2d>((int)image2.NumPoints2D);
		foreach (Point2D point in image2.Points2D)
		{
			points2.Add(point.Xy);
		}

		var twoViewGeometryOptions = new TwoViewGeometryOptions();
		twoViewGeometryOptions.RansacOptions.MinNumTrials = 30;
		twoViewGeometryOptions.RansacOptions.MaxError = options.InitMaxError;
		twoViewGeometryOptions.RansacOptions.RandomSeed = options.RandomSeed;
		// Delegate estimator selection to the general two-view geometry estimator so it lives
		// in a single place. The relative pose is recovered separately below.
		TwoViewGeometry twoViewGeometry = TwoViewGeometryEstimation.EstimateTwoViewGeometry(
			camera1, points1, camera2, points2, matches, twoViewGeometryOptions);

		if (!TwoViewGeometryEstimation.EstimateTwoViewGeometryPose(camera1, points1, camera2, points2, twoViewGeometry))
		{
			return null;
		}

		if (twoViewGeometry.InlierMatches.Count < options.InitMinNumInliers
			|| Math.Abs(twoViewGeometry.Cam2FromCam1!.Value.Translation.Z) >= options.InitMaxForwardMotion
			|| twoViewGeometry.TriAngle <= MathUtils.DegToRad(options.InitMinTriAngle))
		{
			return null;
		}

		Frame frame1 = databaseCache.Frame(image1.FrameId);
		Frame frame2 = databaseCache.Frame(image2.FrameId);
		Rig rig1 = databaseCache.Rig(frame1.RigId);
		Rig rig2 = databaseCache.Rig(frame2.RigId);

		var info = new InitInfo { ImageId1 = imageId1, ImageId2 = imageId2 };

		// If one or both of the frames are non-trivial, initialize using generalized relative
		// pose solver. Note that we intentionally do this after ensuring that the given image
		// pair has stable two-view geometry.
		if (rig1.NumSensors > 1 || rig2.NumSensors > 1)
		{
			if (!EstimateInitialGeneralizedTwoViewGeometry(
				options, databaseCache, image1, image2, frame1, frame2, rig1, rig2, out Rigid3d cam2FromCam1))
			{
				return null;
			}

			// The generalized solver does not recover intrinsics.
			info.Cam2FromCam1 = cam2FromCam1;
			return info;
		}

		info.Cam2FromCam1 = twoViewGeometry.Cam2FromCam1.Value;

		// Surface the intrinsics estimated by the two-view solver (if any) so the caller can
		// seed the cameras before registering the initial pair.
		info.Camera1 = twoViewGeometry.Camera1;
		info.Camera2 = twoViewGeometry.Camera2;

		return info;
	}

	private static bool EstimateInitialGeneralizedTwoViewGeometry(
		IncrementalMapper.Options options,
		DatabaseCache databaseCache,
		Image origImage1,
		Image origImage2,
		Frame frame1,
		Frame frame2,
		Rig rig1,
		Rig rig2,
		out Rigid3d origCam2FromOrigCam1)
	{
		origCam2FromOrigCam1 = Rigid3d.Identity;

		var points2D1 = new List<Vector2d>();
		var points2D2 = new List<Vector2d>();
		var cameraIdxs1 = new List<int>();
		var cameraIdxs2 = new List<int>();
		var camsFromRig = new List<Rigid3d>();
		var cameras = new List<Camera>();

		var cameraIdToIdx = new Dictionary<uint, int>();
		int MaybeAddCamera(Rig rig, Camera camera)
		{
			if (cameraIdToIdx.TryGetValue(camera.CameraId, out int idx))
			{
				return idx;
			}

			idx = cameras.Count;
			cameraIdToIdx.Add(camera.CameraId, idx);
			cameras.Add(camera);
			camsFromRig.Add(rig.IsRefSensor(camera.SensorId) ? Rigid3d.Identity : rig.SensorFromRig(camera.SensorId));
			return idx;
		}

		var matches = new List<FeatureMatch>();
		foreach (DataId imageId1 in frame1.ImageIds())
		{
			Image image1 = databaseCache.Image((uint)imageId1.Id);
			Camera camera1 = databaseCache.Camera(image1.CameraId);
			int cameraIdx1 = MaybeAddCamera(rig1, camera1);

			foreach (DataId imageId2 in frame2.ImageIds())
			{
				Image image2 = databaseCache.Image((uint)imageId2.Id);
				Camera camera2 = databaseCache.Camera(image2.CameraId);
				int cameraIdx2 = MaybeAddCamera(rig2, camera2);

				databaseCache.CorrespondenceGraph.ExtractMatchesBetweenImages(
					(uint)imageId1.Id, (uint)imageId2.Id, matches);
				foreach (FeatureMatch match in matches)
				{
					points2D1.Add(image1.Points2D[(int)match.Point2DIdx1].Xy);
					points2D2.Add(image2.Points2D[(int)match.Point2DIdx2].Xy);
					cameraIdxs1.Add(cameraIdx1);
					cameraIdxs2.Add(cameraIdx2);
				}
			}
		}

		var ransacOptions = new RansacOptions
		{
			MinNumTrials = 30,
			RandomSeed = options.RandomSeed,
			MaxError = options.InitMaxError,
		};

		Rigid3d? maybeRig2FromRig1 = null;
		Rigid3d? maybePano2FromPano1 = null;
		if (!GeneralizedPoseEstimation.EstimateGeneralizedRelativePose(
			ransacOptions,
			points2D1,
			points2D2,
			cameraIdxs1,
			cameraIdxs2,
			camsFromRig,
			cameras,
			ref maybeRig2FromRig1,
			ref maybePano2FromPano1,
			out int numInliers,
			out bool[] _))
		{
			return false;
		}

		// Note that we already checked for stable geometry (i.e., non-forward motion,
		// sufficient triangulation angle) between the original image pair.
		if (numInliers < options.InitMinNumInliers)
		{
			return false;
		}

		Rigid3d rig2FromRig1 = maybeRig2FromRig1 ?? maybePano2FromPano1!.Value;

		// Recompose the relative transformation between the original images.
		var origCameraId1 = new SensorId(SensorType.Camera, origImage1.CameraId);
		Rigid3d origCam1FromRig1 = rig1.IsRefSensor(origCameraId1) ? Rigid3d.Identity : rig1.SensorFromRig(origCameraId1);

		var origCameraId2 = new SensorId(SensorType.Camera, origImage2.CameraId);
		Rigid3d origCam2FromRig2 = rig2.IsRefSensor(origCameraId2) ? Rigid3d.Identity : rig2.SensorFromRig(origCameraId2);

		origCam2FromOrigCam1 = origCam2FromRig2 * rig2FromRig1 * origCam1FromRig1.Inverse();

		return true;
	}
}
