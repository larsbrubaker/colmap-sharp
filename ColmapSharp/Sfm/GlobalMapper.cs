// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GlobalMapper: port of colmap/sfm/global_mapper.h and .cc, global Structure-from-Motion
// (GLOMAP). Instead of registering images one by one like IncrementalMapper, it estimates all
// frame rotations at once (Estimators/RotationAveraging.cs), builds tracks from the pose
// graph's matches, estimates all positions and points at once (Estimators/GlobalPositioning.cs),
// then refines with iterative bundle adjustment and a final retriangulation pass that reuses
// IncrementalMapper's triangulation and global refinement. Options: GlobalMapperOptions.cs.
// This file holds the lifecycle, rotation averaging, track establishment and global
// positioning; GlobalMapper.Refine.cs holds the bundle adjustment, retriangulation and Solve.
// Tests: ColmapSharp.Tests/Sfm/GlobalMapperTests.cs (global_mapper_test.cc 1:1).
//
// Tier C (outcome): every stage runs a solver or a randomized estimator; the tests compare
// with the synthetic ground truth through ReconstructionNear at COLMAP's bounds.
//
// Translation notes:
// - EstablishTracks walks the pose graph's valid edges in pair-id order, and the tracks in
//   the order their first observation entered the union-find (UnionFind.Parents iterates in
//   insertion order). COLMAP walks hash maps, so the 3D point ids and the order of track
//   elements can differ from COLMAP's; the set of tracks is the same
//   (divergence 100).
// - LOG(ERROR) goes to Util/Log.cs; LOG(INFO) / VLOG and the stage timers are dropped.
// - shared_ptr members are references; THROW_CHECK_NOTNULL is Check.NotNull.

using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Sfm;

/// <summary>
/// Port of colmap::GlobalMapper: global SfM over a database cache. Typical use:
/// BeginReconstruction, then Solve (or the individual stages in the same order).
/// </summary>
public sealed partial class GlobalMapper
{
	private readonly DatabaseCache databaseCache;
	private PoseGraph? poseGraph;
	private Reconstruction? reconstruction;

	/// <summary>Creates a mapper over <paramref name="databaseCache"/>.</summary>
	public GlobalMapper(DatabaseCache databaseCache)
	{
		this.databaseCache = Check.NotNull(databaseCache);
	}

	/// <summary>The current reconstruction; null before BeginReconstruction.</summary>
	public Reconstruction? Reconstruction => reconstruction;

	/// <summary>
	/// Prepares the mapper for a new reconstruction: loads the reconstruction and the pose
	/// graph from the database cache.
	/// </summary>
	public void BeginReconstruction(Reconstruction reconstruction)
	{
		Check.NotNull(reconstruction);
		this.reconstruction = reconstruction;
		this.reconstruction.Load(databaseCache);
		poseGraph = new PoseGraph();
		poseGraph.Load(databaseCache.CorrespondenceGraph);
	}

	/// <summary>Runs rotation averaging to estimate global rotations.</summary>
	public bool RotationAveraging(RotationEstimatorOptions options)
	{
		Reconstruction recon = Check.NotNull(reconstruction);
		PoseGraph graph = Check.NotNull(poseGraph);

		if (graph.Empty)
		{
			Log.Error("Cannot continue with empty pose graph");
			return false;
		}

		// Read pose priors from the database cache.
		IReadOnlyList<PosePrior> posePriors = databaseCache.PosePriors;

		// First pass: solve rotation averaging on all frames, then filter outlier pairs by
		// rotation error and de-register frames outside the largest connected component.
		RotationEstimatorOptions customOptions = options.Clone();
		customOptions.FilterUnregistered = false;
		if (!Estimators.RotationAveraging.RunRotationAveraging(customOptions, graph, recon, posePriors))
		{
			return false;
		}

		// Second pass: re-solve on registered frames only to refine rotations after outlier
		// removal.
		customOptions.FilterUnregistered = true;
		return Estimators.RotationAveraging.RunRotationAveraging(customOptions, graph, recon, posePriors);
	}

	/// <summary>Establishes tracks from the feature matches of the valid pose graph edges.</summary>
	public void EstablishTracks(GlobalMapperOptions options)
	{
		Reconstruction recon = Check.NotNull(reconstruction);
		PoseGraph graph = Check.NotNull(poseGraph);
		Check.Eq(recon.NumPoints3D, 0);

		// Build keypoints map from registered images.
		var imageIdToKeypoints = new Dictionary<uint, List<Vector2d>>();
		foreach (uint imageId in recon.RegImageIds())
		{
			Image image = recon.Image(imageId);
			var points = new List<Vector2d>((int)image.NumPoints2D);
			foreach (Point2D point2D in image.Points2D)
			{
				points.Add(point2D.Xy);
			}

			imageIdToKeypoints.Add(imageId, points);
		}

		CorrespondenceGraph corrGraph = databaseCache.CorrespondenceGraph;

		// Union all matching observations. Pair-id order makes the union-find's insertion
		// order, and so the track order below, deterministic (entry 100).
		var uf = new UnionFind<(uint ImageId, uint Point2DIdx)>();
		var matches = new List<FeatureMatch>();
		foreach ((ulong pairId, _) in graph.ValidEdges().OrderBy(kv => kv.Key))
		{
			(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
			Check.That(imageIdToKeypoints.ContainsKey(imageId1), $"Missing keypoints for image {imageId1}");
			Check.That(imageIdToKeypoints.ContainsKey(imageId2), $"Missing keypoints for image {imageId2}");
			corrGraph.ExtractMatchesBetweenImages(imageId1, imageId2, matches);
			foreach (FeatureMatch match in matches)
			{
				var obs1 = (imageId1, match.Point2DIdx1);
				var obs2 = (imageId2, match.Point2DIdx2);
				// std::pair's operator< is lexicographic, as is ValueTuple's CompareTo.
				if (obs2.CompareTo(obs1) < 0)
				{
					uf.Union(obs1, obs2);
				}
				else
				{
					uf.Union(obs2, obs1);
				}
			}
		}

		// Group observations by their root. The Dictionary never removes, so it iterates in
		// insertion order.
		uf.Compress();
		var trackMap = new Dictionary<(uint ImageId, uint Point2DIdx), List<(uint ImageId, uint Point2DIdx)>>();
		foreach ((var obs, var root) in uf.Parents)
		{
			if (!trackMap.TryGetValue(root, out var observations))
			{
				observations = [];
				trackMap.Add(root, observations);
			}

			observations.Add(obs);
		}

		// Validate tracks, check consistency, and collect valid ones with lengths.
		var candidatePoints3D = new Dictionary<ulong, Point3D>();
		var trackLengths = new List<(int Length, ulong Point3DId)>();
		ulong nextPoint3DId = 0;
		double sqThreshold = options.TrackIntraImageConsistencyThreshold * options.TrackIntraImageConsistencyThreshold;
		// C++ compares with static_cast<size_t> of the int options: a negative value wraps to
		// a huge limit (a negative minimum views per track discards every track, a negative
		// required tracks per view keeps adding tracks).
		ulong minNumViewsPerTrack = unchecked((ulong)options.TrackMinNumViewsPerTrack);

		foreach (List<(uint ImageId, uint Point2DIdx)> observations in trackMap.Values)
		{
			var imageIdSet = new Dictionary<uint, List<Vector2d>>();
			var point3D = new Point3D();
			bool isConsistent = true;

			foreach ((uint imageId, uint featureId) in observations)
			{
				Vector2d xy = imageIdToKeypoints[imageId][(int)featureId];

				if (imageIdSet.TryGetValue(imageId, out List<Vector2d>? existing))
				{
					foreach (Vector2d existingXy in existing)
					{
						if ((existingXy - xy).SquaredNorm > sqThreshold)
						{
							isConsistent = false;
							break;
						}
					}

					if (!isConsistent)
					{
						break;
					}

					existing.Add(xy);
				}
				else
				{
					imageIdSet.Add(imageId, [xy]);
				}

				point3D.Track.AddElement(imageId, featureId);
			}

			if (!isConsistent)
			{
				continue;
			}

			if ((ulong)imageIdSet.Count < minNumViewsPerTrack)
			{
				continue;
			}

			ulong point3DId = nextPoint3DId++;
			trackLengths.Add((point3D.Track.Length, point3DId));
			candidatePoints3D.Add(point3DId, point3D);
		}

		// Sort tracks by length (descending) and select for problem. The ids are unique, so
		// the (length, id) keys have no ties and the unstable sort is deterministic.
		trackLengths.Sort((a, b) => b.CompareTo(a));

		var tracksPerImage = new Dictionary<uint, ulong>();
		long imagesLeft = imageIdToKeypoints.Count;
		// static_cast<size_t> of the int options, as above.
		ulong maxNumTracks = unchecked((ulong)options.KeepMaxNumTracks);
		ulong requiredTracksPerView = unchecked((ulong)options.TrackRequiredTracksPerView);
		foreach ((_, ulong point3DId) in trackLengths)
		{
			// Stop once the global track budget is exhausted. As tracks are sorted by
			// decreasing length, this keeps the longest tracks and bounds memory usage.
			if ((ulong)recon.NumPoints3D >= maxNumTracks)
			{
				break;
			}

			Point3D point3D = candidatePoints3D[point3DId];

			// Check if any image in this track still needs more observations. C++ reads the
			// counts with operator[], which inserts zeros; a missing key reads as zero here.
			bool shouldAdd = point3D.Track.Elements.Any(
				obs => tracksPerImage.GetValueOrDefault(obs.ImageId) <= requiredTracksPerView);
			if (!shouldAdd)
			{
				continue;
			}

			// Update image counts.
			foreach (TrackElement obs in point3D.Track.Elements)
			{
				ulong count = tracksPerImage.GetValueOrDefault(obs.ImageId);
				if (count == requiredTracksPerView)
				{
					--imagesLeft;
				}

				tracksPerImage[obs.ImageId] = count + 1;
			}

			recon.AddPoint3D(point3DId, point3D);

			if (imagesLeft == 0)
			{
				break;
			}
		}
	}

	/// <summary>
	/// Estimates global camera positions and points, then filters the tracks by angular
	/// error (relaxed for cameras without a prior focal length), triangulation angle and
	/// normalized reprojection error, and normalizes the reconstruction.
	/// </summary>
	public bool GlobalPositioning(
		GlobalPositionerOptions options,
		double maxAngularReprojErrorDeg,
		double maxNormalizedReprojError,
		double minTriAngleDeg)
	{
		Reconstruction recon = Check.NotNull(reconstruction);
		if (!Estimators.GlobalPositioning.RunGlobalPositioning(options, Check.NotNull(poseGraph), recon))
		{
			return false;
		}

		// Filter tracks based on the estimation
		var obsManager = new ObservationManager(recon);

		// First pass: use relaxed threshold (2x) for cameras without prior focal.
		obsManager.FilterPoints3DWithLargeReprojectionError(
			2.0 * maxAngularReprojErrorDeg, recon.Point3DIds(), ReprojectionErrorType.Angular);

		// Second pass: apply strict threshold for cameras with prior focal length.
		double maxAngularErrorRad = MathUtils.DegToRad(maxAngularReprojErrorDeg);
		var obsToDelete = new List<(uint ImageId, uint Point2DIdx)>();
		foreach (ulong point3DId in recon.Point3DIds())
		{
			if (!recon.ExistsPoint3D(point3DId))
			{
				continue;
			}

			Point3D point3D = recon.Point3D(point3DId);
			foreach (TrackElement trackEl in point3D.Track.Elements)
			{
				Image image = recon.Image(trackEl.ImageId);
				Camera camera = image.CameraPtr;
				if (!camera.HasPriorFocalLength)
				{
					continue;
				}

				Point2D point2D = image.Points2D[(int)trackEl.Point2DIdx];
				double error = Projection.CalculateAngularReprojectionError(
					point2D.Xy, point3D.Xyz, image.CamFromWorld(), camera);
				if (error > maxAngularErrorRad)
				{
					obsToDelete.Add((trackEl.ImageId, trackEl.Point2DIdx));
				}
			}
		}

		foreach ((uint imageId, uint point2DIdx) in obsToDelete)
		{
			if (recon.Image(imageId).Points2D[(int)point2DIdx].HasPoint3D)
			{
				obsManager.DeleteObservation(imageId, point2DIdx);
			}
		}

		// Filter tracks based on triangulation angle and reprojection error
		obsManager.FilterPoints3DWithSmallTriangulationAngle(minTriAngleDeg, recon.Point3DIds());
		// Set the threshold to be larger to avoid removing too many tracks
		obsManager.FilterPoints3DWithLargeReprojectionError(
			10 * maxNormalizedReprojError, recon.Point3DIds(), ReprojectionErrorType.Normalized);

		// Normalize the structure for numerical stability.
		// TODO (COLMAP): Skip normalization when position priors are used (similar to
		// incremental mapper's !use_prior_position condition).
		recon.Normalize();

		return true;
	}
}
