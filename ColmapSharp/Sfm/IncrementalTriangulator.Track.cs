// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalTriangulator.Track: the private per-track steps of
// colmap/sfm/incremental_triangulator.cc (TriangulateTrack, Find, Create, Continue, Merge,
// Complete, HasCameraBogusParams), used by the public operations in
// IncrementalTriangulator.cs. See that file's header for the translation notes.

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

namespace ColmapSharp.Sfm;

public sealed partial class IncrementalTriangulator
{
	// Enforce exhaustive sampling for track lengths up to this.
	private const int kExhaustiveSamplingThreshold = 15;

	// Need at least this many leftover observations to recursively create another point.
	private const int kMinRecursiveTrackLength = 3;

	// Reused inputs of TriangulateTrack.
	private readonly List<Vector2d> _triPoints = [];
	private readonly List<Rigid3d> _triCamsFromWorld = [];
	private readonly List<Camera> _triCameras = [];

	// Port of the anonymous-namespace TriangulateTrack of incremental_triangulator.cc.
	private bool TriangulateTrack(
		EstimateTriangulationOptions options,
		List<CorrData> corrsData,
		out bool[] inlierMask,
		out Vector3d xyz)
	{
		_triPoints.Clear();
		_triCamsFromWorld.Clear();
		_triCameras.Clear();
		foreach (CorrData corrData in corrsData)
		{
			_triPoints.Add(corrData.Point2D.Xy);
			_triCamsFromWorld.Add(corrData.Image.CamFromWorld());
			_triCameras.Add(corrData.Camera);
		}

		// Enforce exhaustive sampling for small track lengths. C++ changes a copy of the
		// options; the port sets and restores the caller's, which CompleteImage reuses.
		int minNumTrials = options.RansacOptions.MinNumTrials;
		if (_triPoints.Count <= kExhaustiveSamplingThreshold)
		{
			options.RansacOptions.MinNumTrials = (int)MathUtils.NChooseK((ulong)_triPoints.Count, 2);
		}

		try
		{
			return TriangulationEstimation.EstimateTriangulation(
				options, _triPoints, _triCamsFromWorld, _triCameras, out inlierMask, out xyz);
		}
		finally
		{
			options.RansacOptions.MinNumTrials = minNumTrials;
		}
	}

	// Finds (transitive) correspondences to other images that have a pose and a sane camera.
	// Returns how many of them are already triangulated.
	private int Find(Options options, uint imageId, uint point2DIdx, int transitivity, List<CorrData> corrsData)
	{
		_correspondenceGraph.ExtractTransitiveCorrespondences(imageId, point2DIdx, transitivity, _foundCorrs);

		corrsData.Clear();

		int numTriangulated = 0;

		foreach (CorrespondenceGraph.Correspondence corr in _foundCorrs)
		{
			Image corrImage = _reconstruction.Image(corr.ImageId);
			if (!corrImage.HasPose)
			{
				continue;
			}

			Camera corrCamera = corrImage.CameraPtr;
			if (HasCameraBogusParams(options, corrCamera))
			{
				continue;
			}

			var corrData = new CorrData(corr.ImageId, corr.Point2DIdx, corrImage, corrCamera);
			corrsData.Add(corrData);

			if (corrData.Point2D.HasPoint3D)
			{
				numTriangulated += 1;
			}
		}

		return numTriangulated;
	}

	// Tries to create a new 3D point from the given correspondences.
	private int Create(Options options, List<CorrData> corrsData, int depth = 0)
	{
		// Extract correspondences without an existing triangulated observation. Each recursion
		// level filters its parent's list, so it needs its own buffer.
		if (_createCorrsDataByDepth.Count == depth)
		{
			_createCorrsDataByDepth.Add([]);
		}

		List<CorrData> createCorrsData = _createCorrsDataByDepth[depth];
		createCorrsData.Clear();
		foreach (CorrData corrData in corrsData)
		{
			if (!corrData.Point2D.HasPoint3D)
			{
				createCorrsData.Add(corrData);
			}
		}

		if (createCorrsData.Count < 2)
		{
			// Need at least two observations for triangulation.
			return 0;
		}
		else if (options.IgnoreTwoViewTracks && createCorrsData.Count == 2)
		{
			CorrData corrData1 = createCorrsData[0];
			if (_correspondenceGraph.IsTwoViewObservation(corrData1.ImageId, corrData1.Point2DIdx))
			{
				return 0;
			}
		}

		// Setup estimation options.
		var triOptions = new EstimateTriangulationOptions
		{
			MinTriAngle = MathUtils.DegToRad(options.MinAngle),
			ResidualType = TriangulationEstimator.ResidualType.AngularError,
		};
		triOptions.RansacOptions.MaxError = MathUtils.DegToRad(options.CreateMaxAngleError);
		triOptions.RansacOptions.RandomSeed = options.RandomSeed;

		// Estimate triangulation.
		if (!TriangulateTrack(triOptions, createCorrsData, out bool[] inlierMask, out Vector3d xyz))
		{
			return 0;
		}

		// Add inliers to estimated track.
		var track = new Track();
		track.Elements.Capacity = createCorrsData.Count;
		for (int i = 0; i < inlierMask.Length; ++i)
		{
			if (inlierMask[i])
			{
				CorrData corrData = createCorrsData[i];
				track.AddElement(corrData.ImageId, corrData.Point2DIdx);
			}
		}

		// Add estimated point to reconstruction.
		int trackLength = track.Length;
		ulong point3DId = _obsManager.AddPoint3D(xyz, track);
		_modifiedPoint3DIds.Add(point3DId);

		if (createCorrsData.Count - trackLength >= kMinRecursiveTrackLength)
		{
			return trackLength + Create(options, createCorrsData, depth + 1);
		}

		return trackLength;
	}

	// Tries to continue the 3D point of the best-fitting correspondence with the reference
	// observation. Returns 1 if the reference observation was added to a track.
	private int Continue(Options options, in CorrData refCorrData, List<CorrData> corrsData)
	{
		// No need to continue, if the reference observation is triangulated.
		if (refCorrData.Point2D.HasPoint3D)
		{
			return 0;
		}

		double bestAngleError = double.MaxValue;
		int bestIdx = -1;

		Vector2d refXy = refCorrData.Point2D.Xy;
		Rigid3d refCamFromWorld = refCorrData.Image.CamFromWorld();

		for (int idx = 0; idx < corrsData.Count; ++idx)
		{
			ref Point2D corrPoint2D = ref corrsData[idx].Point2D;
			if (!corrPoint2D.HasPoint3D)
			{
				continue;
			}

			Point3D point3D = _reconstruction.Point3D(corrPoint2D.Point3DId);

			double angleError = Projection.CalculateAngularReprojectionError(
				refXy, point3D.Xyz, refCamFromWorld, refCorrData.Camera);
			if (angleError < bestAngleError)
			{
				bestAngleError = angleError;
				bestIdx = idx;
			}
		}

		double maxAngleError = MathUtils.DegToRad(options.ContinueMaxAngleError);
		if (bestAngleError <= maxAngleError && bestIdx != -1)
		{
			ulong point3DId = corrsData[bestIdx].Point2D.Point3DId;
			_obsManager.AddObservation(point3DId, new TrackElement(refCorrData.ImageId, refCorrData.Point2DIdx));
			_modifiedPoint3DIds.Add(point3DId);
			return 1;
		}

		return 0;
	}

	// Tries to merge a 3D point with any of its corresponding 3D points, recursively.
	// Returns the number of merged observations.
	private int Merge(Options options, ulong point3DId)
	{
		if (!_reconstruction.ExistsPoint3D(point3DId))
		{
			return 0;
		}

		double maxSquaredReprojError = options.MergeMaxReprojError * options.MergeMaxReprojError;

		Point3D point3D = _reconstruction.Point3D(point3DId);

		foreach (TrackElement trackEl in point3D.Track.Elements)
		{
			foreach (CorrespondenceGraph.Correspondence corr in _correspondenceGraph.FindCorrespondences(trackEl.ImageId, trackEl.Point2DIdx))
			{
				Image image = _reconstruction.Image(corr.ImageId);
				if (!image.HasPose)
				{
					continue;
				}

				ulong corrPoint3DId = image.Point2DAt(corr.Point2DIdx).Point3DId;
				if (corrPoint3DId == Util.Types.InvalidPoint3DId || corrPoint3DId == point3DId)
				{
					continue;
				}

				// Canonical (min, max) pair so this merge is keyed identically regardless of
				// which side of the pair we are visiting from.
				(ulong, ulong) mergeTrialKey = point3DId < corrPoint3DId ? (point3DId, corrPoint3DId) : (corrPoint3DId, point3DId);
				if (!_mergeTrials.Add(mergeTrialKey))
				{
					continue;
				}

				// Try to merge the two 3D points.

				Point3D corrPoint3D = _reconstruction.Point3D(corrPoint3DId);

				// Weighted average of point locations, depending on track length.
				Vector3d mergedXyz =
					((point3D.Track.Length * point3D.Xyz) + (corrPoint3D.Track.Length * corrPoint3D.Xyz))
					/ (point3D.Track.Length + corrPoint3D.Track.Length);

				// Only accept the merge if all track elements of both tracks are inliers.
				if (AllTrackElementsWithin(point3D.Track, mergedXyz, maxSquaredReprojError)
					&& AllTrackElementsWithin(corrPoint3D.Track, mergedXyz, maxSquaredReprojError))
				{
					int numMerged = point3D.Track.Length + corrPoint3D.Track.Length;

					ulong mergedPoint3DId = _obsManager.MergePoints3D(point3DId, corrPoint3DId);

					_modifiedPoint3DIds.Remove(point3DId);
					_modifiedPoint3DIds.Remove(corrPoint3DId);
					_modifiedPoint3DIds.Add(mergedPoint3DId);

					// Merge the merged 3D point and return, as the original points are deleted
					// (so neither enumeration above is touched again).
					int numMergedRecursive = Merge(options, mergedPoint3DId);
					return numMergedRecursive > 0 ? numMergedRecursive : numMerged;
				}
			}
		}

		return 0;
	}

	// Whether every observation of a track reprojects within the threshold at xyz (Merge's
	// inner loop, which C++ breaks out of at the first outlier).
	private bool AllTrackElementsWithin(Track track, Vector3d xyz, double maxSquaredReprojError)
	{
		foreach (TrackElement testTrackEl in track.Elements)
		{
			Image testImage = _reconstruction.Image(testTrackEl.ImageId);
			Vector2d testXy = testImage.Point2DAt(testTrackEl.Point2DIdx).Xy;
			if (Projection.CalculateSquaredReprojectionError(testXy, xyz, testImage.CamFromWorld(), testImage.CameraPtr)
				> maxSquaredReprojError)
			{
				return false;
			}
		}

		return true;
	}

	// Tries to transitively complete the track of a 3D point (breadth first, one level per
	// transitivity step). Returns the number of added observations.
	private int Complete(Options options, ulong point3DId)
	{
		int numCompleted = 0;

		if (!_reconstruction.ExistsPoint3D(point3DId))
		{
			return numCompleted;
		}

		double maxSquaredReprojError = options.CompleteMaxReprojError * options.CompleteMaxReprojError;

		Point3D point3D = _reconstruction.Point3D(point3DId);

		// Reuse the BFS scratch buffers across calls. The queue is a copy of the track, which
		// AddObservation grows below.
		_completeCurrQueue.Clear();
		_completeCurrQueue.AddRange(point3D.Track.Elements);
		_completeNextQueue.Clear();
		_completeVisited.Clear();

		// Seed visited with the existing track members so the BFS never tries to re-add them.
		foreach (TrackElement el in _completeCurrQueue)
		{
			_completeVisited.Add(VisitedKey(el.ImageId, el.Point2DIdx));
		}

		int maxTransitivity = options.CompleteMaxTransitivity;
		for (int transitivity = 1; transitivity <= maxTransitivity; ++transitivity)
		{
			while (_completeCurrQueue.Count > 0)
			{
				TrackElement queueElem = _completeCurrQueue[^1];
				_completeCurrQueue.RemoveAt(_completeCurrQueue.Count - 1);

				foreach (CorrespondenceGraph.Correspondence corr in _correspondenceGraph.FindCorrespondences(queueElem.ImageId, queueElem.Point2DIdx))
				{
					// Two queue entries at the same transitivity level can share
					// correspondences. Dedupe before the (more expensive) reprojection check.
					if (!_completeVisited.Add(VisitedKey(corr.ImageId, corr.Point2DIdx)))
					{
						continue;
					}

					Image image = _reconstruction.Image(corr.ImageId);
					if (!image.HasPose)
					{
						continue;
					}

					Point2D point2D = image.Point2DAt(corr.Point2DIdx);
					if (point2D.HasPoint3D)
					{
						continue;
					}

					Camera camera = image.CameraPtr;
					if (HasCameraBogusParams(options, camera))
					{
						continue;
					}

					if (Projection.CalculateSquaredReprojectionError(point2D.Xy, point3D.Xyz, image.CamFromWorld(), camera)
						> maxSquaredReprojError)
					{
						continue;
					}

					// Success, add observation to point track.
					_obsManager.AddObservation(point3DId, new TrackElement(corr.ImageId, corr.Point2DIdx));
					_modifiedPoint3DIds.Add(point3DId);

					// Recursively complete track for this new correspondence.
					if (transitivity < maxTransitivity)
					{
						_completeNextQueue.Add(new TrackElement(corr.ImageId, corr.Point2DIdx));
					}

					numCompleted += 1;
				}
			}

			if (_completeNextQueue.Count == 0)
			{
				break;
			}

			(_completeCurrQueue, _completeNextQueue) = (_completeNextQueue, _completeCurrQueue);
		}

		return numCompleted;
	}

	private static ulong VisitedKey(uint imageId, uint point2DIdx) => ((ulong)imageId << 32) | point2DIdx;

	// Whether a camera has bogus parameters, cached per camera id until the next ClearCaches.
	private bool HasCameraBogusParams(Options options, Camera camera)
	{
		if (!_cameraHasBogusParams.TryGetValue(camera.CameraId, out bool hasBogusParams))
		{
			hasBogusParams = camera.HasBogusParams(options.MinFocalLengthRatio, options.MaxFocalLengthRatio, options.MaxExtraParam);
			_cameraHasBogusParams.Add(camera.CameraId, hasBogusParams);
		}

		return hasBogusParams;
	}
}
