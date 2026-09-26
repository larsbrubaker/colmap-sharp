// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ObservationManager (filters): the Filter* methods and FindFramesToFilter of
// colmap/sfm/observation_manager.cc. They delete 3D points and observations through the
// bookkeeping operations in ObservationManager.cs so the statistics stay in sync. Tests:
// ColmapSharp.Tests/Sfm/ObservationManagerTests.cs (observation_manager_test.cc 1:1).
//
// Tier A (exact): the reprojection and triangulation-angle tests are the same scalar
// expressions as COLMAP's (Scene/Projection.cs, Geometry/Triangulation.cs).
//
// Translation notes:
// - COLMAP passes FlatHashSets of point/image ids and iterates them in hash order. Here the
//   sets are the caller's IReadOnlySet, iterated in its own order. The order cannot change
//   any result: filtering one point only deletes that point or its own observations, the
//   counters are sums, and the projection-center cache holds per-image constants.
// - The free function MergeAndFilterReconstructions is a static method here; it merges with
//   Estimators/Alignment.Merge.cs and then filters through a fresh ObservationManager. COLMAP
//   has no test for it; ColmapSharp.Tests/Sfm/ObservationManagerTests.Merge.cs is C#-only.

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Sfm;

public sealed partial class ObservationManager
{
	/// <summary>
	/// Merges <paramref name="srcReconstruction"/> into <paramref name="tgtReconstruction"/>
	/// (Alignment.MergeReconstructions) and then filters every 3D point observation of the
	/// target whose reprojection error exceeds <paramref name="maxReprojError"/> (no
	/// triangulation-angle filter). Returns false, leaving the target untouched, when the
	/// merge fails. Port of colmap::MergeAndFilterReconstructions.
	/// </summary>
	public static bool MergeAndFilterReconstructions(
		double maxReprojError, Reconstruction srcReconstruction, Reconstruction tgtReconstruction)
	{
		if (!Alignment.MergeReconstructions(maxReprojError, srcReconstruction, tgtReconstruction))
		{
			return false;
		}

		new ObservationManager(tgtReconstruction).FilterAllPoints3D(maxReprojError, minTriAngle: 0);
		return true;
	}

	/// <summary>
	/// Filters 3D points with large reprojection error (in pixels) or insufficient
	/// triangulation angle (in degrees). Returns the number of filtered observations.
	/// </summary>
	public int FilterPoints3D(double maxReprojError, double minTriAngle, IReadOnlySet<ulong> point3DIds)
	{
		int numFilteredObservations = 0;
		numFilteredObservations += FilterPoints3DWithLargeReprojectionError(maxReprojError, point3DIds);
		numFilteredObservations += FilterPoints3DWithSmallTriangulationAngle(minTriAngle, point3DIds);
		return numFilteredObservations;
	}

	/// <summary>
	/// <see cref="FilterPoints3D"/> for every 3D point observed in the given images. Returns
	/// the number of filtered observations.
	/// </summary>
	public int FilterPoints3DInImages(double maxReprojError, double minTriAngle, IReadOnlySet<uint> imageIds)
	{
		var point3DIds = new HashSet<ulong>();
		foreach (uint imageId in imageIds)
		{
			Image image = _reconstruction.Image(imageId);
			foreach (Point2D point2D in image.Points2D)
			{
				if (point2D.HasPoint3D)
				{
					point3DIds.Add(point2D.Point3DId);
				}
			}
		}

		return FilterPoints3D(maxReprojError, minTriAngle, point3DIds);
	}

	/// <summary>
	/// <see cref="FilterPoints3D"/> for every 3D point. Returns the number of filtered
	/// observations.
	/// </summary>
	public int FilterAllPoints3D(double maxReprojError, double minTriAngle)
	{
		// Important: First filter observations and points with large reprojection error, so
		// that observations with large reprojection error do not make a point stable through
		// a large triangulation angle.
		HashSet<ulong> point3DIds = _reconstruction.Point3DIds();
		int numFilteredObservations = 0;
		numFilteredObservations += FilterPoints3DWithLargeReprojectionError(maxReprojError, point3DIds);
		numFilteredObservations += FilterPoints3DWithSmallTriangulationAngle(minTriAngle, point3DIds);
		return numFilteredObservations;
	}

	/// <summary>
	/// Filters points with a track shorter than <paramref name="minTrackLength"/>. Returns
	/// the number of filtered observations.
	/// </summary>
	public int FilterPoints3DWithShortTracks(int minTrackLength)
	{
		int numFilteredObservations = 0;
		foreach (ulong point3DId in _reconstruction.Point3DIds())
		{
			Point3D point3D = _reconstruction.Point3D(point3DId);
			if (point3D.Track.Length < minTrackLength)
			{
				numFilteredObservations += point3D.Track.Length;
				DeletePoint3D(point3DId);
			}
		}

		return numFilteredObservations;
	}

	/// <summary>
	/// Filters observations of registered, non-spherical images whose point lies at or
	/// behind the camera. Returns the number of filtered observations.
	/// </summary>
	public int FilterObservationsWithNegativeDepth()
	{
		int numFiltered = 0;
		foreach (uint frameId in _reconstruction.RegFrameIds)
		{
			foreach (DataId dataId in _reconstruction.Frame(frameId).ImageIds())
			{
				Image image = _reconstruction.Image((uint)dataId.Id);
				if (image.CameraPtr.IsSpherical)
				{
					continue;
				}

				Matrix3x4d camFromWorld = image.CamFromWorld().ToMatrix();
				for (uint point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
				{
					Point2D point2D = image.Point2DAt(point2DIdx);
					if (point2D.HasPoint3D)
					{
						Point3D point3D = _reconstruction.Point3D(point2D.Point3DId);
						if (!Projection.HasPointPositiveDepth(camFromWorld, point3D.Xyz))
						{
							DeleteObservation((uint)dataId.Id, point2DIdx);
							numFiltered += 1;
						}
					}
				}
			}
		}

		return numFiltered;
	}

	/// <summary>
	/// Deletes the given points unless some pair of their track's images sees them under at
	/// least <paramref name="minTriAngle"/> degrees. Returns the number of filtered
	/// observations.
	/// </summary>
	public int FilterPoints3DWithSmallTriangulationAngle(double minTriAngle, IReadOnlySet<ulong> point3DIds)
	{
		// Number of filtered observations.
		int numFilteredObservations = 0;

		// Minimum triangulation angle in radians.
		double minTriAngleRad = MathUtils.DegToRad(minTriAngle);

		// Cache for image projection centers.
		var projCenters = new Dictionary<uint, Vector3d>();

		foreach (ulong point3DId in point3DIds)
		{
			if (!_reconstruction.ExistsPoint3D(point3DId))
			{
				continue;
			}

			Point3D point3D = _reconstruction.Point3D(point3DId);
			Vector3d xyz = point3D.Xyz;

			// Calculate triangulation angle for all pairwise combinations of image poses in
			// the track. Only delete point if none of the combinations has a sufficient
			// triangulation angle.
			bool keepPoint = false;
			for (int i1 = 0; i1 < point3D.Track.Length; ++i1)
			{
				uint imageId1 = point3D.Track.Element(i1).ImageId;

				if (!projCenters.TryGetValue(imageId1, out Vector3d projCenter1))
				{
					projCenter1 = _reconstruction.Image(imageId1).ProjectionCenter();
					projCenters.Add(imageId1, projCenter1);
				}

				for (int i2 = 0; i2 < i1; ++i2)
				{
					uint imageId2 = point3D.Track.Element(i2).ImageId;
					Vector3d projCenter2 = projCenters[imageId2];

					double triAngle = Triangulation.CalculateTriangulationAngle(projCenter1, projCenter2, xyz);

					if (triAngle >= minTriAngleRad)
					{
						keepPoint = true;
						break;
					}
				}

				if (keepPoint)
				{
					break;
				}
			}

			if (!keepPoint)
			{
				numFilteredObservations += point3D.Track.Length;
				DeletePoint3D(point3DId);
			}
		}

		return numFilteredObservations;
	}

	/// <summary>
	/// Filters observations with large reprojection error. For PIXEL and NORMALIZED,
	/// <paramref name="maxError"/> is the reprojection error; for ANGULAR, the angular error
	/// in degrees. A point is deleted when at most one of its observations would remain.
	/// Returns the number of filtered observations.
	/// </summary>
	public int FilterPoints3DWithLargeReprojectionError(
		double maxError, IReadOnlySet<ulong> point3DIds, ReprojectionErrorType errorType = ReprojectionErrorType.Pixel)
	{
		int numFilteredObservations = 0;

		// Reused across points: C++ allocates a vector per point.
		var trackElsToDelete = new List<TrackElement>();

		foreach (ulong point3DId in point3DIds)
		{
			if (!_reconstruction.ExistsPoint3D(point3DId))
			{
				continue;
			}

			Point3D point3D = _reconstruction.Point3D(point3DId);

			if (point3D.Track.Length < 2)
			{
				numFilteredObservations += point3D.Track.Length;
				DeletePoint3D(point3DId);
				continue;
			}

			double errorSum = 0.0;
			trackElsToDelete.Clear();
			Vector3d xyz = point3D.Xyz;

			foreach (TrackElement trackEl in point3D.Track.Elements)
			{
				Image image = _reconstruction.Image(trackEl.ImageId);
				Camera camera = image.CameraPtr;
				Vector2d xy = image.Point2DAt(trackEl.Point2DIdx).Xy;

				double observationError = ObservationError(errorType, xy, xyz, image.CamFromWorld(), camera);

				if (observationError > maxError)
				{
					trackElsToDelete.Add(trackEl);
				}
				else
				{
					errorSum += observationError;
				}
			}

			if (trackElsToDelete.Count >= point3D.Track.Length - 1)
			{
				numFilteredObservations += point3D.Track.Length;
				DeletePoint3D(point3DId);
			}
			else
			{
				numFilteredObservations += trackElsToDelete.Count;
				foreach (TrackElement trackEl in trackElsToDelete)
				{
					DeleteObservation(trackEl.ImageId, trackEl.Point2DIdx);
				}

				point3D.Error = errorSum / point3D.Track.Length;
			}
		}

		return numFilteredObservations;
	}

	/// <summary>
	/// Finds frames that should be filtered due to having too few observations or bogus
	/// camera parameters, without de-registering them. Pass them to DeRegisterFrame to reset
	/// their pose. Returns the frame ids in registration order.
	/// </summary>
	public List<uint> FindFramesToFilter(
		double minFocalLengthRatio, double maxFocalLengthRatio, double maxExtraParam, int minNumObservations)
	{
		var frameIds = new List<uint>();
		foreach (uint frameId in _reconstruction.RegFrameIds)
		{
			Frame frame = _reconstruction.Frame(frameId);
			bool bogusCamera = false;
			long numObservations = 0;
			foreach (DataId dataId in frame.ImageIds())
			{
				Image image = _reconstruction.Image((uint)dataId.Id);
				numObservations += image.NumPoints3D;
				if (image.CameraPtr.HasBogusParams(minFocalLengthRatio, maxFocalLengthRatio, maxExtraParam))
				{
					bogusCamera = true;
					break;
				}
			}

			if (bogusCamera || numObservations < minNumObservations)
			{
				frameIds.Add(frameId);
			}
		}

		return frameIds;
	}

	// The error of one observation in the units of the threshold (pixels, normalized chord
	// length, or degrees). Degenerate observations that must always be filtered report an
	// infinite error.
	private static double ObservationError(
		ReprojectionErrorType errorType, Vector2d xy, Vector3d xyz, Rigid3d camFromWorld, Camera camera)
	{
		switch (errorType)
		{
			case ReprojectionErrorType.Pixel:
				return Math.Sqrt(Projection.CalculateSquaredReprojectionError(xy, xyz, camFromWorld, camera));
			case ReprojectionErrorType.Normalized:
			{
				Vector3d point3DInCam = camFromWorld * xyz;
				if (camera.IsPerspective)
				{
					const double kMinDepth = 1e-12;
					Vector2d? camPoint = camera.CamFromImg(xy);
					return point3DInCam.Z >= kMinDepth && camPoint is Vector2d cp
						? (point3DInCam.HNormalized() - cp).Norm
						: double.PositiveInfinity;
				}

				// Omnidirectional cameras (e.g. EQUIRECTANGULAR) have no pinhole z-divide and
				// legitimately observe points behind the local +Z axis, so the cheirality gate
				// and 2D CamFromImg above do not apply. Compare unit bearings instead (chord
				// distance ~= angle for small errors, consistent with the normalized threshold).
				Vector3d? camRay = camera.CamRayFromImg(xy);
				return camRay is Vector3d ray
					? (point3DInCam.Normalized() - ray).Norm
					: double.PositiveInfinity;
			}

			case ReprojectionErrorType.Angular:
				return MathUtils.RadToDeg(Projection.CalculateAngularReprojectionError(xy, xyz, camFromWorld, camera));
			default:
				throw new ArgumentOutOfRangeException(nameof(errorType), errorType, "Unknown ReprojectionErrorType");
		}
	}
}
