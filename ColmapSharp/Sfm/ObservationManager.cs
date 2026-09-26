// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ObservationManager: port of colmap/sfm/observation_manager.h and .cc, the bookkeeping the
// incremental mapper keeps next to a Reconstruction (Scene/Reconstruction.cs) and a
// CorrespondenceGraph (Scene/CorrespondenceGraph.cs): per image, how many of its points have
// correspondences, how many of those correspondences see a registered image or a
// triangulated point, and a VisibilityPyramid (Scene/VisibilityPyramid.cs) scoring how
// evenly those points are spread; per image pair, how many correspondences are
// triangulated. Adding, merging and deleting points and observations goes through here so
// those counts stay in sync. This file holds construction, the statistics and the
// point/observation/frame operations; ObservationManager.Filter.cs holds the Filter*
// methods and FindFramesToFilter. Tests: ColmapSharp.Tests/Sfm/ObservationManagerTests.cs
// (observation_manager_test.cc 1:1).
//
// Tier A (exact): integer bookkeeping, plus the filters' scalar geometry.
//
// Translation notes:
// - Iteration order. COLMAP's stat maps are absl flat hash maps. Here they are Dictionarys
//   that are only ever added to, so they iterate in insertion order: image stats in
//   ascending image id (the reconstruction's IdMap order, docs/CPP_DIVERGENCES.md entry 21)
//   then AddImage order; ImagePairs in the correspondence graph's pair order, then AddImage
//   order (entry 50). No count here depends on that order.
// - C++ `image_stats_[id]` / `image_pair_stats_[id]` insert a default entry when the id is
//   missing; CollectionsMarshal.GetValueRefOrAddDefault does the same here. With no
//   correspondence graph, COLMAP leaves an image's num_observations and num_correspondences
//   uninitialized; they are 0 here.
// - point2D_t counters are uint; size_t counts returned by the filters are int (the unit
//   Track.Length uses). The pyramid score is ulong, as in VisibilityPyramid.
// - Check messages are only formatted on failure: the increment/decrement paths run for
//   every correspondence of every triangulated point.

using System.Runtime.InteropServices;

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Sfm;

/// <summary>Port of colmap::ReprojectionErrorType: the error metric used to filter 3D point observations.</summary>
public enum ReprojectionErrorType
{
	/// <summary>Reprojection error in pixels.</summary>
	Pixel = 0,

	/// <summary>Reprojection error in normalized camera coordinates.</summary>
	Normalized = 1,

	/// <summary>Angle between the observation's ray and the point, in degrees.</summary>
	Angular = 2,
}

/// <summary>
/// Port of colmap::ObservationManager: keeps per-image and per-image-pair observation
/// statistics of a reconstruction in sync while points and frames are added and removed.
/// </summary>
public sealed partial class ObservationManager
{
	/// <summary>The number of levels in the 3D point multi-resolution visibility pyramid.</summary>
	public const int kNumPoint3DVisibilityPyramidLevels = 6;

	private readonly Reconstruction _reconstruction;
	private readonly CorrespondenceGraph? _correspondenceGraph;

	// These stat maps are fully populated at construction and only their values are mutated
	// thereafter (no key insert/erase during mapping, only AddImage adds).
	private readonly Dictionary<ulong, ImagePairStat> _imagePairStats;
	private readonly Dictionary<uint, ImageStat> _imageStats;

	/// <summary>Per image pair: triangulated and total correspondences.</summary>
	public struct ImagePairStat
	{
		/// <summary>The number of triangulated correspondences between two images.</summary>
		public uint NumTriCorrs;

		/// <summary>The number of total correspondences/matches between two images.</summary>
		public uint NumTotalCorrs;
	}

	private sealed class ImageStat
	{
		// The number of image points that have at least one correspondence to another image.
		public uint NumObservations;

		// The sum of correspondences per image point.
		public uint NumCorrespondences;

		// The sum of correspondences that have a corresponding registered image.
		public uint NumVisibleCorrespondences;

		// The number of 2D points which have at least one corresponding 2D point in another
		// image that is part of a 3D point track, i.e. the number of entries of
		// NumCorrespondencesHavePoint3D above zero.
		public uint NumVisiblePoints3D;

		// Per image point, the number of correspondences that have a 3D point.
		public uint[] NumCorrespondencesHavePoint3D = [];

		// Distribution of triangulated correspondences in the image.
		public VisibilityPyramid Point3DVisibilityPyramid = new();
	}

	/// <summary>
	/// Builds the statistics for <paramref name="reconstruction"/>. Without a correspondence
	/// graph only the point/observation operations and filters are meaningful; the
	/// correspondence counts stay 0. Images that are already registered with triangulated
	/// points (a model loaded from disk) are counted as such.
	/// </summary>
	public ObservationManager(Reconstruction reconstruction, CorrespondenceGraph? correspondenceGraph = null)
	{
		_reconstruction = reconstruction;
		_correspondenceGraph = correspondenceGraph;

		// Add image pairs.
		_imagePairStats = new Dictionary<ulong, ImagePairStat>(correspondenceGraph?.NumImagePairs ?? 0);
		if (correspondenceGraph is not null)
		{
			foreach ((ulong pairId, uint numMatches) in correspondenceGraph.NumMatchesBetweenAllImages())
			{
				_imagePairStats.TryAdd(pairId, new ImagePairStat { NumTotalCorrs = numMatches });
			}
		}

		// Add image stats.
		_imageStats = new Dictionary<uint, ImageStat>(reconstruction.NumImages);
		foreach ((uint imageId, Image image) in reconstruction.Images)
		{
			_imageStats.TryAdd(imageId, InitImageStat(imageId, image));
		}

		// If an existing model was loaded from disk and there were already images registered
		// previously, we need to initialize the observation bookkeeping.
		foreach (uint imageId in reconstruction.RegImageIds())
		{
			Image image = reconstruction.Image(imageId);
			for (uint point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
			{
				if (image.Point2DAt(point2DIdx).HasPoint3D)
				{
					SetObservationAsTriangulated(imageId, point2DIdx, isContinuedPoint3D: false);
				}

				if (correspondenceGraph is not null)
				{
					AddVisibleCorrespondences(imageId, point2DIdx);
				}
			}
		}
	}

	/// <summary>The reconstruction whose observations are managed.</summary>
	public Reconstruction Reconstruction => _reconstruction;

	/// <summary>
	/// The statistics of every image pair with matches. Iterates in the correspondence
	/// graph's pair order, then in the order AddImage added pairs (docs/CPP_DIVERGENCES.md,
	/// entry 50).
	/// </summary>
	public IReadOnlyDictionary<ulong, ImagePairStat> ImagePairs => _imagePairStats;

	/// <summary>
	/// Adds image stats for streaming/online SfM, so that the image can be registered and
	/// triangulated without rebuilding the ObservationManager. The image must already be
	/// added to the Reconstruction and CorrespondenceGraph. O(N) per call in the number of
	/// existing images, for the image pair stats update.
	/// </summary>
	public void AddImage(uint imageId)
	{
		Check.That(!_imageStats.ContainsKey(imageId), $"Image {imageId} already exists in the ObservationManager");
		Check.That(_reconstruction.ExistsImage(imageId), $"Image {imageId} must be added to the Reconstruction first");
		if (_correspondenceGraph is not null)
		{
			Check.That(_correspondenceGraph.ExistsImage(imageId), $"Image {imageId} must be added to the CorrespondenceGraph first");
		}

		Image image = _reconstruction.Image(imageId);
		_imageStats.Add(imageId, InitImageStat(imageId, image));

		if (_correspondenceGraph is null)
		{
			return;
		}

		// Add image pair stats for all pairs involving the new image and refresh the cached
		// stats for existing images, whose observation/correspondence counts may have
		// increased when AddTwoViewGeometry added new correspondences.
		foreach ((uint otherImageId, ImageStat otherStats) in _imageStats)
		{
			if (otherImageId == imageId)
			{
				continue;
			}

			uint numMatches = _correspondenceGraph.NumMatchesBetweenImages(imageId, otherImageId);
			if (numMatches > 0)
			{
				ulong pairId = ImagePairToPairId(imageId, otherImageId);
				_imagePairStats.TryAdd(pairId, new ImagePairStat { NumTotalCorrs = numMatches });

				otherStats.NumObservations = _correspondenceGraph.NumObservationsForImage(otherImageId);
				otherStats.NumCorrespondences = _correspondenceGraph.NumCorrespondencesForImage(otherImageId);
			}
		}

		// Propagate visibility from already-triangulated points. In the batch pipeline, the
		// constructor handles this by iterating all registered images and propagating
		// triangulation visibility to their correspondences. Since this image was not present
		// during construction, it missed that propagation. We catch up here by scanning the
		// new image for correspondences to points that are already triangulated in other
		// images.
		for (uint point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
		{
			foreach (CorrespondenceGraph.Correspondence corr in _correspondenceGraph.FindCorrespondences(imageId, point2DIdx))
			{
				Image corrImage = _reconstruction.Image(corr.ImageId);
				if (corrImage.HasPose && corrImage.Point2DAt(corr.Point2DIdx).HasPoint3D)
				{
					IncrementCorrespondenceHasPoint3D(imageId, point2DIdx);
				}
			}
		}
	}

	/// <summary>Adds a new 3D point and marks its track's observations as triangulated; returns its id.</summary>
	public ulong AddPoint3D(Vector3d xyz, Track track, Vector3ub color = default)
	{
		ulong point3DId = _reconstruction.AddPoint3D(xyz, track, color);

		foreach (TrackElement trackEl in track.Elements)
		{
			SetObservationAsTriangulated(trackEl.ImageId, trackEl.Point2DIdx, isContinuedPoint3D: false);
		}

		return point3DId;
	}

	/// <summary>Adds an observation to an existing 3D point.</summary>
	public void AddObservation(ulong point3DId, TrackElement trackEl)
	{
		_reconstruction.AddObservation(point3DId, trackEl);
		SetObservationAsTriangulated(trackEl.ImageId, trackEl.Point2DIdx, isContinuedPoint3D: true);
	}

	/// <summary>Deletes a 3D point and all its references in the observed images.</summary>
	public void DeletePoint3D(ulong point3DId)
	{
		// Note: Do not change order of these instructions, especially with respect to
		// ResetTriObservations.
		Track track = _reconstruction.Point3D(point3DId).Track;
		foreach (TrackElement trackEl in track.Elements)
		{
			ResetTriObservations(trackEl.ImageId, trackEl.Point2DIdx, isDeletedPoint3D: true);
		}

		_reconstruction.DeletePoint3D(point3DId);
	}

	/// <summary>
	/// Deletes one observation from an image and the corresponding 3D point. This deletes the
	/// entire 3D point if its track has two elements before the call.
	/// </summary>
	public void DeleteObservation(uint imageId, uint point2DIdx)
	{
		// Note: Do not change order of these instructions, especially with respect to
		// ResetTriObservations.
		Image image = _reconstruction.Image(imageId);
		ulong point3DId = image.Point2DAt(point2DIdx).Point3DId;
		Point3D point3D = _reconstruction.Point3D(point3DId);

		if (point3D.Track.Length <= 2)
		{
			DeletePoint3D(point3DId);
			return;
		}

		ResetTriObservations(imageId, point2DIdx, isDeletedPoint3D: false);
		_reconstruction.DeleteObservation(imageId, point2DIdx);
	}

	/// <summary>Merges two 3D points and returns the id of the merged point.</summary>
	public ulong MergePoints3D(ulong point3DId1, ulong point3DId2)
	{
		foreach (TrackElement trackEl in _reconstruction.Point3D(point3DId1).Track.Elements)
		{
			ResetTriObservations(trackEl.ImageId, trackEl.Point2DIdx, isDeletedPoint3D: true);
		}

		foreach (TrackElement trackEl in _reconstruction.Point3D(point3DId2).Track.Elements)
		{
			ResetTriObservations(trackEl.ImageId, trackEl.Point2DIdx, isDeletedPoint3D: true);
		}

		ulong mergedPoint3DId = _reconstruction.MergePoints3D(point3DId1, point3DId2);

		// C++ copies the merged track; SetObservationAsTriangulated does not touch tracks, so
		// iterating the stored one is the same.
		foreach (TrackElement trackEl in _reconstruction.Point3D(mergedPoint3DId).Track.Elements)
		{
			SetObservationAsTriangulated(trackEl.ImageId, trackEl.Point2DIdx, isContinuedPoint3D: false);
		}

		return mergedPoint3DId;
	}

	/// <summary>Registers an existing frame and counts its images' correspondences as visible.</summary>
	public void RegisterFrame(uint frameId)
	{
		Frame frame = _reconstruction.Frame(frameId);
		if (_correspondenceGraph is not null)
		{
			foreach (DataId dataId in frame.ImageIds())
			{
				Image image = _reconstruction.Image((uint)dataId.Id);
				uint numPoints2D = image.NumPoints2D;
				for (uint point2DIdx = 0; point2DIdx < numPoints2D; ++point2DIdx)
				{
					AddVisibleCorrespondences((uint)dataId.Id, point2DIdx);
				}
			}
		}

		_reconstruction.RegisterFrame(frameId);
	}

	/// <summary>
	/// De-registers an existing frame: its images' correspondences stop being visible and
	/// their observations are deleted.
	/// </summary>
	public void DeRegisterFrame(uint frameId)
	{
		Frame frame = _reconstruction.Frame(frameId);
		foreach (DataId dataId in frame.ImageIds())
		{
			Image image = _reconstruction.Image((uint)dataId.Id);
			uint numPoints2D = image.NumPoints2D;
			for (uint point2DIdx = 0; point2DIdx < numPoints2D; ++point2DIdx)
			{
				if (_correspondenceGraph is not null)
				{
					foreach (CorrespondenceGraph.Correspondence corr in _correspondenceGraph.FindCorrespondences((uint)dataId.Id, point2DIdx))
					{
						ImageStat stats = GetOrAddImageStat(corr.ImageId);
						if (stats.NumVisibleCorrespondences == 0)
						{
							Check.Gt(stats.NumVisibleCorrespondences, 0u, $"Visible correspondences underflow for image {corr.ImageId} when deregistering frame {frameId}");
						}

						stats.NumVisibleCorrespondences -= 1;
					}
				}

				if (image.Point2DAt(point2DIdx).HasPoint3D)
				{
					DeleteObservation((uint)dataId.Id, point2DIdx);
				}
			}
		}

		_reconstruction.DeRegisterFrame(frameId);
	}

	/// <summary>
	/// The number of observations, i.e. the number of image points that have at least one
	/// correspondence to another image.
	/// </summary>
	public uint NumObservations(uint imageId) => _imageStats[imageId].NumObservations;

	/// <summary>The number of correspondences for all image points.</summary>
	public uint NumCorrespondences(uint imageId) => _imageStats[imageId].NumCorrespondences;

	/// <summary>The number of correspondences of all image points to registered images.</summary>
	public uint NumVisibleCorrespondences(uint imageId) => _imageStats[imageId].NumVisibleCorrespondences;

	/// <summary>
	/// The number of observations that see a triangulated point, i.e. the number of image
	/// points that have at least one correspondence to a triangulated point in another image.
	/// </summary>
	public uint NumVisiblePoints3D(uint imageId) => _imageStats[imageId].NumVisiblePoints3D;

	/// <summary>
	/// The score of triangulated observations. In contrast to NumVisiblePoints3D, this score
	/// also captures the distribution of triangulated observations in the image. This is
	/// useful to select the next best image in incremental reconstruction, because a more
	/// uniform distribution of observations results in more robust registration.
	/// </summary>
	public ulong Point3DVisibilityScore(uint imageId) => _imageStats[imageId].Point3DVisibilityPyramid.Score;

	/// <summary>
	/// Indicates that another image has a point that is triangulated and has a
	/// correspondence to this image point.
	/// </summary>
	public void IncrementCorrespondenceHasPoint3D(uint imageId, uint point2DIdx)
	{
		Image image = _reconstruction.Image(imageId);
		Vector2d xy = image.Point2DAt(point2DIdx).Xy;
		ImageStat stats = _imageStats[imageId];

		stats.NumCorrespondencesHavePoint3D[point2DIdx] += 1;
		if (stats.NumCorrespondencesHavePoint3D[point2DIdx] == 1)
		{
			stats.NumVisiblePoints3D += 1;
		}

		stats.Point3DVisibilityPyramid.SetPoint(xy.X, xy.Y);

		Check.Le(stats.NumVisiblePoints3D, stats.NumObservations);
	}

	/// <summary>
	/// Indicates that another image has a point that is not triangulated any more and has a
	/// correspondence to this image point. This assumes that IncrementCorrespondenceHasPoint3D
	/// was called for the same image point and correspondence before.
	/// </summary>
	public void DecrementCorrespondenceHasPoint3D(uint imageId, uint point2DIdx)
	{
		Image image = _reconstruction.Image(imageId);
		Vector2d xy = image.Point2DAt(point2DIdx).Xy;
		ImageStat stats = _imageStats[imageId];

		if (stats.NumCorrespondencesHavePoint3D[point2DIdx] == 0)
		{
			Check.Gt(stats.NumCorrespondencesHavePoint3D[point2DIdx], 0u, $"Correspondence counter underflow for image {imageId} point2D {point2DIdx}");
		}

		stats.NumCorrespondencesHavePoint3D[point2DIdx] -= 1;
		if (stats.NumCorrespondencesHavePoint3D[point2DIdx] == 0)
		{
			stats.NumVisiblePoints3D -= 1;
		}

		stats.Point3DVisibilityPyramid.ResetPoint(xy.X, xy.Y);

		Check.Le(stats.NumVisiblePoints3D, stats.NumObservations);
	}

	/// <summary>
	/// COLMAP's operator&lt;&lt;. C++ streams the correspondence graph's shared_ptr, i.e. its
	/// address; .NET has no stable address, so a non-null graph prints its own ToString
	/// (docs/CPP_DIVERGENCES.md, entry 50).
	/// </summary>
	public override string ToString() =>
		$"ObservationManager(reconstruction={_reconstruction}, correspondence_graph={(_correspondenceGraph is null ? "null" : _correspondenceGraph.ToString())})";

	private ImageStat InitImageStat(uint imageId, Image image)
	{
		Camera camera = image.CameraPtr;
		var imageStat = new ImageStat
		{
			Point3DVisibilityPyramid = new VisibilityPyramid(kNumPoint3DVisibilityPyramidLevels, camera.Width, camera.Height),
			NumCorrespondencesHavePoint3D = new uint[image.NumPoints2D],
		};
		if (_correspondenceGraph is not null && _correspondenceGraph.ExistsImage(imageId))
		{
			imageStat.NumObservations = _correspondenceGraph.NumObservationsForImage(imageId);
			imageStat.NumCorrespondences = _correspondenceGraph.NumCorrespondencesForImage(imageId);
		}

		return imageStat;
	}

	// C++ `image_stats_[image_id]`: inserts a default stat when the image is unknown.
	private ImageStat GetOrAddImageStat(uint imageId)
	{
		ref ImageStat? stats = ref CollectionsMarshal.GetValueRefOrAddDefault(_imageStats, imageId, out _);
		stats ??= new ImageStat();
		return stats;
	}

	// Counts the correspondences of one image point as seen by a registered image.
	private void AddVisibleCorrespondences(uint imageId, uint point2DIdx)
	{
		foreach (CorrespondenceGraph.Correspondence corr in _correspondenceGraph!.FindCorrespondences(imageId, point2DIdx))
		{
			GetOrAddImageStat(corr.ImageId).NumVisibleCorrespondences += 1;
		}
	}

	private void SetObservationAsTriangulated(uint imageId, uint point2DIdx, bool isContinuedPoint3D)
	{
		if (_correspondenceGraph is null)
		{
			return;
		}

		Image image = _reconstruction.Image(imageId);
		Check.That(image.HasPose);

		Point2D point2D = image.Point2DAt(point2DIdx);
		Check.That(point2D.HasPoint3D);
		ulong point3DId = point2D.Point3DId;

		foreach (CorrespondenceGraph.Correspondence corr in _correspondenceGraph.FindCorrespondences(imageId, point2DIdx))
		{
			Image corrImage = _reconstruction.Image(corr.ImageId);
			ulong corrPoint3DId = corrImage.Point2DAt(corr.Point2DIdx).Point3DId;
			IncrementCorrespondenceHasPoint3D(corr.ImageId, corr.Point2DIdx);
			// Update number of shared 3D points between image pairs and make sure to only
			// count the correspondences once (not twice forward and backward).
			if (point3DId == corrPoint3DId && (isContinuedPoint3D || imageId < corr.ImageId))
			{
				ulong pairId = ImagePairToPairId(imageId, corr.ImageId);
				ref ImagePairStat stats = ref CollectionsMarshal.GetValueRefOrAddDefault(_imagePairStats, pairId, out _);
				stats.NumTriCorrs += 1;
				if (stats.NumTriCorrs > stats.NumTotalCorrs)
				{
					Check.Le(stats.NumTriCorrs, stats.NumTotalCorrs, $"The correspondence graph must not contain duplicate matches: {corr.ImageId} {corr.Point2DIdx}");
				}
			}
		}
	}

	private void ResetTriObservations(uint imageId, uint point2DIdx, bool isDeletedPoint3D)
	{
		if (_correspondenceGraph is null)
		{
			return;
		}

		Image image = _reconstruction.Image(imageId);
		Check.That(image.HasPose);

		Point2D point2D = image.Point2DAt(point2DIdx);
		Check.That(point2D.HasPoint3D);
		ulong point3DId = point2D.Point3DId;

		foreach (CorrespondenceGraph.Correspondence corr in _correspondenceGraph.FindCorrespondences(imageId, point2DIdx))
		{
			Image corrImage = _reconstruction.Image(corr.ImageId);
			ulong corrPoint3DId = corrImage.Point2DAt(corr.Point2DIdx).Point3DId;
			DecrementCorrespondenceHasPoint3D(corr.ImageId, corr.Point2DIdx);
			// Update number of shared 3D points between image pairs and make sure to only
			// count the correspondences once (not twice forward and backward).
			if (point3DId == corrPoint3DId && (!isDeletedPoint3D || imageId < corr.ImageId))
			{
				ulong pairId = ImagePairToPairId(imageId, corr.ImageId);
				ref ImagePairStat stats = ref CollectionsMarshal.GetValueRefOrAddDefault(_imagePairStats, pairId, out _);
				Check.Gt(stats.NumTriCorrs, 0u, "The scene graph must not contain duplicate matches");
				stats.NumTriCorrs -= 1;
			}
		}
	}
}
