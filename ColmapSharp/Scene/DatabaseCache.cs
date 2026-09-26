// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// DatabaseCache: port of colmap/scene/database_cache.h and database_cache.cc. It reads the
// rigs, cameras, frames, images (with their keypoints as 2D points), pose priors and
// verified two-view geometries of a Database (Database.cs / InMemoryDatabase.cs) once and
// builds the CorrespondenceGraph (CorrespondenceGraph.cs) from them, so the mapper can
// create many Reconstructions (Reconstruction.Load, Reconstruction.Database.cs) without
// going back to the database. Tests: ColmapSharp.Tests/Scene/DatabaseCacheTests.cs
// (database_cache_test.cc 1:1).
//
// Tier A (exact): pure bookkeeping.
//
// Translation notes:
// - Iteration order (docs/CPP_DIVERGENCES.md, entry 33): COLMAP keeps the objects in
//   NodeHashMaps; here they are in Util/IdMap.cs and enumerate in ascending id order.
//   The correspondence graph is built from the two-view geometries in the order the
//   database returns them (COLMAP does the same), and CreateFromCache copies pairs in the
//   source graph's insertion order, so per-point correspondence lists are deterministic.
// - shared_ptr<DatabaseCache> / shared_ptr<CorrespondenceGraph> are plain references. The
//   const and non-const CorrespondenceGraph() overloads are the one property.
// - Add* take their argument by value in C++, so they store a Clone(); the caller's object
//   stays independent. Objects read from the database are fresh and are stored directly.
// - COLMAP's LOG(INFO) timing messages are dropped (PORTING_PLAN.md, Phase 4).

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::DatabaseCache: caches the contents of the database in memory, used to
/// quickly create new reconstruction instances when multiple models are reconstructed.
/// </summary>
public sealed class DatabaseCache
{
	/// <summary>Port of colmap::DatabaseCache::Options.</summary>
	public sealed class Options
	{
		/// <summary>Only load image pairs with a minimum number of matches.</summary>
		public int MinNumMatches { get; set; }

		/// <summary>Whether to ignore watermark image pairs.</summary>
		public bool IgnoreWatermarks { get; set; }

		/// <summary>
		/// Whether to use only load the data for a subset of the images. Notice that if one
		/// image of a frame is included, all other images in the same frame will also be
		/// included. All images are used if empty.
		/// </summary>
		public HashSet<string> ImageNames { get; set; } = [];

		/// <summary>
		/// Whether to load all candidate images regardless of whether they have
		/// correspondences. If false (default), only images that participate in at least
		/// one valid match pair are loaded.
		/// </summary>
		public bool LoadAllImages { get; set; }

		/// <summary>Whether to convert pose priors to ENU coordinate system.</summary>
		public bool ConvertPosePriorsToEnu { get; set; }
	}

	private readonly IdMap<uint, Rig> _rigs = new();
	private readonly IdMap<uint, Camera> _cameras = new();
	private readonly IdMap<uint, Frame> _frames = new();
	private readonly IdMap<uint, Image> _images = new();
	private List<PosePrior> _posePriors = [];

	/// <summary>Number of rigs.</summary>
	public int NumRigs => _rigs.Count;

	/// <summary>Number of cameras.</summary>
	public int NumCameras => _cameras.Count;

	/// <summary>Number of frames.</summary>
	public int NumFrames => _frames.Count;

	/// <summary>Number of images.</summary>
	public int NumImages => _images.Count;

	/// <summary>Number of pose priors.</summary>
	public int NumPosePriors => _posePriors.Count;

	/// <summary>All rigs, in ascending id order.</summary>
	public IReadOnlyDictionary<uint, Rig> Rigs => _rigs;

	/// <summary>All cameras, in ascending id order.</summary>
	public IReadOnlyDictionary<uint, Camera> Cameras => _cameras;

	/// <summary>All frames, in ascending id order.</summary>
	public IReadOnlyDictionary<uint, Frame> Frames => _frames;

	/// <summary>All images, in ascending id order.</summary>
	public IReadOnlyDictionary<uint, Image> Images => _images;

	/// <summary>All pose priors, in the order they were read or added.</summary>
	public IReadOnlyList<PosePrior> PosePriors => _posePriors;

	/// <summary>
	/// The correspondence graph. It is mutable, so callers can update two-view geometries
	/// (COLMAP's non-const CorrespondenceGraph() overload).
	/// </summary>
	public CorrespondenceGraph CorrespondenceGraph { get; private set; } = new();

	/// <summary>The rig with the given id; throws if it does not exist.</summary>
	public Rig Rig(uint rigId) => _rigs[rigId];

	/// <summary>The camera with the given id; throws if it does not exist.</summary>
	public Camera Camera(uint cameraId) => _cameras[cameraId];

	/// <summary>The frame with the given id; throws if it does not exist.</summary>
	public Frame Frame(uint frameId) => _frames[frameId];

	/// <summary>The image with the given id; throws if it does not exist.</summary>
	public Image Image(uint imageId) => _images[imageId];

	/// <summary>Whether the rig exists.</summary>
	public bool ExistsRig(uint rigId) => _rigs.ContainsKey(rigId);

	/// <summary>Whether the camera exists.</summary>
	public bool ExistsCamera(uint cameraId) => _cameras.ContainsKey(cameraId);

	/// <summary>Whether the frame exists.</summary>
	public bool ExistsFrame(uint frameId) => _frames.ContainsKey(frameId);

	/// <summary>Whether the image exists.</summary>
	public bool ExistsImage(uint imageId) => _images.ContainsKey(imageId);

	/// <summary>Creates a cache and loads it from the database.</summary>
	public static DatabaseCache Create(Database database, Options options)
	{
		var cache = new DatabaseCache();
		cache.Load(database, options);
		return cache;
	}

	/// <summary>Adds a rig; throws if its id exists.</summary>
	public void AddRig(Rig rig)
	{
		Check.That(!ExistsRig(rig.RigId));
		_rigs.TryAdd(rig.RigId, rig.Clone());
	}

	/// <summary>Adds a camera; throws if its id exists.</summary>
	public void AddCamera(Camera camera)
	{
		Check.That(!ExistsCamera(camera.CameraId));
		_cameras.TryAdd(camera.CameraId, camera.Clone());
	}

	/// <summary>Adds a frame; throws if its id exists.</summary>
	public void AddFrame(Frame frame)
	{
		Check.That(!ExistsFrame(frame.FrameId));
		_frames.TryAdd(frame.FrameId, frame.Clone());
	}

	/// <summary>Adds an image and its node in the correspondence graph; throws if its id exists.</summary>
	public void AddImage(Image image)
	{
		Check.That(!ExistsImage(image.ImageId));
		CorrespondenceGraph.AddImage(image.ImageId, (int)image.NumPoints2D);
		_images.TryAdd(image.ImageId, image.Clone());
	}

	/// <summary>Appends a pose prior.</summary>
	public void AddPosePrior(PosePrior posePrior) => _posePriors.Add(posePrior);

	/// <summary>
	/// Finds an image by name (linear search); null if there is none. With several images of
	/// the same name, the one with the smallest id (entry 33).
	/// </summary>
	public Image? FindImageWithName(string name)
	{
		foreach (Image image in _images.Values)
		{
			if (image.Name == name)
			{
				return image;
			}
		}

		return null;
	}

	/// <summary>Load cameras, images, features, and matches from database.</summary>
	public void Load(Database database, Options options)
	{
		bool hasRigs = database.NumRigs() > 0;
		bool hasFrames = database.NumFrames() > 0;

		foreach (Rig rig in database.ReadAllRigs())
		{
			_rigs.TryAdd(rig.RigId, rig);
		}

		foreach (Camera camera in database.ReadAllCameras())
		{
			if (!hasRigs)
			{
				// For backwards compatibility with old databases from before having support
				// for rigs/frames, we create a rig for each camera.
				var rig = new Rig { RigId = camera.CameraId };
				rig.AddRefSensor(camera.SensorId);
				_rigs.TryAdd(rig.RigId, rig);
			}

			_cameras.TryAdd(camera.CameraId, camera);
		}

		foreach (Frame frame in database.ReadAllFrames())
		{
			_frames.TryAdd(frame.FrameId, frame);
		}

		List<(ulong PairId, TwoViewGeometry TwoViewGeometry)> twoViewGeometries = database.ReadTwoViewGeometries();

		var frameIds = new HashSet<uint>();
		var imageToFrameId = new Dictionary<uint, uint>();

		List<Image> images = database.ReadAllImages();
		foreach (Image image in images)
		{
			// For backwards compatibility with old databases from before having support for
			// rigs/frames, we create a frame for each image.
			if (hasFrames)
			{
				Check.That(image.HasFrameId);
			}
			else
			{
				var frame = new Frame { FrameId = image.ImageId };
				frame.SetRigId(image.CameraId);
				frame.AddDataId(image.DataId);
				image.SetFrameId(frame.FrameId);
				_frames.TryAdd(frame.FrameId, frame);
			}

			imageToFrameId.TryAdd(image.ImageId, image.FrameId);
		}

		// Determines for which images data should be loaded.
		foreach (Image image in images)
		{
			if (options.ImageNames.Count == 0 || options.ImageNames.Contains(image.Name))
			{
				frameIds.Add(image.FrameId);
			}
		}

		// Collect all images that are connected in the correspondence graph.
		var connectedFrameIds = new HashSet<uint>();
		if (!options.LoadAllImages)
		{
			foreach ((ulong pairId, TwoViewGeometry twoViewGeometry) in twoViewGeometries)
			{
				if (UseInlierMatchesCheck(options, twoViewGeometry.Config, twoViewGeometry.InlierMatches.Count))
				{
					(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
					uint frameId1 = imageToFrameId[imageId1];
					uint frameId2 = imageToFrameId[imageId2];
					if (frameIds.Contains(frameId1) && frameIds.Contains(frameId2))
					{
						connectedFrameIds.Add(frameId1);
						connectedFrameIds.Add(frameId2);
					}
				}
			}
		}

		HashSet<uint> loadFrameIds = options.LoadAllImages ? frameIds : connectedFrameIds;

		// Remove frames that should not be loaded.
		foreach (uint frameId in _frames.Keys.Where(frameId => !loadFrameIds.Contains(frameId)).ToList())
		{
			_frames.Remove(frameId);
		}

		// Load images and their keypoints. When load_all_images is false, only images with
		// correspondences are loaded, as images without matches are not useful for SfM. When
		// load_all_images is true, all candidate images are loaded so that their keypoints
		// are populated (e.g., for triangulation on an existing reconstruction).
		foreach (Image image in images)
		{
			if (!loadFrameIds.Contains(image.FrameId))
			{
				continue;
			}

			image.SetPoints2D(FeatureKeypointsToPointsVector(database.ReadKeypoints(image.ImageId)));
			_images.TryAdd(image.ImageId, image);
		}

		_posePriors = database.ReadAllPosePriors();
		if (options.ConvertPosePriorsToEnu)
		{
			ConvertPosePriorsToEnu();
		}

		// Build correspondence graph.
		CorrespondenceGraph = new CorrespondenceGraph();
		foreach ((uint imageId, Image image) in _images)
		{
			CorrespondenceGraph.AddImage(imageId, (int)image.NumPoints2D);
		}

		foreach ((ulong pairId, TwoViewGeometry twoViewGeometry) in twoViewGeometries)
		{
			if (UseInlierMatchesCheck(options, twoViewGeometry.Config, twoViewGeometry.InlierMatches.Count))
			{
				(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
				uint frameId1 = imageToFrameId[imageId1];
				uint frameId2 = imageToFrameId[imageId2];
				if (frameIds.Contains(frameId1) && frameIds.Contains(frameId2))
				{
					CorrespondenceGraph.AddTwoViewGeometry(imageId1, imageId2, twoViewGeometry);
				}
			}
		}

		CorrespondenceGraph.FinalizeGraph();
	}

	/// <summary>
	/// Creates a filtered database cache from an existing cache containing only the
	/// specified images and their associated data.
	/// </summary>
	public static DatabaseCache CreateFromCache(DatabaseCache databaseCache, Options options)
	{
		var cache = new DatabaseCache();

		// Collect candidate image ids matching the name filter. Empty image_names means use
		// all images.
		var candidateImageIds = new HashSet<uint>();
		foreach ((uint imageId, Image image) in databaseCache.Images)
		{
			if (options.ImageNames.Count == 0 || options.ImageNames.Contains(image.Name))
			{
				candidateImageIds.Add(imageId);
			}
		}

		CorrespondenceGraph sourceGraph = databaseCache.CorrespondenceGraph;

		var connectedImageIds = new HashSet<uint>();
		if (!options.LoadAllImages)
		{
			foreach ((ulong pairId, uint numMatches) in sourceGraph.NumMatchesBetweenAllImages())
			{
				(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
				if (!candidateImageIds.Contains(imageId1) || !candidateImageIds.Contains(imageId2))
				{
					continue;
				}

				TwoViewGeometry twoViewGeometry = sourceGraph.ExtractTwoViewGeometry(imageId1, imageId2, extractInlierMatches: false);
				if (!UseInlierMatchesCheck(options, twoViewGeometry.Config, numMatches))
				{
					continue;
				}

				connectedImageIds.Add(imageId1);
				connectedImageIds.Add(imageId2);
			}
		}

		HashSet<uint> loadImageIds = options.LoadAllImages ? candidateImageIds : connectedImageIds;

		// Collect frame ids for images to load.
		var filteredFrameIds = new HashSet<uint>();
		foreach (uint imageId in loadImageIds)
		{
			filteredFrameIds.Add(databaseCache.Image(imageId).FrameId);
		}

		// Copy all images of filtered frames (not just the images matching the name filter).
		// This is needed for multi-camera rigs where the generalized pose solver needs all
		// images of a frame.
		var filteredCameraIds = new HashSet<uint>();
		foreach ((uint imageId, Image image) in databaseCache.Images)
		{
			if (filteredFrameIds.Contains(image.FrameId))
			{
				cache._images.TryAdd(imageId, image.Clone());
				filteredCameraIds.Add(image.CameraId);
			}
		}

		// Copy filtered frames and collect rig ids.
		var filteredRigIds = new HashSet<uint>();
		foreach ((uint frameId, Frame frame) in databaseCache.Frames)
		{
			if (filteredFrameIds.Contains(frameId))
			{
				cache._frames.TryAdd(frameId, frame.Clone());
				filteredRigIds.Add(frame.RigId);
			}
		}

		foreach ((uint cameraId, Camera camera) in databaseCache.Cameras)
		{
			if (filteredCameraIds.Contains(cameraId))
			{
				cache._cameras.TryAdd(cameraId, camera.Clone());
			}
		}

		foreach ((uint rigId, Rig rig) in databaseCache.Rigs)
		{
			if (filteredRigIds.Contains(rigId))
			{
				cache._rigs.TryAdd(rigId, rig.Clone());
			}
		}

		cache._posePriors = [.. databaseCache.PosePriors];
		if (options.ConvertPosePriorsToEnu)
		{
			cache.ConvertPosePriorsToEnu();
		}

		// Build filtered correspondence graph with all images from connected frames.
		cache.CorrespondenceGraph = new CorrespondenceGraph();
		foreach ((uint imageId, Image image) in cache._images)
		{
			cache.CorrespondenceGraph.AddImage(imageId, (int)image.NumPoints2D);
		}

		// Copy correspondences between all image pairs in the cache.
		foreach (ulong pairId in sourceGraph.ImagePairs())
		{
			(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);
			if (cache._images.ContainsKey(imageId1) && cache._images.ContainsKey(imageId2))
			{
				cache.CorrespondenceGraph.AddTwoViewGeometry(
					imageId1,
					imageId2,
					sourceGraph.ExtractTwoViewGeometry(imageId1, imageId2, extractInlierMatches: true));
			}
		}

		cache.CorrespondenceGraph.FinalizeGraph();
		return cache;
	}

	private static bool UseInlierMatchesCheck(Options options, TwoViewGeometry.ConfigurationType config, long numMatches) =>
		numMatches >= options.MinNumMatches
		&& (!options.IgnoreWatermarks || config != TwoViewGeometry.ConfigurationType.Watermark);

	private static Vector2d[] FeatureKeypointsToPointsVector(List<FeatureKeypoint> keypoints)
	{
		var points = new Vector2d[keypoints.Count];
		for (int i = 0; i < keypoints.Count; ++i)
		{
			points[i] = new Vector2d(keypoints[i].X, keypoints[i].Y);
		}

		return points;
	}

	private void ConvertPosePriorsToEnu()
	{
		bool priorIsGps = true;

		var gpsPriorPositions = new List<Vector3d>();
		var coordinateSystems = new HashSet<PosePriorCoordinateSystem>();
		foreach (PosePrior posePrior in _posePriors)
		{
			coordinateSystems.Add(posePrior.CoordinateSystem);
			if (posePrior.CoordinateSystem != PosePriorCoordinateSystem.Wgs84)
			{
				priorIsGps = false;
			}
			else
			{
				gpsPriorPositions.Add(posePrior.Position);
			}
		}

		Check.Le(coordinateSystems.Count, 1, "Inconsistent coordinate systems defined in pose priors");

		// If GPS priors are available, convert them to Cartesian ENU coordinates.
		if (priorIsGps && gpsPriorPositions.Count > 0)
		{
			// GPS reference to be used for EllipsoidToENU conversion.
			double refLat = gpsPriorPositions[0].X;
			double refLon = gpsPriorPositions[0].Y;
			double refAlt = gpsPriorPositions[0].Z;

			var gpsTransform = new GPSTransform(GPSTransform.Ellipsoid.WGS84);
			Vector3d[] xyzPrior = gpsTransform.EllipsoidToENU(gpsPriorPositions, refLat, refLon, refAlt);

			for (int i = 0; i < _posePriors.Count; ++i)
			{
				PosePrior posePrior = _posePriors[i];
				posePrior.Position = xyzPrior[i];
				posePrior.CoordinateSystem = PosePriorCoordinateSystem.Cartesian;
				_posePriors[i] = posePrior;
			}
		}
	}
}
