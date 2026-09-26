// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureMatcherCache: port of colmap/controllers/matcher_cache.h and .cc - the cache that
// the feature matching controllers and the pair generators (Controllers/PairGenerator.cs and
// neighbors) read the Database (Scene/Database.cs) through, so a matching run touches the
// database as little as possible. Cameras, frames, images and pose priors are loaded once,
// in full, on first use; keypoints, descriptors, their existence flags and the per-image
// FeatureDescriptorIndex go through ThreadSafeLRUCaches (Util/Cache.cs) of `cacheSize`
// entries. Tests: ColmapSharp.Tests/Controllers/FeatureMatcherCacheTests.cs
// (matcher_cache_test.cc 1:1).
//
// Tier A (exact): pure bookkeeping.
//
// Translation notes:
// - std::shared_ptr<Database> becomes the Database reference; std::shared_ptr<T> results
//   are the cached objects themselves, shared by every caller: do not mutate them (COLMAP's
//   callers copy before modifying, e.g. `auto keypoints = *cache->GetKeypoints(id)`).
// - The std::mutex guarding the database is a Lock. The Maybe* loaders take it one after
//   another (MaybeLoadImages first loads frames), exactly like COLMAP's non-nested locking.
// - NodeHashMap caches become Dictionaries. Nothing iterates them except GetImageIds and
//   GetFrameIds, which sort the ids like COLMAP, so hash order never leaks.
// - FindImagePosePriorOrNull returns a copy of the (struct) PosePrior or null instead of a
//   pointer into the cache.

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>
/// Port of colmap::FeatureMatcherCache: cache for feature matching to minimize database
/// access during matching. Thread safe.
/// </summary>
public sealed class FeatureMatcherCache
{
	private readonly Database database;
	private readonly Lock databaseMutex = new();
	private readonly ThreadSafeLRUCache<uint, List<FeatureKeypoint>> keypointsCache;
	private readonly ThreadSafeLRUCache<uint, FeatureDescriptors> descriptorsCache;
	private readonly ThreadSafeLRUCache<uint, bool> keypointsExistsCache;
	private readonly ThreadSafeLRUCache<uint, bool> descriptorsExistsCache;
	private readonly ThreadSafeLRUCache<uint, FeatureDescriptorIndex> descriptorIndexCache;
	private Dictionary<uint, Camera>? camerasCache;
	private Dictionary<uint, Frame>? framesCache;
	private Dictionary<uint, Image>? imagesCache;
	private Dictionary<uint, PosePrior>? posePriorsCache;
	private long? maxNumKeypoints;

	/// <summary>
	/// A cache over <paramref name="database"/> that keeps the features of at most
	/// <paramref name="cacheSize"/> images in memory.
	/// </summary>
	public FeatureMatcherCache(int cacheSize, Database database)
	{
		this.database = Check.NotNull(database);
		descriptorIndexCache = new ThreadSafeLRUCache<uint, FeatureDescriptorIndex>(cacheSize, imageId =>
		{
			FeatureDescriptors descriptors = GetDescriptors(imageId);
			var index = FeatureDescriptorIndex.Create();
			index.Build(descriptors.ToFloat());
			return index;
		});
		keypointsCache = new ThreadSafeLRUCache<uint, List<FeatureKeypoint>>(cacheSize, imageId =>
		{
			lock (databaseMutex)
			{
				return this.database.ReadKeypoints(imageId);
			}
		});
		descriptorsCache = new ThreadSafeLRUCache<uint, FeatureDescriptors>(cacheSize, imageId =>
		{
			lock (databaseMutex)
			{
				return this.database.ReadDescriptors(imageId);
			}
		});
		keypointsExistsCache = new ThreadSafeLRUCache<uint, bool>(cacheSize, imageId =>
		{
			lock (databaseMutex)
			{
				return this.database.ExistsKeypoints(imageId);
			}
		});
		descriptorsExistsCache = new ThreadSafeLRUCache<uint, bool>(cacheSize, imageId =>
		{
			lock (databaseMutex)
			{
				return this.database.ExistsDescriptors(imageId);
			}
		});
	}

	/// <summary>
	/// Executes a function that accesses the database. Thread safe: only one function can
	/// access the database at a time.
	/// </summary>
	public void AccessDatabase(Action<Database> func)
	{
		lock (databaseMutex)
		{
			func(database);
		}
	}

	/// <summary>The camera; throws when it does not exist.</summary>
	public Camera GetCamera(uint cameraId)
	{
		MaybeLoadCameras();
		return camerasCache![cameraId];
	}

	/// <summary>The frame; throws when it does not exist.</summary>
	public Frame GetFrame(uint frameId)
	{
		MaybeLoadFrames();
		return framesCache![frameId];
	}

	/// <summary>The image; throws when it does not exist.</summary>
	public Image GetImage(uint imageId)
	{
		MaybeLoadImages();
		return imagesCache![imageId];
	}

	/// <summary>The pose prior of the image (keyed by its camera data id), or null.</summary>
	public PosePrior? FindImagePosePriorOrNull(uint imageId)
	{
		MaybeLoadPosePriors();
		return posePriorsCache!.TryGetValue(imageId, out PosePrior posePrior) ? posePrior : null;
	}

	/// <summary>The keypoints of the image (shared: do not modify).</summary>
	public List<FeatureKeypoint> GetKeypoints(uint imageId) => keypointsCache.Get(imageId);

	/// <summary>The descriptors of the image (shared: do not modify).</summary>
	public FeatureDescriptors GetDescriptors(uint imageId) => descriptorsCache.Get(imageId);

	/// <summary>The raw matches of the pair, read from the database.</summary>
	public List<FeatureMatch> GetMatches(uint imageId1, uint imageId2)
	{
		lock (databaseMutex)
		{
			return database.ReadMatches(imageId1, imageId2);
		}
	}

	/// <summary>The two-view geometry of the pair, read from the database.</summary>
	public TwoViewGeometry GetTwoViewGeometry(uint imageId1, uint imageId2)
	{
		lock (databaseMutex)
		{
			return database.ReadTwoViewGeometry(imageId1, imageId2);
		}
	}

	/// <summary>All frame ids, sorted for deterministic behavior.</summary>
	public List<uint> GetFrameIds()
	{
		MaybeLoadFrames();
		var frameIds = new List<uint>(framesCache!.Keys);
		frameIds.Sort();
		return frameIds;
	}

	/// <summary>All image ids, sorted for deterministic behavior.</summary>
	public List<uint> GetImageIds()
	{
		MaybeLoadImages();
		var imageIds = new List<uint>(imagesCache!.Keys);
		imageIds.Sort();
		return imageIds;
	}

	/// <summary>The cache of per-image descriptor indexes, built on first use.</summary>
	public ThreadSafeLRUCache<uint, FeatureDescriptorIndex> GetFeatureDescriptorIndexCache() => descriptorIndexCache;

	/// <summary>Whether keypoints exist for the image (cached).</summary>
	public bool ExistsKeypoints(uint imageId) => keypointsExistsCache.Get(imageId);

	/// <summary>Whether descriptors exist for the image (cached).</summary>
	public bool ExistsDescriptors(uint imageId) => descriptorsExistsCache.Get(imageId);

	/// <summary>Whether raw matches exist for the pair.</summary>
	public bool ExistsMatches(uint imageId1, uint imageId2)
	{
		lock (databaseMutex)
		{
			return database.ExistsMatches(imageId1, imageId2);
		}
	}

	/// <summary>Whether a two-view geometry exists for the pair.</summary>
	public bool ExistsTwoViewGeometry(uint imageId1, uint imageId2)
	{
		lock (databaseMutex)
		{
			return database.ExistsTwoViewGeometry(imageId1, imageId2);
		}
	}

	/// <summary>Whether the pair has a two-view geometry with at least one inlier match.</summary>
	public bool ExistsInlierMatches(uint imageId1, uint imageId2)
	{
		lock (databaseMutex)
		{
			if (!database.ExistsTwoViewGeometry(imageId1, imageId2))
			{
				return false;
			}

			return database.ReadTwoViewGeometry(imageId1, imageId2).InlierMatches.Count != 0;
		}
	}

	/// <summary>Updates the stored two-view geometry of the pair.</summary>
	public void UpdateTwoViewGeometry(uint imageId1, uint imageId2, TwoViewGeometry twoViewGeometry)
	{
		lock (databaseMutex)
		{
			database.UpdateTwoViewGeometry(imageId1, imageId2, twoViewGeometry);
		}
	}

	/// <summary>Writes the raw matches of the pair.</summary>
	public void WriteMatches(uint imageId1, uint imageId2, IReadOnlyList<FeatureMatch> matches)
	{
		lock (databaseMutex)
		{
			database.WriteMatches(imageId1, imageId2, matches);
		}
	}

	/// <summary>Writes the two-view geometry of the pair.</summary>
	public void WriteTwoViewGeometry(uint imageId1, uint imageId2, TwoViewGeometry twoViewGeometry)
	{
		lock (databaseMutex)
		{
			database.WriteTwoViewGeometry(imageId1, imageId2, twoViewGeometry);
		}
	}

	/// <summary>Deletes the raw matches of the pair.</summary>
	public void DeleteMatches(uint imageId1, uint imageId2)
	{
		lock (databaseMutex)
		{
			database.DeleteMatches(imageId1, imageId2);
		}
	}

	/// <summary>Deletes the two-view geometry of the pair.</summary>
	public void DeleteTwoViewGeometry(uint imageId1, uint imageId2)
	{
		lock (databaseMutex)
		{
			database.DeleteTwoViewGeometry(imageId1, imageId2);
		}
	}

	/// <summary>Deletes the inlier matches of the pair, keeping its two-view geometry row.</summary>
	public void DeleteInlierMatches(uint imageId1, uint imageId2)
	{
		lock (databaseMutex)
		{
			database.DeleteInlierMatches(imageId1, imageId2);
		}
	}

	/// <summary>The largest keypoint count of any image, read once and then cached.</summary>
	public long MaxNumKeypoints()
	{
		lock (databaseMutex)
		{
			maxNumKeypoints ??= database.MaxNumKeypoints();
			return maxNumKeypoints.Value;
		}
	}

	private void MaybeLoadCameras()
	{
		lock (databaseMutex)
		{
			if (camerasCache != null)
			{
				return;
			}

			var cameras = new Dictionary<uint, Camera>();
			foreach (Camera camera in database.ReadAllCameras())
			{
				cameras.TryAdd(camera.CameraId, camera);
			}

			camerasCache = cameras;
		}
	}

	private void MaybeLoadFrames()
	{
		lock (databaseMutex)
		{
			if (framesCache != null)
			{
				return;
			}

			var frames = new Dictionary<uint, Frame>();
			foreach (Frame frame in database.ReadAllFrames())
			{
				frames.TryAdd(frame.FrameId, frame);
			}

			framesCache = frames;
		}
	}

	private void MaybeLoadImages()
	{
		MaybeLoadFrames();

		lock (databaseMutex)
		{
			if (imagesCache != null)
			{
				return;
			}

			var images = new Dictionary<uint, Image>();
			foreach (Image image in database.ReadAllImages())
			{
				images.TryAdd(image.ImageId, image);
			}

			imagesCache = images;
		}
	}

	private void MaybeLoadPosePriors()
	{
		MaybeLoadImages();

		lock (databaseMutex)
		{
			if (posePriorsCache != null)
			{
				return;
			}

			// Only pose priors of camera data (images) are kept, keyed by image id.
			var posePriors = new Dictionary<uint, PosePrior>();
			foreach (PosePrior posePrior in database.ReadAllPosePriors())
			{
				if (posePrior.CorrDataId.SensorId.Type == SensorType.Camera)
				{
					uint imageId = (uint)posePrior.CorrDataId.Id;
					Check.That(posePriors.TryAdd(imageId, posePrior), $"Duplicate pose prior for image {imageId}");
				}
			}

			posePriorsCache = posePriors;
		}
	}
}
