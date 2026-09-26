// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Database: port of colmap/scene/database.h and .cc - the abstract store of rigs, cameras,
// frames, images, pose priors, keypoints, descriptors, raw matches and verified two-view
// geometries that feature extraction and matching fill and the mappers read, plus
// Database::Merge, DatabaseTransaction and LoadRandomDatabaseDescriptors. COLMAP's only
// implementation is SQLite (database_sqlite.cc), which is out of scope (native code,
// docs/LICENSE_AUDIT.md); InMemoryDatabase.cs implements the same observable behavior in
// managed memory. Tests: ColmapSharp.Tests/Scene/DatabaseTests.cs (database_test.cc 1:1).
//
// Tier A (exact): pure bookkeeping.
//
// Translation notes:
// - Database::Open / Register (the path-keyed factory registry) are not ported: there are no
//   database files; construct an InMemoryDatabase.
// - For image pairs the order of the ids does not matter: rows are stored under the
//   normalized pair (smaller id first) and swapped back on read, like COLMAP.
// - Counts (size_t) are long; ids keep their COLMAP widths (Util/Types.cs).
// - Read* of a missing entry returns a default-constructed value, exactly like COLMAP (the
//   caller is responsible for making sure the entry exists).
// - DatabaseTransaction's std::mutex becomes a SemaphoreSlim, so a transaction may be ended
//   on another thread than the one that began it (async code).

using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::Database: reads and writes images, features, cameras, matches, etc.
/// For image pairs, the order of <c>imageId1</c> and <c>imageId2</c> does not matter.
/// Wrap many calls in a <see cref="DatabaseTransaction"/> to group them.
/// </summary>
public abstract partial class Database : IDisposable
{
	// Ensures that only one transaction is active at the same time.
	private readonly SemaphoreSlim transactionMutex = new(1, 1);

	/// <summary>Closes the database; any later operation throws. Closing twice is allowed.</summary>
	public abstract void Close();

	/// <summary>Closes the database.</summary>
	public void Dispose()
	{
		Close();
		GC.SuppressFinalize(this);
	}

	/// <summary>Whether the rig exists.</summary>
	public abstract bool ExistsRig(uint rigId);

	/// <summary>Whether the camera exists.</summary>
	public abstract bool ExistsCamera(uint cameraId);

	/// <summary>Whether the frame exists.</summary>
	public abstract bool ExistsFrame(uint frameId);

	/// <summary>Whether the image exists.</summary>
	public abstract bool ExistsImage(uint imageId);

	/// <summary>Whether an image with this name exists.</summary>
	public abstract bool ExistsImageWithName(string name);

	/// <summary>
	/// Whether the pose prior exists. <paramref name="isDeprecatedImagePrior"/> must be passed
	/// as false: COLMAP throws for the old image-keyed pose prior API (its default).
	/// </summary>
	public abstract bool ExistsPosePrior(uint posePriorId, bool isDeprecatedImagePrior = true);

	/// <summary>Whether keypoints exist for the image.</summary>
	public abstract bool ExistsKeypoints(uint imageId);

	/// <summary>Whether descriptors exist for the image.</summary>
	public abstract bool ExistsDescriptors(uint imageId);

	/// <summary>Whether raw matches exist for the image pair.</summary>
	public abstract bool ExistsMatches(uint imageId1, uint imageId2);

	/// <summary>Whether a two-view geometry exists for the image pair.</summary>
	public abstract bool ExistsTwoViewGeometry(uint imageId1, uint imageId2);

	/// <summary>Number of rigs.</summary>
	public abstract long NumRigs();

	/// <summary>Number of cameras.</summary>
	public abstract long NumCameras();

	/// <summary>Number of frames.</summary>
	public abstract long NumFrames();

	/// <summary>Number of images.</summary>
	public abstract long NumImages();

	/// <summary>Number of pose priors.</summary>
	public abstract long NumPosePriors();

	/// <summary>Total number of keypoints over all images.</summary>
	public abstract long NumKeypoints();

	/// <summary>The number of keypoints of the image with the most keypoints.</summary>
	public abstract long MaxNumKeypoints();

	/// <summary>Number of keypoints of one image (0 if none).</summary>
	public abstract long NumKeypointsForImage(uint imageId);

	/// <summary>Total number of descriptors over all images.</summary>
	public abstract long NumDescriptors();

	/// <summary>The number of descriptors of the image with the most descriptors.</summary>
	public abstract long MaxNumDescriptors();

	/// <summary>Number of descriptors of one image (0 if none).</summary>
	public abstract long NumDescriptorsForImage(uint imageId);

	/// <summary>Total number of raw matches over all pairs.</summary>
	public abstract long NumMatches();

	/// <summary>Total number of inlier matches over all two-view geometries.</summary>
	public abstract long NumInlierMatches();

	/// <summary>Number of image pairs with a raw matches entry.</summary>
	public abstract long NumMatchedImagePairs();

	/// <summary>Number of image pairs with a two-view geometry entry.</summary>
	public abstract long NumVerifiedImagePairs();

	/// <summary>Reads a rig (a default Rig if missing).</summary>
	public abstract Rig ReadRig(uint rigId);

	/// <summary>The rig containing the sensor (as reference or non-reference sensor), if any.</summary>
	public abstract Rig? ReadRigWithSensor(SensorId sensorId);

	/// <summary>All rigs, ordered by id.</summary>
	public abstract List<Rig> ReadAllRigs();

	/// <summary>Reads a camera (a default Camera if missing).</summary>
	public abstract Camera ReadCamera(uint cameraId);

	/// <summary>All cameras, ordered by id.</summary>
	public abstract List<Camera> ReadAllCameras();

	/// <summary>Reads a frame (a default Frame if missing).</summary>
	public abstract Frame ReadFrame(uint frameId);

	/// <summary>All frames, ordered by id.</summary>
	public abstract List<Frame> ReadAllFrames();

	/// <summary>Reads an image (a default Image if missing); its frame id comes from the frames.</summary>
	public abstract Image ReadImage(uint imageId);

	/// <summary>The image with this name, if any.</summary>
	public abstract Image? ReadImageWithName(string name);

	/// <summary>All images, ordered by id.</summary>
	public abstract List<Image> ReadAllImages();

	/// <summary>Reads a pose prior (a default PosePrior if missing); see <see cref="ExistsPosePrior"/>.</summary>
	public abstract PosePrior ReadPosePrior(uint posePriorId, bool isDeprecatedImagePrior = true);

	/// <summary>All pose priors, ordered by id.</summary>
	public abstract List<PosePrior> ReadAllPosePriors();

	/// <summary>The stored keypoint blob (N x 2, 4 or 6 floats; 0 x 0 if missing).</summary>
	public abstract RowMajorMatrix<float> ReadKeypointsBlob(uint imageId);

	/// <summary>The image's keypoints (empty if missing).</summary>
	public abstract List<FeatureKeypoint> ReadKeypoints(uint imageId);

	/// <summary>The image's descriptors (empty and UNDEFINED if missing).</summary>
	public abstract FeatureDescriptors ReadDescriptors(uint imageId);

	/// <summary>The raw matches blob of a pair, oriented as asked (0 x 2 if missing).</summary>
	public abstract RowMajorMatrix<uint> ReadMatchesBlob(uint imageId1, uint imageId2);

	/// <summary>The raw matches of a pair, oriented as asked (empty if missing).</summary>
	public abstract List<FeatureMatch> ReadMatches(uint imageId1, uint imageId2);

	/// <summary>All non-empty raw match blobs, by pair id, in the stored (normalized) orientation.</summary>
	public abstract List<(ulong PairId, RowMajorMatrix<uint> Matches)> ReadAllMatchesBlob();

	/// <summary>All non-empty raw matches, by pair id, in the stored (normalized) orientation.</summary>
	public abstract List<(ulong PairId, List<FeatureMatch> Matches)> ReadAllMatches();

	/// <summary>The number of raw matches of every pair with at least one.</summary>
	public abstract List<(ulong PairId, int NumMatches)> ReadNumMatches();

	/// <summary>The two-view geometry of a pair, inverted if asked in swapped order.</summary>
	public abstract TwoViewGeometry ReadTwoViewGeometry(uint imageId1, uint imageId2);

	/// <summary>
	/// All two-view geometries that carry anything (inliers, a matrix, a pose or a camera),
	/// by pair id, in the stored (normalized) orientation.
	/// </summary>
	public abstract List<(ulong PairId, TwoViewGeometry TwoViewGeometry)> ReadTwoViewGeometries();

	/// <summary>The inlier count of every two-view geometry with at least one inlier match.</summary>
	public abstract List<(ulong PairId, int NumInliers)> ReadTwoViewGeometryNumInliers();

	/// <summary>Adds a rig; returns its id (new unless <paramref name="useRigId"/>).</summary>
	public abstract uint WriteRig(Rig rig, bool useRigId = false);

	/// <summary>Adds a camera; returns its id (new unless <paramref name="useCameraId"/>).</summary>
	public abstract uint WriteCamera(Camera camera, bool useCameraId = false);

	/// <summary>Adds a frame; returns its id (new unless <paramref name="useFrameId"/>).</summary>
	public abstract uint WriteFrame(Frame frame, bool useFrameId = false);

	/// <summary>Adds an image; returns its id (new unless <paramref name="useImageId"/>).</summary>
	public abstract uint WriteImage(Image image, bool useImageId = false);

	/// <summary>Adds a pose prior; returns its id (new unless <paramref name="usePosePriorId"/>).</summary>
	public abstract uint WritePosePrior(PosePrior posePrior, bool usePosePriorId = false);

	/// <summary>Adds an image's keypoints (stored as an N x 6 blob).</summary>
	public abstract void WriteKeypoints(uint imageId, IReadOnlyList<FeatureKeypoint> keypoints);

	/// <summary>Adds an image's keypoint blob (N x 2, 4 or 6).</summary>
	public abstract void WriteKeypoints(uint imageId, RowMajorMatrix<float> blob);

	/// <summary>Adds an image's descriptors.</summary>
	public abstract void WriteDescriptors(uint imageId, FeatureDescriptors descriptors);

	/// <summary>Adds the raw matches of a pair.</summary>
	public abstract void WriteMatches(uint imageId1, uint imageId2, IReadOnlyList<FeatureMatch> matches);

	/// <summary>Adds the raw matches blob (N x 2) of a pair.</summary>
	public abstract void WriteMatches(uint imageId1, uint imageId2, RowMajorMatrix<uint> blob);

	/// <summary>Adds the two-view geometry of a pair; throws if one exists.</summary>
	public abstract void WriteTwoViewGeometry(uint imageId1, uint imageId2, TwoViewGeometry twoViewGeometry);

	/// <summary>Updates an existing rig (reference sensor and sensors).</summary>
	public abstract void UpdateRig(Rig rig);

	/// <summary>Updates an existing camera.</summary>
	public abstract void UpdateCamera(Camera camera);

	/// <summary>Updates an existing frame (rig id and data ids).</summary>
	public abstract void UpdateFrame(Frame frame);

	/// <summary>Updates an existing image (name and camera id).</summary>
	public abstract void UpdateImage(Image image);

	/// <summary>Updates an existing pose prior.</summary>
	public abstract void UpdatePosePrior(PosePrior posePrior);

	/// <summary>Updates an existing image's keypoints.</summary>
	public abstract void UpdateKeypoints(uint imageId, IReadOnlyList<FeatureKeypoint> keypoints);

	/// <summary>Updates an existing image's keypoint blob.</summary>
	public abstract void UpdateKeypoints(uint imageId, RowMajorMatrix<float> blob);

	/// <summary>Replaces an existing two-view geometry; does nothing if there is none.</summary>
	public abstract void UpdateTwoViewGeometry(uint imageId1, uint imageId2, TwoViewGeometry twoViewGeometry);

	/// <summary>Deletes the raw matches of a pair.</summary>
	public abstract void DeleteMatches(uint imageId1, uint imageId2);

	/// <summary>Deletes the two-view geometry of a pair.</summary>
	public abstract void DeleteTwoViewGeometry(uint imageId1, uint imageId2);

	/// <summary>Clears the inlier matches of a pair's two-view geometry, keeping the rest.</summary>
	public abstract void DeleteInlierMatches(uint imageId1, uint imageId2);

	/// <summary>Clears every table.</summary>
	public abstract void ClearAllTables();

	/// <summary>Clears the rigs (and, through the rig reference, their frames).</summary>
	public abstract void ClearRigs();

	/// <summary>Clears the cameras; fails while images still reference them.</summary>
	public abstract void ClearCameras();

	/// <summary>Clears the frames.</summary>
	public abstract void ClearFrames();

	/// <summary>Clears the images and with them their keypoints and descriptors.</summary>
	public abstract void ClearImages();

	/// <summary>Clears the pose priors.</summary>
	public abstract void ClearPosePriors();

	/// <summary>Clears the descriptors.</summary>
	public abstract void ClearDescriptors();

	/// <summary>Clears the keypoints.</summary>
	public abstract void ClearKeypoints();

	/// <summary>Clears the raw matches.</summary>
	public abstract void ClearMatches();

	/// <summary>Clears the two-view geometries.</summary>
	public abstract void ClearTwoViewGeometries();

	/// <summary>Begins a transaction (use <see cref="DatabaseTransaction"/>).</summary>
	public abstract void BeginTransaction();

	/// <summary>Ends a transaction (use <see cref="DatabaseTransaction"/>).</summary>
	public abstract void EndTransaction();

	internal SemaphoreSlim TransactionMutex => transactionMutex;
}

/// <summary>
/// Port of colmap::DatabaseTransaction: begins a transaction on construction and ends it on
/// Dispose, holding the database's transaction lock in between so only one transaction is
/// active at a time.
/// </summary>
public sealed class DatabaseTransaction : IDisposable
{
	private Database? database;

	/// <summary>Waits for the database's transaction lock and begins a transaction.</summary>
	public DatabaseTransaction(Database database)
	{
		Check.NotNull(database);
		database.TransactionMutex.Wait();
		try
		{
			database.BeginTransaction();
		}
		catch
		{
			database.TransactionMutex.Release();
			throw;
		}

		this.database = database;
	}

	/// <summary>Ends the transaction and releases the lock.</summary>
	public void Dispose()
	{
		Database? active = Interlocked.Exchange(ref database, null);
		if (active is null)
		{
			return;
		}

		try
		{
			active.EndTransaction();
		}
		finally
		{
			active.TransactionMutex.Release();
		}
	}
}
