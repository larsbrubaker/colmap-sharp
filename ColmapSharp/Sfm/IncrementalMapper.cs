// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalMapper: port of colmap/sfm/incremental_mapper.h and .cc, the class that holds
// the state of one incremental reconstruction and provides every step of it: seeding from an
// image pair, registering the next images, triangulating, bundle adjusting and filtering. The
// algorithms it forwards to live in IncrementalMapperImpl.cs. This file holds the options,
// the lifecycle (Begin/EndReconstruction), the initial pair, the getters and the
// registration statistics; IncrementalMapper.Register.cs holds next-image registration
// (central, generalized and structure-less); IncrementalMapper.Adjust.cs holds
// triangulation, bundle adjustment, refinement loops and filtering. Tests:
// ColmapSharp.Tests/Sfm/IncrementalMapperTests.cs (incremental_mapper_test.cc 1:1).
//
// Tier C (outcome): registration and bundle adjustment run RANSAC and the solver; the
// registration statistics are exact bookkeeping.
//
// Translation notes:
// - The statistics maps are Dictionaries. C++ reads several of them with operator[], which
//   inserts a zero entry; CountAt does the same, so NumRegFramesPerRig and
//   NumRegImagesPerCamera hold the same keys as in COLMAP.
// - shared_ptr members are nullable references; THROW_CHECK_NOTNULL is Check.NotNull.
// - size_t counts are int, as in the triangulator.

using System.Runtime.InteropServices;

using ColmapSharp.Geometry;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Sfm;

/// <summary>
/// Port of colmap::IncrementalMapper: provides all functionality for the incremental
/// reconstruction procedure. Typical use: BeginReconstruction, FindInitialImagePair and
/// RegisterInitialImagePair, then repeatedly FindNextImages / RegisterNextImage with local or
/// global bundle adjustment, and finally EndReconstruction.
/// </summary>
public sealed partial class IncrementalMapper
{
	/// <summary>Port of IncrementalMapper::Options::ImageSelectionMethod.</summary>
	public enum ImageSelectionMethod
	{
		/// <summary>Rank by the number of visible 3D points.</summary>
		MaxVisiblePointsNum,

		/// <summary>Rank by the ratio of visible 3D points to observations.</summary>
		MaxVisiblePointsRatio,

		/// <summary>Rank by the 3D point visibility score (spatial distribution).</summary>
		MinUncertainty,
	}

	/// <summary>Port of IncrementalMapper::Options.</summary>
	public sealed class Options
	{
		/// <summary>Minimum number of inliers for initial image pair.</summary>
		public int InitMinNumInliers { get; set; } = 100;

		/// <summary>Maximum error in pixels for two-view geometry estimation for initial image pair.</summary>
		public double InitMaxError { get; set; } = 4.0;

		/// <summary>Maximum forward motion for initial image pair.</summary>
		public double InitMaxForwardMotion { get; set; } = 0.95;

		/// <summary>Minimum triangulation angle for initial image pair.</summary>
		public double InitMinTriAngle { get; set; } = 16.0;

		/// <summary>Maximum number of trials to use an image for initialization.</summary>
		public int InitMaxRegTrials { get; set; } = 2;

		/// <summary>Maximum reprojection error in absolute pose estimation.</summary>
		public double AbsPoseMaxError { get; set; } = 12.0;

		/// <summary>Minimum number of inliers in absolute pose estimation.</summary>
		public int AbsPoseMinNumInliers { get; set; } = 30;

		/// <summary>Minimum inlier ratio in absolute pose estimation.</summary>
		public double AbsPoseMinInlierRatio { get; set; } = 0.25;

		/// <summary>Whether to estimate the focal length in absolute pose estimation.</summary>
		public bool AbsPoseRefineFocalLength { get; set; } = true;

		/// <summary>Whether to estimate the extra parameters in absolute pose estimation.</summary>
		public bool AbsPoseRefineExtraParams { get; set; } = true;

		/// <summary>Number of images to optimize in local bundle adjustment.</summary>
		public int BaLocalNumImages { get; set; } = 6;

		/// <summary>Minimum triangulation for images to be chosen in local bundle adjustment.</summary>
		public double BaLocalMinTriAngle { get; set; } = 6;

		/// <summary>
		/// Whether to ignore redundant 3D points in bundle adjustment when jointly optimizing
		/// all parameters. If this is enabled, then the bundle adjustment problem is first
		/// solved with a reduced set of 3D points and then the remaining 3D points are
		/// optimized in a second step with all other parameters fixed. Points explicitly
		/// configured as constant or variable are not ignored. This is only activated when
		/// the reconstruction has reached sufficient size with at least 10 registered frames.
		/// </summary>
		public bool BaGlobalIgnoreRedundantPoints3D { get; set; }

		/// <summary>
		/// The minimum coverage gain for any 3D point to be included in global bundle
		/// adjustment. A larger value means more 3D points are pruned.
		/// </summary>
		public double BaGlobalIgnoreRedundantPoints3DMinCoverageGain { get; set; } = 0.05;

		/// <summary>
		/// Thresholds for bogus camera parameters. Images with bogus camera parameters are
		/// filtered and ignored in triangulation. Opening angle of ~130deg.
		/// </summary>
		public double MinFocalLengthRatio { get; set; } = 0.1;

		/// <summary>Opening angle of ~5deg.</summary>
		public double MaxFocalLengthRatio { get; set; } = 10;

		/// <summary>Maximum absolute value of the extra parameters.</summary>
		public double MaxExtraParam { get; set; } = 1;

		/// <summary>Maximum reprojection error in pixels for observations.</summary>
		public double FilterMaxReprojError { get; set; } = 4.0;

		/// <summary>Minimum triangulation angle in degrees for stable 3D points.</summary>
		public double FilterMinTriAngle { get; set; } = 1.5;

		/// <summary>Maximum number of trials to register an image.</summary>
		public int MaxRegTrials { get; set; } = 3;

		/// <summary>If reconstruction is provided as input, fix the existing image poses.</summary>
		public bool FixExistingFrames { get; set; }

		/// <summary>
		/// Rigs for which to fix the sensor_from_rig transformation, independent of
		/// ba_refine_sensor_from_rig.
		/// </summary>
		public HashSet<uint> ConstantRigs { get; set; } = [];

		/// <summary>
		/// Cameras for which to fix the camera parameters independent of refine_focal_length,
		/// refine_principal_point, and refine_extra_params.
		/// </summary>
		public HashSet<uint> ConstantCameras { get; set; } = [];

		/// <summary>Whether to use prior camera positions.</summary>
		public bool UsePriorPosition { get; set; }

		/// <summary>Whether to use a robust loss on prior locations.</summary>
		public bool UseRobustLossOnPriorPosition { get; set; }

		/// <summary>Threshold on the residual for the robust loss (chi2 for 3DOF at 95% = 7.815).</summary>
		public double PriorPositionLossScale { get; set; } = 7.815;

		/// <summary>Number of threads.</summary>
		public int NumThreads { get; set; } = -1;

		/// <summary>PRNG seed for all stochastic methods during reconstruction.</summary>
		public int RandomSeed { get; set; } = -1;

		/// <summary>Method to find and select next best image to register.</summary>
		public ImageSelectionMethod ImageSelectionMethod { get; set; } = ImageSelectionMethod.MinUncertainty;

		/// <summary>Port of Options::Check (CHECK_OPTION_*: false on a violation).</summary>
		public bool Check() =>
			InitMinNumInliers > 0
			&& InitMaxError > 0.0
			&& InitMaxForwardMotion >= 0.0
			&& InitMaxForwardMotion <= 1.0
			&& InitMinTriAngle >= 0.0
			&& InitMaxRegTrials >= 1
			&& AbsPoseMaxError > 0.0
			&& AbsPoseMinNumInliers > 0
			&& AbsPoseMinInlierRatio >= 0.0
			&& AbsPoseMinInlierRatio <= 1.0
			&& BaLocalNumImages >= 2
			&& BaLocalMinTriAngle >= 0.0
			&& BaGlobalIgnoreRedundantPoints3DMinCoverageGain >= 0.0
			&& MinFocalLengthRatio >= 0.0
			&& MaxFocalLengthRatio >= MinFocalLengthRatio
			&& MaxExtraParam >= 0.0
			&& FilterMaxReprojError >= 0.0
			&& FilterMinTriAngle >= 0.0
			&& MaxRegTrials >= 1
			&& NumThreads >= -1
			&& RandomSeed >= -1;

		/// <summary>A deep copy (C++ copies the options struct by value).</summary>
		public Options Clone()
		{
			var clone = (Options)MemberwiseClone();
			clone.ConstantRigs = [.. ConstantRigs];
			clone.ConstantCameras = [.. ConstantCameras];
			return clone;
		}
	}

	/// <summary>Port of IncrementalMapper::LocalBundleAdjustmentReport.</summary>
	public sealed class LocalBundleAdjustmentReport
	{
		/// <summary>Observations merged into other tracks after the adjustment.</summary>
		public int NumMergedObservations { get; set; }

		/// <summary>Observations added by track completion after the adjustment.</summary>
		public int NumCompletedObservations { get; set; }

		/// <summary>Observations removed by filtering.</summary>
		public int NumFilteredObservations { get; set; }

		/// <summary>Observations in the adjusted problem (residuals / 2).</summary>
		public int NumAdjustedObservations { get; set; }
	}

	// Port of IncrementalMapper::RegistrationStatistics.
	private sealed class RegistrationStatistics
	{
		// Number of images that are registered in at least one reconstruction.
		public int NumTotalRegImages;

		// Number of shared images between current reconstruction and all other previous
		// reconstructions.
		public int NumSharedRegImages;

		// Images and image pairs that have been used for initialization. Each image and
		// image pair is only tried once for initialization.
		public readonly Dictionary<uint, int> InitNumRegTrials = [];
		public readonly HashSet<ulong> InitImagePairs = [];

		// The number of registered frames/images per rig/camera. This information is used to
		// avoid duplicate refinement of rig/camera parameters and degradation of already
		// refined rig/camera parameters in local bundle adjustment when multiple frames share
		// rigs or images share intrinsics.
		public readonly Dictionary<uint, int> NumRegFramesPerRig = [];
		public readonly Dictionary<uint, int> NumRegImagesPerCamera = [];

		// The number of reconstructions in which images are registered.
		public readonly Dictionary<uint, int> NumRegistrations = [];

		// Number of trials to register image in current reconstruction. Used to set an upper
		// bound to the number of trials to register an image.
		public readonly Dictionary<uint, int> NumRegTrials = [];
		public readonly Dictionary<uint, int> NumStructureLessRegTrials = [];
	}

	// Holds all necessary data from database in memory.
	private readonly DatabaseCache _databaseCache;

	// Holds data of the reconstruction.
	private Reconstruction? _reconstruction;

	// Responsible for keeping track of 3D point statistics.
	private ObservationManager? _obsManager;

	// Responsible for incremental triangulation.
	private IncrementalTriangulator? _triangulator;

	private readonly RegistrationStatistics _regStats = new();

	// Frames that have been filtered in current reconstruction.
	private readonly HashSet<uint> _filteredFrames = [];

	// Frames that were registered before beginning the reconstruction. This frame list will
	// be non-empty, if the reconstruction is continued from an existing reconstruction.
	private HashSet<uint> _existingFrameIds = [];

	/// <summary>
	/// Creates an incremental mapper. The database cache must live for the entire life-time
	/// of the incremental mapper.
	/// </summary>
	public IncrementalMapper(DatabaseCache databaseCache)
	{
		_databaseCache = databaseCache;
	}

	/// <summary>The current reconstruction; null outside Begin/EndReconstruction.</summary>
	public Reconstruction? Reconstruction => _reconstruction;

	/// <summary>The observation manager of the current reconstruction.</summary>
	public ObservationManager ObservationManager => Check.NotNull(_obsManager);

	/// <summary>The triangulator of the current reconstruction.</summary>
	public IncrementalTriangulator Triangulator => Check.NotNull(_triangulator);

	/// <summary>Frames that have been filtered in the current reconstruction.</summary>
	public IReadOnlySet<uint> FilteredFrames => _filteredFrames;

	/// <summary>Frames that were registered when the reconstruction began.</summary>
	public IReadOnlySet<uint> ExistingFrameIds => _existingFrameIds;

	/// <summary>The number of registered frames per rig.</summary>
	public IReadOnlyDictionary<uint, int> NumRegFramesPerRig => _regStats.NumRegFramesPerRig;

	/// <summary>The number of registered images per camera.</summary>
	public IReadOnlyDictionary<uint, int> NumRegImagesPerCamera => _regStats.NumRegImagesPerCamera;

	/// <summary>Number of images that are registered in at least one reconstruction.</summary>
	public int NumTotalRegImages => _regStats.NumTotalRegImages;

	/// <summary>
	/// Number of shared images between current reconstruction and all other previous
	/// reconstructions.
	/// </summary>
	public int NumSharedRegImages => _regStats.NumSharedRegImages;

	// C++'s map operator[] read: the count, inserting 0 for a missing key.
	internal static int CountAt(Dictionary<uint, int> counts, uint key) =>
		CollectionsMarshal.GetValueRefOrAddDefault(counts, key, out _);

	/// <summary>
	/// Prepares the mapper for a new reconstruction, which might have existing registered
	/// images (in which case RegisterNextImage must be called) or which is empty (in which
	/// case RegisterInitialImagePair must be called).
	/// </summary>
	public void BeginReconstruction(Reconstruction reconstruction)
	{
		Check.That(_reconstruction is null);
		_reconstruction = reconstruction;
		_reconstruction.Load(_databaseCache);
		_obsManager = new ObservationManager(_reconstruction, _databaseCache.CorrespondenceGraph);
		_triangulator = new IncrementalTriangulator(_databaseCache.CorrespondenceGraph, _reconstruction, _obsManager);

		_regStats.NumSharedRegImages = 0;
		_regStats.NumRegFramesPerRig.Clear();
		_regStats.NumRegImagesPerCamera.Clear();
		foreach (uint frameId in _reconstruction.RegFrameIds)
		{
			RegisterFrameEvent(frameId);
		}

		_existingFrameIds = [.. reconstruction.RegFrameIds];

		_filteredFrames.Clear();
		_regStats.NumRegTrials.Clear();
		_regStats.NumStructureLessRegTrials.Clear();
	}

	/// <summary>
	/// Cleans up the mapper after the current reconstruction is done. If the model is
	/// discarded, the number of total and shared registered images will be updated
	/// accordingly.
	/// </summary>
	public void EndReconstruction(bool discard)
	{
		Reconstruction reconstruction = Check.NotNull(_reconstruction);

		if (discard)
		{
			// Need to copy data, because de-registration removes elements from the
			// underlying vector.
			var regFrameIds = reconstruction.RegFrameIds.ToList();
			foreach (uint frameId in regFrameIds)
			{
				Check.NotNull(_obsManager).DeRegisterFrame(frameId);
				DeRegisterFrameEvent(frameId);
			}
		}

		_triangulator = null;
		_obsManager = null;
		reconstruction.TearDown();
		_reconstruction = null;
	}

	/// <summary>
	/// Finds the initial image pair to seed the incremental reconstruction. The image pair
	/// should be passed to RegisterInitialImagePair. This function automatically ignores
	/// image pairs that failed to register previously. On input, a valid
	/// <paramref name="imageId1"/> or <paramref name="imageId2"/> constrains the search; on
	/// success all three arguments hold the chosen pair.
	/// </summary>
	public bool FindInitialImagePair(Options options, ref uint imageId1, ref uint imageId2, ref Rigid3d cam2FromCam1)
	{
		Reconstruction reconstruction = Check.NotNull(_reconstruction);
		IncrementalMapperImpl.InitInfo? initInfo = IncrementalMapperImpl.FindInitialImagePair(
			options,
			_databaseCache,
			reconstruction,
			_regStats.InitNumRegTrials,
			_regStats.NumRegistrations,
			_regStats.InitImagePairs,
			imageId1,
			imageId2);
		if (initInfo is null)
		{
			return false;
		}

		imageId1 = initInfo.ImageId1;
		imageId2 = initInfo.ImageId2;
		cam2FromCam1 = initInfo.Cam2FromCam1;
		SeedEstimatedInitialCameras(reconstruction, imageId1, imageId2, initInfo.Camera1, initInfo.Camera2);
		return true;
	}

	/// <summary>
	/// Estimates the two-view geometry of an image pair and checks whether it is suitable
	/// for initialization.
	/// </summary>
	public bool EstimateInitialTwoViewGeometry(Options options, uint imageId1, uint imageId2, ref Rigid3d cam2FromCam1)
	{
		IncrementalMapperImpl.InitInfo? initInfo =
			IncrementalMapperImpl.EstimateInitialTwoViewGeometry(options, _databaseCache, imageId1, imageId2);
		if (initInfo is null)
		{
			return false;
		}

		cam2FromCam1 = initInfo.Cam2FromCam1;
		SeedEstimatedInitialCameras(
			Check.NotNull(_reconstruction), imageId1, imageId2, initInfo.Camera1, initInfo.Camera2);
		return true;
	}

	/// <summary>
	/// Finds the best next images to register in the incremental reconstruction, best first.
	/// The images should be passed to RegisterNextImage (or RegisterNextStructureLessImage
	/// when <paramref name="structureLess"/>). This function automatically ignores images
	/// that failed to register for max_reg_trials.
	/// </summary>
	public List<uint> FindNextImages(Options options, bool structureLess = false) =>
		IncrementalMapperImpl.FindNextImages(
			options,
			Check.NotNull(_obsManager),
			_filteredFrames,
			structureLess ? _regStats.NumStructureLessRegTrials : _regStats.NumRegTrials,
			structureLess);

	/// <summary>Attempts to seed the reconstruction from an image pair.</summary>
	public void RegisterInitialImagePair(Options options, uint imageId1, uint imageId2, Rigid3d cam2FromCam1)
	{
		Reconstruction reconstruction = Check.NotNull(_reconstruction);
		ObservationManager obsManager = Check.NotNull(_obsManager);
		Check.Eq(reconstruction.NumRegFrames, 0);

		Check.That(options.Check());

		Increment(_regStats.InitNumRegTrials, imageId1);
		Increment(_regStats.InitNumRegTrials, imageId2);
		Increment(_regStats.NumRegTrials, imageId1);
		Increment(_regStats.NumRegTrials, imageId2);

		ulong pairId = ImagePairToPairId(imageId1, imageId2);
		_regStats.InitImagePairs.Add(pairId);

		Image image1 = reconstruction.Image(imageId1);
		Image image2 = reconstruction.Image(imageId2);

		// Apply two-view geometry.
		image1.FramePtr.SetCamFromWorld(image1.CameraId, Rigid3d.Identity);
		image2.FramePtr.SetCamFromWorld(image2.CameraId, cam2FromCam1);

		// Update reconstruction.
		obsManager.RegisterFrame(image1.FrameId);
		RegisterFrameEvent(image1.FrameId);
		obsManager.RegisterFrame(image2.FrameId);
		RegisterFrameEvent(image2.FrameId);
	}

	/// <summary>
	/// Resets registration statistics for initialization. This can be used when relaxing the
	/// initialization thresholds, such that previously tried pairs will be tried again.
	/// </summary>
	public void ResetInitializationStats()
	{
		_regStats.InitImagePairs.Clear();
		_regStats.InitNumRegTrials.Clear();
	}

	/// <summary>
	/// The 3D points changed since the last ClearModifiedPoints3D. This is the
	/// triangulator's live set: later triangulation, merging and filtering change it.
	/// </summary>
	public IReadOnlySet<ulong> GetModifiedPoints3D() => Check.NotNull(_triangulator).GetModifiedPoints3D();

	/// <summary>Clears the collection of changed 3D points.</summary>
	public void ClearModifiedPoints3D() => Check.NotNull(_triangulator).ClearModifiedPoints3D();

	/// <summary>
	/// Finds the local bundle for the given image in the reconstruction. The local bundle is
	/// defined as the images that are most connected, i.e. maximum number of shared 3D
	/// points, to the given image.
	/// </summary>
	public List<uint> FindLocalBundle(Options options, uint imageId) =>
		IncrementalMapperImpl.FindLocalBundle(options, imageId, Check.NotNull(_reconstruction));

	private static void Increment(Dictionary<uint, int> counts, uint key) =>
		CollectionsMarshal.GetValueRefOrAddDefault(counts, key, out _) += 1;

	// Seeds the intrinsics estimated for the initial image pair onto the reconstruction,
	// leaving them optimizable for bundle adjustment. No-op for cameras whose intrinsics were
	// not estimated (null). C++ assigns the whole camera through CameraPtr.
	private static void SeedEstimatedInitialCameras(
		Reconstruction reconstruction, uint imageId1, uint imageId2, Camera? camera1, Camera? camera2)
	{
		if (camera1 is not null)
		{
			AssignCamera(reconstruction.Image(imageId1).CameraPtr, camera1);
		}

		if (camera2 is not null)
		{
			AssignCamera(reconstruction.Image(imageId2).CameraPtr, camera2);
		}
	}

	private static void AssignCamera(Camera target, Camera source)
	{
		target.CameraId = source.CameraId;
		target.ModelId = source.ModelId;
		target.Width = source.Width;
		target.Height = source.Height;
		target.Params = (double[])source.Params.Clone();
		target.HasPriorFocalLength = source.HasPriorFocalLength;
	}

	// Resets the parameters of a reconstruction camera to the database's values. The copy
	// goes into the existing array (bundle adjustment writes Camera.Params in place, see
	// Reconstruction.cs), unless a seeded camera changed the parameter count.
	private void ResetCameraParams(Camera camera)
	{
		double[] databaseParams = _databaseCache.Camera(camera.CameraId).Params;
		if (camera.Params.Length == databaseParams.Length)
		{
			databaseParams.CopyTo(camera.Params, 0);
		}
		else
		{
			camera.Params = (double[])databaseParams.Clone();
		}
	}

	// Register / de-register frame in current reconstruction and update the (shared)
	// registration statistics.
	private void RegisterFrameEvent(uint frameId)
	{
		Reconstruction reconstruction = Check.NotNull(_reconstruction);
		Frame frame = reconstruction.Frame(frameId);

		Increment(_regStats.NumRegFramesPerRig, frame.RigId);

		foreach (DataId dataId in frame.ImageIds())
		{
			uint imageId = (uint)dataId.Id;
			Image image = reconstruction.Image(imageId);

			Increment(_regStats.NumRegImagesPerCamera, image.CameraId);

			ref int numRegsForImage = ref CollectionsMarshal.GetValueRefOrAddDefault(_regStats.NumRegistrations, imageId, out _);
			numRegsForImage += 1;
			if (numRegsForImage == 1)
			{
				_regStats.NumTotalRegImages += 1;
			}
			else if (numRegsForImage > 1)
			{
				_regStats.NumSharedRegImages += 1;
			}
		}
	}

	private void DeRegisterFrameEvent(uint frameId)
	{
		Reconstruction reconstruction = Check.NotNull(_reconstruction);
		Frame frame = reconstruction.Frame(frameId);

		ref int numRegFramesForRig = ref CollectionsMarshal.GetValueRefOrNullRef(_regStats.NumRegFramesPerRig, frame.RigId);
		Check.That(!System.Runtime.CompilerServices.Unsafe.IsNullRef(ref numRegFramesForRig));
		Check.Gt(numRegFramesForRig, 0);
		numRegFramesForRig -= 1;

		foreach (DataId dataId in frame.ImageIds())
		{
			uint imageId = (uint)dataId.Id;
			Image image = reconstruction.Image(imageId);

			ref int numRegImagesForCamera =
				ref CollectionsMarshal.GetValueRefOrNullRef(_regStats.NumRegImagesPerCamera, image.CameraId);
			Check.That(!System.Runtime.CompilerServices.Unsafe.IsNullRef(ref numRegImagesForCamera));
			Check.Gt(numRegImagesForCamera, 0);
			numRegImagesForCamera -= 1;

			ref int numRegsForImage = ref CollectionsMarshal.GetValueRefOrAddDefault(_regStats.NumRegistrations, imageId, out _);
			numRegsForImage -= 1;
			if (numRegsForImage == 0)
			{
				_regStats.NumTotalRegImages -= 1;
			}
			else if (numRegsForImage > 0)
			{
				_regStats.NumSharedRegImages -= 1;
			}
		}
	}
}
