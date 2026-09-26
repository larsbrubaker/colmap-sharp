// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// IncrementalTriangulator: port of colmap/sfm/incremental_triangulator.h and .cc, which
// creates, continues, completes, merges and re-triangulates 3D points while the incremental
// mapper registers images. It reads correspondences from a CorrespondenceGraph
// (Scene/CorrespondenceGraph.cs), estimates points with EstimateTriangulation
// (Estimators/TriangulationEstimation.cs) and writes every change through an
// ObservationManager (Sfm/ObservationManager.cs) so the mapper's statistics stay in sync.
// This file holds the options, construction and the public operations;
// IncrementalTriangulator.Track.cs holds the private per-track steps (Find, Create,
// Continue, Merge, Complete) and the bogus-camera cache. Tests:
// ColmapSharp.Tests/Sfm/IncrementalTriangulatorTests.cs (incremental_triangulator_test.cc 1:1).
//
// Tier C (outcome): Create goes through RANSAC triangulation; everything else is exact
// bookkeeping on top of it.
//
// Translation notes:
// - CorrData. COLMAP stores a `const Point2D*` and re-reads it after the reconstruction
//   changed (Create's recursion and TriangulateImage's Continue-then-Create both rely on
//   seeing an observation that was just triangulated). Point2D is a struct here, so CorrData
//   keeps the image and the index and reads the point through Image.Point2DAt each time.
// - Buffers. COLMAP reuses `found_corrs_`, the BFS queues and the visited set across calls;
//   so does the port, and it also reuses the per-call vectors COLMAP allocates (Find's
//   output, Create's filtered copy per recursion depth, TriangulateTrack's inputs and
//   Retriangulate's matches), because these run for every observation of every image.
// - Iteration order. Retriangulate visits ObservationManager.ImagePairs in insertion order
//   (docs/CPP_DIVERGENCES.md, entry 50); CompleteAllTracks/MergeAllTracks visit
//   Reconstruction.Point3DIds in ascending id order (entry 21); the modified-point set,
//   merge trials and visited set are HashSets, whose order is deterministic for the same
//   sequence of operations. COLMAP iterates absl hash sets and maps there (entry 51).
// - The merge-trial and BFS-visited keys are packed into one ulong / (ulong, ulong)
//   instead of COLMAP's std::pair + PairHash; only membership is used.
// - COLMAP's operator<< prints the graph with its own operator<<, which ToString mirrors.

using System.Runtime.InteropServices;

using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Sfm;

/// <summary>
/// Port of colmap::IncrementalTriangulator: triangulates points during the incremental
/// reconstruction. It holds the state and provides all functionality for triangulation.
/// </summary>
public sealed partial class IncrementalTriangulator
{
	/// <summary>Port of IncrementalTriangulator::Options.</summary>
	public sealed class Options
	{
		/// <summary>Maximum transitivity to search for correspondences.</summary>
		public int MaxTransitivity { get; set; } = 1;

		/// <summary>Maximum angular error to create new triangulations.</summary>
		public double CreateMaxAngleError { get; set; } = 2.0;

		/// <summary>Maximum angular error to continue existing triangulations.</summary>
		public double ContinueMaxAngleError { get; set; } = 2.0;

		/// <summary>Maximum reprojection error in pixels to merge triangulations.</summary>
		public double MergeMaxReprojError { get; set; } = 4.0;

		/// <summary>Maximum reprojection error to complete an existing triangulation.</summary>
		public double CompleteMaxReprojError { get; set; } = 4.0;

		/// <summary>Maximum transitivity for track completion.</summary>
		public int CompleteMaxTransitivity { get; set; } = 5;

		/// <summary>Maximum angular error to re-triangulate under-reconstructed image pairs.</summary>
		public double ReMaxAngleError { get; set; } = 5.0;

		/// <summary>
		/// Minimum ratio of common triangulations between an image pair over the number of
		/// correspondences between that image pair to be considered as under-reconstructed.
		/// </summary>
		public double ReMinRatio { get; set; } = 0.2;

		/// <summary>Maximum number of trials to re-triangulate an image pair.</summary>
		public int ReMaxTrials { get; set; } = 1;

		/// <summary>Minimum pairwise triangulation angle for a stable triangulation.</summary>
		public double MinAngle { get; set; } = 1.5;

		/// <summary>Whether to ignore two-view tracks.</summary>
		public bool IgnoreTwoViewTracks { get; set; } = true;

		/// <summary>
		/// Thresholds for bogus camera parameters. Images with bogus camera parameters are
		/// ignored in triangulation.
		/// </summary>
		public double MinFocalLengthRatio { get; set; } = 0.1;

		/// <summary>See <see cref="MinFocalLengthRatio"/>.</summary>
		public double MaxFocalLengthRatio { get; set; } = 10.0;

		/// <summary>See <see cref="MinFocalLengthRatio"/>.</summary>
		public double MaxExtraParam { get; set; } = 1.0;

		/// <summary>PRNG seed for all stochastic methods during triangulation.</summary>
		public int RandomSeed { get; set; } = -1;

		/// <summary>Port of Options::Check (CHECK_OPTION_*: false on a violation).</summary>
		public bool Check() =>
			MaxTransitivity >= 0
			&& CreateMaxAngleError > 0
			&& ContinueMaxAngleError > 0
			&& MergeMaxReprojError > 0
			&& CompleteMaxReprojError > 0
			&& CompleteMaxTransitivity >= 0
			&& ReMaxAngleError > 0
			&& ReMinRatio >= 0
			&& ReMinRatio <= 1
			&& ReMaxTrials >= 0
			&& MinAngle > 0
			&& RandomSeed >= -1;

		/// <summary>A copy (C++ copies the options struct by value).</summary>
		public Options Clone() => (Options)MemberwiseClone();
	}

	/// <summary>
	/// Port of IncrementalTriangulator::CorrData: one correspondence / track element with the
	/// lookups triangulation needs. The 2D point is read live from the image (see the file
	/// header), so it reflects triangulations made after this was captured.
	/// </summary>
	private readonly struct CorrData
	{
		public readonly uint ImageId;
		public readonly uint Point2DIdx;
		public readonly Image Image;
		public readonly Camera Camera;

		public CorrData(uint imageId, uint point2DIdx, Image image, Camera camera)
		{
			ImageId = imageId;
			Point2DIdx = point2DIdx;
			Image = image;
			Camera = camera;
		}

		public ref Point2D Point2D => ref Image.Point2DAt(Point2DIdx);
	}

	// Correspondence graph used to retrieve correspondence information for triangulation.
	private readonly CorrespondenceGraph _correspondenceGraph;

	// Reconstruction of the model. Modified when triangulating new points.
	private readonly Reconstruction _reconstruction;

	// Keeps track of the 3D point statistics.
	private readonly ObservationManager _obsManager;

	// Cache for cameras with bogus parameters.
	private readonly Dictionary<uint, bool> _cameraHasBogusParams = [];

	// Cache for tried track merges to avoid duplicate merge trials, keyed on the canonical
	// (min, max) pair of 3D point ids so each merge is recorded once in either direction.
	private readonly HashSet<(ulong, ulong)> _mergeTrials = [];

	// Cache for found correspondences in the graph.
	private readonly List<CorrespondenceGraph.Correspondence> _foundCorrs = [];

	// Reusable BFS scratch buffers for Complete().
	private List<TrackElement> _completeCurrQueue = [];
	private List<TrackElement> _completeNextQueue = [];

	// Dedupes (image_id, point2D_idx) pairs reached by Complete()'s BFS, packed into a ulong,
	// so the reprojection check is not redone for correspondences shared across parents.
	private readonly HashSet<ulong> _completeVisited = [];

	// Number of trials to retriangulate image pair.
	private readonly Dictionary<ulong, int> _reNumTrials = [];

	// Changed 3D points, i.e. if a 3D point is modified (created, continued, deleted,
	// merged, etc.). Cleared by ClearModifiedPoints3D.
	private readonly HashSet<ulong> _modifiedPoint3DIds = [];

	// Reused per-call buffers (COLMAP allocates these as locals; see the file header).
	private readonly List<CorrData> _corrsData = [];
	private readonly List<List<CorrData>> _createCorrsDataByDepth = [];
	private readonly List<CorrData> _retriCorrsData = [];
	private readonly List<ulong> _missingPoint3DIds = [];
	private readonly List<FeatureMatch> _matches = [];

	/// <summary>
	/// Creates a new incremental triangulator. Both the correspondence graph and the
	/// reconstruction must outlive the triangulator. Without an observation manager, one is
	/// created for the reconstruction and graph.
	/// </summary>
	public IncrementalTriangulator(
		CorrespondenceGraph correspondenceGraph,
		Reconstruction reconstruction,
		ObservationManager? obsManager = null)
	{
		_correspondenceGraph = correspondenceGraph;
		_reconstruction = reconstruction;
		_obsManager = obsManager ?? new ObservationManager(reconstruction, correspondenceGraph);
	}

	/// <summary>
	/// Triangulates observations of an image: creates new points, continues existing points,
	/// and merges separate points if the image bridges tracks. The image must be registered
	/// with its pose set. Returns the number of triangulated observations.
	/// </summary>
	public int TriangulateImage(Options options, uint imageId)
	{
		Check.That(options.Check());

		int numTris = 0;

		ClearCaches();

		Image image = _reconstruction.Image(imageId);
		if (!image.HasPose)
		{
			return numTris;
		}

		if (HasCameraBogusParams(options, image.CameraPtr))
		{
			return numTris;
		}

		// Every observation of the image becomes the reference correspondence once.
		List<CorrData> corrsData = _corrsData;

		// Try to triangulate all image observations.
		for (uint point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
		{
			int numTriangulated = Find(options, imageId, point2DIdx, options.MaxTransitivity, corrsData);
			if (corrsData.Count == 0)
			{
				continue;
			}

			var refCorrData = new CorrData(imageId, point2DIdx, image, image.CameraPtr);

			if (numTriangulated == 0)
			{
				corrsData.Add(refCorrData);
				numTris += Create(options, corrsData);
			}
			else
			{
				// Continue correspondences to existing 3D points.
				numTris += Continue(options, refCorrData, corrsData);

				// Create points from correspondences that are not continued.
				corrsData.Add(refCorrData);
				numTris += Create(options, corrsData);
			}
		}

		return numTris;
	}

	/// <summary>
	/// Completes triangulations for an image: tries to create new tracks for not yet
	/// triangulated observations and to complete existing tracks. Returns the number of
	/// completed observations.
	/// </summary>
	public int CompleteImage(Options options, uint imageId)
	{
		Check.That(options.Check());

		int numTris = 0;

		ClearCaches();

		Image image = _reconstruction.Image(imageId);
		if (!image.HasPose)
		{
			return numTris;
		}

		Camera camera = image.CameraPtr;
		if (HasCameraBogusParams(options, camera))
		{
			return numTris;
		}

		// Setup estimation options.
		var triOptions = new EstimateTriangulationOptions
		{
			MinTriAngle = Mathematics.MathUtils.DegToRad(options.MinAngle),
			ResidualType = TriangulationEstimator.ResidualType.ReprojectionError,
		};
		triOptions.RansacOptions.MaxError = options.CompleteMaxReprojError;
		triOptions.RansacOptions.RandomSeed = options.RandomSeed;

		List<CorrData> corrsData = _corrsData;

		for (uint point2DIdx = 0; point2DIdx < image.NumPoints2D; ++point2DIdx)
		{
			Point2D point2D = image.Point2DAt(point2DIdx);
			if (point2D.HasPoint3D)
			{
				// Complete existing track.
				numTris += Complete(options, point2D.Point3DId);
				continue;
			}

			if (options.IgnoreTwoViewTracks && _correspondenceGraph.IsTwoViewObservation(imageId, point2DIdx))
			{
				continue;
			}

			int numTriangulated = Find(options, imageId, point2DIdx, options.MaxTransitivity, corrsData);
			if (numTriangulated != 0 || corrsData.Count == 0)
			{
				continue;
			}

			corrsData.Add(new CorrData(imageId, point2DIdx, image, camera));

			// Estimate triangulation.
			if (!TriangulateTrack(triOptions, corrsData, out bool[] inlierMask, out LinearAlgebra.Vector3d xyz))
			{
				continue;
			}

			// Add inliers to estimated track.
			var track = new Track();
			track.Elements.Capacity = corrsData.Count;
			for (int i = 0; i < inlierMask.Length; ++i)
			{
				if (inlierMask[i])
				{
					CorrData corrData = corrsData[i];
					track.AddElement(corrData.ImageId, corrData.Point2DIdx);
					numTris += 1;
				}
			}

			ulong point3DId = _obsManager.AddPoint3D(xyz, track);
			_modifiedPoint3DIds.Add(point3DId);
		}

		return numTris;
	}

	/// <summary>
	/// Completes the tracks of specific 3D points: recursively adds observations that might
	/// have failed to triangulate before due to inaccurate poses, etc. Returns the number of
	/// completed observations.
	/// </summary>
	public int CompleteTracks(Options options, IEnumerable<ulong> point3DIds)
	{
		Check.That(options.Check());

		int numCompleted = 0;

		ClearCaches();

		foreach (ulong point3DId in point3DIds)
		{
			numCompleted += Complete(options, point3DId);
		}

		return numCompleted;
	}

	/// <summary>Completes the tracks of all 3D points. Returns the number of completed observations.</summary>
	public int CompleteAllTracks(Options options)
	{
		Check.That(options.Check());

		int numCompleted = 0;

		ClearCaches();

		// Point3DIds is a snapshot, as in C++ (it returns a set by value).
		foreach (ulong point3DId in _reconstruction.Point3DIds())
		{
			numCompleted += Complete(options, point3DId);
		}

		return numCompleted;
	}

	/// <summary>Merges the tracks of specific 3D points. Returns the number of merged observations.</summary>
	public int MergeTracks(Options options, IEnumerable<ulong> point3DIds)
	{
		Check.That(options.Check());

		int numMerged = 0;

		ClearCaches();

		foreach (ulong point3DId in point3DIds)
		{
			numMerged += Merge(options, point3DId);
		}

		return numMerged;
	}

	/// <summary>Merges the tracks of all 3D points. Returns the number of merged observations.</summary>
	public int MergeAllTracks(Options options)
	{
		Check.That(options.Check());

		int numMerged = 0;

		ClearCaches();

		foreach (ulong point3DId in _reconstruction.Point3DIds())
		{
			numMerged += Merge(options, point3DId);
		}

		return numMerged;
	}

	/// <summary>
	/// Re-triangulates under-reconstructed image pairs, which usually occur in a drifting
	/// reconstruction. A pair is under-reconstructed if its ratio of triangulated over total
	/// correspondences is below <see cref="Options.ReMinRatio"/>. Returns the number of
	/// triangulated observations.
	/// </summary>
	public int Retriangulate(Options options)
	{
		Check.That(options.Check());

		int numTris = 0;

		ClearCaches();

		Options reOptions = options.Clone();
		reOptions.ContinueMaxAngleError = options.ReMaxAngleError;

		List<FeatureMatch> matches = _matches;
		List<CorrData> retriCorrsData = _retriCorrsData;

		// Continue/Create update the pair statistics' values while this loop runs; that is
		// fine because only existing keys are written (no Dictionary version change), and each
		// pair's ratio is read when the loop reaches it, as in C++.
		foreach ((ulong pairId, ObservationManager.ImagePairStat stat) in _obsManager.ImagePairs)
		{
			// Only perform retriangulation for under-reconstructed image pairs.
			double triRatio = (double)stat.NumTriCorrs / (double)stat.NumTotalCorrs;
			if (triRatio >= options.ReMinRatio)
			{
				continue;
			}

			// Check if images are registered yet.

			(uint imageId1, uint imageId2) = PairIdToImagePair(pairId);

			Image image1 = _reconstruction.Image(imageId1);
			if (!image1.HasPose)
			{
				continue;
			}

			Image image2 = _reconstruction.Image(imageId2);
			if (!image2.HasPose)
			{
				continue;
			}

			// Only perform retriangulation for a maximum number of trials.

			ref int numReTrials = ref CollectionsMarshal.GetValueRefOrAddDefault(_reNumTrials, pairId, out _);
			if (numReTrials >= options.ReMaxTrials)
			{
				continue;
			}

			numReTrials += 1;

			Camera camera1 = image1.CameraPtr;
			Camera camera2 = image2.CameraPtr;
			if (HasCameraBogusParams(options, camera1) || HasCameraBogusParams(options, camera2))
			{
				continue;
			}

			// Find correspondences and perform retriangulation.

			_correspondenceGraph.ExtractMatchesBetweenImages(imageId1, imageId2, matches);

			foreach (FeatureMatch match in matches)
			{
				bool hasPoint3D1 = image1.Point2DAt(match.Point2DIdx1).HasPoint3D;
				bool hasPoint3D2 = image2.Point2DAt(match.Point2DIdx2).HasPoint3D;

				// Two cases are possible here: both points belong to the same 3D point or to
				// different 3D points. In the former case, there is nothing to do. In the
				// latter case, we do not attempt retriangulation, as retriangulated
				// correspondences are very likely bogus and would therefore destroy both 3D
				// points if merged.
				if (hasPoint3D1 && hasPoint3D2)
				{
					continue;
				}

				var corrData1 = new CorrData(imageId1, match.Point2DIdx1, image1, camera1);
				var corrData2 = new CorrData(imageId2, match.Point2DIdx2, image2, camera2);

				retriCorrsData.Clear();
				if (hasPoint3D1 && !hasPoint3D2)
				{
					retriCorrsData.Add(corrData1);
					numTris += Continue(reOptions, corrData2, retriCorrsData);
				}
				else if (!hasPoint3D1 && hasPoint3D2)
				{
					retriCorrsData.Add(corrData2);
					numTris += Continue(reOptions, corrData1, retriCorrsData);
				}
				else if (!hasPoint3D1 && !hasPoint3D2)
				{
					retriCorrsData.Add(corrData1);
					retriCorrsData.Add(corrData2);

					// Do not use larger triangulation threshold as this causes significant
					// drift when creating points (options vs. reOptions).
					numTris += Create(options, retriCorrsData);
				}

				// Else both points have a 3D point, but we do not want to merge points in
				// retriangulation.
			}
		}

		return numTris;
	}

	/// <summary>Indicates that a 3D point has been modified.</summary>
	public void AddModifiedPoint3D(ulong point3DId) => _modifiedPoint3DIds.Add(point3DId);

	/// <summary>
	/// The 3D points changed since the last <see cref="ClearModifiedPoints3D"/>, without
	/// those that no longer exist in the reconstruction.
	/// </summary>
	public IReadOnlySet<ulong> GetModifiedPoints3D()
	{
		// First remove any missing 3D points from the set.
		_missingPoint3DIds.Clear();
		foreach (ulong point3DId in _modifiedPoint3DIds)
		{
			if (!_reconstruction.ExistsPoint3D(point3DId))
			{
				_missingPoint3DIds.Add(point3DId);
			}
		}

		foreach (ulong point3DId in _missingPoint3DIds)
		{
			_modifiedPoint3DIds.Remove(point3DId);
		}

		return _modifiedPoint3DIds;
	}

	/// <summary>Clears the collection of changed 3D points.</summary>
	public void ClearModifiedPoints3D() => _modifiedPoint3DIds.Clear();

	/// <summary>COLMAP's operator&lt;&lt;: the reconstruction and the correspondence graph.</summary>
	public override string ToString() =>
		$"IncrementalTriangulator(reconstruction={_reconstruction}, correspondence_graph={_correspondenceGraph})";

	// Clears the cache of bogus camera parameters and merge trials.
	private void ClearCaches()
	{
		_cameraHasBogusParams.Clear();
		_mergeTrials.Clear();
		_foundCorrs.Clear();
	}
}
