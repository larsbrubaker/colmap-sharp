// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureMatching: port of colmap/controllers/feature_matching.h and .cc - the feature
// matchers (exhaustive, sequential, spatial, transitive, imported image pairs), the
// geometric verifier over existing matches, rig verification, and the importer of feature
// matches from a text file (FeaturePairsMatchingOptions of pairing.h). Each runs a
// PairGenerator (Controllers/PairGenerator.cs and neighbors) over a FeatureMatcherCache and
// hands the batches to FeatureMatchingUtils.cs's controllers. Tests:
// ColmapSharp.Tests/Controllers/FeatureMatchingTests.cs (feature_matching_test.cc).
//
// Tier C (outcome): the results go through RANSAC.
//
// Translation notes:
// - Each Create*FeatureMatcher Thread becomes a synchronous method taking the Database (COLMAP
//   opens it from a path), an IProgress (one step per pair batch, out of the generator's
//   NumBatches, or a Total of 0 where that is unknown; the "in %.3fs" LOG(INFO) lines) and a CancellationToken (Thread::Stop), which throws OperationCanceledException
//   between or inside batches; finished batches stay written, as in COLMAP.
// - RigVerification's ThreadPool computes the frame pairs in parallel and writes them in
//   std::map (frame pair) order, so the database does not depend on the thread count
//   (COLMAP's tasks write as they finish).
// - The vocabulary-tree matcher is excluded (colmap/retrieval is out of scope,
//   Controllers/PairingOptions.cs).

using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap::FeaturePairsMatchingOptions (pairing.h).</summary>
public sealed class FeaturePairsMatchingOptions
{
	/// <summary>Whether to geometrically verify the given matches.</summary>
	public bool VerifyMatches { get; set; } = true;

	/// <summary>Path to the file with the matches.</summary>
	public string MatchListPath { get; set; } = "";

	/// <summary>Port of FeaturePairsMatchingOptions::Check.</summary>
	public bool Check() => true;
}

/// <summary>Port of colmap::GeometricVerifierOptions: options for the geometric verifier.</summary>
public sealed class GeometricVerifierOptions
{
	/// <summary>Number of threads for geometric verification.</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>
	/// Whether to perform rig verification at the end. Unnecessary when we have existing
	/// relative poses for each matched pair.
	/// </summary>
	public bool RigVerification { get; set; }

	/// <summary>
	/// Whether to use the existing relative pose stored in TwoViewGeometry in the database.
	/// If no TwoViewGeometry is found we will fall back to geometric verification with RANSAC.
	/// </summary>
	public bool UseExistingRelativePose { get; set; }

	/// <summary>A copy (the C++ value copy).</summary>
	public GeometricVerifierOptions Clone() => (GeometricVerifierOptions)MemberwiseClone();
}

/// <summary>Port of colmap/controllers/feature_matching.h.</summary>
public static partial class FeatureMatching
{
	/// <summary>The <see cref="ControllerProgress.Stage"/> of the feature matchers.</summary>
	public const string MatchingStage = "Feature matching";

	/// <summary>The <see cref="ControllerProgress.Stage"/> of the geometric verifier.</summary>
	public const string VerificationStage = "Geometric verification";

	/// <summary>The <see cref="ControllerProgress.Stage"/> of rig verification.</summary>
	public const string RigVerificationStage = "Rig verification";

	/// <summary>
	/// Port of CreateExhaustiveFeatureMatcher: exhaustively match images by processing each
	/// block of the exhaustive match matrix in one batch.
	/// </summary>
	public static void MatchExhaustive(
		Database database,
		ExhaustivePairingOptions pairingOptions,
		FeatureMatchingOptions matchingOptions,
		TwoViewGeometryOptions geometryOptions,
		IProgress<ControllerProgress>? progress = null,
		CancellationToken cancellationToken = default)
	{
		var cache = new FeatureMatcherCache(pairingOptions.CacheSize(), database);
		RunMatcher(database, cache, () => new ExhaustivePairGenerator(pairingOptions, cache),
			matchingOptions, geometryOptions, progress, cancellationToken);
	}

	/// <summary>
	/// Port of CreateSequentialFeatureMatcher: match each image against its neighbors in
	/// image-name order.
	/// </summary>
	public static void MatchSequential(
		Database database,
		SequentialPairingOptions pairingOptions,
		FeatureMatchingOptions matchingOptions,
		TwoViewGeometryOptions geometryOptions,
		IProgress<ControllerProgress>? progress = null,
		CancellationToken cancellationToken = default)
	{
		var cache = new FeatureMatcherCache(pairingOptions.CacheSize(), database);
		RunMatcher(database, cache, () => new SequentialPairGenerator(pairingOptions, cache),
			matchingOptions, geometryOptions, progress, cancellationToken);
	}

	/// <summary>
	/// Port of CreateSpatialFeatureMatcher: match images against spatial nearest neighbors
	/// using prior location information.
	/// </summary>
	public static void MatchSpatial(
		Database database,
		SpatialPairingOptions pairingOptions,
		FeatureMatchingOptions matchingOptions,
		TwoViewGeometryOptions geometryOptions,
		IProgress<ControllerProgress>? progress = null,
		CancellationToken cancellationToken = default)
	{
		var cache = new FeatureMatcherCache(pairingOptions.CacheSize(), database);
		RunMatcher(database, cache, () => new SpatialPairGenerator(pairingOptions, cache),
			matchingOptions, geometryOptions, progress, cancellationToken);
	}

	/// <summary>
	/// Port of CreateTransitiveFeatureMatcher: if image pairs A-B and B-C match but A-C has
	/// not been matched, attempt to match A-C, for multiple iterations.
	/// </summary>
	public static void MatchTransitive(
		Database database,
		TransitivePairingOptions pairingOptions,
		FeatureMatchingOptions matchingOptions,
		TwoViewGeometryOptions geometryOptions,
		IProgress<ControllerProgress>? progress = null,
		CancellationToken cancellationToken = default)
	{
		var cache = new FeatureMatcherCache(pairingOptions.CacheSize(), database);
		RunMatcher(database, cache, () => new TransitivePairGenerator(pairingOptions, cache),
			matchingOptions, geometryOptions, progress, cancellationToken);
	}

	/// <summary>
	/// Port of CreateImagePairsFeatureMatcher: match the image pairs listed in
	/// <see cref="ImportedPairingOptions.MatchListPath"/> ("name1 name2" per line).
	/// </summary>
	public static void MatchImagePairs(
		Database database,
		ImportedPairingOptions pairingOptions,
		FeatureMatchingOptions matchingOptions,
		TwoViewGeometryOptions geometryOptions,
		IProgress<ControllerProgress>? progress = null,
		CancellationToken cancellationToken = default)
	{
		var cache = new FeatureMatcherCache(pairingOptions.CacheSize(), database);
		RunMatcher(database, cache, () => new ImportedPairGenerator(pairingOptions, cache),
			matchingOptions, geometryOptions, progress, cancellationToken);
	}

	/// <summary>
	/// Port of CreateGeometricVerifier: geometric verification of the existing matched
	/// image pairs.
	/// </summary>
	public static void VerifyGeometry(
		Database database,
		GeometricVerifierOptions verifierOptions,
		ExistingMatchedPairingOptions pairingOptions,
		TwoViewGeometryOptions geometryOptions,
		IProgress<ControllerProgress>? progress = null,
		CancellationToken cancellationToken = default)
	{
		var cache = new FeatureMatcherCache(pairingOptions.CacheSize(), database);
		var verifier = new GeometricVerifierController(verifierOptions, geometryOptions, cache);
		Check.That(geometryOptions.Check());

		if (!verifier.Setup())
		{
			return;
		}

		var pairGenerator = new ExistingMatchedPairGenerator(pairingOptions, cache);
		int totalBatches = pairGenerator.NumBatches;
		int numBatches = 0;
		while (!pairGenerator.HasFinished())
		{
			cancellationToken.ThrowIfCancellationRequested();
			verifier.Verify(pairGenerator.Next(), cancellationToken);
			numBatches += 1;
			progress?.Report(new ControllerProgress(VerificationStage, numBatches, totalBatches, ""));
		}

		if (verifier.Options.RigVerification)
		{
			RigVerification(database, cache, geometryOptions, verifier.Options.NumThreads, cancellationToken);
		}
	}

	// Port of FeatureMatcherThread::Run.
	private static void RunMatcher(
		Database database,
		FeatureMatcherCache cache,
		Func<PairGenerator> pairGeneratorFactory,
		FeatureMatchingOptions matchingOptions,
		TwoViewGeometryOptions geometryOptions,
		IProgress<ControllerProgress>? progress,
		CancellationToken cancellationToken)
	{
		var matcher = new FeatureMatcherController(matchingOptions, geometryOptions, cache);
		Check.That(matchingOptions.Check());
		Check.That(geometryOptions.Check());

		if (!matcher.Setup())
		{
			return;
		}

		PairGenerator pairGenerator = Check.NotNull(pairGeneratorFactory());
		int totalBatches = pairGenerator.NumBatches;
		int numBatches = 0;
		while (!pairGenerator.HasFinished())
		{
			cancellationToken.ThrowIfCancellationRequested();
			matcher.Match(pairGenerator.Next(), cancellationToken);
			numBatches += 1;
			progress?.Report(new ControllerProgress(MatchingStage, numBatches, totalBatches, ""));
		}

		// Notice that we run rig verification after feature matching, because feature
		// matching operates on pairs of images instead of pairs of frames. Rig verification
		// operates on pairs of frames and we require all image pairs between two frames to be
		// matched before running rig verification.
		if (!matchingOptions.SkipGeometricVerification && matchingOptions.RigVerification)
		{
			RigVerification(database, cache, geometryOptions, matchingOptions.NumThreads, cancellationToken);
		}
	}

	// Port of the anonymous RigVerification.
	private static void RigVerification(
		Database database,
		FeatureMatcherCache cache,
		TwoViewGeometryOptions geometryOptions,
		int numThreads,
		CancellationToken cancellationToken)
	{
		var rigs = new Dictionary<uint, Rig>();
		foreach (Rig rig in database.ReadAllRigs())
		{
			rigs[rig.RigId] = rig;
		}

		var imageToFrameIds = new Dictionary<uint, uint>();
		foreach (Frame frame in database.ReadAllFrames())
		{
			foreach (DataId dataId in frame.ImageIds())
			{
				imageToFrameIds[(uint)dataId.Id] = frame.FrameId;
			}
		}

		// std::map: frame pairs in ascending order.
		var framePairStats = new SortedDictionary<(uint, uint), (int NumImagePairs, int NumMatches)>();
		foreach ((ulong imagePairId, int pairNumMatches) in database.ReadNumMatches())
		{
			if (pairNumMatches == 0)
			{
				continue;
			}

			(uint imageId1, uint imageId2) = PairIdToImagePair(imagePairId);
			uint frameId1 = imageToFrameIds[imageId1];
			uint frameId2 = imageToFrameIds[imageId2];
			if (frameId1 > frameId2)
			{
				(frameId1, frameId2) = (frameId2, frameId1);
			}

			framePairStats.TryGetValue((frameId1, frameId2), out var stats);
			framePairStats[(frameId1, frameId2)] = (stats.NumImagePairs + 1, stats.NumMatches + pairNumMatches);
		}

		var framePairs = new List<(uint, uint)>();
		foreach (((uint, uint) framePair, var stats) in framePairStats)
		{
			// If the frame pair has only matches between one pair of images, then there is no
			// need to run rig verification, as there are no rig constraints.
			if (stats.NumImagePairs <= 1 || stats.NumMatches < geometryOptions.MinNumInliers)
			{
				continue;
			}

			framePairs.Add(framePair);
		}

		var results = new List<((uint ImageId1, uint ImageId2) ImagePair, TwoViewGeometry Geometry)>[framePairs.Count];
		var parallelOptions = new ParallelOptions
		{
			MaxDegreeOfParallelism = Threading.GetEffectiveNumThreads(numThreads),
			CancellationToken = cancellationToken,
		};
		Parallel.For(0, framePairs.Count, parallelOptions, i =>
		{
			(uint frameId1, uint frameId2) = framePairs[i];
			results[i] = VerifyFramePair(cache, rigs, geometryOptions, frameId1, frameId2);
		});

		foreach (var frameResults in results)
		{
			foreach (((uint imageId1, uint imageId2), TwoViewGeometry twoViewGeometry) in frameResults)
			{
				cache.DeleteTwoViewGeometry(imageId1, imageId2);
				cache.WriteTwoViewGeometry(imageId1, imageId2, twoViewGeometry);
			}
		}
	}

	// The ThreadPool task of RigVerification for one frame pair.
	private static List<((uint ImageId1, uint ImageId2) ImagePair, TwoViewGeometry Geometry)> VerifyFramePair(
		FeatureMatcherCache cache,
		Dictionary<uint, Rig> rigs,
		TwoViewGeometryOptions geometryOptions,
		uint frameId1,
		uint frameId2)
	{
		Frame frame1 = cache.GetFrame(frameId1);
		Frame frame2 = cache.GetFrame(frameId2);
		Rig rig1 = rigs[frame1.RigId];
		Rig rig2 = rigs[frame2.RigId];

		var images = new Dictionary<uint, Image>();
		var cameras = new Dictionary<uint, Camera>();
		void AddImagesAndCameras(Frame frame)
		{
			foreach (DataId dataId in frame.ImageIds())
			{
				// data_t's 64-bit id holds an image_t here.
				uint imageId = (uint)dataId.Id;
				Image image = cache.GetImage(imageId).Clone();
				image.SetPoints2D(FeatureUtils.FeatureKeypointsToPointsVector(cache.GetKeypoints(imageId)));
				images[imageId] = image;
				cameras[image.CameraId] = cache.GetCamera(image.CameraId);
			}
		}

		AddImagesAndCameras(frame1);
		AddImagesAndCameras(frame2);

		var matches = new List<((uint ImageId1, uint ImageId2) ImagePair, List<FeatureMatch> Matches)>();
		foreach (DataId dataId1 in frame1.ImageIds())
		{
			uint imageId1 = (uint)dataId1.Id;
			foreach (DataId dataId2 in frame2.ImageIds())
			{
				uint imageId2 = (uint)dataId2.Id;
				// If verifying within the same frame, then skip redundant image pairs,
				// whereas different frames are guaranteed to have different image pairs.
				// Note that verifying within the same frame can be useful when the images
				// have some overlap but the matches between image pairs are not enough alone
				// but accumulating them over the whole frame can lead to a successful
				// verification.
				if ((frameId1 == frameId2 && imageId1 <= imageId2) || !cache.ExistsMatches(imageId1, imageId2))
				{
					continue;
				}

				matches.Add(((imageId1, imageId2), cache.GetMatches(imageId1, imageId2)));
			}
		}

		return FeatureMatcherController.WithFreshPrng(geometryOptions, () =>
			TwoViewGeometryEstimation.EstimateRigTwoViewGeometries(rig1, rig2, images, cameras, matches, geometryOptions));
	}
}
