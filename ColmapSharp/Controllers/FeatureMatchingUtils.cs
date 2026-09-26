// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureMatchingUtils: port of colmap/controllers/feature_matching_utils.h and .cc -
// FeatureMatcherController (match a batch of image pairs, verify them geometrically,
// optionally re-match guided by the verified geometry, and write matches and two-view
// geometries through the FeatureMatcherCache) and GeometricVerifierController (verify the
// existing matches of a batch). The per-pair work is FeatureMatcherWorker and VerifierWorker.
// Controllers/FeatureMatching.cs drives both over a PairGenerator. Tests:
// ColmapSharp.Tests/Controllers/FeatureMatchingUtilsTests.cs (feature_matching_utils_test.cc).
//
// Tier C (outcome) where RANSAC runs (the verification), Tier A for the bookkeeping.
//
// Translation notes:
// - COLMAP's worker threads between JobQueues (matcher -> verifier -> guided matcher ->
//   output) become one Parallel.For over the batch's pairs, each pair running the same stages
//   in the same order, with one FeatureMatcher per worker (NumThreads = 1, "prevent nested
//   threading"). COLMAP writes the outputs in completion order; here they are written in the
//   order of the input pairs, so the database does not depend on the thread count.
// - Each verification runs on a fresh PRNG when RANSAC is unseeded (random_seed -1), so
//   every pair sees the stream a fresh COLMAP verifier thread would; COLMAP's verifier
//   threads continue their streams across pairs (docs/CPP_DIVERGENCES.md entry 71). The
//   worker's own PRNG is restored afterwards: Parallel.For also runs iterations on the
//   calling thread, whose generator COLMAP's separate worker threads never touch.
// - COLMAP deletes a pair's stale matches / two-view geometry before queueing it; here the
//   deletes wait until the batch's results are written, so a cancelled batch leaves the
//   database as it was.
// - Pairs that COLMAP counts as outputs but never queues (a verifier batch pair without
//   matches, or an existing-matches pair when no verifiers exist) hang COLMAP's Pop; here
//   they are skipped or passed straight to the output (entry 70).
// - The GPU matchers (SiftGPU, CUDA) and the OpenGL context are excluded.
// - FeatureMatcherImage has no pose prior (the port's matchers do not use it).

using ColmapSharp.Estimators;
using ColmapSharp.Feature;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap::FeatureMatcherData: one image pair's job.</summary>
public sealed class FeatureMatcherData
{
	/// <summary>The first image.</summary>
	public uint ImageId1 { get; set; } = InvalidImageId;

	/// <summary>The second image.</summary>
	public uint ImageId2 { get; set; } = InvalidImageId;

	/// <summary>The raw matches.</summary>
	public List<FeatureMatch> Matches { get; set; } = [];

	/// <summary>The verified two-view geometry.</summary>
	public TwoViewGeometry TwoViewGeometry { get; set; } = new();
}

/// <summary>
/// Port of colmap::FeatureMatcherController: multi-threaded SIFT feature matcher, which
/// writes the computed results to the database and skips already matched image pairs.
/// </summary>
public sealed class FeatureMatcherController
{
	private readonly FeatureMatchingOptions matchingOptions;
	private readonly TwoViewGeometryOptions geometryOptions;
	private readonly FeatureMatcherCache cache;
	private readonly int numThreads;
	private readonly bool verify;
	private FeatureMatchingOptions? workerOptions;
	private FeatureMatchingOptions? guidedWorkerOptions;
	private bool isSetup;

	/// <summary>Port of the FeatureMatcherController constructor.</summary>
	public FeatureMatcherController(
		FeatureMatchingOptions matchingOptions, TwoViewGeometryOptions geometryOptions, FeatureMatcherCache cache)
	{
		this.matchingOptions = matchingOptions.Clone();
		this.geometryOptions = geometryOptions.Clone();
		this.cache = cache;
		Check.That(this.matchingOptions.Check());
		Check.That(this.geometryOptions.Check());
		Check.Eq(this.geometryOptions.RansacOptions.NumThreads, 1,
			"Parallel RANSAC is not supported inside multi-threaded matching");

		numThreads = Threading.GetEffectiveNumThreads(this.matchingOptions.NumThreads);
		Check.Gt(numThreads, 0);

		// If skip_geometric_verification, match directly to the output.
		verify = !(this.matchingOptions.SkipGeometricVerification && !this.matchingOptions.GuidedMatching);
	}

	/// <summary>
	/// Port of FeatureMatcherController::Setup: prepares the per-worker matcher options (the
	/// worker threads' start-up in COLMAP). Throws if a matcher cannot be created.
	/// </summary>
	public bool Setup()
	{
		workerOptions = WorkerOptions(guided: false);
		if (matchingOptions.GuidedMatching)
		{
			guidedWorkerOptions = WorkerOptions(guided: true);
		}

		// Fail here, like COLMAP's CheckValidSetup, when the matcher type is unavailable.
		FeatureMatcher.Create(workerOptions);
		isSetup = true;
		return true;
	}

	/// <summary>Port of FeatureMatcherController::Match: matches one batch of image pairs.</summary>
	public void Match(IReadOnlyList<(uint ImageId1, uint ImageId2)> imagePairs, CancellationToken cancellationToken = default)
	{
		Check.That(isSetup);
		if (imagePairs.Count == 0)
		{
			return;
		}

		// Match the image pairs.
		var imagePairIds = new HashSet<ulong>();
		var jobs = new List<(FeatureMatcherData Data, bool ExistsMatches, bool ExistsTwoViewGeometry)>();
		foreach ((uint imageId1, uint imageId2) in imagePairs)
		{
			// Avoid self-matches.
			if (imageId1 == imageId2)
			{
				continue;
			}

			// Avoid duplicate image pairs.
			if (!imagePairIds.Add(ImagePairToPairId(imageId1, imageId2)))
			{
				continue;
			}

			// Avoid self-matches within a frame.
			if (matchingOptions.SkipImagePairsInSameFrame)
			{
				Image image1 = cache.GetImage(imageId1);
				Image image2 = cache.GetImage(imageId2);
				if (image1.HasFrameId && image2.HasFrameId && image1.FrameId == image2.FrameId)
				{
					continue;
				}
			}

			bool existsMatches = cache.ExistsMatches(imageId1, imageId2);
			bool existsTwoViewGeometry = cache.ExistsTwoViewGeometry(imageId1, imageId2);
			if (existsMatches && existsTwoViewGeometry)
			{
				continue;
			}

			// If only one of the matches or inlier matches exist, we recompute them from
			// scratch and delete the existing results (when the results are written, so that
			// database constraints do not fail on an existing result).
			var data = new FeatureMatcherData { ImageId1 = imageId1, ImageId2 = imageId2 };
			if (existsMatches)
			{
				data.Matches = cache.GetMatches(imageId1, imageId2);
			}

			jobs.Add((data, existsMatches, existsTwoViewGeometry));
		}

		var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = numThreads, CancellationToken = cancellationToken };
		Parallel.For(
			0,
			jobs.Count,
			parallelOptions,
			() => new WorkerMatchers(),
			(i, _, matchers) =>
			{
				(FeatureMatcherData data, bool existsMatches) = (jobs[i].Data, jobs[i].ExistsMatches);
				if (!existsMatches)
				{
					matchers.Matcher ??= FeatureMatcher.Create(workerOptions!);
					MatchPair(matchers.Matcher, cache, geometryOptions, guided: false, data);
				}

				if (verify)
				{
					VerifyPair(geometryOptions, cache, useExistingRelativePose: false, data);
				}

				if (matchingOptions.GuidedMatching)
				{
					matchers.GuidedMatcher ??= FeatureMatcher.Create(guidedWorkerOptions!);
					MatchPair(matchers.GuidedMatcher, cache, geometryOptions, guided: true, data);
				}

				return matchers;
			},
			_ => { });

		// Write results to database.
		foreach ((FeatureMatcherData output, bool existsMatches, bool existsTwoViewGeometry) in jobs)
		{
			if (existsTwoViewGeometry)
			{
				cache.DeleteTwoViewGeometry(output.ImageId1, output.ImageId2);
			}

			if (existsMatches)
			{
				cache.DeleteMatches(output.ImageId1, output.ImageId2);
			}

			if (output.Matches.Count < geometryOptions.MinNumInliers)
			{
				output.Matches = [];
			}

			if (output.TwoViewGeometry.InlierMatches.Count < geometryOptions.MinNumInliers)
			{
				output.TwoViewGeometry = new TwoViewGeometry();
			}

			cache.WriteMatches(output.ImageId1, output.ImageId2, output.Matches);
			cache.WriteTwoViewGeometry(output.ImageId1, output.ImageId2, output.TwoViewGeometry);
		}
	}

	private FeatureMatchingOptions WorkerOptions(bool guided)
	{
		FeatureMatchingOptions options = matchingOptions.Clone();
		// Prevent nested threading.
		options.NumThreads = 1;
		// The first matching is always without guided matching.
		options.GuidedMatching = guided;
		if (options.Type == FeatureMatcherType.SiftBruteForce)
		{
			// COLMAP's "bit ugly" injection of the shared descriptor index cache.
			options.Sift.CpuDescriptorIndexCache = cache.GetFeatureDescriptorIndexCache();
		}

		// Minimize the amount of allocated memory by computing the maximum number of
		// descriptors for any image over the whole database.
		options.MaxNumMatches = (int)Math.Min(options.MaxNumMatches, cache.MaxNumKeypoints());
		return options;
	}

	// The matchers of one Parallel.For worker (COLMAP's matcher and guided matcher threads).
	private sealed class WorkerMatchers
	{
		public FeatureMatcher? Matcher;
		public FeatureMatcher? GuidedMatcher;
	}

	// Port of the loop body of FeatureMatcherWorker::Run.
	internal static void MatchPair(
		FeatureMatcher matcher, FeatureMatcherCache cache, TwoViewGeometryOptions geometryOptions, bool guided, FeatureMatcherData data)
	{
		if (!cache.ExistsDescriptors(data.ImageId1) || !cache.ExistsDescriptors(data.ImageId2))
		{
			return;
		}

		FeatureMatcherImage image1 = MatcherImage(cache, data.ImageId1);
		FeatureMatcherImage image2 = MatcherImage(cache, data.ImageId2);
		if (guided)
		{
			matcher.MatchGuided(geometryOptions.RansacOptions.MaxError, image1, image2, data.TwoViewGeometry);
		}
		else
		{
			var matches = new List<FeatureMatch>();
			matcher.Match(image1, image2, matches);
			data.Matches = matches;
		}
	}

	private static FeatureMatcherImage MatcherImage(FeatureMatcherCache cache, uint imageId) => new()
	{
		ImageId = imageId,
		Camera = cache.GetCamera(cache.GetImage(imageId).CameraId),
		Keypoints = cache.GetKeypoints(imageId),
		Descriptors = cache.GetDescriptors(imageId),
	};

	// Port of the loop body of VerifierWorker::Run.
	internal static void VerifyPair(
		TwoViewGeometryOptions options, FeatureMatcherCache cache, bool useExistingRelativePose, FeatureMatcherData data)
	{
		if (data.Matches.Count < options.MinNumInliers)
		{
			return;
		}

		Camera camera1 = cache.GetCamera(cache.GetImage(data.ImageId1).CameraId);
		Camera camera2 = cache.GetCamera(cache.GetImage(data.ImageId2).CameraId);
		Vector2d[] points1 = FeatureUtils.FeatureKeypointsToPointsVector(cache.GetKeypoints(data.ImageId1));
		Vector2d[] points2 = FeatureUtils.FeatureKeypointsToPointsVector(cache.GetKeypoints(data.ImageId2));

		if (useExistingRelativePose && data.TwoViewGeometry.Cam2FromCam1.HasValue)
		{
			// No RANSAC: the inliers are scored against the known pose.
			data.TwoViewGeometry = TwoViewGeometryEstimation.TwoViewGeometryFromKnownRelativePose(
				camera1,
				points1,
				camera2,
				points2,
				data.TwoViewGeometry.Cam2FromCam1.Value,
				data.Matches,
				options.MinNumInliers,
				options.RansacOptions.MaxError);
		}
		else
		{
			data.TwoViewGeometry = WithFreshPrng(options, () => TwoViewGeometryEstimation.EstimateTwoViewGeometry(
				camera1, points1, camera2, points2, data.Matches, options));
		}
	}

	/// <summary>
	/// Runs <paramref name="estimate"/> as on a fresh COLMAP worker thread. An unseeded RANSAC
	/// (random_seed -1) draws from the thread's PRNG, so that PRNG starts from the default seed,
	/// and the result does not depend on which worker ran the pair (entry 71). The thread's own
	/// PRNG is restored afterwards (a seeded RANSAC reseeds it too), so neither the caller of a
	/// Parallel.For nor the thread pool sees the estimation's draws.
	/// </summary>
	internal static T WithFreshPrng<T>(TwoViewGeometryOptions options, Func<T> estimate)
	{
		Mt19937? callerPrng = RandomUtils.Prng;
		try
		{
			if (options.RansacOptions.RandomSeed == -1)
			{
				RandomUtils.SetPRNGSeed();
			}

			return estimate();
		}
		finally
		{
			RandomUtils.Prng = callerPrng;
		}
	}
}

/// <summary>
/// Port of colmap::GeometricVerifierController: verifies the existing matches of batches of
/// image pairs and writes their two-view geometries.
/// </summary>
public sealed class GeometricVerifierController
{
	private readonly TwoViewGeometryOptions geometryOptions;
	private readonly FeatureMatcherCache cache;
	// COLMAP builds its verifier threads in the constructor, so the thread count and
	// use_existing_relative_pose are fixed there; later changes through Options only reach
	// what reads Options afterwards (rig verification's thread count).
	private readonly int numThreads;
	private readonly bool useExistingRelativePose;
	private bool isSetup;

	/// <summary>Port of the GeometricVerifierController constructor.</summary>
	public GeometricVerifierController(
		GeometricVerifierOptions options, TwoViewGeometryOptions geometryOptions, FeatureMatcherCache cache)
	{
		this.geometryOptions = geometryOptions.Clone();
		this.cache = cache;
		Options = options.Clone();
		Check.That(this.geometryOptions.Check());

		numThreads = Threading.GetEffectiveNumThreads(Options.NumThreads);
		useExistingRelativePose = Options.UseExistingRelativePose;
	}

	/// <summary>Port of GeometricVerifierController::Options (mutable, like the C++ reference).</summary>
	public GeometricVerifierOptions Options { get; }

	/// <summary>Port of GeometricVerifierController::Setup.</summary>
	public bool Setup()
	{
		isSetup = true;
		return true;
	}

	/// <summary>Port of GeometricVerifierController::Verify: verifies one batch of image pairs.</summary>
	public void Verify(IReadOnlyList<(uint ImageId1, uint ImageId2)> imagePairs, CancellationToken cancellationToken = default)
	{
		Check.That(isSetup);
		if (imagePairs.Count == 0)
		{
			return;
		}

		// Verify the matches from the image pairs.
		var imagePairIds = new HashSet<ulong>();
		// Pairs to verify, and pairs whose stale two-view geometry is only deleted (no matches).
		var jobs = new List<FeatureMatcherData>();
		var deleteOnly = new List<(uint, uint)>();
		foreach ((uint imageId1, uint imageId2) in imagePairs)
		{
			// Avoid self-matches.
			if (imageId1 == imageId2)
			{
				continue;
			}

			// Avoid duplicate image pairs.
			if (!imagePairIds.Add(ImagePairToPairId(imageId1, imageId2)))
			{
				continue;
			}

			bool existsMatches = cache.ExistsMatches(imageId1, imageId2);
			bool existsInlierMatches = cache.ExistsInlierMatches(imageId1, imageId2);
			if (existsMatches && existsInlierMatches)
			{
				continue;
			}

			// If only one of the matches or inlier matches exist, we recompute them from
			// scratch and delete the existing results (when the results are written).
			// COLMAP counts a pair without matches as an output but never queues it (entry 70).
			if (!existsMatches)
			{
				if (existsInlierMatches)
				{
					deleteOnly.Add((imageId1, imageId2));
				}

				continue;
			}

			var data = new FeatureMatcherData
			{
				ImageId1 = imageId1,
				ImageId2 = imageId2,
				Matches = cache.GetMatches(imageId1, imageId2),
			};
			// There exists a two view geometry without inlier matches (a geometry with inlier
			// matches is the one COLMAP deletes first, so it is not read).
			if (!existsInlierMatches && cache.ExistsTwoViewGeometry(imageId1, imageId2))
			{
				data.TwoViewGeometry = cache.GetTwoViewGeometry(imageId1, imageId2);
			}

			jobs.Add(data);
		}

		var parallelOptions = new ParallelOptions
		{
			MaxDegreeOfParallelism = numThreads,
			CancellationToken = cancellationToken,
		};
		Parallel.For(0, jobs.Count, parallelOptions, i =>
			FeatureMatcherController.VerifyPair(geometryOptions, cache, useExistingRelativePose, jobs[i]));

		// Write results to database.
		foreach ((uint imageId1, uint imageId2) in deleteOnly)
		{
			cache.DeleteTwoViewGeometry(imageId1, imageId2);
		}

		foreach (FeatureMatcherData output in jobs)
		{
			if (output.Matches.Count < geometryOptions.MinNumInliers)
			{
				output.Matches = [];
			}

			if (output.TwoViewGeometry.InlierMatches.Count < geometryOptions.MinNumInliers)
			{
				output.TwoViewGeometry = new TwoViewGeometry();
			}

			if (cache.ExistsTwoViewGeometry(output.ImageId1, output.ImageId2))
			{
				cache.DeleteTwoViewGeometry(output.ImageId1, output.ImageId2);
			}

			cache.WriteTwoViewGeometry(output.ImageId1, output.ImageId2, output.TwoViewGeometry);
		}
	}
}
