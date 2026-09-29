// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Ransac: colmap/optim/ransac.h - RANSACOptions, the RANSAC report and the RANSAC loop
// itself (sample, estimate, score, keep the best, stop once the dynamic trial count says the
// best model is found with the requested confidence). Generic over the estimator
// (Estimator.cs), the support measurer (SupportMeasurement.cs) and the sampler (Sampler.cs
// and its three implementations). LoRansac.cs derives from it and adds local optimization.
// Tests: ColmapSharp.Tests/Optim/RansacTests.cs (ransac_test.cc 1:1).
//
// Tier C (outcome) for Estimate, since it runs through the estimator's decompositions;
// ComputeNumTrials and the sampling sequence are Tier A (the same seed draws the same
// samples as COLMAP).
//
// Translation notes:
// - The class template's three parameters become seven type parameters, because C# has no
//   associated types: the estimator's X_t/Y_t/M_t (TX, TY, TModel) and the measurer's
//   Support (TSupport) are spelled out. Ransac<TEstimator, TX, TY, TModel> supplies COLMAP's
//   defaults (InlierSupportMeasurer, RandomSampler).
// - Samplers and support measurers are constrained to classes: all of COLMAP's are, the
//   samplers carry per-instance state that must not be copied, and it lets the constructor
//   default the sampler to TSampler.Create(TEstimator.MinNumSamples) as COLMAP defaults it
//   to Sampler(Estimator::kMinNumSamples).
// - size_t trial counts are ulong; RANSACOptions keeps COLMAP's int fields and the same
//   int <-> size_t conversions (see the constructor).
// - Threads: COLMAP runs the trial loop in an OpenMP parallel region when num_threads > 1,
//   each thread with its own sampler seeded random_seed + thread index, sharing an atomic
//   trial counter; which thread wins a trial is a scheduling race, so its parallel result
//   is not reproducible. This port always runs the loop once, on the calling thread, which
//   is exactly COLMAP built without OpenMP ("the block runs once serially"). num_threads is
//   still validated as COLMAP does (including the RandomSampler-only check), so options
//   that COLMAP rejects are rejected here. divergence 17.
// - The per-thread PRNG seeding (SetPRNGSeed(random_seed) when random_seed != -1 and the
//   sampler is randomized) reseeds the calling thread's PRNG, as COLMAP's serial path does.

using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Optim;

/// <summary>Port of colmap::RANSACOptions.</summary>
public struct RansacOptions
{
	/// <summary>
	/// Maximum error for a sample to be considered as an inlier. Note that the residual of an
	/// estimator corresponds to a squared error.
	/// </summary>
	public double MaxError = 0.0;

	/// <summary>
	/// A priori assumed minimum inlier ratio, which determines the maximum number of
	/// iterations. Only applies if smaller than <see cref="MaxNumTrials"/>.
	/// </summary>
	public double MinInlierRatio = 0.1;

	/// <summary>
	/// Abort the iteration if minimum probability that one sample is free from outliers is
	/// reached.
	/// </summary>
	public double Confidence = 0.99;

	/// <summary>
	/// The num_trials_multiplier to the dynamically computed maximum number of iterations
	/// based on the specified confidence value.
	/// </summary>
	public double DynNumTrialsMultiplier = 3.0;

	/// <summary>Minimum number of random trials to estimate model from random subset.</summary>
	public int MinNumTrials = 0;

	/// <summary>Maximum number of random trials to estimate model from random subset.</summary>
	public int MaxNumTrials = int.MaxValue;

	/// <summary>
	/// PRNG seed for randomized samplers. Set to -1 for nondeterministic behavior, or a fixed
	/// value to make results reproducible.
	/// </summary>
	public int RandomSeed = -1;

	/// <summary>
	/// Number of threads for parallel RANSAC. 1 = serial (default). -1 uses all available
	/// hardware threads. This port always runs serially (see the file header).
	/// </summary>
	public int NumThreads = 1;

	/// <summary>COLMAP's default options.</summary>
	public RansacOptions()
	{
	}

	/// <summary>Port of RANSACOptions::Check: throws if an option is out of range.</summary>
	public readonly void Check()
	{
		Util.Check.Gt(MaxError, 0);
		Util.Check.Ge(MinInlierRatio, 0);
		Util.Check.Le(MinInlierRatio, 1);
		Util.Check.Ge(Confidence, 0);
		Util.Check.Le(Confidence, 1);
		Util.Check.Le(MinNumTrials, MaxNumTrials);
		Util.Check.Ge(RandomSeed, -1);
		Util.Check.Ge(NumThreads, -1);
		Util.Check.Ne(NumThreads, 0);
	}
}

/// <summary>Port of RANSAC::Report: the result of a RANSAC (or LO-RANSAC) estimation.</summary>
public sealed class RansacReport<TModel, TSupport>
	where TSupport : struct, IRansacSupport
{
	/// <summary>Whether the estimation was successful.</summary>
	public bool Success { get; set; }

	/// <summary>The number of RANSAC trials / iterations.</summary>
	public ulong NumTrials { get; set; }

	/// <summary>The support of the estimated model.</summary>
	public TSupport Support { get; set; } = new TSupport();

	/// <summary>Boolean mask which is true if a sample is an inlier.</summary>
	public bool[] InlierMask { get; set; } = [];

	/// <summary>The estimated model.</summary>
	public TModel Model { get; set; } = default!;
}

/// <summary>
/// RANSAC (RANdom SAmple Consensus). Port of colmap::RANSAC&lt;Estimator, SupportMeasurer,
/// Sampler&gt;.
/// </summary>
public class Ransac<TEstimator, TX, TY, TModel, TSupportMeasurer, TSupport, TSampler>
	where TEstimator : IEstimator<TX, TY, TModel>
	where TSupportMeasurer : class, ISupportMeasurer<TSupport>
	where TSupport : struct, IRansacSupport
	where TSampler : class, ISampler<TSampler>
{
	/// <summary>
	/// Create a RANSAC estimator. <paramref name="sampler"/> defaults to a sampler of
	/// TEstimator.MinNumSamples elements, as COLMAP's does.
	/// </summary>
	public Ransac(
		RansacOptions options,
		TEstimator estimator,
		TSupportMeasurer supportMeasurer,
		TSampler? sampler = null)
	{
		Estimator = estimator;
		SupportMeasurer = supportMeasurer;
		Sampler = sampler ?? TSampler.Create(TEstimator.MinNumSamples);
		Options = options;
		options.Check();

		// Determine max_num_trials based on assumed `min_inlier_ratio`.
		const ulong NumSamples = 100000;
		ulong dynMaxNumTrials = ComputeNumTrials(
			(ulong)(Options.MinInlierRatio * NumSamples),
			NumSamples,
			Options.Confidence,
			Options.DynNumTrialsMultiplier);
		// std::min<size_t>(int, size_t) assigned back to the int field: the int converts to
		// size_t (a negative value becomes huge) and the minimum truncates back to int.
		Options.MaxNumTrials = unchecked((int)Math.Min(unchecked((ulong)(long)Options.MaxNumTrials), dynMaxNumTrials));
	}

	/// <summary>The minimal-sample estimator.</summary>
	public TEstimator Estimator { get; set; }

	/// <summary>The support measurer.</summary>
	public TSupportMeasurer SupportMeasurer { get; set; }

	/// <summary>
	/// The sampler. RANSAC initializes it to read MaxNumSamples; the trials themselves draw
	/// from a per-run sampler of the same type, as COLMAP's per-thread sampler.
	/// </summary>
	public TSampler Sampler { get; set; }

	/// <summary>The options, with MaxNumTrials already capped by the constructor.</summary>
	protected RansacOptions Options;

	/// <summary>
	/// Determine the maximum number of trials required to sample at least one outlier-free
	/// random set of samples with the specified confidence, given the inlier ratio.
	/// </summary>
	/// <param name="numInliers">The number of inliers.</param>
	/// <param name="numSamples">The total number of samples.</param>
	/// <param name="confidence">Confidence that one sample is outlier-free.</param>
	/// <param name="numTrialsMultiplier">Multiplication factor to number of trials.</param>
	/// <returns>The required number of iterations.</returns>
	public static ulong ComputeNumTrials(ulong numInliers, ulong numSamples, double confidence, double numTrialsMultiplier)
	{
		double probFailure = 1 - confidence;
		if (probFailure <= 0)
		{
			return ulong.MaxValue;
		}

		// Not using pow(inlier_ratio, Estimator::kMinNumSamples).
		// See "Fixing the RANSAC stopping criterion"
		// by Schönberger, Larsson, Pollefeys, 2025.
		double probInlier = 1.0;
		for (int i = 0; i < TEstimator.MinNumSamples; ++i)
		{
			// size_t arithmetic, as in COLMAP: with fewer inliers than samples the difference
			// wraps to a huge value, but an earlier factor is then already zero.
			probInlier *= (double)unchecked(numInliers - (ulong)i) / (double)unchecked(numSamples - (ulong)i);
		}

		double probOutlier = 1 - probInlier;
		if (probOutlier <= 0)
		{
			return 1;
		}

		// Prevent division by zero below.
		if (probOutlier == 1.0)
		{
			return ulong.MaxValue;
		}

		return (ulong)Math.Ceiling(Math.Log(probFailure) / Math.Log(probOutlier) * numTrialsMultiplier);
	}

	/// <summary>
	/// Robustly estimate model with RANSAC (RANdom SAmple Consensus).
	/// </summary>
	/// <param name="x">Independent variables.</param>
	/// <param name="y">Dependent variables.</param>
	/// <returns>The report with the results of the estimation.</returns>
	public RansacReport<TModel, TSupport> Estimate(ReadOnlySpan<TX> x, ReadOnlySpan<TY> y)
	{
		Check.Eq(x.Length, y.Length);

		int numSamples = x.Length;

		var report = new RansacReport<TModel, TSupport>();
		report.Success = false;
		report.NumTrials = 0;

		if (numSamples < TEstimator.MinNumSamples)
		{
			return report;
		}

		var bestSupport = new TSupport();
		TModel bestModel = default!;
		bool hasBestModel = false;

		double maxResidual = Options.MaxError * Options.MaxError;
		ulong minNumTrials = unchecked((ulong)(long)Options.MinNumTrials);

		Sampler.Initialize(numSamples);
		ulong maxNumTrials = Math.Min(unchecked((ulong)(long)Options.MaxNumTrials), Sampler.MaxNumSamples());

		CheckNumThreads(Options.NumThreads, "Parallel RANSAC only supports RandomSampler");

		ulong trialCounter = 0;
		ulong dynMaxNumTrials = maxNumTrials;
		bool abortFlag = false;

		// COLMAP's per-thread copies of the mutable objects; one "thread" here (file header).
		TSampler threadSampler = TSampler.Create(TEstimator.MinNumSamples);
		threadSampler.Initialize(numSamples);
		TEstimator threadEstimator = Estimator;
		TSupportMeasurer threadSupportMeasurer = SupportMeasurer;

		SeedThreadPrng(Options.RandomSeed);

		// Working buffers.
		var residuals = new double[numSamples];
		var xRand = new TX[TEstimator.MinNumSamples];
		var yRand = new TY[TEstimator.MinNumSamples];
		var sampleModels = new List<TModel>();

		while (true)
		{
			ulong currThreadTrial = trialCounter++;
			if (currThreadTrial >= maxNumTrials || abortFlag)
			{
				break;
			}

			threadSampler.SampleXY<TSampler, TX, TY>(x, y, xRand, yRand);

			// Estimate model for current subset.
			sampleModels.Clear();
			threadEstimator.Estimate(xRand, yRand, sampleModels);

			// Iterate through all estimated models.
			foreach (TModel sampleModel in sampleModels)
			{
				threadEstimator.Residuals(x, y, sampleModel, residuals);

				TSupport support = threadSupportMeasurer.Evaluate(residuals, maxResidual);

				// Save as best subset if better than all previous subsets.
				if (threadSupportMeasurer.IsLeftBetter(support, bestSupport))
				{
					bestSupport = support;
					bestModel = sampleModel;
					hasBestModel = true;

					dynMaxNumTrials = ComputeNumTrials(
						(ulong)bestSupport.NumInliers,
						(ulong)numSamples,
						Options.Confidence,
						Options.DynNumTrialsMultiplier);
				}

				if (currThreadTrial >= dynMaxNumTrials && currThreadTrial >= minNumTrials)
				{
					abortFlag = true;
					break;
				}
			}

			if (abortFlag)
			{
				break;
			}
		}

		report.NumTrials = trialCounter;

		if (!hasBestModel)
		{
			return report;
		}

		report.Support = bestSupport;
		report.Model = bestModel;

		// No valid model was found.
		if (report.Support.NumInliers < TEstimator.MinNumSamples)
		{
			return report;
		}

		report.Success = true;

		// Determine inlier mask. Note that this calculates the residuals for the best model
		// twice, but saves to copy and fill the inlier mask for each evaluated model. Some
		// benchmarking revealed that this approach is faster.
		var finalResiduals = new double[numSamples];
		Estimator.Residuals(x, y, report.Model, finalResiduals);
		report.InlierMask = InlierMaskOf(finalResiduals, maxResidual);

		return report;
	}

	/// <summary>
	/// COLMAP's thread-count validation: GetEffectiveNumThreads, then parallel RANSAC is only
	/// allowed with RandomSampler. The loop itself runs serially either way (file header).
	/// </summary>
	protected static void CheckNumThreads(int numThreads, string message)
	{
		int effectiveNumThreads = Threading.GetEffectiveNumThreads(numThreads);
		if (typeof(TSampler) != typeof(RandomSampler))
		{
			Check.Eq(effectiveNumThreads, 1, message);
		}
	}

	/// <summary>
	/// Seed the calling thread's PRNG with <paramref name="randomSeed"/> (plus thread index 0)
	/// when the sampler is randomized and the seed is not -1, as COLMAP seeds each thread.
	/// </summary>
	protected static void SeedThreadPrng(int randomSeed)
	{
		if (TSampler.IsRandomized && randomSeed != -1)
		{
			// SetPRNGSeed takes an unsigned seed; the int converts as in C++.
			RandomUtils.SetPRNGSeed(unchecked((uint)randomSeed));
		}
	}

	/// <summary>The inlier mask: residual &lt;= max residual.</summary>
	protected static bool[] InlierMaskOf(ReadOnlySpan<double> residuals, double maxResidual)
	{
		var inlierMask = new bool[residuals.Length];
		for (int i = 0; i < residuals.Length; ++i)
		{
			inlierMask[i] = residuals[i] <= maxResidual;
		}

		return inlierMask;
	}
}

/// <summary>
/// RANSAC with COLMAP's default support measurer and sampler
/// (<see cref="InlierSupportMeasurer"/>, <see cref="RandomSampler"/>), i.e.
/// <c>RANSAC&lt;Estimator&gt;</c>.
/// </summary>
public class Ransac<TEstimator, TX, TY, TModel>
	: Ransac<TEstimator, TX, TY, TModel, InlierSupportMeasurer, InlierSupportMeasurer.Support, RandomSampler>
	where TEstimator : IEstimator<TX, TY, TModel>
{
	/// <summary>Create a RANSAC estimator with the default measurer and sampler.</summary>
	public Ransac(RansacOptions options, TEstimator estimator)
		: base(options, estimator, new InlierSupportMeasurer())
	{
	}
}
