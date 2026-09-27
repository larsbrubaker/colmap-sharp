// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// LoRansac: colmap/optim/loransac.h - LO-RANSAC (Chum, Matas and Kittler, DAGM 2003): RANSAC
// (Ransac.cs, which it derives from for the options, members and ComputeNumTrials) plus a
// local optimization that re-estimates the model from the inliers of every new best sample
// model, up to 10 times while the support keeps improving. The local estimator is an
// ILocalEstimator (Estimator.cs), which also stands in for COLMAP's compile-time Refine
// detection. Tests: ColmapSharp.Tests/Optim/LoRansacTests.cs (loransac_test.cc 1:1).
//
// Tier C (outcome), like Ransac.cs; the sampling sequence is Tier A.
//
// Translation notes:
// - Same threading decision as Ransac.cs: the trial loop runs once on the calling thread,
//   as in COLMAP built without OpenMP; the speculative-check-then-commit structure COLMAP
//   uses for its parallel region is kept, and serially it is the plain algorithm
//   (docs/CPP_DIVERGENCES.md, entry 17).
// - COLMAP std::swap()s its two residual vectors to keep the best local model's residuals
//   without copying; the two arrays are swapped by reference here the same way.
// - Estimate hides Ransac.Estimate (`new`) as the C++ member hides the base's: neither is
//   virtual.

using System.Runtime.InteropServices;

using ColmapSharp.Util;

namespace ColmapSharp.Optim;

/// <summary>
/// Implementation of LO-RANSAC (Locally Optimized RANSAC).
///
/// "Locally Optimized RANSAC" Ondrej Chum, Jiri Matas, Josef Kittler, DAGM 2003.
///
/// Port of colmap::LORANSAC&lt;Estimator, LocalEstimator, SupportMeasurer, Sampler&gt;.
/// </summary>
public class LoRansac<TEstimator, TLocalEstimator, TX, TY, TModel, TSupportMeasurer, TSupport, TSampler>
	: Ransac<TEstimator, TX, TY, TModel, TSupportMeasurer, TSupport, TSampler>
	where TEstimator : IEstimator<TX, TY, TModel>
	where TLocalEstimator : ILocalEstimator<TX, TY, TModel>
	where TSupportMeasurer : class, ISupportMeasurer<TSupport>
	where TSupport : struct, IRansacSupport
	where TSampler : class, ISampler<TSampler>
{
	// Recursive local optimization to expand the inlier set stops after this many rounds.
	private const int MaxNumLocalTrials = 10;

	/// <summary>
	/// Create a LO-RANSAC estimator. <paramref name="sampler"/> defaults to a sampler of
	/// TEstimator.MinNumSamples elements, as COLMAP's does.
	/// </summary>
	public LoRansac(
		RansacOptions options,
		TEstimator estimator,
		TLocalEstimator localEstimator,
		TSupportMeasurer supportMeasurer,
		TSampler? sampler = null)
		: base(options, estimator, supportMeasurer, sampler)
	{
		LocalEstimator = localEstimator;
	}

	/// <summary>The local optimizer.</summary>
	public TLocalEstimator LocalEstimator { get; set; }

	/// <summary>
	/// Robustly estimate model with LO-RANSAC.
	/// </summary>
	/// <param name="x">Independent variables.</param>
	/// <param name="y">Dependent variables.</param>
	/// <returns>The report with the results of the estimation.</returns>
	public new RansacReport<TModel, TSupport> Estimate(ReadOnlySpan<TX> x, ReadOnlySpan<TY> y)
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

		double maxResidual = Options.MaxError * Options.MaxError;
		ulong minNumTrials = unchecked((ulong)(long)Options.MinNumTrials);

		Sampler.Initialize(numSamples);
		ulong maxNumTrials = Math.Min(unchecked((ulong)(long)Options.MaxNumTrials), Sampler.MaxNumSamples());

		var bestSupport = new TSupport();
		TModel bestModel = default!;
		bool hasBestModel = false;
		bool bestModelIsLocal = false;

		CheckNumThreads(Options.NumThreads, "Parallel LORANSAC only supports RandomSampler");

		ulong trialCounter = 0;
		ulong dynMaxNumTrials = maxNumTrials;
		bool abortFlag = false;

		// COLMAP's per-thread copies of the mutable objects; one "thread" here (file header).
		TSampler threadSampler = TSampler.Create(TEstimator.MinNumSamples);
		threadSampler.Initialize(numSamples);
		TEstimator threadEstimator = Estimator;
		TLocalEstimator threadLocalEstimator = LocalEstimator;
		TSupportMeasurer threadSupportMeasurer = SupportMeasurer;

		SeedThreadPrng(Options.RandomSeed);

		// Working buffers.
		var residuals = new double[numSamples];
		var bestLocalResiduals = new double[numSamples];

		var xInlier = new List<TX>();
		var yInlier = new List<TY>();

		var xRand = new TX[TEstimator.MinNumSamples];
		var yRand = new TY[TEstimator.MinNumSamples];
		var sampleModels = new List<TModel>();
		var localModels = new List<TModel>();

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

			// Iterate through all estimated models. The first one that beats the best support
			// seeds the local optimization, so the order a minimal solver returns its models in
			// can change the result (docs/CPP_DIVERGENCES.md entry 124).
			foreach (TModel sampleModel in sampleModels)
			{
				threadEstimator.Residuals(x, y, sampleModel, residuals);

				TSupport support = threadSupportMeasurer.Evaluate(residuals, maxResidual);

				// Speculatively check if better than global best (COLMAP re-checks under its
				// lock below; serially the two checks see the same best support).
				bool isBetter = threadSupportMeasurer.IsLeftBetter(support, bestSupport);

				if (isBetter)
				{
					// Do local optimization.
					TSupport localBestSupport = support;
					TModel localBestModel = sampleModel;
					bool localBestIsLocal = false;

					// Estimate locally optimized model from inliers.
					if (support.NumInliers > TEstimator.MinNumSamples
						&& support.NumInliers >= TLocalEstimator.MinNumSamples)
					{
						// Recursive local optimization to expand inlier set.
						for (int localNumTrials = 0; localNumTrials < MaxNumLocalTrials; ++localNumTrials)
						{
							xInlier.Clear();
							yInlier.Clear();
							xInlier.EnsureCapacity(numSamples);
							yInlier.EnsureCapacity(numSamples);
							for (int i = 0; i < residuals.Length; ++i)
							{
								if (residuals[i] <= maxResidual)
								{
									xInlier.Add(x[i]);
									yInlier.Add(y[i]);
								}
							}

							localModels.Clear();
							threadLocalEstimator.EstimateLocal(
								CollectionsMarshal.AsSpan(xInlier),
								CollectionsMarshal.AsSpan(yInlier),
								localBestModel,
								localModels);

							bool improvedSupport = false;
							foreach (TModel localModel in localModels)
							{
								threadLocalEstimator.Residuals(x, y, localModel, residuals);

								TSupport localSupport = threadSupportMeasurer.Evaluate(residuals, maxResidual);

								// Check if locally optimized model is better.
								if (threadSupportMeasurer.IsLeftBetter(localSupport, localBestSupport))
								{
									localBestSupport = localSupport;
									localBestModel = localModel;
									localBestIsLocal = true;
									improvedSupport = true;
									(residuals, bestLocalResiduals) = (bestLocalResiduals, residuals);
								}
							}

							// Keep expanding only while the refit improves the support.
							if (!improvedSupport)
							{
								break;
							}

							// Swap back the residuals, so we can extract the best inlier set in
							// the next recursion of local optimization.
							(residuals, bestLocalResiduals) = (bestLocalResiduals, residuals);
						}
					}

					// Commit local optimization result to global best.
					if (threadSupportMeasurer.IsLeftBetter(localBestSupport, bestSupport))
					{
						bestSupport = localBestSupport;
						bestModel = localBestModel;
						hasBestModel = true;
						bestModelIsLocal = localBestIsLocal;

						dynMaxNumTrials = ComputeNumTrials(
							(ulong)bestSupport.NumInliers,
							(ulong)numSamples,
							Options.Confidence,
							Options.DynNumTrialsMultiplier);
					}
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
		if (bestModelIsLocal)
		{
			LocalEstimator.Residuals(x, y, report.Model, finalResiduals);
		}
		else
		{
			Estimator.Residuals(x, y, report.Model, finalResiduals);
		}

		report.InlierMask = InlierMaskOf(finalResiduals, maxResidual);

		return report;
	}
}

/// <summary>
/// LO-RANSAC with COLMAP's default support measurer and sampler
/// (<see cref="InlierSupportMeasurer"/>, <see cref="RandomSampler"/>), i.e.
/// <c>LORANSAC&lt;Estimator, LocalEstimator&gt;</c>.
/// </summary>
public class LoRansac<TEstimator, TLocalEstimator, TX, TY, TModel>
	: LoRansac<TEstimator, TLocalEstimator, TX, TY, TModel, InlierSupportMeasurer, InlierSupportMeasurer.Support, RandomSampler>
	where TEstimator : IEstimator<TX, TY, TModel>
	where TLocalEstimator : ILocalEstimator<TX, TY, TModel>
{
	/// <summary>Create a LO-RANSAC estimator with the default measurer and sampler.</summary>
	public LoRansac(RansacOptions options, TEstimator estimator, TLocalEstimator localEstimator)
		: base(options, estimator, localEstimator, new InlierSupportMeasurer())
	{
	}
}
