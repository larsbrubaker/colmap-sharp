// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatchCpu: the PatchMatchCuda class of patch_match_cuda.h/.cu, run on the CPU - it
// sets up one PatchMatch problem (the prefiltered reference image, the source layers, the
// rotated pose tables, the random or given initial depth and normal maps, the initial costs)
// and runs the iteration schedule: num_iterations x 4 sweeps, each a top-to-bottom pass over
// every column (PatchMatchCpu.Sweep.cs) followed by a 90-degree counter-clockwise rotation
// of every map, so the four sweeps of an iteration run in all four directions and the maps
// end in their original orientation. PatchMatch.cs (Run and the getters) is the public entry
// point. Tests: ColmapSharp.Tests/Mvs/PatchMatchRunTests.cs (C#-only).
//
// Tier C (docs/CPP_DIVERGENCES.md, entries 86, 95, 96). Deterministic: every parallel loop
// writes disjoint outputs and random draws are keyed on the pixel, so the result is the
// same for any thread count.
//
// Translation notes:
// - GpuMat buffers become Mat planes; each rotation writes into a second buffer that is then
//   swapped in, as COLMAP's Rotate does with freshly allocated GpuMats.
// - The per-column global workspace (forward messages and sampling probabilities) becomes a
//   scratch array per worker thread, which every column fully initializes before use.
// - The CUDA timers and logging are not ported; progress goes to an IProgress<double> (the
//   fraction of sweeps done) and cancellation is checked between sweeps and inside them.

namespace ColmapSharp.Mvs;

/// <summary>
/// Port of colmap::mvs::PatchMatchCuda on the CPU. Internal: PatchMatch.Run checks the
/// problem (map sizes included) before constructing it.
/// </summary>
internal sealed partial class PatchMatchCpu
{
	private readonly PatchMatchOptions options;
	private readonly PatchMatch.Problem problem;
	private readonly ulong seed;
	private readonly int refWidth;
	private readonly int refHeight;
	private readonly int windowRadius;
	private readonly int numSamples;
	private readonly float geomConsistencyRegularizer;
	private readonly float geomConsistencyMaxCost;
	private readonly float filterMinNcc;
	private readonly float filterMinTriangulationAngle;
	private readonly int filterMinNumConsistent;
	private readonly float filterGeomConsistencyMaxCost;
	private readonly PatchMatchLikelihood likelihood;
	private readonly PatchMatchTransforms transforms;
	private readonly PatchMatchSourceImages srcImages;
	private readonly PatchMatchSourceDepthMaps? srcDepthMaps;

	private int rotation;
	private PatchMatchRefImage refImage;
	private Mat<float> depthMap;
	private Mat<float> normalMap;
	private Mat<float> selProbMap;
	private Mat<float> prevSelProbMap;
	private Mat<float> costMap;
	private Mat<byte> consistencyMask = new(0, 0, 0);

	/// <summary>
	/// Sets up <paramref name="problem"/> (already checked by PatchMatch.Check) under
	/// <paramref name="options"/>: InitRefImage, InitSourceImages, InitTransforms and
	/// InitWorkspaceMemory. <paramref name="seed"/> keys the random draws.
	/// </summary>
	public PatchMatchCpu(PatchMatchOptions options, PatchMatch.Problem problem, ulong seed = PatchMatchRandom.DefaultSeed)
	{
		this.options = options.Clone();
		this.problem = problem;
		this.seed = seed;
		List<Image> images = Util.Check.NotNull(problem.Images);
		Image ref0 = images[problem.RefImageIdx];
		refWidth = ref0.GetWidth();
		refHeight = ref0.GetHeight();
		windowRadius = options.WindowRadius;
		numSamples = options.NumSamples;
		geomConsistencyRegularizer = (float)options.GeomConsistencyRegularizer;
		geomConsistencyMaxCost = (float)options.GeomConsistencyMaxCost;
		filterMinNcc = (float)options.FilterMinNcc;

		// DEG2RAD(deg) is deg * 0.0174532925199432 in double, stored as float.
		filterMinTriangulationAngle = (float)(options.FilterMinTriangulationAngle * 0.0174532925199432);
		filterMinNumConsistent = options.FilterMinNumConsistent;
		filterGeomConsistencyMaxCost = (float)options.FilterGeomConsistencyMaxCost;
		likelihood = new PatchMatchLikelihood(
			(float)options.NccSigma, (float)(options.MinTriangulationAngle * 0.0174532925199432), (float)options.IncidentAngleSigma);

		// InitTransforms (first: the random normals need the reference calibration).
		transforms = new PatchMatchTransforms(images, problem.RefImageIdx, problem.SrcImageIdxs);

		// InitRefImage.
		refImage = new PatchMatchRefImage(refWidth, refHeight);
		refImage.Filter(
			ref0.GetBitmap().RowMajorData, options.WindowRadius, options.WindowStep,
			(float)options.SigmaSpatial, (float)options.SigmaColor, options.NumThreads);

		// InitSourceImages: layers in the problem's source order.
		var sources = new List<Image>(problem.SrcImageIdxs.Count);
		foreach (int imageIdx in problem.SrcImageIdxs)
		{
			sources.Add(images[imageIdx]);
		}

		srcImages = new PatchMatchSourceImages(sources);
		if (options.GeomConsistency)
		{
			List<DepthMap> allDepthMaps = Util.Check.NotNull(problem.DepthMaps);
			var srcDepths = new List<DepthMap>(problem.SrcImageIdxs.Count);
			foreach (int imageIdx in problem.SrcImageIdxs)
			{
				srcDepths.Add(allDepthMaps[imageIdx]);
			}

			srcDepthMaps = new PatchMatchSourceDepthMaps(srcDepths, srcImages.MaxWidth, srcImages.MaxHeight);
		}

		// InitWorkspaceMemory.
		int numSrc = problem.SrcImageIdxs.Count;
		depthMap = new Mat<float>(refWidth, refHeight, 1);
		normalMap = new Mat<float>(refWidth, refHeight, 3);
		if (options.GeomConsistency)
		{
			Array.Copy(Util.Check.NotNull(problem.DepthMaps)[problem.RefImageIdx].Data, depthMap.Data, depthMap.Data.Length);
			Array.Copy(Util.Check.NotNull(problem.NormalMaps)[problem.RefImageIdx].Data, normalMap.Data, normalMap.Data.Length);
		}
		else
		{
			InitRandomDepthAndNormalMaps((float)options.DepthMin, (float)options.DepthMax);
		}

		selProbMap = new Mat<float>(refWidth, refHeight, numSrc);
		prevSelProbMap = new Mat<float>(refWidth, refHeight, numSrc);
		prevSelProbMap.Fill(0.5f);
		costMap = new Mat<float>(refWidth, refHeight, numSrc);
	}

	/// <summary>
	/// Runs the initial cost and all sweeps. Port of PatchMatchCuda::Run
	/// (RunWithWindowSizeAndStep).
	/// </summary>
	public void Run(CancellationToken cancellationToken = default, IProgress<double>? progress = null)
	{
		PatchMatchKernel.ComputeInitialCost(costMap, depthMap, normalMap, NewPhotoConsistency(), options.NumThreads, cancellationToken);

		float totalNumSteps = options.NumIterations * 4;
		for (int iter = 0; iter < options.NumIterations; ++iter)
		{
			for (int sweep = 0; sweep < 4; ++sweep)
			{
				cancellationToken.ThrowIfCancellationRequested();

				bool lastSweep = iter == options.NumIterations - 1 && sweep == 3;
				var sweepOptions = new SweepOptions
				{
					// std::pow(2.0f, float) is the float overload.
					Perturbation = 1.0f / MathF.Pow(2.0f, iter + sweep / 4.0f),
					PrevSelProbWeight = (iter * 4 + sweep) / totalNumSteps,
					Phase = PatchMatchRandom.SweepPhase(iter, sweep),
					GeomConsistencyTerm = options.GeomConsistency,
					FilterPhotoConsistency = lastSweep && options.Filter,
					FilterGeomConsistency = lastSweep && options.Filter && options.GeomConsistency,
				};

				if (lastSweep && options.Filter)
				{
					consistencyMask = new Mat<byte>(costMap.GetWidth(), costMap.GetHeight(), costMap.GetDepth());
				}

				RunSweep(sweepOptions, cancellationToken);

				Rotate();

				if (lastSweep && options.Filter)
				{
					var rotatedMask = new Mat<byte>(costMap.GetWidth(), costMap.GetHeight(), costMap.GetDepth());
					consistencyMask.Rotate(rotatedMask, options.NumThreads);
					consistencyMask = rotatedMask;
				}

				progress?.Report((iter * 4 + sweep + 1) / (double)totalNumSteps);
			}
		}
	}

	/// <summary>The estimated depth map (0 where filtered). Port of PatchMatchCuda::GetDepthMap.</summary>
	public DepthMap GetDepthMap() => new(CopyOf(depthMap), (float)options.DepthMin, (float)options.DepthMax);

	/// <summary>The estimated normal map (0 where filtered). Port of PatchMatchCuda::GetNormalMap.</summary>
	public NormalMap GetNormalMap() => new(CopyOf(normalMap));

	/// <summary>
	/// The selection probability of every source image per pixel after the last sweep.
	/// Port of PatchMatchCuda::GetSelProbMap.
	/// </summary>
	public Mat<float> GetSelProbMap() => CopyOf(prevSelProbMap);

	/// <summary>
	/// For every pixel with consistent source images: col, row, count, then the image
	/// indices (of the problem's images). Port of PatchMatchCuda::GetConsistentImageIdxs.
	/// </summary>
	public List<int> GetConsistentImageIdxs()
	{
		var consistentImageIdxs = new List<int>();
		var pixelConsistentImageIdxs = new List<int>(consistencyMask.GetDepth());
		for (int r = 0; r < consistencyMask.GetHeight(); ++r)
		{
			for (int c = 0; c < consistencyMask.GetWidth(); ++c)
			{
				pixelConsistentImageIdxs.Clear();
				for (int d = 0; d < consistencyMask.GetDepth(); ++d)
				{
					if (consistencyMask.Get(r, c, d) != 0)
					{
						pixelConsistentImageIdxs.Add(problem.SrcImageIdxs[d]);
					}
				}

				if (pixelConsistentImageIdxs.Count > 0)
				{
					consistentImageIdxs.Add(c);
					consistentImageIdxs.Add(r);
					consistentImageIdxs.Add(pixelConsistentImageIdxs.Count);
					consistentImageIdxs.AddRange(pixelConsistentImageIdxs);
				}
			}
		}

		return consistentImageIdxs;
	}

	private void RunSweep(SweepOptions sweepOptions, CancellationToken cancellationToken)
	{
		PatchMatchPhotoConsistency pcc = NewPhotoConsistency();
		int scratchSize = 2 * costMap.GetDepth() + 2 * pcc.WindowCount;
		ParallelOptions parallelOptions = Mat<float>.ParallelOptionsFor(options.NumThreads);
		parallelOptions.CancellationToken = cancellationToken;
		Parallel.For(
			0,
			costMap.GetWidth(),
			parallelOptions,
			() => new float[scratchSize],
			(col, _, scratch) =>
			{
				SweepColumn(col, sweepOptions, pcc, scratch);
				return scratch;
			},
			_ => { });
	}

	private PatchMatchPhotoConsistency NewPhotoConsistency() =>
		new(refImage, srcImages, transforms.Poses(rotation), new PatchMatchFrame(transforms, rotation),
			options.WindowRadius, options.WindowStep, (float)options.SigmaSpatial, (float)options.SigmaColor);

	/// <summary>
	/// FillWithRandomNumbers(depth_min, depth_max) and InitNormalMap: a uniform depth and a
	/// uniform camera-facing normal per pixel.
	/// </summary>
	private void InitRandomDepthAndNormalMaps(float depthMin, float depthMax)
	{
		var frame = new PatchMatchFrame(transforms, 0);
		float[] depths = depthMap.Data;
		float[] normals = normalMap.Data;
		int planeSize = refWidth * refHeight;
		Parallel.For(0, refHeight, Mat<float>.ParallelOptionsFor(options.NumThreads), row =>
		{
			Span<float> normal = stackalloc float[3];
			for (int col = 0; col < refWidth; ++col)
			{
				int pixel = row * refWidth + col;
				var depthRandom = new PatchMatchRandom(seed, row, col, PatchMatchRandom.InitDepthPhase);
				depths[pixel] = depthRandom.NextUniform() * (depthMax - depthMin) + depthMin;

				var normalRandom = new PatchMatchRandom(seed, row, col, PatchMatchRandom.InitNormalPhase);
				PatchMatchKernel.GenerateRandomNormal(frame, row, col, ref normalRandom, normal);
				WriteNormal(normals, planeSize, pixel, normal);
			}
		});
	}

	/// <summary>
	/// Rotates the reference image and every map by 90 degrees counter-clockwise; the
	/// selection probabilities become the previous ones. Port of PatchMatchCuda::Rotate.
	/// </summary>
	private void Rotate()
	{
		rotation = (rotation + 1) % 4;
		int numThreads = options.NumThreads;

		depthMap = Rotated(depthMap, numThreads);

		PatchMatchKernel.RotateNormalMap(normalMap, numThreads);
		normalMap = Rotated(normalMap, numThreads);

		var rotatedRefImage = new PatchMatchRefImage(refImage.Image.GetHeight(), refImage.Image.GetWidth());
		refImage.Image.Rotate(rotatedRefImage.Image, numThreads);
		refImage.SumImage.Rotate(rotatedRefImage.SumImage, numThreads);
		refImage.SquaredSumImage.Rotate(rotatedRefImage.SquaredSumImage, numThreads);
		refImage = rotatedRefImage;

		// The rotated selection probabilities become the previous ones; the next sweep
		// rewrites every entry of its fresh selection probability map (the backward pass
		// runs first).
		Mat<float> newPrev = new(selProbMap.GetHeight(), selProbMap.GetWidth(), selProbMap.GetDepth());
		selProbMap.Rotate(newPrev, numThreads);
		prevSelProbMap = newPrev;
		selProbMap = new Mat<float>(newPrev.GetWidth(), newPrev.GetHeight(), newPrev.GetDepth());

		costMap = Rotated(costMap, numThreads);
	}

	private static Mat<float> Rotated(Mat<float> mat, int numThreads)
	{
		var rotated = new Mat<float>(mat.GetHeight(), mat.GetWidth(), mat.GetDepth());
		mat.Rotate(rotated, numThreads);
		return rotated;
	}

	private static Mat<float> CopyOf(Mat<float> mat)
	{
		var copy = new Mat<float>(mat.GetWidth(), mat.GetHeight(), mat.GetDepth());
		Array.Copy(mat.Data, copy.Data, mat.Data.Length);
		return copy;
	}
}
