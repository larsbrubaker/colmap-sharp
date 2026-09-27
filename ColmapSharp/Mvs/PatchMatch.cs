// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatch: the PatchMatch class of colmap/mvs/patch_match.h and .cc - one stereo problem
// (a reference image, its source images, and the inputs they index), its validation, and
// the run that estimates the reference image's depth and normal maps. PatchMatchController.cs
// builds the problems; PatchMatchOptions.cs holds the settings; PatchMatchCpu.cs is the CPU
// port of patch_match_cuda.cu that Run delegates to. RunAsync with a host-provided compute
// device runs PatchMatchGpu.cs (the WGSL kernels) instead when PatchMatchGpuPlan.cs says the
// device can hold the problem, and falls back to PatchMatchCpu otherwise, recording which
// backend ran and why (docs/CPP_DIVERGENCES.md, entry 136). Tests:
// ColmapSharp.Tests/Mvs/PatchMatchTests.cs, PatchMatchRunTests.cs and PatchMatchBackendTests.cs
// (C#-only; COLMAP has no patch_match_test.cc).
//
// Translation notes:
// - Problem holds the image, depth map and normal map lists by reference, like COLMAP's
//   pointers: the controller fills them per problem.
// - Check drops COLMAP's gpu_index checks (docs/CPP_DIVERGENCES.md, entry 86): the GPU, when
//   there is one, is the IComputeDevice the host passes to RunAsync, not an index.
// - Run and RunAsync take a CancellationToken and an IProgress<double> (the fraction of sweeps
//   done).
// - A GPU error is not a reason to fall back: it propagates, so a broken device is noticed
//   rather than silently costing a CPU run's time.
// - Problem::Print is not ported (it is a LOG(INFO) listing).

using ColmapSharp.Compute;

namespace ColmapSharp.Mvs;

/// <summary>Where a <see cref="PatchMatch"/> run computed its maps.</summary>
public enum PatchMatchBackend
{
	/// <summary>On the CPU (PatchMatchCpu).</summary>
	Cpu,

	/// <summary>On the host's compute device (PatchMatchGpu).</summary>
	Gpu,
}

/// <summary>
/// The results of a finished run, whichever backend computed them: the getters PatchMatchCpu
/// and PatchMatchGpu share, so PatchMatch reads either the same way.
/// </summary>
internal interface IPatchMatchResult
{
	DepthMap GetDepthMap();

	NormalMap GetNormalMap();

	Mat<float> GetSelProbMap();

	List<int> GetConsistentImageIdxs();
}

/// <summary>
/// Port of colmap::mvs::PatchMatch: estimates the depth and normal map of a reference image
/// from its source images.
/// </summary>
public sealed class PatchMatch
{
	private readonly PatchMatchOptions options;
	private readonly Problem problem;
	private IPatchMatchResult? result;

	/// <summary>Port of PatchMatch::Problem: one reference image and its source images.</summary>
	public sealed class Problem
	{
		/// <summary>Index of the reference image.</summary>
		public int RefImageIdx { get; set; } = -1;

		/// <summary>Indices of the source images.</summary>
		public List<int> SrcImageIdxs { get; set; } = new();

		/// <summary>Input images for the photometric consistency term.</summary>
		public List<Image>? Images { get; set; }

		/// <summary>Input depth maps for the geometric consistency term.</summary>
		public List<DepthMap>? DepthMaps { get; set; }

		/// <summary>Input normal maps for the geometric consistency term.</summary>
		public List<NormalMap>? NormalMaps { get; set; }

		/// <summary>
		/// A copy whose source index list is its own; the input lists are shared, as the
		/// C++ copy shares the pointers.
		/// </summary>
		public Problem Clone()
		{
			var copy = (Problem)MemberwiseClone();
			copy.SrcImageIdxs = new List<int>(SrcImageIdxs);
			return copy;
		}
	}

	/// <summary>A problem to solve with the given options (both are copied, as in C++).</summary>
	public PatchMatch(PatchMatchOptions options, Problem problem)
	{
		this.options = Util.Check.NotNull(options).Clone();
		this.problem = Util.Check.NotNull(problem).Clone();
	}

	/// <summary>
	/// Checks the options and the problem for validity: valid options, no duplicate images,
	/// the reference image not among the sources, every used image grey with a bitmap of its
	/// own size and a calibration with only fx, fy, cx and cy, and (with geometric
	/// consistency) depth maps and the reference normal map of matching size. Throws with
	/// COLMAP's "Check failed" message otherwise. Port of PatchMatch::Check.
	/// </summary>
	public void Check()
	{
		Util.Check.That(options.Check());

		List<Image> images = Util.Check.NotNull(problem.Images);
		if (options.GeomConsistency)
		{
			Util.Check.NotNull(problem.DepthMaps);
			Util.Check.NotNull(problem.NormalMaps);
			Util.Check.Eq(problem.DepthMaps!.Count, images.Count);
			Util.Check.Eq(problem.NormalMaps!.Count, images.Count);
		}

		Util.Check.Gt(problem.SrcImageIdxs.Count, 0);

		// Check that there are no duplicate images and that the reference image
		// is not defined as a source image. COLMAP's std::set iterates in ascending order,
		// so the checks below fail on the same image as COLMAP's.
		var uniqueImageIdxs = new SortedSet<int>(problem.SrcImageIdxs) { problem.RefImageIdx };
		Util.Check.Eq(problem.SrcImageIdxs.Count + 1, uniqueImageIdxs.Count);

		// Check that input data is well-formed.
		foreach (int imageIdx in uniqueImageIdxs)
		{
			string idx = imageIdx.ToString(System.Globalization.CultureInfo.InvariantCulture);
			Util.Check.Ge(imageIdx, 0, idx);
			Util.Check.Lt(imageIdx, images.Count, idx);

			Image image = images[imageIdx];
			Util.Check.Gt(image.GetBitmap().Width, 0, idx);
			Util.Check.Gt(image.GetBitmap().Height, 0, idx);
			Util.Check.That(image.GetBitmap().IsGrey, idx);
			Util.Check.Eq(image.GetWidth(), image.GetBitmap().Width, idx);
			Util.Check.Eq(image.GetHeight(), image.GetBitmap().Height, idx);

			// Make sure, the calibration matrix only contains fx, fy, cx, cy.
			ReadOnlySpan<float> k = image.GetK();
			Util.Check.Lt(MathF.Abs(k[1] - 0.0f), 1e-6f, idx);
			Util.Check.Lt(MathF.Abs(k[3] - 0.0f), 1e-6f, idx);
			Util.Check.Lt(MathF.Abs(k[6] - 0.0f), 1e-6f, idx);
			Util.Check.Lt(MathF.Abs(k[7] - 0.0f), 1e-6f, idx);
			Util.Check.Lt(MathF.Abs(k[8] - 1.0f), 1e-6f, idx);

			if (options.GeomConsistency)
			{
				Util.Check.Lt(imageIdx, problem.DepthMaps!.Count, idx);
				DepthMap depthMap = problem.DepthMaps[imageIdx];
				Util.Check.Eq(image.GetWidth(), depthMap.GetWidth(), idx);
				Util.Check.Eq(image.GetHeight(), depthMap.GetHeight(), idx);
			}
		}

		if (options.GeomConsistency)
		{
			Image refImage = images[problem.RefImageIdx];
			NormalMap refNormalMap = problem.NormalMaps![problem.RefImageIdx];
			Util.Check.Eq(refImage.GetWidth(), refNormalMap.GetWidth());
			Util.Check.Eq(refImage.GetHeight(), refNormalMap.GetHeight());
		}
	}

	/// <summary>
	/// Where the last run computed its maps. Meaningful only after a run completed: it is reset
	/// when a run starts, and a failed or cancelled run leaves it undefined.
	/// </summary>
	public PatchMatchBackend Backend { get; private set; }

	/// <summary>
	/// Why the last run used the CPU although a compute device was given: a sentence for the
	/// user ending "Using the CPU." (e.g. the image needs more GPU memory than one buffer can
	/// hold). Null when the GPU ran or no device was given. Meaningful only after a run
	/// completed: it is reset when a run starts, and a failed or cancelled run leaves it
	/// undefined.
	/// </summary>
	public string? FallbackReason { get; private set; }

	/// <summary>
	/// Checks the problem and runs PatchMatch on it on the CPU: num_iterations x 4 sweeps, with
	/// geometric consistency and filtering as the options ask. Port of PatchMatch::Run.
	/// </summary>
	public void Run(CancellationToken cancellationToken = default, IProgress<double>? progress = null)
	{
		BeginRun();
		RunCpu(cancellationToken, progress);
	}

	/// <summary>
	/// Checks the problem and runs PatchMatch on it: on <paramref name="device"/> when one is
	/// given and it can hold the problem (<see cref="Backend"/> becomes
	/// <see cref="PatchMatchBackend.Gpu"/>), else on the CPU, with the reason in
	/// <see cref="FallbackReason"/>. A device failure during a GPU run faults the task; it
	/// does not fall back to the CPU.
	/// <para>
	/// For hosts: a CPU run (no device, or a fallback) runs synchronously on the calling
	/// thread before the returned task completes, so a UI host should start this call off its
	/// UI thread. The GPU path awaits the device and never blocks on it, so this is the entry
	/// point for any device, including one that cannot be waited on synchronously.
	/// </para>
	/// </summary>
	public async Task RunAsync(IComputeDevice? device, CancellationToken cancellationToken = default, IProgress<double>? progress = null)
	{
		BeginRun();
		if (device != null)
		{
			// Planned from the problem's sizes alone, so a fallback costs no GPU-side setup.
			PatchMatchGpuProblemShape shape = PatchMatchGpu.ShapeOf(problem);
			if (PatchMatchGpuPlan.TryCreate(shape, options, device.Limits, null, out _, out string? reason))
			{
				var gpu = new PatchMatchGpu(options, problem);
				await gpu.RunAsync(device, cancellationToken, progress).ConfigureAwait(false);
				result = gpu;
				Backend = PatchMatchBackend.Gpu;
				return;
			}

			FallbackReason = reason;
		}

		RunCpu(cancellationToken, progress);
	}

	// Checks the problem and clears the last run's results: results become readable only
	// after a complete run; a cancelled or failed one leaves none.
	private void BeginRun()
	{
		Check();
		result = null;
		Backend = PatchMatchBackend.Cpu;
		FallbackReason = null;
	}

	private void RunCpu(CancellationToken cancellationToken, IProgress<double>? progress)
	{
		var run = new PatchMatchCpu(options, problem);
		run.Run(cancellationToken, progress);
		result = run;
	}

	/// <summary>The estimated depth map (0 where filtered). Port of PatchMatch::GetDepthMap.</summary>
	public DepthMap GetDepthMap() => RunResult.GetDepthMap();

	/// <summary>The estimated normal map (0 where filtered). Port of PatchMatch::GetNormalMap.</summary>
	public NormalMap GetNormalMap() => RunResult.GetNormalMap();

	/// <summary>
	/// The per-pixel selection probability of each source image (in the problem's source
	/// order). Port of PatchMatch::GetSelProbMap.
	/// </summary>
	public Mat<float> GetSelProbMap() => RunResult.GetSelProbMap();

	/// <summary>
	/// The source images each pixel is consistent with (empty unless filtering ran).
	/// Port of PatchMatch::GetConsistencyGraph.
	/// </summary>
	public ConsistencyGraph GetConsistencyGraph()
	{
		Image refImage = problem.Images![problem.RefImageIdx];
		return new ConsistencyGraph(refImage.GetWidth(), refImage.GetHeight(), RunResult.GetConsistentImageIdxs().ToArray());
	}

	// COLMAP dereferences a null patch_match_cuda_ before Run; here that is a clear error.
	private IPatchMatchResult RunResult =>
		result ?? throw new InvalidOperationException("PatchMatch.Run must be called before reading its results.");
}
