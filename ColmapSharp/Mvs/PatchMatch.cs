// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatch: the PatchMatch class of colmap/mvs/patch_match.h and .cc - one stereo problem
// (a reference image, its source images, and the inputs they index), its validation, and
// the run that estimates the reference image's depth and normal maps. PatchMatchController.cs
// builds the problems; PatchMatchOptions.cs holds the settings; PatchMatchCpu.cs is the CPU
// port of patch_match_cuda.cu that Run delegates to. Tests:
// ColmapSharp.Tests/Mvs/PatchMatchTests.cs and PatchMatchRunTests.cs (C#-only; COLMAP has no
// patch_match_test.cc).
//
// Translation notes:
// - Problem holds the image, depth map and normal map lists by reference, like COLMAP's
//   pointers: the controller fills them per problem.
// - Check drops COLMAP's gpu_index checks (no GPU here; docs/CPP_DIVERGENCES.md, entry 86).
// - Run takes a CancellationToken and an IProgress<double> (the fraction of sweeps done).
// - Problem::Print is not ported (it is a LOG(INFO) listing).

using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>
/// Port of colmap::mvs::PatchMatch: estimates the depth and normal map of a reference image
/// from its source images.
/// </summary>
public sealed class PatchMatch
{
	private readonly PatchMatchOptions options;
	private readonly Problem problem;
	private PatchMatchCpu? patchMatchCpu;

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
	/// Checks the problem and runs PatchMatch on it: num_iterations x 4 sweeps, with
	/// geometric consistency and filtering as the options ask. Port of PatchMatch::Run.
	/// </summary>
	public void Run(CancellationToken cancellationToken = default, IProgress<double>? progress = null)
	{
		Check();
		patchMatchCpu = null;

		// Results become readable only after a complete run; a cancelled or failed one
		// leaves none.
		var run = new PatchMatchCpu(options, problem);
		run.Run(cancellationToken, progress);
		patchMatchCpu = run;
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
	private PatchMatchCpu RunResult =>
		patchMatchCpu ?? throw new InvalidOperationException("PatchMatch.Run must be called before reading its results.");
}
