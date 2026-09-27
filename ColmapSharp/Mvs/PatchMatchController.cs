// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatchController: the PatchMatchController of colmap/mvs/patch_match.h and .cc, which
// processes all problems of a workspace: reading the workspace (ReadWorkspace), turning
// stereo/patch-match.cfg into problems with their source images (ReadProblems), and per
// problem (ProcessProblem) skipping it when its outputs exist, loading its inputs with its
// effective options (SetUpProblem), running PatchMatch and writing the depth, normal and
// (optionally) consistency-graph .bin files. With geometric consistency, Run first computes
// photometric maps for every problem without filtering, then the geometric ones from them.
// PatchMatch.cs runs one problem; Workspace.cs holds the data.
// Tests: ColmapSharp.Tests/Mvs/PatchMatchControllerTests.cs (C#-only; COLMAP has no
// patch_match_test.cc).
//
// patch-match.cfg lists problems as pairs of lines: a reference image name, then its source
// images as either "__all__" (every other image), "__auto__, N" (the N images sharing the
// most sparse points with it, among those with a large enough triangulation angle) or a
// comma-separated list of image names. Empty lines and lines starting with '#' are skipped.
//
// Translation notes:
// - "__auto__" ranks by shared point count only in COLMAP (std::partial_sort, ties in
//   libc++'s unspecified order); ties go to the lower image index here
//   (docs/CPP_DIVERGENCES.md, entry 84).
// - SetUpProblem collects the used images in a FlatHashSet in COLMAP, whose iteration order
//   becomes the order of the source images; here it is the reference image, then the
//   problem's source images in their configured order (entry 85).
// - COLMAP's gpu_index and ReadGpuIndices are not ported (entry 86): the host may instead
//   supply one IComputeDevice (ComputeDevice), which every problem runs on when it fits,
//   falling back to the CPU per problem (entry 136).
// - Bitmaps come from the host through an IBitmapSource (IBitmapSource.cs); "does the image
//   exist" asks the source, as Workspace.HasBitmap does.
// - LOG(WARNING)/LOG(ERROR) go to Util/Log.cs; the LOG(INFO) lines and timers are not
//   ported, and progress is reported per problem through IProgress<ControllerProgress>.
// - COLMAP runs one problem per GPU in parallel; here problems run one after another, each
//   PatchMatch run itself parallel (NumThreads on the CPU, or on the one device), which
//   gives the same outputs (docs/CPP_DIVERGENCES.md, entry 122).
// - Run blocks; RunAsync awaits the compute device, for a host (the browser) whose device
//   cannot be waited on synchronously.
// - Cancellation is COLMAP's CheckIfStopped, checked before each problem; as in COLMAP a
//   stop is not an error, so Run returns normally. Unlike COLMAP, which finishes the
//   problem in flight, the token also reaches PatchMatch.RunAsync, so a stop aborts the running
//   problem and it writes nothing (entry 122).
// - A Model already in memory can stand in for <workspace>/sparse (as for Workspace).

using System.Globalization;

using ColmapSharp.Compute;
using ColmapSharp.Controllers;
using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>Port of colmap::mvs::PatchMatchController: PatchMatch stereo over a whole workspace.</summary>
public sealed class PatchMatchController
{
	private readonly PatchMatchOptions options;
	private readonly string workspacePath;
	private readonly string workspaceFormat;
	private readonly string pmvsOptionName;
	private readonly string configPath;
	private readonly IBitmapSource bitmapSource;
	private readonly Model? inMemoryModel;

	// Only one problem at a time reads from the workspace (COLMAP's workspace_mutex_).
	private readonly object workspaceLock = new();

	private Workspace? workspace;
	private List<(float Min, float Max)> depthRanges = new();

	/// <summary>
	/// A controller for the workspace at <paramref name="workspacePath"/> in
	/// <paramref name="workspaceFormat"/> ("COLMAP" or "PMVS", with the PMVS option file
	/// <paramref name="pmvsOptionName"/>), with image pixels from
	/// <paramref name="bitmapSource"/>. <paramref name="configPath"/> overrides
	/// &lt;workspace&gt;/&lt;stereo folder&gt;/patch-match.cfg when not empty.
	/// </summary>
	public PatchMatchController(
		PatchMatchOptions options,
		string workspacePath,
		string workspaceFormat,
		string pmvsOptionName,
		IBitmapSource bitmapSource,
		string configPath = "")
	{
		this.options = Check.NotNull(options).Clone();
		this.workspacePath = workspacePath;
		this.workspaceFormat = workspaceFormat;
		this.pmvsOptionName = pmvsOptionName;
		this.bitmapSource = Check.NotNull(bitmapSource);
		this.configPath = configPath;
	}

	/// <summary>
	/// A controller for a COLMAP-format workspace at <paramref name="workspacePath"/> whose
	/// sparse model is <paramref name="model"/> (already in memory, e.g. built with
	/// Model.ReadFromCOLMAP from a Reconstruction) instead of &lt;workspace&gt;/sparse. The
	/// workspace works on its own copy of the model. Configuration and outputs live under
	/// &lt;workspace&gt;/stereo as usual.
	/// </summary>
	public PatchMatchController(
		PatchMatchOptions options,
		Model model,
		string workspacePath,
		IBitmapSource bitmapSource,
		string configPath = "")
		: this(options, workspacePath, "COLMAP", "", bitmapSource, configPath)
	{
		inMemoryModel = Check.NotNull(model);
	}

	/// <summary>
	/// Runs PatchMatch stereo on every problem of patch-match.cfg, writing
	/// &lt;stereo&gt;/depth_maps, normal_maps (and, if WriteConsistencyGraph,
	/// consistency_graphs)/&lt;image name&gt;.{photometric,geometric}.bin. Problems whose
	/// outputs already exist are skipped. Cancelling <paramref name="cancellationToken"/>
	/// stops after writing the finished problems. Port of PatchMatchController::Run.
	/// This blocks on the <see cref="ComputeDevice"/>'s work, so the device must report
	/// <see cref="IComputeDevice.SupportsBlockingWait"/>: blocking must be safe on any thread,
	/// including one the host marshals device calls to. With a device that reports false (the
	/// browser, or a host marshalling to a thread it may block) this throws
	/// <see cref="InvalidOperationException"/>; use <see cref="RunAsync"/> there.
	/// </summary>
	public void Run(CancellationToken cancellationToken = default, IProgress<ControllerProgress>? progress = null)
	{
		if (ComputeDevice != null && !ComputeDevice.SupportsBlockingWait)
		{
			throw new InvalidOperationException(
				"This compute device cannot be waited on synchronously (e.g. in the browser); use RunAsync instead of Run.");
		}

		// Without a device every await in RunAsync completes synchronously; with one, the device
		// promised that a blocking wait completes, and the awaits use ConfigureAwait(false).
		// GetResult rethrows the run's own exception rather than an AggregateException.
		RunAsync(cancellationToken, progress).GetAwaiter().GetResult();
	}

	/// <summary>
	/// The host's compute device to run each problem's PatchMatch on, or null for the CPU. A
	/// problem the device cannot hold runs on the CPU, with the reason (PatchMatch's
	/// FallbackReason) logged as a warning; a device error fails the run. Only when a device is
	/// set does each progress report's Message gain " (GPU)" or " (CPU)" after the reference
	/// image's name, saying where that problem ran.
	/// </summary>
	public IComputeDevice? ComputeDevice { get; init; }

	/// <summary>
	/// <see cref="Run"/>, awaiting the <see cref="ComputeDevice"/>'s work instead of blocking
	/// on it: the entry point for a device that cannot be waited on synchronously. Problems
	/// still run one at a time (docs/CPP_DIVERGENCES.md, entry 122); a problem on the CPU runs
	/// synchronously on the calling thread inside this call, so a UI host should start it off
	/// its UI thread.
	/// </summary>
	public async Task RunAsync(CancellationToken cancellationToken = default, IProgress<ControllerProgress>? progress = null)
	{
		ReadWorkspace();
		ReadProblems();

		int total = Problems.Count * (options.GeomConsistency ? 2 : 1);
		int done = 0;

		// If geometric consistency is enabled, then photometric output must be computed first
		// for all images without filtering.
		if (options.GeomConsistency)
		{
			PatchMatchOptions photometricOptions = options.Clone();
			photometricOptions.GeomConsistency = false;
			photometricOptions.Filter = false;

			for (int problemIdx = 0; problemIdx < Problems.Count; ++problemIdx)
			{
				(bool finished, PatchMatchBackend? backend) =
					await ProcessProblemAsync(photometricOptions, problemIdx, cancellationToken).ConfigureAwait(false);
				if (!finished)
				{
					return;
				}

				progress?.Report(new ControllerProgress("PatchMatch photometric", ++done, total, ProblemMessage(problemIdx, backend)));
			}
		}

		for (int problemIdx = 0; problemIdx < Problems.Count; ++problemIdx)
		{
			(bool finished, PatchMatchBackend? backend) =
				await ProcessProblemAsync(options, problemIdx, cancellationToken).ConfigureAwait(false);
			if (!finished)
			{
				return;
			}

			string stage = options.GeomConsistency ? "PatchMatch geometric" : "PatchMatch photometric";
			progress?.Report(new ControllerProgress(stage, ++done, total, ProblemMessage(problemIdx, backend)));
		}
	}

	/// <summary>
	/// Processes problem <paramref name="problemIdx"/> under <paramref name="problemOptions"/>
	/// (unless its outputs exist). Finished is false when stopped by
	/// <paramref name="cancellationToken"/>; Backend is where PatchMatch ran, null when the
	/// problem was skipped or stopped. Port of PatchMatchController::ProcessProblem.
	/// </summary>
	internal async Task<(bool Finished, PatchMatchBackend? Backend)> ProcessProblemAsync(
		PatchMatchOptions problemOptions, int problemIdx, CancellationToken cancellationToken)
	{
		if (cancellationToken.IsCancellationRequested)
		{
			return (false, null);
		}

		string stereoFolder = Workspace.GetOptions().StereoFolder;
		string outputType = problemOptions.GeomConsistency ? "geometric" : "photometric";
		string fileName = RefImageName(problemIdx) + "." + outputType + ".bin";
		string depthMapPath = Path.Combine(workspacePath, stereoFolder, "depth_maps", fileName);
		string normalMapPath = Path.Combine(workspacePath, stereoFolder, "normal_maps", fileName);
		string consistencyGraphPath = Path.Combine(workspacePath, stereoFolder, "consistency_graphs", fileName);

		if (File.Exists(depthMapPath) && File.Exists(normalMapPath)
			&& (!problemOptions.WriteConsistencyGraph || File.Exists(consistencyGraphPath)))
		{
			return (true, null);
		}

		(PatchMatchOptions patchMatchOptions, PatchMatch.Problem problem) = SetUpProblem(problemOptions, problemIdx);

		var patchMatch = new PatchMatch(patchMatchOptions, problem);
		ProblemRunning?.Invoke(problemIdx);
		try
		{
			await patchMatch.RunAsync(ComputeDevice, cancellationToken).ConfigureAwait(false);
		}
		catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
		{
			return (false, null);
		}

		if (patchMatch.FallbackReason != null)
		{
			Log.Warning($"PatchMatch {outputType} {RefImageName(problemIdx)}: {patchMatch.FallbackReason}");
		}

		patchMatch.GetDepthMap().Write(depthMapPath);
		patchMatch.GetNormalMap().Write(normalMapPath);
		if (problemOptions.WriteConsistencyGraph)
		{
			patchMatch.GetConsistencyGraph().Write(consistencyGraphPath);
		}

		return (true, patchMatch.Backend);
	}

	// The reference image's name, and with a compute device where its problem ran (nothing
	// for a skipped problem, whose maps came from an earlier run).
	private string ProblemMessage(int problemIdx, PatchMatchBackend? backend)
	{
		string name = RefImageName(problemIdx);
		return ComputeDevice == null || backend == null
			? name
			: name + (backend == PatchMatchBackend.Gpu ? " (GPU)" : " (CPU)");
	}

	/// <summary>
	/// Called with the problem index right before a problem's PatchMatch run starts (after
	/// its inputs are loaded). A test hook for cancelling mid-problem.
	/// </summary>
	internal Action<int>? ProblemRunning { get; set; }

	private string RefImageName(int problemIdx) => Workspace.GetModel().GetImageName(Problems[problemIdx].RefImageIdx);

	/// <summary>The problems read by <see cref="ReadProblems"/>.</summary>
	internal List<PatchMatch.Problem> Problems { get; } = new();

	/// <summary>The workspace read by <see cref="ReadWorkspace"/>.</summary>
	internal Workspace Workspace => Check.NotNull(workspace);

	/// <summary>
	/// Opens the workspace as a CachedWorkspace of grey images (importing a PMVS workspace
	/// first) and computes each image's depth range from the sparse model.
	/// Port of PatchMatchController::ReadWorkspace.
	/// </summary>
	internal void ReadWorkspace()
	{
		var workspaceOptions = new Workspace.Options();

		string workspaceFormatLowerCase = workspaceFormat.ToLowerInvariant();
		if (workspaceFormatLowerCase == "pmvs")
		{
			workspaceOptions.StereoFolder = "stereo-" + pmvsOptionName;
		}

		workspaceOptions.MaxImageSize = options.MaxImageSize;
		workspaceOptions.ImageAsRgb = false;
		workspaceOptions.CacheSize = options.CacheSize;
		workspaceOptions.WorkspacePath = workspacePath;
		workspaceOptions.WorkspaceFormat = workspaceFormat;
		workspaceOptions.InputType = options.GeomConsistency ? "photometric" : "";

		workspace = inMemoryModel == null
			? new CachedWorkspace(workspaceOptions, bitmapSource)
			: new CachedWorkspace(workspaceOptions, inMemoryModel, bitmapSource);

		if (workspaceFormatLowerCase == "pmvs")
		{
			Workspace.ImportPMVSWorkspace(workspace, pmvsOptionName);
		}

		depthRanges = workspace.GetModel().ComputeDepthRanges();
	}

	/// <summary>
	/// Reads patch-match.cfg into <see cref="Problems"/>; a reference image left without
	/// source images is ignored. Port of PatchMatchController::ReadProblems.
	/// </summary>
	internal void ReadProblems()
	{
		Problems.Clear();

		Model model = Workspace.GetModel();

		string path = configPath.Length == 0
			? Path.Combine(workspacePath, Workspace.GetOptions().StereoFolder, "patch-match.cfg")
			: configPath;
		List<string> config = Workspace.ReadTextFileLines(path);

		List<SortedDictionary<int, int>>? sharedNumPoints = null;
		List<SortedDictionary<int, float>>? triangulationAngles = null;

		float minTriangulationAngleRad = (float)MathUtils.DegToRad(options.MinTriangulationAngle);

		string refImageName = "";
		var problemConfigs = new List<(string RefImageName, List<string> SrcImageNames)>();

		foreach (string rawLine in config)
		{
			string configLine = CppLineTokens.Trim(rawLine);

			if (configLine.Length == 0 || configLine[0] == '#')
			{
				continue;
			}

			if (refImageName.Length == 0)
			{
				refImageName = configLine;
				continue;
			}

			// COLMAP collects the reference indices into a set it never reads; the lookup
			// still throws for an unknown reference image before any problem is built.
			model.GetImageIdx(refImageName);

			problemConfigs.Add((refImageName, Misc.CsvToStringVector(configLine)));

			refImageName = "";
		}

		foreach ((string problemRefImageName, List<string> srcImageNames) in problemConfigs)
		{
			var problem = new PatchMatch.Problem { RefImageIdx = model.GetImageIdx(problemRefImageName) };

			if (srcImageNames.Count == 1 && srcImageNames[0] == "__all__")
			{
				// Use all images as source images.
				for (int imageIdx = 0; imageIdx < model.Images.Count; ++imageIdx)
				{
					if (imageIdx != problem.RefImageIdx)
					{
						problem.SrcImageIdxs.Add(imageIdx);
					}
				}
			}
			else if (srcImageNames.Count == 2 && srcImageNames[0] == "__auto__")
			{
				// Use maximum number of overlapping images as source images. Overlapping
				// will be sorted based on the number of shared points to the reference
				// image and the top ranked images are selected. Note that images are only
				// selected if some points have a sufficient triangulation angle.
				sharedNumPoints ??= model.ComputeSharedPoints();
				if (triangulationAngles == null)
				{
					const float TriangulationAnglePercentile = 75;
					triangulationAngles = model.ComputeTriangulationAngles(TriangulationAnglePercentile);
				}

				// std::stoll, then converted to size_t: a negative count wraps to a huge one.
				ulong maxNumSrcImages = unchecked((ulong)ParseStoll(srcImageNames[1]));

				SortedDictionary<int, int> overlappingImages = sharedNumPoints[problem.RefImageIdx];
				SortedDictionary<int, float> overlappingTriangulationAngles = triangulationAngles[problem.RefImageIdx];

				var srcImages = new List<(int ImageIdx, int NumShared)>(overlappingImages.Count);
				foreach ((int imageIdx, int numShared) in overlappingImages)
				{
					if (overlappingTriangulationAngles[imageIdx] >= minTriangulationAngleRad)
					{
						srcImages.Add((imageIdx, numShared));
					}
				}

				int effMaxNumSrcImages = (int)Math.Min((ulong)srcImages.Count, maxNumSrcImages);

				// COLMAP partial_sorts by shared count only; ties go to the lower image index
				// (divergence 84). The order is then total, so a full sort gives the same
				// prefix as partial_sort.
				srcImages.Sort(static (a, b) => a.NumShared != b.NumShared
					? b.NumShared.CompareTo(a.NumShared)
					: a.ImageIdx.CompareTo(b.ImageIdx));

				for (int i = 0; i < effMaxNumSrcImages; ++i)
				{
					problem.SrcImageIdxs.Add(srcImages[i].ImageIdx);
				}
			}
			else
			{
				foreach (string srcImageName in srcImageNames)
				{
					problem.SrcImageIdxs.Add(model.GetImageIdx(srcImageName));
				}
			}

			if (problem.SrcImageIdxs.Count == 0)
			{
				Log.Warning($"Ignoring reference image {problemRefImageName}, because it has no source images.");
			}
			else
			{
				Problems.Add(problem);
			}
		}
	}

	/// <summary>
	/// The effective options and the loaded problem for <see cref="Problems"/>[problemIdx]
	/// under <paramref name="problemOptions"/>: the depth range from the sparse model when
	/// not set, sigma_spatial defaulting to the window radius, filter_min_num_consistent
	/// capped by the number of source images, and the bitmaps (plus, with geometric
	/// consistency, the depth and normal maps) of the used images read from the workspace.
	/// With AllowMissingFiles, a source image without its files is dropped, from the
	/// returned problem and from <see cref="Problems"/> (as COLMAP updates its stored
	/// problem). The first half of PatchMatchController::ProcessProblem.
	/// </summary>
	internal (PatchMatchOptions Options, PatchMatch.Problem Problem) SetUpProblem(PatchMatchOptions problemOptions, int problemIdx)
	{
		Model model = Workspace.GetModel();
		PatchMatch.Problem problem = Problems[problemIdx].Clone();

		PatchMatchOptions patchMatchOptions = problemOptions.Clone();

		if (patchMatchOptions.DepthMin < 0 || patchMatchOptions.DepthMax < 0)
		{
			patchMatchOptions.DepthMin = depthRanges[problem.RefImageIdx].Min;
			patchMatchOptions.DepthMax = depthRanges[problem.RefImageIdx].Max;
			Check.That(
				patchMatchOptions.DepthMin > 0 && patchMatchOptions.DepthMax > 0,
				" - You must manually set the minimum and maximum depth, since no sparse model is provided in the workspace.");
		}

		if (patchMatchOptions.SigmaSpatial <= 0.0f)
		{
			patchMatchOptions.SigmaSpatial = patchMatchOptions.WindowRadius;
		}

		var images = new List<Image>(model.Images.Count);
		foreach (Image image in model.Images)
		{
			images.Add(image.Clone());
		}

		var depthMaps = new List<DepthMap>();
		var normalMaps = new List<NormalMap>();
		if (problemOptions.GeomConsistency)
		{
			for (int i = 0; i < model.Images.Count; i++)
			{
				depthMaps.Add(new DepthMap());
				normalMaps.Add(new NormalMap());
			}
		}

		problem.Images = images;
		problem.DepthMaps = depthMaps;
		problem.NormalMaps = normalMaps;

		// Collect all used images in current problem: the reference image, then the source
		// images in order, each once (divergence 85).
		var usedImageIdxs = new List<int> { problem.RefImageIdx };
		foreach (int srcImageIdx in problem.SrcImageIdxs)
		{
			if (!usedImageIdxs.Contains(srcImageIdx))
			{
				usedImageIdxs.Add(srcImageIdx);
			}
		}

		patchMatchOptions.FilterMinNumConsistent = Math.Min(usedImageIdxs.Count - 1, patchMatchOptions.FilterMinNumConsistent);

		lock (workspaceLock)
		{
			// Filter by file existence first (fast, no heavy I/O).
			var srcImageIdxs = new List<int>();
			var validImageIdxs = new List<int>();
			foreach (int imageIdx in usedImageIdxs)
			{
				bool missing = !Workspace.HasBitmap(imageIdx)
					|| (problemOptions.GeomConsistency && !Workspace.HasDepthMap(imageIdx))
					|| (problemOptions.GeomConsistency && !Workspace.HasNormalMap(imageIdx));

				if (missing)
				{
					string name = model.GetImageName(imageIdx);
					if (problemOptions.AllowMissingFiles)
					{
						Log.Warning(string.Create(
							CultureInfo.InvariantCulture,
							$"Skipping source image {imageIdx}: {name} for missing image or depth/normal map"));
						continue;
					}

					// COLMAP only logs here; reading the missing input below then fails.
					Log.Error(string.Create(
						CultureInfo.InvariantCulture, $"Missing image or map dependency for image {imageIdx}: {name}"));
				}

				validImageIdxs.Add(imageIdx);
				if (imageIdx != problem.RefImageIdx)
				{
					srcImageIdxs.Add(imageIdx);
				}
			}

			// Read images in parallel; each task writes only its own image's slots.
			var parallelOptions = new ParallelOptions
			{
				MaxDegreeOfParallelism = Threading.GetEffectiveNumThreads(options.NumThreads),
			};
			Parallel.ForEach(validImageIdxs, parallelOptions, imageIdx =>
			{
				images[imageIdx].SetBitmap(Workspace.GetBitmap(imageIdx));
				if (problemOptions.GeomConsistency)
				{
					depthMaps[imageIdx] = Workspace.GetDepthMap(imageIdx);
					normalMaps[imageIdx] = Workspace.GetNormalMap(imageIdx);
				}
			});

			// COLMAP updates the controller's own problem (it holds a reference), so a later
			// pass (the geometric one) starts from the sources that had their files.
			problem.SrcImageIdxs = srcImageIdxs;
			Problems[problemIdx].SrcImageIdxs = new List<int>(srcImageIdxs);
		}

		return (patchMatchOptions, problem);
	}

	/// <summary>
	/// std::stoll: leading white space, an optional sign, then the leading digits (trailing
	/// characters are ignored); no digits throws, like std::invalid_argument, and a value
	/// outside long throws, like std::out_of_range.
	/// </summary>
	internal static long ParseStoll(string text)
	{
		ReadOnlySpan<char> s = text.AsSpan().TrimStart(CppLineTokens.CppWhitespace);
		int start = s.Length > 0 && (s[0] == '-' || s[0] == '+') ? 1 : 0;
		int end = start;
		while (end < s.Length && char.IsAsciiDigit(s[end]))
		{
			end++;
		}

		if (end == start)
		{
			throw new FormatException($"stoll: no conversion for \"{text}\"");
		}

		if (!long.TryParse(s[..end], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long value))
		{
			throw new OverflowException($"stoll: \"{text}\" is out of range");
		}

		return value;
	}
}
