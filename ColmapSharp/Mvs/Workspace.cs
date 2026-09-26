// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Workspace: colmap/mvs/workspace.h and workspace.cc - the MVS model plus per-image bitmaps,
// depth maps and normal maps, either all loaded up front (Workspace.Load) or loaded on
// demand under a memory budget (CachedWorkspace, over Util/Cache.cs
// MemoryConstrainedLRUCache). Fusion and PatchMatch read their inputs through it. This file
// also holds ImportPMVSWorkspace. Tests: ColmapSharp.Tests/Mvs/WorkspaceTests.cs
// (workspace_test.cc 1:1).
//
// Tier A (exact) bookkeeping; rescaling goes through Bitmap.Rescale (Tier B, divergence 9)
// and DepthMap/NormalMap.Downsize (exact).
//
// Translation notes:
// - Image files are decoded by the host: the workspace reads bitmaps through an
//   IBitmapSource (Bitmap::Read in COLMAP), and HasBitmap asks the source instead of the
//   file system. Depth and normal maps are COLMAP .bin files under
//   <workspace>/<stereo folder>/{depth,normal}_maps, as in COLMAP.
// - The model can be given in memory (a Model built with Model.ReadFromCOLMAP from a
//   Reconstruction) instead of being read from options.WorkspacePath. The workspace
//   deep-copies it (Model.Clone), because MaxImageSize downsizes the workspace's images
//   and C++ holds its Model by value.
// - COLMAP's ThreadPool in Load becomes Parallel.ForEach; each task writes only its own
//   image's slots. The LOG(INFO)/LOG(WARNING) lines and the timer are not ported (the
//   library does not log yet; PORTING_PLAN.md Phase 4).
// - CachedWorkspace keeps COLMAP's locking: a cache lock around cache access and a lock per
//   cached image around its lazy loads.

using System.Globalization;

using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>Port of colmap::mvs::Workspace: the model and per-image MVS data, loaded up front.</summary>
public class Workspace
{
	/// <summary>Port of colmap::mvs::Workspace::Options.</summary>
	public sealed class Options
	{
		/// <summary>The maximum cache size in gigabytes.</summary>
		public double CacheSize { get; set; } = 32.0;

		/// <summary>The number of threads to use when pre-loading workspace.</summary>
		public int NumThreads { get; set; } = -1;

		/// <summary>Maximum image size in either dimension.</summary>
		public int MaxImageSize { get; set; } = -1;

		/// <summary>Whether to read image as RGB or gray scale.</summary>
		public bool ImageAsRgb { get; set; } = true;

		/// <summary>Location of the workspace.</summary>
		public string WorkspacePath { get; set; } = "";

		/// <summary>Type of the workspace ("COLMAP" or "PMVS").</summary>
		public string WorkspaceFormat { get; set; } = "";

		/// <summary>The kind of depth/normal maps to read ("photometric" or "geometric").</summary>
		public string InputType { get; set; } = "";

		/// <summary>The stereo folder under the workspace.</summary>
		public string StereoFolder { get; set; } = "stereo";

		/// <summary>A copy of these options.</summary>
		public Options Clone() => (Options)MemberwiseClone();
	}

	/// <summary>The options (input type lower-cased).</summary>
	protected readonly Options options;

	/// <summary>The sparse model.</summary>
	protected readonly Model model;

	/// <summary>Where bitmaps come from.</summary>
	protected readonly IBitmapSource bitmapSource;

	private readonly string depthMapPath;
	private readonly string normalMapPath;
	private Bitmap?[] bitmaps = [];
	private DepthMap?[] depthMaps = [];
	private NormalMap?[] normalMaps = [];

	/// <summary>
	/// A workspace whose model is read from options.WorkspacePath in options.WorkspaceFormat,
	/// with bitmaps from <paramref name="bitmapSource"/>.
	/// </summary>
	public Workspace(Options options, IBitmapSource bitmapSource)
		: this(ReadModel(options, bitmapSource), options, bitmapSource)
	{
	}

	/// <summary>
	/// A workspace over a model already in memory, with bitmaps from
	/// <paramref name="bitmapSource"/>. The workspace works on its own deep copy of
	/// <paramref name="model"/> (MaxImageSize downsizes the copy's images), so the caller's
	/// model is never changed, as when C++ copies the Model into the workspace.
	/// </summary>
	public Workspace(Options options, Model model, IBitmapSource bitmapSource)
		: this(Check.NotNull(model).Clone(), options, bitmapSource)
	{
	}

	/// <summary>The shared constructor; <paramref name="ownedModel"/> must not be held by anyone else.</summary>
	private Workspace(Model ownedModel, Options options, IBitmapSource bitmapSource)
	{
		this.options = options.Clone();
		this.options.InputType = this.options.InputType.ToLowerInvariant();
		model = ownedModel;
		this.bitmapSource = Check.NotNull(bitmapSource);
		if (this.options.MaxImageSize > 0)
		{
			foreach (Image image in this.model.Images)
			{
				image.Downsize(this.options.MaxImageSize, this.options.MaxImageSize);
			}
		}

		depthMapPath = Path.Combine(this.options.WorkspacePath, this.options.StereoFolder, "depth_maps");
		normalMapPath = Path.Combine(this.options.WorkspacePath, this.options.StereoFolder, "normal_maps");
	}

	private static Model ReadModel(Options options, IBitmapSource bitmapSource)
	{
		var model = new Model();
		model.Read(options.WorkspacePath, options.WorkspaceFormat, bitmapSource);
		return model;
	}

	/// <summary>The options.</summary>
	public Options GetOptions() => options;

	/// <summary>The sparse model.</summary>
	public Model GetModel() => model;

	/// <summary>
	/// Loads the bitmap, depth map and normal map of the named images (resized to the model's
	/// image sizes); images whose bitmap or depth map does not exist are skipped. Does
	/// nothing in a CachedWorkspace, which loads on demand.
	/// </summary>
	public virtual void Load(IReadOnlyList<string> imageNames)
	{
		int numImages = model.Images.Count;
		Array.Resize(ref bitmaps, numImages);
		Array.Resize(ref depthMaps, numImages);
		Array.Resize(ref normalMaps, numImages);

		var imageIdxs = new List<int>(imageNames.Count);
		foreach (string imageName in imageNames)
		{
			int imageIdx = model.GetImageIdx(imageName);
			if (HasBitmap(imageIdx) && HasDepthMap(imageIdx))
			{
				imageIdxs.Add(imageIdx);
			}
		}

		var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Threading.GetEffectiveNumThreads(options.NumThreads) };
		Parallel.ForEach(imageIdxs, parallelOptions, LoadWorkspaceData);
	}

	private void LoadWorkspaceData(int imageIdx)
	{
		int width = model.Images[imageIdx].GetWidth();
		int height = model.Images[imageIdx].GetHeight();

		// Read and rescale bitmap
		Bitmap bitmap = bitmapSource.Read(GetBitmapPath(imageIdx), options.ImageAsRgb);
		if (bitmap.Width != width || bitmap.Height != height)
		{
			bitmap.Rescale(width, height);
		}

		bitmaps[imageIdx] = bitmap;

		// Read and rescale depth map
		var depthMap = new DepthMap();
		depthMap.Read(GetDepthMapPath(imageIdx));
		if (depthMap.GetWidth() != width || depthMap.GetHeight() != height)
		{
			depthMap.Downsize(width, height);
		}

		depthMaps[imageIdx] = depthMap;

		// Read and rescale normal map
		var normalMap = new NormalMap();
		normalMap.Read(GetNormalMapPath(imageIdx));
		if (normalMap.GetWidth() != width || normalMap.GetHeight() != height)
		{
			normalMap.Downsize(width, height);
		}

		normalMaps[imageIdx] = normalMap;
	}

	/// <summary>The bitmap of an image (loaded by Load).</summary>
	public virtual Bitmap GetBitmap(int imageIdx) => Check.NotNull(bitmaps[imageIdx]);

	/// <summary>The depth map of an image (loaded by Load).</summary>
	public virtual DepthMap GetDepthMap(int imageIdx) => Check.NotNull(depthMaps[imageIdx]);

	/// <summary>The normal map of an image (loaded by Load).</summary>
	public virtual NormalMap GetNormalMap(int imageIdx) => Check.NotNull(normalMaps[imageIdx]);

	/// <summary>The path (bitmap source key) of an image's bitmap.</summary>
	public string GetBitmapPath(int imageIdx) => model.Images[imageIdx].GetPath();

	/// <summary>The path of an image's depth map file.</summary>
	public string GetDepthMapPath(int imageIdx) => Path.Combine(depthMapPath, GetFileName(imageIdx));

	/// <summary>The path of an image's normal map file.</summary>
	public string GetNormalMapPath(int imageIdx) => Path.Combine(normalMapPath, GetFileName(imageIdx));

	/// <summary>Whether the bitmap source has the image's bitmap.</summary>
	public bool HasBitmap(int imageIdx) => bitmapSource.Exists(GetBitmapPath(imageIdx));

	/// <summary>Whether the image's depth map file exists.</summary>
	public bool HasDepthMap(int imageIdx) => File.Exists(GetDepthMapPath(imageIdx));

	/// <summary>Whether the image's normal map file exists.</summary>
	public bool HasNormalMap(int imageIdx) => File.Exists(GetNormalMapPath(imageIdx));

	/// <summary>"&lt;image name&gt;.&lt;input type&gt;.bin".</summary>
	protected string GetFileName(int imageIdx) =>
		string.Create(CultureInfo.InvariantCulture, $"{model.GetImageName(imageIdx)}.{options.InputType}.bin");

	/// <summary>
	/// Imports a PMVS workspace into the COLMAP workspace format: creates the stereo folders
	/// and writes patch-match.cfg and fusion.cfg for the images listed on the "timages"
	/// lines of the PMVS option file <paramref name="optionName"/>.
	/// Port of colmap::mvs::ImportPMVSWorkspace.
	/// </summary>
	public static void ImportPMVSWorkspace(Workspace workspace, string optionName)
	{
		string workspacePath = workspace.GetOptions().WorkspacePath;
		string stereoFolder = workspace.GetOptions().StereoFolder;

		Directory.CreateDirectory(Path.Combine(workspacePath, stereoFolder));
		Directory.CreateDirectory(Path.Combine(workspacePath, stereoFolder, "depth_maps"));
		Directory.CreateDirectory(Path.Combine(workspacePath, stereoFolder, "normal_maps"));
		Directory.CreateDirectory(Path.Combine(workspacePath, stereoFolder, "consistency_graphs"));

		foreach (string line in ReadTextFileLines(Path.Combine(workspacePath, optionName)))
		{
			if (!line.StartsWith("timages", StringComparison.Ordinal))
			{
				continue;
			}

			// StringSplit compresses repeated delimiters; the line is already trimmed.
			string[] elems = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
			int numImages = ParseStoull(elems[1]);

			var imageIdxs = new List<int>();
			if (numImages == -1)
			{
				Check.Eq(elems.Length, 4);
				int rangeLower = ParseStoull(elems[2]);
				int rangeUpper = ParseStoull(elems[3]);
				Check.Lt(rangeLower, rangeUpper);
				numImages = rangeUpper - rangeLower;
				for (int i = 0; i < numImages; i++)
				{
					imageIdxs.Add(rangeLower + i);
				}
			}
			else
			{
				Check.Eq(numImages + 2, elems.Length);
				for (int i = 2; i < elems.Length; ++i)
				{
					imageIdxs.Add(ParseStoull(elems[i]));
				}
			}

			var imageNames = new List<string>(imageIdxs.Count);
			foreach (int imageIdx in imageIdxs)
			{
				imageNames.Add(workspace.GetModel().GetImageName(imageIdx));
			}

			List<List<int>> overlappingImages = workspace.GetModel().GetMaxOverlappingImagesFromPMVS();

			string patchMatchPath = Path.Combine(workspacePath, stereoFolder, "patch-match.cfg");
			string fusionPath = Path.Combine(workspacePath, stereoFolder, "fusion.cfg");
			using var patchMatchFile = new StreamWriter(FileOpen.OpenWrite(patchMatchPath));
			using var fusionFile = new StreamWriter(FileOpen.OpenWrite(fusionPath));
			for (int i = 0; i < imageNames.Count; ++i)
			{
				string refImageName = imageNames[i];
				patchMatchFile.Write(refImageName + "\n");
				if (overlappingImages.Count == 0)
				{
					patchMatchFile.Write("__auto__, 20\n");
				}
				else
				{
					foreach (int imageIdx in overlappingImages[i])
					{
						patchMatchFile.Write(workspace.GetModel().GetImageName(imageIdx) + ", ");
					}

					patchMatchFile.Write("\n");
				}

				fusionFile.Write(refImageName + "\n");
			}
		}
	}

	/// <summary>
	/// std::stoull narrowed to int as COLMAP assigns it: leading whitespace and an optional
	/// sign, then the leading digits (trailing characters are ignored); "-1" parses as
	/// 2^64 - 1, which becomes -1 again. No digits throws, like std::invalid_argument.
	/// </summary>
	private static int ParseStoull(string text)
	{
		ReadOnlySpan<char> s = text.AsSpan().TrimStart(CppLineTokens.CppWhitespace);
		bool negative = s.Length > 0 && s[0] == '-';
		if (s.Length > 0 && (s[0] == '-' || s[0] == '+'))
		{
			s = s[1..];
		}

		int numDigits = 0;
		while (numDigits < s.Length && char.IsAsciiDigit(s[numDigits]))
		{
			numDigits++;
		}

		if (numDigits == 0)
		{
			throw new FormatException($"stoull: no conversion for \"{text}\"");
		}

		ulong magnitude = ulong.Parse(s[..numDigits], NumberStyles.None, CultureInfo.InvariantCulture);
		ulong value = negative ? unchecked(0 - magnitude) : magnitude;
		return unchecked((int)value);
	}

	/// <summary>COLMAP's ReadTextFileLines: std::getline lines, StringTrim'd, empty lines skipped.</summary>
	private static List<string> ReadTextFileLines(string path)
	{
		var lines = new List<string>();
		foreach (string rawLine in File.ReadAllText(path).Split('\n'))
		{
			string line = CppLineTokens.Trim(rawLine);
			if (line.Length > 0)
			{
				lines.Add(line);
			}
		}

		return lines;
	}
}
