// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Fusion: colmap/mvs/fusion.h and fusion.cc - StereoFusion, which fuses the per-image depth
// and normal maps of a Workspace (Workspace.cs) into a colored, oriented point cloud
// (Util/PlyTypes.cs PlyPoint) with, per point, the indices of the images that saw it; plus
// WritePointsVisibility/ReadPointsVisibility for the .vis file next to fused.ply. Options
// are in StereoFusionOptions.cs. WriteFusedPly is the "ply" branch of COLMAP's
// RunStereoFuserImpl (exe/mvs.cc). Tests: ColmapSharp.Tests/Mvs/FusionTests.cs
// (fusion_test.cc 1:1, plus an oracle fixture against pycolmap.stereo_fusion).
//
// Tier C (outcome). The float products go through Mvs/Image.cs's projection matrices, whose
// inverse is not bit-identical to Eigen's (divergence 63), so positions agree with COLMAP to
// float rounding (under 1e-6 on the oracle fixtures), not bit for bit. Sums avoid FMA and follow
// Eigen's grouping as far as it could be established (Sum3/Sum4): P * X.homogeneous() is
// Eigen's 3x3 block product plus the last column; the 3-term reductions group as
// (a0 + a1) + a2, measured; the 4-term 3x4 * 4-vector reductions group pairwise, which is
// inferred from arm64's vectorized horizontal add and not certain (an x86 build of COLMAP
// may group differently).
//
// Translation notes:
// - COLMAP fuses the rows of an image in 10-row tasks on a thread pool, and the tasks race
//   on the shared fused-pixel masks, so its output depends on scheduling. Here the
//   traversal runs on one thread in row order, which is exactly COLMAP with num_threads = 1
//   (and COLMAP's use_cache path, which always runs one thread). NumThreads only sets the
//   workspace loading parallelism (docs/CPP_DIVERGENCES.md, entry 87).
// - A point's visibility is COLMAP's FlatHashSet (std::unordered_set) iterated into a
//   vector, in hash order; here it is in ascending image index (entry 88).
// - Images and masks are decoded by the host through IBitmapSource; masks are read as grey
//   with Read(path, asRgb: false), and "the mask file exists" is the source's Exists. A mask
//   that exists but cannot be decoded throws (entry 89).
// - CheckIfStopped becomes a CancellationToken checked where COLMAP checks: after loading
//   and before each image. As in COLMAP (BaseController), a stop is not an error: Run
//   returns normally with the points fused so far. Progress is reported after each image.
// - LOG(WARNING) goes to Util/Log.cs; LOG(INFO) lines and timers are not ported.

using System.Runtime.InteropServices;

using ColmapSharp.Mathematics;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>Port of colmap::mvs::StereoFusion: fuses per-view depth and normal maps into a consistent 3D point cloud.</summary>
public sealed class StereoFusion
{
	// Using a row stride of 10 to avoid starting parallel processing in rows that are too
	// close to each other which may lead to duplicated work, since nearby pixels are likely
	// to get fused into the same point. Kept because it fixes COLMAP's traversal order even
	// on one thread: rows are visited in blocks, which on one thread is plain row order.
	private const int RowStride = 10;

	private readonly StereoFusionOptions options;
	private readonly string workspacePath;
	private readonly string workspaceFormat;
	private readonly string pmvsOptionName;
	private readonly string inputType;
	private readonly IBitmapSource bitmapSource;
	private readonly float maxSquaredReprojError;
	private readonly float minCosNormalError;

	private Workspace? workspace;
	private bool[] usedImages = [];
	private bool[] fusedImages = [];
	private List<List<int>> overlappingImages = new();

	// Contains image masks of pre-masked and already fused pixels. Initialized from image
	// masks if provided in StereoFusionOptions.
	private Mat<byte>[] fusedPixelMasks = [];
	private (int Width, int Height)[] depthMapSizes = [];
	private (float X, float Y)[] bitmapScales = [];
	private float[][] projections = [];        // P_: 3x4 row-major
	private float[][] inverseProjections = []; // inv_P_: 3x4 row-major
	private float[][] inverseRotations = [];   // inv_R_: 3x3 row-major

	// Already fused points.
	private readonly List<PlyPoint> fusedPoints = new();
	private readonly List<List<int>> fusedPointsVisibility = new();

	// Per-call scratch of Fuse, reused so the per-pixel loop does not allocate.
	private readonly List<FusionData> fusionQueue = new();
	private readonly List<float> fusedPointX = new();
	private readonly List<float> fusedPointY = new();
	private readonly List<float> fusedPointZ = new();
	private readonly List<float> fusedPointNx = new();
	private readonly List<float> fusedPointNy = new();
	private readonly List<float> fusedPointNz = new();
	private readonly List<byte> fusedPointR = new();
	private readonly List<byte> fusedPointG = new();
	private readonly List<byte> fusedPointB = new();
	private readonly List<int> fusedPointVisibility = new();

	/// <summary>
	/// A fusion over the workspace at <paramref name="workspacePath"/> ("COLMAP" or "PMVS"
	/// format), reading "photometric" or "geometric" depth and normal maps, with bitmaps
	/// (and masks) from <paramref name="bitmapSource"/>.
	/// </summary>
	public StereoFusion(
		StereoFusionOptions options,
		string workspacePath,
		string workspaceFormat,
		string pmvsOptionName,
		string inputType,
		IBitmapSource bitmapSource)
	{
		this.options = Check.NotNull(options).Clone();
		this.workspacePath = workspacePath;
		this.workspaceFormat = workspaceFormat;
		this.pmvsOptionName = pmvsOptionName;
		this.inputType = inputType;
		this.bitmapSource = Check.NotNull(bitmapSource);
		maxSquaredReprojError = (float)(this.options.MaxReprojError * this.options.MaxReprojError);
		minCosNormalError = (float)Math.Cos(MathUtils.DegToRad(this.options.MaxNormalError));
		Check.That(this.options.Check());
	}

	/// <summary>The fused 3D points (position, unit normal, color).</summary>
	public IReadOnlyList<PlyPoint> GetFusedPoints() => fusedPoints;

	/// <summary>Per fused point, the indices (into the model's images) of the images that observe it.</summary>
	public IReadOnlyList<List<int>> GetFusedPointsVisibility() => fusedPointsVisibility;

	/// <summary>
	/// Runs the depth map fusion. Cancelling <paramref name="cancellationToken"/> stops
	/// before the next image, keeping the points fused so far (no exception, as COLMAP's
	/// stopped controllers). <paramref name="progress"/> is told after each image.
	/// </summary>
	public void Run(CancellationToken cancellationToken = default, IProgress<StereoFusionProgress>? progress = null)
	{
		fusedPoints.Clear();
		fusedPointsVisibility.Clear();

		var workspaceOptions = new Workspace.Options();
		if (workspaceFormat.ToLowerInvariant() == "pmvs")
		{
			workspaceOptions.StereoFolder = "stereo-" + pmvsOptionName;
		}

		workspaceOptions.NumThreads = options.NumThreads;
		workspaceOptions.MaxImageSize = options.MaxImageSize;
		workspaceOptions.ImageAsRgb = true;
		workspaceOptions.CacheSize = options.CacheSize;
		workspaceOptions.WorkspacePath = workspacePath;
		workspaceOptions.WorkspaceFormat = workspaceFormat;
		workspaceOptions.InputType = inputType;

		List<string> imageNames = Workspace.ReadTextFileLines(
			Path.Combine(workspacePath, workspaceOptions.StereoFolder, "fusion.cfg"));
		if (options.UseCache)
		{
			workspace = new CachedWorkspace(workspaceOptions, bitmapSource);
		}
		else
		{
			workspace = new Workspace(workspaceOptions, bitmapSource);
			workspace.Load(imageNames);
		}

		if (cancellationToken.IsCancellationRequested)
		{
			return;
		}

		Model model = workspace.GetModel();

		const double MinTriangulationAngle = 0;
		overlappingImages = model.GetMaxOverlappingImagesFromPMVS().Count == 0
			? model.GetMaxOverlappingImages(options.CheckNumImages, MinTriangulationAngle)
			: model.GetMaxOverlappingImagesFromPMVS();

		int numImages = model.Images.Count;
		usedImages = new bool[numImages];
		fusedImages = new bool[numImages];
		fusedPixelMasks = new Mat<byte>[numImages];
		depthMapSizes = new (int, int)[numImages];
		bitmapScales = new (float, float)[numImages];
		projections = new float[numImages][];
		inverseProjections = new float[numImages][];
		inverseRotations = new float[numImages][];

		Span<float> k = stackalloc float[9];
		foreach (string imageName in imageNames)
		{
			int imageIdx = model.GetImageIdx(imageName);

			if (!workspace.HasBitmap(imageIdx) || !workspace.HasDepthMap(imageIdx) || !workspace.HasNormalMap(imageIdx))
			{
				Log.Warning($"Ignoring image {imageName}, because input does not exist.");
				continue;
			}

			Image image = model.Images[imageIdx];
			DepthMap depthMap = workspace.GetDepthMap(imageIdx);

			usedImages[imageIdx] = true;

			InitFusedPixelMask(imageIdx, depthMap.GetWidth(), depthMap.GetHeight());

			depthMapSizes[imageIdx] = (depthMap.GetWidth(), depthMap.GetHeight());

			bitmapScales[imageIdx] = (
				(float)depthMap.GetWidth() / image.GetWidth(),
				(float)depthMap.GetHeight() / image.GetHeight());

			image.GetK().CopyTo(k);
			k[0] *= bitmapScales[imageIdx].X;
			k[2] *= bitmapScales[imageIdx].X;
			k[4] *= bitmapScales[imageIdx].Y;
			k[5] *= bitmapScales[imageIdx].Y;

			projections[imageIdx] = new float[12];
			inverseProjections[imageIdx] = new float[12];
			MvsGeometry.ComposeProjectionMatrix(k, image.GetR(), image.GetT(), projections[imageIdx]);
			MvsGeometry.ComposeInverseProjectionMatrix(k, image.GetR(), image.GetT(), inverseProjections[imageIdx]);
			ReadOnlySpan<float> r = image.GetR();
			inverseRotations[imageIdx] = [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
		}

		int numFusedImages = 0;
		for (int imageIdx = 0; imageIdx >= 0; imageIdx = FindNextImage(overlappingImages, usedImages, fusedImages, imageIdx))
		{
			if (cancellationToken.IsCancellationRequested)
			{
				break;
			}

			// COLMAP visits index 0 first even if it is unused; its depth map size is then
			// (0, 0), so nothing is fused from it.
			(int width, int height) = depthMapSizes[imageIdx];
			Mat<byte>? fusedPixelMask = fusedPixelMasks[imageIdx];

			for (int rowStart = 0; rowStart < height; rowStart += RowStride)
			{
				int rowEnd = Math.Min(height, rowStart + RowStride);
				for (int row = rowStart; row < rowEnd; ++row)
				{
					for (int col = 0; col < width; ++col)
					{
						if (fusedPixelMask!.Get(row, col) > 0)
						{
							continue;
						}

						Fuse(imageIdx, row, col);
					}
				}
			}

			numFusedImages += 1;
			fusedImages[imageIdx] = true;
			progress?.Report(new StereoFusionProgress(numFusedImages, numImages, fusedPoints.Count));
		}

		if (fusedPoints.Count == 0)
		{
			Log.Warning(
				"Could not fuse any points. This is likely caused by incorrect settings - filtering must be enabled for "
				+ "the last call to patch match stereo.");
		}
	}

	/// <summary>
	/// Writes the fused points as a binary PLY at <paramref name="path"/> and their
	/// visibility to <c>path + ".vis"</c>: the "ply" output of COLMAP's RunStereoFuserImpl.
	/// </summary>
	public void WriteFusedPly(string path)
	{
		Ply.WriteBinaryPlyPoints(path, fusedPoints);
		WritePointsVisibility(path + ".vis", fusedPointsVisibility);
	}

	/// <summary>
	/// Use the sparse model to find most connected image that has not yet been fused. This is
	/// used as a heuristic to ensure that the workspace cache reuses already cached images as
	/// efficient as possible. -1 once every used image is fused.
	/// Port of colmap::mvs::internal::FindNextImage.
	/// </summary>
	internal static int FindNextImage(List<List<int>> overlappingImages, bool[] usedImages, bool[] fusedImages, int prevImageIdx)
	{
		Check.Eq(usedImages.Length, fusedImages.Length);

		foreach (int imageIdx in overlappingImages[prevImageIdx])
		{
			if (usedImages[imageIdx] && !fusedImages[imageIdx])
			{
				return imageIdx;
			}
		}

		// If none of the overlapping images are not yet fused, simply return the first image
		// that has not yet been fused.
		for (int imageIdx = 0; imageIdx < fusedImages.Length; ++imageIdx)
		{
			if (usedImages[imageIdx] && !fusedImages[imageIdx])
			{
				return imageIdx;
			}
		}

		return -1;
	}

	private void InitFusedPixelMask(int imageIdx, int width, int height)
	{
		string maskImageName = workspace!.GetModel().GetImageName(imageIdx);
		string maskPath = Path.Combine(options.MaskPath, maskImageName + ".png");
		if (!bitmapSource.Exists(maskPath) && Path.GetExtension(maskImageName) == ".png")
		{
			maskPath = Path.Combine(options.MaskPath, maskImageName);
		}

		var fusedPixelMask = new Mat<byte>(width, height, 1);
		if (options.MaskPath.Length != 0 && bitmapSource.Exists(maskPath))
		{
			// An unreadable mask throws, where COLMAP silently fuses unmasked (divergence 89).
			Bitmap mask = bitmapSource.Read(maskPath, asRgb: false);
			mask.Rescale(width, height, Bitmap.RescaleFilter.Box);
			for (int row = 0; row < height; ++row)
			{
				for (int col = 0; col < width; ++col)
				{
					BitmapColor<byte>? color = mask.GetPixel(col, row);
					fusedPixelMask.Set(row, col, (byte)(color is not { } c || c.R == 0 ? 1 : 0));
				}
			}
		}
		else
		{
			fusedPixelMask.Fill(0);
		}

		fusedPixelMasks[imageIdx] = fusedPixelMask;
	}

	private void Fuse(int refImageIdx, int refRow, int refCol)
	{
		// Next points to fuse.
		fusionQueue.Clear();
		fusionQueue.Add(new FusionData(refImageIdx, refRow, refCol, 0));

		float refX = 0, refY = 0, refZ = 0;
		float refNx = 0, refNy = 0, refNz = 0;

		// Points of different pixels of the currently point to be fused.
		fusedPointX.Clear();
		fusedPointY.Clear();
		fusedPointZ.Clear();
		fusedPointNx.Clear();
		fusedPointNy.Clear();
		fusedPointNz.Clear();
		fusedPointR.Clear();
		fusedPointG.Clear();
		fusedPointB.Clear();
		fusedPointVisibility.Clear();

		Workspace ws = workspace!;
		while (fusionQueue.Count > 0)
		{
			FusionData data = fusionQueue[^1];
			int imageIdx = data.ImageIdx;
			int row = data.Row;
			int col = data.Col;
			int traversalDepth = data.TraversalDepth;

			fusionQueue.RemoveAt(fusionQueue.Count - 1);

			// Check if pixel already fused.
			Mat<byte> fusedPixelMask = fusedPixelMasks[imageIdx];
			if (fusedPixelMask.Get(row, col) > 0)
			{
				continue;
			}

			float depth = ws.GetDepthMap(imageIdx).Get(row, col);

			// Pixels with negative depth are filtered.
			if (depth <= 0.0f)
			{
				continue;
			}

			// If the traversal depth is greater than zero, the initial reference pixel has
			// already been added and we need to check for consistency.
			if (traversalDepth > 0)
			{
				// Project reference point into current view.
				float[] p = projections[imageIdx];
				float projX = Sum4(p[0] * refX, p[1] * refY, p[2] * refZ, p[3] * 1.0f);
				float projY = Sum4(p[4] * refX, p[5] * refY, p[6] * refZ, p[7] * 1.0f);
				float projZ = Sum4(p[8] * refX, p[9] * refY, p[10] * refZ, p[11] * 1.0f);

				// Depth error of reference depth with current depth.
				float depthError = Math.Abs((projZ - depth) / depth);
				if (depthError > options.MaxDepthError)
				{
					continue;
				}

				// Reprojection error reference point in the current view.
				float colDiff = projX / projZ - col;
				float rowDiff = projY / projZ - row;
				float squaredReprojError = colDiff * colDiff + rowDiff * rowDiff;
				if (squaredReprojError > maxSquaredReprojError)
				{
					continue;
				}
			}

			// Determine normal direction in global reference frame.
			NormalMap normalMap = ws.GetNormalMap(imageIdx);
			float n0 = normalMap.Get(row, col, 0);
			float n1 = normalMap.Get(row, col, 1);
			float n2 = normalMap.Get(row, col, 2);
			float[] invR = inverseRotations[imageIdx];
			float nx = Sum3(invR[0] * n0, invR[1] * n1, invR[2] * n2);
			float ny = Sum3(invR[3] * n0, invR[4] * n1, invR[5] * n2);
			float nz = Sum3(invR[6] * n0, invR[7] * n1, invR[8] * n2);

			// Check for consistent normal direction with reference normal.
			if (traversalDepth > 0)
			{
				float cosNormalError = Sum3(refNx * nx, refNy * ny, refNz * nz);
				if (cosNormalError < minCosNormalError)
				{
					continue;
				}
			}

			// Determine 3D location of current depth value.
			float[] invP = inverseProjections[imageIdx];
			float hx = col * depth;
			float hy = row * depth;
			float x = Sum4(invP[0] * hx, invP[1] * hy, invP[2] * depth, invP[3] * 1.0f);
			float y = Sum4(invP[4] * hx, invP[5] * hy, invP[6] * depth, invP[7] * 1.0f);
			float z = Sum4(invP[8] * hx, invP[9] * hy, invP[10] * depth, invP[11] * 1.0f);

			// Read the color of the pixel.
			(float scaleX, float scaleY) = bitmapScales[imageIdx];
			BitmapColor<byte> color = ws.GetBitmap(imageIdx)
				.InterpolateNearestNeighbor(col / scaleX, row / scaleY)
				?? new BitmapColor<byte>(0);

			// Set the current pixel as visited.
			fusedPixelMask.Set(row, col, 1);

			// Pixels out of bounds are filtered
			if (x < options.BoundingBoxMin.X || y < options.BoundingBoxMin.Y || z < options.BoundingBoxMin.Z
				|| x > options.BoundingBoxMax.X || y > options.BoundingBoxMax.Y || z > options.BoundingBoxMax.Z)
			{
				continue;
			}

			// Accumulate statistics for fused point.
			fusedPointX.Add(x);
			fusedPointY.Add(y);
			fusedPointZ.Add(z);
			fusedPointNx.Add(nx);
			fusedPointNy.Add(ny);
			fusedPointNz.Add(nz);
			fusedPointR.Add(color.R);
			fusedPointG.Add(color.G);
			fusedPointB.Add(color.B);
			if (!fusedPointVisibility.Contains(imageIdx))
			{
				fusedPointVisibility.Add(imageIdx);
			}

			// Remember the first pixel as the reference.
			if (traversalDepth == 0)
			{
				(refX, refY, refZ) = (x, y, z);
				(refNx, refNy, refNz) = (nx, ny, nz);
			}

			if (fusedPointX.Count >= options.MaxNumPixels)
			{
				break;
			}

			if (traversalDepth >= options.MaxTraversalDepth - 1)
			{
				continue;
			}

			foreach (int nextImageIdx in overlappingImages[imageIdx])
			{
				if (!usedImages[nextImageIdx] || fusedImages[nextImageIdx])
				{
					continue;
				}

				float[] np = projections[nextImageIdx];
				float nextX = Sum3(np[0] * x, np[1] * y, np[2] * z) + np[3];
				float nextY = Sum3(np[4] * x, np[5] * y, np[6] * z) + np[7];
				float nextZ = Sum3(np[8] * x, np[9] * y, np[10] * z) + np[11];
				// static_cast<int>(std::round(float)): .NET saturates out-of-range values and
				// converts NaN to 0, the same as arm64 C++ (fcvtzs); x86 C++ gives INT_MIN for
				// both. A NaN pixel (0, 0) passes the bounds check here as on arm64.
				int nextCol = (int)MathF.Round(nextX / nextZ, MidpointRounding.AwayFromZero);
				int nextRow = (int)MathF.Round(nextY / nextZ, MidpointRounding.AwayFromZero);

				(int nextWidth, int nextHeight) = depthMapSizes[nextImageIdx];
				if (nextCol < 0 || nextRow < 0 || nextCol >= nextWidth || nextRow >= nextHeight)
				{
					continue;
				}

				fusionQueue.Add(new FusionData(nextImageIdx, nextRow, nextCol, traversalDepth + 1));
			}
		}

		int numPixels = fusedPointX.Count;
		if (numPixels >= options.MinNumPixels)
		{
			float fusedNx = (float)MathUtils.Median<float>(CollectionsMarshal.AsSpan(fusedPointNx));
			float fusedNy = (float)MathUtils.Median<float>(CollectionsMarshal.AsSpan(fusedPointNy));
			float fusedNz = (float)MathUtils.Median<float>(CollectionsMarshal.AsSpan(fusedPointNz));
			float fusedNormalNorm = MathF.Sqrt(Sum3(fusedNx * fusedNx, fusedNy * fusedNy, fusedNz * fusedNz));
			// std::numeric_limits<float>::epsilon(), not float.Epsilon.
			const float FloatEpsilon = 1.1920929E-07f;
			if (fusedNormalNorm < FloatEpsilon)
			{
				return;
			}

			var fusedPoint = new PlyPoint
			{
				X = MedianFloat(fusedPointX),
				Y = MedianFloat(fusedPointY),
				Z = MedianFloat(fusedPointZ),
				Nx = fusedNx / fusedNormalNorm,
				Ny = fusedNy / fusedNormalNorm,
				Nz = fusedNz / fusedNormalNorm,
				R = MedianColor(fusedPointR),
				G = MedianColor(fusedPointG),
				B = MedianColor(fusedPointB),
			};

			fusedPoints.Add(fusedPoint);
			var visibility = new List<int>(fusedPointVisibility);
			visibility.Sort();
			fusedPointsVisibility.Add(visibility);
		}
	}

	// Eigen's reductions of 3-element float products (the 3x3 products, dot and squaredNorm).
	// Measured against the stereo_fusion_options oracle (per-pixel normals): (a0 + a1) + a2
	// reproduces pycolmap's normals far more often than a0 + (a1 + a2) or an FMA on the last
	// term (31 vs 134 vs 98 mismatched components of 387); see the file header.
	private static float Sum3(float a0, float a1, float a2) => (a0 + a1) + a2;

	// The 3x4 * 4-vector products. Inferred, not certain: pairwise, as a vectorized 4-lane
	// horizontal add (arm64 faddp) sums; this matched pycolmap best on both fixtures, but the
	// positions also go through the inverse projection (entry 63), so no fixture proves it.
	private static float Sum4(float a0, float a1, float a2, float a3) => (a0 + a1) + (a2 + a3);

	private static float MedianFloat(List<float> values) =>
		(float)MathUtils.Median<float>(CollectionsMarshal.AsSpan(values));

	// TruncateCast<float, uint8_t>(std::round(Median(values))): std::round on the double
	// median, then the implicit narrowing to float of TruncateCast's parameter.
	private static byte MedianColor(List<byte> values) =>
		MathUtils.TruncateCast<float, byte>((float)Math.Round(
			MathUtils.Median<byte>(CollectionsMarshal.AsSpan(values)),
			MidpointRounding.AwayFromZero));

	/// <summary>
	/// Writes the visibility information into a binary file of the following format:
	/// <c>&lt;num_points : uint64&gt;</c>, then per point
	/// <c>&lt;num_visible_images : uint32&gt;&lt;image_idx : uint32&gt;...</c>, all little-endian.
	/// Note that an image_idx does not correspond to the image_id of a Reconstruction, but the
	/// index of the image in the mvs::Model, which is the location of the image in the
	/// images.bin/.txt. Port of colmap::mvs::WritePointsVisibility.
	/// </summary>
	public static void WritePointsVisibility(string path, IReadOnlyList<IReadOnlyList<int>> pointsVisibility)
	{
		using FileStream file = FileOpen.OpenWrite(path);
		using var writer = new BinaryWriter(file);

		writer.Write((ulong)pointsVisibility.Count);

		foreach (IReadOnlyList<int> visibility in pointsVisibility)
		{
			writer.Write((uint)visibility.Count);
			foreach (int imageIdx in visibility)
			{
				writer.Write(unchecked((uint)imageIdx));
			}
		}
	}

	/// <summary>
	/// Reads per-point visibility data from a binary .vis file produced by the stereo fusion
	/// pipeline (the format of <see cref="WritePointsVisibility"/>). Throws if the file's
	/// point count is not <paramref name="numPoints"/>.
	/// Port of colmap::mvs::ReadPointsVisibility.
	/// </summary>
	public static List<List<int>> ReadPointsVisibility(string path, long numPoints)
	{
		using FileStream file = FileOpen.OpenRead(path);
		using var reader = new BinaryReader(file);

		ulong fileNumPoints = reader.ReadUInt64();
		Check.Eq(fileNumPoints, (ulong)numPoints);

		var visibility = new List<List<int>>((int)numPoints);
		for (long i = 0; i < numPoints; ++i)
		{
			uint numVisible = reader.ReadUInt32();
			var point = new List<int>((int)Math.Min(numVisible, 1024u));
			for (uint j = 0; j < numVisible; ++j)
			{
				point.Add(unchecked((int)reader.ReadUInt32()));
			}

			visibility.Add(point);
		}

		return visibility;
	}

	private readonly record struct FusionData(int ImageIdx, int Row, int Col, int TraversalDepth);
}
