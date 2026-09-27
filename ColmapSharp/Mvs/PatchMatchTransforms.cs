// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatchTransforms: PatchMatchCuda::InitTransforms (patch_match_cuda.cu) - the
// reference calibration and the per-source pose tables for each of the four 90-degree
// rotations PatchMatch sweeps the reference image in. COLMAP keeps the calibration in
// __constant__ memory (ref_K, ref_inv_K) and the pose tables in 2D textures; the CPU port
// keeps them as flat float arrays the kernel indexes directly. Also here: the map from a
// pixel of a rotated frame back to the original reference image, which keys
// PatchMatchRandom's streams. Neighbors: PatchMatchTextures.cs (source image layers),
// PatchMatchRefImage.cs (reference image). Tests:
// ColmapSharp.Tests/Mvs/PatchMatchInputsTests.cs (C#-only).
//
// Tier A: the same float operations in the same order as COLMAP, through MvsGeometry
// (Image.cs) for the pose algebra.

namespace ColmapSharp.Mvs;

/// <summary>
/// The rotated reference calibrations and per-source pose tables of a PatchMatch problem.
/// Port of PatchMatchCuda::InitTransforms.
/// </summary>
public sealed class PatchMatchTransforms
{
	/// <summary>Offset of the source calibration {fx, cx, fy, cy} in a pose row.</summary>
	public const int KOffset = 0;

	/// <summary>Offset of the row-major relative rotation (reference to source).</summary>
	public const int ROffset = 4;

	/// <summary>Offset of the relative translation.</summary>
	public const int TOffset = 13;

	/// <summary>Offset of the source projection center in the reference frame.</summary>
	public const int COffset = 16;

	/// <summary>Offset of the 3x4 row-major projection matrix K [R | T].</summary>
	public const int POffset = 19;

	/// <summary>Offset of the 3x4 row-major inverse projection matrix.</summary>
	public const int InvPOffset = 31;

	/// <summary>Floats per source image in a pose table (kNumTformParams).</summary>
	public const int NumTformParams = 4 + 9 + 3 + 3 + 12 + 12;

	// Matrix for 90deg rotation around Z-axis in counter-clockwise direction.
	private static readonly float[] RZ90 = [0, 1, 0, -1, 0, 0, 0, 0, 1];

	private readonly float[][] refK = new float[4][];
	private readonly float[][] refInvK = new float[4][];
	private readonly float[][] poses = new float[4][];

	/// <summary>
	/// The transforms for reference image <paramref name="refImageIdx"/> and source images
	/// <paramref name="srcImageIdxs"/> of <paramref name="images"/>.
	/// </summary>
	public PatchMatchTransforms(IReadOnlyList<Image> images, int refImageIdx, IReadOnlyList<int> srcImageIdxs)
	{
		Image refImage = images[refImageIdx];
		RefWidth = refImage.GetWidth();
		RefHeight = refImage.GetHeight();
		NumSrcImages = srcImageIdxs.Count;

		// Calibration as {fx, cx, fy, cy} per rotation.
		ReadOnlySpan<float> k = refImage.GetK();
		for (int i = 0; i < 4; ++i)
		{
			refK[i] = [k[0], k[2], k[4], k[5]];
		}

		// Rotated by 90 degrees.
		(refK[1][0], refK[1][2]) = (refK[1][2], refK[1][0]);
		(refK[1][1], refK[1][3]) = (refK[1][3], refK[1][1]);
		refK[1][3] = (RefWidth - 1) - refK[1][3];

		// Rotated by 180 degrees.
		refK[2][1] = (RefWidth - 1) - refK[2][1];
		refK[2][3] = (RefHeight - 1) - refK[2][3];

		// Rotated by 270 degrees.
		(refK[3][0], refK[3][2]) = (refK[3][2], refK[3][0]);
		(refK[3][1], refK[3][3]) = (refK[3][3], refK[3][1]);
		refK[3][1] = (RefHeight - 1) - refK[3][1];

		// Extract 1/fx, -cx/fx, 1/fy, -cy/fy.
		for (int i = 0; i < 4; ++i)
		{
			refInvK[i] =
			[
				1.0f / refK[i][0],
				-refK[i][1] / refK[i][0],
				1.0f / refK[i][2],
				-refK[i][3] / refK[i][2],
			];
		}

		// Generate rotated versions of camera poses.
		float[] rotatedR = refImage.GetR().ToArray();
		float[] rotatedT = refImage.GetT().ToArray();
		Span<float> relR = stackalloc float[9];
		Span<float> relT = stackalloc float[3];
		for (int i = 0; i < 4; ++i)
		{
			float[] table = new float[NumTformParams * NumSrcImages];
			int offset = 0;
			foreach (int imageIdx in srcImageIdxs)
			{
				Image image = images[imageIdx];
				ReadOnlySpan<float> imageK = image.GetK();
				table[offset + 0] = imageK[0];
				table[offset + 1] = imageK[2];
				table[offset + 2] = imageK[4];
				table[offset + 3] = imageK[5];
				Span<float> row = table.AsSpan(offset, NumTformParams);

				MvsGeometry.ComputeRelativePose(rotatedR, rotatedT, image.GetR(), image.GetT(), relR, relT);
				relR.CopyTo(row[ROffset..]);
				relT.CopyTo(row[TOffset..]);
				MvsGeometry.ComputeProjectionCenter(relR, relT, row.Slice(COffset, 3));
				MvsGeometry.ComposeProjectionMatrix(imageK, relR, relT, row.Slice(POffset, 12));
				MvsGeometry.ComposeInverseProjectionMatrix(imageK, relR, relT, row.Slice(InvPOffset, 12));
				offset += NumTformParams;
			}

			poses[i] = table;
			MvsGeometry.RotatePose(RZ90, rotatedR, rotatedT);
		}
	}

	// The transforms of a GPU pose table (FromPoseTable): every value already computed.
	private PatchMatchTransforms(int refWidth, int refHeight, int numSrcImages)
	{
		RefWidth = refWidth;
		RefHeight = refHeight;
		NumSrcImages = numSrcImages;
	}

	/// <summary>
	/// The transforms held by a GPU PatchMatch pose table (patch_match_geometry.wgsl's contract:
	/// floats [8 r, 8 r + 4) are RefK(r), [8 r + 4, 8 r + 8) RefInvK(r), and Poses(r) starts at
	/// float 32 + NumTformParams r S), for a <paramref name="refWidth"/> x
	/// <paramref name="refHeight"/> reference image and <paramref name="numSrcImages"/> sources.
	/// Lets Mvs/Testing/ReferenceComputeDevice run the CPU kernels on the table the GPU reads.
	/// </summary>
	internal static PatchMatchTransforms FromPoseTable(ReadOnlySpan<float> table, int refWidth, int refHeight, int numSrcImages)
	{
		int tableSize = 32 + 4 * NumTformParams * numSrcImages;
		if (table.Length < tableSize)
		{
			throw new ArgumentException($"A pose table for {numSrcImages} source images needs {tableSize} floats, but has {table.Length}.", nameof(table));
		}

		var transforms = new PatchMatchTransforms(refWidth, refHeight, numSrcImages);
		for (int r = 0; r < 4; ++r)
		{
			transforms.refK[r] = table.Slice(8 * r, 4).ToArray();
			transforms.refInvK[r] = table.Slice(8 * r + 4, 4).ToArray();
			transforms.poses[r] = table.Slice(32 + NumTformParams * r * numSrcImages, NumTformParams * numSrcImages).ToArray();
		}

		return transforms;
	}

	/// <summary>Width of the unrotated reference image.</summary>
	public int RefWidth { get; }

	/// <summary>Height of the unrotated reference image.</summary>
	public int RefHeight { get; }

	/// <summary>Number of source images (rows of each pose table).</summary>
	public int NumSrcImages { get; }

	/// <summary>
	/// The reference calibration {fx, cx, fy, cy} for <paramref name="rotation"/> (0..3,
	/// in units of 90 degrees counter-clockwise), COLMAP's ref_K.
	/// </summary>
	public ReadOnlySpan<float> RefK(int rotation) => refK[rotation];

	/// <summary>
	/// The inverse reference calibration {1/fx, -cx/fx, 1/fy, -cy/fy} for
	/// <paramref name="rotation"/>, COLMAP's ref_inv_K.
	/// </summary>
	public ReadOnlySpan<float> RefInvK(int rotation) => refInvK[rotation];

	/// <summary>
	/// The pose table for <paramref name="rotation"/>: <see cref="NumTformParams"/> floats per
	/// source image, laid out as the *Offset constants (COLMAP's poses_texture_[rotation]).
	/// </summary>
	public float[] Poses(int rotation) => poses[rotation];

	/// <summary>
	/// The pixel of the original reference image (refWidth x refHeight) that pixel
	/// (<paramref name="row"/>, <paramref name="col"/>) of the frame rotated
	/// <paramref name="rotation"/> times by Mat.Rotate came from.
	/// </summary>
	public static (int Row, int Col) ToOriginalPixel(int rotation, int row, int col, int refWidth, int refHeight)
	{
		// Mat.Rotate maps (y, x) of a frame of width w to (w - 1 - x, y); undo it once per
		// rotation. The frame before rotation k is refWidth wide when k - 1 is even.
		for (int k = rotation; k > 0; --k)
		{
			int prevWidth = (k - 1) % 2 == 0 ? refWidth : refHeight;
			(row, col) = (col, prevWidth - 1 - row);
		}

		return (row, col);
	}
}
