// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchGpu: one PatchMatch problem run on a host-provided compute device (IComputeDevice,
// ColmapSharp/Compute/) instead of the CPU. Not a direct COLMAP port: it stands where
// PatchMatchCuda (patch_match_cuda.h/.cu) stands, and runs the kernels of Mvs/Shaders/ in the
// schedule PatchMatchCpu.cs runs its loops in (PORTING_PLAN.md Phase 13). This file builds the
// problem's inputs from the same objects PatchMatchCpu builds from - the filtered reference
// image, the source layers, the source depth maps, the rotated pose tables and the likelihood
// constants - packs every buffer and uniform in the layouts patch_match_layout.wgsl,
// patch_match_geometry.wgsl, patch_match_textures.wgsl and patch_match_rotate.wgsl document, and
// decodes the read-back maps into the result types PatchMatchCpu exposes.
// PatchMatchGpu.Run.cs creates the device resources and records the schedule;
// PatchMatchGpuPlan.cs sizes them; PatchMatchGpuKernels.cs names the kernels and their bindings.
// Tests: ColmapSharp.Tests/Mvs/PatchMatchGpuTests.cs (C#-only, on RecordingComputeDevice).
//
// Every per-problem float the kernels read is computed here with exactly the C# expression
// PatchMatchCpu uses (and stored as the same float), so the GPU kernels start from the CPU's
// constants; only the kernels' own arithmetic is Tier C.

using System.Buffers.Binary;
using System.Runtime.InteropServices;

using ColmapSharp.Compute;

namespace ColmapSharp.Mvs;

/// <summary>
/// PatchMatch on an <see cref="IComputeDevice"/>, with the getters of <see cref="PatchMatchCpu"/>.
/// Internal: PatchMatch checks the problem (map sizes included) before constructing it.
/// </summary>
internal sealed partial class PatchMatchGpu
{
	private readonly PatchMatchOptions options;
	private readonly PatchMatch.Problem problem;
	private readonly ulong seed;
	private readonly int refWidth;
	private readonly int refHeight;
	private readonly int numSrc;
	private readonly PatchMatchLikelihood likelihood;
	private readonly BilateralWeightComputer bilateral;
	private readonly PatchMatchTransforms transforms;
	private readonly PatchMatchSourceImages srcImages;
	private readonly PatchMatchSourceDepthMaps? srcDepthMaps;
	private readonly PatchMatchRefImage refImage;

	private Mat<float>? depthMap;
	private Mat<float>? normalMap;
	private Mat<float>? selProbMap;
	private Mat<byte> consistencyMask = new(0, 0, 0);

	/// <summary>
	/// Sets up <paramref name="problem"/> (already checked by PatchMatch.Check) under
	/// <paramref name="options"/>, as PatchMatchCpu's constructor does, without touching a device.
	/// <paramref name="seed"/> keys the random draws.
	/// </summary>
	public PatchMatchGpu(PatchMatchOptions options, PatchMatch.Problem problem, ulong seed = PatchMatchRandom.DefaultSeed)
	{
		this.options = options.Clone();
		this.problem = problem;
		this.seed = seed;
		List<Image> images = Util.Check.NotNull(problem.Images);
		Image ref0 = images[problem.RefImageIdx];
		refWidth = ref0.GetWidth();
		refHeight = ref0.GetHeight();
		numSrc = problem.SrcImageIdxs.Count;

		// The same constructions as PatchMatchCpu, so every constant is the same float.
		likelihood = new PatchMatchLikelihood(
			(float)options.NccSigma, (float)(options.MinTriangulationAngle * 0.0174532925199432), (float)options.IncidentAngleSigma);
		bilateral = new BilateralWeightComputer((float)options.SigmaSpatial, (float)options.SigmaColor);
		transforms = new PatchMatchTransforms(images, problem.RefImageIdx, problem.SrcImageIdxs);

		refImage = new PatchMatchRefImage(refWidth, refHeight);
		refImage.Filter(
			ref0.GetBitmap().RowMajorData, options.WindowRadius, options.WindowStep,
			(float)options.SigmaSpatial, (float)options.SigmaColor, options.NumThreads);

		var sources = new List<Image>(numSrc);
		foreach (int imageIdx in problem.SrcImageIdxs)
		{
			sources.Add(images[imageIdx]);
		}

		srcImages = new PatchMatchSourceImages(sources);
		if (options.GeomConsistency)
		{
			List<DepthMap> allDepthMaps = Util.Check.NotNull(problem.DepthMaps);
			var srcDepths = new List<DepthMap>(numSrc);
			foreach (int imageIdx in problem.SrcImageIdxs)
			{
				srcDepths.Add(allDepthMaps[imageIdx]);
			}

			srcDepthMaps = new PatchMatchSourceDepthMaps(srcDepths, srcImages.MaxWidth, srcImages.MaxHeight);
		}
	}

	/// <summary>
	/// Rows per sweep_band dispatch when positive, in place of the plan's (the GPU side of
	/// PatchMatchCpu.SweepBandHeight). Only tests set it, to run several bands per sweep on a
	/// small problem (PatchMatchGpuTwinTests).
	/// </summary>
	internal int SweepBandRows { get; init; }

	/// <summary>The sizes <see cref="PatchMatchGpuPlan"/> plans this problem for.</summary>
	public PatchMatchGpuProblemShape ProblemShape => new(refWidth, refHeight, numSrc, srcImages.MaxWidth, srcImages.MaxHeight);

	/// <summary>The per-problem values baked into the kernels' text.</summary>
	public PatchMatchGpuShaderShape ShaderShape => new(
		refWidth, refHeight, numSrc, srcImages.MaxWidth, srcImages.MaxHeight,
		options.WindowRadius, options.WindowStep, options.NumSamples, options.GeomConsistency);

	/// <summary>The estimated depth map (0 where filtered). Mirrors PatchMatchCpu.GetDepthMap.</summary>
	public DepthMap GetDepthMap() => new(CopyOf(Result(depthMap)), (float)options.DepthMin, (float)options.DepthMax);

	/// <summary>The estimated normal map (0 where filtered). Mirrors PatchMatchCpu.GetNormalMap.</summary>
	public NormalMap GetNormalMap() => new(CopyOf(Result(normalMap)));

	/// <summary>
	/// The selection probability of every source image per pixel after the last sweep.
	/// Mirrors PatchMatchCpu.GetSelProbMap.
	/// </summary>
	public Mat<float> GetSelProbMap() => CopyOf(Result(selProbMap));

	/// <summary>
	/// For every pixel with consistent source images: col, row, count, then the image indices
	/// (of the problem's images); empty when the run did not filter. Mirrors
	/// PatchMatchCpu.GetConsistentImageIdxs.
	/// </summary>
	public List<int> GetConsistentImageIdxs()
	{
		Result(depthMap);
		return PatchMatchCpu.ConsistentImageIdxs(consistencyMask, problem.SrcImageIdxs);
	}

	/// <summary>
	/// The sweep settings of sweep <paramref name="sweep"/> of iteration <paramref name="iter"/>,
	/// computed exactly as PatchMatchCpu.Run computes its SweepOptions.
	/// </summary>
	internal (float Perturbation, float PrevSelProbWeight, int Phase, bool FilterPhoto, bool FilterGeom) SweepSettings(int iter, int sweep)
	{
		float totalNumSteps = options.NumIterations * 4;
		bool lastSweep = iter == options.NumIterations - 1 && sweep == 3;
		return (
			1.0f / MathF.Pow(2.0f, iter + sweep / 4.0f),
			(iter * 4 + sweep) / totalNumSteps,
			PatchMatchRandom.SweepPhase(iter, sweep),
			lastSweep && options.Filter,
			lastSweep && options.Filter && options.GeomConsistency);
	}

	// ---- Buffer contents ----

	/// <summary>The byte / 255 table: 256 floats, four per vec4 (PmByteToUnit).</summary>
	internal static byte[] PackByteTable() => MemoryMarshal.AsBytes(PatchMatchSourceImages.ByteToUnitTable).ToArray();

	/// <summary>
	/// The pose table of all four rotations (patch_match_geometry.wgsl's contract): floats
	/// [8 r, 8 r + 4) RefK(r), [8 r + 4, 8 r + 8) RefInvK(r), then Poses(r) of every source at
	/// 32 + 43 (r S + s).
	/// </summary>
	internal byte[] PackPoseTable()
	{
		int rowsOffset = 4 * PatchMatchGpuPlan.PoseTableHeaderVec4;
		float[] table = new float[rowsOffset + 4 * PatchMatchTransforms.NumTformParams * numSrc];
		for (int r = 0; r < 4; ++r)
		{
			transforms.RefK(r).CopyTo(table.AsSpan(8 * r, 4));
			transforms.RefInvK(r).CopyTo(table.AsSpan(8 * r + 4, 4));
			transforms.Poses(r).CopyTo(table.AsSpan(rowsOffset + r * numSrc * PatchMatchTransforms.NumTformParams));
		}

		return MemoryMarshal.AsBytes(table.AsSpan()).ToArray();
	}

	/// <summary>The PmProblem uniform (patch_match_layout.wgsl, 64 bytes).</summary>
	internal byte[] PackProblem()
	{
		// As PatchMatchCpu's fields: DEG2RAD in double, stored as float.
		float filterMinNcc = (float)options.FilterMinNcc;
		float filterMinTriangulationAngle = (float)(options.FilterMinTriangulationAngle * 0.0174532925199432);

		var bytes = new byte[PatchMatchGpuKernels.ProblemUniformSize];
		Span<byte> b = bytes;
		PutF32(b, 0, likelihood.CosMinTriangulationAngle);
		PutF32(b, 4, likelihood.InvIncidentAngleSigmaSquare);
		PutF32(b, 8, likelihood.InvNccSigmaSquare);
		PutF32(b, 12, likelihood.NccNormFactor);
		PutU32(b, 16, (uint)seed);
		PutU32(b, 20, (uint)(seed >> 32));
		PutF32(b, 24, (float)options.DepthMin);
		PutF32(b, 28, (float)options.DepthMax);
		PutF32(b, 32, (float)options.GeomConsistencyRegularizer);
		PutF32(b, 36, (float)options.GeomConsistencyMaxCost);
		PutF32(b, 40, likelihood.ComputeNCCProb(1.0f - filterMinNcc));
		PutF32(b, 44, MathF.Cos(filterMinTriangulationAngle));
		PutF32(b, 48, (float)options.FilterGeomConsistencyMaxCost);
		PutU32(b, 52, (uint)options.FilterMinNumConsistent);
		PutF32(b, 56, bilateral.SpatialNormalization);
		PutF32(b, 60, bilateral.ColorNormalization);
		return bytes;
	}

	/// <summary>
	/// The PmSweep uniform of sweep <paramref name="sweep"/> of iteration <paramref name="iter"/>
	/// (32 bytes). The maps are in rotation <paramref name="sweep"/> during it.
	/// </summary>
	internal byte[] PackSweep(int iter, int sweep)
	{
		var s = SweepSettings(iter, sweep);
		var bytes = new byte[PatchMatchGpuKernels.SweepUniformSize];
		Span<byte> b = bytes;
		PutU32(b, 0, (uint)sweep);
		PutU32(b, 4, (uint)s.Phase);
		PutF32(b, 8, s.Perturbation);

		// perturbation * M_PI is a double product passed as a float (PatchMatchCpu.SweepRows).
		PutF32(b, 12, (float)(s.Perturbation * Math.PI));
		PutF32(b, 16, s.PrevSelProbWeight);
		PutU32(b, 20, s.FilterPhoto ? 1u : 0u);
		PutU32(b, 24, s.FilterGeom ? 1u : 0u);
		return bytes;
	}

	/// <summary>The PmBand uniform of rows [<paramref name="rowStart"/>, <paramref name="rowEnd"/>) (16 bytes).</summary>
	internal static byte[] PackBand(int rowStart, int rowEnd)
	{
		var bytes = new byte[PatchMatchGpuKernels.BandUniformSize];
		PutU32(bytes, 0, (uint)rowStart);
		PutU32(bytes, 4, (uint)rowEnd);
		return bytes;
	}

	/// <summary>The PmRotate uniform (patch_match_rotate.wgsl, 32 bytes).</summary>
	internal static byte[] PackRotate(int srcWidth, int srcHeight, int numPlanes, int srcOffset, int dstOffset)
	{
		var bytes = new byte[PatchMatchGpuKernels.RotateUniformSize];
		Span<byte> b = bytes;
		PutU32(b, 0, (uint)srcWidth);
		PutU32(b, 4, (uint)srcHeight);
		PutU32(b, 8, (uint)numPlanes);
		PutU32(b, 12, (uint)srcOffset);
		PutU32(b, 16, (uint)dstOffset);
		return bytes;
	}

	/// <summary>The unrotated reference planes [image byte / 255 | sum | squared sum].</summary>
	internal byte[] PackReference()
	{
		int planeSize = refWidth * refHeight;
		float[] planes = new float[3 * planeSize];
		ReadOnlySpan<float> byteToUnit = PatchMatchSourceImages.ByteToUnitTable;
		byte[] image = refImage.Image.Data;
		for (int i = 0; i < planeSize; ++i)
		{
			planes[i] = byteToUnit[image[i]];
		}

		refImage.SumImage.Data.CopyTo(planes, planeSize);
		refImage.SquaredSumImage.Data.CopyTo(planes, 2 * planeSize);
		return MemoryMarshal.AsBytes(planes.AsSpan()).ToArray();
	}

	/// <summary>
	/// The initial depth and normal planes of a geometric run (the problem's maps of the
	/// reference image, as PatchMatchCpu copies them); the column states after them start at
	/// zero. A photometric run starts from init_random instead and gets no initial data.
	/// </summary>
	internal byte[] PackInitialState()
	{
		int planeSize = refWidth * refHeight;
		float[] planes = new float[4 * planeSize];
		Array.Copy(Util.Check.NotNull(problem.DepthMaps)[problem.RefImageIdx].Data, planes, planeSize);
		Array.Copy(Util.Check.NotNull(problem.NormalMaps)[problem.RefImageIdx].Data, 0, planes, planeSize, 3 * planeSize);
		return MemoryMarshal.AsBytes(planes.AsSpan()).ToArray();
	}

	/// <summary>The first sweep's previous selection probabilities: 0.5 everywhere, as PatchMatchCpu fills them.</summary>
	internal byte[] PackInitialPrevSelProbs()
	{
		float[] prev = new float[numSrc * refWidth * refHeight];
		Array.Fill(prev, 0.5f);
		return MemoryMarshal.AsBytes(prev.AsSpan()).ToArray();
	}

	/// <summary>The source layers' bytes, zero-padded to whole u32 words (patch_match_textures.wgsl).</summary>
	internal byte[] PackSourceImages(long size)
	{
		var bytes = new byte[size];
		srcImages.RawData.CopyTo(bytes);
		return bytes;
	}

	/// <summary>The source depth layers, or null for a photometric run (a zeroed 4-byte dummy).</summary>
	internal byte[]? PackSourceDepths() =>
		srcDepthMaps == null ? null : MemoryMarshal.AsBytes(srcDepthMaps.RawData).ToArray();

	// ---- Read-back decoding ----

	/// <summary>
	/// Decodes the final state buffer's depth and normal planes (the maps are back in rotation
	/// 0 after 4 x iterations sweeps), the final selection probabilities, and the mask (u32 per
	/// (source, pixel), nonzero where consistent) when the run filtered.
	/// </summary>
	private void Decode(ReadOnlySpan<byte> state, ReadOnlySpan<byte> selProbs, ReadOnlySpan<byte> mask)
	{
		int planeSize = refWidth * refHeight;
		ReadOnlySpan<float> stateFloats = MemoryMarshal.Cast<byte, float>(state);
		var depth = new Mat<float>(refWidth, refHeight, 1);
		stateFloats[..planeSize].CopyTo(depth.Data);
		var normal = new Mat<float>(refWidth, refHeight, 3);
		stateFloats.Slice(planeSize, 3 * planeSize).CopyTo(normal.Data);
		var sel = new Mat<float>(refWidth, refHeight, numSrc);
		MemoryMarshal.Cast<byte, float>(selProbs).CopyTo(sel.Data);

		if (mask.IsEmpty)
		{
			consistencyMask = new Mat<byte>(0, 0, 0);
		}
		else
		{
			ReadOnlySpan<uint> words = MemoryMarshal.Cast<byte, uint>(mask);
			consistencyMask = new Mat<byte>(refWidth, refHeight, numSrc);
			byte[] maskData = consistencyMask.Data;
			for (int i = 0; i < maskData.Length; ++i)
			{
				maskData[i] = words[i] != 0 ? (byte)1 : (byte)0;
			}
		}

		depthMap = depth;
		normalMap = normal;
		selProbMap = sel;
	}

	private static Mat<float> Result(Mat<float>? mat) =>
		mat ?? throw new InvalidOperationException("PatchMatchGpu.RunAsync has not completed, so there are no results yet.");

	private static Mat<float> CopyOf(Mat<float> mat)
	{
		var copy = new Mat<float>(mat.GetWidth(), mat.GetHeight(), mat.GetDepth());
		Array.Copy(mat.Data, copy.Data, mat.Data.Length);
		return copy;
	}

	private static void PutF32(Span<byte> bytes, int offset, float value) => BinaryPrimitives.WriteSingleLittleEndian(bytes[offset..], value);

	private static void PutU32(Span<byte> bytes, int offset, uint value) => BinaryPrimitives.WriteUInt32LittleEndian(bytes[offset..], value);
}
