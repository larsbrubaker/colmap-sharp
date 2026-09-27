// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ReferenceComputeDeviceFixture: C#-only test support for ReferenceComputeDeviceTests. Packs a
// small PatchMatch problem (PatchMatchRunTests' synthetic scene) into the GPU buffers the
// kernels read - pose table, problem uniform, reference planes, source layers - following the
// layouts of Mvs/Shaders/patch_match_layout.wgsl and patch_match_geometry.wgsl, and builds the
// same problem's CPU structures so a test can compare a kernel twin against a direct CPU call.

using System.Runtime.InteropServices;

using ColmapSharp.Compute;
using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Testing;

using Image = ColmapSharp.Mvs.Image;

namespace ColmapSharp.Tests.Mvs;

/// <summary>A packed synthetic PatchMatch problem on a <see cref="ReferenceComputeDevice"/>.</summary>
internal sealed class ReferenceComputeDeviceFixture
{
	public const int Width = 13;
	public const int Height = 9;
	public const ulong Seed = PatchMatchRandom.DefaultSeed;

	public ReferenceComputeDeviceFixture()
	{
		(List<Image> images, _, _) = PatchMatchRunTests.Scene(Width, Height);
		Options = PatchMatchRunTests.Options(1, geomConsistency: false, filter: true, numThreads: 1);
		Problem = PatchMatchRunTests.Problem(images);
		Transforms = new PatchMatchTransforms(images, 0, Problem.SrcImageIdxs);
		RefImage = new PatchMatchRefImage(Width, Height);
		RefImage.Filter(images[0].GetBitmap().RowMajorData, Options.WindowRadius, Options.WindowStep, (float)Options.SigmaSpatial, (float)Options.SigmaColor, 1);
		SrcImages = new PatchMatchSourceImages([images[1], images[2]]);
		Likelihood = new PatchMatchLikelihood(
			(float)Options.NccSigma, (float)(Options.MinTriangulationAngle * 0.0174532925199432), (float)Options.IncidentAngleSigma);
		Shape = new PatchMatchGpuShaderShape(
			Width, Height, NumSrc, SrcImages.MaxWidth, SrcImages.MaxHeight, Options.WindowRadius, Options.WindowStep, Options.NumSamples, false);

		ByteToUnit = Device.CreateBuffer(ComputeBufferKind.Uniform, 1024, MemoryMarshal.AsBytes(PatchMatchSourceImages.ByteToUnitTable), "byte_to_unit");
		Poses = Device.CreateBuffer(ComputeBufferKind.Uniform, 4 * PoseTable().Length, MemoryMarshal.AsBytes(PoseTable().AsSpan()), "poses");
		ProblemUniform = Device.CreateBuffer(ComputeBufferKind.Uniform, PatchMatchGpuKernels.ProblemUniformSize, ProblemBytes(), "problem");
		float[] reference = [.. Units(RefImage.Image.Data), .. RefImage.SumImage.Data, .. RefImage.SquaredSumImage.Data];
		Reference = Device.CreateBuffer(ComputeBufferKind.Storage, 4 * reference.Length, MemoryMarshal.AsBytes(reference.AsSpan()), "reference");
		byte[] layers = PadTo4(SrcImages.Data.ToArray());
		SourceImages = Device.CreateBuffer(ComputeBufferKind.Storage, layers.Length, layers, "source_images");
		SourceDepths = Device.CreateBuffer(ComputeBufferKind.Storage, 4, default, "source_depths_dummy");

		// No initial data: the device fills them with its garbage word.
		State = Device.CreateBuffer(ComputeBufferKind.Storage, 4L * (4 * PlaneSize + Math.Max(Width, Height) * (NumSrc + 4)), default, "state");
		Costs = Device.CreateBuffer(ComputeBufferKind.Storage, 4L * NumSrc * PlaneSize, default, "costs");
	}

	public static int PlaneSize => Width * Height;

	public static int NumSrc => 2;

	public ReferenceComputeDevice Device { get; } = new();

	public PatchMatchOptions Options { get; }

	public PatchMatch.Problem Problem { get; }

	public PatchMatchTransforms Transforms { get; }

	public PatchMatchRefImage RefImage { get; }

	public PatchMatchSourceImages SrcImages { get; }

	public PatchMatchLikelihood Likelihood { get; }

	public PatchMatchGpuShaderShape Shape { get; }

	public IComputeBuffer ByteToUnit { get; }

	public IComputeBuffer Poses { get; }

	public IComputeBuffer ProblemUniform { get; }

	public IComputeBuffer Reference { get; }

	public IComputeBuffer SourceImages { get; }

	public IComputeBuffer SourceDepths { get; }

	public IComputeBuffer State { get; }

	public IComputeBuffer Costs { get; }

	/// <summary>Workgroups for <paramref name="elements"/> elements, one row of workgroups.</summary>
	public static uint Groups(long elements) => (uint)((elements + PatchMatchGpuKernels.WorkgroupSize - 1) / PatchMatchGpuKernels.WorkgroupSize);

	/// <summary>The kernel composed for <paramref name="groupsX"/> workgroups along x.</summary>
	public IComputeKernel Kernel(PatchMatchGpuKernel kernel, uint groupsX) =>
		Device.CreateKernel(PatchMatchGpuKernels.Descriptor(kernel, Shape, groupsX));

	/// <summary>A group-0 bind group of <paramref name="kernel"/> from (binding, buffer) pairs, whole buffers.</summary>
	public IComputeBindGroup Group(IComputeKernel kernel, int group, params (int Binding, IComputeBuffer Buffer)[] entries) =>
		Device.CreateBindGroup(kernel, group, entries.Select(e => new ComputeBufferBinding(e.Binding, e.Buffer, 0, e.Buffer.Size)).ToArray());

	/// <summary>A uniform buffer holding <paramref name="words"/> (ints or float bits).</summary>
	public IComputeBuffer Uniform(params uint[] words) =>
		Device.CreateBuffer(ComputeBufferKind.Uniform, 4 * words.Length, MemoryMarshal.AsBytes(words.AsSpan()));

	/// <summary>Reads <paramref name="count"/> floats of <paramref name="buffer"/> (flushing).</summary>
	public async Task<float[]> ReadFloats(IComputeBuffer buffer, int count)
	{
		byte[] bytes = new byte[4 * count];
		await Device.ReadBufferAsync(buffer, 0, bytes);
		return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
	}

	/// <summary>patch_match_geometry.wgsl's pose table: 4 x (RefK, RefInvK), then the 4 pose tables.</summary>
	public float[] PoseTable()
	{
		var table = new List<float>();
		for (int r = 0; r < 4; ++r)
		{
			table.AddRange(Transforms.RefK(r).ToArray());
			table.AddRange(Transforms.RefInvK(r).ToArray());
		}

		for (int r = 0; r < 4; ++r)
		{
			table.AddRange(Transforms.Poses(r));
		}

		// A uniform's size is a multiple of 16 bytes.
		while (table.Count % 4 != 0)
		{
			table.Add(0);
		}

		return [.. table];
	}

	/// <summary>The PmProblem uniform (patch_match_layout.wgsl), from the options as PatchMatchCpu derives them.</summary>
	private byte[] ProblemBytes()
	{
		float sigmaNcc = (float)Options.NccSigma;
		float sigmaInc = (float)Options.IncidentAngleSigma;
		float sigmaSpatial = (float)Options.SigmaSpatial;
		float sigmaColor = (float)Options.SigmaColor;
		float[] floats =
		[
			MathF.Cos((float)(Options.MinTriangulationAngle * 0.0174532925199432)),
			-0.5f / (sigmaInc * sigmaInc),
			-0.5f / (sigmaNcc * sigmaNcc),
			PatchMatchLikelihood.ComputeNCCCostNormFactor(sigmaNcc),
			BitConverter.UInt32BitsToSingle((uint)Seed),
			BitConverter.UInt32BitsToSingle((uint)(Seed >> 32)),
			(float)Options.DepthMin,
			(float)Options.DepthMax,
			(float)Options.GeomConsistencyRegularizer,
			(float)Options.GeomConsistencyMaxCost,
			Likelihood.ComputeNCCProb(1.0f - (float)Options.FilterMinNcc),
			MathF.Cos((float)(Options.FilterMinTriangulationAngle * 0.0174532925199432)),
			(float)Options.FilterGeomConsistencyMaxCost,
			BitConverter.Int32BitsToSingle(Options.FilterMinNumConsistent),
			1.0f / (2.0f * sigmaSpatial * sigmaSpatial),
			1.0f / (2.0f * sigmaColor * sigmaColor),
		];
		return MemoryMarshal.AsBytes(floats.AsSpan()).ToArray();
	}

	private static float[] Units(byte[] bytes) => bytes.Select(b => PatchMatchSourceImages.ByteToUnitTable[b]).ToArray();

	private static byte[] PadTo4(byte[] bytes) => bytes.Length % 4 == 0 ? bytes : [.. bytes, .. new byte[4 - bytes.Length % 4]];
}
