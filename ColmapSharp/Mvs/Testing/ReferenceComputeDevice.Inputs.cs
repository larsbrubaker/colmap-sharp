// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// The buffer <-> CPU structure adapters of ReferenceComputeDevice's kernel twins
// (ReferenceComputeDevice.Kernels.cs), and the two rotation twins, which need nothing else.
// Not a COLMAP port (PORTING_PLAN.md Phase 13). The layouts read here are the ones
// patch_match_layout.wgsl, patch_match_geometry.wgsl (the pose table), patch_match_textures.wgsl
// and patch_match_rotate.wgsl document; the CPU structures are rebuilt through the internal
// constructors that take already-derived values (PatchMatchTransforms.FromPoseTable,
// PatchMatchLikelihood.FromParameters, BilateralWeightComputer.FromNormalizations, ...), so
// nothing the GPU was given is re-derived differently here.

using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace ColmapSharp.Mvs.Testing;

public sealed partial class ReferenceComputeDevice
{
	/// <summary>The PmProblem uniform, and the transforms of the pose table next to it.</summary>
	private sealed record ProblemUniform(
		PatchMatchLikelihood Likelihood,
		ulong Seed,
		float DepthMin,
		float DepthMax,
		float GeomConsistencyRegularizer,
		float GeomConsistencyMaxCost,
		float FilterMinNccProb,
		float FilterCosMinTriangulationAngle,
		float FilterGeomConsistencyMaxCost,
		int FilterMinNumConsistent,
		BilateralWeightComputer Bilateral,
		PatchMatchTransforms Transforms);

	/// <summary>The PmSweep uniform.</summary>
	private readonly record struct SweepUniform(
		int Rotation,
		int Phase,
		float Perturbation,
		float NormalPerturbation,
		float PrevSelProbWeight,
		bool FilterPhotoConsistency,
		bool FilterGeomConsistency);

	private sealed partial class TwinRun
	{
		// byte / 255.0f back to the byte, for the reference image plane.
		private static readonly Dictionary<uint, byte> UnitToByte = CreateUnitToByte();

		public void RotatePlanes()
		{
			(uint srcWidth, uint srcHeight, uint numPlanes, int srcOffset, int dstOffset) = this.Rotation();
			int planeSize = checked((int)(srcWidth * srcHeight));
			Span<uint> src = this.RotateBinding(PatchMatchGpuKernels.RotateSourceBinding, srcOffset + (long)numPlanes * planeSize);
			Span<uint> dst = this.RotateBinding(PatchMatchGpuKernels.RotateDestinationBinding, dstOffset + (long)numPlanes * planeSize);
			int[] origin = RotatedIndices((int)srcWidth, (int)srcHeight);
			var plane = new Mat<uint>((int)srcWidth, (int)srcHeight, 1);
			var rotated = new Mat<uint>((int)srcHeight, (int)srcWidth, 1);
			for (int p = 0; p < numPlanes; ++p)
			{
				src.Slice(srcOffset + p * planeSize, planeSize).CopyTo(plane.Data);
				plane.Rotate(rotated, 1);
				for (int d = 0; d < planeSize; ++d)
				{
					// Element p P + origin[d] moves the word that lands at d.
					if (this.coverage.Covers((long)p * planeSize + origin[d]))
					{
						dst[dstOffset + p * planeSize + d] = rotated.Data[d];
					}
				}
			}
		}

		public void RotateNormals()
		{
			(uint srcWidth, uint srcHeight, _, int srcOffset, int dstOffset) = this.Rotation();
			int planeSize = checked((int)(srcWidth * srcHeight));
			Span<uint> src = this.RotateBinding(PatchMatchGpuKernels.RotateSourceBinding, srcOffset + 3L * planeSize);
			Span<uint> dst = this.RotateBinding(PatchMatchGpuKernels.RotateDestinationBinding, dstOffset + 3L * planeSize);
			var normals = new Mat<float>((int)srcWidth, (int)srcHeight, 3);
			MemoryMarshal.Cast<uint, float>(src.Slice(srcOffset, 3 * planeSize)).CopyTo(normals.Data);

			// PatchMatchCpu.Rotate's order: rotate each normal, then move the map.
			PatchMatchKernel.RotateNormalMap(normals, 1);
			var rotated = new Mat<float>((int)srcHeight, (int)srcWidth, 3);
			normals.Rotate(rotated, 1);
			int[] origin = RotatedIndices((int)srcWidth, (int)srcHeight);
			for (int d = 0; d < planeSize; ++d)
			{
				if (this.coverage.Covers(origin[d]))
				{
					for (int k = 0; k < 3; ++k)
					{
						dst[dstOffset + k * planeSize + d] = BitConverter.SingleToUInt32Bits(rotated.Data[k * planeSize + d]);
					}
				}
			}
		}

		private ProblemUniform Problem()
		{
			ReadOnlySpan<byte> u = this.bindings.Uniform(0, PatchMatchGpuKernels.ProblemBinding, PatchMatchGpuKernels.ProblemUniformSize, "problem");
			var likelihood = PatchMatchLikelihood.FromParameters(F(u, 0), F(u, 4), F(u, 8), F(u, 12));
			ulong seed = BinaryPrimitives.ReadUInt32LittleEndian(u[16..]) | ((ulong)BinaryPrimitives.ReadUInt32LittleEndian(u[20..]) << 32);
			Span<float> poseTable = this.bindings.Floats(0, PatchMatchGpuKernels.PosesBinding);
			var transforms = PatchMatchTransforms.FromPoseTable(poseTable, this.RefWidth, this.RefHeight, this.NumSrc);
			return new ProblemUniform(
				likelihood, seed, F(u, 24), F(u, 28), F(u, 32), F(u, 36), F(u, 40), F(u, 44), F(u, 48),
				BinaryPrimitives.ReadInt32LittleEndian(u[52..]),
				BilateralWeightComputer.FromNormalizations(F(u, 56), F(u, 60)),
				transforms);
		}

		private SweepUniform Sweep()
		{
			ReadOnlySpan<byte> u = this.bindings.Uniform(PatchMatchGpuKernels.SweepGroup, 0, PatchMatchGpuKernels.SweepUniformSize, "sweep");
			int rotation = BinaryPrimitives.ReadInt32LittleEndian(u);
			if (rotation is < 0 or > 3)
			{
				throw new InvalidOperationException($"The sweep uniform's rotation is {rotation}; it must be 0..3.");
			}

			return new SweepUniform(
				rotation, BinaryPrimitives.ReadInt32LittleEndian(u[4..]), F(u, 8), F(u, 12), F(u, 16),
				BinaryPrimitives.ReadUInt32LittleEndian(u[20..]) != 0, BinaryPrimitives.ReadUInt32LittleEndian(u[24..]) != 0);
		}

		private (uint SrcWidth, uint SrcHeight, uint NumPlanes, int SrcOffset, int DstOffset) Rotation()
		{
			ReadOnlySpan<byte> u = this.bindings.Uniform(PatchMatchGpuKernels.SweepGroup, 0, PatchMatchGpuKernels.RotateUniformSize, "rotate");
			return (
				BinaryPrimitives.ReadUInt32LittleEndian(u), BinaryPrimitives.ReadUInt32LittleEndian(u[4..]),
				BinaryPrimitives.ReadUInt32LittleEndian(u[8..]), checked((int)BinaryPrimitives.ReadUInt32LittleEndian(u[12..])),
				checked((int)BinaryPrimitives.ReadUInt32LittleEndian(u[16..])));
		}

		private Span<uint> RotateBinding(int binding, long words)
		{
			this.bindings.Floats(binding, words, binding == PatchMatchGpuKernels.RotateSourceBinding ? "rotation source" : "rotation destination");
			return this.bindings.Words(0, binding);
		}

		/// <summary>
		/// For each position of a rotated srcWidth x srcHeight plane, the index of the source
		/// element Mat.Rotate moves there (Mat.Rotate applied to the indices themselves).
		/// </summary>
		private static int[] RotatedIndices(int srcWidth, int srcHeight)
		{
			var indices = new Mat<int>(srcWidth, srcHeight, 1);
			for (int i = 0; i < indices.Data.Length; ++i)
			{
				indices.Data[i] = i;
			}

			var rotated = new Mat<int>(srcHeight, srcWidth, 1);
			indices.Rotate(rotated, 1);
			return rotated.Data;
		}

		/// <summary>The depth and normal planes of the state buffer, in rotation <paramref name="rotation"/>'s frame.</summary>
		private (Mat<float> Depth, Mat<float> Normal) ReadDepthNormal(int rotation)
		{
			(int width, int height) = this.Frame(rotation);
			Span<float> state = this.bindings.Floats(PatchMatchGpuKernels.StateBinding, 4L * this.PlaneSize, "state");
			var depth = new Mat<float>(width, height, 1);
			var normal = new Mat<float>(width, height, 3);
			state[..this.PlaneSize].CopyTo(depth.Data);
			state.Slice(this.PlaneSize, 3 * this.PlaneSize).CopyTo(normal.Data);
			return (depth, normal);
		}

		private void WriteDepthNormal(Mat<float> depth, Mat<float> normal)
		{
			Span<float> state = this.bindings.Floats(0, PatchMatchGpuKernels.StateBinding);
			depth.Data.CopyTo(state);
			normal.Data.CopyTo(state[this.PlaneSize..]);
		}

		/// <summary>The state buffer, checked to hold the column states of <paramref name="width"/> columns.</summary>
		private Span<float> StateWithColumns(int width) =>
			this.bindings.Floats(PatchMatchGpuKernels.StateBinding, 4L * this.PlaneSize + (long)width * PatchMatchCpu.ColumnStateSize(this.NumSrc), "state");

		private Span<float> ColumnState(Span<float> state, int col)
		{
			int size = PatchMatchCpu.ColumnStateSize(this.NumSrc);
			return state.Slice(4 * this.PlaneSize + col * size, size);
		}

		/// <summary>The S planes of a map binding, in rotation <paramref name="rotation"/>'s frame.</summary>
		private Mat<float> ReadPlanes(int binding, int rotation, string what)
		{
			(int width, int height) = this.Frame(rotation);
			var planes = new Mat<float>(width, height, this.NumSrc);
			this.bindings.Floats(binding, planes.Data.Length, what)[..planes.Data.Length].CopyTo(planes.Data);
			return planes;
		}

		private void WritePlanes(int binding, Mat<float> planes) => planes.Data.CopyTo(this.bindings.Floats(0, binding));

		private (int Width, int Height) Frame(int rotation) => (rotation & 1) == 0
			? (this.RefWidth, this.RefHeight)
			: (this.RefHeight, this.RefWidth);

		private PatchMatchPhotoConsistency PhotoConsistency(ProblemUniform problem, int rotation)
		{
			(int width, int height) = this.Frame(rotation);
			int planeSize = this.PlaneSize;
			Span<float> reference = this.bindings.Floats(PatchMatchGpuKernels.ReferenceBinding, 3L * planeSize, "reference");
			var refImage = new PatchMatchRefImage(width, height);
			for (int i = 0; i < planeSize; ++i)
			{
				uint bits = BitConverter.SingleToUInt32Bits(reference[i]);
				refImage.Image.Data[i] = UnitToByte.TryGetValue(bits, out byte value)
					? value
					: throw new InvalidOperationException($"Reference texel {i} is {reference[i]:R}, not byte / 255.0f for any byte.");
			}

			reference.Slice(planeSize, planeSize).CopyTo(refImage.SumImage.Data);
			reference.Slice(2 * planeSize, planeSize).CopyTo(refImage.SquaredSumImage.Data);
			return new PatchMatchPhotoConsistency(
				refImage, this.SourceImages(), problem.Transforms.Poses(rotation), new PatchMatchFrame(problem.Transforms, rotation),
				this.twin.Int("PM_WINDOW_RADIUS"), this.twin.Int("PM_WINDOW_STEP"), problem.Bilateral);
		}

		private PatchMatchSourceImages SourceImages()
		{
			ReadOnlySpan<float> table = this.bindings.Floats(PatchMatchGpuKernels.ByteToUnitBinding, 256, "byte-to-unit");
			ReadOnlySpan<float> expected = PatchMatchSourceImages.ByteToUnitTable;
			if (!MemoryMarshal.AsBytes(table[..256]).SequenceEqual(MemoryMarshal.AsBytes(expected)))
			{
				throw new InvalidOperationException("The byte-to-unit uniform is not byte / 255.0f for every byte, the table the CPU samples with.");
			}

			int maxWidth = this.twin.Int("PM_SRC_MAX_WIDTH");
			int maxHeight = this.twin.Int("PM_SRC_MAX_HEIGHT");
			int size = checked(maxWidth * maxHeight * this.NumSrc);
			Span<byte> bytes = this.bindings.Bytes(0, PatchMatchGpuKernels.SourceImagesBinding);
			if (bytes.Length < size)
			{
				throw new InvalidOperationException($"The source image binding has {bytes.Length} bytes; the layers need {size}.");
			}

			return new PatchMatchSourceImages(bytes[..size].ToArray(), maxWidth, maxHeight, this.NumSrc);
		}

		private PatchMatchSourceDepthMaps SourceDepths()
		{
			int maxWidth = this.twin.Int("PM_SRC_MAX_WIDTH");
			int maxHeight = this.twin.Int("PM_SRC_MAX_HEIGHT");
			int size = checked(maxWidth * maxHeight * this.NumSrc);
			float[] data = this.bindings.Floats(PatchMatchGpuKernels.SourceDepthsBinding, size, "source depth")[..size].ToArray();
			return new PatchMatchSourceDepthMaps(data, maxWidth, maxHeight, this.NumSrc);
		}

		/// <summary>
		/// A kernel-level PatchMatchCpu over the given maps, with the source inputs rebuilt from
		/// their bindings when <paramref name="sources"/> / <paramref name="depths"/> say the
		/// kernel reads them.
		/// </summary>
		private PatchMatchCpu Cpu(
			ProblemUniform problem, int rotation, Mat<float> depth, Mat<float> normal, Mat<float> costs, Mat<float> selProbs,
			Mat<float> prevSelProbs, bool sources, bool depths, Mat<byte>? mask = null)
		{
			bool hasWindow = this.twin.Kind is PatchMatchGpuKernel.InitialCost or PatchMatchGpuKernel.SweepBand;
			var parameters = new PatchMatchCpu.KernelParameters(
				this.RefWidth,
				this.RefHeight,
				problem.Seed,
				hasWindow ? this.twin.Int("PM_WINDOW_RADIUS") : 0,
				this.twin.Kind == PatchMatchGpuKernel.SweepBand ? this.twin.Int("PM_NUM_SAMPLES") : 0,
				problem.GeomConsistencyRegularizer,
				problem.GeomConsistencyMaxCost,
				problem.FilterMinNumConsistent,
				problem.FilterGeomConsistencyMaxCost,
				problem.Likelihood,
				problem.Transforms,
				sources ? this.SourceImages() : null,
				depths ? this.SourceDepths() : null);
			return new PatchMatchCpu(
				parameters, rotation, new PatchMatchRefImage(0, 0), depth, normal, costs, selProbs, prevSelProbs, mask ?? new Mat<byte>(0, 0, 0));
		}

		private static float F(ReadOnlySpan<byte> bytes, int offset) => BinaryPrimitives.ReadSingleLittleEndian(bytes[offset..]);

		private static Dictionary<uint, byte> CreateUnitToByte()
		{
			var map = new Dictionary<uint, byte>();
			ReadOnlySpan<float> table = PatchMatchSourceImages.ByteToUnitTable;
			for (int b = 0; b < 256; ++b)
			{
				map.Add(BitConverter.SingleToUInt32Bits(table[b]), (byte)b);
			}

			return map;
		}
	}
}
