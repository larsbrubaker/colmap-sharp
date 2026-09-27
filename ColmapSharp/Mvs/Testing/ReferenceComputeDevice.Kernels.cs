// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// The kernel twins of ReferenceComputeDevice (ReferenceComputeDevice.cs): for each GPU
// PatchMatch entry point (Mvs/Shaders/patch_match_*.wgsl), the CPU run of one dispatch. Not a
// COLMAP port (PORTING_PLAN.md Phase 13). Each twin reads the kernel's inputs out of the bound
// buffers into the CPU structures (Mat planes, PatchMatchRefImage, PatchMatchSourceImages,
// PatchMatchTransforms from the pose table, the likelihood and bilateral weight from the
// problem uniform), runs the production CPU code for every element the dispatch reaches, and
// copies the outputs back. Column states are used in place, as spans of the state buffer.
// Copying back a whole map is safe for elements the dispatch did not reach: the CPU code left
// them as they were read, and the copies move bits.
//
// Where the WGSL's contract differs from the CPU's, the twin follows the WGSL:
// - filter_pixels writes 0 or 1 to every (source, pixel) mask entry of the cost buffer; the CPU
//   writes only the 1s of a zeroed mask, so the twin runs FilterPixel on a zeroed mask and then
//   stores every entry.
// - The per-sweep and per-problem values arrive derived (filter_min_ncc_prob, the filter angle's
//   cosine, the likelihood's four floats, the bilateral normalizations); the twin passes them to
//   the CPU code as they are. normal_perturbation must be (float)(perturbation * Math.PI), the
//   value SweepRows computes; any other value faults, since SweepRows cannot take it.
// - A binding smaller than what the kernel indexes, or a band outside the frame, faults (the WGSL
//   would read or write out of bounds, which WebGPU leaves undefined).

using System.Buffers.Binary;

using ColmapSharp.Compute.Testing;

namespace ColmapSharp.Mvs.Testing;

public sealed partial class ReferenceComputeDevice
{
	private void Execute(DispatchCommand dispatch)
	{
		TwinKernel twin = this.kernels[dispatch.Kernel];
		var run = new TwinRun(twin, new TwinBindings(dispatch), new DispatchCoverage(twin.GroupsX, dispatch.X, dispatch.Y));
		switch (twin.Kind)
		{
			case PatchMatchGpuKernel.InitRandom:
				run.InitRandom();
				break;
			case PatchMatchGpuKernel.InitialCost:
				run.InitialCost();
				break;
			case PatchMatchGpuKernel.BackwardMessages:
				run.BackwardMessages();
				break;
			case PatchMatchGpuKernel.SweepBand:
				run.SweepBand();
				break;
			case PatchMatchGpuKernel.FilterPixels:
				run.FilterPixels();
				break;
			case PatchMatchGpuKernel.RotatePlanes:
				run.RotatePlanes();
				break;
			default:
				run.RotateNormals();
				break;
		}
	}

	/// <summary>One dispatch of a PatchMatch kernel on the CPU.</summary>
	private sealed partial class TwinRun
	{
		private readonly TwinKernel twin;
		private readonly TwinBindings bindings;
		private readonly DispatchCoverage coverage;

		public TwinRun(TwinKernel twin, TwinBindings bindings, DispatchCoverage coverage)
		{
			this.twin = twin;
			this.bindings = bindings;
			this.coverage = coverage;
		}

		private int RefWidth => this.twin.Int("PM_REF_WIDTH");

		private int RefHeight => this.twin.Int("PM_REF_HEIGHT");

		private int PlaneSize => this.RefWidth * this.RefHeight;

		private int NumSrc => this.twin.Int("PM_NUM_SRC_IMAGES");

		public void InitRandom()
		{
			ProblemUniform problem = this.Problem();
			(Mat<float> depth, Mat<float> normal) = this.ReadDepthNormal(0);
			PatchMatchCpu cpu = this.Cpu(problem, 0, depth, normal, Empty(), Empty(), Empty(), sources: false, depths: false);
			var frame = new PatchMatchFrame(problem.Transforms, 0);
			for (int pixel = 0; pixel < this.PlaneSize; ++pixel)
			{
				if (this.coverage.Covers(pixel))
				{
					cpu.InitRandomPixel(frame, pixel / this.RefWidth, pixel % this.RefWidth, problem.DepthMin, problem.DepthMax);
				}
			}

			this.WriteDepthNormal(depth, normal);
		}

		public void InitialCost()
		{
			ProblemUniform problem = this.Problem();
			(Mat<float> depth, Mat<float> normal) = this.ReadDepthNormal(0);
			Mat<float> costs = this.ReadPlanes(PatchMatchGpuKernels.CostsBinding, 0, "cost");
			PatchMatchPhotoConsistency pcc = this.PhotoConsistency(problem, 0);
			Span<float> pixelNormal = stackalloc float[3];
			int planeSize = this.PlaneSize;

			// ComputeInitialCost's loop body, one (source image, pixel) element at a time.
			for (long index = 0; index < (long)planeSize * this.NumSrc; ++index)
			{
				if (!this.coverage.Covers(index))
				{
					continue;
				}

				int imageIdx = (int)(index / planeSize);
				int pixel = (int)(index % planeSize);
				pixelNormal[0] = normal.Data[pixel];
				pixelNormal[1] = normal.Data[planeSize + pixel];
				pixelNormal[2] = normal.Data[2 * planeSize + pixel];
				costs.Data[index] = pcc.Compute(pixel / this.RefWidth, pixel % this.RefWidth, depth.Data[pixel], pixelNormal, imageIdx);
			}

			this.WritePlanes(PatchMatchGpuKernels.CostsBinding, costs);
		}

		public void BackwardMessages()
		{
			SweepUniform sweep = this.Sweep();
			ProblemUniform problem = this.Problem();
			int width = FrameWidth(sweep.Rotation, this.RefWidth, this.RefHeight);
			(Mat<float> depth, Mat<float> normal) = this.ReadDepthNormal(sweep.Rotation);
			Mat<float> costs = this.ReadPlanes(PatchMatchGpuKernels.CostsBinding, sweep.Rotation, "cost");
			Mat<float> selProbs = this.ReadPlanes(PatchMatchGpuKernels.SelProbsBinding, sweep.Rotation, "selection probability");
			PatchMatchCpu cpu = this.Cpu(problem, sweep.Rotation, depth, normal, costs, selProbs, Empty(), sources: false, depths: false);
			Span<float> state = this.StateWithColumns(width);
			for (long index = 0; index < (long)width * this.NumSrc; ++index)
			{
				if (!this.coverage.Covers(index))
				{
					continue;
				}

				int imageIdx = (int)(index / width);
				int col = (int)(index % width);
				Span<float> columnState = this.ColumnState(state, col);
				cpu.BackwardMessage(col, imageIdx, columnState);
				if (imageIdx == 0)
				{
					cpu.InitColumnPrevious(col, columnState);
				}
			}

			this.WritePlanes(PatchMatchGpuKernels.SelProbsBinding, selProbs);
		}

		public void SweepBand()
		{
			SweepUniform sweep = this.Sweep();
			ProblemUniform problem = this.Problem();
			ReadOnlySpan<byte> band = this.bindings.Uniform(PatchMatchGpuKernels.BandGroup, 0, PatchMatchGpuKernels.BandUniformSize, "band");
			int rowStart = BinaryPrimitives.ReadInt32LittleEndian(band);
			int rowEnd = BinaryPrimitives.ReadInt32LittleEndian(band[4..]);
			int width = FrameWidth(sweep.Rotation, this.RefWidth, this.RefHeight);
			int height = FrameWidth(sweep.Rotation + 1, this.RefWidth, this.RefHeight);
			if (rowStart < 0 || rowEnd > height)
			{
				throw new InvalidOperationException($"The band [{rowStart}, {rowEnd}) runs outside the frame's {height} rows.");
			}

			float normalPerturbation = (float)(sweep.Perturbation * Math.PI);
			if (BitConverter.SingleToUInt32Bits(normalPerturbation) != BitConverter.SingleToUInt32Bits(sweep.NormalPerturbation))
			{
				throw new InvalidOperationException(
					$"The sweep uniform's normal_perturbation ({sweep.NormalPerturbation:R}) is not (float)(perturbation * Math.PI) ({normalPerturbation:R}), the value SweepRows uses.");
			}

			bool geometric = this.twin.Int("PM_GEOM_CONSISTENCY") != 0;
			(Mat<float> depth, Mat<float> normal) = this.ReadDepthNormal(sweep.Rotation);
			Mat<float> costs = this.ReadPlanes(PatchMatchGpuKernels.CostsBinding, sweep.Rotation, "cost");
			Mat<float> selProbs = this.ReadPlanes(PatchMatchGpuKernels.SelProbsBinding, sweep.Rotation, "selection probability");
			Mat<float> prevSelProbs = this.ReadPlanes(PatchMatchGpuKernels.PrevSelProbsBinding, sweep.Rotation, "previous selection probability");
			PatchMatchCpu cpu = this.Cpu(problem, sweep.Rotation, depth, normal, costs, selProbs, prevSelProbs, sources: true, depths: geometric);
			PatchMatchPhotoConsistency pcc = this.PhotoConsistency(problem, sweep.Rotation);
			var options = new PatchMatchCpu.SweepOptions
			{
				Perturbation = sweep.Perturbation,
				PrevSelProbWeight = sweep.PrevSelProbWeight,
				Phase = sweep.Phase,
				GeomConsistencyTerm = geometric,
			};

			Span<float> state = this.StateWithColumns(width);
			float[] rowScratch = new float[PatchMatchCpu.RowScratchSize(this.NumSrc, pcc.WindowCount)];

			// A column is one invocation (serial) or one workgroup (cooperative), and element
			// col * perColumn is its first. Either way the twin runs the column's rows with the
			// CPU's SweepRows, not the cooperative WGSL: it checks that scheme's dispatch shape and
			// coverage, while a real-GPU test pins the kernel itself.
			int perColumn = PatchMatchGpuKernels.SweepInvocationsPerColumn(this.twin.Int("PM_SWEEP_COOPERATIVE") != 0);
			for (int col = 0; col < width; ++col)
			{
				if (this.coverage.Covers((long)col * perColumn))
				{
					cpu.SweepRows(col, rowStart, rowEnd, options, pcc, this.ColumnState(state, col), rowScratch);
				}
			}

			this.WriteDepthNormal(depth, normal);
			this.WritePlanes(PatchMatchGpuKernels.CostsBinding, costs);
			this.WritePlanes(PatchMatchGpuKernels.SelProbsBinding, selProbs);
		}

		public void FilterPixels()
		{
			SweepUniform sweep = this.Sweep();
			ProblemUniform problem = this.Problem();
			int width = FrameWidth(sweep.Rotation, this.RefWidth, this.RefHeight);
			int height = FrameWidth(sweep.Rotation + 1, this.RefWidth, this.RefHeight);
			int planeSize = this.PlaneSize;
			(Mat<float> depth, Mat<float> normal) = this.ReadDepthNormal(sweep.Rotation);
			Mat<float> selProbs = this.ReadPlanes(PatchMatchGpuKernels.SelProbsBinding, sweep.Rotation, "selection probability");
			var mask = new Mat<byte>(width, height, this.NumSrc);
			PatchMatchCpu cpu = this.Cpu(
				problem, sweep.Rotation, depth, normal, new Mat<float>(width, height, this.NumSrc), selProbs, Empty(),
				sources: false, depths: sweep.FilterGeomConsistency, mask);
			var options = new PatchMatchCpu.SweepOptions
			{
				FilterPhotoConsistency = sweep.FilterPhotoConsistency,
				FilterGeomConsistency = sweep.FilterGeomConsistency,
			};

			float[] poses = problem.Transforms.Poses(sweep.Rotation);
			var frame = new PatchMatchFrame(problem.Transforms, sweep.Rotation);
			Span<uint> maskWords = this.bindings.Words(0, PatchMatchGpuKernels.CostsBinding);
			this.bindings.Floats(PatchMatchGpuKernels.CostsBinding, (long)planeSize * this.NumSrc, "mask");
			Span<float> bestNormal = stackalloc float[3];
			for (int pixel = 0; pixel < planeSize; ++pixel)
			{
				if (!this.coverage.Covers(pixel))
				{
					continue;
				}

				bestNormal[0] = normal.Data[pixel];
				bestNormal[1] = normal.Data[planeSize + pixel];
				bestNormal[2] = normal.Data[2 * planeSize + pixel];
				cpu.FilterPixel(
					options, poses, frame, pixel / width, pixel % width, depth.Data[pixel], bestNormal,
					problem.FilterMinNccProb, problem.FilterCosMinTriangulationAngle);

				// The WGSL contract: every entry of the pixel gets 0 or 1.
				for (int imageIdx = 0; imageIdx < this.NumSrc; ++imageIdx)
				{
					maskWords[imageIdx * planeSize + pixel] = mask.Data[imageIdx * planeSize + pixel];
				}
			}

			this.WriteDepthNormal(depth, normal);
		}

		private static Mat<float> Empty() => new(0, 0, 0);

		/// <summary>The width of the frame rotated <paramref name="rotation"/> times (its height is the next rotation's width).</summary>
		private static int FrameWidth(int rotation, int refWidth, int refHeight) => (rotation & 1) == 0 ? refWidth : refHeight;
	}
}
