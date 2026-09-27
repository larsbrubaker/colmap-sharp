// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PatchMatchCpu.Kernels: a PatchMatchCpu over maps and parameters that are given rather than
// derived from a PatchMatch.Problem. Not a COLMAP port (PORTING_PLAN.md Phase 13). It exists so
// the CPU twin of the GPU kernels (Mvs/Testing/ReferenceComputeDevice.cs) runs the production
// sweep code - InitRandomPixel, BackwardMessage, InitColumnPrevious, SweepRows, FilterPixel in
// PatchMatchCpu.cs and PatchMatchCpu.Sweep.cs - on maps it reads out of the GPU buffers, with
// the parameters the GPU uniforms carry, instead of re-implementing any of that math.

namespace ColmapSharp.Mvs;

internal sealed partial class PatchMatchCpu
{
	/// <summary>
	/// Everything a kernel-level PatchMatchCpu needs that the public constructor would derive
	/// from the options and the problem: the values of the GPU's constants and problem uniform,
	/// and the inputs rebuilt from its buffers.
	/// </summary>
	internal readonly record struct KernelParameters(
		int RefWidth,
		int RefHeight,
		ulong Seed,
		int WindowRadius,
		int NumSamples,
		float GeomConsistencyRegularizer,
		float GeomConsistencyMaxCost,
		int FilterMinNumConsistent,
		float FilterGeomConsistencyMaxCost,
		PatchMatchLikelihood Likelihood,
		PatchMatchTransforms Transforms,
		PatchMatchSourceImages? SrcImages,
		PatchMatchSourceDepthMaps? SrcDepthMaps);

	/// <summary>
	/// A PatchMatchCpu in the middle of a run: rotation <paramref name="rotation"/>, over the
	/// given maps (used, not copied). Only the kernel-level methods may be called on it; Run
	/// and FilterPixels need options it does not have (filter_min_ncc and the filter angle
	/// arrive already derived, as FilterPixel's arguments).
	/// </summary>
	internal PatchMatchCpu(
		in KernelParameters parameters,
		int rotation,
		PatchMatchRefImage refImage,
		Mat<float> depthMap,
		Mat<float> normalMap,
		Mat<float> costMap,
		Mat<float> selProbMap,
		Mat<float> prevSelProbMap,
		Mat<byte> consistencyMask)
	{
		// One thread: the caller runs one GPU element at a time.
		options = new PatchMatchOptions { NumThreads = 1 };
		problem = new PatchMatch.Problem();
		seed = parameters.Seed;
		refWidth = parameters.RefWidth;
		refHeight = parameters.RefHeight;
		windowRadius = parameters.WindowRadius;
		numSamples = parameters.NumSamples;
		geomConsistencyRegularizer = parameters.GeomConsistencyRegularizer;
		geomConsistencyMaxCost = parameters.GeomConsistencyMaxCost;

		// Consumed only by FilterPixels, which a kernel-level instance never runs.
		filterMinNcc = float.NaN;
		filterMinTriangulationAngle = float.NaN;
		filterMinNumConsistent = parameters.FilterMinNumConsistent;
		filterGeomConsistencyMaxCost = parameters.FilterGeomConsistencyMaxCost;
		likelihood = parameters.Likelihood;
		transforms = parameters.Transforms;

		// The photometric kernels take their source images through PatchMatchPhotoConsistency,
		// so an instance that runs none of them has no use for the field.
		srcImages = parameters.SrcImages!;
		srcDepthMaps = parameters.SrcDepthMaps;
		this.rotation = rotation;
		this.refImage = refImage;
		this.depthMap = depthMap;
		this.normalMap = normalMap;
		this.costMap = costMap;
		this.selProbMap = selProbMap;
		this.prevSelProbMap = prevSelProbMap;
		this.consistencyMask = consistencyMask;
	}
}
