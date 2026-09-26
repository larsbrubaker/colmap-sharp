// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// CovariantSift: COLMAP's CovariantSiftCPUFeatureExtractor (colmap/feature/sift.cc) - SIFT
// through VLFeat's covariant detector (VLFeat/VlCovDet*.cs) with optional affine shape
// estimation, and descriptors from warped patches (vl_imgradient_polar_f +
// vl_sift_calc_raw_descriptor), optionally domain-size pooled (DSP-SIFT: the mean descriptor
// over several patch scales). CreateSiftFeatureExtractor (Sift.cs) picks it for
// estimate_affine_shape, domain_size_pooling or force_covariant_extractor. Tests:
// ColmapSharp.Tests/Feature/SiftTests.cs (sift_test.cc rows), VlCovDetTests.cs (bit-exact
// against unfused VLFeat) and CovariantSiftOracleTests.cs (pycolmap).
//
// Tier A against unfused VLFeat, like Sift.cs; against the macOS arm64 wheel, which fuses
// multiply-adds, within the tolerance of docs/CPP_DIVERGENCES.md entry 41.
//
// Translation notes:
// - Sort: COLMAP orders the features by (octave, level) descending with std::sort, which
//   leaves equal keys in an implementation-defined order. Here ties keep detection order
//   (a stable sort). libc++ insertion-sorts short ranges, which is stable too, so the orders
//   agree there; above that they can differ among equal keys, and so can the one keypoint
//   max_num_features truncation keeps from the next group (entry 57).
// - DSP-SIFT's mean is Eigen's colwise().mean() of a row-major matrix, which vectorizes
//   across columns and adds the rows in blocks of four (PooledMean has the order). The
//   per-scale raw descriptors are pinned bit for bit against VLFeat; the summation order is
//   only checked against the wheel, within entry 41's tolerance.
// - An image too small for the detector's octaves (shorter side below MinimumImageSide, 16
//   pixels by default) throws an ArgumentException saying so; COLMAP's VLFeat reads out of
//   bounds there (undefined behavior), so no result is lost.
// - Extract replaces the keypoints (COLMAP appends to what the caller passed, always an empty
//   vector) and leaves the outputs untouched when cancelled.
// - Cancellation (not in COLMAP) is checked while the scale space is built, per detection
//   octave, while shapes and orientations are estimated, and before each keypoint's
//   descriptor. The checks never touch the numbers.

using ColmapSharp.Feature.VLFeat;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;

namespace ColmapSharp.Feature;

/// <summary>Port of colmap's CovariantSiftCPUFeatureExtractor.</summary>
public sealed class CovariantSiftCpuFeatureExtractor : FeatureExtractor
{
	/// <summary>Patch resolution: the descriptor patch is 2 * 15 + 1 = 31 pixels wide.</summary>
	internal const int PatchResolution = 15;

	/// <summary>Descriptor patch side.</summary>
	internal const int PatchSide = (2 * PatchResolution) + 1;

	private const int MaxOctaveResolution = 1000;
	private const double PatchRelativeExtent = 7.5;
	private const double PatchRelativeSmoothing = 1;
	private const double PatchStep = PatchRelativeExtent / PatchResolution;

	// The descriptor's sigma in patch pixels: 7.5 / 7.5 / 0.5 = 2.
	private const double Sigma = PatchRelativeExtent / (3.0 * (4 + 1) / 2) / PatchStep;

	private readonly SiftExtractionOptions options;

	/// <summary>Creates the extractor; the options must pass Check.</summary>
	public CovariantSiftCpuFeatureExtractor(FeatureExtractionOptions options)
	{
		Util.Check.That(options.Check());

		// COLMAP logs that darkness adaptivity is only available for GLSL SiftGPU; it has no
		// effect here either.
		this.options = options.Sift.Clone();
	}

	/// <summary>
	/// Port of CovariantSiftCPUFeatureExtractor::Extract: detects DoG features (coarsest
	/// octave first), optionally adapts their affine shape and assigns orientations, keeps
	/// whole (octave, level) groups until max_num_features is reached, and computes their
	/// descriptors from warped patches.
	/// </summary>
	public override bool Extract(
		Bitmap bitmap,
		List<FeatureKeypoint> keypoints,
		FeatureDescriptors? descriptors,
		CancellationToken cancellationToken = default)
	{
		Util.Check.That(bitmap.IsGrey);
		Util.Check.NotNull(keypoints);
		Checkpoint(cancellationToken);

		// Not in COLMAP: VLFeat's scale space needs octave max(first_octave, 0) to exist, and
		// reads out of bounds when the image is too small for it.
		int minimumSide = MinimumImageSide(options.FirstOctave);
		if (Math.Min(bitmap.Width, bitmap.Height) < minimumSide)
		{
			throw new ArgumentException(
				$"The image ({bitmap.Width} x {bitmap.Height} pixels) is too small for covariant SIFT with first octave "
				+ $"{options.FirstOctave}: its shorter side must be at least {minimumSide} pixels. "
				+ "Use a larger image, or a lower first octave.",
				nameof(bitmap));
		}

		// Set up the covariant SIFT detector.
		var covdet = new VlCovDet();
		Util.Check.Le(options.OctaveResolution, MaxOctaveResolution);
		covdet.FirstOctave = options.FirstOctave;
		covdet.OctaveResolution = options.OctaveResolution;
		covdet.PeakThreshold = options.PeakThreshold;
		covdet.EdgeThreshold = options.EdgeThreshold;

		{
			byte[] dataUint8 = bitmap.RowMajorData;
			var dataFloat = new float[dataUint8.Length];
			for (int i = 0; i < dataUint8.Length; ++i)
			{
				dataFloat[i] = (float)dataUint8[i] / 255.0f;
			}

			covdet.PutImage(dataFloat, bitmap.Width, bitmap.Height, cancellationToken);
		}

		Checkpoint(cancellationToken);
		covdet.Detect(options.MaxNumFeatures, cancellationToken);

		if (options.EstimateAffineShape)
		{
			Checkpoint(cancellationToken);
			covdet.ExtractAffineShape(cancellationToken);
		}

		if (!options.Upright)
		{
			Checkpoint(cancellationToken);
			covdet.ExtractOrientations(cancellationToken);
		}

		// Sort features by detected octave and scale, both descending (stable, see header).
		VlCovDetFeature[] features = SortByOctaveAndLevel(covdet.Features);

		// Copy the detected keypoints and clamp when the maximum number of features is reached.
		var newKeypoints = new List<FeatureKeypoint>();
		int prevOctaveScaleIdx = int.MaxValue;
		for (int i = 0; i < features.Length; ++i)
		{
			VlFrameOrientedEllipse frame = features[i].Frame;
			newKeypoints.Add(new FeatureKeypoint(
				(float)(frame.X + 0.5), (float)(frame.Y + 0.5), frame.A11, frame.A12, frame.A21, frame.A22));

			int octaveScaleIdx = (features[i].O * MaxOctaveResolution) + features[i].S;
			Util.Check.Le(octaveScaleIdx, prevOctaveScaleIdx);

			if (octaveScaleIdx != prevOctaveScaleIdx && newKeypoints.Count >= options.MaxNumFeatures)
			{
				break;
			}

			prevOctaveScaleIdx = octaveScaleIdx;
		}

		RowMajorMatrix<byte>? descriptorData = null;
		if (descriptors != null)
		{
			descriptorData = ComputeDescriptors(covdet, features, newKeypoints.Count, cancellationToken);
		}

		keypoints.Clear();
		keypoints.AddRange(newKeypoints);
		if (descriptors != null)
		{
			descriptors.Data = descriptorData!;
			descriptors.Type = FeatureExtractorType.Sift;
		}

		return true;
	}

	/// <summary>
	/// Eigen's colwise().mean() of a row-major float matrix (rows x cols) into
	/// <paramref name="mean"/>, in the order its vectorized partial reduction adds the rows:
	/// the first row, then blocks of four rows as ((a + b) + (c + d)) while the block ends
	/// before row (rows - 1) &amp; ~3, then the remaining rows one at a time; divided by the
	/// row count. For the default 10 scales: r0 + ((r1 + r2) + (r3 + r4)) + ((r5 + r6) +
	/// (r7 + r8)) + r9.
	/// </summary>
	internal static void PooledMean(ReadOnlySpan<float> rows, int numRows, int cols, Span<float> mean)
	{
		int size4 = (numRows - 1) & ~3;
		for (int c = 0; c < cols; ++c)
		{
			float p = rows[c];
			int i = 1;
			for (; i < size4; i += 4)
			{
				float ab = rows[(i * cols) + c] + rows[((i + 1) * cols) + c];
				float cd = rows[((i + 2) * cols) + c] + rows[((i + 3) * cols) + c];
				p += ab + cd;
			}

			for (; i < numRows; ++i)
			{
				p += rows[(i * cols) + c];
			}

			mean[c] = p / (float)numRows;
		}
	}

	/// <summary>
	/// The smallest shorter image side the detector accepts for <paramref name="firstOctave"/>:
	/// VLFeat keeps octaves while (15 * 2^o + 1) pixels fit, and needs octave
	/// max(first_octave, 0), so 15 * 2^max(first_octave, 0) + 1 (16 pixels by default).
	/// </summary>
	public static int MinimumImageSide(int firstOctave)
	{
		return (15 << Math.Max(firstOctave, 0)) + 1;
	}

	/// <summary>
	/// Test diagnostic: called at each of the extractor's own cancellation checks (the
	/// detector's inner checks use the token only).
	/// </summary>
	internal Action? CancellationCheckpoint { get; set; }

	/// <summary>The VLFeat SIFT filter COLMAP uses only for its descriptor parameters (magnif 3).</summary>
	internal static VlSiftFilter CreateDescriptorFilter()
	{
		return new VlSiftFilter(16, 16, 1, 3, 0) { Magnif = 3.0 };
	}

	/// <summary>
	/// The per-scale half of COLMAP's descriptor loop: for each DSP scale s, the frame's shape
	/// scaled by <c>dspMinScale + s * dspScaleStep</c> (float arithmetic, as in COLMAP), a
	/// 31 x 31 patch warped from the scale space, its polar gradient, and the raw SIFT
	/// descriptor of the patch center, into row s of <paramref name="scaledDescriptors"/>.
	/// </summary>
	internal static void ComputeRawDescriptors(
		VlCovDet covdet,
		VlSiftFilter sift,
		in VlFrameOrientedEllipse frame,
		float dspMinScale,
		float dspScaleStep,
		int dspNumScales,
		float[] patch,
		float[] patchXY,
		Span<float> scaledDescriptors)
	{
		const int Dim = SiftCpuFeatureExtractor.SiftDescriptorDim;
		for (int s = 0; s < dspNumScales; ++s)
		{
			// float + int * float, then widened.
			double dspScale = dspMinScale + (s * dspScaleStep);

			VlFrameOrientedEllipse scaledFrame = frame;
			scaledFrame.A11 = (float)(scaledFrame.A11 * dspScale);
			scaledFrame.A12 = (float)(scaledFrame.A12 * dspScale);
			scaledFrame.A21 = (float)(scaledFrame.A21 * dspScale);
			scaledFrame.A22 = (float)(scaledFrame.A22 * dspScale);

			covdet.ExtractPatchForFrame(patch, PatchResolution, PatchRelativeExtent, PatchRelativeSmoothing, scaledFrame);

			VlImOpv.ImGradientPolarF(patchXY, 0, patchXY, 1, 2, 2 * PatchSide, patch, 0, PatchSide, PatchSide, PatchSide);

			sift.CalcRawDescriptor(
				patchXY, scaledDescriptors.Slice(s * Dim, Dim), PatchSide, PatchSide, PatchResolution, PatchResolution, Sigma, 0);
		}
	}

	private void Checkpoint(CancellationToken cancellationToken)
	{
		CancellationCheckpoint?.Invoke();
		cancellationToken.ThrowIfCancellationRequested();
	}

	private static VlCovDetFeature[] SortByOctaveAndLevel(ReadOnlySpan<VlCovDetFeature> features)
	{
		var order = new int[features.Length];
		for (int i = 0; i < order.Length; ++i)
		{
			order[i] = i;
		}

		VlCovDetFeature[] source = features.ToArray();
		Array.Sort(order, (a, b) =>
		{
			VlCovDetFeature f1 = source[a];
			VlCovDetFeature f2 = source[b];
			if (f1.O != f2.O)
			{
				return f2.O.CompareTo(f1.O);
			}

			if (f1.S != f2.S)
			{
				return f2.S.CompareTo(f1.S);
			}

			return a.CompareTo(b);
		});

		var sorted = new VlCovDetFeature[order.Length];
		for (int i = 0; i < order.Length; ++i)
		{
			sorted[i] = source[order[i]];
		}

		return sorted;
	}

	private RowMajorMatrix<byte> ComputeDescriptors(
		VlCovDet covdet, VlCovDetFeature[] features, int numKeypoints, CancellationToken cancellationToken)
	{
		var data = new RowMajorMatrix<byte>(numKeypoints, SiftCpuFeatureExtractor.SiftDescriptorDim);
		var patch = new float[PatchSide * PatchSide];
		var patchXY = new float[2 * PatchSide * PatchSide];

		// COLMAP keeps these as float.
		float dspMinScale = 1;
		float dspScaleStep = 0;
		int dspNumScales = 1;
		if (options.DomainSizePooling)
		{
			dspMinScale = (float)options.DspMinScale;
			dspScaleStep = (float)((options.DspMaxScale - options.DspMinScale) / options.DspNumScales);
			dspNumScales = options.DspNumScales;
		}

		var descriptor = new float[SiftCpuFeatureExtractor.SiftDescriptorDim];
		var scaledDescriptors = new float[dspNumScales * SiftCpuFeatureExtractor.SiftDescriptorDim];

		VlSiftFilter sift = CreateDescriptorFilter();

		for (int i = 0; i < numKeypoints; ++i)
		{
			Checkpoint(cancellationToken);
			ComputeRawDescriptors(
				covdet, sift, features[i].Frame, dspMinScale, dspScaleStep, dspNumScales, patch, patchXY, scaledDescriptors);

			if (options.DomainSizePooling)
			{
				PooledMean(scaledDescriptors, dspNumScales, SiftCpuFeatureExtractor.SiftDescriptorDim, descriptor);
			}
			else
			{
				scaledDescriptors.AsSpan(0, SiftCpuFeatureExtractor.SiftDescriptorDim).CopyTo(descriptor);
			}

			switch (options.Normalization)
			{
				case SiftNormalization.L2:
					FeatureUtils.L2NormalizeRow(descriptor);
					break;
				case SiftNormalization.L1Root:
					FeatureUtils.L1RootNormalizeRow(descriptor);
					break;
				default:
					throw new InvalidOperationException("Normalization type not supported");
			}

			FeatureUtils.ToUnsignedByte(descriptor, data.Row(i));
		}

		return SiftCpuFeatureExtractor.TransformVLFeatToUBCFeatureDescriptors(data);
	}
}
