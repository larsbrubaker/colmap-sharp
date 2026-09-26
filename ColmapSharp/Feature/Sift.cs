// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Sift: the CPU SIFT extraction half of colmap/feature/sift.h and sift.cc -
// SiftExtractionOptions, TransformVLFeatToUBCFeatureDescriptors and SiftCPUFeatureExtractor
// (VLFeat's SIFT, VLFeat/VlSiftFilter*.cs, driven octave by octave). Neighbors:
// FeatureUtils.cs (descriptor normalization and quantization), Sensor/Bitmap.cs (the grey
// input image). Tests: ColmapSharp.Tests/Feature/SiftTests.cs (sift_test.cc) and
// SiftOracleTests.cs (C#-only, pycolmap fixtures).
//
// Not here yet: CovariantSiftCPUFeatureExtractor (estimate_affine_shape, domain-size
// pooling, force_covariant_extractor) needs VLFeat's covdet, which is not ported, so this
// extractor rejects those options exactly like COLMAP's SiftCPUFeatureExtractor does; the
// CreateSiftFeatureExtractor factory and FeatureExtractionOptions land with
// feature/extractor. The matching half of sift.cc is SiftMatcher.cs. SiftGPU is excluded
// (docs/LICENSE_AUDIT.md).
//
// Tier A (exact) against unfused VLFeat: the VLFeat layer matches VLFeat compiled without
// FMA contraction bit for bit (VlSiftFilterTests). The macOS arm64 pycolmap wheel fuses
// multiply-adds, so against it the same keypoints come out with last-bit drift
// (SiftOracleTests, docs/CPP_DIVERGENCES.md entry 41). The descriptor normalization sums in
// Eigen's NEON reduction order (FeatureUtils.cs).

using ColmapSharp.Feature.VLFeat;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Feature;

/// <summary>Port of SiftExtractionOptions::Normalization.</summary>
public enum SiftNormalization
{
	/// <summary>
	/// L1_ROOT: L1-normalizes each descriptor followed by element-wise square rooting. This
	/// normalization is usually better than standard L2-normalization. See "Three things
	/// everyone should know to improve object retrieval", Relja Arandjelovic and Andrew
	/// Zisserman, CVPR 2012.
	/// </summary>
	L1Root = 0,

	/// <summary>L2: each vector is L2-normalized.</summary>
	L2 = 1,
}

/// <summary>Port of colmap::SiftExtractionOptions.</summary>
public sealed class SiftExtractionOptions
{
	/// <summary>Maximum number of features to detect, keeping larger-scale features.</summary>
	public int MaxNumFeatures { get; set; } = 8192;

	/// <summary>First octave in the pyramid, i.e. -1 upsamples the image by one level.</summary>
	public int FirstOctave { get; set; } = -1;

	/// <summary>Number of octaves.</summary>
	public int NumOctaves { get; set; } = 4;

	/// <summary>Number of levels per octave.</summary>
	public int OctaveResolution { get; set; } = 3;

	/// <summary>Peak threshold for detection.</summary>
	public double PeakThreshold { get; set; } = 0.02 / 3;

	/// <summary>Edge threshold for detection.</summary>
	public double EdgeThreshold { get; set; } = 10.0;

	/// <summary>
	/// Estimate affine shape of SIFT features in the form of oriented ellipses as opposed to
	/// original SIFT which estimates oriented disks (covariant extractor only).
	/// </summary>
	public bool EstimateAffineShape { get; set; }

	/// <summary>Maximum number of orientations per keypoint if not estimate_affine_shape.</summary>
	public int MaxNumOrientations { get; set; } = 2;

	/// <summary>Fix the orientation to 0 for upright features.</summary>
	public bool Upright { get; set; }

	/// <summary>
	/// Whether to adapt the feature detection depending on the image darkness. Only
	/// available in the OpenGL SiftGPU version, which is excluded, so it has no effect.
	/// </summary>
	public bool DarknessAdaptivity { get; set; }

	/// <summary>
	/// Domain-size pooling: an average SIFT descriptor across multiple scales around the
	/// detected scale ("Domain-Size Pooling in Local Descriptors and Network Architectures",
	/// J. Dong and S. Soatto, CVPR 2015). Covariant extractor only.
	/// </summary>
	public bool DomainSizePooling { get; set; }

	/// <summary>Smallest domain-size pooling scale.</summary>
	public double DspMinScale { get; set; } = 1.0 / 6.0;

	/// <summary>Largest domain-size pooling scale.</summary>
	public double DspMaxScale { get; set; } = 3.0;

	/// <summary>Number of domain-size pooling scales.</summary>
	public int DspNumScales { get; set; } = 10;

	/// <summary>Whether to force usage of the covariant VLFeat implementation.</summary>
	public bool ForceCovariantExtractor { get; set; }

	/// <summary>Descriptor normalization.</summary>
	public SiftNormalization Normalization { get; set; } = SiftNormalization.L1Root;

	/// <summary>Port of SiftExtractionOptions::Check (CHECK_OPTION_*: false on a violation).</summary>
	public bool Check()
	{
		if (!(MaxNumFeatures > 0) || !(OctaveResolution > 0) || !(PeakThreshold > 0.0) ||
			!(EdgeThreshold > 0.0) || !(MaxNumOrientations > 0))
		{
			return false;
		}

		if (DomainSizePooling)
		{
			if (!(DspMinScale > 0) || !(DspMaxScale >= DspMinScale) || !(DspNumScales > 0))
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>A copy (C++ copy construction).</summary>
	public SiftExtractionOptions Clone() => (SiftExtractionOptions)MemberwiseClone();
}

/// <summary>
/// Port of colmap's SiftCPUFeatureExtractor: VLFeat SIFT on a grey bitmap. One instance can
/// extract from many images on one thread; it keeps its VLFeat filter while the image size
/// stays the same.
/// </summary>
public sealed class SiftCpuFeatureExtractor
{
	/// <summary>kSiftDescriptorDim.</summary>
	public const int SiftDescriptorDim = 128;

	private readonly SiftExtractionOptions options;
	private VlSiftFilter? sift;

	/// <summary>Creates the extractor; the options must pass Check and not ask for the covariant extractor.</summary>
	public SiftCpuFeatureExtractor(SiftExtractionOptions options)
	{
		this.options = options.Clone();
		Util.Check.That(this.options.Check());
		Util.Check.That(!this.options.EstimateAffineShape);
		Util.Check.That(!this.options.DomainSizePooling);
		Util.Check.That(!this.options.ForceCovariantExtractor);
	}

	/// <summary>
	/// Port of SiftCPUFeatureExtractor::Extract: detects keypoints (and, when
	/// <paramref name="descriptors"/> is given, computes their descriptors) in a grey bitmap.
	/// Keypoints come out octave by octave, DoG level by level; when there are more than
	/// max_num_features, whole lowest (finest) DoG levels are dropped.
	/// </summary>
	public bool Extract(Bitmap bitmap, List<FeatureKeypoint> keypoints, FeatureDescriptors? descriptors)
	{
		Util.Check.That(bitmap.IsGrey);
		Util.Check.NotNull(keypoints);

		if (sift == null || sift.Width != bitmap.Width || sift.Height != bitmap.Height)
		{
			sift = new VlSiftFilter(
				bitmap.Width, bitmap.Height, options.NumOctaves, options.OctaveResolution, options.FirstOctave);
		}

		sift.PeakThreshold = options.PeakThreshold;
		sift.EdgeThreshold = options.EdgeThreshold;

		// Iterate through octaves.
		var levelNumFeatures = new List<int>();
		var levelKeypoints = new List<List<FeatureKeypoint>>();
		var levelDescriptors = new List<List<byte[]>>();
		Span<double> angles = stackalloc double[4];
		var desc = new float[SiftDescriptorDim];
		bool firstOctave = true;
		while (true)
		{
			if (firstOctave)
			{
				byte[] dataUint8 = bitmap.RowMajorData;
				var dataFloat = new float[dataUint8.Length];
				for (int i = 0; i < dataUint8.Length; ++i)
				{
					dataFloat[i] = (float)dataUint8[i] / 255.0f;
				}

				if (!sift.ProcessFirstOctave(dataFloat))
				{
					break;
				}

				firstOctave = false;
			}
			else if (!sift.ProcessNextOctave())
			{
				break;
			}

			// Detect keypoints.
			sift.Detect();

			// Extract detected keypoints.
			ReadOnlySpan<VlSiftKeypoint> vlKeypoints = sift.Keypoints;
			int numKeypoints = vlKeypoints.Length;
			if (numKeypoints == 0)
			{
				continue;
			}

			// Extract features with different orientations per DOG level.
			int prevLevel = -1;
			for (int i = 0; i < numKeypoints; ++i)
			{
				if (vlKeypoints[i].IS != prevLevel)
				{
					// Add containers for new DOG level.
					levelNumFeatures.Add(0);
					levelKeypoints.Add(new List<FeatureKeypoint>());
					levelDescriptors.Add(new List<byte[]>());
				}

				levelNumFeatures[^1] += 1;
				prevLevel = vlKeypoints[i].IS;

				// Extract feature orientations.
				int numOrientations;
				if (options.Upright)
				{
					numOrientations = 1;
					angles[0] = 0.0;
				}
				else
				{
					numOrientations = sift.CalcKeypointOrientations(angles, vlKeypoints[i]);
				}

				// Note that this is different from SiftGPU, which selects the top global
				// maxima as orientations while this selects the first two local maxima. It
				// is not clear which procedure is better.
				int numUsedOrientations = Math.Min(numOrientations, options.MaxNumOrientations);

				for (int o = 0; o < numUsedOrientations; ++o)
				{
					levelKeypoints[^1].Add(new FeatureKeypoint(
						vlKeypoints[i].X + 0.5f,
						vlKeypoints[i].Y + 0.5f,
						vlKeypoints[i].Sigma,
						(float)angles[o]));
					if (descriptors != null)
					{
						// COLMAP reuses one 1x128 buffer and normalizes it in place. When
						// VLFeat rejects a keypoint (out of bounds) it returns without
						// writing, so that keypoint gets the previous descriptor, already
						// normalized, normalized a second time. desc is reused the same way.
						sift.CalcKeypointDescriptor(desc, vlKeypoints[i], angles[o]);
						levelDescriptors[^1].Add(NormalizeAndQuantize(desc));
					}
				}
			}
		}

		// Determine how many DOG levels to keep to satisfy max_num_features option.
		int firstLevelToKeep = 0;
		int numFeatures = 0;
		int numFeaturesWithOrientations = 0;
		for (int i = levelKeypoints.Count - 1; i >= 0; --i)
		{
			numFeatures += levelNumFeatures[i];
			numFeaturesWithOrientations += levelKeypoints[i].Count;
			if (numFeatures > options.MaxNumFeatures)
			{
				firstLevelToKeep = i;
				break;
			}
		}

		// Extract the features to be kept.
		keypoints.Clear();
		keypoints.Capacity = Math.Max(keypoints.Capacity, numFeaturesWithOrientations);
		for (int i = firstLevelToKeep; i < levelKeypoints.Count; ++i)
		{
			keypoints.AddRange(levelKeypoints[i]);
		}

		// Compute the descriptors for the detected keypoints.
		if (descriptors != null)
		{
			var data = new RowMajorMatrix<byte>(numFeaturesWithOrientations, SiftDescriptorDim);
			int k = 0;
			for (int i = firstLevelToKeep; i < levelKeypoints.Count; ++i)
			{
				foreach (byte[] row in levelDescriptors[i])
				{
					row.CopyTo(data.Row(k));
					k += 1;
				}
			}

			descriptors.Data = TransformVLFeatToUBCFeatureDescriptors(data);
			descriptors.Type = FeatureExtractorType.Sift;
		}

		return true;
	}

	/// <summary>
	/// Port of TransformVLFeatToUBCFeatureDescriptors: VLFeat stores the 8 orientation bins
	/// of each spatial cell in the opposite rotational order from the original SIFT (UBC)
	/// format that SiftGPU also uses; this reorders them.
	/// </summary>
	public static RowMajorMatrix<byte> TransformVLFeatToUBCFeatureDescriptors(RowMajorMatrix<byte> vlfeatDescriptors)
	{
		var ubc = new RowMajorMatrix<byte>(vlfeatDescriptors.Rows, vlfeatDescriptors.Cols);
		ReadOnlySpan<int> q = [0, 7, 6, 5, 4, 3, 2, 1];
		for (int n = 0; n < vlfeatDescriptors.Rows; ++n)
		{
			for (int i = 0; i < 4; ++i)
			{
				for (int j = 0; j < 4; ++j)
				{
					for (int k = 0; k < 8; ++k)
					{
						ubc[n, (8 * (j + (4 * i))) + q[k]] = vlfeatDescriptors[n, (8 * (j + (4 * i))) + k];
					}
				}
			}
		}

		return ubc;
	}

	// The normalization and FeatureDescriptorsToUnsignedByte of one descriptor. desc is
	// normalized in place, like COLMAP's reused 1x128 matrix.
	private byte[] NormalizeAndQuantize(float[] desc)
	{
		switch (options.Normalization)
		{
			case SiftNormalization.L2:
				FeatureUtils.L2NormalizeRow(desc);
				break;
			case SiftNormalization.L1Root:
				FeatureUtils.L1RootNormalizeRow(desc);
				break;
			default:
				throw new InvalidOperationException("Normalization type not supported");
		}

		var row = new byte[SiftDescriptorDim];
		FeatureUtils.ToUnsignedByte(desc, row);
		return row;
	}
}
