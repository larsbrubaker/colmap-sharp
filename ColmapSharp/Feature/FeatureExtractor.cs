// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureExtractor: colmap/feature/extractor.h and extractor.cc - FeatureExtractionOptions
// and the FeatureExtractor interface (Extract) that the extraction controller drives.
// Neighbors: Sift.cs (SiftExtractionOptions, the SIFT CPU extractor and
// SiftFeatureExtractors.CreateSiftFeatureExtractor), FeatureMatcher.cs (the matching
// counterpart). Tests: FeatureExtractorTests.cs (extractor_test.cc).
//
// Out of scope (docs/LICENSE_AUDIT.md): the GPU extractor (SiftGPU, excluded) and the ONNX
// extractors (ALIKED, LoMa). So FeatureExtractionOptions has no use_gpu / gpu_index and no
// aliked / loma sub-options, and Create throws for the ONNX extractor types, as
// FeatureMatcher.Create does for the ONNX matchers. The per-type answers that need no
// sub-options (RequiresRGB, RequiresOpenGL, EffMaxImageSize) are kept for every type.
//
// Extract takes a CancellationToken, which COLMAP's interface does not have: MatterCAD must
// be able to cancel a multi-second extraction in the middle of an image. It never changes
// the result of an extraction that runs to completion.
//
// FeatureExtractionTypeOptions' shared_ptr sub-options with deep-copying copy construction
// become a reference-typed Sift property and Clone(), which deep-copies it.

using ColmapSharp.Sensor;

namespace ColmapSharp.Feature;

/// <summary>Port of colmap::FeatureExtractionOptions (the in-scope SIFT CPU fields).</summary>
public sealed class FeatureExtractionOptions
{
	/// <summary>Options of the given extractor type.</summary>
	public FeatureExtractionOptions(FeatureExtractorType type = FeatureExtractorType.Sift)
	{
		Type = type;
	}

	/// <summary>The extractor type.</summary>
	public FeatureExtractorType Type { get; set; }

	/// <summary>SIFT extraction options (FeatureExtractionTypeOptions::sift).</summary>
	public SiftExtractionOptions Sift { get; set; } = new();

	/// <summary>
	/// Maximum image size, otherwise image will be down-scaled. If non-positive, the
	/// appropriate size is selected automatically based on the extractor type.
	/// </summary>
	public int MaxImageSize { get; set; } = -1;

	/// <summary>Number of threads for feature extraction (-1: all cores).</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>Port of FeatureExtractionOptions::RequiresRGB: whether the extractor needs RGB (not grey) images.</summary>
	public bool RequiresRGB()
	{
		return Type switch
		{
			FeatureExtractorType.Sift => false,
			FeatureExtractorType.AlikedN16Rot or FeatureExtractorType.AlikedN32 or
			FeatureExtractorType.LomaB or FeatureExtractorType.LomaB128 => true,
			_ => throw UnknownType(Type),
		};
	}

	/// <summary>
	/// Port of FeatureExtractionOptions::RequiresOpenGL. Only the excluded SiftGPU extractor
	/// (use_gpu without the covariant options) needs OpenGL, so every known type answers false.
	/// </summary>
	public bool RequiresOpenGL()
	{
		return Type switch
		{
			FeatureExtractorType.Sift or
			FeatureExtractorType.AlikedN16Rot or FeatureExtractorType.AlikedN32 or
			FeatureExtractorType.LomaB or FeatureExtractorType.LomaB128 => false,
			_ => throw UnknownType(Type),
		};
	}

	/// <summary>
	/// Port of FeatureExtractionOptions::EffMaxImageSize: <see cref="MaxImageSize"/> when
	/// positive, otherwise the extractor type's default (3200 for SIFT, 1600 for the learned
	/// extractors).
	/// </summary>
	public int EffMaxImageSize()
	{
		if (MaxImageSize > 0)
		{
			return MaxImageSize;
		}

		return Type switch
		{
			FeatureExtractorType.Sift => 3200,
			FeatureExtractorType.AlikedN16Rot or FeatureExtractorType.AlikedN32 or
			FeatureExtractorType.LomaB or FeatureExtractorType.LomaB128 => 1600,
			_ => throw UnknownType(Type),
		};
	}

	/// <summary>
	/// Port of FeatureExtractionOptions::Check (CHECK_OPTION_*: false on a violation). The
	/// ONNX extractor types are not available here, so they fail the check.
	/// </summary>
	public bool Check()
	{
		if (!(EffMaxImageSize() > 0))
		{
			return false;
		}

		return Type switch
		{
			FeatureExtractorType.Sift => Util.Check.NotNull(Sift).Check(),
			_ => false,
		};
	}

	/// <summary>A deep copy (C++ copy construction, which deep-copies the sub-options).</summary>
	public FeatureExtractionOptions Clone()
	{
		var copy = (FeatureExtractionOptions)MemberwiseClone();
		copy.Sift = Sift.Clone();
		return copy;
	}

	// ThrowUnknownFeatureExtractorType (a std::runtime_error in COLMAP).
	internal static InvalidOperationException UnknownType(FeatureExtractorType type) =>
		new($"Unknown feature extractor type: {type.ToColmapString()}");
}

/// <summary>Port of colmap::FeatureExtractor.</summary>
public abstract class FeatureExtractor
{
	/// <summary>
	/// Port of FeatureExtractor::Create: the SIFT CPU extractor for SIFT. The ONNX
	/// extractors (ALIKED, LoMa) are excluded from this port.
	/// </summary>
	public static FeatureExtractor Create(FeatureExtractionOptions options)
	{
		return options.Type switch
		{
			FeatureExtractorType.Sift => SiftFeatureExtractors.CreateSiftFeatureExtractor(options),
			FeatureExtractorType.AlikedN16Rot or FeatureExtractorType.AlikedN32 or
			FeatureExtractorType.LomaB or FeatureExtractorType.LomaB128 =>
				throw new NotSupportedException($"Feature extractor {options.Type.ToColmapString()} needs ONNX models, which ColmapSharp does not include."),
			_ => throw FeatureExtractionOptions.UnknownType(options.Type),
		};
	}

	/// <summary>
	/// Extracts the features of <paramref name="bitmap"/> into <paramref name="keypoints"/>
	/// (replaced) and, when given, <paramref name="descriptors"/>. Returns false when
	/// extraction failed. Throws <see cref="OperationCanceledException"/> when
	/// <paramref name="cancellationToken"/> is cancelled mid-extraction; the outputs are then
	/// left as they were.
	/// </summary>
	public abstract bool Extract(
		Bitmap bitmap,
		List<FeatureKeypoint> keypoints,
		FeatureDescriptors? descriptors,
		CancellationToken cancellationToken = default);
}
