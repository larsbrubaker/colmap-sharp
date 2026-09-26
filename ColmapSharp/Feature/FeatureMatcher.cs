// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureMatcher: colmap/feature/matcher.h and matcher.cc - FeatureMatchingOptions and the
// FeatureMatcher interface (Match, MatchGuided) that the matching controllers drive.
// Neighbors: SiftMatcher.cs (SiftMatchingOptions and the SIFT CPU matcher, the only matcher
// in scope), Scene/TwoViewGeometry.cs (MatchGuided's input and output). Tests:
// FeatureMatcherTests.cs (matcher_test.cc).
//
// Out of scope (docs/LICENSE_AUDIT.md): the GPU matchers (SiftGPU, excluded) and the ONNX
// matchers (LightGlue, ALIKED, LoMa). So FeatureMatchingOptions has no use_gpu / gpu_index
// and no aliked / loma sub-options, and Create throws for the ONNX matcher types. Their
// options would only exist to be rejected.
//
// FeatureMatchingTypeOptions' shared_ptr sub-options with deep-copying copy construction
// become a reference-typed Sift property and Clone(), which deep-copies it.

using ColmapSharp.Scene;

namespace ColmapSharp.Feature;

/// <summary>Port of colmap::FeatureMatchingOptions (the in-scope SIFT CPU fields).</summary>
public sealed class FeatureMatchingOptions
{
	/// <summary>Options of the given matcher type.</summary>
	public FeatureMatchingOptions(FeatureMatcherType type = FeatureMatcherType.SiftBruteForce)
	{
		Type = type;
	}

	/// <summary>The matcher type.</summary>
	public FeatureMatcherType Type { get; set; }

	/// <summary>SIFT matching options (FeatureMatchingTypeOptions::sift).</summary>
	public SiftMatchingOptions Sift { get; set; } = new();

	/// <summary>Number of threads for feature matching and geometric verification (-1: all cores).</summary>
	public int NumThreads { get; set; } = -1;

	/// <summary>Maximum number of matches.</summary>
	public int MaxNumMatches { get; set; } = 32768;

	/// <summary>Whether to perform guided matching.</summary>
	public bool GuidedMatching { get; set; }

	/// <summary>
	/// Skips the geometric verification stage and forwards matches unchanged. This option is
	/// ignored when guided matching is enabled, because guided matching depends on the
	/// two-view geometry produced by geometric verification.
	/// </summary>
	public bool SkipGeometricVerification { get; set; }

	/// <summary>
	/// Whether to perform geometric verification using rig constraints between pairs of
	/// non-trivial frames. If disabled, performs geometric two-view verification for
	/// non-trivial frames without rig constraints. Ignored when SkipGeometricVerification.
	/// </summary>
	public bool RigVerification { get; set; }

	/// <summary>
	/// Whether to skip matching images within the same frame. This is useful for the case of
	/// non-overlapping cameras in a rig.
	/// </summary>
	public bool SkipImagePairsInSameFrame { get; set; }

	/// <summary>
	/// Port of FeatureMatchingOptions::RequiresOpenGL. Only the excluded GPU matcher needs
	/// OpenGL, so every known type answers false.
	/// </summary>
	public bool RequiresOpenGL()
	{
		return Type switch
		{
			FeatureMatcherType.SiftBruteForce or FeatureMatcherType.SiftLightGlue or
			FeatureMatcherType.AlikedBruteForce or FeatureMatcherType.AlikedLightGlue or
			FeatureMatcherType.LomaBruteForce or FeatureMatcherType.LomaB or FeatureMatcherType.LomaB128 or
			FeatureMatcherType.LomaR or FeatureMatcherType.LomaL or FeatureMatcherType.LomaG => false,
			_ => throw new InvalidOperationException($"Unknown feature matcher type: {Type.ToColmapString()}"),
		};
	}

	/// <summary>
	/// Port of FeatureMatchingOptions::Check (CHECK_OPTION_*: false on a violation). The ONNX
	/// matcher types are not available here, so they fail the check.
	/// </summary>
	public bool Check()
	{
		if (!(MaxNumMatches >= 0))
		{
			return false;
		}

		return Type switch
		{
			FeatureMatcherType.SiftBruteForce => Util.Check.NotNull(Sift).Check(),
			_ => false,
		};
	}

	/// <summary>A deep copy (C++ copy construction, which deep-copies the sub-options).</summary>
	public FeatureMatchingOptions Clone()
	{
		var copy = (FeatureMatchingOptions)MemberwiseClone();
		copy.Sift = Sift.Clone();
		return copy;
	}
}

/// <summary>Port of colmap::FeatureMatcher::Image: one image's features as seen by a matcher.</summary>
public sealed class FeatureMatcherImage
{
	/// <summary>
	/// Unique identifier for the image. Allows a matcher to cache some computations per image
	/// in consecutive calls to matching.
	/// </summary>
	public uint ImageId { get; init; } = Util.Types.InvalidImageId;

	/// <summary>The image's camera.</summary>
	public Camera? Camera { get; init; }

	/// <summary>The keypoints (required by guided matching).</summary>
	public IReadOnlyList<FeatureKeypoint>? Keypoints { get; init; }

	/// <summary>The descriptors, one row per keypoint.</summary>
	public FeatureDescriptors? Descriptors { get; init; }
}

/// <summary>Port of colmap::FeatureMatcher.</summary>
public abstract class FeatureMatcher
{
	/// <summary>
	/// Port of FeatureMatcher::Create: the SIFT CPU matcher for SIFT_BRUTEFORCE. The ONNX
	/// matchers (LightGlue, ALIKED, LoMa) are excluded from this port.
	/// </summary>
	public static FeatureMatcher Create(FeatureMatchingOptions options)
	{
		return options.Type switch
		{
			FeatureMatcherType.SiftBruteForce or FeatureMatcherType.SiftLightGlue => SiftFeatureMatchers.CreateSiftFeatureMatcher(options),
			FeatureMatcherType.AlikedBruteForce or FeatureMatcherType.AlikedLightGlue or
			FeatureMatcherType.LomaBruteForce or FeatureMatcherType.LomaB or FeatureMatcherType.LomaB128 or
			FeatureMatcherType.LomaR or FeatureMatcherType.LomaL or FeatureMatcherType.LomaG =>
				throw new NotSupportedException($"Feature matcher {options.Type.ToColmapString()} needs ONNX models, which ColmapSharp does not include."),
			_ => throw new InvalidOperationException($"Unknown feature matcher type: {options.Type.ToColmapString()}"),
		};
	}

	/// <summary>Matches the features of two images into <paramref name="matches"/> (cleared first).</summary>
	public abstract void Match(FeatureMatcherImage image1, FeatureMatcherImage image2, List<FeatureMatch> matches);

	/// <summary>
	/// Matches the features of two images restricted to pairs consistent with the two-view
	/// geometry within <paramref name="maxError"/> pixels, into its InlierMatches.
	/// </summary>
	public abstract void MatchGuided(
		double maxError, FeatureMatcherImage image1, FeatureMatcherImage image2, TwoViewGeometry twoViewGeometry);
}
