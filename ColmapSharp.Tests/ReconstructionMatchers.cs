// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionMatchers: the ReconstructionEq and ReconstructionNear gmock matchers of
// colmap/scene/reconstruction_matchers.h, as predicates: EXPECT_THAT(a,
// ReconstructionEq(b)) becomes Assert.That(ReconstructionEq(a, b)).IsTrue(). Test
// infrastructure next to Rigid3dMatchers.cs; first user is Scene/ReconstructionTests.cs.
//
// ReconstructionEq, as in COLMAP: the rig, camera, frame, image and 3D point maps compare
// equal as std::unordered_map ==, i.e. the same key set with operator== equal values.
// ReconstructionNear compares object counts and the observation count, optionally aligns
// lhs to rhs with Alignment.AlignReconstructionsViaProjCenters (LO-RANSAC, so it draws from
// the thread PRNG as COLMAP's does), and bounds every common image's rotation and projection
// center error (Alignment.ComputeImageAlignmentError). The explain text ("have different
// rigs", ...) is returned for failure messages. Tests: Scene/ReconstructionMatchersTests.cs
// (reconstruction_matchers_test.cc).

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Tests;

/// <summary>
/// Port of COLMAP's ReconstructionEq and ReconstructionNear matchers.
/// </summary>
internal static class ReconstructionMatchers
{
	/// <summary>ReconstructionEq(rhs) applied to lhs: exact equality of all objects.</summary>
	public static bool ReconstructionEq(Reconstruction lhs, Reconstruction rhs) => ExplainReconstructionEq(lhs, rhs) is null;

	/// <summary>Null when lhs matches ReconstructionEq(rhs), else COLMAP's explanation.</summary>
	public static string? ExplainReconstructionEq(Reconstruction lhs, Reconstruction rhs)
	{
		if (!MapsEqual(lhs.Rigs, rhs.Rigs))
		{
			return " have different rigs";
		}

		if (!MapsEqual(lhs.Cameras, rhs.Cameras))
		{
			return " have different cameras";
		}

		if (!MapsEqual(lhs.Frames, rhs.Frames))
		{
			return " have different frames";
		}

		if (!MapsEqual(lhs.Images, rhs.Images))
		{
			return " have different images";
		}

		if (!MapsEqual(lhs.Points3D, rhs.Points3D))
		{
			return " have different points";
		}

		return null;
	}

	/// <summary>
	/// ReconstructionNear(rhs, ...) applied to lhs: approximate equality of two
	/// reconstructions, optionally after aligning the two reconstruction worlds through
	/// common shared registered images.
	/// </summary>
	public static bool ReconstructionNear(
		Reconstruction lhs,
		Reconstruction rhs,
		double maxRotationErrorDeg = 1e-6,
		double maxProjCenterError = 1e-6,
		double? maxScaleError = null,
		double numObsTolerance = 0.0,
		bool align = true) =>
		ExplainReconstructionNear(lhs, rhs, maxRotationErrorDeg, maxProjCenterError, maxScaleError, numObsTolerance, align) is null;

	/// <summary>Null when lhs matches ReconstructionNear(rhs, ...), else COLMAP's explanation.</summary>
	public static string? ExplainReconstructionNear(
		Reconstruction lhs,
		Reconstruction rhs,
		double maxRotationErrorDeg = 1e-6,
		double maxProjCenterError = 1e-6,
		double? maxScaleError = null,
		double numObsTolerance = 0.0,
		bool align = true)
	{
		// The matcher's constructor checks.
		Check.Ge(maxRotationErrorDeg, 0.0);
		Check.Ge(maxProjCenterError, 0.0);
		if (maxScaleError.HasValue)
		{
			Check.Ge(maxScaleError.Value, 0.0);
		}

		Check.Ge(numObsTolerance, 0.0);

		if (lhs.NumRigs != rhs.NumRigs)
		{
			return $" have different number of rigs: {lhs.NumRigs} vs {rhs.NumRigs}";
		}

		if (lhs.NumCameras != rhs.NumCameras)
		{
			return $" have different number of cameras: {lhs.NumCameras} vs {rhs.NumCameras}";
		}

		if (lhs.NumFrames != rhs.NumFrames)
		{
			return $" have different number of frames: {lhs.NumFrames} vs {rhs.NumFrames}";
		}

		if (lhs.NumImages != rhs.NumImages)
		{
			return $" have different number of images: {lhs.NumImages} vs {rhs.NumImages}";
		}

		if (lhs.NumRegImages != rhs.NumRegImages)
		{
			return $" have different number of registered images: {lhs.NumRegImages} vs {rhs.NumRegImages}";
		}

		long lhsNumObs = lhs.ComputeNumObservations();
		long rhsNumObs = rhs.ComputeNumObservations();
		// With one side zero a ratio is inf (or NaN), which fails the tolerance as in C++.
		if ((lhsNumObs != 0 || rhsNumObs != 0) &&
			Math.Max(Math.Abs(1 - (lhsNumObs / (double)rhsNumObs)), Math.Abs(1 - (rhsNumObs / (double)lhsNumObs))) > numObsTolerance)
		{
			return $" have different number of observations: {lhsNumObs} vs {rhsNumObs}";
		}

		var rhsFromLhs = new Sim3d();
		if (align)
		{
			// std::numeric_limits<double>::epsilon(), so a zero tolerance still aligns.
			if (!Alignment.AlignReconstructionsViaProjCenters(
				lhs,
				rhs,
				maxProjCenterError: Math.Max(maxProjCenterError, 2.220446049250313E-16),
				ref rhsFromLhs))
			{
				return " failed to align";
			}

			if (maxScaleError.HasValue && !(Math.Abs(rhsFromLhs.Scale - 1.0) < maxScaleError.Value))
			{
				return $" have different scale: {rhsFromLhs.Scale}";
			}
		}

		List<ImageAlignmentError> errors = Alignment.ComputeImageAlignmentError(lhs, rhs, rhsFromLhs);
		Check.Eq(errors.Count, rhs.NumImages);
		foreach (ImageAlignmentError error in errors)
		{
			if (error.RotationErrorDeg > maxRotationErrorDeg)
			{
				return $"Image with name {error.ImageName} exceeds rotation error threshold: {error.RotationErrorDeg} vs {maxRotationErrorDeg}";
			}

			if (error.ProjCenterError > maxProjCenterError)
			{
				return $"Image with name {error.ImageName} exceeds projection center error threshold: {error.ProjCenterError} vs {maxProjCenterError}";
			}
		}

		return null;
	}

	private static bool MapsEqual<TKey, TValue>(IReadOnlyDictionary<TKey, TValue> a, IReadOnlyDictionary<TKey, TValue> b)
		where TValue : IEquatable<TValue>
	{
		if (a.Count != b.Count)
		{
			return false;
		}

		foreach (var (key, value) in a)
		{
			if (!b.TryGetValue(key, out TValue? other) || !value.Equals(other))
			{
				return false;
			}
		}

		return true;
	}
}
