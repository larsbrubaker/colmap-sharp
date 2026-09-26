// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionMatchers: the ReconstructionEq gmock matcher of
// colmap/scene/reconstruction_matchers.h, as a predicate: EXPECT_THAT(a,
// ReconstructionEq(b)) becomes Assert.That(ReconstructionEq(a, b)).IsTrue(). Test
// infrastructure next to Rigid3dMatchers.cs; first user is Scene/ReconstructionTests.cs.
//
// Semantics, as in COLMAP: the rig, camera, frame, image and 3D point maps compare equal as
// std::unordered_map ==, i.e. the same key set with operator== equal values. The explain
// text ("have different rigs", ...) is returned for failure messages.
//
// Not ported yet: ReconstructionNear, which aligns the reconstructions with
// AlignReconstructionsViaProjCenters and ComputeImageAlignmentError
// (estimators/alignment, Phase 6). Tests: Scene/ReconstructionMatchersTests.cs
// (reconstruction_matchers_test.cc; Reconstruction.Near waits for ReconstructionNear).

using ColmapSharp.Scene;

namespace ColmapSharp.Tests;

/// <summary>
/// Port of COLMAP's ReconstructionEq matcher.
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
