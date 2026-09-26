// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureMatcherTests: colmap/feature/matcher_test.cc. Tests ColmapSharp/Feature/FeatureMatcher.cs.
//
// The ONNX matchers (LightGlue, ALIKED, LoMa) are excluded, so FeatureMatchingOptions has no
// aliked / loma sub-options:
// - Copy and CopyAssignment are ported without their options.aliked->brute_force.min_cossim
//   lines (skipped: ALIKED options, ONNX).
// - RequiresOpenGL is ported over all ten types (it needs no matcher).
// - Check is ported for SIFT_BRUTEFORCE only; the other nine types (SIFT_LIGHTGLUE and the
//   ALIKED / LoMa types) are skipped: they need ONNX sub-options, and Check rejects them here.

using ColmapSharp.Feature;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Feature;

public class FeatureMatcherTests
{
	private static readonly FeatureMatcherType[] AllTypes =
	[
		FeatureMatcherType.SiftBruteForce,
		FeatureMatcherType.SiftLightGlue,
		FeatureMatcherType.AlikedBruteForce,
		FeatureMatcherType.AlikedLightGlue,
		FeatureMatcherType.LomaBruteForce,
		FeatureMatcherType.LomaB,
		FeatureMatcherType.LomaB128,
		FeatureMatcherType.LomaR,
		FeatureMatcherType.LomaL,
		FeatureMatcherType.LomaG,
	];

	[Test]
	public async Task FeatureMatchingOptions_Copy()
	{
		var options = new FeatureMatchingOptions();
		options.MaxNumMatches += 100;
		options.Sift.MaxRatio *= 0.1;

		FeatureMatchingOptions copy = options.Clone();

		// Verify fields are copied
		await Assert.That(copy.MaxNumMatches).IsEqualTo(options.MaxNumMatches);
		await Assert.That(copy.Sift.MaxRatio).IsEqualTo(options.Sift.MaxRatio);

		// Verify deep copy of shared_ptr (different pointer instances)
		await Assert.That(ReferenceEquals(options.Sift, copy.Sift)).IsFalse();
	}

	[Test]
	public async Task FeatureMatchingOptions_CopyAssignment()
	{
		var options = new FeatureMatchingOptions();
		options.MaxNumMatches += 100;
		options.Sift.MaxRatio *= 0.1;

		var assigned = new FeatureMatchingOptions();
		assigned = options.Clone();

		// Verify fields are copied.
		await Assert.That(assigned.MaxNumMatches).IsEqualTo(options.MaxNumMatches);
		await Assert.That(assigned.Sift.MaxRatio).IsEqualTo(options.Sift.MaxRatio);

		// Verify deep copy (different pointer instances).
		await Assert.That(ReferenceEquals(options.Sift, assigned.Sift)).IsFalse();

		// Test self-assignment: a C# reference assignment to itself keeps the same instances.
		SiftMatchingOptions siftPtr = options.Sift;
		int prevMaxNumMatches = options.MaxNumMatches;
		FeatureMatchingOptions selfRef = options;
		options = selfRef;
		await Assert.That(ReferenceEquals(options.Sift, siftPtr)).IsTrue();
		await Assert.That(options.MaxNumMatches).IsEqualTo(prevMaxNumMatches);
	}

	[Test]
	public async Task FeatureMatchingOptions_RequiresOpenGL()
	{
		var options = new FeatureMatchingOptions();
		foreach (FeatureMatcherType type in AllTypes)
		{
			options.Type = type;
			await Assert.That(options.RequiresOpenGL()).IsFalse();
		}
	}

	[Test]
	public async Task FeatureMatchingOptions_Check()
	{
		var options = new FeatureMatchingOptions();
		options.Type = FeatureMatcherType.SiftBruteForce;
		await Assert.That(options.Check()).IsTrue();
	}
}
