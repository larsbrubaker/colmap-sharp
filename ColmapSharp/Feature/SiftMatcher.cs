// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// SiftMatcher: the CPU matching half of colmap/feature/sift.h and sift.cc -
// SiftMatchingOptions, FindBestMatches{OneWay}{BruteForce,Index}, the guided filters,
// UseEssentialMatrixForGuidedMatching, SiftCPUFeatureMatcher and CreateSiftFeatureMatcher.
// Neighbors: SiftMatchKernels.cs (the fused integer distance scan), FeatureDescriptorIndex.cs
// (the k-NN index of the default, non brute-force path), FeatureMatcher.cs (the interface),
// Sift.cs (the extractor half of sift.cc). Tests: SiftMatcherTests.cs (sift_test.cc's CPU
// matcher cases).
//
// Tier A (exact) given the same nearest neighbours: every distance is an integer (see
// SiftMatchKernels.cs) and the ratio / distance tests are COLMAP's float expressions
// (MathF.Acos / MathF.Sqrt on the same floats). The guided filters evaluate COLMAP's float
// (F, H) and double (E) expressions in Eigen's left-to-right order without FMA; Apple clang
// may contract them, which can only flip a pair lying exactly on the max_error boundary.
//
// Two differences in how, not what:
// - COLMAP's brute-force and guided paths fill an N x M float matrix, then scan rows and the
//   transposed matrix. Here one fused integer scan yields both directions (identical
//   results, SiftMatchKernels.cs header), parallel over rows, deterministic.
// - The default path searches FeatureDescriptorIndex, which is exact here (faiss is
//   approximate for 512+ descriptors; divergence 42).
// The GPU matcher (SiftGPU) and SIFT LightGlue (ONNX) are excluded.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Feature;

/// <summary>Port of colmap::SiftMatchingOptions (minus the ONNX LightGlue options).</summary>
public sealed class SiftMatchingOptions
{
	/// <summary>Maximum distance ratio between first and second best match.</summary>
	public double MaxRatio { get; set; } = 0.8;

	/// <summary>Maximum distance to best match.</summary>
	public double MaxDistance { get; set; } = 0.7;

	/// <summary>Whether to enable cross checking in matching.</summary>
	public bool CrossCheck { get; set; } = true;

	/// <summary>Whether to use brute-force instead of index based CPU matching.</summary>
	public bool CpuBruteForceMatcher { get; set; }

	/// <summary>
	/// Cache for reusing descriptor index for feature matching. Required unless
	/// <see cref="CpuBruteForceMatcher"/>. Shared, not copied, by <see cref="Clone"/>.
	/// </summary>
	public ThreadSafeLRUCache<uint, FeatureDescriptorIndex>? CpuDescriptorIndexCache { get; set; }

	/// <summary>Port of SiftMatchingOptions::Check (CHECK_OPTION_*: false on a violation).</summary>
	public bool Check() => MaxRatio > 0.0 && MaxDistance > 0.0;

	/// <summary>A copy (C++ copy construction; the cache pointer is shared).</summary>
	public SiftMatchingOptions Clone() => (SiftMatchingOptions)MemberwiseClone();
}

/// <summary>Port of the SIFT matcher factory of colmap/feature/sift.h.</summary>
public static class SiftFeatureMatchers
{
	/// <summary>
	/// Port of CreateSiftFeatureMatcher: the SIFT CPU matcher. SIFT_LIGHTGLUE (ONNX) and the
	/// GPU matcher are excluded from this port.
	/// </summary>
	public static FeatureMatcher CreateSiftFeatureMatcher(FeatureMatchingOptions options)
	{
		return options.Type switch
		{
			FeatureMatcherType.SiftBruteForce => new SiftCpuFeatureMatcher(options),
			FeatureMatcherType.SiftLightGlue =>
				throw new NotSupportedException("SIFT LightGlue matching needs ONNX models, which ColmapSharp does not include."),
			_ => throw new InvalidOperationException($"Unknown feature matcher type: {options.Type.ToColmapString()}"),
		};
	}
}

/// <summary>Port of colmap's SiftCPUFeatureMatcher.</summary>
public sealed class SiftCpuFeatureMatcher : FeatureMatcher
{
	private readonly FeatureMatchingOptions options;
	private uint prevImageId1 = Types.InvalidImageId;
	private uint prevImageId2 = Types.InvalidImageId;
	private FeatureDescriptorIndex? index1;
	private FeatureDescriptorIndex? index2;

	/// <summary>A matcher with a copy of <paramref name="options"/>, which must pass Check.</summary>
	public SiftCpuFeatureMatcher(FeatureMatchingOptions options)
	{
		this.options = options.Clone();
		Util.Check.That(this.options.Check());
	}

	/// <inheritdoc/>
	public override void Match(FeatureMatcherImage image1, FeatureMatcherImage image2, List<FeatureMatch> matches)
	{
		Util.Check.NotNull(matches);
		ThrowCheckFeatureTypesMatch(image1, image2);

		matches.Clear();
		FetchIndexes(image1, image2);

		RowMajorMatrix<byte> d1 = image1.Descriptors!.Data;
		RowMajorMatrix<byte> d2 = image2.Descriptors!.Data;
		if (d1.Rows == 0 || d2.Rows == 0)
		{
			return;
		}

		SiftMatchingOptions sift = options.Sift;
		if (sift.CpuBruteForceMatcher)
		{
			var rows = new Top2[d1.Rows];
			Top2[]? cols = sift.CrossCheck ? new Top2[d2.Rows] : null;
			SiftMatchKernels.ScanDistances(d1, d2, useL2: false, guidedFilter: null, options.NumThreads, rows, cols);
			CollectMatches(
				OneWayBruteForce(rows, (float)sift.MaxRatio, (float)sift.MaxDistance),
				cols is null ? null : OneWayBruteForce(cols, (float)sift.MaxRatio, (float)sift.MaxDistance),
				matches);
			return;
		}

		RowMajorMatrix<int> indices1To2 = new(0, 0);
		RowMajorMatrix<float> l2Dists1To2 = new(0, 0);
		RowMajorMatrix<int> indices2To1 = new(0, 0);
		RowMajorMatrix<float> l2Dists2To1 = new(0, 0);

		Util.Check.NotNull(index2).Search(2, image1.Descriptors.ToFloat(), ref indices1To2, ref l2Dists1To2);
		if (sift.CrossCheck)
		{
			Util.Check.NotNull(index1).Search(2, image2.Descriptors.ToFloat(), ref indices2To1, ref l2Dists2To1);
		}

		FindBestMatchesIndex(
			indices1To2, l2Dists1To2, indices2To1, l2Dists2To1, (float)sift.MaxRatio, (float)sift.MaxDistance, sift.CrossCheck, matches);
	}

	/// <inheritdoc/>
	public override void MatchGuided(
		double maxError, FeatureMatcherImage image1, FeatureMatcherImage image2, TwoViewGeometry twoViewGeometry)
	{
		Util.Check.NotNull(twoViewGeometry);
		ThrowCheckFeatureTypesMatch(image1, image2, checkKeypoints: true);

		twoViewGeometry.InlierMatches.Clear();
		FetchIndexes(image1, image2);

		Func<int, int, bool>? guidedFilter = SiftGuidedFilters.Create(maxError, image1, image2, twoViewGeometry);
		if (guidedFilter is null)
		{
			return;
		}

		// The guided filter indexes per-feature geometry (bearings with Jacobians, or the
		// keypoints themselves) by descriptor row, so the two must align.
		Util.Check.Eq(image1.Keypoints!.Count, image1.Descriptors!.Data.Rows);
		Util.Check.Eq(image2.Keypoints!.Count, image2.Descriptors!.Data.Rows);

		// COLMAP scans the full guided L2 matrix with FindBestMatchesIndex over every column
		// (indices LinSpaced), in both directions.
		RowMajorMatrix<byte> d1 = image1.Descriptors.Data;
		RowMajorMatrix<byte> d2 = image2.Descriptors.Data;
		var rows = new Top2[d1.Rows];
		var cols = new Top2[d2.Rows];
		SiftMatchKernels.ScanDistances(d1, d2, useL2: true, guidedFilter, options.NumThreads, rows, cols);

		SiftMatchingOptions sift = options.Sift;
		float maxRatio = (float)sift.MaxRatio;
		float maxDistance = (float)sift.MaxDistance;
		bool crossCheck = sift.CrossCheck && d2.Rows != 0;
		CollectMatches(
			OneWayIndex(rows, maxRatio, maxDistance),
			crossCheck ? OneWayIndex(cols, maxRatio, maxDistance) : null,
			twoViewGeometry.InlierMatches);
	}

	/// <summary>
	/// Port of FindBestMatchesIndex: the ratio / distance tests on each query's neighbours
	/// (<paramref name="indices2To1"/> is only used for cross-checking when it has rows).
	/// </summary>
	internal static void FindBestMatchesIndex(
		RowMajorMatrix<int> indices1To2,
		RowMajorMatrix<float> l2Dists1To2,
		RowMajorMatrix<int> indices2To1,
		RowMajorMatrix<float> l2Dists2To1,
		float maxRatio,
		float maxDistance,
		bool crossCheck,
		List<FeatureMatch> matches)
	{
		matches.Clear();
		int[] matches1To2 = FindBestMatchesOneWayIndex(indices1To2, l2Dists1To2, maxRatio, maxDistance);
		int[]? matches2To1 = crossCheck && indices2To1.Rows != 0
			? FindBestMatchesOneWayIndex(indices2To1, l2Dists2To1, maxRatio, maxDistance)
			: null;
		CollectMatches(matches1To2, matches2To1, matches);
	}

	// Port of FindBestMatchesOneWayIndex over explicit neighbour lists.
	private static int[] FindBestMatchesOneWayIndex(
		RowMajorMatrix<int> indices, RowMajorMatrix<float> l2Dists, float maxRatio, float maxDistance)
	{
		var result = new int[indices.Rows];
		for (int d1Idx = 0; d1Idx < indices.Rows; ++d1Idx)
		{
			int bestD2Idx = -1;
			float bestL2Dist = float.MaxValue;
			float secondBestL2Dist = float.MaxValue;
			for (int nIdx = 0; nIdx < indices.Cols; ++nIdx)
			{
				int d2Idx = indices[d1Idx, nIdx];
				float l2Dist = l2Dists[d1Idx, nIdx];
				if (l2Dist < bestL2Dist)
				{
					bestD2Idx = d2Idx;
					secondBestL2Dist = bestL2Dist;
					bestL2Dist = l2Dist;
				}
				else if (l2Dist < secondBestL2Dist)
				{
					secondBestL2Dist = l2Dist;
				}
			}

			result[d1Idx] = PassesIndexTests(bestD2Idx, bestL2Dist, secondBestL2Dist, maxRatio, maxDistance) ? bestD2Idx : -1;
		}

		return result;
	}

	// FindBestMatchesOneWayIndex on the fused scan's states (keys are negated L2 distances).
	private static int[] OneWayIndex(Top2[] states, float maxRatio, float maxDistance)
	{
		var result = new int[states.Length];
		for (int i = 0; i < states.Length; ++i)
		{
			Top2 s = states[i];
			float best = s.BestIdx == -1 ? float.MaxValue : -s.Best;
			float second = s.Second == -int.MaxValue ? float.MaxValue : -s.Second;
			result[i] = PassesIndexTests(s.BestIdx, best, second, maxRatio, maxDistance) ? s.BestIdx : -1;
		}

		return result;
	}

	private static bool PassesIndexTests(int bestD2Idx, float bestL2Dist, float secondBestL2Dist, float maxRatio, float maxDistance)
	{
		// Check if any match found.
		if (bestD2Idx == -1)
		{
			return false;
		}

		// Check if match distance passes threshold (kSqSiftDescriptorNorm * max_distance *
		// max_distance, evaluated in float).
		float maxL2Dist = SiftMatchKernels.SqSiftDescriptorNorm * maxDistance * maxDistance;
		if (bestL2Dist > maxL2Dist)
		{
			return false;
		}

		// Check if match passes ratio test. Keep this comparison >= in order to ensure that the
		// case of best == second_best is detected.
		return !(MathF.Sqrt(bestL2Dist) >= maxRatio * MathF.Sqrt(secondBestL2Dist));
	}

	// Port of FindBestMatchesOneWayBruteForce's tests on the fused scan's dot-product states.
	private static int[] OneWayBruteForce(Top2[] states, float maxRatio, float maxDistance)
	{
		const float kInvSqDescriptorNorm = (float)(1.0 / SiftMatchKernels.SqSiftDescriptorNorm);
		var result = new int[states.Length];
		for (int i = 0; i < states.Length; ++i)
		{
			Top2 s = states[i];
			result[i] = -1;

			// Check if any match found.
			if (s.BestIdx == -1)
			{
				continue;
			}

			// Convert to L2 distance in which the thresholds are defined.
			float bestDistNormed = MathF.Acos(MathF.Min(kInvSqDescriptorNorm * s.Best, 1.0f));

			// Check if match distance passes threshold.
			if (bestDistNormed > maxDistance)
			{
				continue;
			}

			float secondBestDistNormed = MathF.Acos(MathF.Min(kInvSqDescriptorNorm * s.Second, 1.0f));

			// Check if match passes ratio test. Keep this comparison >= in order to ensure that
			// the case of best == second_best is detected.
			if (bestDistNormed >= maxRatio * secondBestDistNormed)
			{
				continue;
			}

			result[i] = s.BestIdx;
		}

		return result;
	}

	// The cross-check (or one-way) assembly shared by FindBestMatchesBruteForce and
	// FindBestMatchesIndex; matches are in ascending point2D_idx1 order.
	private static void CollectMatches(int[] matches1To2, int[]? matches2To1, List<FeatureMatch> matches)
	{
		matches.Clear();
		for (int i1 = 0; i1 < matches1To2.Length; ++i1)
		{
			int i2 = matches1To2[i1];
			if (i2 == -1)
			{
				continue;
			}

			if (matches2To1 is null || matches2To1[i2] == i1)
			{
				matches.Add(new FeatureMatch((uint)i1, (uint)i2));
			}
		}
	}

	private void FetchIndexes(FeatureMatcherImage image1, FeatureMatcherImage image2)
	{
		SiftMatchingOptions sift = options.Sift;
		if (!sift.CpuBruteForceMatcher && (prevImageId1 == Types.InvalidImageId || prevImageId1 != image1.ImageId))
		{
			index1 = Util.Check.NotNull(sift.CpuDescriptorIndexCache).Get(image1.ImageId);
			prevImageId1 = image1.ImageId;
		}

		if (!sift.CpuBruteForceMatcher && (prevImageId2 == Types.InvalidImageId || prevImageId2 != image2.ImageId))
		{
			index2 = Util.Check.NotNull(sift.CpuDescriptorIndexCache).Get(image2.ImageId);
			prevImageId2 = image2.ImageId;
		}
	}

	private static void ThrowCheckFeatureTypesMatch(FeatureMatcherImage image1, FeatureMatcherImage image2, bool checkKeypoints = false)
	{
		Util.Check.NotNull(image1.Descriptors);
		Util.Check.NotNull(image2.Descriptors);
		Util.Check.That(image1.Descriptors.Type == FeatureExtractorType.Sift);
		Util.Check.That(image2.Descriptors.Type == FeatureExtractorType.Sift);
		Util.Check.Eq(image1.Descriptors.Data.Cols, SiftMatchKernels.Dim);
		Util.Check.Eq(image2.Descriptors.Data.Cols, SiftMatchKernels.Dim);
		if (checkKeypoints)
		{
			Util.Check.NotNull(image1.Camera);
			Util.Check.NotNull(image2.Camera);
			Util.Check.NotNull(image1.Keypoints);
			Util.Check.NotNull(image2.Keypoints);
			Util.Check.Eq(image1.Descriptors.Data.Rows, image1.Keypoints.Count);
			Util.Check.Eq(image2.Descriptors.Data.Rows, image2.Keypoints.Count);
		}
	}
}
