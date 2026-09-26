// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureMatch: the FeatureMatch struct of colmap/feature/types.h, a pair of feature indices
// matched between two images. FeatureMatches (std::vector<FeatureMatch>) is List<FeatureMatch>.
// The first piece of feature/types.h to be ported, because Scene/TwoViewGeometry and
// Scene/CorrespondenceGraph need it; keypoints, descriptors and the matrix conversions join
// it with the feature phase. Tests: ColmapSharp.Tests/Feature/FeatureTypesTests.cs
// (FeatureMatches.Nominal of feature/types_test.cc).
//
// Tier A (exact): plain bookkeeping.
//
// Default value: COLMAP's default constructor sets both indices to kInvalidPoint2DIdx, and
// std::vector<FeatureMatch>(n) value-initializes n such matches. A C# struct's default
// (array elements, default(T)) is all-zero bits instead, which would silently read as the
// valid match (0, 0). So each index is stored plus one (unchecked): zero bits decode to
// uint.MaxValue = kInvalidPoint2DIdx, and default(FeatureMatch) == new FeatureMatch().
// Never reinterpret the memory (MemoryMarshal.Cast to uint pairs, raw file or matrix I/O):
// the ids are stored +1, so always convert through the properties. The same holds for
// TrackElement, CorrespondenceGraph.Correspondence and Point2D, which use the same encoding.

using ColmapSharp.Util;

namespace ColmapSharp.Feature;

/// <summary>
/// Port of colmap::FeatureMatch: feature <see cref="Point2DIdx1"/> in the first image matches
/// feature <see cref="Point2DIdx2"/> in the second. Defaults to two invalid indices.
/// </summary>
public struct FeatureMatch : IEquatable<FeatureMatch>
{
	// Index + 1, so that the all-zero default decodes to InvalidPoint2DIdx (header).
	private uint point2DIdx1PlusOne;
	private uint point2DIdx2PlusOne;

	/// <summary>A match between two invalid indices, like COLMAP's default constructor.</summary>
	public FeatureMatch()
	{
	}

	/// <summary>A match between the given feature indices.</summary>
	public FeatureMatch(uint point2DIdx1, uint point2DIdx2)
	{
		Point2DIdx1 = point2DIdx1;
		Point2DIdx2 = point2DIdx2;
	}

	/// <summary>Feature index in the first image (point2D_idx1).</summary>
	public uint Point2DIdx1
	{
		readonly get => unchecked(point2DIdx1PlusOne - 1);
		set => point2DIdx1PlusOne = unchecked(value + 1);
	}

	/// <summary>Feature index in the second image (point2D_idx2).</summary>
	public uint Point2DIdx2
	{
		readonly get => unchecked(point2DIdx2PlusOne - 1);
		set => point2DIdx2PlusOne = unchecked(value + 1);
	}

	/// <summary>Equality of both indices.</summary>
	public static bool operator ==(FeatureMatch left, FeatureMatch right) => left.Equals(right);

	/// <summary>Inequality of either index.</summary>
	public static bool operator !=(FeatureMatch left, FeatureMatch right) => !left.Equals(right);

	/// <inheritdoc/>
	public readonly bool Equals(FeatureMatch other)
	{
		return point2DIdx1PlusOne == other.point2DIdx1PlusOne && point2DIdx2PlusOne == other.point2DIdx2PlusOne;
	}

	/// <inheritdoc/>
	public override readonly bool Equals(object? obj) => obj is FeatureMatch other && Equals(other);

	/// <inheritdoc/>
	public override readonly int GetHashCode() => PairHash.Instance.GetHashCode((Point2DIdx1, Point2DIdx2));

	/// <inheritdoc/>
	public override readonly string ToString() => $"FeatureMatch({Point2DIdx1}, {Point2DIdx2})";
}
