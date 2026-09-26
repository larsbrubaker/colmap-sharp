// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AlignedBox3d: axis-aligned 3D box of doubles, the replacement for Eigen::AlignedBox3d
// (Geometry/Bbox.cs, and later the scene's bounding-box queries). Written here to Eigen's
// documented semantics; Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md). Minimal:
// only what COLMAP's callers ported so far use. Tests: ColmapSharp.Tests/Geometry/BboxTests.cs.
//
// Semantics, as Eigen documents them: the default box is empty, with min at +double.MaxValue
// and max at double.MinValue (Eigen's setEmpty uses highest() and lowest()), so extending
// it by any box gives that box. Extend takes the coefficient-wise min and max, and
// Diagonal is max - min. Tier A: no arithmetic beyond one subtraction per coefficient.

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Axis-aligned 3D box. Replacement for Eigen::AlignedBox3d.
/// </summary>
public readonly struct AlignedBox3d : IEquatable<AlignedBox3d>
{
	/// <summary>The minimum corner, Eigen's min().</summary>
	public readonly Vector3d Min;

	/// <summary>The maximum corner, Eigen's max().</summary>
	public readonly Vector3d Max;

	/// <summary>The empty box, like Eigen's default constructor (setEmpty()).</summary>
	public AlignedBox3d()
	{
		Min = new Vector3d(double.MaxValue, double.MaxValue, double.MaxValue);
		Max = new Vector3d(double.MinValue, double.MinValue, double.MinValue);
	}

	/// <summary>The box with the given corners.</summary>
	public AlignedBox3d(Vector3d min, Vector3d max)
	{
		Min = min;
		Max = max;
	}

	/// <summary>
	/// Whether the point lies in the box, borders included: Eigen's contains(p), min &lt;= p
	/// and p &lt;= max coefficient-wise (so a NaN coordinate is never contained).
	/// </summary>
	public bool Contains(Vector3d point) =>
		Min.X <= point.X && Min.Y <= point.Y && Min.Z <= point.Z
		&& point.X <= Max.X && point.Y <= Max.Y && point.Z <= Max.Z;

	/// <summary>max - min, Eigen's diagonal().</summary>
	public Vector3d Diagonal() => Max - Min;

	/// <summary>
	/// The smallest box containing this box and <paramref name="other"/>, Eigen's
	/// extend(box), which takes the coefficient-wise min and max of the corners.
	/// </summary>
	public AlignedBox3d Extend(AlignedBox3d other)
	{
		return new AlignedBox3d(
			new Vector3d(MinOf(Min.X, other.Min.X), MinOf(Min.Y, other.Min.Y), MinOf(Min.Z, other.Min.Z)),
			new Vector3d(MaxOf(Max.X, other.Max.X), MaxOf(Max.Y, other.Max.Y), MaxOf(Max.Z, other.Max.Z)));
	}

	/// <summary>Eigen's operator== on boxes: both corners compare equal.</summary>
	public static bool operator ==(AlignedBox3d a, AlignedBox3d b) => a.Min == b.Min && a.Max == b.Max;

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(AlignedBox3d a, AlignedBox3d b) => !(a == b);

	/// <summary>.NET equality of both corners.</summary>
	public bool Equals(AlignedBox3d other) => Min.Equals(other.Min) && Max.Equals(other.Max);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is AlignedBox3d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => HashCode.Combine(Min, Max);

	/// <inheritdoc/>
	public override string ToString() => $"[min={Min}, max={Max}]";

	// Eigen's cwiseMin/cwiseMax are std::min/std::max per coefficient:
	// min(a, b) is (b < a) ? b : a, max(a, b) is (a < b) ? b : a.
	private static double MinOf(double a, double b) => b < a ? b : a;

	private static double MaxOf(double a, double b) => a < b ? b : a;
}
