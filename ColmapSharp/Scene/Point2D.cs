// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Point2D: port of colmap/scene/point2d.h and point2d.cc, one keypoint of an image and the
// 3D point it observes, if any. Image holds a list of these; Point3D/Track hold the reverse
// link. Tests: ColmapSharp.Tests/Scene/Point2DTests.cs (point2d_test.cc 1:1).
//
// Tier A (exact): plain bookkeeping; ToString matches COLMAP's operator<< text.
//
// Translation notes:
// - A mutable struct, like the C++ value type: copies are independent, and a list of them
//   stores no per-point object.
// - Default value: COLMAP's default Point2D has point3D_id = kInvalidPoint3DId, but a C#
//   struct's default (array elements, default(T)) is all-zero bits, which would read as
//   3D point 0. So the id is stored plus one (unchecked): zero bits decode to
//   ulong.MaxValue = kInvalidPoint3DId, and default(Point2D) == new Point2D().

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::Point2D: image coordinates of a keypoint and the id of the 3D point it
/// observes (<see cref="Types.InvalidPoint3DId"/> when it observes none).
/// </summary>
public struct Point2D : IEquatable<Point2D>
{
	// point3D_id + 1, so that the all-zero default decodes to InvalidPoint3DId (header).
	private ulong point3DIdPlusOne;

	/// <summary>A point at (0, 0) observing no 3D point.</summary>
	public Point2D()
	{
	}

	/// <summary>
	/// The image coordinates in pixels, starting at the upper left corner with 0.
	/// </summary>
	public Vector2d Xy { get; set; }

	/// <summary>
	/// The identifier of the observed 3D point, or <see cref="Types.InvalidPoint3DId"/> if the
	/// 2D point is not part of a 3D point track.
	/// </summary>
	public ulong Point3DId
	{
		readonly get => unchecked(point3DIdPlusOne - 1);
		set => point3DIdPlusOne = unchecked(value + 1);
	}

	/// <summary>Whether the 2D point observes a 3D point.</summary>
	public readonly bool HasPoint3D => Point3DId != Types.InvalidPoint3DId;

	/// <summary>Equality of coordinates and 3D point id.</summary>
	public static bool operator ==(Point2D left, Point2D right) => left.Equals(right);

	/// <summary>Inequality of coordinates or 3D point id.</summary>
	public static bool operator !=(Point2D left, Point2D right) => !left.Equals(right);

	/// <inheritdoc/>
	public readonly bool Equals(Point2D other) => Xy == other.Xy && point3DIdPlusOne == other.point3DIdPlusOne;

	/// <inheritdoc/>
	public override readonly bool Equals(object? obj) => obj is Point2D other && Equals(other);

	/// <inheritdoc/>
	public override readonly int GetHashCode() => HashCode.Combine(Xy, point3DIdPlusOne);

	/// <summary>COLMAP's operator&lt;&lt;: "Point2D(xy=[x, y], point3D_id=id)", -1 when invalid.</summary>
	public override readonly string ToString()
	{
		string point3DId = HasPoint3D ? Point3DId.ToString(System.Globalization.CultureInfo.InvariantCulture) : "-1";
		return $"Point2D(xy=[{CppStreamFormat.FormatDouble(Xy.X)}, {CppStreamFormat.FormatDouble(Xy.Y)}], point3D_id={point3DId})";
	}
}
