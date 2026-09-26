// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Point3D: port of colmap/scene/point3d.h and point3d.cc, a triangulated 3D point with its
// mean reprojection error, color and Track of observations (Track.cs). The reconstruction
// keys these by point3D_t. Tests: ColmapSharp.Tests/Scene/Point3DTests.cs (point3d_test.cc
// 1:1).
//
// Tier A (exact): plain bookkeeping; ToString matches COLMAP's operator<< text.
//
// Translation notes:
// - A class, because it owns a Track (a list); C++ copy-assignment becomes Clone(), which
//   deep-copies the track. Equality is COLMAP's value equality via Equals and ==.
// - The position lives in XyzParams, a 3-value array that stays the same object for the
//   point's lifetime: bundle adjustment registers it as a parameter block, the way COLMAP
//   registers point3D.xyz.data(), and the solver writes the result straight into it
//   (Geometry/Rigid3dStorage.cs does the same for poses). Xyz is a view of that array.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::Point3D: position, mean reprojection error, observations and color.
/// </summary>
public sealed class Point3D : IEquatable<Point3D>
{
	/// <summary>The 3D position of the point, read from and written to <see cref="XyzParams"/>.</summary>
	public Vector3d Xyz
	{
		get => new(XyzParams[0], XyzParams[1], XyzParams[2]);
		set
		{
			XyzParams[0] = value.X;
			XyzParams[1] = value.Y;
			XyzParams[2] = value.Z;
		}
	}

	/// <summary>
	/// colmap::Point3D::xyz as [x, y, z]; an optimizer may register it as one parameter block
	/// and write into it (see the file header).
	/// </summary>
	public double[] XyzParams { get; } = new double[3];

	/// <summary>The mean reprojection error in pixels; -1 when not computed.</summary>
	public double Error { get; set; } = -1.0;

	/// <summary>The track of the point as a list of image observations.</summary>
	public Track Track { get; set; } = new();

	/// <summary>The color of the point in the range [0, 255].</summary>
	public Vector3ub Color { get; set; }

	/// <summary>Whether the reprojection error has been computed (error != -1).</summary>
	public bool HasError => Error != -1;

	/// <summary>A deep copy (C++ copy construction).</summary>
	public Point3D Clone()
	{
		return new Point3D { Xyz = Xyz, Error = Error, Track = Track.Clone(), Color = Color };
	}

	/// <summary>Value equality, COLMAP's operator==.</summary>
	public static bool operator ==(Point3D? left, Point3D? right) => left is null ? right is null : left.Equals(right);

	/// <summary>Value inequality.</summary>
	public static bool operator !=(Point3D? left, Point3D? right) => !(left == right);

	/// <inheritdoc/>
	public bool Equals(Point3D? other)
	{
		return other is not null && Xyz == other.Xyz && Color == other.Color && Error == other.Error && Track == other.Track;
	}

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Point3D other && Equals(other);

	/// <summary>Hash of the value; a Point3D is mutable, so do not key a map on one you then change.</summary>
	public override int GetHashCode() => HashCode.Combine(Xyz, Color, Error, Track);

	/// <summary>COLMAP's operator&lt;&lt;: "Point3D(xyz=[x, y, z], track_len=n)".</summary>
	public override string ToString()
	{
		return $"Point3D(xyz=[{CppStreamFormat.FormatDouble(Xyz.X)}, {CppStreamFormat.FormatDouble(Xyz.Y)}, "
			+ $"{CppStreamFormat.FormatDouble(Xyz.Z)}], track_len={Track.Length})";
	}
}
