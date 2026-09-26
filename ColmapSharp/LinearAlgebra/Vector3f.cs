// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Vector3f: fixed-size 3-vector of floats, the replacement for Eigen::Vector3f where COLMAP
// keeps single-precision positions (mvs/delaunay_meshing's input points and camera centers).
// Only storage and equality: the float arithmetic that uses it is written out at the call
// site, with its summation order stated there. Written to Eigen's documented semantics;
// Eigen is not ported (docs/LICENSE_AUDIT.md). Sibling of Vector3d.

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Fixed-size 3-vector of floats. Replacement for Eigen::Vector3f.
/// </summary>
public readonly struct Vector3f : IEquatable<Vector3f>
{
	/// <summary>First coefficient, Eigen's x().</summary>
	public readonly float X;

	/// <summary>Second coefficient, Eigen's y().</summary>
	public readonly float Y;

	/// <summary>Third coefficient, Eigen's z().</summary>
	public readonly float Z;

	/// <summary>Creates the vector (x, y, z).</summary>
	public Vector3f(float x, float y, float z)
	{
		X = x;
		Y = y;
		Z = z;
	}

	/// <summary>Coefficient-wise equality.</summary>
	public static bool operator ==(Vector3f left, Vector3f right) => left.Equals(right);

	/// <summary>Coefficient-wise inequality.</summary>
	public static bool operator !=(Vector3f left, Vector3f right) => !left.Equals(right);

	/// <inheritdoc/>
	public bool Equals(Vector3f other) => X.Equals(other.X) && Y.Equals(other.Y) && Z.Equals(other.Z);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Vector3f other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => HashCode.Combine(X, Y, Z);

	/// <inheritdoc/>
	public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"({X:R}, {Y:R}, {Z:R})");
}
