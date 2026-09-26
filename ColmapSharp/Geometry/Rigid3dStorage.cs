// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Rigid3dStorage: mutable, address-stable backing for a Rigid3d that an optimizer writes in
// place. Not a port of a COLMAP file; it replaces the thing COLMAP gets for free from
// `Rigid3d&`: Ceres parameter blocks that point straight into a pose's quaternion
// coefficients and translation (bundle_adjustment_ceres.cc passes
// `rig_from_world.rotation().coeffs().data()` and `rig_from_world.translation().data()`).
// Geometry/Rigid3d.cs is a readonly struct, so a pose owner (Scene/Frame.cs) keeps its pose
// in one of these for its whole lifetime, and Phase 8's bundle adjustment registers the two
// arrays as parameter blocks and writes into them directly; the owner's Rigid3d view reads
// the arrays back, so no copy-out step can be forgotten.
//
// Layout (the one Ceres sees in COLMAP): Rotation = [qx, qy, qz, qw] (Eigen's coeffs()
// memory order, not the (w, x, y, z) constructor order), Translation = [tx, ty, tz]. Two
// arrays rather than one double[7] because COLMAP uses two parameter blocks per pose (the
// quaternion one carries a manifold). Round-tripping a Rigid3d through the arrays is exact.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Geometry;

/// <summary>
/// Mutable storage of a <see cref="Rigid3d"/> as two parameter arrays that stay the same
/// objects for the storage's lifetime, so an optimizer can hold on to them.
/// </summary>
public sealed class Rigid3dStorage
{
	/// <summary>Creates storage holding the identity transform.</summary>
	public Rigid3dStorage()
	{
		Value = new Rigid3d();
	}

	/// <summary>Creates storage holding <paramref name="value"/>.</summary>
	public Rigid3dStorage(Rigid3d value)
	{
		Value = value;
	}

	/// <summary>Quaternion coefficients [qx, qy, qz, qw]; an optimizer may write into it.</summary>
	public double[] Rotation { get; } = new double[4];

	/// <summary>Translation [tx, ty, tz]; an optimizer may write into it.</summary>
	public double[] Translation { get; } = new double[3];

	/// <summary>The stored transform, read from and written to the arrays.</summary>
	public Rigid3d Value
	{
		get => new(
			new Quaterniond(Rotation[3], Rotation[0], Rotation[1], Rotation[2]),
			new Vector3d(Translation[0], Translation[1], Translation[2]));
		set
		{
			Rotation[0] = value.Rotation.X;
			Rotation[1] = value.Rotation.Y;
			Rotation[2] = value.Rotation.Z;
			Rotation[3] = value.Rotation.W;
			Translation[0] = value.Translation.X;
			Translation[1] = value.Translation.Y;
			Translation[2] = value.Translation.Z;
		}
	}
}
