// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Rigid3dStorage: mutable, address-stable backing for a Rigid3d that an optimizer writes in
// place. Not a port of a COLMAP file; it replaces the thing COLMAP gets for free from
// `Rigid3d&`: a Ceres parameter block that points straight into the pose.
// Geometry/Rigid3d.cs is a readonly struct, so a pose owner (Scene/Frame.cs for
// rig_from_world, Sensor/Rig.cs for sensor_from_rig) keeps its pose in one of these for its
// whole lifetime, and Phase 8's bundle adjustment registers Params as a parameter block and
// writes into it directly; the owner's Rigid3d view reads the array back, so no copy-out
// step can be forgotten.
//
// Layout (the one Ceres sees in COLMAP 4.2.0): Params = [qx, qy, qz, qw, tx, ty, tz], which
// is colmap::Rigid3d::params (Eigen's coeffs() memory order for the quaternion, not the
// (w, x, y, z) constructor order). COLMAP registers `rig_from_world.params.data()` and
// `sensor_from_rig.params.data()` as ONE 7-value block each, with the product manifold
// EigenQuaternion x Euclidean<3> (bundle_adjustment_ceres.cc), and its pose cost functors
// (Estimators/CostFunctions/) take that 7-value layout. Rotation and Translation are views
// (ArraySegment slices) into the same array, not separate storage. Round-tripping a Rigid3d
// through Params is exact.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Geometry;

/// <summary>
/// Mutable storage of a <see cref="Rigid3d"/> as one 7-value parameter array
/// [qx, qy, qz, qw, tx, ty, tz] that stays the same object for the storage's lifetime, so an
/// optimizer can hold on to it.
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

	/// <summary>
	/// colmap::Rigid3d::params, [qx, qy, qz, qw, tx, ty, tz]; an optimizer may register it as
	/// one parameter block and write into it.
	/// </summary>
	public double[] Params { get; } = new double[7];

	/// <summary>The quaternion coefficients [qx, qy, qz, qw]: a view of Params[0..4].</summary>
	public ArraySegment<double> Rotation => new(Params, 0, 4);

	/// <summary>The translation [tx, ty, tz]: a view of Params[4..7].</summary>
	public ArraySegment<double> Translation => new(Params, 4, 3);

	/// <summary>The stored transform, read from and written to Params.</summary>
	public Rigid3d Value
	{
		get => new(
			new Quaterniond(Params[3], Params[0], Params[1], Params[2]),
			new Vector3d(Params[4], Params[5], Params[6]));
		set
		{
			Params[0] = value.Rotation.X;
			Params[1] = value.Rotation.Y;
			Params[2] = value.Rotation.Z;
			Params[3] = value.Rotation.W;
			Params[4] = value.Translation.X;
			Params[5] = value.Translation.Y;
			Params[6] = value.Translation.Z;
		}
	}
}
