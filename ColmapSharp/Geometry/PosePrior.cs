// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PosePrior: colmap/geometry/pose_prior.h/.cc - a measured prior on a sensor's position
// (e.g. GPS, in WGS84 or Cartesian coordinates, with covariance) and gravity direction,
// its NaN-aware equality and operator<<, and the EXIF-orientation / gravity helpers that
// decide how many quarter turns make an image upright. Uses the sensor ids of
// Util/Types.cs. Tests: ColmapSharp.Tests/Geometry/PosePriorTests.cs
// (pose_prior_test.cc 1:1). Scalar code, Tier A.
//
// PosePrior is a mutable struct, like the C++ struct it ports, so assignment copies it.
// Its parameterless constructor sets COLMAP's defaults (NaN position, covariance and
// gravity; invalid ids; UNDEFINED coordinate system); default(PosePrior) skips that
// constructor and has zeros, so always create one with new PosePrior().
// COLMAP's LOG(WARNING)/LOG(ERROR) for unsupported EXIF orientations have no logging
// counterpart here; the returned null carries the outcome.

using System.Text;

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Geometry;

/// <summary>
/// The coordinate system of a pose prior's position. Port of
/// colmap::PosePrior::CoordinateSystem (MAKE_ENUM_CLASS, start -1).
/// </summary>
public enum PosePriorCoordinateSystem
{
	/// <summary>UNDEFINED.</summary>
	Undefined = -1,

	/// <summary>WGS84: latitude, longitude, altitude.</summary>
	Wgs84 = 0,

	/// <summary>CARTESIAN.</summary>
	Cartesian = 1,
}

/// <summary>
/// A prior on a sensor's pose. Port of colmap::PosePrior.
/// </summary>
public struct PosePrior : IEquatable<PosePrior>
{
	/// <summary>kInvalidPosePriorId.</summary>
	public const uint InvalidPosePriorId = Types.InvalidPosePriorId;

	/// <summary>A prior with COLMAP's defaults: nothing known.</summary>
	public PosePrior()
	{
		PosePriorId = InvalidPosePriorId;
		CorrDataId = Types.InvalidDataId;
		Position = new Vector3d(double.NaN, double.NaN, double.NaN);
		PositionCovariance = Matrix3d.Ones * double.NaN;
		CoordinateSystem = PosePriorCoordinateSystem.Undefined;
		Gravity = new Vector3d(double.NaN, double.NaN, double.NaN);
	}

	/// <summary>The unique identifier of the pose prior.</summary>
	public uint PosePriorId { get; set; }

	/// <summary>
	/// The identifier of the associated sensor for which this prior defines the
	/// pose. For example, this can refer to a camera or an IMU sensor.
	/// </summary>
	public DataId CorrDataId { get; set; }

	/// <summary>The position of the associated sensor in the world coordinate system.</summary>
	public Vector3d Position { get; set; }

	/// <summary>The position covariance in the Cartesian world coordinate system.</summary>
	public Matrix3d PositionCovariance { get; set; }

	/// <summary>The coordinate system of the position in the world.</summary>
	public PosePriorCoordinateSystem CoordinateSystem { get; set; }

	/// <summary>The gravity (down) in the sensor coordinate system.</summary>
	public Vector3d Gravity { get; set; }

	/// <summary>Whether every position coordinate is finite.</summary>
	public readonly bool HasPosition() => AllFinite(Position);

	/// <summary>Whether every covariance entry is finite.</summary>
	public readonly bool HasPositionCov()
	{
		for (int i = 0; i < 3; ++i)
		{
			if (!AllFinite(PositionCovariance.Row(i)))
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>Whether every gravity coordinate is finite.</summary>
	public readonly bool HasGravity() => AllFinite(Gravity);

	/// <summary>Equality with NaN == NaN in the vectors and matrix, as COLMAP defines it.</summary>
	public static bool operator ==(PosePrior left, PosePrior right) => left.Equals(right);

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(PosePrior left, PosePrior right) => !left.Equals(right);

	/// <inheritdoc/>
	public readonly bool Equals(PosePrior other)
	{
		return PosePriorId == other.PosePriorId
			&& CorrDataId == other.CorrDataId
			&& CoordinateSystem == other.CoordinateSystem
			&& IsNaNEqual(Position, other.Position)
			&& IsNaNEqual(PositionCovariance.Row(0), other.PositionCovariance.Row(0))
			&& IsNaNEqual(PositionCovariance.Row(1), other.PositionCovariance.Row(1))
			&& IsNaNEqual(PositionCovariance.Row(2), other.PositionCovariance.Row(2))
			&& IsNaNEqual(Gravity, other.Gravity);
	}

	/// <inheritdoc/>
	public override readonly bool Equals(object? obj) => obj is PosePrior other && Equals(other);

	/// <inheritdoc/>
	public override readonly int GetHashCode() => HashCode.Combine(PosePriorId, CorrDataId, CoordinateSystem);

	/// <summary>
	/// COLMAP's operator&lt;&lt;: e.g. "PosePrior(pose_prior_id=0, corr_data_id=(CAMERA, 1, 2),
	/// position=[0, 0, 0], position_covariance=[1, 0, 0, 0, 1, 0, 0, 0, 1],
	/// coordinate_system=CARTESIAN, gravity=[0, 0, 1])", numbers as a default ostream
	/// prints them and the matrix row by row.
	/// </summary>
	public override readonly string ToString()
	{
		var builder = new StringBuilder();
		builder.Append("PosePrior(pose_prior_id=").Append(PosePriorId)
			.Append(", corr_data_id=(").Append(CorrDataId.SensorId.Type.ToColmapString())
			.Append(", ").Append(CorrDataId.SensorId.Id)
			.Append(", ").Append(CorrDataId.Id)
			.Append("), position=[").Append(Format(Position))
			.Append("], position_covariance=[")
			.Append(Format(PositionCovariance.Row(0))).Append(", ")
			.Append(Format(PositionCovariance.Row(1))).Append(", ")
			.Append(Format(PositionCovariance.Row(2)))
			.Append("], coordinate_system=").Append(CoordinateSystemToString(CoordinateSystem))
			.Append(", gravity=[").Append(Format(Gravity)).Append("])");
		return builder.ToString();
	}

	/// <summary>COLMAP's CoordinateSystemToString: "UNDEFINED", "WGS84" or "CARTESIAN".</summary>
	public static string CoordinateSystemToString(PosePriorCoordinateSystem coordinateSystem) => coordinateSystem switch
	{
		PosePriorCoordinateSystem.Undefined => "UNDEFINED",
		PosePriorCoordinateSystem.Wgs84 => "WGS84",
		PosePriorCoordinateSystem.Cartesian => "CARTESIAN",
		_ => throw new ArgumentOutOfRangeException(nameof(coordinateSystem)),
	};

	/// <summary>
	/// The gravity direction in image coordinates for the four upright-or-rotated EXIF
	/// orientations; null for mirrored (2, 4, 5, 7) and unknown orientations.
	/// Port of colmap::GravityFromExifOrientation.
	/// </summary>
	public static Vector3d? GravityFromExifOrientation(int orientation) => orientation switch
	{
		1 => new Vector3d(0, 1, 0), // Normal
		3 => new Vector3d(0, -1, 0), // Rotate 180
		6 => new Vector3d(1, 0, 0), // Rotate 90 CW
		8 => new Vector3d(-1, 0, 0), // Rotate 270 CW
		_ => null,
	};

	/// <summary>
	/// The number of 90 degree counter-clockwise rotations (0-3) that make an image
	/// upright, i.e. bring its gravity to +y. Port of colmap::ComputeRot90FromGravity.
	/// </summary>
	public static int ComputeRot90FromGravity(Vector3d gravity)
	{
		// Calculate the angle of gravity in image space, then find number of 90 deg
		// CCW rotations needed to make the image upright (where gravity is at pi/2).
		double angle = Math.Atan2(gravity.Y, gravity.X);
		const double HalfPi = Math.PI / 2.0;

		// std::round rounds halfway cases away from zero; C#'s default is to even.
		int rot90Ccw = (int)Math.Round((angle - HalfPi) / HalfPi, MidpointRounding.AwayFromZero) % 4;
		if (rot90Ccw < 0)
		{
			rot90Ccw += 4;
		}

		return rot90Ccw;
	}

	private static bool AllFinite(Vector3d v) => double.IsFinite(v.X) && double.IsFinite(v.Y) && double.IsFinite(v.Z);

	// Handle NaNs explicitly and consider them equal, whereas the default C++
	// comparison operator returns false for a NaN == NaN comparison.
	private static bool IsNaNEqual(Vector3d left, Vector3d right)
	{
		for (int i = 0; i < 3; ++i)
		{
			if (double.IsNaN(left[i]) != double.IsNaN(right[i]) || (!double.IsNaN(left[i]) && left[i] != right[i]))
			{
				return false;
			}
		}

		return true;
	}

	private static string Format(Vector3d v) =>
		$"{CppStreamFormat.FormatDouble(v.X)}, {CppStreamFormat.FormatDouble(v.Y)}, {CppStreamFormat.FormatDouble(v.Z)}";
}
