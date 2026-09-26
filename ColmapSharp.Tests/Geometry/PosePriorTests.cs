// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PosePriorTests: colmap/geometry/pose_prior_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name. Tests ColmapSharp/Geometry/PosePrior.cs. Tier A
// (exact equality, the exact operator<< string). The C++ static functions
// GravityFromExifOrientation and ComputeRot90FromGravity are static members of PosePrior.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Geometry;

public class PosePriorTests
{
	private static PosePrior NominalPrior() => new()
	{
		PosePriorId = 0,
		CorrDataId = new DataId(new SensorId(SensorType.Camera, 1), 2),
		Position = Vector3d.Zero,
		PositionCovariance = Matrix3d.Identity,
		CoordinateSystem = PosePriorCoordinateSystem.Cartesian,
		Gravity = Vector3d.UnitZ,
	};

	[Test]
	public async Task PosePrior_Equals()
	{
		PosePrior prior = NominalPrior();
		PosePrior other = prior;
		bool equalAtStart = prior == other;
		prior.Position = new Vector3d(1, prior.Position.Y, prior.Position.Z);
		bool notEqual = prior != other;
		other.Position = new Vector3d(1, other.Position.Y, other.Position.Z);
		bool equalAgain = prior == other;
		using (Assert.Multiple())
		{
			await Assert.That(equalAtStart).IsTrue();
			await Assert.That(notEqual).IsTrue();
			await Assert.That(equalAgain).IsTrue();
		}
	}

	[Test]
	public async Task PosePrior_NaNEquals()
	{
		var prior = new PosePrior();
		PosePrior other = prior;
		bool equalAtStart = prior == other;
		prior.Position = new Vector3d(1, 2, double.NaN);
		other.Position = new Vector3d(1, 2, double.NaN);
		bool equalPosition = prior == other;
		prior.PositionCovariance = new Matrix3d(double.NaN, 0, 0, 0, 1, 0, 0, 0, 1);
		other.PositionCovariance = new Matrix3d(double.NaN, 0, 0, 0, 1, 0, 0, 0, 1);
		bool equalCovariance = prior == other;
		other.Position = new Vector3d(other.Position.X, other.Position.Y, 3);
		bool notEqual = prior != other;
		using (Assert.Multiple())
		{
			await Assert.That(equalAtStart).IsTrue();
			await Assert.That(equalPosition).IsTrue();
			await Assert.That(equalCovariance).IsTrue();
			await Assert.That(notEqual).IsTrue();
		}
	}

	[Test]
	public async Task PosePrior_Print()
	{
		PosePrior prior = NominalPrior();
		await Assert.That(prior.ToString()).IsEqualTo(
			"PosePrior(pose_prior_id=0, corr_data_id=(CAMERA, 1, 2), "
			+ "position=[0, 0, 0], "
			+ "position_covariance=[1, 0, 0, 0, 1, 0, 0, 0, 1], "
			+ "coordinate_system=CARTESIAN, gravity=[0, 0, 1])");
	}

	[Test]
	public async Task PosePrior_GravityFromExifOrientation()
	{
		using (Assert.Multiple())
		{
			await Assert.That(PosePrior.GravityFromExifOrientation(1) == new Vector3d(0, 1, 0)).IsTrue();
			await Assert.That(PosePrior.GravityFromExifOrientation(3) == new Vector3d(0, -1, 0)).IsTrue();
			await Assert.That(PosePrior.GravityFromExifOrientation(6) == new Vector3d(1, 0, 0)).IsTrue();
			await Assert.That(PosePrior.GravityFromExifOrientation(8) == new Vector3d(-1, 0, 0)).IsTrue();
			await Assert.That(PosePrior.GravityFromExifOrientation(2).HasValue).IsFalse();
			await Assert.That(PosePrior.GravityFromExifOrientation(4).HasValue).IsFalse();
			await Assert.That(PosePrior.GravityFromExifOrientation(5).HasValue).IsFalse();
			await Assert.That(PosePrior.GravityFromExifOrientation(7).HasValue).IsFalse();
			await Assert.That(PosePrior.GravityFromExifOrientation(0).HasValue).IsFalse();
			await Assert.That(PosePrior.GravityFromExifOrientation(42).HasValue).IsFalse();
			await Assert.That(PosePrior.GravityFromExifOrientation(-1).HasValue).IsFalse();
		}
	}

	[Test]
	public async Task PosePrior_ComputeRot90FromGravity()
	{
		using (Assert.Multiple())
		{
			// Normal.
			await Assert.That(PosePrior.ComputeRot90FromGravity(new Vector3d(0, 1, 0))).IsEqualTo(0);

			// Gravity is +x (right). Need 90 CW (270 CCW) to make upright.
			await Assert.That(PosePrior.ComputeRot90FromGravity(new Vector3d(1, 0, 0))).IsEqualTo(3);

			// Gravity is -y (up). Need 180 CCW to make upright.
			await Assert.That(PosePrior.ComputeRot90FromGravity(new Vector3d(0, -1, 0))).IsEqualTo(2);

			// Gravity is -x (left). Need 90 CCW to make upright.
			await Assert.That(PosePrior.ComputeRot90FromGravity(new Vector3d(-1, 0, 0))).IsEqualTo(1);

			// Robustness to slight inaccuracies.
			await Assert.That(PosePrior.ComputeRot90FromGravity(new Vector3d(0.01, 0.99, 0.1))).IsEqualTo(0);
			await Assert.That(PosePrior.ComputeRot90FromGravity(new Vector3d(0.99, -0.01, 0.1))).IsEqualTo(3);
		}
	}
}
