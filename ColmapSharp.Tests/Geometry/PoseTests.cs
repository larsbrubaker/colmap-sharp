// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PoseTests: colmap/geometry/pose_test.cc ported 1:1, one method per gtest TEST(Suite,
// Name) named Suite_Name, same checks and tolerances. Tests ColmapSharp/Geometry/Pose.cs.
// Tier B (SVD, QR, slerp) where the function goes through a decomposition; COLMAP's
// tolerances are the bar. The comparisons against pycolmap are in
// GeometryTwoViewOracleTests (C#-only).
//
// Each test seeds the PRNG with 0, as COLMAP's gtest_main does, and draws everything
// before its first await (the PRNG is per thread). Loops collect their per-iteration
// results and assert them together.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Geometry;

public class PoseTests
{
	private static Rigid3d RandomRigid3d()
	{
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d();
		return new Rigid3d(rotation, translation);
	}

	private static Vector3d RandomAngleAxisVector()
	{
		AngleAxisd aa = AngleAxisd.FromQuaternion(RandomEigen.RandomEigenQuaterniond());
		return aa.Angle * aa.Axis;
	}

	private static MatrixXd Columns(params Vector3d[] columns)
	{
		var m = new MatrixXd(3, columns.Length);
		for (int i = 0; i < columns.Length; ++i)
		{
			m.SetCol(i, VectorXd.From(columns[i]));
		}

		return m;
	}

	[Test]
	public async Task ComputeClosestRotationMatrix_Nominal()
	{
		Matrix3d a = Matrix3d.Identity;
		using (Assert.Multiple())
		{
			await Assert.That(EigenMatrixNear(Pose.ComputeClosestRotationMatrix(a), a, 1e-6)).IsTrue();
			await Assert.That(EigenMatrixNear(Pose.ComputeClosestRotationMatrix(2 * a), a, 1e-6)).IsTrue();
		}
	}

	[Test]
	public async Task DecomposeProjectionMatrix_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		var failures = new List<int>();
		for (int i = 1; i < 100; ++i)
		{
			var refK = new Matrix3d(i, 0, i, 0, i, 2 * i, 0, 0, i);
			Rigid3d camFromWorld = RandomRigid3d();
			Matrix3x4d p = refK * camFromWorld.ToMatrix();
			Pose.DecomposeProjectionMatrix(p, out Matrix3d k, out Matrix3d r, out Vector3d t);
			if (!EigenMatrixNear(refK, k, 1e-6)
				|| !EigenMatrixNear(camFromWorld.Rotation.ToRotationMatrix(), r, 1e-6)
				|| !EigenMatrixNear(camFromWorld.Translation, t, 1e-6))
			{
				failures.Add(i);
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task RotationMatrixToAngleAxis_Roundtrip()
	{
		RandomUtils.SetPRNGSeed(0);
		Matrix3d r = RandomEigen.RandomEigenQuaterniond().ToRotationMatrix();
		await Assert.That(EigenMatrixNear(Pose.AngleAxisToRotationMatrix(Pose.RotationMatrixToAngleAxis(r)), r, 1e-6)).IsTrue();
	}

	[Test]
	public async Task AngleAxisToRotationMatrix_Roundtrip()
	{
		RandomUtils.SetPRNGSeed(0);
		Vector3d w = RandomAngleAxisVector();
		await Assert.That(EigenMatrixNear(Pose.RotationMatrixToAngleAxis(Pose.AngleAxisToRotationMatrix(w)), w, 1e-6)).IsTrue();
	}

	private static async Task CheckEulerRoundtrip(double rx, double ry, double rz)
	{
		(double rxx, double ryy, double rzz) = Pose.RotationMatrixToEulerAngles(Pose.EulerAnglesToRotationMatrix(rx, ry, rz));
		using (Assert.Multiple())
		{
			await Assert.That(rxx).IsEqualTo(rx).Within(1e-6);
			await Assert.That(ryy).IsEqualTo(ry).Within(1e-6);
			await Assert.That(rzz).IsEqualTo(rz).Within(1e-6);
		}
	}

	[Test]
	public async Task EulerAngles_X() => await CheckEulerRoundtrip(0.3, 0, 0);

	[Test]
	public async Task EulerAngles_Y() => await CheckEulerRoundtrip(0, 0.3, 0);

	[Test]
	public async Task EulerAngles_Z() => await CheckEulerRoundtrip(0, 0, 0.3);

	[Test]
	public async Task EulerAngles_XYZ() => await CheckEulerRoundtrip(0.1, 0.2, 0.3);

	[Test]
	public async Task AverageQuaternions_Nominal()
	{
		Vector4d identity = Quaterniond.Identity.Coeffs;
		using (Assert.Multiple())
		{
			await Assert.That(Pose.AverageQuaternions([Quaterniond.Identity], [1.0]).Coeffs == identity).IsTrue();
			await Assert.That(Pose.AverageQuaternions([Quaterniond.Identity], [2.0]).Coeffs == identity).IsTrue();
			await Assert.That(Pose.AverageQuaternions([Quaterniond.Identity, Quaterniond.Identity], [1.0, 1.0]).Coeffs == identity).IsTrue();
			await Assert.That(Pose.AverageQuaternions([Quaterniond.Identity, Quaterniond.Identity], [1.0, 2.0]).Coeffs == identity).IsTrue();
			await Assert.That(Pose.AverageQuaternions([Quaterniond.Identity, new Quaterniond(2, 0, 0, 0)], [1.0, 2.0]).Coeffs == identity).IsTrue();
			await Assert.That(EigenMatrixNear(
				Pose.AverageQuaternions([Quaterniond.Identity, new Quaterniond(1, 1, 0, 0)], [1.0, 1.0]).Coeffs,
				new Quaterniond(0.92388, 0.382683, 0, 0).Coeffs,
				1e-6)).IsTrue();
			await Assert.That(EigenMatrixNear(
				Pose.AverageQuaternions([Quaterniond.Identity, new Quaterniond(1, 1, 0, 0)], [1.0, 2.0]).Coeffs,
				new Quaterniond(0.850651, 0.525731, 0, 0).Coeffs,
				1e-6)).IsTrue();
		}
	}

	[Test]
	public async Task InterpolateCameraPoses_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		Rigid3d camFromWorld1 = RandomRigid3d();
		Rigid3d camFromWorld2 = RandomRigid3d();

		Rigid3d interp1 = Pose.InterpolateCameraPoses(camFromWorld1, camFromWorld2, 0);
		Rigid3d interp2 = Pose.InterpolateCameraPoses(camFromWorld1, camFromWorld2, 1);
		Rigid3d interp3 = Pose.InterpolateCameraPoses(camFromWorld1, camFromWorld2, 0.5);
		using (Assert.Multiple())
		{
			await Assert.That(EigenMatrixNear(interp1.Translation, camFromWorld1.Translation)).IsTrue();
			await Assert.That(EigenMatrixNear(interp2.Translation, camFromWorld2.Translation)).IsTrue();
			await Assert.That(EigenMatrixNear(
				interp3.Translation, (camFromWorld1.Translation + camFromWorld2.Translation) / 2)).IsTrue();
		}
	}

	[Test]
	public async Task CheckCheirality_Nominal()
	{
		var cam2FromCam1 = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0));
		var rays1 = new List<Vector3d>();
		var rays2 = new List<Vector3d>();
		var validIndices = new List<int>();

		rays1.Add(new Vector3d(0, 0, 1).Normalized());
		rays2.Add(new Vector3d(0.1, 0, 1).Normalized());
		bool result1 = Pose.CheckCheirality(cam2FromCam1, rays1, rays2, validIndices);
		int count1 = validIndices.Count;

		rays1.Add(new Vector3d(0, 0, 1).Normalized());
		rays2.Add(new Vector3d(-0.1, 0, 1).Normalized());
		bool result2 = Pose.CheckCheirality(cam2FromCam1, rays1, rays2, validIndices);
		int count2 = validIndices.Count;

		// rays2[1][0] = 0.2 (the component only, the ray is not renormalized).
		rays2[1] = new Vector3d(0.2, rays2[1].Y, rays2[1].Z);
		bool result3 = Pose.CheckCheirality(cam2FromCam1, rays1, rays2, validIndices);
		int count3 = validIndices.Count;

		rays2[0] = new Vector3d(-0.2, rays2[0].Y, rays2[0].Z);
		rays2[1] = new Vector3d(-0.2, rays2[1].Y, rays2[1].Z);
		bool result4 = Pose.CheckCheirality(cam2FromCam1, rays1, rays2, validIndices);
		int count4 = validIndices.Count;

		using (Assert.Multiple())
		{
			await Assert.That(result1).IsTrue();
			await Assert.That(count1).IsEqualTo(1);
			await Assert.That(result2).IsTrue();
			await Assert.That(count2).IsEqualTo(1);
			await Assert.That(result3).IsTrue();
			await Assert.That(count3).IsEqualTo(2);
			await Assert.That(result4).IsFalse();
			await Assert.That(count4).IsEqualTo(0);
		}
	}

	[Test]
	public async Task AverageUnitVectors_Nominal()
	{
		VectorXd avg1 = Pose.AverageUnitVectors(Columns(Vector3d.UnitX));
		VectorXd avg2 = Pose.AverageUnitVectors(Columns(Vector3d.UnitZ, Vector3d.UnitZ));
		MatrixXd xy = Columns(Vector3d.UnitX, Vector3d.UnitY);
		VectorXd avg3 = Pose.AverageUnitVectors(xy);
		VectorXd avg4 = Pose.AverageUnitVectors(xy, new VectorXd([3.0, 1.0]));
		VectorXd avg5 = Pose.AverageUnitVectors(Columns(
			new Vector3d(1, 0.1, 0).Normalized(),
			new Vector3d(1, -0.1, 0).Normalized(),
			new Vector3d(1, 0, 0.1).Normalized(),
			new Vector3d(-1, 0, 0)));
		MatrixXd vectors4d = MatrixXd.FromColumnMajor(4, 2, [1, 0, 0, 0, 0, 1, 0, 0]);
		VectorXd avg6 = Pose.AverageUnitVectors(vectors4d);
		using (Assert.Multiple())
		{
			await Assert.That(EigenMatrixNear(avg1.ToVector3d(), Vector3d.UnitX, 1e-6)).IsTrue();
			await Assert.That(EigenMatrixNear(avg2.ToVector3d(), Vector3d.UnitZ, 1e-6)).IsTrue();
			await Assert.That(avg3.Length).IsEqualTo(3);
			await Assert.That(avg3.Norm()).IsEqualTo(1.0).Within(1e-6);
			await Assert.That(avg4[0]).IsGreaterThan(avg4[1]);
			await Assert.That(avg5[0]).IsGreaterThan(0);
			await Assert.That(avg6.Length).IsEqualTo(4);
			await Assert.That(avg6.Norm()).IsEqualTo(1.0).Within(1e-6);
		}
	}

	[Test]
	public async Task AverageDirections_Nominal()
	{
		Vector3d[] vectors = [new Vector3d(1, 0.1, 0), new Vector3d(1, -0.1, 0)];
		Vector3d avg1 = Pose.AverageDirections(vectors);
		Vector3d avg2 = Pose.AverageDirections(vectors, [1.0, 1.0]);
		using (Assert.Multiple())
		{
			await Assert.That(EigenMatrixNear(avg1, Vector3d.UnitX, 1e-6)).IsTrue();
			await Assert.That(EigenMatrixNear(avg2, Vector3d.UnitX, 1e-6)).IsTrue();
		}
	}

	[Test]
	public async Task GravityAlignedRotation_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		Vector3d gravity1 = new Vector3d(0.5, 0.5, 0.5).Normalized();
		Matrix3d r1 = Pose.GravityAlignedRotation(gravity1);
		Matrix3d r2 = Pose.GravityAlignedRotation(Vector3d.UnitY);

		var failures = new List<int>();
		for (int i = 0; i < 100; ++i)
		{
			Vector3d gravity = RandomEigen.RandomEigenVector3d().Normalized();
			Matrix3d r = Pose.GravityAlignedRotation(gravity);
			if (!EigenMatrixNear(r.Transpose() * r, Matrix3d.Identity, 1e-6) || Math.Abs(r.Determinant() - 1.0) > 1e-6)
			{
				failures.Add(i);
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(EigenMatrixNear(r1.Col(1), gravity1, 1e-6)).IsTrue();
			await Assert.That(EigenMatrixNear(r2.Col(1), Vector3d.UnitY, 1e-6)).IsTrue();
			await Assert.That(r2.Determinant()).IsEqualTo(1.0).Within(1e-6);
			await Assert.That(failures).IsEmpty();
		}
	}

	[Test]
	public async Task YAxisAngleFromRotation_Roundtrip()
	{
		RandomUtils.SetPRNGSeed(0);
		var failures = new List<int>();
		for (int i = 0; i < 100; ++i)
		{
			double angle = RandomUtils.RandomUniformReal(-Math.PI, Math.PI);
			Matrix3d r = Pose.RotationFromYAxisAngle(angle);
			double recoveredAngle = Pose.YAxisAngleFromRotation(r);
			if (!EigenMatrixNear(Pose.RotationFromYAxisAngle(recoveredAngle), r, 1e-6))
			{
				failures.Add(i);
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task RotationFromYAxisAngle_Nominal()
	{
		Matrix3d r0 = Pose.RotationFromYAxisAngle(0);
		Matrix3d r90 = Pose.RotationFromYAxisAngle(Math.PI / 2);
		using (Assert.Multiple())
		{
			await Assert.That(EigenMatrixNear(r0, Matrix3d.Identity, 1e-6)).IsTrue();
			await Assert.That(Pose.YAxisAngleFromRotation(r0)).IsEqualTo(0.0).Within(1e-6);
			await Assert.That(EigenMatrixNear(r90 * Vector3d.UnitX, -Vector3d.UnitZ, 1e-6)).IsTrue();
		}
	}

	[Test]
	public async Task QuaternionFromAngleAxis_Zero()
	{
		Quaterniond q = Pose.QuaternionFromAngleAxis(Vector3d.Zero);
		using (Assert.Multiple())
		{
			await Assert.That(q.W).IsEqualTo(1.0).Within(1e-12);
			await Assert.That(q.Vec.Norm).IsEqualTo(0.0).Within(1e-12);
		}
	}

	[Test]
	public async Task QuaternionFromAngleAxis_SmallAngle()
	{
		// Just above and below the threshold - results should be continuous.
		Vector3d axis = new Vector3d(1, 2, 3).Normalized();
		Quaterniond qSmall = Pose.QuaternionFromAngleAxis(axis * 1e-11);
		Quaterniond qMedium = Pose.QuaternionFromAngleAxis(axis * 1e-9);

		// Small angle should NOT snap to identity - it should preserve direction.
		Vector3d recovered = AngleAxisd.FromQuaternion(qSmall).Axis.Normalized();
		using (Assert.Multiple())
		{
			// Both should be near identity but preserve direction.
			await Assert.That(qSmall.AngularDistance(qMedium)).IsEqualTo(0.0).Within(1e-8);
			await Assert.That(Math.Abs(recovered.Dot(axis))).IsEqualTo(1.0).Within(1e-6);
		}
	}

	[Test]
	public async Task QuaternionFromAngleAxis_Roundtrip()
	{
		RandomUtils.SetPRNGSeed(0);
		var failures = new List<int>();
		for (int i = 0; i < 100; ++i)
		{
			AngleAxisd aa = AngleAxisd.FromQuaternion(RandomEigen.RandomEigenQuaterniond());
			Quaterniond q = Pose.QuaternionFromAngleAxis(aa.Angle * aa.Axis);
			if (!EigenMatrixNear(q.ToRotationMatrix(), aa.ToRotationMatrix(), 1e-10))
			{
				failures.Add(i);
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task LeftJacobianFromAngleAxis_IdentityAtZero()
	{
		Matrix3d jl = Pose.LeftJacobianFromAngleAxis(Vector3d.Zero);
		await Assert.That(EigenMatrixNear(jl, Matrix3d.Identity, 1e-10)).IsTrue();
	}

	[Test]
	public async Task LeftJacobianFromAngleAxis_RelationToRight()
	{
		// Jr(w) = Jl(-w) for all w.
		RandomUtils.SetPRNGSeed(0);
		var failures = new List<int>();
		for (int i = 0; i < 100; ++i)
		{
			Vector3d omega = RandomAngleAxisVector();
			if (!EigenMatrixNear(Pose.RightJacobianFromAngleAxis(omega), Pose.LeftJacobianFromAngleAxis(-omega), 1e-10))
			{
				failures.Add(i);
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task RightJacobianFromAngleAxis_SmallAngle()
	{
		// Near zero, Jr ~ I - 0.5 * [w]_x.
		var omega = new Vector3d(1e-12, 2e-12, 3e-12);
		Matrix3d jr = Pose.RightJacobianFromAngleAxis(omega);
		Matrix3d expected = Matrix3d.Identity - 0.5 * Rigid3d.CrossProductMatrix(omega);
		await Assert.That(EigenMatrixNear(jr, expected, 1e-10)).IsTrue();
	}

	// Numeric Jacobian column k of Exp at omega: Log(R^T Exp(omega + eps e_k)) / eps for
	// the right Jacobian, Log(Exp(omega + eps e_k) R^T) / eps for the left one.
	private static Matrix3d NumericJacobian(Vector3d omega, bool right)
	{
		const double Eps = 1e-7;
		Matrix3d r = Pose.AngleAxisToRotationMatrix(omega);
		var columns = new Vector3d[3];
		for (int k = 0; k < 3; ++k)
		{
			Vector3d dw = k switch { 0 => new Vector3d(Eps, 0, 0), 1 => new Vector3d(0, Eps, 0), _ => new Vector3d(0, 0, Eps) };
			Matrix3d rPerturbed = Pose.AngleAxisToRotationMatrix(omega + dw);
			Matrix3d dR = right ? r.Transpose() * rPerturbed : rPerturbed * r.Transpose();
			columns[k] = Pose.RotationMatrixToAngleAxis(dR) / Eps;
		}

		return Matrix3d.FromColumns(columns[0], columns[1], columns[2]);
	}

	[Test]
	public async Task RightJacobianFromAngleAxis_NumericDerivative()
	{
		// Verify Jr by numeric differentiation of Exp(w + dw) ~ Exp(w) * Exp(Jr*dw).
		RandomUtils.SetPRNGSeed(0);
		var failures = new List<int>();
		for (int i = 0; i < 50; ++i)
		{
			Vector3d omega = RandomAngleAxisVector();
			if (!EigenMatrixNear(Pose.RightJacobianFromAngleAxis(omega), NumericJacobian(omega, right: true), 1e-5))
			{
				failures.Add(i);
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task LeftJacobianFromAngleAxis_NumericDerivative()
	{
		// Verify Jl by numeric differentiation of Exp(w + dw) ~ Exp(Jl*dw) * Exp(w).
		RandomUtils.SetPRNGSeed(0);
		var failures = new List<int>();
		for (int i = 0; i < 50; ++i)
		{
			Vector3d omega = RandomAngleAxisVector();
			if (!EigenMatrixNear(Pose.LeftJacobianFromAngleAxis(omega), NumericJacobian(omega, right: false), 1e-5))
			{
				failures.Add(i);
			}
		}

		await Assert.That(failures).IsEmpty();
	}
}
