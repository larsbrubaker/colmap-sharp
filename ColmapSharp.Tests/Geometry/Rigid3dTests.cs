// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Rigid3dTests: colmap/geometry/rigid3_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Geometry/Rigid3d.cs. The bit-exact and tolerance comparisons against
// pycolmap are in GeometryOracleTests (C#-only), which states each function's tier.
//
// Eigen::Matrix<double, 12, 12> and <6, 12> are MatrixXd here, and
// RandomEigenMatrixd<12, 12> is RandomEigenMatrixXd(12, 12) (same draw order).
//
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does,
// and each test draws everything before its first await (the PRNG is per thread and an
// await may resume elsewhere).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Geometry;

public class Rigid3dTests
{
	private static Rigid3d TestRigid3d()
	{
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d();
		return new Rigid3d(rotation, translation);
	}

	[Test]
	public async Task CrossProductMatrix_Nominal()
	{
		var refMatrix = new Matrix3d(0, -3, 2, 3, 0, -1, -2, 1, 0);
		using (Assert.Multiple())
		{
			await Assert.That(Rigid3d.CrossProductMatrix(new Vector3d(0, 0, 0)) == Matrix3d.Zero).IsTrue();
			await Assert.That(Rigid3d.CrossProductMatrix(new Vector3d(1, 2, 3)) == refMatrix).IsTrue();
		}
	}

	[Test]
	public async Task Rigid3d_Default()
	{
		var tform = new Rigid3d();
		using (Assert.Multiple())
		{
			await Assert.That(tform.Rotation.Coeffs == Quaterniond.Identity.Coeffs).IsTrue();
			await Assert.That(tform.Translation == Vector3d.Zero).IsTrue();
		}
	}

	[Test]
	public async Task Rigid3d_Equals()
	{
		var tform = new Rigid3d();
		Rigid3d other = tform;
		bool equalAtStart = tform == other;
		tform = tform with { Translation = new Vector3d(1, tform.Translation.Y, tform.Translation.Z) };
		bool notEqualAfterChange = tform != other;
		other = other with { Translation = new Vector3d(1, other.Translation.Y, other.Translation.Z) };
		bool equalAgain = tform == other;
		using (Assert.Multiple())
		{
			await Assert.That(equalAtStart).IsTrue();
			await Assert.That(notEqualAfterChange).IsTrue();
			await Assert.That(equalAgain).IsTrue();
		}
	}

	[Test]
	public async Task Rigid3d_Print()
	{
		var tform = new Rigid3d();
		await Assert.That(tform.ToString())
			.IsEqualTo("Rigid3d(rotation_xyzw=[0, 0, 0, 1], translation=[0, 0, 0])");
	}

	[Test]
	public async Task Rigid3d_Inverse()
	{
		Rigid3d bFromA = TestRigid3d();
		Rigid3d aFromB = bFromA.Inverse();
		var results = new List<bool>();
		for (int i = 0; i < 100; ++i)
		{
			Vector3d xInA = RandomEigen.RandomEigenVector3d();
			Vector3d xInB = bFromA * xInA;
			results.Add(EigenMatrixNear(aFromB * xInB, xInA, 1e-6));
		}

		await Assert.That(results.All(r => r)).IsTrue();
	}

	[Test]
	public async Task Rigid3d_TgtOriginInSrc()
	{
		Rigid3d bFromA = TestRigid3d();
		Vector3d originBInA = bFromA.TgtOriginInSrc();
		await Assert.That((bFromA * originBInA - Vector3d.Zero).Norm).IsLessThan(1e-6);
	}

	[Test]
	public async Task Rigid3d_ToMatrix()
	{
		Rigid3d bFromA = TestRigid3d();
		Matrix3x4d bFromAMat = bFromA.ToMatrix();
		var errors = new List<double>();
		for (int i = 0; i < 100; ++i)
		{
			Vector3d xInA = RandomEigen.RandomEigenVector3d();
			errors.Add((bFromA * xInA - bFromAMat * xInA.Homogeneous()).Norm);
		}

		await Assert.That(errors.Max()).IsLessThan(1e-6);
	}

	[Test]
	public async Task Rigid3d_FromMatrix()
	{
		Rigid3d b1FromA = TestRigid3d();
		Rigid3d b2FromA = Rigid3d.FromMatrix(b1FromA.ToMatrix());
		var results = new List<bool>();
		for (int i = 0; i < 100; ++i)
		{
			Vector3d xInA = RandomEigen.RandomEigenVector3d();
			results.Add(EigenMatrixNear(b1FromA * xInA, b2FromA * xInA, 1e-6));
		}

		await Assert.That(results.All(r => r)).IsTrue();
	}

	[Test]
	public async Task Rigid3d_ApplyNoRotation()
	{
		var bFromA = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 2, 3));
		await Assert.That((bFromA * new Vector3d(1, 2, 3) - new Vector3d(2, 4, 6)).Norm).IsLessThan(1e-6);
	}

	[Test]
	public async Task Rigid3d_ApplyNoTranslation()
	{
		var bFromA = new Rigid3d(new AngleAxisd(Math.PI / 2, Vector3d.UnitX).ToQuaternion(), Vector3d.Zero);
		await Assert.That((bFromA * new Vector3d(1, 2, 3) - new Vector3d(1, -3, 2)).Norm).IsLessThan(1e-6);
	}

	[Test]
	public async Task Rigid3d_ApplyRotationTranslation()
	{
		var bFromA = new Rigid3d(new AngleAxisd(Math.PI / 2, Vector3d.UnitX).ToQuaternion(), new Vector3d(1, 2, 3));
		await Assert.That((bFromA * new Vector3d(1, 2, 3) - new Vector3d(2, -1, 5)).Norm).IsLessThan(1e-6);
	}

	[Test]
	public async Task Rigid3d_ApplyChain()
	{
		Rigid3d bFromA = TestRigid3d();
		Rigid3d cFromB = TestRigid3d();
		Rigid3d dFromC = TestRigid3d();
		Vector3d xInA = RandomEigen.RandomEigenVector3d();
		Vector3d xInB = bFromA * xInA;
		Vector3d xInC = cFromB * xInB;
		Vector3d xInD = dFromC * xInC;
		await Assert.That(dFromC * (cFromB * (bFromA * xInA)) == xInD).IsTrue();
	}

	[Test]
	public async Task Rigid3d_Compose()
	{
		Rigid3d bFromA = TestRigid3d();
		Rigid3d cFromB = TestRigid3d();
		Rigid3d dFromC = TestRigid3d();
		Rigid3d dFromA = dFromC * cFromB * bFromA;
		Vector3d xInA = RandomEigen.RandomEigenVector3d();
		Vector3d xInB = bFromA * xInA;
		Vector3d xInC = cFromB * xInB;
		Vector3d xInD = dFromC * xInC;
		await Assert.That(EigenMatrixNear(dFromA * xInA, xInD, 1e-6)).IsTrue();
	}

	[Test]
	public async Task Rigid3d_Adjoint()
	{
		Rigid3d bFromA = TestRigid3d();
		Matrix6d adjoint = bFromA.Adjoint();
		Matrix6d adjointInv = bFromA.AdjointInverse();
		Rigid3d aFromB = bFromA.Inverse();
		Matrix6d adjointAFromB = aFromB.Adjoint();
		using (Assert.Multiple())
		{
			await Assert.That(EigenMatrixNear(adjoint * adjointInv, Matrix6d.Identity, 1e-6)).IsTrue();
			await Assert.That(EigenMatrixNear(adjointInv, adjointAFromB, 1e-6)).IsTrue();
		}
	}

	[Test]
	public async Task Rigid3d_CovarianceForInverse()
	{
		Rigid3d bFromA = TestRigid3d();
		Matrix6d a = RandomEigen.RandomEigenMatrix6d();
		Matrix6d covBFromA = a * a.Transpose();
		Matrix6d covAFromB = Rigid3d.GetCovarianceForRigid3dInverse(bFromA, covBFromA);
		Rigid3d aFromB = bFromA.Inverse();
		Matrix6d covBFromATest = Rigid3d.GetCovarianceForRigid3dInverse(aFromB, covAFromB);
		await Assert.That(EigenMatrixNear(covBFromATest, covBFromA, 1e-6)).IsTrue();
	}

	[Test]
	public async Task Rigid3d_CovarianceForRelativeRigid3d_PerfectCorrelation()
	{
		Rigid3d worldFromA = TestRigid3d();
		Rigid3d worldFromB = TestRigid3d();
		MatrixXd a = RandomEigen.RandomEigenMatrixXd(6, 6);
		MatrixXd covarSubblock = a * a.Transpose();
		// Two poses are perfectly correlated in world frame
		var covarWorldFromCam = new MatrixXd(12, 12);
		covarWorldFromCam.SetBlock(0, 0, covarSubblock);
		covarWorldFromCam.SetBlock(0, 6, covarSubblock);
		covarWorldFromCam.SetBlock(6, 0, covarSubblock);
		covarWorldFromCam.SetBlock(6, 6, covarSubblock);
		// Invert poses
		Rigid3d aFromWorld = worldFromA.Inverse();
		Rigid3d bFromWorld = worldFromB.Inverse();
		MatrixXd j0 = MatrixXd.Zero(12, 12);
		j0.SetBlock(0, 0, MatrixXd.From(-worldFromA.AdjointInverse()));
		j0.SetBlock(6, 6, MatrixXd.From(-worldFromB.AdjointInverse()));
		MatrixXd covarCamFromWorld = j0 * covarWorldFromCam * j0.Transpose();
		// Calculate relative pose covariance, which should be a zero matrix.
		Matrix6d bCovFromA = Rigid3d.GetCovarianceForRelativeRigid3d(aFromWorld, bFromWorld, covarCamFromWorld);
		await Assert.That(bCovFromA.Norm()).IsLessThan(1e-6);
	}

	[Test]
	public async Task Rigid3d_CovarianceForRelativeRigid3d()
	{
		Rigid3d aFromWorld = TestRigid3d();
		Rigid3d bFromWorld = TestRigid3d();
		MatrixXd a = RandomEigen.RandomEigenMatrixXd(12, 12);
		MatrixXd covar = a * a.Transpose();

		// Ours (in left convention)
		Matrix6d bCovFromA = Rigid3d.GetCovarianceForRelativeRigid3d(aFromWorld, bFromWorld, covar);

		// Use the equations from the right convention as a reference.
		// The covariance in left (right) equals to the covariance of pose inverse in
		// right (left).

		// Convert to right convention. To estimate covariance of T_2T_1^{-1} in left,
		// We can equivalently estimate covariance of T_1T_2^{-1} in right.
		MatrixXd j0 = MatrixXd.Zero(12, 12);
		// the covariance of T_1^{-1} in left corresponds to the covariance of T_1 in
		// right
		j0.SetBlock(0, 0, MatrixXd.From(-aFromWorld.AdjointInverse()));
		// the covariance of T_2 in left corresponds to the covariance of T_2^{-1} in
		// right
		j0.SetBlock(6, 6, MatrixXd.Identity(6));
		// Get the covariance of (T_1, T_2^{-1}) in right
		MatrixXd covarInRight = j0 * covar * j0.Transpose();

		// Compose T_1T_2^{-1} in right
		// [Reference] Joan Sola, Jeremie Deray, Dinesh Atchuthan, A micro Lie theory
		// for state estimation in robotics, 2018.
		// Eqs. (177) and (178)
		var jInRight = new MatrixXd(6, 12);
		jInRight.SetBlock(0, 0, MatrixXd.From(bFromWorld.Adjoint()));
		jInRight.SetBlock(0, 6, MatrixXd.Identity(6));
		Matrix6d aCovFromBRight = (jInRight * covarInRight * jInRight.Transpose()).ToMatrix6d();
		await Assert.That(EigenMatrixNear(bCovFromA, aCovFromBRight, 1e-6)).IsTrue();
	}

	[Test]
	public async Task Rigid3d_CovariancePropagation_Composed_vs_Relative()
	{
		Rigid3d aFromB = TestRigid3d();
		Rigid3d bFromC = TestRigid3d();
		MatrixXd a = RandomEigen.RandomEigenMatrixXd(12, 12);
		MatrixXd covar = a * a.Transpose();

		// Covariance for the composed rigid3d
		Matrix6d aCovFromCComposed = Rigid3d.GetCovarianceForComposedRigid3d(aFromB, covar);

		// Invert b_from_c and switch order
		Rigid3d cFromB = bFromC.Inverse();
		MatrixXd j0 = MatrixXd.Zero(12, 12);
		j0.SetBlock(6, 0, MatrixXd.Identity(6));
		j0.SetBlock(0, 6, MatrixXd.From(-bFromC.AdjointInverse()));
		MatrixXd covarXFromB = j0 * covar * j0.Transpose();
		Matrix6d aCovFromCRelative = Rigid3d.GetCovarianceForRelativeRigid3d(cFromB, aFromB, covarXFromB);

		// Check consistency
		await Assert.That(EigenMatrixNear(aCovFromCComposed, aCovFromCRelative, 1e-6)).IsTrue();
	}
}
