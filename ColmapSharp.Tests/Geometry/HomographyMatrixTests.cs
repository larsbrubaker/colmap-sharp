// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// HomographyMatrixTests: colmap/geometry/homography_matrix_test.cc ported 1:1, one method
// per gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Geometry/HomographyMatrix.cs. Tier B; the pycolmap comparison of
// PoseFromHomographyMatrix is in GeometryTwoViewOracleTests (C#-only).
// Each test seeds the PRNG with 0, as COLMAP's gtest_main does.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;
using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Geometry;

public class HomographyMatrixTests
{
	// Note that the test case values are obtained from OpenCV.
	[Test]
	public async Task DecomposeHomographyMatrix_Nominal()
	{
		Matrix3d h = new Matrix3d(
			2.649157564634028, 4.583875997496426, 70.694447785121326,
			-1.072756858861583, 3.533262150437228, 1513.656999614321649,
			0.001303887589576, 0.003042206876298, 1) * 3;

		var k = new Matrix3d(640, 0, 320, 0, 640, 240, 0, 0, 1);

		var cams2FromCams1 = new List<Rigid3d>();
		var normals = new List<Vector3d>();
		HomographyMatrix.DecomposeHomographyMatrix(h, k, k, cams2FromCams1, normals);

		var refRotation = new Matrix3d(
			0.43307983549125, 0.545749113549648, -0.717356090899523,
			-0.85630229674426, 0.497582023798831, -0.138414255706431,
			0.281404038139784, 0.67421809131173, 0.682818960388909);
		var refTranslation = new Vector3d(1.826751712278038, 1.264718492450820, 0.195080809998819);
		var refNormal = new Vector3d(-0.244875830334816, -0.480857890778889, -0.841909446789566);

		bool refSolutionExists = false;
		for (int i = 0; i < 4 && i < cams2FromCams1.Count; ++i)
		{
			const double Eps = 1e-6;
			if ((cams2FromCams1[i].Rotation.ToRotationMatrix() - refRotation).Norm() < Eps
				&& (cams2FromCams1[i].Translation - refTranslation).Norm < Eps
				&& (normals[i] - refNormal).Norm < Eps)
			{
				refSolutionExists = true;
				break;
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(cams2FromCams1.Count).IsEqualTo(4);
			await Assert.That(normals.Count).IsEqualTo(4);
			await Assert.That(refSolutionExists).IsTrue();
		}
	}

	[Test]
	public async Task DecomposeHomographyMatrix_Random()
	{
		RandomUtils.SetPRNGSeed(0);
		const int NumIters = 100;
		const double Epsilon = 1e-6;
		Matrix3d identity = Matrix3d.Identity;

		var failures = new List<string>();
		for (int i = 0; i < NumIters; ++i)
		{
			Matrix3d h = RandomEigen.RandomEigenMatrix3d();
			if (Math.Abs(h.Determinant()) < Epsilon)
			{
				continue;
			}

			var cams2FromCams1 = new List<Rigid3d>();
			var normals = new List<Vector3d>();
			HomographyMatrix.DecomposeHomographyMatrix(h, identity, identity, cams2FromCams1, normals);

			if (cams2FromCams1.Count != 4 || normals.Count != 4)
			{
				failures.Add($"iteration {i}: {cams2FromCams1.Count} poses, {normals.Count} normals");
			}

			// Test that each candidate rotation is a rotation.
			foreach (Rigid3d cam2FromCam1 in cams2FromCams1)
			{
				Matrix3d r = cam2FromCam1.Rotation.ToRotationMatrix();
				Matrix3d orthogError = r.Transpose() * r - identity;
				double infinityNorm = 0;
				for (int row = 0; row < 3; ++row)
				{
					for (int col = 0; col < 3; ++col)
					{
						infinityNorm = Math.Max(infinityNorm, Math.Abs(orthogError[row, col]));
					}
				}

				// Orthogonal, with determinant 1.
				if (!(infinityNorm < Epsilon) || !(Math.Abs(r.Determinant() - 1.0) <= Epsilon))
				{
					failures.Add($"iteration {i}: not a rotation");
				}
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	[Test]
	public async Task PoseFromHomographyMatrix_Nominal()
	{
		Matrix3d k1 = Matrix3d.Identity;
		Matrix3d k2 = Matrix3d.Identity;
		Quaterniond refRotation = new Quaterniond(1, 0.1, 0.2, 0.3).Normalized();
		var refTranslation = new Vector3d(1, 0, 0);
		var refNormal = new Vector3d(0, 0, -1);
		Matrix3d h = HomographyMatrix.HomographyMatrixFromPose(
			k1, k2, refRotation.ToRotationMatrix(), refTranslation, refNormal, 1);

		Vector3d[] rays1 =
		[
			new Vector3d(0.1, 0.1, 1).Normalized(),
			new Vector3d(0.4, 0.1, 1).Normalized(),
			new Vector3d(0.1, 0.4, 1).Normalized(),
			new Vector3d(0.4, 0.4, 1).Normalized(),
			new Vector3d(0.0, 0.0, 1).Normalized(),
		];

		var rays2 = new List<Vector3d>();
		double minZ = double.MaxValue;
		foreach (Vector3d ray1 in rays1)
		{
			Vector3d ray2 = h * ray1;
			minZ = Math.Min(minZ, ray2.Z);
			rays2.Add(ray2.Normalized());
		}

		await Assert.That(minZ).IsGreaterThan(0);

		var points3D = new List<Vector3d>();
		HomographyMatrix.PoseFromHomographyMatrix(
			h, k1, k2, rays1, rays2, out Rigid3d cam2FromCam1, out Vector3d normal, points3D);

		using (Assert.Multiple())
		{
			await Assert.That(Rigid3dNear(
				new Rigid3d(cam2FromCam1.Rotation, cam2FromCam1.Translation.Normalized()),
				new Rigid3d(refRotation, refTranslation.Normalized()),
				1e-6,
				1e-6)).IsTrue();
			await Assert.That(EigenMatrixNear(normal, refNormal, 1e-5)).IsTrue();
			await Assert.That(points3D.Count).IsEqualTo(rays1.Length);
		}
	}

	[Test]
	public async Task HomographyMatrixFromPose_PureRotation()
	{
		Matrix3d h = HomographyMatrix.HomographyMatrixFromPose(
			Matrix3d.Identity, Matrix3d.Identity, Matrix3d.Identity, new Vector3d(0, 0, 0), new Vector3d(-1, 0, 0), double.PositiveInfinity);
		await Assert.That(h == Matrix3d.Identity).IsTrue();
	}

	[Test]
	public async Task HomographyMatrixFromPose_PlanarScene()
	{
		Matrix3d h = HomographyMatrix.HomographyMatrixFromPose(
			Matrix3d.Identity, Matrix3d.Identity, Matrix3d.Identity, new Vector3d(1, 0, 0), new Vector3d(-1, 0, 0), 1);
		var hRef = new Matrix3d(2, 0, 0, 0, 1, 0, 0, 0, 1);
		await Assert.That(h == hRef).IsTrue();
	}
}
