// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// EssentialMatrixTests: colmap/geometry/essential_matrix_test.cc ported 1:1, one method
// per gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Geometry/EssentialMatrix.cs. Tier B where the SVD is involved
// (decomposition, pose recovery, epipoles); the pycolmap comparisons are in
// GeometryTwoViewOracleTests (C#-only).
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

public class EssentialMatrixTests
{
	// A pinhole geometry to exercise the tangent Sampson error against a known
	// closed-form answer.
	private const double Focal = 650.0;
	private const double PrincipalX = 512.0;
	private const double PrincipalY = 384.0;

	private static readonly Vector3d[] FourPoints =
	[
		new Vector3d(0, 0, 1),
		new Vector3d(0, 0.1, 1),
		new Vector3d(0.1, 0, 1),
		new Vector3d(0.1, 0.1, 1),
	];

	private static Vector2d PinholeImgFromCam(Vector3d camPoint) =>
		new(Focal * camPoint.X / camPoint.Z + PrincipalX, Focal * camPoint.Y / camPoint.Z + PrincipalY);

	// Normalized image plane representative (u, v, 1) of a pixel.
	private static Vector3d PinholeNormalizedFromImg(Vector2d imagePoint) =>
		new((imagePoint.X - PrincipalX) / Focal, (imagePoint.Y - PrincipalY) / Focal, 1.0);

	// d(u, v, 1) / d(x, y) for a pinhole: constant and diagonal.
	private static Matrix3x2d PinholeNormalizedJacobian() => new(1.0 / Focal, 0, 0, 1.0 / Focal, 0, 0);

	// d(unit ray) / d(x, y) for a pinhole, via the normalization quotient rule.
	private static Matrix3x2d PinholeUnitRayJacobian(Vector3d normalized)
	{
		double norm = normalized.Norm;
		Matrix3d outer = Matrix3d.FromColumns(normalized * normalized.X, normalized * normalized.Y, normalized * normalized.Z);
		Matrix3d dnormalize = (Matrix3d.Identity - outer / (norm * norm)) / norm;
		return dnormalize * PinholeNormalizedJacobian();
	}

	private static Rigid3d TangentTestPose() => new(
		new AngleAxisd(0.15, new Vector3d(0.3, 1, 0.2).Normalized()).ToQuaternion(),
		new Vector3d(1.0, 0.2, 0.1).Normalized());

	[Test]
	public async Task DecomposeEssentialMatrix_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		var cam2FromCam1 = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), new Vector3d(0.5, 1, 1).Normalized());
		Matrix3d rotMat = cam2FromCam1.Rotation.ToRotationMatrix();
		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);

		EssentialMatrix.DecomposeEssentialMatrix(e, out Matrix3d r1, out Matrix3d r2, out Vector3d t);

		using (Assert.Multiple())
		{
			await Assert.That((r1 - rotMat).Norm() < 1e-10 || (r2 - rotMat).Norm() < 1e-10).IsTrue();
			await Assert.That((t - cam2FromCam1.Translation).Norm < 1e-10 || (t + cam2FromCam1.Translation).Norm < 1e-10).IsTrue();
		}
	}

	[Test]
	public async Task EssentialMatrixFromPose_Nominal()
	{
		var expected = new Matrix3d(0, -1, 0, 1, 0, 0, 0, 0, 0);
		using (Assert.Multiple())
		{
			await Assert.That(EssentialMatrix.EssentialMatrixFromPose(new Rigid3d(Quaterniond.Identity, new Vector3d(0, 0, 1))) == expected).IsTrue();
			await Assert.That(EssentialMatrix.EssentialMatrixFromPose(new Rigid3d(Quaterniond.Identity, new Vector3d(0, 0, 2))) == expected).IsTrue();
		}
	}

	[Test]
	public async Task PoseFromEssentialMatrix_Nominal()
	{
		var cam1FromWorld = new Rigid3d();
		var cam2FromWorld = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0).Normalized());
		Rigid3d cam2FromCam1 = cam2FromWorld * cam1FromWorld.Inverse();
		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);

		Vector3d[] rays1 = FourPoints.Select(p => (cam1FromWorld * p).Normalized()).ToArray();
		Vector3d[] rays2 = FourPoints.Select(p => (cam2FromWorld * p).Normalized()).ToArray();

		var validIndices = new List<int>();
		EssentialMatrix.PoseFromEssentialMatrix(e, rays1, rays2, out Rigid3d cam2FromCam1Est, validIndices);

		using (Assert.Multiple())
		{
			await Assert.That(validIndices.Count).IsEqualTo(4);
			await Assert.That(Rigid3dNear(cam2FromCam1Est, cam2FromCam1)).IsTrue();
		}
	}

	[Test]
	public async Task FindOptimalImageObservations_Nominal()
	{
		var cam1FromWorld = new Rigid3d();
		var cam2FromWorld = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0).Normalized());
		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(cam2FromWorld * cam1FromWorld.Inverse());

		// Test if perfect projection is equivalent to optimal image observations.
		var mismatches = new List<int>();
		for (int i = 0; i < FourPoints.Length; ++i)
		{
			Vector2d point1 = (cam1FromWorld * FourPoints[i]).HNormalized();
			Vector2d point2 = (cam2FromWorld * FourPoints[i]).HNormalized();
			EssentialMatrix.FindOptimalImageObservations(e, point1, point2, out Vector2d optimal1, out Vector2d optimal2);
			if (!EigenMatrixNear(point1, optimal1) || !EigenMatrixNear(point2, optimal2))
			{
				mismatches.Add(i);
			}
		}

		await Assert.That(mismatches).IsEmpty();
	}

	[Test]
	public async Task EpipoleFromEssentialMatrix_Nominal()
	{
		var cam2FromCam1 = new Rigid3d(Quaterniond.Identity, new Vector3d(0, 0, -1).Normalized());
		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);

		Vector3d leftEpipole = EssentialMatrix.EpipoleFromEssentialMatrix(e, true);
		Vector3d rightEpipole = EssentialMatrix.EpipoleFromEssentialMatrix(e, false);
		using (Assert.Multiple())
		{
			await Assert.That(EigenMatrixNear(leftEpipole, new Vector3d(0, 0, 1))).IsTrue();
			await Assert.That(EigenMatrixNear(rightEpipole, new Vector3d(0, 0, 1))).IsTrue();
		}
	}

	[Test]
	public async Task InvertEssentialMatrix_Nominal()
	{
		var mismatches = new List<int>();
		for (int i = 1; i < 10; ++i)
		{
			var cam2FromCam1 = new Rigid3d(
				Quaterniond.FromRotationMatrix(Pose.EulerAnglesToRotationMatrix(0, 0.1, 0)),
				new Vector3d(0, 0, i).Normalized());
			Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);
			Matrix3d invInvE = EssentialMatrix.InvertEssentialMatrix(EssentialMatrix.InvertEssentialMatrix(e));
			if (!EigenMatrixNear(e, invInvE))
			{
				mismatches.Add(i);
			}
		}

		await Assert.That(mismatches).IsEmpty();
	}

	private static readonly Matrix3d K1 = new(2, 0, 1, 0, 3, 2, 0, 0, 1);
	private static readonly Matrix3d K2 = new(3, 0, 2, 0, 4, 1, 0, 0, 1);

	private static Matrix3d RandomEssentialMatrix()
	{
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d();
		return EssentialMatrix.EssentialMatrixFromPose(new Rigid3d(rotation, translation));
	}

	[Test]
	public async Task FundamentalFromEssentialMatrix_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		Matrix3d e = RandomEssentialMatrix();
		Matrix3d f = EssentialMatrix.FundamentalFromEssentialMatrix(K2, e, K1);
		var x = new Vector3d(3, 2, 1);
		using (Assert.Multiple())
		{
			await Assert.That(EigenMatrixNear(K2.Transpose().Inverse() * e * x, f * K1 * x)).IsTrue();
			await Assert.That(EigenMatrixNear(e * K1.Inverse() * x, K2.Transpose() * f * x)).IsTrue();
		}
	}

	[Test]
	public async Task EssentialFromFundamentalMatrix_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		Matrix3d e = RandomEssentialMatrix();
		Matrix3d f = EssentialMatrix.FundamentalFromEssentialMatrix(K2, e, K1);
		await Assert.That(EigenMatrixNear(EssentialMatrix.EssentialFromFundamentalMatrix(K2, f, K1), e, 1e-6)).IsTrue();
	}

	[Test]
	public async Task ComputeSquaredSampsonError_Nominal()
	{
		Vector2d[] points1 = [new Vector2d(0, 0), new Vector2d(0, 0), new Vector2d(0, 0)];
		Vector2d[] points2 = [new Vector2d(2, 0), new Vector2d(2, 1), new Vector2d(2, 2)];

		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0)));

		var residuals = new List<double>();
		EssentialMatrix.ComputeSquaredSampsonError(points1, points2, e, residuals);

		using (Assert.Multiple())
		{
			await Assert.That(residuals.Count).IsEqualTo(3);
			await Assert.That(residuals[0]).IsEqualTo(0);
			await Assert.That(residuals[1]).IsEqualTo(0.5);
			await Assert.That(residuals[2]).IsEqualTo(2);
		}
	}

	// With the normalized image plane representative (u, v, 1), whose Jacobian
	// w.r.t. pixels is the constant 1/f, the tangent Sampson error is *exactly*
	// f^2 times the classical Sampson error. This is an algebraic identity, so it
	// pins down the whole formula - numerator, both gradient chains, and the
	// denominator - to machine precision.
	[Test]
	public async Task ComputeSquaredTangentSampsonError_PinholeMatchesScaledSampsonExactly()
	{
		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(TangentTestPose());
		Matrix3x2d jNorm = PinholeNormalizedJacobian();

		// Deliberately includes badly mismatched pairs: the identity is exact for
		// arbitrary inputs, not only for near-inliers.
		double maxRelDiff = 0;
		double minScaledSampson = double.MaxValue;
		foreach (double x1 in new[] { 20.0, 512.0, 1000.0 })
		{
			foreach (double y1 in new[] { 30.0, 384.0, 740.0 })
			{
				foreach (double x2 in new[] { 45.0, 512.0, 980.0 })
				{
					foreach (double y2 in new[] { 60.0, 384.0, 700.0 })
					{
						Vector3d m1 = PinholeNormalizedFromImg(new Vector2d(x1, y1));
						Vector3d m2 = PinholeNormalizedFromImg(new Vector2d(x2, y2));
						double tangentSampson = EssentialMatrix.ComputeSquaredTangentSampsonError(
							new CamRayWithJac(m1, jNorm), new CamRayWithJac(m2, jNorm), e);
						double scaledSampson = Focal * Focal * EssentialMatrix.ComputeSquaredSampsonError(m1, m2, e);
						minScaledSampson = Math.Min(minScaledSampson, scaledSampson);
						maxRelDiff = Math.Max(maxRelDiff, Math.Abs(tangentSampson - scaledSampson) / scaledSampson);
					}
				}
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(minScaledSampson).IsGreaterThan(0.0);
			await Assert.That(maxRelDiff).IsLessThanOrEqualTo(1e-14);
		}
	}

	// With unit bearing vectors the agreement is only first order: rescaling the
	// homogeneous representative by a function of the measurements changes the
	// Sampson approximation by a term proportional to the residual itself. The
	// relative discrepancy is therefore expected to shrink linearly as the
	// correspondence approaches the epipolar variety.
	[Test]
	public async Task ComputeSquaredTangentSampsonError_UnitRaysAgreeToFirstOrder()
	{
		Rigid3d cam2FromCam1 = TangentTestPose();
		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);

		var point3DInCam1 = new Vector3d(0.35, -0.2, 4.0);
		Vector2d imagePoint1 = PinholeImgFromCam(point3DInCam1);
		Vector2d imagePoint2 = PinholeImgFromCam(cam2FromCam1 * point3DInCam1);

		// Displace image 2 perpendicular to the epipolar line of image_point1, so
		// the offset is entirely "epipolar error" rather than a slide along the line.
		Vector3d m1 = PinholeNormalizedFromImg(imagePoint1);
		Vector3d epipolarLine2 = e * m1;
		Vector2d perpendicular = epipolarLine2.Head2().Normalized();

		var failures = new List<string>();
		double prevRelDiff = double.MaxValue;
		double prevRatio = 0.0;
		foreach (double offset in new[] { 1.0, 0.1, 0.01, 0.001 })
		{
			Vector3d m2 = PinholeNormalizedFromImg(imagePoint2 + offset * perpendicular);
			double tangentSampson = EssentialMatrix.ComputeSquaredTangentSampsonError(
				new CamRayWithJac(m1.Normalized(), PinholeUnitRayJacobian(m1)),
				new CamRayWithJac(m2.Normalized(), PinholeUnitRayJacobian(m2)),
				e);
			double scaledSampson = Focal * Focal * EssentialMatrix.ComputeSquaredSampsonError(m1, m2, e);

			// The residual is a squared distance, so it must scale quadratically with
			// the displacement; its square root is strictly below the offset.
			if (!(Math.Sqrt(tangentSampson) < offset))
			{
				failures.Add($"sqrt(residual) >= offset at {offset}");
			}

			double ratio = tangentSampson / (offset * offset);
			if (prevRatio > 0.0 && !(Math.Abs(ratio - prevRatio) / prevRatio <= 1e-2))
			{
				failures.Add($"ratio drift at {offset}");
			}

			prevRatio = ratio;

			double relDiff = Math.Abs(tangentSampson - scaledSampson) / scaledSampson;

			// Each tenfold reduction of the residual must reduce the discrepancy.
			if (!(relDiff < prevRelDiff))
			{
				failures.Add($"discrepancy did not shrink at {offset}");
			}

			prevRelDiff = relDiff;
		}

		using (Assert.Multiple())
		{
			await Assert.That(failures).IsEmpty();

			// At a milli-pixel residual the two formulations are indistinguishable.
			await Assert.That(prevRelDiff).IsLessThan(1e-5);
		}
	}

	[Test]
	public async Task ComputeSquaredTangentSampsonError_DegenerateDenominatorReturnsMax()
	{
		Matrix3x2d jNorm = PinholeNormalizedJacobian();
		double zeroE = EssentialMatrix.ComputeSquaredTangentSampsonError(
			new CamRayWithJac(new Vector3d(0, 0, 1), jNorm), new CamRayWithJac(new Vector3d(0, 0, 1), jNorm), Matrix3d.Zero);

		// The CamRayWithJac.Zero sentinel that callers substitute for an unprojectable
		// point must be rejected for any essential matrix: its zero ray and Jacobian
		// force the denominator to zero regardless of a (nonzero) E.
		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(
			new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0.1, 0.2).Normalized()));
		double sentinel = EssentialMatrix.ComputeSquaredTangentSampsonError(
			CamRayWithJac.Zero, new CamRayWithJac(new Vector3d(0, 0, 1), jNorm), e);
		using (Assert.Multiple())
		{
			await Assert.That(zeroE).IsEqualTo(double.MaxValue);
			await Assert.That(sentinel).IsEqualTo(double.MaxValue);
		}
	}

	[Test]
	public async Task ComputeSquaredTangentSampsonError_VectorOverloadAndCheirality()
	{
		var cam1FromWorld = new Rigid3d();
		var cam2FromWorld = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0).Normalized());
		Rigid3d cam2FromCam1 = cam2FromWorld * cam1FromWorld.Inverse();
		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);

		var camRays1WithJac = new CamRayWithJac[FourPoints.Length];
		var camRays2WithJac = new CamRayWithJac[FourPoints.Length];
		for (int i = 0; i < FourPoints.Length; ++i)
		{
			Vector3d cam1Point = cam1FromWorld * FourPoints[i];
			Vector3d cam2Point = cam2FromWorld * FourPoints[i];
			camRays1WithJac[i] = new CamRayWithJac(cam1Point.Normalized(), PinholeUnitRayJacobian(cam1Point / cam1Point.Z));
			camRays2WithJac[i] = new CamRayWithJac(cam2Point.Normalized(), PinholeUnitRayJacobian(cam2Point / cam2Point.Z));
		}

		var residuals = new List<double>();
		EssentialMatrix.ComputeSquaredTangentSampsonError(camRays1WithJac, camRays2WithJac, e, residuals);
		int residualCount = residuals.Count;
		double maxResidual = residuals.Max();

		// Flipping one correspondence behind both cameras leaves the epipolar
		// constraint satisfied but must be rejected once cheirality is enforced.
		camRays1WithJac[1] = camRays1WithJac[1] with { Ray = -camRays1WithJac[1].Ray };
		camRays2WithJac[1] = camRays2WithJac[1] with { Ray = -camRays2WithJac[1].Ray };

		var plainResiduals = new List<double>();
		EssentialMatrix.ComputeSquaredTangentSampsonError(camRays1WithJac, camRays2WithJac, e, plainResiduals);

		var cheiralResiduals = new List<double>();
		EssentialMatrix.ComputeSquaredTangentSampsonErrorWithCheirality(camRays1WithJac, camRays2WithJac, e, cheiralResiduals);

		await Assert.That(residualCount).IsEqualTo(FourPoints.Length);
		await Assert.That(cheiralResiduals.Count).IsEqualTo(FourPoints.Length);
		using (Assert.Multiple())
		{
			await Assert.That(maxResidual).IsLessThan(1e-16);
			await Assert.That(plainResiduals[1]).IsLessThan(1e-16);
			await Assert.That(cheiralResiduals[1]).IsEqualTo(double.MaxValue);
			await Assert.That(cheiralResiduals[0]).IsEqualTo(plainResiduals[0]);
			await Assert.That(cheiralResiduals[2]).IsEqualTo(plainResiduals[2]);
			await Assert.That(cheiralResiduals[3]).IsEqualTo(plainResiduals[3]);
		}
	}
}
