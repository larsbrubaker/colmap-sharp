// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PoseEstimationTests: colmap/estimators/pose_test.cc ported 1:1. TEST(Suite, Name) becomes
// Suite_Name; same synthetic scenes and tolerances. Tests ColmapSharp/Estimators/
// PoseEstimation.cs (and, through the covariance checks, Solver/Covariance.cs). Tier C:
// RANSAC and the nonlinear refinement are judged by outcome at COLMAP's tolerances.
//
// Each test seeds the PRNG with 0, as COLMAP's gtest_main does, and draws everything before
// its first await (the PRNG is per thread).

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Estimators;

public class PoseEstimationTests
{
	private sealed class AbsolutePoseProblem
	{
		public Reconstruction Reconstruction { get; } = new();

		public Image Image { get; set; } = null!;

		public Camera Camera { get; set; } = null!;

		public List<Vector2d> Points2D { get; } = [];

		public List<Vector3d> Points3D { get; } = [];
	}

	private static AbsolutePoseProblem CreateAbsolutePoseTestData(
		CameraModelId cameraModelId = CameraModelId.SimpleRadial, double[]? cameraParams = null)
	{
		var problem = new AbsolutePoseProblem();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 2,
			NumPoints3D = 100,
			CameraModelId = cameraModelId,
			CameraParams = cameraParams ?? [1280, 512, 384, 0.05],
		};
		Synthetic.SynthesizeDataset(options, problem.Reconstruction);

		problem.Image = problem.Reconstruction.Image(1);
		problem.Camera = problem.Image.CameraPtr.Clone();
		ColmapSharp.Util.Check.That(problem.Camera.ModelId == cameraModelId, "camera model");
		foreach (Point2D point2D in problem.Image.Points2D)
		{
			if (point2D.HasPoint3D)
			{
				problem.Points2D.Add(point2D.Xy);
				problem.Points3D.Add(problem.Reconstruction.Point3D(point2D.Point3DId).Xyz);
			}
		}

		return problem;
	}

	private static List<CamRayWithJac> ProjectedRaysWithJac(Camera camera, Rigid3d camFromWorld, List<Vector3d> points3D)
	{
		var rays = new List<CamRayWithJac>(points3D.Count);
		foreach (Vector3d point3D in points3D)
		{
			rays.Add(camera.CamRayFromImgWithJac(camera.ImgFromCam(camFromWorld * point3D)!.Value)!.Value);
		}

		return rays;
	}

	[Test]
	public async Task EstimateAbsolutePose_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		AbsolutePoseProblem problem = CreateAbsolutePoseTestData();

		var options = new AbsolutePoseEstimationOptions();
		var camFromWorld = new Rigid3d();
		Camera camera = problem.Camera.Clone();
		bool success = PoseEstimation.EstimateAbsolutePose(
			options, problem.Points2D, problem.Points3D, ref camFromWorld, camera, out int numInliers, out bool[] inlierMask);

		await Assert.That(success).IsTrue();
		await Assert.That(Rigid3dNear(camFromWorld, problem.Image.CamFromWorld(), 1e-6, 1e-6)).IsTrue();
		await Assert.That(camera == problem.Camera).IsTrue();
		await Assert.That(numInliers).IsEqualTo(problem.Points2D.Count);
		await Assert.That(inlierMask.All(x => x)).IsTrue();
	}

	[Test]
	public async Task EstimateAbsolutePose_EstimateFocalLength()
	{
		RandomUtils.SetPRNGSeed(0);
		AbsolutePoseProblem problem = CreateAbsolutePoseTestData();

		var options = new AbsolutePoseEstimationOptions { EstimateFocalLength = true };
		var camFromWorld = new Rigid3d();
		Camera camera = problem.Camera.Clone();
		bool success = PoseEstimation.EstimateAbsolutePose(
			options, problem.Points2D, problem.Points3D, ref camFromWorld, camera, out int numInliers, out bool[] inlierMask);

		await Assert.That(success).IsTrue();
		await Assert.That(Rigid3dNear(camFromWorld, problem.Image.CamFromWorld(), 1e-3, 1e-2)).IsTrue();
		await Assert.That(camera.FocalLength()).IsEqualTo(problem.Camera.FocalLength()).Within(5);
		camera.SetFocalLength(problem.Camera.FocalLength());
		await Assert.That(camera == problem.Camera).IsTrue();
		await Assert.That(numInliers).IsEqualTo(problem.Points2D.Count);
		await Assert.That(inlierMask.All(x => x)).IsTrue();
	}

	[Test]
	public async Task EstimateAbsolutePose_EstimateSeparateFocalLengths()
	{
		RandomUtils.SetPRNGSeed(0);
		// PINHOLE camera with distinct fx and fy (fx, fy, cx, cy).
		AbsolutePoseProblem problem = CreateAbsolutePoseTestData(CameraModelId.Pinhole, [1280, 1000, 512, 384]);

		var options = new AbsolutePoseEstimationOptions { EstimateFocalLength = true };
		var camFromWorld = new Rigid3d();
		Camera camera = problem.Camera.Clone();
		bool success = PoseEstimation.EstimateAbsolutePose(
			options, problem.Points2D, problem.Points3D, ref camFromWorld, camera, out int numInliers, out bool[] inlierMask);

		await Assert.That(success).IsTrue();
		await Assert.That(Rigid3dNear(camFromWorld, problem.Image.CamFromWorld(), 1e-3, 1e-2)).IsTrue();
		await Assert.That(camera.FocalLengthX()).IsEqualTo(problem.Camera.FocalLengthX()).Within(5);
		await Assert.That(camera.FocalLengthY()).IsEqualTo(problem.Camera.FocalLengthY()).Within(5);
		await Assert.That(numInliers).IsEqualTo(problem.Points2D.Count);
		await Assert.That(inlierMask.All(x => x)).IsTrue();
	}

	[Test]
	public async Task EstimateRelativePose_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 512.0, 1024, 1024);
		var cam1FromWorld = new Rigid3d();
		var cam2FromWorld = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0.1, 0.2).Normalized());

		// EstimateRelativePose scores with the pixel-unit tangent Sampson error, so
		// it takes rays with their unprojection Jacobians (CamRayWithJac). Points are
		// placed in front of both cameras and projected to pixels.
		var camRays1WithJac = new List<CamRayWithJac>();
		var camRays2WithJac = new List<CamRayWithJac>();
		for (int i = 0; i < 100; ++i)
		{
			Vector3d point3D = RandomEigen.RandomEigenVector3d() + new Vector3d(0, 0, 3);
			camRays1WithJac.AddRange(ProjectedRaysWithJac(camera, cam1FromWorld, [point3D]));
			camRays2WithJac.AddRange(ProjectedRaysWithJac(camera, cam2FromWorld, [point3D]));
		}

		var options = new RansacOptions { MaxError = 1.0 };  // pixels
		var cam2FromCam1 = new Rigid3d();
		bool success = PoseEstimation.EstimateRelativePose(
			options, camRays1WithJac, camRays2WithJac, ref cam2FromCam1, out int numInliers, out bool[] inlierMask);

		await Assert.That(success).IsTrue();
		await Assert.That(Rigid3dNear(cam2FromCam1, cam2FromWorld * cam1FromWorld.Inverse(), 1e-3, 1e-3)).IsTrue();
		await Assert.That(numInliers).IsEqualTo(camRays1WithJac.Count);
		await Assert.That(inlierMask.All(x => x)).IsTrue();
	}

	[Test]
	public async Task EstimateRelativePose_ZeroSentinelRaysExcluded()
	{
		RandomUtils.SetPRNGSeed(0);
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 512.0, 1024, 1024);
		var cam1FromWorld = new Rigid3d();
		var cam2FromWorld = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0.1, 0.2).Normalized());

		var camRays1WithJac = new List<CamRayWithJac>();
		var camRays2WithJac = new List<CamRayWithJac>();
		for (int i = 0; i < 100; ++i)
		{
			Vector3d point3D = RandomEigen.RandomEigenVector3d() + new Vector3d(0, 0, 3);
			camRays1WithJac.AddRange(ProjectedRaysWithJac(camera, cam1FromWorld, [point3D]));
			camRays2WithJac.AddRange(ProjectedRaysWithJac(camera, cam2FromWorld, [point3D]));
		}

		// Emulate unprojectable correspondences: CamRayFromImgWithJac returns nullopt
		// and callers substitute the CamRayWithJac::Zero() sentinel. These must be
		// rejected (infinite tangent Sampson residual) and never counted as inliers,
		// while the pose is still recovered from the remaining correspondences.
		var isZeroed = new bool[camRays1WithJac.Count];
		foreach (int i in new[] { 3, 17, 42, 88 })
		{
			camRays1WithJac[i] = CamRayWithJac.Zero;
			camRays2WithJac[i] = CamRayWithJac.Zero;
			isZeroed[i] = true;
		}

		var options = new RansacOptions { MaxError = 1.0 };  // pixels
		var cam2FromCam1 = new Rigid3d();
		bool success = PoseEstimation.EstimateRelativePose(
			options, camRays1WithJac, camRays2WithJac, ref cam2FromCam1, out int numInliers, out bool[] inlierMask);

		await Assert.That(success).IsTrue();
		await Assert.That(Rigid3dNear(cam2FromCam1, cam2FromWorld * cam1FromWorld.Inverse(), 1e-3, 1e-3)).IsTrue();
		await Assert.That(numInliers).IsEqualTo(camRays1WithJac.Count - 4);
		await Assert.That(inlierMask.Length).IsEqualTo(isZeroed.Length);
		for (int i = 0; i < inlierMask.Length; ++i)
		{
			await Assert.That(inlierMask[i]).IsEqualTo(!isZeroed[i]);
		}
	}

	[Test]
	public async Task RefineAbsolutePose_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
		bool[] inlierMask = Enumerable.Repeat(true, problem.Points2D.Count).ToArray();

		var options = new AbsolutePoseRefinementOptions();
		Rigid3d camFromWorld = problem.Image.CamFromWorld();
		Vector3d axis = RandomEigen.RandomEigenVector3d();
		Vector3d translation = 0.1 * RandomEigen.RandomEigenVector3d();
		camFromWorld *= new Rigid3d(Quaterniond.FromAngleAxis(new AngleAxisd(0.1, axis)), translation);
		Camera camera = problem.Camera.Clone();
		bool success = PoseEstimation.RefineAbsolutePose(
			options, inlierMask, problem.Points2D, problem.Points3D, ref camFromWorld, camera, out Matrix6d camFromWorldCov);

		await Assert.That(success).IsTrue();
		await Assert.That(Rigid3dNear(camFromWorld, problem.Image.CamFromWorld(), 1e-6, 1e-6)).IsTrue();
		await Assert.That(camFromWorld.Rotation.Norm).IsEqualTo(1.0).Within(1e-6);
		await Assert.That(camera == problem.Camera).IsTrue();
		await Assert.That(camFromWorldCov != Matrix6d.Zero).IsTrue();
	}

	[Test]
	public async Task RefineAbsolutePose_RefineFocalLength()
	{
		RandomUtils.SetPRNGSeed(0);
		AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
		bool[] inlierMask = Enumerable.Repeat(true, problem.Points2D.Count).ToArray();

		var options = new AbsolutePoseRefinementOptions { RefineFocalLength = true };
		Rigid3d camFromWorld = problem.Image.CamFromWorld();
		Camera camera = problem.Camera.Clone();
		camera.SetFocalLength(0.9 * camera.FocalLength());
		bool success = PoseEstimation.RefineAbsolutePose(
			options, inlierMask, problem.Points2D, problem.Points3D, ref camFromWorld, camera, out Matrix6d camFromWorldCov);

		await Assert.That(success).IsTrue();
		await Assert.That(Rigid3dNear(camFromWorld, problem.Image.CamFromWorld(), 1e-3, 1e-3)).IsTrue();
		await Assert.That(camFromWorld.Rotation.Norm).IsEqualTo(1.0).Within(1e-6);
		await Assert.That(camera.FocalLength()).IsEqualTo(problem.Camera.FocalLength()).Within(5);
		camera.SetFocalLength(problem.Camera.FocalLength());
		await Assert.That(camera == problem.Camera).IsTrue();
		await Assert.That(camFromWorldCov != Matrix6d.Zero).IsTrue();
	}

	[Test]
	public async Task RefineAbsolutePose_RefineExtraParams()
	{
		RandomUtils.SetPRNGSeed(0);
		AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
		bool[] inlierMask = Enumerable.Repeat(true, problem.Points2D.Count).ToArray();

		var options = new AbsolutePoseRefinementOptions { RefineExtraParams = true };
		Rigid3d camFromWorld = problem.Image.CamFromWorld();
		Camera camera = problem.Camera.Clone();
		camera.Params[3] += 0.1;
		bool success = PoseEstimation.RefineAbsolutePose(
			options, inlierMask, problem.Points2D, problem.Points3D, ref camFromWorld, camera, out Matrix6d camFromWorldCov);

		await Assert.That(success).IsTrue();
		await Assert.That(Rigid3dNear(camFromWorld, problem.Image.CamFromWorld(), 1e-3, 1e-3)).IsTrue();
		await Assert.That(camFromWorld.Rotation.Norm).IsEqualTo(1.0).Within(1e-6);
		await Assert.That(camera.Params[3]).IsEqualTo(problem.Camera.Params[3]).Within(1e-3);
		camera.Params[3] = problem.Camera.Params[3];
		await Assert.That(camera == problem.Camera).IsTrue();
		await Assert.That(camFromWorldCov != Matrix6d.Zero).IsTrue();
	}

	[Test]
	public async Task RefineAbsolutePose_PositionPrior()
	{
		RandomUtils.SetPRNGSeed(0);
		AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
		// Isolate the position-prior-only refinement path without reprojection terms.
		bool[] inlierMask = new bool[problem.Points2D.Count];

		var options = new AbsolutePoseRefinementOptions
		{
			UsePositionPrior = true,
			PositionPriorInWorld = new Vector3d(1.0, 2.0, 3.0),
			PositionPriorCovariance = Matrix3d.Identity,
		};
		var camFromWorld = new Rigid3d(
			Quaterniond.FromAngleAxis(new AngleAxisd(0.2, Vector3d.UnitY)), new Vector3d(0.3, -0.5, 0.7));
		double ComputePositionError(Rigid3d toCheck) =>
			(toCheck.Inverse().Translation - options.PositionPriorInWorld).Norm;
		double initialError = ComputePositionError(camFromWorld);
		Camera camera = problem.Camera.Clone();
		bool success = PoseEstimation.RefineAbsolutePose(
			options, inlierMask, problem.Points2D, problem.Points3D, ref camFromWorld, camera);

		await Assert.That(success).IsTrue();
		await Assert.That(ComputePositionError(camFromWorld)).IsLessThan(initialError);
		await Assert.That(camFromWorld.Rotation.Norm).IsEqualTo(1.0).Within(1e-6);
	}

	[Test]
	public async Task RefineAbsolutePose_PositionPriorCovariance()
	{
		RandomUtils.SetPRNGSeed(0);
		AbsolutePoseProblem problem = CreateAbsolutePoseTestData();
		bool[] inlierMask = Enumerable.Repeat(true, problem.Points2D.Count).ToArray();

		var weakPriorOptions = new AbsolutePoseRefinementOptions
		{
			UsePositionPrior = true,
			PositionPriorInWorld = problem.Image.CamFromWorld().Inverse().Translation + new Vector3d(1.0, -0.7, 0.5),
			// Large covariance = weak prior (high uncertainty).
			PositionPriorCovariance = Matrix3d.Identity,
		};

		AbsolutePoseRefinementOptions strongPriorOptions = weakPriorOptions.Clone();
		// Small covariance = strong prior (low uncertainty).
		strongPriorOptions.PositionPriorCovariance = 0.0001 * Matrix3d.Identity;

		var initialCamFromWorld = new Rigid3d(
			Quaterniond.FromAngleAxis(new AngleAxisd(0.1, Vector3d.UnitX)),
			problem.Image.CamFromWorld().Translation + new Vector3d(0.2, 0.1, -0.1));
		Camera weakPriorCamera = problem.Camera.Clone();
		Camera strongPriorCamera = problem.Camera.Clone();
		Rigid3d weakPriorCamFromWorld = initialCamFromWorld;
		Rigid3d strongPriorCamFromWorld = initialCamFromWorld;

		bool weakSuccess = PoseEstimation.RefineAbsolutePose(
			weakPriorOptions, inlierMask, problem.Points2D, problem.Points3D, ref weakPriorCamFromWorld, weakPriorCamera);
		bool strongSuccess = PoseEstimation.RefineAbsolutePose(
			strongPriorOptions, inlierMask, problem.Points2D, problem.Points3D, ref strongPriorCamFromWorld, strongPriorCamera);

		await Assert.That(weakSuccess).IsTrue();
		await Assert.That(weakPriorCamFromWorld.Rotation.Norm).IsEqualTo(1.0).Within(1e-6);
		await Assert.That(strongSuccess).IsTrue();
		await Assert.That(strongPriorCamFromWorld.Rotation.Norm).IsEqualTo(1.0).Within(1e-6);

		double ComputePositionError(Rigid3d toCheck) =>
			(toCheck.Inverse().Translation - weakPriorOptions.PositionPriorInWorld).Norm;
		await Assert.That(ComputePositionError(strongPriorCamFromWorld)).IsLessThan(ComputePositionError(weakPriorCamFromWorld));
	}

	[Test]
	public async Task RefineEssentialMatrix_Nominal()
	{
		RandomUtils.SetPRNGSeed(0);
		var cam1FromWorld = new Rigid3d();
		var cam2FromWorld = new Rigid3d(Quaterniond.Identity, new Vector3d(1, 0, 0).Normalized());
		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(cam2FromWorld * cam1FromWorld.Inverse());

		var points3D = new Vector3d[150];
		for (int i = 0; i < points3D.Length / 3; ++i)
		{
			points3D[3 * i + 0] = new Vector3d(i * 0.01, 0, 1);
			points3D[3 * i + 1] = new Vector3d(0, i * 0.01, 1);
			points3D[3 * i + 2] = new Vector3d(i * 0.01, i * 0.01, 1);
		}

		// Score with the pixel-unit tangent Sampson error, so each ray carries its
		// unprojection Jacobian. A spherical camera keeps every direction valid.
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.Equirectangular, 0.0, 1000, 500);
		var camRays1WithJac = new List<CamRayWithJac>();
		var camRays2WithJac = new List<CamRayWithJac>();
		foreach (Vector3d point3D in points3D)
		{
			Vector3d ray1 = (cam1FromWorld * point3D).Normalized();
			Vector3d ray2 = (cam2FromWorld * point3D).Normalized();
			camRays1WithJac.Add(camera.CamRayFromImgWithJac(camera.ImgFromCam(ray1)!.Value)!.Value);
			camRays2WithJac.Add(camera.CamRayFromImgWithJac(camera.ImgFromCam(ray2)!.Value)!.Value);
		}

		var cam2FromWorldPerturbed = new Rigid3d(Quaterniond.Identity, new Vector3d(1.02, 0.02, 0.01).Normalized());
		Matrix3d ePerturbed = EssentialMatrix.EssentialMatrixFromPose(cam2FromWorldPerturbed * cam1FromWorld.Inverse());

		Matrix3d eRefined = ePerturbed;
		var options = new SolverOptions();
		PoseEstimation.RefineEssentialMatrix(
			options, camRays1WithJac, camRays2WithJac, Enumerable.Repeat(true, camRays1WithJac.Count).ToArray(), ref eRefined);

		await Assert.That((e - eRefined).Norm()).IsLessThanOrEqualTo((e - ePerturbed).Norm());
	}
}
