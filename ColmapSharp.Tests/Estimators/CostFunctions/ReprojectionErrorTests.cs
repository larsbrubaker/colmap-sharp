// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReprojectionErrorTests: colmap/estimators/cost_functions/reprojection_error_test.cc ported
// 1:1, one method per gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances.
// Tests ColmapSharp/Estimators/CostFunctions/ReprojectionError.cs,
// AnalyticalReprojectionError.cs and (for the covariance case) CostFunctionUtils.cs, through
// CostFunction.Evaluate; the analytic Jacobians are probed with Solver/GradientChecker.cs,
// the port of the ceres::GradientChecker the C++ uses. The templated helpers
// TestAnalyticalReprojError<CameraModel> become generic methods over the model struct.
// Tier: exact where the C++ uses EXPECT_EQ; the gradient checks keep kJacEps = 1e-4 and
// kResEps = 1e-9.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Sensor;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class ReprojectionErrorTests
{
	private const double EquirectangularCameraWidth = 1000;
	private const double EquirectangularCameraHeight = 500;

	private static double[] PoseParams(Rigid3d pose) =>
		[pose.Rotation.X, pose.Rotation.Y, pose.Rotation.Z, pose.Rotation.W, pose.Translation.X, pose.Translation.Y, pose.Translation.Z];

	private static ArraySegment<double>[] Blocks(params double[][] blocks) => blocks.Select(b => new ArraySegment<double>(b)).ToArray();

	private static async Task ExpectResiduals(CostFunction costFunction, ArraySegment<double>[] parameters, double expected0, double expected1)
	{
		var residuals = new double[2];
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(residuals[0]).IsEqualTo(expected0);
		await Assert.That(residuals[1]).IsEqualTo(expected1);
	}

	[Test]
	public async Task ReprojErrorCostFunctor_Nominal()
	{
		var costFunction = ReprojErrorCostFunctor<SimplePinholeCameraModel>.Create(new Vector2d(0, 0));
		double[] camFromWorld = [0, 0, 0, 1, 0, 0, 0];
		double[] point3D = [0, 0, 1];
		double[] cameraParams = [1, 0, 0];
		ArraySegment<double>[] parameters = Blocks(point3D, camFromWorld, cameraParams);
		await ExpectResiduals(costFunction, parameters, 0, 0);

		point3D[1] = 1;
		await ExpectResiduals(costFunction, parameters, 0, 1);

		cameraParams[0] = 2;
		await ExpectResiduals(costFunction, parameters, 0, 2);

		point3D[0] = -1;
		await ExpectResiduals(costFunction, parameters, -2, 2);

		point3D[2] = -1;
		await ExpectResiduals(costFunction, parameters, 0, 0);
	}

	private static Rigid3d RandomTestPose()
	{
		double angle = RandomUtils.RandomUniformReal(0.0, 2 * Math.PI);
		Quaterniond rotation = new AngleAxisd(angle, new Vector3d(0.1, -0.1, 1).Normalized()).ToQuaternion();
		return new Rigid3d(rotation, new Vector3d(1, 2, 3));
	}

	// Restrict to a realistic field of view. The OpenCV/rational models are only
	// well-conditioned for moderate normalized radii, where the finite-difference (Ridders)
	// gradient check is reliable; the exact analytical Jacobians are validated over the full
	// range against Jets in ModelsJacobianTests.
	private static bool OutsideTestFieldOfView(Rigid3d camFromWorld, Vector3d point3D)
	{
		Vector3d pointInCam = camFromWorld * point3D;
		double hx = pointInCam.X / pointInCam.Z;
		double hy = pointInCam.Y / pointInCam.Z;
		return pointInCam.Z < 0.5 || Math.Sqrt(hx * hx + hy * hy) > 0.8;
	}

	private static async Task ExpectProbeMatchesAutoDiff(CostFunction analytical, CostFunction autoDiff, ArraySegment<double>[] parameterBlocks)
	{
		const double JacEps = 1e-4;
		const double ResEps = 1e-9;

		var autoDiffResiduals = new double[2];
		await Assert.That(autoDiff.Evaluate(parameterBlocks, autoDiffResiduals, default)).IsTrue();

		var gradientChecker = new GradientChecker(analytical, new NumericDiffOptions());
		var results = new GradientChecker.ProbeResults();
		bool probed = gradientChecker.Probe(parameterBlocks, JacEps, results);
		await Assert.That(probed).IsTrue().Because(results.ErrorLog);
		await Assert.That(results.Residuals[0]).IsEqualTo(autoDiffResiduals[0]).Within(ResEps);
		await Assert.That(results.Residuals[1]).IsEqualTo(autoDiffResiduals[1]).Within(ResEps);
	}

	// Validates the fully-variable analytical reprojection error cost function against both
	// the numeric Jacobian and the autodiff residual. The numeric (Ridders) Jacobian
	// comparison uses a looser tolerance than the residual comparison, since finite
	// differences on higher-order distortion models do not reach full double precision; the
	// exact analytical Jacobians are verified separately against Jets in ModelsJacobianTests.
	private static async Task TestAnalyticalReprojError<TModel>(double[] cameraParams)
		where TModel : struct, ICameraModel<TModel>
	{
		RandomUtils.SetPRNGSeed(42);
		var point2D = new Vector2d(200, 300);
		var analyticalCostFunction = new AnalyticalReprojErrorCostFunction<TModel>(point2D);
		CostFunction autoDiffCostFunction = ReprojErrorCostFunctor<TModel>.Create(point2D);

		foreach (double x in new double[] { -1, 0, 1 })
		{
			foreach (double y in new double[] { -1, 0, 1 })
			{
				foreach (double z in new double[] { 0, 1, 2, 3 })
				{
					Rigid3d camFromWorld = RandomTestPose();
					var point3D = new Vector3d(x, y, z);
					if (OutsideTestFieldOfView(camFromWorld, point3D))
					{
						continue;
					}

					ArraySegment<double>[] parameterBlocks = Blocks([x, y, z], PoseParams(camFromWorld), (double[])cameraParams.Clone());
					await ExpectProbeMatchesAutoDiff(analyticalCostFunction, autoDiffCostFunction, parameterBlocks);
				}
			}
		}
	}

	// Validates the fixed-pose analytical reprojection error cost function against both the
	// numeric Jacobian and the autodiff residual. See TestAnalyticalReprojError for the
	// tolerance rationale.
	private static async Task TestAnalyticalReprojErrorConstantPose<TModel>(double[] cameraParams)
		where TModel : struct, ICameraModel<TModel>
	{
		RandomUtils.SetPRNGSeed(42);
		var point2D = new Vector2d(200, 300);
		foreach (double x in new double[] { -1, 0, 1 })
		{
			foreach (double y in new double[] { -1, 0, 1 })
			{
				foreach (double z in new double[] { 0, 1, 2, 3 })
				{
					Rigid3d camFromWorld = RandomTestPose();
					var point3D = new Vector3d(x, y, z);
					if (OutsideTestFieldOfView(camFromWorld, point3D))
					{
						continue;
					}

					var analyticalCostFunction = new AnalyticalReprojErrorConstantPoseCostFunction<TModel>(point2D, camFromWorld);
					CostFunction autoDiffCostFunction = ReprojErrorConstantPoseCostFunctor<TModel>.Create(point2D, camFromWorld);
					ArraySegment<double>[] parameterBlocks = Blocks([x, y, z], (double[])cameraParams.Clone());
					await ExpectProbeMatchesAutoDiff(analyticalCostFunction, autoDiffCostFunction, parameterBlocks);
				}
			}
		}
	}

	private static readonly double[] ThinPrismParams =
		[200, 210, 100, 120, -0.05, 0.02, -0.001, 0.001, 0.001, 0.002, 0.001, -0.001];

	private static readonly double[] RadTanThinPrismParams =
		[200, 210, 100, 120, -0.05, 0.02, -0.005, 0.001, 0.0005, 0.0002, -0.001, 0.001, 0.001, -0.001, 0.0005, -0.0005];

	[Test]
	public async Task ReprojErrorCostFunctor_AnalyticalVersusAutoDiff()
	{
		await TestAnalyticalReprojError<SimplePinholeCameraModel>([200, 100, 120]);
		await TestAnalyticalReprojError<PinholeCameraModel>([200, 210, 100, 120]);
		await TestAnalyticalReprojError<SimpleRadialCameraModel>([200, 100, 120, 0.1]);
		await TestAnalyticalReprojError<RadialCameraModel>([200, 100, 120, 0.1, 0.05]);
		await TestAnalyticalReprojError<OpenCVCameraModel>([200, 210, 100, 120, -0.1, 0.05, -0.001, 0.002]);
		await TestAnalyticalReprojError<FullOpenCVCameraModel>(
			[200, 210, 100, 120, -0.1, 0.05, -0.001, 0.002, 0.01, 0.02, -0.02, 0.01]);
		await TestAnalyticalReprojError<FOVCameraModel>([200, 210, 100, 120, 0.9]);
		await TestAnalyticalReprojError<SimpleRadialFisheyeCameraModel>([200, 100, 120, 0.1]);
		await TestAnalyticalReprojError<RadialFisheyeCameraModel>([200, 100, 120, 0.1, 0.02]);
		await TestAnalyticalReprojError<OpenCVFisheyeCameraModel>([200, 210, 100, 120, -0.05, 0.02, -0.001, 0.001]);
		await TestAnalyticalReprojError<ThinPrismFisheyeCameraModel>(ThinPrismParams);
		await TestAnalyticalReprojError<RadTanThinPrismFisheyeModel>(RadTanThinPrismParams);
		await TestAnalyticalReprojError<SimpleFisheyeCameraModel>([200, 100, 120]);
		await TestAnalyticalReprojError<FisheyeCameraModel>([200, 210, 100, 120]);
		await TestAnalyticalReprojError<SimpleDivisionCameraModel>([200, 100, 120, 0.1]);
		await TestAnalyticalReprojError<DivisionCameraModel>([200, 210, 100, 120, 0.1]);
		await TestAnalyticalReprojError<EUCMCameraModel>([200, 210, 100, 120, 0.6, 1.2]);
		await TestAnalyticalReprojError<EquirectangularCameraModel>([1000, 500]);
	}

	[Test]
	public async Task ReprojErrorConstantPoseCostFunctor_AnalyticalVersusAutoDiff()
	{
		await TestAnalyticalReprojErrorConstantPose<SimplePinholeCameraModel>([200, 100, 120]);
		await TestAnalyticalReprojErrorConstantPose<PinholeCameraModel>([200, 210, 100, 120]);
		await TestAnalyticalReprojErrorConstantPose<SimpleRadialCameraModel>([200, 100, 120, 0.1]);
		await TestAnalyticalReprojErrorConstantPose<RadialCameraModel>([200, 100, 120, 0.1, 0.05]);
		await TestAnalyticalReprojErrorConstantPose<OpenCVCameraModel>([200, 210, 100, 120, -0.1, 0.05, -0.001, 0.002]);
		await TestAnalyticalReprojErrorConstantPose<FullOpenCVCameraModel>(
			[200, 210, 100, 120, -0.1, 0.05, -0.001, 0.002, 0.01, 0.02, -0.02, 0.01]);
		await TestAnalyticalReprojErrorConstantPose<FOVCameraModel>([200, 210, 100, 120, 0.9]);
		await TestAnalyticalReprojErrorConstantPose<SimpleRadialFisheyeCameraModel>([200, 100, 120, 0.1]);
		await TestAnalyticalReprojErrorConstantPose<RadialFisheyeCameraModel>([200, 100, 120, 0.1, 0.02]);
		await TestAnalyticalReprojErrorConstantPose<OpenCVFisheyeCameraModel>([200, 210, 100, 120, -0.05, 0.02, -0.001, 0.001]);
		await TestAnalyticalReprojErrorConstantPose<ThinPrismFisheyeCameraModel>(ThinPrismParams);
		await TestAnalyticalReprojErrorConstantPose<RadTanThinPrismFisheyeModel>(RadTanThinPrismParams);
		await TestAnalyticalReprojErrorConstantPose<SimpleFisheyeCameraModel>([200, 100, 120]);
		await TestAnalyticalReprojErrorConstantPose<FisheyeCameraModel>([200, 210, 100, 120]);
		await TestAnalyticalReprojErrorConstantPose<SimpleDivisionCameraModel>([200, 100, 120, 0.1]);
		await TestAnalyticalReprojErrorConstantPose<DivisionCameraModel>([200, 210, 100, 120, 0.1]);
		await TestAnalyticalReprojErrorConstantPose<EUCMCameraModel>([200, 210, 100, 120, 0.6, 1.2]);
		await TestAnalyticalReprojErrorConstantPose<EquirectangularCameraModel>([1000, 500]);
	}

	[Test]
	public async Task ReprojErrorConstantPoseCostFunctor_Nominal()
	{
		var camFromWorld = new Rigid3d();
		var costFunction = ReprojErrorConstantPoseCostFunctor<SimplePinholeCameraModel>.Create(new Vector2d(0, 0), camFromWorld);
		double[] point3D = [0, 0, 1];
		double[] cameraParams = [1, 0, 0];
		ArraySegment<double>[] parameters = Blocks(point3D, cameraParams);
		await ExpectResiduals(costFunction, parameters, 0, 0);

		point3D[1] = 1;
		await ExpectResiduals(costFunction, parameters, 0, 1);

		cameraParams[0] = 2;
		await ExpectResiduals(costFunction, parameters, 0, 2);

		point3D[0] = -1;
		await ExpectResiduals(costFunction, parameters, -2, 2);
	}

	[Test]
	public async Task ReprojErrorConstantPoint3DCostFunctor_Nominal()
	{
		var point2D = new Vector2d(0, 0);
		double[] point3D = [0, 0, 1];

		double[] camFromWorld = [0, 0, 0, 1, 0, 0, 0];
		double[] cameraParams = [1, 0, 0];
		ArraySegment<double>[] parameters = Blocks(camFromWorld, cameraParams);

		Vector3d P() => new(point3D[0], point3D[1], point3D[2]);

		await ExpectResiduals(ReprojErrorConstantPoint3DCostFunctor<SimplePinholeCameraModel>.Create(point2D, P()), parameters, 0, 0);

		point3D[1] = 1;
		await ExpectResiduals(ReprojErrorConstantPoint3DCostFunctor<SimplePinholeCameraModel>.Create(point2D, P()), parameters, 0, 1);

		cameraParams[0] = 2;
		await ExpectResiduals(ReprojErrorConstantPoint3DCostFunctor<SimplePinholeCameraModel>.Create(point2D, P()), parameters, 0, 2);

		point3D[0] = -1;
		await ExpectResiduals(ReprojErrorConstantPoint3DCostFunctor<SimplePinholeCameraModel>.Create(point2D, P()), parameters, -2, 2);
	}

	[Test]
	public async Task CovarianceWeightedCostFunctor_ReprojErrorCostFunctor()
	{
		double[] camFromWorld = [0, 0, 0, 1, 0, 0, 0];
		double[] point3D = [-1, 1, 1];
		double[] cameraParams = [2, 0, 0];
		ArraySegment<double>[] parameters = Blocks(point3D, camFromWorld, cameraParams);

		var costFunction1 = CovarianceWeightedCostFunctor.Create(
			MatrixXd.Identity(2), new ReprojErrorCostFunctor<SimplePinholeCameraModel>(new Vector2d(0, 0)));
		await ExpectResiduals(costFunction1, parameters, -2, 2);

		MatrixXd cov = MatrixXd.Identity(2);
		cov[0, 0] = 4;
		cov[1, 1] = 4;
		var costFunction2 = CovarianceWeightedCostFunctor.Create(
			cov, new ReprojErrorCostFunctor<SimplePinholeCameraModel>(new Vector2d(0, 0)));
		await ExpectResiduals(costFunction2, parameters, -1, 1);
	}

	[Test]
	public async Task RigReprojErrorCostFunctor_Nominal()
	{
		var costFunction = RigReprojErrorCostFunctor<SimplePinholeCameraModel>.Create(new Vector2d(0, 0));
		double[] camFromRig = [0, 0, 0, 1, 0, 0, -1];
		double[] rigFromWorld = [0, 0, 0, 1, 0, 0, 1];
		double[] point3D = [0, 0, 1];
		double[] cameraParams = [1, 0, 0];
		ArraySegment<double>[] parameters = Blocks(point3D, camFromRig, rigFromWorld, cameraParams);
		await ExpectResiduals(costFunction, parameters, 0, 0);

		point3D[1] = 1;
		await ExpectResiduals(costFunction, parameters, 0, 1);

		cameraParams[0] = 2;
		await ExpectResiduals(costFunction, parameters, 0, 2);

		point3D[0] = -1;
		await ExpectResiduals(costFunction, parameters, -2, 2);
	}

	[Test]
	public async Task RigReprojErrorConstantRigCostFunctor_Nominal()
	{
		var camFromRig = new Rigid3d(Quaterniond.Identity, new Vector3d(0, 0, -1));
		var costFunction = RigReprojErrorConstantRigCostFunctor<SimplePinholeCameraModel>.Create(new Vector2d(0, 0), camFromRig);

		double[] rigFromWorld = [0, 0, 0, 1, 0, 0, 1];
		double[] point3D = [0, 0, 1];
		double[] cameraParams = [1, 0, 0];
		ArraySegment<double>[] parameters = Blocks(point3D, rigFromWorld, cameraParams);
		await ExpectResiduals(costFunction, parameters, 0, 0);

		point3D[1] = 1;
		await ExpectResiduals(costFunction, parameters, 0, 1);

		cameraParams[0] = 2;
		await ExpectResiduals(costFunction, parameters, 0, 2);

		point3D[0] = -1;
		await ExpectResiduals(costFunction, parameters, -2, 2);
	}

	private static double[] Wrap<TModel>(double[] residuals)
		where TModel : struct, ICameraModel<TModel>
	{
		double[] cameraParams = [EquirectangularCameraWidth, EquirectangularCameraHeight];
		ReprojectionErrors.WrapEquirectangularHorizontalSeam<TModel, Real>(Real.Cast(cameraParams), Real.CastWritable(residuals));
		return residuals;
	}

	[Test]
	public async Task WrapEquirectangularHorizontalSeam_Nominal()
	{
		// No-op for non-periodic models, regardless of residual magnitude.
		double[] r = Wrap<SimplePinholeCameraModel>([993.0, -7.0]);
		await Assert.That(r[0]).IsEqualTo(993.0);
		await Assert.That(r[1]).IsEqualTo(-7.0);

		// Equirectangular folds the x-residual into [-w/2, w/2); y is untouched.
		r = Wrap<EquirectangularCameraModel>([1000.0, 5.0]); // Exactly one period -> 0.
		await Assert.That(r[0]).IsEqualTo(0.0);
		await Assert.That(r[1]).IsEqualTo(5.0);

		r = Wrap<EquirectangularCameraModel>([998.0, 0.0]); // Just under a period -> -2.
		await Assert.That(r[0]).IsEqualTo(-2.0);

		r = Wrap<EquirectangularCameraModel>([-998.0, 0.0]);
		await Assert.That(r[0]).IsEqualTo(2.0);

		r = Wrap<EquirectangularCameraModel>([-3.0, 0.0]); // Already minimal -> unchanged.
		await Assert.That(r[0]).IsEqualTo(-3.0);

		r = Wrap<EquirectangularCameraModel>([500.0, 0.0]); // +w/2 boundary folds to -w/2.
		await Assert.That(r[0]).IsEqualTo(-500.0);
	}

	private static async Task ExpectEquirectangularSeamWrap(Func<Vector2d, CostFunction> create, ArraySegment<double>[] parameters)
	{
		CostFunction costFunction = create(new Vector2d(2, EquirectangularCameraHeight / 2));
		await ExpectResiduals(costFunction, parameters, -2, 0); // Raw ~w folded into [-w/2, w/2).
	}

	private static double[] EquirectangularParams() => [EquirectangularCameraWidth, EquirectangularCameraHeight];

	[Test]
	public async Task ReprojErrorCostFunctor_EquirectangularSeamWrap()
	{
		double[] camFromWorld = [0, 0, 0, 1, 0, 0, 0];
		double[] point3D = [0, 0, -1]; // Azimuth = ±π -> projects to x = w.
		await ExpectEquirectangularSeamWrap(
			point2D => ReprojErrorCostFunctor<EquirectangularCameraModel>.Create(point2D),
			Blocks(point3D, camFromWorld, EquirectangularParams()));
	}

	[Test]
	public async Task ReprojErrorConstantPoseCostFunctor_EquirectangularSeamWrap()
	{
		var camFromWorld = new Rigid3d();
		double[] point3D = [0, 0, -1];
		await ExpectEquirectangularSeamWrap(
			point2D => ReprojErrorConstantPoseCostFunctor<EquirectangularCameraModel>.Create(point2D, camFromWorld),
			Blocks(point3D, EquirectangularParams()));
	}

	[Test]
	public async Task ReprojErrorConstantPoint3DCostFunctor_EquirectangularSeamWrap()
	{
		var point3D = new Vector3d(0, 0, -1);
		double[] camFromWorld = [0, 0, 0, 1, 0, 0, 0];
		await ExpectEquirectangularSeamWrap(
			point2D => ReprojErrorConstantPoint3DCostFunctor<EquirectangularCameraModel>.Create(point2D, point3D),
			Blocks(camFromWorld, EquirectangularParams()));
	}

	[Test]
	public async Task RigReprojErrorCostFunctor_EquirectangularSeamWrap()
	{
		double[] camFromRig = [0, 0, 0, 1, 0, 0, 0];
		double[] rigFromWorld = [0, 0, 0, 1, 0, 0, 0];
		double[] point3D = [0, 0, -1];
		await ExpectEquirectangularSeamWrap(
			point2D => RigReprojErrorCostFunctor<EquirectangularCameraModel>.Create(point2D),
			Blocks(point3D, camFromRig, rigFromWorld, EquirectangularParams()));
	}

	[Test]
	public async Task RigReprojErrorConstantRigCostFunctor_EquirectangularSeamWrap()
	{
		var camFromRig = new Rigid3d();
		double[] rigFromWorld = [0, 0, 0, 1, 0, 0, 0];
		double[] point3D = [0, 0, -1];
		await ExpectEquirectangularSeamWrap(
			point2D => RigReprojErrorConstantRigCostFunctor<EquirectangularCameraModel>.Create(point2D, camFromRig),
			Blocks(point3D, rigFromWorld, EquirectangularParams()));
	}
}
