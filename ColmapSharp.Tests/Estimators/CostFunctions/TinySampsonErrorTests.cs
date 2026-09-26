// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TinySampsonErrorTests: colmap/estimators/cost_functions/tiny_sampson_error_test.cc ported
// 1:1 (all ten cases), named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Estimators/CostFunctions/TinySampsonError.cs and TinyRelativePoseSampsonError.cs.
// The C++ calls the functors' operator(); here the analytic form is Evaluate(p, r, jacobian)
// and the generic form Evaluate(Real.Cast(p), Real.CastWritable(r)).

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class TinySampsonErrorTests
{
	// The batched functor's squared residuals match ComputeSquaredTangentSampsonError at
	// several 7-parameter poses.
	[Test]
	public async Task TinyTangentSampsonErrorCostFunctor_MatchesSquaredTangentSampsonError()
	{
		CamRayWithJac[] camRays1WithJac =
		[
			Make(new Vector3d(0.1, 0.2, 1), new Matrix3x2d(1.0, 0.1, 0.05, 1.0, 0.2, -0.1)),
			Make(new Vector3d(-0.3, 0.1, 1), new Matrix3x2d(0.9, -0.1, 0.15, 1.1, -0.05, 0.2)),
			Make(new Vector3d(0.2, -0.25, 1), new Matrix3x2d(1.05, 0.0, 0.0, 0.95, 0.1, 0.1)),
		];
		CamRayWithJac[] camRays2WithJac =
		[
			Make(new Vector3d(0.15, -0.1, 1), new Matrix3x2d(1.0, 0.05, -0.1, 1.0, 0.2, 0.0)),
			Make(new Vector3d(0.05, 0.3, 1), new Matrix3x2d(0.8, 0.2, 0.1, 1.2, 0.0, -0.15)),
			Make(new Vector3d(-0.2, -0.15, 1), new Matrix3x2d(1.1, -0.05, 0.05, 0.9, -0.1, 0.1)),
		];

		var functor = new TinyTangentSampsonErrorCostFunctor(camRays1WithJac, camRays2WithJac);

		Quaterniond[] quaternions =
		[
			new AngleAxisd(0.9, new Vector3d(-1, 0.5, 2).Normalized()).ToQuaternion(),
			new AngleAxisd(0.3, new Vector3d(0.2, -1, 0.7).Normalized()).ToQuaternion(),
		];
		Vector3d[] translations =
		[
			new Vector3d(1.0, -2.0, 0.5).Normalized(),
			new Vector3d(-0.4, 0.8, 1.2).Normalized(),
		];

		var log = new ExpectationLog();
		for (int k = 0; k < quaternions.Length; ++k)
		{
			Quaterniond q = quaternions[k].Normalized();
			Vector3d t = translations[k];
			double[] cam2FromCam1 = PoseParams(q, t);

			var residuals = new double[camRays1WithJac.Length];
			if (!log.True(functor.Evaluate(cam2FromCam1, residuals, default), $"k={k}: functor succeeds"))
			{
				break;
			}

			Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(new Rigid3d(q, t));
			for (int i = 0; i < camRays1WithJac.Length; ++i)
			{
				double expected = EssentialMatrix.ComputeSquaredTangentSampsonError(camRays1WithJac[i], camRays2WithJac[i], e);
				log.Near(residuals[i] * residuals[i], expected, 1e-9, $"k={k} residual[{i}]^2");
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// The closed-form 7-parameter Jacobian matches central finite differences.
	[Test]
	public async Task TinyTangentSampsonErrorCostFunctor_JacobianMatchesFiniteDifference()
	{
		CamRayWithJac[] camRays1WithJac = TangentTestRays1();
		CamRayWithJac[] camRays2WithJac = TangentTestRays2();
		var functor = new TinyTangentSampsonErrorCostFunctor(camRays1WithJac, camRays2WithJac);
		int n = camRays1WithJac.Length;

		Quaterniond q = new AngleAxisd(0.7, new Vector3d(0.2, -1, 0.5).Normalized()).ToQuaternion();
		Vector3d t = new Vector3d(0.6, -0.3, 1.0).Normalized();
		double[] p = PoseParams(q, t);

		var residuals = new double[n];
		var jacobian = new double[n * 7];
		await Assert.That(functor.Evaluate(p, residuals, jacobian)).IsTrue();

		var log = new ExpectationLog();
		CheckAgainstFiniteDifferences(log, p, jacobian, n, (x, r) => functor.Evaluate(x, r, default));
		await Assert.That(log.Failures).IsEmpty();
	}

	// The analytic functor agrees with the independent autodiff
	// TangentSampsonErrorCostFunctor (the one RefineRelativePose uses) on both the residual
	// and the 7-parameter Jacobian, pinning the hand-derived Jacobian to a second
	// implementation rather than to finite differences alone.
	[Test]
	public async Task TinyTangentSampsonErrorCostFunctor_MatchesAutodiffCostFunctor()
	{
		CamRayWithJac[] camRays1WithJac = TangentTestRays1();
		CamRayWithJac[] camRays2WithJac = TangentTestRays2();
		var functor = new TinyTangentSampsonErrorCostFunctor(camRays1WithJac, camRays2WithJac);
		int n = camRays1WithJac.Length;

		Quaterniond q = new AngleAxisd(0.6, new Vector3d(0.3, -0.7, 0.5).Normalized()).ToQuaternion();
		Vector3d t = new Vector3d(0.4, 0.9, -0.2).Normalized();
		double[] p = PoseParams(q, t);

		var residuals = new double[n];
		var jacobian = new double[n * 7];
		await Assert.That(functor.Evaluate(p, residuals, jacobian)).IsTrue();

		var log = new ExpectationLog();
		for (int i = 0; i < n; ++i)
		{
			CostFunction cost = TangentSampsonErrorCostFunctor.Create(camRays1WithJac[i], camRays2WithJac[i]);
			var residualAd = new double[1];
			var jacobianAd = new double[7];
			ArraySegment<double>[] paramPtrs = [new(p)];
			ArraySegment<double>[] jacobianAdPtrs = [new(jacobianAd)];
			if (!log.True(cost.Evaluate(paramPtrs, residualAd, jacobianAdPtrs), $"i={i}: autodiff succeeds"))
			{
				break;
			}

			log.Near(residuals[i], residualAd[0], 1e-9, $"residual[{i}]");
			for (int l = 0; l < 7; ++l)
			{
				log.Near(jacobian[i + l * n], jacobianAd[l], 1e-9, $"J({i}, {l})");
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// The batched focal functor's residuals match the pixel-space squared Sampson error of
	// the fundamental matrix F = diag(1/f, 1/f, 1) * E * diag(1/f, 1/f, 1) implied by the
	// pose and shared focal, at several poses and focal lengths.
	[Test]
	public async Task TinyFocalSampsonErrorCostFunctor_MatchesSquaredSampsonError()
	{
		// Principal-point-centered image points (u - cx, v - cy), not calibrated rays.
		Vector2d[] points1 = [new(120.0, -45.0), new(-200.0, 80.0), new(33.0, 210.0)];
		Vector2d[] points2 = [new(95.0, -60.0), new(-180.0, 100.0), new(50.0, 190.0)];

		var functor = new TinyFocalSampsonErrorCostFunctor(points1, points2);

		Quaterniond[] quaternions =
		[
			new AngleAxisd(0.7, new Vector3d(0.3, -1.0, 0.5).Normalized()).ToQuaternion(),
			new AngleAxisd(0.25, new Vector3d(-0.6, 0.4, 1.0).Normalized()).ToQuaternion(),
		];
		Vector3d[] translations =
		[
			new Vector3d(1.0, -0.5, 2.0).Normalized(),
			new Vector3d(-0.7, 1.1, 0.3).Normalized(),
		];
		double[] focals = [900.0, 1500.0];

		var log = new ExpectationLog();
		for (int k = 0; k < quaternions.Length; ++k)
		{
			Quaterniond q = quaternions[k].Normalized();
			Vector3d t = translations[k];
			double focal = focals[k];
			double[] parameters = [.. PoseParams(q, t), Math.Log(focal)];

			var residuals = new double[points1.Length];
			if (!log.True(functor.Evaluate(Real.Cast(parameters), Real.CastWritable(residuals)), $"k={k}: functor succeeds"))
			{
				break;
			}

			// Independent reference: build the pixel-space fundamental matrix and evaluate the
			// squared Sampson error with a matrix-based implementation.
			Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(new Rigid3d(q, t));
			double invF = 1.0 / focal;
			Matrix3d kInv = Matrix3d.FromDiagonal(new Vector3d(invF, invF, 1.0));
			Matrix3d f = kInv * e * kInv;
			var expectedSquared = new List<double>();
			EssentialMatrix.ComputeSquaredSampsonError(points1, points2, f, expectedSquared);

			for (int i = 0; i < points1.Length; ++i)
			{
				log.Near(residuals[i] * residuals[i], expectedSquared[i], 1e-9, $"k={k} residual[{i}]^2");
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// The functor's squared residuals match ComputeSquaredTangentSampsonError under
	// M = E * K1inv, with the uncalibrated view's constant measurement Jacobian
	// d(x, y, 1)/d(x, y), at several poses and focal lengths.
	[Test]
	public async Task TinyOneSidedFocalTangentSampsonErrorCostFunctor_MatchesSquaredTangentSampsonError()
	{
		Vector2d[] points1 = OneSidedFocalTestPoints1();
		CamRayWithJac[] camRays2WithJac = OneSidedFocalTestRays2();
		var functor = new TinyOneSidedFocalTangentSampsonErrorCostFunctor(points1, camRays2WithJac);

		Quaterniond[] quaternions =
		[
			new AngleAxisd(0.7, new Vector3d(0.3, -1.0, 0.5).Normalized()).ToQuaternion(),
			new AngleAxisd(0.25, new Vector3d(-0.6, 0.4, 1.0).Normalized()).ToQuaternion(),
		];
		Vector3d[] translations =
		[
			new Vector3d(1.0, -0.5, 2.0).Normalized(),
			new Vector3d(-0.7, 1.1, 0.3).Normalized(),
		];
		double[] focals1 = [900.0, 1500.0];

		var j1 = new Matrix3x2d(1, 0, 0, 1, 0, 0);

		var log = new ExpectationLog();
		for (int k = 0; k < quaternions.Length; ++k)
		{
			Quaterniond q = quaternions[k].Normalized();
			Vector3d t = translations[k];
			double focal1 = focals1[k];
			double[] parameters = [.. PoseParams(q, t), Math.Log(focal1)];

			var residuals = new double[points1.Length];
			if (!log.True(functor.Evaluate(parameters, residuals, default), $"k={k}: functor succeeds"))
			{
				break;
			}

			Matrix3d m = EssentialMatrix.EssentialMatrixFromPose(new Rigid3d(q, t)) *
				Matrix3d.FromDiagonal(new Vector3d(1.0 / focal1, 1.0 / focal1, 1.0));
			for (int i = 0; i < points1.Length; ++i)
			{
				double expected = EssentialMatrix.ComputeSquaredTangentSampsonError(
					points1[i].Homogeneous(), j1, camRays2WithJac[i].Ray, camRays2WithJac[i].Jacobian, m);
				log.Near(residuals[i] * residuals[i], expected, 1e-9, $"k={k} residual[{i}]^2");
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// The closed-form 8-parameter Jacobian matches central finite differences. This covers
	// the log-focal column, whose correctness rests on the unknown focal entering only
	// through M and never through a measurement Jacobian.
	[Test]
	public async Task TinyOneSidedFocalTangentSampsonErrorCostFunctor_JacobianMatchesFiniteDifference()
	{
		Vector2d[] points1 = OneSidedFocalTestPoints1();
		CamRayWithJac[] camRays2WithJac = OneSidedFocalTestRays2();
		var functor = new TinyOneSidedFocalTangentSampsonErrorCostFunctor(points1, camRays2WithJac);
		int n = points1.Length;

		Quaterniond q = new AngleAxisd(0.7, new Vector3d(0.2, -1, 0.5).Normalized()).ToQuaternion();
		Vector3d t = new Vector3d(0.6, -0.3, 1.0).Normalized();
		double[] p = [.. PoseParams(q, t), Math.Log(1100.0)];

		var residuals = new double[n];
		var jacobian = new double[n * 8];
		await Assert.That(functor.Evaluate(p, residuals, jacobian)).IsTrue();

		var log = new ExpectationLog();
		CheckAgainstFiniteDifferences(log, p, jacobian, n, (x, r) => functor.Evaluate(x, r, default));
		await Assert.That(log.Failures).IsEmpty();
	}

	// The analytic functor agrees with its own autodiff wrapper on both the residual and the
	// 8-parameter Jacobian, pinning the hand-derived Jacobian to a second implementation
	// rather than to finite differences alone.
	[Test]
	public async Task TinyOneSidedFocalTangentSampsonErrorCostFunctor_MatchesAutodiffFunction()
	{
		Vector2d[] points1 = OneSidedFocalTestPoints1();
		CamRayWithJac[] camRays2WithJac = OneSidedFocalTestRays2();
		var functor = new TinyOneSidedFocalTangentSampsonErrorCostFunctor(points1, camRays2WithJac);
		var autodiff = functor.CreateAutoDiffFunction();
		int n = points1.Length;

		Quaterniond q = new AngleAxisd(0.6, new Vector3d(0.3, -0.7, 0.5).Normalized()).ToQuaternion();
		Vector3d t = new Vector3d(0.4, 0.9, -0.2).Normalized();
		double[] p = [.. PoseParams(q, t), Math.Log(1250.0)];

		var residuals = new double[n];
		var jacobian = new double[n * 8];
		await Assert.That(functor.Evaluate(p, residuals, jacobian)).IsTrue();

		var residualsAd = new double[n];
		var jacobianAd = new double[n * 8];
		await Assert.That(autodiff.Evaluate(p, residualsAd, jacobianAd)).IsTrue();

		var log = new ExpectationLog();
		for (int i = 0; i < n; ++i)
		{
			log.Near(residuals[i], residualsAd[i], 1e-9, $"residual[{i}]");
			for (int l = 0; l < 8; ++l)
			{
				log.Near(jacobian[i + l * n], jacobianAd[i + l * n], 1e-9, $"J({i}, {l})");
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// Central differences with step 1e-6 against the column-major n x p.Length Jacobian,
	// within 1e-5, as the C++ tests loop.
	private static void CheckAgainstFiniteDifferences(
		ExpectationLog log, double[] p, double[] jacobian, int n, Func<double[], double[], bool> evaluate)
	{
		const double kEps = 1e-6;
		for (int l = 0; l < p.Length; ++l)
		{
			double[] pPlus = (double[])p.Clone();
			double[] pMinus = (double[])p.Clone();
			pPlus[l] += kEps;
			pMinus[l] -= kEps;
			var resPlus = new double[n];
			var resMinus = new double[n];
			evaluate(pPlus, resPlus);
			evaluate(pMinus, resMinus);
			for (int i = 0; i < n; ++i)
			{
				double finiteDiff = (resPlus[i] - resMinus[i]) / (2 * kEps);
				log.Near(jacobian[i + l * n], finiteDiff, 1e-5, $"J({i}, {l})");
			}
		}
	}

	private static CamRayWithJac Make(Vector3d ray, Matrix3x2d jac) => new(ray.Normalized(), jac);

	// The two-ray sets of the Jacobian tests (the first two rays of the residual test).
	private static CamRayWithJac[] TangentTestRays1() =>
	[
		Make(new Vector3d(0.1, 0.2, 1), new Matrix3x2d(1.0, 0.1, 0.05, 1.0, 0.2, -0.1)),
		Make(new Vector3d(-0.3, 0.1, 1), new Matrix3x2d(0.9, -0.1, 0.15, 1.1, -0.05, 0.2)),
	];

	private static CamRayWithJac[] TangentTestRays2() =>
	[
		Make(new Vector3d(0.15, -0.1, 1), new Matrix3x2d(1.0, 0.05, -0.1, 1.0, 0.2, 0.0)),
		Make(new Vector3d(0.05, 0.3, 1), new Matrix3x2d(0.8, 0.2, 0.1, 1.2, 0.0, -0.15)),
	];

	// Centered image points of the uncalibrated view and bearing rays with unprojection
	// Jacobians for the calibrated view. The rays deliberately include one with z < 0, which
	// no pinhole image plane could represent but a spherical camera observes.
	private static Vector2d[] OneSidedFocalTestPoints1() => [new(120.0, -45.0), new(-200.0, 80.0), new(33.0, 210.0)];

	private static CamRayWithJac[] OneSidedFocalTestRays2() =>
	[
		Make(new Vector3d(0.08, -0.05, 1.0), new Matrix3x2d(1.1e-3, 5e-5, -1e-4, 1.2e-3, 2e-4, 0.0)),
		Make(new Vector3d(-0.9, 0.4, 0.2), new Matrix3x2d(9e-4, -1e-4, 1.5e-4, 1.1e-3, -5e-5, 2e-4)),
		Make(new Vector3d(0.3, 0.7, -0.6), new Matrix3x2d(8e-4, 2e-4, 1e-4, 1.3e-3, 0.0, -1.5e-4)),
	];

	// {q.x, q.y, q.z, q.w, t.x, t.y, t.z}.
	private static double[] PoseParams(Quaterniond q, Vector3d t) => [q.X, q.Y, q.Z, q.W, t.X, t.Y, t.Z];

	// The factorized parameterization always yields a rank-2 matrix with singular
	// values (1, sigma, 0), which is what makes the truncation step of the 8-point
	// algorithm unnecessary.
	[Test]
	public async Task TinyFundamentalSampsonErrorCostFunctor_ParameterizationIsRankTwo()
	{
		Quaterniond qU = new AngleAxisd(0.8, new Vector3d(0.3, -1, 0.5).Normalized()).ToQuaternion();
		Quaterniond qV = new AngleAxisd(-0.4, new Vector3d(1, 0.2, -0.7).Normalized()).ToQuaternion();
		const double sigma = 0.6;
		double[] parameters = Params(qU, qV, sigma);

		Matrix3d f = TinyFundamentalSampsonErrorCostFunctor.FundamentalFromParams(parameters);
		Vector3d singularValues = Svd3d.Compute(f).SingularValues;
		using (Assert.Multiple())
		{
			await Assert.That(singularValues.X).IsEqualTo(1.0).Within(1e-12);
			await Assert.That(singularValues.Y).IsEqualTo(sigma).Within(1e-12);
			await Assert.That(singularValues.Z).IsEqualTo(0.0).Within(1e-12);
		}
	}

	// The batched functor's squared residuals match ComputeSquaredSampsonError at
	// several points in the 9-parameter space.
	[Test]
	public async Task TinyFundamentalSampsonErrorCostFunctor_MatchesSquaredSampsonError()
	{
		Vector2d[] points1 = FundamentalTestPoints1();
		Vector2d[] points2 = FundamentalTestPoints2();
		var functor = new TinyFundamentalSampsonErrorCostFunctor(points1, points2);

		Quaterniond[] quaternionsU =
		[
			new AngleAxisd(0.9, new Vector3d(-1, 0.5, 2).Normalized()).ToQuaternion(),
			new AngleAxisd(0.2, new Vector3d(0.4, 1, -0.3).Normalized()).ToQuaternion(),
		];
		Quaterniond[] quaternionsV =
		[
			new AngleAxisd(0.3, new Vector3d(0.2, -1, 0.7).Normalized()).ToQuaternion(),
			new AngleAxisd(-0.6, new Vector3d(1, 0.1, 0.2).Normalized()).ToQuaternion(),
		];
		double[] sigmas = [0.85, 0.35];

		var log = new ExpectationLog();
		for (int k = 0; k < sigmas.Length; ++k)
		{
			double[] parameters = Params(quaternionsU[k].Normalized(), quaternionsV[k].Normalized(), sigmas[k]);

			var residuals = new double[points1.Length];
			if (!log.True(functor.Evaluate(parameters, residuals, default), $"k={k}: functor succeeds"))
			{
				break;
			}

			var expected = new List<double>();
			EssentialMatrix.ComputeSquaredSampsonError(
				points1, points2, TinyFundamentalSampsonErrorCostFunctor.FundamentalFromParams(parameters), expected);
			for (int i = 0; i < points1.Length; ++i)
			{
				log.Near(residuals[i] * residuals[i], expected[i], 1e-12, $"k={k} residual[{i}]^2");
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// The closed-form 9-parameter Jacobian matches central finite differences.
	[Test]
	public async Task TinyFundamentalSampsonErrorCostFunctor_JacobianMatchesFiniteDifference()
	{
		Vector2d[] points1 = FundamentalTestPoints1();
		Vector2d[] points2 = FundamentalTestPoints2();
		var functor = new TinyFundamentalSampsonErrorCostFunctor(points1, points2);
		int n = points1.Length;

		Quaterniond qU = new AngleAxisd(0.7, new Vector3d(0.2, -1, 0.5).Normalized()).ToQuaternion();
		Quaterniond qV = new AngleAxisd(0.5, new Vector3d(-0.6, 0.3, 1).Normalized()).ToQuaternion();
		double[] p = Params(qU, qV, 0.7);

		var residuals = new double[n];
		var jacobian = new double[n * 9];
		await Assert.That(functor.Evaluate(p, residuals, jacobian)).IsTrue();

		const double kEps = 1e-6;
		var log = new ExpectationLog();
		for (int l = 0; l < 9; ++l)
		{
			double[] pPlus = (double[])p.Clone();
			double[] pMinus = (double[])p.Clone();
			pPlus[l] += kEps;
			pMinus[l] -= kEps;
			var resPlus = new double[n];
			var resMinus = new double[n];
			functor.Evaluate(pPlus, resPlus, default);
			functor.Evaluate(pMinus, resMinus, default);
			for (int i = 0; i < n; ++i)
			{
				double finiteDiff = (resPlus[i] - resMinus[i]) / (2 * kEps);
				log.Near(jacobian[i + l * n], finiteDiff, 1e-5, $"J({i}, {l})");
			}
		}

		await Assert.That(log.Failures).IsEmpty();
	}

	// Image points shared by the fundamental matrix functor tests, in the
	// normalized frame the refiner solves in.
	private static Vector2d[] FundamentalTestPoints1() =>
		[new(0.10, 0.20), new(-0.30, 0.10), new(0.20, -0.25), new(-0.15, -0.35), new(0.40, 0.05)];

	private static Vector2d[] FundamentalTestPoints2() =>
		[new(0.15, -0.10), new(0.05, 0.30), new(-0.20, -0.15), new(0.25, 0.35), new(-0.05, 0.45)];

	// {qU.x, qU.y, qU.z, qU.w, qV.x, qV.y, qV.z, qV.w, sigma}.
	private static double[] Params(Quaterniond qU, Quaterniond qV, double sigma) =>
		[qU.X, qU.Y, qU.Z, qU.W, qV.X, qV.Y, qV.Z, qV.W, sigma];
}
