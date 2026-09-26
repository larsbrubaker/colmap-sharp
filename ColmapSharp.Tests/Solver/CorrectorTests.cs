// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/corrector_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// CorrectorTests (C#-only in the sense of CLAUDE.md: Ceres' test, not COLMAP's): the robust
// loss correction of ColmapSharp/Solver/ResidualBlock.cs, same names, values and
// tolerances. The death tests become "throws": Ceres CHECK-fails, the port throws through
// Check.Gt. The random cases draw from mt19937 with libc++'s uniform_real_distribution, as
// Ceres' test does (on the macOS build); the assertions do not depend on the values drawn.

using ColmapSharp.Mathematics;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class CorrectorTests
{
	// If rho[1] is zero, the Corrector constructor should crash.
	[Test]
	public async Task ZeroGradientDeathTest()
	{
		double[] kRho = [0.0, 0.0, 1.0];
		await Assert.That(() => new Corrector(1.0, kRho)).Throws<ArgumentException>();
	}

	// If rho[1] is negative, the Corrector constructor should crash.
	[Test]
	public async Task NegativeGradientDeathTest()
	{
		double[] kRho = [0.0, -0.1, 1.0];
		await Assert.That(() => new Corrector(1.0, kRho)).Throws<ArgumentException>();
	}

	[Test]
	public async Task ScalarCorrection()
	{
		double[] residuals = [Math.Sqrt(3.0)];
		double[] jacobian = [10.0];
		double sqNorm = residuals[0] * residuals[0];
		double[] kRho = [sqNorm, 0.1, -0.01];

		// In light of the rho'' < 0 clamping now implemented in corrector.cc, alpha = 0
		// whenever rho'' < 0.
		const double kAlpha = 0.0;

		// Thus the expected value of the residual is residual[i] * sqrt(kRho[1]) / (1.0 - kAlpha).
		double kExpectedResidual = residuals[0] * Math.Sqrt(kRho[1]) / (1 - kAlpha);

		// The jacobian in this case will be sqrt(kRho[1]) * (1 - kAlpha) * jacobian.
		double kExpectedJacobian = Math.Sqrt(kRho[1]) * (1 - kAlpha) * jacobian[0];

		var c = new Corrector(sqNorm, kRho);
		c.CorrectJacobian(1, 1, residuals, jacobian);
		c.CorrectResiduals(residuals);

		await Assert.That(residuals[0]).IsEqualTo(kExpectedResidual).Within(1e-6);
		await Assert.That(jacobian[0]).IsEqualTo(kExpectedJacobian).Within(1e-6);
	}

	[Test]
	public async Task ScalarCorrectionZeroResidual()
	{
		double[] residuals = [0.0];
		double[] jacobian = [10.0];
		double sqNorm = residuals[0] * residuals[0];
		double[] kRho = [0.0, 0.1, -0.01];
		var c = new Corrector(sqNorm, kRho);

		// The alpha equation is 1/2 alpha^2 - alpha + 0.0 = 0, i.e. alpha = 1.0 - sqrt(1.0),
		// alpha = 0.0. Thus the expected value of the residual is residual[i] * sqrt(kRho[1]).
		double kExpectedResidual = residuals[0] * Math.Sqrt(kRho[1]);

		// The jacobian in this case will be sqrt(kRho[1]) * jacobian.
		double kExpectedJacobian = Math.Sqrt(kRho[1]) * jacobian[0];

		c.CorrectJacobian(1, 1, residuals, jacobian);
		c.CorrectResiduals(residuals);

		await Assert.That(residuals[0]).IsEqualTo(kExpectedResidual).Within(1e-6);
		await Assert.That(jacobian[0]).IsEqualTo(kExpectedJacobian).Within(1e-6);
	}

	// Scaling behaviour for one dimensional functions.
	[Test]
	public async Task ScalarCorrectionAlphaClamped()
	{
		double[] residuals = [Math.Sqrt(3.0)];
		double[] jacobian = [10.0];
		double sqNorm = residuals[0] * residuals[0];
		double[] kRho = [3, 0.1, -0.1];

		// rho[2] < 0 -> alpha = 0.0
		const double kAlpha = 0.0;

		// Thus the expected value of the residual is residual[i] * sqrt(kRho[1]) / (1.0 - kAlpha).
		double kExpectedResidual = residuals[0] * Math.Sqrt(kRho[1]) / (1.0 - kAlpha);

		// The jacobian in this case will be scaled by sqrt(rho[1]) * (1 - alpha) * J.
		double kExpectedJacobian = Math.Sqrt(kRho[1]) * (1.0 - kAlpha) * jacobian[0];

		var c = new Corrector(sqNorm, kRho);
		c.CorrectJacobian(1, 1, residuals, jacobian);
		c.CorrectResiduals(residuals);

		await Assert.That(residuals[0]).IsEqualTo(kExpectedResidual).Within(1e-6);
		await Assert.That(jacobian[0]).IsEqualTo(kExpectedJacobian).Within(1e-6);
	}

	// Test that the corrected multidimensional residual and jacobians match the expected
	// values and the resulting modified normal equations match the robustified gauss newton
	// approximation.
	[Test]
	public async Task MultidimensionalGaussNewtonApproximation()
	{
		var prng = new Mt19937();
		double maxResidualError = 0.0;
		double maxJacobianError = 0.0;
		double maxGradientError = 0.0;
		for (int iter = 0; iter < 10000; ++iter)
		{
			// Initialize the jacobian (3 x 2, row-major like Ceres' MatrixRef) and residual.
			double[] jac = new double[6];
			double[] res = new double[3];
			for (int i = 0; i < 6; i++)
			{
				jac[i] = LibcxxRandom.UniformReal(prng, 0.0, 1.0);
			}

			for (int i = 0; i < 3; i++)
			{
				res[i] = LibcxxRandom.UniformReal(prng, 0.0, 1.0);
			}

			double sqNorm = Dot(res, res);
			double[] rho = [sqNorm, LibcxxRandom.UniformReal(prng, 0.0, 1.0), LibcxxRandom.UniformReal(prng, -1.0, 1.0)];

			// If rho[2] > 0, then the curvature correction to the correction and the gauss
			// newton approximation will match. Otherwise, we will clamp alpha to 0.
			double kD = 1 + (2 * rho[2] / rho[1] * sqNorm);
			double kAlpha = rho[2] > 0.0 ? 1 - Math.Sqrt(kD) : 0.0;

			// Ground truth values.
			double[] gRes = new double[3];
			double[] gJac = new double[6];
			double[] gGrad = new double[2];
			for (int r = 0; r < 3; r++)
			{
				gRes[r] = Math.Sqrt(rho[1]) / (1.0 - kAlpha) * res[r];
			}

			for (int c = 0; c < 2; c++)
			{
				double rTj = 0.0;
				for (int r = 0; r < 3; r++)
				{
					rTj += res[r] * jac[(r * 2) + c];
				}

				for (int r = 0; r < 3; r++)
				{
					gJac[(r * 2) + c] = Math.Sqrt(rho[1]) * (jac[(r * 2) + c] - (kAlpha / sqNorm * res[r] * rTj));
				}

				gGrad[c] = rho[1] * rTj;
			}

			var corrector = new Corrector(sqNorm, rho);
			corrector.CorrectJacobian(3, 2, res, jac);
			corrector.CorrectResiduals(res);

			// Corrected gradient.
			double[] cGrad = new double[2];
			for (int c = 0; c < 2; c++)
			{
				for (int r = 0; r < 3; r++)
				{
					cGrad[c] += jac[(r * 2) + c] * res[r];
				}
			}

			maxResidualError = Math.Max(maxResidualError, DiffNorm(gRes, res));
			maxJacobianError = Math.Max(maxJacobianError, DiffNorm(gJac, jac));
			maxGradientError = Math.Max(maxGradientError, DiffNorm(gGrad, cGrad));
		}

		await Assert.That(maxResidualError).IsEqualTo(0.0).Within(1e-10);
		await Assert.That(maxJacobianError).IsEqualTo(0.0).Within(1e-10);
		await Assert.That(maxGradientError).IsEqualTo(0.0).Within(1e-10);
	}

	[Test]
	public async Task MultidimensionalGaussNewtonApproximationZeroResidual()
	{
		var prng = new Mt19937();
		double maxResidualError = 0.0;
		double maxJacobianError = 0.0;
		double maxGradientError = 0.0;
		double maxHessianError = 0.0;
		for (int iter = 0; iter < 10000; ++iter)
		{
			// Initialize the jacobian; zero residuals.
			double[] jac = new double[6];
			double[] res = new double[3];
			for (int i = 0; i < 6; i++)
			{
				jac[i] = LibcxxRandom.UniformReal(prng, 0.0, 1.0);
			}

			double sqNorm = Dot(res, res);
			double[] rho = [sqNorm, LibcxxRandom.UniformReal(prng, 0.0, 1.0), LibcxxRandom.UniformReal(prng, -1.0, 1.0)];

			// Ground truth values: with r = 0 the r r' terms vanish.
			double[] gRes = new double[3];
			double[] gJac = new double[6];
			double[] gGrad = new double[2];
			double[] gHess = new double[4];
			for (int i = 0; i < 6; i++)
			{
				gJac[i] = Math.Sqrt(rho[1]) * jac[i];
			}

			for (int a = 0; a < 2; a++)
			{
				for (int b = 0; b < 2; b++)
				{
					double jtj = 0.0;
					for (int r = 0; r < 3; r++)
					{
						jtj += jac[(r * 2) + a] * jac[(r * 2) + b];
					}

					gHess[(a * 2) + b] = rho[1] * jtj;
				}
			}

			var corrector = new Corrector(sqNorm, rho);
			corrector.CorrectJacobian(3, 2, res, jac);
			corrector.CorrectResiduals(res);

			// Corrected gradient and hessian.
			double[] cGrad = new double[2];
			double[] cHess = new double[4];
			for (int a = 0; a < 2; a++)
			{
				for (int r = 0; r < 3; r++)
				{
					cGrad[a] += jac[(r * 2) + a] * res[r];
				}

				for (int b = 0; b < 2; b++)
				{
					for (int r = 0; r < 3; r++)
					{
						cHess[(a * 2) + b] += jac[(r * 2) + a] * jac[(r * 2) + b];
					}
				}
			}

			maxResidualError = Math.Max(maxResidualError, DiffNorm(gRes, res));
			maxJacobianError = Math.Max(maxJacobianError, DiffNorm(gJac, jac));
			maxGradientError = Math.Max(maxGradientError, DiffNorm(gGrad, cGrad));
			maxHessianError = Math.Max(maxHessianError, DiffNorm(gHess, cHess));
		}

		await Assert.That(maxResidualError).IsEqualTo(0.0).Within(1e-10);
		await Assert.That(maxJacobianError).IsEqualTo(0.0).Within(1e-10);
		await Assert.That(maxGradientError).IsEqualTo(0.0).Within(1e-10);
		await Assert.That(maxHessianError).IsEqualTo(0.0).Within(1e-10);
	}

	private static double Dot(double[] a, double[] b)
	{
		double sum = 0.0;
		for (int i = 0; i < a.Length; i++)
		{
			sum += a[i] * b[i];
		}

		return sum;
	}

	private static double DiffNorm(double[] a, double[] b)
	{
		double sum = 0.0;
		for (int i = 0; i < a.Length; i++)
		{
			double d = a[i] - b[i];
			sum += d * d;
		}

		return Math.Sqrt(sum);
	}
}
