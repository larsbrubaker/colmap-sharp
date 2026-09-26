// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/loss_function_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// LossFunctionTests: Ceres' own loss-function tests, same names, values and tolerances,
// for ColmapSharp/Solver/LossFunctions.cs. COLMAP has no loss-function test of its own.
// Each loss's rho'(s) and rho''(s) are checked against symmetric finite differences of
// rho(s), and rho at s = 0 against its closed form. Ceres' LossFunctionWrapper test is not
// ported because the wrapper is not (COLMAP never uses it).

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class LossFunctionTests
{
	// Compares the values of rho'(s) and rho''(s) computed by the callback with estimates
	// obtained by symmetric finite differencing of rho(s).
	private static async Task AssertLossFunctionIsValid(LossFunction loss, double s)
	{
		await Assert.That(s).IsGreaterThan(0.0);

		// Evaluate rho(s), rho'(s) and rho''(s).
		double[] rho = new double[3];
		loss.Evaluate(s, rho);

		// Use symmetric finite differencing to estimate rho'(s) and rho''(s).
		const double kH = 1e-4;
		double[] fwd = new double[3];
		double[] bwd = new double[3];
		loss.Evaluate(s + kH, fwd);
		loss.Evaluate(s - kH, bwd);

		// First derivative.
		double fd1 = (fwd[0] - bwd[0]) / (2 * kH);
		await Assert.That(Math.Abs(fd1 - rho[1])).IsLessThanOrEqualTo(1e-6);

		// Second derivative.
		double fd2 = (fwd[0] - 2 * rho[0] + bwd[0]) / (kH * kH);
		await Assert.That(Math.Abs(fd2 - rho[2])).IsLessThanOrEqualTo(1e-6);
	}

	private static async Task AssertRhoAtZero(LossFunction loss, double rho0, double rho1, double rho2)
	{
		double[] rho = new double[3];
		loss.Evaluate(0.0, rho);
		await Assert.That(Math.Abs(rho[0] - rho0)).IsLessThanOrEqualTo(1e-6);
		await Assert.That(Math.Abs(rho[1] - rho1)).IsLessThanOrEqualTo(1e-6);
		await Assert.That(Math.Abs(rho[2] - rho2)).IsLessThanOrEqualTo(1e-6);
	}

	// Try two values of the scaling a = 0.7 and 1.3 (where scaling makes sense) and of the
	// squared norm s = 0.357 and 1.792. For the Huber loss this exercises both code paths
	// (small and large values of s).
	[Test]
	public async Task TrivialLoss()
	{
		await AssertLossFunctionIsValid(new TrivialLoss(), 0.357);
		await AssertLossFunctionIsValid(new TrivialLoss(), 1.792);
		// Check that at s = 0: rho = [0, 1, 0].
		await AssertRhoAtZero(new TrivialLoss(), 0.0, 1.0, 0.0);
	}

	[Test]
	public async Task HuberLoss()
	{
		await AssertLossFunctionIsValid(new HuberLoss(0.7), 0.357);
		await AssertLossFunctionIsValid(new HuberLoss(0.7), 1.792);
		await AssertLossFunctionIsValid(new HuberLoss(1.3), 0.357);
		await AssertLossFunctionIsValid(new HuberLoss(1.3), 1.792);
		// Check that at s = 0: rho = [0, 1, 0].
		await AssertRhoAtZero(new HuberLoss(0.7), 0.0, 1.0, 0.0);
	}

	[Test]
	public async Task SoftLOneLoss()
	{
		await AssertLossFunctionIsValid(new SoftLOneLoss(0.7), 0.357);
		await AssertLossFunctionIsValid(new SoftLOneLoss(0.7), 1.792);
		await AssertLossFunctionIsValid(new SoftLOneLoss(1.3), 0.357);
		await AssertLossFunctionIsValid(new SoftLOneLoss(1.3), 1.792);
		// Check that at s = 0: rho = [0, 1, -1 / (2 * a^2)].
		await AssertRhoAtZero(new SoftLOneLoss(0.7), 0.0, 1.0, -0.5 / (0.7 * 0.7));
	}

	[Test]
	public async Task CauchyLoss()
	{
		await AssertLossFunctionIsValid(new CauchyLoss(0.7), 0.357);
		await AssertLossFunctionIsValid(new CauchyLoss(0.7), 1.792);
		await AssertLossFunctionIsValid(new CauchyLoss(1.3), 0.357);
		await AssertLossFunctionIsValid(new CauchyLoss(1.3), 1.792);
		// Check that at s = 0: rho = [0, 1, -1 / a^2].
		await AssertRhoAtZero(new CauchyLoss(0.7), 0.0, 1.0, -1.0 / (0.7 * 0.7));
	}

	[Test]
	public async Task ArctanLoss()
	{
		await AssertLossFunctionIsValid(new ArctanLoss(0.7), 0.357);
		await AssertLossFunctionIsValid(new ArctanLoss(0.7), 1.792);
		await AssertLossFunctionIsValid(new ArctanLoss(1.3), 0.357);
		await AssertLossFunctionIsValid(new ArctanLoss(1.3), 1.792);
		// Check that at s = 0: rho = [0, 1, 0].
		await AssertRhoAtZero(new ArctanLoss(0.7), 0.0, 1.0, 0.0);
	}

	[Test]
	public async Task TolerantLoss()
	{
		await AssertLossFunctionIsValid(new TolerantLoss(0.7, 0.4), 0.357);
		await AssertLossFunctionIsValid(new TolerantLoss(0.7, 0.4), 1.792);
		await AssertLossFunctionIsValid(new TolerantLoss(0.7, 0.4), 55.5);
		await AssertLossFunctionIsValid(new TolerantLoss(1.3, 0.1), 0.357);
		await AssertLossFunctionIsValid(new TolerantLoss(1.3, 0.1), 1.792);
		await AssertLossFunctionIsValid(new TolerantLoss(1.3, 0.1), 55.5);
		// Check the value at zero is actually zero.
		double[] rho = new double[3];
		new TolerantLoss(0.7, 0.4).Evaluate(0.0, rho);
		await Assert.That(Math.Abs(rho[0])).IsLessThanOrEqualTo(1e-6);
		// Check that loss before and after the approximation threshold are good.
		// A threshold of 36.7 is used by the implementation.
		await AssertLossFunctionIsValid(new TolerantLoss(20.0, 1.0), 20.0 + 36.6);
		await AssertLossFunctionIsValid(new TolerantLoss(20.0, 1.0), 20.0 + 36.7);
		await AssertLossFunctionIsValid(new TolerantLoss(20.0, 1.0), 20.0 + 36.8);
		await AssertLossFunctionIsValid(new TolerantLoss(20.0, 1.0), 20.0 + 1000.0);
	}

	[Test]
	public async Task TukeyLoss()
	{
		await AssertLossFunctionIsValid(new TukeyLoss(0.7), 0.357);
		await AssertLossFunctionIsValid(new TukeyLoss(0.7), 1.792);
		await AssertLossFunctionIsValid(new TukeyLoss(1.3), 0.357);
		await AssertLossFunctionIsValid(new TukeyLoss(1.3), 1.792);
		// Check that at s = 0: rho = [0, 1, -2 / a^2].
		await AssertRhoAtZero(new TukeyLoss(0.7), 0.0, 1.0, -2.0 / (0.7 * 0.7));
	}

	[Test]
	public async Task ComposedLoss()
	{
		var c1 = new ComposedLoss(new HuberLoss(0.7), new CauchyLoss(1.3));
		await AssertLossFunctionIsValid(c1, 0.357);
		await AssertLossFunctionIsValid(c1, 1.792);

		var c2 = new ComposedLoss(new CauchyLoss(0.7), new HuberLoss(1.3));
		await AssertLossFunctionIsValid(c2, 0.357);
		await AssertLossFunctionIsValid(c2, 1.792);
	}

	[Test]
	public async Task ScaledLoss()
	{
		// Wrap a few loss functions, and a few scale factors.
		await AssertLossFunctionIsValid(new ScaledLoss(null, 6), 0.323);
		await AssertLossFunctionIsValid(new ScaledLoss(new TrivialLoss(), 10), 0.357);
		await AssertLossFunctionIsValid(new ScaledLoss(new HuberLoss(0.7), 0.1), 1.792);
		await AssertLossFunctionIsValid(new ScaledLoss(new SoftLOneLoss(1.3), 0.1), 1.792);
		await AssertLossFunctionIsValid(new ScaledLoss(new CauchyLoss(1.3), 10), 1.792);
		await AssertLossFunctionIsValid(new ScaledLoss(new ArctanLoss(1.3), 10), 1.792);
		await AssertLossFunctionIsValid(new ScaledLoss(new TolerantLoss(1.3, 0.1), 10), 1.792);
		await AssertLossFunctionIsValid(
			new ScaledLoss(new ComposedLoss(new HuberLoss(0.8), new TolerantLoss(1.3, 0.5)), 10),
			1.792);
	}
}
