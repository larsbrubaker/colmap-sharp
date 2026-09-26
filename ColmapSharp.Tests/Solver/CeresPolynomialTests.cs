// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/polynomial_test.cc and the ExpectClose /
// ExpectArraysClose helpers of internal/ceres/test_util.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// CeresPolynomialTests (Ceres' test, not COLMAP's): ColmapSharp/Solver/CeresPolynomial.cs,
// same 26 cases, values and tolerances (Ceres names one of them "Polymomial.
// ConstantInterpolatingPolynomial"; the typo is dropped here). Tier B: the roots of degree
// three and up come from LinearAlgebra/EigenSolver.cs.
// - FindPolynomialRoots always returns both parts here (C# out parameters); the
//   NullPointerAs*Part / BothOutputArgumentsNull cases check only the parts Ceres asks for,
//   so they still exercise the same root finding.
// Not ported: line_search_minimizer_test.cc and line_search_preprocessor_test.cc test Ceres'
// line search minimizer, which is not ported; Ceres has no direct test of ArmijoLineSearch.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class CeresPolynomialTests
{
	// For IEEE-754 doubles, machine precision is about 2e-16.
	private const double Epsilon = 1e-13;
	private const double EpsilonLoose = 1e-9;

	// Return the constant polynomial p(x) = value.
	private static VectorXd ConstantPolynomial(double value) => new([value]);

	// Return the polynomial p(x) = poly(x) * (x - root).
	private static VectorXd AddRealRoot(VectorXd poly, double root)
	{
		var poly2 = new VectorXd(poly.Length + 1);
		for (int i = 0; i < poly.Length; i++)
		{
			poly2[i] += poly[i];
		}

		for (int i = 0; i < poly.Length; i++)
		{
			poly2[i + 1] -= root * poly[i];
		}

		return poly2;
	}

	// Return the polynomial p(x) = poly(x) * (x - real - imag*i) * (x - real + imag*i).
	private static VectorXd AddComplexRootPair(VectorXd poly, double real, double imag)
	{
		var poly2 = new VectorXd(poly.Length + 2);

		// Multiply poly by x^2 - 2real + abs(real,imag)^2
		for (int i = 0; i < poly.Length; i++)
		{
			poly2[i] += poly[i];
		}

		for (int i = 0; i < poly.Length; i++)
		{
			poly2[i + 1] -= 2 * real * poly[i];
		}

		for (int i = 0; i < poly.Length; i++)
		{
			poly2[i + 2] += (real * real + imag * imag) * poly[i];
		}

		return poly2;
	}

	// test_util.cc ExpectClose: relative difference, absolute when either value is zero.
	private static async Task ExpectClose(double x, double y, double maxAbsRelativeDifference)
	{
		if (double.IsInfinity(x) && double.IsInfinity(y))
		{
			await Assert.That(double.IsNegative(x)).IsEqualTo(double.IsNegative(y));
			return;
		}

		if (double.IsNaN(x) && double.IsNaN(y))
		{
			return;
		}

		double absoluteDifference = Math.Abs(x - y);
		double relativeDifference = absoluteDifference / Math.Max(Math.Abs(x), Math.Abs(y));
		if (x == 0 || y == 0)
		{
			// If x or y is exactly zero, then relative difference doesn't have any meaning.
			// Take the absolute difference instead.
			relativeDifference = absoluteDifference;
		}

		await Assert.That(Math.Abs(relativeDifference)).IsLessThanOrEqualTo(maxAbsRelativeDifference);
	}

	// Run a test with the polynomial defined by the real roots.
	private static async Task RunPolynomialTestRealRoots(double[] realRoots, bool useReal, bool useImaginary, double epsilon)
	{
		VectorXd poly = ConstantPolynomial(1.23);
		foreach (double root in realRoots)
		{
			poly = AddRealRoot(poly, root);
		}

		bool success = CeresPolynomial.FindPolynomialRoots(poly, out VectorXd real, out VectorXd imaginary);
		await Assert.That(success).IsTrue();
		int n = realRoots.Length;
		if (useReal)
		{
			await Assert.That(real.Length).IsEqualTo(n);

			// Needed because the roots are not returned in sorted order.
			double[] sorted = real.AsSpan().ToArray();
			Array.Sort(sorted);
			for (int i = 0; i < n; i++)
			{
				await ExpectClose(sorted[i], realRoots[i], epsilon);
			}
		}

		if (useImaginary)
		{
			await Assert.That(imaginary.Length).IsEqualTo(n);
			for (int i = 0; i < n; i++)
			{
				await ExpectClose(imaginary[i], 0.0, epsilon);
			}
		}
	}

	private static double Norm(VectorXd a, VectorXd b)
	{
		double sum = 0.0;
		for (int i = 0; i < a.Length; i++)
		{
			sum += (a[i] - b[i]) * (a[i] - b[i]);
		}

		return Math.Sqrt(sum);
	}

	[Test]
	public async Task InvalidPolynomialOfZeroLengthIsRejected()
	{
		bool success = CeresPolynomial.FindPolynomialRoots(new VectorXd(0), out _, out _);
		await Assert.That(success).IsFalse();
	}

	[Test]
	public async Task ConstantPolynomialReturnsNoRoots()
	{
		bool success = CeresPolynomial.FindPolynomialRoots(ConstantPolynomial(1.23), out VectorXd real, out VectorXd imag);
		await Assert.That(success).IsTrue();
		await Assert.That(real.Length).IsEqualTo(0);
		await Assert.That(imag.Length).IsEqualTo(0);
	}

	[Test]
	public async Task LinearPolynomialWithPositiveRootWorks() => await RunPolynomialTestRealRoots([42.42], true, true, Epsilon);

	[Test]
	public async Task LinearPolynomialWithNegativeRootWorks() => await RunPolynomialTestRealRoots([-42.42], true, true, Epsilon);

	[Test]
	public async Task QuadraticPolynomialWithPositiveRootsWorks() => await RunPolynomialTestRealRoots([1.0, 42.42], true, true, Epsilon);

	[Test]
	public async Task QuadraticPolynomialWithOneNegativeRootWorks() => await RunPolynomialTestRealRoots([-42.42, 1.0], true, true, Epsilon);

	[Test]
	public async Task QuadraticPolynomialWithTwoNegativeRootsWorks() => await RunPolynomialTestRealRoots([-42.42, -1.0], true, true, Epsilon);

	[Test]
	public async Task QuadraticPolynomialWithCloseRootsWorks() => await RunPolynomialTestRealRoots([42.42, 42.43], true, false, EpsilonLoose);

	[Test]
	public async Task QuadraticPolynomialWithComplexRootsWorks()
	{
		VectorXd poly = AddComplexRootPair(ConstantPolynomial(1.23), 42.42, 4.2);
		bool success = CeresPolynomial.FindPolynomialRoots(poly, out VectorXd real, out VectorXd imag);

		await Assert.That(success).IsTrue();
		await Assert.That(real.Length).IsEqualTo(2);
		await Assert.That(imag.Length).IsEqualTo(2);
		await ExpectClose(real[0], 42.42, Epsilon);
		await ExpectClose(real[1], 42.42, Epsilon);
		await ExpectClose(Math.Abs(imag[0]), 4.2, Epsilon);
		await ExpectClose(Math.Abs(imag[1]), 4.2, Epsilon);
		await ExpectClose(Math.Abs(imag[0] + imag[1]), 0.0, Epsilon);
	}

	[Test]
	public async Task QuarticPolynomialWorks() => await RunPolynomialTestRealRoots([1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5], true, true, Epsilon);

	[Test]
	public async Task QuarticPolynomialWithTwoClustersOfCloseRootsWorks() =>
		await RunPolynomialTestRealRoots([1.23e-1, 2.46e-1, 1.23e+5, 2.46e+5], true, true, EpsilonLoose);

	[Test]
	public async Task QuarticPolynomialWithTwoZeroRootsWorks() =>
		await RunPolynomialTestRealRoots([-42.42, 0.0, 0.0, 42.42], true, true, 2 * EpsilonLoose);

	[Test]
	public async Task QuarticMonomialWorks() => await RunPolynomialTestRealRoots([0.0, 0.0, 0.0, 0.0], true, true, Epsilon);

	[Test]
	public async Task NullPointerAsImaginaryPartWorks() =>
		await RunPolynomialTestRealRoots([1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5], true, false, Epsilon);

	[Test]
	public async Task NullPointerAsRealPartWorks() =>
		await RunPolynomialTestRealRoots([1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5], false, true, Epsilon);

	[Test]
	public async Task BothOutputArgumentsNullWorks() =>
		await RunPolynomialTestRealRoots([1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5], false, false, Epsilon);

	[Test]
	public async Task DifferentiateConstantPolynomial()
	{
		// p(x) = 1;
		VectorXd derivative = CeresPolynomial.DifferentiatePolynomial(new VectorXd([1.0]));
		await Assert.That(derivative.Length).IsEqualTo(1);
		await Assert.That(derivative[0]).IsEqualTo(0.0);
	}

	[Test]
	public async Task DifferentiateQuadraticPolynomial()
	{
		// p(x) = x^2 + 2x + 3;
		VectorXd derivative = CeresPolynomial.DifferentiatePolynomial(new VectorXd([1.0, 2.0, 3.0]));
		await Assert.That(derivative.Length).IsEqualTo(2);
		await Assert.That(derivative[0]).IsEqualTo(2.0);
		await Assert.That(derivative[1]).IsEqualTo(2.0);
	}

	[Test]
	public async Task MinimizeConstantPolynomial()
	{
		// p(x) = 1;
		double minX = 0.0;
		double maxX = 1.0;
		CeresPolynomial.MinimizePolynomial(new VectorXd([1.0]), minX, maxX, out double optimalX, out double optimalValue);
		await Assert.That(optimalValue).IsEqualTo(1.0);
		await Assert.That(optimalX).IsLessThanOrEqualTo(maxX);
		await Assert.That(optimalX).IsGreaterThanOrEqualTo(minX);
	}

	[Test]
	public async Task MinimizeLinearPolynomial()
	{
		// p(x) = x - 2 (Ceres' comment; the coefficients are x + 2)
		CeresPolynomial.MinimizePolynomial(new VectorXd([1.0, 2.0]), 0.0, 1.0, out double optimalX, out double optimalValue);
		await Assert.That(optimalX).IsEqualTo(0.0);
		await Assert.That(optimalValue).IsEqualTo(2.0);
	}

	[Test]
	public async Task MinimizeQuadraticPolynomial()
	{
		// p(x) = x^2 - 3 x + 2
		// min_x = 3/2
		// min_value = -1/4;
		var polynomial = new VectorXd([1.0, -3.0, 2.0]);
		CeresPolynomial.MinimizePolynomial(polynomial, -2.0, 2.0, out double optimalX, out double optimalValue);
		await Assert.That(optimalX).IsEqualTo(3.0 / 2.0);
		await Assert.That(optimalValue).IsEqualTo(-1.0 / 4.0);

		CeresPolynomial.MinimizePolynomial(polynomial, -2.0, 1.0, out optimalX, out optimalValue);
		await Assert.That(optimalX).IsEqualTo(1.0);
		await Assert.That(optimalValue).IsEqualTo(0.0);

		CeresPolynomial.MinimizePolynomial(polynomial, 2.0, 3.0, out optimalX, out optimalValue);
		await Assert.That(optimalX).IsEqualTo(2.0);
		await Assert.That(optimalValue).IsEqualTo(0.0);
	}

	[Test]
	public async Task ConstantInterpolatingPolynomial()
	{
		// p(x) = 1.0
		var truePolynomial = new VectorXd([1.0]);
		FunctionSample[] samples = [new FunctionSample(1.0, 1.0)];
		VectorXd polynomial = CeresPolynomial.FindInterpolatingPolynomial(samples);
		await Assert.That(Norm(truePolynomial, polynomial)).IsLessThanOrEqualTo(1e-15);
	}

	[Test]
	public async Task LinearInterpolatingPolynomial()
	{
		// p(x) = 2x - 1
		var truePolynomial = new VectorXd([2.0, -1.0]);
		FunctionSample[] samples = [new FunctionSample(1.0, 1.0, 2.0)];
		VectorXd polynomial = CeresPolynomial.FindInterpolatingPolynomial(samples);
		await Assert.That(Norm(truePolynomial, polynomial)).IsLessThanOrEqualTo(1e-15);
	}

	[Test]
	public async Task QuadraticInterpolatingPolynomial()
	{
		// p(x) = 2x^2 + 3x + 2
		var truePolynomial = new VectorXd([2.0, 3.0, 2.0]);
		FunctionSample[] samples = [new FunctionSample(1.0, 7.0, 7.0), new FunctionSample(-3.0, 11.0)];
		VectorXd polynomial = CeresPolynomial.FindInterpolatingPolynomial(samples);
		await Assert.That(Norm(truePolynomial, polynomial)).IsLessThanOrEqualTo(1e-15);
	}

	[Test]
	public async Task DeficientCubicInterpolatingPolynomial()
	{
		// p(x) = 2x^2 + 3x + 2
		var truePolynomial = new VectorXd([0.0, 2.0, 3.0, 2.0]);
		FunctionSample[] samples = [new FunctionSample(1.0, 7.0, 7.0), new FunctionSample(-3.0, 11.0, -9)];
		VectorXd polynomial = CeresPolynomial.FindInterpolatingPolynomial(samples);
		await Assert.That(Norm(truePolynomial, polynomial)).IsLessThanOrEqualTo(1e-14);
	}

	[Test]
	public async Task CubicInterpolatingPolynomialFromValues()
	{
		// p(x) = x^3 + 2x^2 + 3x + 2
		var truePolynomial = new VectorXd([1.0, 2.0, 3.0, 2.0]);
		FunctionSample[] samples =
		[
			new FunctionSample(1.0, CeresPolynomial.EvaluatePolynomial(truePolynomial, 1.0)),
			new FunctionSample(-3.0, CeresPolynomial.EvaluatePolynomial(truePolynomial, -3.0)),
			new FunctionSample(2.0, CeresPolynomial.EvaluatePolynomial(truePolynomial, 2.0)),
			new FunctionSample(0.0, CeresPolynomial.EvaluatePolynomial(truePolynomial, 0.0)),
		];
		VectorXd polynomial = CeresPolynomial.FindInterpolatingPolynomial(samples);
		await Assert.That(Norm(truePolynomial, polynomial)).IsLessThanOrEqualTo(1e-14);
	}

	[Test]
	public async Task CubicInterpolatingPolynomialFromValuesAndOneGradient()
	{
		// p(x) = x^3 + 2x^2 + 3x + 2
		var truePolynomial = new VectorXd([1.0, 2.0, 3.0, 2.0]);
		VectorXd trueGradientPolynomial = CeresPolynomial.DifferentiatePolynomial(truePolynomial);
		FunctionSample[] samples =
		[
			new FunctionSample(1.0, CeresPolynomial.EvaluatePolynomial(truePolynomial, 1.0)),
			new FunctionSample(-3.0, CeresPolynomial.EvaluatePolynomial(truePolynomial, -3.0)),
			new FunctionSample(
				2.0,
				CeresPolynomial.EvaluatePolynomial(truePolynomial, 2.0),
				CeresPolynomial.EvaluatePolynomial(trueGradientPolynomial, 2.0)),
		];
		VectorXd polynomial = CeresPolynomial.FindInterpolatingPolynomial(samples);
		await Assert.That(Norm(truePolynomial, polynomial)).IsLessThanOrEqualTo(1e-14);
	}

	[Test]
	public async Task CubicInterpolatingPolynomialFromValuesAndGradients()
	{
		// p(x) = x^3 + 2x^2 + 3x + 2
		var truePolynomial = new VectorXd([1.0, 2.0, 3.0, 2.0]);
		VectorXd trueGradientPolynomial = CeresPolynomial.DifferentiatePolynomial(truePolynomial);
		FunctionSample[] samples =
		[
			new FunctionSample(
				-3.0,
				CeresPolynomial.EvaluatePolynomial(truePolynomial, -3.0),
				CeresPolynomial.EvaluatePolynomial(trueGradientPolynomial, -3.0)),
			new FunctionSample(
				2.0,
				CeresPolynomial.EvaluatePolynomial(truePolynomial, 2.0),
				CeresPolynomial.EvaluatePolynomial(trueGradientPolynomial, 2.0)),
		];
		VectorXd polynomial = CeresPolynomial.FindInterpolatingPolynomial(samples);
		await Assert.That(Norm(truePolynomial, polynomial)).IsLessThanOrEqualTo(1e-14);
	}
}
