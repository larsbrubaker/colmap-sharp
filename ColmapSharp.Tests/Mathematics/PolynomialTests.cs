// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PolynomialTests: colmap/math/polynomial_test.cc 1:1 (same test names, expected values
// and tolerances) for ColmapSharp/Mathematics/Polynomial.cs. EigenMatrixNear(a, b, tol) is
// COLMAP's per-coefficient |a - b| <= tol matcher (util/eigen_matchers.h).

using System.Numerics;

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class PolynomialTests
{
	private delegate bool RootFinder(VectorXd coeffs, out VectorXd real, out VectorXd imag);

	private static VectorXd V(params double[] values) => new(values);

	private static async Task ExpectNear(VectorXd actual, VectorXd expected, double tolerance)
	{
		await Assert.That(actual.Length).IsEqualTo(expected.Length);
		for (int i = 0; i < expected.Length; i++)
		{
			await Assert.That(Math.Abs(actual[i] - expected[i])).IsLessThanOrEqualTo(tolerance).Because($"[{i}] {actual[i]:R} vs {expected[i]:R}");
		}
	}

	private static async Task ExpectEqual(VectorXd actual, VectorXd expected)
	{
		// EXPECT_EQ on Eigen vectors: same size and every coefficient equal, in order.
		bool equal = actual.AsSpan().SequenceEqual(expected.AsSpan());
		await Assert.That(equal).IsTrue()
			.Because($"{string.Join(", ", actual.AsSpan().ToArray())} vs {string.Join(", ", expected.AsSpan().ToArray())}");
	}

	// CHECK_EQUAL_RESULT in polynomial_test.cc.
	private static async Task CheckEqualResult(RootFinder find1, VectorXd coeffs1, RootFinder find2, VectorXd coeffs2)
	{
		bool success1 = find1(coeffs1, out VectorXd real1, out VectorXd imag1);
		bool success2 = find2(coeffs2, out VectorXd real2, out VectorXd imag2);
		await Assert.That(success1).IsEqualTo(success2);
		if (success1)
		{
			await ExpectEqual(real1, real2);
			await ExpectEqual(imag1, imag2);
		}
	}

	[Test]
	public async Task EvaluatePolynomial_Nominal()
	{
		await Assert.That(Polynomial.EvaluatePolynomial(V(1, -3, 3, -5, 10), 1)).IsEqualTo(1 - 3 + 3 - 5 + 10);
		await Assert.That(Math.Abs(Polynomial.EvaluatePolynomial(V(1, -3, 3, -5), 2.0) - (1 * 2 * 2 * 2 - 3 * 2 * 2 + 3 * 2 - 5)))
			.IsLessThanOrEqualTo(1e-6);
	}

	[Test]
	public async Task FindLinearPolynomialRoots_Nominal()
	{
		await Assert.That(Polynomial.FindLinearPolynomialRoots(V(3, -2), out VectorXd real, out VectorXd imag)).IsTrue();
		await Assert.That(real[0]).IsEqualTo(2.0 / 3.0);
		await Assert.That(imag[0]).IsEqualTo(0.0);
		Complex value = Polynomial.EvaluatePolynomial(V(3, -2), new Complex(real[0], imag[0]));
		await Assert.That(Math.Abs(value.Real)).IsLessThanOrEqualTo(1e-6);
		await Assert.That(Math.Abs(value.Imaginary)).IsLessThanOrEqualTo(1e-6);

		await Assert.That(Polynomial.FindLinearPolynomialRoots(V(0, 1), out _, out _)).IsFalse();
	}

	[Test]
	public async Task FindQuadraticPolynomialRoots_Real()
	{
		VectorXd coeffs = V(3, -2, -4);  // negative b
		await Assert.That(Polynomial.FindQuadraticPolynomialRoots(coeffs, out VectorXd real, out VectorXd imag)).IsTrue();
		await ExpectNear(real, V(-0.868517092, 1.535183758), 1e-6);
		await ExpectEqual(imag, V(0, 0));
		await Assert.That(Math.Abs(Polynomial.EvaluatePolynomial(coeffs, new Complex(real[0], imag[0])).Real)).IsLessThanOrEqualTo(1e-6);
		await Assert.That(Math.Abs(Polynomial.EvaluatePolynomial(coeffs, new Complex(real[1], imag[1])).Imaginary)).IsLessThanOrEqualTo(1e-6);
		coeffs = V(1, 5, 2);  // positive b
		await Assert.That(Polynomial.FindQuadraticPolynomialRoots(coeffs, out real, out imag)).IsTrue();
		await ExpectNear(real, V(-4.561552812808831, -0.4384471871911697), 1e-6);
		await ExpectEqual(imag, V(0, 0));
		await Assert.That(Math.Abs(Polynomial.EvaluatePolynomial(coeffs, new Complex(real[0], imag[0])).Real)).IsLessThanOrEqualTo(1e-6);
		await Assert.That(Math.Abs(Polynomial.EvaluatePolynomial(coeffs, new Complex(real[1], imag[1])).Real)).IsLessThanOrEqualTo(1e-6);
	}

	[Test]
	public async Task FindQuadraticPolynomialRoots_Complex()
	{
		VectorXd coeffs = V(0.276025076998578, 0.679702676853675, 0.655098003973841);
		await Assert.That(Polynomial.FindQuadraticPolynomialRoots(coeffs, out VectorXd real, out VectorXd imag)).IsTrue();
		await ExpectNear(real, V(-1.231233560813707, -1.231233560813707), 1e-6);
		await ExpectNear(imag, V(0.925954520440279, -0.925954520440279), 1e-6);
		await Assert.That(Math.Abs(Polynomial.EvaluatePolynomial(coeffs, new Complex(real[0], imag[0])).Real)).IsLessThanOrEqualTo(1e-6);
		await Assert.That(Math.Abs(Polynomial.EvaluatePolynomial(coeffs, new Complex(real[1], imag[1])).Imaginary)).IsLessThanOrEqualTo(1e-6);
	}

	[Test]
	public async Task FindQuadraticPolynomialRoots_ZeroLeadingCoefficient()
	{
		await Assert.That(Polynomial.FindQuadraticPolynomialRoots(V(0, 2, -4), out VectorXd real, out VectorXd imag)).IsTrue();
		await Assert.That(real.Length).IsEqualTo(1);
		await Assert.That(imag.Length).IsEqualTo(1);
		await Assert.That(real[0]).IsEqualTo(2.0);
		await Assert.That(imag[0]).IsEqualTo(0.0);
	}

	[Test]
	public async Task FindQuadraticPolynomialRoots_OnlyZeroSolution()
	{
		await Assert.That(Polynomial.FindQuadraticPolynomialRoots(V(0, 2, 0), out VectorXd real, out VectorXd imag)).IsTrue();
		await Assert.That(real.Length).IsEqualTo(1);
		await Assert.That(imag.Length).IsEqualTo(1);
		await Assert.That(real[0]).IsEqualTo(0.0);
		await Assert.That(imag[0]).IsEqualTo(0.0);
	}

	[Test]
	public async Task FindQuadraticPolynomialRoots_OnlyLeadingCoefficientNonZero()
	{
		await Assert.That(Polynomial.FindQuadraticPolynomialRoots(V(5, 0, 0), out VectorXd real, out VectorXd imag)).IsTrue();
		await Assert.That(real.Length).IsEqualTo(1);
		await Assert.That(imag.Length).IsEqualTo(1);
		await Assert.That(real[0]).IsEqualTo(0.0);
		await Assert.That(imag[0]).IsEqualTo(0.0);
	}

	[Test]
	public async Task FindCubicPolynomialRoots_SingleRoot()
	{
		VectorXd coeffs = V(1, 0.276025076998578, 0.679702676853675, 0.655098003973841);
		await Assert.That(Polynomial.FindCubicPolynomialRoots(coeffs[1], coeffs[2], coeffs[3], out Vector3d real)).IsEqualTo(1);
		await Assert.That(Math.Abs(real.X - -0.68359403879256575)).IsLessThanOrEqualTo(1e-6);
		await Assert.That(Math.Abs(Polynomial.EvaluatePolynomial(coeffs, new Complex(real.X, 0)).Real)).IsLessThanOrEqualTo(1e-6);
	}

	[Test]
	public async Task FindCubicPolynomialRoots_MultiRoot()
	{
		VectorXd coeffs = V(1, -3, -3, 5);
		await Assert.That(Polynomial.FindCubicPolynomialRoots(coeffs[1], coeffs[2], coeffs[3], out Vector3d roots)).IsEqualTo(3);
		double[] sorted = [roots.X, roots.Y, roots.Z];
		Array.Sort(sorted);
		VectorXd real = V(sorted);
		await ExpectNear(real, V(-1.4494897427831781, 1, 3.4494897427831783), 1e-6);
		for (int i = 0; i < 3; ++i)
		{
			await Assert.That(Math.Abs(Polynomial.EvaluatePolynomial(coeffs, new Complex(real[i], 0)).Real)).IsLessThanOrEqualTo(1e-6);
		}

		await Assert.That(Polynomial.FindPolynomialRootsDurandKerner(coeffs, out VectorXd realDurandKerner, out _)).IsTrue();
		double[] sortedDurandKerner = realDurandKerner.AsSpan().ToArray();
		Array.Sort(sortedDurandKerner);
		await ExpectNear(real, V(sortedDurandKerner), 1e-4);
	}

	[Test]
	public async Task FindPolynomialRootsDurandKerner_Nominal()
	{
		await Assert.That(Polynomial.FindPolynomialRootsDurandKerner(V(10, -5, 3, -3, 1), out VectorXd real, out VectorXd imag)).IsTrue();
		// Reference values generated with OpenCV/Matlab.
		await ExpectNear(real, V(-0.201826, -0.201826, 0.451826, 0.451826), 1e-6);
		await ExpectNear(imag, V(-0.627696, 0.627696, 0.160867, -0.160867), 1e-6);
	}

	[Test]
	public async Task FindPolynomialRootsDurandKerner_LinearQuadratic()
	{
		await CheckEqualResult(Polynomial.FindPolynomialRootsDurandKerner, V(1, 2), Polynomial.FindLinearPolynomialRoots, V(1, 2));
		await CheckEqualResult(Polynomial.FindPolynomialRootsDurandKerner, V(0, 0, 1, 2), Polynomial.FindLinearPolynomialRoots, V(1, 2));
		await CheckEqualResult(Polynomial.FindPolynomialRootsDurandKerner, V(1, 2, 3), Polynomial.FindQuadraticPolynomialRoots, V(1, 2, 3));
		await CheckEqualResult(Polynomial.FindPolynomialRootsDurandKerner, V(0, 0, 1, 2, 3), Polynomial.FindQuadraticPolynomialRoots, V(1, 2, 3));
	}

	[Test]
	public async Task FindPolynomialRootsCompanionMatrix_Nominal()
	{
		await Assert.That(Polynomial.FindPolynomialRootsCompanionMatrix(V(10, -5, 3, -3, 1), out VectorXd real, out VectorXd imag)).IsTrue();
		// Reference values generated with OpenCV/Matlab.
		await ExpectNear(real, V(-0.201826, -0.201826, 0.451826, 0.451826), 1e-6);
		await ExpectNear(imag, V(0.627696, -0.627696, 0.160867, -0.160867), 1e-6);
	}

	[Test]
	public async Task FindPolynomialRootsCompanionMatrix_LinearQuadratic()
	{
		await CheckEqualResult(Polynomial.FindPolynomialRootsCompanionMatrix, V(1, 2), Polynomial.FindLinearPolynomialRoots, V(1, 2));
		await CheckEqualResult(Polynomial.FindPolynomialRootsCompanionMatrix, V(0, 0, 1, 2), Polynomial.FindLinearPolynomialRoots, V(1, 2));
		await CheckEqualResult(Polynomial.FindPolynomialRootsCompanionMatrix, V(1, 2, 3), Polynomial.FindQuadraticPolynomialRoots, V(1, 2, 3));
		await CheckEqualResult(Polynomial.FindPolynomialRootsCompanionMatrix, V(0, 0, 1, 2, 3), Polynomial.FindQuadraticPolynomialRoots, V(1, 2, 3));
	}

	[Test]
	public async Task FindPolynomialRootsCompanionMatrix_ZeroSolution()
	{
		await Assert.That(Polynomial.FindPolynomialRootsCompanionMatrix(V(10, -5, 3, -3, 0), out VectorXd real, out VectorXd imag)).IsTrue();
		// Reference values generated with Matlab.
		await ExpectNear(real, V(0.692438, -0.0962191, -0.0962191, 0), 1e-6);
		await ExpectNear(imag, V(0, 0.651148, -0.651148, 0), 1e-6);
	}

	[Test]
	public async Task FindPolynomialRootsCompanionMatrix_OnlyZeroSolution()
	{
		await Assert.That(Polynomial.FindPolynomialRootsCompanionMatrix(V(0, 0, 5, 0, 0, 0), out VectorXd real, out VectorXd imag)).IsTrue();
		await Assert.That(real.Length).IsEqualTo(1);
		await Assert.That(imag.Length).IsEqualTo(1);
		await Assert.That(real[0]).IsEqualTo(0.0);
		await Assert.That(imag[0]).IsEqualTo(0.0);
	}
}
