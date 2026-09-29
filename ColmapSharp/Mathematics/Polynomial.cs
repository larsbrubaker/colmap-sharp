// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Polynomial: colmap/math/polynomial.h and polynomial.cc - evaluation and real/complex
// roots of polynomials given by their coefficients in decreasing order of degree
// (coeffs[0] x^n + ... + coeffs[n]): closed forms for degree 1, 2 and 3 (depressed-cubic
// Cardano / trigonometric form plus one Newton step), Durand-Kerner iteration, and the
// companion-matrix eigenvalues through LinearAlgebra/EigenSolver. Tests:
// ColmapSharp.Tests/Mathematics/PolynomialTests.cs (polynomial_test.cc 1:1).
//
// Tiers: EvaluatePolynomial (real) and the linear/quadratic roots are plain arithmetic in
// COLMAP's order, Tier A. The cubic goes through cbrt/acos/cos (platform libm vs .NET,
// divergence 114),
// Durand-Kerner through std::complex division (libc++ scales by logb/scalbn, .NET's
// Complex uses Smith's algorithm), and the companion matrix through EigenSolver (our
// Francis QR, not Eigen's), so those are Tier B; polynomial_test.cc compares them with
// tolerances. The companion matrix's root order is EigenSolver's Schur-block order, which
// matches Eigen's on the test cases (a complex pair lists +imag first).
//
// C++'s nullptr outputs become C# out parameters that are always filled.

using System.Numerics;

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Mathematics;

/// <summary>
/// Polynomial evaluation and root finding. Port of colmap/math/polynomial.h.
/// </summary>
public static class Polynomial
{
	/// <summary>Horner evaluation at a real x. Port of colmap::EvaluatePolynomial&lt;double&gt;.</summary>
	public static double EvaluatePolynomial(VectorXd coeffs, double x)
	{
		double value = 0.0;
		for (int i = 0; i < coeffs.Length; i++)
		{
			value = value * x + coeffs[i];
		}

		return value;
	}

	/// <summary>Horner evaluation at a complex x. Port of colmap::EvaluatePolynomial&lt;std::complex&lt;double&gt;&gt;.</summary>
	public static Complex EvaluatePolynomial(VectorXd coeffs, Complex x)
	{
		Complex value = 0.0;
		for (int i = 0; i < coeffs.Length; i++)
		{
			value = value * x + coeffs[i];
		}

		return value;
	}

	/// <summary>
	/// The root of coeffs[0] x + coeffs[1]. False when coeffs[0] == 0.
	/// Port of colmap::FindLinearPolynomialRoots.
	/// </summary>
	public static bool FindLinearPolynomialRoots(VectorXd coeffs, out VectorXd real, out VectorXd imag)
	{
		Check.Eq(coeffs.Length, 2);
		real = new VectorXd(0);
		imag = new VectorXd(0);
		if (coeffs[0] == 0)
		{
			return false;
		}

		real = new VectorXd([-coeffs[1] / coeffs[0]]);
		imag = new VectorXd(1);
		return true;
	}

	/// <summary>
	/// The roots of coeffs[0] x^2 + coeffs[1] x + coeffs[2] by the cancellation-free
	/// quadratic formula. Port of colmap::FindQuadraticPolynomialRoots.
	/// </summary>
	public static bool FindQuadraticPolynomialRoots(VectorXd coeffs, out VectorXd real, out VectorXd imag)
	{
		Check.Eq(coeffs.Length, 3);

		double a = coeffs[0];
		if (a == 0)
		{
			return FindLinearPolynomialRoots(coeffs.Tail(2), out real, out imag);
		}

		double b = coeffs[1];
		double c = coeffs[2];
		if (b == 0 && c == 0)
		{
			real = new VectorXd(1);
			imag = new VectorXd(1);
			return true;
		}

		double d = b * b - 4 * a * c;

		if (d >= 0)
		{
			double sqrtD = Math.Sqrt(d);
			real = new VectorXd(2);
			if (b >= 0)
			{
				real[0] = (-b - sqrtD) / (2 * a);
				real[1] = (2 * c) / (-b - sqrtD);
			}
			else
			{
				real[0] = (2 * c) / (-b + sqrtD);
				real[1] = (-b + sqrtD) / (2 * a);
			}

			imag = new VectorXd(2);
		}
		else
		{
			real = VectorXd.Constant(2, -b / (2 * a));
			imag = new VectorXd(2);
			imag[0] = Math.Sqrt(-d) / (2 * a);
			imag[1] = -imag[0];
		}

		return true;
	}

	/// <summary>
	/// Real roots of the monic cubic x^3 + c2 x^2 + c1 x + c0; returns their count (1 or 3),
	/// each refined by one Newton step. Port of colmap::FindCubicPolynomialRoots.
	/// </summary>
	public static int FindCubicPolynomialRoots(double c2, double c1, double c0, out Vector3d real)
	{
		const double k2PiOver3 = 2.09439510239319526263557236234192;
		const double k4PiOver3 = 4.18879020478639052527114472468384;
		double c2Over3 = c2 / 3.0;
		double a = c1 - c2 * c2Over3;
		double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
		double c = b * b / 4.0 + a * a * a / 27.0;
		Span<double> roots = stackalloc double[3];
		int numRoots;
		if (c > 0)
		{
			c = Math.Sqrt(c);
			b *= -0.5;
			roots[0] = Math.Cbrt(b + c) + Math.Cbrt(b - c) - c2Over3;
			numRoots = 1;
		}
		else
		{
			c = 3.0 * b / (2.0 * a) * Math.Sqrt(-3.0 / a);
			double d = 2.0 * Math.Sqrt(-a / 3.0);
			double acosOver3 = Math.Acos(c) / 3.0;
			roots[0] = d * Math.Cos(acosOver3) - c2Over3;
			roots[1] = d * Math.Cos(acosOver3 - k2PiOver3) - c2Over3;
			roots[2] = d * Math.Cos(acosOver3 - k4PiOver3) - c2Over3;
			numRoots = 3;
		}

		// Single Newton iteration.
		for (int i = 0; i < numRoots; ++i)
		{
			double x = roots[i];
			double x2 = x * x;
			double x3 = x * x2;
			double dx = -(x3 + c2 * x2 + c1 * x + c0) / (3 * x2 + 2 * c2 * x + c1);
			roots[i] += dx;
		}

		// C++ leaves the unused entries of the caller's Vector3d untouched; they are zero here.
		real = new Vector3d(roots[0], roots[1], roots[2]);
		return numRoots;
	}

	/// <summary>
	/// All complex roots by Durand-Kerner (Weierstrass) iteration; degrees 1 and 2 use the
	/// closed forms. False for a constant. Port of colmap::FindPolynomialRootsDurandKerner.
	/// </summary>
	public static bool FindPolynomialRootsDurandKerner(VectorXd coeffsAll, out VectorXd real, out VectorXd imag)
	{
		Check.Ge(coeffsAll.Length, 2);

		VectorXd coeffs = RemoveLeadingZeros(coeffsAll);

		int degree = coeffs.Length - 1;

		real = new VectorXd(0);
		imag = new VectorXd(0);
		if (degree <= 0)
		{
			return false;
		}
		else if (degree == 1)
		{
			return FindLinearPolynomialRoots(coeffs, out real, out imag);
		}
		else if (degree == 2)
		{
			return FindQuadraticPolynomialRoots(coeffs, out real, out imag);
		}

		// Initialize roots.
		var roots = new Complex[degree];
		roots[degree - 1] = new Complex(1, 0);
		for (int i = degree - 2; i >= 0; --i)
		{
			roots[i] = roots[i + 1] * new Complex(1, 1);
		}

		// Iterative solver.
		const int kMaxNumIterations = 100;
		const double kMaxRootChange = 1e-10;
		for (int iter = 0; iter < kMaxNumIterations; ++iter)
		{
			double maxRootChange = 0.0;
			for (int i = 0; i < degree; ++i)
			{
				Complex rootI = roots[i];
				Complex numerator = coeffs[0];
				Complex denominator = coeffs[0];
				for (int j = 0; j < degree; ++j)
				{
					numerator = numerator * rootI + coeffs[j + 1];
					if (i != j)
					{
						denominator *= rootI - roots[j];
					}
				}

				Complex rootIChange = numerator / denominator;
				roots[i] = rootI - rootIChange;
				maxRootChange = Math.Max(maxRootChange, Math.Abs(rootIChange.Real));
				maxRootChange = Math.Max(maxRootChange, Math.Abs(rootIChange.Imaginary));
			}

			// Break, if roots do not change anymore.
			if (maxRootChange < kMaxRootChange)
			{
				break;
			}
		}

		real = new VectorXd(degree);
		imag = new VectorXd(degree);
		for (int i = 0; i < degree; i++)
		{
			real[i] = roots[i].Real;
			imag[i] = roots[i].Imaginary;
		}

		return true;
	}

	/// <summary>
	/// All complex roots as the eigenvalues of the companion matrix; degrees 1 and 2 use the
	/// closed forms, and trailing zero coefficients add a root at zero (listed last). False
	/// for a constant or when the eigen solver fails.
	/// Port of colmap::FindPolynomialRootsCompanionMatrix.
	/// </summary>
	public static bool FindPolynomialRootsCompanionMatrix(VectorXd coeffsAll, out VectorXd real, out VectorXd imag)
	{
		Check.Ge(coeffsAll.Length, 2);

		VectorXd coeffs = RemoveLeadingZeros(coeffsAll);

		int degree = coeffs.Length - 1;

		real = new VectorXd(0);
		imag = new VectorXd(0);
		if (degree <= 0)
		{
			return false;
		}
		else if (degree == 1)
		{
			return FindLinearPolynomialRoots(coeffs, out real, out imag);
		}
		else if (degree == 2)
		{
			return FindQuadraticPolynomialRoots(coeffs, out real, out imag);
		}

		// Remove the coefficients where zero is a solution.
		coeffs = RemoveTrailingZeros(coeffs);

		// Check if only zero is a solution.
		if (coeffs.Length == 1)
		{
			real = new VectorXd(1);
			imag = new VectorXd(1);
			return true;
		}

		// Fill the companion matrix.
		int size = coeffs.Length - 1;
		var companion = new MatrixXd(size, size);
		for (int i = 1; i < size; ++i)
		{
			companion[i, i - 1] = 1;
		}

		for (int j = 0; j < size; j++)
		{
			companion[0, j] = -coeffs[j + 1] / coeffs[0];
		}

		// Solve for the roots of the polynomial.
		var solver = new EigenSolver(companion, computeEigenvectors: false);
		if (solver.Info != ComputationInfo.Success)
		{
			return false;
		}

		Complex[] eigenvalues = solver.Eigenvalues();

		// If there are trailing zeros, we must add zero as a solution.
		int effectiveDegree = size < degree ? coeffs.Length : size;

		real = new VectorXd(effectiveDegree);
		imag = new VectorXd(effectiveDegree);
		for (int i = 0; i < size; i++)
		{
			real[i] = eigenvalues[i].Real;
			imag[i] = eigenvalues[i].Imaginary;
		}

		return true;
	}

	/// <summary>Remove leading zero coefficients.</summary>
	private static VectorXd RemoveLeadingZeros(VectorXd coeffs)
	{
		int numZeros = 0;
		for (; numZeros < coeffs.Length; ++numZeros)
		{
			if (coeffs[numZeros] != 0)
			{
				break;
			}
		}

		return coeffs.Tail(coeffs.Length - numZeros);
	}

	/// <summary>Remove trailing zero coefficients.</summary>
	private static VectorXd RemoveTrailingZeros(VectorXd coeffs)
	{
		int numZeros = 0;
		for (; numZeros < coeffs.Length; ++numZeros)
		{
			if (coeffs[coeffs.Length - 1 - numZeros] != 0)
			{
				break;
			}
		}

		return coeffs.Head(coeffs.Length - numZeros);
	}
}
