// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/polynomial.cc, internal/ceres/polynomial.h
// and internal/ceres/function_sample.h (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Ceres' own polynomial helpers, which its line search uses to pick a step size: fit the
// polynomial through function samples (values and derivatives), then minimize it on an
// interval by checking the ends, the middle and the real parts of the derivative's roots.
// These differ from COLMAP's (Mathematics/Polynomial.cs): the companion matrix is balanced
// by powers of two before the eigenvalues are taken, and the interpolation system is solved
// with full-pivoting LU at a zero threshold. The eigenvalues come from
// LinearAlgebra/EigenSolver.cs and the LU from LinearAlgebra/FullPivLU.cs, both written from
// the published algorithms (Tier B). TrustRegionLineSearch.cs is the caller.
// Polynomials are coefficient vectors, highest degree first.

using System.Numerics;

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>
/// ceres::internal::FunctionSample: a sample of a univariate function (the line search's
/// cost along the step) with its value and derivative, each possibly invalid.
/// </summary>
internal struct FunctionSample
{
	/// <summary>An invalid sample at 0.</summary>
	public FunctionSample()
	{
	}

	/// <summary>A sample with a valid value.</summary>
	public FunctionSample(double x, double value)
	{
		X = x;
		Value = value;
		ValueIsValid = true;
	}

	/// <summary>A sample with a valid value and derivative.</summary>
	public FunctionSample(double x, double value, double gradient)
	{
		X = x;
		Value = value;
		ValueIsValid = true;
		Gradient = gradient;
		GradientIsValid = true;
	}

	/// <summary>The argument.</summary>
	public double X;

	/// <summary>The function value.</summary>
	public double Value;

	/// <summary>Whether <see cref="Value"/> is valid.</summary>
	public bool ValueIsValid;

	/// <summary>The derivative.</summary>
	public double Gradient;

	/// <summary>Whether <see cref="Gradient"/> is valid.</summary>
	public bool GradientIsValid;
}

/// <summary>Ceres' polynomial.cc.</summary>
internal static class CeresPolynomial
{
	/// <summary>Horner evaluation, highest degree first (ceres::internal::EvaluatePolynomial).</summary>
	public static double EvaluatePolynomial(VectorXd polynomial, double x)
	{
		double v = 0.0;
		for (int i = 0; i < polynomial.Length; i++)
		{
			v = v * x + polynomial[i];
		}

		return v;
	}

	/// <summary>
	/// FindPolynomialRoots: the (complex) roots after dropping leading zeros; closed forms up
	/// to degree 2, the balanced companion matrix's eigenvalues above. A constant polynomial
	/// has no roots (and returns true); false if the eigenvalue iteration fails.
	/// </summary>
	public static bool FindPolynomialRoots(VectorXd polynomialIn, out VectorXd real, out VectorXd imaginary)
	{
		real = new VectorXd(0);
		imaginary = new VectorXd(0);
		if (polynomialIn.Length == 0)
		{
			return false;
		}

		VectorXd polynomial = RemoveLeadingZeros(polynomialIn);
		int degree = polynomial.Length - 1;

		// Is the polynomial constant? It is correct that there are no roots.
		if (degree == 0)
		{
			return true;
		}

		if (degree == 1)
		{
			real = new VectorXd([-polynomial[1] / polynomial[0]]);
			imaginary = new VectorXd(1);
			return true;
		}

		if (degree == 2)
		{
			FindQuadraticPolynomialRoots(polynomial, out real, out imaginary);
			return true;
		}

		// The degree is now known to be at least 3. For cubic or higher roots we use the
		// method of companion matrices. Divide by leading term.
		double leadingTerm = polynomial[0];
		for (int i = 0; i < polynomial.Length; i++)
		{
			polynomial[i] /= leadingTerm;
		}

		MatrixXd companion = BuildCompanionMatrix(polynomial);
		BalanceCompanionMatrix(companion);
		var solver = new EigenSolver(companion, computeEigenvectors: false);
		if (solver.Info != ComputationInfo.Success)
		{
			return false;
		}

		Complex[] eigenvalues = solver.Eigenvalues();
		real = new VectorXd(eigenvalues.Length);
		imaginary = new VectorXd(eigenvalues.Length);
		for (int i = 0; i < eigenvalues.Length; i++)
		{
			real[i] = eigenvalues[i].Real;
			imaginary[i] = eigenvalues[i].Imaginary;
		}

		return true;
	}

	/// <summary>DifferentiatePolynomial: the derivative; a constant's is the zero constant.</summary>
	public static VectorXd DifferentiatePolynomial(VectorXd polynomial)
	{
		int degree = polynomial.Length - 1;
		Check.Ge(degree, 0);

		// Degree zero polynomials are constants, and their derivative does not result in a
		// smaller degree polynomial, just a degree zero polynomial with value zero.
		if (degree == 0)
		{
			return new VectorXd(1);
		}

		var derivative = new VectorXd(degree);
		for (int i = 0; i < degree; i++)
		{
			derivative[i] = (degree - i) * polynomial[i];
		}

		return derivative;
	}

	/// <summary>
	/// MinimizePolynomial: the minimum of the polynomial on [xMin, xMax], checking the middle,
	/// the ends and the real parts of the critical points, in that order (strictly smaller
	/// wins, so the first of equal values is kept).
	/// </summary>
	public static void MinimizePolynomial(VectorXd polynomial, double xMin, double xMax, out double optimalX, out double optimalValue)
	{
		// We start by inspecting the middle of the interval. Technically this is not needed,
		// but we do this to make this code as close to the minFunc package as possible.
		optimalX = (xMin + xMax) / 2.0;
		optimalValue = EvaluatePolynomial(polynomial, optimalX);

		double xMinValue = EvaluatePolynomial(polynomial, xMin);
		if (xMinValue < optimalValue)
		{
			optimalValue = xMinValue;
			optimalX = xMin;
		}

		double xMaxValue = EvaluatePolynomial(polynomial, xMax);
		if (xMaxValue < optimalValue)
		{
			optimalValue = xMaxValue;
			optimalX = xMax;
		}

		// If the polynomial is linear or constant, we are done.
		if (polynomial.Length <= 2)
		{
			return;
		}

		VectorXd derivative = DifferentiatePolynomial(polynomial);
		if (!FindPolynomialRoots(derivative, out VectorXd rootsReal, out _))
		{
			// Ceres logs "Unable to find the critical points of the interpolating polynomial."
			return;
		}

		// This is a bit of an overkill, as some of the roots may actually have a complex
		// part, but its simpler to just check these values.
		for (int i = 0; i < rootsReal.Length; i++)
		{
			double root = rootsReal[i];
			if (root < xMin || root > xMax)
			{
				continue;
			}

			double value = EvaluatePolynomial(polynomial, root);
			if (value < optimalValue)
			{
				optimalValue = value;
				optimalX = root;
			}
		}
	}

	/// <summary>
	/// FindInterpolatingPolynomial: the polynomial of degree (constraints - 1) matching every
	/// valid value and derivative of the samples.
	/// </summary>
	public static VectorXd FindInterpolatingPolynomial(ReadOnlySpan<FunctionSample> samples)
	{
		int numConstraints = 0;
		foreach (FunctionSample sample in samples)
		{
			numConstraints += (sample.ValueIsValid ? 1 : 0) + (sample.GradientIsValid ? 1 : 0);
		}

		int degree = numConstraints - 1;
		var lhs = new MatrixXd(numConstraints, numConstraints);
		var rhs = new VectorXd(numConstraints);
		int row = 0;
		foreach (FunctionSample sample in samples)
		{
			if (sample.ValueIsValid)
			{
				for (int j = 0; j <= degree; j++)
				{
					lhs[row, j] = Math.Pow(sample.X, degree - j);
				}

				rhs[row] = sample.Value;
				row++;
			}

			if (sample.GradientIsValid)
			{
				for (int j = 0; j < degree; j++)
				{
					lhs[row, j] = (degree - j) * Math.Pow(sample.X, degree - j - 1);
				}

				rhs[row] = sample.Gradient;
				row++;
			}
		}

		// Ceres: "This is a hack" (ceres-solver issue 248): full-pivoting LU with a zero
		// threshold, so a singular system still gives an answer.
		return new FullPivLU(lhs).Solve(rhs, 0.0);
	}

	/// <summary>
	/// MinimizeInterpolatingPolynomial: minimizes the interpolating polynomial on [xMin, xMax],
	/// also trying the samples inside the interval.
	/// </summary>
	public static void MinimizeInterpolatingPolynomial(
		ReadOnlySpan<FunctionSample> samples, double xMin, double xMax, out double optimalX, out double optimalValue)
	{
		VectorXd polynomial = FindInterpolatingPolynomial(samples);
		MinimizePolynomial(polynomial, xMin, xMax, out optimalX, out optimalValue);
		foreach (FunctionSample sample in samples)
		{
			if (sample.X < xMin || sample.X > xMax)
			{
				continue;
			}

			double value = EvaluatePolynomial(polynomial, sample.X);
			if (value < optimalValue)
			{
				optimalX = sample.X;
				optimalValue = value;
			}
		}
	}

	private static VectorXd RemoveLeadingZeros(VectorXd polynomialIn)
	{
		int i = 0;
		while (i < polynomialIn.Length - 1 && polynomialIn[i] == 0.0)
		{
			i++;
		}

		return polynomialIn.Tail(polynomialIn.Length - i);
	}

	private static void FindQuadraticPolynomialRoots(VectorXd polynomial, out VectorXd real, out VectorXd imaginary)
	{
		double a = polynomial[0];
		double b = polynomial[1];
		double c = polynomial[2];
		double d = b * b - 4 * a * c;
		double sqrtD = Math.Sqrt(Math.Abs(d));
		real = new VectorXd(2);
		imaginary = new VectorXd(2);

		// Real roots.
		if (d >= 0)
		{
			// Stable quadratic roots according to BKP Horn.
			// http://people.csail.mit.edu/bkph/articles/Quadratics.pdf
			if (b >= 0)
			{
				real[0] = (-b - sqrtD) / (2.0 * a);
				real[1] = (2.0 * c) / (-b - sqrtD);
			}
			else
			{
				real[0] = (2.0 * c) / (-b + sqrtD);
				real[1] = (-b + sqrtD) / (2.0 * a);
			}

			return;
		}

		// Use the normal quadratic formula for the complex case.
		real[0] = -b / (2.0 * a);
		real[1] = -b / (2.0 * a);
		imaginary[0] = sqrtD / (2.0 * a);
		imaginary[1] = -sqrtD / (2.0 * a);
	}

	// Ones on the subdiagonal, the negated (monic) coefficients, lowest degree first, in the
	// last column.
	private static MatrixXd BuildCompanionMatrix(VectorXd polynomial)
	{
		int degree = polynomial.Length - 1;
		var companion = new MatrixXd(degree, degree);
		for (int i = 1; i < degree; i++)
		{
			companion[i, i - 1] = 1.0;
		}

		for (int i = 0; i < degree; i++)
		{
			companion[i, degree - 1] = -polynomial[degree - i];
		}

		return companion;
	}

	// Greedily scales row/column pairs of the off-diagonal part by powers of two until the
	// 1-norms stop improving; the diagonal is left alone.
	private static void BalanceCompanionMatrix(MatrixXd companion)
	{
		int degree = companion.Rows;
		MatrixXd offDiagonal = companion.Clone();
		for (int i = 0; i < degree; i++)
		{
			offDiagonal[i, i] = 0.0;
		}

		// gamma <= 1 controls how much a change in the scaling has to lower the 1-norm of the
		// companion matrix to be accepted. gamma = 1 seems to lead to cycles (numerical
		// issues?), so we set it slightly lower.
		const double gamma = 0.9;

		// Greedily scale row/column pairs until there is no change.
		bool scalingHasChanged;
		do
		{
			scalingHasChanged = false;
			for (int i = 0; i < degree; i++)
			{
				double rowNorm = 0.0;
				double colNorm = 0.0;
				for (int j = 0; j < degree; j++)
				{
					rowNorm += Math.Abs(offDiagonal[i, j]);
					colNorm += Math.Abs(offDiagonal[j, i]);
				}

				// Decompose row_norm/col_norm into mantissa * 2^exponent, where
				// 0.5 <= mantissa < 1, and keep only the exponent.
				int exponent = FrexpExponent(rowNorm / colNorm) / 2;
				if (exponent != 0)
				{
					double scaledColNorm = Math.ScaleB(colNorm, exponent);
					double scaledRowNorm = Math.ScaleB(rowNorm, -exponent);
					if (scaledColNorm + scaledRowNorm < gamma * (colNorm + rowNorm))
					{
						// Accept the new scaling. (Multiplication by powers of 2 should not
						// introduce rounding errors (ignoring non-normalized numbers and over-
						// or underflow))
						scalingHasChanged = true;
						double rowScale = Math.ScaleB(1.0, -exponent);
						double colScale = Math.ScaleB(1.0, exponent);
						for (int j = 0; j < degree; j++)
						{
							offDiagonal[i, j] *= rowScale;
						}

						for (int j = 0; j < degree; j++)
						{
							offDiagonal[j, i] *= colScale;
						}
					}
				}
			}
		}
		while (scalingHasChanged);

		for (int i = 0; i < degree; i++)
		{
			for (int j = 0; j < degree; j++)
			{
				companion[i, j] = i == j ? companion[i, j] : offDiagonal[i, j];
			}
		}
	}

	// std::frexp's exponent: x = m * 2^e with 0.5 <= |m| < 1; 0 for zero, and (as libc++ on
	// macOS and glibc return) 0 for infinity and NaN.
	private static int FrexpExponent(double x)
	{
		if (x == 0.0 || !double.IsFinite(x))
		{
			return 0;
		}

		return Math.ILogB(x) + 1;
	}
}
