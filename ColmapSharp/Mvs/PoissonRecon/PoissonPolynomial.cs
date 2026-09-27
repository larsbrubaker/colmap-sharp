// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonPolynomial: thirdparty/PoissonRecon/Polynomial.h and Polynomial.inl - a dense
// polynomial in the monomial basis (coefficients[i] multiplies t^i) with the operations the
// B-spline machinery needs: evaluation, definite integrals, products, scaling and shifting of
// the argument, derivatives, and the de Boor B-spline pieces (BSplineComponent,
// BSplineComponentValues, BinomialCoefficients). It is the leaf of the PoissonRecon port;
// BSplineElements, BSplineEvaluationData, BSplineIntegrationData and BSplineData build on it.
// Named PoissonPolynomial so it cannot collide with ColmapSharp.Mathematics.Polynomial.
//
// Tier A: every operation performs the C++ arithmetic in the C++ order, so results are
// bit-identical to a -ffp-contract=off build of the C++ (oracle/poisson_bspline_harness.cc,
// PoissonBSplineOracleTests). COLMAP itself builds PoissonRecon with -ffast-math
// (docs/CPP_DIVERGENCES.md, entry 74), so the pipeline as a whole is Tier C.
//
// Translation notes:
// - The C++ template parameter Degree becomes a runtime degree (Coefficients.Length - 1).
//   Assigning a lower-degree polynomial into a higher-degree slot (Polynomial<D>'s converting
//   constructor) is WithDegree, which pads with zeros as the C++ does.
// - Polynomial<0>::derivative returns a degree-0 zero polynomial, not a degree -1 one.
// - The degree-0/1/2 operator() specializations are the general Horner loop written out,
//   so one loop serves all degrees with the same operation order.
// - getSolutions (with Factor.h's linear and quadratic Factor) is ported for degrees 1 and 2,
//   the only ones level-set extraction solves; like the C++ generic template, other degrees
//   throw. Factor's complex roots are kept as (real, imaginary) pairs.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// A polynomial in the monomial basis. Port of PoissonRecon's <c>Polynomial&lt;Degree&gt;</c>.
/// </summary>
public sealed class PoissonPolynomial
{
	/// <summary>Coefficients, lowest power first: the value is sum of Coefficients[i] * t^i.</summary>
	public readonly double[] Coefficients;

	/// <summary>The zero polynomial of the given degree.</summary>
	public PoissonPolynomial(int degree)
	{
		if (degree < 0)
		{
			throw new ArgumentOutOfRangeException(nameof(degree), "Polynomial degree must be non-negative.");
		}

		Coefficients = new double[degree + 1];
	}

	/// <summary>A polynomial with the given coefficients (copied), lowest power first.</summary>
	public PoissonPolynomial(ReadOnlySpan<double> coefficients)
	{
		Coefficients = coefficients.ToArray();
	}

	/// <summary>The (template) degree: the number of stored coefficients minus one.</summary>
	public int Degree => Coefficients.Length - 1;

	/// <summary>
	/// This polynomial as one of another degree: extra coefficients are zero, missing ones are
	/// dropped. Port of the converting constructor <c>Polynomial&lt;D&gt;(const Polynomial&lt;D2&gt;&amp;)</c>.
	/// </summary>
	public PoissonPolynomial WithDegree(int degree)
	{
		var q = new PoissonPolynomial(degree);
		for (int i = 0; i <= degree && i < Coefficients.Length; i++)
		{
			q.Coefficients[i] = Coefficients[i];
		}

		return q;
	}

	/// <summary>Horner evaluation at t. Port of <c>operator()( double t )</c>.</summary>
	public double Evaluate(double t)
	{
		double[] c = Coefficients;
		double v = c[^1];
		for (int d = c.Length - 2; d >= 0; d--)
		{
			v = v * t + c[d];
		}

		return v;
	}

	/// <summary>
	/// The definite integral over [tMin, tMax]. Port of <c>integral( double tMin , double tMax )</c>;
	/// the powers are built incrementally (and left alone at +/-DBL_MAX) exactly as there.
	/// </summary>
	public double Integral(double tMin, double tMax)
	{
		double v = 0;
		double t1 = tMin;
		double t2 = tMax;
		for (int i = 0; i < Coefficients.Length; i++)
		{
			v += Coefficients[i] * (t2 - t1) / (i + 1);
			if (t1 != -double.MaxValue && t1 != double.MaxValue)
			{
				t1 *= tMin;
			}

			if (t2 != -double.MaxValue && t2 != double.MaxValue)
			{
				t2 *= tMax;
			}
		}

		return v;
	}

	/// <summary>The antiderivative that vanishes at 0 (degree + 1). Port of <c>integral( void )</c>.</summary>
	public PoissonPolynomial Antiderivative()
	{
		var p = new PoissonPolynomial(Degree + 1);
		p.Coefficients[0] = 0;
		for (int i = 0; i < Coefficients.Length; i++)
		{
			p.Coefficients[i + 1] = Coefficients[i] / (i + 1);
		}

		return p;
	}

	/// <summary>
	/// The derivative: degree - 1, or the degree-0 zero polynomial for a constant. Port of
	/// <c>derivative( void )</c>.
	/// </summary>
	public PoissonPolynomial Derivative()
	{
		if (Degree == 0)
		{
			return new PoissonPolynomial(0);
		}

		var p = new PoissonPolynomial(Degree - 1);
		for (int i = 0; i < Degree; i++)
		{
			p.Coefficients[i] = Coefficients[i + 1] * (i + 1);
		}

		return p;
	}

	/// <summary>True if every coefficient is zero. Port of <c>isZero</c>.</summary>
	public bool IsZero()
	{
		foreach (double c in Coefficients)
		{
			if (c != 0)
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>In-place sum with a polynomial of the same degree. Port of <c>operator +=</c>.</summary>
	public void AddInPlace(PoissonPolynomial p)
	{
		RequireSameDegree(p);
		for (int i = 0; i < Coefficients.Length; i++)
		{
			Coefficients[i] += p.Coefficients[i];
		}
	}

	/// <summary>In-place difference with a polynomial of the same degree. Port of <c>operator -=</c>.</summary>
	public void SubtractInPlace(PoissonPolynomial p)
	{
		RequireSameDegree(p);
		for (int i = 0; i < Coefficients.Length; i++)
		{
			Coefficients[i] -= p.Coefficients[i];
		}
	}

	/// <summary>The product with a scalar. Port of <c>operator * ( double s )</c>.</summary>
	public PoissonPolynomial Multiply(double s)
	{
		var q = new PoissonPolynomial(Degree);
		for (int i = 0; i < Coefficients.Length; i++)
		{
			q.Coefficients[i] = Coefficients[i] * s;
		}

		return q;
	}

	/// <summary>The quotient by a scalar. Port of <c>operator / ( double s )</c>.</summary>
	public PoissonPolynomial Divide(double s)
	{
		var q = new PoissonPolynomial(Degree);
		for (int i = 0; i < Coefficients.Length; i++)
		{
			q.Coefficients[i] = Coefficients[i] / s;
		}

		return q;
	}

	/// <summary>The product of two polynomials (degree is the sum). Port of <c>operator * ( const Polynomial&lt;D2&gt;&amp; )</c>.</summary>
	public PoissonPolynomial Multiply(PoissonPolynomial p)
	{
		var q = new PoissonPolynomial(Degree + p.Degree);
		for (int i = 0; i < Coefficients.Length; i++)
		{
			for (int j = 0; j < p.Coefficients.Length; j++)
			{
				q.Coefficients[i + j] += Coefficients[i] * p.Coefficients[j];
			}
		}

		return q;
	}

	/// <summary>
	/// q(t) = p(t / s): coefficient i is multiplied by s^-i, with the running factor divided by s
	/// each step. Port of <c>scale( double s )</c>.
	/// </summary>
	public PoissonPolynomial Scale(double s)
	{
		var q = new PoissonPolynomial(Coefficients);
		double s2 = 1.0;
		for (int i = 0; i < Coefficients.Length; i++)
		{
			q.Coefficients[i] *= s2;
			s2 /= s;
		}

		return q;
	}

	/// <summary>q(t) = p(t - t0), expanded binomially. Port of <c>shift( double t )</c>.</summary>
	public PoissonPolynomial Shift(double t)
	{
		var q = new PoissonPolynomial(Degree);
		for (int i = 0; i < Coefficients.Length; i++)
		{
			double temp = 1;
			for (int j = i; j >= 0; j--)
			{
				q.Coefficients[j] += Coefficients[i] * temp;
				temp *= -t * j;
				temp /= i - j + 1;
			}
		}

		return q;
	}

	/// <summary>
	/// The i-th piece (0 &lt;= i &lt;= degree) of the degree-<paramref name="degree"/> uniform
	/// B-spline on [0,1], indexed as in de Boor's algorithm so that BSplineComponent(0)(1) = 0
	/// for degree &gt; 0. Port of <c>Polynomial&lt;Degree&gt;::BSplineComponent</c>.
	/// </summary>
	public static PoissonPolynomial BSplineComponent(int degree, int i)
	{
		var p = new PoissonPolynomial(degree);
		if (degree == 0)
		{
			p.Coefficients[0] = 1.0;
			return p;
		}

		// B_d^i(x) = \int_x^1 B_{d-1}^{i}(y) dy + \int_0^x B_{d-1}^{i-1} y dy
		//          = \int_0^1 B_{d-1}^{i}(y) dy - \int_0^x B_{d-1}^{i}(y) dy + \int_0^x B_{d-1}^{i-1} y dy
		if (i < degree)
		{
			PoissonPolynomial lower = BSplineComponent(degree - 1, i).Antiderivative();
			p.SubtractInPlace(lower);
			p.Coefficients[0] += lower.Evaluate(1);
		}

		if (i > 0)
		{
			PoissonPolynomial lower = BSplineComponent(degree - 1, i - 1).Antiderivative();
			p.AddInPlace(lower);
		}

		return p;
	}

	/// <summary>
	/// The values at x in [0,1] of the degree + 1 B-spline pieces overlapping a unit cell, by the
	/// Cox-de Boor recurrence. Port of <c>Polynomial&lt;Degree&gt;::BSplineComponentValues</c>.
	/// </summary>
	public static void BSplineComponentValues(int degree, double x, Span<double> values)
	{
		if (degree == 0)
		{
			values[0] = 1.0;
			return;
		}

		double scale = 1.0 / degree;
		BSplineComponentValues(degree - 1, x, values[1..]);
		values[0] = values[1] * (1.0 - x) * scale;
		for (int i = 1; i < degree; i++)
		{
			double x1 = x - i + degree;
			double x2 = -x + i + 1;
			values[i] = (values[i] * x1 + values[i + 1] * x2) * scale;
		}

		values[degree] *= x * scale;
	}

	/// <summary>
	/// Row <paramref name="degree"/> of Pascal's triangle. Port of
	/// <c>Polynomial&lt;Degree&gt;::BinomialCoefficients</c>.
	/// </summary>
	public static void BinomialCoefficients(int degree, Span<int> coefficients)
	{
		if (degree == 0)
		{
			coefficients[0] = 1;
			return;
		}

		BinomialCoefficients(degree - 1, coefficients);
		int leftValue = 0;
		for (int i = 0; i < degree; i++)
		{
			int temp = coefficients[i];
			coefficients[i] += leftValue;
			leftValue = temp;
		}

		coefficients[degree] = 1;
	}

	/// <summary>
	/// Writes the real solutions t of p(t) = <paramref name="c"/> (roots whose imaginary part is
	/// at most <paramref name="eps"/> in magnitude) to <paramref name="roots"/> and returns how
	/// many there are. Port of <c>Polynomial&lt;1&gt;::getSolutions</c> and
	/// <c>Polynomial&lt;2&gt;::getSolutions</c> with Factor.h's <c>Factor</c>.
	/// </summary>
	public int GetSolutions(double c, Span<double> roots, double eps)
	{
		Span<double> real = stackalloc double[2];
		Span<double> imaginary = stackalloc double[2];
		int factorCount = Degree switch
		{
			1 => FactorLinear(Coefficients[1], Coefficients[0] - c, real, imaginary, eps),
			2 => FactorQuadratic(Coefficients[2], Coefficients[1], Coefficients[0] - c, real, imaginary, eps),
			_ => throw new NotSupportedException($"Can't solve polynomial of degree: {Degree}"),
		};
		int count = 0;
		for (int i = 0; i < factorCount; i++)
		{
			if (Math.Abs(imaginary[i]) <= eps)
			{
				roots[count++] = real[i];
			}
		}

		return count;
	}

	// Factor( a1 , a0 , roots , EPS ).
	private static int FactorLinear(double a1, double a0, Span<double> real, Span<double> imaginary, double eps)
	{
		if (Math.Abs(a1) <= eps)
		{
			return 0;
		}

		real[0] = -a0 / a1;
		imaginary[0] = 0;
		return 1;
	}

	// Factor( a2 , a1 , a0 , roots , EPS ).
	private static int FactorQuadratic(double a2, double a1, double a0, Span<double> real, Span<double> imaginary, double eps)
	{
		if (Math.Abs(a2) <= eps)
		{
			return FactorLinear(a1, a0, real, imaginary, eps);
		}

		double d = (a1 * a1) - (4 * a0 * a2);
		a1 /= 2 * a2;
		if (d < 0)
		{
			d = Math.Sqrt(-d) / (2 * a2);
			real[0] = -a1;
			imaginary[0] = -d;
			real[1] = -a1;
			imaginary[1] = d;
		}
		else
		{
			d = Math.Sqrt(d) / (2 * a2);
			real[0] = -a1 - d;
			imaginary[0] = 0;
			real[1] = -a1 + d;
			imaginary[1] = 0;
		}

		return 2;
	}

	private void RequireSameDegree(PoissonPolynomial p)
	{
		if (p.Degree != Degree)
		{
			throw new ArgumentException($"Polynomial degrees differ: {Degree} and {p.Degree}.", nameof(p));
		}
	}
}
