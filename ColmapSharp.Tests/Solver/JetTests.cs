// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// JetTests (C#-only; COLMAP has no test of ceres::Jet, and Ceres' jet_test.cc is not part of
// this port): the elementary-function rules of ColmapSharp/Solver/Jet.cs that the camera
// model oracle does not reach, because IterativeUndistortion only runs +, -, * and / on
// Jets. Each rule is checked two ways:
// - against its closed-form derivative, written out here, bit for bit where the rule's
//   formula is the closed form's own grouping (Ceres 2.2.0's), and
// - against a central finite difference of the value part, (f(a + h) - f(a - h)) / 2h,
//   which checks the calculus independently of how the formula is grouped.
// Both derivative slots are checked, with a seed that differs per slot, so a rule that
// mixes up v0 and v1 fails.

using ColmapSharp.Solver;

using Jet2 = ColmapSharp.Solver.Jet<ColmapSharp.Solver.Grad2>;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class JetTests
{
	// Finite-difference step and tolerance: central differences have O(h^2) truncation and
	// O(eps / h) rounding error, so h = 1e-5 leaves about 1e-10 relative; 1e-8 is loose
	// enough for every function and point below and tight enough to catch a wrong rule.
	private const double H = 1e-5;
	private const double FiniteDifferenceTolerance = 1e-8;

	// Seeds for the two derivative slots: d/dt of the input a + t*(V0Seed, V1Seed).
	private const double V0Seed = 1.0;
	private const double V1Seed = -0.5;

	private static Jet2 Input(double a) => Jet2.Create(a, [V0Seed, V1Seed]);

	private delegate Jet2 JetFunction(in Jet2 a);

	private static async Task CheckFiniteDifference(JetFunction f, Func<double, double> g, double a)
	{
		Jet2 result = f(Input(a));
		double slope = (g(a + H) - g(a - H)) / (2 * H);
		double scale = Math.Max(1.0, Math.Abs(slope));
		await Assert.That(Math.Abs(result.Derivative(0) - slope * V0Seed)).IsLessThanOrEqualTo(FiniteDifferenceTolerance * scale);
		await Assert.That(Math.Abs(result.Derivative(1) - slope * V1Seed)).IsLessThanOrEqualTo(FiniteDifferenceTolerance * scale);
		await Assert.That(result.A).IsEqualTo(g(a));
	}

	[Test]
	[Arguments(0.25)]
	[Arguments(2.0)]
	[Arguments(9.5)]
	public async Task Sqrt_MatchesClosedFormAndFiniteDifference(double a)
	{
		Jet2 r = Jet2.Sqrt(Input(a));
		double twoAInverse = 1.0 / (2.0 * Math.Sqrt(a));
		await Assert.That(r.A).IsEqualTo(Math.Sqrt(a));
		await Assert.That(r.Derivative(0)).IsEqualTo(V0Seed * twoAInverse);
		await Assert.That(r.Derivative(1)).IsEqualTo(V1Seed * twoAInverse);
		await CheckFiniteDifference(Jet2.Sqrt, Math.Sqrt, a);
	}

	[Test]
	[Arguments(-1.3)]
	[Arguments(0.7)]
	public async Task Abs_MatchesClosedFormAndFiniteDifference(double a)
	{
		Jet2 r = Jet2.Abs(Input(a));
		double sign = Math.Sign(a);
		await Assert.That(r.A).IsEqualTo(Math.Abs(a));
		await Assert.That(r.Derivative(0)).IsEqualTo(sign * V0Seed);
		await Assert.That(r.Derivative(1)).IsEqualTo(sign * V1Seed);
		await CheckFiniteDifference(Jet2.Abs, Math.Abs, a);
	}

	// Ceres 2.1+ (pycolmap 4.2.0 links 2.2.0) uses copysign(1, a), so -0.0 takes the negative
	// branch: value +0.0, derivative -v. The older `a < 0 ? -f : f` would give +v here.
	[Test]
	public async Task Abs_NegativeZeroUsesCopysign()
	{
		Jet2 r = Jet2.Abs(Input(-0.0));
		await Assert.That(double.IsNegative(r.A)).IsFalse();
		await Assert.That(r.A).IsEqualTo(0.0);
		await Assert.That(r.Derivative(0)).IsEqualTo(-V0Seed);
		await Assert.That(r.Derivative(1)).IsEqualTo(-V1Seed);

		Jet2 positive = Jet2.Abs(Input(0.0));
		await Assert.That(positive.Derivative(0)).IsEqualTo(V0Seed);
		await Assert.That(positive.Derivative(1)).IsEqualTo(V1Seed);
	}

	[Test]
	[Arguments(-2.0)]
	[Arguments(0.3)]
	[Arguments(1.4)]
	public async Task Sin_MatchesClosedFormAndFiniteDifference(double a)
	{
		Jet2 r = Jet2.Sin(Input(a));
		await Assert.That(r.A).IsEqualTo(Math.Sin(a));
		await Assert.That(r.Derivative(0)).IsEqualTo(Math.Cos(a) * V0Seed);
		await Assert.That(r.Derivative(1)).IsEqualTo(Math.Cos(a) * V1Seed);
		await CheckFiniteDifference(Jet2.Sin, Math.Sin, a);
	}

	[Test]
	[Arguments(-2.0)]
	[Arguments(0.3)]
	[Arguments(1.4)]
	public async Task Cos_MatchesClosedFormAndFiniteDifference(double a)
	{
		Jet2 r = Jet2.Cos(Input(a));
		await Assert.That(r.A).IsEqualTo(Math.Cos(a));
		await Assert.That(r.Derivative(0)).IsEqualTo(-Math.Sin(a) * V0Seed);
		await Assert.That(r.Derivative(1)).IsEqualTo(-Math.Sin(a) * V1Seed);
		await CheckFiniteDifference(Jet2.Cos, Math.Cos, a);
	}

	[Test]
	[Arguments(-1.2)]
	[Arguments(0.3)]
	[Arguments(0.9)]
	public async Task Tan_MatchesClosedFormAndFiniteDifference(double a)
	{
		Jet2 r = Jet2.Tan(Input(a));
		double tanA = Math.Tan(a);
		double secSquared = 1.0 + tanA * tanA;
		await Assert.That(r.A).IsEqualTo(tanA);
		await Assert.That(r.Derivative(0)).IsEqualTo(secSquared * V0Seed);
		await Assert.That(r.Derivative(1)).IsEqualTo(secSquared * V1Seed);
		await CheckFiniteDifference(Jet2.Tan, Math.Tan, a);
	}

	[Test]
	[Arguments(-3.0)]
	[Arguments(0.0)]
	[Arguments(0.8)]
	public async Task Atan_MatchesClosedFormAndFiniteDifference(double a)
	{
		Jet2 r = Jet2.Atan(Input(a));
		double derivative = 1.0 / (1.0 + a * a);
		await Assert.That(r.A).IsEqualTo(Math.Atan(a));
		await Assert.That(r.Derivative(0)).IsEqualTo(derivative * V0Seed);
		await Assert.That(r.Derivative(1)).IsEqualTo(derivative * V1Seed);
		await CheckFiniteDifference(Jet2.Atan, Math.Atan, a);
	}

	// atan2(y, x) with y and x depending on different variables: y = y0 + t0, x = x0 + t1,
	// so d/dt0 = x / (x^2 + y^2) and d/dt1 = -y / (x^2 + y^2).
	[Test]
	[Arguments(0.6, 1.1)]
	[Arguments(-0.4, -2.0)]
	[Arguments(1.5, -0.3)]
	public async Task Atan2_MatchesClosedFormAndFiniteDifference(double y, double x)
	{
		Jet2 r = Jet2.Atan2(Jet2.Variable(y, 0), Jet2.Variable(x, 1));
		double inverseNormSquared = 1.0 / (x * x + y * y);
		await Assert.That(r.A).IsEqualTo(Math.Atan2(y, x));
		await Assert.That(r.Derivative(0)).IsEqualTo(inverseNormSquared * (-y * 0 + x * 1));
		await Assert.That(r.Derivative(1)).IsEqualTo(inverseNormSquared * (-y * 1 + x * 0));

		double dy = (Math.Atan2(y + H, x) - Math.Atan2(y - H, x)) / (2 * H);
		double dx = (Math.Atan2(y, x + H) - Math.Atan2(y, x - H)) / (2 * H);
		await Assert.That(Math.Abs(r.Derivative(0) - dy)).IsLessThanOrEqualTo(FiniteDifferenceTolerance);
		await Assert.That(Math.Abs(r.Derivative(1) - dx)).IsLessThanOrEqualTo(FiniteDifferenceTolerance);
	}

	// C++ `1.0 - alpha` with alpha a Jet: Ceres' operator-(T, Jet) gives (s - a, -v), with a
	// true negation (so a +0.0 derivative becomes -0.0, unlike Jet(s) - alpha's 0 - v).
	[Test]
	public async Task DoubleMinusJet_NegatesDerivative()
	{
		Jet2 r = 1.0 - Input(0.3);
		await Assert.That(r.A).IsEqualTo(1.0 - 0.3);
		await Assert.That(r.Derivative(0)).IsEqualTo(-V0Seed);
		await Assert.That(r.Derivative(1)).IsEqualTo(-V1Seed);

		Jet2 zeroSlope = 1.0 - Jet2.FromDouble(0.3);
		await Assert.That(double.IsNegative(zeroSlope.Derivative(0))).IsTrue();
		await CheckFiniteDifference((in Jet2 j) => 1.0 - j, a => 1.0 - a, 0.3);
	}

	// The Phase 7 rules (exp, log, the inverse trig functions, pow): value, closed form in
	// Ceres' grouping, and finite difference.
	[Test]
	[Arguments(-1.5)]
	[Arguments(0.4)]
	[Arguments(2.2)]
	public async Task Exp_MatchesClosedFormAndFiniteDifference(double a)
	{
		Jet2 r = Jet2.Exp(Input(a));
		await Assert.That(r.Derivative(0)).IsEqualTo(Math.Exp(a) * V0Seed);
		await CheckFiniteDifference(Jet2.Exp, Math.Exp, a);
	}

	[Test]
	[Arguments(0.3)]
	[Arguments(2.5)]
	public async Task Log_MatchesClosedFormAndFiniteDifference(double a)
	{
		Jet2 r = Jet2.Log(Input(a));
		await Assert.That(r.Derivative(1)).IsEqualTo(V1Seed * (1.0 / a));
		await CheckFiniteDifference(Jet2.Log, Math.Log, a);
	}

	[Test]
	[Arguments(-0.7)]
	[Arguments(0.2)]
	public async Task Asin_MatchesClosedFormAndFiniteDifference(double a)
	{
		Jet2 r = Jet2.Asin(Input(a));
		await Assert.That(r.Derivative(0)).IsEqualTo(1.0 / Math.Sqrt(1.0 - a * a) * V0Seed);
		await CheckFiniteDifference(Jet2.Asin, Math.Asin, a);
	}

	[Test]
	[Arguments(-0.7)]
	[Arguments(0.2)]
	public async Task Acos_MatchesClosedFormAndFiniteDifference(double a)
	{
		Jet2 r = Jet2.Acos(Input(a));
		await Assert.That(r.Derivative(0)).IsEqualTo(-1.0 / Math.Sqrt(1.0 - a * a) * V0Seed);
		await CheckFiniteDifference(Jet2.Acos, Math.Acos, a);
	}

	[Test]
	[Arguments(0.6, 2.5)]
	[Arguments(3.0, -1.5)]
	public async Task PowConstantExponent_MatchesClosedFormAndFiniteDifference(double a, double p)
	{
		Jet2 r = Jet2.Pow(Input(a), p);
		await Assert.That(r.Derivative(0)).IsEqualTo(p * Math.Pow(a, p - 1.0) * V0Seed);
		await CheckFiniteDifference((in Jet2 j) => Jet2.Pow(j, p), x => Math.Pow(x, p), a);
	}

	// pow(f, g) with f = f0 + t0 and g = g0 + t1: d/dt0 = g f^(g-1), d/dt1 = f^g log f.
	[Test]
	[Arguments(1.7, 0.8)]
	[Arguments(0.4, 2.3)]
	public async Task PowJetExponent_MatchesFiniteDifference(double f, double g)
	{
		Jet2 r = Jet2.Pow(Jet2.Variable(f, 0), Jet2.Variable(g, 1));
		double df = (Math.Pow(f + H, g) - Math.Pow(f - H, g)) / (2 * H);
		double dg = (Math.Pow(f, g + H) - Math.Pow(f, g - H)) / (2 * H);
		await Assert.That(r.A).IsEqualTo(Math.Pow(f, g));
		await Assert.That(Math.Abs(r.Derivative(0) - df)).IsLessThanOrEqualTo(FiniteDifferenceTolerance);
		await Assert.That(Math.Abs(r.Derivative(1) - dg)).IsLessThanOrEqualTo(FiniteDifferenceTolerance);
	}

	// Ceres' special cases: 0^g for g > 1 is a constant zero; a negative base with an
	// integral exponent differentiates the base and is NaN in the exponent's slots.
	[Test]
	public async Task PowJetExponent_SpecialCases()
	{
		Jet2 zero = Jet2.Pow(Jet2.Variable(0.0, 0), Jet2.Variable(2.0, 1));
		await Assert.That(zero.A).IsEqualTo(0.0);
		await Assert.That(zero.Derivative(0)).IsEqualTo(0.0);
		await Assert.That(zero.Derivative(1)).IsEqualTo(0.0);

		Jet2 negative = Jet2.Pow(Jet2.Variable(-2.0, 0), Jet2.Variable(3.0, 1));
		await Assert.That(negative.A).IsEqualTo(-8.0);
		await Assert.That(negative.Derivative(0)).IsEqualTo(12.0);
		await Assert.That(double.IsNaN(negative.Derivative(1))).IsTrue();
	}

	[Test]
	public async Task Hypot_MatchesClosedForm()
	{
		const double x = 0.3, y = -1.2, z = 0.7;
		Jet2 two = Jet2.Hypot(Jet2.Variable(x, 0), Jet2.Variable(y, 1));
		await Assert.That(two.A).IsEqualTo(double.Hypot(x, y));
		await Assert.That(Math.Abs(two.Derivative(0) - x / double.Hypot(x, y))).IsLessThanOrEqualTo(1e-15);
		await Assert.That(Math.Abs(two.Derivative(1) - y / double.Hypot(x, y))).IsLessThanOrEqualTo(1e-15);

		Jet2 three = Jet2.Hypot(Jet2.Variable(x, 0), Jet2.FromDouble(y), Jet2.Variable(z, 1));
		double norm = Math.Sqrt(x * x + y * y + z * z);
		await Assert.That(three.A).IsEqualTo(norm);
		await Assert.That(Math.Abs(three.Derivative(0) - x / norm)).IsLessThanOrEqualTo(1e-15);
		await Assert.That(Math.Abs(three.Derivative(1) - z / norm)).IsLessThanOrEqualTo(1e-15);
	}

	// libc++'s three-argument hypot rescales out-of-range arguments instead of overflowing.
	[Test]
	public async Task Hypot3_DoesNotOverflowOrUnderflow()
	{
		// Multiples of a power of two, so 3-4-5 is exact and only over- or underflow could break it.
		double big = Math.ScaleB(1.0, 1000);
		double tiny = Math.ScaleB(1.0, -1000);
		await Assert.That(ScalarMath.Hypot(3 * big, 4 * big, 0.0)).IsEqualTo(5 * big);
		await Assert.That(ScalarMath.Hypot(3 * tiny, 0.0, 4 * tiny)).IsEqualTo(5 * tiny);
		await Assert.That(ScalarMath.Hypot(1.0, 2.0, 2.0)).IsEqualTo(3.0);
	}

	// ceres::fmin/fmax: NaN is missing data, and a value tie averages the two Jets.
	[Test]
	public async Task MinMax_HandleNaNAndTies()
	{
		Jet2 small = Jet2.Variable(1.0, 0);
		Jet2 large = Jet2.Variable(2.0, 1);
		await Assert.That(Jet2.Min(small, large).Derivative(0)).IsEqualTo(1.0);
		await Assert.That(Jet2.Max(small, large).Derivative(1)).IsEqualTo(1.0);

		Jet2 nan = Jet2.FromDouble(double.NaN);
		await Assert.That(Jet2.Min(nan, large).A).IsEqualTo(2.0);
		await Assert.That(Jet2.Max(small, nan).A).IsEqualTo(1.0);

		Jet2 tie = Jet2.Max(Jet2.Variable(1.0, 0), Jet2.Variable(1.0, 1));
		await Assert.That(tie.A).IsEqualTo(1.0);
		await Assert.That(tie.Derivative(0)).IsEqualTo(0.5);
		await Assert.That(tie.Derivative(1)).IsEqualTo(0.5);
		await Assert.That(ScalarMath.FMax(double.NaN, 3.0)).IsEqualTo(3.0);
		await Assert.That(ScalarMath.FMin(3.0, double.NaN)).IsEqualTo(3.0);
	}

	// The mixed operators have Ceres' own rules, which are not Jet(s) op x: s / g is
	// s / g.a with slope -s / g.a^2 (not s * (1 / g.a)), f / s goes through 1 / s.
	[Test]
	public async Task MixedOperators_FollowCeresRules()
	{
		const double s = 0.7;
		Jet2 g = Input(1.3);
		Jet2 quotient = s / g;
		await Assert.That(quotient.A).IsEqualTo(s / 1.3);
		await Assert.That(quotient.Derivative(0)).IsEqualTo(V0Seed * (-s / (1.3 * 1.3)));

		Jet2 byScalar = g / s;
		double sInverse = 1.0 / s;
		await Assert.That(byScalar.A).IsEqualTo(1.3 * sInverse);
		await Assert.That(byScalar.Derivative(1)).IsEqualTo(V1Seed * sInverse);

		Jet2 product = s * g;
		await Assert.That(product.Derivative(1)).IsEqualTo(V1Seed * s);
		await Assert.That((g + s).Derivative(0)).IsEqualTo(V0Seed);
		await Assert.That((g - s).A).IsEqualTo(1.3 - s);
		await Assert.That(Jet2.Floor(Input(1.7)).A).IsEqualTo(1.0);
		await Assert.That(Jet2.Floor(Input(1.7)).Derivative(0)).IsEqualTo(0.0);
	}

	// Variable(value, k) seeds exactly slot k, for any width.
	[Test]
	public async Task Variable_SeedsOneSlot()
	{
		var jet = Jet<Grad15>.Variable(2.5, 11);
		await Assert.That(jet.A).IsEqualTo(2.5);
		for (int i = 0; i < Jet<Grad15>.Size; i++)
		{
			await Assert.That(jet.Derivative(i)).IsEqualTo(i == 11 ? 1.0 : 0.0);
		}
	}
}
