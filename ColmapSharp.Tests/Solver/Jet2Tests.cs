// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Jet2Tests (C#-only; COLMAP has no test of ceres::Jet, and Ceres' jet_test.cc is not part of
// this port): the elementary-function rules of ColmapSharp/Solver/Jet2.cs that the camera
// model oracle does not reach, because IterativeUndistortion only runs +, -, * and / on
// Jets. Each rule is checked two ways:
// - against its closed-form derivative, written out here, bit for bit where the rule's
//   formula is the closed form's own grouping (Ceres 2.2.0's), and
// - against a central finite difference of the value part, (f(a + h) - f(a - h)) / 2h,
//   which checks the calculus independently of how the formula is grouped.
// Both derivative slots are checked, with a seed that differs per slot, so a rule that
// mixes up v0 and v1 fails.

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class Jet2Tests
{
	// Finite-difference step and tolerance: central differences have O(h^2) truncation and
	// O(eps / h) rounding error, so h = 1e-5 leaves about 1e-10 relative; 1e-8 is loose
	// enough for every function and point below and tight enough to catch a wrong rule.
	private const double H = 1e-5;
	private const double FiniteDifferenceTolerance = 1e-8;

	// Seeds for the two derivative slots: d/dt of the input a + t*(V0Seed, V1Seed).
	private const double V0Seed = 1.0;
	private const double V1Seed = -0.5;

	private static Jet2 Input(double a) => new(a, V0Seed, V1Seed);

	private static async Task CheckFiniteDifference(Func<Jet2, Jet2> f, Func<double, double> g, double a)
	{
		Jet2 result = f(Input(a));
		double slope = (g(a + H) - g(a - H)) / (2 * H);
		double scale = Math.Max(1.0, Math.Abs(slope));
		await Assert.That(Math.Abs(result.V0 - slope * V0Seed)).IsLessThanOrEqualTo(FiniteDifferenceTolerance * scale);
		await Assert.That(Math.Abs(result.V1 - slope * V1Seed)).IsLessThanOrEqualTo(FiniteDifferenceTolerance * scale);
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
		await Assert.That(r.V0).IsEqualTo(V0Seed * twoAInverse);
		await Assert.That(r.V1).IsEqualTo(V1Seed * twoAInverse);
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
		await Assert.That(r.V0).IsEqualTo(sign * V0Seed);
		await Assert.That(r.V1).IsEqualTo(sign * V1Seed);
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
		await Assert.That(r.V0).IsEqualTo(-V0Seed);
		await Assert.That(r.V1).IsEqualTo(-V1Seed);

		Jet2 positive = Jet2.Abs(Input(0.0));
		await Assert.That(positive.V0).IsEqualTo(V0Seed);
		await Assert.That(positive.V1).IsEqualTo(V1Seed);
	}

	[Test]
	[Arguments(-2.0)]
	[Arguments(0.3)]
	[Arguments(1.4)]
	public async Task Sin_MatchesClosedFormAndFiniteDifference(double a)
	{
		Jet2 r = Jet2.Sin(Input(a));
		await Assert.That(r.A).IsEqualTo(Math.Sin(a));
		await Assert.That(r.V0).IsEqualTo(Math.Cos(a) * V0Seed);
		await Assert.That(r.V1).IsEqualTo(Math.Cos(a) * V1Seed);
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
		await Assert.That(r.V0).IsEqualTo(-Math.Sin(a) * V0Seed);
		await Assert.That(r.V1).IsEqualTo(-Math.Sin(a) * V1Seed);
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
		await Assert.That(r.V0).IsEqualTo(secSquared * V0Seed);
		await Assert.That(r.V1).IsEqualTo(secSquared * V1Seed);
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
		await Assert.That(r.V0).IsEqualTo(derivative * V0Seed);
		await Assert.That(r.V1).IsEqualTo(derivative * V1Seed);
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
		await Assert.That(r.V0).IsEqualTo(inverseNormSquared * (-y * 0 + x * 1));
		await Assert.That(r.V1).IsEqualTo(inverseNormSquared * (-y * 1 + x * 0));

		double dy = (Math.Atan2(y + H, x) - Math.Atan2(y - H, x)) / (2 * H);
		double dx = (Math.Atan2(y, x + H) - Math.Atan2(y, x - H)) / (2 * H);
		await Assert.That(Math.Abs(r.V0 - dy)).IsLessThanOrEqualTo(FiniteDifferenceTolerance);
		await Assert.That(Math.Abs(r.V1 - dx)).IsLessThanOrEqualTo(FiniteDifferenceTolerance);
	}

	// C++ `1.0 - alpha` with alpha a Jet: Ceres' operator-(T, Jet) gives (s - a, -v), with a
	// true negation (so a +0.0 derivative becomes -0.0, unlike Jet(s) - alpha's 0 - v).
	[Test]
	public async Task DoubleMinusJet_NegatesDerivative()
	{
		Jet2 r = 1.0 - Input(0.3);
		await Assert.That(r.A).IsEqualTo(1.0 - 0.3);
		await Assert.That(r.V0).IsEqualTo(-V0Seed);
		await Assert.That(r.V1).IsEqualTo(-V1Seed);

		Jet2 zeroSlope = 1.0 - Jet2.FromDouble(0.3);
		await Assert.That(double.IsNegative(zeroSlope.V0)).IsTrue();
		await CheckFiniteDifference(j => 1.0 - j, a => 1.0 - a, 0.3);
	}
}
