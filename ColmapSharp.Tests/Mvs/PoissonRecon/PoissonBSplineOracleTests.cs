// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonBSplineOracleTests (C#-only, not a COLMAP test): the PoissonRecon polynomial and
// B-spline port (ColmapSharp/Mvs/PoissonRecon) against TestData/oracle/poisson_bspline.json,
// which oracle/fixture_poisson_bspline.py records from oracle/poisson_bspline_harness.cc -
// the same PoissonRecon templates COLMAP vendors, built with -ffp-contract=off. Tier A: every
// number must be bit-identical (doubles compared by their bits, so -0.0 != +0.0).
//
// COLMAP has no test for these internals (poisson_meshing_test.cc only runs the whole
// pipeline; it is ported with the pipeline). Each producer below mirrors the matching
// Dump* function of the harness and emits the same case names in the same value order.

using System.Globalization;
using System.Text.Json;

using ColmapSharp.Mvs.PoissonRecon;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs.PoissonRecon;

public class PoissonBSplineOracleTests
{
	private const string Fixture = "poisson_bspline.json";

	[Test]
	public async Task Polynomials_MatchHarnessBitForBit() => await CompareGroup(AddPolynomialCases);

	[Test]
	public async Task SupportAndOverlapSizes_MatchHarness() => await CompareGroup(AddSizeCases);

	[Test]
	public async Task BSplineElements_MatchHarness() => await CompareGroup(AddElementCases);

	[Test]
	public async Task EvaluationData_MatchesHarnessBitForBit() => await CompareGroup(AddEvaluationCases);

	[Test]
	public async Task IntegrationData_MatchesHarnessBitForBit() => await CompareGroup(AddIntegrationCases);

	[Test]
	public async Task SparseEvaluator_MatchesHarnessBitForBit() => await CompareGroup(AddSparseCases);

	[Test]
	public async Task OutOfRangeDegreesAndDerivatives_Throw()
	{
		await Assert.That(() => BSplineSupportSizes.For(3)).Throws<ArgumentOutOfRangeException>();
		await Assert.That(() => BSplineOverlapSizes.For(1, 3)).Throws<ArgumentOutOfRangeException>();
		await Assert.That(() => BSplineEvaluationData.For(FemSignature.Of(3, BoundaryType.Free))).Throws<ArgumentOutOfRangeException>();
		var integrator = new BSplineIntegrator(5, 7, 1, 0, 2, child: false);
		await Assert.That(() => integrator.Dot(1, 1, 1, 1)).Throws<ArgumentOutOfRangeException>();
		var evaluator = new BSplineTabulatedEvaluator(5, 0, BSplineSamplePoints.Center, 2);
		await Assert.That(() => evaluator.Value(1, 1, 1)).Throws<ArgumentOutOfRangeException>();
	}

	[Test]
	public async Task StdMinMax_FollowTheCppSemantics()
	{
		using (Assert.Multiple())
		{
			// The first argument wins on NaN and on signed-zero ties, as with std::min/std::max.
			await Assert.That(double.IsNaN(StdMinMax.StdMax(double.NaN, 1.0))).IsTrue();
			await Assert.That(StdMinMax.StdMax(1.0, double.NaN)).IsEqualTo(1.0);
			await Assert.That(double.IsNegative(StdMinMax.StdMax(-0.0, 0.0))).IsTrue();
			await Assert.That(double.IsNegative(StdMinMax.StdMin(0.0, -0.0))).IsFalse();
			await Assert.That(StdMinMax.StdMin(3, 2)).IsEqualTo(2);
		}
	}

	[Test]
	public async Task EveryFixtureCase_HasAProducer()
	{
		var all = new Cases();
		AddPolynomialCases(all);
		AddSizeCases(all);
		AddElementCases(all);
		AddEvaluationCases(all);
		AddIntegrationCases(all);
		AddSparseCases(all);
		var fixtureNames = OracleFixture.Load(Fixture).GetProperty("cases").EnumerateObject().Select(p => p.Name).ToHashSet();
		await Assert.That(string.Join(" ", fixtureNames.Except(all.Keys))).IsEqualTo(string.Empty);
		await Assert.That(string.Join(" ", all.Keys.Except(fixtureNames))).IsEqualTo(string.Empty);
	}

	private static async Task CompareGroup(Action<Cases> producer)
	{
		var cases = new Cases();
		producer(cases);
		JsonElement expected = OracleFixture.Load(Fixture).GetProperty("cases");
		var mismatches = new List<string>();
		foreach (var (name, actual) in cases)
		{
			if (!expected.TryGetProperty(name, out JsonElement values))
			{
				mismatches.Add($"{name}: not in the fixture");
				continue;
			}

			string? mismatch = FirstMismatch(values, actual);
			if (mismatch != null)
			{
				mismatches.Add($"{name}: {mismatch}");
			}
		}

		await Assert.That(string.Join("\n", mismatches.Take(20))).IsEqualTo(string.Empty);
		await Assert.That(cases.Count).IsGreaterThan(0);
	}

	private static string? FirstMismatch(JsonElement expected, List<object> actual)
	{
		int count = expected.GetArrayLength();
		if (count != actual.Count)
		{
			return $"expected {count} values, got {actual.Count}";
		}

		int i = 0;
		foreach (JsonElement e in expected.EnumerateArray())
		{
			object a = actual[i];

			// The fixture writes harness "f" values as JSON floats (Python repr always has a '.'
			// or an exponent) and "i" values as bare integers, so the kind must agree too.
			string raw = e.GetRawText();
			bool expectedFloat = raw.Contains('.') || raw.Contains('e') || raw.Contains('E');
			if (expectedFloat != a is double)
			{
				return $"value {i}: expected a{(expectedFloat ? " float" : "n integer")} ({raw}), got a{(a is double ? " float" : "n integer")} ({a})";
			}

			if (a is double d)
			{
				double x = e.GetDouble();
				if (BitConverter.DoubleToInt64Bits(x) != BitConverter.DoubleToInt64Bits(d))
				{
					return $"value {i}: expected {x.ToString("R", CultureInfo.InvariantCulture)}, got {d.ToString("R", CultureInfo.InvariantCulture)}";
				}
			}
			else if (e.GetInt64() != (long)a)
			{
				return $"value {i}: expected {e.GetInt64()}, got {a}";
			}

			i++;
		}

		return null;
	}

	// ---- Producers, one per Dump* function of oracle/poisson_bspline_harness.cc ----

	private static void AddPolynomialCases(Cases cases)
	{
		for (int degree = 0; degree <= 3; degree++)
		{
			string deg = $"deg{degree}";
			for (int i = 0; i <= degree; i++)
			{
				cases.F($"poly/component/{deg}/i{i}", PoissonPolynomial.BSplineComponent(degree, i).Coefficients);
			}

			var values = new List<double>();
			var v = new double[degree + 1];
			foreach (double x in new[] { 0.0, 0.1, 0.25, 0.3137, 0.5, 0.7, 0.9, 1.0 })
			{
				PoissonPolynomial.BSplineComponentValues(degree, x, v);
				values.AddRange(v);
			}

			cases.F($"poly/componentvalues/{deg}", values);
			var b = new int[degree + 1];
			PoissonPolynomial.BinomialCoefficients(degree, b);
			cases.I($"poly/binomial/{deg}", b.Select(x => (long)x));
		}

		PoissonPolynomial p = PoissonPolynomial.BSplineComponent(2, 1);
		cases.F("poly/ops/shift", p.Shift(0.3).Coefficients);
		cases.F("poly/ops/scale", p.Scale(0.7).Coefficients);
		cases.F("poly/ops/scaleshift", p.Scale(1.0 / 3).Shift(5.0 / 3).Coefficients);
		cases.F("poly/ops/eval", [p.Evaluate(-0.4), p.Evaluate(0.0), p.Evaluate(0.3137), p.Evaluate(1.0), p.Evaluate(2.5)]);
		cases.F("poly/ops/integral", [p.Integral(-0.2, 1.3), p.Integral(0.0, 1.0), p.Integral(0.7, 0.1)]);
		cases.F("poly/ops/product", p.Multiply(PoissonPolynomial.BSplineComponent(1, 0)).Coefficients);
		cases.F("poly/ops/antiderivative", p.Antiderivative().Coefficients);
		cases.F("poly/ops/derivative", p.Derivative().Coefficients);
		cases.F("poly/ops/muldiv", p.Multiply(3.0).Divide(7).Coefficients);
	}

	private static void AddSizeCases(Cases cases)
	{
		for (int degree = 0; degree <= FemSignature.MaxDegree; degree++)
		{
			BSplineSupportSizes s = BSplineSupportSizes.For(degree);
			cases.I($"sizes/deg{degree}",
			[
				s.Inset, s.SupportStart, s.SupportEnd, s.ChildSupportStart, s.ChildSupportEnd,
				s.CornerStart, s.CornerEnd, s.ChildCornerStart, s.ChildCornerEnd, s.BCornerStart,
				s.BCornerEnd, s.ChildBCornerStart, s.ChildBCornerEnd, s.UpSampleStart,
				s.UpSampleEnd, s.DownSample0Start, s.DownSample0End, s.DownSample1Start,
				s.DownSample1End, s.Nodes(0), s.Nodes(3),
			]);
		}

		foreach (var (d1, d2) in new[] { (0, 0), (0, 1), (1, 2), (2, 1), (2, 2), (1, 1) })
		{
			BSplineOverlapSizes o = BSplineOverlapSizes.For(d1, d2);
			cases.I($"overlap/deg{d1}x{d2}",
			[
				o.OverlapStart, o.OverlapEnd, o.ChildOverlapStart, o.ChildOverlapEnd,
				o.OverlapSupportStart, o.OverlapSupportEnd, o.ChildOverlapSupportStart,
				o.ChildOverlapSupportEnd, o.ParentOverlap0Start, o.ParentOverlap0End,
				o.ParentOverlap1Start, o.ParentOverlap1End,
			]);
		}
	}

	private static List<long> Flatten(BSplineElements e)
	{
		var values = new List<long>();
		for (int i = 0; i < e.Count; i++)
		{
			values.AddRange(e[i].Select(x => (long)x));
		}

		values.Add(e.Denominator);
		return values;
	}

	private static void AddElementCases(Cases cases)
	{
		for (int degree = 0; degree <= 2; degree++)
		{
			for (int b = 0; b < 3; b++)
			{
				foreach (int res in new[] { 1, 2, 4, 5 })
				{
					for (int off = -2; off <= res + 2; off++)
					{
						string name = $"elements/deg{degree}/b{b}/res{res}/off{off}";
						var e = new BSplineElements(degree, res, off, (BoundaryType)b);
						cases.I(name, Flatten(e));
						BSplineElements up = e.UpSample();
						cases.I(name + "/up2", Flatten(up.UpSample()));
						if (degree >= 1)
						{
							cases.I(name + "/up/d1", Flatten(up.Differentiate(1)));
						}
					}
				}
			}
		}
	}

	private static void AddEvaluationCases(Cases cases)
	{
		for (int sig = 0; sig <= 8; sig++)
		{
			BSplineEvaluationData e = BSplineEvaluationData.For(sig);
			for (int depth = 0; depth <= 3; depth++)
			{
				string key = $"sig{sig}/depth{depth}";
				int res = 1 << depth;
				var values = new List<double>();
				var integrals = new List<double>();
				for (int off = e.BeginAt(depth) - 1; off <= e.EndAt(depth); off++)
				{
					for (int d = 0; d <= e.Degree + 1; d++)
					{
						for (int k = -1; k <= 4 * res + 1; k++)
						{
							values.Add(e.Value(depth, off, k / (4.0 * res), d));
						}

						values.Add(e.Value(depth, off, 0.3137, d));
						values.Add(e.Value(depth, off, 0.8123, d));
						integrals.Add(e.Integral(depth, off, -0.5, 2.0, d));
						integrals.Add(e.Integral(depth, off, 0.1, 0.6, d));
						integrals.Add(e.Integral(depth, off, 0.3137, 0.8123, d));
						integrals.Add(e.Integral(depth, off, 0.6, 0.1, d));
					}
				}

				cases.F("value/" + key, values);
				cases.F("integral/" + key, integrals);

				var up = new BSplineUpSampleEvaluator(sig, depth);
				var ups = new List<double>();
				for (int p = e.BeginAt(depth) - 1; p <= e.EndAt(depth); p++)
				{
					for (int c = e.BeginAt(depth + 1) - 1; c <= e.EndAt(depth + 1); c++)
					{
						ups.Add(up.Value(p, c));
					}
				}

				cases.F("upsample/" + key, ups);

				var center = new BSplineTabulatedEvaluator(sig, e.Degree, BSplineSamplePoints.Center, depth);
				var childCenter = new BSplineTabulatedEvaluator(sig, e.Degree, BSplineSamplePoints.ChildCenter, depth);
				var corner = new BSplineTabulatedEvaluator(sig, e.Degree, BSplineSamplePoints.Corner, depth);
				var childCorner = new BSplineTabulatedEvaluator(sig, e.Degree, BSplineSamplePoints.ChildCorner, depth);
				var c0 = new List<double>();
				var c1 = new List<double>();
				var c2 = new List<double>();
				var c3 = new List<double>();
				for (int f = e.BeginAt(depth) - 1; f <= e.EndAt(depth); f++)
				{
					for (int d = 0; d <= e.Degree; d++)
					{
						for (int c = -1; c <= res + 1; c++)
						{
							c0.Add(center.Value(f, c, d));
							c2.Add(corner.Value(f, c, d));
						}

						for (int c = -1; c <= 2 * res + 1; c++)
						{
							c1.Add(childCenter.Value(f, c, d));
							c3.Add(childCorner.Value(f, c, d));
						}
					}
				}

				cases.F("center/" + key, c0);
				cases.F("childcenter/" + key, c1);
				cases.F("corner/" + key, c2);
				cases.F("childcorner/" + key, c3);
			}
		}
	}

	private static void AddIntegrationCases(Cases cases)
	{
		(int, int)[] pairs = [(5, 5), (5, 7), (7, 5), (7, 7), (0, 0), (0, 5), (5, 0), (8, 8), (3, 3), (4, 4), (6, 7)];
		foreach (var (sig1, sig2) in pairs)
		{
			var data = new BSplineIntegrationData(sig1, sig2);
			BSplineEvaluationData e1 = BSplineEvaluationData.For(sig1);
			BSplineEvaluationData e2 = BSplineEvaluationData.For(sig2);
			int deg1 = data.Degree1;
			int deg2 = data.Degree2;
			string pair = $"sig{sig1}x{sig2}";
			for (int depth = 0; depth <= 3; depth++)
			{
				var same = new BSplineIntegrator(sig1, sig2, deg1, deg2, depth, child: false);
				var child = new BSplineIntegrator(sig1, sig2, deg1, deg2, depth, child: true);
				var s = new List<double>();
				var c = new List<double>();
				for (int o1 = e1.BeginAt(depth) - 1; o1 <= e1.EndAt(depth); o1++)
				{
					for (int d1 = 0; d1 <= deg1; d1++)
					{
						for (int d2 = 0; d2 <= deg2; d2++)
						{
							for (int o2 = e2.BeginAt(depth) - 1; o2 <= e2.EndAt(depth); o2++)
							{
								s.Add(same.Dot(o1, o2, d1, d2));
							}

							for (int o2 = e2.BeginAt(depth + 1) - 1; o2 <= e2.EndAt(depth + 1); o2++)
							{
								c.Add(child.Dot(o1, o2, d1, d2));
							}
						}
					}
				}

				cases.F($"integrator/{pair}/depth{depth}", s);
				cases.F($"childintegrator/{pair}/depth{depth}", c);
			}

			var far = new List<double>();
			for (int o1 = e1.BeginAt(1); o1 < e1.EndAt(1); o1++)
			{
				for (int o2 = e2.BeginAt(3); o2 < e2.EndAt(3); o2++)
				{
					far.Add(data.Dot(0, 0, 1, o1, 3, o2));
					far.Add(data.Dot(deg1, deg2, 1, o1, 3, o2));
				}
			}

			for (int o1 = e1.BeginAt(3); o1 < e1.EndAt(3); o1++)
			{
				for (int o2 = e2.BeginAt(1); o2 < e2.EndAt(1); o2++)
				{
					far.Add(data.Dot(deg1, 0, 3, o1, 1, o2));
				}
			}

			cases.F($"dotfar/{pair}", far);
		}
	}

	private static void AddSparseCases(Cases cases)
	{
		foreach (var (sig, derivatives) in new[] { (5, 1), (7, 1), (0, 0), (8, 2), (4, 1) })
		{
			var data = new BSplineData(sig, derivatives, 4);
			BSplineEvaluationData e = BSplineEvaluationData.For(sig);
			int left = -e.Sizes.SupportStart;
			var values = new List<double>();
			for (int depth = 0; depth <= 4; depth++)
			{
				int res = 1 << depth;
				foreach (double p in new[] { 0.0, 0.0625, 0.3137, 0.5, 0.8123, 0.99 })
				{
					int pIdx = (int)(p * res);
					for (int f = e.BeginAt(depth); f < e.EndAt(depth); f++)
					{
						int column = pIdx - f + left;
						if (column < 0 || column > e.Degree)
						{
							continue;
						}

						for (int d = 0; d <= derivatives; d++)
						{
							values.Add(data[depth].Value(p, f, d));
						}
					}
				}
			}

			cases.F($"sparse/sig{sig}/d{derivatives}", values);
		}
	}

	// Case name -> values (boxed doubles or longs), in production order.
	private sealed class Cases : Dictionary<string, List<object>>
	{
		public void F(string name, IEnumerable<double> values) => Add(name, values.Select(v => (object)v).ToList());

		public void I(string name, IEnumerable<long> values) => Add(name, values.Select(v => (object)v).ToList());
	}
}
