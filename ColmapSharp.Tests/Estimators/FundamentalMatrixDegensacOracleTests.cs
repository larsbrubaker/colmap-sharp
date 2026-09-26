// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// FundamentalMatrixDegensacOracleTests (C#-only; fundamental_matrix_degensac_test.cc checks
// outcomes on generated scenes, not COLMAP's actual model): EstimateFundamentalMatrixDegensac
// (Estimators/FundamentalMatrixDegensac.cs) against COLMAP through pycolmap 4.2.0. The
// fixture is written by oracle/fundamental_matrix_degensac.py into
// TestData/oracle/fundamental_matrix_degensac.json; that script explains why the F of
// estimate_two_view_geometry with use_degensac is exactly DEGENSAC's model.
//
// Tier C with a tight bound: the RANSAC seed is fixed and the draws are Tier A, so the same
// samples, degeneracy decisions and completions happen on both sides and F agrees to the
// rounding of the solvers (Tier B), not merely to the same basin. F is compared up to scale
// and sign, after normalizing both to unit Frobenius norm.

using System.Text.Json;

using ColmapSharp.Estimators;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public class FundamentalMatrixDegensacOracleTests
{
	private const string Fixture = "fundamental_matrix_degensac.json";

	// Without Sampson refinement F comes straight from the 7/8-point solvers or the
	// plane-and-parallax completion: largest gap observed 7.2e-16 (plane95).
	private const double SampledBound = 1e-12;

	// The Sampson refinement is iterative (TinySolver vs COLMAP's), so its stopping point
	// differs in the last digits: largest gap observed 8.2e-9 (plane98).
	private const double RefinedBound = 1e-7;

	[Test]
	public async Task EstimateFundamentalMatrixDegensac_MatchesPycolmap()
	{
		var gaps = new List<(string Name, double Gap, double Bound)>();
		foreach (JsonElement c in OracleFixture.Load(Fixture).GetProperty("cases").EnumerateArray())
		{
			Vector2d[] points1 = Vec2s(OracleFixture.Doubles(c.GetProperty("points1")));
			Vector2d[] points2 = Vec2s(OracleFixture.Doubles(c.GetProperty("points2")));
			Matrix3d expected = Mat3(OracleFixture.Doubles(c.GetProperty("F")));

			var options = new FundamentalMatrixDegensacOptions
			{
				Ransac = new RansacOptions
				{
					MaxError = 1.0,
					Confidence = 0.9999,
					MinInlierRatio = 0.1,
					MinNumTrials = 0,
					MaxNumTrials = 10000,
					RandomSeed = 0,
				},
				UseSampsonRefinement = c.GetProperty("use_sampson_refinement").GetBoolean(),
			};
			var report = FundamentalMatrixDegensac.EstimateFundamentalMatrixDegensac(points1, points2, options);

			Matrix3d actual = report.Model / report.Model.Norm();
			Matrix3d expectedUnit = expected / expected.Norm();
			double gap = Math.Min((actual - expectedUnit).Norm(), (actual + expectedUnit).Norm());
			double bound = options.UseSampsonRefinement ? RefinedBound : SampledBound;
			gaps.Add((c.GetProperty("name").GetString()!, report.Success ? gap : double.PositiveInfinity, bound));
		}

		using (Assert.Multiple())
		{
			await Assert.That(gaps.Count).IsEqualTo(8);
			foreach (var (name, gap, bound) in gaps)
			{
				await Assert.That(gap).IsLessThan(bound).Because($"{name}: gap {gap:R}");
			}
		}
	}

	private static Vector2d[] Vec2s(double[] values) =>
		Enumerable.Range(0, values.Length / 2).Select(i => new Vector2d(values[2 * i], values[2 * i + 1])).ToArray();

	private static Matrix3d Mat3(double[] rowMajor) => new(
		rowMajor[0], rowMajor[1], rowMajor[2],
		rowMajor[3], rowMajor[4], rowMajor[5],
		rowMajor[6], rowMajor[7], rowMajor[8]);
}
