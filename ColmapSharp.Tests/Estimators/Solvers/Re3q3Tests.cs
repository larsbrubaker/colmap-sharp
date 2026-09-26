// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Re3q3Tests: C#-only tests (PoseLib has no re3q3 unit test and COLMAP none either) for
// ColmapSharp/Estimators/Solvers/PoseLib/Re3q3.cs. They pin the random change-of-variables
// branch, which the ported absolute_pose_test.cc cases never reach
// (docs/CPP_DIVERGENCES.md entry 26). Tier B: solutions are checked against the system.

using ColmapSharp.Estimators.Solvers.PoseLib;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class Re3q3Tests
{
	// Coefficient order: x^2, xy, xz, y^2, yz, z^2, x, y, z, 1. Column-major 3 x 10.
	private static double Evaluate(ReadOnlySpan<double> coeffs, int k, double x, double y, double z)
	{
		ReadOnlySpan<double> monomials = [x * x, x * y, x * z, y * y, y * z, z * z, x, y, z, 1.0];
		double sum = 0.0;
		for (int j = 0; j < 10; ++j)
		{
			sum += coeffs[k + 3 * j] * monomials[j];
		}

		return sum;
	}

	/// <summary>
	/// C#-only: x^2 = 1, y^2 = 4, z^2 = 9. The only quadratic monomials are x^2, y^2 and
	/// z^2, so all three elimination matrices (y^2 z^2 yz / x^2 z^2 xz / y^2 x^2 xy) have a
	/// zero column and determinant 0, which sends re3q3 through the random change of
	/// variables. Every returned solution must satisfy the original system, and all eight real
	/// solutions (+-1, +-2, +-3) must be found.
	/// </summary>
	[Test]
	public async Task CSharpOnly_DegenerateEliminationUsesRandomVarChange()
	{
		double[] coeffs = new double[30];
		coeffs[0 + 3 * 0] = 1.0;  // x^2
		coeffs[0 + 3 * 9] = -1.0; // -1
		coeffs[1 + 3 * 3] = 1.0;  // y^2
		coeffs[1 + 3 * 9] = -4.0; // -4
		coeffs[2 + 3 * 5] = 1.0;  // z^2
		coeffs[2 + 3 * 9] = -9.0; // -9

		double[] solutions = new double[24];
		int n = Re3q3.Solve(coeffs, solutions);

		var found = new List<(double X, double Y, double Z)>();
		var failures = new List<string>();
		for (int i = 0; i < n; ++i)
		{
			double x = solutions[3 * i], y = solutions[3 * i + 1], z = solutions[3 * i + 2];
			found.Add((x, y, z));
			for (int k = 0; k < 3; ++k)
			{
				double r = Evaluate(coeffs, k, x, y, z);
				if (!(Math.Abs(r) < 1e-8))
				{
					failures.Add($"solution {i} ({x}, {y}, {z}): equation {k} residual {r}");
				}
			}
		}

		foreach (double ex in new[] { 1.0, -1.0 })
		{
			foreach (double ey in new[] { 2.0, -2.0 })
			{
				foreach (double ez in new[] { 3.0, -3.0 })
				{
					if (!found.Any(s => Math.Abs(s.X - ex) < 1e-8 && Math.Abs(s.Y - ey) < 1e-8 && Math.Abs(s.Z - ez) < 1e-8))
					{
						failures.Add($"missing solution ({ex}, {ey}, {ez}); found {n}");
					}
				}
			}
		}

		await Assert.That(failures).IsEmpty();
	}
}
