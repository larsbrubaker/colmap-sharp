// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Gp3pTests: C#-only tests (PoseLib's gp3p has no unit test and COLMAP tests it only through
// GP3PEstimator and RANSAC) for ColmapSharp/Estimators/Solvers/PoseLib/Gp3p.cs and
// Re3q3.SolveRotation. The expected poses are the output of PoseLib fa7280f's gp3p built with
// clang++ and Eigen 3.4 in a scratch harness on the same input. PoseLib pre-rotates the
// problem by a std::rand rotation, so its solutions vary with the seed at the 1e-10 level
// (seeds 1..5 all gave these four poses); the port draws that rotation from a fixed seed
// (docs/CPP_DIVERGENCES.md entry 29). Tier B: the solution set is compared
// order-insensitively, since its order depends on the random pre-rotation.

using ColmapSharp.Estimators.Solvers.PoseLib;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.Solvers;

public class Gp3pTests
{
	/// <summary>
	/// C#-only: three rays from distinct origins; gp3p must return the same four poses as
	/// C++ PoseLib (to 1e-8), one of which is the ground truth.
	/// </summary>
	[Test]
	public async Task CSharpOnly_MatchesPoseLibSolutionSet()
	{
		Matrix3d r = new AngleAxisd(0.7, new Vector3d(0.3, -0.5, 0.8).Normalized()).ToRotationMatrix();
		var t = new Vector3d(0.2, -0.4, 1.5);
		Vector3d[] p = [new(0.1, 0.0, 0.0), new(-0.2, 0.3, 0.1), new(0.0, -0.1, 0.4)];
		Vector3d[] bigX = [new(1.0, 0.5, 4.0), new(-1.2, 0.3, 5.0), new(0.4, -0.9, 3.0)];
		var x = new Vector3d[3];
		for (int i = 0; i < 3; ++i)
		{
			x[i] = (r * bigX[i] + t - p[i]).Normalized();
		}

		var output = new List<CameraPose>();
		int n = Gp3p.Solve(p, x, bigX, output);

		// PoseLib output, seed 2 (row-major R, then t).
		(Matrix3d R, Vector3d T)[] expected =
		[
			(new Matrix3d(0.78643831294203415, -0.55660005156953363, -0.26778939958422493, 0.48461296604437926, 0.82483142522211972, -0.2912102215028109, 0.38296873642447177, 0.099244660102391982, 0.91841463640482346),
				new Vector3d(0.20000000000001467, -0.40000000000002667, 1.5000000000000004)),
			(new Matrix3d(-0.3115249563007132, 0.89677384268464011, -0.31424333991106707, -0.6225007385407948, -0.44245516960232001, -0.64554337840949683, -0.71794500633266001, -0.0054861615730680824, 0.69607820675064291),
				new Vector3d(1.5938683320549281, 3.9649710278062558, -6.5890832575233063)),
			(new Matrix3d(-0.092892833061926972, -0.17092584302288621, 0.98089514105873965, -0.97539153337595708, -0.18222090593954188, -0.12412452641227711, 0.1999556905522531, -0.96828709462921603, -0.14979260392383642),
				new Vector3d(-3.2590970512765134, 2.1199117032871451, -3.7918864601009084)),
			(new Matrix3d(0.058701789199897658, 0.20584854725259422, -0.97682161909876219, 0.29517353781895167, 0.93117681433605992, 0.21396804203990566, 0.9536386540593742, -0.30089220002675948, -0.0060992988844521534),
				new Vector3d(3.3727273062725383, -2.2987449769690533, 4.9474907494556586)),
		];

		var missing = new List<string>();
		foreach ((Matrix3d er, Vector3d et) in expected)
		{
			bool found = output.Any(pose =>
				MaxAbs(pose.R() - er) < 1e-8 && (pose.T - et).CwiseAbs() is var d && Math.Max(Math.Max(d.X, d.Y), d.Z) < 1e-8);
			if (!found)
			{
				missing.Add($"R={er} t={et}");
			}
		}

		await Assert.That(n).IsEqualTo(4);
		await Assert.That(output.Count).IsEqualTo(4);
		await Assert.That(missing).IsEmpty();
	}

	private static double MaxAbs(Matrix3d m)
	{
		double max = 0.0;
		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				max = Math.Max(max, Math.Abs(m[i, j]));
			}
		}

		return max;
	}
}
