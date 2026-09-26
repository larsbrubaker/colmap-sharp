// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BundleAdjustmentProblemTests (C#-only; COLMAP's bundle adjustment tests arrive with
// Phase 8): a small synthetic bundle adjustment through Problem, the evaluator, every linear
// solver and the Levenberg-Marquardt loop (ColmapSharp/Solver/*). 20 points seen by three
// SIMPLE_RADIAL views (TestReprojectionFunctor from AutoDiffCostFunctionTests), blocks laid
// out as COLMAP's bundle adjuster does: point (3), rotation (4, Eigen quaternion manifold),
// translation (3), shared camera (4, principal point held by a SubsetManifold). Views 0 and
// 1 are held constant, which fixes the gauge, so the optimum is unique and is the ground
// truth the exact observations were made from.
// - The evaluator's gradient (robust loss, manifolds) agrees with central finite
//   differences of its cost in the tangent space, and the dense and block-sparse Jacobians
//   hold the same bits.
// - Every linear solver, with and without a robust loss, recovers the ground truth, and on
//   noisy observations every solver (the Schur ones included) reaches DENSE_QR's optimum.
// - Results are bit-identical for 1 and 4 evaluation threads.
// - Cancellation and callbacks stop the solve with Ceres' termination types and leave the
//   parameters where Ceres would.

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class BundleAdjustmentProblemTests
{
	private const int NumPoints = 20;
	private const int NumViews = 3;

	// Parameter arrays of one synthetic scene, ground truth or perturbed.
	private sealed class Scene
	{
		public double[][] Points { get; } = new double[NumPoints][];

		public double[][] Rotations { get; } = new double[NumViews][];

		public double[][] Translations { get; } = new double[NumViews][];

		public double[] Camera { get; set; } = [];

		public double[][] Observations { get; } = new double[NumViews * NumPoints][];

		public IEnumerable<double[]> AllBlocks() =>
			Points.Concat(Rotations).Concat(Translations).Append(Camera);
	}

	private static double[] AxisAngleQuaternion(double ax, double ay, double az, double angle)
	{
		double norm = Math.Sqrt((ax * ax) + (ay * ay) + (az * az));
		double s = Math.Sin(angle / 2) / norm;
		return [ax * s, ay * s, az * s, Math.Cos(angle / 2)];
	}

	private static Scene GroundTruth()
	{
		var scene = new Scene();
		for (int i = 0; i < NumPoints; i++)
		{
			scene.Points[i] = [-1.0 + (0.5 * (i % 5)), -0.75 + (0.5 * (i / 5)), 5.0 + (0.3 * Math.Sin(i))];
		}

		scene.Rotations[0] = [0.0, 0.0, 0.0, 1.0];
		scene.Translations[0] = [0.0, 0.0, 0.0];
		scene.Rotations[1] = AxisAngleQuaternion(0.0, 1.0, 0.0, 0.1);
		scene.Translations[1] = [-0.5, 0.0, 0.05];
		scene.Rotations[2] = AxisAngleQuaternion(0.2, 1.0, 0.1, -0.12);
		scene.Translations[2] = [0.6, 0.1, -0.1];
		scene.Camera = [800.0, 320.0, 240.0, -0.05];

		// Exact observations: the projection of the ground truth (the residual against 0, 0).
		for (int v = 0; v < NumViews; v++)
		{
			for (int i = 0; i < NumPoints; i++)
			{
				double[] projection = new double[2];
				CostFunction projector = Reprojection(0.0, 0.0);
				projector.Evaluate(
					[scene.Points[i], scene.Rotations[v], scene.Translations[v], scene.Camera], projection, []);
				scene.Observations[(v * NumPoints) + i] = projection;
			}
		}

		return scene;
	}

	private static Scene Perturbed()
	{
		Scene scene = GroundTruth();
		for (int i = 0; i < NumPoints; i++)
		{
			scene.Points[i][0] += 0.03 * Math.Cos(i);
			scene.Points[i][1] -= 0.02 * Math.Sin(2 * i);
			scene.Points[i][2] += 0.05 * Math.Cos(3 * i);
		}

		scene.Rotations[2] = AxisAngleQuaternion(0.25, 1.0, 0.05, -0.1);
		scene.Translations[2][0] += 0.02;
		scene.Translations[2][1] -= 0.03;
		scene.Translations[2][2] += 0.01;
		scene.Camera[0] += 8.0;
		scene.Camera[3] += 0.01;
		return scene;
	}

	private static AutoDiffCostFunction<TestReprojectionFunctor, Grad14> Reprojection(double x, double y) =>
		new(new TestReprojectionFunctor(x, y), 2, 3, 4, 3, 4);

	private static Problem BuildProblem(Scene scene, LossFunction? loss)
	{
		var problem = new Problem();
		for (int v = 0; v < NumViews; v++)
		{
			for (int i = 0; i < NumPoints; i++)
			{
				double[] observation = scene.Observations[(v * NumPoints) + i];
				problem.AddResidualBlock(
					Reprojection(observation[0], observation[1]),
					loss,
					scene.Points[i],
					scene.Rotations[v],
					scene.Translations[v],
					scene.Camera);
			}

			problem.SetManifold(scene.Rotations[v], QuaternionManifold.EigenOrder());
		}

		problem.SetManifold(scene.Camera, new SubsetManifold(4, [1, 2]));
		for (int v = 0; v < 2; v++)
		{
			problem.SetParameterBlockConstant(scene.Rotations[v]);
			problem.SetParameterBlockConstant(scene.Translations[v]);
		}

		return problem;
	}

	public static IEnumerable<(LinearSolverType, string)> SolverAndLoss()
	{
		foreach (LinearSolverType type in CeresExampleTests.AllLinearSolvers())
		{
			yield return (type, "trivial");
			yield return (type, "cauchy");
			yield return (type, "huber");
		}
	}

	private static LossFunction? MakeLoss(string name) => name switch
	{
		"cauchy" => new CauchyLoss(1.0),
		"huber" => new HuberLoss(2.0),
		_ => null,
	};

	[Test]
	[MethodDataSource(nameof(SolverAndLoss))]
	public async Task RecoversGroundTruth(LinearSolverType linearSolver, string loss)
	{
		Scene truth = GroundTruth();
		Scene scene = Perturbed();
		Problem problem = BuildProblem(scene, MakeLoss(loss));
		var options = new SolverOptions { LinearSolverType = linearSolver, MaxNumIterations = 100 };
		SolverSummary summary = LeastSquaresSolver.Solve(options, problem);

		await Assert.That(summary.IsSolutionUsable).IsTrue();
		await Assert.That(summary.InitialCost).IsGreaterThan(1.0);
		await Assert.That(summary.FinalCost).IsLessThan(1e-12);
		await Assert.That(summary.NumResidualsReduced).IsEqualTo(2 * NumViews * NumPoints);

		// 20 points, one rotation (3), one translation, focal and k.
		await Assert.That(summary.NumEffectiveParametersReduced).IsEqualTo((3 * NumPoints) + 3 + 3 + 2);
		for (int i = 0; i < NumPoints; i++)
		{
			for (int k = 0; k < 3; k++)
			{
				await Assert.That(scene.Points[i][k]).IsEqualTo(truth.Points[i][k]).Within(1e-6);
			}
		}

		for (int k = 0; k < 4; k++)
		{
			await Assert.That(scene.Rotations[2][k]).IsEqualTo(truth.Rotations[2][k]).Within(1e-8);
			await Assert.That(scene.Camera[k]).IsEqualTo(truth.Camera[k]).Within(1e-5);
		}

		for (int k = 0; k < 3; k++)
		{
			await Assert.That(scene.Translations[2][k]).IsEqualTo(truth.Translations[2][k]).Within(1e-7);
		}

		// The constant blocks and the held principal point are untouched.
		await Assert.That(scene.Rotations[1].SequenceEqual(truth.Rotations[1])).IsTrue();
		await Assert.That(scene.Camera[1]).IsEqualTo(320.0);
		await Assert.That(scene.Camera[2]).IsEqualTo(240.0);
	}

	// Observations off by a deterministic pseudo-noise of about a pixel, so the optimum has
	// a non-zero cost and is not the ground truth.
	private static Scene NoisyPerturbed()
	{
		Scene scene = Perturbed();
		for (int k = 0; k < scene.Observations.Length; k++)
		{
			scene.Observations[k][0] += 0.8 * Math.Sin(1.7 * k);
			scene.Observations[k][1] += 0.8 * Math.Cos(2.3 * k);
		}

		return scene;
	}

	[Test]
	[MethodDataSource(typeof(CeresExampleTests), nameof(CeresExampleTests.AllLinearSolvers))]
	public async Task EverySolver_ReachesTheSameOptimum(LinearSolverType linearSolver)
	{
		var options = new SolverOptions
		{
			MaxNumIterations = 100,
			FunctionTolerance = 1e-12,
			GradientTolerance = 1e-12,
			ParameterTolerance = 1e-12,
		};
		Scene reference = NoisyPerturbed();
		options.LinearSolverType = LinearSolverType.DenseQr;
		SolverSummary expected = LeastSquaresSolver.Solve(options, BuildProblem(reference, new CauchyLoss(1.0)));
		Scene scene = NoisyPerturbed();
		options.LinearSolverType = linearSolver;
		SolverSummary summary = LeastSquaresSolver.Solve(options, BuildProblem(scene, new CauchyLoss(1.0)));

		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.Convergence);
		await Assert.That(expected.FinalCost).IsGreaterThan(1.0);
		await Assert.That(summary.FinalCost).IsEqualTo(expected.FinalCost).Within(1e-9 * expected.FinalCost);
		foreach ((double[] p, double[] q) in scene.AllBlocks().Zip(reference.AllBlocks()))
		{
			for (int k = 0; k < p.Length; k++)
			{
				await Assert.That(p[k]).IsEqualTo(q[k]).Within(1e-6 * Math.Max(1.0, Math.Abs(q[k])));
			}
		}
	}

	[Test]
	public async Task Gradient_MatchesFiniteDifferencesOfTheCost()
	{
		Problem problem = BuildProblem(Perturbed(), new CauchyLoss(2.0));
		Program reduced = problem.Program.CreateReducedProgram(out _, out _)!;
		await Assert.That(problem.Program.SetParameterBlockStatePtrsToUserStatePtrs()).IsTrue();
		var evaluator = new ProgramEvaluator(reduced, denseJacobian: false, numEliminateBlocks: 0, numThreads: 1);
		double[] x = new double[evaluator.NumParameters];
		reduced.ParameterBlocksToStateVector(x);
		double[] residuals = new double[evaluator.NumResiduals];
		double[] gradient = new double[evaluator.NumEffectiveParameters];
		SparseMatrix jacobian = evaluator.CreateJacobian();
		await Assert.That(evaluator.Evaluate(x, out _, residuals, gradient, jacobian)).IsTrue();

		double[] delta = new double[evaluator.NumEffectiveParameters];
		double[] moved = new double[x.Length];
		double maxError = 0.0;
		for (int i = 0; i < delta.Length; i++)
		{
			const double h = 1e-6;
			delta[i] = h;
			evaluator.Plus(x, delta, moved);
			evaluator.Evaluate(moved, out double forward, null, null, null);
			delta[i] = -h;
			evaluator.Plus(x, delta, moved);
			evaluator.Evaluate(moved, out double backward, null, null, null);
			delta[i] = 0.0;
			double fd = (forward - backward) / (2 * h);
			maxError = Math.Max(maxError, Math.Abs(gradient[i] - fd) / Math.Max(1.0, Math.Abs(fd)));
		}

		await Assert.That(maxError).IsLessThan(1e-5);

		// The dense evaluator writes the same Jacobian values as the block-sparse one.
		var denseEvaluator = new ProgramEvaluator(reduced, denseJacobian: true, numEliminateBlocks: 0, numThreads: 1);
		var dense = (DenseSparseMatrix)denseEvaluator.CreateJacobian();
		await Assert.That(denseEvaluator.Evaluate(x, out _, new double[residuals.Length], null, dense)).IsTrue();
		double[] unit = new double[delta.Length];
		double[] column = new double[residuals.Length];
		int mismatches = 0;
		for (int c = 0; c < unit.Length; c++)
		{
			unit[c] = 1.0;
			Array.Clear(column);
			jacobian.RightMultiplyAndAccumulate(unit, column);
			unit[c] = 0.0;
			for (int r = 0; r < column.Length; r++)
			{
				mismatches += column[r] == dense.Matrix[r, c] ? 0 : 1;
			}
		}

		await Assert.That(mismatches).IsEqualTo(0);
	}

	[Test]
	[MethodDataSource(typeof(CeresExampleTests), nameof(CeresExampleTests.AllLinearSolvers))]
	public async Task ThreadCount_DoesNotChangeTheResult(LinearSolverType linearSolver)
	{
		Scene single = Perturbed();
		Scene parallel = Perturbed();
		var options = new SolverOptions { LinearSolverType = linearSolver, NumThreads = 1 };
		SolverSummary a = LeastSquaresSolver.Solve(options, BuildProblem(single, new CauchyLoss(1.0)));
		options.NumThreads = 4;
		SolverSummary b = LeastSquaresSolver.Solve(options, BuildProblem(parallel, new CauchyLoss(1.0)));

		await Assert.That(b.FinalCost).IsEqualTo(a.FinalCost);
		await Assert.That(b.Iterations.Count).IsEqualTo(a.Iterations.Count);
		foreach ((double[] p, double[] q) in single.AllBlocks().Zip(parallel.AllBlocks()))
		{
			await Assert.That(p.SequenceEqual(q)).IsTrue();
		}
	}

	[Test]
	public async Task Cancellation_StopsWithUserSuccessAtTheBestPointSoFar()
	{
		Scene scene = Perturbed();
		double[][] before = [.. scene.AllBlocks().Select(block => (double[])block.Clone())];
		using var cts = new CancellationTokenSource();
		cts.Cancel();
		SolverSummary summary = LeastSquaresSolver.Solve(new SolverOptions(), BuildProblem(scene, null), cts.Token);

		// Checked after iteration 0, as COLMAP's CancellationCallback is.
		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.UserSuccess);
		await Assert.That(summary.IsSolutionUsable).IsTrue();
		await Assert.That(summary.Iterations.Count).IsEqualTo(1);
		foreach ((double[] p, double[] q) in scene.AllBlocks().Zip(before))
		{
			await Assert.That(p.SequenceEqual(q)).IsTrue();
		}
	}

	private sealed class AbortAfter(int iteration) : IIterationCallback
	{
		public CallbackReturnType Invoke(IterationSummary summary) =>
			summary.Iteration >= iteration ? CallbackReturnType.SolverAbort : CallbackReturnType.SolverContinue;
	}

	[Test]
	public async Task AbortingCallback_RestoresTheParameters()
	{
		Scene scene = Perturbed();
		double[][] before = [.. scene.AllBlocks().Select(block => (double[])block.Clone())];
		var options = new SolverOptions();
		options.Callbacks.Add(new AbortAfter(2));
		SolverSummary summary = LeastSquaresSolver.Solve(options, BuildProblem(scene, null));

		await Assert.That(summary.TerminationType).IsEqualTo(TerminationType.UserFailure);
		await Assert.That(summary.Message).IsEqualTo("User callback returned SOLVER_ABORT.");
		await Assert.That(summary.Iterations.Count).IsEqualTo(3);
		await Assert.That(summary.FinalCost).IsLessThan(summary.InitialCost);
		foreach ((double[] p, double[] q) in scene.AllBlocks().Zip(before))
		{
			await Assert.That(p.SequenceEqual(q)).IsTrue();
		}
	}
}
