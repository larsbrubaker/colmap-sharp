// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/solver.cc (Solver::Solve, Minimize,
// PostSolveSummarize), internal/ceres/trust_region_preprocessor.cc and
// internal/ceres/solver_utils.h (SetSummaryFinalCost) (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// ceres::Solve for the trust-region minimizer: validate the options, point every block at
// the user's memory, build the reduced program (constant blocks out, their residuals' cost
// into fixed_cost), set up the evaluator, Jacobian, linear solver and LM strategy, minimize
// (TrustRegionMinimizer.cs), then write the solution back to the user's arrays if it is
// usable (CONVERGENCE, NO_CONVERGENCE, USER_SUCCESS; otherwise the original values are
// restored) and fill in the summary.
//
// The parameter blocks keep Problem's insertion order except for the Schur solvers, which
// put the eliminated blocks first (SchurOrdering.cs); Ceres also reorders for SuiteSparse's
// own fill-reducing ordering, which the simplicial Cholesky here does itself
// (docs/CPP_DIVERGENCES.md entry 22).

namespace ColmapSharp.Solver;

/// <summary>ceres::Solve.</summary>
public static class LeastSquaresSolver
{
	/// <summary>
	/// Solves <paramref name="problem"/>, writing the solution into the parameter blocks'
	/// arrays. Cancelling <paramref name="cancellationToken"/> stops at the end of the
	/// current iteration like COLMAP's CancellationCallback: termination USER_SUCCESS, the best
	/// point so far written back, no exception.
	/// </summary>
	public static SolverSummary Solve(SolverOptions options, Problem problem, CancellationToken cancellationToken = default)
	{
		ArgumentNullException.ThrowIfNull(options);
		ArgumentNullException.ThrowIfNull(problem);
		var clock = System.Diagnostics.Stopwatch.StartNew();
		var summary = new SolverSummary();
		string? error = options.Validate();
		if (error is not null)
		{
			summary.Message = error;
			return summary;
		}

		Program program = problem.Program;
		summary.NumParameterBlocks = program.ParameterBlocks.Count;
		summary.NumParameters = program.NumParameters;
		summary.NumEffectiveParameters = program.NumEffectiveParameters;
		summary.NumResidualBlocks = program.ResidualBlocks.Count;
		summary.NumResiduals = program.NumResiduals;
		summary.LinearSolverTypeUsed = options.LinearSolverType;
		summary.NumThreadsUsed = options.NumThreads;

		try
		{
			if (!program.SetParameterBlockStatePtrsToUserStatePtrs())
			{
				summary.Message = "Manifold::PlusJacobian failed at the initial parameter values.";
				return summary;
			}

			Preprocessed? pp = Preprocess(options, program, summary, out error);
			if (pp is null)
			{
				summary.Message = error;
			}
			else
			{
				Minimize(options, pp, summary, cancellationToken);
				summary.NumResidualEvaluations = pp.Evaluator?.NumResidualEvaluations ?? 0;
				summary.NumJacobianEvaluations = pp.Evaluator?.NumJacobianEvaluations ?? 0;
				summary.NumLinearSolves = pp.LinearSolver?.NumSolves ?? 0;
				SetSummaryFinalCost(summary);
				Program reduced = pp.ReducedProgram;
				summary.NumParameterBlocksReduced = reduced.ParameterBlocks.Count;
				summary.NumParametersReduced = reduced.NumParameters;
				summary.NumEffectiveParametersReduced = reduced.NumEffectiveParameters;
				summary.NumResidualBlocksReduced = reduced.ResidualBlocks.Count;
				summary.NumResidualsReduced = reduced.NumResiduals;
			}
		}
		finally
		{
			// Restore the problem's own view of its blocks, whatever happened.
			program.SetParameterBlockStatePtrsToUserStatePtrs();
			program.SetParameterOffsetsAndIndex();
		}

		summary.TotalTimeInSeconds = clock.Elapsed.TotalSeconds;
		return summary;
	}

	private sealed class Preprocessed(Program reducedProgram, double fixedCost)
	{
		public Program ReducedProgram { get; } = reducedProgram;

		public double FixedCost { get; } = fixedCost;

		public ProgramEvaluator? Evaluator { get; set; }

		public LinearSolver? LinearSolver { get; set; }

		public SparseMatrix? Jacobian { get; set; }

		public LevenbergMarquardtStrategy? Strategy { get; set; }
	}

	// TrustRegionPreprocessor::Preprocess.
	private static Preprocessed? Preprocess(SolverOptions options, Program program, SolverSummary summary, out string error)
	{
		if (!program.ParameterBlocksAreFinite(out error))
		{
			return null;
		}

		Program? reduced = program.CreateReducedProgram(out double fixedCost, out error);
		if (reduced is null)
		{
			return null;
		}

		summary.FixedCost = fixedCost;
		var pp = new Preprocessed(reduced, fixedCost);
		if (reduced.ParameterBlocks.Count == 0)
		{
			return pp;
		}

		// SetupLinearSolver: the Schur solvers eliminate an independent set of blocks, which
		// reorders the program (ReorderProgramForSchurTypeLinearSolver).
		LinearSolverType type = options.LinearSolverType;
		int numEliminateBlocks = LinearSolver.IsSchurType(type) ? SchurOrdering.ReorderProgramForSchurTypeLinearSolver(reduced) : 0;
		pp.LinearSolver = LinearSolver.Create(new LinearSolverOptions(
			type,
			options.PreconditionerType,
			numEliminateBlocks,
			options.MinLinearSolverIterations,
			options.MaxLinearSolverIterations));
		bool dense = type is LinearSolverType.DenseQr or LinearSolverType.DenseNormalCholesky;
		pp.Evaluator = new ProgramEvaluator(reduced, dense, numEliminateBlocks, options.NumThreads);
		pp.Jacobian = pp.Evaluator.CreateJacobian();
		pp.Strategy = new LevenbergMarquardtStrategy(
			pp.LinearSolver,
			options.InitialTrustRegionRadius,
			options.MaxTrustRegionRadius,
			options.MinLmDiagonal,
			options.MaxLmDiagonal);
		return pp;
	}

	// solver.cc Minimize.
	private static void Minimize(SolverOptions options, Preprocessed pp, SolverSummary summary, CancellationToken cancellationToken)
	{
		Program program = pp.ReducedProgram;
		if (program.ParameterBlocks.Count == 0)
		{
			summary.Message = "Function tolerance reached. No non-constant parameter blocks found.";
			summary.TerminationType = TerminationType.Convergence;
			summary.InitialCost = summary.FixedCost;
			summary.FinalCost = summary.FixedCost;
			return;
		}

		var reducedParameters = new double[program.NumParameters];
		program.ParameterBlocksToStateVector(reducedParameters);
		double[] originalReducedParameters = (double[])reducedParameters.Clone();

		var minimizer = new TrustRegionMinimizer(options, pp.Evaluator!, pp.Jacobian!, pp.Strategy!, summary, cancellationToken);
		minimizer.Minimize(reducedParameters);

		program.StateVectorToParameterBlocks(summary.IsSolutionUsable ? reducedParameters : originalReducedParameters);
		program.CopyParameterBlockStateToUserState();
	}

	// SetSummaryFinalCost: the minimizer may take nonmonotonic steps, so the final cost is
	// the least cost over all iterations, not the last one.
	private static void SetSummaryFinalCost(SolverSummary summary)
	{
		summary.FinalCost = summary.InitialCost;
		foreach (IterationSummary iteration in summary.Iterations)
		{
			summary.FinalCost = Math.Min(iteration.Cost, summary.FinalCost);
		}
	}
}
