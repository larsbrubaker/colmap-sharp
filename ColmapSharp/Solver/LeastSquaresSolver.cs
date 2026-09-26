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
// put the eliminated blocks first (SchurOrdering.cs), chosen automatically or taken from
// the user's SolverOptions.LinearSolverOrdering; Ceres also reorders for SuiteSparse's own
// fill-reducing ordering, which the simplicial Cholesky here does itself
// (docs/CPP_DIVERGENCES.md entry 22). A problem whose variable blocks have bounds must start
// feasible and runs the projected line search (TrustRegionMinimizer.cs).

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
		summary.NumLineSearchSteps = 0;

		try
		{
			if (!program.SetParameterBlockStatePtrsToUserStatePtrs())
			{
				summary.Message = "Manifold::PlusJacobian failed at the initial parameter values.";
				return summary;
			}

			Preprocessed? pp = Preprocess(options, problem, summary, out error);
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
	private static Preprocessed? Preprocess(SolverOptions options, Problem problem, SolverSummary summary, out string error)
	{
		Program program = problem.Program;

		// IsProgramValid.
		if (!program.ParameterBlocksAreFinite(out error) || !program.IsFeasible(out error))
		{
			return null;
		}

		var removedParameterBlocks = new List<ParameterBlock>();
		Program? reduced = program.CreateReducedProgram(removedParameterBlocks, out double fixedCost, out error);
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

		// SetupLinearSolver. A user ordering loses the blocks the reduction removed (on a
		// copy: Ceres edits the caller's ordering). If that empties the first elimination
		// group, a Schur solver has nothing to eliminate and Ceres switches solvers
		// (docs/CPP_DIVERGENCES.md entry 37).
		LinearSolverType type = options.LinearSolverType;
		ParameterBlockOrdering? ordering = null;
		if (options.LinearSolverOrdering is not null)
		{
			ordering = options.LinearSolverOrdering.Clone();

			// Ceres CHECKs (and aborts) on an empty ordering in MinNonZeroGroup; here the solve
			// fails with the reason instead.
			if (ordering.NumGroups == 0)
			{
				error = "Solver::Options::linear_solver_ordering is empty (Check failed: NumGroups() != 0). Leave it unset to let the solver choose, or add every parameter block to a group.";
				return null;
			}

			int minGroupId = ordering.MinNonZeroGroup();
			ordering.Remove(removedParameterBlocks.ConvertAll(b => b.UserState));
			if (LinearSolver.IsSchurType(type) && ordering.NumGroups == 0)
			{
				error = "Solver::Options::linear_solver_ordering holds only parameter blocks that are constant or unused (Check failed: NumGroups() != 0), so the Schur solver has no ordering for the variable blocks.";
				return null;
			}

			if (LinearSolver.IsSchurType(type) && minGroupId != ordering.MinNonZeroGroup())
			{
				if (!LinearSolverForZeroEBlocks(type, out type, out error))
				{
					return null;
				}

				summary.LinearSolverTypeUsed = type;
			}
		}

		// ReorderProgram: the Schur solvers eliminate the first group of the ordering, or an
		// independent set they choose, which reorders the program.
		int numEliminateBlocks = 0;
		if (LinearSolver.IsSchurType(type))
		{
			if (!SchurOrdering.ReorderProgramForSchurTypeLinearSolver(problem, ordering, reduced, out numEliminateBlocks, out error))
			{
				return null;
			}
		}
		else if (type == LinearSolverType.SparseNormalCholesky && ordering is not null && ordering.NumElements != reduced.ParameterBlocks.Count)
		{
			// ReorderProgramForSparseCholesky's check; the fill-reducing ordering itself is
			// the simplicial Cholesky's own (docs/CPP_DIVERGENCES.md entry 22).
			error = $"The program has {reduced.ParameterBlocks.Count} parameter blocks, but the parameter block ordering has {ordering.NumElements} parameter blocks.";
			return null;
		}

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

		var minimizer = new TrustRegionMinimizer(
			options, pp.Evaluator!, pp.Jacobian!, pp.Strategy!, summary, program.IsBoundsConstrained(), cancellationToken);
		minimizer.Minimize(reducedParameters);

		program.StateVectorToParameterBlocks(summary.IsSolutionUsable ? reducedParameters : originalReducedParameters);
		program.CopyParameterBlockStateToUserState();
	}

	// LinearSolver::LinearSolverForZeroEBlocks: the solver to use when a user ordering leaves
	// the Schur solver nothing to eliminate. ITERATIVE_SCHUR would become CGNR, which is not
	// ported, so that case fails with a message rather than running something else.
	private static bool LinearSolverForZeroEBlocks(LinearSolverType given, out LinearSolverType type, out string error)
	{
		error = string.Empty;
		type = given switch
		{
			LinearSolverType.SparseSchur => LinearSolverType.SparseNormalCholesky,
			LinearSolverType.DenseSchur => LinearSolverType.DenseQr,
			_ => given,
		};
		if (given == LinearSolverType.IterativeSchur)
		{
			error = "No E blocks: Ceres switches ITERATIVE_SCHUR to CGNR, which is not ported. Use SPARSE_SCHUR or give the first elimination group a variable block.";
			return false;
		}

		return true;
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
