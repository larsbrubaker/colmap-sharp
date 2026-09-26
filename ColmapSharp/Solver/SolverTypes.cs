// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 include/ceres/solver.h, include/ceres/iteration_callback.h,
// include/ceres/types.h and internal/ceres/solver.cc (Options::IsValid, BriefReport)
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The public types around LeastSquaresSolver.Solve: the options COLMAP sets, the summary it
// reads (termination type, costs, iteration counts, residual counts, BriefReport, message,
// IsSolutionUsable), per-iteration summaries and the iteration callback COLMAP's
// CancellationCallback implements. Defaults are Ceres 2.2's. Options that only select
// Ceres features not ported (line search minimizer, dogleg, inner iterations, logging to
// stdout, the GPU and library switches) are absent rather than accepted and ignored.
// FullReport (a long table COLMAP only logs when print_summary is on) is not ported yet.

using System.Globalization;
using System.Text;

namespace ColmapSharp.Solver;

/// <summary>ceres::TerminationType.</summary>
public enum TerminationType
{
	/// <summary>A convergence tolerance was met.</summary>
	Convergence,

	/// <summary>An iteration or time limit was reached; the solution is usable.</summary>
	NoConvergence,

	/// <summary>The solver failed; the parameters are left as they were.</summary>
	Failure,

	/// <summary>A callback (or cancellation) stopped the solve; the solution is usable.</summary>
	UserSuccess,

	/// <summary>A callback aborted the solve; the parameters are left as they were.</summary>
	UserFailure,
}

/// <summary>ceres::CallbackReturnType.</summary>
public enum CallbackReturnType
{
	/// <summary>Keep going.</summary>
	SolverContinue,

	/// <summary>Stop with USER_FAILURE.</summary>
	SolverAbort,

	/// <summary>Stop with USER_SUCCESS, keeping the best solution so far.</summary>
	SolverTerminateSuccessfully,
}

/// <summary>ceres::IterationSummary: the state after one minimizer iteration.</summary>
public sealed record IterationSummary
{
	/// <summary>Iteration number; 0 is the initial evaluation.</summary>
	public int Iteration { get; init; }

	/// <summary>Whether the linear solver produced a step that decreases the model.</summary>
	public bool StepIsValid { get; init; }

	/// <summary>Whether the step was accepted without decreasing the minimum cost.</summary>
	public bool StepIsNonmonotonic { get; init; }

	/// <summary>Whether the step was accepted.</summary>
	public bool StepIsSuccessful { get; init; }

	/// <summary>Cost at this iteration's point (the candidate's for a rejected step), fixed cost included.</summary>
	public double Cost { get; init; }

	/// <summary>Current cost minus the candidate's.</summary>
	public double CostChange { get; init; }

	/// <summary>Infinity norm of the (projected) gradient.</summary>
	public double GradientMaxNorm { get; init; }

	/// <summary>2-norm of the (projected) gradient.</summary>
	public double GradientNorm { get; init; }

	/// <summary>2-norm of the step.</summary>
	public double StepNorm { get; init; }

	/// <summary>Actual over predicted cost decrease.</summary>
	public double RelativeDecrease { get; init; }

	/// <summary>Trust region radius after the iteration.</summary>
	public double TrustRegionRadius { get; init; }

	/// <summary>Forcing sequence value (iterative linear solvers).</summary>
	public double Eta { get; init; }

	/// <summary>Iterations the linear solver used.</summary>
	public int LinearSolverIterations { get; init; }

	/// <summary>Seconds spent in the iteration.</summary>
	public double IterationTimeInSeconds { get; init; }

	/// <summary>Seconds spent computing the step.</summary>
	public double StepSolverTimeInSeconds { get; init; }

	/// <summary>Seconds since the solve started.</summary>
	public double CumulativeTimeInSeconds { get; init; }
}

/// <summary>ceres::IterationCallback: called after every minimizer iteration.</summary>
public interface IIterationCallback
{
	/// <summary>Inspects the iteration and says whether to continue.</summary>
	CallbackReturnType Invoke(IterationSummary summary);
}

/// <summary>ceres::Solver::Options, the trust-region / Levenberg-Marquardt subset.</summary>
public sealed class SolverOptions
{
	/// <summary>Maximum number of iterations.</summary>
	public int MaxNumIterations { get; set; } = 50;

	/// <summary>Maximum wall time.</summary>
	public double MaxSolverTimeInSeconds { get; set; } = 1e9;

	/// <summary>Threads for Jacobian evaluation; results do not depend on it.</summary>
	public int NumThreads { get; set; } = 1;

	/// <summary>Initial trust region radius (1 / initial LM damping).</summary>
	public double InitialTrustRegionRadius { get; set; } = 1e4;

	/// <summary>Largest trust region radius.</summary>
	public double MaxTrustRegionRadius { get; set; } = 1e16;

	/// <summary>The solve converges when the radius falls below this.</summary>
	public double MinTrustRegionRadius { get; set; } = 1e-32;

	/// <summary>Smallest relative decrease for a step to be accepted.</summary>
	public double MinRelativeDecrease { get; set; } = 1e-3;

	/// <summary>Lower clamp of the LM diagonal (diag(J'J)).</summary>
	public double MinLmDiagonal { get; set; } = 1e-6;

	/// <summary>Upper clamp of the LM diagonal.</summary>
	public double MaxLmDiagonal { get; set; } = 1e32;

	/// <summary>The solve fails after this many invalid steps in a row.</summary>
	public int MaxNumConsecutiveInvalidSteps { get; set; } = 5;

	/// <summary>Converged when |cost change| / cost &lt;= this.</summary>
	public double FunctionTolerance { get; set; } = 1e-6;

	/// <summary>Converged when the max-norm of the projected gradient &lt;= this.</summary>
	public double GradientTolerance { get; set; } = 1e-10;

	/// <summary>Converged when |step| &lt;= this * (|x| + this).</summary>
	public double ParameterTolerance { get; set; } = 1e-8;

	/// <summary>Accept steps that increase the cost within the nonmonotonic window.</summary>
	public bool UseNonmonotonicSteps { get; set; }

	/// <summary>Window of the nonmonotonic step acceptance.</summary>
	public int MaxConsecutiveNonmonotonicSteps { get; set; } = 5;

	/// <summary>The linear solver for the LM step.</summary>
	public LinearSolverType LinearSolverType { get; set; } = LinearSolverType.SparseNormalCholesky;

	/// <summary>Preconditioner of ITERATIVE_SCHUR (Ceres' default: JACOBI).</summary>
	public PreconditionerType PreconditionerType { get; set; } = PreconditionerType.Jacobi;

	/// <summary>Minimum iterations of an iterative linear solver.</summary>
	public int MinLinearSolverIterations { get; set; }

	/// <summary>Maximum iterations of an iterative linear solver.</summary>
	public int MaxLinearSolverIterations { get; set; } = 500;

	/// <summary>Forcing sequence for iterative linear solvers.</summary>
	public double Eta { get; set; } = 1e-1;

	/// <summary>Scale the Jacobian columns by 1 / (1 + norm) before solving.</summary>
	public bool JacobiScaling { get; set; } = true;

	/// <summary>Called after every iteration, in order, until one does not continue.</summary>
	public List<IIterationCallback> Callbacks { get; } = [];

	/// <summary>Solver::Options::IsValid: null if valid, else Ceres' error message.</summary>
	public string? Validate()
	{
		return Ge(MaxNumIterations, 0, "max_num_iterations")
			?? Ge(MaxSolverTimeInSeconds, 0.0, "max_solver_time_in_seconds")
			?? Ge(FunctionTolerance, 0.0, "function_tolerance")
			?? Ge(GradientTolerance, 0.0, "gradient_tolerance")
			?? Ge(ParameterTolerance, 0.0, "parameter_tolerance")
			?? Gt(NumThreads, 0, "num_threads")
			?? Gt(InitialTrustRegionRadius, 0.0, "initial_trust_region_radius")
			?? Gt(MinTrustRegionRadius, 0.0, "min_trust_region_radius")
			?? Gt(MaxTrustRegionRadius, 0.0, "max_trust_region_radius")
			?? LeOption(MinTrustRegionRadius, MaxTrustRegionRadius, "min_trust_region_radius", "max_trust_region_radius")
			?? LeOption(MinTrustRegionRadius, InitialTrustRegionRadius, "min_trust_region_radius", "initial_trust_region_radius")
			?? LeOption(InitialTrustRegionRadius, MaxTrustRegionRadius, "initial_trust_region_radius", "max_trust_region_radius")
			?? Ge(MinRelativeDecrease, 0.0, "min_relative_decrease")
			?? Ge(MinLmDiagonal, 0.0, "min_lm_diagonal")
			?? Ge(MaxLmDiagonal, 0.0, "max_lm_diagonal")
			?? LeOption(MinLmDiagonal, MaxLmDiagonal, "min_lm_diagonal", "max_lm_diagonal")
			?? Ge(MaxNumConsecutiveInvalidSteps, 0, "max_num_consecutive_invalid_steps")
			?? Gt(Eta, 0.0, "eta")
			?? Ge(MinLinearSolverIterations, 0, "min_linear_solver_iterations")
			?? Ge(MaxLinearSolverIterations, 0, "max_linear_solver_iterations")
			?? LeOption(MinLinearSolverIterations, MaxLinearSolverIterations, "min_linear_solver_iterations", "max_linear_solver_iterations")
			?? (UseNonmonotonicSteps ? Gt(MaxConsecutiveNonmonotonicSteps, 0, "max_consecutive_nonmonotonic_steps") : null);
	}

	private static string? Ge(double value, double bound, string name) =>
		value >= bound ? null : Violation(name, value, $"{name} >= {Format(bound)}");

	private static string? Gt(double value, double bound, string name) =>
		value > bound ? null : Violation(name, value, $"{name} > {Format(bound)}");

	private static string Violation(string name, double value, string constraint) =>
		$"Invalid configuration. Solver::Options::{name} = {Format(value)}. Violated constraint: Solver::Options::{constraint}";

	private static string? LeOption(double x, double y, string xName, string yName) =>
		x <= y
			? null
			: $"Invalid configuration. Solver::Options::{xName} = {Format(x)}. Solver::Options::{yName} = {Format(y)}. "
				+ $"Violated constraint: Solver::Options::{xName}<= Solver::Options::{yName}.";

	private static string Format(double value) => Util.CppStreamFormat.FormatDouble(value);
}

/// <summary>ceres::Solver::Summary, the fields COLMAP reads plus the counters behind them.</summary>
public sealed class SolverSummary
{
	/// <summary>Why the solver stopped.</summary>
	public TerminationType TerminationType { get; internal set; } = TerminationType.Failure;

	/// <summary>Human-readable reason for stopping.</summary>
	public string Message { get; internal set; } = "ceres::Solve was not called.";

	/// <summary>Cost at the start, fixed cost included (-1 if not evaluated).</summary>
	public double InitialCost { get; internal set; } = -1.0;

	/// <summary>Cost at the end, fixed cost included (-1 if not evaluated).</summary>
	public double FinalCost { get; internal set; } = -1.0;

	/// <summary>Cost of the residual blocks that depend only on constant parameters.</summary>
	public double FixedCost { get; internal set; } = -1.0;

	/// <summary>Every iteration, the initial evaluation (iteration 0) first.</summary>
	public List<IterationSummary> Iterations { get; } = [];

	/// <summary>Number of accepted steps (iteration 0 counts as one, as in Ceres).</summary>
	public int NumSuccessfulSteps { get; internal set; } = -1;

	/// <summary>Number of rejected steps.</summary>
	public int NumUnsuccessfulSteps { get; internal set; } = -1;

	/// <summary>Parameter blocks in the problem.</summary>
	public int NumParameterBlocks { get; internal set; } = -1;

	/// <summary>Scalar parameters in the problem.</summary>
	public int NumParameters { get; internal set; } = -1;

	/// <summary>Tangent-space parameters in the problem.</summary>
	public int NumEffectiveParameters { get; internal set; } = -1;

	/// <summary>Residual blocks in the problem.</summary>
	public int NumResidualBlocks { get; internal set; } = -1;

	/// <summary>Residuals in the problem.</summary>
	public int NumResiduals { get; internal set; } = -1;

	/// <summary>Parameter blocks after removing the constant ones.</summary>
	public int NumParameterBlocksReduced { get; internal set; } = -1;

	/// <summary>Scalar parameters after reduction.</summary>
	public int NumParametersReduced { get; internal set; } = -1;

	/// <summary>Tangent-space parameters after reduction.</summary>
	public int NumEffectiveParametersReduced { get; internal set; } = -1;

	/// <summary>Residual blocks after reduction.</summary>
	public int NumResidualBlocksReduced { get; internal set; } = -1;

	/// <summary>Residuals after reduction (COLMAP's BA summary num_residuals).</summary>
	public int NumResidualsReduced { get; internal set; } = -1;

	/// <summary>Residual-only evaluations.</summary>
	public int NumResidualEvaluations { get; internal set; } = -1;

	/// <summary>Jacobian evaluations.</summary>
	public int NumJacobianEvaluations { get; internal set; } = -1;

	/// <summary>Linear solves.</summary>
	public int NumLinearSolves { get; internal set; } = -1;

	/// <summary>The linear solver used.</summary>
	public LinearSolverType LinearSolverTypeUsed { get; internal set; }

	/// <summary>Threads used.</summary>
	public int NumThreadsUsed { get; internal set; } = -1;

	/// <summary>Wall time of the whole solve.</summary>
	public double TotalTimeInSeconds { get; internal set; } = -1.0;

	/// <summary>Solver::Summary::IsSolutionUsable: CONVERGENCE, NO_CONVERGENCE or USER_SUCCESS.</summary>
	public bool IsSolutionUsable =>
		TerminationType is TerminationType.Convergence or TerminationType.NoConvergence or TerminationType.UserSuccess;

	/// <summary>Solver::Summary::BriefReport, character for character.</summary>
	public string BriefReport()
	{
		var report = new StringBuilder("Ceres Solver Report: ");
		report.Append(CultureInfo.InvariantCulture, $"Iterations: {NumSuccessfulSteps + NumUnsuccessfulSteps}, ");
		report.Append("Initial cost: ").Append(FormatE(InitialCost)).Append(", ");
		report.Append("Final cost: ").Append(FormatE(FinalCost)).Append(", ");
		report.Append("Termination: ").Append(TerminationTypeToString(TerminationType));
		return report.ToString();
	}

	/// <summary>ceres::TerminationTypeToString.</summary>
	public static string TerminationTypeToString(TerminationType type) => type switch
	{
		TerminationType.Convergence => "CONVERGENCE",
		TerminationType.NoConvergence => "NO_CONVERGENCE",
		TerminationType.Failure => "FAILURE",
		TerminationType.UserSuccess => "USER_SUCCESS",
		TerminationType.UserFailure => "USER_FAILURE",
		_ => "UNKNOWN",
	};

	/// <summary>printf's "%e": six fraction digits and an exponent of at least two digits.</summary>
	internal static string FormatE(double value)
	{
		if (double.IsNaN(value))
		{
			return double.IsNegative(value) ? "-nan" : "nan";
		}

		if (double.IsInfinity(value))
		{
			return value < 0 ? "-inf" : "inf";
		}

		// "E6" rounds the exact binary value correctly (unlike a custom format, which rounds
		// a 15-digit intermediate); only the exponent's width differs from printf.
		string s = value.ToString("E6", CultureInfo.InvariantCulture);
		int e = s.IndexOf('E', StringComparison.Ordinal);
		int exponent = int.Parse(s.AsSpan(e + 1), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
		return string.Concat(
			s.AsSpan(0, e),
			exponent < 0 ? "e-" : "e+",
			Math.Abs(exponent).ToString("00", CultureInfo.InvariantCulture));
	}
}
