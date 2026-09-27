// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/conjugate_gradients_solver.h,
// internal/ceres/iterative_schur_complement_solver.cc,
// internal/ceres/schur_jacobi_preconditioner.cc and internal/ceres/preconditioner.h
// (IdentityPreconditioner, SparseMatrixPreconditionerWrapper) (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// ITERATIVE_SCHUR: preconditioned conjugate gradients on the implicit Schur complement
// (ImplicitSchurComplement.cs), then back substitution for the E blocks. The preconditioners
// COLMAP can select: IDENTITY, JACOBI ((F'F + D_f^2)^-1, block diagonal) and SCHUR_JACOBI (the
// inverted diagonal blocks of the Schur complement itself, computed by SchurEliminator.cs;
// COLMAP's choice for large bundle adjustments). Not ported: the power-series (SPSE)
// preconditioner and initialization and the visibility-based CLUSTER_* preconditioners,
// which COLMAP never selects, and use_explicit_schur_complement (off by default).
// Vector reductions (dot products, norms) are sequential sums, so the iterates do not depend
// on the thread count; Ceres' parallel reductions (and Eigen's vectorized norm) sum in a
// different order (docs/CPP_DIVERGENCES.md entry 35).

namespace ColmapSharp.Solver;

/// <summary>A symmetric linear operator y += A x (ceres::internal::LinearOperator for CG).</summary>
internal interface ILinearOperator
{
	/// <summary>y += A x.</summary>
	void RightMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y);
}

/// <summary>ceres::internal::ConjugateGradientsSolverOptions.</summary>
internal readonly record struct ConjugateGradientsSolverOptions(
	int MinNumIterations = 1,
	int MaxNumIterations = 1,
	int ResidualResetPeriod = 10,
	double RTolerance = 0.0,
	double QTolerance = 0.0);

/// <summary>ceres::internal::ConjugateGradientsSolver.</summary>
internal static class ConjugateGradients
{
	/// <summary>
	/// Solves lhs x = rhs by preconditioned CG, starting from <paramref name="solution"/>.
	/// Converges when Nash and Sofer's quadratic-model criterion
	/// zeta = i (Q_i - Q_{i-1}) / Q_i drops below the q tolerance, or when |r| &lt;= r
	/// tolerance * |rhs|.
	/// </summary>
	public static LinearSolverSummary Solve(
		ConjugateGradientsSolverOptions options,
		ILinearOperator lhs,
		ReadOnlySpan<double> rhs,
		ILinearOperator preconditioner,
		Span<double> solution)
	{
		int n = rhs.Length;
		var p = new double[n];
		var r = new double[n];
		var z = new double[n];
		var tmp = new double[n];
		var summary = (Type: LinearSolverTerminationType.NoConvergence, Iterations: 0, Message: "Maximum number of iterations reached.");

		double normRhs = Norm(rhs);
		if (normRhs == 0.0)
		{
			solution.Clear();
			return new LinearSolverSummary(LinearSolverTerminationType.Success, 0, "Convergence. |b| = 0.");
		}

		double tolR = options.RTolerance * normRhs;

		lhs.RightMultiplyAndAccumulate(solution, tmp);
		Axpby(1.0, rhs, -1.0, tmp, r);
		double normR = Norm(r);
		if (options.MinNumIterations == 0 && normR <= tolR)
		{
			return new LinearSolverSummary(
				LinearSolverTerminationType.Success, 0, $"Convergence. |r| = {SolverSummary.FormatE(normR)} <= {SolverSummary.FormatE(tolR)}.");
		}

		double rho = 1.0;

		// Initial value of the quadratic model Q = x'Ax - 2 * b'x.
		Axpby(1.0, rhs, 1.0, r, tmp);
		double q0 = -Dot(solution, tmp);

		for (summary.Iterations = 1; ; summary.Iterations++)
		{
			Array.Clear(z);
			preconditioner.RightMultiplyAndAccumulate(r, z);

			double lastRho = rho;
			rho = Dot(r, z);
			if (IsZeroOrInfinity(rho))
			{
				summary = (LinearSolverTerminationType.Failure, summary.Iterations, $"Numerical failure. rho = r'z = {SolverSummary.FormatE(rho)}.");
				break;
			}

			if (summary.Iterations == 1)
			{
				z.CopyTo(p, 0);
			}
			else
			{
				double beta = rho / lastRho;
				if (IsZeroOrInfinity(beta))
				{
					summary = (
						LinearSolverTerminationType.Failure,
						summary.Iterations,
						$"Numerical failure. beta = rho_n / rho_{{n-1}} = {SolverSummary.FormatE(beta)}, rho_n = {SolverSummary.FormatE(rho)}, rho_{{n-1}} = {SolverSummary.FormatE(lastRho)}");
					break;
				}

				Axpby(1.0, z, beta, p, p);
			}

			double[] q = z;
			Array.Clear(q);
			lhs.RightMultiplyAndAccumulate(p, q);
			double pq = Dot(p, q);
			if (pq <= 0 || double.IsInfinity(pq))
			{
				summary = (
					LinearSolverTerminationType.NoConvergence,
					summary.Iterations,
					$"Matrix is indefinite, no more progress can be made. p'q = {SolverSummary.FormatE(pq)}. |p| = {SolverSummary.FormatE(Norm(p))}, |q| = {SolverSummary.FormatE(Norm(q))}");
				break;
			}

			double alpha = rho / pq;
			if (double.IsInfinity(alpha))
			{
				summary = (
					LinearSolverTerminationType.Failure,
					summary.Iterations,
					$"Numerical failure. alpha = rho / pq = {SolverSummary.FormatE(alpha)}, rho = {SolverSummary.FormatE(rho)}, pq = {SolverSummary.FormatE(pq)}.");
				break;
			}

			Axpby(1.0, solution, alpha, p, solution);

			// Ideally we would just use the update r = r - alpha*q to keep track of the
			// residual vector. However this estimate tends to drift over time due to round
			// off errors. Thus every residual_reset_period iterations, we calculate the
			// residual as r = b - Ax. We do not do this every iteration because this requires
			// an additional matrix vector multiply which would double the complexity of the
			// CG algorithm.
			if (summary.Iterations % options.ResidualResetPeriod == 0)
			{
				Array.Clear(tmp);
				lhs.RightMultiplyAndAccumulate(solution, tmp);
				Axpby(1.0, rhs, -1.0, tmp, r);
			}
			else
			{
				Axpby(1.0, r, -alpha, q, r);
			}

			// Quadratic model based termination.
			//   Q1 = x'Ax - 2 * b' x.
			// Since A x = b - r, Q1 = -x'(b + r).
			Axpby(1.0, rhs, 1.0, r, tmp);
			double q1 = -Dot(solution, tmp);

			// For PSD matrices A, let
			//
			//   Q(x) = x'Ax - 2b'x
			//
			// be the cost of the quadratic function defined by A and b. Then, the solver
			// terminates at iteration i if
			//
			//   i * (Q(x_i) - Q(x_i-1)) / Q(x_i) < q_tolerance.
			//
			// This termination criterion is more useful when using CG to solve the Newton
			// step. This particular convergence test comes from Stephen Nash's work on
			// truncated Newton methods. References:
			//
			//   1. Stephen G. Nash & Ariela Sofer, Assessing A Search Direction Within A
			//   Truncated Newton Method, Operation Research Letters 9(1990) 219-221.
			//
			//   2. Stephen G. Nash, A Survey of Truncated Newton Methods, Journal of
			//   Computational and Applied Mathematics, 124(1-2), 45-59, 2000.
			double zeta = summary.Iterations * (q1 - q0) / q1;
			if (zeta < options.QTolerance && summary.Iterations >= options.MinNumIterations)
			{
				summary = (
					LinearSolverTerminationType.Success,
					summary.Iterations,
					$"Iteration: {summary.Iterations} Convergence: zeta = {SolverSummary.FormatE(zeta)} < {SolverSummary.FormatE(options.QTolerance)}. |r| = {SolverSummary.FormatE(Norm(r))}");
				break;
			}

			q0 = q1;

			// Residual based termination.
			normR = Norm(r);
			if (normR <= tolR && summary.Iterations >= options.MinNumIterations)
			{
				summary = (
					LinearSolverTerminationType.Success,
					summary.Iterations,
					$"Iteration: {summary.Iterations} Convergence. |r| = {SolverSummary.FormatE(normR)} <= {SolverSummary.FormatE(tolR)}.");
				break;
			}

			if (summary.Iterations >= options.MaxNumIterations)
			{
				break;
			}
		}

		return new LinearSolverSummary(summary.Type, summary.Iterations, summary.Message);
	}

	private static bool IsZeroOrInfinity(double x) => x == 0.0 || double.IsInfinity(x);

	private static double Dot(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
	{
		double sum = 0.0;
		for (int i = 0; i < x.Length; i++)
		{
			sum += x[i] * y[i];
		}

		return sum;
	}

	private static double Norm(ReadOnlySpan<double> x) => Math.Sqrt(Dot(x, x));

	// z = a * x + b * y; z may alias x or y.
	private static void Axpby(double a, ReadOnlySpan<double> x, double b, ReadOnlySpan<double> y, Span<double> z)
	{
		for (int i = 0; i < z.Length; i++)
		{
			z[i] = (a * x[i]) + (b * y[i]);
		}
	}
}

/// <summary>ceres::PreconditionerType, the ones COLMAP can select.</summary>
public enum PreconditionerType
{
	/// <summary>No preconditioning.</summary>
	Identity,

	/// <summary>Block Jacobi of the F part: (F'F + D_f^2)^-1 for ITERATIVE_SCHUR.</summary>
	Jacobi,

	/// <summary>The inverted diagonal blocks of the Schur complement (COLMAP's choice for large problems).</summary>
	SchurJacobi,
}

/// <summary>ceres::internal::IterativeSchurComplementSolver.</summary>
internal sealed class IterativeSchurComplementSolver(
	int numEliminateBlocks, PreconditionerType preconditionerType, int minNumIterations, int maxNumIterations, int numThreads = 1) : LinearSolver
{
	private ImplicitSchurComplement? schurComplement;
	private CompressedRowBlockStructure? structure;
	private SchurEliminator? schurJacobiEliminator;
	private BlockRandomAccessDiagonalMatrix? schurJacobi;
	private double[] reducedSolution = [];

	/// <inheritdoc/>
	protected override LinearSolverSummary SolveImpl(
		SparseMatrix a, ReadOnlySpan<double> b, ReadOnlySpan<double> d, Span<double> x, double qTolerance, double rTolerance)
	{
		var matrix = (BlockSparseMatrix)a;
		CompressedRowBlockStructure bs = matrix.Structure;
		if (!ReferenceEquals(structure, bs))
		{
			structure = bs;
			schurComplement = new ImplicitSchurComplement(numEliminateBlocks, preconditionerType == PreconditionerType.Jacobi);
			schurJacobiEliminator = null;
			schurJacobi = null;
		}

		schurComplement!.Init(matrix, d, b);
		int numSchurComplementBlocks = bs.Cols.Length - numEliminateBlocks;
		if (numSchurComplementBlocks == 0)
		{
			// No parameter blocks left in the schur complement.
			schurComplement.BackSubstitute([], x);
			return new LinearSolverSummary(LinearSolverTerminationType.Success, 0, string.Empty);
		}

		// Initialize the solution to the Schur complement system.
		if (reducedSolution.Length != schurComplement.NumRows)
		{
			reducedSolution = new double[schurComplement.NumRows];
		}

		Array.Clear(reducedSolution);
		ILinearOperator preconditioner = UpdatePreconditioner(matrix, d);

		var cgOptions = new ConjugateGradientsSolverOptions(
			MinNumIterations: minNumIterations,
			MaxNumIterations: maxNumIterations,
			ResidualResetPeriod: 10,
			RTolerance: rTolerance,
			QTolerance: qTolerance);
		LinearSolverSummary summary = ConjugateGradients.Solve(
			cgOptions, new SchurOperator(schurComplement), schurComplement.Rhs, preconditioner, reducedSolution);
		if (summary.TerminationType is not LinearSolverTerminationType.Failure and not LinearSolverTerminationType.FatalError)
		{
			schurComplement.BackSubstitute(reducedSolution, x);
		}

		return summary;
	}

	// CreatePreconditioner + Preconditioner::Update.
	private ILinearOperator UpdatePreconditioner(BlockSparseMatrix a, ReadOnlySpan<double> d)
	{
		switch (preconditionerType)
		{
			case PreconditionerType.Identity:
				return new IdentityOperator();
			case PreconditionerType.Jacobi:
				return new DiagonalOperator(schurComplement!.BlockDiagonalFtFInverse!);
			case PreconditionerType.SchurJacobi:
				if (schurJacobi is null)
				{
					schurJacobi = new BlockRandomAccessDiagonalMatrix(SchurEliminator.ReducedBlocks(a.Structure, numEliminateBlocks));
					schurJacobiEliminator = new SchurEliminator(numThreads);
					schurJacobiEliminator.Init(numEliminateBlocks, a.Structure);
				}

				// Compute a subset of the entries of the Schur complement and invert them.
				schurJacobiEliminator!.Eliminate(a, [], d, schurJacobi, []);
				schurJacobi.Invert();
				return new DiagonalOperator(schurJacobi);
			default:
				throw new ArgumentOutOfRangeException(nameof(preconditionerType), preconditionerType, "Unknown Preconditioner Type");
		}
	}

	private sealed class SchurOperator(ImplicitSchurComplement schurComplement) : ILinearOperator
	{
		// ImplicitSchurComplement assigns y rather than accumulating; CG always passes a
		// zeroed y, so the two agree.
		public void RightMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y) =>
			schurComplement.RightMultiplyAndAccumulate(x, y);
	}

	private sealed class DiagonalOperator(BlockRandomAccessDiagonalMatrix matrix) : ILinearOperator
	{
		public void RightMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y) =>
			matrix.RightMultiplyAndAccumulate(x, y);
	}
}

/// <summary>ceres::internal::IdentityPreconditioner: y += x.</summary>
internal sealed class IdentityOperator : ILinearOperator
{
	/// <inheritdoc/>
	public void RightMultiplyAndAccumulate(ReadOnlySpan<double> x, Span<double> y)
	{
		for (int i = 0; i < y.Length; i++)
		{
			y[i] += x[i];
		}
	}
}
