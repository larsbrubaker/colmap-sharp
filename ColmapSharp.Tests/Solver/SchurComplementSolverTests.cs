// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/schur_complement_solver_test.cc and
// internal/ceres/iterative_schur_complement_solver_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// SchurComplementSolverTests and IterativeSchurComplementSolverTests (Ceres' tests, not
// COLMAP's): DENSE_SCHUR, SPARSE_SCHUR (ColmapSharp/Solver/SchurComplementSolvers.cs) and
// ITERATIVE_SCHUR (IterativeSchurSolver.cs) against DENSE_QR on the same problems, with and
// without the diagonal, same tolerances (|x - x_qr| / num_cols <= 1e-10 for the direct
// solvers, |x - x_qr| < 1e-14 for the iterative one with r_tolerance 1e-12).
// - Ceres names the dense cases by back end (EIGEN) and the sparse ones by library and
//   ordering. The one sparse back end here is the simplicial LLT with AMD, so the Eigen-sparse
//   AMD and NATURAL cases both run it (the ordering is the Cholesky's own either way,
//   divergence 35).
// Not ported: the LAPACK, SuiteSparse, Accelerate and NESDIS (METIS) variants (those
// libraries are not ported), and IterativeSchurComplementSolverTest's
// NormalProblemSchurJacobiWithPowerSeriesExpansionInitialization and
// NormalProblemPowerSeriesExpansionPreconditioner (the SPSE preconditioner and
// initialization are not ported; COLMAP never selects them).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class SchurComplementSolverTests
{
	private static double[] QrSolution(LinearLeastSquaresProblem problem, double[] d)
	{
		var x = new double[problem.A.NumCols];
		LinearSolver.Create(LinearSolverType.DenseQr).Solve(problem.DenseSparseA(), problem.B, d, x);
		return x;
	}

	private static async Task ComputeAndCompareSolutions(int problemId, bool regularization, LinearSolverType linearSolverType)
	{
		LinearLeastSquaresProblem problem = LinearLeastSquaresProblems.FromId(problemId);
		int numCols = problem.A.NumCols;
		double[] d = regularization ? problem.D : [];
		double[] expected = QrSolution(problem, d);

		LinearSolver solver = LinearSolver.Create(
			new LinearSolverOptions(linearSolverType, NumEliminateBlocks: problem.NumEliminateBlocks));
		var x = new double[numCols];
		LinearSolverSummary summary = solver.Solve(problem.A, problem.B, d, x);
		await Assert.That(summary.TerminationType).IsEqualTo(LinearSolverTerminationType.Success);
		await Assert.That((new VectorXd(expected) - new VectorXd(x)).Norm() / numCols).IsEqualTo(0.0).Within(1e-10);
	}

	[Test]
	public async Task DenseSchurWithEigenSmallProblem()
	{
		await ComputeAndCompareSolutions(2, false, LinearSolverType.DenseSchur);
		await ComputeAndCompareSolutions(2, true, LinearSolverType.DenseSchur);
	}

	[Test]
	public async Task DenseSchurWithEigenLargeProblem()
	{
		await ComputeAndCompareSolutions(3, false, LinearSolverType.DenseSchur);
		await ComputeAndCompareSolutions(3, true, LinearSolverType.DenseSchur);
	}

	[Test]
	public async Task DenseSchurWithEigenVaryingFBlockSize()
	{
		await ComputeAndCompareSolutions(4, true, LinearSolverType.DenseSchur);
	}

	[Test]
	public async Task SparseSchurWithEigenSparseSmallProblemAMD()
	{
		await ComputeAndCompareSolutions(2, false, LinearSolverType.SparseSchur);
		await ComputeAndCompareSolutions(2, true, LinearSolverType.SparseSchur);
	}

	[Test]
	public async Task SparseSchurWithEigenSparseSmallProblemNATURAL()
	{
		await ComputeAndCompareSolutions(2, false, LinearSolverType.SparseSchur);
		await ComputeAndCompareSolutions(2, true, LinearSolverType.SparseSchur);
	}

	[Test]
	public async Task SparseSchurWithEigenSparseLargeProblemAMD()
	{
		await ComputeAndCompareSolutions(3, false, LinearSolverType.SparseSchur);
		await ComputeAndCompareSolutions(3, true, LinearSolverType.SparseSchur);
	}

	[Test]
	public async Task SparseSchurWithEigenSparseLargeProblemNATURAL()
	{
		await ComputeAndCompareSolutions(3, false, LinearSolverType.SparseSchur);
		await ComputeAndCompareSolutions(3, true, LinearSolverType.SparseSchur);
	}
}

public class IterativeSchurComplementSolverTests
{
	private const double Epsilon = 1e-14;

	private static async Task TestSolver(LinearLeastSquaresProblem problem, double[] d, PreconditionerType preconditionerType)
	{
		int numCols = problem.A.NumCols;
		var referenceSolution = new double[numCols];
		LinearSolver.Create(LinearSolverType.DenseQr).Solve(problem.DenseSparseA(), problem.B, d, referenceSolution);

		LinearSolver isc = LinearSolver.Create(new LinearSolverOptions(
			LinearSolverType.IterativeSchur,
			preconditionerType,
			problem.NumEliminateBlocks,
			MinNumIterations: 1,
			MaxNumIterations: numCols));
		var iscSol = new double[numCols];
		isc.Solve(problem.A, problem.B, d, iscSol, qTolerance: 0.0, rTolerance: 1e-12);
		await Assert.That((new VectorXd(iscSol) - new VectorXd(referenceSolution)).Norm()).IsLessThan(Epsilon);
	}

	[Test]
	public async Task NormalProblemSchurJacobi()
	{
		LinearLeastSquaresProblem problem = LinearLeastSquaresProblems.FromId(2);
		await TestSolver(problem, [], PreconditionerType.SchurJacobi);
		await TestSolver(problem, problem.D, PreconditionerType.SchurJacobi);
	}

	[Test]
	public async Task ProblemWithNoFBlocks()
	{
		LinearLeastSquaresProblem problem = LinearLeastSquaresProblems.FromId(3);
		await TestSolver(problem, [], PreconditionerType.SchurJacobi);
		await TestSolver(problem, problem.D, PreconditionerType.SchurJacobi);
	}
}
