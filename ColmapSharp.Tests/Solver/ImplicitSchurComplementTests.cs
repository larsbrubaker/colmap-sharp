// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/implicit_schur_complement_test.cc
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ImplicitSchurComplementTests (Ceres' test, not COLMAP's):
// ColmapSharp/Solver/ImplicitSchurComplement.cs against the explicit Schur complement
// SchurEliminator.cs builds, with and without the LM diagonal, same problem (id 2) and the
// same 1e-14 absolute tolerance: every column of S, every column of the power-series
// operator Z, the reduced right-hand side and the back-substituted solution.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class ImplicitSchurComplementTests
{
	private const double Epsilon = 1e-14;

	// The explicit reduced system (full symmetric lhs), its rhs and the solution through
	// the SchurEliminator.
	private static (MatrixXd Lhs, VectorXd Rhs, VectorXd Solution) ReducedLinearSystemAndSolution(
		LinearLeastSquaresProblem problem, double[] d)
	{
		CompressedRowBlockStructure bs = problem.A.Structure;
		var blhs = new BlockRandomAccessDenseMatrix(SchurEliminator.ReducedBlocks(bs, problem.NumEliminateBlocks));
		int numSchurRows = blhs.NumRows;
		var eliminator = new SchurEliminator();
		eliminator.Init(problem.NumEliminateBlocks, bs);
		var rhs = new double[numSchurRows];
		eliminator.Eliminate(problem.A, problem.B, d, blhs, rhs);

		// lhs_ref is an upper triangular matrix. Construct a full version of lhs_ref in lhs
		// by transposing lhs_ref, choosing the strictly lower triangular part of the matrix
		// and adding it to lhs_ref.
		MatrixXd lhs = MatrixXd.FromRowMajor(numSchurRows, numSchurRows, blhs.Values);
		for (int c = 0; c < numSchurRows; c++)
		{
			for (int r = c + 1; r < numSchurRows; r++)
			{
				lhs[r, c] = lhs[c, r];
			}
		}

		int numCols = problem.A.NumCols;
		var solution = new double[numCols];
		VectorXd schurSolution = new LLT(lhs).Solve(new VectorXd(rhs));
		schurSolution.AsSpan().CopyTo(solution.AsSpan(numCols - numSchurRows));
		eliminator.BackSubstitute(problem.A, problem.B, d, schurSolution.AsSpan(), solution);
		return (lhs, new VectorXd(rhs), new VectorXd(solution));
	}

	private static async Task TestImplicitSchurComplement(LinearLeastSquaresProblem problem, double[] d)
	{
		(MatrixXd lhs, VectorXd rhs, VectorXd referenceSolution) = ReducedLinearSystemAndSolution(problem, d);

		var isc = new ImplicitSchurComplement(problem.NumEliminateBlocks, computeFtfInverse: true);
		isc.Init(problem.A, d, problem.B);

		int numCols = problem.A.NumCols;
		int numFCols = lhs.Cols;
		int numECols = numCols - numFCols;
		MatrixXd aDense = problem.DenseA();
		MatrixXd e = aDense.LeftCols(numECols);
		MatrixXd f = aDense.RightCols(numFCols);
		var de = new MatrixXd(numECols, numECols);
		var df = new MatrixXd(numFCols, numFCols);
		if (d.Length > 0)
		{
			for (int i = 0; i < numECols; i++)
			{
				de[i, i] = d[i];
			}

			for (int i = 0; i < numFCols; i++)
			{
				df[i, i] = d[numECols + i];
			}
		}

		// Z = (block_diagonal(F'F))^-1 F'E (E'E)^-1 E'F
		// Here, assuming that block_diagonal(F'F) == diagonal(F'F)
		MatrixXd ftf = f.TransposeTimesSelf() + df;
		var ftfDiagonalInverse = new MatrixXd(numFCols, numFCols);
		for (int i = 0; i < numFCols; i++)
		{
			ftfDiagonalInverse[i, i] = 1.0 / ftf[i, i];
		}

		MatrixXd eteInverse = new PartialPivLU(e.TransposeTimesSelf() + de).Inverse();
		MatrixXd zReference = ftfDiagonalInverse * f.TransposeTimes(e) * eteInverse * e.TransposeTimes(f);

		for (int i = 0; i < numFCols; i++)
		{
			VectorXd x = VectorXd.Unit(numFCols, i);
			VectorXd y = lhs * x;
			var z = new double[numFCols];
			isc.RightMultiplyAndAccumulate(x.AsSpan(), z);

			// The i^th column of the implicit schur complement is the same as the explicit
			// schur complement.
			await Assert.That((y - new VectorXd(z)).Norm()).IsLessThanOrEqualTo(Epsilon);

			y = zReference * x;
			Array.Clear(z);
			isc.InversePowerSeriesOperatorRightMultiplyAccumulate(x.AsSpan(), z);

			// The i^th column of operator Z stored implicitly is the same as its explicit
			// version.
			await Assert.That((y - new VectorXd(z)).Norm()).IsLessThanOrEqualTo(Epsilon);
		}

		// Compare the rhs of the reduced linear system
		await Assert.That((new VectorXd(isc.Rhs) - rhs).Norm()).IsLessThanOrEqualTo(Epsilon);

		// Reference solution to the f_block.
		VectorXd referenceFSol = new LLT(lhs).Solve(rhs);

		// Backsubstituted solution from the implicit schur solver using the reference
		// solution to the f_block.
		var sol = new double[numCols];
		isc.BackSubstitute(referenceFSol.AsSpan(), sol);
		await Assert.That((new VectorXd(sol) - referenceSolution).Norm()).IsLessThanOrEqualTo(Epsilon);
	}

	// Verify that the Schur Complement matrix implied by the ImplicitSchurComplement class
	// matches the one explicitly computed by the SchurComplement solver. We do this with and
	// without regularization to check that the support for the LM diagonal is correct.
	[Test]
	public async Task SchurMatrixValuesTest()
	{
		LinearLeastSquaresProblem problem = LinearLeastSquaresProblems.FromId(2);
		await TestImplicitSchurComplement(problem, []);
		await TestImplicitSchurComplement(problem, problem.D);
	}
}
