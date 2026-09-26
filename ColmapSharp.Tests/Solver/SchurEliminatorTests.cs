// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/schur_eliminator_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// SchurEliminatorTests (Ceres' test, not COLMAP's): ColmapSharp/Solver/SchurEliminator.cs
// against the reduced system and solution computed densely from H = J'J + D^2, same problems
// and the same 1e-14 relative tolerances.
// - Ceres runs each case twice, with the static block structure (template specializations)
//   and without; only the dynamic eliminator is ported, so each case runs once. Ceres'
//   VaryingFBlockSizeWithStaticStructure is ported as VaryingFBlockSizeWithoutStaticStructure:
//   the same problem, through the dynamic eliminator.
// Not ported: SchurEliminatorForOneFBlock.MatchesSchurEliminator (Ceres' fixed-size <2, 3, 6>
// specialization for a single F block is not ported; see docs/CPP_DIVERGENCES.md entry 35).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class SchurEliminatorTests
{
	private sealed record Reference(MatrixXd LhsExpected, VectorXd RhsExpected, VectorXd SolExpected);

	// Compute the golden values for the reduced linear system and the solution to the linear
	// least squares problem using dense linear algebra.
	private static Reference ComputeReferenceSolution(LinearLeastSquaresProblem problem, double[] d)
	{
		MatrixXd j = problem.DenseA();
		var f = new VectorXd(problem.B);
		int numCols = j.Cols;
		MatrixXd h = j.TransposeTimesSelf();
		for (int i = 0; i < numCols; i++)
		{
			h[i, i] += d[i] * d[i];
		}

		VectorXd g = j.TransposeTimes(f);
		CompressedRowBlockStructure bs = problem.A.Structure;
		int numEliminateCols = 0;
		for (int i = 0; i < problem.NumEliminateBlocks; i++)
		{
			numEliminateCols += bs.Cols[i].Size;
		}

		int schurSize = numCols - numEliminateCols;
		MatrixXd p = h.Block(0, 0, numEliminateCols, numEliminateCols);
		MatrixXd q = h.Block(0, numEliminateCols, numEliminateCols, schurSize);
		MatrixXd r = h.Block(numEliminateCols, numEliminateCols, schurSize, schurSize);
		int row = 0;
		for (int i = 0; i < problem.NumEliminateBlocks; i++)
		{
			int blockSize = bs.Cols[i].Size;
			MatrixXd block = p.Block(row, row, blockSize, blockSize);
			p.SetBlock(row, row, new LLT(block).Solve(MatrixXd.Identity(blockSize)));
			row += blockSize;
		}

		MatrixXd full = r - (q.TransposeTimes(p) * q);
		var lhsExpected = new MatrixXd(schurSize, schurSize);
		for (int c = 0; c < schurSize; c++)
		{
			for (int rr = 0; rr <= c; rr++)
			{
				lhsExpected[rr, c] = full[rr, c];
			}
		}

		VectorXd rhsExpected = g.Tail(schurSize) - (q.TransposeTimes(p) * g.Head(numEliminateCols));
		VectorXd solExpected = new LLT(h).Solve(g);
		return new Reference(lhsExpected, rhsExpected, solExpected);
	}

	private static double UpperFrobeniusNorm(MatrixXd m)
	{
		double sum = 0.0;
		for (int c = 0; c < m.Cols; c++)
		{
			for (int r = 0; r < m.Rows; r++)
			{
				// selfadjointView<Upper>: the strictly upper part counts twice.
				double v = r <= c ? m[r, c] : m[c, r];
				sum += v * v;
			}
		}

		return Math.Sqrt(sum);
	}

	private static async Task EliminateSolveAndCompare(
		LinearLeastSquaresProblem problem, double[] diagonal, Reference reference, double relativeTolerance)
	{
		CompressedRowBlockStructure bs = problem.A.Structure;
		var lhs = new BlockRandomAccessDenseMatrix(SchurEliminator.ReducedBlocks(bs, problem.NumEliminateBlocks));
		int numCols = problem.A.NumCols;
		int schurSize = lhs.NumRows;
		var rhs = new double[schurSize];

		var eliminator = new SchurEliminator();
		eliminator.Init(problem.NumEliminateBlocks, bs);
		eliminator.Eliminate(problem.A, problem.B, diagonal, lhs, rhs);

		// The row-major upper triangle as a column-major matrix is the lower triangle; the
		// transpose puts it back in the upper triangle.
		MatrixXd lhsRef = MatrixXd.FromRowMajor(schurSize, schurSize, lhs.Values);
		VectorXd reducedSol = new LLT(MatrixXd.FromColumnMajor(schurSize, schurSize, lhs.Values)).Solve(new VectorXd(rhs));

		// Solution to the linear least squares problem.
		var sol = new double[numCols];
		reducedSol.AsSpan().CopyTo(sol.AsSpan(numCols - schurSize));
		eliminator.BackSubstitute(problem.A, problem.B, diagonal, reducedSol.AsSpan(), sol);

		// lhs_expected holds only the upper triangle, so its norm is over that.
		double diff = UpperFrobeniusNorm(lhsRef - reference.LhsExpected);
		double expectedNorm = Math.Sqrt(reference.LhsExpected.AsSpan().ToArray().Sum(v => v * v));
		await Assert.That(diff / expectedNorm).IsEqualTo(0.0).Within(relativeTolerance);
		await Assert.That((new VectorXd(rhs) - reference.RhsExpected).Norm() / reference.RhsExpected.Norm())
			.IsEqualTo(0.0).Within(relativeTolerance);
		await Assert.That((new VectorXd(sol) - reference.SolExpected).Norm() / reference.SolExpected.Norm())
			.IsEqualTo(0.0).Within(relativeTolerance);
	}

	[Test]
	public async Task ScalarProblemNoRegularization()
	{
		LinearLeastSquaresProblem problem = LinearLeastSquaresProblems.FromId(2);
		var zero = new double[problem.A.NumCols];
		Reference reference = ComputeReferenceSolution(problem, zero);
		await EliminateSolveAndCompare(problem, zero, reference, 1e-14);
	}

	[Test]
	public async Task ScalarProblemWithRegularization()
	{
		LinearLeastSquaresProblem problem = LinearLeastSquaresProblems.FromId(2);
		Reference reference = ComputeReferenceSolution(problem, problem.D);
		await EliminateSolveAndCompare(problem, problem.D, reference, 1e-14);
	}

	[Test]
	public async Task VaryingFBlockSizeWithoutStaticStructure()
	{
		LinearLeastSquaresProblem problem = LinearLeastSquaresProblems.FromId(4);
		Reference reference = ComputeReferenceSolution(problem, problem.D);
		await EliminateSolveAndCompare(problem, problem.D, reference, 1e-14);
	}
}
