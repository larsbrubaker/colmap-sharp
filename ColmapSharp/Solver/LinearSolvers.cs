// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/linear_solver.h/.cc,
// internal/ceres/dense_qr_solver.cc, internal/ceres/dense_normal_cholesky_solver.cc,
// internal/ceres/sparse_normal_cholesky_solver.cc, internal/ceres/dense_qr.cc and
// internal/ceres/dense_cholesky.cc (Eigen back ends) (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// The linear solvers of the Levenberg-Marquardt step: find x minimizing
// |A x - b|^2 + |D x|^2 for the (scaled) Jacobian A, residuals b and LM diagonal D.
// - DENSE_QR: QR of the augmented [A; diag(D)] (LinearAlgebra/HouseholderQR.cs, where Ceres
//   uses Eigen's HouseholderQR).
// - DENSE_NORMAL_CHOLESKY: dense LLT of A'A + D^2 (LinearAlgebra/LLT.cs, Eigen's LLT).
// - SPARSE_NORMAL_CHOLESKY: A'A + D^2 formed from BlockSparseMatrix's block structure and
//   factored by LinearAlgebra/SimplicialCholesky.cs (LLT, AMD ordering), the stand-in for
//   Ceres' SuiteSparse/Eigen sparse Cholesky (divergences 13 and 22).
// The Schur solvers (DENSE_SCHUR, SPARSE_SCHUR: SchurComplementSolvers.cs; ITERATIVE_SCHUR:
// IterativeSchurSolver.cs) are created here too; they need the Jacobian's first
// num_eliminate_blocks column blocks to be the E blocks (SchurOrdering.cs).

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Solver;

/// <summary>ceres::LinearSolverType (all but CGNR, which COLMAP never selects).</summary>
public enum LinearSolverType
{
	/// <summary>Dense QR of the Jacobian; small problems, best accuracy.</summary>
	DenseQr,

	/// <summary>Dense Cholesky of the normal equations.</summary>
	DenseNormalCholesky,

	/// <summary>Sparse Cholesky of the normal equations.</summary>
	SparseNormalCholesky,

	/// <summary>Eliminate the E blocks (points), dense Cholesky of the reduced camera matrix.</summary>
	DenseSchur,

	/// <summary>Eliminate the E blocks, sparse Cholesky of the reduced camera matrix.</summary>
	SparseSchur,

	/// <summary>Preconditioned conjugate gradients on the implicit Schur complement.</summary>
	IterativeSchur,
}

/// <summary>
/// The LinearSolver::Options that choose and configure a solver: the Schur solvers eliminate
/// the first <paramref name="NumEliminateBlocks"/> column blocks (Ceres' elimination_groups[0]).
/// <paramref name="NumThreads"/> is Ceres' num_threads; the Schur solvers give the same result
/// for every value.
/// </summary>
internal readonly record struct LinearSolverOptions(
	LinearSolverType Type,
	PreconditionerType PreconditionerType = PreconditionerType.Jacobi,
	int NumEliminateBlocks = 0,
	int MinNumIterations = 1,
	int MaxNumIterations = 1,
	int NumThreads = 1);

/// <summary>ceres::internal::LinearSolverTerminationType.</summary>
internal enum LinearSolverTerminationType
{
	/// <summary>Solved.</summary>
	Success,

	/// <summary>An iterative solver ran out of iterations.</summary>
	NoConvergence,

	/// <summary>Numerical failure (e.g. not positive definite); the minimizer treats the step as invalid.</summary>
	Failure,

	/// <summary>Unrecoverable; the solve stops.</summary>
	FatalError,
}

/// <summary>ceres::internal::LinearSolver::Summary.</summary>
internal readonly record struct LinearSolverSummary(
	LinearSolverTerminationType TerminationType, int NumIterations, string Message);

/// <summary>ceres::internal::LinearSolver.</summary>
internal abstract class LinearSolver
{
	/// <summary>Number of Solve calls.</summary>
	public int NumSolves { get; private set; }

	/// <summary>Creates the solver for a type with default options (the direct solvers).</summary>
	public static LinearSolver Create(LinearSolverType type) => Create(new LinearSolverOptions(type));

	/// <summary>LinearSolver::Create.</summary>
	public static LinearSolver Create(LinearSolverOptions options) => options.Type switch
	{
		LinearSolverType.DenseQr => new DenseQrSolver(),
		LinearSolverType.DenseNormalCholesky => new DenseNormalCholeskySolver(),
		LinearSolverType.SparseNormalCholesky => new SparseNormalCholeskySolver(),
		LinearSolverType.DenseSchur => new DenseSchurComplementSolver(options.NumEliminateBlocks, options.NumThreads),
		LinearSolverType.SparseSchur => new SparseSchurComplementSolver(options.NumEliminateBlocks, options.NumThreads),
		LinearSolverType.IterativeSchur => new IterativeSchurComplementSolver(
			options.NumEliminateBlocks, options.PreconditionerType, options.MinNumIterations, options.MaxNumIterations, options.NumThreads),
		_ => throw new ArgumentOutOfRangeException(nameof(options), options.Type, "Unknown linear solver type."),
	};

	/// <summary>True for the solvers that eliminate E blocks (ceres::IsSchurType).</summary>
	public static bool IsSchurType(LinearSolverType type) =>
		type is LinearSolverType.DenseSchur or LinearSolverType.SparseSchur or LinearSolverType.IterativeSchur;

	/// <summary>
	/// Solves min |A x - b|^2 + |D x|^2 (D may be empty for no regularization) into
	/// <paramref name="x"/>. The tolerances are LinearSolver::PerSolveOptions' q_tolerance
	/// and r_tolerance, which only the iterative solver reads.
	/// </summary>
	public LinearSolverSummary Solve(
		SparseMatrix a, ReadOnlySpan<double> b, ReadOnlySpan<double> d, Span<double> x, double qTolerance = 0.0, double rTolerance = 0.0)
	{
		NumSolves++;
		return SolveImpl(a, b, d, x, qTolerance, rTolerance);
	}

	/// <summary>The solver-specific work.</summary>
	protected abstract LinearSolverSummary SolveImpl(
		SparseMatrix a, ReadOnlySpan<double> b, ReadOnlySpan<double> d, Span<double> x, double qTolerance, double rTolerance);
}

/// <summary>ceres::internal::DenseQRSolver.</summary>
internal sealed class DenseQrSolver : LinearSolver
{
	// Reused across iterations while the shape stays the same (it does for a whole solve).
	private MatrixXd? lhs;
	private VectorXd? rhs;

	/// <inheritdoc/>
	protected override LinearSolverSummary SolveImpl(
		SparseMatrix a, ReadOnlySpan<double> b, ReadOnlySpan<double> d, Span<double> x, double qTolerance, double rTolerance)
	{
		MatrixXd matrix = ((DenseSparseMatrix)a).Matrix;
		int numRows = matrix.Rows;
		int numCols = matrix.Cols;
		int numAugmentedRows = numRows + (d.IsEmpty ? 0 : numCols);
		if (lhs is null || lhs.Rows != numAugmentedRows || lhs.Cols != numCols)
		{
			lhs = new MatrixXd(numAugmentedRows, numCols);
			rhs = new VectorXd(numAugmentedRows);
		}

		lhs.AsSpan().Clear();
		rhs!.AsSpan().Clear();
		for (int c = 0; c < numCols; c++)
		{
			matrix.ColumnSpan(c).CopyTo(lhs.ColumnSpan(c));
			if (!d.IsEmpty)
			{
				lhs[numRows + c, c] = d[c];
			}
		}

		b[..numRows].CopyTo(rhs.AsSpan());
		new HouseholderQR(lhs).Solve(rhs).AsSpan().CopyTo(x);
		return new LinearSolverSummary(LinearSolverTerminationType.Success, 1, "Success.");
	}
}

/// <summary>ceres::internal::DenseNormalCholeskySolver.</summary>
internal sealed class DenseNormalCholeskySolver : LinearSolver
{
	// Reused across iterations while the shape stays the same.
	private MatrixXd? lhs;
	private VectorXd? rhs;

	/// <inheritdoc/>
	protected override LinearSolverSummary SolveImpl(
		SparseMatrix a, ReadOnlySpan<double> b, ReadOnlySpan<double> d, Span<double> x, double qTolerance, double rTolerance)
	{
		MatrixXd matrix = ((DenseSparseMatrix)a).Matrix;
		int numRows = matrix.Rows;
		int numCols = matrix.Cols;
		if (lhs is null || lhs.Cols != numCols)
		{
			lhs = new MatrixXd(numCols, numCols);
			rhs = new VectorXd(numCols);
		}

		for (int j = 0; j < numCols; j++)
		{
			ReadOnlySpan<double> colJ = matrix.ColumnSpan(j);
			for (int i = j; i < numCols; i++)
			{
				ReadOnlySpan<double> colI = matrix.ColumnSpan(i);
				double sum = 0.0;
				for (int r = 0; r < numRows; r++)
				{
					sum += colI[r] * colJ[r];
				}

				lhs[i, j] = sum;
				lhs[j, i] = sum;
			}

			double dot = 0.0;
			for (int r = 0; r < numRows; r++)
			{
				dot += colJ[r] * b[r];
			}

			rhs![j] = dot;
			if (!d.IsEmpty)
			{
				lhs[j, j] += d[j] * d[j];
			}
		}

		var llt = new LLT(lhs);
		if (llt.Info != ComputationInfo.Success)
		{
			return new LinearSolverSummary(
				LinearSolverTerminationType.Failure, 1, "Eigen failure. Unable to perform dense Cholesky factorization.");
		}

		llt.Solve(rhs!).AsSpan().CopyTo(x);
		return new LinearSolverSummary(LinearSolverTerminationType.Success, 1, "Success.");
	}
}

/// <summary>
/// ceres::internal::SparseNormalCholeskySolver: A'A + D^2 in lower-triangle CSC form, built
/// from the Jacobian's block structure (Ceres' InnerProductComputer), factored by a
/// simplicial LLT. The pattern, the CSC matrix, its symbolic analysis and the value offset
/// of every cell pair are computed once per Jacobian structure; an iteration only
/// accumulates values and factors.
/// </summary>
internal sealed class SparseNormalCholeskySolver : LinearSolver
{
	private readonly SimplicialCholesky cholesky = new(SimplicialCholeskyKind.LLT, SparseOrdering.Amd, SymmetricPart.Lower);
	private CompressedRowBlockStructure? structure;
	private SparseMatrixCsc? lhs;
	private int[] colPtr = [];
	private VectorXd? rhs;

	// For every row block, every cell pair (p, q <= p) in order: the offset such that entry
	// (row m of the column block I of cell p, column k of the column block J of cell q) is at
	// colPtr[J.Position + k] + pairOffset - k + m; see BuildPattern.
	private int[] pairOffsets = [];

	/// <inheritdoc/>
	protected override LinearSolverSummary SolveImpl(
		SparseMatrix a, ReadOnlySpan<double> b, ReadOnlySpan<double> d, Span<double> x, double qTolerance, double rTolerance)
	{
		var matrix = (BlockSparseMatrix)a;
		int numCols = matrix.NumCols;
		if (!ReferenceEquals(structure, matrix.Structure))
		{
			BuildPattern(matrix.Structure);
			cholesky.AnalyzePattern(lhs!);
			rhs = new VectorXd(numCols);
		}

		rhs!.AsSpan().Clear();
		matrix.LeftMultiplyAndAccumulate(b, rhs.AsSpan());
		ComputeInnerProduct(matrix, d);
		if (!cholesky.Factorize(lhs!))
		{
			return new LinearSolverSummary(
				LinearSolverTerminationType.Failure, 1, "Eigen failure. Unable to find numeric factorization.");
		}

		cholesky.Solve(rhs).AsSpan().CopyTo(x);
		return new LinearSolverSummary(LinearSolverTerminationType.Success, 1, "Success.");
	}

	// The lower triangle of A'A has a block (I, J), I >= J, wherever column blocks I and J
	// share a row block; the diagonal blocks are always present (the LM diagonal D lands on
	// them). Column c of block J stores the rows of J from c down, then every partner I > J
	// in increasing order: row indices come out sorted because column block positions grow
	// with the block index.
	private void BuildPattern(CompressedRowBlockStructure bs)
	{
		structure = bs;
		Block[] cols = bs.Cols;
		var partners = new SortedSet<int>[cols.Length];
		for (int j = 0; j < cols.Length; j++)
		{
			partners[j] = [];
		}

		int numPairs = 0;
		foreach (CompressedRow row in bs.Rows)
		{
			Cell[] cells = row.Cells;
			numPairs += cells.Length * (cells.Length + 1) / 2;
			for (int p = 0; p < cells.Length; p++)
			{
				for (int q = 0; q < p; q++)
				{
					// Cells are sorted by block id, so cells[p] is the lower block.
					partners[cells[q].BlockId].Add(cells[p].BlockId);
				}
			}
		}

		int n = 0;
		foreach (Block col in cols)
		{
			n += col.Size;
		}

		// prefix[(J, I)]: rows of the partners of J that come before I in a column of J.
		var prefix = new Dictionary<(int J, int I), int>();
		colPtr = new int[n + 1];
		var rows = new List<int>();
		for (int j = 0; j < cols.Length; j++)
		{
			int offset = 0;
			foreach (int i in partners[j])
			{
				prefix[(j, i)] = offset;
				offset += cols[i].Size;
			}

			for (int k = 0; k < cols[j].Size; k++)
			{
				int c = cols[j].Position + k;
				for (int m = k; m < cols[j].Size; m++)
				{
					rows.Add(cols[j].Position + m);
				}

				foreach (int i in partners[j])
				{
					for (int m = 0; m < cols[i].Size; m++)
					{
						rows.Add(cols[i].Position + m);
					}
				}

				colPtr[c + 1] = rows.Count;
			}
		}

		// Entry (m, k) of pair (I, J) sits at colPtr[c] + m - k for I == J (the diagonal
		// block starts at row c), and at colPtr[c] + (size_J - k) + prefix + m for I > J.
		// Both are colPtr[c] + pairOffset - k + m, with pairOffset 0 or size_J + prefix, so
		// one int per pair carries the whole placement.
		pairOffsets = new int[numPairs];
		int pair = 0;
		foreach (CompressedRow row in bs.Rows)
		{
			Cell[] cells = row.Cells;
			for (int p = 0; p < cells.Length; p++)
			{
				for (int q = 0; q <= p; q++)
				{
					int i = cells[p].BlockId;
					int j = cells[q].BlockId;
					pairOffsets[pair++] = i == j ? 0 : cols[j].Size + prefix[(j, i)];
				}
			}
		}

		int[] rowIndices = [.. rows];
		lhs = SparseMatrixCsc.FromCsc(n, n, colPtr, rowIndices, new double[rowIndices.Length]);
	}

	// values = lower triangle of A'A + D^2, accumulated row block by row block in order, so
	// the sums are the same on every run.
	private void ComputeInnerProduct(BlockSparseMatrix matrix, ReadOnlySpan<double> d)
	{
		Span<double> values = lhs!.Values;
		values.Clear();
		Block[] cols = structure!.Cols;
		double[] a = matrix.Values;
		int pair = 0;
		foreach (CompressedRow row in structure.Rows)
		{
			int rowSize = row.Block.Size;
			Cell[] cells = row.Cells;
			for (int p = 0; p < cells.Length; p++)
			{
				for (int q = 0; q <= p; q++, pair++)
				{
					Cell cellI = cells[p];
					Cell cellJ = cells[q];
					bool diagonal = cellI.BlockId == cellJ.BlockId;
					int sizeI = cols[cellI.BlockId].Size;
					int sizeJ = cols[cellJ.BlockId].Size;
					int positionJ = cols[cellJ.BlockId].Position;
					int pairOffset = pairOffsets[pair];
					for (int k = 0; k < sizeJ; k++)
					{
						int columnStart = colPtr[positionJ + k] + pairOffset - k;
						for (int m = diagonal ? k : 0; m < sizeI; m++)
						{
							double sum = 0.0;
							for (int r = 0; r < rowSize; r++)
							{
								sum += a[cellI.Position + (r * sizeI) + m] * a[cellJ.Position + (r * sizeJ) + k];
							}

							values[columnStart + m] += sum;
						}
					}
				}
			}
		}

		if (!d.IsEmpty)
		{
			for (int c = 0; c < colPtr.Length - 1; c++)
			{
				values[colPtr[c]] += d[c] * d[c];
			}
		}
	}
}
