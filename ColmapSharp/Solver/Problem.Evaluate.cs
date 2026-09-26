// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 include/ceres/problem.h (Problem::EvaluateOptions),
// include/ceres/crs_matrix.h and internal/ceres/problem_impl.cc (ProblemImpl::Evaluate)
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Problem::Evaluate: cost, residuals, gradient and Jacobian of the problem at the values in
// the user's arrays, without solving. COLMAP calls it twice: covariance estimation takes the
// Jacobian (CRS) over a chosen list of parameter blocks, and view-graph calibration takes the
// residuals without the loss function. The evaluation runs on a temporary program over the
// chosen residual and parameter blocks (all by default, in problem order); blocks left out
// of an explicit list are held constant for the call, so they get no Jacobian columns.
// Constant blocks in the list keep their (zero) columns, as in Ceres. ProgramEvaluator.cs does
// the work with the block-sparse Jacobian, converted to CRS row by row with each row's columns
// ascending, the layout Ceres' CompressedRowJacobianWriter produces.

namespace ColmapSharp.Solver;

/// <summary>ceres::Problem::EvaluateOptions.</summary>
public sealed class ProblemEvaluateOptions
{
	/// <summary>
	/// The parameter blocks, in Jacobian column order. Empty means every block of the problem
	/// in insertion order; otherwise the blocks not listed are treated as constant.
	/// </summary>
	public List<ArraySegment<double>> ParameterBlocks { get; } = [];

	/// <summary>The residual blocks, in residual order. Empty means every one, in insertion order.</summary>
	public List<ResidualBlockId> ResidualBlocks { get; } = [];

	/// <summary>Apply each residual block's loss function (default true).</summary>
	public bool ApplyLossFunction { get; set; } = true;

	/// <summary>Threads for the evaluation; the result does not depend on it.</summary>
	public int NumThreads { get; set; } = 1;
}

/// <summary>ceres::CRSMatrix: a compressed row sparse matrix.</summary>
public sealed class CRSMatrix
{
	/// <summary>Number of rows.</summary>
	public int NumRows { get; set; }

	/// <summary>Number of columns.</summary>
	public int NumCols { get; set; }

	/// <summary>Row i's entries are Cols/Values[Rows[i] .. Rows[i + 1]); NumRows + 1 long.</summary>
	public List<int> Rows { get; } = [];

	/// <summary>Column index of each entry.</summary>
	public List<int> Cols { get; } = [];

	/// <summary>Value of each entry.</summary>
	public List<double> Values { get; } = [];
}

/// <content>Problem::Evaluate.</content>
public sealed partial class Problem
{
	/// <summary>
	/// Problem::Evaluate: evaluates at the current values of the parameter blocks. Each of
	/// <paramref name="residuals"/>, <paramref name="gradient"/> and
	/// <paramref name="jacobian"/> is optional and is resized and filled; the gradient and the
	/// Jacobian's columns follow the tangent sizes of the chosen parameter blocks. Returns
	/// false if a residual block fails to evaluate or the cost is not finite, in which case
	/// <paramref name="cost"/> is 0 and the Jacobian is left untouched.
	/// </summary>
	public bool Evaluate(
		ProblemEvaluateOptions options,
		out double cost,
		List<double>? residuals,
		List<double>? gradient,
		CRSMatrix? jacobian)
	{
		ArgumentNullException.ThrowIfNull(options);
		cost = 0.0;

		// If the user supplied residual blocks, then use them, otherwise take the residual
		// blocks from the underlying program.
		var program = new Program();
		if (options.ResidualBlocks.Count > 0)
		{
			foreach (ResidualBlockId id in options.ResidualBlocks)
			{
				program.ResidualBlocks.Add(id.Block);
			}
		}
		else
		{
			program.ResidualBlocks.AddRange(Program.ResidualBlocks);
		}

		var variableParameterBlocks = new List<ParameterBlock>();
		if (options.ParameterBlocks.Count == 0)
		{
			// The user did not provide any parameter blocks, so default to using all the
			// parameter blocks in the order that they are in the underlying program object.
			program.ParameterBlocks.AddRange(Program.ParameterBlocks);
		}
		else
		{
			var included = new HashSet<ParameterBlock>(ReferenceEqualityComparer.Instance);
			for (int i = 0; i < options.ParameterBlocks.Count; i++)
			{
				ParameterBlock block = Find(options.ParameterBlocks[i])
					?? throw new ArgumentException(
						$"No known parameter block for Problem::Evaluate::Options.parameter_blocks[{i}]");
				program.ParameterBlocks.Add(block);
				included.Add(block);
			}

			// The user may have only supplied a subset of parameter blocks, so the variable
			// ones not supplied are made constant during the evaluation (so that they get no
			// Jacobian columns) and variable again afterwards.
			foreach (ParameterBlock block in Program.ParameterBlocks)
			{
				if (!included.Contains(block) && !block.IsConstant)
				{
					variableParameterBlocks.Add(block);
					block.IsSetConstant = true;
				}
			}
		}

		try
		{
			// Setup the Parameter indices and offsets before an evaluator can be constructed
			// and used.
			program.SetParameterOffsetsAndIndex();
			var evaluator = new ProgramEvaluator(program, denseJacobian: false, numEliminateBlocks: 0, Math.Max(1, options.NumThreads));
			double[]? residualValues = residuals is null ? null : new double[evaluator.NumResiduals];
			double[]? gradientValues = gradient is null ? null : new double[evaluator.NumEffectiveParameters];
			BlockSparseMatrix? blockJacobian = jacobian is null ? null : (BlockSparseMatrix)evaluator.CreateJacobian();

			// Point the state at the user's memory, so the state vector holds the current values.
			program.SetParameterBlockStatePtrsToUserStatePtrs();
			var parameters = new double[program.NumParameters];
			program.ParameterBlocksToStateVector(parameters);

			bool status = evaluator.Evaluate(
				parameters, out double evaluatedCost, residualValues, gradientValues, blockJacobian, options.ApplyLossFunction);

			// Ceres evaluates straight into the caller's vectors, so they are filled whatever
			// the status.
			CopyInto(residualValues, residuals);
			CopyInto(gradientValues, gradient);
			if (status)
			{
				cost = evaluatedCost;
				if (jacobian is not null)
				{
					ToCrsMatrix(blockJacobian!, jacobian);
				}
			}

			return status;
		}
		finally
		{
			// Make the parameter blocks that were temporarily marked constant, variable again.
			foreach (ParameterBlock block in variableParameterBlocks)
			{
				block.IsSetConstant = false;
			}

			Program.SetParameterBlockStatePtrsToUserStatePtrs();
			Program.SetParameterOffsetsAndIndex();
		}
	}

	private static void CopyInto(double[]? values, List<double>? destination)
	{
		if (values is null || destination is null)
		{
			return;
		}

		destination.Clear();
		destination.AddRange(values);
	}

	// BlockSparseMatrix -> CRSMatrix, rows in order, each row's cells by increasing column
	// block (the structure keeps them sorted) and each cell's columns in order.
	private static void ToCrsMatrix(BlockSparseMatrix matrix, CRSMatrix crs)
	{
		crs.NumRows = matrix.NumRows;
		crs.NumCols = matrix.NumCols;
		crs.Rows.Clear();
		crs.Cols.Clear();
		crs.Values.Clear();
		crs.Rows.Add(0);
		CompressedRowBlockStructure structure = matrix.Structure;
		double[] values = matrix.Values;
		foreach (CompressedRow row in structure.Rows)
		{
			for (int r = 0; r < row.Block.Size; r++)
			{
				foreach (Cell cell in row.Cells)
				{
					Block col = structure.Cols[cell.BlockId];
					int start = cell.Position + (r * col.Size);
					for (int c = 0; c < col.Size; c++)
					{
						crs.Cols.Add(col.Position + c);
						crs.Values.Add(values[start + c]);
					}
				}

				crs.Rows.Add(crs.Cols.Count);
			}
		}
	}
}
