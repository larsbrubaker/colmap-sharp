// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Replaces Ceres Solver 2.2.0 include/ceres/covariance.h and internal/ceres/covariance_impl.cc
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md) for the one way COLMAP uses it: default
// Covariance::Options, Compute(parameter_blocks, problem), then
// GetCovarianceMatrixInTangentSpace(parameter_blocks, out). Callers: Estimators/PoseEstimation.cs
// (RefineAbsolutePose) and Estimators/GeneralizedPoseEstimation.cs
// (RefineGeneralizedAbsolutePose). Tests: ColmapSharp.Tests/Solver/CovarianceTests.cs.
//
// Ceres' default algorithm is SPARSE_QR: evaluate the tangent-space Jacobian J of the whole
// problem (loss function applied; constant blocks and blocks no residual uses have no columns), QR-factor it with a
// column-pivoting sparse QR, fail if it is rank deficient, and read the covariance
// (J'J)^-1 = P R^-1 R^-T P' column by column. Here the Jacobian is dense and factored with
// the column-pivoting Householder QR of LinearAlgebra/ColPivHouseholderQR.cs (Eigen's sparse
// QR is MPL-2.0 and not ported). The result is the same matrix up to round-off
// (docs/CPP_DIVERGENCES.md, entry 45). Covariance of a constant block is zero, as in Ceres.
// Dense is fine for COLMAP's callers: a single pose (plus a camera) against its observations.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>
/// ceres::Covariance with default options (SPARSE_QR, loss function applied), for the
/// tangent-space covariance of a few parameter blocks.
/// </summary>
public sealed class Covariance
{
	private Problem? problem;
	private MatrixXd? fullCovariance;
	private readonly Dictionary<(double[] Array, int Offset), int> columnOffsets = [];

	/// <summary>
	/// Covariance::Compute: computes the covariance of the problem's variable blocks at their
	/// current values. Returns false if a residual block fails to evaluate or the Jacobian is
	/// rank deficient (the covariance is then undefined).
	/// </summary>
	public bool Compute(IReadOnlyList<ArraySegment<double>> parameterBlocks, Problem problem)
	{
		ArgumentNullException.ThrowIfNull(problem);
		for (int i = 0; i < parameterBlocks.Count; i++)
		{
			Check.That(problem.HasParameterBlock(parameterBlocks[i]), "Covariance::Compute called with a block not in the problem.");
			for (int j = 0; j < i; j++)
			{
				if (ReferenceEquals(parameterBlocks[i].Array, parameterBlocks[j].Array)
					&& parameterBlocks[i].Offset == parameterBlocks[j].Offset)
				{
					throw new InvalidOperationException(
						$"Covariance::Compute called with duplicate blocks at indices ({j}, {i})");
				}
			}
		}

		this.problem = problem;
		fullCovariance = null;
		columnOffsets.Clear();

		Program program = problem.Program;
		program.SetParameterOffsetsAndIndex();
		if (!program.SetParameterBlockStatePtrsToUserStatePtrs())
		{
			return false;
		}

		// Columns: the variable blocks some residual block uses, in problem order. As in Ceres'
		// ComputeCovarianceSparsity, constant blocks and blocks no residual uses are treated as
		// constant (zero covariance); an unused block would otherwise be an all-zero column
		// and make the Jacobian rank deficient.
		var blocksInUse = new HashSet<ParameterBlock>(ReferenceEqualityComparer.Instance);
		foreach (ResidualBlock residualBlock in program.ResidualBlocks)
		{
			blocksInUse.UnionWith(residualBlock.ParameterBlocks);
		}

		int numCols = 0;
		foreach (ParameterBlock block in program.ParameterBlocks)
		{
			if (!block.IsConstant && blocksInUse.Contains(block))
			{
				columnOffsets[(block.UserState.Array!, block.UserState.Offset)] = numCols;
				numCols += block.TangentSize;
			}
		}

		if (numCols == 0)
		{
			// Nothing to do, all zeros covariance matrix.
			return true;
		}

		MatrixXd? jacobian = EvaluateJacobian(program, numCols);
		if (jacobian is null)
		{
			return false;
		}

		var qr = new ColPivHouseholderQR(jacobian);
		if (qr.Rank() < numCols)
		{
			return false;
		}

		// R is n x n upper triangular with J P = Q R, so (J'J)^-1 = P (R'R)^-1 P'. Column i of
		// (R'R)^-1 solves R' y = e_i, then R x = y.
		MatrixXd r = qr.MatrixR();
		int[] permutation = qr.ColsPermutationIndices();
		var rtrInverse = new MatrixXd(numCols, numCols);
		var solution = new double[numCols];
		for (int i = 0; i < numCols; i++)
		{
			Array.Clear(solution);
			solution[i] = 1.0;
			for (int row = 0; row < numCols; row++)
			{
				double sum = solution[row];
				for (int k = 0; k < row; k++)
				{
					sum -= r[k, row] * solution[k];
				}

				solution[row] = sum / r[row, row];
			}

			for (int row = numCols - 1; row >= 0; row--)
			{
				double sum = solution[row];
				for (int k = row + 1; k < numCols; k++)
				{
					sum -= r[row, k] * solution[k];
				}

				solution[row] = sum / r[row, row];
			}

			for (int row = 0; row < numCols; row++)
			{
				rtrInverse[row, i] = solution[row];
			}
		}

		// Column j of J P is column permutation[j] of J.
		fullCovariance = new MatrixXd(numCols, numCols);
		for (int i = 0; i < numCols; i++)
		{
			for (int j = 0; j < numCols; j++)
			{
				fullCovariance[permutation[i], permutation[j]] = rtrInverse[i, j];
			}
		}

		return true;
	}

	/// <summary>
	/// Covariance::GetCovarianceMatrixInTangentSpace: the joint covariance of
	/// <paramref name="parameterBlocks"/>, block by block in the given order, as a dense
	/// (sum of tangent sizes)^2 matrix.
	/// </summary>
	public MatrixXd GetCovarianceMatrixInTangentSpace(IReadOnlyList<ArraySegment<double>> parameterBlocks)
	{
		Check.That(problem is not null, "Covariance::GetCovarianceMatrix called before Covariance::Compute");
		var sizes = new int[parameterBlocks.Count];
		var starts = new int[parameterBlocks.Count + 1];
		for (int i = 0; i < sizes.Length; i++)
		{
			sizes[i] = problem!.ParameterBlockTangentSize(parameterBlocks[i]);
			starts[i + 1] = starts[i] + sizes[i];
		}

		var result = new MatrixXd(starts[^1], starts[^1]);
		for (int i = 0; i < sizes.Length; i++)
		{
			if (!TryColumn(parameterBlocks[i], out int ci))
			{
				continue;
			}

			for (int j = 0; j < sizes.Length; j++)
			{
				if (!TryColumn(parameterBlocks[j], out int cj))
				{
					continue;
				}

				for (int a = 0; a < sizes[i]; a++)
				{
					for (int b = 0; b < sizes[j]; b++)
					{
						result[starts[i] + a, starts[j] + b] = fullCovariance![ci + a, cj + b];
					}
				}
			}
		}

		return result;
	}

	private bool TryColumn(ArraySegment<double> block, out int column)
	{
		column = 0;
		return fullCovariance is not null && columnOffsets.TryGetValue((block.Array!, block.Offset), out column);
	}

	// Problem::Evaluate's Jacobian (dense, loss function applied) over the variable blocks.
	private MatrixXd? EvaluateJacobian(Program program, int numCols)
	{
		var jacobian = new MatrixXd(program.NumResiduals, numCols);
		int rowOffset = 0;
		foreach (ResidualBlock residualBlock in program.ResidualBlocks)
		{
			int numResiduals = residualBlock.NumResiduals;
			int numBlocks = residualBlock.NumParameterBlocks;
			var views = new ArraySegment<double>[numBlocks];
			for (int i = 0; i < numBlocks; i++)
			{
				ParameterBlock block = residualBlock.ParameterBlocks[i];
				if (!block.IsConstant)
				{
					// Every variable block of a residual block is in use, so it has a column.
					views[i] = new ArraySegment<double>(new double[numResiduals * block.TangentSize]);
				}
			}

			bool anyVariable = Array.Exists(views, v => v.Array is not null);
			var residuals = new double[numResiduals];
			var scratch = new double[residualBlock.NumScratchDoublesForEvaluate()];
			if (!residualBlock.Evaluate(
				true,
				out _,
				residuals,
				anyVariable ? views : ReadOnlySpan<ArraySegment<double>>.Empty,
				scratch,
				new ArraySegment<double>[numBlocks],
				new ArraySegment<double>[numBlocks]))
			{
				return null;
			}

			for (int i = 0; i < numBlocks; i++)
			{
				if (views[i].Array is null)
				{
					continue;
				}

				ParameterBlock block = residualBlock.ParameterBlocks[i];
				int col = columnOffsets[(block.UserState.Array!, block.UserState.Offset)];
				int tangent = block.TangentSize;
				for (int r = 0; r < numResiduals; r++)
				{
					for (int c = 0; c < tangent; c++)
					{
						jacobian[rowOffset + r, col + c] = views[i][r * tangent + c];
					}
				}
			}

			rowOffset += numResiduals;
		}

		return jacobian;
	}
}
