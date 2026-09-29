// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RotationAveragingSolver: the solver half of colmap/estimators/rotation_averaging_impl.h
// and .cc. Solves a RotationAveragingProblem (RotationAveragingProblem.cs) by L1 regression
// (Optim/LeastAbsoluteDeviations.cs, ADMM) followed by iteratively reweighted least squares
// over the normal equations (Optim/SparseCholesky.cs). Tests:
// ColmapSharp.Tests/Estimators/RotationAveragingTests.cs.
//
// Tier C (iterative).
//
// Translation notes:
// - COLMAP's L1 phase doubles l1_solver_options.max_num_iterations (capped at 100) after each
//   non-converged iteration; its LeastAbsoluteDeviationSolver holds a reference to those
//   options, so the next solve uses the raised cap. The C# solver copies its options, so the
//   cap is raised through LeastAbsoluteDeviationSolver.MaxNumIterations.
// - SupernodalCholmodLLT selects the managed LLT-with-LDLT-fallback solver
//   (divergence 13), as does the IRLS phase.
// - LOG(ERROR) messages go to Util/Log.cs; VLOG messages are dropped, like LOG(INFO).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <summary>
/// Solves the rotation averaging problem using L1 regression followed by IRLS.
/// Port of colmap::RotationAveragingSolver.
/// </summary>
public sealed class RotationAveragingSolver
{
	private readonly RotationEstimatorOptions _options;

	/// <summary>A solver with a copy of <paramref name="options"/>.</summary>
	public RotationAveragingSolver(RotationEstimatorOptions options)
	{
		_options = options.Clone();
	}

	/// <summary>Solves the rotation averaging problem; false if a linear solve failed.</summary>
	public bool Solve(RotationAveragingProblem problem)
	{
		if (_options.MaxNumL1Iterations > 0 && !SolveL1Regression(problem))
		{
			return false;
		}

		if (_options.MaxNumIrlsIterations > 0 && !SolveIrls(problem))
		{
			return false;
		}

		return true;
	}

	private static bool HasNaN(VectorXd v)
	{
		foreach (double value in v.AsSpan())
		{
			if (double.IsNaN(value))
			{
				return true;
			}
		}

		return false;
	}

	/// <summary>L1 robust loss minimization phase.</summary>
	private bool SolveL1Regression(RotationAveragingProblem problem)
	{
		var l1SolverOptions = new LeastAbsoluteDeviationSolver.Options
		{
			MaxNumIterations = 10,
			Solver = LeastAbsoluteDeviationSolver.Options.SolverType.SupernodalCholmodLLT,
			RidgeRegularization = _options.RidgeRegularization,
		};

		// For weighted rotation averaging, the row-scaled constraint matrix and residuals solve
		// the weighted problem min ||D(Ax-b)||_1 = sum_i w_i |r_i|.
		SparseMatrixCsc l1Matrix = problem.WeightedConstraintMatrix();

		var l1Solver = new LeastAbsoluteDeviationSolver(l1SolverOptions, l1Matrix);
		if (!l1Solver.Valid)
		{
			Log.Error("L1 regression linear solver factorization failed");
			return false;
		}

		double currNorm = 0;
		var step = new VectorXd(problem.NumParameters);

		for (int iteration = 0; iteration < _options.MaxNumL1Iterations; iteration++)
		{
			problem.ComputeResiduals();

			double prevNorm = currNorm;

			step.AsSpan().Clear();
			if (!l1Solver.Solve(problem.WeightedResiduals(), step) || HasNaN(step))
			{
				Log.Error($"L1 regression solve failed (iteration {iteration})");
				return false;
			}

			currNorm = step.Norm();
			problem.UpdateState(step);

			// Check convergence.
			const double kEps = 1e-12;
			if (problem.AverageStepSize(step) < _options.L1StepConvergenceThreshold
				|| Math.Abs(prevNorm - currNorm) < kEps)
			{
				break;
			}

			l1Solver.MaxNumIterations = Math.Min(l1Solver.MaxNumIterations * 2, 100);
		}

		return true;
	}

	/// <summary>
	/// Computes the IRLS weight of every residual row; null if any weight is NaN.
	/// </summary>
	private VectorXd? ComputeIrlsWeights(RotationAveragingProblem problem, double sigma)
	{
		var weights = new VectorXd(problem.NumResiduals);
		VectorXd residuals = problem.Residuals;

		foreach (RotationAveragingProblem.PairConstraint constraint in problem.PairConstraints.Values)
		{
			int row = constraint.RowIndex;
			double errSquared;
			bool is1Dof = false;
			if (constraint.Constraint is RotationAveragingProblem.GravityAligned1Dof c1)
			{
				// 1-DOF: Y-axis error plus xz_error.
				double residual = residuals[row];
				errSquared = residual * residual + c1.XzError;
				is1Dof = true;
			}
			else
			{
				// 3-DOF: full rotation error.
				errSquared = residuals[row] * residuals[row]
					+ residuals[row + 1] * residuals[row + 1]
					+ residuals[row + 2] * residuals[row + 2];
			}

			// Compute the weight.
			double w = 0;
			if (_options.WeightType == RotationWeightType.GemanMcClure)
			{
				double tmp = errSquared + sigma * sigma;
				w = sigma * sigma / (tmp * tmp);
			}
			else if (_options.WeightType == RotationWeightType.HalfNorm)
			{
				// Exponent for half-norm weight: (p - 2) / 2 where p = 0.5.
				const double kHalfNormExponent = (0.5 - 2) / 2;
				w = Math.Pow(errSquared, kHalfNormExponent);
			}

			if (double.IsNaN(w))
			{
				Log.Error("nan weight!");
				return null;
			}

			// Set weights for appropriate number of equations.
			int rows = is1Dof ? 1 : 3;
			for (int i = 0; i < rows; i++)
			{
				weights[row + i] = w;
			}
		}

		// Set gauge-fixing weights to 1.
		int gaugeRows = problem.NumGaugeFixingResiduals;
		for (int i = problem.NumResiduals - gaugeRows; i < problem.NumResiduals; i++)
		{
			weights[i] = 1;
		}

		// Fold the residual-space reweighting W into the IRLS weights so the solver can
		// operate on the plain constraint matrix, scaling each robust weight by W
		// (gauge-fixing rows have W = 1 and are left unchanged).
		if (problem.ResidualReweighting is VectorXd reweighting)
		{
			for (int i = 0; i < weights.Length; i++)
			{
				weights[i] *= reweighting[i];
			}
		}

		return weights;
	}

	/// <summary>Iteratively reweighted least squares phase.</summary>
	private bool SolveIrls(RotationAveragingProblem problem)
	{
		var solver = new SparseCholeskyWithFallbackSolver();
		bool patternAnalyzed = false;

		double sigma = MathUtils.DegToRad(_options.IrlsLossParameterSigma);

		// The constraint matrix A is constant across iterations; only the per-row weights
		// change. ComputeIrlsWeights folds the optional residual-space reweighting W into the
		// robust weights, so we solve the normal equations A^T D A x = A^T D b directly with
		// the plain constraint matrix.
		SparseMatrixCsc constraintMatrix = problem.ConstraintMatrix;
		SparseMatrixCsc constraintMatrixT = constraintMatrix.Transpose();

		for (int iteration = 0; iteration < _options.MaxNumIrlsIterations; iteration++)
		{
			problem.ComputeResiduals();

			VectorXd? weightsIrls = ComputeIrlsWeights(problem, sigma);
			if (weightsIrls is null)
			{
				return false;
			}

			// A^T D: column i of A^T is row i of A, scaled by its weight.
			SparseMatrixCsc atWeight = constraintMatrixT.Clone();
			ScaleColumns(atWeight, weightsIrls);
			SparseMatrixCsc atWeightA = atWeight * constraintMatrix;

			// Optionally add a small Tikhonov ridge to stabilize poorly conditioned but
			// mathematically PD systems. When zero, no regularization is added.
			if (_options.RidgeRegularization > 0)
			{
				atWeightA = atWeightA.AddToDiagonal(_options.RidgeRegularization);
			}

			if (!patternAnalyzed)
			{
				solver.AnalyzePattern(atWeightA);
				patternAnalyzed = true;
			}

			if (!solver.Factorize(atWeightA))
			{
				Log.Error($"IRLS Cholesky factorization failed (iteration {iteration})");
				return false;
			}

			VectorXd atWeightResiduals = atWeight * problem.Residuals;
			if (!solver.Solve(atWeightResiduals, out VectorXd step) || HasNaN(step))
			{
				Log.Error($"IRLS solve failed (iteration {iteration})");
				return false;
			}

			problem.UpdateState(step);

			if (problem.AverageStepSize(step) < _options.IrlsStepConvergenceThreshold)
			{
				break;
			}
		}

		return true;
	}

	private static void ScaleColumns(SparseMatrixCsc matrix, VectorXd scale)
	{
		ReadOnlySpan<int> colPtr = matrix.ColPtr;
		Span<double> values = matrix.Values;
		for (int j = 0; j < matrix.Cols; j++)
		{
			for (int p = colPtr[j]; p < colPtr[j + 1]; p++)
			{
				values[p] *= scale[j];
			}
		}
	}
}
