// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/levenberg_marquardt_strategy.cc and
// internal/ceres/trust_region_step_evaluator.cc (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The two policies TrustRegionMinimizer.cs delegates to:
// - LevenbergMarquardtStrategy computes the step min |J s + f|^2 + |D s|^2 with
//   D = sqrt(clamp(diag(J'J)) / radius) through a LinearSolver (LinearSolvers.cs), and grows
//   or shrinks the radius from the step quality (Nielsen's update, as Ceres does it).
// - TrustRegionStepEvaluator measures a step's quality (actual over predicted decrease),
//   with Ceres' nonmonotonic reference-cost bookkeeping.

using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>ceres::internal::TrustRegionStrategy::Summary.</summary>
internal readonly record struct TrustRegionStrategySummary(LinearSolverTerminationType TerminationType, int NumIterations);

/// <summary>ceres::internal::LevenbergMarquardtStrategy.</summary>
internal sealed class LevenbergMarquardtStrategy
{
	private readonly LinearSolver linearSolver;
	private readonly double maxRadius;
	private readonly double minDiagonal;
	private readonly double maxDiagonal;
	private double radius;
	private double decreaseFactor = 2.0;
	private bool reuseDiagonal;
	private double[] diagonal = [];
	private double[] lmDiagonal = [];

	/// <summary>Creates the strategy.</summary>
	public LevenbergMarquardtStrategy(
		LinearSolver linearSolver, double initialRadius, double maxRadius, double minDiagonal, double maxDiagonal)
	{
		this.linearSolver = linearSolver;
		radius = initialRadius;
		this.maxRadius = maxRadius;
		this.minDiagonal = minDiagonal;
		this.maxDiagonal = maxDiagonal;
		Check.Gt(minDiagonal, 0.0);
		Check.Le(minDiagonal, maxDiagonal);
		Check.Gt(maxRadius, 0.0);
	}

	/// <summary>Current trust region radius.</summary>
	public double Radius => radius;

	/// <summary>
	/// Computes the LM step for <paramref name="jacobian"/> and <paramref name="residuals"/>
	/// into <paramref name="step"/> (already negated, so x + step decreases the model).
	/// <paramref name="eta"/> is the forcing sequence value an iterative linear solver
	/// stops at (Solver::Options::eta).
	/// </summary>
	public TrustRegionStrategySummary ComputeStep(SparseMatrix jacobian, ReadOnlySpan<double> residuals, Span<double> step, double eta)
	{
		int numParameters = jacobian.NumCols;
		if (!reuseDiagonal)
		{
			if (diagonal.Length != numParameters)
			{
				diagonal = new double[numParameters];
			}

			jacobian.SquaredColumnNorm(diagonal);
			for (int i = 0; i < numParameters; i++)
			{
				diagonal[i] = Math.Min(Math.Max(diagonal[i], minDiagonal), maxDiagonal);
			}
		}

		if (lmDiagonal.Length != numParameters)
		{
			lmDiagonal = new double[numParameters];
		}

		for (int i = 0; i < numParameters; i++)
		{
			lmDiagonal[i] = Math.Sqrt(diagonal[i] / radius);
		}

		// Invalidate the output array step so that we can detect if the linear solver
		// generated numerical garbage. This is known to happen for the DENSE_QR and then
		// DENSE_SCHUR solver when the Jacobian is severely rank deficient and mu is too small.
		ArrayValidity.Invalidate(step[..numParameters]);
		// Disable r_tolerance checking. Since we only care about termination via the
		// q_tolerance. As Nash and Sofer show, r_tolerance based termination is essentially
		// useless in Truncated Newton methods.
		LinearSolverSummary linearSolverSummary = linearSolver.Solve(
			jacobian, residuals, lmDiagonal, step, qTolerance: eta, rTolerance: -1.0);
		LinearSolverTerminationType terminationType = linearSolverSummary.TerminationType;
		if (terminationType is LinearSolverTerminationType.Success or LinearSolverTerminationType.NoConvergence)
		{
			if (!ArrayValidity.IsValid(step[..numParameters]))
			{
				// "Linear solver failure. Failed to compute a finite step."
				terminationType = LinearSolverTerminationType.Failure;
			}
			else
			{
				for (int i = 0; i < numParameters; i++)
				{
					step[i] = -step[i];
				}
			}
		}

		reuseDiagonal = true;
		return new TrustRegionStrategySummary(terminationType, linearSolverSummary.NumIterations);
	}

	/// <summary>Grows the radius after an accepted step of quality <paramref name="stepQuality"/>.</summary>
	public void StepAccepted(double stepQuality)
	{
		Check.Gt(stepQuality, 0.0);
		radius = radius / Math.Max(1.0 / 3.0, 1.0 - Math.Pow(2.0 * stepQuality - 1.0, 3));
		radius = Math.Min(maxRadius, radius);
		decreaseFactor = 2.0;
		reuseDiagonal = false;
	}

	/// <summary>Shrinks the radius after a rejected step, faster each time in a row.</summary>
	public void StepRejected()
	{
		radius = radius / decreaseFactor;
		decreaseFactor *= 2.0;
		reuseDiagonal = true;
	}

	/// <summary>TrustRegionStrategy::StepIsInvalid: treated as a rejection of the worst quality.</summary>
	public void StepIsInvalid() => StepRejected();
}

/// <summary>ceres::internal::TrustRegionStepEvaluator.</summary>
internal sealed class TrustRegionStepEvaluator
{
	private readonly int maxConsecutiveNonmonotonicSteps;
	private double minimumCost;
	private double currentCost;
	private double referenceCost;
	private double candidateCost;
	private double accumulatedReferenceModelCostChange;
	private double accumulatedCandidateModelCostChange;
	private int numConsecutiveNonmonotonicSteps;

	/// <summary>Starts at <paramref name="initialCost"/>; 0 steps of window means monotonic.</summary>
	public TrustRegionStepEvaluator(double initialCost, int maxConsecutiveNonmonotonicSteps)
	{
		this.maxConsecutiveNonmonotonicSteps = maxConsecutiveNonmonotonicSteps;
		minimumCost = initialCost;
		currentCost = initialCost;
		referenceCost = initialCost;
		candidateCost = initialCost;
	}

	/// <summary>
	/// The step quality: the larger of the relative decrease against the current cost and
	/// against the (nonmonotonic) reference cost.
	/// </summary>
	public double StepQuality(double cost, double modelCostChange)
	{
		// If the function evaluation for this step was a failure, in which case the
		// TrustRegionMinimizer would have set the cost to double max, then the step quality
		// is the lowest possible.
		if (cost >= double.MaxValue)
		{
			return double.MinValue;
		}

		double relativeDecrease = (currentCost - cost) / modelCostChange;
		double historicalRelativeDecrease =
			(referenceCost - cost) / (accumulatedReferenceModelCostChange + modelCostChange);
		return Math.Max(relativeDecrease, historicalRelativeDecrease);
	}

	/// <summary>Records an accepted step.</summary>
	public void StepAccepted(double cost, double modelCostChange)
	{
		// Algorithm 10.1.2 from Trust Region Methods by Conn, Gould & Toint.
		currentCost = cost;
		accumulatedCandidateModelCostChange += modelCostChange;
		accumulatedReferenceModelCostChange += modelCostChange;
		if (currentCost < minimumCost)
		{
			minimumCost = currentCost;
			numConsecutiveNonmonotonicSteps = 0;
			candidateCost = currentCost;
			accumulatedCandidateModelCostChange = 0.0;
		}
		else
		{
			// We have a non-monotonic step.
			numConsecutiveNonmonotonicSteps++;
			if (currentCost > candidateCost)
			{
				candidateCost = currentCost;
				accumulatedCandidateModelCostChange = 0.0;
			}
		}

		// We have exceeded the maximum number of non-monotonic steps; reset the reference
		// cost to the candidate cost.
		if (numConsecutiveNonmonotonicSteps == maxConsecutiveNonmonotonicSteps)
		{
			referenceCost = candidateCost;
			accumulatedReferenceModelCostChange = accumulatedCandidateModelCostChange;
		}
	}
}
