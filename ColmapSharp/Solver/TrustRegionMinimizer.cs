// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/trust_region_minimizer.cc and
// internal/ceres/minimizer.cc (RunCallbacks) (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The trust-region loop of ceres::Solve with the Levenberg-Marquardt strategy
// (TrustRegionStrategy.cs): evaluate, compute a step in the Jacobi-scaled space, evaluate
// the candidate, accept or reject by step quality, and stop on the function, gradient or
// parameter tolerance, the iteration or time limit, the minimum radius, too many invalid
// steps, a callback, or cancellation. On a bounds-constrained problem the start point is first
// projected onto the bounds, and every valid step is shortened by a projected Armijo line
// search (TrustRegionLineSearch.cs) before the candidate is evaluated; Plus does the
// projection (ParameterBlock.cs). Not ported, because nothing COLMAP builds enables them:
// inner iterations (use_inner_iterations) and the dogleg strategy.
// LeastSquaresSolver.cs sets it up and writes the result back.

using System.Diagnostics;
using System.Globalization;

namespace ColmapSharp.Solver;

/// <summary>ceres::internal::TrustRegionMinimizer (Levenberg-Marquardt).</summary>
internal sealed class TrustRegionMinimizer
{
	private readonly SolverOptions options;
	private readonly ProgramEvaluator evaluator;
	private readonly SparseMatrix jacobian;
	private readonly LevenbergMarquardtStrategy strategy;
	private readonly SolverSummary summary;
	private readonly CancellationToken cancellationToken;
	private readonly Stopwatch clock = Stopwatch.StartNew();
	private readonly int numResiduals;
	private readonly int numEffectiveParameters;
	private readonly bool isConstrained;
	private LineSearchFunction? lineSearchFunction;
	private ArmijoLineSearch? lineSearch;

	private double[] x = [];
	private double[] candidateX = [];
	private double[] residuals = [];
	private double[] gradient = [];
	private double[] trustRegionStep = [];
	private double[] delta = [];
	private double[] modelResiduals = [];
	private double[] negativeGradient = [];
	private double[] projectedGradientStep = [];
	private double[] jacobianScaling = [];

	private IterationSummary iterationSummary = new();
	private TrustRegionStepEvaluator? stepEvaluator;
	private double xCost;
	private double candidateCost;
	private double minimumCost;
	private double modelCostChange;
	private int numConsecutiveInvalidSteps;
	private double iterationStartTime;

	/// <summary>Creates the minimizer over an evaluator and its Jacobian.</summary>
	public TrustRegionMinimizer(
		SolverOptions options,
		ProgramEvaluator evaluator,
		SparseMatrix jacobian,
		LevenbergMarquardtStrategy strategy,
		SolverSummary summary,
		bool isConstrained,
		CancellationToken cancellationToken)
	{
		this.isConstrained = isConstrained;
		this.options = options;
		this.evaluator = evaluator;
		this.jacobian = jacobian;
		this.strategy = strategy;
		this.summary = summary;
		this.cancellationToken = cancellationToken;
		numResiduals = evaluator.NumResiduals;
		numEffectiveParameters = evaluator.NumEffectiveParameters;
	}

	/// <summary>
	/// Minimizes starting from, and writing the best point to, <paramref name="parameters"/>
	/// (the reduced program's state vector).
	/// </summary>
	public void Minimize(double[] parameters)
	{
		iterationStartTime = Now;
		Init(parameters);
		if (!IterationZero())
		{
			return;
		}

		stepEvaluator = new TrustRegionStepEvaluator(
			xCost, options.UseNonmonotonicSteps ? options.MaxConsecutiveNonmonotonicSteps : 0);

		bool atLeastOneSuccessfulStep = false;
		while (FinalizeIterationAndCheckIfMinimizerCanContinue(parameters))
		{
			iterationStartTime = Now;
			double previousGradientNorm = iterationSummary.GradientNorm;
			double previousGradientMaxNorm = iterationSummary.GradientMaxNorm;
			iterationSummary = new IterationSummary { Iteration = summary.Iterations[^1].Iteration + 1 };

			if (!ComputeTrustRegionStep())
			{
				return;
			}

			if (!iterationSummary.StepIsValid)
			{
				if (!HandleInvalidStep())
				{
					return;
				}

				continue;
			}

			if (isConstrained && options.MaxNumLineSearchStepSizeIterations > 0)
			{
				// Use a projected line search to enforce the bounds constraints and improve
				// the quality of the step.
				DoLineSearch();
			}

			ComputeCandidatePointAndEvaluateCost();

			if (atLeastOneSuccessfulStep && ParameterToleranceReached())
			{
				return;
			}

			if (FunctionToleranceReached())
			{
				return;
			}

			if (IsStepSuccessful())
			{
				atLeastOneSuccessfulStep = true;
				if (!HandleSuccessfulStep())
				{
					return;
				}

				continue;
			}

			// Declare the step unsuccessful and inform the trust region strategy.
			iterationSummary = iterationSummary with
			{
				StepIsSuccessful = false,
				Cost = candidateCost + summary.FixedCost,

				// When the step is unsuccessful, we do not compute the gradient (or update
				// x), so we preserve its value from the last successful iteration.
				GradientNorm = previousGradientNorm,
				GradientMaxNorm = previousGradientMaxNorm,
			};
			strategy.StepRejected();
		}
	}

	private double Now => clock.Elapsed.TotalSeconds;

	private void Init(double[] parameters)
	{
		summary.TerminationType = TerminationType.NoConvergence;
		summary.NumSuccessfulSteps = 0;
		summary.NumUnsuccessfulSteps = 0;
		summary.IsConstrained = isConstrained;
		numConsecutiveInvalidSteps = 0;
		if (isConstrained)
		{
			lineSearchFunction = new LineSearchFunction(evaluator);
			lineSearch = new ArmijoLineSearch(options, lineSearchFunction);
		}

		x = (double[])parameters.Clone();
		candidateX = new double[parameters.Length];
		residuals = new double[numResiduals];
		trustRegionStep = new double[numEffectiveParameters];
		delta = new double[numEffectiveParameters];
		gradient = new double[numEffectiveParameters];
		modelResiduals = new double[numResiduals];
		negativeGradient = new double[numEffectiveParameters];
		projectedGradientStep = new double[parameters.Length];
		jacobianScaling = new double[numEffectiveParameters];
		Array.Fill(jacobianScaling, 1.0);

		xCost = double.MaxValue;
		minimumCost = xCost;
		modelCostChange = 0.0;
	}

	private bool IterationZero()
	{
		iterationSummary = new IterationSummary { Iteration = 0, Eta = options.Eta };
		if (isConstrained)
		{
			// Project the starting point onto the bounds: x = Plus(x, 0).
			Array.Clear(delta);
			if (!evaluator.Plus(x, delta, candidateX))
			{
				summary.Message = "Unable to project initial point onto the feasible set.";
				summary.TerminationType = TerminationType.Failure;
				return false;
			}

			candidateX.AsSpan().CopyTo(x);
		}

		if (!EvaluateGradientAndJacobian())
		{
			summary.Message = "Initial residual and Jacobian evaluation failed.";
			return false;
		}

		summary.InitialCost = xCost + summary.FixedCost;
		iterationSummary = iterationSummary with { StepIsValid = true, StepIsSuccessful = true };
		return true;
	}

	// Evaluates the cost, residuals, gradient and Jacobian at x, applies the Jacobi scaling
	// (computed once, at iteration 0) and the gradient norms of the projected gradient step.
	private bool EvaluateGradientAndJacobian()
	{
		if (!evaluator.Evaluate(x, out xCost, residuals, gradient, jacobian))
		{
			summary.Message = "Residual and Jacobian evaluation failed.";
			summary.TerminationType = TerminationType.Failure;
			return false;
		}

		iterationSummary = iterationSummary with { Cost = xCost + summary.FixedCost };

		if (options.JacobiScaling)
		{
			if (iterationSummary.Iteration == 0)
			{
				// Compute a scaling vector that is used to improve the conditioning of the
				// Jacobian: jacobian_scaling = diag(J'J)^{-1/2}, with a 1 + to avoid division
				// by zero.
				jacobian.SquaredColumnNorm(jacobianScaling);
				for (int i = 0; i < jacobian.NumCols; i++)
				{
					jacobianScaling[i] = 1.0 / (1.0 + Math.Sqrt(jacobianScaling[i]));
				}
			}

			// This must be done after the gradient is evaluated, since the gradient is the
			// unscaled J'f.
			jacobian.ScaleColumns(jacobianScaling);
		}

		// The gradient exists in the local tangent space. To account for the bounds
		// constraints correctly, instead of just computing the norm of the gradient vector,
		// we compute the norm of x - Plus(x, -gradient).
		for (int i = 0; i < gradient.Length; i++)
		{
			negativeGradient[i] = -gradient[i];
		}

		if (!evaluator.Plus(x, negativeGradient, projectedGradientStep))
		{
			summary.Message = "projected_gradient_step = Plus(x, -gradient) failed.";
			summary.TerminationType = TerminationType.Failure;
			return false;
		}

		double maxNorm = 0.0;
		double squaredNorm = 0.0;
		for (int i = 0; i < x.Length; i++)
		{
			double d = x[i] - projectedGradientStep[i];
			maxNorm = Math.Max(maxNorm, Math.Abs(d));
			squaredNorm += d * d;
		}

		iterationSummary = iterationSummary with { GradientMaxNorm = maxNorm, GradientNorm = Math.Sqrt(squaredNorm) };
		return true;
	}

	private bool FinalizeIterationAndCheckIfMinimizerCanContinue(double[] parameters)
	{
		if (iterationSummary.StepIsSuccessful)
		{
			summary.NumSuccessfulSteps++;
			if (xCost < minimumCost)
			{
				minimumCost = xCost;
				x.AsSpan().CopyTo(parameters);
				iterationSummary = iterationSummary with { StepIsNonmonotonic = false };
			}
			else
			{
				iterationSummary = iterationSummary with { StepIsNonmonotonic = true };
			}
		}
		else
		{
			summary.NumUnsuccessfulSteps++;
		}

		double now = Now;
		iterationSummary = iterationSummary with
		{
			TrustRegionRadius = strategy.Radius,
			IterationTimeInSeconds = now - iterationStartTime,
			CumulativeTimeInSeconds = now,
		};
		summary.Iterations.Add(iterationSummary);

		return RunCallbacks()
			&& !MaxSolverTimeReached()
			&& !MaxSolverIterationsReached()
			&& !GradientToleranceReached()
			&& !MinTrustRegionRadiusReached();
	}

	// Minimizer::RunCallbacks, with cancellation acting as COLMAP's CancellationCallback
	// (estimators/bundle_adjustment_ceres.cc), which returns SOLVER_TERMINATE_SUCCESSFULLY.
	private bool RunCallbacks()
	{
		CallbackReturnType status = CallbackReturnType.SolverContinue;
		for (int i = 0; status == CallbackReturnType.SolverContinue && i < options.Callbacks.Count; i++)
		{
			status = options.Callbacks[i].Invoke(iterationSummary);
		}

		if (status == CallbackReturnType.SolverContinue && cancellationToken.IsCancellationRequested)
		{
			summary.TerminationType = TerminationType.UserSuccess;
			summary.Message = "Cancelled: the solve stopped at the best point so far.";
			return false;
		}

		switch (status)
		{
			case CallbackReturnType.SolverContinue:
				return true;
			case CallbackReturnType.SolverTerminateSuccessfully:
				summary.TerminationType = TerminationType.UserSuccess;
				summary.Message = "User callback returned SOLVER_TERMINATE_SUCCESSFULLY.";
				return false;
			case CallbackReturnType.SolverAbort:
				summary.TerminationType = TerminationType.UserFailure;
				summary.Message = "User callback returned SOLVER_ABORT.";
				return false;
			default:
				throw new InvalidOperationException("Unknown type of user callback status");
		}
	}

	// Computes the step and the model's predicted cost change.
	private bool ComputeTrustRegionStep()
	{
		double strategyStartTime = Now;
		iterationSummary = iterationSummary with { StepIsValid = false };
		TrustRegionStrategySummary strategySummary = strategy.ComputeStep(jacobian, residuals, trustRegionStep, options.Eta);
		if (strategySummary.TerminationType == LinearSolverTerminationType.FatalError)
		{
			summary.Message =
				"Linear solver failed due to unrecoverable non-numeric causes. Please see the error log for clues. ";
			summary.TerminationType = TerminationType.Failure;
			return false;
		}

		iterationSummary = iterationSummary with
		{
			StepSolverTimeInSeconds = Now - strategyStartTime,
			LinearSolverIterations = strategySummary.NumIterations,
		};

		if (strategySummary.TerminationType == LinearSolverTerminationType.Failure)
		{
			return true;
		}

		// new_model_cost = 1/2 [f + J * step]^2 = 1/2 [ f'f + 2f'J * step + step' * J' * J * step ]
		// model_cost_change = cost - new_model_cost
		//                   = f'f/2 - 1/2 [ f'f + 2f'J * step + step' * J' * J * step]
		//                   = -f'J * step - step' * J' * J * step / 2
		//                   = -(J * step)'(f + J * step / 2)
		Array.Clear(modelResiduals);
		jacobian.RightMultiplyAndAccumulate(trustRegionStep, modelResiduals);
		double dot = 0.0;
		for (int i = 0; i < numResiduals; i++)
		{
			dot += modelResiduals[i] * (residuals[i] + modelResiduals[i] / 2.0);
		}

		modelCostChange = -dot;

		// A step that does not decrease the model is invalid: the linear solve went wrong.
		bool valid = modelCostChange > 0.0;
		iterationSummary = iterationSummary with { StepIsValid = valid };
		if (valid)
		{
			// Undo the Jacobian column scaling.
			for (int i = 0; i < numEffectiveParameters; i++)
			{
				delta[i] = trustRegionStep[i] * jacobianScaling[i];
			}

			numConsecutiveInvalidSteps = 0;
		}

		return true;
	}

	// DoLineSearch: an Armijo search along delta from x, starting at the full step, scales
	// delta by the step length it finds (and leaves it alone if it fails). Ceres notes this
	// "does not do anything illegal but is incorrect and not terribly effective"
	// (ceres-solver issue 187); it is ported as is.
	private void DoLineSearch()
	{
		lineSearchFunction!.Init(x, delta);
		double directionalDerivative = 0.0;
		for (int i = 0; i < numEffectiveParameters; i++)
		{
			directionalDerivative += gradient[i] * delta[i];
		}

		LineSearchSummary lineSearchSummary = lineSearch!.Search(1.0, xCost, directionalDerivative);
		summary.NumLineSearchSteps += lineSearchSummary.NumIterations;
		if (lineSearchSummary.Success)
		{
			double stepLength = lineSearchSummary.OptimalPoint.X;
			for (int i = 0; i < numEffectiveParameters; i++)
			{
				delta[i] *= stepLength;
			}
		}
	}

	private bool HandleInvalidStep()
	{
		if (++numConsecutiveInvalidSteps >= options.MaxNumConsecutiveInvalidSteps)
		{
			summary.Message = string.Format(
				CultureInfo.InvariantCulture,
				"Number of consecutive invalid steps more than Solver::Options::max_num_consecutive_invalid_steps: {0}",
				options.MaxNumConsecutiveInvalidSteps);
			summary.TerminationType = TerminationType.Failure;
			return false;
		}

		strategy.StepIsInvalid();

		// We are going to try and reduce the trust region radius and solve again. To do
		// this, we are going to treat this iteration as an unsuccessful iteration. Since the
		// various callbacks are still executed, we are going to fill the iteration summary
		// with data that assumes a step of length zero and no progress.
		IterationSummary previous = summary.Iterations[^1];
		iterationSummary = iterationSummary with
		{
			Cost = xCost + summary.FixedCost,
			CostChange = 0.0,
			GradientMaxNorm = previous.GradientMaxNorm,
			GradientNorm = previous.GradientNorm,
			StepNorm = 0.0,
			RelativeDecrease = 0.0,
			Eta = options.Eta,
		};
		return true;
	}

	private bool MaxSolverTimeReached()
	{
		double totalSolverTime = Now;
		if (totalSolverTime < options.MaxSolverTimeInSeconds)
		{
			return false;
		}

		summary.Message = string.Format(
			CultureInfo.InvariantCulture,
			"Maximum solver time reached. Total solver time: {0} >= {1}.",
			SolverSummary.FormatE(totalSolverTime),
			SolverSummary.FormatE(options.MaxSolverTimeInSeconds));
		summary.TerminationType = TerminationType.NoConvergence;
		return true;
	}

	private bool MaxSolverIterationsReached()
	{
		if (iterationSummary.Iteration < options.MaxNumIterations)
		{
			return false;
		}

		summary.Message = string.Format(
			CultureInfo.InvariantCulture,
			"Maximum number of iterations reached. Number of iterations: {0}.",
			iterationSummary.Iteration);
		summary.TerminationType = TerminationType.NoConvergence;
		return true;
	}

	private bool GradientToleranceReached()
	{
		if (!iterationSummary.StepIsSuccessful || iterationSummary.GradientMaxNorm > options.GradientTolerance)
		{
			return false;
		}

		summary.Message = string.Format(
			CultureInfo.InvariantCulture,
			"Gradient tolerance reached. Gradient max norm: {0} <= {1}",
			SolverSummary.FormatE(iterationSummary.GradientMaxNorm),
			SolverSummary.FormatE(options.GradientTolerance));
		summary.TerminationType = TerminationType.Convergence;
		return true;
	}

	private bool MinTrustRegionRadiusReached()
	{
		if (iterationSummary.TrustRegionRadius > options.MinTrustRegionRadius)
		{
			return false;
		}

		summary.Message = string.Format(
			CultureInfo.InvariantCulture,
			"Minimum trust region radius reached. Trust region radius: {0} <= {1}",
			SolverSummary.FormatE(iterationSummary.TrustRegionRadius),
			SolverSummary.FormatE(options.MinTrustRegionRadius));
		summary.TerminationType = TerminationType.Convergence;
		return true;
	}

	// Solver::Options::parameter_tolerance based convergence check.
	private bool ParameterToleranceReached()
	{
		double xNormSquared = 0.0;
		double stepNormSquared = 0.0;
		for (int i = 0; i < x.Length; i++)
		{
			xNormSquared += x[i] * x[i];
			double d = x[i] - candidateX[i];
			stepNormSquared += d * d;
		}

		double xNorm = Math.Sqrt(xNormSquared);
		double stepNorm = Math.Sqrt(stepNormSquared);
		iterationSummary = iterationSummary with { StepNorm = stepNorm };
		double stepSizeTolerance = options.ParameterTolerance * (xNorm + options.ParameterTolerance);
		if (stepNorm > stepSizeTolerance)
		{
			return false;
		}

		summary.Message = string.Format(
			CultureInfo.InvariantCulture,
			"Parameter tolerance reached. Relative step_norm: {0} <= {1}.",
			SolverSummary.FormatE(stepNorm / (xNorm + options.ParameterTolerance)),
			SolverSummary.FormatE(options.ParameterTolerance));
		summary.TerminationType = TerminationType.Convergence;
		return true;
	}

	// Solver::Options::function_tolerance based convergence check.
	private bool FunctionToleranceReached()
	{
		double costChange = xCost - candidateCost;
		iterationSummary = iterationSummary with { CostChange = costChange };
		double absoluteFunctionTolerance = options.FunctionTolerance * xCost;
		if (Math.Abs(costChange) > absoluteFunctionTolerance)
		{
			return false;
		}

		summary.Message = string.Format(
			CultureInfo.InvariantCulture,
			"Function tolerance reached. |cost_change|/cost: {0} <= {1}",
			SolverSummary.FormatE(Math.Abs(costChange) / xCost),
			SolverSummary.FormatE(options.FunctionTolerance));
		summary.TerminationType = TerminationType.Convergence;
		return true;
	}

	// candidate_x = Plus(x, delta); a failed Plus or evaluation is a step of infinite cost.
	private void ComputeCandidatePointAndEvaluateCost()
	{
		if (!evaluator.Plus(x, delta, candidateX))
		{
			candidateCost = double.MaxValue;
			return;
		}

		if (!evaluator.Evaluate(candidateX, out candidateCost, null, null, null))
		{
			candidateCost = double.MaxValue;
		}
	}

	private bool IsStepSuccessful()
	{
		double relativeDecrease = stepEvaluator!.StepQuality(candidateCost, modelCostChange);
		iterationSummary = iterationSummary with { RelativeDecrease = relativeDecrease };
		return relativeDecrease > options.MinRelativeDecrease;
	}

	private bool HandleSuccessfulStep()
	{
		// A copy, as Ceres does: candidate_x must keep the accepted point, because if the
		// next Plus fails it is left as is and ParameterToleranceReached measures x - candidate_x.
		candidateX.AsSpan().CopyTo(x);
		if (!EvaluateGradientAndJacobian())
		{
			return false;
		}

		iterationSummary = iterationSummary with { StepIsSuccessful = true };
		strategy.StepAccepted(iterationSummary.RelativeDecrease);
		stepEvaluator!.StepAccepted(candidateCost, modelCostChange);
		return true;
	}
}
