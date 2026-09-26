// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/line_search.cc (LineSearchFunction,
// LineSearch::InterpolatingPolynomialMinimizingStepSize, ArmijoLineSearch) and
// internal/ceres/line_search.h (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The projected line search the trust-region minimizer runs on a bounds-constrained problem
// (TrustRegionMinimizer.cs, DoLineSearch): after the LM step is computed, an Armijo
// backtracking search along it picks a step length, each trial point being Plus(x, t * delta),
// which projects onto the bounds (ParameterBlock.cs). Trial steps are shortened by
// minimizing a polynomial fitted to the samples so far (CeresPolynomial.cs). Only the Armijo
// search is here: the Wolfe search belongs to Ceres' line search minimizer, which COLMAP
// never selects. The search's timing statistics are not ported (log-only). The buffers are
// allocated once per solve, so a search allocates only the polynomial fit's small matrices.

using System.Globalization;

using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>
/// ceres::internal::LineSearchFunction: the cost along a direction from a position,
/// f(t) = cost(Plus(position, t * direction)), and its derivative direction . gradient.
/// </summary>
internal sealed class LineSearchFunction
{
	private readonly ProgramEvaluator evaluator;
	private readonly double[] position;
	private readonly double[] direction;
	private readonly double[] scaledDirection;
	private readonly double[] vectorX;
	private readonly double[] vectorGradient;

	/// <summary>Creates the function over <paramref name="evaluator"/>'s state space.</summary>
	public LineSearchFunction(ProgramEvaluator evaluator)
	{
		this.evaluator = evaluator;
		position = new double[evaluator.NumParameters];
		vectorX = new double[evaluator.NumParameters];
		direction = new double[evaluator.NumEffectiveParameters];
		scaledDirection = new double[evaluator.NumEffectiveParameters];
		vectorGradient = new double[evaluator.NumEffectiveParameters];
	}

	/// <summary>Sets the position and the direction.</summary>
	public void Init(ReadOnlySpan<double> newPosition, ReadOnlySpan<double> newDirection)
	{
		newPosition.CopyTo(position);
		newDirection.CopyTo(direction);
	}

	/// <summary>
	/// Evaluates f (and with <paramref name="evaluateGradient"/>, f') at <paramref name="x"/>.
	/// A failed Plus or evaluation, or a non-finite value, leaves the value invalid.
	/// </summary>
	public FunctionSample Evaluate(double x, bool evaluateGradient)
	{
		var output = new FunctionSample { X = x };
		for (int i = 0; i < direction.Length; i++)
		{
			scaledDirection[i] = x * direction[i];
		}

		if (!evaluator.Plus(position, scaledDirection, vectorX))
		{
			return output;
		}

		bool evalStatus = evaluator.Evaluate(
			vectorX, out double value, null, evaluateGradient ? vectorGradient : null, null);
		output.Value = value;
		if (!evalStatus || !double.IsFinite(value))
		{
			return output;
		}

		output.ValueIsValid = true;
		if (!evaluateGradient)
		{
			return output;
		}

		double gradient = 0.0;
		for (int i = 0; i < direction.Length; i++)
		{
			gradient += direction[i] * vectorGradient[i];
		}

		output.Gradient = gradient;
		if (!double.IsFinite(gradient))
		{
			return output;
		}

		output.GradientIsValid = true;
		return output;
	}

	/// <summary>The infinity norm of the direction.</summary>
	public double DirectionInfinityNorm()
	{
		double norm = 0.0;
		foreach (double value in direction)
		{
			norm = Math.Max(norm, Math.Abs(value));
		}

		return norm;
	}
}

/// <summary>ceres::internal::LineSearch::Summary, the fields the minimizer reads.</summary>
internal struct LineSearchSummary
{
	/// <summary>Whether a step satisfying the sufficient decrease condition was found.</summary>
	public bool Success;

	/// <summary>The accepted sample.</summary>
	public FunctionSample OptimalPoint;

	/// <summary>Function evaluations, the initial trial included.</summary>
	public int NumFunctionEvaluations;

	/// <summary>Gradient evaluations.</summary>
	public int NumGradientEvaluations;

	/// <summary>Backtracking iterations.</summary>
	public int NumIterations;

	/// <summary>Why the search failed.</summary>
	public string Error;
}

/// <summary>ceres::internal::ArmijoLineSearch with the options of the trust-region minimizer.</summary>
internal sealed class ArmijoLineSearch(SolverOptions options, LineSearchFunction function)
{
	/// <summary>
	/// ArmijoLineSearch::DoSearch: from <paramref name="stepSizeEstimate"/>, backtrack until
	/// f(t) &lt;= f(0) + sufficient_decrease * f'(0) * t. <paramref name="initialCost"/> and
	/// <paramref name="initialGradient"/> are f(0) and f'(0).
	/// </summary>
	public LineSearchSummary Search(double stepSizeEstimate, double initialCost, double initialGradient)
	{
		Check.Ge(stepSizeEstimate, 0.0);
		Check.Gt(options.LineSearchSufficientFunctionDecrease, 0.0);
		Check.Lt(options.LineSearchSufficientFunctionDecrease, 1.0);
		Check.Gt(options.MaxNumLineSearchStepSizeIterations, 0);
		var summary = new LineSearchSummary { Error = string.Empty };

		// Note initial_cost & initial_gradient are evaluated at step_size = 0, not
		// step_size_estimate, which is our starting guess.
		var initialPosition = new FunctionSample(0.0, initialCost, initialGradient);
		double descentDirectionMaxNorm = function.DirectionInfinityNorm();
		var previous = new FunctionSample();

		// As the Armijo line search algorithm always uses the initial point, for which both
		// the function value and derivative are known, when fitting a minimizing polynomial,
		// we can fit up to a quadratic without requiring the gradient at the current query
		// point.
		bool evaluateGradient = options.LineSearchInterpolationType == LineSearchInterpolationType.Cubic;

		summary.NumFunctionEvaluations++;
		if (evaluateGradient)
		{
			summary.NumGradientEvaluations++;
		}

		FunctionSample current = function.Evaluate(stepSizeEstimate, evaluateGradient);
		while (!current.ValueIsValid
			|| current.Value > initialCost + options.LineSearchSufficientFunctionDecrease * initialGradient * current.X)
		{
			// If current.value_is_valid is false, we treat it as if the cost at that point is
			// not large enough to satisfy the sufficient decrease condition.
			summary.NumIterations++;
			if (summary.NumIterations >= options.MaxNumLineSearchStepSizeIterations)
			{
				summary.Error = string.Format(
					CultureInfo.InvariantCulture,
					"Line search failed: Armijo failed to find a point satisfying the sufficient decrease condition within specified max_num_iterations: {0}.",
					options.MaxNumLineSearchStepSizeIterations);
				return summary;
			}

			double stepSize = InterpolatingPolynomialMinimizingStepSize(
				options.LineSearchInterpolationType,
				initialPosition,
				previous,
				current,
				options.MaxLineSearchStepContraction * current.X,
				options.MinLineSearchStepContraction * current.X);

			if (stepSize * descentDirectionMaxNorm < options.MinLineSearchStepSize)
			{
				summary.Error = string.Concat(
					"Line search failed: step_size too small: ",
					SolverSummary.FormatE(stepSize, 5),
					" with descent_direction_max_norm: ",
					SolverSummary.FormatE(descentDirectionMaxNorm, 5),
					".");
				return summary;
			}

			previous = current;
			summary.NumFunctionEvaluations++;
			if (evaluateGradient)
			{
				summary.NumGradientEvaluations++;
			}

			current = function.Evaluate(stepSize, evaluateGradient);
		}

		summary.OptimalPoint = current;
		summary.Success = true;
		return summary;
	}

	/// <summary>
	/// LineSearch::InterpolatingPolynomialMinimizingStepSize: the next trial step in
	/// [minStepSize, maxStepSize], by halving (BISECTION, or an invalid sample) or by
	/// minimizing the polynomial through the lower bound and the current (and, when valid,
	/// previous) samples.
	/// </summary>
	public static double InterpolatingPolynomialMinimizingStepSize(
		LineSearchInterpolationType interpolationType,
		FunctionSample lowerbound,
		FunctionSample previous,
		FunctionSample current,
		double minStepSize,
		double maxStepSize)
	{
		if (!current.ValueIsValid || (interpolationType == LineSearchInterpolationType.Bisection && maxStepSize <= current.X))
		{
			// Either: sample is invalid; or we are using BISECTION and contracting the step
			// size. (std::min(std::max(a, lo), hi).)
			double halved = current.X * 0.5;
			double clampedBelow = halved < minStepSize ? minStepSize : halved;
			return maxStepSize < clampedBelow ? maxStepSize : clampedBelow;
		}

		if (interpolationType == LineSearchInterpolationType.Bisection)
		{
			Check.Gt(maxStepSize, current.X);

			// We are expanding the search using BISECTION interpolation, which is defined to
			// mean always taking the maximum step size.
			return maxStepSize;
		}

		// Only check if lower-bound is valid here, where it is required to avoid replicating
		// current.value_is_valid == false behaviour in WolfeLineSearch.
		Check.That(lowerbound.ValueIsValid, "Ceres bug: lower-bound sample for interpolation is invalid, please contact the developers!");

		// Select step size by interpolating the function and gradient values and minimizing
		// the corresponding polynomial.
		Span<FunctionSample> samples = stackalloc FunctionSample[3];
		int count = 0;
		samples[count++] = lowerbound;
		if (interpolationType == LineSearchInterpolationType.Quadratic)
		{
			// Two point interpolation using function values and the gradient at the lower
			// bound; three point with the previous value when it is valid.
			samples[count++] = new FunctionSample(current.X, current.Value);
			if (previous.ValueIsValid)
			{
				samples[count++] = new FunctionSample(previous.X, previous.Value);
			}
		}
		else
		{
			// CUBIC: two (or three) point interpolation using the function values and the
			// gradients.
			samples[count++] = current;
			if (previous.ValueIsValid)
			{
				samples[count++] = previous;
			}
		}

		CeresPolynomial.MinimizeInterpolatingPolynomial(samples[..count], minStepSize, maxStepSize, out double stepSize, out _);
		return stepSize;
	}
}
