// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 include/ceres/gradient_checker.h,
// internal/ceres/gradient_checker.cc, internal/ceres/is_close.cc,
// include/ceres/numeric_diff_options.h and the RIDDERS path of
// include/ceres/internal/numeric_diff.h / include/ceres/dynamic_numeric_diff_cost_function.h
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GradientChecker: ceres::GradientChecker, which compares a CostFunction's Jacobians with
// numeric ones from Ridders' adaptive finite differences (RiddersNumericDiffCostFunction
// below, Ceres' DynamicNumericDiffCostFunction<CostFunction, RIDDERS>). COLMAP uses it to
// validate its analytic reprojection Jacobians (reprojection_error_test.cc; tested here by
// ColmapSharp.Tests/Estimators/CostFunctions/ReprojectionErrorTests.cs).
//
// Scope: COLMAP always passes no manifolds, so the checker compares ambient Jacobians only
// (Ceres' local_jacobians equal jacobians then). The per-entry error table Ceres builds with
// StringAppendF is reproduced in a plain format for the failure message.

using System.Globalization;
using System.Text;

namespace ColmapSharp.Solver;

/// <summary>ceres::NumericDiffOptions: the step-size controls of numeric differentiation.</summary>
public sealed class NumericDiffOptions
{
	/// <summary>Relative step for forward/central differences.</summary>
	public double RelativeStepSize { get; init; } = 1e-6;

	/// <summary>Initial relative step of Ridders' method.</summary>
	public double RiddersRelativeInitialStepSize { get; init; } = 1e-2;

	/// <summary>Maximal number of Ridders extrapolations.</summary>
	public int MaxNumRiddersExtrapolations { get; init; } = 10;

	/// <summary>Convergence criterion on the extrapolation error.</summary>
	public double RiddersEpsilon { get; init; } = 1e-12;

	/// <summary>Factor by which the step shrinks per extrapolation.</summary>
	public double RiddersStepShrinkFactor { get; init; } = 2.0;
}

/// <summary>
/// ceres::DynamicNumericDiffCostFunction&lt;CostFunction, RIDDERS&gt;: the Jacobians of a
/// cost function by Ridders' method (C.J.F. Ridders, "Accurate computation of F'(x) and
/// F'(x) F"(x)", Advances in Engineering Software 4(2), 1982), evaluating the wrapped
/// function without Jacobians at shifted parameters.
/// </summary>
public sealed class RiddersNumericDiffCostFunction : CostFunction
{
	private readonly CostFunction _function;
	private readonly NumericDiffOptions _options;

	/// <summary>Wraps <paramref name="function"/>; it is not owned.</summary>
	public RiddersNumericDiffCostFunction(CostFunction function, NumericDiffOptions options)
		: base(function.NumResiduals, function.ParameterBlockSizes.ToArray())
	{
		_function = function;
		_options = options;
	}

	/// <inheritdoc/>
	public override bool Evaluate(
		ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
	{
		ReadOnlySpan<int> sizes = ParameterBlockSizes;

		// Work on a copy of the parameters so the caller's are never mutated.
		var copies = new ArraySegment<double>[sizes.Length];
		for (int b = 0; b < sizes.Length; b++)
		{
			copies[b] = new ArraySegment<double>(parameters[b].AsSpan(0, sizes[b]).ToArray());
		}

		if (!_function.Evaluate(copies, residuals, default))
		{
			return false;
		}

		if (jacobians.IsEmpty)
		{
			return true;
		}

		int numResiduals = NumResiduals;
		for (int b = 0; b < sizes.Length; b++)
		{
			if (jacobians[b].Array is null)
			{
				continue;
			}

			if (!EvaluateJacobianForParameterBlock(copies, b, numResiduals, jacobians[b].AsSpan(0, numResiduals * sizes[b])))
			{
				return false;
			}
		}

		return true;
	}

	// NumericDiff::EvaluateJacobianForParameterBlock with kMethod = RIDDERS; the Jacobian is
	// row-major numResiduals x blockSize.
	private bool EvaluateJacobianForParameterBlock(ArraySegment<double>[] parameters, int block, int numResiduals, Span<double> jacobian)
	{
		double[] xPlusDelta = parameters[block].Array!;
		double[] x = (double[])xPlusDelta.Clone();
		int blockSize = x.Length;

		// It is not a good idea to make the step size arbitrarily small. This will lead to
		// problems with round off and numerical instability when dividing by the step size.
		// The general recommendation is to not go down below sqrt(epsilon). For Ridders'
		// method, the initial step size is required to be large, thus
		// ridders_relative_initial_step_size is used.
		double minStepSize = Math.Max(Math.Sqrt(2.220446049250313E-16), _options.RiddersRelativeInitialStepSize);

		var column = new double[numResiduals];
		for (int j = 0; j < blockSize; j++)
		{
			double stepSize = Math.Abs(x[j]) * _options.RiddersRelativeInitialStepSize;
			double delta = Math.Max(minStepSize, stepSize);
			if (!EvaluateRiddersJacobianColumn(parameters, xPlusDelta, x, j, delta, numResiduals, column))
			{
				return false;
			}

			for (int r = 0; r < numResiduals; r++)
			{
				jacobian[r * blockSize + j] = column[r];
			}
		}

		return true;
	}

	// NumericDiff::EvaluateRiddersJacobianColumn: a Romberg tableau of central differences
	// over shrinking steps, extrapolated by Richardson's method.
	private bool EvaluateRiddersJacobianColumn(
		ArraySegment<double>[] parameters, double[] xPlusDelta, double[] x, int parameterIndex, double delta, int numResiduals, double[] residuals)
	{
		int maxExtrapolations = _options.MaxNumRiddersExtrapolations;

		// In order for the algorithm to converge, the step size should be initialized to a
		// value that is large enough to produce a significant change in the function. As the
		// derivative is estimated, the step size decreases. By default, the step sizes are
		// chosen so that the middle column of the Romberg tableau uses the input delta.
		double currentStepSize = delta * Math.Pow(_options.RiddersStepShrinkFactor, maxExtrapolations / 2);

		// Double-buffered candidate columns (column-major numResiduals x maxExtrapolations).
		var current = new double[numResiduals * maxExtrapolations];
		var previous = new double[numResiduals * maxExtrapolations];
		var temp = new double[numResiduals];

		// The computational error of the derivative, initially large, then the difference
		// between current and previous extrapolations.
		double normError = double.MaxValue;

		// Loop over decreasing step sizes until the error is smaller than ridders_epsilon,
		// the maximal order of extrapolation is reached, or extrapolation becomes
		// numerically unstable.
		for (int i = 0; i < maxExtrapolations; i++)
		{
			if (!EvaluateCentralColumn(parameters, xPlusDelta, x, parameterIndex, currentStepSize, temp, current.AsSpan(0, numResiduals)))
			{
				return false;
			}

			if (i == 0)
			{
				current.AsSpan(0, numResiduals).CopyTo(residuals);
			}

			currentStepSize /= _options.RiddersStepShrinkFactor;

			// Extrapolation factor for the Richardson acceleration method.
			double richardsonFactor = _options.RiddersStepShrinkFactor * _options.RiddersStepShrinkFactor;
			for (int k = 1; k <= i; k++)
			{
				for (int r = 0; r < numResiduals; r++)
				{
					current[k * numResiduals + r] =
						(richardsonFactor * current[(k - 1) * numResiduals + r] - previous[(k - 1) * numResiduals + r])
						/ (richardsonFactor - 1.0);
				}

				richardsonFactor *= _options.RiddersStepShrinkFactor * _options.RiddersStepShrinkFactor;

				double candidateError = Math.Max(
					ColumnDistance(current, k, current, k - 1, numResiduals),
					ColumnDistance(current, k, previous, k - 1, numResiduals));

				// If the error has decreased, update results.
				if (candidateError <= normError)
				{
					normError = candidateError;
					current.AsSpan(k * numResiduals, numResiduals).CopyTo(residuals);

					// If the error is small enough, stop.
					if (normError < _options.RiddersEpsilon)
					{
						break;
					}
				}
			}

			// After breaking out of the inner loop, declare convergence.
			if (normError < _options.RiddersEpsilon)
			{
				break;
			}

			// Check to see if the current gradient estimate is numerically unstable. If so,
			// bail out and return the last stable result.
			if (i > 0)
			{
				double tableauError = ColumnDistance(current, i, previous, i - 1, numResiduals);
				if (tableauError >= 2 * normError)
				{
					break;
				}
			}

			(current, previous) = (previous, current);
		}

		return true;
	}

	// NumericDiff::EvaluateJacobianColumn with the CENTRAL rule Ridders uses:
	// (f(x + delta) - f(x - delta)) * (1 / delta / 2).
	private bool EvaluateCentralColumn(
		ArraySegment<double>[] parameters, double[] xPlusDelta, double[] x, int parameterIndex, double delta, double[] temp, Span<double> column)
	{
		xPlusDelta[parameterIndex] = x[parameterIndex] + delta;
		if (!_function.Evaluate(parameters, column, default))
		{
			xPlusDelta[parameterIndex] = x[parameterIndex];
			return false;
		}

		double oneOverDelta = 1.0 / delta;
		xPlusDelta[parameterIndex] = x[parameterIndex] - delta;
		if (!_function.Evaluate(parameters, temp, default))
		{
			xPlusDelta[parameterIndex] = x[parameterIndex];
			return false;
		}

		for (int r = 0; r < column.Length; r++)
		{
			column[r] -= temp[r];
		}

		oneOverDelta /= 2;
		xPlusDelta[parameterIndex] = x[parameterIndex];
		for (int r = 0; r < column.Length; r++)
		{
			column[r] *= oneOverDelta;
		}

		return true;
	}

	private static double ColumnDistance(double[] a, int colA, double[] b, int colB, int rows)
	{
		double sum = 0;
		for (int r = 0; r < rows; r++)
		{
			double d = a[colA * rows + r] - b[colB * rows + r];
			sum += d * d;
		}

		return Math.Sqrt(sum);
	}
}

/// <summary>
/// ceres::GradientChecker: compares a cost function's Jacobians against Ridders numeric
/// differentiation, entry by entry, with a relative tolerance.
/// </summary>
public sealed class GradientChecker
{
	private readonly CostFunction _function;
	private readonly RiddersNumericDiffCostFunction _finiteDiffCostFunction;

	/// <summary>Creates a checker for <paramref name="function"/> (no manifolds).</summary>
	public GradientChecker(CostFunction function, NumericDiffOptions options)
	{
		_function = function;
		_finiteDiffCostFunction = new RiddersNumericDiffCostFunction(function, options);
	}

	/// <summary>ceres::GradientChecker::ProbeResults.</summary>
	public sealed class ProbeResults
	{
		/// <summary>The status of the probe.</summary>
		public bool ReturnValue { get; set; }

		/// <summary>The residuals computed by the cost function under test.</summary>
		public double[] Residuals { get; set; } = [];

		/// <summary>Row-major Jacobians of the cost function, one per block.</summary>
		public double[][] Jacobians { get; set; } = [];

		/// <summary>Row-major numeric Jacobians, one per block.</summary>
		public double[][] NumericJacobians { get; set; } = [];

		/// <summary>The maximum relative error found in any Jacobian entry.</summary>
		public double MaximumRelativeError { get; set; }

		/// <summary>Why the probe failed, if it did.</summary>
		public string ErrorLog { get; set; } = string.Empty;
	}

	/// <summary>
	/// ceres::internal::IsClose: whether x and y agree within the relative precision. If
	/// either is exactly zero the relative difference is meaningless, so the absolute
	/// difference is compared instead.
	/// </summary>
	public static bool IsClose(double x, double y, double relativePrecision, out double relativeError, out double absoluteError)
	{
		absoluteError = Math.Abs(x - y);
		relativeError = absoluteError / Math.Max(Math.Abs(x), Math.Abs(y));
		if (x == 0 || y == 0)
		{
			relativeError = absoluteError;
		}

		return relativeError < Math.Abs(relativePrecision);
	}

	/// <summary>
	/// ceres::GradientChecker::Probe: evaluates both Jacobians at <paramref name="parameters"/>
	/// and returns whether every entry agrees within <paramref name="relativePrecision"/>
	/// and the residuals with and without Jacobians agree.
	/// </summary>
	public bool Probe(ReadOnlySpan<ArraySegment<double>> parameters, double relativePrecision, ProbeResults results)
	{
		int numResiduals = _function.NumResiduals;
		results.MaximumRelativeError = 0.0;
		results.ReturnValue = true;
		results.ErrorLog = string.Empty;

		// Evaluate the derivative using the user supplied code.
		if (!EvaluateCostFunction(_function, parameters, out double[] residuals, out double[][] jacobians))
		{
			results.ErrorLog = "Function evaluation with Jacobians failed.";
			results.ReturnValue = false;
		}

		results.Residuals = residuals;
		results.Jacobians = jacobians;

		// Evaluate the derivative using numeric derivatives.
		if (!EvaluateCostFunction(_finiteDiffCostFunction, parameters, out double[] finiteDiffResiduals, out double[][] numericJacobians))
		{
			results.ErrorLog += "\nFunction evaluation with numerical differentiation failed.";
			results.ReturnValue = false;
		}

		results.NumericJacobians = numericJacobians;
		if (!results.ReturnValue)
		{
			return false;
		}

		for (int i = 0; i < numResiduals; i++)
		{
			if (!IsClose(residuals[i], finiteDiffResiduals[i], relativePrecision, out _, out _))
			{
				results.ErrorLog = "Function evaluation with and without Jacobians resulted in different residuals.";
				return false;
			}
		}

		// See if any elements have relative error larger than the threshold.
		int numBadJacobianComponents = 0;
		double worstRelativeError = 0;
		var errorLog = new StringBuilder();
		ReadOnlySpan<int> sizes = _function.ParameterBlockSizes;
		for (int k = 0; k < sizes.Length; k++)
		{
			errorLog.Append(CultureInfo.InvariantCulture, $"========== Jacobian for block {k}: ({numResiduals} by {sizes[k]})) ==========\n");
			for (int i = 0; i < numResiduals; i++)
			{
				for (int j = 0; j < sizes[k]; j++)
				{
					double termJacobian = jacobians[k][i * sizes[k] + j];
					double finiteJacobian = numericJacobians[k][i * sizes[k] + j];
					bool bad = !IsClose(termJacobian, finiteJacobian, relativePrecision, out double relativeError, out double absoluteError);
					worstRelativeError = Math.Max(worstRelativeError, relativeError);
					errorLog.Append(CultureInfo.InvariantCulture, $"{k} {i} {j} user {termJacobian:G17} num {finiteJacobian:G17} abs {absoluteError:G6} rel {relativeError:G6}");
					if (bad)
					{
						numBadJacobianComponents++;
						errorLog.Append(CultureInfo.InvariantCulture, $" ------ ({k},{i},{j}) Relative error worse than {relativePrecision}");
					}

					errorLog.Append('\n');
				}
			}
		}

		results.MaximumRelativeError = worstRelativeError;

		// Since there were some bad errors, dump comprehensive debug info.
		if (numBadJacobianComponents > 0)
		{
			results.ErrorLog = string.Create(
				CultureInfo.InvariantCulture,
				$"\nDetected {numBadJacobianComponents} bad Jacobian component(s). Worst relative error was {worstRelativeError}.\n\n{errorLog}");
			return false;
		}

		return true;
	}

	// EvaluateCostFunction of gradient_checker.cc without manifolds: zeroed residuals and
	// Jacobians, then one evaluation with every Jacobian requested.
	private static bool EvaluateCostFunction(
		CostFunction function, ReadOnlySpan<ArraySegment<double>> parameters, out double[] residuals, out double[][] jacobians)
	{
		ReadOnlySpan<int> sizes = function.ParameterBlockSizes;
		int numResiduals = function.NumResiduals;
		jacobians = new double[sizes.Length][];
		var jacobianSegments = new ArraySegment<double>[sizes.Length];
		for (int b = 0; b < sizes.Length; b++)
		{
			jacobians[b] = new double[numResiduals * sizes[b]];
			jacobianSegments[b] = new ArraySegment<double>(jacobians[b]);
		}

		residuals = new double[numResiduals];
		return function.Evaluate(parameters, residuals, jacobianSegments);
	}
}
