// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 include/ceres/tiny_solver_autodiff_function.h
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Adapts an autodiff functor (IAutoDiffFunctor, the same contract AutoDiffCostFunction.cs
// wraps) to Optim/TinySolver.cs's ITinySolverFunction: with no Jacobian requested it runs
// the functor on doubles (Real); otherwise on Jet<TGrad> with every parameter a variable,
// one pass, and copies the derivative parts out into the column-major Jacobian.
//
// Ceres' template arguments <CostFunctor, kNumResiduals, kNumParameters> become:
// kNumParameters = TGrad.Length (Ceres' jet is Jet<T, kNumParameters>, so the gradient width
// *is* the parameter count), and kNumResiduals is a constructor argument (Ceres reads the
// functor's NumResiduals() when it is Eigen::Dynamic). Like Ceres, the Jet buffers are
// members reused across calls, so one instance is not thread safe.

using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>
/// Port of ceres::TinySolverAutoDiffFunction: an <see cref="ITinySolverFunction"/> whose
/// Jacobian comes from evaluating <typeparamref name="TFunctor"/> on
/// <see cref="Jet{TGrad}"/>, with <c>TGrad.Length</c> parameters.
/// </summary>
public sealed class TinySolverAutoDiffFunction<TFunctor, TGrad> : ITinySolverFunction
	where TFunctor : IAutoDiffFunctor
	where TGrad : unmanaged, IJetGradient
{
	// ceres::kImpossibleValue: residual Jets start as this so that a functor that forgets to
	// write a residual is visible rather than reading stale values.
	private const double ImpossibleValue = 1e302;

	// Not readonly: a struct functor in a readonly field would be defensively copied per call.
	private TFunctor costFunctor;

	private readonly Jet<TGrad>[] jetParameters;
	private readonly Jet<TGrad>[] jetResiduals;
	private readonly Jet<TGrad> impossibleJet;

	/// <summary>Wraps <paramref name="costFunctor"/>, which produces <paramref name="numResiduals"/> residuals.</summary>
	public TinySolverAutoDiffFunction(TFunctor costFunctor, int numResiduals)
	{
		Check.Ge(numResiduals, 0);
		this.costFunctor = costFunctor;
		NumResiduals = numResiduals;
		jetParameters = new Jet<TGrad>[TGrad.Length];
		jetResiduals = new Jet<TGrad>[numResiduals];
		Span<double> impossible = stackalloc double[TGrad.Length];
		impossible.Fill(ImpossibleValue);
		impossibleJet = Jet<TGrad>.Create(ImpossibleValue, impossible);
	}

	/// <summary>Ceres' NUM_PARAMETERS = kNumParameters, the Jet width.</summary>
	public static int NumParameters => TGrad.Length;

	/// <inheritdoc/>
	public int NumResiduals { get; }

	/// <summary>The wrapped functor.</summary>
	public TFunctor Functor => costFunctor;

	/// <inheritdoc/>
	public bool Evaluate(ReadOnlySpan<double> parameters, Span<double> residuals, Span<double> jacobian)
	{
		int numParameters = TGrad.Length;
		if (jacobian.IsEmpty)
		{
			// No Jacobian requested, so just directly call the cost function with doubles,
			// skipping jets and derivatives.
			return costFunctor.Evaluate(Real.Cast(parameters[..numParameters]), Real.CastWritable(residuals[..NumResiduals]));
		}

		// Initialize the input jets with the passed parameters.
		for (int i = 0; i < numParameters; ++i)
		{
			jetParameters[i] = Jet<TGrad>.Variable(parameters[i], i);
		}

		// Initialize the output jets such that we can detect user errors.
		for (int i = 0; i < NumResiduals; ++i)
		{
			jetResiduals[i] = impossibleJet;
		}

		// Execute the cost function, but with jets to find the derivative.
		if (!costFunctor.Evaluate<Jet<TGrad>>(jetParameters, jetResiduals))
		{
			return false;
		}

		// Copy the Jacobian out of the derivative part of the residual jets.
		int n = NumResiduals;
		for (int r = 0; r < n; ++r)
		{
			residuals[r] = jetResiduals[r].A;
			for (int c = 0; c < numParameters; ++c)
			{
				jacobian[c * n + r] = jetResiduals[r].Derivative(c);
			}
		}

		return true;
	}
}
