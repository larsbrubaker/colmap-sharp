// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 include/ceres/cost_function.h,
// include/ceres/autodiff_cost_function.h and include/ceres/dynamic_autodiff_cost_function.h
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The residual-block contract bundle adjustment and every other COLMAP refinement is
// written against:
// - CostFunction is ceres::CostFunction: NumResiduals, the parameter block sizes, and
//   Evaluate(parameters, residuals, jacobians). Ceres passes `double const* const*` and
//   `double**`; here each block is an ArraySegment<double>, a (array, offset, count) view.
//   A segment can cover a whole per-block array (a parameter block the Problem owns) or a
//   slice of one shared per-thread buffer (the evaluator's Jacobian scratch), so the
//   evaluator can lay blocks out however it likes without allocating: the span of segments
//   lives on its stack or in a reused array. Jacobians are row-major
//   NumResiduals x blockSize; a default segment (null Array) skips that block, and an empty
//   span asks for no Jacobians at all. ArraySegment rather than a custom ref struct because
//   it is the BCL's own zero-allocation array view, stores in ordinary arrays for reuse, and
//   converts to Span implicitly.
// - IAutoDiffFunctor is the C# shape of a Ceres functor's templated operator(): one generic
//   method over T : IScalar<T>, given every parameter block concatenated in order (the
//   functor slices them at the offsets its block sizes imply) and the residual span.
//   C++ `bool operator()(const T* p0, const T* p1, T* r)` becomes
//   `bool Evaluate<T>(ReadOnlySpan<T> p, Span<T> r)` with p0 = p[..3], p1 = p[3..10], ...
//   Make functors structs: the generic method is then specialized per T with no virtual
//   dispatch.
// - AutoDiffCostFunction<TFunctor, TGrad> is ceres::AutoDiffCostFunction. With no Jacobians
//   requested it evaluates the functor on Real (doubles). Otherwise it evaluates on
//   Jet<TGrad>, seeding the variables of the blocks whose Jacobian is wanted TGrad.Length at
//   a time, as ceres::DynamicAutoDiffCostFunction strides; Ceres' fixed-size
//   AutoDiffCostFunction seeds all of them in one Jet of width N, which is the fastest choice
//   here too (JetGradients.cs has every width COLMAP needs). The derivative bits are the same
//   either way, because Jet computes each slot from the value parts and that slot alone
//   (Jet.cs), and the residuals come from the first pass's value parts, as Ceres takes them
//   from the Jets' `a`. Evaluation allocates nothing (the Jets live on the stack up to
//   16 KB), so one instance may be evaluated from several threads, provided the functor's
//   Evaluate does not mutate it (IAutoDiffFunctor's contract).

using System.Runtime.CompilerServices;

using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>ceres::CostFunction: a residual block's residuals and Jacobians.</summary>
public abstract class CostFunction
{
	private readonly int[] parameterBlockSizes;

	/// <summary>Creates a cost function with the given residual count and block sizes.</summary>
	protected CostFunction(int numResiduals, params int[] parameterBlockSizes)
	{
		Check.Gt(numResiduals, 0);
		Check.Gt(parameterBlockSizes.Length, 0);
		foreach (int size in parameterBlockSizes)
		{
			Check.Gt(size, 0);
		}

		NumResiduals = numResiduals;
		this.parameterBlockSizes = (int[])parameterBlockSizes.Clone();
		foreach (int size in parameterBlockSizes)
		{
			NumParameters += size;
		}
	}

	/// <summary>Number of residuals, Ceres' <c>num_residuals()</c>.</summary>
	public int NumResiduals { get; }

	/// <summary>Size of each parameter block, Ceres' <c>parameter_block_sizes()</c>.</summary>
	public ReadOnlySpan<int> ParameterBlockSizes => parameterBlockSizes;

	/// <summary>The total size of all parameter blocks.</summary>
	public int NumParameters { get; }

	/// <summary>
	/// Computes the residuals and, where asked, the Jacobians.
	/// <paramref name="parameters"/> holds one view per block, each at least the block's
	/// size. <paramref name="jacobians"/> is empty for no Jacobians, or holds one view per
	/// block: <c>default</c> (null Array) to skip that block, else room for a row-major
	/// NumResiduals x blockSize matrix. Returns false if the residuals could not be computed
	/// (the solver then rejects the step).
	/// </summary>
	public abstract bool Evaluate(
		ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians);
}

/// <summary>
/// A residual functor generic over the scalar type: the C# form of a Ceres functor's
/// <c>template &lt;typename T&gt; bool operator()(...) const</c>.
/// </summary>
public interface IAutoDiffFunctor
{
	/// <summary>
	/// Computes the residuals from the parameter blocks, concatenated in block order.
	/// Returns false if they cannot be computed. Must not mutate the functor (C++'s
	/// <c>const</c>): one cost function is evaluated from several threads at once, and its
	/// thread safety rests on the functor being read-only.
	/// </summary>
	bool Evaluate<T>(ReadOnlySpan<T> parameters, Span<T> residuals)
		where T : struct, IScalar<T>;
}

/// <summary>
/// ceres::AutoDiffCostFunction: a <see cref="CostFunction"/> whose Jacobians come from
/// evaluating <typeparamref name="TFunctor"/> on <see cref="Jet{TGrad}"/>, differentiating
/// <c>TGrad.Length</c> variables per pass.
/// </summary>
public sealed class AutoDiffCostFunction<TFunctor, TGrad> : CostFunction
	where TFunctor : IAutoDiffFunctor
	where TGrad : unmanaged, IJetGradient
{
	// Stack budget for the parameter and residual Jets together; beyond it they go on the
	// heap. Every COLMAP cost function fits: the largest, 33 parameters and 2 residuals at
	// width 33, is 35 * 34 * 8 = 9520 bytes.
	private const int MaxStackBytes = 16 * 1024;

	// Not readonly: a struct functor in a readonly field would be defensively copied per call.
	private TFunctor functor;

	/// <summary>Wraps a functor with the given residual count and parameter block sizes.</summary>
	public AutoDiffCostFunction(TFunctor functor, int numResiduals, params int[] parameterBlockSizes)
		: base(numResiduals, parameterBlockSizes)
	{
		this.functor = functor;
	}

	/// <summary>The wrapped functor.</summary>
	public TFunctor Functor => functor;

	/// <inheritdoc/>
	public override bool Evaluate(
		ReadOnlySpan<ArraySegment<double>> parameters, Span<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
	{
		ReadOnlySpan<int> sizes = ParameterBlockSizes;
		Check.Eq(parameters.Length, sizes.Length);
		int total = NumParameters;
		int numResiduals = NumResiduals;

		if (jacobians.IsEmpty)
		{
			Span<double> x = total * sizeof(double) <= MaxStackBytes ? stackalloc double[total] : new double[total];
			for (int b = 0, offset = 0; b < sizes.Length; offset += sizes[b], b++)
			{
				parameters[b].AsSpan(0, sizes[b]).CopyTo(x[offset..]);
			}

			return functor.Evaluate(Real.Cast(x), Real.CastWritable(residuals[..numResiduals]));
		}

		Check.Eq(jacobians.Length, sizes.Length);
		int jetCount = total + numResiduals;
		Span<Jet<TGrad>> jets = jetCount * Unsafe.SizeOf<Jet<TGrad>>() <= MaxStackBytes
			? stackalloc Jet<TGrad>[jetCount]
			: new Jet<TGrad>[jetCount];
		Span<Jet<TGrad>> jetX = jets[..total];
		Span<Jet<TGrad>> jetResiduals = jets[total..];

		int numActive = 0;
		for (int b = 0; b < sizes.Length; b++)
		{
			if (jacobians[b].Array is not null)
			{
				numActive += sizes[b];
			}
		}

		int stride = TGrad.Length;

		// At least one pass, even when no block wants a Jacobian, so the residuals still
		// come from Jet value parts as they would in Ceres.
		for (int chunkStart = 0; chunkStart == 0 || chunkStart < numActive; chunkStart += stride)
		{
			Seed(parameters, jacobians, chunkStart, jetX);
			if (!functor.Evaluate<Jet<TGrad>>(jetX, jetResiduals))
			{
				return false;
			}

			if (chunkStart == 0)
			{
				for (int r = 0; r < numResiduals; r++)
				{
					residuals[r] = jetResiduals[r].A;
				}
			}

			ExtractJacobians(jacobians, chunkStart, jetResiduals);
		}

		return true;
	}

	// Loads every parameter as a Jet: the active ones (in blocks whose Jacobian is wanted)
	// with index in [chunkStart, chunkStart + stride) become variables, the rest constants.
	private void Seed(
		ReadOnlySpan<ArraySegment<double>> parameters, ReadOnlySpan<ArraySegment<double>> jacobians, int chunkStart, Span<Jet<TGrad>> jetX)
	{
		ReadOnlySpan<int> sizes = ParameterBlockSizes;
		int stride = TGrad.Length;
		int active = 0;
		for (int b = 0, offset = 0; b < sizes.Length; offset += sizes[b], b++)
		{
			ReadOnlySpan<double> block = parameters[b].AsSpan(0, sizes[b]);
			bool wanted = jacobians[b].Array is not null;
			for (int c = 0; c < block.Length; c++)
			{
				int k = active - chunkStart;
				jetX[offset + c] = wanted && k >= 0 && k < stride
					? Jet<TGrad>.Variable(block[c], k)
					: Jet<TGrad>.FromDouble(block[c]);
				if (wanted)
				{
					active++;
				}
			}
		}
	}

	private void ExtractJacobians(ReadOnlySpan<ArraySegment<double>> jacobians, int chunkStart, ReadOnlySpan<Jet<TGrad>> jetResiduals)
	{
		ReadOnlySpan<int> sizes = ParameterBlockSizes;
		int stride = TGrad.Length;
		int numResiduals = NumResiduals;
		int active = 0;
		for (int b = 0; b < sizes.Length; b++)
		{
			if (jacobians[b].Array is null)
			{
				continue;
			}

			int size = sizes[b];
			Span<double> jacobian = jacobians[b].AsSpan(0, numResiduals * size);
			for (int c = 0; c < size; c++, active++)
			{
				int k = active - chunkStart;
				if (k < 0 || k >= stride)
				{
					continue;
				}

				for (int r = 0; r < numResiduals; r++)
				{
					jacobian[r * size + c] = jetResiduals[r].Derivative(k);
				}
			}
		}
	}
}
