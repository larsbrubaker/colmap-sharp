// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/residual_block.cc and
// internal/ceres/corrector.cc (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// A residual block ties a CostFunction and an optional LossFunction to its parameter blocks
// and evaluates them the way the solver needs: residuals, the cost 1/2 rho(|f|^2), and
// Jacobians in the tangent space (the cost function's ambient Jacobian times the manifold's
// Plus Jacobian), all corrected for the robust loss (Corrector, Triggs et al., "Bundle
// Adjustment - A Modern Synthesis", 1999, section 4.3, as Ceres implements it).
// ProgramEvaluator.cs calls Evaluate from several threads at once: each call only writes its
// own residual and Jacobian slots and the caller's scratch.

using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>ceres::internal::ResidualBlock.</summary>
internal sealed class ResidualBlock
{
	/// <summary>Creates a residual block over the given parameter blocks.</summary>
	public ResidualBlock(CostFunction costFunction, LossFunction? lossFunction, ParameterBlock[] parameterBlocks, int index)
	{
		CostFunction = costFunction;
		LossFunction = lossFunction;
		ParameterBlocks = parameterBlocks;
		Index = index;
	}

	/// <summary>The cost function.</summary>
	public CostFunction CostFunction { get; }

	/// <summary>The robust loss, or null for plain least squares.</summary>
	public LossFunction? LossFunction { get; }

	/// <summary>The parameter blocks, in the cost function's order.</summary>
	public ParameterBlock[] ParameterBlocks { get; }

	/// <summary>Position in the owning program.</summary>
	public int Index { get; set; }

	/// <summary>Number of residuals.</summary>
	public int NumResiduals => CostFunction.NumResiduals;

	/// <summary>Number of parameter blocks.</summary>
	public int NumParameterBlocks => ParameterBlocks.Length;

	/// <summary>
	/// Scratch Evaluate needs: room for the residuals when the caller does not want them, plus
	/// an ambient Jacobian for every block with a manifold.
	/// </summary>
	public int NumScratchDoublesForEvaluate()
	{
		int scratchDoubles = 1;
		foreach (ParameterBlock block in ParameterBlocks)
		{
			if (block.PlusJacobian is not null)
			{
				scratchDoubles += block.Size;
			}
		}

		return scratchDoubles * NumResiduals;
	}

	/// <summary>
	/// Evaluates the block at its parameter blocks' current State. <paramref name="residuals"/>
	/// is empty when the caller only wants the cost; <paramref name="jacobians"/> is empty for
	/// no Jacobians, otherwise one row-major NumResiduals x TangentSize view per block, with a
	/// default (null Array) view to skip a block. <paramref name="parameterViews"/> and
	/// <paramref name="evalJacobians"/> are caller scratch of at least NumParameterBlocks
	/// entries, <paramref name="scratch"/> at least <see cref="NumScratchDoublesForEvaluate"/>.
	/// Returns false if the cost function fails or leaves a non-finite or unwritten value.
	/// </summary>
	public bool Evaluate(
		bool applyLossFunction,
		out double cost,
		Span<double> residuals,
		ReadOnlySpan<ArraySegment<double>> jacobians,
		double[] scratch,
		ArraySegment<double>[] parameterViews,
		ArraySegment<double>[] evalJacobians)
	{
		cost = 0.0;
		int numParameterBlocks = NumParameterBlocks;
		int numResiduals = NumResiduals;
		for (int i = 0; i < numParameterBlocks; i++)
		{
			parameterViews[i] = ParameterBlocks[i].State;
		}

		// Blocks with a manifold get their ambient Jacobian in scratch; the product with the
		// Plus Jacobian then lands in the caller's tangent-space view.
		int scratchOffset = 0;
		bool wantJacobians = !jacobians.IsEmpty;
		if (wantJacobians)
		{
			for (int i = 0; i < numParameterBlocks; i++)
			{
				ParameterBlock block = ParameterBlocks[i];
				if (jacobians[i].Array is not null && block.PlusJacobian is not null)
				{
					evalJacobians[i] = new ArraySegment<double>(scratch, scratchOffset, numResiduals * block.Size);
					scratchOffset += numResiduals * block.Size;
				}
				else
				{
					evalJacobians[i] = jacobians[i].Array is null
						? default
						: jacobians[i].Slice(0, numResiduals * block.TangentSize);
				}
			}
		}

		bool outputtingResiduals = !residuals.IsEmpty;
		if (!outputtingResiduals)
		{
			residuals = scratch.AsSpan(scratchOffset, numResiduals);
		}
		else
		{
			residuals = residuals[..numResiduals];
		}

		ReadOnlySpan<ArraySegment<double>> evalJacobianSpan =
			wantJacobians ? evalJacobians.AsSpan(0, numParameterBlocks) : ReadOnlySpan<ArraySegment<double>>.Empty;

		// InvalidateEvaluation: a value the cost function leaves unwritten is caught below.
		ArrayValidity.Invalidate(residuals);
		foreach (ArraySegment<double> jacobian in evalJacobianSpan)
		{
			if (jacobian.Array is not null)
			{
				ArrayValidity.Invalidate(jacobian);
			}
		}

		if (!CostFunction.Evaluate(parameterViews.AsSpan(0, numParameterBlocks), residuals, evalJacobianSpan))
		{
			return false;
		}

		if (!IsEvaluationValid(residuals, evalJacobianSpan))
		{
			return false;
		}

		double squaredNorm = 0.0;
		foreach (double r in residuals)
		{
			squaredNorm += r * r;
		}

		if (wantJacobians)
		{
			for (int i = 0; i < numParameterBlocks; i++)
			{
				ParameterBlock block = ParameterBlocks[i];
				if (jacobians[i].Array is not null && block.PlusJacobian is not null)
				{
					MultiplyByPlusJacobian(evalJacobians[i], numResiduals, block, jacobians[i]);
				}
			}
		}

		if (LossFunction is null || !applyLossFunction)
		{
			cost = 0.5 * squaredNorm;
			return true;
		}

		Span<double> rho = stackalloc double[3];
		LossFunction.Evaluate(squaredNorm, rho);
		cost = 0.5 * rho[0];

		if (!wantJacobians && !outputtingResiduals)
		{
			return true;
		}

		var corrector = new Corrector(squaredNorm, rho);
		if (wantJacobians)
		{
			for (int i = 0; i < numParameterBlocks; i++)
			{
				if (jacobians[i].Array is not null)
				{
					int tangentSize = ParameterBlocks[i].TangentSize;
					corrector.CorrectJacobian(numResiduals, tangentSize, residuals, jacobians[i].AsSpan(0, numResiduals * tangentSize));
				}
			}
		}

		if (outputtingResiduals)
		{
			corrector.CorrectResiduals(residuals);
		}

		return true;
	}

	private static bool IsEvaluationValid(ReadOnlySpan<double> residuals, ReadOnlySpan<ArraySegment<double>> jacobians)
	{
		if (!ArrayValidity.IsValid(residuals))
		{
			return false;
		}

		foreach (ArraySegment<double> jacobian in jacobians)
		{
			if (jacobian.Array is not null && !ArrayValidity.IsValid(jacobian))
			{
				return false;
			}
		}

		return true;
	}

	// tangent (r x t) = ambient (r x s) * plus_jacobian (s x t), all row-major; each entry a
	// left-to-right sum over the ambient index.
	private static void MultiplyByPlusJacobian(
		ReadOnlySpan<double> ambient, int numResiduals, ParameterBlock block, Span<double> tangent)
	{
		int size = block.Size;
		int tangentSize = block.TangentSize;
		double[] plus = block.PlusJacobian!;
		for (int r = 0; r < numResiduals; r++)
		{
			for (int c = 0; c < tangentSize; c++)
			{
				double sum = 0.0;
				for (int k = 0; k < size; k++)
				{
					sum += ambient[r * size + k] * plus[k * tangentSize + c];
				}

				tangent[r * tangentSize + c] = sum;
			}
		}
	}
}

/// <summary>
/// ceres::internal::Corrector: rescales a residual block's residuals and Jacobian so that the
/// Gauss-Newton model of rho(|f|^2)/2 matches its gradient and (a curvature-correct
/// approximation of) its Hessian.
/// </summary>
internal readonly struct Corrector
{
	private readonly double sqrtRho1;
	private readonly double residualScaling;
	private readonly double alphaSqNorm;

	/// <summary>Builds the correction for squared norm s and rho = (rho, rho', rho'').</summary>
	public Corrector(double sqNorm, ReadOnlySpan<double> rho)
	{
		Check.Ge(sqNorm, 0.0);
		sqrtRho1 = Math.Sqrt(rho[1]);

		// If sq_norm = 0.0, the correction becomes trivially zero. Ceres also skips the
		// correction when rho'' <= 0 (the common robust-loss outlier region), where the
		// alpha below would need the square root of something that can go negative; there
		// the Jacobian is only scaled, which is always a valid (if cruder) model.
		if (sqNorm == 0.0 || rho[2] <= 0.0)
		{
			residualScaling = sqrtRho1;
			alphaSqNorm = 0.0;
			return;
		}

		// We now require that the first derivative of the loss function be positive only if
		// the second derivative is positive. This is because the loss functions do not
		// satisfy rho' > 0 everywhere (Tukey).
		Check.Gt(rho[1], 0.0);

		// Calculate the smaller of the two solutions to the equation
		//   0.5 * alpha^2 - alpha - rho'' / rho' * z'z = 0.
		// Start by calculating the discriminant D.
		double d = 1.0 + 2.0 * sqNorm * rho[2] / rho[1];

		// Since both rho[1] and rho[2] are guaranteed to be positive at this point, we know
		// that D > 1.0.
		double alpha = 1.0 - Math.Sqrt(d);

		// Calculate the constants needed by the correction routines.
		residualScaling = sqrtRho1 / (1 - alpha);
		alphaSqNorm = alpha / sqNorm;
	}

	/// <summary>residuals *= sqrt(rho') / (1 - alpha).</summary>
	public void CorrectResiduals(Span<double> residuals)
	{
		for (int i = 0; i < residuals.Length; i++)
		{
			residuals[i] *= residualScaling;
		}
	}

	/// <summary>J = sqrt(rho') (I - alpha r r' / |r|^2) J for a row-major rows x cols J.</summary>
	public void CorrectJacobian(int numRows, int numCols, ReadOnlySpan<double> residuals, Span<double> jacobian)
	{
		// The common case (rho[2] <= 0).
		if (alphaSqNorm == 0.0)
		{
			for (int i = 0; i < numRows * numCols; i++)
			{
				jacobian[i] *= sqrtRho1;
			}

			return;
		}

		// Equation 10 of the Ceres derivation: J = sqrt(rho) * (J - alpha^2 r * r' J).
		for (int c = 0; c < numCols; c++)
		{
			double rTransposeJ = 0.0;
			for (int r = 0; r < numRows; r++)
			{
				rTransposeJ += jacobian[r * numCols + c] * residuals[r];
			}

			for (int r = 0; r < numRows; r++)
			{
				jacobian[r * numCols + c] = sqrtRho1 * (jacobian[r * numCols + c] - alphaSqNorm * residuals[r] * rTransposeJ);
			}
		}
	}
}
