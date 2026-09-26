// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/program.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// A Program is an ordered list of parameter blocks and residual blocks. Problem.cs owns the
// full one; LeastSquaresSolver.cs builds the reduced one the minimizer works on (constant
// blocks and residual blocks that only touch them removed, their cost folded into
// fixed_cost) and lays out the state vector (ambient sizes) and the delta vector (tangent
// sizes) in block order. ProgramEvaluator.cs evaluates it. The bounds checks
// (IsBoundsConstrained, IsFeasible) and IsParameterBlockSetIndependent serve the
// preprocessing in LeastSquaresSolver.cs and SchurOrdering.cs.

using System.Globalization;

namespace ColmapSharp.Solver;

/// <summary>ceres::internal::Program.</summary>
internal sealed class Program
{
	/// <summary>Creates an empty program.</summary>
	public Program()
	{
	}

	/// <summary>A copy of <paramref name="other"/>'s block lists (Ceres' copy constructor).</summary>
	internal Program(Program other)
	{
		ParameterBlocks = [.. other.ParameterBlocks];
		ResidualBlocks = [.. other.ResidualBlocks];
	}

	/// <summary>The parameter blocks, in state-vector order.</summary>
	public List<ParameterBlock> ParameterBlocks { get; } = [];

	/// <summary>The residual blocks, in residual-vector order.</summary>
	public List<ResidualBlock> ResidualBlocks { get; } = [];

	/// <summary>Total ambient size of the parameter blocks.</summary>
	public int NumParameters
	{
		get
		{
			int n = 0;
			foreach (ParameterBlock block in ParameterBlocks)
			{
				n += block.Size;
			}

			return n;
		}
	}

	/// <summary>Total tangent size of the parameter blocks.</summary>
	public int NumEffectiveParameters
	{
		get
		{
			int n = 0;
			foreach (ParameterBlock block in ParameterBlocks)
			{
				n += block.TangentSize;
			}

			return n;
		}
	}

	/// <summary>Total number of residuals.</summary>
	public int NumResiduals
	{
		get
		{
			int n = 0;
			foreach (ResidualBlock block in ResidualBlocks)
			{
				n += block.NumResiduals;
			}

			return n;
		}
	}

	/// <summary>Largest scratch any residual block needs for Evaluate.</summary>
	public int MaxScratchDoublesNeededForEvaluate()
	{
		int max = 0;
		foreach (ResidualBlock block in ResidualBlocks)
		{
			max = Math.Max(max, block.NumScratchDoublesForEvaluate());
		}

		return max;
	}

	/// <summary>Most residuals in one residual block.</summary>
	public int MaxResidualsPerResidualBlock()
	{
		int max = 0;
		foreach (ResidualBlock block in ResidualBlocks)
		{
			max = Math.Max(max, block.NumResiduals);
		}

		return max;
	}

	/// <summary>Most parameter blocks in one residual block.</summary>
	public int MaxParametersPerResidualBlock()
	{
		int max = 0;
		foreach (ResidualBlock block in ResidualBlocks)
		{
			max = Math.Max(max, block.NumParameterBlocks);
		}

		return max;
	}

	/// <summary>
	/// Program::CreateReducedProgram: a copy without constant parameter blocks and without
	/// the residual blocks that depend only on constant ones, whose cost (with the loss) goes
	/// to <paramref name="fixedCost"/>. Returns null with <paramref name="error"/> set if such a
	/// residual block fails to evaluate.
	/// </summary>
	public Program? CreateReducedProgram(List<ParameterBlock> removedParameterBlocks, out double fixedCost, out string error)
	{
		var reduced = new Program(this);
		if (!reduced.RemoveFixedBlocks(removedParameterBlocks, out fixedCost, out error))
		{
			return null;
		}

		reduced.SetParameterOffsetsAndIndex();
		return reduced;
	}

	private bool RemoveFixedBlocks(List<ParameterBlock> removedParameterBlocks, out double fixedCost, out string error)
	{
		error = string.Empty;
		removedParameterBlocks.Clear();
		fixedCost = 0.0;
		var scratch = new double[MaxScratchDoublesNeededForEvaluate()];
		int maxBlocks = MaxParametersPerResidualBlock();
		var views = new ArraySegment<double>[maxBlocks];
		var evalJacobians = new ArraySegment<double>[maxBlocks];

		// Mark all the parameters as unused. Abuse the index member of the parameter blocks
		// for the marking.
		foreach (ParameterBlock block in ParameterBlocks)
		{
			block.Index = -1;
		}

		// Filter out residual blocks that have all-constant parameters, and mark all the
		// parameter blocks that appear in residuals.
		int numActiveResidualBlocks = 0;
		for (int i = 0; i < ResidualBlocks.Count; i++)
		{
			ResidualBlock residualBlock = ResidualBlocks[i];
			bool allConstant = true;
			foreach (ParameterBlock parameterBlock in residualBlock.ParameterBlocks)
			{
				if (!parameterBlock.IsConstant)
				{
					allConstant = false;
					parameterBlock.Index = 1;
				}
			}

			if (!allConstant)
			{
				ResidualBlocks[numActiveResidualBlocks++] = residualBlock;
				continue;
			}

			// The residual is constant and will be removed, so its cost is added to the
			// variable fixed_cost.
			if (!residualBlock.Evaluate(true, out double cost, [], [], scratch, views, evalJacobians))
			{
				error = string.Format(
					CultureInfo.InvariantCulture,
					"Evaluation of the residual {0} failed during removal of fixed residual blocks.",
					i);
				return false;
			}

			fixedCost += cost;
		}

		ResidualBlocks.RemoveRange(numActiveResidualBlocks, ResidualBlocks.Count - numActiveResidualBlocks);

		// Filter out unused or fixed parameter blocks.
		int numActiveParameterBlocks = 0;
		for (int i = 0; i < ParameterBlocks.Count; i++)
		{
			ParameterBlock parameterBlock = ParameterBlocks[i];
			if (parameterBlock.Index != -1)
			{
				ParameterBlocks[numActiveParameterBlocks++] = parameterBlock;
			}
			else
			{
				removedParameterBlocks.Add(parameterBlock);
			}
		}

		ParameterBlocks.RemoveRange(numActiveParameterBlocks, ParameterBlocks.Count - numActiveParameterBlocks);
		return true;
	}

	/// <summary>Assigns each block its index and state/delta offsets, in list order.</summary>
	public void SetParameterOffsetsAndIndex()
	{
		// Set positions for all parameters appearing as arguments to residuals to one past
		// the end of the parameter block array.
		foreach (ResidualBlock residualBlock in ResidualBlocks)
		{
			foreach (ParameterBlock parameterBlock in residualBlock.ParameterBlocks)
			{
				parameterBlock.Index = -1;
			}
		}

		// For parameters that appear in the program, set their position and offset.
		int stateOffset = 0;
		int deltaOffset = 0;
		for (int i = 0; i < ParameterBlocks.Count; i++)
		{
			ParameterBlock block = ParameterBlocks[i];
			block.Index = i;
			block.StateOffset = stateOffset;
			block.DeltaOffset = deltaOffset;
			stateOffset += block.Size;
			deltaOffset += block.TangentSize;
		}

		for (int i = 0; i < ResidualBlocks.Count; i++)
		{
			ResidualBlocks[i].Index = i;
		}
	}

	/// <summary>Points every variable block at its slice of <paramref name="state"/>.</summary>
	public bool StateVectorToParameterBlocks(double[] state)
	{
		int offset = 0;
		foreach (ParameterBlock block in ParameterBlocks)
		{
			if (!block.IsConstant && !block.SetState(new ArraySegment<double>(state, offset, block.Size)))
			{
				return false;
			}

			offset += block.Size;
		}

		return true;
	}

	/// <summary>Copies every block's current state into <paramref name="state"/>.</summary>
	public void ParameterBlocksToStateVector(double[] state)
	{
		int offset = 0;
		foreach (ParameterBlock block in ParameterBlocks)
		{
			block.GetState(new ArraySegment<double>(state, offset, block.Size));
			offset += block.Size;
		}
	}

	/// <summary>Writes every block's current state back to the user's memory.</summary>
	public void CopyParameterBlockStateToUserState()
	{
		foreach (ParameterBlock block in ParameterBlocks)
		{
			block.GetState(block.UserState);
		}
	}

	/// <summary>Points every block back at the user's memory (refreshing Plus Jacobians).</summary>
	public bool SetParameterBlockStatePtrsToUserStatePtrs()
	{
		foreach (ParameterBlock block in ParameterBlocks)
		{
			if (!block.IsConstant)
			{
				if (!block.SetState(block.UserState))
				{
					return false;
				}
			}
			else
			{
				block.ResetStateToUserState();
			}
		}

		return true;
	}

	/// <summary>
	/// state_plus_delta = Plus(state, delta), block by block. Each block writes only its own
	/// slice, so the blocks run in parallel with results independent of the thread count.
	/// </summary>
	public bool Plus(double[] state, double[] delta, double[] statePlusDelta, int numThreads)
	{
		List<ParameterBlock> blocks = ParameterBlocks;
		if (numThreads <= 1 || blocks.Count < 2)
		{
			foreach (ParameterBlock block in blocks)
			{
				if (!PlusBlock(block, state, delta, statePlusDelta))
				{
					return false;
				}
			}

			return true;
		}

		int failed = 0;
		Parallel.For(
			0,
			blocks.Count,
			new ParallelOptions { MaxDegreeOfParallelism = numThreads },
			i =>
			{
				if (Volatile.Read(ref failed) == 0 && !PlusBlock(blocks[i], state, delta, statePlusDelta))
				{
					Volatile.Write(ref failed, 1);
				}
			});
		return failed == 0;
	}

	private static bool PlusBlock(ParameterBlock block, double[] state, double[] delta, double[] statePlusDelta) =>
		block.Plus(
			state.AsSpan(block.StateOffset, block.Size),
			delta.AsSpan(block.DeltaOffset, block.TangentSize),
			statePlusDelta.AsSpan(block.StateOffset, block.Size));

	/// <summary>Program::IsBoundsConstrained: some variable block has a finite bound.</summary>
	public bool IsBoundsConstrained()
	{
		foreach (ParameterBlock block in ParameterBlocks)
		{
			if (block.IsConstant)
			{
				continue;
			}

			for (int j = 0; j < block.Size; j++)
			{
				if (block.LowerBoundForParameter(j) > -double.MaxValue || block.UpperBoundForParameter(j) < double.MaxValue)
				{
					return true;
				}
			}
		}

		return false;
	}

	/// <summary>
	/// Program::IsFeasible: constant blocks must start inside their bounds (Ceres cannot move
	/// them), and variable blocks must have a non-empty box.
	/// </summary>
	public bool IsFeasible(out string error)
	{
		error = string.Empty;
		foreach (ParameterBlock block in ParameterBlocks)
		{
			ReadOnlySpan<double> parameters = block.UserState.AsSpan();
			for (int j = 0; j < block.Size; j++)
			{
				double lowerBound = block.LowerBoundForParameter(j);
				double upperBound = block.UpperBoundForParameter(j);
				if (block.IsConstant && (parameters[j] < lowerBound || parameters[j] > upperBound))
				{
					error = string.Format(
						CultureInfo.InvariantCulture,
						"ParameterBlock with size {0} has at least one infeasible value.\nFirst infeasible value is at index: {1}.\nLower bound: {2}, value: {3}, upper bound: {4}\nParameter block values: {5}",
						block.Size,
						j,
						SolverSummary.FormatE(lowerBound),
						SolverSummary.FormatE(parameters[j]),
						SolverSummary.FormatE(upperBound),
						FormatValues(parameters));
					return false;
				}

				if (!block.IsConstant && lowerBound >= upperBound)
				{
					error = string.Format(
						CultureInfo.InvariantCulture,
						"ParameterBlock with size {0} has at least one infeasible bound.\nFirst infeasible bound is at index: {1}.\nLower bound: {2}, upper bound: {3}\nParameter block values: {4}",
						block.Size,
						j,
						SolverSummary.FormatE(lowerBound),
						SolverSummary.FormatE(upperBound),
						FormatValues(parameters));
					return false;
				}
			}
		}

		return true;
	}

	/// <summary>
	/// Program::IsParameterBlockSetIndependent: no residual block touches two blocks of
	/// <paramref name="independentSet"/>.
	/// </summary>
	public bool IsParameterBlockSetIndependent(IReadOnlySet<ParameterBlock> independentSet)
	{
		foreach (ResidualBlock residualBlock in ResidualBlocks)
		{
			int count = 0;
			foreach (ParameterBlock parameterBlock in residualBlock.ParameterBlocks)
			{
				count += independentSet.Contains(parameterBlock) ? 1 : 0;
			}

			if (count > 1)
			{
				return false;
			}
		}

		return true;
	}

	// AppendArrayToString: each value as "%12g ", uninitialized ones as "Uninitialized ".
	private static string FormatValues(ReadOnlySpan<double> values)
	{
		var text = new System.Text.StringBuilder();
		foreach (double value in values)
		{
			text.Append(value == ArrayValidity.ImpossibleValue
				? "Uninitialized "
				: Util.CppStreamFormat.FormatDouble(value).PadLeft(12) + " ");
		}

		return text.ToString();
	}

	/// <summary>Program::ParameterBlocksAreFinite: every parameter value is finite.</summary>
	public bool ParameterBlocksAreFinite(out string error)
	{
		error = string.Empty;
		foreach (ParameterBlock block in ParameterBlocks)
		{
			ReadOnlySpan<double> values = block.UserState.AsSpan();
			for (int i = 0; i < values.Length; i++)
			{
				if (!double.IsFinite(values[i]))
				{
					error = string.Format(
						CultureInfo.InvariantCulture,
						"ParameterBlock with size {0} has at least one invalid value.\nFirst invalid value is at index: {1}.",
						block.Size,
						i);
					return false;
				}
			}
		}

		return true;
	}
}
