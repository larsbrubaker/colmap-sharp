// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/program_evaluator.h,
// internal/ceres/evaluator.cc, internal/ceres/dense_jacobian_writer.h and
// internal/ceres/block_evaluate_preparer.cc (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Evaluates a reduced Program at a state vector: total cost, residual vector, gradient and
// Jacobian. The residual blocks run in parallel (Parallel.For over num_threads contiguous chunks);
// each one writes only its own residual slots, its own Jacobian cells (block-sparse: in
// place; dense: its rows, copied from per-worker scratch) and its own entry of a per-block
// cost array. Ceres instead sums cost and gradient per thread and then across threads, which
// makes its last bits depend on the thread count and the scheduling. Here the cost is summed
// over residual blocks in order and the gradient is J'r computed after the parallel pass, so
// the result is the same for any thread count (docs/CPP_DIVERGENCES.md, entry 18). A gradient
// asked for without a Jacobian (the bounded line search, Problem.Evaluate) goes through a
// Jacobian and residual vector the evaluator owns, allocated on first use and reused.

using System.Runtime.ExceptionServices;

using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>ceres::internal::ProgramEvaluator for the dense and block-sparse Jacobians.</summary>
internal sealed class ProgramEvaluator
{
	private readonly Program program;
	private readonly int numThreads;
	private readonly bool denseJacobian;
	private readonly BlockJacobianLayout? blockLayout;
	private readonly int[] residualLayout;
	private readonly double[] blockCosts;
	private readonly Scratch[] scratchPool;
	private readonly Exception?[] chunkExceptions;
	private readonly ParallelOptions parallelOptions;
	private readonly Action<int> chunkBody;

	// The outputs of the parallel evaluation in progress (the evaluator is not reentrant).
	private double[]? currentResiduals;
	private SparseMatrix? currentJacobian;
	private bool currentApplyLossFunction = true;
	private int failed;

	// Scratch for a gradient evaluated without the caller's Jacobian or residuals.
	private SparseMatrix? gradientJacobian;
	private double[]? gradientResiduals;

	/// <summary>
	/// Creates the evaluator. <paramref name="denseJacobian"/> selects DenseSparseMatrix
	/// (DENSE_QR, DENSE_NORMAL_CHOLESKY) over BlockSparseMatrix (the sparse and Schur solvers);
	/// <paramref name="numEliminateBlocks"/> is the Schur E-block count (0 otherwise).
	/// </summary>
	public ProgramEvaluator(Program program, bool denseJacobian, int numEliminateBlocks, int numThreads)
	{
		this.program = program;
		this.numThreads = Math.Max(1, numThreads);
		this.denseJacobian = denseJacobian;
		if (!denseJacobian)
		{
			blockLayout = new BlockJacobianLayout(program, numEliminateBlocks);
		}

		List<ResidualBlock> residualBlocks = program.ResidualBlocks;
		residualLayout = new int[residualBlocks.Count];
		for (int i = 0, position = 0; i < residualBlocks.Count; i++)
		{
			residualLayout[i] = position;
			position += residualBlocks[i].NumResiduals;
		}

		blockCosts = new double[residualBlocks.Count];
		scratchPool = new Scratch[this.numThreads];
		for (int t = 0; t < scratchPool.Length; t++)
		{
			scratchPool[t] = new Scratch(program, denseJacobian);
		}

		chunkExceptions = new Exception?[this.numThreads];
		parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = this.numThreads };
		chunkBody = EvaluateChunk;

		NumParameters = program.NumParameters;
		NumEffectiveParameters = program.NumEffectiveParameters;
		NumResiduals = program.NumResiduals;
	}

	/// <summary>Ambient size of the state vector.</summary>
	public int NumParameters { get; }

	/// <summary>Tangent size (Jacobian columns).</summary>
	public int NumEffectiveParameters { get; }

	/// <summary>Number of residuals (Jacobian rows).</summary>
	public int NumResiduals { get; }

	/// <summary>Number of Evaluate calls asking for neither a gradient nor a Jacobian.</summary>
	public int NumResidualEvaluations { get; private set; }

	/// <summary>Number of Evaluate calls asking for a gradient or a Jacobian.</summary>
	public int NumJacobianEvaluations { get; private set; }

	/// <summary>A zero Jacobian of the evaluator's kind.</summary>
	public SparseMatrix CreateJacobian() =>
		denseJacobian ? new DenseSparseMatrix(NumResiduals, NumEffectiveParameters) : blockLayout!.CreateJacobian();

	/// <summary>
	/// Evaluates at <paramref name="state"/>. <paramref name="residuals"/>,
	/// <paramref name="gradient"/> and <paramref name="jacobian"/> are optional. Without
	/// <paramref name="applyLossFunction"/> the cost and residuals are the plain squared ones
	/// (Problem::EvaluateOptions::apply_loss_function). Returns false if a residual block
	/// fails or the cost is not finite.
	/// </summary>
	public bool Evaluate(
		double[] state,
		out double cost,
		double[]? residuals,
		double[]? gradient,
		SparseMatrix? jacobian,
		bool applyLossFunction = true)
	{
		cost = 0.0;

		// program_evaluator.h times a call as "Evaluator::Jacobian" when a gradient or a
		// Jacobian is asked for, "Evaluator::Residual" otherwise.
		bool countAsJacobianEvaluation = gradient is not null || jacobian is not null;
		if (gradient is not null && (jacobian is null || residuals is null))
		{
			// The gradient is J'r, so it needs both; borrow the evaluator's own.
			jacobian ??= gradientJacobian ??= CreateJacobian();
			residuals ??= gradientResiduals ??= new double[NumResiduals];
		}

		if (!countAsJacobianEvaluation)
		{
			NumResidualEvaluations++;
		}
		else
		{
			NumJacobianEvaluations++;
		}

		if (!program.StateVectorToParameterBlocks(state))
		{
			return false;
		}

		if (residuals is not null)
		{
			Array.Clear(residuals);
		}

		jacobian?.SetZero();

		int numResidualBlocks = program.ResidualBlocks.Count;
		bool aborted = false;
		if (numThreads == 1 || numResidualBlocks < 2)
		{
			Scratch scratch = scratchPool[0];
			for (int i = 0; i < numResidualBlocks && !aborted; i++)
			{
				aborted = !EvaluateResidualBlock(i, scratch, residuals, jacobian, applyLossFunction);
			}
		}
		else
		{
			// Contiguous chunks, one per worker slot, each with its own scratch; the body
			// delegate is cached and reads the call's outputs from fields, so a parallel
			// evaluation allocates nothing of its own.
			currentResiduals = residuals;
			currentJacobian = jacobian;
			currentApplyLossFunction = applyLossFunction;
			failed = 0;
			Array.Clear(chunkExceptions);
			Parallel.For(0, scratchPool.Length, parallelOptions, chunkBody);
			currentResiduals = null;
			currentJacobian = null;

			// A cost function that throws surfaces its own exception, as it does on one
			// thread, rather than Parallel.For's AggregateException; the lowest chunk's wins.
			foreach (Exception? exception in chunkExceptions)
			{
				if (exception is not null)
				{
					ExceptionDispatchInfo.Capture(exception).Throw();
				}
			}

			aborted = failed != 0;
		}

		if (aborted)
		{
			return false;
		}

		for (int i = 0; i < numResidualBlocks; i++)
		{
			cost += blockCosts[i];
		}

		if (gradient is not null)
		{
			Array.Clear(gradient);
			jacobian!.LeftMultiplyAndAccumulate(residuals!, gradient);
		}

		// Ceres logs "Accumulated cost = ... is not a finite number. Evaluation failed."
		return double.IsFinite(cost);
	}

	private bool EvaluateResidualBlock(int i, Scratch scratch, double[]? residuals, SparseMatrix? jacobian, bool applyLossFunction)
	{
		ResidualBlock residualBlock = program.ResidualBlocks[i];
		int numResiduals = residualBlock.NumResiduals;
		ParameterBlock[] blocks = residualBlock.ParameterBlocks;
		Span<double> blockResiduals = residuals is not null
			? residuals.AsSpan(residualLayout[i], numResiduals)
			: jacobian is not null
				? scratch.ResidualBlockResiduals.AsSpan(0, numResiduals)
				: [];

		ReadOnlySpan<ArraySegment<double>> blockJacobians = [];
		if (jacobian is not null)
		{
			PrepareJacobians(i, residualBlock, jacobian, scratch);
			blockJacobians = scratch.JacobianViews.AsSpan(0, blocks.Length);
		}

		if (!residualBlock.Evaluate(
				applyLossFunction, out double blockCost, blockResiduals, blockJacobians, scratch.EvaluateScratch, scratch.ParameterViews, scratch.EvalJacobians))
		{
			return false;
		}

		blockCosts[i] = blockCost;
		if (jacobian is DenseSparseMatrix dense)
		{
			WriteDense(i, residualBlock, scratch, dense);
		}

		return true;
	}

	// Evaluates chunk t of the residual blocks with scratch t, stopping at the first failure
	// here or in another chunk.
	private void EvaluateChunk(int t)
	{
		int n = program.ResidualBlocks.Count;
		int chunks = scratchPool.Length;
		int start = (int)((long)n * t / chunks);
		int end = (int)((long)n * (t + 1) / chunks);
		try
		{
			for (int i = start; i < end && Volatile.Read(ref failed) == 0; i++)
			{
				if (!EvaluateResidualBlock(i, scratchPool[t], currentResiduals, currentJacobian, currentApplyLossFunction))
				{
					Volatile.Write(ref failed, 1);
				}
			}
		}
		catch (Exception exception)
		{
			chunkExceptions[t] = exception;
			Volatile.Write(ref failed, 1);
		}
	}

	// BlockEvaluatePreparer / ScratchEvaluatePreparer: where each block's Jacobian goes.
	private void PrepareJacobians(int i, ResidualBlock residualBlock, SparseMatrix jacobian, Scratch scratch)
	{
		ParameterBlock[] blocks = residualBlock.ParameterBlocks;
		int numResiduals = residualBlock.NumResiduals;
		if (jacobian is BlockSparseMatrix blockSparse)
		{
			int[] positions = blockLayout![i];
			for (int j = 0, k = 0; j < blocks.Length; j++)
			{
				scratch.JacobianViews[j] = blocks[j].IsConstant
					? default
					: new ArraySegment<double>(blockSparse.Values, positions[k++], numResiduals * blocks[j].TangentSize);
			}

			return;
		}

		for (int j = 0, offset = 0; j < blocks.Length; j++)
		{
			if (blocks[j].IsConstant)
			{
				scratch.JacobianViews[j] = default;
				continue;
			}

			int size = numResiduals * blocks[j].TangentSize;
			scratch.JacobianViews[j] = new ArraySegment<double>(scratch.JacobianScratch, offset, size);
			offset += size;
		}
	}

	// DenseJacobianWriter::Write: copies the block Jacobians into the residual block's rows.
	private void WriteDense(int i, ResidualBlock residualBlock, Scratch scratch, DenseSparseMatrix dense)
	{
		ParameterBlock[] blocks = residualBlock.ParameterBlocks;
		int numResiduals = residualBlock.NumResiduals;
		int rowOffset = residualLayout[i];
		for (int j = 0; j < blocks.Length; j++)
		{
			if (blocks[j].IsConstant)
			{
				continue;
			}

			int tangentSize = blocks[j].TangentSize;
			ReadOnlySpan<double> cell = scratch.JacobianViews[j];
			for (int c = 0; c < tangentSize; c++)
			{
				Span<double> column = dense.Matrix.ColumnSpan(blocks[j].DeltaOffset + c);
				for (int r = 0; r < numResiduals; r++)
				{
					column[rowOffset + r] = cell[r * tangentSize + c];
				}
			}
		}
	}

	/// <summary>Program::Plus through the evaluator (Evaluator::Plus).</summary>
	public bool Plus(double[] state, double[] delta, double[] statePlusDelta) =>
		program.Plus(state, delta, statePlusDelta, numThreads);

	// Per-worker buffers; one worker uses one at a time.
	private sealed class Scratch
	{
		public Scratch(Program program, bool denseJacobian)
		{
			int maxBlocks = Math.Max(1, program.MaxParametersPerResidualBlock());
			EvaluateScratch = new double[program.MaxScratchDoublesNeededForEvaluate()];
			ResidualBlockResiduals = new double[program.MaxResidualsPerResidualBlock()];
			ParameterViews = new ArraySegment<double>[maxBlocks];
			EvalJacobians = new ArraySegment<double>[maxBlocks];
			JacobianViews = new ArraySegment<double>[maxBlocks];
			int maxDerivatives = 0;
			if (denseJacobian)
			{
				foreach (ResidualBlock block in program.ResidualBlocks)
				{
					int derivatives = 0;
					foreach (ParameterBlock parameterBlock in block.ParameterBlocks)
					{
						if (!parameterBlock.IsConstant)
						{
							derivatives += block.NumResiduals * parameterBlock.TangentSize;
						}
					}

					maxDerivatives = Math.Max(maxDerivatives, derivatives);
				}
			}

			JacobianScratch = new double[maxDerivatives];
		}

		public double[] EvaluateScratch { get; }

		public double[] ResidualBlockResiduals { get; }

		public ArraySegment<double>[] ParameterViews { get; }

		public ArraySegment<double>[] EvalJacobians { get; }

		public ArraySegment<double>[] JacobianViews { get; }

		public double[] JacobianScratch { get; }
	}
}
