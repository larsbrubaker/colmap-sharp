// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 include/ceres/problem.h and internal/ceres/problem_impl.cc
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ceres::Problem: the nonlinear least-squares problem, built from parameter blocks (user
// memory) and residual blocks (cost function + optional loss over some parameter blocks).
// COLMAP passes raw pointers into its own objects (Camera.params.data(), a pose's
// quaternion coefficients, a point's xyz). The C# identity of a block is an
// ArraySegment<double>'s (array, offset) pair: a whole double[] converts implicitly (offset
// 0), and a slice of a larger array is a distinct block, so one buffer can hold many blocks.
// As in Ceres, the block's size comes from the cost function (or AddParameterBlock), blocks
// may not overlap, and the solver writes the solution back into the user's memory when
// LeastSquaresSolver.Solve ends.
//
// Only the part of the API COLMAP calls is here (see the grep in the Phase 7 notes):
// AddParameterBlock (with manifold), AddResidualBlock, Set/IsParameterBlockConstant,
// SetParameterBlockVariable, SetManifold/GetManifold, HasParameterBlock, the counts, the
// per-coordinate bounds (Set/GetParameterLowerBound/UpperBound: view-graph calibration and
// the global positioner) and Problem::Evaluate (Problem.Evaluate.cs: covariance estimation
// and view-graph calibration). RemoveResidualBlock/RemoveParameterBlock,
// EvaluateResidualBlock and the evaluation callback are unused by COLMAP and not ported.
// Ceres' ownership options have no C# meaning: cost, loss and manifold objects may be shared
// between blocks and problems freely.

using System.Runtime.CompilerServices;

using ColmapSharp.Util;

namespace ColmapSharp.Solver;

/// <summary>Opaque handle to a residual block (ceres::ResidualBlockId).</summary>
public sealed class ResidualBlockId
{
	internal ResidualBlockId(ResidualBlock block) => Block = block;

	internal ResidualBlock Block { get; }
}

/// <summary>ceres::Problem: parameter blocks and the residual blocks over them.</summary>
public sealed partial class Problem
{
	// Blocks on each array, sorted by offset, for lookup and the aliasing check.
	private readonly Dictionary<double[], List<ParameterBlock>> blocksByArray = new(ReferenceEqualityComparer.Instance);

	/// <summary>The problem's program: every block, in insertion order.</summary>
	internal Program Program { get; } = new();

	/// <summary>Number of parameter blocks.</summary>
	public int NumParameterBlocks => Program.ParameterBlocks.Count;

	/// <summary>Total number of scalar parameters (ambient sizes).</summary>
	public int NumParameters => Program.NumParameters;

	/// <summary>Number of residual blocks.</summary>
	public int NumResidualBlocks => Program.ResidualBlocks.Count;

	/// <summary>Total number of residuals.</summary>
	public int NumResiduals => Program.NumResiduals;

	/// <summary>
	/// Adds a parameter block over the whole of <paramref name="values"/> (a double[] converts
	/// implicitly), optionally with a manifold. Adding an existing block again only checks the
	/// size and, if given, sets the manifold.
	/// </summary>
	public void AddParameterBlock(ArraySegment<double> values, Manifold? manifold = null)
	{
		ParameterBlock block = InternalAddParameterBlock(values, values.Count);
		if (manifold is not null)
		{
			block.SetManifold(manifold);
		}
	}

	/// <summary>
	/// Adds a residual block. Each parameter block is identified by its segment's array and
	/// offset and takes its size from the cost function; blocks not yet in the problem are
	/// added. A null <paramref name="lossFunction"/> means plain least squares.
	/// </summary>
	public ResidualBlockId AddResidualBlock(
		CostFunction costFunction, LossFunction? lossFunction, params ArraySegment<double>[] parameterBlocks)
	{
		ArgumentNullException.ThrowIfNull(costFunction);
		ReadOnlySpan<int> sizes = costFunction.ParameterBlockSizes;
		Check.Eq(
			parameterBlocks.Length,
			sizes.Length,
			$"Number of blocks input is {parameterBlocks.Length} and the number expected by the cost function is {sizes.Length}.");

		// Check the sizes match the cost function, and for duplicate parameter blocks.
		for (int i = 0; i < parameterBlocks.Length; i++)
		{
			for (int j = 0; j < i; j++)
			{
				if (ReferenceEquals(parameterBlocks[i].Array, parameterBlocks[j].Array)
					&& parameterBlocks[i].Offset == parameterBlocks[j].Offset)
				{
					throw new ArgumentException(
						"Duplicate parameter blocks in a residual parameter are not allowed. "
						+ $"Parameter blocks {j} and {i} are duplicates.");
				}
			}
		}

		var blocks = new ParameterBlock[parameterBlocks.Length];
		for (int i = 0; i < parameterBlocks.Length; i++)
		{
			blocks[i] = InternalAddParameterBlock(parameterBlocks[i], sizes[i]);
		}

		for (int i = 0; i < blocks.Length; i++)
		{
			Check.Eq(
				sizes[i],
				blocks[i].Size,
				$"The cost function expects parameter block {i} of size {sizes[i]} but was given a block of size {blocks[i].Size}");
		}

		var residualBlock = new ResidualBlock(costFunction, lossFunction, blocks, Program.ResidualBlocks.Count);
		Program.ResidualBlocks.Add(residualBlock);
		return new ResidualBlockId(residualBlock);
	}

	/// <summary>True if the block starting at <paramref name="values"/> is in the problem.</summary>
	public bool HasParameterBlock(ArraySegment<double> values) => Find(values) is not null;

	/// <summary>Holds the block constant during optimization.</summary>
	public void SetParameterBlockConstant(ArraySegment<double> values) =>
		FindOrThrow(values, "set constant").IsSetConstant = true;

	/// <summary>Lets a constant block vary again.</summary>
	public void SetParameterBlockVariable(ArraySegment<double> values) =>
		FindOrThrow(values, "set varying").IsSetConstant = false;

	/// <summary>True if the block is held constant (Ceres: set constant, or a zero-dimensional tangent space).</summary>
	public bool IsParameterBlockConstant(ArraySegment<double> values) =>
		FindOrThrow(values, "queried for constancy").IsConstant;

	/// <summary>Sets (or with null, clears) the block's manifold.</summary>
	public void SetManifold(ArraySegment<double> values, Manifold? manifold) =>
		FindOrThrow(values, "have its manifold set").SetManifold(manifold);

	/// <summary>The block's manifold, or null.</summary>
	public Manifold? GetManifold(ArraySegment<double> values) =>
		FindOrThrow(values, "have its manifold queried").Manifold;

	/// <summary>
	/// Sets the lower bound of coordinate <paramref name="index"/> of the block
	/// (-double.MaxValue, Ceres' -std::numeric_limits&lt;double&gt;::max(), means none).
	/// </summary>
	public void SetParameterLowerBound(ArraySegment<double> values, int index, double lowerBound) =>
		FindOrThrow(values, "given a lower bound on one of its components").SetLowerBound(index, lowerBound);

	/// <summary>Sets the upper bound of coordinate <paramref name="index"/> (double.MaxValue means none).</summary>
	public void SetParameterUpperBound(ArraySegment<double> values, int index, double upperBound) =>
		FindOrThrow(values, "given an upper bound on one of its components").SetUpperBound(index, upperBound);

	/// <summary>The lower bound of coordinate <paramref name="index"/> (-double.MaxValue if none).</summary>
	public double GetParameterLowerBound(ArraySegment<double> values, int index) =>
		FindOrThrow(values, "queried for the lower bound of one of its components").LowerBoundForParameter(index);

	/// <summary>The upper bound of coordinate <paramref name="index"/> (double.MaxValue if none).</summary>
	public double GetParameterUpperBound(ArraySegment<double> values, int index) =>
		FindOrThrow(values, "queried for the upper bound of one of its components").UpperBoundForParameter(index);

	/// <summary>Ambient size of the block.</summary>
	public int ParameterBlockSize(ArraySegment<double> values) => FindOrThrow(values, "have its size queried").Size;

	/// <summary>Tangent size of the block.</summary>
	public int ParameterBlockTangentSize(ArraySegment<double> values) =>
		FindOrThrow(values, "have its tangent size queried").TangentSize;

	private ParameterBlock InternalAddParameterBlock(ArraySegment<double> values, int size)
	{
		double[] array = values.Array
			?? throw new ArgumentException($"Null array passed to AddParameterBlock for a parameter with size {size}");
		Check.Gt(size, 0);
		if (values.Count < size)
		{
			throw new ArgumentException(
				$"The parameter block at offset {values.Offset} needs {size} values but the segment holds {values.Count}.");
		}

		if (!blocksByArray.TryGetValue(array, out List<ParameterBlock>? list))
		{
			list = [];
			blocksByArray.Add(array, list);
		}

		int position = LowerBound(list, values.Offset);
		if (position < list.Count && list[position].UserState.Offset == values.Offset)
		{
			ParameterBlock existing = list[position];
			if (existing.Size != size)
			{
				throw new ArgumentException(
					"Tried adding a parameter block with the same memory location twice, but with different block sizes. "
					+ $"Original size was {existing.Size} but new size is {size}");
			}

			return existing;
		}

		// Before adding the parameter block, also check that it doesn't alias any other
		// parameter blocks.
		int start = values.Offset;
		int end = start + size;
		if (position > 0)
		{
			ParameterBlock previous = list[position - 1];
			CheckForNoAliasing(previous.UserState.Offset, previous.Size, start, end);
		}

		if (position < list.Count)
		{
			ParameterBlock next = list[position];
			CheckForNoAliasing(next.UserState.Offset, next.Size, start, end);
		}

		var block = new ParameterBlock(new ArraySegment<double>(array, start, size), Program.ParameterBlocks.Count);
		list.Insert(position, block);
		Program.ParameterBlocks.Add(block);
		return block;
	}

	private static void CheckForNoAliasing(int existingStart, int existingSize, int newStart, int newEnd)
	{
		int existingEnd = existingStart + existingSize;
		if (newStart < existingEnd && existingStart < newEnd)
		{
			throw new ArgumentException(
				$"Aliasing detected between existing parameter block at offset {existingStart} and has size {existingSize} "
				+ $"with new parameter block that has offset {newStart} and size {newEnd - newStart}.");
		}
	}

	private static int LowerBound(List<ParameterBlock> list, int offset)
	{
		int lo = 0;
		int hi = list.Count;
		while (lo < hi)
		{
			int mid = (lo + hi) >>> 1;
			if (list[mid].UserState.Offset < offset)
			{
				lo = mid + 1;
			}
			else
			{
				hi = mid;
			}
		}

		return lo;
	}

	/// <summary>The block starting at (array, offset), or null (Ceres' parameter_map lookup).</summary>
	internal ParameterBlock? Find(double[] array, int offset) => Find(new ArraySegment<double>(array, offset, 0));

	private ParameterBlock? Find(ArraySegment<double> values)
	{
		if (values.Array is null || !blocksByArray.TryGetValue(values.Array, out List<ParameterBlock>? list))
		{
			return null;
		}

		int position = LowerBound(list, values.Offset);
		return position < list.Count && list[position].UserState.Offset == values.Offset ? list[position] : null;
	}

	private ParameterBlock FindOrThrow(ArraySegment<double> values, string action, [CallerMemberName] string caller = "")
	{
		return Find(values) ?? throw new ArgumentException(
			$"{caller}: parameter block not found. You must add the parameter block to the problem before it can be {action}.");
	}
}
