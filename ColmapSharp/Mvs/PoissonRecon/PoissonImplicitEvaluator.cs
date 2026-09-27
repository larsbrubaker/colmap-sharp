// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonImplicitEvaluator: point evaluation of the solved implicit function and the iso-value
// Poisson::Solver::Solve derives from it (thirdparty/PoissonRecon/Reconstructors.h, "Get the
// iso-value"). Ports FEMTree::_MultiThreadedEvaluator< Sigs , 0 > (constructor and values(),
// FEMTree.Evaluation.inl), FEMTree::_getValues (the value-only path Solve uses),
// FEMTree::coarseCoefficients (FEMTree.System.inl) and the iso-value average. Evaluation goes
// through PoissonPointEvaluator (the per-axis basis values) and PoissonMultigrid.UpSample (the
// coarse coefficients). The solution comes from PoissonSystem.Solve; the level-set extraction
// uses the iso-value. Tier A against oracle/poisson_levelset_harness.cc.
//
// Translation notes:
// - The C++ keeps one neighbor key per thread and sums per-thread partial sums in thread
//   order; the port evaluates the samples in order on one thread, which is what the
//   single-threaded C++ does (the multi-threaded order is divergence 123's).
// - _Evaluator::set also tabulates the center and corner stencils; values() does not read
//   them. The level-set extractor's corner stencils are PoissonCornerEvaluator's.
// - Float/double mix as upstream: each basis value is a double narrowed to float, the value is
//   a float sum of float products, and the averages are double sums of float terms.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Evaluates the implicit function with coefficients indexed by node index at points inside
/// known leaves. Port of PoissonRecon's <c>MultiThreadedEvaluator&lt; Sigs , 0 &gt;</c> used
/// single-threaded.
/// </summary>
public sealed class PoissonImplicitEvaluator
{
	private const int ProgressInterval = 1 << 16;

	private readonly FemTree tree;
	private readonly float[] coefficients;
	private readonly float[] coarseCoefficients;
	private readonly PoissonPointEvaluator pointEvaluator;
	private readonly NeighborKey pointKey;
	private readonly int[] childWindow;
	private readonly int maxDepth;

	/// <summary>
	/// An evaluator of <paramref name="coefficients"/> (the solution, indexed by node index) for
	/// basis <paramref name="signature"/> on the finalized tree. Sets the FEM_FLAG_1 flags for the
	/// signature. Port of the <c>_MultiThreadedEvaluator</c> constructor.
	/// </summary>
	public PoissonImplicitEvaluator(FemTree tree, SortedTreeNodes sorted, int signature, float[] coefficients)
	{
		this.tree = tree;
		this.coefficients = coefficients;
		PoissonMultigrid.SetFem1ValidityFlags(tree, sorted, signature);
		maxDepth = PoissonMultigrid.MaxDepth(tree);
		coarseCoefficients = CoarseCoefficients(tree, sorted, signature, coefficients);
		pointEvaluator = new PoissonPointEvaluator(signature, 0, maxDepth);
		BSplineSupportSizes support = BSplineSupportSizes.For(FemSignature.Degree(signature));

		// ConstPointSupportKey: the functions whose support contains a node's cell.
		pointKey = new NeighborKey(tree, support.SupportEnd, -support.SupportStart, resetOnMissing: false);
		pointKey.Set(maxDepth + tree.DepthOffset);
		childWindow = new int[pointKey.WindowSize];
	}

	/// <summary>The prolonged coarser solution per node (FEMTree::coarseCoefficients), indexed by node index.</summary>
	public float[] Coarse => coarseCoefficients;

	/// <summary>
	/// The coefficients below the finest depth with every coarser depth's contribution
	/// up-sampled into each finer one: the copy of the coefficients of depths 0..maxDepth-1,
	/// then, coarse to fine, each depth's up-sampled into the next. Port of
	/// <c>coarseCoefficients( DenseNodeData )</c>.
	/// </summary>
	public static float[] CoarseCoefficients(FemTree tree, SortedTreeNodes sorted, int signature, float[] coefficients)
	{
		int treeMaxDepth = PoissonMultigrid.MaxDepth(tree);
		int end = PoissonMultigrid.End(tree, sorted, treeMaxDepth - 1);
		var coarse = new float[end];
		int begin = PoissonMultigrid.Begin(tree, sorted, 0);
		Array.Copy(coefficients, begin, coarse, begin, end - begin);
		var rp = new RestrictionProlongation(signature);
		for (int d = 1; d < treeMaxDepth; d++)
		{
			PoissonMultigrid.UpSample(tree, sorted, rp, d, coarse, 0, coarse, 0, 1);
		}

		return coarse;
	}

	/// <summary>
	/// The value at unit-cube point (<paramref name="x"/>, <paramref name="y"/>,
	/// <paramref name="z"/>) inside leaf <paramref name="node"/>: the node's (or its first
	/// non-ghost ancestor's) functions against the solution, its parent's against the coarse
	/// coefficients, and, below the finest depth, the functions of the child cell holding the
	/// point. Port of <c>values( p , thread , node )[0]</c> (<c>_getValues</c>).
	/// </summary>
	public float Value(float x, float y, float z, int node)
	{
		pointKey.GetNeighbors(node);

		// Nudging evaluation point into the interior
		x = Nudge(x);
		y = Nudge(y);
		z = Nudge(z);

		float value = 0;
		int depth = tree.LocalDepth(node);
		while (tree.IsGhost(node))
		{
			node = tree.Parent(node);
			depth--;
		}

		Init(depth, x, y, z);
		value = AddToValues(value, pointKey.Window(tree.Depth(node)), coefficients);
		if (depth > 0)
		{
			Init(depth - 1, x, y, z);
			value = AddToValues(value, pointKey.Window(tree.Depth(tree.Parent(node))), coarseCoefficients);
		}

		// If there could be finer neighbors whose support overlaps the point
		if (depth < maxDepth)
		{
			// _centerAndWidth, in the unit cube's frame.
			float width = (float)(1.0 / (1 << depth));
			int cIdx = 0;
			if (x > (float)(tree.LocalOffset(node, 0) + 0.5) * width)
			{
				cIdx |= 1;
			}

			if (y > (float)(tree.LocalOffset(node, 1) + 0.5) * width)
			{
				cIdx |= 2;
			}

			if (z > (float)(tree.LocalOffset(node, 2) + 0.5) * width)
			{
				cIdx |= 4;
			}

			if (pointKey.GetChildNeighbors(cIdx, tree.Depth(node), childWindow) != 0)
			{
				Init(depth + 1, x, y, z);
				value = AddToValues(value, childWindow, coefficients);
			}
		}

		return value;
	}

	/// <summary>
	/// The iso-value: the weight-averaged value at the samples' mean positions, over the samples
	/// of positive weight, in sample order, with the double sums it divides. Port of Solve's
	/// "Get the iso-value" block. The token is checked, and progress reported, every 65536 samples.
	/// </summary>
	public PoissonIsoValue IsoValue(PoissonSampleSet samples, IProgress<PoissonProgress>? progress = null, CancellationToken cancellationToken = default)
	{
		double valueSum = 0, weightSum = 0;
		int count = samples.Count;
		for (int j = 0; j < count; j++)
		{
			if (j % ProgressInterval == 0)
			{
				cancellationToken.ThrowIfCancellationRequested();
				progress?.Report(new PoissonProgress(PoissonStage.IsoValue, (double)j / count));
			}

			float w = samples.Weight(j);
			if (w > 0)
			{
				float value = Value(samples.Position(j, 0) / w, samples.Position(j, 1) / w, samples.Position(j, 2) / w, samples.Node(j));
				weightSum += w;
				valueSum += value * w;
			}
		}

		progress?.Report(new PoissonProgress(PoissonStage.IsoValue, 1));
		return new PoissonIsoValue((float)(valueSum / weightSum), valueSum, weightSum);
	}

	private static float Nudge(float p) => p == 0 ? (float)(0.0 + 1e-6) : p == 1 ? (float)(1.0 - 1e-6) : p;

	// initEvaluationState( p , depth , state ): the point's cell at the depth, from the point
	// widened to double.
	private void Init(int depth, float x, float y, float z)
	{
		int res = 1 << depth;
		pointEvaluator.Init(depth, x, y, z, (int)((double)x * res), (int)((double)y * res), (int)((double)z * res));
	}

	// _getValues' AddToValues: the window's valid functions, in window order.
	private float AddToValues(float value, int[] window, float[] source)
	{
		for (int i = 0; i < window.Length; i++)
		{
			int n = window[i];
			if (PoissonMultigrid.IsValidFem1Node(tree, n))
			{
				float basis = (float)pointEvaluator.Value(tree.LocalOffset(n, 0), tree.LocalOffset(n, 1), tree.LocalOffset(n, 2));
				value += source[tree.NodeIndex(n)] * basis;
			}
		}

		return value;
	}
}

/// <summary>
/// Solve's iso-value (<c>implicit.isoValue</c>) with the weighted value sum and the weight sum
/// it is the ratio of.
/// </summary>
public readonly record struct PoissonIsoValue(float Value, double ValueSum, double WeightSum);
