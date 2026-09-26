// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonSystem: the per-depth system assembly of FEMTree's multigrid solver
// (thirdparty/PoissonRecon/FEMTree.System.inl) for Solve's system - the gradient inner
// product of the degree-1 Neumann basis (FemSystemIntegrator) plus the point-interpolation
// term (PoissonInterpolation entries weighted by SystemDual): _getSliceMatrixAndProlongationConstraints
// sets one matrix row per valid node of a depth (_setMatrixRowAndGetConstraintFromProlongation,
// with _addPointValues for the interpolation part), the inverse diagonal, and the constraint
// the coarser (prolonged) solution already meets (_getConstraintFromProlongedSolution with
// _getInterpolationConstraintFromProlongedSolution); _getProlongedMatrixRowSize counts a
// node's valid parent-depth neighbors. The solver (a later slice) relaxes these rows.
// Tier A against oracle/poisson_system_harness.cc (the "slice*" and "prolongedrowsize" cases).
//
// Translation notes:
// - One instance holds the scratch windows and the row buffer, so the per-node loops do not
//   allocate; the C++ keeps them on the stack per node. The windows have the key's own radii
//   (one ring), so NeighborKey reads the cached parent window and never recurses.
// - The C++ runs the rows through ThreadPool::ParallelFor; each row writes only its own slot,
//   and the port runs them in order.
// - Float/double mixing follows the C++ types: the point values and matrix entries are float
//   (Real), stencils and integrals double, SystemDual's product double (the double
//   CumulativeDerivativeValues overload), partialDotDValues float.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Matrix rows and prolongation constraints for one depth of Solve's system. Port of the matching
/// members of PoissonRecon's <c>FEMTree&lt;3,float&gt;</c>.
/// </summary>
public sealed class PoissonSystem
{
	private const int Dim = 3;
	private readonly FemTree tree;
	private readonly SortedTreeNodes sorted;
	private readonly FemSystemIntegrator f;
	private readonly PoissonPointEvaluator? bsData;
	private readonly SparseNodeData? interpolation;
	private readonly float systemWeight;
	private readonly int degree;
	private readonly int overlapRadius;
	private readonly int width;
	private readonly int center;
	private readonly BSplineSupportSizes support;
	private readonly int[][] parentLoopData;
	private readonly NeighborKey neighborKey;
	private readonly int[] neighbors;
	private readonly int[] pNeighbors;
	private readonly float[] pointValues;
	private readonly int[] off = new int[Dim];
	private readonly int[] other = new int[Dim];

	/// <summary>
	/// The assembly for system <paramref name="f"/> on the finalized tree, with point evaluation
	/// <paramref name="bsData"/> over <paramref name="interpolation"/> (PoissonInterpolation's
	/// layout) whose SystemDual weight is <paramref name="systemWeight"/>; both null for no
	/// interpolation term. Sets the FEM_FLAG_1 flags for the system's signature.
	/// </summary>
	public PoissonSystem(FemTree tree, SortedTreeNodes sorted, FemSystemIntegrator f, PoissonPointEvaluator? bsData, SparseNodeData? interpolation, float systemWeight)
	{
		this.tree = tree;
		this.sorted = sorted;
		this.f = f;
		this.bsData = bsData;
		this.interpolation = interpolation;
		this.systemWeight = systemWeight;
		degree = FemSignature.Degree(f.Signature);
		BSplineOverlapSizes overlap = BSplineOverlapSizes.For(degree, degree);
		overlapRadius = -overlap.OverlapStart;
		width = overlap.OverlapSize;
		center = ((overlapRadius * width) + overlapRadius) * width + overlapRadius;
		support = BSplineSupportSizes.For(degree);
		parentLoopData = PoissonMultigrid.ParentOverlapLoopData(degree, degree);
		neighborKey = new NeighborKey(tree, 1, 1, resetOnMissing: false);
		neighbors = new int[width * width * width];
		pNeighbors = new int[width * width * width];
		pointValues = new float[width * width * width];
		PoissonMultigrid.SetFem1ValidityFlags(tree, sorted, f.Signature);
	}

	/// <summary>
	/// Sets matrix row i - nBegin for every valid node i in [nBegin, nEnd) of
	/// <paramref name="depth"/>, its inverse diagonal (0 for Dirichlet elements), and the
	/// constraint the prolonged coarser solution (indexed by node index; null for none) meets
	/// there; invalid nodes get an empty row and a zero constraint. <c>f</c> must be initialized
	/// at the depth, with <paramref name="ccStencil"/> and <paramref name="pcStencils"/> its
	/// stencils. Port of <c>_getSliceMatrixAndProlongationConstraints</c>.
	/// </summary>
	public int GetSliceMatrixAndProlongationConstraints(PoissonSystemMatrix matrix, float[]? diagonalR, int depth, int nBegin, int nEnd, float[]? prolongedSolution, float[]? constraints, double[] ccStencil, double[][] pcStencils)
	{
		int range = nEnd - nBegin;
		matrix.Resize(range);
		neighborKey.Set(depth + tree.DepthOffset);
		for (int i = 0; i < range; i++)
		{
			int node = sorted.TreeNodes[i + nBegin];
			if (!PoissonMultigrid.IsValidFem1Node(tree, node))
			{
				if (constraints != null)
				{
					constraints[i] = 0;
				}

				continue;
			}

			neighborKey.GetNeighbors(overlapRadius, overlapRadius, node, pNeighbors, neighbors);
			float constraint = SetMatrixRowAndGetConstraintFromProlongation(node, i, matrix, nBegin, pcStencils, ccStencil, prolongedSolution);
			if (constraints != null)
			{
				constraints[i] = constraint;
			}

			if (diagonalR != null)
			{
				diagonalR[i] = (tree.Flags(node) & FemTree.DirichletElementFlag) != 0 ? 0f : 1f / matrix.Value(i, 0);
			}
		}

		return 1;
	}

	/// <summary>
	/// The number of valid (FEM_FLAG_1) nodes among the parent-depth neighbors in
	/// <paramref name="parentWindow"/> (a one-ring window of the parent) that overlap
	/// <paramref name="node"/>; 0 for a root. Port of <c>_getProlongedMatrixRowSize</c>.
	/// </summary>
	public int GetProlongedMatrixRowSize(int node, int[] parentWindow)
	{
		int count = 0;
		if (tree.Parent(node) != FemTree.None)
		{
			foreach (int idx in parentLoopData[tree.ChildIndexInParent(node)])
			{
				if (PoissonMultigrid.IsValidFem1Node(tree, parentWindow[idx]))
				{
					count++;
				}
			}
		}

		return count;
	}

	/// <summary>
	/// The interpolation constraint of <paramref name="node"/>: over the interpolation entries of
	/// the space nodes in its window (<paramref name="window"/>, one ring) whose support reaches it,
	/// the dual value times the node's function at the entry. Port of
	/// <c>_getInterpolationConstraintFromProlongedSolution</c> (whose prolonged solution argument
	/// is unused upstream: the dual values were set from it by _setPointValuesFromProlongedSolution).
	/// </summary>
	public float GetInterpolationConstraintFromProlongedSolution(int[] window, int node)
	{
		if (interpolation == null || bsData == null)
		{
			return 0;
		}

		float temp = 0;
		if (!PoissonMultigrid.IsValidFem1Node(tree, node))
		{
			return temp;
		}

		int ox = tree.LocalOffset(node, 0), oy = tree.LocalOffset(node, 1), oz = tree.LocalOffset(node, 2);
		int start = overlapRadius + support.SupportStart;
		int end = start + support.SupportSize;
		for (int i0 = start; i0 < end; i0++)
		{
			for (int i1 = start; i1 < end; i1++)
			{
				for (int i2 = start; i2 < end; i2++)
				{
					int pNode = window[((i0 * width) + i1) * width + i2];
					if (!PoissonMultigrid.IsValidSpaceNode(tree, pNode))
					{
						continue;
					}

					int slot = interpolation.Index(tree.NodeIndex(pNode));
					if (slot == -1)
					{
						continue;
					}

					bsData.Init(tree.LocalDepth(pNode), interpolation.Value(slot, 0), interpolation.Value(slot, 1), interpolation.Value(slot, 2), tree.LocalOffset(pNode, 0), tree.LocalOffset(pNode, 1), tree.LocalOffset(pNode, 2));
					float value = (float)bsData.Value(ox, oy, oz);
					temp += interpolation.Value(slot, 4) * value;
				}
			}
		}

		return temp;
	}

	// Port of _setMatrixRowAndGetConstraintFromProlongation: row `row` of the matrix for the
	// center of `neighbors`, and the constraint met by the prolonged solution.
	private float SetMatrixRowAndGetConstraintFromProlongation(int node, int row, PoissonSystemMatrix matrix, int offset, double[][] pcStencils, double[] ccStencil, float[]? prolongedSolution)
	{
		float constraint = 0;
		int count = 0;
		if ((tree.Flags(node) & FemTree.DirichletElementFlag) != 0)
		{
			matrix.SetRowSize(row, count);
			return constraint;
		}

		int d = tree.LocalDepth(node);
		ReadOffset(node, off);
		if (d > 0 && prolongedSolution != null)
		{
			constraint = GetConstraintFromProlongedSolution(node, prolongedSolution, pcStencils[tree.ChildIndexInParent(node)]);
		}

		bool isInterior = PoissonMultigrid.IsInteriorlyOverlapped(degree, degree, d, off[0], off[1], off[2]);
		Array.Clear(pointValues);
		AddPointValues(node, d);
		int nodeIndex = tree.NodeIndex(node);
		matrix.SetRowSize(row, neighbors.Length);
		if (isInterior)
		{
			// General case, so try to make fast
			matrix.Column(row, count) = nodeIndex - offset;
			matrix.Value(row, count++) = (float)(pointValues[center] + ccStencil[center]);
			for (int i = 0; i < neighbors.Length; i++)
			{
				int n = neighbors[i];
				if (i != center && PoissonMultigrid.IsValidFem1Node(tree, n) && (tree.Flags(n) & FemTree.DirichletElementFlag) == 0)
				{
					matrix.Column(row, count) = tree.NodeIndex(n) - offset;
					matrix.Value(row, count++) = (float)(pointValues[i] + ccStencil[i]);
				}
			}
		}
		else
		{
			matrix.Column(row, count) = nodeIndex - offset;
			matrix.Value(row, count++) = (float)f.CcIntegrate(off, off) + pointValues[center];
			for (int i0 = 0; i0 < width; i0++)
			{
				other[0] = off[0] - overlapRadius + i0;
				for (int i1 = 0; i1 < width; i1++)
				{
					other[1] = off[1] - overlapRadius + i1;
					for (int i2 = 0; i2 < width; i2++)
					{
						other[2] = off[2] - overlapRadius + i2;
						int i = ((i0 * width) + i1) * width + i2;
						int n = neighbors[i];
						if (n != node && PoissonMultigrid.IsValidFem1Node(tree, n) && (tree.Flags(n) & FemTree.DirichletElementFlag) == 0)
						{
							matrix.Column(row, count) = tree.NodeIndex(n) - offset;
							matrix.Value(row, count++) = (float)f.CcIntegrate(other, off) + pointValues[i];
						}
					}
				}
			}
		}

		matrix.SetRowSize(row, count);
		return constraint;
	}

	// Port of _getConstraintFromProlongedSolution: the coarser solution's contribution through
	// the parent-child stencil (or exact integrals near the boundary), plus the interpolation
	// constraint.
	private float GetConstraintFromProlongedSolution(int node, float[] prolongedSolution, double[] stencil)
	{
		if (tree.LocalDepth(node) <= 0)
		{
			return 0;
		}

		// This is a conservative estimate as we only need to make sure that the parent nodes don't overlap the child (not the parent itself)
		bool isInterior = PoissonMultigrid.IsInteriorlyOverlapped(tree, degree, degree, tree.Parent(node));

		// Offset the constraints using the solution from lower resolutions.
		float constraint = 0;
		foreach (int idx in parentLoopData[tree.ChildIndexInParent(node)])
		{
			int n = pNeighbors[idx];
			if (!PoissonMultigrid.IsValidFem1Node(tree, n))
			{
				continue;
			}

			float s;
			if (isInterior)
			{
				s = (float)stencil[idx];
			}
			else
			{
				ReadOffset(n, other);
				s = (float)f.PcIntegrate(other, off);
			}

			constraint += prolongedSolution[tree.NodeIndex(n)] * s;
		}

		return constraint + GetInterpolationConstraintFromProlongedSolution(neighbors, node);
	}

	// Port of _addPointValues (Dim = 3, PointD = 0): for every interpolation entry in the space
	// nodes whose support reaches the center node, the center function's value there times
	// SystemDual times the entry's weight, spread over the window entries whose functions are
	// supported on the point.
	private void AddPointValues(int node, int d)
	{
		if (interpolation == null || bsData == null)
		{
			return;
		}

		int leftSupport = -support.SupportStart;
		int leftPointSupport = support.SupportEnd;
		int rightPointSupport = -support.SupportStart;
		int loopStart = overlapRadius - leftSupport;
		int loopEnd = overlapRadius + support.SupportEnd + 1;
		for (int j0 = loopStart; j0 < loopEnd; j0++)
		{
			int idx0 = j0 - overlapRadius;
			for (int j1 = loopStart; j1 < loopEnd; j1++)
			{
				int idx1 = j1 - overlapRadius;
				for (int j2 = loopStart; j2 < loopEnd; j2++)
				{
					int idx2 = j2 - overlapRadius;
					int spaceNode = neighbors[((j0 * width) + j1) * width + j2];
					if (!PoissonMultigrid.IsValidSpaceNode(tree, spaceNode))
					{
						continue;
					}

					int slot = interpolation.Index(tree.NodeIndex(spaceNode));
					if (slot == -1)
					{
						continue;
					}

					// Compute the partial evaluation of all B-splines (and derivatives) that are supported on the point
					bsData.Init(d, interpolation.Value(slot, 0), interpolation.Value(slot, 1), interpolation.Value(slot, 2), off[0] + idx0, off[1] + idx1, off[2] + idx2);

					// The value (and derivatives) of the function of the center node at this point
					double values = (float)bsData.Value(off[0], off[1], off[2]);
					double dualValue = values * (double)systemWeight * (double)interpolation.Value(slot, 3);

					// Compute the bounds of nodes which can be supported on the point
					float dualAsReal = (float)dualValue;
					int start0 = idx0 + overlapRadius - leftPointSupport, end0 = idx0 + overlapRadius + rightPointSupport + 1;
					int start1 = idx1 + overlapRadius - leftPointSupport, end1 = idx1 + overlapRadius + rightPointSupport + 1;
					int firstI = idx2 + overlapRadius - leftPointSupport;
					for (int a0 = start0; a0 < end0; a0++)
					{
						for (int a1 = start1; a1 < end1; a1++)
						{
							float dot = 0;
							dot += (float)(bsData.SubValue(a0 - overlapRadius + off[0], a1 - overlapRadius + off[1]) * dualAsReal);
							double partialDot = dot;
							int sliceBase = ((a0 * width) + a1) * width + firstI;
							for (int i = 0; i < support.SupportSize; i++)
							{
								if (PoissonMultigrid.IsValidFem1Node(tree, neighbors[sliceBase + i]))
								{
									pointValues[sliceBase + i] += (float)(bsData.AxisValue(2, i) * partialDot);
								}
							}
						}
					}
				}
			}
		}
	}

	private void ReadOffset(int node, int[] offset)
	{
		for (int k = 0; k < Dim; k++)
		{
			offset[k] = tree.LocalOffset(node, k);
		}
	}
}
