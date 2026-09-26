// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonMultigrid: the per-node helpers FEMTree's system assembly and solver share, on the
// finalized tree (FemTree + SortedTreeNodes): the FEM validity flags (_setFEM1ValidityFlags,
// _setFEM2ValidityFlags, isValidFEMNode, _isValidFEM1Node/_isValidFEM2Node in FEMTree.inl/.h),
// BaseFEMIntegrator's IsInteriorlyOverlapped and ParentOverlapBounds loop tables
// (WindowLoopData, FEMTree.h), and the restriction (_downSample) and prolongation (_upSample)
// of per-node values between adjacent depths (FEMTree.System.inl). PoissonFemConstraints
// assembles the right-hand side with them; the solver (a later slice) restricts residuals and
// prolongs solutions with the same two functions. Tier A (oracle/poisson_system_harness.cc,
// through PoissonFemConstraints).
//
// Translation notes:
// - Only isotropic 3D signatures (all axes alike), as everywhere in the port.
// - Per-node values are float arrays with `width` floats per node (1 for constraints and
//   solutions, 3 for the normal coefficients), indexed by node index minus a caller-given
//   offset, standing in for the C++'s Pointer( C ) + _sNodesBegin( depth ) arithmetic. A
//   Point< float , 3 > update `c += (C)( x * (Real)s )` is per component, as float.
// - The C++ runs the loops through ThreadPool::ParallelFor; each node writes only its own
//   slot there, and here they run in order.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// FEM validity flags, overlap tests and two-scale transfers on the finalized tree. Port of the
/// matching members of PoissonRecon's <c>FEMTree&lt;3,float&gt;</c>.
/// </summary>
public static class PoissonMultigrid
{
	/// <summary>The first sorted index at a local depth. Port of <c>_sNodesBegin( d )</c>.</summary>
	public static int Begin(FemTree tree, SortedTreeNodes sorted, int localDepth) => sorted.Begin(localDepth + tree.DepthOffset);

	/// <summary>The end of the sorted indices at a local depth. Port of <c>_sNodesEnd( d )</c>.</summary>
	public static int End(FemTree tree, SortedTreeNodes sorted, int localDepth) => sorted.End(localDepth + tree.DepthOffset);

	/// <summary>The finalized tree's local max depth (FEMTree::_maxDepth after setSortedTreeNodes).</summary>
	public static int MaxDepth(FemTree tree) => tree.MaxDepth(tree.Root) - tree.DepthOffset;

	/// <summary>
	/// Sets FEM_FLAG_1 on every sorted node that supports a <paramref name="signature"/> function,
	/// unless the flags are already set for that signature. Port of <c>_setFEM1ValidityFlags</c>.
	/// </summary>
	public static void SetFem1ValidityFlags(FemTree tree, SortedTreeNodes sorted, int signature)
	{
		if (tree.FemSignature1 == signature)
		{
			return;
		}

		tree.FemSignature1 = signature;
		SetFemFlags(tree, sorted, signature, FemTree.FemFlag1);
	}

	/// <summary>The same for FEM_FLAG_2. Port of <c>_setFEM2ValidityFlags</c>.</summary>
	public static void SetFem2ValidityFlags(FemTree tree, SortedTreeNodes sorted, int signature)
	{
		if (tree.FemSignature2 == signature)
		{
			return;
		}

		tree.FemSignature2 = signature;
		SetFemFlags(tree, sorted, signature, FemTree.FemFlag2);
	}

	/// <summary>Active with FEM_FLAG_1 set (None is invalid). Port of <c>_isValidFEM1Node</c>.</summary>
	public static bool IsValidFem1Node(FemTree tree, int node) => !tree.IsGhost(node) && (tree.Flags(node) & FemTree.FemFlag1) != 0;

	/// <summary>Active with FEM_FLAG_2 set (None is invalid). Port of <c>_isValidFEM2Node</c>.</summary>
	public static bool IsValidFem2Node(FemTree tree, int node) => !tree.IsGhost(node) && (tree.Flags(node) & FemTree.FemFlag2) != 0;

	/// <summary>
	/// True if every degree-<paramref name="degree2"/> function overlapping the
	/// degree-<paramref name="degree1"/> function at (depth, off) is interiorly supported, on all
	/// axes. Port of <c>BaseFEMIntegrator::IsInteriorlyOverlapped</c>.
	/// </summary>
	public static bool IsInteriorlyOverlapped(int degree1, int degree2, int depth, int x, int y, int z)
	{
		if (depth < 0)
		{
			return false;
		}

		BSplineOverlapSizes overlap = BSplineOverlapSizes.For(degree1, degree2);
		BSplineSupportSizes support = BSplineSupportSizes.For(degree2);
		int begin = -overlap.OverlapStart - support.SupportStart;
		int end = (1 << depth) - overlap.OverlapEnd - support.SupportEnd;
		return x >= begin && x < end && y >= begin && y < end && z >= begin && z < end;
	}

	/// <summary>True for a node at an interiorly overlapped local depth and offset.</summary>
	public static bool IsInteriorlyOverlapped(FemTree tree, int degree1, int degree2, int node) =>
		IsInteriorlyOverlapped(degree1, degree2, tree.LocalDepth(node), tree.LocalOffset(node, 0), tree.LocalOffset(node, 1), tree.LocalOffset(node, 2));

	/// <summary>
	/// Per child corner, the indices into an OverlapSize&lt;degree1,degree2&gt;^3 window of the
	/// degree-1 functions (around the parent) that overlap a degree-2 child at that corner, in
	/// window order. Port of <c>WindowLoopData</c> built from
	/// <c>BaseFEMIntegrator::ParentOverlapBounds( degree1 , degree2 , corner , ... )</c>.
	/// </summary>
	public static int[][] ParentOverlapLoopData(int degree1, int degree2)
	{
		BSplineOverlapSizes overlap = BSplineOverlapSizes.For(degree1, degree2);
		int n = overlap.OverlapSize;
		return CornerLoopData(n, (c, d) => (overlap.ParentOverlapStart((c >> d) & 1) - overlap.OverlapStart, overlap.ParentOverlapEnd((c >> d) & 1) - overlap.OverlapStart + 1));
	}

	/// <summary>
	/// Adds to each valid coarse node at highDepth - 1 the restriction of the fine values at
	/// <paramref name="highDepth"/>: the up-sample stencil's weights of its valid children
	/// (FEM_FLAG_1), or the exact coefficients near the boundary. Values are indexed by node
	/// index minus the offsets. Port of <c>_downSample</c> (the FEM_FLAG_1 flags must be set for
	/// the restriction's signature).
	/// </summary>
	public static void DownSample(FemTree tree, SortedTreeNodes sorted, RestrictionProlongation rp, int highDepth, float[] fine, int fineOffset, float[] coarse, int coarseOffset, int width)
	{
		int lowDepth = highDepth - 1;
		if (lowDepth < 0)
		{
			return;
		}

		BSplineSupportSizes sizes = BSplineSupportSizes.For(FemSignature.Degree(rp.Signature));
		int degree = sizes.Degree;
		var neighborKey = new NeighborKey(tree, -sizes.UpSampleStart, sizes.UpSampleEnd, resetOnMissing: false);
		neighborKey.Set(lowDepth + tree.DepthOffset);
		rp.Init(highDepth);
		double[] upSampleStencil = rp.SetStencil();
		var neighbors = new int[neighborKey.WindowSize];
		Span<int> off = stackalloc int[3];
		Span<int> childOff = stackalloc int[3];
		int begin = Begin(tree, sorted, lowDepth), end = End(tree, sorted, lowDepth);
		for (int i = begin; i < end; i++)
		{
			int pNode = sorted.TreeNodes[i];
			if (!IsValidFem1Node(tree, pNode))
			{
				continue;
			}

			int d = tree.LocalDepth(pNode);
			for (int k = 0; k < 3; k++)
			{
				off[k] = tree.LocalOffset(pNode, k);
			}

			neighborKey.GetNeighbors(pNode);
			Array.Fill(neighbors, FemTree.None);
			neighborKey.GetChildNeighbors(0, d + tree.DepthOffset, neighbors);
			int coarseSlot = (i - coarseOffset) * width;

			// Want to make sure test if contained children are interior.
			// This is more conservative because we are test that overlapping children are interior
			bool isInterior = IsInteriorlyOverlapped(degree, degree, d, off[0], off[1], off[2]);
			for (int j = 0; j < neighbors.Length; j++)
			{
				int node = neighbors[j];
				if (!IsValidFem1Node(tree, node))
				{
					continue;
				}

				float s;
				if (isInterior)
				{
					s = (float)upSampleStencil[j];
				}
				else
				{
					for (int k = 0; k < 3; k++)
					{
						childOff[k] = tree.LocalOffset(node, k);
					}

					s = (float)rp.UpSampleCoefficient(off, childOff);
				}

				int fineSlot = (tree.NodeIndex(node) - fineOffset) * width;
				for (int k = 0; k < width; k++)
				{
					coarse[coarseSlot + k] += fine[fineSlot + k] * s;
				}
			}
		}
	}

	/// <summary>
	/// Adds to each valid fine node at <paramref name="highDepth"/> the prolongation of the
	/// coarse values at highDepth - 1: the down-sample stencil's weights of the valid parent
	/// neighbors (FEM_FLAG_1), or the exact coefficients near the boundary. Values are indexed by
	/// node index minus the offsets. Port of <c>_upSample</c> (the FEM_FLAG_1 flags must be set for
	/// the prolongation's signature).
	/// </summary>
	public static void UpSample(FemTree tree, SortedTreeNodes sorted, RestrictionProlongation rp, int highDepth, float[] coarse, int coarseOffset, float[] fine, int fineOffset, int width)
	{
		int lowDepth = highDepth - 1;
		if (lowDepth < 0)
		{
			return;
		}

		BSplineSupportSizes sizes = BSplineSupportSizes.For(FemSignature.Degree(rp.Signature));
		int degree = sizes.Degree;
		var neighborKey = new NeighborKey(tree, -sizes.DownSample0Start, sizes.DownSample1End, resetOnMissing: false);
		neighborKey.Set(lowDepth + tree.DepthOffset);
		rp.Init(highDepth);
		double[][] downSampleStencils = rp.SetStencils();
		int[][] loopData = CornerLoopData(rp.DownSampleSize, (c, d) => (sizes.DownSampleStart((c >> d) & 1) - sizes.DownSample0Start, -sizes.DownSample0Start + sizes.DownSampleEnd((c >> d) & 1) + 1));
		Span<int> off = stackalloc int[3];
		Span<int> parentOff = stackalloc int[3];
		int begin = Begin(tree, sorted, highDepth), end = End(tree, sorted, highDepth);

		// For Dirichlet constraints, can't get to all children from parents because boundary nodes are invalid
		for (int i = begin; i < end; i++)
		{
			int cNode = sorted.TreeNodes[i];
			if (!IsValidFem1Node(tree, cNode))
			{
				continue;
			}

			int c = tree.ChildIndexInParent(cNode);
			int parent = tree.Parent(cNode);
			int[] neighbors = neighborKey.GetNeighbors(parent);

			// Want to make sure test if contained children are interior.
			// This is more conservative because we are test that overlapping children are interior
			bool isInterior = IsInteriorlyOverlapped(tree, degree, degree, parent);
			int fineSlot = (tree.NodeIndex(cNode) - fineOffset) * width;
			double[] stencil = downSampleStencils[c];
			if (!isInterior)
			{
				for (int k = 0; k < 3; k++)
				{
					off[k] = tree.LocalOffset(cNode, k);
				}
			}

			foreach (int idx in loopData[c])
			{
				int node = neighbors[idx];
				if (!IsValidFem1Node(tree, node))
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
					for (int k = 0; k < 3; k++)
					{
						parentOff[k] = tree.LocalOffset(node, k);
					}

					s = (float)rp.UpSampleCoefficient(parentOff, off);
				}

				int coarseSlot = (tree.NodeIndex(node) - coarseOffset) * width;
				for (int k = 0; k < width; k++)
				{
					fine[fineSlot + k] += coarse[coarseSlot + k] * s;
				}
			}
		}
	}

	// WindowLoopData: per corner c, the window indices ((i0 * n) + i1) * n + i2 with
	// bounds(c, d) = [start, end) on each axis d, axis 0 outermost (WindowLoop's order).
	private static int[][] CornerLoopData(int n, Func<int, int, (int Start, int End)> bounds)
	{
		var data = new int[8][];
		for (int c = 0; c < 8; c++)
		{
			var (s0, e0) = bounds(c, 0);
			var (s1, e1) = bounds(c, 1);
			var (s2, e2) = bounds(c, 2);
			var indices = new List<int>();
			for (int i0 = s0; i0 < e0; i0++)
			{
				for (int i1 = s1; i1 < e1; i1++)
				{
					for (int i2 = s2; i2 < e2; i2++)
					{
						indices.Add(((i0 * n) + i1) * n + i2);
					}
				}
			}

			data[c] = indices.ToArray();
		}

		return data;
	}

	// The flag loop of _setFEM{1,2}ValidityFlags with isValidFEMNode: active, local depth >= 0
	// and a function of the signature at the offset on every axis.
	private static void SetFemFlags(FemTree tree, SortedTreeNodes sorted, int signature, byte flag)
	{
		BSplineEvaluationData data = BSplineEvaluationData.For(signature);
		for (int i = 0; i < sorted.Size; i++)
		{
			int node = sorted.TreeNodes[i];
			byte flags = (byte)(tree.Flags(node) & ~flag);
			int d = tree.LocalDepth(node);
			if (!tree.IsGhost(node) && d >= 0 &&
				!data.OutOfBounds(d, tree.LocalOffset(node, 0)) && !data.OutOfBounds(d, tree.LocalOffset(node, 1)) && !data.OutOfBounds(d, tree.LocalOffset(node, 2)))
			{
				flags |= flag;
			}

			tree.SetFlags(node, flags);
		}
	}
}
