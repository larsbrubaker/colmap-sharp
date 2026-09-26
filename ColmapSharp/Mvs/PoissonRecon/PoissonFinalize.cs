// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonFinalize: FEMTree::_finalizeForMultigrid and setSortedTreeNodes (thirdparty/
// PoissonRecon/FEMTree.inl) as Poisson::Solver::Solve runs them (no Dirichlet constraints):
// re-root the tree until the widest basis functions at depth 0 fit inside the global grid,
// complete the tree up to the base depth (_setFullDepth), refine wherever the add-node functor
// asks (_refine, with Solve's "depth <= fullDepth"), clear the flags and mark the children of
// nodes below the base depth that are not to be added as ghosts, clip (_clipTree) the subtrees
// below the base depth that carry no normal data, create and un-ghost the elements that
// support prolongation (_supportApproximateProlongation), clear the Dirichlet element marks
// (_markNonBaseDirichletElements), and sort the active nodes, set their space flags and re-key
// the node data and interpolation info to the sorted indices (setSortedTreeNodes). The FEM
// system and multigrid solver (later slices) work on the result. Tier A against
// oracle/poisson_tree_harness.cc: the "density*" runs step by step, and the "density*final"
// runs, which call the vendored finalizeForMultigrid, for the composed FinalizeForMultigrid.
//
// Translation notes: nodes are created with the plain child initializer (_refine and
// _setFullDepth run with ThreadSafe = false) except in _supportApproximateProlongation, which
// uses the thread-safe numbering (FemTree.InitChildren); the recursions keep the C++'s visiting
// order, which fixes the node numbering. _clipTree's and _supportApproximateProlongation's
// ParallelFors touch disjoint subtrees or run in node order with one key here, as with one
// thread in the C++.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Shaping the octree for the multigrid solver. Port of the tree-shaping part of FEMTree's
/// <c>_finalizeForMultigrid</c>.
/// </summary>
public static class PoissonFinalize
{
	/// <summary>
	/// Adds root levels until the degree-<paramref name="maxDegree"/> free basis at local depth
	/// 0 is indexable within the global grid, then renumbers the tree. Port of the re-rooting
	/// loop of <c>_finalizeForMultigrid</c> and the following <c>_init</c>.
	/// </summary>
	public static void Reroot(FemTree tree, int maxDegree)
	{
		int sig = FemSignature.Of(maxDegree, BoundaryType.Free);
		while (tree.LocalInset(0) + BSplineEvaluationData.Begin(sig, 0) < 0
			|| tree.LocalInset(0) + BSplineEvaluationData.End(sig, 0) > (1 << tree.DepthOffset))
		{
			tree.AddRootLevel();
		}

		tree.Init();
	}

	/// <summary>
	/// Refines every node above local depth <paramref name="depth"/> whose degree-maxDegree
	/// free function is in bounds (and every node above the cube). Port of
	/// <c>_setFullDepth&lt;false&gt;( IsotropicUIntPack&lt;Dim,MaxDegree&gt; , allocator , depth )</c>.
	/// </summary>
	public static void SetFullDepth(FemTree tree, int maxDegree, int depth)
	{
		int sig = FemSignature.Of(maxDegree, BoundaryType.Free);
		if (!tree.HasChildren(tree.Root))
		{
			tree.InitChildren(tree.Root);
		}

		for (int c = 0; c < FemTree.ChildCount; c++)
		{
			SetFullDepth(tree, sig, tree.FirstChild(tree.Root) + c, depth);
		}
	}

	/// <summary>
	/// Refines every node one of whose in-bounds children the functor wants (and every node
	/// above the cube). Port of <c>_refine&lt;false&gt;( IsotropicUIntPack&lt;Dim,MaxDegree&gt; ,
	/// allocator , addNodeFunctor )</c>; the functor gets the child's local depth and offset.
	/// </summary>
	public static void Refine(FemTree tree, int maxDegree, Func<int, int, int, int, bool> addNode)
	{
		int sig = FemSignature.Of(maxDegree, BoundaryType.Free);
		if (!tree.HasChildren(tree.Root))
		{
			tree.InitChildren(tree.Root);
		}

		for (int c = 0; c < FemTree.ChildCount; c++)
		{
			Refine(tree, sig, addNode, tree.FirstChild(tree.Root) + c);
		}
	}

	/// <summary>
	/// Clears every flag but the scratch flag, then, in pre-order, sets each node's parent's
	/// ghost flag to "not added and deeper than the base depth" (so the last child visited
	/// decides). Port of the flag pass of <c>_finalizeForMultigrid</c> without Dirichlet data.
	/// </summary>
	public static void MarkGhosts(FemTree tree, int baseDepth, Func<int, int, int, int, bool> addNode)
	{
		tree.ProcessNodes(tree.Root, node =>
		{
			tree.SetFlags(node, (byte)(tree.Flags(node) & FemTree.ScratchFlag));
			bool added = addNode(tree.LocalDepth(node), tree.LocalOffset(node, 0), tree.LocalOffset(node, 1), tree.LocalOffset(node, 2));
			tree.SetGhostFlag(node, !added && tree.LocalDepth(node) > baseDepth);
		});
	}

	/// <summary>
	/// Below local depth <paramref name="fullDepth"/>, marks the children of every node as
	/// ghosts (pruned) when none of their subtrees has data. Port of <c>_clipTree( f , fullDepth )</c>.
	/// </summary>
	public static void ClipTree(FemTree tree, Func<int, bool> hasData, int fullDepth)
	{
		var regularNodes = new List<int>();
		tree.ProcessNodes(tree.Root, node =>
		{
			if (tree.LocalDepth(node) == fullDepth)
			{
				regularNodes.Add(node);
			}

			return tree.LocalDepth(node) < fullDepth;
		});

		// Get the data status of each node
		var nodeHasData = new byte[tree.NodeCount];
		foreach (int regular in regularNodes)
		{
			tree.ProcessNodes(regular, node =>
			{
				if (tree.NodeIndex(node) != -1)
				{
					nodeHasData[tree.NodeIndex(node)] = hasData(node) ? (byte)1 : (byte)0;
				}
			});
		}

		// Pull the data status from the leaves
		foreach (int regular in regularNodes)
		{
			PullHasDataFromChildren(tree, nodeHasData, regular);
		}

		// Mark all children of a node as ghost if none of them have data
		foreach (int regular in regularNodes)
		{
			tree.ProcessNodes(regular, node =>
			{
				if (!tree.HasChildren(node))
				{
					return;
				}

				int brood = tree.FirstChild(node);
				byte childHasData = 0;
				for (int c = 0; c < FemTree.ChildCount; c++)
				{
					if (tree.NodeIndex(brood + c) != -1)
					{
						childHasData |= nodeHasData[tree.NodeIndex(brood + c)];
					}
				}

				for (int c = 0; c < FemTree.ChildCount; c++)
				{
					tree.SetGhostFlag(brood + c, childHasData == 0);
				}
			});
		}
	}

	/// <summary>
	/// True if the node or a descendant has a normal-field entry with a nonzero component.
	/// Port of <c>FEMTree::HasNormalDataFunctor</c>.
	/// </summary>
	public static bool HasNormalData(FemTree tree, SparseNodeData normals, int node)
	{
		int slot = normals.Index(tree.NodeIndex(node));
		if (slot != -1)
		{
			for (int d = 0; d < 3; d++)
			{
				if (normals.Value(slot, d) != 0)
				{
					return true;
				}
			}
		}

		if (tree.HasChildren(node))
		{
			for (int c = 0; c < FemTree.ChildCount; c++)
			{
				if (HasNormalData(tree, normals, tree.FirstChild(node) + c))
				{
					return true;
				}
			}
		}

		return false;
	}

	/// <summary>
	/// For each depth d from maxDepth - 1 down to baseDepth + 1, creates (thread-safe numbering)
	/// and un-ghosts the depth-d neighbors, within the degree-maxDegree overlap radius, of every
	/// node whose children are active, so every element at d + 1 has its supporting elements
	/// at d. Port of <c>_supportApproximateProlongation&lt;MaxDegree&gt;</c>.
	/// </summary>
	public static void SupportApproximateProlongation(FemTree tree, int maxDegree, int maxDepth, int baseDepth)
	{
		int overlapRadius = -BSplineOverlapSizes.For(maxDegree, maxDegree).OverlapStart;
		var key = new NeighborKey(tree, overlapRadius, overlapRadius, resetOnMissing: true);
		key.Set(maxDepth - 1 + tree.DepthOffset);
		for (int d = maxDepth - 1; d > baseDepth; d--)
		{
			// Compute the set of nodes at depth d that have (non-ghost) children at depth d+1.
			var nodes = new List<int>();
			int depth = d;
			tree.ProcessNodes(tree.Root, node =>
			{
				if (tree.LocalDepth(node) == depth && tree.HasChildren(node) && tree.IsActive(tree.FirstChild(node)))
				{
					nodes.Add(node);
				}

				return tree.LocalDepth(node) < depth;
			});

			// Make sure that all finite elements whose support overlaps the support of the finite elements indexed by those nodes are in the tree.
			foreach (int node in nodes)
			{
				foreach (int neighbor in key.GetNeighbors(node, createNodes: true, threadSafe: true))
				{
					tree.SetGhostFlag(neighbor, false);
				}
			}
		}
	}

	/// <summary>
	/// Clears the Dirichlet-element flag of every node that is not a ghost, in the subtrees of
	/// the nodes at the base depth (stopping at ghosts). Port of
	/// <c>_markNonBaseDirichletElements&lt;SystemDegree&gt;</c> for trees without Dirichlet node
	/// flags: the C++ sets each flag to "some leaf in the element's support is a Dirichlet
	/// node" (via setLeafNeighbors), which is false everywhere when, as in Solve without an
	/// envelope, the flag pass has cleared all Dirichlet node flags and no later step sets any.
	/// </summary>
	public static void MarkNonBaseDirichletElements(FemTree tree, int baseDepth)
	{
		var baseNodes = new List<int>();
		tree.ProcessNodes(tree.Root, node =>
		{
			if ((tree.Flags(node) & FemTree.DirichletNodeFlag) != 0)
			{
				throw new InvalidOperationException("Dirichlet node flags are not supported: the port has no envelope constraints.");
			}

			if (tree.LocalDepth(node) == baseDepth)
			{
				baseNodes.Add(node);
			}
		});
		foreach (int node in baseNodes)
		{
			ClearDirichletElements(tree, node);
		}
	}

	/// <summary>
	/// Clears then sets the space flag of every sorted node: local depth at least 0 and every
	/// offset inside [0, 2^depth). Port of <c>_setSpaceValidityFlags</c>.
	/// </summary>
	public static void SetSpaceValidityFlags(FemTree tree, SortedTreeNodes sorted)
	{
		for (int i = 0; i < sorted.Size; i++)
		{
			int node = sorted.TreeNodes[i];
			byte flags = (byte)(tree.Flags(node) & ~FemTree.SpaceFlag);
			if (IsValidSpaceNode(tree, node))
			{
				flags |= FemTree.SpaceFlag;
			}

			tree.SetFlags(node, flags);
		}
	}

	/// <summary>
	/// Sorts the active nodes (SortedTreeNodes, renumbering them to their sorted positions),
	/// sets the space flags, sets the index of inactive nodes to -1, and re-keys the node data
	/// to the new indices. Returns the map (new index to old). Port of
	/// <c>setSortedTreeNodes( interpolationInfos , data )</c>.
	/// </summary>
	public static int[] SetSortedTreeNodes(FemTree tree, SortedTreeNodes sorted, params SparseNodeData?[] data)
	{
		int[] map = sorted.Reset(tree, tree.Root);
		SetSpaceValidityFlags(tree, sorted);
		tree.ProcessNodes(tree.Root, node =>
		{
			if (!tree.IsActive(node))
			{
				tree.SetNodeIndex(node, -1);
			}
		});
		foreach (SparseNodeData? field in data)
		{
			field?.RemapIndices(map, sorted.Size);
		}

		return map;
	}

	/// <summary>
	/// Solve's <c>finalizeForMultigrid&lt;MaxDegree,SystemDegree&gt;( baseDepth , d &lt;= fullDepth ,
	/// HasNormalDataFunctor , iInfo , ( normals , density , aux ) )</c> without Dirichlet data:
	/// shapes the tree, supports prolongation, sorts the nodes and re-keys the data. Returns the
	/// sorted nodes; <paramref name="map"/> is the new-to-old index map.
	/// </summary>
	public static SortedTreeNodes FinalizeForMultigrid(
		FemTree tree,
		int maxDegree,
		int baseDepth,
		int fullDepth,
		SparseNodeData normals,
		out int[] map,
		params SparseNodeData?[] data)
	{
		bool AddNode(int d, int x, int y, int z) => d <= fullDepth;
		Reroot(tree, maxDegree);
		SetFullDepth(tree, maxDegree, baseDepth);
		Refine(tree, maxDegree, AddNode);
		MarkGhosts(tree, baseDepth, AddNode);
		ClipTree(tree, node => tree.LocalDepth(node) <= fullDepth || HasNormalData(tree, normals, node), baseDepth);

		// It is possible for the tree to have become shallower after clipping
		int maxDepth = tree.MaxDepth(tree.Root) - tree.DepthOffset;
		SupportApproximateProlongation(tree, maxDegree, maxDepth, baseDepth);
		MarkNonBaseDirichletElements(tree, baseDepth);
		var sorted = new SortedTreeNodes();
		map = SetSortedTreeNodes(tree, sorted, data);
		return sorted;
	}

	private static void ClearDirichletElements(FemTree tree, int node)
	{
		if ((tree.Flags(node) & FemTree.GhostFlag) != 0)
		{
			return;
		}

		tree.SetFlags(node, (byte)(tree.Flags(node) & ~FemTree.DirichletElementFlag));
		if (tree.HasChildren(node))
		{
			for (int c = 0; c < FemTree.ChildCount; c++)
			{
				ClearDirichletElements(tree, tree.FirstChild(node) + c);
			}
		}
	}

	// FEMTree::isValidSpaceNode.
	private static bool IsValidSpaceNode(FemTree tree, int node)
	{
		int d = tree.LocalDepth(node);
		if (d < 0)
		{
			return false;
		}

		int res = 1 << d;
		for (int dd = 0; dd < 3; dd++)
		{
			int off = tree.LocalOffset(node, dd);
			if (off < 0 || off >= res)
			{
				return false;
			}
		}

		return true;
	}

	private static void SetFullDepth(FemTree tree, int sig, int node, int depth)
	{
		int d = tree.LocalDepth(node);
		bool refine = d < depth && (d < 0 || !OutOfBounds(tree, sig, node));
		if (!refine)
		{
			return;
		}

		if (!tree.HasChildren(node))
		{
			tree.InitChildren(node);
		}

		for (int c = 0; c < FemTree.ChildCount; c++)
		{
			SetFullDepth(tree, sig, tree.FirstChild(node) + c, depth);
		}
	}

	private static void Refine(FemTree tree, int sig, Func<int, int, int, int, bool> addNode, int node)
	{
		int d = tree.LocalDepth(node);
		int childDepth = d + 1;
		int ox = tree.LocalOffset(node, 0), oy = tree.LocalOffset(node, 1), oz = tree.LocalOffset(node, 2);
		bool refine = d < 0;
		for (int c = 0; c < FemTree.ChildCount; c++)
		{
			int cx = ox * 2 + (c & 1);
			int cy = oy * 2 + ((c >> 1) & 1);
			int cz = oz * 2 + ((c >> 2) & 1);
			refine |= !OutOfBounds(sig, childDepth, cx, cy, cz) && addNode(childDepth, cx, cy, cz);
		}

		if (!refine)
		{
			return;
		}

		if (!tree.HasChildren(node))
		{
			tree.InitChildren(node);
		}

		for (int c = 0; c < FemTree.ChildCount; c++)
		{
			Refine(tree, sig, addNode, tree.FirstChild(node) + c);
		}
	}

	private static byte PullHasDataFromChildren(FemTree tree, byte[] nodeHasData, int node)
	{
		int index = tree.NodeIndex(node);
		if (index == -1)
		{
			return 0;
		}

		byte hasData = nodeHasData[index];
		if (tree.HasChildren(node))
		{
			for (int c = 0; c < FemTree.ChildCount; c++)
			{
				hasData |= PullHasDataFromChildren(tree, nodeHasData, tree.FirstChild(node) + c);
			}
		}

		nodeHasData[index] = hasData;
		return hasData;
	}

	// FEMIntegrator::IsOutOfBounds for an isotropic signature at the node's local depth/offset.
	private static bool OutOfBounds(FemTree tree, int sig, int node) =>
		OutOfBounds(sig, tree.LocalDepth(node), tree.LocalOffset(node, 0), tree.LocalOffset(node, 1), tree.LocalOffset(node, 2));

	private static bool OutOfBounds(int sig, int depth, int x, int y, int z)
	{
		if (depth < 0)
		{
			return true;
		}

		BSplineEvaluationData data = BSplineEvaluationData.For(sig);
		return data.OutOfBounds(depth, x) || data.OutOfBounds(depth, y) || data.OutOfBounds(depth, z);
	}
}
