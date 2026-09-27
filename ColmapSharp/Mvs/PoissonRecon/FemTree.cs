// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// FemTree: the octree of thirdparty/PoissonRecon - RegularTreeNode<3,FEMTreeNodeData,...>
// (RegularTree.h/.inl: broods of eight children, depth and offset per node, pre-order
// traversal, ResetDepthAndOffset) together with the parts of FEMTree<3,float> that own it
// (FEMTree.h/.inl: the global root with its first brood, the "space root" one level down,
// the node-index counter every new node takes its index from, FEMTreeNodeData's flags and
// resetNodeIndices). NeighborKey, SortedTreeNodes, PoissonSampleSet, PoissonDensity and
// PoissonSplat operate on it; their per-node data lives in SparseNodeData keyed by NodeIndex.
//
// Storage: millions of input points make millions of nodes, so nodes live in chunked,
// growable arrays indexed by a node handle ("slot") instead of C++ node objects: parent,
// first child (a brood's eight children are contiguous, child c at FirstChild + c, as in the
// C++ allocator), depth, x/y/z offset, node index and flags. A slot is the node's identity
// (the C++ pointer); NodeIndex is FEMTreeNodeData::nodeIndex, which resetNodeIndices and
// SortedTreeNodes renumber. Tier A: the tree shape, the offsets and the node indices equal
// the C++'s (oracle/poisson_tree_harness.cc).
//
// Translation notes:
// - FEMTree's constructor creates the global root (index 0) and its brood (indices 1..8);
//   the unit cube is the root's last child (depth offset 1), so a node's depth here is its
//   global depth and the local (unit-cube) depth is Depth - DepthOffset (LocalDepth).
//   Finalizing re-roots the tree (AddRootLevel, Init), raising the depth offset. The stages
//   before it temporarily renumber the space root's subtree to depth 0 / offset 0
//   (ExtractSubTree, FEMTreeNode::SubTreeExtractor) and restore it afterwards.
// - processNodes is recursive in the C++; here it is too (the recursion depth is the tree
//   depth, at most ~20), with the same "visit, then descend if the functor returned true"
//   order.
// - Nodes are never deleted: pruning marks the parents of pruned broods as ghosts
//   (SetGhostFlag), as the C++ does.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// The FEM octree with flat node storage. Port of PoissonRecon's <c>RegularTreeNode</c> and
/// the tree-owning parts of <c>FEMTree&lt;3,float&gt;</c>.
/// </summary>
public sealed class FemTree
{
	/// <summary>The handle of "no node" (a C++ null pointer).</summary>
	public const int None = -1;

	/// <summary>The number of children of a node (1 &lt;&lt; Dim).</summary>
	public const int ChildCount = 8;

	/// <summary>FEMTreeNodeData flag: part of the partition of the unit cube.</summary>
	public const byte SpaceFlag = 1 << 0;

	/// <summary>FEMTreeNodeData flag: indexes a valid finite element (first signature).</summary>
	public const byte FemFlag1 = 1 << 1;

	/// <summary>FEMTreeNodeData flag: indexes a valid finite element (second signature).</summary>
	public const byte FemFlag2 = 1 << 2;

	/// <summary>FEMTreeNodeData flag: the finite elements should evaluate to zero on this node.</summary>
	public const byte DirichletNodeFlag = 1 << 3;

	/// <summary>FEMTreeNodeData flag: the coefficient of this node is locked to zero.</summary>
	public const byte DirichletElementFlag = 1 << 4;

	/// <summary>FEMTreeNodeData flag: the element's support overlaps geometry constraints.</summary>
	public const byte GeometrySupportedFlag = 1 << 5;

	/// <summary>FEMTreeNodeData flag: the children are pruned out.</summary>
	public const byte GhostFlag = 1 << 6;

	/// <summary>FEMTreeNodeData flag: scratch space for algorithms.</summary>
	public const byte ScratchFlag = 1 << 7;

	// Chunked, so growth never copies (ChunkedArray). Depth and offsets are stored in the
	// widths of COLMAP's build (depth_and_offset_type is unsigned short, USE_DEEP_TREE_NODES
	// unset); offsets truncate to 16 bits on assignment exactly as the C++ does, and depths
	// never exceed a byte (COLMAP's default depth is 13 plus the depth offset).
	private readonly ChunkedArray<int> parent = new();
	private readonly ChunkedArray<int> firstChild = new();
	private readonly ChunkedArray<byte> depth = new();
	private readonly ChunkedArray<ushort> offsets = new();
	private readonly ChunkedArray<int> nodeIndex = new();
	private readonly ChunkedArray<byte> flags = new();

	/// <summary>
	/// A tree with the global root and its first brood, the unit cube being the root's last
	/// child. Port of <c>FEMTree( blockSize )</c> and <c>_init</c>.
	/// </summary>
	public FemTree()
	{
		// Initialize the root
		Root = Allocate(1);
		parent[Root] = None;
		nodeIndex[Root] = NodeCount++;
		InitChildren(Root);
		DepthOffset = 1;
		Init();
	}

	/// <summary>
	/// The signature the FEM_FLAG_1 flags were last set for, or -1 when they are stale
	/// (FEMTree::_femSigs1, memset to -1 by the constructor and setSortedTreeNodes). The flags are
	/// only recomputed when a caller asks for a different signature (PoissonFemConstraints).
	/// </summary>
	public int FemSignature1 { get; set; } = -1;

	/// <summary>The same for FEM_FLAG_2 (FEMTree::_femSigs2).</summary>
	public int FemSignature2 { get; set; } = -1;

	/// <summary>The global root (FEMTree::_tree).</summary>
	public int Root { get; }

	/// <summary>The node covering the unit cube (FEMTree::spaceRoot).</summary>
	public int SpaceRoot { get; private set; }

	/// <summary>
	/// How many levels the space root sits below the global root (FEMTree::_depthOffset): 1
	/// after construction, larger once finalizing re-roots the tree (<see cref="AddRootLevel"/>).
	/// </summary>
	public int DepthOffset { get; private set; }

	/// <summary>
	/// The offset of the unit cube's first cell at a local depth within the global grid. Port of
	/// <c>FEMTree::_localInset</c>.
	/// </summary>
	public int LocalInset(int localDepth) => DepthOffset == 0 ? 0 : 1 << (localDepth + DepthOffset - 1);

	/// <summary>The node's depth below the unit cube (negative above it). Port of <c>_localDepth</c>.</summary>
	public int LocalDepth(int node) => depth[node] - DepthOffset;

	/// <summary>
	/// The node's offset along axis d within the unit cube's grid at its local depth, or -1
	/// for nodes above the cube. Port of <c>_localDepthAndOffset</c>.
	/// </summary>
	public int LocalOffset(int node, int d)
	{
		int localDepth = depth[node] - DepthOffset;
		return localDepth < 0 ? -1 : offsets[3 * node + d] - LocalInset(localDepth);
	}

	/// <summary>
	/// Inserts a new brood between the global root and its children, so the old children hang
	/// off the new brood's last node and the unit cube moves one level down, and increments the
	/// depth offset. The old last child's children move to the old first child. Port of the
	/// re-rooting loop body of <c>_finalizeForMultigrid</c> (with <c>NewBrood</c>).
	/// </summary>
	public void AddRootLevel()
	{
		int oldChildren = firstChild[Root];
		if (oldChildren == None)
		{
			throw new InvalidOperationException("Expected children");
		}

		// NewBrood: numbered in child order; depth and offsets are reset by Init below.
		int newChildren = Allocate(ChildCount);
		for (int idx = 0; idx < ChildCount; idx++)
		{
			int child = newChildren + idx;
			firstChild[child] = None;
			nodeIndex[child] = NodeCount++;
			flags[child] = 0;
		}

		int last = oldChildren + ChildCount - 1;
		if (firstChild[last] != None)
		{
			for (int c = 0; c < ChildCount; c++)
			{
				parent[firstChild[last] + c] = oldChildren;
			}

			firstChild[oldChildren] = firstChild[last];
			firstChild[last] = None;
		}

		for (int c = 0; c < ChildCount; c++)
		{
			parent[oldChildren + c] = newChildren + ChildCount - 1;
		}

		firstChild[newChildren + ChildCount - 1] = oldChildren;
		for (int c = 0; c < ChildCount; c++)
		{
			parent[newChildren + c] = Root;
		}

		firstChild[Root] = newChildren;
		DepthOffset++;
	}

	/// <summary>
	/// Renumbers depths and offsets from the global root and finds the space root: the root's
	/// last child, then first children down to the depth offset. Port of <c>FEMTree::_init</c>.
	/// </summary>
	public void Init()
	{
		ResetDepthAndOffset(Root, 0, 0, 0, 0);
		int spaceRoot = Root;
		for (int d = 0; d < DepthOffset; d++)
		{
			if (firstChild[spaceRoot] == None)
			{
				throw new InvalidOperationException($"Expected child node: {d} / {DepthOffset}");
			}

			spaceRoot = d == 0 ? firstChild[spaceRoot] + ChildCount - 1 : firstChild[spaceRoot];
		}

		SpaceRoot = spaceRoot;
	}

	/// <summary>
	/// Sets or clears the ghost flag of the node's parent (children of a ghost are pruned).
	/// Port of <c>SetGhostFlag&lt;Dim&gt;( node , flag )</c>.
	/// </summary>
	public void SetGhostFlag(int node, bool flag)
	{
		if (node != None && parent[node] != None)
		{
			int p = parent[node];
			flags[p] = flag ? (byte)(flags[p] | GhostFlag) : (byte)(flags[p] & ~GhostFlag);
		}
	}

	/// <summary>The next node index to hand out (FEMTree::_nodeCount).</summary>
	public int NodeCount { get; private set; }

	/// <summary>The number of allocated node slots.</summary>
	public int SlotCount { get; private set; }

	/// <summary>The parent's slot, or <see cref="None"/> for the root.</summary>
	public int Parent(int node) => parent[node];

	/// <summary>The first child's slot (children are contiguous), or <see cref="None"/> for a leaf.</summary>
	public int FirstChild(int node) => firstChild[node];

	/// <summary>True if the node has children.</summary>
	public bool HasChildren(int node) => firstChild[node] != None;

	/// <summary>The node's depth below the global root. Port of <c>depth()</c>.</summary>
	public int Depth(int node) => depth[node];

	/// <summary>The node's offset along axis d at its depth. Port of <c>depthAndOffset</c>.</summary>
	public int Offset(int node, int d) => offsets[3 * node + d];

	/// <summary>FEMTreeNodeData::nodeIndex.</summary>
	public int NodeIndex(int node) => nodeIndex[node];

	/// <summary>Sets FEMTreeNodeData::nodeIndex.</summary>
	public void SetNodeIndex(int node, int index) => nodeIndex[node] = index;

	/// <summary>FEMTreeNodeData::flags.</summary>
	public byte Flags(int node) => flags[node];

	/// <summary>Sets FEMTreeNodeData::flags.</summary>
	public void SetFlags(int node, byte value) => flags[node] = value;

	/// <summary>
	/// True for the root, the absent node, and children of ghost nodes. Port of
	/// <c>GetGhostFlag&lt;Dim&gt;( node )</c>.
	/// </summary>
	public bool IsGhost(int node) => node == None || parent[node] == None || (flags[parent[node]] & GhostFlag) != 0;

	/// <summary>Port of <c>IsActiveNode&lt;Dim&gt;</c>.</summary>
	public bool IsActive(int node) => !IsGhost(node);

	/// <summary>The index of a child within its brood (C++ <c>node - node->parent->children</c>).</summary>
	public int ChildIndexInParent(int node) => node - firstChild[parent[node]];

	/// <summary>
	/// Creates the node's eight children, numbering them from the node counter in child order.
	/// Port of <c>initChildren&lt;false&gt;</c> with FEMTree's node initializer.
	/// </summary>
	/// <remarks>
	/// <paramref name="threadSafe"/> selects the C++ <c>initChildren&lt;true&gt;</c>
	/// (<c>_initChildren_s</c>), which the creating neighbor lookups of the density and splatting
	/// stages use. Its node-initializer loop is nested inside the per-child loop, so every child
	/// is numbered eight times: the counter advances by 64 per brood and child c keeps
	/// index base + 56 + c. That is upstream behavior (harmless there: the indices stay unique,
	/// and resetNodeIndices later renumbers densely), and the port reproduces it because node
	/// indices key every SparseNodeData and so decide the order of their entries.
	/// </remarks>
	public void InitChildren(int node, bool threadSafe = false)
	{
		int brood = Allocate(ChildCount);
		firstChild[node] = brood;
		int d = depth[node];
		if (threadSafe)
		{
			NodeCount += ChildCount * (ChildCount - 1);
		}

		for (int idx = 0; idx < ChildCount; idx++)
		{
			int child = brood + idx;
			parent[child] = node;
			firstChild[child] = None;
			nodeIndex[child] = NodeCount++;
			flags[child] = 0;
			depth[child] = (byte)(d + 1);
			for (int dim = 0; dim < 3; dim++)
			{
				offsets[3 * child + dim] = (ushort)((offsets[3 * node + dim] << 1) | ((idx >> dim) & 1));
			}
		}
	}

	/// <summary>
	/// Sets the node's depth and offset and renumbers its subtree to match. Port of
	/// <c>ResetDepthAndOffset</c>.
	/// </summary>
	public void ResetDepthAndOffset(int node, int nodeDepth, int x, int y, int z)
	{
		depth[node] = (byte)nodeDepth;
		offsets[3 * node] = (ushort)x;
		offsets[3 * node + 1] = (ushort)y;
		offsets[3 * node + 2] = (ushort)z;
		if (firstChild[node] != None)
		{
			for (int c = 0; c < ChildCount; c++)
			{
				ResetDepthAndOffset(firstChild[node] + c, nodeDepth + 1, (x << 1) | (c & 1), (y << 1) | ((c >> 1) & 1), (z << 1) | ((c >> 2) & 1));
			}
		}
	}

	/// <summary>
	/// The node's cell center and width in its root's frame, in float. Port of
	/// <c>centerAndWidth</c>.
	/// </summary>
	public void CenterAndWidth(int node, Span<float> center, out float width)
	{
		width = (float)(1.0 / (1 << depth[node]));
		for (int d = 0; d < 3; d++)
		{
			center[d] = (float)(0.5 + offsets[3 * node + d]) * width;
		}
	}

	/// <summary>
	/// The node's cell corner and width in its root's frame, in float. Port of
	/// <c>startAndWidth</c>.
	/// </summary>
	public void StartAndWidth(int node, Span<float> start, out float width)
	{
		width = (float)(1.0 / (1 << depth[node]));
		for (int d = 0; d < 3; d++)
		{
			start[d] = (float)offsets[3 * node + d] * width;
		}
	}

	/// <summary>
	/// The node's cell corner and width in the unit cube's frame (local depth and offset), in
	/// float. Port of <c>FEMTree::_startAndWidth</c>; the same as <see cref="StartAndWidth"/>
	/// inside a <see cref="ExtractSubTree"/> scope, where the depth offset is 0.
	/// </summary>
	public void LocalStartAndWidth(int node, Span<float> start, out float width)
	{
		int localDepth = LocalDepth(node);
		width = localDepth >= 0 ? (float)(1.0 / (1 << localDepth)) : (float)(1.0 * (1 << -localDepth));
		for (int d = 0; d < 3; d++)
		{
			start[d] = (float)LocalOffset(node, d) * width;
		}
	}

	/// <summary>
	/// Makes <paramref name="node"/> look like a root until the returned scope is disposed: no
	/// parent, depth 0, offset 0, its subtree renumbered to match, and a depth offset of 0. Port of
	/// <c>FEMTree::SubTreeExtractor</c> (and <c>RegularTreeNode::SubTreeExtractor</c>), which
	/// every FEMTree stage wraps around its work on the space root, so depths inside are local
	/// (unit-cube) depths and neighbor windows stop at the cube's faces.
	/// </summary>
	public SubTreeScope ExtractSubTree(int node) => new(this, node);

	/// <summary>The scope returned by <see cref="ExtractSubTree"/>; disposing restores the node.</summary>
	public readonly struct SubTreeScope : IDisposable
	{
		private readonly FemTree tree;
		private readonly int node;
		private readonly int savedParent;
		private readonly int savedDepth;
		private readonly int savedX;
		private readonly int savedY;
		private readonly int savedZ;
		private readonly int savedDepthOffset;

		internal SubTreeScope(FemTree tree, int node)
		{
			this.tree = tree;
			this.node = node;
			savedParent = tree.parent[node];
			savedDepth = tree.depth[node];
			savedX = tree.offsets[3 * node];
			savedY = tree.offsets[3 * node + 1];
			savedZ = tree.offsets[3 * node + 2];
			savedDepthOffset = tree.DepthOffset;
			tree.parent[node] = None;
			tree.ResetDepthAndOffset(node, 0, 0, 0, 0);

			// FEMTree::SubTreeExtractor zeroes the depth offset too, so local depths are the
			// extracted subtree's depths.
			tree.DepthOffset = 0;
		}

		/// <summary>Restores the node's parent, depth and offset (and its subtree's).</summary>
		public void Dispose()
		{
			tree.ResetDepthAndOffset(node, savedDepth, savedX, savedY, savedZ);
			tree.parent[node] = savedParent;
			tree.DepthOffset = savedDepthOffset;
		}
	}

	/// <summary>
	/// The child whose cell contains p (bit d set when p is past the center along d). Port of
	/// <c>ChildIndex</c>.
	/// </summary>
	public static int ChildIndex(ReadOnlySpan<float> center, ReadOnlySpan<float> p)
	{
		int cIndex = 0;
		for (int d = 0; d < 3; d++)
		{
			if (p[d] > center[d])
			{
				cIndex |= 1 << d;
			}
		}

		return cIndex;
	}

	/// <summary>The depth of the deepest node below this one (0 for a leaf). Port of <c>maxDepth</c>.</summary>
	public int MaxDepth(int node)
	{
		if (firstChild[node] == None)
		{
			return 0;
		}

		int c = 0;
		for (int i = 0; i < ChildCount; i++)
		{
			int d = MaxDepth(firstChild[node] + i);
			if (i == 0 || d > c)
			{
				c = d;
			}
		}

		return c + 1;
	}

	/// <summary>
	/// Pre-order traversal: visits the node, and descends into a node's children only if the
	/// functor returned true for it. Port of <c>processNodes</c> with a bool functor.
	/// </summary>
	public void ProcessNodes(int node, Func<int, bool> nodeFunctor)
	{
		if (nodeFunctor(node) && firstChild[node] != None)
		{
			ProcessChildNodes(node, nodeFunctor);
		}
	}

	/// <summary>Pre-order traversal of every node. Port of <c>processNodes</c> with a void functor.</summary>
	public void ProcessNodes(int node, Action<int> nodeFunctor)
	{
		ProcessNodes(node, n =>
		{
			nodeFunctor(n);
			return true;
		});
	}

	/// <summary>
	/// Renumbers every node in pre-order from 0 and clears the given flags. Returns map with
	/// map[new index] = old index (sized by the old node count, like the C++). Port of
	/// <c>resetNodeIndices( flagsToClear , data )</c>; the caller reorders its node data with the map.
	/// </summary>
	public int[] ResetNodeIndices(byte flagsToClear)
	{
		byte mask = (byte)~flagsToClear;
		var map = new int[NodeCount];
		NodeCount = 0;
		ProcessNodes(Root, n =>
		{
			int old = nodeIndex[n];
			nodeIndex[n] = NodeCount++;
			map[nodeIndex[n]] = old;
			flags[n] &= mask;
		});
		return map;
	}

	private void ProcessChildNodes(int node, Func<int, bool> nodeFunctor)
	{
		int brood = firstChild[node];
		for (int c = 0; c < ChildCount; c++)
		{
			if (nodeFunctor(brood + c) && firstChild[brood + c] != None)
			{
				ProcessChildNodes(brood + c, nodeFunctor);
			}
		}
	}

	private int Allocate(int count)
	{
		int start = SlotCount;
		int needed = start + count;
		parent.EnsureCapacity(needed);
		firstChild.EnsureCapacity(needed);
		depth.EnsureCapacity(needed);
		offsets.EnsureCapacity(3L * needed);
		nodeIndex.EnsureCapacity(needed);
		flags.EnsureCapacity(needed);

		for (int i = start; i < needed; i++)
		{
			firstChild[i] = None;
			flags[i] = 0;
		}

		SlotCount = needed;
		return start;
	}
}
