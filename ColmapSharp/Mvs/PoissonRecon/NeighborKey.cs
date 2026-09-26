// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// NeighborKey: RegularTreeNode::NeighborKey and ::ConstNeighborKey from
// thirdparty/PoissonRecon/RegularTree.h/.inl for Dim = 3 with isotropic radii - the cached,
// per-depth windows of same-depth neighbors (offsets -left..+right along each axis) that
// every splatting, constraint and evaluation loop of FEMTree walks. A window at depth d is
// derived from the parent's window at depth d-1; the creating variant (getNeighbors<true,..>)
// adds the missing children of existing parent neighbors, handing out node indices in window
// order. Tier A: same windows and same creation order as the C++
// (oracle/poisson_tree_harness.cc).
//
// Translation notes:
// - A window is W^3 node handles (W = left + right + 1), indexed ((i0 * W) + i1) * W + i2 with
//   axis 0 outermost, as Window::StaticWindow lays them out; the C++ recursion over axes
//   (_Run) is the triple loop in NeighborsLoop.
// - NeighborKey recomputes a cached window whose center matches but that holds a null
//   (neighbors may have been created since); ConstNeighborKey does not. The constructor's
//   resetOnMissing flag selects which.
// - All of PoissonRecon's keys in the Poisson path are isotropic, so one radius pair serves
//   all three axes. The radius overloads (windows of other radii, used by the FEM system)
//   take an isotropic pair too.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Cached same-depth neighbor windows along a root-to-node path. Port of PoissonRecon's
/// <c>NeighborKey</c> (resetOnMissing = true) and <c>ConstNeighborKey</c> (false).
/// </summary>
public sealed class NeighborKey
{
	private readonly FemTree tree;
	private readonly bool resetOnMissing;
	private int[][] neighbors = [];

	/// <summary>A key for windows reaching <paramref name="leftRadius"/> below and <paramref name="rightRadius"/> above.</summary>
	public NeighborKey(FemTree tree, int leftRadius, int rightRadius, bool resetOnMissing)
	{
		this.tree = tree;
		this.resetOnMissing = resetOnMissing;
		LeftRadius = leftRadius;
		RightRadius = rightRadius;
		Width = leftRadius + rightRadius + 1;
		WindowSize = Width * Width * Width;
		CenterIndex = ((leftRadius * Width) + leftRadius) * Width + leftRadius;
		KeyDepth = -1;
	}

	/// <summary>The radius toward lower offsets.</summary>
	public int LeftRadius { get; }

	/// <summary>The radius toward higher offsets.</summary>
	public int RightRadius { get; }

	/// <summary>The window width along each axis.</summary>
	public int Width { get; }

	/// <summary>The number of entries in a window.</summary>
	public int WindowSize { get; }

	/// <summary>The index of the center (the node itself) in a window. Port of <c>CenterIndex</c>.</summary>
	public int CenterIndex { get; }

	/// <summary>The deepest depth the key caches. Port of <c>depth()</c>.</summary>
	public int KeyDepth { get; private set; }

	/// <summary>Allocates (cleared) windows for depths 0..depth. Port of <c>set( depth )</c>.</summary>
	public void Set(int depth)
	{
		KeyDepth = depth;
		neighbors = new int[Math.Max(depth + 1, 0)][];
		for (int d = 0; d <= depth; d++)
		{
			neighbors[d] = new int[WindowSize];
			Array.Fill(neighbors[d], FemTree.None);
		}
	}

	/// <summary>The cached window at a depth (valid after GetNeighbors of a node at that depth).</summary>
	public int[] Window(int depth) => neighbors[depth];

	/// <summary>
	/// The window around <paramref name="node"/>: entry ((i0 * W) + i1) * W + i2 is the node at
	/// offset (i0 - left, i1 - left, i2 - left), or <see cref="FemTree.None"/>. With
	/// <paramref name="createNodes"/>, missing children of existing parent neighbors are created.
	/// Port of <c>getNeighbors&lt;CreateNodes,ThreadSafe&gt;( node , ... )</c> (and ConstNeighborKey's);
	/// <paramref name="threadSafe"/> selects how created children are numbered (see
	/// <see cref="FemTree.InitChildren"/>). The returned array is the key's cache: it changes with
	/// the next call.
	/// </summary>
	public int[] GetNeighbors(int node, bool createNodes = false, bool threadSafe = false)
	{
		int nodeDepth = tree.Depth(node);
		int[] window = neighbors[nodeDepth];

		// This is required in case the neighbors have been constructed between the last call to getNeighbors and this one
		if (resetOnMissing && node == window[CenterIndex])
		{
			bool reset = false;
			for (int i = 0; i < WindowSize; i++)
			{
				if (window[i] == FemTree.None)
				{
					reset = true;
				}
			}

			if (reset)
			{
				window[CenterIndex] = FemTree.None;
			}
		}

		if (node != window[CenterIndex])
		{
			for (int d = nodeDepth + 1; d <= KeyDepth && neighbors[d][CenterIndex] != FemTree.None; d++)
			{
				neighbors[d][CenterIndex] = FemTree.None;
			}

			Array.Fill(window, FemTree.None);
			int parent = tree.Parent(node);
			if (parent == FemTree.None)
			{
				window[CenterIndex] = node;
			}
			else
			{
				int[] parentWindow = GetNeighbors(parent, createNodes, threadSafe);
				NeighborsLoop(parentWindow, window, tree.ChildIndexInParent(node), createNodes, threadSafe);
			}
		}

		return window;
	}

	/// <summary>
	/// The window of child <paramref name="cIdx"/> of the center node cached at depth
	/// <paramref name="d"/>, written to <paramref name="childNeighbors"/>; returns how many
	/// entries are non-null (0 if nothing is cached there). Port of
	/// <c>getChildNeighbors( cIdx , d , childNeighbors )</c>.
	/// </summary>
	public int GetChildNeighbors(int cIdx, int d, int[] childNeighbors, bool createNodes = false, bool threadSafe = false)
	{
		int[] parentWindow = neighbors[d];

		// Check that we actually have a center node
		if (parentWindow[CenterIndex] == FemTree.None)
		{
			return 0;
		}

		return NeighborsLoop(parentWindow, childNeighbors, cIdx, createNodes, threadSafe);
	}

	/// <summary>
	/// The window of <paramref name="node"/> with radii (<paramref name="leftRadius"/>,
	/// <paramref name="rightRadius"/>), which may differ from the key's own, written to
	/// <paramref name="window"/> ((L + R + 1)^3 entries). Taken from the key's cached parent
	/// window when that reaches far enough (half the radii, rounded up), else built recursively
	/// from a parent window of those half radii. Port of ConstNeighborKey's <c>getNeighbors(
	/// UIntPack&lt;L...&gt; , UIntPack&lt;R...&gt; , node , neighbors )</c>. It never creates
	/// nodes: NeighborKey's creating versions of these overloads do not compile upstream (their
	/// Window::Index use is ill-formed), so no PoissonRecon code path reaches them.
	/// </summary>
	public void GetNeighbors(int leftRadius, int rightRadius, int node, int[] window)
	{
		int width = leftRadius + rightRadius + 1;
		Array.Fill(window, FemTree.None, 0, width * width * width);
		if (node == FemTree.None)
		{
			return;
		}

		int parent = tree.Parent(node);
		int parentLeft = (leftRadius + 1) / 2;
		int parentRight = (rightRadius + 1) / 2;

		// If we are at the root of the tree, we are done
		if (parent == FemTree.None)
		{
			window[((leftRadius * width) + leftRadius) * width + leftRadius] = node;
		}

		// If we can get the data from the the key for the parent node, do that
		else if (parentLeft <= LeftRadius && parentRight <= RightRadius)
		{
			GetNeighbors(parent);
			NeighborsLoop(neighbors[tree.Depth(node) - 1], LeftRadius, RightRadius, window, leftRadius, rightRadius, tree.ChildIndexInParent(node), false, false);
		}

		// Otherwise recurse
		else
		{
			int parentWidth = parentLeft + parentRight + 1;
			var parentWindow = new int[parentWidth * parentWidth * parentWidth];
			GetNeighbors(parentLeft, parentRight, parent, parentWindow);
			NeighborsLoop(parentWindow, parentLeft, parentRight, window, leftRadius, rightRadius, tree.ChildIndexInParent(node), false, false);
		}
	}

	/// <summary>
	/// The window of <paramref name="node"/> with radii (L, R), built from the same-radii window
	/// of its parent, which is left in <paramref name="parentWindow"/>. Port of
	/// ConstNeighborKey's <c>getNeighbors( UIntPack&lt;L...&gt; , UIntPack&lt;R...&gt; , node , pNeighbors , neighbors )</c>.
	/// </summary>
	public void GetNeighbors(int leftRadius, int rightRadius, int node, int[] parentWindow, int[] window)
	{
		int parent = tree.Parent(node);
		if (parent == FemTree.None)
		{
			GetNeighbors(leftRadius, rightRadius, node, window);
			return;
		}

		GetNeighbors(leftRadius, rightRadius, parent, parentWindow);
		NeighborsLoop(parentWindow, leftRadius, rightRadius, window, leftRadius, rightRadius, tree.ChildIndexInParent(node), false, false);
	}

	private int NeighborsLoop(int[] parentWindow, int[] childWindow, int cIdx, bool createNodes, bool threadSafe) =>
		NeighborsLoop(parentWindow, LeftRadius, RightRadius, childWindow, LeftRadius, RightRadius, cIdx, createNodes, threadSafe);

	// Port of _NeighborsLoop/_Run: child window entry i (per axis, radii cLeft/cRight) sits in
	// parent entry ((i + c + 2*cLeft) >> 1) - cLeft + pLeft (radii pLeft/pRight), as child
	// (i + c + 2*cLeft) & 1 of it; axis 0 is the outermost loop, as in the C++ recursion.
	private int NeighborsLoop(int[] parentWindow, int pLeft, int pRight, int[] childWindow, int cLeft, int cRight, int cIdx, bool createNodes, bool threadSafe)
	{
		int pWidth = pLeft + pRight + 1;
		int cWidth = cLeft + cRight + 1;
		int c0 = cIdx & 1;
		int c1 = (cIdx >> 1) & 1;
		int c2 = (cIdx >> 2) & 1;
		int count = 0;
		for (int i0 = -cLeft; i0 <= cRight; i0++)
		{
			int s0 = i0 + c0 + (cLeft << 1);
			int p0 = (s0 >> 1) - cLeft + pLeft;
			int ci0 = i0 + cLeft;
			for (int i1 = -cLeft; i1 <= cRight; i1++)
			{
				int s1 = i1 + c1 + (cLeft << 1);
				int p1 = (s1 >> 1) - cLeft + pLeft;
				int ci1 = i1 + cLeft;
				for (int i2 = -cLeft; i2 <= cRight; i2++)
				{
					int s2 = i2 + c2 + (cLeft << 1);
					int p2 = (s2 >> 1) - cLeft + pLeft;
					int ci2 = i2 + cLeft;
					int corner = (s0 & 1) | ((s1 & 1) << 1) | ((s2 & 1) << 2);
					int pNode = parentWindow[((p0 * pWidth) + p1) * pWidth + p2];
					int slot = ((ci0 * cWidth) + ci1) * cWidth + ci2;
					if (pNode == FemTree.None)
					{
						childWindow[slot] = FemTree.None;
						continue;
					}

					if (createNodes && !tree.HasChildren(pNode))
					{
						tree.InitChildren(pNode, threadSafe);
					}

					if (tree.HasChildren(pNode))
					{
						childWindow[slot] = tree.FirstChild(pNode) + corner;
						count++;
					}
					else
					{
						childWindow[slot] = FemTree.None;
					}
				}
			}
		}

		return count;
	}
}
