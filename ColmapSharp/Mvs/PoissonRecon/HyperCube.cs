// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// HyperCube: the element algebra of a D-dimensional cube that the level-set extractor indexes
// corners, edges, faces and cells with. Ports MarchingCubes.h's HyperCube::Cube< D > (its
// Element< K >, factor, directions, antipodal, Overlap, OverlapElements, IncidentCube,
// IncidentElement, CellOffset, IsOriented, MCIndex, ElementMCIndex, HasMCRoots). HyperCubeTables
// caches these per (D, K) as FEMTree.LevelSet.inl does. Tier A against
// oracle/poisson_hypercube_harness.cc.
//
// Translation notes: upstream resolves D and K at compile time through recursive templates;
// here they are runtime arguments and the same recursions run once, when HyperCubeTables is
// built. An element is its index; an incident-cube index of a K-element is an Element< 0 > of
// the (D-K)-cube, as IncidentCubeIndex< K > is upstream. Element indices sort the K-elements of
// the back face first, then those spanning axis D-1, then those of the front face.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>Where an element lies along one axis of a cube. Port of <c>HyperCube::Direction</c>.</summary>
public enum HyperCubeDirection
{
	/// <summary>Along the axis's back (lower) side.</summary>
	Back = 0,
	/// <summary>Spanning the axis.</summary>
	Cross = 1,
	/// <summary>Along the axis's front (upper) side.</summary>
	Front = 2,
}

/// <summary>
/// The element algebra of a D-dimensional cube. Port of PoissonRecon's
/// <c>HyperCube::Cube&lt; D &gt;</c>.
/// </summary>
public static class HyperCube
{
	/// <summary>The opposite direction (Cross stays Cross). Port of <c>HyperCube::Opposite</c>.</summary>
	public static HyperCubeDirection Opposite(HyperCubeDirection dir) => dir == HyperCubeDirection.Back ? HyperCubeDirection.Front : dir == HyperCubeDirection.Front ? HyperCubeDirection.Back : HyperCubeDirection.Cross;

	/// <summary>The number of K-dimensional elements of a D-cube. Port of <c>HyperCube::ElementNum</c>.</summary>
	public static int ElementNum(int d, int k)
	{
		if (k > d || k < 0)
		{
			return 0;
		}

		if (d == k)
		{
			return 1;
		}

		return k == 0 ? 2 * ElementNum(d - 1, 0) : (2 * ElementNum(d - 1, k)) + ElementNum(d - 1, k - 1);
	}

	/// <summary>
	/// The number of K2-elements overlapping a K1-element (contained in it when K1 &gt;= K2,
	/// containing it otherwise). Port of <c>HyperCube::OverlapElementNum</c>.
	/// </summary>
	public static int OverlapElementNum(int d, int k1, int k2)
	{
		if (k1 >= k2)
		{
			return ElementNum(k1, k2);
		}

		return k2 == d ? 1 : OverlapElementNum(d - 1, k1, k2) + OverlapElementNum(d - 1, k1, k2 - 1);
	}

	/// <summary>The number of cubes incident on a K-element. Port of <c>Cube::IncidentCubeNum</c>.</summary>
	public static int IncidentCubeNum(int d, int k) => ElementNum(d - k, 0);

	/// <summary>The element with the given direction along axis D-1 and index in the (D-1)-cube. Port of <c>Element( dir , coIndex )</c>.</summary>
	public static int Element(int d, int k, HyperCubeDirection dir, int coIndex)
	{
		if (d == 0)
		{
			return coIndex;
		}

		return dir switch
		{
			HyperCubeDirection.Back => coIndex,
			HyperCubeDirection.Cross when k != 0 => coIndex + ElementNum(d - 1, k),
			HyperCubeDirection.Front => coIndex + ElementNum(d - 1, k) + (k != 0 ? ElementNum(d - 1, k - 1) : 0),
			_ => throw new ArgumentException($"Bad direction: {dir}"),
		};
	}

	/// <summary>The direction along axis D-1 and the index in the (D-1)-cube. Port of <c>Element::factor</c>.</summary>
	public static void Factor(int d, int k, int index, out HyperCubeDirection dir, out int coIndex)
	{
		if (d == k)
		{
			dir = HyperCubeDirection.Cross;
			coIndex = 0;
			return;
		}

		int back = ElementNum(d - 1, k);
		if (index < back)
		{
			dir = HyperCubeDirection.Back;
			coIndex = index;
		}
		else if (k != 0 && index < back + ElementNum(d - 1, k - 1))
		{
			dir = HyperCubeDirection.Cross;
			coIndex = index - back;
		}
		else
		{
			dir = HyperCubeDirection.Front;
			coIndex = index - back - (k != 0 ? ElementNum(d - 1, k - 1) : 0);
		}
	}

	/// <summary>The direction of the element along each axis. Port of <c>Element::directions</c>.</summary>
	public static void Directions(int d, int k, int index, Span<HyperCubeDirection> dirs)
	{
		if (d == k)
		{
			for (int i = 0; i < d; i++)
			{
				dirs[i] = HyperCubeDirection.Cross;
			}

			return;
		}

		Factor(d, k, index, out HyperCubeDirection dir, out int coIndex);
		dirs[d - 1] = dir;
		Directions(d - 1, dir == HyperCubeDirection.Cross ? k - 1 : k, coIndex, dirs);
	}

	/// <summary>The element opposite through the cube's center. Port of <c>Element::antipodal</c>.</summary>
	public static int Antipodal(int d, int k, int index)
	{
		if (d == k)
		{
			return index;
		}

		Factor(d, k, index, out HyperCubeDirection dir, out int coIndex);
		return dir == HyperCubeDirection.Cross ? Element(d, k, dir, Antipodal(d - 1, k - 1, coIndex)) : Element(d, k, Opposite(dir), Antipodal(d - 1, k, coIndex));
	}

	/// <summary>Whether one element contains the other. Port of <c>Cube::Overlap</c>.</summary>
	public static bool Overlap(int d, int k1, int e1, int k2, int e2)
	{
		if (k1 < k2)
		{
			return Overlap(d, k2, e2, k1, e1);
		}

		Span<HyperCubeDirection> dir1 = stackalloc HyperCubeDirection[d];
		Span<HyperCubeDirection> dir2 = stackalloc HyperCubeDirection[d];
		Directions(d, k1, e1, dir1);
		Directions(d, k2, e2, dir2);
		for (int i = 0; i < d; i++)
		{
			if (dir1[i] != HyperCubeDirection.Cross && dir1[i] != dir2[i])
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>
	/// The K-element of the D-cube that is K-element <paramref name="subElement"/> of the
	/// KD-element <paramref name="subCube"/>. Port of <c>Element( subCube , subElement )</c>.
	/// </summary>
	public static int FromSubCube(int d, int kd, int subCube, int k, int subElement)
	{
		if (kd == k)
		{
			return subCube;
		}

		if (d == kd)
		{
			return subElement;
		}

		Factor(d, kd, subCube, out HyperCubeDirection dir, out int coIndex);
		if (dir != HyperCubeDirection.Cross)
		{
			return Element(d, k, dir, FromSubCube(d - 1, kd, coIndex, k, subElement));
		}

		Factor(kd, k, subElement, out HyperCubeDirection subDir, out int subCoIndex);
		if (subDir != HyperCubeDirection.Cross)
		{
			return Element(d, k, subDir, FromSubCube(d - 1, kd - 1, coIndex, k, subCoIndex));
		}

		// A vertex never spans an axis, so K = 0 does not reach this (upstream leaves it unset).
		return Element(d, k, subDir, FromSubCube(d - 1, kd - 1, coIndex, k - 1, subCoIndex));
	}

	/// <summary>
	/// The K2-elements overlapping K1-element <paramref name="e"/>, in upstream's order. Port of
	/// <c>Cube::OverlapElements</c>.
	/// </summary>
	public static void OverlapElements(int d, int k1, int e, int k2, Span<int> es)
	{
		if (k1 >= k2)
		{
			int count = ElementNum(k1, k2);
			for (int i = 0; i < count; i++)
			{
				es[i] = FromSubCube(d, k1, e, k2, i);
			}

			return;
		}

		if (d == k2)
		{
			es[0] = 0;
			return;
		}

		Factor(d, k1, e, out HyperCubeDirection dir, out int coIndex);
		if (dir != HyperCubeDirection.Cross)
		{
			int n1 = OverlapElementNum(d - 1, k1, k2), n2 = OverlapElementNum(d - 1, k1, k2 - 1);
			Span<int> sub = stackalloc int[Math.Max(n1, n2)];
			OverlapElements(d - 1, k1, coIndex, k2, sub);
			for (int i = 0; i < n1; i++)
			{
				es[i] = Element(d, k2, dir, sub[i]);
			}

			OverlapElements(d - 1, k1, coIndex, k2 - 1, sub);
			for (int i = 0; i < n2; i++)
			{
				es[n1 + i] = Element(d, k2, HyperCubeDirection.Cross, sub[i]);
			}
		}
		else
		{
			int n = OverlapElementNum(d - 1, k1 - 1, k2 - 1);
			Span<int> sub = stackalloc int[n];
			OverlapElements(d - 1, k1 - 1, coIndex, k2 - 1, sub);
			for (int i = 0; i < n; i++)
			{
				es[i] = Element(d, k2, HyperCubeDirection.Cross, sub[i]);
			}
		}
	}

	/// <summary>
	/// The index (an Element&lt; 0 &gt; of the (D-K)-cube) of the incident cube that has
	/// <paramref name="e"/> as its own element. Port of <c>Cube::IncidentCube</c>.
	/// </summary>
	public static int IncidentCube(int d, int k, int e)
	{
		if (d == k)
		{
			return 0;
		}

		Factor(d, k, e, out HyperCubeDirection dir, out int coIndex);
		if (dir == HyperCubeDirection.Cross)
		{
			return IncidentCube(d - 1, k - 1, coIndex);
		}

		return Element(d - k, 0, Opposite(dir), IncidentCube(d - 1, k, coIndex));
	}

	/// <summary>
	/// The element of incident cube <paramref name="ic"/> that is <paramref name="e"/>. Port of
	/// <c>Cube::IncidentElement</c>.
	/// </summary>
	public static int IncidentElement(int d, int k, int e, int ic)
	{
		if (d == k)
		{
			return e;
		}

		Factor(d, k, e, out HyperCubeDirection eDir, out int eCoIndex);
		Factor(d - k, 0, ic, out HyperCubeDirection dDir, out int dCoIndex);
		if (eDir == HyperCubeDirection.Cross)
		{
			return Element(d, k, eDir, IncidentElement(d - 1, k - 1, eCoIndex, ic));
		}

		return Element(d, k, eDir == dDir ? Opposite(eDir) : eDir, IncidentElement(d - 1, k, eCoIndex, dCoIndex));
	}

	/// <summary>
	/// The offset, each coordinate in {-1, 0, 1}, of incident cube <paramref name="ic"/> of
	/// <paramref name="e"/> from the cube. Port of <c>Cube::CellOffset( e , d , x )</c>.
	/// </summary>
	public static void CellOffset(int d, int k, int e, int ic, Span<int> x)
	{
		if (d == k)
		{
			for (int i = 0; i < d; i++)
			{
				x[i] = 0;
			}

			return;
		}

		Factor(d, k, e, out HyperCubeDirection eDir, out int eCoIndex);
		Factor(d - k, 0, ic, out HyperCubeDirection dDir, out int dCoIndex);
		int front = dDir == HyperCubeDirection.Back ? 0 : 1;
		if (eDir == HyperCubeDirection.Cross)
		{
			x[d - 1] = 0;
			CellOffset(d - 1, k - 1, eCoIndex, ic, x);
		}
		else
		{
			x[d - 1] = (eDir == HyperCubeDirection.Back ? -1 : 0) + front;
			CellOffset(d - 1, k, eCoIndex, dCoIndex, x);
		}
	}

	/// <summary>
	/// The same offset linearized into the 3^D neighbor window, the last axis fastest (NeighborKey's order). Port of
	/// <c>Cube::CellOffset( e , d )</c>.
	/// </summary>
	public static int CellOffset(int d, int k, int e, int ic)
	{
		if (d == k)
		{
			// WindowIndex< 3^D , 1^D >::Index: the center of the window.
			int index = 0;
			for (int i = 0; i < d; i++)
			{
				index = (index * 3) + 1;
			}

			return index;
		}

		Factor(d, k, e, out HyperCubeDirection eDir, out int eCoIndex);
		Factor(d - k, 0, ic, out HyperCubeDirection dDir, out int dCoIndex);
		int front = dDir == HyperCubeDirection.Back ? 0 : 1;
		return eDir switch
		{
			HyperCubeDirection.Cross => 1 + (CellOffset(d - 1, k - 1, eCoIndex, ic) * 3),
			HyperCubeDirection.Back => front + (CellOffset(d - 1, k, eCoIndex, dCoIndex) * 3),
			_ => 1 + front + (CellOffset(d - 1, k, eCoIndex, dCoIndex) * 3),
		};
	}

	/// <summary>Whether a face (a (D-1)-element) faces outward. Port of <c>Cube::IsOriented</c>.</summary>
	public static bool IsOriented(int d, int e)
	{
		int dim = d;
		int k = d - 1;
		HyperCubeDirection dir;
		while (true)
		{
			Factor(dim, k, e, out dir, out int coIndex);
			if (dim == 1 || dir != HyperCubeDirection.Cross)
			{
				dim--;
				break;
			}

			e = coIndex;
			dim--;
			k--;
		}

		return (dir == HyperCubeDirection.Front) ^ (((d - dim - 1) & 1) != 0);
	}

	/// <summary>
	/// The marching-cubes index of the 2^D corner values: bit c set when corner c is below
	/// <paramref name="iso"/>. Port of <c>Cube::MCIndex</c>.
	/// </summary>
	public static int McIndex(ReadOnlySpan<float> values, float iso)
	{
		int mcIdx = 0;
		for (int c = 0; c < values.Length; c++)
		{
			if (values[c] < iso)
			{
				mcIdx |= 1 << c;
			}
		}

		return mcIdx;
	}

	/// <summary>The sub-index of <paramref name="mcIndex"/> over the corners of a K-element. Port of <c>Cube::ElementMCIndex</c>.</summary>
	public static int ElementMcIndex(int d, int k, int element, int mcIndex)
	{
		if (d == k)
		{
			return mcIndex;
		}

		int shift = (1 << d) / 2;
		int mask = (1 << shift) - 1;
		int mcIndex0 = mcIndex & mask, mcIndex1 = (mcIndex >> shift) & mask;
		Factor(d, k, element, out HyperCubeDirection dir, out int coIndex);
		if (dir == HyperCubeDirection.Cross)
		{
			int subShift = (1 << k) / 2;
			return ElementMcIndex(d - 1, k - 1, coIndex, mcIndex0) | (ElementMcIndex(d - 1, k - 1, coIndex, mcIndex1) << subShift);
		}

		return ElementMcIndex(d - 1, k, coIndex, dir == HyperCubeDirection.Back ? mcIndex0 : mcIndex1);
	}

	/// <summary>Whether the iso-surface crosses the D-cube. Port of <c>Cube::HasMCRoots</c>.</summary>
	public static bool HasMcRoots(int d, int mcIndex)
	{
		int mask = (1 << (1 << d)) - 1;
		return mcIndex != 0 && (mcIndex & mask) != mask;
	}
}
