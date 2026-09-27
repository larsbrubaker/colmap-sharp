// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// HyperCubeTables: HyperCube's answers cached per cube dimension and element dimension, as the
// level-set extractor reads them in its per-node loops. Ports FEMTree.LevelSet.inl's
// LevelSetExtraction::HyperCubeTables< D , K1 , K2 > (Overlap, OverlapElements) and
// HyperCubeTables< D , K > (CellOffset, IncidentElementCoIndex, IncidentElementIndex,
// CellOffsetAntipodal, IncidentCube, Directions), which SetHyperCubeTables< 3 > and < 2 > fill
// for every D in 1..3. Upstream fills static arrays at the start of each extraction; here they
// are built once, on first use. Tier A against oracle/poisson_hypercube_harness.cc.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// The per-element tables of the K-elements of a D-cube. Port of
/// <c>LevelSetExtraction::HyperCubeTables&lt; D , K &gt;</c>.
/// </summary>
public sealed class HyperCubeElementTable
{
	internal HyperCubeElementTable(int d, int k)
	{
		ElementNum = HyperCube.ElementNum(d, k);
		IncidentCubeNum = HyperCube.IncidentCubeNum(d, k);
		CellOffset = new int[ElementNum][];
		IncidentElementCoIndex = new int[ElementNum][];
		IncidentElementIndex = new int[ElementNum][];
		CellOffsetAntipodal = new int[ElementNum];
		IncidentCube = new int[ElementNum];
		Directions = new HyperCubeDirection[ElementNum][];
		for (int e = 0; e < ElementNum; e++)
		{
			CellOffset[e] = new int[IncidentCubeNum];
			IncidentElementCoIndex[e] = new int[IncidentCubeNum];
			IncidentElementIndex[e] = new int[IncidentCubeNum];
			for (int i = 0; i < IncidentCubeNum; i++)
			{
				CellOffset[e][i] = HyperCube.CellOffset(d, k, e, i);
				int incident = HyperCube.IncidentElement(d, k, e, i);
				HyperCube.Factor(d, k, incident, out _, out int coIndex);
				IncidentElementCoIndex[e][i] = coIndex;
				IncidentElementIndex[e][i] = incident;
			}

			IncidentCube[e] = HyperCube.IncidentCube(d, k, e);
			CellOffsetAntipodal[e] = HyperCube.CellOffset(d, k, e, HyperCube.Antipodal(d - k, 0, IncidentCube[e]));
			Directions[e] = new HyperCubeDirection[d];
			HyperCube.Directions(d, k, e, Directions[e]);
		}
	}

	/// <summary>The number of K-elements.</summary>
	public int ElementNum { get; }

	/// <summary>The number of cubes incident on each K-element.</summary>
	public int IncidentCubeNum { get; }

	/// <summary>[element][incident cube]: the incident cube's index in the 3^D neighbor window.</summary>
	public int[][] CellOffset { get; }

	/// <summary>[element][incident cube]: the co-index of the element within that cube.</summary>
	public int[][] IncidentElementCoIndex { get; }

	/// <summary>[element][incident cube]: the index of the element within that cube.</summary>
	public int[][] IncidentElementIndex { get; }

	/// <summary>[element]: the window index of the incident cube opposite the cube itself.</summary>
	public int[] CellOffsetAntipodal { get; }

	/// <summary>[element]: which incident cube the cube itself is.</summary>
	public int[] IncidentCube { get; }

	/// <summary>[element][axis]: the element's direction along each axis.</summary>
	public HyperCubeDirection[][] Directions { get; }
}

/// <summary>
/// The overlaps of the K1-elements of a D-cube with its K2-elements. Port of
/// <c>LevelSetExtraction::HyperCubeTables&lt; D , K1 , K2 &gt;</c>.
/// </summary>
public sealed class HyperCubeOverlapTable
{
	internal HyperCubeOverlapTable(int d, int k1, int k2)
	{
		int elementNum1 = HyperCube.ElementNum(d, k1), elementNum2 = HyperCube.ElementNum(d, k2);
		OverlapElementNum = HyperCube.OverlapElementNum(d, k1, k2);
		Overlap = new bool[elementNum1][];
		OverlapElements = new int[elementNum1][];
		for (int e = 0; e < elementNum1; e++)
		{
			Overlap[e] = new bool[elementNum2];
			for (int e2 = 0; e2 < elementNum2; e2++)
			{
				Overlap[e][e2] = HyperCube.Overlap(d, k1, e, k2, e2);
			}

			OverlapElements[e] = new int[OverlapElementNum];
			HyperCube.OverlapElements(d, k1, e, k2, OverlapElements[e]);
		}
	}

	/// <summary>The number of K2-elements overlapping each K1-element.</summary>
	public int OverlapElementNum { get; }

	/// <summary>[K1-element][K2-element]: whether one contains the other.</summary>
	public bool[][] Overlap { get; }

	/// <summary>[K1-element][i]: the K2-elements overlapping it, in upstream's order.</summary>
	public int[][] OverlapElements { get; }
}

/// <summary>
/// The cached tables of the 1-, 2- and 3-cubes. Port of what
/// <c>LevelSetExtraction::SetHyperCubeTables&lt; 3 &gt;</c> and <c>&lt; 2 &gt;</c> fill.
/// </summary>
public static class HyperCubeTables
{
	private const int MaxDim = 3;

	// [d][k] and [d][k1][k2], d in 1..MaxDim.
	private static readonly HyperCubeElementTable[][] Elements = BuildElements();
	private static readonly HyperCubeOverlapTable[][][] Overlaps = BuildOverlaps();

	/// <summary>The tables of the K-elements of the D-cube.</summary>
	public static HyperCubeElementTable Of(int d, int k) => Elements[d][k];

	/// <summary>The overlap tables of the K1- and K2-elements of the D-cube.</summary>
	public static HyperCubeOverlapTable Of(int d, int k1, int k2) => Overlaps[d][k1][k2];

	private static HyperCubeElementTable[][] BuildElements()
	{
		var tables = new HyperCubeElementTable[MaxDim + 1][];
		tables[0] = [];
		for (int d = 1; d <= MaxDim; d++)
		{
			tables[d] = new HyperCubeElementTable[d + 1];
			for (int k = 0; k <= d; k++)
			{
				tables[d][k] = new HyperCubeElementTable(d, k);
			}
		}

		return tables;
	}

	private static HyperCubeOverlapTable[][][] BuildOverlaps()
	{
		var tables = new HyperCubeOverlapTable[MaxDim + 1][][];
		tables[0] = [];
		for (int d = 1; d <= MaxDim; d++)
		{
			tables[d] = new HyperCubeOverlapTable[d + 1][];
			for (int k1 = 0; k1 <= d; k1++)
			{
				tables[d][k1] = new HyperCubeOverlapTable[d + 1];
				for (int k2 = 0; k2 <= d; k2++)
				{
					tables[d][k1][k2] = new HyperCubeOverlapTable(d, k1, k2);
				}
			}
		}

		return tables;
	}
}
