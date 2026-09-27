// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// LevelSetSliceValues: what the level-set extractor keeps per slice (a plane z = slice at one
// depth): its cell indices (LevelSetCellIndices), the implicit function's value and gradient at
// each distinct corner, each leaf's marching-squares index on the slice, and the scratch flags
// saying which corners are already set. Ports the corner part of FEMTree.LevelSet.3D.inl's
// _LevelSetExtractor< ... , 3 , ... >::SliceValues (reset, cornerValues, cornerGradients,
// mcIndices) and SliceValues::Scratch (reset, cSet), and XSliceValues' slab cell indices and
// slab number. LevelSetSlabValues pairs them as SlabValues does (two of each, by parity). Filled
// by PoissonLevelSetExtractor. Like upstream, the arrays grow but never shrink or clear on reuse:
// a corner is read only once its cSet flag is set.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Per-slice corner data of the level-set extractor. Port of the corner part of PoissonRecon's
/// <c>SliceValues</c> and <c>SliceValues::Scratch</c>.
/// </summary>
public sealed class LevelSetSliceValues
{
	/// <summary>The slice's cell indices. Port of <c>cellIndices</c>.</summary>
	public LevelSetCellIndices CellIndices { get; } = new();

	/// <summary>The slice number (-1 before the first reset). Port of <c>slice()</c>.</summary>
	public int Slice { get; private set; } = -1;

	/// <summary>The value at each corner, by corner index. Port of <c>cornerValues</c>.</summary>
	public float[] CornerValues { get; private set; } = [];

	/// <summary>The gradient at each corner, three floats per corner index, or null. Port of <c>cornerGradients</c>.</summary>
	public float[]? CornerGradients { get; private set; }

	/// <summary>Each node's marching-squares index, by sorted index minus the node offset. Port of <c>mcIndices</c>.</summary>
	public byte[] McIndices { get; private set; } = [];

	/// <summary>Whether each corner has been set. Port of <c>Scratch::cSet</c>.</summary>
	public byte[] CornerSet { get; private set; } = [];

	/// <summary>
	/// Starts slice <paramref name="slice"/> after <see cref="CellIndices"/> is set. Port of
	/// <c>SliceValues::reset( slice , computeGradients )</c> followed by
	/// <c>Scratch::reset( cellIndices )</c>'s corner flags.
	/// </summary>
	public void Reset(int slice, bool computeGradients)
	{
		Slice = slice;
		if (McIndices.Length < CellIndices.Size)
		{
			McIndices = new byte[CellIndices.Size];
		}

		int corners = CellIndices.Count(0);
		if (CornerValues.Length < corners)
		{
			CornerValues = new float[corners];
			CornerGradients = computeGradients ? new float[3 * corners] : null;
		}

		// Scratch::reset: a fresh, zeroed flag per corner.
		CornerSet = new byte[corners];
	}
}

/// <summary>
/// The extractor's per-depth pair of slices and slabs, addressed by parity. Port of
/// <c>SlabValues</c> (its slice values, and its x-slice values' cell indices and slab number).
/// </summary>
public sealed class LevelSetSlabValues
{
	private readonly LevelSetSliceValues[] slices = [new(), new()];
	private readonly LevelSetCellIndices[] slabCellIndices = [new(), new()];
	private readonly int[] slabs = [-1, -1];

	/// <summary>The values of slice <paramref name="slice"/> (by parity). Port of <c>sliceValues( idx )</c>.</summary>
	public LevelSetSliceValues SliceValues(int slice) => slices[slice & 1];

	/// <summary>The cell indices of slab <paramref name="slab"/> (by parity). Port of <c>xSliceValues( idx ).cellIndices</c>.</summary>
	public LevelSetCellIndices SlabCellIndices(int slab) => slabCellIndices[slab & 1];

	/// <summary>The slab number last reset at this parity. Port of <c>xSliceValues( idx ).slab()</c>.</summary>
	public int Slab(int slab) => slabs[slab & 1];

	/// <summary>Records the slab number. Port of <c>XSliceValues::reset( slab )</c>'s slab.</summary>
	public void ResetSlab(int slab) => slabs[slab & 1] = slab;
}
