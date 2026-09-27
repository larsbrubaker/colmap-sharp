// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// LevelSetKeys: the keys the level-set extractor names corners, edges and faces by, independent
// of which node sees them, so iso-vertices on a shared edge merge. Ports FEMTree.LevelSet.inl's
// LevelSetExtraction::Key< 3 > and KeyGenerator< 3 >: per axis, a corner of the finest (max key
// depth) grid is its offset times 4 (divisible by 4) and an edge is the midpoint of its corners
// plus one (odd). Tier A against oracle/poisson_levelset2_harness.cc.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>A per-axis corner/edge key of a cell. Port of <c>LevelSetExtraction::Key&lt; 3 &gt;</c>.</summary>
public readonly record struct LevelSetKey(uint X, uint Y, uint Z)
{
	/// <summary>The unset key (every index -1 as unsigned). Port of <c>Key()</c>.</summary>
	public static readonly LevelSetKey Unset = new(uint.MaxValue, uint.MaxValue, uint.MaxValue);

	/// <summary>The index along one axis.</summary>
	public uint this[int axis] => axis == 0 ? X : axis == 1 ? Y : Z;
}

/// <summary>
/// Builds <see cref="LevelSetKey"/>s at a fixed max key depth. Port of
/// <c>LevelSetExtraction::KeyGenerator&lt; 3 &gt;</c>.
/// </summary>
public sealed class LevelSetKeyGenerator(int maxDepth)
{
	/// <summary>The depth keys are expressed at.</summary>
	public int MaxDepth { get; } = maxDepth;

	/// <summary>The key of a K-element of the cell at <paramref name="depth"/> and offset. Port of <c>operator()( depth , offset , e )</c>.</summary>
	public LevelSetKey Key(int k, int depth, int x, int y, int z, int element)
	{
		HyperCubeDirection[] dirs = HyperCubeTables.Of(3, k).Directions[element];
		return new LevelSetKey(Index(depth, x, dirs[0]), Index(depth, y, dirs[1]), Index(depth, z, dirs[2]));
	}

	/// <summary>A corner's index: divisible by 4. Port of <c>cornerIndex</c>.</summary>
	public uint CornerIndex(int depth, int offset) => unchecked((uint)offset << (MaxDepth + 2 - depth));

	/// <summary>An edge's index: odd. Port of <c>edgeIndex</c>.</summary>
	public uint EdgeIndex(int depth, int offset) => unchecked(((CornerIndex(depth, offset) + CornerIndex(depth, offset + 1)) / 2) + 1);

	/// <summary>The index along one axis in direction <paramref name="dir"/>. Port of <c>index</c>.</summary>
	public uint Index(int depth, int offset, HyperCubeDirection dir) =>
		dir == HyperCubeDirection.Cross ? EdgeIndex(depth, offset) : CornerIndex(depth, offset + (dir == HyperCubeDirection.Back ? 0 : 1));
}
