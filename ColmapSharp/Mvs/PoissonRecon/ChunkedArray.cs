// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ChunkedArray: a growable array stored in fixed-size blocks, for the PoissonRecon port's
// per-node storage (FemTree). A reconstruction at COLMAP's default depth makes tens of
// millions of nodes; growing one contiguous array by doubling would need a transient copy of
// hundreds of megabytes, which browser-wasm's linear memory handles badly. Blocks are
// allocated as the array grows and never move, so growth costs only the new block. This
// plays the role of PoissonRecon's block Allocator / NestedVector; it is written here and
// carries no upstream code.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>A growable array of <typeparamref name="T"/> in 65536-element blocks.</summary>
internal sealed class ChunkedArray<T>
{
	private const int LogBlockSize = 16;
	private const int BlockSize = 1 << LogBlockSize;
	private const int Mask = BlockSize - 1;

	private T[][] blocks = new T[4][];
	private int blockCount;

	/// <summary>The number of elements that can be addressed without growing.</summary>
	public long Capacity => (long)blockCount << LogBlockSize;

	/// <summary>Element i (i must be below <see cref="Capacity"/>).</summary>
	public ref T this[int i] => ref blocks[i >> LogBlockSize][i & Mask];

	/// <summary>Makes elements below <paramref name="size"/> addressable; new elements are default.</summary>
	public void EnsureCapacity(long size)
	{
		while (Capacity < size)
		{
			if (blockCount == blocks.Length)
			{
				Array.Resize(ref blocks, blocks.Length * 2);
			}

			blocks[blockCount++] = new T[BlockSize];
		}
	}
}
