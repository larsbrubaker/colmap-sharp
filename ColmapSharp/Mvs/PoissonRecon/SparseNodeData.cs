// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// SparseNodeData: SparseNodeData<Data,...> from thirdparty/PoissonRecon/FEMTree.h for Data a
// fixed number of floats (a density, a normal, a color...): a grow-only map from node index
// (FEMTreeNodeData::nodeIndex) to a slot in a packed array, slots handed out in the order
// nodes are first touched (at()). Later stages iterate the packed array by slot (e.g. Solve
// negates the normal field slot by slot), so the slot order is part of the port's contract
// and matches the C++ (PoissonTreeOracleTests).
//
// Translation notes: the C++ stores Data values in a NestedVector; here Width floats per slot
// in one float[]. NestedVector::resize only grows, so Reserve never truncates. The C++ at()
// is thread-safe under a mutex; the port's callers are sequential.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Per-node float vectors for a subset of nodes. Port of PoissonRecon's <c>SparseNodeData</c>.
/// </summary>
public class SparseNodeData
{
	private int[] indices = [];
	private float[] data;

	/// <summary>Empty data with <paramref name="width"/> floats per node.</summary>
	public SparseNodeData(int width)
	{
		if (width < 1)
		{
			throw new ArgumentOutOfRangeException(nameof(width), "Each entry needs at least one float.");
		}

		Width = width;
		data = new float[16 * width];
	}

	/// <summary>The number of floats per entry.</summary>
	public int Width { get; }

	/// <summary>The number of entries. Port of <c>size()</c>.</summary>
	public int Count { get; private set; }

	/// <summary>The packed values: entry i occupies [i * Width, (i + 1) * Width).</summary>
	public Span<float> Values => data.AsSpan(0, Count * Width);

	/// <summary>Grows the index map to cover node indices below <paramref name="size"/>. Port of <c>reserve</c>.</summary>
	public void Reserve(int size)
	{
		if (size > indices.Length)
		{
			int old = indices.Length;
			Array.Resize(ref indices, size);
			Array.Fill(indices, -1, old, size - old);
		}
	}

	/// <summary>The entry of a node index, or -1. Port of <c>index( idx )</c>.</summary>
	public int Index(int nodeIndex) => nodeIndex < 0 || nodeIndex >= indices.Length ? -1 : indices[nodeIndex];

	/// <summary>
	/// The entry of a node index, appending a zero entry if it has none. Port of
	/// <c>at( node , zero )</c> with a zero default.
	/// </summary>
	public int At(int nodeIndex)
	{
		if (nodeIndex + 1 > indices.Length)
		{
			// NestedVector grows geometrically in blocks; the growth policy is invisible to callers.
			Reserve(Math.Max(nodeIndex + 1, 2 * indices.Length));
		}

		int slot = indices[nodeIndex];
		if (slot == -1)
		{
			slot = Count;
			if ((slot + 1) * Width > data.Length)
			{
				Array.Resize(ref data, Math.Max((slot + 1) * Width, 2 * data.Length));
			}

			Array.Clear(data, slot * Width, Width);
			indices[nodeIndex] = slot;
			Count = slot + 1;
		}

		return slot;
	}

	/// <summary>The k-th float of an entry.</summary>
	public ref float Value(int slot, int k = 0) => ref data[slot * Width + k];
}

/// <summary>
/// The kernel density estimate: per-node splatted sample weights with the parameters that
/// interpret them. Port of <c>FEMTree::DensityEstimator&lt;DensityDegree&gt;</c>.
/// </summary>
public sealed class DensityEstimator : SparseNodeData
{
	/// <summary>Port of <c>DensityEstimator( kernelDepth , coDimension , samplesPerNode )</c>.</summary>
	public DensityEstimator(int degree, int kernelDepth, int coDimension, float samplesPerNode)
		: base(1)
	{
		Degree = FemSignature.CheckDegree(degree);
		KernelDepth = kernelDepth;
		CoDimension = coDimension;
		SamplesPerNode = samplesPerNode;
	}

	/// <summary>The B-spline degree of the splatting kernel (DensityDegree).</summary>
	public int Degree { get; }

	/// <summary>The (local) depth the density is splatted to.</summary>
	public int KernelDepth { get; }

	/// <summary>The co-dimension of the sampled surface (1 for a surface in 3D).</summary>
	public int CoDimension { get; }

	/// <summary>The target number of samples per node.</summary>
	public float SamplesPerNode { get; }
}
