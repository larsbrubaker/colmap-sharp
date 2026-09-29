// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonSplat: the sample splatting of thirdparty/PoissonRecon - FEMTree::
// setInterpolatedDataField (FEMTree.inl) with _getSamplesPerNode, _getSampleDepthAndWeight and
// both forms of _splatPointData (FEMTree.WeightedSamples.inl), which turn the oriented samples
// into the normal field the Poisson system's right-hand side is built from; and
// updateExtrapolatedDataField with _multiSplatPointData, which spreads the samples' colors
// over every ancestor of their nodes. Solve's post-processing of both fields (negating the
// normals, scaling the colors by 32^depth) is here too. Runs after PoissonDensity; the FEM
// system (next slice) reads the normal field. Tier A against oracle/poisson_tree_harness.cc
// (the "density*" runs), in the order below.
//
// Translation notes:
// - The C++ splats in a ThreadPool::ParallelFor over samples; COLMAP runs it on all hardware
//   threads, so the float accumulation order there is not deterministic. The port splats
//   sequentially in sample order, which is what a single-threaded run does
//   (divergence 106).
// - The density form of _splatPointData calls its Splat lambda for the node and, with
//   weight 1 - dx, for the node's parent, but the lambda ignores its node argument and splats
//   into `temp` both times (and scales by temp's width). The port reproduces that.
// - Solve's kernel density degree (2) equals the normals' degree, so the C++ uses one key for
//   both the density lookups and the splats (oneKey); the port shares one NeighborKey too,
//   since its cache state is observable through node creation order.
// - log( float ) is the float overload in the C++; the port rounds the double logarithm to
//   float (divergence 116).

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Splatting of the samples' normals and auxiliary data into the octree. Port of FEMTree's
/// data-field setters used by <c>Poisson::Solver::Solve</c>.
/// </summary>
public static class PoissonSplat
{
	private const int NormalDegree = 2;

	/// <summary>
	/// The normal field: for each valid sample, its unit average normal times its weight,
	/// splatted at the depth the density estimate assigns it, then negated (Solve). Also returns
	/// the weighted sums of sample depth and weight (Solve's pointDepthAndWeight: data, weight).
	/// Port of <c>setInterpolatedDataField</c> with Solve's ConversionFunction, followed by
	/// Solve's <c>(*normalInfo)[i] *= -1</c>.
	/// </summary>
	public static SparseNodeData SetNormalField(
		PoissonSampleSet samples,
		DensityEstimator density,
		int minDepth,
		int maxDepth,
		float minDepthCutoff,
		out (float DepthSum, float WeightSum, float TotalWeight) pointDepthAndWeight,
		CancellationToken cancellationToken = default,
		IProgress<PoissonProgress>? progress = null)
	{
		FemTree tree = samples.Tree;
		using FemTree.SubTreeScope scope = tree.ExtractSubTree(tree.SpaceRoot);
		BSplineSupportSizes sizes = BSplineSupportSizes.For(NormalDegree);

		// oneKey: DensityDegree == the normals' degree, so one key serves both.
		var key = new NeighborKey(tree, sizes.SupportEnd, -sizes.SupportStart, resetOnMissing: true);
		key.Set(maxDepth);
		var dataField = new SparseNodeData(3);
		float depthAndWeightSum = 0;
		float depthSum = 0;
		float weightSum = 0;
		Span<float> p = stackalloc float[3];
		Span<float> n = stackalloc float[3];
		Span<float> center = stackalloc float[3];
		for (int i = 0; i < samples.Count; i++)
		{
			if ((i & 0xFFFF) == 0)
			{
				cancellationToken.ThrowIfCancellationRequested();
				progress?.Report(new PoissonProgress(PoissonStage.Splat, (double)i / samples.Count));
			}

			float sampleWeight = samples.Weight(i);
			if (!(sampleWeight > 0))
			{
				continue;
			}

			for (int d = 0; d < 3; d++)
			{
				p[d] = samples.Position(i, d) / sampleWeight;
				n[d] = samples.Normal(i, d) / sampleWeight;
			}

			// "Point sample is out of bounds" (a warning in the C++) skips the sample.
			if (!InBounds(p))
			{
				continue;
			}

			// Solve's ConversionFunction: normalize, rejecting a zero average normal.
			float l = MathF.Sqrt(0.0f + n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
			if (l == 0)
			{
				continue;
			}

			for (int d = 0; d < 3; d++)
			{
				n[d] /= l;
			}

			depthAndWeightSum += sampleWeight;
			for (int d = 0; d < 3; d++)
			{
				n[d] *= sampleWeight;
			}

			(float rawDepth, float weight) = SplatPointData(tree, density, minDepthCutoff, p, n, dataField, key, minDepth, maxDepth, 3, 0.0f, center);
			depthSum += rawDepth * sampleWeight;
			weightSum += weight * sampleWeight;
		}

		// The per-thread sums are added into a zero point (one thread here).
		pointDepthAndWeight = (0.0f + depthSum, 0.0f + weightSum, 0.0f + depthAndWeightSum);

		Span<float> values = dataField.Values;
		for (int k = 0; k < values.Length; k++)
		{
			values[k] *= -1.0f;
		}

		progress?.Report(new PoissonProgress(PoissonStage.Splat, 1.0));
		return dataField;
	}

	/// <summary>
	/// The auxiliary (color) field: each sample's summed auxiliary data and weight, splatted
	/// with the degree-0 basis into its node and every ancestor up to the unit cube (scaled by
	/// 4^depth), then scaled by perLevelDataScaleFactor^depth. Entries are
	/// (data..., weight). Port of Solve's <c>setExtrapolatedDataField&lt;DataSig,false,...&gt;</c>
	/// call (without a density) and its per-level scaling.
	/// </summary>
	public static SparseNodeData SetAuxField(PoissonSampleSet samples, float perLevelDataScaleFactor)
	{
		FemTree tree = samples.Tree;
		int aux = samples.AuxPerPoint;
		var dataField = new SparseNodeData(aux + 1);
		int maxDepth = tree.MaxDepth(tree.SpaceRoot);

		// No SubTreeExtractor here: depths are global and the local depth is Depth - 1.
		var dataKey = new NeighborKey(tree, 0, 0, resetOnMissing: true);
		dataKey.Set(maxDepth + tree.DepthOffset);
		Span<float> p = stackalloc float[3];
		Span<float> v = stackalloc float[aux + 1];
		for (int i = 0; i < samples.Count; i++)
		{
			float sampleWeight = samples.Weight(i);
			for (int d = 0; d < 3; d++)
			{
				p[d] = sampleWeight == 0 ? samples.Position(i, d) : samples.Position(i, d) / sampleWeight;
			}

			// "Point is out of bounds" (a warning in the C++) skips the sample.
			if (!InBounds(p))
			{
				continue;
			}

			// _multiSplatPointData without a density: weight 1, degree-0 values 1.
			for (int k = 0; k < aux; k++)
			{
				v[k] = samples.Aux(i, k) * 1.0f;
			}

			v[aux] = sampleWeight * 1.0f;
			int node = samples.Node(i);
			dataKey.GetNeighbors(node);
			for (int current = node; tree.Depth(current) - tree.DepthOffset >= 0; current = tree.Parent(current))
			{
				int localDepth = tree.Depth(current) - tree.DepthOffset;
				float levelScale = (float)Math.Pow(1 << localDepth, 2);
				int neighbor = dataKey.Window(tree.Depth(current))[0];
				if (tree.IsActive(neighbor))
				{
					int slot = dataField.At(tree.NodeIndex(neighbor));
					for (int k = 0; k <= aux; k++)
					{
						dataField.Value(slot, k) += v[k] * levelScale * (float)1.0;
					}
				}
			}
		}

		// Solve: (*clr) *= pow( perLevelDataScaleFactor , tree.depth( n ) ) for every node.
		tree.ProcessNodes(tree.Root, node =>
		{
			int slot = dataField.Index(tree.NodeIndex(node));
			if (slot != -1)
			{
				float scale = (float)Math.Pow(perLevelDataScaleFactor, tree.Depth(node) - tree.DepthOffset);
				for (int k = 0; k <= aux; k++)
				{
					dataField.Value(slot, k) *= scale;
				}
			}
		});
		return dataField;
	}

	/// <summary>
	/// The density read back at position from the kernel around node. Port of
	/// <c>_getSamplesPerNode</c>.
	/// </summary>
	internal static float GetSamplesPerNode(FemTree tree, DensityEstimator density, int node, ReadOnlySpan<float> position, NeighborKey weightKey)
	{
		float weight = 0;
		int size = BSplineSupportSizes.For(density.Degree).SupportSize;
		int[] neighbors = weightKey.GetNeighbors(node);
		Span<double> values = stackalloc double[3 * size];
		Span<float> start = stackalloc float[3];
		tree.LocalStartAndWidth(node, start, out float w);
		for (int dim = 0; dim < 3; dim++)
		{
			PoissonPolynomial.BSplineComponentValues(density.Degree, (position[dim] - start[dim]) / w, values.Slice(dim * size, size));
		}

		for (int i0 = 0; i0 < size; i0++)
		{
			double s1 = 1.0 * values[i0];
			for (int i1 = 0; i1 < size; i1++)
			{
				double s2 = s1 * values[size + i1];
				for (int i2 = 0; i2 < size; i2++)
				{
					double s3 = s2 * values[2 * size + i2];
					int neighbor = neighbors[(i0 * size + i1) * size + i2];
					if (neighbor != FemTree.None)
					{
						int slot = density.Index(tree.NodeIndex(neighbor));
						if (slot != -1)
						{
							weight += (float)(s3 * density.Value(slot));
						}
					}
				}
			}
		}

		return weight;
	}

	/// <summary>
	/// The depth at which the density around position reaches samplesPerNode, and the sample's
	/// area weight. Port of <c>_getSampleDepthAndWeight( density , node , position , key , depth , weight )</c>.
	/// </summary>
	internal static (float Depth, float Weight) GetSampleDepthAndWeight(FemTree tree, DensityEstimator density, int node, ReadOnlySpan<float> position, NeighborKey weightKey)
	{
		int temp = node;
		while (tree.LocalDepth(temp) > density.KernelDepth)
		{
			temp = tree.Parent(temp);
		}

		float depth;
		float samplesPerNode = GetSamplesPerNode(tree, density, temp, position, weightKey);
		if (samplesPerNode >= density.SamplesPerNode)
		{
			depth = (float)(tree.LocalDepth(temp) + LogF(samplesPerNode / density.SamplesPerNode) / (Math.Log(2.0) * (3 - density.CoDimension)));
		}
		else
		{
			float fineSamplesPerNode = samplesPerNode;
			float coarseSamplesPerNode = samplesPerNode;
			while (coarseSamplesPerNode < density.SamplesPerNode && tree.LocalDepth(temp) != 0)
			{
				temp = tree.Parent(temp);
				fineSamplesPerNode = coarseSamplesPerNode;
				coarseSamplesPerNode = GetSamplesPerNode(tree, density, temp, position, weightKey);
			}

			// Rather than assuming that the number of samples per node scales by a factor of 2^(Dim-CoDim),
			// use the fact that between the coarse and fine levels the samples per node scaled by coarseSamplesPerNode / fineSamplesPerNode
			depth = tree.LocalDepth(temp) + (LogF(coarseSamplesPerNode / density.SamplesPerNode) / LogF(coarseSamplesPerNode / fineSamplesPerNode));
			samplesPerNode = coarseSamplesPerNode;
		}

		float nodeWidth = (float)(1.0 / (1 << tree.LocalDepth(temp)));
		float weight = (float)Math.Pow(nodeWidth, 3 - density.CoDimension) / samplesPerNode;
		return (depth, weight);
	}

	// The density form of _splatPointData< true , true , WeightDegree , V , DataSigs... >.
	private static (float RawDepth, float Weight) SplatPointData(
		FemTree tree,
		DensityEstimator density,
		float minDepthCutoff,
		ReadOnlySpan<float> position,
		ReadOnlySpan<float> v,
		SparseNodeData dataField,
		NeighborKey key,
		int minDepth,
		int maxDepth,
		int dim,
		float depthBias,
		Span<float> center)
	{
		// Get the depth and weight at position
		int temp = tree.SpaceRoot;
		center.Fill(0.5f);
		float width = 1.0f;
		while (tree.Depth(temp) < density.KernelDepth)
		{
			if (!tree.HasChildren(temp) || !tree.IsActive(tree.FirstChild(temp)))
			{
				break;
			}

			int cIndex = FemTree.ChildIndex(center, position);
			temp = tree.FirstChild(temp) + cIndex;
			width /= 2;
			Step(center, cIndex, width);
		}

		(float depth, float weight) = GetSampleDepthAndWeight(tree, density, temp, position, key);
		depth += depthBias;
		if (depth < minDepthCutoff)
		{
			return (-1.0f, 0.0f);
		}

		float rawDepth = depth;
		if (depth < minDepth)
		{
			depth = minDepth;
		}

		if (depth > maxDepth)
		{
			depth = maxDepth;
		}

		int topDepth = (int)MathF.Ceiling(depth);
		double dx = 1.0 - (topDepth - depth);
		if (topDepth <= minDepth)
		{
			topDepth = minDepth;
			dx = 1;
		}
		else if (topDepth > maxDepth)
		{
			topDepth = maxDepth;
			dx = 1;
		}

		while (tree.Depth(temp) > topDepth)
		{
			temp = tree.Parent(temp);
		}

		while (tree.Depth(temp) < topDepth)
		{
			if (!tree.HasChildren(temp))
			{
				tree.InitChildren(temp, threadSafe: true);
			}

			int cIndex = FemTree.ChildIndex(center, position);
			temp = tree.FirstChild(temp) + cIndex;
			width /= 2;
			Step(center, cIndex, width);
		}

		// The Splat lambda: always into temp, whatever node it is handed (see the file header).
		Span<float> scaled = stackalloc float[3];
		Splat(tree, temp, position, v, weight, dim, (float)dx, scaled, dataField, key);
		if (Math.Abs(1.0 - dx) > 1e-6)
		{
			Splat(tree, temp, position, v, weight, dim, (float)(1.0 - dx), scaled, dataField, key);
		}

		return (rawDepth, weight);
	}

	// The density form's Splat lambda: v * weight / width^dim * fraction into node's neighbors.
	private static void Splat(FemTree tree, int node, ReadOnlySpan<float> position, ReadOnlySpan<float> v, float weight, int dim, float fraction, Span<float> scaled, SparseNodeData dataField, NeighborKey key)
	{
		double nodeWidth = 1.0 / (1 << tree.Depth(node));
		float volume = (float)Math.Pow(nodeWidth, dim);
		for (int d = 0; d < 3; d++)
		{
			scaled[d] = v[d] * weight / volume * fraction;
		}

		SplatPointData(tree, node, position, scaled, dataField, key);
	}

	// The node form of _splatPointData< true , true , V , DataSigs... > for degree-2 data:
	// adds v times the B-spline weights at position into the 3x3x3 neighbors of node.
	private static void SplatPointData(FemTree tree, int node, ReadOnlySpan<float> position, ReadOnlySpan<float> v, SparseNodeData dataField, NeighborKey dataKey)
	{
		const int Size = 3;
		int[] neighbors = dataKey.GetNeighbors(node, createNodes: true, threadSafe: true);
		Span<float> start = stackalloc float[3];
		Span<double> values = stackalloc double[3 * Size];
		tree.StartAndWidth(node, start, out float w);
		for (int d = 0; d < 3; d++)
		{
			PoissonPolynomial.BSplineComponentValues(NormalDegree, (position[d] - start[d]) / w, values.Slice(d * Size, Size));
		}

		for (int i0 = 0; i0 < Size; i0++)
		{
			double s1 = 1.0 * values[i0];
			for (int i1 = 0; i1 < Size; i1++)
			{
				double s2 = s1 * values[Size + i1];
				for (int i2 = 0; i2 < Size; i2++)
				{
					double s3 = s2 * values[2 * Size + i2];
					int neighbor = neighbors[(i0 * Size + i1) * Size + i2];
					if (tree.IsActive(neighbor))
					{
						int slot = dataField.At(tree.NodeIndex(neighbor));
						float f = (float)s3;
						for (int k = 0; k < 3; k++)
						{
							dataField.Value(slot, k) += v[k] * f;
						}
					}
				}
			}
		}
	}

	private static void Step(Span<float> center, int cIndex, float width)
	{
		for (int d = 0; d < 3; d++)
		{
			if (((cIndex >> d) & 1) != 0)
			{
				center[d] += width / 2;
			}
			else
			{
				center[d] -= width / 2;
			}
		}
	}

	// FEMTree::_InBounds.
	private static bool InBounds(ReadOnlySpan<float> p)
	{
		for (int d = 0; d < 3; d++)
		{
			if (p[d] < 0 || p[d] > 1)
			{
				return false;
			}
		}

		return true;
	}

	/// <summary>log( float ): the double logarithm rounded to float (divergence 116).</summary>
	internal static float LogF(float x) => (float)Math.Log(x);
}
