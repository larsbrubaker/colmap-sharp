// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonInterpolation: the approximate point-interpolation constraints of thirdparty/
// PoissonRecon - FEMTree::InitializeApproximatePointInterpolationInfo (FEMTree.h) with
// _densifyInterpolationInfoAndSetDualConstraints and _setInterpolationInfoFromChildren
// (FEMTree.inl), for Solve's value-only (PointD = 0) screening term. Each active node gets a
// DualPointInfo: the samples' mean position in its subtree, their weight scaled by
// 2^(local depth * adaptiveExponent - maxDepth * (adaptiveExponent - 1)), and the dual value
// (target * pointWeight-scaled area) times that weight. The FEM system (next slices) adds
// these as interpolation constraints and the screening term of its matrix. Solve builds it
// after the splats and before finalizing the tree (which reorders it with the node data).
// Tier A against oracle/poisson_tree_harness.cc (the "interpolation" cases).
//
// Entries are (position x, y, z, weight, dual value) in a SparseNodeData of width 5, created
// in the C++'s order: the samples' nodes in sample order, then interior nodes as the
// post-order child pass first adds into them.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Solve's point-interpolation constraints. Port of PoissonRecon's
/// <c>ApproximatePointInterpolationInfo&lt;Real,0,ConstraintDual,SystemDual&gt;</c> data.
/// </summary>
public static class PoissonInterpolation
{
	/// <summary>Floats per entry: position (3), weight, dual value.</summary>
	public const int Width = 5;

	/// <summary>
	/// Builds the interpolation data for the samples with Solve's ConstraintDual (target value,
	/// weight). Port of <c>InitializeApproximatePointInterpolationInfo&lt;Real,0&gt;( tree ,
	/// samples , ConstraintDual( target , weight ) , ... , maxDepth , adaptiveExponent )</c>.
	/// </summary>
	public static SparseNodeData Build(PoissonSampleSet samples, float targetValue, float constraintWeight, int maxDepth, int adaptiveExponent)
	{
		FemTree tree = samples.Tree;
		var iInfo = new SparseNodeData(Width);
		for (int i = 0; i < samples.Count; i++)
		{
			int node = samples.Node(i);
			while (!tree.IsActive(node))
			{
				node = tree.Parent(node);
			}

			float weight = samples.Weight(i);
			if (weight != 0)
			{
				int slot = iInfo.At(tree.NodeIndex(node));
				for (int d = 0; d < 3; d++)
				{
					iInfo.Value(slot, d) += samples.Position(i, d);
				}

				iInfo.Value(slot, 3) += weight;

				// constraintDual( p ) = target * weight (the C++ recomputes the product per call).
				iInfo.Value(slot, 4) += (targetValue * constraintWeight) * weight;
			}
		}

		// Set the interior values
		SetFromChildren(tree, iInfo, tree.SpaceRoot);

		Span<float> values = iInfo.Values;
		for (int slot = 0; slot < iInfo.Count; slot++)
		{
			float w = values[slot * Width + 3];
			for (int k = 0; k < Width; k++)
			{
				values[slot * Width + k] /= w;
			}

			values[slot * Width + 3] = w;
		}

		// Set the average position and scale the weights
		tree.ProcessNodes(tree.Root, node =>
		{
			if (!tree.IsActive(node))
			{
				return;
			}

			int slot = iInfo.Index(tree.NodeIndex(node));
			if (slot == -1)
			{
				return;
			}

			int e = (tree.LocalDepth(node) * adaptiveExponent) - (maxDepth * (adaptiveExponent - 1));
			if (e < 0)
			{
				iInfo.Value(slot, 3) /= 1 << -e;
			}
			else
			{
				iInfo.Value(slot, 3) *= 1 << e;
			}

			iInfo.Value(slot, 4) *= iInfo.Value(slot, 3);
		});
		return iInfo;
	}

	/// <summary>
	/// The point weight of Solve's constraints: pointDepthAndWeight.value()[1], i.e. the summed
	/// sample weights over the summed sample counts. Port of <c>ProjectiveData::operator Data</c>.
	/// </summary>
	public static float AverageSampleWeight(float weightSum, float totalWeight) =>
		totalWeight != 0 ? weightSum / totalWeight : weightSum * totalWeight;

	// Port of _setInterpolationInfoFromChildren: adds each node's children's entries (post
	// order) into it, creating its entry; returns whether the subtree has data.
	private static bool SetFromChildren(FemTree tree, SparseNodeData iInfo, int node)
	{
		int firstChild = tree.FirstChild(node);
		if (firstChild != FemTree.None && tree.IsActive(firstChild))
		{
			bool hasChildData = false;
			Span<float> t = stackalloc float[Width];
			for (int c = 0; c < FemTree.ChildCount; c++)
			{
				if (SetFromChildren(tree, iInfo, firstChild + c))
				{
					int childSlot = iInfo.At(tree.NodeIndex(firstChild + c));

					// DualPointInfo::operator +=: position, weight, dual values.
					for (int k = 0; k < Width; k++)
					{
						t[k] += iInfo.Value(childSlot, k);
					}

					hasChildData = true;
				}
			}

			if (hasChildData && tree.IsActive(node))
			{
				int slot = iInfo.At(tree.NodeIndex(node));
				for (int k = 0; k < Width; k++)
				{
					iInfo.Value(slot, k) += t[k];
				}
			}

			return hasChildData;
		}

		return iInfo.Index(tree.NodeIndex(node)) != -1;
	}
}
