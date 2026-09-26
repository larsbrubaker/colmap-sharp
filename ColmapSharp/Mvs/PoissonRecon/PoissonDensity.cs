// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonDensity: the kernel density estimator of thirdparty/PoissonRecon -
// FEMTree::setDensityEstimator / updateDensityEstimator (FEMTree.inl) and
// _GetScaleValue / _addWeightContribution (FEMTree.WeightedSamples.inl). Every node up to the
// kernel depth that holds samples in its subtree splats the subtree's total weight, at the
// subtree's mean position, into the 3x3x3 neighbors around it with degree-2 B-spline
// weights (creating missing neighbors), normalized so a flat sheet of samples reads back
// one sample per unit area. Solve calls it right after resetNodeIndices, and the normal
// splatting (PoissonSplat) reads it to pick each sample's splat depth. Tier A against
// oracle/poisson_tree_harness.cc (the "density*" cases): same values, same entry order, same
// created nodes.
//
// Translation notes:
// - The recursion (SetDensity) is kept: it returns each subtree's summed ProjectiveData in
//   child order, exactly as the C++ accumulates it, and its depth is the tree depth.
// - The C++ caches ScaleValue in a function-local static per (CoDim, Degree); here a static
//   field for the (1, 2) instantiation COLMAP uses and a computation otherwise.
// - The weights are accumulated with AddAtomic in the C++, from one thread in COLMAP's
//   single-threaded use of this stage; the port is sequential (the recursion is sequential in
//   the C++ too).

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Kernel density estimation over the octree samples. Port of FEMTree's density estimator.
/// </summary>
public static class PoissonDensity
{
	private static readonly float SurfaceScaleValue2 = GetScaleValue(1, 2, 10);

	/// <summary>
	/// Builds the density estimator of the samples, splatted at depths 0..splatDepth (clamped
	/// to the tree), with the given co-dimension and kernel degree. Port of
	/// <c>setDensityEstimator&lt;CoDim,DensityDegree&gt;( samples , splatDepth , samplesPerNode )</c>.
	/// </summary>
	public static DensityEstimator SetDensityEstimator(PoissonSampleSet samples, int coDimension, int degree, int splatDepth, float samplesPerNode)
	{
		FemTree tree = samples.Tree;
		int maxDepth = tree.MaxDepth(tree.SpaceRoot);
		splatDepth = StdMinMax.StdMax(0, StdMinMax.StdMin(splatDepth, maxDepth));
		var density = new DensityEstimator(degree, splatDepth, coDimension, samplesPerNode);
		UpdateDensityEstimator(density, samples, 0, splatDepth);
		return density;
	}

	/// <summary>
	/// Splats the samples into <paramref name="density"/> at depths minSplatDepth..maxSplatDepth.
	/// Port of <c>updateDensityEstimator&lt;CoDim,DensityDegree&gt;( density , samples , min , max )</c>.
	/// </summary>
	public static void UpdateDensityEstimator(DensityEstimator density, PoissonSampleSet samples, int minSplatDepth, int maxSplatDepth)
	{
		FemTree tree = samples.Tree;
		using FemTree.SubTreeScope scope = tree.ExtractSubTree(tree.SpaceRoot);
		int maxDepth = tree.MaxDepth(tree.SpaceRoot);
		maxSplatDepth = StdMinMax.StdMax(0, StdMinMax.StdMin(maxSplatDepth, maxDepth));
		minSplatDepth = StdMinMax.StdMax(0, StdMinMax.StdMin(minSplatDepth, maxDepth));
		if (minSplatDepth > maxSplatDepth)
		{
			throw new InvalidOperationException("Minimum splat depth exceeds maximum splat depth");
		}

		BSplineSupportSizes sizes = BSplineSupportSizes.For(density.Degree);

		// PointSupportKey< Degree >: left radius SupportEnd, right radius -SupportStart.
		var densityKey = new NeighborKey(tree, sizes.SupportEnd, -sizes.SupportStart, resetOnMissing: true);
		densityKey.Set(maxSplatDepth);

		// Initialize the map from node indices to samples
		var sampleMap = new int[tree.NodeCount];
		Array.Fill(sampleMap, -1);
		for (int i = 0; i < samples.Count; i++)
		{
			if (samples.Weight(i) > 0)
			{
				sampleMap[tree.NodeIndex(samples.Node(i))] = i;
			}
		}

		var context = new SplatContext(tree, samples, density, densityKey, sampleMap, minSplatDepth, maxSplatDepth);
		Span<float> total = stackalloc float[4];
		context.SetDensity(tree.SpaceRoot, total);
	}

	/// <summary>
	/// The normalization of a degree-<paramref name="degree"/> splat: 1 over the density a
	/// sheet of co-dimension <paramref name="coDimension"/> of unit-weight points reads back at
	/// p (in a node's [0,1]^3 frame). Port of <c>_GetScaleValue&lt;CoDim,Degree&gt;( p )</c>.
	/// </summary>
	public static float GetScaleValue(int coDimension, int degree, ReadOnlySpan<float> p)
	{
		BSplineSupportSizes sizes = BSplineSupportSizes.For(degree);
		int pointSupportStart = -sizes.SupportEnd;
		int pointSupportEnd = -sizes.SupportStart;
		int n = pointSupportEnd - pointSupportStart + 1;
		Span<double> splineValues = stackalloc double[3 * (degree + 1)];

		// Evaluate the B-spline component functions at the position
		for (int d = 0; d < 3; d++)
		{
			PoissonPolynomial.BSplineComponentValues(degree, p[d], splineValues.Slice(d * (degree + 1), degree + 1));
		}

		// Get the values with which the center point splats into its neighbors
		Span<double> splatValues = stackalloc double[n * n * n];
		Span<double> densityValues = stackalloc double[n * n * n];
		for (int i0 = 0; i0 < n; i0++)
		{
			double s1 = 1.0 * splineValues[i0];
			for (int i1 = 0; i1 < n; i1++)
			{
				double s2 = s1 * splineValues[(degree + 1) + i1];
				for (int i2 = 0; i2 < n; i2++)
				{
					int w = (i0 * n + i1) * n + i2;
					splatValues[w] = s2 * splineValues[2 * (degree + 1) + i2];
					densityValues[w] = 0;
				}
			}
		}

		// Splat from points along the hyperplane: a point at node i contributes to the
		// evaluation at node 0 if - PointSupportEnd - BSplineSupportEnd <= i <= - PointSupportStart - BSplineSupportStart.
		int nbStart = -pointSupportEnd - sizes.SupportEnd;
		int nbEnd = -pointSupportStart - sizes.SupportStart;
		Span<int> nb = stackalloc int[3];
		Span<int> si = stackalloc int[3];
		for (nb[0] = nbStart; nb[0] <= nbEnd; nb[0]++)
		{
			for (nb[1] = nbStart; nb[1] <= nbEnd; nb[1]++)
			{
				for (nb[2] = nbStart; nb[2] <= nbEnd; nb[2]++)
				{
					// Check that the neighboring point's node lies on the hyperplane
					bool validNeighbor = true;
					for (int d = 0; d < coDimension; d++)
					{
						if (nb[d] != 0)
						{
							validNeighbor = false;
						}
					}

					if (!validNeighbor)
					{
						continue;
					}

					// Iterate over all B-Splines supported on the neighboring point
					for (si[0] = pointSupportStart; si[0] <= pointSupportEnd; si[0]++)
					{
						for (si[1] = pointSupportStart; si[1] <= pointSupportEnd; si[1]++)
						{
							for (si[2] = pointSupportStart; si[2] <= pointSupportEnd; si[2]++)
							{
								bool inRange = true;
								int target = 0;
								int source = 0;
								for (int d = 0; d < 3; d++)
								{
									int idx = nb[d] + si[d] - pointSupportStart;
									if (idx < 0 || idx >= n)
									{
										inRange = false;
									}

									target = target * n + idx;
									source = source * n + (si[d] - pointSupportStart);
								}

								if (inRange)
								{
									densityValues[target] += splatValues[source];
								}
							}
						}
					}
				}
			}
		}

		double scaleValue = 0;
		for (int w = 0; w < n * n * n; w++)
		{
			scaleValue += splatValues[w] * densityValues[w];
		}

		return (float)(1.0 / scaleValue);
	}

	/// <summary>
	/// The mean of <see cref="GetScaleValue(int,int,ReadOnlySpan{float})"/> over a res^3 grid of
	/// cell centers. Port of <c>_GetScaleValue&lt;CoDim,Degree&gt;( res )</c>.
	/// </summary>
	public static float GetScaleValue(int coDimension, int degree, int res)
	{
		Span<float> p = stackalloc float[3];
		float dx = (float)(1.0 / res);
		uint count = 0;
		float scaleValueSum = 0;
		for (int i0 = 0; i0 < res; i0++)
		{
			p[0] = dx / 2 + dx * i0;
			for (int i1 = 0; i1 < res; i1++)
			{
				p[1] = dx / 2 + dx * i1;
				for (int i2 = 0; i2 < res; i2++)
				{
					p[2] = dx / 2 + dx * i2;
					count++;
					scaleValueSum += GetScaleValue(coDimension, degree, p);
				}
			}
		}

		return scaleValueSum / count;
	}

	/// <summary>
	/// Adds weight times the scaled B-spline kernel centered at position into the density of
	/// the node's neighbors (created if missing). Port of <c>_addWeightContribution</c>.
	/// </summary>
	internal static void AddWeightContribution(FemTree tree, DensityEstimator density, int node, ReadOnlySpan<float> position, NeighborKey weightKey, float weight)
	{
		float scaleValue = density.CoDimension == 1 && density.Degree == 2
			? SurfaceScaleValue2
			: GetScaleValue(density.CoDimension, density.Degree, 10);
		int size = BSplineSupportSizes.For(density.Degree).SupportSize;
		// _addWeightContribution< ThreadSafe=true , ... >: getNeighbors< true , true >.
		int[] neighbors = weightKey.GetNeighbors(node, createNodes: true, threadSafe: true);
		density.Reserve(tree.NodeCount);

		// Evaluate the B-spline components at the position
		Span<double> values = stackalloc double[3 * size];
		Span<float> start = stackalloc float[3];
		tree.StartAndWidth(node, start, out float w);
		for (int dim = 0; dim < 3; dim++)
		{
			PoissonPolynomial.BSplineComponentValues(density.Degree, (position[dim] - start[dim]) / w, values.Slice(dim * size, size));
		}

		weight *= scaleValue;
		double s0 = weight;
		for (int i0 = 0; i0 < size; i0++)
		{
			double s1 = s0 * values[i0];
			for (int i1 = 0; i1 < size; i1++)
			{
				double s2 = s1 * values[size + i1];
				for (int i2 = 0; i2 < size; i2++)
				{
					double s3 = s2 * values[2 * size + i2];
					int neighbor = neighbors[(i0 * size + i1) * size + i2];
					if (neighbor != FemTree.None)
					{
						density.Value(density.At(tree.NodeIndex(neighbor))) += (float)s3;
					}
				}
			}
		}
	}

	// The state of updateDensityEstimator's recursive SetDensity lambda.
	private sealed class SplatContext(
		FemTree tree,
		PoissonSampleSet samples,
		DensityEstimator density,
		NeighborKey densityKey,
		int[] sampleMap,
		int minSplatDepth,
		int maxSplatDepth)
	{
		// Writes the subtree's summed (x, y, z, weight) to total.
		public void SetDensity(int node, Span<float> total)
		{
			total.Clear();
			int d = tree.Depth(node);
			int idx = tree.NodeIndex(node);
			if (tree.HasChildren(node))
			{
				int brood = tree.FirstChild(node);
				Span<float> child = stackalloc float[4];
				for (int c = 0; c < FemTree.ChildCount; c++)
				{
					SetDensity(brood + c, child);
					for (int k = 0; k < 4; k++)
					{
						total[k] += child[k];
					}
				}
			}

			if (idx < sampleMap.Length && sampleMap[idx] != -1)
			{
				int s = sampleMap[idx];
				total[0] += samples.Position(s, 0);
				total[1] += samples.Position(s, 1);
				total[2] += samples.Position(s, 2);
				total[3] += samples.Weight(s);
			}

			if (d >= minSplatDepth && d <= maxSplatDepth && total[3] > 0)
			{
				Span<float> p = stackalloc float[3];
				for (int k = 0; k < 3; k++)
				{
					p[k] = total[k] / total[3];
				}

				AddWeightContribution(tree, density, node, p, densityKey, total[3]);
			}
		}
	}
}
