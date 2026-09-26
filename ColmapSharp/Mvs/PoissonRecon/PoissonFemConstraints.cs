// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonFemConstraints: FEMTree::addFEMConstraints / _addFEMConstraints
// (thirdparty/PoissonRecon/FEMTree.h and FEMTree.System.inl) for Solve's Poisson constraints:
// the integral of the divergence of the splatted normal field (PoissonSplat, degree-2
// Dirichlet-derivative coefficients) against every degree-1 Neumann test function of the
// finalized tree (PoissonFinalize). Same-depth contributions are gathered per test node
// through the FemConstraintIntegrator stencil (or exact integrals near the boundary); finer
// normals reach coarser test functions two ways, as upstream: each normal scatters to the
// test functions around its parent (parent-child stencils), and the scattered values are
// restricted further down with PoissonMultigrid.DownSample; coarser normals reach finer test
// functions by prolonging the accumulated normal coefficients (PoissonMultigrid.UpSample) and
// integrating them against each test node's parent neighbors (child-parent stencils). Tier A
// against oracle/poisson_system_harness.cc (the "femconstraints" cases).
//
// Translation notes:
// - The C++ runs each depth through ThreadPool::ParallelFor with atomic float adds into the
//   coarser depth; the order of those adds reaches the float sums, so the port runs the nodes
//   in sorted order, which is the C++'s order with ThreadPool::NONE (the harness's setting).
// - hasCoarserCoefficients starts true upstream, so the coarse-to-fine pass always runs; the
//   port keeps it unconditional.
// - The normal coefficients are prolonged with the TEST basis' restriction/prolongation
//   (F.tRestrictionProlongation(), degree 1 with FEM_FLAG_1), not the normals' own degree-2
//   basis, exactly as upstream does.
// - _StencilDot< double , float , 3 >: dot = 0f; dot += data[d] * (float)stencil[d] per axis.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// Solve's normal-divergence right-hand side. Port of PoissonRecon's <c>FEMTree::addFEMConstraints</c>
/// for the Poisson constraint.
/// </summary>
public static class PoissonFemConstraints
{
	/// <summary>The test basis: degree 1, Neumann (Solve's Sigs).</summary>
	public const int TestSignature = 5;

	/// <summary>The normal field's basis: degree 2, the derivative boundary of Neumann (Solve's NormalSigs).</summary>
	public const int NormalSignature = 7;

	/// <summary>
	/// Solve's constraint integrator: test functions with one derivative against the normals'
	/// functions with none, weights[d][e_d][0] = 1 (the divergence).
	/// </summary>
	public static FemConstraintIntegrator CreateDivergenceIntegrator()
	{
		var f = new FemConstraintIntegrator(TestSignature, 1, NormalSignature, 0, 3);
		for (int d = 0; d < 3; d++)
		{
			f.Weights[d, FemConstraintIntegrator.DerivativeIndex(1, d == 0 ? 1 : 0, d == 1 ? 1 : 0, d == 2 ? 1 : 0), 0] = 1;
		}

		return f;
	}

	/// <summary>
	/// Adds to <paramref name="constraints"/> (one float per sorted node) the integral of
	/// <paramref name="f"/>'s constraint over the <paramref name="normals"/> field against every
	/// test function at local depths 0..min(<paramref name="maxDepth"/>, tree max depth). Port of
	/// <c>addFEMConstraints( F , normalInfo , constraints , maxDepth )</c>.
	/// </summary>
	public static void AddFemConstraints(FemTree tree, SortedTreeNodes sorted, FemConstraintIntegrator f, SparseNodeData normals, float[] constraints, int maxDepth)
	{
		PoissonMultigrid.SetFem1ValidityFlags(tree, sorted, f.TestSignature);
		PoissonMultigrid.SetFem2ValidityFlags(tree, sorted, f.ConstraintSignature);
		int femDegree = FemSignature.Degree(f.TestSignature);
		int cDegree = FemSignature.Degree(f.ConstraintSignature);
		BSplineOverlapSizes femC = BSplineOverlapSizes.For(femDegree, cDegree);
		BSplineOverlapSizes cFem = BSplineOverlapSizes.For(cDegree, femDegree);
		int treeMaxDepth = PoissonMultigrid.MaxDepth(tree);

		// To set the constraints, we iterate over the splatted normals and compute the dot-product of the divergence of the normal field with all the basis functions.
		// Within the same depth: set directly as a gather
		// Coarser depths
		maxDepth = StdMinMax.StdMin(maxDepth, treeMaxDepth);
		var coarserConstraints = new float[PoissonMultigrid.End(tree, sorted, maxDepth - 1)];
		int[][] cfemLoopData = PoissonMultigrid.ParentOverlapLoopData(cDegree, femDegree);
		int[][] femcLoopData = PoissonMultigrid.ParentOverlapLoopData(femDegree, cDegree);
		var rp = new RestrictionProlongation(f.TestSignature);
		var neighbors = new int[Cube(femC.OverlapSize)];
		var pNeighbors = new int[Cube(cFem.OverlapSize)];
		var off = new int[3];
		var other = new int[3];
		var integral = new double[3];

		// Iterate from fine to coarse, setting the constraints @(depth) and the cumulative constraints @(depth-1)
		for (int d = maxDepth; d >= 0; d--)
		{
			f.Init(d);
			double[] stencil = f.SetStencil();
			double[][] stencils = f.SetParentChildStencils();
			var neighborKey = new NeighborKey(tree, 1, 1, resetOnMissing: false);
			neighborKey.Set(d + tree.DepthOffset);
			int begin = PoissonMultigrid.Begin(tree, sorted, d), end = PoissonMultigrid.End(tree, sorted, d);
			for (int i = begin; i < end; i++)
			{
				if (d < maxDepth)
				{
					constraints[i] += coarserConstraints[i];
				}

				int node = sorted.TreeNodes[i];
				neighborKey.GetNeighbors(-femC.OverlapStart, femC.OverlapEnd, node, neighbors);
				ReadOffset(tree, node, off);

				// Set constraints from current depth
				// Gather the constraints from _node into the constraint stored with node
				if (PoissonMultigrid.IsValidFem1Node(tree, node))
				{
					bool isInterior = PoissonMultigrid.IsInteriorlyOverlapped(femDegree, cDegree, d, off[0], off[1], off[2]);
					for (int j = 0; j < neighbors.Length; j++)
					{
						int slot = PoissonMultigrid.IsValidFem2Node(tree, neighbors[j]) ? normals.Index(tree.NodeIndex(neighbors[j])) : -1;
						if (slot == -1)
						{
							continue;
						}

						if (isInterior)
						{
							constraints[i] += StencilDot(stencil, j, normals, slot);
						}
						else
						{
							ReadOffset(tree, neighbors[j], other);
							f.CcIntegrate(off, other, integral);
							constraints[i] += StencilDot(integral, 0, normals, slot);
						}
					}
				}

				int dataSlot = PoissonMultigrid.IsValidFem2Node(tree, node) ? normals.Index(tree.NodeIndex(node)) : -1;
				if (dataSlot == -1 || IsZero(normals, dataSlot) || d == 0)
				{
					continue;
				}

				// Set the _constraints for the parents
				int parent = tree.Parent(node);
				bool isInterior2 = PoissonMultigrid.IsInteriorlyOverlapped(tree, cDegree, femDegree, parent);
				int cIdx = tree.ChildIndexInParent(node);
				double[] pcStencil = stencils[cIdx];
				neighborKey.GetNeighbors(-cFem.OverlapStart, cFem.OverlapEnd, parent, neighbors);
				foreach (int idx in cfemLoopData[cIdx])
				{
					int n = neighbors[idx];
					if (n == FemTree.None)
					{
						continue;
					}

					float value;
					if (isInterior2)
					{
						value = StencilDot(pcStencil, idx, normals, dataSlot);
					}
					else
					{
						ReadOffset(tree, n, other);
						f.PcIntegrate(other, off, integral);
						value = StencilDot(integral, 0, normals, dataSlot);
					}

					coarserConstraints[tree.NodeIndex(n)] += value;
				}
			}

			if (d > 0 && d < maxDepth)
			{
				PoissonMultigrid.DownSample(tree, sorted, rp, d, coarserConstraints, 0, coarserConstraints, 0, 1);
			}
		}

		// hasCoarserCoefficients is initialized to true upstream, so this always runs.
		var coefficients = new float[3 * PoissonMultigrid.End(tree, sorted, maxDepth - 1)];
		for (int d = maxDepth - 1; d >= 0; d--)
		{
			int end = PoissonMultigrid.End(tree, sorted, d);
			for (int i = PoissonMultigrid.Begin(tree, sorted, d); i < end; i++)
			{
				int slot = normals.Index(tree.NodeIndex(sorted.TreeNodes[i]));
				if (slot != -1)
				{
					for (int k = 0; k < 3; k++)
					{
						coefficients[3 * i + k] += normals.Value(slot, k);
					}
				}
			}
		}

		// Coarse-to-fine up-sampling of coefficients
		for (int d = 1; d < maxDepth; d++)
		{
			PoissonMultigrid.UpSample(tree, sorted, rp, d, coefficients, 0, coefficients, 0, 3);
		}

		// Compute the contribution from all coarser depths
		for (int d = 1; d <= maxDepth; d++)
		{
			f.Init(d);
			double[][] stencils = f.SetChildParentStencils();
			var neighborKey = new NeighborKey(tree, 1, 1, resetOnMissing: false);
			neighborKey.Set(d - 1 + tree.DepthOffset);
			int end = PoissonMultigrid.End(tree, sorted, d);
			for (int i = PoissonMultigrid.Begin(tree, sorted, d); i < end; i++)
			{
				int node = sorted.TreeNodes[i];
				if (!PoissonMultigrid.IsValidFem1Node(tree, node))
				{
					continue;
				}

				int parent = tree.Parent(node);
				neighborKey.GetNeighbors(-femC.OverlapStart, femC.OverlapEnd, parent, pNeighbors);
				bool isInterior = PoissonMultigrid.IsInteriorlyOverlapped(tree, femDegree, cDegree, parent);
				int corner = tree.ChildIndexInParent(node);
				double[] cpStencil = stencils[corner];
				ReadOffset(tree, node, off);
				float constraint = 0;
				foreach (int idx in femcLoopData[corner])
				{
					int n = pNeighbors[idx];
					if (!PoissonMultigrid.IsValidFem2Node(tree, n))
					{
						continue;
					}

					if (isInterior)
					{
						constraint += StencilDot(cpStencil, idx, coefficients, tree.NodeIndex(n));
					}
					else
					{
						ReadOffset(tree, n, other);
						f.CpIntegrate(off, other, integral);
						constraint += StencilDot(integral, 0, coefficients, tree.NodeIndex(n));
					}
				}

				constraints[i] += constraint;
			}
		}

		int allEnd = PoissonMultigrid.End(tree, sorted, treeMaxDepth);
		for (int i = PoissonMultigrid.Begin(tree, sorted, 0); i < allEnd; i++)
		{
			int node = sorted.TreeNodes[i];
			if (PoissonMultigrid.IsValidFem1Node(tree, node) && (tree.Flags(node) & FemTree.DirichletElementFlag) != 0)
			{
				constraints[i] *= 0f;
			}
		}
	}

	private static int Cube(int x) => x * x * x;

	private static void ReadOffset(FemTree tree, int node, int[] off)
	{
		for (int k = 0; k < 3; k++)
		{
			off[k] = tree.LocalOffset(node, k);
		}
	}

	// _StencilDot< double , float , 3 >( stencil[entry] , normal ).
	private static float StencilDot(double[] stencil, int entry, SparseNodeData normals, int slot)
	{
		float dot = 0;
		for (int k = 0; k < 3; k++)
		{
			dot += normals.Value(slot, k) * (float)stencil[3 * entry + k];
		}

		return dot;
	}

	// The same over the accumulated coefficients (three floats per node index).
	private static float StencilDot(double[] stencil, int entry, float[] coefficients, int nodeIndex)
	{
		float dot = 0;
		for (int k = 0; k < 3; k++)
		{
			dot += coefficients[3 * nodeIndex + k] * (float)stencil[3 * entry + k];
		}

		return dot;
	}

	// _IsZero( Point< float , 3 > ): every component == 0.
	private static bool IsZero(SparseNodeData normals, int slot) =>
		normals.Value(slot, 0) == 0 && normals.Value(slot, 1) == 0 && normals.Value(slot, 2) == 0;
}
