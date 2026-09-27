// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonLevelSetExtractor.IsoVertices: the iso-vertices on the edges that lie in a slice (the
// x and y edges of the leaves' back or front faces). Ports FEMTree.LevelSet.3D.inl's
// _LevelSetExtractor< ... , 3 , ... >::SetSliceIsoVertices (both forms), the slice-edge form of
// GetIsoVertex, Extract's SetSliceIso vertex loop and FinalizeSlice's edge part, and
// FEMTree.Evaluation.inl's _addEvaluation / _accumulate for the degree-0 color field. Each
// vertex goes to the sink (Vertices) in the order upstream's vertexStream.write sees it; its
// edge key is recorded on the slice and, when the edge borders a coarser leaf, pushed to the
// coarser slices (or, across an odd slice, to the coarser slab) that share it. The cross-slice
// (slab) edges' vertices are in PoissonLevelSetExtractor.XSliceIsoVertices.cs; the Hermite
// root (AverageRoot), the push-down test (IsNeeded) and the color evaluation (DataAt) here are
// shared by both GetIsoVertex forms. Tier A against oracle/poisson_levelset3_harness.cc and,
// interleaved with the slab vertices as Extract runs them, oracle/poisson_levelset4_harness.cc.
//
// Translation notes:
// - COLMAP runs Extract with nonLinearFit on (it never passes --linearFit) and gradientNormals
//   off (no --gradients), so the vertex gradient is always zero and the root comes from the
//   Hermite quadratic, falling back to the linear root as upstream does.
// - The float/double mix of GetIsoVertex is kept: corner gradients times width are float
//   products widened to double, 3*(x1-x0) is a float, the linear root is a float quotient.
// - Upstream counts clamped roots in a static atomic (_BadRootCount) that Extract only logs;
//   here it is BadRootCount on the extractor.
// - Upstream sets the vertices in a ThreadPool::ParallelFor over the slice's leaves and numbers
//   them by an atomic counter; the port runs the leaves in order, which is what a
//   single-threaded run does (docs/CPP_DIVERGENCES.md, entry 123). No hash-map order reaches
//   the numbering: the vertex order is the leaf and edge loop order.

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// An iso-surface vertex: position in the unit cube, gradient, density-derived depth and
/// auxiliary data (colors). Port of <c>_LevelSetExtractor::Vertex</c>.
/// </summary>
public readonly record struct LevelSetVertex(float X, float Y, float Z, float GradientX, float GradientY, float GradientZ, float Depth, float[] Data);

public sealed partial class PoissonLevelSetExtractor
{
	// Reconstructor::WeightDegree and FEMDegreeAndBType< Reconstructor::DataDegree , BOUNDARY_FREE >.
	private const int WeightDegree = 2;
	private static readonly int DataSignature = FemSignature.Of(0, BoundaryType.Free);

	private readonly DensityEstimator? density;
	private readonly SparseNodeData? data;
	private readonly float[] zeroData;
	private readonly PoissonPointEvaluator? dataEvaluator;
	private readonly float[] vertexPosition = new float[3];
	private readonly float[] cellStart = new float[3];
	private readonly double[] roots = new double[2];

	/// <summary>The edge and face key generator at the finest depth. Port of Extract's <c>keyGenerator</c>.</summary>
	public LevelSetKeyGenerator KeyGenerator { get; }

	/// <summary>The iso-vertices in output order (a vertex's index is its position here). Port of the vertex stream.</summary>
	public List<LevelSetVertex> Vertices { get; } = [];

	/// <summary>How many roots fell outside (0, 1) and were clamped. Port of <c>_BadRootCount</c>.</summary>
	public long BadRootCount { get; private set; }

	/// <summary>
	/// Sets the iso-vertices on the slice edges of the slices on the plane
	/// <paramref name="sliceAtMaxDepth"/>, finest depth first. Port of the vertex loop of
	/// Extract's <c>SetSliceIso</c> (without a boundary).
	/// </summary>
	public void SetSliceIsoVertices(int sliceAtMaxDepth)
	{
		for (int d = MaxDepth, o = sliceAtMaxDepth; d >= FullDepth; d--, o >>= 1)
		{
			SetSliceIsoVertices(d, o);
			if ((o & 1) != 0)
			{
				break;
			}
		}
	}

	/// <summary>
	/// Moves each slice's recorded edge keys on the plane <paramref name="sliceAtMaxDepth"/> into
	/// its edge-vertex map. Port of the edge part of Extract's <c>FinalizeSlice</c>.
	/// </summary>
	public void FinalizeSliceEdges(int sliceAtMaxDepth)
	{
		for (int d = MaxDepth, o = sliceAtMaxDepth; d >= FullDepth; d--, o >>= 1)
		{
			SlabValues[d].SliceValues(o).SetEdgesFromScratch();
			if ((o & 1) != 0)
			{
				break;
			}
		}
	}

	/// <summary>
	/// The iso-vertices on the edges of slice <paramref name="slice"/> at <paramref name="depth"/>,
	/// from the leaves behind it and then those in front. Port of
	/// <c>SetSliceIsoVertices( ... , depth , fullDepth , slice , ... )</c>.
	/// </summary>
	public void SetSliceIsoVertices(int depth, int slice)
	{
		if (slice > 0)
		{
			SetSliceIsoVertices(depth, slice, HyperCubeDirection.Front);
		}

		if (slice < (1 << depth))
		{
			SetSliceIsoVertices(depth, slice, HyperCubeDirection.Back);
		}
	}

	private void SetSliceIsoVertices(int depth, int slice, HyperCubeDirection zDir)
	{
		LevelSetSliceValues sValues = SlabValues[depth].SliceValues(slice);
		int globalDepth = depth + tree.DepthOffset;
		BSplineSupportSizes weightSupport = BSplineSupportSizes.For(WeightDegree);

		// Const keys, fresh per call as upstream's per-thread keys are.
		var neighborKey = new NeighborKey(tree, 1, 1, resetOnMissing: false);
		var weightKey = new NeighborKey(tree, weightSupport.SupportEnd, -weightSupport.SupportStart, resetOnMissing: false);
		var dataKey = new NeighborKey(tree, 0, 0, resetOnMissing: false);
		neighborKey.Set(globalDepth);
		weightKey.Set(globalDepth);
		dataKey.Set(globalDepth);

		int nodeSlice = slice - (zDir == HyperCubeDirection.Back ? 0 : 1) + tree.LocalInset(depth);
		int end = sorted.End(globalDepth, nodeSlice);
		HyperCubeOverlapTable edgeFaces = HyperCubeTables.Of(3, 1, 2);
		HyperCubeOverlapTable faceCorners = HyperCubeTables.Of(3, 2, 0);
		for (int i = sorted.Begin(globalDepth, nodeSlice); i < end; i++)
		{
			int leaf = sorted.TreeNodes[i];
			if (!PoissonMultigrid.IsValidSpaceNode(tree, leaf) || tree.IsActive(tree.FirstChild(leaf)))
			{
				continue;
			}

			int mcIndex = sValues.McIndices[i - sValues.CellIndices.NodeOffset];
			if (!HyperCube.HasMcRoots(2, mcIndex))
			{
				continue;
			}

			int leafIndex = tree.NodeIndex(leaf);
			neighborKey.GetNeighbors(leaf);
			if (density != null)
			{
				weightKey.GetNeighbors(leaf);
			}

			if (data != null)
			{
				dataKey.GetNeighbors(leaf);
			}

			for (int sliceEdge = 0; sliceEdge < HyperCube.ElementNum(2, 1); sliceEdge++)
			{
				if (!HyperCube.HasMcRoots(1, HyperCube.ElementMcIndex(2, 1, sliceEdge, mcIndex)))
				{
					continue;
				}

				int e = HyperCube.Element(3, 1, zDir, sliceEdge);
				int vIndex = sValues.CellIndices.Index(1, leafIndex, sliceEdge);
				if (sValues.EdgeSet[vIndex] != 0)
				{
					continue;
				}

				LevelSetKey key = KeyGenerator.Key(1, depth, tree.LocalOffset(leaf, 0), tree.LocalOffset(leaf, 1), tree.LocalOffset(leaf, 2), e);
				LevelSetVertex vertex = GetIsoVertex(weightKey, dataKey, leaf, leafIndex, sliceEdge, zDir, sValues);
				sValues.EdgeSet[vIndex] = 1;
				int vertexIndex = Vertices.Count;
				Vertices.Add(vertex);
				sValues.EdgeKeys[vIndex] = key;
				sValues.EdgeKeyValues.Add((key, vertexIndex));

				// We only need to pass the iso-vertex down if the edge it lies on is adjacent to a coarser leaf
				if (!IsNeeded(neighborKey, e, depth))
				{
					continue;
				}

				int[] faces = edgeFaces.OverlapElements[e];
				for (int k = 0; k < edgeFaces.OverlapElementNum; k++)
				{
					int node = leaf;
					int nodeDepth = depth;
					int nodeSliceIndex = slice;
					bool cross = false;
					while (PoissonMultigrid.IsValidSpaceNode(tree, tree.Parent(node)) && faceCorners.Overlap[faces[k]][tree.ChildIndexInParent(node)])
					{
						if ((nodeSliceIndex & 1) != 0)
						{
							cross = true;
						}

						node = tree.Parent(node);
						nodeDepth--;
						nodeSliceIndex >>= 1;
						if (nodeDepth >= FullDepth && !cross)
						{
							LevelSetSliceValues coarse = SlabValues[nodeDepth].SliceValues(nodeSliceIndex);
							coarse.EdgeSet[coarse.CellIndices.Index(1, tree.NodeIndex(node), sliceEdge)] = 1;
						}

						if (cross)
						{
							SlabValues[nodeDepth].SlabEdgeKeyValues(nodeSliceIndex).Add((key, vertexIndex));
						}
						else
						{
							SlabValues[nodeDepth].SliceValues(nodeSliceIndex).EdgeKeyValues.Add((key, vertexIndex));
						}

						if (!IsNeeded(neighborKey, e, nodeDepth))
						{
							break;
						}
					}
				}
			}
		}
	}

	// SetSliceIsoVertices' and SetXSliceIsoVertices' IsNeeded: whether edge e of the leaf is
	// adjacent to a coarser leaf at depth d, i.e. whether one of the other cubes around the
	// edge is missing from the leaf's one-ring there.
	private bool IsNeeded(NeighborKey neighborKey, int e, int d)
	{
		HyperCubeElementTable edges3 = HyperCubeTables.Of(3, 1);
		bool isNeeded = false;
		int myIncidentCube = edges3.IncidentCube[e];
		int[] window = neighborKey.Window(d + tree.DepthOffset);
		for (int ic = 0; ic < edges3.IncidentCubeNum; ic++)
		{
			if (ic != myIncidentCube)
			{
				isNeeded |= !PoissonMultigrid.IsValidSpaceNode(tree, window[edges3.CellOffset[e][ic]]);
			}
		}

		return isNeeded;
	}

	// GetIsoVertex( ... , node , _e , zDir , sValues , vertex , zeroData ) with nonLinearFit on
	// and gradientNormals off.
	private LevelSetVertex GetIsoVertex(NeighborKey weightKey, NeighborKey dataKey, int node, int nodeIndex, int sliceEdge, HyperCubeDirection zDir, LevelSetSliceValues sValues)
	{
		int[] corners = HyperCubeTables.Of(2, 1, 0).OverlapElements[sliceEdge];
		int i0 = sValues.CellIndices.Index(0, nodeIndex, corners[0]);
		int i1 = sValues.CellIndices.Index(0, nodeIndex, corners[1]);
		float x0 = sValues.CornerValues[i0];
		float x1 = sValues.CornerValues[i1];
		Span<float> s = cellStart;
		Span<float> position = vertexPosition;
		tree.LocalStartAndWidth(node, s, out float width);
		int o = 0;
		float start = 0;
		HyperCubeDirection[] dirs = HyperCubeTables.Of(2, 1).Directions[sliceEdge];
		for (int d = 0; d < 2; d++)
		{
			if (dirs[d] == HyperCubeDirection.Cross)
			{
				o = d;
				start = s[d];
				int other = (d + 1) % 2;
				position[other] = s[other] + (width * (dirs[other] == HyperCubeDirection.Back ? 0 : 1));
			}
		}

		position[2] = s[2] + (width * (zDir == HyperCubeDirection.Back ? 0 : 1));

		// Float products, widened.
		double averageRoot = AverageRoot(x0, x1, sValues.CornerGradients![(3 * i0) + o] * width, sValues.CornerGradients[(3 * i1) + o] * width);
		position[o] = (float)(start + (width * averageRoot));

		// gradientNormals is off, so both corner gradients are zero and so is the blend.
		float gradient = (0f * (float)(1.0 - averageRoot)) + (0f * (float)averageRoot);
		float depth = 1f;
		if (density != null)
		{
			depth = PoissonSplat.GetSampleDepthAndWeight(tree, density, node, position, weightKey).Depth;
		}

		float[] dataValue = data == null ? [] : DataAt(dataKey, s[0] + (width / 2), s[1] + (width / 2), s[2] + (width / 2));
		return new LevelSetVertex(position[0], position[1], position[2], gradient, gradient, gradient, depth, dataValue);
	}

	// Both GetIsoVertex forms' root: the Hermite quadratic through the edge's end values x0, x1
	// with end derivatives dx0, dx1 (the corner gradients times the edge length), its roots in
	// [0, 1] averaged, falling back to the linear root, then clamped to [0, 1] (counted as bad
	// when it lands on or outside an end).
	private double AverageRoot(float x0, float x1, double dx0, double dx1)
	{
		double averageRoot;
		bool rootFound = false;
		{
			// The scaling will turn the Hermite Spline into a quadratic
			double scl = (x1 - x0) / ((dx1 + dx0) / 2);
			dx0 *= scl;
			dx1 *= scl;

			// Hermite Spline
			var p = new PoissonPolynomial(2);
			p.Coefficients[0] = x0;
			p.Coefficients[1] = dx0;
			p.Coefficients[2] = (3 * (x1 - x0)) - dx1 - (2 * dx0);
			int rCount = 0;
			int rootCount = p.GetSolutions(isoValue, roots, 0);
			averageRoot = 0;
			for (int i = 0; i < rootCount; i++)
			{
				if (roots[i] >= 0 && roots[i] <= 1)
				{
					averageRoot += roots[i];
					rCount++;
				}
			}

			if (rCount != 0)
			{
				rootFound = true;
			}

			averageRoot /= rCount;
		}

		if (!rootFound)
		{
			// We have a linear function L, with L(0) = x0 and L(1) = x1
			// => L(t) = x0 + t * (x1-x0)
			// => L(t) = isoValue <=> t = ( isoValue - x0 ) / ( x1 - x0 )
			if (x0 == x1)
			{
				throw new InvalidOperationException($"Not a zero-crossing root: {x0} {x1}");
			}

			averageRoot = (isoValue - x0) / (x1 - x0);
		}

		if (averageRoot <= 0 || averageRoot >= 1)
		{
			BadRootCount++;
			if (averageRoot < 0)
			{
				averageRoot = 0;
			}

			if (averageRoot > 1)
			{
				averageRoot = 1;
			}
		}

		return averageRoot;
	}

	// GetIsoVertex's DataDegree == 0 branch: _addEvaluation of the data at the cell center
	// (cx, cy, cz) into ProjectiveData( zeroData ), then its value, or zeroData if it has no weight.
	private float[] DataAt(NeighborKey dataKey, float cx, float cy, float cz)
	{
		int width1 = data!.Width;
		int channels = width1 - 1;
		Span<float> center = [cx, cy, cz];
		Span<float> sum = stackalloc float[width1];
		for (int k = 0; k < channels; k++)
		{
			sum[k] = zeroData[k];
		}

		sum[channels] = 0;

		// _accumulate: every depth from the unit cube down to the key's depth.
		int keyDepth = dataKey.KeyDepth;
		for (int d = tree.DepthOffset; d <= keyDepth; d++)
		{
			int node = dataKey.Window(d)[0];
			if (node == FemTree.None)
			{
				throw new InvalidOperationException($"Point is not centered on a node: ({center[0]}, {center[1]}, {center[2]}) 0 @ {d}");
			}

			int localDepth = tree.LocalDepth(node);
			int res = 1 << localDepth;
			dataEvaluator!.Init(localDepth, center[0], center[1], center[2], (int)((double)center[0] * res), (int)((double)center[1] * res), (int)((double)center[2] * res));
			if (!PoissonMultigrid.IsValidFem1Node(tree, node))
			{
				continue;
			}

			int slot = data.Index(tree.NodeIndex(node));
			if (slot == -1)
			{
				continue;
			}

			float scale = (float)dataEvaluator.Value(tree.LocalOffset(node, 0), tree.LocalOffset(node, 1), tree.LocalOffset(node, 2));
			for (int k = 0; k < width1; k++)
			{
				sum[k] += data.Value(slot, k) * scale;
			}
		}

		var value = new float[channels];
		float weight = sum[channels];
		for (int k = 0; k < channels; k++)
		{
			value[k] = weight != 0 ? sum[k] / weight : zeroData[k];
		}

		return value;
	}
}
