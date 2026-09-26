// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Kazhdan's PoissonRecon as vendored by COLMAP (MIT, see THIRD_PARTY_NOTICES.md).
//
// PoissonSampleSet: the "read in the samples" block of Reconstructor::Poisson::Solver::Solve
// (thirdparty/PoissonRecon/Reconstructors.h) with what it calls -
// PointExtent::GetXForm, SolutionParameters::testAndSet, TransformedInputOrientedSampleStream
// (positions through the model-to-unit-cube transform, normals through its inverse transpose
// scaled by |det|^(1/3)) and FEMTreeInitializer::Initialize (FEMTree.Initialize.inl: descend
// from the unit-cube root to the requested depth, creating nodes, and hand each valid in-cube
// sample to Solve's Process lambda). Process accumulates, per finest node, the weighted
// position (a ProjectiveData), the unit normal and the auxiliary data (colors). The density,
// splatting and solver slices consume these samples. Tier A against
// oracle/poisson_tree_harness.cc.
//
// Input: flat float arrays, three floats per point for positions and normals and
// AuxPerPoint floats per point for the auxiliary data (COLMAP's red, green, blue; empty for
// none). The C++ reads the same values from COLMAP's PLY stream in file order, and the
// accumulation order is the input order, so the sums match.
//
// Cancellation and progress: the token is checked, and progress reported, every 65536
// points (docs/CPP_DIVERGENCES.md, entry 76, for the in-memory API).

namespace ColmapSharp.Mvs.PoissonRecon;

/// <summary>
/// The oriented samples of a point set gathered into the finest octree nodes. Port of the
/// sample-reading stage of PoissonRecon's <c>Poisson::Solver::Solve</c>.
/// </summary>
public sealed class PoissonSampleSet
{
	private const int ProgressInterval = 1 << 16;

	private int[] sampleNode = new int[64];
	private float[] samplePosition = new float[3 * 64];
	private float[] sampleWeight = new float[64];
	private float[] sampleNormal = new float[3 * 64];
	private float[] sampleAux;

	private PoissonSampleSet(FemTree tree, int auxPerPoint)
	{
		Tree = tree;
		AuxPerPoint = auxPerPoint;
		sampleAux = new float[Math.Max(auxPerPoint, 1) * 64];
	}

	/// <summary>The octree the samples were inserted into.</summary>
	public FemTree Tree { get; }

	/// <summary>The number of auxiliary values per point.</summary>
	public int AuxPerPoint { get; }

	/// <summary>The transform from model coordinates into the unit cube.</summary>
	public PoissonXForm ModelToUnitCube { get; private set; } = PoissonXForm.Identity(4);

	/// <summary>Its inverse (Implicit::unitCubeToModel).</summary>
	public PoissonXForm UnitCubeToModel { get; private set; } = PoissonXForm.Identity(4);

	/// <summary>
	/// The normal transform: the inverse transpose of ModelToUnitCube's linear part times
	/// |det|^(1/3) (TransformedInputOrientedSampleStream::_normalXForm).
	/// </summary>
	public PoissonXForm NormalXForm { get; private set; } = PoissonXForm.Identity(3);

	/// <summary>The number of input points that were valid and inside the cube (Solve's pointCount).</summary>
	public long PointCount { get; private set; }

	/// <summary>The number of samples (distinct finest nodes that received points).</summary>
	public int Count { get; private set; }

	/// <summary>The node handle of sample i (FEMTree::PointSample::node).</summary>
	public int Node(int i) => sampleNode[i];

	/// <summary>Component d of sample i's weighted position sum (PointSample::sample.data).</summary>
	public float Position(int i, int d) => samplePosition[3 * i + d];

	/// <summary>Sample i's total weight (PointSample::sample.weight).</summary>
	public float Weight(int i) => sampleWeight[i];

	/// <summary>Component d of sample i's weighted sum of unit normals.</summary>
	public float Normal(int i, int d) => sampleNormal[3 * i + d];

	/// <summary>Auxiliary value k of sample i's weighted sum.</summary>
	public float Aux(int i, int k) => sampleAux[AuxPerPoint * i + k];

	/// <summary>
	/// Computes the unit-cube transform, completes <paramref name="parameters"/> (testAndSet),
	/// builds the octree to parameters.Depth and accumulates the samples. Port of Solve's
	/// sample-reading block.
	/// </summary>
	public static PoissonSampleSet Build(
		ReadOnlySpan<float> positions,
		ReadOnlySpan<float> normals,
		ReadOnlySpan<float> aux,
		int auxPerPoint,
		PoissonSolutionParameters parameters,
		CancellationToken cancellationToken = default,
		IProgress<PoissonProgress>? progress = null)
	{
		int pointTotal = positions.Length / 3;
		if (positions.Length != 3 * pointTotal || normals.Length != positions.Length)
		{
			throw new ArgumentException("positions and normals must hold three floats per point, the same number of points.");
		}

		if (auxPerPoint < 0 || aux.Length != auxPerPoint * pointTotal)
		{
			throw new ArgumentException("aux must hold auxPerPoint floats per point.", nameof(aux));
		}

		var set = new PoissonSampleSet(new FemTree(), auxPerPoint);
		PoissonXForm modelToUnitCube = PoissonXForm.Identity(4);
		if (parameters.Scale > 0)
		{
			modelToUnitCube = PointExtent.GetXForm(positions, parameters.Scale, (int)parameters.AlignDir).Multiply(modelToUnitCube);
		}

		set.ModelToUnitCube = modelToUnitCube;
		set.UnitCubeToModel = modelToUnitCube.Inverse();
		parameters.TestAndSet(set.UnitCubeToModel);

		// TransformedInputOrientedSampleStream's normal transform.
		PoissonXForm normalXForm = modelToUnitCube.UpperLeft().Inverse().Transpose()
			.Multiply((float)PowOneThird.Pow(MathF.Abs(modelToUnitCube.Determinant())));

		set.NormalXForm = normalXForm;
		set.Insert(positions, normals, aux, modelToUnitCube, normalXForm, (int)parameters.Depth, parameters.Confidence, cancellationToken, progress);
		return set;
	}

	private void Insert(
		ReadOnlySpan<float> positions,
		ReadOnlySpan<float> normals,
		ReadOnlySpan<float> aux,
		PoissonXForm positionXForm,
		PoissonXForm normalXForm,
		int maxDepth,
		bool confidence,
		CancellationToken cancellationToken,
		IProgress<PoissonProgress>? progress)
	{
		FemTree tree = Tree;
		int root = tree.SpaceRoot;

		// FEMTreeNode::SubTreeExtractor: treat the unit-cube node as a depth-0 root while inserting.
		using FemTree.SubTreeScope scope = tree.ExtractSubTree(root);

		int pointTotal = positions.Length / 3;
		var nodeToIndexMap = new int[Math.Max(tree.SlotCount, 64)];
		Array.Fill(nodeToIndexMap, -1);
		Span<float> p = stackalloc float[3];
		Span<float> n = stackalloc float[3];
		Span<float> center = stackalloc float[3];
		long pointCount = 0;
		for (int i = 0; i < pointTotal; i++)
		{
			if (i % ProgressInterval == 0)
			{
				cancellationToken.ThrowIfCancellationRequested();
				progress?.Report(new PoissonProgress(PoissonStage.TreeBuild, (double)i / pointTotal));
			}

			positionXForm.TransformPoint(positions.Slice(3 * i, 3), p);
			normalXForm.TransformVector(normals.Slice(3 * i, 3), n);

			// Check if the data is good
			float squareNorm = 0.0f + n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
			if (!(squareNorm > 0 && float.IsFinite(squareNorm)))
			{
				continue;
			}

			// Check that the position is in-range
			int leaf = Leaf(root, p, maxDepth, center);
			if (leaf == FemTree.None)
			{
				continue;
			}

			// Process the data (Solve's Process lambda)
			float l = MathF.Sqrt(squareNorm);
			float weight = confidence ? l : 1.0f;
			for (int d = 0; d < 3; d++)
			{
				n[d] /= l;
			}

			if (leaf >= nodeToIndexMap.Length)
			{
				int old = nodeToIndexMap.Length;
				Array.Resize(ref nodeToIndexMap, Math.Max(leaf + 1, 2 * old));
				Array.Fill(nodeToIndexMap, -1, old, nodeToIndexMap.Length - old);
			}

			int idx = nodeToIndexMap[leaf];
			ReadOnlySpan<float> d0 = aux.Slice(AuxPerPoint * i, AuxPerPoint);
			if (idx == -1)
			{
				idx = Count;
				nodeToIndexMap[leaf] = idx;
				Grow(idx + 1);
				Count = idx + 1;
				sampleNode[idx] = leaf;
				for (int d = 0; d < 3; d++)
				{
					samplePosition[3 * idx + d] = p[d] * weight;
					sampleNormal[3 * idx + d] = n[d] * weight;
				}

				sampleWeight[idx] = weight;
				for (int k = 0; k < AuxPerPoint; k++)
				{
					sampleAux[AuxPerPoint * idx + k] = d0[k] * weight;
				}
			}
			else
			{
				for (int d = 0; d < 3; d++)
				{
					samplePosition[3 * idx + d] += p[d] * weight;
					sampleNormal[3 * idx + d] += n[d] * weight;
				}

				sampleWeight[idx] += weight;
				for (int k = 0; k < AuxPerPoint; k++)
				{
					sampleAux[AuxPerPoint * idx + k] += d0[k] * weight;
				}
			}

			pointCount++;
		}

		PointCount = pointCount;
		progress?.Report(new PoissonProgress(PoissonStage.TreeBuild, 1.0));
	}

	// Port of Initialize's Leaf lambda: the depth-maxDepth node containing p, creating nodes
	// on the way; None when p is outside [0,1]^3.
	private int Leaf(int root, ReadOnlySpan<float> p, int maxDepth, Span<float> center)
	{
		for (int d = 0; d < 3; d++)
		{
			if (p[d] < 0 || p[d] > 1)
			{
				return FemTree.None;
			}
		}

		FemTree tree = Tree;
		tree.CenterAndWidth(root, center, out float width);
		int depth = tree.Depth(root);
		int node = root;
		while (depth < maxDepth)
		{
			if (!tree.HasChildren(node))
			{
				tree.InitChildren(node);
			}

			int cIndex = FemTree.ChildIndex(center, p);
			node = tree.FirstChild(node) + cIndex;
			width /= 2;
			depth++;
			for (int dd = 0; dd < 3; dd++)
			{
				if (((cIndex >> dd) & 1) != 0)
				{
					center[dd] += width / 2;
				}
				else
				{
					center[dd] -= width / 2;
				}
			}
		}

		return node;
	}

	private void Grow(int count)
	{
		if (count <= sampleNode.Length)
		{
			return;
		}

		int capacity = Math.Max(count, 2 * sampleNode.Length);
		Array.Resize(ref sampleNode, capacity);
		Array.Resize(ref samplePosition, 3 * capacity);
		Array.Resize(ref sampleWeight, capacity);
		Array.Resize(ref sampleNormal, 3 * capacity);
		Array.Resize(ref sampleAux, Math.Max(AuxPerPoint, 1) * capacity);
	}
}
