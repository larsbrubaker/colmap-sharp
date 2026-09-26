// poisson_system_harness.cc: the system-assembly steps of PoissonRecon's
// Poisson::Solver::Solve (thirdparty/PoissonRecon/Reconstructors.h, MIT, as vendored by
// COLMAP 4.2.0) after finalizeForMultigrid: addFEMConstraints (the divergence of the normal
// field, FEMTree.System.inl's _addFEMConstraints) and addInterpolationConstraints
// (_addInterpolationConstraints), then the solver's per-depth matrix rows and prolongation
// constraints (_getSliceMatrixAndProlongationConstraints, _getProlongedMatrixRowSize). Built and run by
// oracle/fixture_poisson_tree.py, which writes ColmapSharp.Tests/TestData/oracle/poisson_system.json
// (read by ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.System.cs). Not part of any
// build. The shared set-up and output format are in oracle/poisson_harness.h; the stages up to
// finalizeForMultigrid are checked step by step by oracle/poisson_tree_harness.cc, so here they
// run through the vendored functions and print nothing.

#include "poisson_harness.h"

namespace {

typedef IsotropicUIntPack<Dim, FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature> Sigs;
typedef IsotropicUIntPack<Dim, FEMDegreeAndBType<Reconstructor::Poisson::NormalDegree, DerivativeBoundary<BOUNDARY_NEUMANN, 1>::BType>::Signature> NormalSigs;
typedef DirectSum<Real, Point<Real, Dim>, DirectSum<Real, Color>> InternalNormalAndAuxData;
typedef DirectSum<Real, Color> InternalAuxData;

// Solve's steps up to and including finalizeForMultigrid, with COLMAP's options (depth,
// fullDepth = depth when depth < 5, pointWeight 1, everything else default), as
// oracle/poisson_tree_harness.cc's kFinal runs execute them.
void Run(const std::string& name, int depth, const std::vector<Sample>& input) {
  {
    std::vector<double> flat;
    for (const Sample& s : input)
      for (int k = 0; k < 3; k++) flat.push_back(s.p[k]);
    for (const Sample& s : input)
      for (int k = 0; k < 3; k++) flat.push_back(s.n[k]);
    for (const Sample& s : input)
      for (int k = 0; k < 3; k++) flat.push_back(s.c[k]);
    PrintF(name + "/input", flat);
  }

  Reconstructor::Poisson::SolutionParameters<Real> params;
  params.depth = depth;
  params.fullDepth = depth < 5 ? depth : 5;
  params.baseDepth = (unsigned int)-1, params.solveDepth = (unsigned int)-1, params.kernelDepth = (unsigned int)-1;
  params.alignDir = Dim - 1;
  params.pointWeight = 1.f;

  FEMTree<Dim, Real> tree(MEMORY_ALLOCATOR_BLOCK_SIZE);
  VectorStream pointStream(input);
  std::vector<typename FEMTree<Dim, Real>::PointSample> samples;
  std::vector<InternalNormalAndAuxData> sampleNormalAndAuxData;
  XForm<Real, Dim + 1> modelToUnitCube = XForm<Real, Dim + 1>::Identity();
  pointStream.reset();
  modelToUnitCube = PointExtent::GetXForm<Real, Dim, true, Point<Real, Dim>, Color>(pointStream, Point<Real, Dim>(), Color(), params.scale, params.alignDir) * modelToUnitCube;
  pointStream.reset();
  params.template testAndSet<Dim>(modelToUnitCube.inverse());
  {
    Reconstructor::TransformedInputOrientedSampleStream<Real, Dim, Color> _pointStream(modelToUnitCube, pointStream);
    std::vector<node_index_type> nodeToIndexMap;
    auto IsValid = [&](const Point<Real, Dim>& p, const Point<Real, Dim>& n, Color d) {
      Real l = Point<Real, Dim>::SquareNorm(n);
      return l > 0 && std::isfinite(l);
    };
    auto Process = [&](FEMTreeNode& node, const Point<Real, Dim>& p, Point<Real, Dim>& n, Color d) {
      Real l = (Real)Length(n);
      Real weight = (Real)1.;
      n /= l;
      node_index_type nodeIndex = node.nodeData.nodeIndex;
      if (nodeIndex >= (node_index_type)nodeToIndexMap.size()) nodeToIndexMap.resize(nodeIndex + 1, -1);
      node_index_type idx = nodeToIndexMap[nodeIndex];
      if (idx == -1) {
        idx = (node_index_type)samples.size();
        nodeToIndexMap[nodeIndex] = idx;
        samples.resize(idx + 1), samples[idx].node = &node;
        sampleNormalAndAuxData.resize(idx + 1);
        samples[idx].sample = ProjectiveData<Point<Real, Dim>, Real>(p * weight, weight);
        sampleNormalAndAuxData[idx] = InternalNormalAndAuxData(n, DirectSum<Real, Color>(d)) * weight;
      } else {
        samples[idx].sample += ProjectiveData<Point<Real, Dim>, Real>(p * weight, weight);
        sampleNormalAndAuxData[idx] += InternalNormalAndAuxData(n, DirectSum<Real, Color>(d)) * weight;
      }
      return true;
    };
    FEMTreeInitializer<Dim, Real>::template Initialize<decltype(IsValid), decltype(Process), Point<Real, Dim>, Color>(tree.spaceRoot(), _pointStream, Point<Real, Dim>(), Color(), params.depth, tree.nodeAllocators.size() ? tree.nodeAllocators[0] : nullptr, tree.initializer(), IsValid, Process);
  }
  tree.resetNodeIndices(0, std::make_tuple());

  auto* estimator = tree.template setDensityEstimator<1, Reconstructor::WeightDegree>(samples, params.kernelDepth, params.samplesPerNode);
  ProjectiveData<Point<Real, 2>, Real> pointDepthAndWeight;
  std::function<bool(InternalNormalAndAuxData, Point<Real, Dim>&)> ConversionFunction = [](InternalNormalAndAuxData in, Point<Real, Dim>& out) {
    Point<Real, Dim> n = in.template get<0>();
    Real l = (Real)Length(n);
    if (!l) return false;
    out = n / l;
    return true;
  };
  auto* normalInfo = new SparseNodeData<Point<Real, Dim>, NormalSigs>();
  *normalInfo = tree.setInterpolatedDataField(Point<Real, Dim>(), NormalSigs(), samples, sampleNormalAndAuxData, estimator, params.baseDepth, params.depth, params.lowDepthCutOff, pointDepthAndWeight, ConversionFunction);
  for (size_t i = 0; i < normalInfo->size(); i++) (*normalInfo)[i] *= (Real)-1.;
  static const unsigned int DataSig = FEMDegreeAndBType<Reconstructor::DataDegree, BOUNDARY_FREE>::Signature;
  auto PointSampleFunctor = [&](size_t i) -> const typename FEMTree<Dim, Real>::PointSample& { return samples[i]; };
  auto AuxDataSampleFunctor = [&](size_t i) -> const InternalAuxData& { return sampleNormalAndAuxData[i].template get<1>(); };
  auto auxData = tree.template setExtrapolatedDataField<DataSig, false, Reconstructor::WeightDegree, InternalAuxData>(InternalAuxData(Color()), samples.size(), PointSampleFunctor, AuxDataSampleFunctor, (typename FEMTree<Dim, Real>::template DensityEstimator<Reconstructor::WeightDegree>*)nullptr);
  typedef typename FEMTree<Dim, Real>::template InterpolationInfo<Real, 0> InterpolationInfo;
  InterpolationInfo* iInfo = FEMTree<Dim, Real>::template InitializeApproximatePointInterpolationInfo<Real, 0>(tree, samples, Reconstructor::Poisson::ConstraintDual<Dim, Real>((Real)0.5, params.pointWeight * pointDepthAndWeight.value()[1]), Reconstructor::Poisson::SystemDual<Dim, Real>(params.pointWeight * pointDepthAndWeight.value()[1]), true, params.depth, 1);
  {
    typename FEMTree<Dim, Real>::template HasNormalDataFunctor<NormalSigs> hasNormalDataFunctor(*normalInfo);
    auto hasDataFunctor = [&](const FEMTreeNode* node) { return hasNormalDataFunctor(node); };
    auto addNodeFunctor = [&](int d, const int off[Dim]) { return d <= (int)params.fullDepth; };
    tree.template finalizeForMultigrid<2, 1>(params.baseDepth, addNodeFunctor, hasDataFunctor, std::make_tuple(iInfo), std::make_tuple(normalInfo, estimator, &auxData));
  }
  PrintI(name + "/sortedslices", [&] {
    std::vector<long long> s;
    for (int d = 0; d < tree._sNodes.levels(); d++) s.push_back(tree._sNodes.begin(d)), s.push_back(tree._sNodes.end(d));
    return s;
  }());

  // Solve's Poisson constraints: addFEMConstraints( F , *normalInfo , constraints , solveDepth )
  // with the divergence integrator; also with a shallower maxDepth (coarser levels only), which
  // Solve does not use but which takes _addFEMConstraints' early-depth paths.
  const int solveDepth = params.depth;
  typename FEMIntegrator::template Constraint<Sigs, IsotropicUIntPack<Dim, 1>, NormalSigs, IsotropicUIntPack<Dim, 0>, Dim> F;
  unsigned int derivatives2[Dim] = {0, 0, 0};
  for (int d = 0; d < (int)Dim; d++) {
    unsigned int derivatives1[Dim];
    for (int dd = 0; dd < (int)Dim; dd++) derivatives1[dd] = dd == d ? 1 : 0;
    F.weights[d][TensorDerivatives<IsotropicUIntPack<Dim, 1>>::Index(derivatives1)][TensorDerivatives<IsotropicUIntPack<Dim, 0>>::Index(derivatives2)] = 1;
  }
  auto dump = [&](const std::string& caseName, const DenseNodeData<Real, Sigs>& c) {
    std::vector<double> out;
    for (size_t i = 0; i < c.size(); i++) out.push_back(c[i]);
    PrintF(name + "/" + caseName, out);
  };
  {
    DenseNodeData<Real, Sigs> shallow = tree.initDenseNodeData(Sigs());
    tree.addFEMConstraints(F, *normalInfo, shallow, solveDepth - 2);
    dump("femconstraintsshallow", shallow);
  }
  DenseNodeData<Real, Sigs> constraints = tree.initDenseNodeData(Sigs());
  tree.addFEMConstraints(F, *normalInfo, constraints, solveDepth);
  dump("femconstraints", constraints);

  // Solve's point constraints (pointWeight > 0): addInterpolationConstraints( constraints ,
  // solveDepth , iInfo ), on top of the FEM constraints.
  tree.addInterpolationConstraints(constraints, solveDepth, std::make_tuple(iInfo));
  dump("interpolationconstraints", constraints);

  // The solver's per-depth assembly, as _solveSystemGS calls it (whole depth, F initialized at
  // the depth with setStencil<false> and setStencils<true>, bsData( solveDepth )), with a
  // synthetic prolonged solution so the prolongation constraints are not trivially zero. Only
  // depths of at most kSliceLimit nodes are dumped, to keep the fixture small.
  {
    const size_t kSliceLimit = 12000;
    typename FEMIntegrator::template System<Sigs, IsotropicUIntPack<Dim, 1>> S({0., 1.});
    typename FEMIntegrator::template PointEvaluator<Sigs, IsotropicUIntPack<Dim, 1>> bsData(solveDepth);
    std::vector<Real> prolonged(tree._sNodesEnd(tree._maxDepth - 1));
    for (size_t i = 0; i < prolonged.size(); i++) prolonged[i] = (Real)((long long)(i * 37 % 101) - 50) / (Real)64;
    std::vector<long long> depths, rowSizes, columns, prolongedRowSizes;
    std::vector<double> values, sliceConstraints, diagonal;
    for (int d = 1; d <= tree._maxDepth; d++) {
      node_index_type begin = tree._sNodesBegin(d), end = tree._sNodesEnd(d);
      if ((size_t)(end - begin) > kSliceLimit) continue;
      depths.push_back(d);
      S.init(d);
      typename FEMTree<Dim, Real>::template CCStencil<IsotropicUIntPack<Dim, 1>> cc;
      typename FEMTree<Dim, Real>::template PCStencils<IsotropicUIntPack<Dim, 1>> pc;
      S.template setStencil<false>(cc);
      S.template setStencils<true>(pc);
      typename FEMTree<Dim, Real>::template SystemMatrixType<5, 5, 5> M;
      std::vector<Real> diag(end - begin), cons(end - begin);
      tree._getSliceMatrixAndProlongationConstraints(Sigs(), S, M, &diag[0], bsData, d, begin, end, &prolonged[0], &cons[0], cc, pc, std::make_tuple(iInfo));
      for (node_index_type i = 0; i < end - begin; i++) {
        rowSizes.push_back(M.rowSize(i));
        for (size_t j = 0; j < M.rowSize(i); j++) columns.push_back(M[i][j].N), values.push_back(M[i][j].Value);
        sliceConstraints.push_back(cons[i]);
        diagonal.push_back(tree._isValidFEM1Node(tree._sNodes.treeNodes[i + begin]) ? diag[i] : 0.);
      }
      // _getProlongedMatrixRowSize of every valid node, with its parent's one-ring window.
      typename FEMTree<Dim, Real>::ConstOneRingNeighborKey key;
      key.set(tree._localToGlobal(d));
      for (node_index_type i = begin; i < end; i++) {
        const FEMTreeNode* node = tree._sNodes.treeNodes[i];
        if (!tree._isValidFEM1Node(node)) continue;
        typename FEMTreeNode::template ConstNeighbors<IsotropicUIntPack<Dim, 3>> pNeighbors;
        key.getNeighbors(IsotropicUIntPack<Dim, 1>(), IsotropicUIntPack<Dim, 1>(), node->parent, pNeighbors);
        prolongedRowSizes.push_back(tree.template _getProlongedMatrixRowSize<5, 5, 5>(node, pNeighbors));
      }
    }
    PrintI(name + "/slicedepths", depths);
    PrintI(name + "/slicerowsizes", rowSizes);
    PrintI(name + "/slicecolumns", columns);
    PrintF(name + "/slicevalues", values);
    PrintF(name + "/sliceconstraints", sliceConstraints);
    PrintF(name + "/slicediagonal", diagonal);
    PrintI(name + "/prolongedrowsizes", prolongedRowSizes);
  }

  delete normalInfo;
  delete iInfo;
  delete estimator;
}

}  // namespace

int main() {
  ThreadPool::ParallelizationType = ThreadPool::NONE;
  Run("system3", 3, MakeInput(100));
  Run("system5", 5, MakeInput(300));
  Run("system6", 6, MakeInput(600));
  Run("system8", 8, MakeInput(500));
  return 0;
}
