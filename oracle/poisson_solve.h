// poisson_solve.h: Poisson::Solver::Solve's steps up to and including finalizeForMultigrid
// (thirdparty/PoissonRecon/Reconstructors.h, MIT, as vendored by COLMAP 4.2.0) with COLMAP's
// options (depth, fullDepth = depth when depth < 5, pointWeight 1, everything else default),
// as oracle/poisson_tree_harness.cc's kFinal runs execute them, and Solve's constraints and
// linear solve. Shared by oracle/poisson_system_harness.cc (which dumps the system assembly
// and the solve) and oracle/poisson_levelset_harness.cc (which dumps what follows the solve).
// Not part of any build.

#pragma once

#include "poisson_harness.h"

namespace {

typedef IsotropicUIntPack<Dim, FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature> Sigs;
typedef IsotropicUIntPack<Dim, FEMDegreeAndBType<Reconstructor::Poisson::NormalDegree, DerivativeBoundary<BOUNDARY_NEUMANN, 1>::BType>::Signature> NormalSigs;
typedef DirectSum<Real, Point<Real, Dim>, DirectSum<Real, Color>> InternalNormalAndAuxData;
typedef DirectSum<Real, Color> InternalAuxData;
static const unsigned int DataSig = FEMDegreeAndBType<Reconstructor::DataDegree, BOUNDARY_FREE>::Signature;
typedef typename FEMTree<Dim, Real>::template InterpolationInfo<Real, 0> InterpolationInfo;

struct PoissonRun {
  Reconstructor::Poisson::SolutionParameters<Real> params;
  FEMTree<Dim, Real> tree;
  std::vector<typename FEMTree<Dim, Real>::PointSample> samples;
  std::vector<InternalNormalAndAuxData> sampleNormalAndAuxData;
  typename FEMTree<Dim, Real>::template DensityEstimator<Reconstructor::WeightDegree>* estimator = nullptr;
  SparseNodeData<Point<Real, Dim>, NormalSigs>* normalInfo = nullptr;
  SparseNodeData<ProjectiveData<InternalAuxData, Real>, IsotropicUIntPack<Dim, DataSig>> auxData;
  InterpolationInfo* iInfo = nullptr;

  PoissonRun() : tree(MEMORY_ALLOCATOR_BLOCK_SIZE) {}
  ~PoissonRun() {
    delete normalInfo;
    delete iInfo;
    delete estimator;
  }

  void Prepare(int depth, const std::vector<Sample>& input) {
    params.depth = depth;
    params.fullDepth = depth < 5 ? depth : 5;
    params.baseDepth = (unsigned int)-1, params.solveDepth = (unsigned int)-1, params.kernelDepth = (unsigned int)-1;
    params.alignDir = Dim - 1;
    params.pointWeight = 1.f;

    VectorStream pointStream(input);
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

    estimator = tree.template setDensityEstimator<1, Reconstructor::WeightDegree>(samples, params.kernelDepth, params.samplesPerNode);
    ProjectiveData<Point<Real, 2>, Real> pointDepthAndWeight;
    std::function<bool(InternalNormalAndAuxData, Point<Real, Dim>&)> ConversionFunction = [](InternalNormalAndAuxData in, Point<Real, Dim>& out) {
      Point<Real, Dim> n = in.template get<0>();
      Real l = (Real)Length(n);
      if (!l) return false;
      out = n / l;
      return true;
    };
    normalInfo = new SparseNodeData<Point<Real, Dim>, NormalSigs>();
    *normalInfo = tree.setInterpolatedDataField(Point<Real, Dim>(), NormalSigs(), samples, sampleNormalAndAuxData, estimator, params.baseDepth, params.depth, params.lowDepthCutOff, pointDepthAndWeight, ConversionFunction);
    for (size_t i = 0; i < normalInfo->size(); i++) (*normalInfo)[i] *= (Real)-1.;
    auto PointSampleFunctor = [&](size_t i) -> const typename FEMTree<Dim, Real>::PointSample& { return samples[i]; };
    auto AuxDataSampleFunctor = [&](size_t i) -> const InternalAuxData& { return sampleNormalAndAuxData[i].template get<1>(); };
    auxData = tree.template setExtrapolatedDataField<DataSig, false, Reconstructor::WeightDegree, InternalAuxData>(InternalAuxData(Color()), samples.size(), PointSampleFunctor, AuxDataSampleFunctor, (typename FEMTree<Dim, Real>::template DensityEstimator<Reconstructor::WeightDegree>*)nullptr);
    iInfo = FEMTree<Dim, Real>::template InitializeApproximatePointInterpolationInfo<Real, 0>(tree, samples, Reconstructor::Poisson::ConstraintDual<Dim, Real>((Real)0.5, params.pointWeight * pointDepthAndWeight.value()[1]), Reconstructor::Poisson::SystemDual<Dim, Real>(params.pointWeight * pointDepthAndWeight.value()[1]), true, params.depth, 1);
    {
      typename FEMTree<Dim, Real>::template HasNormalDataFunctor<NormalSigs> hasNormalDataFunctor(*normalInfo);
      auto hasDataFunctor = [&](const FEMTreeNode* node) { return hasNormalDataFunctor(node); };
      auto addNodeFunctor = [&](int d, const int off[Dim]) { return d <= (int)params.fullDepth; };
      tree.template finalizeForMultigrid<2, 1>(params.baseDepth, addNodeFunctor, hasDataFunctor, std::make_tuple(iInfo), std::make_tuple(normalInfo, estimator, &auxData));
    }
  }

  // Solve's divergence integrator for addFEMConstraints.
  typename FEMIntegrator::template Constraint<Sigs, IsotropicUIntPack<Dim, 1>, NormalSigs, IsotropicUIntPack<Dim, 0>, Dim> DivergenceIntegrator() const {
    typename FEMIntegrator::template Constraint<Sigs, IsotropicUIntPack<Dim, 1>, NormalSigs, IsotropicUIntPack<Dim, 0>, Dim> F;
    unsigned int derivatives2[Dim] = {0, 0, 0};
    for (int d = 0; d < (int)Dim; d++) {
      unsigned int derivatives1[Dim];
      for (int dd = 0; dd < (int)Dim; dd++) derivatives1[dd] = dd == d ? 1 : 0;
      F.weights[d][TensorDerivatives<IsotropicUIntPack<Dim, 1>>::Index(derivatives1)][TensorDerivatives<IsotropicUIntPack<Dim, 0>>::Index(derivatives2)] = 1;
    }
    return F;
  }

  // Solve's SolverInfo for solveSystem.
  typename FEMTree<Dim, Real>::SolverInfo SolverInfo() const {
    typename FEMTree<Dim, Real>::SolverInfo sInfo;
    sInfo.cgDepth = 0, sInfo.cascadic = true, sInfo.vCycles = 1, sInfo.iters = params.iters, sInfo.cgAccuracy = params.cgSolverAccuracy, sInfo.verbose = false, sInfo.showResidual = false, sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE, sInfo.sliceBlockSize = 1;
    sInfo.baseVCycles = params.baseVCycles;
    return sInfo;
  }

  // Solve's constraints (FEM, then point) and linear solve, after Prepare: the solution.
  DenseNodeData<Real, Sigs> SolveSystem() {
    const int solveDepth = params.depth;
    DenseNodeData<Real, Sigs> constraints = tree.initDenseNodeData(Sigs());
    auto divergence = DivergenceIntegrator();
    tree.addFEMConstraints(divergence, *normalInfo, constraints, solveDepth);
    tree.addInterpolationConstraints(constraints, solveDepth, std::make_tuple(iInfo));
    typename FEMIntegrator::template System<Sigs, IsotropicUIntPack<Dim, 1>> F({0., 1.});
    return tree.solveSystem(Sigs(), F, constraints, params.baseDepth, params.solveDepth, SolverInfo(), std::make_tuple(iInfo));
  }
};

}  // namespace
