// poisson_tree_harness.cc: runs the first stages of PoissonRecon's Poisson::Solver::Solve
// (thirdparty/PoissonRecon/Reconstructors.h, MIT, as vendored by COLMAP 4.2.0) - the
// bounding transform (PointExtent::GetXForm), SolutionParameters::testAndSet, the transformed
// sample stream and FEMTreeInitializer::Initialize with Solve's IsValid/Process lambdas -
// then resetNodeIndices, SortedTreeNodes and the neighbor keys, and prints the tree and the
// accumulated samples; the density runs go on through the density estimator, the normal and
// color fields, the interpolation constraints and finalizeForMultigrid. Built and run by
// oracle/fixture_poisson_tree.py, which writes ColmapSharp.Tests/TestData/oracle/poisson_tree.json.
// Not part of any build. The shared set-up and output format are in oracle/poisson_harness.h.
//
// The IsValid/Process lambdas and the parameter set-up are copied from Solve (COLMAP's
// options: depth, fullDepth = depth when depth < 5, everything else default). The auxiliary
// data is a Point<float,3> (the red/green/blue of COLMAP's PLY input; PoissonRecon's
// DynamicFactory stores them as three Reals, so the arithmetic is the same).

#include "poisson_harness.h"

namespace {

template <unsigned int L, unsigned int R>
void DumpNeighbors(const std::string& name, const FEMTreeNode& root, int maxDepth) {
  typedef typename FEMTreeNode::template ConstNeighborKey<IsotropicUIntPack<Dim, L>, IsotropicUIntPack<Dim, R>> Key;
  Key key;
  key.set(maxDepth);
  std::vector<long long> out;
  size_t i = 0;
  root.processNodes([&](const FEMTreeNode* n) {
    if (i++ % 7) return;
    const auto& neighbors = key.getNeighbors(n);
    const unsigned int size = (L + R + 1) * (L + R + 1) * (L + R + 1);
    for (unsigned int k = 0; k < size; k++) {
      const FEMTreeNode* m = neighbors.neighbors.data[k];
      out.push_back(m ? m->nodeData.nodeIndex : -1);
    }
  });
  PrintI(name, out);
}

enum Mode { kTree, kDensity, kConfidence, kFinal };

// kTree: the tree-stage cases. kDensity: the same stage printed quietly, then Solve's next
// steps (setDensityEstimator<1,WeightDegree=2>, the fields, the interpolation info and
// _finalizeForMultigrid step by step) and their results. kFinal: all of that quietly on a
// second tree built from the same input, but calling the real finalizeForMultigrid<2,1>, and
// printing only the sorted-tree cases, so the step-by-step copy is checked against the vendored
// function. kConfidence: the samples with params.confidence set (normal lengths as weights).
void Run(const std::string& name, int depth, const std::vector<Sample>& input, Mode mode) {
  const bool density = mode == kDensity || mode == kFinal;
  if (mode != kFinal) {
    std::vector<double> flat;
    for (const Sample& s : input)
      for (int k = 0; k < 3; k++) flat.push_back(s.p[k]);
    for (const Sample& s : input)
      for (int k = 0; k < 3; k++) flat.push_back(s.n[k]);
    for (const Sample& s : input)
      for (int k = 0; k < 3; k++) flat.push_back(s.c[k]);
    PrintF(name + "/input", flat);
  }
  quiet = mode != kTree;

  // Solve's parameters with COLMAP's options.
  Reconstructor::Poisson::SolutionParameters<Real> params;
  params.depth = depth;
  params.fullDepth = depth < 5 ? depth : 5;
  params.baseDepth = (unsigned int)-1, params.solveDepth = (unsigned int)-1, params.kernelDepth = (unsigned int)-1;
  params.alignDir = Dim - 1;
  params.pointWeight = 1.f;  // COLMAP's PoissonMeshingOptions::point_weight default
  params.confidence = mode == kConfidence;

  FEMTree<Dim, Real> tree(MEMORY_ALLOCATOR_BLOCK_SIZE);
  VectorStream pointStream(input);
  std::vector<typename FEMTree<Dim, Real>::PointSample> samples;
  typedef DirectSum<Real, Point<Real, Dim>, DirectSum<Real, Color>> InternalNormalAndAuxData;
  std::vector<InternalNormalAndAuxData> sampleNormalAndAuxData;

  XForm<Real, Dim + 1> modelToUnitCube = XForm<Real, Dim + 1>::Identity();
  pointStream.reset();
  modelToUnitCube = PointExtent::GetXForm<Real, Dim, true, Point<Real, Dim>, Color>(pointStream, Point<Real, Dim>(), Color(), params.scale, params.alignDir) * modelToUnitCube;
  XForm<Real, Dim + 1> unitCubeToModel = modelToUnitCube.inverse();
  pointStream.reset();
  params.template testAndSet<Dim>(unitCubeToModel);

  {
    std::vector<double> x;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) x.push_back(modelToUnitCube(i, j));
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) x.push_back(unitCubeToModel(i, j));
    XForm<Real, Dim> normalXForm = XForm<Real, Dim>(modelToUnitCube).inverse().transpose() * (Real)pow(fabs(modelToUnitCube.determinant()), 1. / Dim);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) x.push_back(normalXForm(i, j));
    PrintF(name + "/xform", x);
    PrintI(name + "/params", {params.depth, params.solveDepth, params.fullDepth, params.baseDepth, params.kernelDepth});
  }

  size_t pointCount;
  {
    Reconstructor::TransformedInputOrientedSampleStream<Real, Dim, Color> _pointStream(modelToUnitCube, pointStream);
    std::vector<node_index_type> nodeToIndexMap;
    auto IsValid = [&](const Point<Real, Dim>& p, const Point<Real, Dim>& n, Color d) {
      Real l = Point<Real, Dim>::SquareNorm(n);
      return l > 0 && std::isfinite(l);
    };
    auto Process = [&](FEMTreeNode& node, const Point<Real, Dim>& p, Point<Real, Dim>& n, Color d) {
      Real l = (Real)Length(n);
      Real weight = params.confidence ? l : (Real)1.;
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
      pointCount = FEMTreeInitializer<Dim, Real>::template Initialize<decltype(IsValid), decltype(Process), Point<Real, Dim>, Color>(tree.spaceRoot(), _pointStream, Point<Real, Dim>(), Color(), params.depth, tree.nodeAllocators.size() ? tree.nodeAllocators[0] : nullptr, tree.initializer(), IsValid, Process);
    }
    PrintI(name + "/pointcount", {(long long)pointCount, (long long)tree.nodeCount()});
    PrintI(name + "/tree", DumpTree(tree.tree()));
    {
      std::vector<double> s;
      for (size_t i = 0; i < samples.size(); i++) {
        s.push_back(samples[i].node->nodeData.nodeIndex);
        for (int k = 0; k < 3; k++) s.push_back(samples[i].sample.data[k]);
        s.push_back(samples[i].sample.weight);
        const Point<Real, Dim>& n = sampleNormalAndAuxData[i].template get<0>();
        const Color& c = sampleNormalAndAuxData[i].template get<1>().template get<0>();
        for (int k = 0; k < 3; k++) s.push_back(n[k]);
        for (int k = 0; k < 3; k++) s.push_back(c[k]);
      }
      if (mode == kConfidence) quiet = false;
      PrintF(name + "/samples", s);
      if (mode == kConfidence) return;
    }

    tree.resetNodeIndices(0, std::make_tuple());
    PrintI(name + "/reset", DumpTree(tree.tree()));
    {
      std::vector<long long> s;
      for (size_t i = 0; i < samples.size(); i++) s.push_back(samples[i].node->nodeData.nodeIndex);
      PrintI(name + "/samplenodes", s);
    }

    if (density) {
      quiet = mode == kFinal;
      const long long nodesBefore = tree.nodeCount();
      auto* estimator = tree.template setDensityEstimator<1, Reconstructor::WeightDegree>(samples, params.kernelDepth, params.samplesPerNode);
      // Per node with an entry, in pre-order: node index, the estimator's data slot and the value.
      std::vector<double> values;
      tree.tree().processNodes([&](const FEMTreeNode* n) {
        const Real* w = (*estimator)(n);
        if (!w) return;
        values.push_back(n->nodeData.nodeIndex), values.push_back(estimator->index(n)), values.push_back(*w);
      });
      PrintF(name + "/density", values);
      PrintI(name + "/densitytree", DumpTree(tree.tree()));
      PrintI(name + "/densityinfo", {(long long)estimator->size(), estimator->kernelDepth(), estimator->coDimension(), nodesBefore, (long long)tree.nodeCount()});

      // Solve's next two steps, copied: the normal field (setInterpolatedDataField with the
      // normalizing ConversionFunction, then the negation) and the auxiliary (color) field
      // (setExtrapolatedDataField without a density, then the per-level scaling).
      {
        typedef IsotropicUIntPack<Dim, FEMDegreeAndBType<Reconstructor::Poisson::NormalDegree, DerivativeBoundary<BOUNDARY_NEUMANN, 1>::BType>::Signature> NormalSigs;
        ProjectiveData<Point<Real, 2>, Real> pointDepthAndWeight;
        std::function<bool(InternalNormalAndAuxData, Point<Real, Dim>&)> ConversionFunction = [](InternalNormalAndAuxData in, Point<Real, Dim>& out) {
          Point<Real, Dim> n = in.template get<0>();
          Real l = (Real)Length(n);
          if (!l) return false;
          out = n / l;
          return true;
        };
        auto normalInfo = tree.setInterpolatedDataField(Point<Real, Dim>(), NormalSigs(), samples, sampleNormalAndAuxData, estimator, params.baseDepth, params.depth, params.lowDepthCutOff, pointDepthAndWeight, ConversionFunction);
        for (size_t i = 0; i < normalInfo.size(); i++) normalInfo[i] *= (Real)-1.;
        std::vector<double> normals;
        tree.tree().processNodes([&](const FEMTreeNode* n) {
          const Point<Real, Dim>* v = normalInfo(n);
          if (!v) return;
          normals.push_back(n->nodeData.nodeIndex), normals.push_back(normalInfo.index(n));
          for (int k = 0; k < 3; k++) normals.push_back((*v)[k]);
        });
        PrintF(name + "/normals", normals);
        PrintF(name + "/pointdepthandweight", {pointDepthAndWeight.data[0], pointDepthAndWeight.data[1], pointDepthAndWeight.weight});
        PrintI(name + "/normaltree", DumpTree(tree.tree()));

        typedef DirectSum<Real, Color> InternalAuxData;
        static const unsigned int DataSig = FEMDegreeAndBType<Reconstructor::DataDegree, BOUNDARY_FREE>::Signature;
        auto PointSampleFunctor = [&](size_t i) -> const typename FEMTree<Dim, Real>::PointSample& { return samples[i]; };
        auto AuxDataSampleFunctor = [&](size_t i) -> const InternalAuxData& { return sampleNormalAndAuxData[i].template get<1>(); };
        auto auxData = tree.template setExtrapolatedDataField<DataSig, false, Reconstructor::WeightDegree, InternalAuxData>(InternalAuxData(Color()), samples.size(), PointSampleFunctor, AuxDataSampleFunctor, (typename FEMTree<Dim, Real>::template DensityEstimator<Reconstructor::WeightDegree>*)nullptr);
        tree.tree().processNodes([&](const FEMTreeNode* n) {
          ProjectiveData<InternalAuxData, Real>* clr = auxData(n);
          if (clr) (*clr) *= (Real)pow((Real)params.perLevelDataScaleFactor, tree.depth(n));
        });
        std::vector<double> colors;
        tree.tree().processNodes([&](const FEMTreeNode* n) {
          const ProjectiveData<InternalAuxData, Real>* c = auxData(n);
          if (!c) return;
          colors.push_back(n->nodeData.nodeIndex), colors.push_back(auxData.index(n));
          for (int k = 0; k < 3; k++) colors.push_back(c->data.template get<0>()[k]);
          colors.push_back(c->weight);
        });
        PrintF(name + "/colors", colors);
        PrintI(name + "/splatinfo", {(long long)normalInfo.size(), (long long)auxData.size(), (long long)tree.nodeCount()});

        // Solve's point interpolation constraints (approximate, since exactInterpolation is off):
        // per active node, the weighted mean position, the weight scaled by 2^(local depth) and
        // the dual value pointWeight * (estimated area) * 0.5 times that weight.
        typedef typename FEMTree<Dim, Real>::template InterpolationInfo<Real, 0> InterpolationInfo;
        InterpolationInfo* iInfo = FEMTree<Dim, Real>::template InitializeApproximatePointInterpolationInfo<Real, 0>(tree, samples, Reconstructor::Poisson::ConstraintDual<Dim, Real>((Real)0.5, params.pointWeight * pointDepthAndWeight.value()[1]), Reconstructor::Poisson::SystemDual<Dim, Real>(params.pointWeight * pointDepthAndWeight.value()[1]), true, params.depth, 1);
        typedef typename FEMTree<Dim, Real>::template ApproximatePointInterpolationInfo<Real, 0, Reconstructor::Poisson::ConstraintDual<Dim, Real>, Reconstructor::Poisson::SystemDual<Dim, Real>> Approximate;
        // Per node with an entry, in pre-order: node index, entry, position, weight, dual value.
        auto dumpInterpolation = [&](const std::string& caseName) {
          const auto& iData = static_cast<Approximate*>(iInfo)->iData;
          std::vector<double> out;
          tree.tree().processNodes([&](const FEMTreeNode* n) {
            const auto* d = iData(n);
            if (!d) return;
            out.push_back(n->nodeData.nodeIndex), out.push_back(iData.index(n));
            for (int k = 0; k < 3; k++) out.push_back(d->position[k]);
            out.push_back(d->weight), out.push_back(d->dualValues[0]);
          });
          PrintF(name + "/" + caseName, out);
        };
        dumpInterpolation("interpolation");
        PrintI(name + "/interpolationinfo", {(long long)static_cast<Approximate*>(iInfo)->iData.size(), (long long)tree.nodeCount()});

        // The first half of _finalizeForMultigrid< false , MaxDegree=2 , SystemDegree=1 >, as
        // Solve calls it (addNodeFunctor: d <= fullDepth, hasDataFunctor: HasNormalDataFunctor),
        // copied up to the clip because the function is monolithic: re-rooting until the
        // degree-2 functions at depth 0 fit, _setFullDepth, _refine, the flag pass and
        // _clipTree. Protected members are reached through the #define in poisson_harness.h.
        {
          typedef FEMTree<Dim, Real> Tree;
          static const unsigned int MaxDegree = 2;
          typename Tree::template HasNormalDataFunctor<NormalSigs> hasNormalDataFunctor(normalInfo);
          auto hasDataFunctor = [&](const FEMTreeNode* node) { return hasNormalDataFunctor(node); };
          auto addNodeFunctor = [&](int d, const int off[Dim]) { return d <= (int)params.fullDepth; };
          Allocator<FEMTreeNode>* nodeAllocator = tree.nodeAllocators.size() ? tree.nodeAllocators[0] : NULL;
          std::vector<node_index_type> map;
          if (mode == kFinal) {
            map = tree.template finalizeForMultigrid<MaxDegree, 1>(params.baseDepth, addNodeFunctor, hasDataFunctor, std::make_tuple(iInfo), std::make_tuple(&normalInfo, estimator, &auxData));
            quiet = false;
          } else {
          tree._baseDepth = params.baseDepth;
          while (tree._localInset(0) + BSplineEvaluationData<FEMDegreeAndBType<MaxDegree>::Signature>::Begin(0) < 0 || tree._localInset(0) + BSplineEvaluationData<FEMDegreeAndBType<MaxDegree>::Signature>::End(0) > (1 << tree._depthOffset)) {
            FEMTreeNode* oldChildren = tree._tree.children;
            FEMTreeNode* newChildren = FEMTreeNode::NewBrood(nodeAllocator, tree._nodeInitializer);
            if (oldChildren[(1 << Dim) - 1].children) {
              for (int c = 0; c < (1 << Dim); c++) oldChildren[(1 << Dim) - 1].children[c].parent = oldChildren;
              oldChildren[0].children = oldChildren[(1 << Dim) - 1].children;
              oldChildren[(1 << Dim) - 1].children = NULL;
            }
            for (int c = 0; c < (1 << Dim); c++) oldChildren[c].parent = newChildren + (1 << Dim) - 1;
            newChildren[(1 << Dim) - 1].children = oldChildren;
            for (int c = 0; c < (1 << Dim); c++) newChildren[c].parent = &tree._tree;
            tree._tree.children = newChildren;
            tree._depthOffset++;
        }
        tree._init();
        tree._maxDepth = tree._spaceRoot->maxDepth();
        PrintI(name + "/reroot", {tree._depthOffset, (long long)tree.nodeCount(), tree._maxDepth});
        PrintI(name + "/reroottree", DumpTree(tree.tree()));

        tree.template _setFullDepth<false>(IsotropicUIntPack<Dim, MaxDegree>(), nodeAllocator, tree._baseDepth);
        PrintI(name + "/fulldepthtree", DumpTree(tree.tree()));
        tree.template _refine<false>(IsotropicUIntPack<Dim, MaxDegree>(), nodeAllocator, addNodeFunctor);
        PrintI(name + "/refinetree", DumpTree(tree.tree()));

        auto _addNodeFunctor = [&](const FEMTreeNode* node) {
          int d, off[Dim];
          tree._localDepthAndOffset(node, d, off);
          return addNodeFunctor(d, off);
        };
        tree._tree.processNodes([&](FEMTreeNode* node) {
          node->nodeData.flags &= (FEMTreeNodeData::SCRATCH_FLAG);
          SetGhostFlag(node, !_addNodeFunctor(node) && tree._localDepth(node) > tree._baseDepth);
        });
        std::vector<long long> flags;
        tree._tree.processNodes([&](const FEMTreeNode* node) { flags.push_back(node->nodeData.flags); });
        PrintI(name + "/ghostflags", flags);

        tree._clipTree([&](const FEMTreeNode* node) { return _addNodeFunctor(node) || hasDataFunctor(node); }, tree._baseDepth);
        tree._maxDepth = tree._tree.maxDepth() - tree._depthOffset;
        flags.clear();
        tree._tree.processNodes([&](const FEMTreeNode* node) { flags.push_back(node->nodeData.flags); });
        PrintI(name + "/clipflags", flags);
        PrintI(name + "/clipinfo", {tree._maxDepth, (long long)tree.nodeCount(), (long long)tree.activeNodes(), (long long)tree.ghostNodes()});

        // The second half: supporting prolongation, the (here trivial) Dirichlet element
        // marks, and the sorted node list with the node data and interpolation info remapped.
        tree.template _supportApproximateProlongation<MaxDegree>();
        PrintI(name + "/prolongtree", DumpTree(tree.tree()));
        tree.template _markNonBaseDirichletElements<1>();
        flags.clear();
        tree._tree.processNodes([&](const FEMTreeNode* node) { flags.push_back(node->nodeData.flags); });
        PrintI(name + "/prolongflags", flags);
        map = tree.setSortedTreeNodes(std::make_tuple(iInfo), std::make_tuple(&normalInfo, estimator, &auxData));
        }
        PrintI(name + "/sortedmap", std::vector<long long>(map.begin(), map.end()));
        PrintI(name + "/sortedtree2", DumpTree(tree.tree()));
        std::vector<long long> sortedFlags;
        tree._tree.processNodes([&](const FEMTreeNode* node) { sortedFlags.push_back(node->nodeData.flags); });
        PrintI(name + "/sortedflags", sortedFlags);
        std::vector<long long> slices;
        for (int d = 0; d < tree._sNodes.levels(); d++) slices.push_back(tree._sNodes.begin(d)), slices.push_back(tree._sNodes.end(d));
        PrintI(name + "/sortedslices", slices);
        std::vector<double> remapped;
        tree.tree().processNodes([&](const FEMTreeNode* n) {
          const Point<Real, Dim>* v = normalInfo(n);
          const Real* w = (*estimator)(n);
          const ProjectiveData<InternalAuxData, Real>* c = auxData(n);
          if (!v && !w && !c) return;
          remapped.push_back(n->nodeData.nodeIndex);
          remapped.push_back(normalInfo.index(n)), remapped.push_back(estimator->index(n)), remapped.push_back(auxData.index(n));
        });
        PrintI(name + "/remappedslots", std::vector<long long>(remapped.begin(), remapped.end()));
        dumpInterpolation("remappedinterpolation");
      }
      delete iInfo;
    }
    delete estimator;
    return;
  }

  const int maxDepth = tree.tree().maxDepth();
  DumpNeighbors<1, 1>(name + "/neighbors11", tree.tree(), maxDepth);
  DumpNeighbors<1, 2>(name + "/neighbors12", tree.tree(), maxDepth);
  DumpNeighbors<2, 2>(name + "/neighbors22", tree.tree(), maxDepth);

  {
    SortedTreeNodes<Dim> sorted;
    std::vector<node_index_type> map;
    sorted.reset(const_cast<FEMTreeNode&>(tree.tree()), map);
    std::vector<long long> s;
    s.push_back(sorted.levels());
    for (int d = 0; d < sorted.levels(); d++)
      for (int slice = 0; slice <= (1 << d); slice++) s.push_back(sorted.begin(d, slice));
    for (node_index_type m : map) s.push_back(m);
    PrintI(name + "/sorted", s);
    PrintI(name + "/sortedtree", DumpTree(tree.tree()));
  }

  // Creating neighbor key: refine around every finest node (in traversal order).
  {
    std::vector<FEMTreeNode*> finest;
    tree.spaceRoot().processNodes([&](FEMTreeNode* n) { if (!n->children && n->depth() == maxDepth) finest.push_back(n); });
    typedef typename FEMTreeNode::template NeighborKey<IsotropicUIntPack<Dim, 1>, IsotropicUIntPack<Dim, 1>> Key;
    Key key;
    key.set(maxDepth);
    for (FEMTreeNode* n : finest) key.template getNeighbors<true, false>(n, tree.nodeAllocators.size() ? tree.nodeAllocators[0] : nullptr, tree.initializer());
    PrintI(name + "/created", DumpTree(tree.tree()));
  }

  // Neighbor-key caching and traversal details the stages above do not reach.
  {
    FEMTreeNode& root = const_cast<FEMTreeNode&>(tree.tree());
    Allocator<FEMTreeNode>* allocator = tree.nodeAllocators.size() ? tree.nodeAllocators[0] : nullptr;
    const int md = root.maxDepth();

    // processNodes with a bool functor that prunes.
    std::vector<long long> pruned;
    root.processNodes([&](const FEMTreeNode* n) {
      pruned.push_back(n->nodeData.nodeIndex);
      return n->depth() < 2 || (n->nodeData.nodeIndex % 3) == 0;
    });
    PrintI(name + "/pruned", pruned);

    // A resetting (non-const) key used without creating nodes, as the density and splat
    // stages' keys are between creations.
    {
      typename FEMTreeNode::template NeighborKey<IsotropicUIntPack<Dim, 1>, IsotropicUIntPack<Dim, 1>> key;
      key.set(md);
      std::vector<long long> out;
      size_t i = 0;
      root.processNodes([&](FEMTreeNode* n) {
        if (i++ % 7) return;
        const auto& w = key.template getNeighbors<false, false>(n, allocator, tree.initializer());
        for (unsigned int k = 0; k < 27; k++) out.push_back(w.neighbors.data[k] ? w.neighbors.data[k]->nodeData.nodeIndex : -1);
      });
      PrintI(name + "/resetkey11", out);
    }

    // getChildNeighbors of every 5th interior node.
    {
      typename FEMTreeNode::template ConstNeighborKey<IsotropicUIntPack<Dim, 1>, IsotropicUIntPack<Dim, 1>> key;
      key.set(md);
      typename FEMTreeNode::template ConstNeighborKey<IsotropicUIntPack<Dim, 1>, IsotropicUIntPack<Dim, 1>>::NeighborType childNeighbors;
      std::vector<long long> out;
      size_t i = 0;
      root.processNodes([&](const FEMTreeNode* n) {
        if (!n->children || n->depth() >= md || i++ % 5) return;
        key.getNeighbors(n);
        for (int c = 0; c < 8; c++) {
          out.push_back(key.getChildNeighbors(c, n->depth(), childNeighbors));
          for (unsigned int k = 0; k < 27; k++) out.push_back(childNeighbors.neighbors.data[k] ? childNeighbors.neighbors.data[k]->nodeData.nodeIndex : -1);
        }
      });
      PrintI(name + "/childneighbors", out);
    }

    // Cached windows across node creation: a resetting key must recompute a window that
    // had nulls (the reset rule), and a const key must drop deeper cached windows when a
    // shallower query moves (the clearing loop).
    {
      typedef typename FEMTreeNode::template NeighborKey<IsotropicUIntPack<Dim, 2>, IsotropicUIntPack<Dim, 2>> Key;
      typedef typename FEMTreeNode::template ConstNeighborKey<IsotropicUIntPack<Dim, 2>, IsotropicUIntPack<Dim, 2>> ConstKey;
      Key resetKey, createKey;
      ConstKey constKey;
      resetKey.set(md), createKey.set(md), constKey.set(md);
      std::vector<FEMTreeNode*> targets;
      size_t i = 0;
      root.processNodes([&](FEMTreeNode* n) {
        if (n->depth() == md && (i++ % 11) == 0 && targets.size() < 12) targets.push_back(n);
      });
      std::vector<long long> out;
      auto dump = [&](const FEMTreeNode* const* data) {
        for (unsigned int k = 0; k < 125; k++) out.push_back(data[k] ? data[k]->nodeData.nodeIndex : -1);
      };
      for (FEMTreeNode* n : targets) {
        dump(resetKey.template getNeighbors<false, false>(n, allocator, tree.initializer()).neighbors.data);
        dump(constKey.getNeighbors(n).neighbors.data);
        createKey.template getNeighbors<true, false>(n, allocator, tree.initializer());
        dump(resetKey.template getNeighbors<false, false>(n, allocator, tree.initializer()).neighbors.data);
        FEMTreeNode* parent = n->parent;
        FEMTreeNode* other = parent->parent->children + (((parent - parent->parent->children) + 1) % 8);
        constKey.getNeighbors(other);
        dump(constKey.getNeighbors(n).neighbors.data);
      }
      PrintI(name + "/cache", out);
      PrintI(name + "/cachetree", DumpTree(tree.tree()));
    }

    // Windows with radii other than the key's (ConstNeighborKey's getNeighbors( UIntPack ,
    // UIntPack , ... ) overloads, which the FEM system uses; NeighborKey's versions of these do
    // not compile upstream, so nothing reaches them): from the key's parent window when half
    // the radii fit in the key's, else recursively; and the pNeighbors form.
    {
      typedef typename FEMTreeNode::template ConstNeighborKey<IsotropicUIntPack<Dim, 1>, IsotropicUIntPack<Dim, 1>> ConstKey;
      ConstKey constKey;
      constKey.set(md);
      typename FEMTreeNode::template ConstNeighbors<IsotropicUIntPack<Dim, 5>> c22, p22;
      typename FEMTreeNode::template ConstNeighbors<UIntPack<4, 4, 4>> c12;
      typename FEMTreeNode::template ConstNeighbors<IsotropicUIntPack<Dim, 7>> c33;
      typename FEMTreeNode::template ConstNeighbors<UIntPack<6, 6, 6>> c32;
      std::vector<long long> out;
      auto dump = [&](const auto& w, unsigned int size) {
        for (unsigned int k = 0; k < size; k++) out.push_back(w.neighbors.data[k] ? w.neighbors.data[k]->nodeData.nodeIndex : -1);
      };
      size_t i = 0;
      root.processNodes([&](const FEMTreeNode* n) {
        if (i++ % 9) return;
        constKey.getNeighbors(IsotropicUIntPack<Dim, 2>(), IsotropicUIntPack<Dim, 2>(), n, c22), dump(c22, 125);
        constKey.getNeighbors(IsotropicUIntPack<Dim, 1>(), IsotropicUIntPack<Dim, 2>(), n, c12), dump(c12, 64);
        constKey.getNeighbors(IsotropicUIntPack<Dim, 3>(), IsotropicUIntPack<Dim, 3>(), n, c33), dump(c33, 343);
        constKey.getNeighbors(IsotropicUIntPack<Dim, 3>(), IsotropicUIntPack<Dim, 2>(), n, c32), dump(c32, 216);
        constKey.getNeighbors(IsotropicUIntPack<Dim, 2>(), IsotropicUIntPack<Dim, 2>(), n, p22, c22), dump(p22, 125), dump(c22, 125);
      });
      PrintI(name + "/radii", out);
    }
  }
}

}  // namespace

int main() {
  ThreadPool::ParallelizationType = ThreadPool::NONE;
  // Each density run is repeated as "<name>final" on the same input (kFinal).
  auto density = [](const std::string& name, int depth, int count) {
    std::vector<Sample> input = MakeInput(count);
    Run(name, depth, input, kDensity);
    Run(name + "final", depth, input, kFinal);
  };
  Run("depth5", 5, MakeInput(300), kTree);
  Run("depth3", 3, MakeInput(100), kTree);
  density("density5", 5, 300);
  density("density3", 3, 100);
  Run("confidence3", 3, MakeInput(100), kConfidence);
  // Depth above the full (and so base) depth, as in COLMAP's default runs: samples then
  // splat at fractional depths, into two levels' worth of weight.
  density("density6", 6, 600);
  // Depth two or more above the full depth, so _supportApproximateProlongation has levels to
  // work on: it creates nodes, un-ghosts them, and the sorted map gets -1 entries.
  density("density8", 8, 500);
  return 0;
}
