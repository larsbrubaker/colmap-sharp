// poisson_levelset4_harness.cc: the level-set extractor's iso-vertices on slice edges and on
// cross-slice (slab) edges (thirdparty/PoissonRecon/FEMTree.LevelSet.3D.inl's
// _LevelSetExtractor< ... , 3 , ... > SetSliceIsoVertices, SetXSliceIsoVertices and both forms of
// GetIsoVertex, MIT, as vendored by COLMAP 4.2.0), as Extract runs them for COLMAP (whole tree,
// nonLinearFit on, gradientNormals off, with the density estimator and the color field, as
// --density and --colors ask): Extract's slab loop with InitSlice, InitSlab, SetSliceValues,
// SetSlabIsoVertices, the vertex loop of SetSliceIso, and the edge parts of FinalizeSlice and
// FinalizeSlab, in Extract's order, leaving out the iso-edges and the polygons (which write no
// vertices without --barycenter, which COLMAP never passes). Extract's lambdas are copied here,
// since they are local to it. The level is Solve's iso-value, and the color field is scaled per
// level as Solve does. Output: every vertex in write order (position, gradient, depth, color),
// the colors alone in full for the smaller runs (so a mismatch can be located without the
// checksum), each finalized slice's and slab's edge-vertex map sorted by key, and the bad-root
// count. A "crafted" run (see Run) reaches the branches the others miss: zeroData colors, a
// clamped root, and pushes below the full depth. Built and run by oracle/fixture_poisson_tree.py, which writes
// ColmapSharp.Tests/TestData/oracle/poisson_levelset4.json (read by
// ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.LevelSet4.cs). Not part of any
// build. The set-up and the solve are oracle/poisson_solve.h's PoissonRun.

#include "poisson_solve.h"

namespace {

typedef _LevelSetExtractor<true, Real, Dim, InternalAuxData> Extractor;
static const unsigned int kSig = FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature;

struct VertexSink {
  std::vector<Extractor::Vertex> vertices;
  size_t write(unsigned int, const Extractor::Vertex& v) {
    vertices.push_back(v);
    return vertices.size() - 1;
  }
};

template <typename KeyValues>
void DumpKeyValues(const KeyValues& lists, std::vector<long long>& out) {
  for (size_t t = 0; t < lists.size(); t++)
    for (const auto& kv : lists[t]) {
      for (int k = 0; k < 3; k++) out.push_back(kv.first.idx[k]);
      out.push_back(kv.second.first);
    }
}

// crafted: the run that reaches the rarer branches of GetIsoVertex and of the vertex push-down.
// Only the even node indices keep their color (the others are zeroed, so some vertices have no
// color weight and take zeroData, which is non-zero here); the level is the median of the
// corner values set on slice 0 at the full depth, so the level set meets the domain's z = 0
// face (whose edges push their vertices below the full depth) and passes exactly through
// corners (roots on an edge end, clamped and counted as bad); and the edge keys pushed below
// the full depth, which Extract never reads back, are dumped at the end.
void Run(const std::string& name, int depth, const std::vector<Sample>& input, bool fullColors, bool crafted = false) {
  PoissonRun run;
  run.Prepare(depth, input);
  DenseNodeData<Real, Sigs> solution = run.SolveSystem();
  auto& tree = run.tree;

  // Solve's per-level scaling of the color field.
  const Real perLevelDataScaleFactor = run.params.perLevelDataScaleFactor;
  tree.tree().processNodes([&](const FEMTreeNode* n) {
    ProjectiveData<InternalAuxData, Real>* clr = run.auxData(n);
    if (clr) (*clr) *= (Real)pow((Real)perLevelDataScaleFactor, tree.depth(n));
    if (clr && crafted && (n->nodeData.nodeIndex & 1)) *clr = ProjectiveData<InternalAuxData, Real>();
  });

  // Solve's iso-value, single-threaded.
  double valueSum = 0, weightSum = 0;
  {
    typename FEMTree<Dim, Real>::template MultiThreadedEvaluator<Sigs, 0> evaluator(&tree, solution);
    for (size_t j = 0; j < run.samples.size(); j++) {
      ProjectiveData<Point<Real, Dim>, Real>& sample = run.samples[j].sample;
      Real w = sample.weight;
      if (w > 0) {
        Real value = evaluator.values(sample.data / sample.weight, 0, run.samples[j].node)[0];
        weightSum += w, valueSum += value * w;
      }
    }
  }
  Real isoValue = (Real)(valueSum / weightSum);

  // Extract's set-up.
  tree._setFEM1ValidityFlags(Sigs());
  LevelSetExtraction::SetHyperCubeTables<Dim>();
  LevelSetExtraction::SetHyperCubeTables<Dim - 1>();
  typename FEMTree<Dim, Real>::LocalOffset start, end;
  for (unsigned int d = 0; d < Dim; d++) start[d] = 0, end[d] = 1;
  int fullDepth = tree.getFullDepth(UIntPack<1, 1, 1>(), 0, start, end);
  unsigned int maxDepth = tree._maxDepth;
  LevelSetExtraction::KeyGenerator<Dim> keyGenerator(maxDepth);
  DenseNodeData<Real, Sigs> coarseCoefficients(tree._sNodesEnd(tree._maxDepth - 1));
  memset(coarseCoefficients(), 0, sizeof(Real) * tree._sNodesEnd(tree._maxDepth - 1));
  for (size_t i = tree._sNodesBegin(0); i < (size_t)tree._sNodesEnd(tree._maxDepth - 1); i++) coarseCoefficients[i] = solution[i];
  typename FEMIntegrator::template RestrictionProlongation<Sigs> rp;
  for (int d = 1; d < tree._maxDepth; d++) tree._upSample(Sigs(), rp, d, (ConstPointer(Real))coarseCoefficients() + tree._sNodesBegin(d - 1), coarseCoefficients() + tree._sNodesBegin(d));
  std::vector<typename FEMTree<Dim, Real>::template _Evaluator<Sigs, 1>> evaluators(tree._maxDepth + 1);
  for (int d = 0; d <= tree._maxDepth; d++) evaluators[d].set(tree._maxDepth);
  std::vector<typename Extractor::SlabValues> slabValues(tree._maxDepth + 1);
  typename FEMIntegrator::template PointEvaluator<IsotropicUIntPack<Dim, DataSig>, ZeroUIntPack<Dim>> pointEvaluator(tree._maxDepth);
  const InternalAuxData zeroData = crafted ? InternalAuxData(Color(0.25f, 0.5f, 0.75f)) : InternalAuxData(Color());
  VertexSink sink;
  const size_t badRootsBefore = Extractor::_BadRootCount;  // A static counter: count this run's.

  std::vector<long long> edgeMaps, slabMaps;
  long long step = 0;
  // A key map sorted by key (upstream's is an unordered_map), after a step/depth/index head.
  auto DumpMap = [&](const auto& map, int d, unsigned int o, std::vector<long long>& out) {
    std::vector<std::array<long long, 4>> entries;
    for (const auto& kv : map) entries.push_back({(long long)kv.first.idx[0], (long long)kv.first.idx[1], (long long)kv.first.idx[2], (long long)kv.second.first});
    std::sort(entries.begin(), entries.end());
    out.push_back(step), out.push_back(d), out.push_back(o), out.push_back(entries.size());
    for (const auto& e : entries) out.insert(out.end(), e.begin(), e.end());
  };
  auto InitSlice = [&](unsigned int sliceAtMaxDepth) {
    for (int d = maxDepth; d >= fullDepth; d--) {
      unsigned int dOff = maxDepth - d, slice = sliceAtMaxDepth >> dOff;
      if (sliceAtMaxDepth != (slice << dOff)) break;
      auto& values = slabValues[d].sliceValues(slice);
      values.cellIndices.set(tree._sNodes, tree._localToGlobal(d), slice + tree._localInset(d));
      values.reset(slice, true);
      slabValues[d].sliceScratch(slice).reset(values.cellIndices);
    }
  };
  auto InitSlab = [&](unsigned int slabAtMaxDepth, bool first) {
    unsigned int slab = slabAtMaxDepth;
    for (int d = maxDepth; d >= fullDepth; d--, slab >>= 1) {
      slabValues[d].xSliceValues(slab).cellIndices.set(tree._sNodes, tree._localToGlobal(d), slab + tree._localInset(d));
      slabValues[d].xSliceValues(slab).reset(slab);
      slabValues[d].xSliceScratch(slab).reset(slabValues[d].xSliceValues(slab).cellIndices);
      if ((slab & 1) && !first) break;
    }
  };
  auto SetSliceValues = [&](unsigned int sliceAtMaxDepth) {
    int d;
    unsigned int o;
    for (d = maxDepth, o = sliceAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      Extractor::template SetSliceCornerValuesAndMCIndices<kSig, kSig, kSig>(tree, solution(), coarseCoefficients(), isoValue, d, fullDepth, o, slabValues, evaluators[d]);
      if (o & 1) break;
    }
  };
  // SetSlabIsoVertices with SetSlabBounds; InteriorSlab holds for every slab of the whole tree.
  auto SetSlabIsoVertices = [&](unsigned int slabAtMaxDepth) {
    int d;
    unsigned int o;
    for (d = maxDepth, o = slabAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      unsigned int start = (o + 0) << (maxDepth - d), end = (o + 1) << (maxDepth - d);
      Real bCoordinate = start / (Real)(1 << maxDepth), fCoordinate = end / (Real)(1 << maxDepth);
      Extractor::template SetXSliceIsoVertices<Reconstructor::WeightDegree, DataSig>(keyGenerator, tree, true, false, &pointEvaluator, run.estimator, &run.auxData, isoValue, d, fullDepth, o, bCoordinate, fCoordinate, sink, slabValues, zeroData);
      if (!(o & 1)) break;
    }
  };
  auto SetSliceIsoVertices = [&](unsigned int sliceAtMaxDepth) {
    int d;
    unsigned int o;
    for (d = maxDepth, o = sliceAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      Extractor::template SetSliceIsoVertices<Reconstructor::WeightDegree, DataSig>(keyGenerator, tree, true, false, &pointEvaluator, run.estimator, &run.auxData, isoValue, d, fullDepth, o, sink, slabValues, zeroData);
      if (o & 1) break;
    }
  };
  auto FinalizeSlice = [&](unsigned int sliceAtMaxDepth) {
    int d;
    unsigned int o;
    for (d = maxDepth, o = sliceAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      auto& values = slabValues[d].sliceValues(o);
      values.setFromScratch(slabValues[d].sliceScratch(o).eKeyValues);
      DumpMap(values.edgeVertexMap, d, o, edgeMaps);
      if (o & 1) break;
    }
  };
  auto FinalizeSlab = [&](unsigned int slabAtMaxDepth) {
    int d;
    unsigned int o;
    for (d = maxDepth, o = slabAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      auto& values = slabValues[d].xSliceValues(o);
      values.setFromScratch(slabValues[d].xSliceScratch(o).eKeyValues);
      DumpMap(values.edgeVertexMap, d, o, slabMaps);
      if (!(o & 1)) break;
    }
  };

  if (crafted) {
    // The corner values do not depend on the level; the loop below re-initializes slice 0.
    InitSlice(0);
    InitSlab(0, true);
    SetSliceValues(0);
    auto& values = slabValues[fullDepth].sliceValues(0);
    std::vector<Real> corners;
    for (size_t i = 0; i < values.cellIndices.counts[0]; i++)
      if (slabValues[fullDepth].sliceScratch(0).cSet[i]) corners.push_back(values.cornerValues[i]);
    std::sort(corners.begin(), corners.end());
    isoValue = corners[corners.size() / 2];
  }
  PrintF(name + "/isovalue", {isoValue});

  // Extract's slab loop, without the iso-edges and the polygons.
  InitSlice(0);
  InitSlab(0, true);
  SetSliceValues(0);
  SetSliceIsoVertices(0);
  FinalizeSlice(0);
  step++;
  for (unsigned int slab = 0; slab < (1u << maxDepth); slab++) {
    InitSlice(slab + 1);
    if (slab != 0) InitSlab(slab, false);
    SetSliceValues(slab + 1);
    SetSlabIsoVertices(slab);
    SetSliceIsoVertices(slab + 1);
    FinalizeSlice(slab + 1);
    FinalizeSlab(slab);
    step++;
  }

  std::vector<double> vertices;
  for (const auto& v : sink.vertices) {
    for (int k = 0; k < 3; k++) vertices.push_back(v.template get<0>()[k]);
    for (int k = 0; k < 3; k++) vertices.push_back(v.template get<1>()[k]);
    vertices.push_back(v.template get<2>());
    for (int k = 0; k < 3; k++) vertices.push_back(v.template get<3>().template get<0>()[k]);
  }
  PrintI(name + "/vertexcount", {(long long)sink.vertices.size()});
  PrintF(name + "/vertices", vertices);
  if (fullColors) {
    std::vector<double> colors;
    for (const auto& v : sink.vertices)
      for (int k = 0; k < 3; k++) colors.push_back(v.template get<3>().template get<0>()[k]);
    PrintF(name + "/vertexcolors", colors);
  }
  PrintI(name + "/edgemaps", edgeMaps);
  PrintI(name + "/slabmaps", slabMaps);
  PrintI(name + "/badroots", {(long long)(Extractor::_BadRootCount - badRootsBefore)});
  if (crafted) {
    // Every key pushed below the full depth, in recording order: depth and parity, then the
    // slice keys and the slab keys (each a count, then key and vertex per entry).
    std::vector<long long> coarseKeys;
    for (int d = fullDepth - 1; d >= 0; d--)
      for (int parity = 0; parity < 2; parity++) {
        std::vector<long long> sliceKeys, slabKeys;
        DumpKeyValues(slabValues[d].sliceScratch(parity).eKeyValues, sliceKeys);
        DumpKeyValues(slabValues[d].xSliceScratch(parity).eKeyValues, slabKeys);
        coarseKeys.push_back(d), coarseKeys.push_back(parity);
        coarseKeys.push_back(sliceKeys.size() / 4), coarseKeys.insert(coarseKeys.end(), sliceKeys.begin(), sliceKeys.end());
        coarseKeys.push_back(slabKeys.size() / 4), coarseKeys.insert(coarseKeys.end(), slabKeys.begin(), slabKeys.end());
      }
    PrintI(name + "/coarsekeys", coarseKeys);
  }
}

}  // namespace

int main() {
  ThreadPool::ParallelizationType = ThreadPool::NONE;
  Run("levelset3", 3, MakeInput(100), true);
  // MakeInput draws from one running generator, so levelset5's points are kept for crafted5.
  const std::vector<Sample> input5 = MakeInput(300);
  Run("levelset5", 5, input5, true);
  Run("levelset6", 6, MakeInput(600), true);
  Run("levelset8", 8, MakeInput(500), false);
  Run("crafted5", 5, input5, true, true);
  return 0;
}
