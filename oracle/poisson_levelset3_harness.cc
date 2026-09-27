// poisson_levelset3_harness.cc: the level-set extractor's iso-vertices on slice edges
// (thirdparty/PoissonRecon/FEMTree.LevelSet.3D.inl's _LevelSetExtractor< ... , 3 , ... >
// SetSliceIsoVertices and GetIsoVertex, MIT, as vendored by COLMAP 4.2.0), as Extract runs them
// for COLMAP (whole tree, nonLinearFit on, gradientNormals off, with the density estimator and
// the color field, as --density and --colors ask): Extract's slab loop with InitSlice,
// InitSlab, SetSliceValues, the vertex loop of SetSliceIso and the edge part of FinalizeSlice,
// leaving out the cross-slice (slab) vertices, the iso-edges and the polygons. Extract's lambdas
// are copied here, since they are local to it. The level is Solve's iso-value, and the color
// field is scaled per level as Solve does. Output: every vertex in write order (position,
// gradient, depth, color), each finalized slice's edge-vertex map sorted by key, the edge keys
// pushed into coarser slabs, and the bad-root count. Built and run by
// oracle/fixture_poisson_tree.py, which writes
// ColmapSharp.Tests/TestData/oracle/poisson_levelset3.json (read by
// ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.LevelSet3.cs). Not part of any
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

void Run(const std::string& name, int depth, const std::vector<Sample>& input) {
  PoissonRun run;
  run.Prepare(depth, input);
  DenseNodeData<Real, Sigs> solution = run.SolveSystem();
  auto& tree = run.tree;

  // Solve's per-level scaling of the color field.
  const Real perLevelDataScaleFactor = run.params.perLevelDataScaleFactor;
  tree.tree().processNodes([&](const FEMTreeNode* n) {
    ProjectiveData<InternalAuxData, Real>* clr = run.auxData(n);
    if (clr) (*clr) *= (Real)pow((Real)perLevelDataScaleFactor, tree.depth(n));
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
  const Real isoValue = (Real)(valueSum / weightSum);
  PrintF(name + "/isovalue", {isoValue});

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
  const InternalAuxData zeroData = InternalAuxData(Color());
  VertexSink sink;
  const size_t badRootsBefore = Extractor::_BadRootCount;  // A static counter: count this run's.

  std::vector<long long> edgeMaps, slabKeys;
  long long step = 0;
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
      // The map, sorted by key.
      std::vector<std::array<long long, 4>> entries;
      for (const auto& kv : values.edgeVertexMap) entries.push_back({(long long)kv.first.idx[0], (long long)kv.first.idx[1], (long long)kv.first.idx[2], (long long)kv.second.first});
      std::sort(entries.begin(), entries.end());
      edgeMaps.push_back(step), edgeMaps.push_back(d), edgeMaps.push_back(o), edgeMaps.push_back(entries.size());
      for (const auto& e : entries) edgeMaps.insert(edgeMaps.end(), e.begin(), e.end());
      if (o & 1) break;
    }
    // Every slab's pushed-down edge keys, in recording order.
    for (int dd = maxDepth; dd >= fullDepth; dd--)
      for (int parity = 0; parity < 2; parity++) {
        std::vector<long long> keys;
        DumpKeyValues(slabValues[dd].xSliceScratch(parity).eKeyValues, keys);
        if (keys.empty()) continue;
        slabKeys.push_back(step), slabKeys.push_back(dd), slabKeys.push_back(parity), slabKeys.push_back(keys.size() / 4);
        slabKeys.insert(slabKeys.end(), keys.begin(), keys.end());
      }
    step++;
  };

  // Extract's slab loop, without the cross-slice vertices, iso-edges and polygons.
  InitSlice(0);
  InitSlab(0, true);
  SetSliceValues(0);
  SetSliceIsoVertices(0);
  FinalizeSlice(0);
  for (unsigned int slab = 0; slab < (1u << maxDepth); slab++) {
    InitSlice(slab + 1);
    if (slab != 0) InitSlab(slab, false);
    SetSliceValues(slab + 1);
    SetSliceIsoVertices(slab + 1);
    FinalizeSlice(slab + 1);
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
  PrintI(name + "/edgemaps", edgeMaps);
  PrintI(name + "/slabkeys", slabKeys);
  PrintI(name + "/badroots", {(long long)(Extractor::_BadRootCount - badRootsBefore)});
}

}  // namespace

int main() {
  ThreadPool::ParallelizationType = ThreadPool::NONE;
  Run("levelset3", 3, MakeInput(100));
  Run("levelset5", 5, MakeInput(300));
  Run("levelset6", 6, MakeInput(600));
  Run("levelset8", 8, MakeInput(500));
  return 0;
}
