// poisson_levelset5_harness.cc: the level-set extractor's iso-edges
// (thirdparty/PoissonRecon/FEMTree.LevelSet.3D.inl's _LevelSetExtractor< ... , 3 , ... >
// CopyFinerSliceIsoEdgeKeys, CopyFinerXSliceIsoEdgeKeys, SetSliceIsoEdges, SetXSliceIsoEdges
// and the SliceValues / XSliceValues setFromScratch forms, MIT, as vendored by COLMAP 4.2.0),
// as Extract runs them for COLMAP (whole tree, nonLinearFit on, gradientNormals off, with the
// density estimator and the color field, as --density and --colors ask): Extract's slab loop
// with InitSlice, InitSlab, SetSliceValues, SetSlabIsoVertices, SetSliceIso (vertices then
// edges), SetSlabIsoEdges, FinalizeSlice and FinalizeSlab, in Extract's order, leaving out only
// IsoSurface (the polygons). Extract's lambdas are copied here, since they are local to it; the
// steps they call are upstream's own. Output, for each finalized slice and slab (after a
// step/depth/index head): the edge keys whose edges are set (own vertices and keys copied from
// finer), the iso-edges of each face that is set, the face-edge map and the vertex-pair map
// (both sorted by key; upstream's are unordered_maps, only ever looked up), plus the vertex
// count. The "levelset3" run is printed in full (poisson_harness.h's KeepFull); the others as
// checksums. Built and run by oracle/fixture_poisson_tree.py, which writes
// ColmapSharp.Tests/TestData/oracle/poisson_levelset5.json (read by
// ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.LevelSet5.cs). Not part of any
// build. The set-up and the solve are oracle/poisson_solve.h's PoissonRun.

#include "poisson_solve.h"

namespace {

typedef _LevelSetExtractor<true, Real, Dim, InternalAuxData> Extractor;
typedef LevelSetExtraction::Key<Dim> Key;
typedef LevelSetExtraction::IsoEdge<Dim> IsoEdge;
static const unsigned int kSig = FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature;

struct VertexSink {
  std::vector<Extractor::Vertex> vertices;
  size_t write(unsigned int, const Extractor::Vertex& v) {
    vertices.push_back(v);
    return vertices.size() - 1;
  }
};

void PushKey(const Key& k, std::vector<long long>& out) {
  for (int i = 0; i < 3; i++) out.push_back((long long)k.idx[i]);
}

std::array<long long, 3> KeyArray(const Key& k) { return {(long long)k.idx[0], (long long)k.idx[1], (long long)k.idx[2]}; }

// The four dumps of one finalized slice or slab.
struct Dumps {
  std::vector<long long> keys, faces, faceMaps, pairs;
};

// Edge keys (index, key) where eSet; faces (index, count, edges) where fSet; the face-edge map
// and the vertex-pair map sorted by key. Each after a step/depth/index/count head.
template <typename Values>
void Dump(const Values& values, const char* eSet, unsigned int eCount, const char* fSet, unsigned int fCount, long long step, int d, unsigned int o, Dumps& out) {
  std::vector<long long> keys, faces, faceMaps, pairs;
  long long n = 0;
  for (unsigned int i = 0; i < eCount; i++)
    if (eSet[i]) keys.push_back(i), PushKey(values.edgeKeys[i], keys), n++;
  out.keys.insert(out.keys.end(), {step, d, (long long)o, n});
  out.keys.insert(out.keys.end(), keys.begin(), keys.end());

  n = 0;
  for (unsigned int i = 0; i < fCount; i++)
    if (fSet[i]) {
      const auto& fe = values.faceEdges[i];
      faces.push_back(i), faces.push_back(fe.count), n++;
      for (int j = 0; j < fe.count; j++) PushKey(fe.edges[j][0], faces), PushKey(fe.edges[j][1], faces);
    }
  out.faces.insert(out.faces.end(), {step, d, (long long)o, n});
  out.faces.insert(out.faces.end(), faces.begin(), faces.end());

  std::vector<std::pair<std::array<long long, 3>, std::vector<long long>>> maps;
  for (const auto& kv : values.faceEdgeMap) {
    std::vector<long long> edges;
    edges.push_back(kv.second.size());
    for (const IsoEdge& e : kv.second) PushKey(e[0], edges), PushKey(e[1], edges);
    maps.push_back({KeyArray(kv.first), edges});
  }
  std::sort(maps.begin(), maps.end());
  out.faceMaps.insert(out.faceMaps.end(), {step, d, (long long)o, (long long)maps.size()});
  for (const auto& m : maps) {
    out.faceMaps.insert(out.faceMaps.end(), m.first.begin(), m.first.end());
    out.faceMaps.insert(out.faceMaps.end(), m.second.begin(), m.second.end());
  }

  std::vector<std::pair<std::array<long long, 3>, std::array<long long, 3>>> pairMap;
  for (const auto& kv : values.vertexPairMap) pairMap.push_back({KeyArray(kv.first), KeyArray(kv.second)});
  std::sort(pairMap.begin(), pairMap.end());
  out.pairs.insert(out.pairs.end(), {step, d, (long long)o, (long long)pairMap.size()});
  for (const auto& p : pairMap) {
    out.pairs.insert(out.pairs.end(), p.first.begin(), p.first.end());
    out.pairs.insert(out.pairs.end(), p.second.begin(), p.second.end());
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

  Dumps slices, slabs;
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
  auto SetSliceIso = [&](unsigned int sliceAtMaxDepth) {
    int d;
    unsigned int o;
    for (d = maxDepth, o = sliceAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      Extractor::template SetSliceIsoVertices<Reconstructor::WeightDegree, DataSig>(keyGenerator, tree, true, false, &pointEvaluator, run.estimator, &run.auxData, isoValue, d, fullDepth, o, sink, slabValues, zeroData);
      if (o & 1) break;
    }
    for (d = maxDepth, o = sliceAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      if (d < tree._maxDepth) Extractor::CopyFinerSliceIsoEdgeKeys(tree, d, fullDepth, o, slabValues);
      Extractor::SetSliceIsoEdges(keyGenerator, tree, d, o, slabValues);
      if (o & 1) break;
    }
  };
  auto SetSlabIsoEdges = [&](unsigned int slabAtMaxDepth) {
    int d;
    unsigned int o;
    for (d = maxDepth, o = slabAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      if (d < tree._maxDepth) Extractor::CopyFinerXSliceIsoEdgeKeys(tree, d, fullDepth, o, slabValues);
      Extractor::SetXSliceIsoEdges(keyGenerator, tree, d, o, slabValues);
      if (!(o & 1)) break;
    }
  };
  auto FinalizeSlice = [&](unsigned int sliceAtMaxDepth) {
    int d;
    unsigned int o;
    for (d = maxDepth, o = sliceAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      auto& values = slabValues[d].sliceValues(o);
      auto& scratch = slabValues[d].sliceScratch(o);
      values.setFromScratch(scratch.vKeyValues);
      values.setFromScratch(scratch.eKeyValues);
      values.setFromScratch(scratch.fKeyValues);
      Dump(values, scratch.eSet, values.cellIndices.counts[1], scratch.fSet, values.cellIndices.counts[2], step, d, o, slices);
      if (o & 1) break;
    }
  };
  auto FinalizeSlab = [&](unsigned int slabAtMaxDepth) {
    int d;
    unsigned int o;
    for (d = maxDepth, o = slabAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      auto& values = slabValues[d].xSliceValues(o);
      auto& scratch = slabValues[d].xSliceScratch(o);
      values.setFromScratch(scratch.vKeyValues);
      values.setFromScratch(scratch.eKeyValues);
      values.setFromScratch(scratch.fKeyValues);
      Dump(values, scratch.eSet, values.cellIndices.counts[0], scratch.fSet, values.cellIndices.counts[1], step, d, o, slabs);
      if (!(o & 1)) break;
    }
  };

  // Extract's slab loop, without IsoSurface.
  InitSlice(0);
  InitSlab(0, true);
  SetSliceValues(0);
  SetSliceIso(0);
  FinalizeSlice(0);
  step++;
  for (unsigned int slab = 0; slab < (1u << maxDepth); slab++) {
    InitSlice(slab + 1);
    if (slab != 0) InitSlab(slab, false);
    SetSliceValues(slab + 1);
    SetSlabIsoVertices(slab);
    SetSliceIso(slab + 1);
    SetSlabIsoEdges(slab);
    FinalizeSlice(slab + 1);
    FinalizeSlab(slab);
    step++;
  }

  PrintI(name + "/vertexcount", {(long long)sink.vertices.size()});
  PrintI(name + "/isoedges/slicekeys", slices.keys);
  PrintI(name + "/isoedges/slicefaces", slices.faces);
  PrintI(name + "/isoedges/slicefacemaps", slices.faceMaps);
  PrintI(name + "/isoedges/slicepairs", slices.pairs);
  PrintI(name + "/isoedges/slabkeys", slabs.keys);
  PrintI(name + "/isoedges/slabfaces", slabs.faces);
  PrintI(name + "/isoedges/slabfacemaps", slabs.faceMaps);
  PrintI(name + "/isoedges/slabpairs", slabs.pairs);
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
