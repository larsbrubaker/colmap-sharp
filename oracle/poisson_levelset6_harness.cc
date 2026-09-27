// poisson_levelset6_harness.cc: the level-set extractor's polygons
// (thirdparty/PoissonRecon/FEMTree.LevelSet.3D.inl's _LevelSetExtractor< ... , 3 , ... >
// SetLevelSet and AddIsoPolygons, with SliceValues' addIsoEdges, setVertexPair and
// setEdgeVertex, and MAT.h's MinimalAreaTriangulation, MIT, as vendored by COLMAP 4.2.0), as
// Extract runs them for COLMAP (whole tree, nonLinearFit on, gradientNormals off, with the
// density estimator and the color field, as --density and --colors ask; addBarycenter on,
// since Reconstructors.h passes forceManifold = !--nonManifold there, polygonMesh and
// flipOrientation off): Extract's whole slab loop with InitSlice, InitSlab, SetSliceValues,
// SetSlabIsoVertices, SetSliceIso, SetSlabIsoEdges, FinalizeSlice, FinalizeSlab and
// IsoSurface, in Extract's order. Extract's lambdas are copied here, since they are local to
// it; the steps they call are upstream's own. Output: the total vertex count (iso-vertices and
// barycenters), the triangle count, every triangle in polygonStream.write order (its size, 3,
// then its vertex indices), and each barycenter vertex written by AddIsoPolygons (its index,
// then position, gradient, depth and color); then MinimalAreaTriangulation alone on crafted
// polygons ("mat"), since the runs need not reach every polygon size. The "levelset3" run is
// printed in full (poisson_harness.h's KeepFull); the others as checksums. Built and run by oracle/fixture_poisson_tree.py, which writes
// ColmapSharp.Tests/TestData/oracle/poisson_levelset6.json (read by
// ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.LevelSet6.cs). Not part of any
// build. The set-up and the solve are oracle/poisson_solve.h's PoissonRun.

#include "poisson_solve.h"

namespace {

typedef _LevelSetExtractor<true, Real, Dim, InternalAuxData> Extractor;
static const unsigned int kSig = FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature;

// The vertex stream; the writes made while IsoSurface runs are the barycenters.
struct VertexSink {
  std::vector<Extractor::Vertex> vertices;
  std::vector<size_t> barycenters;
  bool inIsoSurface = false;
  size_t write(unsigned int, const Extractor::Vertex& v) {
    vertices.push_back(v);
    if (inIsoSurface) barycenters.push_back(vertices.size() - 1);
    return vertices.size() - 1;
  }
};

struct PolygonSink : public OutputDataStream<std::vector<node_index_type>> {
  std::vector<std::vector<node_index_type>> polygons;
  size_t size(void) const { return polygons.size(); }
  size_t write(const std::vector<node_index_type>& p) {
    polygons.push_back(p);
    return polygons.size() - 1;
  }
};

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
  PolygonSink polygons;

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
      if (!(o & 1)) break;
    }
  };
  // IsoSurface; InteriorSlab holds for every slab of the whole tree, and the faceIndexFunctor's
  // slab-bound adjustments never apply (slabStartAtMaxDepth 0, slabEndAtMaxDepth 1<<maxDepth),
  // but they are copied as they are.
  const unsigned int slabStartAtMaxDepth = 0, slabEndAtMaxDepth = 1u << maxDepth;
  auto IsoSurface = [&](unsigned int slabAtMaxDepth) {
    sink.inIsoSurface = true;
    int d;
    int o;
    for (d = maxDepth, o = slabAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      auto faceIndexFunctor = [&](const Extractor::TreeNode* node, typename HyperCube::Cube<Dim>::template Element<2> f) {
        HyperCube::Direction dir;
        unsigned int coIndex;
        f.factor(dir, coIndex);
        int depth, offset[Dim];
        tree.depthAndOffset(node, depth, offset);
        LevelSetExtraction::Key<Dim> key = keyGenerator(depth, offset, f);
        if (dir == HyperCube::BACK && ((((unsigned int)offset[Dim - 1] + 0) << (maxDepth - depth)) < slabStartAtMaxDepth))
          key[Dim - 1] = keyGenerator.cornerIndex(maxDepth, slabStartAtMaxDepth);
        else if (dir == HyperCube::FRONT && ((((unsigned int)offset[Dim - 1] + 1) << (maxDepth - depth)) > slabEndAtMaxDepth))
          key[Dim - 1] = keyGenerator.cornerIndex(maxDepth, slabEndAtMaxDepth);
        return key;
      };
      Extractor::SetLevelSet(keyGenerator, faceIndexFunctor, tree, d, o, slabValues[d].sliceValues(o), slabValues[d].sliceValues(o + 1), slabValues[d].xSliceValues(o), slabValues[d].sliceScratch(o), slabValues[d].sliceScratch(o + 1), slabValues[d].xSliceScratch(o), sink, polygons, false, true, false);
      if (!(o & 1)) break;
    }
    sink.inIsoSurface = false;
  };

  // Extract's slab loop.
  InitSlice(0);
  InitSlab(0, true);
  SetSliceValues(0);
  SetSliceIso(0);
  FinalizeSlice(0);
  for (unsigned int slab = 0; slab < (1u << maxDepth); slab++) {
    InitSlice(slab + 1);
    if (slab != 0) InitSlab(slab, false);
    SetSliceValues(slab + 1);
    SetSlabIsoVertices(slab);
    SetSliceIso(slab + 1);
    SetSlabIsoEdges(slab);
    FinalizeSlice(slab + 1);
    FinalizeSlab(slab);
    IsoSurface(slab);
  }

  std::vector<long long> triangles;
  for (const auto& p : polygons.polygons) {
    triangles.push_back((long long)p.size());
    for (node_index_type v : p) triangles.push_back((long long)v);
  }
  std::vector<long long> centerIndices;
  std::vector<double> centers;
  for (size_t idx : sink.barycenters) {
    const auto& v = sink.vertices[idx];
    centerIndices.push_back((long long)idx);
    for (int k = 0; k < 3; k++) centers.push_back(v.template get<0>()[k]);
    for (int k = 0; k < 3; k++) centers.push_back(v.template get<1>()[k]);
    centers.push_back(v.template get<2>());
    for (int k = 0; k < 3; k++) centers.push_back(v.template get<3>().template get<0>()[k]);
  }
  PrintI(name + "/vertexcount", {(long long)sink.vertices.size()});
  PrintI(name + "/polygoncount", {(long long)polygons.polygons.size()});
  PrintI(name + "/polygons", triangles);
  PrintI(name + "/polygonbarycenterindices", centerIndices);
  PrintF(name + "/polygonbarycenters", centers);
}

// MinimalAreaTriangulation on crafted polygons (the level-set runs may not reach every size):
// for each size 4..9, twice, NextUnit points; printed are the points, the sizes and, per
// polygon, the triangle count and the triangles.
void RunMat() {
  std::vector<double> points, sizes;
  std::vector<long long> triangles;
  for (int n = 4; n <= 9; n++)
    for (int rep = 0; rep < 2; rep++) {
      std::vector<Point<Real, Dim>> vertices(n);
      for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) vertices[i][k] = NextUnit(), points.push_back(vertices[i][k]);
      sizes.push_back(n);
      std::vector<TriangleIndex<node_index_type>> t = MinimalAreaTriangulation<node_index_type, Real, Dim>((ConstPointer(Point<Real, Dim>))GetPointer(vertices), (node_index_type)vertices.size());
      triangles.push_back((long long)t.size());
      for (const auto& tri : t)
        for (int j = 0; j < 3; j++) triangles.push_back((long long)tri.idx[j]);
    }
  PrintF("mat/points", points);
  PrintI("mat/sizes", std::vector<long long>(sizes.begin(), sizes.end()));
  PrintI("mat/triangles", triangles);
}

}  // namespace

int main() {
  ThreadPool::ParallelizationType = ThreadPool::NONE;
  Run("levelset3", 3, MakeInput(100));
  Run("levelset5", 5, MakeInput(300));
  Run("levelset6", 6, MakeInput(600));
  Run("levelset8", 8, MakeInput(500));
  RunMat();
  return 0;
}
