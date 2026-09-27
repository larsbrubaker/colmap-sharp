// poisson_levelset2_harness.cc: the level-set extractor's slice and slab bookkeeping and corner
// values (thirdparty/PoissonRecon/FEMTree.LevelSet.3D.inl's _LevelSetExtractor< ... , 3 , ... >
// and FEMTree.LevelSet.inl's SliceCellIndexData / SlabCellIndexData, MIT, as vendored by COLMAP
// 4.2.0), as Extract runs them for COLMAP (whole tree: slabDepth 0, slab [0,1), no boundaries):
// the full depth, then, in Extract's slab loop, InitSlice, InitSlab and SetSliceValues
// (SetSliceCornerValuesAndMCIndices). Extract's lambdas are copied here, since they are local
// to it. After each step the cell tables of every slice or slab it set, and after
// SetSliceValues every set corner's value and gradient and every leaf's marching-squares index.
// Built and run by oracle/fixture_poisson_tree.py, which writes
// ColmapSharp.Tests/TestData/oracle/poisson_levelset2.json (read by
// ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.LevelSet2.cs). Not part of any
// build. The set-up and the solve are oracle/poisson_solve.h's PoissonRun.

#include "poisson_solve.h"

namespace {

typedef _LevelSetExtractor<true, Real, Dim, InternalAuxData> Extractor;
static const unsigned int kSig = FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature;

template <typename CellIndices>
void DumpCells(const CellIndices& cells, std::vector<long long>& heads, std::vector<long long>& indices, long long step, long long d, long long o) {
  heads.push_back(step), heads.push_back(d), heads.push_back(o), heads.push_back(cells.nodeOffset), heads.push_back(cells.size());
  for (int k = 0; k < 3; k++) heads.push_back(cells.counts[k]);
  for (size_t i = 0; i < cells.size(); i++) {
    for (int j = 0; j < 4; j++) indices.push_back(std::get<0>(cells.tables)[i][j]);
    for (int j = 0; j < 4; j++) indices.push_back(std::get<1>(cells.tables)[i][j]);
    indices.push_back(std::get<2>(cells.tables)[i][0]);
  }
}

void Run(const std::string& name, int depth, const std::vector<Sample>& input) {
  PoissonRun run;
  run.Prepare(depth, input);
  DenseNodeData<Real, Sigs> solution = run.SolveSystem();
  auto& tree = run.tree;
  const Real isoValue = (Real)0.;  // Any level exercises the same bookkeeping; 0 crosses the shell.

  // Extract's set-up.
  tree._setFEM1ValidityFlags(Sigs());
  LevelSetExtraction::SetHyperCubeTables<Dim>();
  LevelSetExtraction::SetHyperCubeTables<Dim - 1>();
  typename FEMTree<Dim, Real>::LocalOffset start, end;
  for (unsigned int d = 0; d < Dim; d++) start[d] = 0, end[d] = 1;
  int fullDepth = tree.getFullDepth(UIntPack<1, 1, 1>(), 0, start, end);
  unsigned int maxDepth = tree._maxDepth;
  DenseNodeData<Real, Sigs> coarseCoefficients(tree._sNodesEnd(tree._maxDepth - 1));
  memset(coarseCoefficients(), 0, sizeof(Real) * tree._sNodesEnd(tree._maxDepth - 1));
  for (size_t i = tree._sNodesBegin(0); i < (size_t)tree._sNodesEnd(tree._maxDepth - 1); i++) coarseCoefficients[i] = solution[i];
  typename FEMIntegrator::template RestrictionProlongation<Sigs> rp;
  for (int d = 1; d < tree._maxDepth; d++) tree._upSample(Sigs(), rp, d, (ConstPointer(Real))coarseCoefficients() + tree._sNodesBegin(d - 1), coarseCoefficients() + tree._sNodesBegin(d));
  std::vector<typename FEMTree<Dim, Real>::template _Evaluator<Sigs, 1>> evaluators(tree._maxDepth + 1);
  for (int d = 0; d <= tree._maxDepth; d++) evaluators[d].set(tree._maxDepth);
  std::vector<typename Extractor::SlabValues> slabValues(tree._maxDepth + 1);
  PrintI(name + "/fulldepth", {fullDepth});

  std::vector<long long> sliceHeads, sliceIndices, slabHeads, slabIndices, cornerFlags, mcIndices;
  std::vector<double> corners;
  long long step = 0;
  auto InitSlice = [&](unsigned int sliceAtMaxDepth) {
    for (int d = maxDepth; d >= fullDepth; d--) {
      unsigned int dOff = maxDepth - d, slice = sliceAtMaxDepth >> dOff;
      if (sliceAtMaxDepth != (slice << dOff)) break;
      auto& values = slabValues[d].sliceValues(slice);
      values.cellIndices.set(tree._sNodes, tree._localToGlobal(d), slice + tree._localInset(d));
      values.reset(slice, true);
      slabValues[d].sliceScratch(slice).reset(values.cellIndices);
      DumpCells(values.cellIndices, sliceHeads, sliceIndices, step, d, slice);
    }
    step++;
  };
  auto InitSlab = [&](unsigned int slabAtMaxDepth, bool first) {
    unsigned int slab = slabAtMaxDepth;
    for (int d = maxDepth; d >= fullDepth; d--, slab >>= 1) {
      slabValues[d].xSliceValues(slab).cellIndices.set(tree._sNodes, tree._localToGlobal(d), slab + tree._localInset(d));
      slabValues[d].xSliceValues(slab).reset(slab);
      slabValues[d].xSliceScratch(slab).reset(slabValues[d].xSliceValues(slab).cellIndices);
      DumpCells(slabValues[d].xSliceValues(slab).cellIndices, slabHeads, slabIndices, step, d, slab);
      if ((slab & 1) && !first) break;
    }
    step++;
  };
  auto SetSliceValues = [&](unsigned int sliceAtMaxDepth) {
    int d;
    unsigned int o;
    for (d = maxDepth, o = sliceAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      Extractor::template SetSliceCornerValuesAndMCIndices<kSig, kSig, kSig>(tree, solution(), coarseCoefficients(), isoValue, d, fullDepth, o, slabValues, evaluators[d]);
      if (o & 1) break;
    }
    // Every slice just finished: set corners, then each processed leaf's index.
    for (d = maxDepth, o = sliceAtMaxDepth; d >= fullDepth; d--, o >>= 1) {
      auto& values = slabValues[d].sliceValues(o);
      auto& scratch = slabValues[d].sliceScratch(o);
      cornerFlags.push_back(step), cornerFlags.push_back(d), cornerFlags.push_back(o);
      for (size_t v = 0; v < values.cellIndices.counts[0]; v++) {
        cornerFlags.push_back(scratch.cSet[v] ? 1 : 0);
        if (scratch.cSet[v]) {
          corners.push_back(values.cornerValues[v]);
          for (int k = 0; k < 3; k++) corners.push_back(values.cornerGradients[v][k]);
        }
      }
      for (int s = (int)o - 1; s <= (int)o; s++) {
        if (s < 0 || s >= (1 << d)) continue;
        for (node_index_type i = tree._sNodesBegin(d, s); i < tree._sNodesEnd(d, s); i++) {
          const FEMTreeNode* leaf = tree._sNodes.treeNodes[i];
          if (tree._isValidSpaceNode(leaf) && !IsActiveNode<Dim>(leaf->children)) mcIndices.push_back(values.mcIndices[i - values.cellIndices.nodeOffset]);
        }
      }
      if (o & 1) break;
    }
    step++;
  };

  // Extract's slab loop, up to its corner values.
  InitSlice(0);
  InitSlab(0, true);
  SetSliceValues(0);
  for (unsigned int slab = 0; slab < (1u << maxDepth); slab++) {
    InitSlice(slab + 1);
    if (slab != 0) InitSlab(slab, false);
    SetSliceValues(slab + 1);
  }
  PrintI(name + "/slicecells", sliceHeads);
  PrintI(name + "/sliceindices", sliceIndices);
  PrintI(name + "/slabcells", slabHeads);
  PrintI(name + "/slabindices", slabIndices);
  PrintI(name + "/cornerflags", cornerFlags);
  PrintF(name + "/corners", corners);
  PrintI(name + "/mcindices", mcIndices);
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
