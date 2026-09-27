// poisson_levelset_harness.cc: what PoissonRecon's Poisson::Solver::Solve
// (thirdparty/PoissonRecon/Reconstructors.h, MIT, as vendored by COLMAP 4.2.0) does after the
// linear solve: the iso-value, the weighted average of the implicit function at the samples
// (MultiThreadedEvaluator< Sigs , 0 >, FEMTree.Evaluation.inl, with FEMTree.System.inl's
// coarseCoefficients), and the start of the level-set extraction (FEMTree.LevelSet.3D.inl):
// the corner values and gradients (_Evaluator< Sigs , 1 >, _getCornerValues). Built and run by oracle/fixture_poisson_tree.py, which writes
// ColmapSharp.Tests/TestData/oracle/poisson_levelset.json (read by
// ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.LevelSet.cs). Not part of any
// build. The set-up and the solve are oracle/poisson_solve.h's PoissonRun, checked stage by
// stage by oracle/poisson_system_harness.cc; here they print nothing.

#include "poisson_solve.h"

namespace {

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

  PoissonRun run;
  run.Prepare(depth, input);
  DenseNodeData<Real, Sigs> solution = run.SolveSystem();
  auto& samples = run.samples;

  // Solve's "Get the iso-value" block, single-threaded (ThreadPool::NONE: one thread, the
  // samples in order), also recording each weighted sample's value.
  double valueSum = 0, weightSum = 0;
  typename FEMTree<Dim, Real>::template MultiThreadedEvaluator<Sigs, 0> evaluator(&run.tree, solution);
  std::vector<double> valueSums(ThreadPool::NumThreads(), 0), weightSums(ThreadPool::NumThreads(), 0);
  std::vector<double> sampleValues;
  ThreadPool::ParallelFor(0, samples.size(), [&](unsigned int thread, size_t j) {
    ProjectiveData<Point<Real, Dim>, Real>& sample = samples[j].sample;
    Real w = sample.weight;
    if (w > 0) {
      Real value = evaluator.values(sample.data / sample.weight, thread, samples[j].node)[0];
      sampleValues.push_back(value);
      weightSums[thread] += w, valueSums[thread] += value * w;
    }
  });
  for (size_t t = 0; t < valueSums.size(); t++) valueSum += valueSums[t], weightSum += weightSums[t];
  Real isoValue = (Real)(valueSum / weightSum);

  std::vector<double> coarse;
  for (size_t i = 0; i < evaluator._coarseCoefficients.size(); i++) coarse.push_back(evaluator._coarseCoefficients[i]);
  PrintF(name + "/coarsecoefficients", coarse);
  PrintF(name + "/samplevalues", sampleValues);
  PrintF(name + "/isovalue", {valueSum, weightSum, isoValue});

  // The level-set extractor's corner evaluation (SetSliceCornerValuesAndMCIndices with corner
  // gradients, as nonLinearFit asks): _getCornerValues< Real , 1 > through a
  // ConstCornerSupportKey and _Evaluator< Sigs , 1 > at every corner of every leaf of every
  // depth, in sorted-node order, with isInterior from the leaf's parent. Value, then gradient.
  {
    typename FEMTree<Dim, Real>::template _Evaluator<Sigs, 1> cornerEvaluator;
    cornerEvaluator.set(run.tree._maxDepth);
    const auto& coarse = evaluator._coarseCoefficients;
    std::vector<double> corners;
    for (int d = 0; d <= run.tree._maxDepth; d++) {
      ConstCornerSupportKey<IsotropicUIntPack<Dim, 1>> key;
      key.set(run.tree._localToGlobal(run.tree._maxDepth));
      for (node_index_type i = run.tree._sNodesBegin(d); i < run.tree._sNodesEnd(d); i++) {
        const FEMTreeNode* leaf = run.tree._sNodes.treeNodes[i];
        if (!run.tree._isValidSpaceNode(leaf) || IsActiveNode<Dim>(leaf->children)) continue;
        bool isInterior = run.tree._isInteriorlySupported(IsotropicUIntPack<Dim, 1>(), leaf->parent);
        key.getNeighbors(leaf);
        for (int c = 0; c < 8; c++) {
          CumulativeDerivativeValues<Real, Dim, 1> p = run.tree.template _getCornerValues<Real, 1>(key, leaf, c, solution(), coarse(), cornerEvaluator, run.tree._maxDepth, isInterior);
          for (int k = 0; k < 4; k++) corners.push_back(p[k]);
        }
      }
    }
    PrintF(name + "/cornervalues", corners);
  }
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
