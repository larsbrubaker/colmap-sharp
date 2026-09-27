// poisson_system_harness.cc: the system-assembly steps of PoissonRecon's
// Poisson::Solver::Solve (thirdparty/PoissonRecon/Reconstructors.h, MIT, as vendored by
// COLMAP 4.2.0) after finalizeForMultigrid: addFEMConstraints (the divergence of the normal
// field, FEMTree.System.inl's _addFEMConstraints) and addInterpolationConstraints
// (_addInterpolationConstraints), then the solver's per-depth matrix rows and prolongation
// constraints (_getSliceMatrixAndProlongationConstraints, _getProlongedMatrixRowSize) and its
// point-constraint transfers (_setPointValuesFromProlongedSolution,
// _updateRestrictedInterpolationConstraints) and the sliced Gauss-Seidel relaxation
// (_solveSystemGS), and the base-depth multigrid's sparse algebra (downSampleMatrix, transpose,
// multiply, setDiagonalR, the Galerkin products R * M * P) and its solve (_solveRegularMG),
// and Solve's cascadic solveSystem. Built and run by
// oracle/fixture_poisson_tree.py, which writes ColmapSharp.Tests/TestData/oracle/poisson_system.json
// (read by ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.System.cs). Not part of any
// build. The set-up up to finalizeForMultigrid is oracle/poisson_solve.h's PoissonRun and the
// output format is in oracle/poisson_harness.h; the stages up to finalizeForMultigrid are
// checked step by step by oracle/poisson_tree_harness.cc, so here they run through the vendored
// functions and print nothing.

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
  FEMTree<Dim, Real>& tree = run.tree;
  const Reconstructor::Poisson::SolutionParameters<Real>& params = run.params;
  auto* normalInfo = run.normalInfo;
  InterpolationInfo* iInfo = run.iInfo;
  PrintI(name + "/sortedslices", [&] {
    std::vector<long long> s;
    for (int d = 0; d < tree._sNodes.levels(); d++) s.push_back(tree._sNodes.begin(d)), s.push_back(tree._sNodes.end(d));
    return s;
  }());

  // Solve's Poisson constraints: addFEMConstraints( F , *normalInfo , constraints , solveDepth )
  // with the divergence integrator; also with a shallower maxDepth (coarser levels only), which
  // Solve does not use but which takes _addFEMConstraints' early-depth paths.
  const int solveDepth = params.depth;
  auto F = run.DivergenceIntegrator();
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

    // The solver's point-constraint transfers: _setPointValuesFromProlongedSolution at every
    // depth (rewriting the entries' dual values from the prolonged solution), then
    // _updateRestrictedInterpolationConstraints at every depth from a synthetic solution into
    // one restricted-constraint array.
    for (int d = 1; d <= tree._maxDepth; d++) tree.template _setPointValuesFromProlongedSolution<0>(d, bsData, (const Real*)&prolonged[0], std::make_tuple(iInfo));
    {
      typedef typename FEMTree<Dim, Real>::template ApproximatePointInterpolationInfo<Real, 0, Reconstructor::Poisson::ConstraintDual<Dim, Real>, Reconstructor::Poisson::SystemDual<Dim, Real>> Approximate;
      const auto& iData = static_cast<Approximate*>(iInfo)->iData;
      std::vector<double> out;
      tree.tree().processNodes([&](const FEMTreeNode* n) {
        const auto* e = iData(n);
        if (e) out.push_back(n->nodeData.nodeIndex), out.push_back(e->dualValues[0]);
      });
      PrintF(name + "/prolongedpointvalues", out);
    }
    std::vector<Real> solution(tree._sNodesEnd(tree._maxDepth)), restricted(tree._sNodesEnd(tree._maxDepth - 1));
    for (size_t i = 0; i < solution.size(); i++) solution[i] = (Real)((long long)(i * 53 % 97) - 48) / (Real)32;
    for (int d = 1; d <= tree._maxDepth; d++) tree.template _updateRestrictedInterpolationConstraints<0>(bsData, d, (const Real*)&solution[0], &restricted[0], std::make_tuple(iInfo));
    PrintF(name + "/restrictedinterpolation", std::vector<double>(restricted.begin(), restricted.end()));

    // _solveSystemGS (sliced) at each small depth from a zero solution, against the constraints
    // above and the synthetic prolonged solution: Solve's prolongation-phase call (8 iterations,
    // coarse to fine, one slice per block), and a restriction-direction call with two slices
    // per block and 3 iterations, so the blocked nBegin/nEnd sub-ranges and the window walk in
    // both directions are covered.
    struct UnitSOR { Real operator[](node_index_type) const { return (Real)1; } };
    auto gs = [&](const std::string& caseName, int iters, bool coarseToFine, unsigned int sliceBlockSize) {
      std::vector<Real> x(tree._sNodesEnd(tree._maxDepth), (Real)0);
      std::vector<double> out;
      for (int d = 1; d <= tree._maxDepth; d++) {
        node_index_type begin = tree._sNodesBegin(d), end = tree._sNodesEnd(d);
        if ((size_t)(end - begin) > kSliceLimit) continue;
        S.init(d);
        typename FEMTree<Dim, Real>::_SolverStats stats;
        tree._solveSystemGS(Sigs(), true, S, bsData, d, &x[0], (const Real*)&prolonged[0], (const Real*)constraints(), [](Real v, Real w) { return v * w; }, iters, coarseToFine, sliceBlockSize, UnitSOR(), stats, false, std::make_tuple(iInfo));
        for (node_index_type i = begin; i < end; i++) out.push_back(x[i]);
      }
      PrintF(name + "/" + caseName, out);
    };
    gs("gsprolongation", 8, true, 1);
    gs("gsblocked", 3, false, 2);

    // The base-depth multigrid's sparse algebra at each small depth: downSampleMatrix R, its
    // transpose P, R * x, P * y added to z, and setDiagonalR of systemMatrix.
    {
      typedef SparseMatrix<Real, matrix_index_type> Matrix;
      std::vector<long long> shape;
      std::vector<double> entries, products, diagonals;
      auto dumpMatrix = [&](const Matrix& m) {
        shape.push_back((long long)m.rows());
        for (size_t i = 0; i < m.rows(); i++) {
          shape.push_back((long long)m.rowSize(i));
          for (size_t j = 0; j < m.rowSize(i); j++) shape.push_back(m[i][j].N), entries.push_back(m[i][j].Value);
        }
      };
      for (int d = 1; d <= tree._maxDepth; d++) {
        size_t high = tree._sNodesSize(d), low = tree._sNodesSize(d - 1);
        if (high > kSliceLimit) continue;
        Matrix R = tree.downSampleMatrix(Sigs(), d);
        Matrix P = R.transpose(high);
        dumpMatrix(R), dumpMatrix(P);
        std::vector<Real> x(high), y(low), z(high);
        for (size_t i = 0; i < high; i++) x[i] = (Real)((long long)(i * 29 % 83) - 41) / (Real)16, z[i] = (Real)((long long)(i * 7 % 13) - 6) / (Real)8;
        R.multiply(&x[0], &y[0]);
        P.multiply(&y[0], &z[0], MULTIPLY_ADD);
        for (Real v : y) products.push_back(v);
        for (Real v : z) products.push_back(v);
        Matrix M = tree.systemMatrix(Sigs(), S, d, std::make_tuple(iInfo));
        std::vector<Real> D(M.rows());
        M.setDiagonalR(&D[0]);
        for (Real v : D) diagonals.push_back(v);
      }
      PrintI(name + "/sparseshape", shape);
      PrintF(name + "/sparseentries", entries);
      PrintF(name + "/sparseproducts", products);
      PrintF(name + "/sparsediagonals", diagonals);

      // _solveRegularMG's Galerkin chain: M[baseDepth] = systemMatrix( baseDepth ), then
      // M[d-1] = R[d-1] * M[d] * P[d-1], each row in the unordered_map order it was built in.
      std::vector<long long> galerkinShape{tree._baseDepth};
      std::vector<double> galerkinEntries;
      Matrix M = tree.systemMatrix(Sigs(), S, tree._baseDepth, std::make_tuple(iInfo));
      for (int d = tree._baseDepth; d > 0; d--) {
        Matrix R = tree.downSampleMatrix(Sigs(), d);
        Matrix P = R.transpose(M.rows());
        M = R * M * P;
        galerkinShape.push_back((long long)M.rows());
        for (size_t i = 0; i < M.rows(); i++) {
          galerkinShape.push_back((long long)M.rowSize(i));
          for (size_t j = 0; j < M.rowSize(i); j++) galerkinShape.push_back(M[i][j].N), galerkinEntries.push_back(M[i][j].Value);
        }
      }
      PrintI(name + "/galerkinshape", galerkinShape);
      PrintF(name + "/galerkinentries", galerkinEntries);

      // _solveRegularMG from a zero solution against the constraints above: Solve's call (one
      // V-cycle, 8 sweeps, cgAccuracy = the float 1e-3) and one with two V-cycles, 3 sweeps and
      // no sweeps at the base depth (maxSolveDepth = baseDepth - 1). Dumps the base level.
      auto regularMG = [&](const std::string& caseName, int maxSolveDepth, int vCycles, int iters) {
        std::vector<Real> x(tree._sNodesEnd(tree._maxDepth), (Real)0);
        typename FEMTree<Dim, Real>::_SolverStats stats;
        tree._solveRegularMG(Sigs(), S, bsData, maxSolveDepth, &x[0], (const Real*)constraints(), [](Real v, Real w) { return v * w; }, vCycles, iters, stats, false, (double)(Real)1e-3, std::make_tuple(iInfo));
        std::vector<double> out;
        for (node_index_type i = tree._sNodesBegin(tree._baseDepth); i < tree._sNodesEnd(tree._baseDepth); i++) out.push_back(x[i]);
        PrintF(name + "/" + caseName, out);
      };
      regularMG("regularmg", tree._baseDepth, 1, 8);
      regularMG("regularmgshallow", tree._baseDepth - 1, 2, 3);
    }

    // Solve's linear solve: solveSystem( Sigs , F , constraints , baseDepth , solveDepth ,
    // _sInfo , iInfo ) with Solve's SolverInfo, run with maxSolveDepth capped at each depth from
    // the base depth up: that is the solution after each depth of the full cascadic solve (the
    // depths are solved coarse to fine, each from the ones before).
    {
      typename FEMTree<Dim, Real>::SolverInfo sInfo = run.SolverInfo();
      for (int k = (int)params.baseDepth; k <= solveDepth; k++) {
        DenseNodeData<Real, Sigs> solution = tree.solveSystem(Sigs(), S, constraints, params.baseDepth, k, sInfo, std::make_tuple(iInfo));
        std::vector<double> out;
        for (size_t i = 0; i < solution.size(); i++) out.push_back(solution[i]);
        PrintF(name + "/solve" + std::to_string(k), out);
      }
    }
  }

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
