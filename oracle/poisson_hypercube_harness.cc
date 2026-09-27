// poisson_hypercube_harness.cc: the level-set extractor's hypercube algebra and tables
// (thirdparty/PoissonRecon/MarchingCubes.h's HyperCube::Cube and FEMTree.LevelSet.inl's
// HyperCubeTables, MIT, as vendored by COLMAP 4.2.0), for every cube dimension the 3D
// extractor sets up (SetHyperCubeTables< 3 > and < 2 >, which recurse down to 1). Built and run
// by oracle/fixture_poisson_tree.py, which writes
// ColmapSharp.Tests/TestData/oracle/poisson_hypercube.json (read by
// ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.HyperCube.cs). Not part of any build.

#include "poisson_harness.h"

namespace {

template <unsigned int D, unsigned int K1, unsigned int K2>
void DumpPair() {
  using T = LevelSetExtraction::HyperCubeTables<D, K1, K2>;
  std::string prefix = "hypercube/d" + std::to_string(D) + "/overlap" + std::to_string(K1) + std::to_string(K2);
  std::vector<long long> overlap, elements;
  for (unsigned int e = 0; e < T::ElementNum1; e++) {
    for (unsigned int e2 = 0; e2 < T::ElementNum2; e2++) overlap.push_back(T::Overlap[e][e2] ? 1 : 0);
    for (unsigned int i = 0; i < T::OverlapElementNum; i++) elements.push_back(T::OverlapElements[e][i].index);
  }
  PrintI(prefix + "/flags", overlap);
  PrintI(prefix + "/elements", elements);
}

template <unsigned int D, unsigned int K>
void DumpSingle() {
  using T = LevelSetExtraction::HyperCubeTables<D, K>;
  using Cube = HyperCube::Cube<D>;
  std::string prefix = "hypercube/d" + std::to_string(D) + "/k" + std::to_string(K);
  std::vector<long long> cellOffset, coIndex, index, antipodalOffset, incident, dirs, antipodal, factor, offsets, mc;
  for (unsigned int e = 0; e < T::ElementNum; e++) {
    typename Cube::template Element<K> element(e);
    for (unsigned int i = 0; i < T::IncidentCubeNum; i++) {
      cellOffset.push_back(T::CellOffset[e][i]);
      coIndex.push_back(T::IncidentElementCoIndex[e][i]);
      index.push_back(T::IncidentElementIndex[e][i]);
      int x[D];
      Cube::CellOffset(element, typename Cube::template IncidentCubeIndex<K>(i), x);
      for (unsigned int d = 0; d < D; d++) offsets.push_back(x[d]);
    }
    antipodalOffset.push_back(T::CellOffsetAntipodal[e]);
    incident.push_back(T::IncidentCube[e].index);
    for (unsigned int d = 0; d < D; d++) dirs.push_back(T::Directions[e][d]);
    antipodal.push_back(element.antipodal().index);
    HyperCube::Direction dir;
    unsigned int co;
    element.factor(dir, co);
    factor.push_back(dir), factor.push_back(co);
    for (unsigned int m = 0; m < (1u << Cube::template ElementNum<0>()); m++) mc.push_back(Cube::ElementMCIndex(element, m));
  }
  PrintI(prefix + "/celloffset", cellOffset);
  PrintI(prefix + "/incidentcoindex", coIndex);
  PrintI(prefix + "/incidentindex", index);
  PrintI(prefix + "/celloffsetantipodal", antipodalOffset);
  PrintI(prefix + "/incidentcube", incident);
  PrintI(prefix + "/directions", dirs);
  PrintI(prefix + "/antipodal", antipodal);
  PrintI(prefix + "/factor", factor);
  PrintI(prefix + "/celloffsetxyz", offsets);
  PrintI(prefix + "/elementmcindex", mc);
  if constexpr (K + 1 == D) {
    std::vector<long long> oriented;
    for (unsigned int e = 0; e < T::ElementNum; e++) oriented.push_back(Cube::IsOriented(typename Cube::template Element<K>(e)) ? 1 : 0);
    PrintI(prefix + "/oriented", oriented);
  }
}

template <unsigned int D, unsigned int K1, unsigned int K2>
void DumpAllPairs() {
  DumpPair<D, K1, K2>();
  if constexpr (K2 < D) DumpAllPairs<D, K1, K2 + 1>();
  else if constexpr (K1 < D) DumpAllPairs<D, K1 + 1, 0>();
}

template <unsigned int D, unsigned int K>
void DumpAllSingles() {
  DumpSingle<D, K>();
  if constexpr (K < D) DumpAllSingles<D, K + 1>();
}

template <unsigned int D>
void DumpCube() {
  std::vector<long long> roots;
  for (unsigned int m = 0; m < (1u << (1u << D)); m++) roots.push_back(HyperCube::Cube<D>::HasMCRoots(m) ? 1 : 0);
  PrintI("hypercube/d" + std::to_string(D) + "/hasmcroots", roots);
  DumpAllPairs<D, 0, 0>();
  DumpAllSingles<D, 0>();
}

}  // namespace

int main() {
  LevelSetExtraction::SetHyperCubeTables<Dim>();
  LevelSetExtraction::SetHyperCubeTables<Dim - 1>();
  DumpCube<1>();
  DumpCube<2>();
  DumpCube<3>();

  // MCIndex over the corner values, with ties at the iso-value (values< iso sets the bit).
  std::vector<long long> mc;
  for (unsigned int m = 0; m < 256; m++) {
    Real values[8];
    for (int c = 0; c < 8; c++) values[c] = (m >> c) & 1 ? (Real)-0.25 : (c % 3 == 0 ? (Real)0.5 : (Real)0.);
    mc.push_back(HyperCube::Cube<3>::MCIndex(values, (Real)0.));
  }
  PrintI("hypercube/mcindex", mc);
  return 0;
}
