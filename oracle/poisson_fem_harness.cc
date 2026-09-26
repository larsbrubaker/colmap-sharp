// poisson_fem_harness.cc: PoissonRecon's FEM integrators as Poisson::Solver::Solve
// (thirdparty/PoissonRecon/Reconstructors.h, MIT, as vendored by COLMAP 4.2.0) sets them up,
// with their stencils per depth and raw integrals near the boundary: the normal-divergence
// constraint, the system F( { 0 , 1 } ) and the restriction/prolongation up/down-sample
// stencils. Built and run by oracle/fixture_poisson_tree.py, which writes
// ColmapSharp.Tests/TestData/oracle/poisson_fem.json (read by PoissonTreeOracleTests.Fem* and
// RestrictionProlongation_MatchesHarness). Not part of any build.

#include "poisson_harness.h"

namespace {

// Solve's FEM constraint integrator (the divergence of the normal field against the degree-1
// Neumann basis: FEMIntegrator::Constraint< Sigs , Iso<1> , NormalSigs , Iso<0> , Dim > with
// weights[d][e_d][0] = 1), its same-depth stencil (setStencil<false>) and parent-child
// stencils (setStencils<true>) per depth, and raw cc/pc/cp integrals near the boundary.
void DumpFemConstraint() {
  typedef IsotropicUIntPack<Dim, FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature> Sigs;
  typedef IsotropicUIntPack<Dim, FEMDegreeAndBType<Reconstructor::Poisson::NormalDegree, DerivativeBoundary<BOUNDARY_NEUMANN, 1>::BType>::Signature> NormalSigs;
  typedef typename FEMIntegrator::template Constraint<Sigs, IsotropicUIntPack<Dim, 1>, NormalSigs, IsotropicUIntPack<Dim, 0>, Dim> Constraint;
  typedef typename BaseFEMIntegrator::template Constraint<IsotropicUIntPack<Dim, 1>, IsotropicUIntPack<Dim, 2>, Dim> BaseConstraint;
  Constraint F;
  unsigned int derivatives2[Dim] = {0, 0, 0};
  for (int d = 0; d < (int)Dim; d++) {
    unsigned int derivatives1[Dim];
    for (int dd = 0; dd < (int)Dim; dd++) derivatives1[dd] = dd == d ? 1 : 0;
    F.weights[d][TensorDerivatives<IsotropicUIntPack<Dim, 1>>::Index(derivatives1)][TensorDerivatives<IsotropicUIntPack<Dim, 0>>::Index(derivatives2)] = 1;
  }
  std::vector<double> cc, pc, raw;
  for (int depth = 0; depth <= 6; depth++) {
    typename BaseConstraint::CCStencil stencil;
    typename BaseConstraint::PCStencils stencils;
    F.init(depth);
    F.template setStencil<false>(stencil);
    F.template setStencils<true>(stencils);
    for (unsigned int i = 0; i < stencil.Size(); i++)
      for (int c = 0; c < (int)Dim; c++) cc.push_back(stencil.data[i][c]);
    for (unsigned int s = 0; s < stencils.Size(); s++)
      for (unsigned int i = 0; i < stencils.data[s].Size(); i++)
        for (int c = 0; c < (int)Dim; c++) pc.push_back(stencils.data[s].data[i][c]);
    if (depth == 3) {
      for (int a = -2; a <= 10; a++)
        for (int b = -2; b <= 10; b++) {
          int o1[Dim] = {a, 1, 7}, o2[Dim] = {b, 2, 7};
          Point<double, Dim> v = F.ccIntegrate(o1, o2);
          for (int c = 0; c < (int)Dim; c++) raw.push_back(v[c]);
          int p1[Dim] = {a / 2, 1, 3}, c2[Dim] = {b, 2, 7};
          v = F.pcIntegrate(p1, c2);
          for (int c = 0; c < (int)Dim; c++) raw.push_back(v[c]);
          v = F.cpIntegrate(c2, p1);
          for (int c = 0; c < (int)Dim; c++) raw.push_back(v[c]);
        }
    }
  }
  PrintF("femconstraint/cc", cc);
  PrintF("femconstraint/pc", pc);
  PrintF("femconstraint/raw", raw);
}

// Solve's system integrator F( { 0 , 1 } ) (FEMIntegrator::System< Sigs , Iso<1> >: the
// gradient inner product of the degree-1 Neumann basis), its same-depth stencil
// (setStencil<false>) and parent-child stencils (setStencils<true>) per depth, raw cc/pc
// integrals near the boundary, and vanishesOnConstants.
void DumpFemSystem() {
  typedef IsotropicUIntPack<Dim, FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature> Sigs;
  typedef typename FEMIntegrator::template System<Sigs, IsotropicUIntPack<Dim, 1>> System;
  typedef typename BaseFEMIntegrator::template System<IsotropicUIntPack<Dim, 1>> BaseSystem;
  System F({0., 1.});
  std::vector<double> cc, pc, raw;
  for (int depth = 0; depth <= 6; depth++) {
    typename BaseSystem::CCStencil stencil;
    typename BaseSystem::PCStencils stencils;
    F.init(depth);
    F.template setStencil<false>(stencil);
    F.template setStencils<true>(stencils);
    for (unsigned int i = 0; i < stencil.Size(); i++) cc.push_back(stencil.data[i]);
    for (unsigned int s = 0; s < stencils.Size(); s++)
      for (unsigned int i = 0; i < stencils.data[s].Size(); i++) pc.push_back(stencils.data[s].data[i]);
    if (depth == 3) {
      for (int a = -2; a <= 10; a++)
        for (int b = -2; b <= 10; b++) {
          int o1[Dim] = {a, 0, 7}, o2[Dim] = {b, 1, 7};
          raw.push_back(F.ccIntegrate(o1, o2));
          int p1[Dim] = {a / 2, 0, 3}, c2[Dim] = {b, 1, 7};
          raw.push_back(F.pcIntegrate(p1, c2));
        }
    }
  }
  PrintF("femsystem/cc", cc);
  PrintF("femsystem/pc", pc);
  PrintF("femsystem/raw", raw);
  PrintI("femsystem/vanishesonconstants", {F.vanishesOnConstants() ? 1 : 0});
}

// The restriction/prolongation of the degree-1 Neumann basis (FEMIntegrator::
// RestrictionProlongation< Sigs >): the up-sample stencil and the down-sample stencils per
// depth, and raw up-sample coefficients near the boundary.
void DumpRestrictionProlongation() {
  typedef IsotropicUIntPack<Dim, FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature> Sigs;
  typedef typename FEMIntegrator::template RestrictionProlongation<Sigs> RP;
  typedef typename BaseFEMIntegrator::template RestrictionProlongation<IsotropicUIntPack<Dim, 1>> BaseRP;
  RP rp;
  std::vector<double> up, down, raw;
  for (int depth = 1; depth <= 6; depth++) {
    typename BaseRP::UpSampleStencil upStencil;
    typename BaseRP::DownSampleStencils downStencils;
    rp.init(depth);
    rp.setStencil(upStencil);
    rp.setStencils(downStencils);
    for (unsigned int i = 0; i < upStencil.Size(); i++) up.push_back(upStencil.data[i]);
    for (unsigned int s = 0; s < downStencils.Size(); s++)
      for (unsigned int i = 0; i < downStencils.data[s].Size(); i++) down.push_back(downStencils.data[s].data[i]);
    if (depth == 3) {
      for (int a = -1; a <= 5; a++)
        for (int b = -2; b <= 9; b++) {
          int pOff[Dim] = {a, 0, 2}, cOff[Dim] = {b, 1, 4};
          raw.push_back(rp.upSampleCoefficient(pOff, cOff));
        }
    }
  }
  PrintF("restrictionprolongation/up", up);
  PrintF("restrictionprolongation/down", down);
  PrintF("restrictionprolongation/raw", raw);
}

}  // namespace

int main() {
  DumpFemConstraint();
  DumpFemSystem();
  DumpRestrictionProlongation();
  return 0;
}
