// poisson_isoroot_harness.cc: the iso-vertex root on crafted edges, where the extraction runs
// rarely or never reach. Two parts:
// - "solutions": thirdparty/PoissonRecon/Polynomial.inl's Polynomial< 1 > and Polynomial< 2 >
//   getSolutions with Factor.h's linear and quadratic Factor (MIT, as vendored by COLMAP 4.2.0),
//   called directly: a real pair, a double root, a complex pair (dropped at EPS 0, kept at a
//   larger EPS), a vanishing (or sub-EPS) leading coefficient falling through to the linear
//   Factor, and a vanishing linear coefficient (no root).
// - "averageroot": the root block of FEMTree.LevelSet.3D.inl's GetIsoVertex (both forms share
//   it; nonLinearFit on), copied verbatim into AverageRoot below because it is inline in
//   GetIsoVertex: the Hermite quadratic's roots in [0, 1] averaged (one root, two roots), the
//   linear fallback (no root in [0, 1], and the NaN scale of end derivatives that sum to zero),
//   the clamp at either end counted as a bad root, and the throw when the ends are equal.
// Output per case: "solutions/<i>" the root count then the roots; "averageroot/<i>" the root,
// whether it counted as bad, and whether it threw. Built and run by oracle/fixture_poisson_tree.py,
// which writes ColmapSharp.Tests/TestData/oracle/poisson_isoroot.json (read by
// ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.IsoRoot.cs). Not part of any build.

#include "poisson_harness.h"

namespace {

// GetIsoVertex's root, lines "double averageRoot;" to the clamp, with sValues' corner values
// and gradients replaced by the edge's values x0, x1 and end derivatives dx0, dx1 (the corner
// gradients times the edge length, as GetIsoVertex forms them).
double AverageRoot(Real isoValue, Real x0, Real x1, double dx0, double dx1, bool& bad) {
  bool nonLinearFit = true;
  bad = false;
  double averageRoot;
  bool rootFound = false;
  if (nonLinearFit) {
    // The scaling will turn the Hermite Spline into a quadratic
    double scl = (x1 - x0) / ((dx1 + dx0) / 2);
    dx0 *= scl, dx1 *= scl;

    // Hermite Spline
    Polynomial<2> P;
    P.coefficients[0] = x0;
    P.coefficients[1] = dx0;
    P.coefficients[2] = 3 * (x1 - x0) - dx1 - 2 * dx0;

    double roots[2];
    int rCount = 0, rootCount = P.getSolutions(isoValue, roots, 0);
    averageRoot = 0;
    for (int i = 0; i < rootCount; i++)
      if (roots[i] >= 0 && roots[i] <= 1) averageRoot += roots[i], rCount++;
    if (rCount) rootFound = true;
    averageRoot /= rCount;
  }
  if (!rootFound) {
    if (x0 == x1) MK_THROW("Not a zero-crossing root: ", x0, " ", x1);
    averageRoot = (isoValue - x0) / (x1 - x0);
  }
  if (averageRoot <= 0 || averageRoot >= 1) {
    bad = true;
    if (averageRoot < 0) averageRoot = 0;
    if (averageRoot > 1) averageRoot = 1;
  }
  return averageRoot;
}

void Solutions() {
  // c0, c1, c2, c, EPS for Polynomial< 2 >.
  const double quadratic[][5] = {
      {1, -3, 2, 0, 0},            // two real roots, 0.5 and 1
      {1, 2, 1, 0, 0},             // a double root
      {1, 0, 1, 0, 0},             // complex pair, dropped at EPS 0
      {1, 0, 1, 0, 2},             // leading coefficient under EPS: linear, with no root
      {1, 0, 4, 0, 1},             // complex pair, kept when |imag| <= EPS
      {-1, 4, -3, 0, 0},           // negative leading coefficient
      {3, 2, 0, 1, 0},             // leading coefficient 0: the linear Factor
      {3, 0, 0, 1, 0},             // linear coefficient 0 too: no root
      {3, 2, 1e-12, 1, 1e-10},     // leading coefficient under EPS: the linear Factor
      {5, 1e-12, 0, 1, 1e-10},     // linear coefficient under EPS: no root
      {0.3, -1.7, 2.9, 0.1, 0},    // an irrational pair
      {0.25, 0.1, 0.3, 0.25, 0},   // a root at exactly 0
      {0.1, 0.7, -0.35, 0.4, 0},   // a real pair after subtracting c
      {0.1, 0.7, 0.5, -0.3, 0},    // a complex pair after subtracting c
  };
  int i = 0;
  for (const auto& q : quadratic) {
    Polynomial<2> p;
    for (int k = 0; k < 3; k++) p.coefficients[k] = q[k];
    double roots[2];
    int count = p.getSolutions(q[3], roots, q[4]);
    std::vector<double> out = {(double)count};
    for (int k = 0; k < count; k++) out.push_back(roots[k]);
    PrintF("solutions/quadratic" + std::to_string(i++), out);
  }

  // c0, c1, c, EPS for Polynomial< 1 >.
  const double linear[][4] = {{1, 2, 0, 0}, {1, 0, 0, 0}, {1, 1e-12, 0, 1e-10}, {-0.3, 0.7, 0.2, 0}};
  i = 0;
  for (const auto& l : linear) {
    Polynomial<1> p;
    p.coefficients[0] = l[0], p.coefficients[1] = l[1];
    double roots[2];
    int count = p.getSolutions(l[2], roots, l[3]);
    std::vector<double> out = {(double)count};
    for (int k = 0; k < count; k++) out.push_back(roots[k]);
    PrintF("solutions/linear" + std::to_string(i++), out);
  }
}

void AverageRoots() {
  // isoValue, x0, x1 (floats), dx0, dx1.
  const double edges[][5] = {
      {0.1, -0.3, 0.7, 0.9, 1.4},     // one root inside
      {0, -0.2, 0.05, 0.01, 0.6},     // a skewed crossing
      {0, -1, 1, 2, 2},               // equal slopes: the quadratic is linear
      {1, 0, 1, 4, -2},               // roots 1/3 and 1, averaged
      {1, -1, 1, 2, 2},               // root exactly at the far end: clamped, bad
      {-1, -1, 1, 2, 2},              // root exactly at the near end: bad
      {0, -1, 1, 1, -1},              // slopes summing to 0: NaN scale, linear fallback
      {0, -1, 1, 0, 0},               // zero slopes: NaN scale, linear fallback
      {2, 0, 1, 1, 1},                // level above both ends: linear root 2, clamped to 1
      {-1, 0, 1, 1, 1},               // level below both ends: linear root -1, clamped to 0
      {0.2, 0.1, 0.9, -3, 5},         // over- and undershooting slopes
      {0, 1, 1, 0, 0},                // equal ends: throws
  };
  int i = 0;
  for (const auto& e : edges) {
    bool bad = false, threw = false;
    double root = 0;
    try {
      root = AverageRoot((Real)e[0], (Real)e[1], (Real)e[2], e[3], e[4], bad);
    } catch (const std::exception&) {
      threw = true;
    }
    PrintF("averageroot/" + std::to_string(i++), {root, bad ? 1.0 : 0.0, threw ? 1.0 : 0.0});
  }
}

}  // namespace

int main() {
  Solutions();
  AverageRoots();
  return 0;
}
