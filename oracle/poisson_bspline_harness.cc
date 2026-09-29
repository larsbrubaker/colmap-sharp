// poisson_bspline_harness.cc: prints the B-spline machinery of Kazhdan's PoissonRecon
// (thirdparty/PoissonRecon/Polynomial.h and BSplineData.h, MIT, as vendored by COLMAP 4.2.0)
// for the degrees and boundary types the port supports. Built and run by
// oracle/fixture_poisson_bspline.py, which turns the output into
// ColmapSharp.Tests/TestData/oracle/poisson_bspline.json. Not part of any build.
//
// Why a harness and not pycolmap: pycolmap exposes only the whole poisson_meshing call, and
// its PoissonRecon is compiled with -ffast-math (divergence 74). These
// templates are header-only, so compiling them here with -ffp-contract=off gives the exact
// numbers of the arithmetic ColmapSharp performs.
//
// Output: one line per case, "<name> <kind> <values...>", kind i = decimal integer,
// f = C99 hex float (%a, exact). ColmapSharp.Tests/Mvs/PoissonRecon/
// PoissonBSplineOracleTests.cs produces the same case names from the C# port; keep the two in
// step: EveryFixtureCase_HasAProducer checks the case names in both directions.

#include "MyMiscellany.h"
#include "BSplineData.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace PoissonRecon;

namespace {

void PrintF(const std::string& name, const std::vector<double>& values) {
  printf("%s f", name.c_str());
  for (double v : values) printf(" %a", v);
  printf("\n");
}

void PrintI(const std::string& name, const std::vector<long long>& values) {
  printf("%s i", name.c_str());
  for (long long v : values) printf(" %lld", v);
  printf("\n");
}

template <int D>
std::vector<double> Coeffs(const Polynomial<D>& p) {
  return std::vector<double>(p.coefficients, p.coefficients + D + 1);
}

// Polynomial<D>: B-spline pieces, recurrence values, binomials.
template <int D>
void DumpPolynomial() {
  const std::string deg = "deg" + std::to_string(D);
  for (int i = 0; i <= D; i++) {
    PrintF("poly/component/" + deg + "/i" + std::to_string(i), Coeffs(Polynomial<D>::BSplineComponent(i)));
  }
  std::vector<double> values;
  const double xs[] = {0.0, 0.1, 0.25, 0.3137, 0.5, 0.7, 0.9, 1.0};
  for (double x : xs) {
    double v[D + 1];
    Polynomial<D>::BSplineComponentValues(x, v);
    values.insert(values.end(), v, v + D + 1);
  }
  PrintF("poly/componentvalues/" + deg, values);
  int b[D + 1];
  Polynomial<D>::BinomialCoefficients(b);
  PrintI("poly/binomial/" + deg, std::vector<long long>(b, b + D + 1));
}

void DumpPolynomialOps() {
  Polynomial<2> p = Polynomial<2>::BSplineComponent(1);
  PrintF("poly/ops/shift", Coeffs(p.shift(0.3)));
  PrintF("poly/ops/scale", Coeffs(p.scale(0.7)));
  PrintF("poly/ops/scaleshift", Coeffs(p.scale(1. / 3).shift(5. / 3)));
  PrintF("poly/ops/eval", {p(-0.4), p(0.0), p(0.3137), p(1.0), p(2.5)});
  PrintF("poly/ops/integral", {p.integral(-0.2, 1.3), p.integral(0.0, 1.0), p.integral(0.7, 0.1)});
  PrintF("poly/ops/product", Coeffs(p * Polynomial<1>::BSplineComponent(0)));
  PrintF("poly/ops/antiderivative", Coeffs(p.integral()));
  PrintF("poly/ops/derivative", Coeffs(p.derivative()));
  PrintF("poly/ops/muldiv", Coeffs(p * 3.0 / 7));
}

template <unsigned int D>
void DumpSizes() {
  typedef BSplineSupportSizes<D> S;
  PrintI("sizes/deg" + std::to_string(D),
         {S::Inset, S::SupportStart, S::SupportEnd, S::ChildSupportStart, S::ChildSupportEnd,
          S::CornerStart, S::CornerEnd, S::ChildCornerStart, S::ChildCornerEnd, S::BCornerStart,
          S::BCornerEnd, S::ChildBCornerStart, S::ChildBCornerEnd, S::UpSampleStart,
          S::UpSampleEnd, S::DownSample0Start, S::DownSample0End, S::DownSample1Start,
          S::DownSample1End, S::Nodes(0), S::Nodes(3)});
}

template <unsigned int D1, unsigned int D2>
void DumpOverlap() {
  typedef BSplineOverlapSizes<D1, D2> O;
  PrintI("overlap/deg" + std::to_string(D1) + "x" + std::to_string(D2),
         {O::OverlapStart, O::OverlapEnd, O::ChildOverlapStart, O::ChildOverlapEnd,
          O::OverlapSupportStart, O::OverlapSupportEnd, O::ChildOverlapSupportStart,
          O::ChildOverlapSupportEnd, O::ParentOverlap0Start, O::ParentOverlap0End,
          O::ParentOverlap1Start, O::ParentOverlap1End});
}

template <unsigned int D>
std::vector<long long> Flatten(const BSplineElements<D>& e) {
  std::vector<long long> out;
  for (size_t i = 0; i < e.size(); i++)
    for (unsigned int j = 0; j <= D; j++) out.push_back(e[i][j]);
  out.push_back(e.denominator);
  return out;
}

// BSplineElements<D>: construction with each boundary, up-sampling, derivatives.
template <unsigned int D>
void DumpElements() {
  const int resolutions[] = {1, 2, 4, 5};
  for (int b = 0; b < 3; b++)
    for (int res : resolutions)
      for (int off = -2; off <= res + 2; off++) {
        const std::string name = "elements/deg" + std::to_string(D) + "/b" + std::to_string(b) +
                                 "/res" + std::to_string(res) + "/off" + std::to_string(off);
        BSplineElements<D> e(res, off, (BoundaryType)b);
        PrintI(name, Flatten(e));
        BSplineElements<D> up;
        e.upSample(up);
        BSplineElements<D> up2;
        up.upSample(up2);
        PrintI(name + "/up2", Flatten(up2));
        if constexpr (D >= 1) {
          BSplineElements<D - 1> de;
          up.template differentiate<1>(de);
          PrintI(name + "/up/d1", Flatten(de));
        }
      }
}

// BSplineEvaluationData<Sig>: Value, Integral, the up-sampling rows and the evaluators.
template <unsigned int Sig>
void DumpEvaluation() {
  typedef BSplineEvaluationData<Sig> E;
  const std::string sig = "sig" + std::to_string(Sig);
  for (int depth = 0; depth <= 3; depth++) {
    const std::string dep = "/depth" + std::to_string(depth);
    const int res = 1 << depth;
    std::vector<double> values, integrals;
    for (int off = E::Begin(depth) - 1; off <= E::End(depth); off++)
      for (unsigned int d = 0; d <= E::Degree + 1; d++) {
        for (int k = -1; k <= 4 * res + 1; k++) values.push_back(E::Value(depth, off, k / (4.0 * res), d));
        values.push_back(E::Value(depth, off, 0.3137, d));
        values.push_back(E::Value(depth, off, 0.8123, d));
        integrals.push_back(E::Integral(depth, off, -0.5, 2.0, d));
        integrals.push_back(E::Integral(depth, off, 0.1, 0.6, d));
        integrals.push_back(E::Integral(depth, off, 0.3137, 0.8123, d));
        integrals.push_back(E::Integral(depth, off, 0.6, 0.1, d));
      }
    PrintF("value/" + sig + dep, values);
    PrintF("integral/" + sig + dep, integrals);

    typename E::UpSampleEvaluator up;
    up.set(depth);
    std::vector<double> ups;
    for (int p = E::Begin(depth) - 1; p <= E::End(depth); p++)
      for (int c = E::Begin(depth + 1) - 1; c <= E::End(depth + 1); c++) ups.push_back(up.value(p, c));
    PrintF("upsample/" + sig + dep, ups);

    typename E::template CenterEvaluator<E::Degree>::Evaluator center;
    typename E::template CenterEvaluator<E::Degree>::ChildEvaluator childCenter;
    typename E::template CornerEvaluator<E::Degree>::Evaluator corner;
    typename E::template CornerEvaluator<E::Degree>::ChildEvaluator childCorner;
    center.set(depth), childCenter.set(depth), corner.set(depth), childCorner.set(depth);
    std::vector<double> c0, c1, c2, c3;
    for (int f = E::Begin(depth) - 1; f <= E::End(depth); f++)
      for (unsigned int d = 0; d <= E::Degree; d++) {
        for (int c = -1; c <= res + 1; c++) c0.push_back(center.value(f, c, d)), c2.push_back(corner.value(f, c, d));
        for (int c = -1; c <= 2 * res + 1; c++) c1.push_back(childCenter.value(f, c, d)), c3.push_back(childCorner.value(f, c, d));
      }
    PrintF("center/" + sig + dep, c0);
    PrintF("childcenter/" + sig + dep, c1);
    PrintF("corner/" + sig + dep, c2);
    PrintF("childcorner/" + sig + dep, c3);
  }
}

// BSplineIntegrationData<Sig1,Sig2>: same-depth and child integrator tables, and Dot across
// two depths (the interior path up-samples twice).
template <unsigned int Sig1, unsigned int Sig2>
void DumpIntegration() {
  typedef BSplineIntegrationData<Sig1, Sig2> I;
  typedef BSplineEvaluationData<Sig1> E1;
  typedef BSplineEvaluationData<Sig2> E2;
  const unsigned int D1 = I::Degree1, D2 = I::Degree2;
  const std::string pair = "sig" + std::to_string(Sig1) + "x" + std::to_string(Sig2);
  for (int depth = 0; depth <= 3; depth++) {
    typename I::FunctionIntegrator::template Integrator<D1, D2> same;
    typename I::FunctionIntegrator::template ChildIntegrator<D1, D2> child;
    same.set(depth), child.set(depth);
    std::vector<double> s, c;
    for (int o1 = E1::Begin(depth) - 1; o1 <= E1::End(depth); o1++)
      for (unsigned int d1 = 0; d1 <= D1; d1++)
        for (unsigned int d2 = 0; d2 <= D2; d2++) {
          for (int o2 = E2::Begin(depth) - 1; o2 <= E2::End(depth); o2++) s.push_back(same.dot(o1, o2, d1, d2));
          for (int o2 = E2::Begin(depth + 1) - 1; o2 <= E2::End(depth + 1); o2++) c.push_back(child.dot(o1, o2, d1, d2));
        }
    PrintF("integrator/" + pair + "/depth" + std::to_string(depth), s);
    PrintF("childintegrator/" + pair + "/depth" + std::to_string(depth), c);
  }
  std::vector<double> far;
  for (int o1 = E1::Begin(1); o1 < E1::End(1); o1++)
    for (int o2 = E2::Begin(3); o2 < E2::End(3); o2++) {
      far.push_back(I::template Dot<0, 0>(1, o1, 3, o2));
      far.push_back(I::template Dot<D1, D2>(1, o1, 3, o2));
    }
  for (int o1 = E1::Begin(3); o1 < E1::End(3); o1++)
    for (int o2 = E2::Begin(1); o2 < E2::End(1); o2++) far.push_back(I::template Dot<D1, 0>(3, o1, 1, o2));
  PrintF("dotfar/" + pair, far);
}

// BSplineData<Sig,D>::SparseBSplineEvaluator: point values at depths 0..4.
template <unsigned int Sig, unsigned int D>
void DumpSparse() {
  typedef BSplineData<Sig, D> B;
  B data(4);
  const int degree = B::Degree;
  const int left = -BSplineSupportSizes<degree>::SupportStart;
  std::vector<double> values;
  for (int depth = 0; depth <= 4; depth++) {
    const int res = 1 << depth;
    const double ps[] = {0.0, 0.0625, 0.3137, 0.5, 0.8123, 0.99};
    for (double p : ps) {
      const int pIdx = (int)(p * res);
      for (int f = BSplineEvaluationData<Sig>::Begin(depth); f < BSplineEvaluationData<Sig>::End(depth); f++) {
        const int column = pIdx - f + left;
        if (column < 0 || column > degree) continue;
        for (unsigned int d = 0; d <= D; d++) values.push_back(data[depth].value(p, f, d));
      }
    }
  }
  PrintF("sparse/sig" + std::to_string(Sig) + "/d" + std::to_string(D), values);
}

}  // namespace

int main() {
  DumpPolynomial<0>(), DumpPolynomial<1>(), DumpPolynomial<2>(), DumpPolynomial<3>();
  DumpPolynomialOps();
  DumpSizes<0>(), DumpSizes<1>(), DumpSizes<2>();
  DumpOverlap<0, 0>(), DumpOverlap<0, 1>(), DumpOverlap<1, 2>(), DumpOverlap<2, 1>(), DumpOverlap<2, 2>(), DumpOverlap<1, 1>();
  DumpElements<0>(), DumpElements<1>(), DumpElements<2>();
  DumpEvaluation<0>(), DumpEvaluation<1>(), DumpEvaluation<2>();
  DumpEvaluation<3>(), DumpEvaluation<4>(), DumpEvaluation<5>();
  DumpEvaluation<6>(), DumpEvaluation<7>(), DumpEvaluation<8>();
  DumpIntegration<5, 5>(), DumpIntegration<5, 7>(), DumpIntegration<7, 5>(), DumpIntegration<7, 7>();
  DumpIntegration<0, 0>(), DumpIntegration<0, 5>(), DumpIntegration<5, 0>(), DumpIntegration<8, 8>();
  DumpIntegration<3, 3>(), DumpIntegration<4, 4>(), DumpIntegration<6, 7>();
  DumpSparse<5, 1>(), DumpSparse<7, 1>(), DumpSparse<0, 0>(), DumpSparse<8, 2>(), DumpSparse<4, 1>();
  return 0;
}
