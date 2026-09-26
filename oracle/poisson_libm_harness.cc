// poisson_libm_harness.cc: the platform libm functions PoissonRecon's solve path calls, over
// many float arguments - pow( x , 1./3 ) (the normal transform's scale, PowOneThird.cs) and
// logf (_getSampleDepthAndWeight, PoissonSplat.LogF). Built and run by
// oracle/fixture_poisson_tree.py, which writes ColmapSharp.Tests/TestData/oracle/poisson_libm.json
// (read by PoissonTreeOracleTests.PowOneThird_* and LogF_MatchesLibm). The tables are printed
// in full, never checksummed. Not part of any build.

#include "poisson_harness.h"

int main() {
  // pow: float arguments (as fabs( float determinant ) is) over many magnitudes.
  std::vector<double> xs, ys;
  unsigned int state = 987654321u;
  for (int i = 0; i < 4000; i++) {
    state = state * 1664525u + 1013904223u;
    float mantissa = 1.0f + (float)(state >> 9) / 8388608.0f;
    float x = std::ldexp(mantissa, (int)(i % 161) - 80);
    xs.push_back(x);
    ys.push_back(pow(fabs(x), 1. / Dim));
  }
  PrintF("powonethird/x", xs);
  PrintF("powonethird/y", ys);
  // logf, which _getSampleDepthAndWeight calls on float ratios (log( float ) resolves to the
  // float overload): positive float arguments over many magnitudes.
  std::vector<double> lx, ly;
  for (int i = 0; i < 4000; i++) {
    state = state * 1664525u + 1013904223u;
    float mantissa = 1.0f + (float)(state >> 9) / 8388608.0f;
    float x = std::ldexp(mantissa, (int)(i % 81) - 40);
    lx.push_back(x);
    ly.push_back(log(x));
  }
  PrintF("logf/x", lx);
  PrintF("logf/y", ly);
  return 0;
}
