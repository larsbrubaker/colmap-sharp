// poisson_harness.h: what the PoissonRecon harnesses share - the vendored headers
// (thirdparty/PoissonRecon, MIT, as vendored by COLMAP 4.2.0) with their private and protected
// members opened, the output format, and the generated input points. Included by
// oracle/poisson_tree_harness.cc, oracle/poisson_fem_harness.cc and
// oracle/poisson_libm_harness.cc, which oracle/fixture_poisson_tree.py builds and runs. Not
// part of any build.
//
// Output: "<name> <kind> <values...>", kind i = integer, f = C99 hex float, c = checksums (see
// PrintChunks). The inputs are generated here from a fixed LCG and printed too, so the C#
// tests replay the same points.

#pragma once

// Every standard and system header PoissonRecon pulls in (the non-Windows, non-optional ones:
// PreProcessor.h + Reconstructors.h, checked with clang++ -H), included before the #define
// below so the redefinition only reaches PoissonRecon's own code, never the standard library.
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <float.h>
#include <fstream>
#include <functional>
#include <future>
#include <inttypes.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mach/mach.h>
#include <math.h>
#include <memory>
#include <mutex>
#include <ostream>
#include <setjmp.h>
#include <shared_mutex>
#include <sstream>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// The finalize stage is monolithic upstream (_finalizeForMultigrid); to compare its steps
// one by one the harness calls FEMTree's protected members directly.
#define protected public
#define private public
#include "PreProcessor.h"
#include "Reconstructors.h"
#undef private
#undef protected

using namespace PoissonRecon;

namespace {

typedef float Real;
static const unsigned int Dim = 3;
typedef Point<Real, 3> Color;
typedef RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type> FEMTreeNode;

// Density runs rebuild the tree stage silently and print only their own cases.
bool quiet = false;

// Long dumps are printed as checksums to keep the fixture small: kind "c", then the value
// count, 1 for floats / 0 for integers, and one FNV-1a hash (masked to 52 bits, so the JSON
// number is exact) per chunk of 256 values, over each value's 64 bits (the integer, or the
// double's bit pattern). The C# test hashes its own values the same way and, on a mismatch,
// reports the first differing chunk and its values. Inputs, the libm tables and the level-set
// vertex colors (so a color mismatch can be located) and the smallest run of the iso-edges
// (poisson_levelset5_harness.cc, so a case can be read in full) stay in full.
const size_t kChunk = 256;
const size_t kFullLimit = 1024;

bool KeepFull(const std::string& name) {
  return name.find("/input") != std::string::npos || name.find("/vertexcolors") != std::string::npos || name.rfind("powonethird/", 0) == 0 || name.rfind("logf/", 0) == 0 ||
         name.rfind("levelset3/isoedges/", 0) == 0;
}

void PrintChunks(const std::string& name, const std::vector<unsigned long long>& bits, bool isFloat) {
  printf("%s c %zu %d", name.c_str(), bits.size(), isFloat ? 1 : 0);
  for (size_t start = 0; start < bits.size(); start += kChunk) {
    unsigned long long h = 14695981039346656037ull;
    for (size_t i = start; i < bits.size() && i < start + kChunk; i++)
      for (int b = 0; b < 8; b++) h = (h ^ ((bits[i] >> (8 * b)) & 0xFF)) * 1099511628211ull;
    printf(" %lld", (long long)(h & ((1ull << 52) - 1)));
  }
  printf("\n");
}

void PrintF(const std::string& name, const std::vector<double>& values) {
  if (quiet) return;
  if (values.size() > kFullLimit && !KeepFull(name)) {
    std::vector<unsigned long long> bits(values.size());
    for (size_t i = 0; i < values.size(); i++) memcpy(&bits[i], &values[i], sizeof(double));
    return PrintChunks(name, bits, true);
  }
  printf("%s f", name.c_str());
  for (double v : values) printf(" %a", v);
  printf("\n");
}

void PrintI(const std::string& name, const std::vector<long long>& values) {
  if (quiet) return;
  if (values.size() > kFullLimit && !KeepFull(name)) {
    std::vector<unsigned long long> bits(values.begin(), values.end());
    return PrintChunks(name, bits, false);
  }
  printf("%s i", name.c_str());
  for (long long v : values) printf(" %lld", v);
  printf("\n");
}

struct Sample {
  Point<Real, Dim> p, n;
  Color c;
};

struct VectorStream : public Reconstructor::InputOrientedSampleStream<Real, Dim, Color> {
  const std::vector<Sample>& samples;
  size_t next = 0;
  explicit VectorStream(const std::vector<Sample>& s) : samples(s) {}
  void reset(void) { next = 0; }
  bool read(Point<Real, Dim>& p, Point<Real, Dim>& n, Color& c) {
    if (next >= samples.size()) return false;
    p = samples[next].p, n = samples[next].n, c = samples[next].c;
    next++;
    return true;
  }
};

unsigned int lcg = 12345u;
float NextUnit() {  // [0,1) with 24 bits
  lcg = lcg * 1664525u + 1013904223u;
  return (float)(lcg >> 8) / 16777216.0f;
}

std::vector<Sample> MakeInput(int count) {
  std::vector<Sample> samples;
  for (int i = 0; i < count; i++) {
    Sample s;
    // A noisy ellipsoid shell in an off-center, anisotropic box.
    float u = NextUnit() * 6.2831853f, v = NextUnit() * 3.1415927f, r = 1.0f + 0.05f * NextUnit();
    s.p = Point<Real, Dim>(3.0f + 2.0f * r * std::cos(u) * std::sin(v), -1.0f + r * std::sin(u) * std::sin(v),
                           0.5f + 0.7f * r * std::cos(v));
    s.n = Point<Real, Dim>(std::cos(u) * std::sin(v), 2.0f * std::sin(u) * std::sin(v), std::cos(v) / 0.7f);
    s.c = Color((float)(int)(NextUnit() * 256), (float)(int)(NextUnit() * 256), (float)(int)(NextUnit() * 256));
    if (i % 37 == 5) s.n = Point<Real, Dim>();                          // zero normal: invalid
    if (i % 53 == 7) s.n = Point<Real, Dim>(NAN, 0.f, 1.f);            // non-finite: invalid
    if (i % 41 == 3 && i > 0) s.p = samples[i - 1].p;                   // coincident with the previous
    samples.push_back(s);
  }
  return samples;
}

std::vector<long long> DumpTree(const FEMTreeNode& root) {
  std::vector<long long> out;
  root.processNodes([&](const FEMTreeNode* n) {
    int d, off[Dim];
    n->depthAndOffset(d, off);
    out.push_back(d), out.push_back(off[0]), out.push_back(off[1]), out.push_back(off[2]);
    out.push_back(n->nodeData.nodeIndex);
  });
  return out;
}

}  // namespace
