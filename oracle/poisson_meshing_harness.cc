// poisson_meshing_harness.cc: COLMAP's PoissonMeshing file function end to end, as upstream
// runs it - src/colmap/mvs/poisson_meshing.cc's argument building and calls, linked against
// the vendored thirdparty/PoissonRecon/PoissonRecon.cpp and SurfaceTrimmer.cpp (MIT, as
// vendored by COLMAP 4.2.0), each compiled as its own translation unit with -DRELEASE as
// COLMAP's CMakeLists.txt does. The thread pool is set as COLMAP sets it for num_threads = 1
// (one thread, ThreadPool::NONE): the single-threaded order the port follows
// (docs/CPP_DIVERGENCES.md, entries 106 and 123).
//
// Usage: harness <depth> <point_weight> <trim> <in.ply> <out.ply>. One case per process:
// PoissonRecon.cpp's and SurfaceTrimmer.cpp's command-line flags are globals that keep their
// "set" state across calls (docs/CPP_DIVERGENCES.md, entry 131). Built and run by
// oracle/fixture_poisson_meshing.py, which stores the output PLYs in
// ColmapSharp.Tests/TestData/oracle/poisson_meshing_exact.json. Not part of any build.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "MultiThreading.h"
#include "PoissonRecon.h"
#include "SurfaceTrimmer.h"

namespace {

int Run(const std::vector<std::string>& args, int (*entry)(int, char**)) {
  std::vector<const char*> argv;
  for (const std::string& arg : args) argv.push_back(arg.c_str());
  return entry((int)argv.size(), const_cast<char**>(argv.data()));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s <depth> <point_weight> <trim> <in.ply> <out.ply>\n", argv[0]);
    return EXIT_FAILURE;
  }
  // The script passes point_weight and trim as Python reprs, which atof reads back exactly.
  const int depth = atoi(argv[1]);
  const double point_weight = atof(argv[2]);
  const double trim = atof(argv[3]);
  const std::string input_path = argv[4], output_path = argv[5];

  PoissonRecon::ThreadPool::SetNumThreads(1);
  PoissonRecon::ThreadPool::ParallelizationType = PoissonRecon::ThreadPool::NONE;

  // poisson_meshing.cc's arguments, with options.color at its default (true); PoissonRecon
  // ignores --colors for PLY input.
  std::vector<std::string> args = {"./poisson_recon", "--in", input_path, "--out", output_path,
                                   "--pointWeight", std::to_string(point_weight), "--depth", std::to_string(depth)};
  if (depth < 5) args.push_back("--fullDepth"), args.push_back(std::to_string(depth));
  args.push_back("--colors");
  if (trim > 0) args.push_back("--density");
  if (Run(args, RunPoissonRecon) != EXIT_SUCCESS) return EXIT_FAILURE;

  if (trim != 0) {
    args = {"./surface_trimmer", "--in", output_path, "--out", output_path, "--trim", std::to_string(trim)};
    if (Run(args, RunSurfaceTrimmer) != EXIT_SUCCESS) return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
