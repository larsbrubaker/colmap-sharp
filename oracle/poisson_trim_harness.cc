// poisson_trim_harness.cc: COLMAP's mesh trimming step end to end - upstream's own
// thirdparty/PoissonRecon/SurfaceTrimmer.cpp (MIT, as vendored by COLMAP 4.2.0), included
// whole and run through RunSurfaceTrimmer with exactly the arguments
// src/colmap/mvs/poisson_meshing.cc passes: --in, --out and --trim std::to_string( trim ), so
// --aRatio keeps its 0.001 default (the island merge runs), --polygonMesh is off (the
// output is triangulated by MAT.h's MinimalAreaTriangulation), --removeIslands, --long and
// --ascii are off (the output keeps the input's binary format). Its input is the
// extracted mesh of oracle/poisson_extract_harness.cc (the same runs, extracted the same way),
// written as the PLY PoissonRecon's WriteMesh writes for COLMAP: float x, y, z and value
// (the density), uchar red, green, blue (the input fused.ply's color properties, converted by
// PlyFile.inl's get_stored_item / write_binary_item), and a uchar-counted int face list. The
// trimmer's output PLY is read back and printed per run and trim value: vertex and triangle
// counts, positions, values, color bytes and triangles ("levelset3" in full through
// poisson_harness.h's KeepFull on "/mesh/", the others as checksums). A crafted grid mesh
// (Crafted, printed in full as "crafted/input") adds the island merges the extracted meshes
// never reach: a small component with two neighbors, where libc++'s hash-container order
// decides the output triangle order. Built and run by
// oracle/fixture_poisson_tree.py, which writes ColmapSharp.Tests/TestData/oracle/poisson_trim.json
// (read by ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.Trim.cs). Not part of any
// build.

#include "poisson_solve.h"

#include "SurfaceTrimmer.cpp"

namespace {

typedef LevelSetExtractor<Real, Dim, InternalAuxData> PublicExtractor;

struct ModelVertexSink : public Reconstructor::OutputLevelSetVertexStream<Real, Dim, InternalAuxData> {
  std::vector<Point<Real, Dim>> positions;
  std::vector<Real> values;
  std::vector<InternalAuxData> data;
  size_t size(void) const { return positions.size(); }
  size_t write(const Point<Real, Dim>& p, const Point<Real, Dim>&, const Real& w, const InternalAuxData& d) {
    positions.push_back(p), values.push_back(w), data.push_back(d);
    return positions.size() - 1;
  }
};

struct PolygonSink : public OutputDataStream<std::vector<node_index_type>> {
  std::vector<std::vector<node_index_type>> polygons;
  size_t size(void) const { return polygons.size(); }
  size_t write(const std::vector<node_index_type>& p) {
    polygons.push_back(p);
    return polygons.size() - 1;
  }
};

// A Real color channel as PLY::Write stores it in a uchar property.
unsigned char PlyUChar(Real value) {
  int int_val;
  unsigned int uint_val;
  long long longlong_val;
  unsigned long long ulonglong_val;
  double double_val;
  get_stored_item(&value, PLY_FLOAT, int_val, uint_val, longlong_val, ulonglong_val, double_val);
  FILE* fp = tmpfile();
  write_binary_item(fp, PLY_BINARY_NATIVE, int_val, uint_val, longlong_val, ulonglong_val, double_val, PLY_UCHAR);
  rewind(fp);
  unsigned char byte = 0;
  if (fread(&byte, 1, 1, fp) != 1) MK_THROW("Failed to read back the PLY byte");
  fclose(fp);
  return byte;
}

struct Mesh {
  std::vector<float> positions, values;
  std::vector<unsigned char> colors;
  std::vector<int> triangles;
};

void WritePly(const std::string& path, const Mesh& mesh) {
  FILE* fp = fopen(path.c_str(), "wb");
  size_t n = mesh.values.size(), m = mesh.triangles.size() / 3;
  fprintf(fp, "ply\nformat binary_little_endian 1.0\nelement vertex %zu\n", n);
  fprintf(fp, "property float x\nproperty float y\nproperty float z\nproperty float value\n");
  fprintf(fp, "property uchar red\nproperty uchar green\nproperty uchar blue\n");
  fprintf(fp, "element face %zu\nproperty list uchar int vertex_indices\nend_header\n", m);
  for (size_t v = 0; v < n; v++) {
    fwrite(&mesh.positions[3 * v], sizeof(float), 3, fp);
    fwrite(&mesh.values[v], sizeof(float), 1, fp);
    fwrite(&mesh.colors[3 * v], 1, 3, fp);
  }
  for (size_t t = 0; t < m; t++) {
    unsigned char three = 3;
    fwrite(&three, 1, 1, fp);
    fwrite(&mesh.triangles[3 * t], sizeof(int), 3, fp);
  }
  fclose(fp);
}

// Reads the trimmer's output, which must have the input's vertex layout and triangle faces.
Mesh ReadPly(const std::string& path) {
  FILE* fp = fopen(path.c_str(), "rb");
  std::string header;
  char line[1024];
  size_t n = 0, m = 0;
  while (fgets(line, sizeof(line), fp)) {
    std::string l(line);
    header += l;
    sscanf(line, "element vertex %zu", &n);
    sscanf(line, "element face %zu", &m);
    if (l == "end_header\n") break;
  }
  const std::string expected =
      "property float x\nproperty float y\nproperty float z\nproperty float value\n"
      "property uchar red\nproperty uchar green\nproperty uchar blue\nelement face";
  if (header.find(expected) == std::string::npos || header.find("binary_little_endian") == std::string::npos)
    MK_THROW("Unexpected trimmer output header:\n", header);
  bool intCount = header.find("property list int int") != std::string::npos;
  if (!intCount && header.find("property list uchar int") == std::string::npos) MK_THROW("Unexpected face list:\n", header);
  Mesh mesh;
  mesh.positions.resize(3 * n), mesh.values.resize(n), mesh.colors.resize(3 * n), mesh.triangles.resize(3 * m);
  for (size_t v = 0; v < n; v++) {
    if (fread(&mesh.positions[3 * v], sizeof(float), 3, fp) != 3 || fread(&mesh.values[v], sizeof(float), 1, fp) != 1 ||
        fread(&mesh.colors[3 * v], 1, 3, fp) != 3)
      MK_THROW("Short vertex read");
  }
  for (size_t t = 0; t < m; t++) {
    int count = 0;
    if (intCount) {
      if (fread(&count, sizeof(int), 1, fp) != 1) MK_THROW("Short face read");
    } else {
      unsigned char c;
      if (fread(&c, 1, 1, fp) != 1) MK_THROW("Short face read");
      count = c;
    }
    if (count != 3) MK_THROW("Not a triangle");
    if (fread(&mesh.triangles[3 * t], sizeof(int), 3, fp) != 3) MK_THROW("Short face read");
  }
  fclose(fp);
  return mesh;
}

void PrintMesh(const std::string& prefix, const Mesh& mesh) {
  PrintI(prefix + "/vertexcount", {(long long)mesh.values.size()});
  PrintI(prefix + "/trianglecount", {(long long)(mesh.triangles.size() / 3)});
  PrintF(prefix + "/positions", std::vector<double>(mesh.positions.begin(), mesh.positions.end()));
  PrintF(prefix + "/values", std::vector<double>(mesh.values.begin(), mesh.values.end()));
  PrintI(prefix + "/colors", std::vector<long long>(mesh.colors.begin(), mesh.colors.end()));
  PrintI(prefix + "/triangles", std::vector<long long>(mesh.triangles.begin(), mesh.triangles.end()));
}

// Extracts as poisson_extract_harness.cc does and returns the mesh WriteMesh would write.
Mesh Extract(int depth, const std::vector<Sample>& input) {
  PoissonRun run;
  run.Prepare(depth, input);
  DenseNodeData<Real, Sigs> solution = run.SolveSystem();
  auto& tree = run.tree;
  const Real perLevelDataScaleFactor = run.params.perLevelDataScaleFactor;
  tree.tree().processNodes([&](const FEMTreeNode* n) {
    ProjectiveData<InternalAuxData, Real>* clr = run.auxData(n);
    if (clr) (*clr) *= (Real)pow((Real)perLevelDataScaleFactor, tree.depth(n));
  });
  double valueSum = 0, weightSum = 0;
  {
    typename FEMTree<Dim, Real>::template MultiThreadedEvaluator<Sigs, 0> evaluator(&tree, solution);
    for (size_t j = 0; j < run.samples.size(); j++) {
      ProjectiveData<Point<Real, Dim>, Real>& sample = run.samples[j].sample;
      Real w = sample.weight;
      if (w > 0) {
        Real value = evaluator.values(sample.data / sample.weight, 0, run.samples[j].node)[0];
        weightSum += w, valueSum += value * w;
      }
    }
  }
  const Real isoValue = (Real)(valueSum / weightSum);
  ModelVertexSink sink;
  PolygonSink polygons;
  Reconstructor::TransformedOutputLevelSetVertexStream<Real, Dim, InternalAuxData> vertexStream(run.unitCubeToModel, sink);
  const InternalAuxData zeroData = InternalAuxData(Color());
  PublicExtractor::Extract(Sigs(), UIntPack<Reconstructor::WeightDegree>(), UIntPack<DataSig>(), tree, run.estimator, &run.auxData, solution, isoValue, vertexStream, polygons, zeroData, true, false, true, false, false);

  Mesh mesh;
  for (size_t v = 0; v < sink.positions.size(); v++) {
    for (int k = 0; k < 3; k++) mesh.positions.push_back(sink.positions[v][k]);
    mesh.values.push_back(sink.values[v]);
    for (int k = 0; k < 3; k++) mesh.colors.push_back(PlyUChar(sink.data[v].template get<0>()[k]));
  }
  for (const auto& p : polygons.polygons)
    for (node_index_type v : p) mesh.triangles.push_back((int)v);
  return mesh;
}

// A crafted mesh: a wavy grid whose density is 6 plus noise, except around a few vertices
// where a high center (7) is ringed by its six neighbors just below 5. Trimmed at 5, each
// ring is a thin below-trim annulus between two kept components (the center's and the rest),
// so it is the small component with two neighbors that the merge folds into its first
// neighbor - the one the hash containers' iteration order puts first, which decides where its
// polygons, and so the output triangles, go. Trimmed at 6 the noise leaves many small
// islands. Printed in full as its input.
Mesh Crafted() {
  const int n = 24;
  Mesh mesh;
  unsigned long long state = 12345;
  auto noise = [&]() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return (double)(state >> 11) / (double)(1ull << 53) - 0.5;
  };
  for (int y = 0; y < n; y++)
    for (int x = 0; x < n; x++) {
      mesh.positions.push_back((float)(x * 0.5));
      mesh.positions.push_back((float)(y * 0.5));
      mesh.positions.push_back((float)(0.3 * sin(0.7 * x) * cos(0.4 * y)));
      mesh.values.push_back((float)(6 + 0.3 * sin(0.8 * x) * cos(0.6 * y) + 0.3 * noise()));
      mesh.colors.push_back((unsigned char)(x * 10)), mesh.colors.push_back((unsigned char)(y * 10)), mesh.colors.push_back((unsigned char)((x * y) % 256));
    }
  const int centers[][2] = {{6, 6}, {12, 12}, {17, 8}, {8, 18}};
  const int ring[][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 0}, {-1, -1}, {0, -1}};
  for (const auto& c : centers) {
    mesh.values[c[1] * n + c[0]] = 7.f;
    for (int k = 0; k < 6; k++) mesh.values[(c[1] + ring[k][1]) * n + c[0] + ring[k][0]] = (float)(4.99 - 0.001 * k);
  }
  for (int y = 0; y + 1 < n; y++)
    for (int x = 0; x + 1 < n; x++) {
      int a = y * n + x, b = a + 1, c = a + n, d = c + 1;
      mesh.triangles.insert(mesh.triangles.end(), {a, b, d, a, d, c});
    }
  return mesh;
}

void Run(const std::string& name, Mesh mesh, const std::vector<double>& trims) {
  float lo = mesh.values[0], hi = mesh.values[0];
  for (float v : mesh.values) lo = std::min(lo, v), hi = std::max(hi, v);
  PrintF(name + "/mesh/valuerange", {lo, hi});
  char dir[] = "/tmp/poisson_trim_XXXXXX";
  if (!mkdtemp(dir)) MK_THROW("mkdtemp failed");
  const std::string in = std::string(dir) + "/in.ply", out = std::string(dir) + "/out.ply";
  WritePly(in, mesh);
  PrintF(name + "/mesh/trims", trims);
  for (size_t t = 0; t < trims.size(); t++) {
    // poisson_meshing.cc's argument list, trim formatted by std::to_string.
    std::vector<std::string> args = {"./surface_trimmer", "--in", in, "--out", out, "--trim", std::to_string(trims[t])};
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    // The command-line parameters are globals: reset what an earlier run set.
    Trim.set = false;
    if (RunSurfaceTrimmer((int)argv.size(), argv.data()) != EXIT_SUCCESS) MK_THROW("RunSurfaceTrimmer failed");
    PrintMesh(name + "/mesh/trim" + std::to_string(t), ReadPly(out));
  }
  remove(in.c_str()), remove(out.c_str()), rmdir(dir);
}

void Run(const std::string& name, int depth, const std::vector<Sample>& input, const std::vector<double>& trims) {
  Run(name, Extract(depth, input), trims);
}

}  // namespace

int main() {
  ThreadPool::ParallelizationType = ThreadPool::NONE;
  // COLMAP's default trim (10, above every density here, so everything is trimmed) and two
  // values inside each run's density range, which split polygons and leave islands to merge.
  Run("levelset3", 3, MakeInput(100), {10.0, 2.6, 2.7, 2.85});
  Run("levelset5", 5, MakeInput(300), {10.0, 3.9, 4.2});
  Run("levelset6", 6, MakeInput(600), {10.0, 4.3, 4.8});
  Run("levelset8", 8, MakeInput(500), {10.0, 4.5, 5.5});
  Mesh crafted = Crafted();
  PrintMesh("crafted/input", crafted);
  Run("crafted", crafted, {5.0, 5.2, 6.0});
  return 0;
}
