// poisson_extract_harness.cc: the level-set extraction end to end as COLMAP's poisson_meshing
// reaches it - thirdparty/PoissonRecon/Reconstructors.h's Implicit::extractLevelSet (MIT, as
// vendored by COLMAP 4.2.0): upstream's own LevelSetExtractor< Real , 3 , Data >::Extract
// (FEMTree.LevelSet.3D.inl, the whole tree, nonLinearFit on, outputGradients off,
// addBarycenter = forceManifold on, polygonMesh and flipOrientation off, with the density
// estimator and the color field, as --density and --colors ask) writing through
// Reconstructors.streams.h's TransformedOutputLevelSetVertexStream with Solve's
// unitCubeToModel (modelToUnitCube.inverse()); then what PoissonRecon.cpp's WriteMesh keeps of
// each vertex for COLMAP (OutputVertexInfo< ... , HasGradients false , HasDensity true ,
// DynamicFactory >: position, value = the density weight, the color channels) with each color
// converted to the input PLY's uchar as PlyFile.inl's get_stored_item (PLY_FLOAT) and
// write_binary_item (PLY_UCHAR) convert it. Output per run: unitCubeToModel, the vertex and
// triangle counts, the positions, values, color bytes and triangles ("levelset3" in full,
// poisson_harness.h's KeepFull; the others as checksums), and for the crafted "vertexpairs*"
// runs, which reach the level set's vertex-pair branches, their input first (the other runs'
// inputs are poisson_levelset.json's); then those two PLY functions alone on
// crafted floats ("plycolor"), outside [0, 256) included. Built and run by
// oracle/fixture_poisson_tree.py, which writes ColmapSharp.Tests/TestData/oracle/poisson_extract.json
// (read by ColmapSharp.Tests/Mvs/PoissonRecon/PoissonTreeOracleTests.Extract.cs). Not part of
// any build. The set-up and the solve are oracle/poisson_solve.h's PoissonRun; the iso-value
// and the color scaling are Solve's, as oracle/poisson_levelset6_harness.cc computes them.

#include "poisson_solve.h"

namespace {

typedef LevelSetExtractor<Real, Dim, InternalAuxData> PublicExtractor;

// The model-side vertex stream: what TransformedOutputLevelSetVertexStream passes on.
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

// A Real color channel written to a uchar PLY property: PLY::Write stores the vertex's Real
// (internal type PLY_FLOAT) and writes the file's type through these two functions.
long long PlyUChar(Real value) {
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
  return (long long)byte;
}

void Run(const std::string& name, int depth, const std::vector<Sample>& input, bool printInput = false) {
  if (printInput) {
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
  auto& tree = run.tree;

  // Solve's per-level scaling of the color field.
  const Real perLevelDataScaleFactor = run.params.perLevelDataScaleFactor;
  tree.tree().processNodes([&](const FEMTreeNode* n) {
    ProjectiveData<InternalAuxData, Real>* clr = run.auxData(n);
    if (clr) (*clr) *= (Real)pow((Real)perLevelDataScaleFactor, tree.depth(n));
  });

  // Solve's iso-value, single-threaded.
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
  PrintF(name + "/isovalue", {isoValue});

  std::vector<double> xform;
  for (int i = 0; i < Dim + 1; i++)
    for (int j = 0; j < Dim + 1; j++) xform.push_back(run.unitCubeToModel(i, j));
  PrintF(name + "/mesh/unitcubetomodel", xform);

  // extractLevelSet: the transformed stream over the output, then Extract.
  ModelVertexSink sink;
  PolygonSink polygons;
  Reconstructor::TransformedOutputLevelSetVertexStream<Real, Dim, InternalAuxData> vertexStream(run.unitCubeToModel, sink);
  const InternalAuxData zeroData = InternalAuxData(Color());
  PublicExtractor::Extract(Sigs(), UIntPack<Reconstructor::WeightDegree>(), UIntPack<DataSig>(), tree, run.estimator, &run.auxData, solution, isoValue, vertexStream, polygons, zeroData, true, false, true, false, false);

  std::vector<double> positions, values;
  std::vector<long long> colors, triangles;
  for (size_t v = 0; v < sink.positions.size(); v++) {
    for (int k = 0; k < 3; k++) positions.push_back(sink.positions[v][k]);
    values.push_back(sink.values[v]);
    for (int k = 0; k < 3; k++) colors.push_back(PlyUChar(sink.data[v].template get<0>()[k]));
  }
  for (const auto& p : polygons.polygons) {
    if (p.size() != 3) MK_THROW("Not a triangle");
    for (node_index_type v : p) triangles.push_back((long long)v);
  }
  PrintI(name + "/mesh/vertexcount", {(long long)sink.positions.size()});
  PrintI(name + "/mesh/trianglecount", {(long long)polygons.polygons.size()});
  PrintF(name + "/mesh/positions", positions);
  PrintF(name + "/mesh/values", values);
  PrintI(name + "/mesh/colors", colors);
  PrintI(name + "/mesh/triangles", triangles);
}

// The "vertexpairs*" inputs: where a coarse leaf's edge borders finer leaves and the surface
// crosses both of its halves, the two iso-vertices are paired and the leaf's loop walks from
// one to the other. Each input restarts the generator (the seed is part of the input), and
// each reaches different pair branches (see PoissonTreeOracleTests.Extract.cs); the colors
// are a function of the index so they draw nothing from it.
Color IndexColor(size_t i) { return Color((float)(i * 37 % 256), (float)(i * 101 % 256), (float)(i * 199 % 256)); }

// A few samples on each of three small spheres outside the shell.
void AddSphere(std::vector<Sample>& samples, float cx, float cy, float cz, float r, int count) {
  for (int i = 0; i < count; i++) {
    Sample s;
    float u = NextUnit() * 6.2831853f, v = NextUnit() * 3.1415927f;
    // __sincosf_stret for the same reason as MakeInput (poisson_harness.h).
    __float2 su = __sincosf_stret(u), sv = __sincosf_stret(v);
    s.n = Point<Real, Dim>(su.__cosval * sv.__sinval, su.__sinval * sv.__sinval, sv.__cosval);
    s.p = Point<Real, Dim>(cx, cy, cz) + s.n * r;
    s.c = IndexColor(samples.size());
    samples.push_back(s);
  }
}

// Two parallel sheets 2 * halfGap apart, facing away from each other.
void AddSheets(std::vector<Sample>& samples, float z, float halfGap, int count) {
  for (int i = 0; i < count; i++) {
    Sample s;
    float x = 1.5f + 3.f * NextUnit(), y = -1.9f + 1.8f * NextUnit();
    bool top = NextUnit() < 0.5f;
    s.p = Point<Real, Dim>(x, y, top ? z + halfGap : z - halfGap);
    s.n = Point<Real, Dim>(0.f, 0.f, top ? 1.f : -1.f);
    s.c = IndexColor(samples.size());
    samples.push_back(s);
  }
}

// The shell alone, denser than the leaves at depth 6 resolve.
std::vector<Sample> MakeVertexPairShell() {
  lcg = 1177u;
  return MakeInput(400);
}

// The shell with three small spheres, 60 samples each.
std::vector<Sample> MakeVertexPairSpheres() {
  lcg = 1059u;
  std::vector<Sample> samples = MakeInput(150);
  AddSphere(samples, 1.6f, -1.8f, 1.0f, 0.03f, 60);
  AddSphere(samples, 4.4f, -0.2f, 1.0f, 0.03f, 60);
  AddSphere(samples, 4.5f, -1.9f, 0.0f, 0.03f, 60);
  return samples;
}

// A thin slab: two sheets 0.02 apart.
std::vector<Sample> MakeVertexPairSheets() {
  lcg = 955u;
  std::vector<Sample> samples;
  AddSheets(samples, 0.5f, 0.01f, 400);
  return samples;
}

// The PLY uchar conversion on crafted floats: in range, fractions, 256 and above (the low byte
// of the unsigned int), negative (0 on arm64, where the unsigned conversion saturates) and at or
// beyond 2^32 (saturating to 255 there).
void RunPlyColor() {
  const std::vector<Real> in = {0.f, 0.25f, 0.999f, 1.f, 127.5f, 254.99f, 255.f, 255.75f, 256.f, 256.5f, 300.f, 511.9f, 65535.5f, 1e6f, -0.25f, -1.f, -300.f, 4294967040.f, 4294967296.f, 1e10f, 1e30f};
  std::vector<double> x(in.begin(), in.end());
  std::vector<long long> y;
  for (Real v : in) y.push_back(PlyUChar(v));
  PrintF("plycolor/in", x);
  PrintI("plycolor/out", y);
}

}  // namespace

int main() {
  ThreadPool::ParallelizationType = ThreadPool::NONE;
  Run("levelset3", 3, MakeInput(100));
  Run("levelset5", 5, MakeInput(300));
  Run("levelset6", 6, MakeInput(600));
  Run("levelset8", 8, MakeInput(500));
  Run("vertexpairs1", 6, MakeVertexPairShell(), true);
  Run("vertexpairs2", 7, MakeVertexPairSpheres(), true);
  Run("vertexpairs3", 7, MakeVertexPairSheets(), true);
  RunPlyColor();
  return 0;
}
