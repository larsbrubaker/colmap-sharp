// poisson_tree_harness.cc: runs the first stage of PoissonRecon's Poisson::Solver::Solve
// (thirdparty/PoissonRecon/Reconstructors.h, MIT, as vendored by COLMAP 4.2.0) - the
// bounding transform (PointExtent::GetXForm), SolutionParameters::testAndSet, the transformed
// sample stream and FEMTreeInitializer::Initialize with Solve's IsValid/Process lambdas -
// then resetNodeIndices, SortedTreeNodes and the neighbor keys, and prints the tree and the
// accumulated samples. Built and run by oracle/fixture_poisson_tree.py, which writes
// ColmapSharp.Tests/TestData/oracle/poisson_tree.json. Not part of any build.
//
// The IsValid/Process lambdas and the parameter set-up are copied from Solve (COLMAP's
// options: depth, fullDepth = depth when depth < 5, everything else default). The auxiliary
// data is a Point<float,3> (the red/green/blue of COLMAP's PLY input; PoissonRecon's
// DynamicFactory stores them as three Reals, so the arithmetic is the same).
//
// Output: "<name> <kind> <values...>", kind i = integer, f = C99 hex float. The inputs are
// generated here from a fixed LCG and printed too, so the C# test replays the same points.

#include "PreProcessor.h"
#include "Reconstructors.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
// reports the first differing chunk and its values. Inputs and the libm tables stay in full.
const size_t kChunk = 256;
const size_t kFullLimit = 1024;

bool KeepFull(const std::string& name) {
  return name.find("/input") != std::string::npos || name.rfind("powonethird/", 0) == 0 || name.rfind("logf/", 0) == 0;
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

template <unsigned int L, unsigned int R>
void DumpNeighbors(const std::string& name, const FEMTreeNode& root, int maxDepth) {
  typedef typename FEMTreeNode::template ConstNeighborKey<IsotropicUIntPack<Dim, L>, IsotropicUIntPack<Dim, R>> Key;
  Key key;
  key.set(maxDepth);
  std::vector<long long> out;
  size_t i = 0;
  root.processNodes([&](const FEMTreeNode* n) {
    if (i++ % 7) return;
    const auto& neighbors = key.getNeighbors(n);
    const unsigned int size = (L + R + 1) * (L + R + 1) * (L + R + 1);
    for (unsigned int k = 0; k < size; k++) {
      const FEMTreeNode* m = neighbors.neighbors.data[k];
      out.push_back(m ? m->nodeData.nodeIndex : -1);
    }
  });
  PrintI(name, out);
}

enum Mode { kTree, kDensity, kConfidence };

// kTree: the tree-stage cases. kDensity: the same stage printed quietly, then Solve's next
// step (setDensityEstimator<1,WeightDegree=2>) and its results. kConfidence: the samples with
// params.confidence set (normal lengths as weights).
void Run(const std::string& name, int depth, int count, Mode mode) {
  const bool density = mode == kDensity;
  std::vector<Sample> input = MakeInput(count);
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
  quiet = mode != kTree;

  // Solve's parameters with COLMAP's options.
  Reconstructor::Poisson::SolutionParameters<Real> params;
  params.depth = depth;
  params.fullDepth = depth < 5 ? depth : 5;
  params.baseDepth = (unsigned int)-1, params.solveDepth = (unsigned int)-1, params.kernelDepth = (unsigned int)-1;
  params.alignDir = Dim - 1;
  params.confidence = mode == kConfidence;

  FEMTree<Dim, Real> tree(MEMORY_ALLOCATOR_BLOCK_SIZE);
  VectorStream pointStream(input);
  std::vector<typename FEMTree<Dim, Real>::PointSample> samples;
  typedef DirectSum<Real, Point<Real, Dim>, DirectSum<Real, Color>> InternalNormalAndAuxData;
  std::vector<InternalNormalAndAuxData> sampleNormalAndAuxData;

  XForm<Real, Dim + 1> modelToUnitCube = XForm<Real, Dim + 1>::Identity();
  pointStream.reset();
  modelToUnitCube = PointExtent::GetXForm<Real, Dim, true, Point<Real, Dim>, Color>(pointStream, Point<Real, Dim>(), Color(), params.scale, params.alignDir) * modelToUnitCube;
  XForm<Real, Dim + 1> unitCubeToModel = modelToUnitCube.inverse();
  pointStream.reset();
  params.template testAndSet<Dim>(unitCubeToModel);

  {
    std::vector<double> x;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) x.push_back(modelToUnitCube(i, j));
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) x.push_back(unitCubeToModel(i, j));
    XForm<Real, Dim> normalXForm = XForm<Real, Dim>(modelToUnitCube).inverse().transpose() * (Real)pow(fabs(modelToUnitCube.determinant()), 1. / Dim);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) x.push_back(normalXForm(i, j));
    PrintF(name + "/xform", x);
    PrintI(name + "/params", {params.depth, params.solveDepth, params.fullDepth, params.baseDepth, params.kernelDepth});
  }

  size_t pointCount;
  {
    Reconstructor::TransformedInputOrientedSampleStream<Real, Dim, Color> _pointStream(modelToUnitCube, pointStream);
    std::vector<node_index_type> nodeToIndexMap;
    auto IsValid = [&](const Point<Real, Dim>& p, const Point<Real, Dim>& n, Color d) {
      Real l = Point<Real, Dim>::SquareNorm(n);
      return l > 0 && std::isfinite(l);
    };
    auto Process = [&](FEMTreeNode& node, const Point<Real, Dim>& p, Point<Real, Dim>& n, Color d) {
      Real l = (Real)Length(n);
      Real weight = params.confidence ? l : (Real)1.;
      n /= l;
      node_index_type nodeIndex = node.nodeData.nodeIndex;
      if (nodeIndex >= (node_index_type)nodeToIndexMap.size()) nodeToIndexMap.resize(nodeIndex + 1, -1);
      node_index_type idx = nodeToIndexMap[nodeIndex];
      if (idx == -1) {
        idx = (node_index_type)samples.size();
        nodeToIndexMap[nodeIndex] = idx;
        samples.resize(idx + 1), samples[idx].node = &node;
        sampleNormalAndAuxData.resize(idx + 1);
        samples[idx].sample = ProjectiveData<Point<Real, Dim>, Real>(p * weight, weight);
        sampleNormalAndAuxData[idx] = InternalNormalAndAuxData(n, DirectSum<Real, Color>(d)) * weight;
      } else {
        samples[idx].sample += ProjectiveData<Point<Real, Dim>, Real>(p * weight, weight);
        sampleNormalAndAuxData[idx] += InternalNormalAndAuxData(n, DirectSum<Real, Color>(d)) * weight;
      }
      return true;
    };
    pointCount = FEMTreeInitializer<Dim, Real>::template Initialize<decltype(IsValid), decltype(Process), Point<Real, Dim>, Color>(tree.spaceRoot(), _pointStream, Point<Real, Dim>(), Color(), params.depth, tree.nodeAllocators.size() ? tree.nodeAllocators[0] : nullptr, tree.initializer(), IsValid, Process);
  }
  PrintI(name + "/pointcount", {(long long)pointCount, (long long)tree.nodeCount()});
  PrintI(name + "/tree", DumpTree(tree.tree()));
  {
    std::vector<double> s;
    for (size_t i = 0; i < samples.size(); i++) {
      s.push_back(samples[i].node->nodeData.nodeIndex);
      for (int k = 0; k < 3; k++) s.push_back(samples[i].sample.data[k]);
      s.push_back(samples[i].sample.weight);
      const Point<Real, Dim>& n = sampleNormalAndAuxData[i].template get<0>();
      const Color& c = sampleNormalAndAuxData[i].template get<1>().template get<0>();
      for (int k = 0; k < 3; k++) s.push_back(n[k]);
      for (int k = 0; k < 3; k++) s.push_back(c[k]);
    }
    if (mode == kConfidence) quiet = false;
    PrintF(name + "/samples", s);
    if (mode == kConfidence) return;
  }

  tree.resetNodeIndices(0, std::make_tuple());
  PrintI(name + "/reset", DumpTree(tree.tree()));
  {
    std::vector<long long> s;
    for (size_t i = 0; i < samples.size(); i++) s.push_back(samples[i].node->nodeData.nodeIndex);
    PrintI(name + "/samplenodes", s);
  }

  if (density) {
    quiet = false;
    const long long nodesBefore = tree.nodeCount();
    auto* estimator = tree.template setDensityEstimator<1, Reconstructor::WeightDegree>(samples, params.kernelDepth, params.samplesPerNode);
    // Per node with an entry, in pre-order: node index, the estimator's data slot and the value.
    std::vector<double> values;
    tree.tree().processNodes([&](const FEMTreeNode* n) {
      const Real* w = (*estimator)(n);
      if (!w) return;
      values.push_back(n->nodeData.nodeIndex), values.push_back(estimator->index(n)), values.push_back(*w);
    });
    PrintF(name + "/density", values);
    PrintI(name + "/densitytree", DumpTree(tree.tree()));
    PrintI(name + "/densityinfo", {(long long)estimator->size(), estimator->kernelDepth(), estimator->coDimension(), nodesBefore, (long long)tree.nodeCount()});

    // Solve's next two steps, copied: the normal field (setInterpolatedDataField with the
    // normalizing ConversionFunction, then the negation) and the auxiliary (color) field
    // (setExtrapolatedDataField without a density, then the per-level scaling).
    {
      typedef IsotropicUIntPack<Dim, FEMDegreeAndBType<Reconstructor::Poisson::NormalDegree, DerivativeBoundary<BOUNDARY_NEUMANN, 1>::BType>::Signature> NormalSigs;
      ProjectiveData<Point<Real, 2>, Real> pointDepthAndWeight;
      std::function<bool(InternalNormalAndAuxData, Point<Real, Dim>&)> ConversionFunction = [](InternalNormalAndAuxData in, Point<Real, Dim>& out) {
        Point<Real, Dim> n = in.template get<0>();
        Real l = (Real)Length(n);
        if (!l) return false;
        out = n / l;
        return true;
      };
      auto normalInfo = tree.setInterpolatedDataField(Point<Real, Dim>(), NormalSigs(), samples, sampleNormalAndAuxData, estimator, params.baseDepth, params.depth, params.lowDepthCutOff, pointDepthAndWeight, ConversionFunction);
      for (size_t i = 0; i < normalInfo.size(); i++) normalInfo[i] *= (Real)-1.;
      std::vector<double> normals;
      tree.tree().processNodes([&](const FEMTreeNode* n) {
        const Point<Real, Dim>* v = normalInfo(n);
        if (!v) return;
        normals.push_back(n->nodeData.nodeIndex), normals.push_back(normalInfo.index(n));
        for (int k = 0; k < 3; k++) normals.push_back((*v)[k]);
      });
      PrintF(name + "/normals", normals);
      PrintF(name + "/pointdepthandweight", {pointDepthAndWeight.data[0], pointDepthAndWeight.data[1], pointDepthAndWeight.weight});
      PrintI(name + "/normaltree", DumpTree(tree.tree()));

      typedef DirectSum<Real, Color> InternalAuxData;
      static const unsigned int DataSig = FEMDegreeAndBType<Reconstructor::DataDegree, BOUNDARY_FREE>::Signature;
      auto PointSampleFunctor = [&](size_t i) -> const typename FEMTree<Dim, Real>::PointSample& { return samples[i]; };
      auto AuxDataSampleFunctor = [&](size_t i) -> const InternalAuxData& { return sampleNormalAndAuxData[i].template get<1>(); };
      auto auxData = tree.template setExtrapolatedDataField<DataSig, false, Reconstructor::WeightDegree, InternalAuxData>(InternalAuxData(Color()), samples.size(), PointSampleFunctor, AuxDataSampleFunctor, (typename FEMTree<Dim, Real>::template DensityEstimator<Reconstructor::WeightDegree>*)nullptr);
      tree.tree().processNodes([&](const FEMTreeNode* n) {
        ProjectiveData<InternalAuxData, Real>* clr = auxData(n);
        if (clr) (*clr) *= (Real)pow((Real)params.perLevelDataScaleFactor, tree.depth(n));
      });
      std::vector<double> colors;
      tree.tree().processNodes([&](const FEMTreeNode* n) {
        const ProjectiveData<InternalAuxData, Real>* c = auxData(n);
        if (!c) return;
        colors.push_back(n->nodeData.nodeIndex), colors.push_back(auxData.index(n));
        for (int k = 0; k < 3; k++) colors.push_back(c->data.template get<0>()[k]);
        colors.push_back(c->weight);
      });
      PrintF(name + "/colors", colors);
      PrintI(name + "/splatinfo", {(long long)normalInfo.size(), (long long)auxData.size(), (long long)tree.nodeCount()});
    }
    delete estimator;
    return;
  }

  const int maxDepth = tree.tree().maxDepth();
  DumpNeighbors<1, 1>(name + "/neighbors11", tree.tree(), maxDepth);
  DumpNeighbors<1, 2>(name + "/neighbors12", tree.tree(), maxDepth);
  DumpNeighbors<2, 2>(name + "/neighbors22", tree.tree(), maxDepth);

  {
    SortedTreeNodes<Dim> sorted;
    std::vector<node_index_type> map;
    sorted.reset(const_cast<FEMTreeNode&>(tree.tree()), map);
    std::vector<long long> s;
    s.push_back(sorted.levels());
    for (int d = 0; d < sorted.levels(); d++)
      for (int slice = 0; slice <= (1 << d); slice++) s.push_back(sorted.begin(d, slice));
    for (node_index_type m : map) s.push_back(m);
    PrintI(name + "/sorted", s);
    PrintI(name + "/sortedtree", DumpTree(tree.tree()));
  }

  // Creating neighbor key: refine around every finest node (in traversal order).
  {
    std::vector<FEMTreeNode*> finest;
    tree.spaceRoot().processNodes([&](FEMTreeNode* n) { if (!n->children && n->depth() == maxDepth) finest.push_back(n); });
    typedef typename FEMTreeNode::template NeighborKey<IsotropicUIntPack<Dim, 1>, IsotropicUIntPack<Dim, 1>> Key;
    Key key;
    key.set(maxDepth);
    for (FEMTreeNode* n : finest) key.template getNeighbors<true, false>(n, tree.nodeAllocators.size() ? tree.nodeAllocators[0] : nullptr, tree.initializer());
    PrintI(name + "/created", DumpTree(tree.tree()));
  }

  // Neighbor-key caching and traversal details the stages above do not reach.
  {
    FEMTreeNode& root = const_cast<FEMTreeNode&>(tree.tree());
    Allocator<FEMTreeNode>* allocator = tree.nodeAllocators.size() ? tree.nodeAllocators[0] : nullptr;
    const int md = root.maxDepth();

    // processNodes with a bool functor that prunes.
    std::vector<long long> pruned;
    root.processNodes([&](const FEMTreeNode* n) {
      pruned.push_back(n->nodeData.nodeIndex);
      return n->depth() < 2 || (n->nodeData.nodeIndex % 3) == 0;
    });
    PrintI(name + "/pruned", pruned);

    // A resetting (non-const) key used without creating nodes, as the density and splat
    // stages' keys are between creations.
    {
      typename FEMTreeNode::template NeighborKey<IsotropicUIntPack<Dim, 1>, IsotropicUIntPack<Dim, 1>> key;
      key.set(md);
      std::vector<long long> out;
      size_t i = 0;
      root.processNodes([&](FEMTreeNode* n) {
        if (i++ % 7) return;
        const auto& w = key.template getNeighbors<false, false>(n, allocator, tree.initializer());
        for (unsigned int k = 0; k < 27; k++) out.push_back(w.neighbors.data[k] ? w.neighbors.data[k]->nodeData.nodeIndex : -1);
      });
      PrintI(name + "/resetkey11", out);
    }

    // getChildNeighbors of every 5th interior node.
    {
      typename FEMTreeNode::template ConstNeighborKey<IsotropicUIntPack<Dim, 1>, IsotropicUIntPack<Dim, 1>> key;
      key.set(md);
      typename FEMTreeNode::template ConstNeighborKey<IsotropicUIntPack<Dim, 1>, IsotropicUIntPack<Dim, 1>>::NeighborType childNeighbors;
      std::vector<long long> out;
      size_t i = 0;
      root.processNodes([&](const FEMTreeNode* n) {
        if (!n->children || n->depth() >= md || i++ % 5) return;
        key.getNeighbors(n);
        for (int c = 0; c < 8; c++) {
          out.push_back(key.getChildNeighbors(c, n->depth(), childNeighbors));
          for (unsigned int k = 0; k < 27; k++) out.push_back(childNeighbors.neighbors.data[k] ? childNeighbors.neighbors.data[k]->nodeData.nodeIndex : -1);
        }
      });
      PrintI(name + "/childneighbors", out);
    }

    // Cached windows across node creation: a resetting key must recompute a window that
    // had nulls (the reset rule), and a const key must drop deeper cached windows when a
    // shallower query moves (the clearing loop).
    {
      typedef typename FEMTreeNode::template NeighborKey<IsotropicUIntPack<Dim, 2>, IsotropicUIntPack<Dim, 2>> Key;
      typedef typename FEMTreeNode::template ConstNeighborKey<IsotropicUIntPack<Dim, 2>, IsotropicUIntPack<Dim, 2>> ConstKey;
      Key resetKey, createKey;
      ConstKey constKey;
      resetKey.set(md), createKey.set(md), constKey.set(md);
      std::vector<FEMTreeNode*> targets;
      size_t i = 0;
      root.processNodes([&](FEMTreeNode* n) {
        if (n->depth() == md && (i++ % 11) == 0 && targets.size() < 12) targets.push_back(n);
      });
      std::vector<long long> out;
      auto dump = [&](const FEMTreeNode* const* data) {
        for (unsigned int k = 0; k < 125; k++) out.push_back(data[k] ? data[k]->nodeData.nodeIndex : -1);
      };
      for (FEMTreeNode* n : targets) {
        dump(resetKey.template getNeighbors<false, false>(n, allocator, tree.initializer()).neighbors.data);
        dump(constKey.getNeighbors(n).neighbors.data);
        createKey.template getNeighbors<true, false>(n, allocator, tree.initializer());
        dump(resetKey.template getNeighbors<false, false>(n, allocator, tree.initializer()).neighbors.data);
        FEMTreeNode* parent = n->parent;
        FEMTreeNode* other = parent->parent->children + (((parent - parent->parent->children) + 1) % 8);
        constKey.getNeighbors(other);
        dump(constKey.getNeighbors(n).neighbors.data);
      }
      PrintI(name + "/cache", out);
      PrintI(name + "/cachetree", DumpTree(tree.tree()));
    }

    // Windows with radii other than the key's (ConstNeighborKey's getNeighbors( UIntPack ,
    // UIntPack , ... ) overloads, which the FEM system uses; NeighborKey's versions of these do
    // not compile upstream, so nothing reaches them): from the key's parent window when half
    // the radii fit in the key's, else recursively; and the pNeighbors form.
    {
      typedef typename FEMTreeNode::template ConstNeighborKey<IsotropicUIntPack<Dim, 1>, IsotropicUIntPack<Dim, 1>> ConstKey;
      ConstKey constKey;
      constKey.set(md);
      typename FEMTreeNode::template ConstNeighbors<IsotropicUIntPack<Dim, 5>> c22, p22;
      typename FEMTreeNode::template ConstNeighbors<UIntPack<4, 4, 4>> c12;
      typename FEMTreeNode::template ConstNeighbors<IsotropicUIntPack<Dim, 7>> c33;
      typename FEMTreeNode::template ConstNeighbors<UIntPack<6, 6, 6>> c32;
      std::vector<long long> out;
      auto dump = [&](const auto& w, unsigned int size) {
        for (unsigned int k = 0; k < size; k++) out.push_back(w.neighbors.data[k] ? w.neighbors.data[k]->nodeData.nodeIndex : -1);
      };
      size_t i = 0;
      root.processNodes([&](const FEMTreeNode* n) {
        if (i++ % 9) return;
        constKey.getNeighbors(IsotropicUIntPack<Dim, 2>(), IsotropicUIntPack<Dim, 2>(), n, c22), dump(c22, 125);
        constKey.getNeighbors(IsotropicUIntPack<Dim, 1>(), IsotropicUIntPack<Dim, 2>(), n, c12), dump(c12, 64);
        constKey.getNeighbors(IsotropicUIntPack<Dim, 3>(), IsotropicUIntPack<Dim, 3>(), n, c33), dump(c33, 343);
        constKey.getNeighbors(IsotropicUIntPack<Dim, 3>(), IsotropicUIntPack<Dim, 2>(), n, c32), dump(c32, 216);
        constKey.getNeighbors(IsotropicUIntPack<Dim, 2>(), IsotropicUIntPack<Dim, 2>(), n, p22, c22), dump(p22, 125), dump(c22, 125);
      });
      PrintI(name + "/radii", out);
    }
  }
}

}  // namespace

int main() {
  ThreadPool::ParallelizationType = ThreadPool::NONE;
  Run("depth5", 5, 300, kTree);
  Run("depth3", 3, 100, kTree);
  Run("density5", 5, 300, kDensity);
  Run("density3", 3, 100, kDensity);
  Run("confidence3", 3, 100, kConfidence);
  // Depth above the full (and so base) depth, as in COLMAP's default runs: samples then
  // splat at fractional depths, into two levels' worth of weight.
  Run("density6", 6, 600, kDensity);

  // libm's pow( x , 1./3 ), which scales the normal transform (PowOneThird.cs): float
  // arguments (as fabs( float determinant ) is) over many magnitudes, then the results.
  {
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
    // logf, which _getSampleDepthAndWeight calls on float ratios (log( float ) resolves to
    // the float overload): positive float arguments over many magnitudes.
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
  }
  return 0;
}
