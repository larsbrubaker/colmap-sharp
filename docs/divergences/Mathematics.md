# Divergences from COLMAP: Mathematics: random numbers, graphs and libm

Part of the divergence log: `docs/CPP_DIVERGENCES.md` is the index and explains the
numbering. Entries are in ascending number; each says what differs, why, and the evidence.

## 1. Real-valued random draws are not fused (no FMA), unlike the macOS pycolmap wheel

**What differs.** `RandomUtils.RandomUniformReal` (for `min != 0`), `RandomGaussian` and
`NormalDistribution` evaluate libc++'s `(b - a) * u + a`, `u*u + v*v` and
`x * stddev + mean` as separate multiply and add. Apple clang's default `-ffp-contract=on`
fuses these into single-rounding FMAs on arm64, so a libc++ build with default flags can
differ from ColmapSharp in the last ulp of those draws. Integer draws, `RandomUniformReal`
with `min == 0`, the shuffles and the mt19937 words are unaffected.

**Why.** CLAUDE.md's "No FMA" rule: ColmapSharp's results must be identical on every
platform and .NET JIT never contracts, while contraction in the C++ build is a compiler and
flags choice, not COLMAP behavior. Matching it would mean `Math.FusedMultiplyAdd` at exactly
the sites a particular clang version chose to fuse.

**Evidence.** `oracle/fixture_random.py` builds `oracle/random_harness.cc` with
`-ffp-contract=off` and with the default; the draws above differ between the two builds
(recorded as `contracted_cases` in `ColmapSharp.Tests/TestData/oracle/random.json`), and
the C# port matches the `-ffp-contract=off` build bit for bit (`RandomOracleTests`). The
pycolmap 4.2.0 wheel's `_core` disassembles (`otool -tv`) to about 36,000 `fmadd`/`fmsub`
family instructions, so it is built with contraction on and most likely produces the
contracted numbers. Downstream, a pycolmap fixture that depends on real-valued draws (e.g.
`synthesize_dataset` noise) can therefore differ in the last ulp and is compared at Tier B
or C, not Tier A.

## 2. Hash-container iteration order in UnionFind and connected components

**What differs.** `UnionFind.Parents` enumerates in insertion order. `FindConnectedComponents`
returns components ordered by their first node in the caller's node enumeration, each listing
its nodes in that order, and `FindLargestConnectedComponent` breaks ties between equally large
components the same way. COLMAP's orders come from `boost::unordered_node_map` /
`unordered_flat_set` (`colmap/util/hash_containers.h`).

**Why.** Reproducing them would mean porting Boost.Unordered's bucket layout and
`boost::hash` mixing for every key type, and Boost is not ported (`docs/LICENSE_AUDIT.md`).
The order is not part of COLMAP's contract: `connected_components_test.cc` compares with
`UnorderedElementsAre`, and `union_find_test.cc` never inspects it. Insertion order is
deterministic and the same on every platform.

**Evidence.** The component *sets*, and every root `Find` returns for the same call sequence,
are identical (ColmapSharp.Tests/Mathematics/UnionFindTests.cs, ConnectedComponentsTests.cs).
Callers that need a reproducible component order should pass nodes in a deterministic order
(for example sorted ids).

## 3. Spanning-tree ties between equal edge weights

**What differs.** `ComputeMaximumSpanningTree` / `ComputeMinimumSpanningTree` run Kruskal with
edges sorted by (cost, input index). COLMAP calls `boost::kruskal_minimum_spanning_tree`,
which pops edges from a `std::priority_queue`, so among equal costs it takes them in heap
order. With tied weights the two can choose different (equally optimal) trees.

**Why.** Boost is not ported; the input-index tie-break is the deterministic, documented rule.
The cost transform itself (`max_weight - w` in float for the maximum tree) is kept, so the
same weights tie on both sides.

**Evidence.** With distinct costs the minimum spanning tree is unique and the parents match
exactly (SpanningTreeTests).

## 4. Stoer-Wagner: which min cut, and which side is labeled 1

**What differs.** `ComputeMinGraphCutStoerWagner` is written from Stoer & Wagner (JACM 1997)
instead of calling `boost::stoer_wagner_min_cut`. The cut weight is the global minimum on
both sides, but when several cuts share it the one reported can differ, and the side labeled
1 is the set of vertices merged into the last vertex of the best phase (Boost's parity map
labels its own choice).

**Why.** Boost is not ported. COLMAP's contract is the minimum weight plus a 0/1 label per
vertex; `graph_cut_test.cc` checks only the weight and the label range.

**Evidence.** The ported tests pass, and GraphCutCrossCheckTests compares the weight and the
labeled cut against exhaustive search on 200 random graphs.

## 5. Boykov-Kolmogorov max-flow with float capacities

**What differs.** `MinSTGraphCut` is written from Boykov & Kolmogorov (PAMI 2004) instead of
calling `boost::boykov_kolmogorov_max_flow`, and it pushes each node's direct
source -> node -> sink flow before the search starts. Both change the order in which flow is
augmented. With integer capacities that order cannot show in the result; with float
capacities (`MinSTGraphCut<float>`, as Delaunay meshing uses it) the returned flow is summed
in a different order, so it can differ from COLMAP's by rounding, and a node whose residual
path to a terminal is only a rounding residue can be labeled on the other side of the cut.

**Why.** Boost is not ported (`docs/LICENSE_AUDIT.md`), and Kolmogorov's own maxflow library
is GPL/research-only, so neither is transcribed. Reproducing Boost's exact float rounding would
mean reproducing its exact augmentation order. The terminal-capacity handling is needed for
scale: storing terminal links as ordinary terminal out-edges made each augmentation rescan
them, which was quadratic on Delaunay-sized graphs.

**Evidence.** For integer capacities the result is Tier A: GraphCutCrossCheckTests compares
the flow and the labels with exhaustive search on 300 random graphs, and the sink-side labels
equal the unique minimal sink-side min cut, which is the same for every maximum flow and so
for Boost too. MinSTGraphCutScalingTests checks, for a 200k-node float grid, that the labeled
cut's capacity equals the returned flow within 1e-3 relative.

## 77. ComputeNormalizedMinGraphCut partitions with our own multilevel bisection, not METIS

**What differs.** COLMAP's `ComputeNormalizedMinGraphCut` (math/graph_cut.cc) calls
`METIS_PartGraphKway` with default options. The port (`Mathematics/GraphCut.cs`) builds the
same CSR graph (vertex indices by first appearance, parallel edges kept) and hands it to
`Mathematics/MultilevelPartitioner.cs`, written here from the published multilevel scheme
(Hendrickson & Leland 1995; Karypis & Kumar, SIAM J. Sci. Comput. 1998; Fiduccia &
Mattheyses 1982): heavy-edge-matching coarsening (followed, when over 10% of the vertices
stay unmatched, by pairing unmatched vertices that share a neighbor and unmatched isolated
vertices, so stars, hub images and many small components still coarsen), greedy graph
growing from eight seeds on the coarsest graph (a vertex too heavy to fit is skipped), FM refinement at every level, and recursive bisection for k parts
(floor(k/2) parts on the first side). METIS's k-way path instead refines all k parts at once
with its own greedy k-way refinement and randomizes its visit orders with GKlib's RNG. So the
labels, which part gets which number, and the exact cut can differ from COLMAP's. Each
bisection allows 3% over its target weight (METIS's k-way `ufactor` default of 30), rounded
up to a whole vertex so that small graphs can always be split. Every tie is broken by vertex
index, so the output is deterministic. Self-loops are ignored (they never cross a cut).

**Why.** Porting the reached METIS subset (coarsening, recursive-bisection initial
partitioning, 2-way and k-way FM refinement, balancing, GKlib's priority queues and RNG) is
well over the ~3k-line budget set for this step, and a close match would also need GKlib's
random stream reproduced exactly. COLMAP's contract, and all its callers need (scene
clustering), is a balanced partition with a small cut; graph_cut_test.cc checks the label
range, that both parts are used, and the component split of a disconnected graph. No METIS
code was read or transcribed, so METIS's notice is not needed.

**Evidence.** Tier C. The four ported `GraphCut_ComputeNormalizedMinGraphCut*` cases in
GraphCutTests pass. NormalizedMinGraphCutTests (C#-only) checks that planted clusters (2-5
dense clusters in a ring of light edges, ten random draws each) come out one part per
cluster, that random graphs of 100-400 vertices split into 1, 2, 3, 5 and 8 parts give
non-empty parts within 10% (plus three vertices) of n/k and identical output on a second
call, that a 100 x 40 unit grid is bisected with 43 cut edges against an optimum of 40, and
that the partitioner's step count grows less than 6x from n = 5k to 20k on a star, a
20-hub graph and disconnected pairs (a quadratic step would give 16x). In Release on an
Apple arm64 laptop, k = 2 takes 42-162 ms for stars of 10k-40k leaves, 21-58 ms for
10k-40k leaves on 20 hubs, 2-22 ms for 10k-40k disconnected pairs, and 0.11 s / 2.4 s for
random graphs of 10k / 100k vertices with 5 edges per vertex.

## 114. libm and MathF results can differ from COLMAP's in the last ulp

**What differs.** Wherever COLMAP calls a transcendental function (`std::log`, `std::exp`,
`std::sin`/`cos`/`atan2`, `std::cbrt`, `std::acos`, their float overloads), the port calls
.NET's `Math`/`MathF` counterpart. .NET forwards most of these to the platform C runtime, but
neither side promises correct rounding, and a C++ build may inline or substitute its own
versions (a vectorized routine, a `sincos` pair), so a result can differ by one ulp. Sites
where this is stated and the tier set accordingly:
- `Optim/Sprt.cs`: the SPRT decision threshold goes through `log`.
- `Feature/FeatureKeypoint.cs`: the scale/orientation constructor, `FromShapeParameters`,
  `ComputeScale*`, `ComputeOrientation` and `ComputeShear` go through `MathF.Sin/Cos/Atan2`.
- `Mathematics/MathUtils.cs`: `Sigmoid`/`ScaleSigmoid` go through `exp`.
- `Mathematics/Polynomial.cs`: the cubic roots go through `cbrt`/`acos`/`cos`.
- `Mathematics/LibcxxRandom.cs`: `NormalDistribution` calls `log` (on macOS both sides call
  the system libm and agree bit for bit).
Entries 7 (sin(a/2) in the angle-axis conversion) and 11 (the UTM latitude) are the cases an
oracle fixture actually caught; entry 12 covers the camera models, where the observed
differences are FMA contraction rather than libm.

**Why.** Reproducing another library's transcendental functions bit for bit would mean
porting that library (and choosing which one: Apple's libm, glibc and MSVC's CRT differ), for
last-ulp effects. .NET's own functions keep ColmapSharp's results the same for a given
runtime and platform.

**Evidence.** The ported tests that cover these sites compare with COLMAP's own tolerances
and pass: `SprtTests`, `FeatureTypesTests`, `MathTests.Sigmoid_Nominal` /
`ScaleSigmoid_Nominal`, `PolynomialTests.FindCubicPolynomialRoots_*` and `RandomTests`.
