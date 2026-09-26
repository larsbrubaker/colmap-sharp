# Deliberate divergences from COLMAP

Each entry: what differs, why, and the evidence. Numbered so code comments can cite them
(`docs/CPP_DIVERGENCES.md`, entry N). Remove an entry when the divergence is gone.

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
