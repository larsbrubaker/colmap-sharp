# colmap-sharp — agent guidelines

A **pure C#** photos/video → mesh reconstructor for MatterCAD. It turns photos into camera
poses, a sparse point cloud, dense depth, and a textured mesh. It started as a port of
[COLMAP](https://github.com/colmap/colmap) 4.2.0 (`REFERENCE`) and that port is its base, but
**the goal is the best reconstruction we can build, not COLMAP parity** (Lars, 2026-09-28).
Change any behavior, default or algorithm when a measurement shows it makes results better;
pull in upstream COLMAP changes only when they help. Remaining port work is in
`PORTING_PLAN.md`.

## The contract (non-negotiable)

1. **Pure managed C#.** No C/C++, no P/Invoke, no native NuGet packages, no linking to COLMAP
   or anything else. Every COLMAP dependency is ported, replaced by C# written here, or
   dropped (`docs/LICENSE_AUDIT.md`). The library must stay trim- and AOT-clean so it runs on
   browser-wasm like manifold-sharp.
2. **MIT, commercial-safe.** Port only permissively licensed code. Never port, transcribe or
   "follow along with" GPL/LGPL/AGPL/MPL or non-commercial code — Eigen, CHOLMOD/CSparse,
   CGAL, LSD, SiftGPU, Qt are all out. Check `docs/LICENSE_AUDIT.md` before porting anything
   that is not COLMAP's own source, and add the upstream's notice to
   `THIRD_PARTY_NOTICES.md` in the same change.
3. **No stubs.** No `NotImplementedException`, no placeholders, no partial implementations.
   If a dependency isn't ported yet, port it first (dependency order, below).
4. **Quality is measured, not asserted.** The yardstick is reconstruction quality on a
   benchmark set (synthetic scenes with known geometry plus real captures): images placed,
   pose error, mesh-to-truth distance, runtime. An improvement lands with the benchmark
   numbers that justify it, and a regression on the benchmark is a bug.
5. **Never weaken a test to make it pass.** Every failure is a real bug, found by
   instrumentation and root-cause analysis. COLMAP's ported tests (one file per `*_test.cc`,
   same names) stay as the regression net for the base port. When an intended improvement
   changes one of their expected values, update it in the same change, stating the new
   behavior and the benchmark evidence; an unexplained change is still a bug.
6. **Differences from COLMAP are recorded, not avoided.** Anything that behaves differently
   from COLMAP gets a numbered divergence entry (what differs, why, the evidence), so a later
   upstream merge knows what was changed on purpose. Entries live by module in
   `docs/divergences/<Module>.md`; `docs/CPP_DIVERGENCES.md` is the index (number, title,
   file), says where a new entry goes and holds the next free number. Code comments cite an
   entry as "divergence N".

## How closely must the base port match COLMAP?

These tiers apply to code that is still a straight port. Code we have deliberately improved
is held to the benchmark instead. Bit-exactness with COLMAP is not achievable everywhere,
because COLMAP's numbers run through Eigen and Ceres, which we replace rather than port. The
bar depends on the code:

| Tier | Code | Bar |
|---|---|---|
| **A — exact** | Scalar, deterministic code: camera models, pose/transform algebra, PRNG and samplers, track/graph bookkeeping, reconstruction file I/O, SIFT's scale space | Bit-identical to COLMAP on the same input. |
| **B — tolerance** | Code that goes through decompositions (SVD, eigen, QR) or solvers: minimal solvers, triangulation, homography/essential estimation | COLMAP's own test tolerances, and oracle fixtures agree within a stated tolerance. Solution *sets* compare order-insensitively only where COLMAP's contract is a set. |
| **C — outcome** | Iterative / randomized pipelines: RANSAC, bundle adjustment, incremental mapping, PatchMatch | Same convergence on oracle fixtures: final cost, registered image count, reprojection error, and poses after Sim3 alignment within stated bounds. |

Tier is decided per function when it is ported, and stated in the test that pins it.

## Verification nets

1. **Ported test suite** — `ColmapSharp.Tests`, one file per `*_test.cc`, same names.
2. **Oracle fixtures** — `oracle/` holds Python scripts that run the pinned `pycolmap` wheel
   (`oracle/requirements.txt`) and write inputs and outputs to `ColmapSharp.Tests/TestData/oracle/`.
   The fixtures are checked in, so the C# build and tests never need Python. pycolmap is a
   test oracle only; nothing in this repo calls it at runtime.
3. **Differential harnesses** — for a Tier A function the suite can only see through a final
   result, instrument both sides (a pycolmap script, or the C++ reference built in a scratchpad)
   and diff intermediates until they agree.

## Reference source

`scripts/fetch-reference.sh` clones COLMAP at the `REFERENCE` commit into `cpp-reference/`
(git-ignored). It is reading material only; the build never touches it. Always read the C++
function *and everything it calls* before porting it.

## C++ → C# translation rules

- **Stable sorts.** `std::sort` is unstable and so is `Array.Sort`, but they don't pick the same
  order among equal keys. Where the order of ties can reach an output, use an explicit
  tie-break (usually the index) and note it. `std::stable_sort` → LINQ `OrderBy` or an index
  tie-break comparator.
- **Hash container iteration order.** COLMAP iterates `std::unordered_map`/`set` in places
  where order leaks into results. .NET `Dictionary` order differs. Where iteration order
  matters, find out what COLMAP actually relies on and make it deterministic here; document
  it in a divergence entry (indexed in `docs/CPP_DIVERGENCES.md`) if the result can differ.
- **PRNG.** COLMAP uses `std::mt19937` behind `colmap/math/random.h`. Port mt19937 exactly.
  `std::uniform_int_distribution` / `uniform_real_distribution` / `normal_distribution` are
  implementation-defined; we match **libc++**, which the macOS pycolmap wheel links. So
  PRNG-dependent fixtures must be generated on macOS (the Linux wheel uses libstdc++).
- **No FMA.** Never introduce `Math.FusedMultiplyAdd` or `System.Numerics.Vector<T>` in math
  paths, so results are identical on every platform. Explicit fixed-width `Vector128<T>` lanes
  are allowed when each lane repeats the scalar IEEE operations in the same order (add, sub,
  mul, div, sqrt; no `FusedMultiplyAdd`, no horizontal sums that reorder a reduction), with a
  test that pins the result bit-identical to the scalar path. Watch out: Apple clang defaults to
  `-ffp-contract=on`, so the macOS arm64 pycolmap wheel *may* fuse `a*b + c` where we don't.
  When a Tier A oracle diff lands exactly on a multiply-add, suspect contraction on the C++
  side before chasing a port bug, and record the case in a divergence entry (indexed in
  `docs/CPP_DIVERGENCES.md`).
- **Numeric constants.** `std::numeric_limits<double>::epsilon()` is `2.220446049250313E-16`;
  C# `double.Epsilon` is the smallest subnormal and is **wrong**. `float` stays `float` —
  COLMAP stores descriptors, bitmaps, and depth maps in single precision.
- **Integer overflow.** C# `unchecked` wraps like C++ unsigned arithmetic; signed overflow is
  UB in C++, so a site that depends on it gets a comment.
- **Eigen semantics.** Quaternions are stored `(x, y, z, w)` in Eigen memory but constructed
  `(w, x, y, z)`; COLMAP's file formats write `qw qx qy qz`. Matrices are column-major. Eigen's
  `normalized()` on a zero vector returns zero. Match the documented behavior, don't port Eigen.
- **Threading.** COLMAP's `ThreadPool` becomes `Parallel.For`/tasks only where each worker
  writes its own slot. Sequential and parallel runs must give the same result.
- **Errors.** `THROW_CHECK*` → `Util/Check.cs` (throws with COLMAP's "Check failed: …" message); `LOG(FATAL)` → exception. Never swallow.
- **Cancellation and progress** go through `CancellationToken` and `IProgress<T>` — MatterCAD
  shows progress and lets the user cancel a long reconstruction.

## Layout and style

- `ColmapSharp/` — the library, one folder per COLMAP module (`Mathematics/` — not `Math`,
  which would shadow `System.Math` — `Geometry/`,
  `Sensor/`, `Scene/`, `Optim/`, `Estimators/`, `Feature/`, `Sfm/`, `Mvs/`, `Controllers/`,
  `Util/`), plus `LinearAlgebra/` and `Solver/` (the Eigen and Ceres replacements).
- `ColmapSharp.Tests/` — TUnit, mirroring the same folders.
- Namespace `ColmapSharp.<Module>`. PascalCase C# names; the C++ name goes in the doc
  comment when it differs enough to be hard to find (`/// Port of colmap::EstimateRigid3d`).
- **Every file starts with a header**: what it is, the C++ file(s) it ports, and how it relates
  to its neighbors.
- **800-line limit per file:** at most 800 lines, blank lines included, in any source file
  (`.cs`, scripts, oracle harnesses, WGSL) and any `.md` doc (plans, notices, divergences),
  enforced by `FileComplianceTests` in the test suite. The limit is a length trigger that
  prompts a refactor, not a measure of content. There are no exemptions. If a port would
  exceed it, split it by responsibility (partial classes along C++ section boundaries are
  fine); a doc that grows past it is split by topic, with an index file pointing at the
  parts. Split rather than squeeze: never trim comments or blank lines to fit. See the
  `file-size-refactoring` skill.
- Value types (`Vector3d`, `Matrix3d`, `Rigid3d`, `Quaterniond`) are `readonly struct`s.
  Hot loops avoid allocation.
- Comments explain *why*. Keep COLMAP's non-obvious comments, since they carry the reasoning.

## Commands

```bash
dotnet build ColmapSharp.sln
dotnet test --project ColmapSharp.Tests/ColmapSharp.Tests.csproj
dotnet test --project ColmapSharp.Tests/ColmapSharp.Tests.csproj -- --treenode-filter "/*/*/<TestClass>/*"
scripts/fetch-reference.sh            # C++ reference into cpp-reference/
oracle/setup.sh && oracle/.venv/bin/python oracle/<script>.py   # regenerate fixtures
```

## Test-first bug fixing

Write a failing test that reproduces the bug, fix the root cause, and watch the test pass. Never skip the reproducing test.

## Git

Commit on `main`; push only to `origin` (github.com/larsbrubaker/colmap-sharp). This repo is
a submodule of MatterCAD at `Submodules/colmap-sharp`: push here first, then bump the pointer
in MatterCAD.

## Orchestration pattern

Same as MatterCAD. The main session acts as planner and orchestrator only; it should not write
or edit code directly. All implementation is delegated to the `implementer` subagent
(`.claude/agents/implementer.md`), one scoped step at a time. All post-change review is
delegated to the `reviewer` subagent (`.claude/agents/reviewer.md`). The main session handles
only planning, architecture decisions, and synthesizing subagent results. When tests fail, use
the `fix-test-failures` agent, which treats every failure as a real bug.

Brief each implementer with one deliverable and a 25-minute budget; a run over 30 minutes is an
error. A deliverable is one coherent slice of a `PORTING_PLAN.md` phase together with its ported
tests. Implementers that may run concurrently get `isolation: "worktree"`.

Working conventions that every brief repeats:
- A new worktree starts with `git reset --hard main`. `cpp-reference/` and `oracle/.venv` are
  git-ignored, so worktrees read them from the main checkout. On a fresh machine run
  `scripts/fetch-reference.sh` and `oracle/setup.sh` first.
- Implementers commit on their own branch and never push or edit `PORTING_PLAN.md`; they list
  what their change makes stale, and the orchestrator prunes the plan when it merges.
- Divergence entry numbers are stable and never reused. Hand each concurrent implementer its own
  range, and take the next free number from the index, `docs/CPP_DIVERGENCES.md`. When
  merging divergences, merge entry by entry rather than by text hunks, in both the module
  file under `docs/divergences/` and the index table, update the index's next free number,
  then check that no conflict marker is left (`FileComplianceTests` also rejects them).
- Keep headers, comments and divergence entries true to the final code in the same commit.
- PoissonRecon stages are ported bit-exact against the vendored C++ through the clang harnesses
  in `oracle/poisson_*_harness.cc` (`oracle/fixture_poisson_tree.py` writes the fixtures);
  follow those tests' pattern for new stages.
