# colmap-sharp — agent guidelines

A **pure C#** port of [COLMAP](https://github.com/colmap/colmap) (Structure-from-Motion and
Multi-View Stereo), for use inside MatterCAD. It turns a set of photos into camera poses, a
sparse point cloud, dense depth, and a mesh. Reference version: `REFERENCE` (COLMAP 4.2.0).
Remaining work and the phase order are in `PORTING_PLAN.md`.

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
4. **COLMAP's tests are the specification, 1:1.** Every `*_test.cc` becomes a C# test file
   with the same test names and the same expected values and tolerances. A test that
   exercises an excluded feature (CUDA, ONNX, LSD, GUI, SQLite files) is listed as skipped
   with the reason in `PORTING_PLAN.md`, never silently dropped. C#-only tests are labeled as
   such and never stand in for a ported one.
5. **Never weaken a test to make it pass.** Every failure is a real bug, found by
   instrumentation and root-cause analysis. For a ported test, COLMAP's expected value is
   the spec and the C# output is the bug.
6. **Deliberate divergence is documented.** Anything that behaves differently from COLMAP on
   purpose (a real upstream bug, a replaced dependency) gets a numbered entry in
   `docs/CPP_DIVERGENCES.md`: what differs, why, and the evidence. Convenience is not a reason.

## How closely must results match?

Bit-exactness with COLMAP is not achievable everywhere, because COLMAP's numbers run through
Eigen and Ceres, which we replace rather than port. The bar depends on the code:

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
  it in `docs/CPP_DIVERGENCES.md` if the result can differ.
- **PRNG.** COLMAP uses `std::mt19937` behind `colmap/math/random.h`. Port mt19937 exactly.
  `std::uniform_int_distribution` / `uniform_real_distribution` / `normal_distribution` are
  implementation-defined (libc++ ≠ libstdc++). Port the one used by the pinned pycolmap wheel
  for the platform the oracle fixtures come from, and record which in `Random.cs`.
- **No FMA.** Never introduce `Math.FusedMultiplyAdd` or `System.Numerics.Vector<T>` in math
  paths; C++ is built without `-ffast-math` and contraction would change bits.
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
- **Errors.** `THROW_CHECK*` → `ArgumentException`/`InvalidOperationException` with the
  same message; `LOG(FATAL)` → exception. Never swallow.
- **Cancellation and progress** go through `CancellationToken` and `IProgress<T>` — MatterCAD
  shows progress and lets the user cancel a long reconstruction.

## Layout and style

- `ColmapSharp/` — the library, one folder per COLMAP module (`Math/`, `Geometry/`,
  `Sensor/`, `Scene/`, `Optim/`, `Estimators/`, `Feature/`, `Sfm/`, `Mvs/`, `Controllers/`,
  `Util/`), plus `LinearAlgebra/` and `Solver/` (the Eigen and Ceres replacements).
- `ColmapSharp.Tests/` — TUnit, mirroring the same folders.
- Namespace `ColmapSharp.<Module>`. PascalCase C# names; the C++ name goes in the doc
  comment when it differs enough to be hard to find (`/// Port of colmap::EstimateRigid3d`).
- **Every file starts with a header**: what it is, the C++ file(s) it ports, and how it relates
  to its neighbors.
- **800-line limit per file, the same rule as MatterCAD:** at most 800 non-empty lines in any
  source file (`.cs`, scripts), enforced by `FileComplianceTests` in the test suite. There
  are no exemptions. If a port would exceed the limit, split it by responsibility (partial
  classes along C++ section boundaries are fine). Never trim comments or blank lines to fit;
  see the `file-size-refactoring` skill.
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
