# Divergences from COLMAP: Optim, Solver and LinearAlgebra: RANSAC, the Ceres replacement and sparse factorizations

Part of the divergence log: `docs/CPP_DIVERGENCES.md` is the index and explains the
numbering. Entries are in ascending number; each says what differs, why, and the evidence.

## 13. Sparse Cholesky: simplicial LLT/LDLT with our AMD instead of CHOLMOD and Eigen

**What differs.** `SparseCholeskyWithFallbackSolver`'s first stage is CHOLMOD's supernodal
LLT in COLMAP and the simplicial LLT of `LinearAlgebra/SimplicialCholesky.cs` here; its
fallback and `LeastAbsoluteDeviationSolver`'s `SimplicialLLT` are Eigen's simplicial
factorizations in COLMAP and ours here. `SolverType.SupernodalCholmodLLT` keeps its name but
selects the managed fallback solver. The fill-reducing ordering is our AMD
(`LinearAlgebra/AmdOrdering.cs`, from the Amestoy-Davis-Duff paper, without dense-row
deferral, initial supervariable detection or mass elimination), so the permutation, and with
it the order of floating-point operations and the last bits of every solution, differ from
CHOLMOD's and Eigen's. Accept/reject decisions follow the same rules (LLT stops at the first
pivot <= 0, LDLT at an exactly zero pivot), but a pivot within rounding of zero can land on
the other side under a different ordering.

Also, once the fallback solver has switched to LDLT, `AnalyzePattern`/`Compute` with a new
pattern re-analyzes the LDLT stage here. COLMAP re-analyzes only the supernodal stage, so its
next `ldlt_.factorize` runs on a stale symbolic analysis (an upstream bug with undefined
results in Eigen); `ComputeNewPatternAfterFallback` pins the fix.

**Why.** CHOLMOD and CSparse are GPL/LGPL and Eigen is MPL-2.0 (`docs/LICENSE_AUDIT.md`);
none of them can be ported. A supernodal code is a performance refinement over the
simplicial one, not a different result, and can replace it later behind the same API.

**Evidence.** `sparse_cholesky_test.cc` and `least_absolute_deviations_test.cc` pass 1:1
(`SparseCholeskyWithFallbackSolverTests`, `LeastAbsoluteDeviationsTests`), including the
singular, indefinite and ridge cases whose outcome depends on the pivot rules. The C#-only
`SimplicialCholeskyTests` pin the solutions to the dense LLT within 1e-10 relative error.

## 16. PROSAC's out-of-range sample index fails a Check instead of reading past the data

**What differs.** COLMAP's `ProgressiveSampler` (ported faithfully in
`Optim/ProgressiveSampler.cs`) makes index `n` the mandatory element of a progressive sample,
and `n` can equal `total_num_samples`: on the first sample when `num_samples ==
total_num_samples`, and in general on the sample where the growth schedule reaches the last
element. COLMAP's `Sampler::SampleXY` then reads `X[total_num_samples]`, past the end of the
`std::vector` (undefined behavior: garbage or a crash). `SampleX`/`SampleXY` in
`Optim/Sampler.cs` check every sampled index against the data length and throw COLMAP's
"Check failed" `ArgumentException` instead.

**Why.** Undefined behavior has no C# equivalent to match, and silently clamping or shifting
the index would change the sampler's Tier A sequence that `progressive_sampler_test.cc` pins.
COLMAP 4.2.0 instantiates `ProgressiveSampler` nowhere outside its own test, so no pipeline
reaches this path; failing loudly keeps any future user from getting a model estimated from
out-of-bounds memory.

**Evidence.** `ProgressiveSamplerTests` (1:1) still pass unchanged. The C#-only
`RansacTests.CSharpOnly_ProgressiveSamplerIndexPastEndFailsCheck` runs RANSAC with PROSAC on
exactly `kMinNumSamples` pairs and expects the Check failure.

## 17. RANSAC and LO-RANSAC always run their trial loop serially

**What differs.** With `RANSACOptions::num_threads > 1` (or -1), COLMAP built with OpenMP
runs the trial loop on several threads, each with its own sampler seeded
`random_seed + thread index`, sharing an atomic trial counter and a mutex-guarded best model.
`Optim/Ransac.cs` and `Optim/LoRansac.cs` validate `num_threads` exactly as COLMAP does
(`Check()`, and "Parallel RANSAC only supports RandomSampler" for any other sampler with more
than one effective thread) and then run the loop once on the calling thread, which is what
COLMAP itself does when built without OpenMP ("the block runs once serially").

**Why.** COLMAP's parallel result depends on thread scheduling (which thread claims which
trial index, and which thread's model reaches the shared best first), so it is not
reproducible even against itself; CLAUDE.md requires sequential and parallel runs to give the
same result. The serial loop with the thread-0 seed is COLMAP's own `num_threads == 1`
behavior and its non-OpenMP build's behavior for every `num_threads`. A deterministic parallel
scheme (for example, fixed per-trial seeds) would be a different algorithm from both COLMAP
builds and is left for when profiling shows RANSAC is a bottleneck.

**Evidence.** `RansacTests.RANSAC_ParallelSimilarityTransform` and
`LoRansacTests.LORANSAC_ParallelSimilarityTransform` (1:1, `num_threads = 4`) pass with
COLMAP's expectations. The C#-only `RansacTests.CSharpOnly_ParallelRequiresRandomSampler`
pins the kept validation.

## 18. Solver cost and gradient are summed in residual-block order for any thread count

**What differs.** Ceres' `ProgramEvaluator` accumulates the cost and the gradient per
thread and then adds the per-thread partial sums, so with `num_threads > 1` the last bits of
the cost, the gradient (and through them every accept/reject decision and tolerance test)
depend on how `ParallelFor` split the residual blocks. `Solver/ProgramEvaluator.cs`
evaluates the residual blocks in parallel but stores each block's cost in its own slot and
sums the slots in residual-block order, and computes the gradient as J'r after the parallel
pass (`BlockSparseMatrix.LeftMultiplyAndAccumulate`, row block by row block). The result
equals Ceres' single-threaded summation order for the cost; the gradient's grouping of
additions differs from Ceres' even at one thread (Ceres adds each residual block's
J_i' r_i into the gradient as it goes, which is the same order for the block-sparse
Jacobian and a column-wise order for the dense one).

**Why.** CLAUDE.md's threading rule: sequential and parallel runs must give the same
result. Ceres' per-thread reduction is scheduling-dependent, so no fixed order could
reproduce it anyway; bundle adjustment is Tier C.

**Evidence.** `BundleAdjustmentProblemTests.ThreadCount_DoesNotChangeTheResult` solves the
same problem with 1 and 4 threads for every linear solver and gets bit-identical parameters,
final cost and iteration counts. `Gradient_MatchesFiniteDifferencesOfTheCost` checks the
gradient against the cost it is the derivative of.

## 22. SPARSE_NORMAL_CHOLESKY factors with the simplicial LLT and its own AMD ordering

**What differs.** Ceres 2.2 (as COLMAP builds it, with SuiteSparse) reorders the reduced
program's parameter blocks with CAMD on the Jacobian's block structure, forms J'J in that
order and factors it with CHOLMOD; with EIGEN_SPARSE it leaves the blocks in order and lets
Eigen's AMD permute the matrix. `Solver/LinearSolvers.cs` keeps the problem's block order,
forms the lower triangle of J'J + D^2 from the block structure, and factors it with
`LinearAlgebra/SimplicialCholesky.cs` (LLT, AMD from `AmdOrdering.cs`), the same stand-in as
entry 13. The steps agree to rounding; the last bits differ, and so can a borderline
accept/reject decision late in a solve.

**Why.** CHOLMOD is GPL/LGPL and Eigen MPL-2.0 (`docs/LICENSE_AUDIT.md`); entry 13 has the
reasoning. A factorization failure is reported as Ceres' Eigen back end reports it
("Eigen failure. Unable to find numeric factorization.", a rejected step).

**Evidence.** `CeresExampleTests.Powell` and `CurveFitting` reach the tutorial's printed
optimum, cost, iteration count and number of rejected steps with SPARSE_NORMAL_CHOLESKY as
with DENSE_QR, and `BundleAdjustmentProblemTests.RecoversGroundTruth` recovers the ground
truth with all three linear solvers.

## 35. The Schur solvers eliminate sequentially with dynamic-size kernels and their own sparse ordering

**What differs.** `Solver/SchurEliminator.cs`, `SchurComplementSolvers.cs`,
`ImplicitSchurComplement.cs`, `IterativeSchurSolver.cs` and `SchurOrdering.cs` port Ceres
2.2's DENSE_SCHUR, SPARSE_SCHUR and ITERATIVE_SCHUR with these differences:
1. The chunks (one E block's rows) are eliminated one after another in row order. Ceres runs
   them in parallel and serializes the updates of each reduced-camera-matrix cell and
   right-hand-side block with a mutex, so with `num_threads > 1` the order of additions into a
   cell depends on scheduling. The order here is Ceres' single-threaded order, for every
   thread count: with several threads the elimination runs in two phases that replay each
   cell's updates in that order (`SchurEliminator.Parallel.cs`, pinned by
   `SchurEliminatorParallelTests` and `SchurDeterminismTests`).
2. Only the dynamic-size eliminator is ported. Ceres picks a template specialization from the
   Jacobian's static block sizes (for COLMAP's BA typically `<2, 3, 6>`, and
   `SchurEliminatorForOneFBlock<2, 3, 6>` when there is a single camera block); those use
   Eigen's fixed-size products and invert 3x3 E'E blocks with Eigen's cofactor `inverse()`,
   where the port uses Ceres' naive loop kernels (`SmallBlas.cs`) and a Cholesky solve of the
   identity (Ceres' own dynamic-size path). Results agree to rounding, not bit for bit. The
   BA shapes (2-row residual blocks, 3-dimensional E blocks) run through unrolled kernels
   (`SmallBlasFixed.cs`) that keep the naive loops' summation order, so they round exactly as
   the naive loops do (`SmallBlasFixedTests`, `SchurDeterminismTests`).
3. SPARSE_SCHUR factors the lower triangle of the reduced camera matrix with
   `LinearAlgebra/SimplicialCholesky.cs` (AMD on the scalar pattern). Ceres with EIGEN_SPARSE
   first reorders the F blocks by AMD on the block pattern of the Schur complement
   (`ReorderSchurComplementColumnsUsingEigen`) and factors in natural order; with SuiteSparse
   it uses CAMD/CHOLMOD. The factorization differs, as for SPARSE_NORMAL_CHOLESKY (entry 22).
4. The CG vector reductions (dot products, norms) are left-to-right sums; Ceres uses Eigen's
   (vectorized) `norm()`/`dot()` or per-thread partial sums.
5. A non-positive-definite E'E or diagonal preconditioner block becomes NaN (the LM step is
   then invalid and the radius shrinks), where Eigen's LLT leaves unspecified values.
6. With no user ordering or a single group, the elimination ordering is Ceres' automatic
   one (the greedy independent set, `ComputeStableSchurOrdering`). A user
   `ParameterBlockOrdering` with several groups (only COLMAP's global positioner sets one) is
   applied as in Ceres, except for the order inside each group (entry 36).

**Why.** CLAUDE.md's threading rule (sequential and parallel runs must give the same result;
Ceres' multithreaded order is not reproducible anyway), and Eigen and SuiteSparse are
replaced rather than ported (CLAUDE.md contract 2). Bundle adjustment is Tier C.

**Evidence.** The ported Ceres tests pass with Ceres' tolerances:
`SchurEliminatorTests` (reduced system and solution vs. the dense reference, 1e-14 relative),
`ImplicitSchurComplementTests.SchurMatrixValuesTest` (1e-14), `SchurComplementSolverTests`
and `IterativeSchurComplementSolverTests` (vs. DENSE_QR, 1e-10 and 1e-14),
`ConjugateGradientsSolverTests` (4 ULP). `BundleAdjustmentProblemTests` runs every Schur
solver: ground truth recovered (`RecoversGroundTruth`), DENSE_QR's optimum on noisy
observations (`EverySolver_ReachesTheSameOptimum`, cost to 1e-9 relative, parameters to
1e-6), and bit-identical results for 1 and 4 threads (`ThreadCount_DoesNotChangeTheResult`).
Not compared against C++ Ceres.

## 36. A user ParameterBlockOrdering keeps each group in insertion order

**What differs.** Ceres' `ParameterBlockOrdering` (`OrderedGroups<double*>`) stores each
group as a `std::set<double*>`, so `ApplyOrdering` lays out the blocks of one group in
increasing heap-address order. `Solver/ParameterBlockOrdering.cs` keeps each group in the
order its blocks were added (`AddElementToGroup`; moving a block to another group appends it
there). The program's parameter block order within a group, and so the Schur eliminator's
chunk order and the reduced camera system's column order, can differ from a given Ceres run.

**Why.** C# has no stable addresses, and in C++ the address order is itself an accident of
the allocator (COLMAP's global positioner adds points, frame centers and rig cameras that
live in separate hash-map nodes), so no run-to-run order exists to match. Insertion order is
deterministic and is the order the caller controls. Groups themselves are still visited in
increasing id order, and every Ceres check (independent first group, element count, unknown
blocks) is the same.

**Evidence.** `OrderedGroupsTests` (Ceres' `ordered_groups_test.cc`) and
`ReorderProgramTests.ApplyOrderingNormal` pass unchanged; neither depends on the order
inside a group. The effect on results is rounding-level (the order of additions into the
reduced system), within the Tier C outcome bar; `ProblemTests.UserOrdering_SchurSolverMatchesDenseQr`
checks the solutions against DENSE_QR.

## 37. Solve does not edit the caller's ordering, and ITERATIVE_SCHUR never falls back to CGNR

**What differs.** 1. Ceres' preprocessor removes the blocks it drops as constant from
`Solver::Options::linear_solver_ordering` itself (a `shared_ptr`, so the caller's object is
edited). `LeastSquaresSolver.Solve` edits a copy; `SolverOptions.LinearSolverOrdering` is
unchanged afterwards. 2. When that removal empties the first elimination group, Ceres swaps
SPARSE_SCHUR for SPARSE_NORMAL_CHOLESKY and DENSE_SCHUR for DENSE_QR (both done here, and
reported in `SolverSummary.LinearSolverTypeUsed`) and ITERATIVE_SCHUR for CGNR; CGNR is not
ported, so that last case ends the solve with FAILURE and a message saying so. 3. An empty
user ordering, or (for a Schur solver) one that holds only blocks the reduction removed,
trips a CHECK in Ceres' `MinNonZeroGroup` and aborts the process; here the solve ends with
FAILURE and a message naming the problem.

**Why.** 1. The in-place edit is a side effect on an input the caller may reuse (COLMAP
rebuilds its ordering before every solve, so it never sees it). 2. CGNR is excluded from the
port (PORTING_PLAN.md, Skipped tests); no COLMAP caller combines ITERATIVE_SCHUR with a user ordering. 3. A
library inside MatterCAD must not abort the host; the CHECK's condition is kept, only its
consequence differs.

**Evidence.** `ProblemTests.UserOrdering_ConstantFirstGroup_SwitchesSolver`: every block of the
first group constant, SPARSE_SCHUR solves as SPARSE_NORMAL_CHOLESKY and the ordering still
holds all six blocks. `ProblemTests.UserOrdering_Empty_Fails` and
`UserOrdering_OnlyConstantBlocks_Fails` cover case 3.

## 53. Problem.GetParameterBlocks lists blocks in insertion order

**What differs.** Ceres' `Problem::GetParameterBlocks` walks the problem's `std::map` keyed by
the blocks' addresses, so it lists them in memory-address order. The port
(`Solver/Problem.cs`) lists them in insertion order. The one caller,
`GetOtherParams` in `Estimators/Covariance.cs` (colmap/estimators/covariance.cc), uses the order
to lay out the "other" blocks (camera intrinsics, sensor_from_rig poses) after the poses in
the Schur complement and in `L_inv`.

**Why.** .NET arrays have no stable address, and address order is an accident of the
allocator in C++ as well. The layout changes only the elimination order inside the sparse
factorization, not the covariances beyond round-off.

**Evidence.** `BACovarianceTests` (covariance_test.cc 1:1) checks `GetOtherParamsCov` against
ceres::Covariance at COLMAP's 1e-8 tolerance.

## 115. Eigen's SIMD evaluation order is not reproduced in norms, reductions and products

**What differs.** Eigen vectorizes fixed- and dynamic-size expressions: a norm or dot product
is summed in packet lanes and then reduced horizontally, a quaternion product on doubles pairs
its terms for SIMD, and matrix products use blocked, vectorized kernels. The port evaluates
these as left-to-right sums in coefficient order, except where an oracle showed Eigen's order
and the port copies it (`LinearAlgebra/Vector4d.cs` reductions, `Quaterniond`'s product,
`Feature/FeatureUtils.cs`' descriptor normalization, `Feature/CovariantSift.cs`' DSP mean).
The remaining sites can differ from COLMAP in the last bits:
- `Solver/Manifolds.cs`: the vector norms in the sphere and quaternion manifolds. Tier C,
  like the solver that consumes them.
- `Estimators/CostFunctions/QuaternionT.cs` and `ReprojectionError.cs`: the quaternion product
  on the residual-only (plain double) path, where Eigen pairs terms for SIMD. Tier B.
- `LinearAlgebra/Matrix6d.cs` and `MatrixXd.cs`: 6x6 and dynamic-size products. Tier B.
The Schur solvers' CG reductions are entry 35, item 4.

**Why.** Eigen is MPL-2.0 and not ported (CLAUDE.md contract 2), and its packet order depends
on the target's SIMD width and the compiler (NEON, SSE and AVX builds reduce differently), so
there is no single order to match; `System.Numerics.Vector<T>` is also banned in math paths
("No FMA"). A plain sequential order gives the same result on every platform.

**Evidence.** `ManifoldTests` (C#-only, the invariants of Ceres' manifold_test_utils.h) and
`ReprojectionErrorTests` (reprojection_error_test.cc 1:1) pass with their tolerances, and the Jet (autodiff) path of the cost
functions already evaluates the product term by term, as Eigen does for Jets. No oracle
fixture exposes a 6x6 or dynamic product directly.

## 124. LO-RANSAC can start its local optimization from a different five-point solution

**What differs.** The five-point solver returns up to ten essential matrices per minimal
sample, and ColmapSharp returns them in a different order than COLMAP: they come from the real
roots of a polynomial whose coefficients depend on the null-space basis, and that basis
differs (entry 27). `Optim/LoRansac.cs` walks a sample's models in order, as COLMAP's
`LORANSAC::Estimate` does, and runs the local optimization from the first model that beats the
best support so far; a later model of the same sample is only scored against the result. So
when two of a sample's models both beat the current best, the two sides refine from different
starting models. The local optimization is capped at ten refit rounds
(`kMaxNumLocalTrials`), so a chain that has not converged by then stops somewhere else, and
the final E and relative pose differ even though the inlier count is the same.

**Why.** Reproducing COLMAP's order would mean reproducing the basis of Eigen's
full-pivoting Householder QR, which entry 27 rules out (Eigen is MPL-2.0). Sorting the
solutions by some canonical key would not match COLMAP either. The algorithm is followed
exactly; only the order of an unordered solution set differs, and RANSAC is Tier C.

**Evidence.** `GlobalSfmOracleTests.CSharpOnly_ViewGraphCalibrationMatchesPycolmap(uncalibrated)`
(fixture `oracle/fixture_global_sfm.py`): 27 of the 28 re-estimated relative poses match
pycolmap 4.2.0 within 6.5e-7 deg. Pair 2-3 has the same configuration and 47 inliers, but its
pose is 0.41 deg (rotation) and 0.25 deg (translation direction) away. Traced in
`LoRansac.Estimate`, the first trial's sample yields models with 5, 8 and 14 inliers in that
order. ColmapSharp starts the local optimization from the 8-inlier model, and after ten
rounds it stops at 46 inliers with score 11.85, still moving. With the sample's models
reversed as an experiment (and entry 19's experimental basis also in place), the chain
starts from the 14-inlier model, converges at score 11.45, and reproduces pycolmap's E and
pose to 2.6e-14 deg. pycolmap's own result is stable: `estimate_essential_matrix` on the same pair gives the same E with `max_num_trials` 1 to
1000, and with every keypoint perturbed by up to 1e-4 px, so the difference is not rounding
noise. Downstream, the global pipeline on the two calibrations agrees to 4.0e-4 deg and 2.5e-5
of the scene radius.
