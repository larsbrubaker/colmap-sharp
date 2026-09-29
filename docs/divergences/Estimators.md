# Divergences from COLMAP: Estimators: minimal solvers, bundle adjustment, rotation averaging and alignment

Part of the divergence log: `docs/CPP_DIVERGENCES.md` is the index and explains the
numbering. Entries are in ascending number; each says what differs, why, and the evidence.

## 19. TinySphereManifold's tangent basis uses Hughes & Moller, not Eigen's unitOrthogonal()

**What differs.** COLMAP's `SphereManifold<3>` (`estimators/cost_functions/tiny_manifold.h`)
builds the tangent basis at x as `b1 = x_hat.unitOrthogonal()`, `b2 = x_hat.cross(b1)`.
`Estimators/CostFunctions/TinyManifold.cs` builds `b1` with the construction of Hughes &
Moller, "Building an Orthonormal Basis from a Unit Vector" (JGT 4(4), 1999): zero the
component of smallest magnitude (the first on a tie), swap the other two and negate one,
normalize. `b2` is the same cross product. The two constructions can pick a different unit
vector in the tangent plane, so the tangent coordinates of a step, and with them TinySolver's
Jacobi scaling and iterates, can differ from COLMAP's while spanning the same plane.

**Why.** Eigen documents only that `unitOrthogonal()` returns some unit vector orthogonal to
the input (its choice for 3-vectors is an implementation detail of MPL-2.0 code we may not
transcribe, CLAUDE.md contract 2). Any orthonormal tangent basis gives a valid retraction and
Plus Jacobian; TinySolver is Tier C, so what must agree is the converged result, which does
not depend on the basis.

**Evidence.** `TinyManifoldTests.SphereManifold_PlusStaysOnUnitSphere` and
`SphereManifold_PlusJacobianMatchesFiniteDiff` (1:1: orthogonality to x within 1e-12 and the
finite-difference Jacobian within 1e-6) and `TinySolverTests.TinySolver_ManifoldConvergesAndStaysOnManifold`
(1:1: the sphere fit converges to target.normalized() within 1e-6) pass. Where TinySolver
stops before converging (its 25-iteration cap in `EssentialMatrixTangentSampsonEstimator.Refine`,
or LO-RANSAC's ten refit rounds), the basis does reach the result, at the rounding level:
the relative poses `GlobalSfmOracleTests` re-estimates agree with pycolmap to 6.5e-7 deg.
In an experiment with another tangent basis (b1 in the x-y plane), one of them agreed to 4e-14
deg instead of 5e-7, so the ~1e-7 deg level of those differences comes from the basis.

## 24. The 7-point fundamental solver takes its null space from unpivoted Householder QR

**What differs.** `FundamentalMatrixSevenPointEstimator::Estimate` gets the 2D null space of
its 7 x 9 constraint system from columns 7 and 8 of the full Q of
`A.fullPivHouseholderQr()` (A is 9 x 7). `Estimators/Solvers/FundamentalMatrixEstimators.cs`
takes the same two columns from an unpivoted Householder QR of A
(`LinearAlgebra/Householder.FactorInPlace` / `ApplyQ`, allocation-free). Both pairs are
orthonormal bases of the same space: in any factorization A = Q R, the first seven columns
of Q contain A's column space, so the last two are orthogonal to every constraint, whatever
the pivoting and whatever A's rank. Only the basis inside that plane differs.

**Why.** Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR (which also swaps
rows) has no published convention precise enough to reproduce Eigen's Q column for column.
The basis does not reach the solutions: the solver finds the rank-2 members of the pencil
{a f1 + b f2}, normalized to unit norm, and that set of matrices (up to sign, which the
Sampson error and COLMAP's F / F(2, 2) comparisons ignore) does not depend on the basis. The
one place it shows is the parametrization lambda f1 + f2 with f1 = q7 - q8: when
|det(f1)| < 1e-16 COLMAP returns no model, and which member of the pencil f1 is depends on
the basis. For a generic sample that is a measure-zero event on either side, so the two can
disagree only on a sample that sits on that numeric knife edge, and then one of them emits
no model for that RANSAC hypothesis.

**Evidence.** `FundamentalMatrixTests.FundamentalSevenPointEstimator_Reference` (COLMAP's
Matlab reference values, 1e-6) and `FundamentalSevenPointEstimator_Nominal` (100 random
problems, at least one model equal to the true F up to scale) pass 1:1.

## 26. re3q3's random change of variables uses a fixed-seed mt19937, not std::rand

**What differs.** PoseLib's `re3q3` (P4Pf's three-quadratics solver,
`Estimators/Solvers/PoseLib/Re3q3.cs`) retries a near-degenerate system (all three
elimination determinants below 1e-10) after a random affine change of variables. PoseLib
draws that from Eigen's `Quaternion::UnitRandom()` and `setRandom()`, which read the C
library's global `std::rand()`. The port draws a uniform rotation (Shoemake, Graphics Gems
III) and a shift in [-1, 1]^3 from a fresh `Mt19937` seeded with 0 on every call, through
`LibcxxRandom.UniformReal`.

**Why.** `std::rand()` is a process-global stream whose sequence depends on the C library
and on every earlier call in the process, so COLMAP's result in this branch is not
reproducible even against itself; porting it would add hidden global state shared across
threads, which CLAUDE.md rules out (sequential and parallel runs must agree). Eigen's
`UnitRandom` code is MPL-2.0 and is not ported, so the rotation follows the published method
it cites. A fixed per-call seed keeps the result a pure function of the input. Only the
degenerate branch is affected; the regular path is a straight port.

**Evidence.** The branch is only reached for near-singular inputs; the 1:1
`AbsolutePoseTests.AbsolutePose_P4PF*` cases (regular path) pass with COLMAP's tolerances.
A change of variables followed by `refine_3q3` against the original coefficients targets the
same system, so the solution set should agree up to solver tolerance whichever random matrix
is drawn. The C#-only `Re3q3Tests.CSharpOnly_DegenerateEliminationUsesRandomVarChange`
(x^2 = 1, y^2 = 4, z^2 = 9: every elimination determinant is 0) finds all eight solutions and
each satisfies the original system to 1e-8. Cross-checked against C++ PoseLib fa7280f built
with clang++ and Eigen 3.4 headers in a scratch harness: for std::srand seeds 1..20 it also
returns 8 solutions with max residual below 3e-9, and for x^2 = 1, y^2 = 4, z = x + y (a
linear third equation) it returns 0 solutions for every seed, as the port does (upstream
behavior, pinned by `Re3q3Tests.CSharpOnly_LinearEquationReturnsNoSolutionsLikePoseLib`).

## 27. The five-point solver takes its null space from unpivoted Householder QR

**What differs.** PoseLib's `relpose_5pt` (the minimal case of COLMAP's
`EssentialMatrixFivePointEstimator`) takes the 4D null space of its five epipolar
constraints from the last four columns of the full Q of
`epipolar_constraints.fullPivHouseholderQr()` (a 9 x 5 matrix).
`Estimators/Solvers/PoseLib/Relpose5pt.cs` takes the same four columns from an unpivoted
Householder QR (`LinearAlgebra/Householder.FactorInPlace` / `ApplyQ`, allocation-free). Both
are orthonormal bases of the same space (in any A = Q R the first five columns of Q contain
A's column space); only the basis inside it differs, and so do the intermediate polynomial
coefficients (`compute_trace_constraints`, the degree-10 determinant polynomial and its
roots, which parametrize the solutions in basis coordinates).

**Why.** Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR has no published
convention precise enough to reproduce Eigen's Q column for column (entry 24 is the same
choice for the 7-point fundamental solver). The solution set does not depend on the basis:
the solver returns every unit-norm E = x N0 + y N1 + z N2 + N3 in the null space that meets
the cubic constraints, and a change of orthonormal basis maps that set onto itself (up to
the sign of each E, which the Sampson errors and COLMAP's tests ignore). What can differ is
numerical conditioning: the back substitution's 1e-6 test for switching to the three-row QR
solve, and the Sturm bracketing tolerance, act on basis-dependent quantities, so a solution
near those thresholds can come out with slightly different rounding, and a near-degenerate
root can be kept on one side and dropped on the other.

**Evidence.** `EssentialMatrixSolverTests.EssentialMatrixFivePointEstimatorTests_Nominal(5)`
(100 random minimal problems, at least one model within COLMAP's 5e-3 of the true E, with
Sampson residuals below 1e-5) passes 1:1, as do the over-determined (20, 1000) and LO-RANSAC
cases that sit on the same solver. Not compared against C++ PoseLib intermediates.

## 29. re3q3_rotation's pre-rotation uses a fixed-seed mt19937, not std::rand

**What differs.** PoseLib's `re3q3_rotation` (used by `gp3p`, which COLMAP's
`GP3PEstimator` calls; `Estimators/Solvers/PoseLib/Re3q3.cs` `SolveRotation`) rotates every
problem by a random rotation R0 before solving in Cayley parameters, so that no solution
lands on the Cayley transform's singularity (a rotation by pi). Unlike re3q3's change of
variables (entry 26), this draw happens on *every* call, not only for degenerate input.
PoseLib takes R0 from Eigen's `Quaternion::UnitRandom()`, i.e. from the process-global
`std::rand()`. The port draws it (Shoemake's uniform rotation) from a fresh `Mt19937` seeded
with 1 on every call.

**Why.** The same as entry 26: `std::rand()` is hidden global state whose sequence depends on
the C library and on every earlier call, so COLMAP's gp3p is not reproducible even against
itself, and porting it would make sequential and parallel runs disagree. Eigen's `UnitRandom`
is MPL-2.0 and not ported. The seed differs from entry 26's so the two draws are not the same
rotation. Consequence: for a fixed input the port always uses the same R0, so an input whose
true rotation is (close to) R0 composed with a rotation by pi is always ill-conditioned here,
where PoseLib would be only for an unlucky draw. That set has measure zero and RANSAC draws
other samples, so no ported test is affected.

**Evidence.** PoseLib fa7280f's gp3p, built with clang++ and Eigen 3.4 headers in a scratch
harness, returns the same four poses for `std::srand` seeds 1..5 on a fixed three-ray input,
agreeing to about 1e-10 across seeds; the port returns the same four poses to 1e-8
(`Gp3pTests.CSharpOnly_MatchesPoseLibSolutionSet`, order-insensitive, since the order depends
on R0). The 1:1 `GeneralizedAbsolutePoseTests.ParameterizedGP3PEstimatorTests_Nominal` cases
pass with COLMAP's tolerances.

## 30. The six-point focal relative pose solvers take their null space from unpivoted Householder QR

**What differs.** PoseLib's `relpose_6pt_shared_focal` and `relpose_6pt_onesided_focal`
(the minimal solvers of COLMAP's `RelativePoseSharedFocalEstimator` and
`RelativePoseOneSidedFocalEstimator`) take the 3D null space of their six epipolar
constraints from the last three columns of the full Q of
`epipolar_constraints.fullPivHouseholderQr()` (a 9 x 6 matrix).
`Estimators/Solvers/PoseLib/Relpose6ptSharedFocal.NullSpace3` (used by both) takes the same
three columns from an unpivoted Householder QR (`LinearAlgebra/Householder.FactorInPlace` /
`ApplyQ`, allocation-free). Both are orthonormal bases of the same space; only the basis
inside it differs, and with it the 280 (shared) or 190 (one-sided) template coefficients,
the action matrix and its eigenvalues, which parametrize F = N0 + x N1 + y N2 in basis
coordinates. The one-sided solver also orders its nine solutions by the eigenvalue order of
`LinearAlgebra/EigenSolver.cs` (Schur order), which may differ from Eigen's.

**Why.** Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR has no published
convention precise enough to reproduce Eigen's Q column for column (entries 24 and 27 are
the same choice for the 7-point and 5-point solvers). In exact arithmetic the solution set does
not depend on the basis (apart from solutions whose F has no N0 component, which the affine
parametrization cannot represent; a measure-zero configuration that moves with the basis).
In floating point it can: the shared-focal solver brackets the real roots of an
ill-conditioned degree-15 characteristic polynomial with Sturm sequences, and which
spurious (non-physical) real roots survive that bracketing depends on the basis, so extra
models can appear or vanish on either side. The output order is not part of the contract: COLMAP treats the models as
a set. What can differ is numerical conditioning near the solvers' thresholds (the Sturm
bracketing tolerance, the 1e-8 tests on imaginary parts and on q = 1/f^2), where a
near-degenerate root can be kept on one side and dropped on the other.

**Evidence.** `RelativePoseSharedFocalTests.RelativePoseSharedFocalEstimator_Nominal` and
`RelativePoseOneSidedFocalTests` (`_Nominal`, `_FullSphereCalibratedRays`) pass 1:1 with
COLMAP's tolerances and failure-rate bounds; with seeds 0 to 4 both failure-rate tests
observed 0 failures in 100 trials (COLMAP's bound is 3). A differential run against the
real PoseLib (the reviewer's harness, fed the same inputs to both sides): one-sided, 574 of
574 models identical; shared focal, 521 of 552 models match to 1e-6, and the true focal is
found in 295 of 300 problems on both sides, the differences being the spurious roots above.

## 39. Bundle adjustment builds its problem in ascending id order

**What differs.** COLMAP's `DefaultBundleAdjuster` (estimators/bundle_adjustment_ceres.cc)
adds residuals by iterating the config's `FlatHashSet`s of images, variable points and
constant points, and `FixGaugeWithThreePoints` picks the three gauge points by iterating a
`FlatHashMap` of per-point observation counts. Abseil's hash containers iterate in an order
that is seeded per process, so COLMAP's residual order, and which three points fix the gauge,
can change from run to run. The port (`Estimators/BundleAdjustmentCeres.Default.cs`,
`BundleAdjustmentCeres.Gauge.cs`) visits the config's images and points in ascending id order
and the observation counts in first-seen order (the order residuals were added), so the
problem layout is the same on every run. The containers COLMAP already orders (`std::set` of
parameterized image and camera ids) are ordered the same way here.

**Why.** CLAUDE.md requires reproducible results; the hash order carries no meaning. The
residual order only changes floating-point summation order inside the solver (Tier C), and
the choice of gauge points only changes which three points stay fixed, never how many.

**Evidence.** `BundleAdjustmentCeresTests` (bundle_adjustment_ceres_test.cc) pass with
COLMAP's exact parameter counts, e.g. `FixGaugeWithThreePoints` and `TwoViewRig` (97 variable
points). `BundleAdjustmentOracleTests` matches pycolmap's final cost to 1e-6 relative and its
poses and points to 1e-6 on a 16-image scene with the TWO_CAMS_FROM_WORLD gauge.

## 44. Rotation averaging lays out its linear system in ascending id order

**What differs.** COLMAP's `RotationAveragingProblem` allocates frame parameters in the
iteration order of a `FlatHashSet<frame_t>`, camera (unknown cam_from_rig) parameters in the
order of a `NodeHashMap<camera_t, int>`, and constraint rows in the order of
`PoseGraph::ValidEdges()` (a `NodeHashMap`); `ComputeResiduals` and the IRLS weights walk the
pair constraints in `NodeHashMap<image_pair_t, …>` order, which is also the order of the
jitter draws near the ±π boundary of the 1-DOF residual. The gauge is fixed at the *first*
active frame in hash order (the first gravity-aligned one, if any).
`Estimators/RotationAveragingProblem*.cs` allocates frames and cameras in ascending id order,
rows in ascending pair id order, visits constraints in that order, and fixes the gauge at the
smallest-id (gravity-aligned, if any) active frame.

The driver code (`Estimators/RotationEstimator.cs`, `Estimators/RotationAveraging.cs`) does the
same where COLMAP walks hash containers: the maximum spanning tree numbers the active images in
ascending id order (COLMAP: `FlatHashSet` order, and node 0 is the tree's root, which is held
at the identity rotation) and adds edges in ascending pair id order (which breaks weight ties);
the stratified gravity subset copies its edges in ascending pair id order;
`InitializeRigRotationsFromImages` walks frames in ascending id order, which fixes the order in
which quaternion samples are summed; `CreateExpandedReconstruction` numbers its singleton rigs
in ascending rig id (then sensor) order and its new frames in ascending frame id (then data id)
order.

**Why.** CLAUDE.md's hash-order rule: the layout changes the floating-point grouping of the
normal equations and the Cholesky ordering, the jitter draw order, and the gauge (the solution
is only defined up to a global rotation, so a different fixed frame gives a globally rotated
result). COLMAP relies on none of these: its tests compare gauge-invariant relative rotations.

**Evidence.** `RotationAveragingTests` (rotation_averaging_test.cc 1:1, all 15 cases) pass
with COLMAP's tolerances, including `DeterministicRandomSeed` and the 1e-12 degree
`RidgeRegularizationDoesNotBiasSolution`; the `CSharpOnly_*` cases recover the synthetic scenes' relative
rotations within COLMAP's 1e-2 degree bar, and `CSharpOnly_DeterministicWithSeed` pins
bit-identical repeat solves.

## 45. Covariance factors a dense Jacobian instead of Ceres' sparse QR

**What differs.** COLMAP's pose refinements (`RefineAbsolutePose`,
`RefineGeneralizedAbsolutePose`) call `ceres::Covariance` with default options: SPARSE_QR,
which factors the problem's sparse tangent-space Jacobian with a column-pivoting sparse QR
(SuiteSparseQR or Eigen's `SparseQR` with COLAMD ordering) and reads (J'J)^-1 from R. The
port (`Solver/Covariance.cs`) builds the same Jacobian densely (loss function applied,
constant blocks and blocks no residual uses without columns, as in Ceres' ComputeCovarianceSparsity) and factors it with the dense column-pivoting Householder
QR of `LinearAlgebra/ColPivHouseholderQR.cs`; rank deficiency (by that QR's rank threshold)
makes `Compute` fail, as in Ceres. The covariance is the same matrix up to round-off.

**Why.** SuiteSparseQR is GPL/LGPL and Eigen's sparse QR is MPL-2.0 (CLAUDE.md contract 2).
COLMAP's covariance callers are single poses (plus a camera) against their observations, a
few columns, where dense is exact enough and fast. The BA covariance
(`estimators/covariance`) is COLMAP's own Schur-based code and does not use this.

**Evidence.** `CovarianceTests` (the SPARSE_QR legs of Ceres' covariance_test.cc
NormalBehavior, ManifoldInTangentSpace and ManifoldInTangentSpaceWithConstantBlocks, at
Ceres' 1e-5 tolerance) and the covariance checks of `PoseEstimationTests` pass.

## 46. IsPanoramicRig compares rig camera origins to the smallest camera index

**What differs.** `IsPanoramicRig` (estimators/generalized_pose.cc) decides whether all
cameras used by a set of correspondences share one optical center: it puts the camera
indices in a `FlatHashSet`, takes the set's first element as the reference, and checks
every other camera's origin with `isApprox(reference, 1e-6)`. The set's iteration order is
unspecified. The port (`Estimators/GeneralizedPoseEstimation.cs`) takes the smallest camera
index as the reference.

**Why.** Hash iteration order is not reproducible across containers (CLAUDE.md, hash
container rule). The choice only matters when origins agree to about 1e-6 relative with some
reference but not another (isApprox is not transitive), i.e. never for a real rig, where
cameras are either co-centered (panoramic) or centimeters apart.

**Evidence.** `GeneralizedPoseEstimationTests` (generalized_pose_test.cc 1:1), whose
EstimateGeneralizedRelativePose_Nominal covers panoramic and non-panoramic rigs, passes.

## 47. Gravity refinement visits error-prone frames and their neighbor pairs in id order

**What.** `GravityRefiner.RefineGravity` (Estimators/GravityRefinement.cs) walks the
error-prone frames, each frame's neighbor pairs and the image adjacency sets in ascending id
order. COLMAP walks absl `FlatHashSet`s, whose order is unspecified.

**Why.** CLAUDE.md requires deterministic iteration. The order reaches the result because
an accepted frame's refined gravity is written into the shared pose prior, which later frames
read as a neighbor gravity; the neighbor order also sets the residual order of each small
solve. There is no order to match: COLMAP's is a property of absl's hash seed.

**Evidence.** `GravityRefinementTests` (gravity_refinement_test.cc 1:1) pass at COLMAP's
1e-2 degree tolerance with 30% gravity outliers.

## 48. Global positioning keeps frame centers and cameras-in-rig in id order

**What.** `GlobalPositioner` (Estimators/GlobalPositioning.cs) keeps its frame centers,
cameras-in-rig and points sorted by id, where COLMAP uses `NodeHashMap`s. The order decides
which frame receives which random initial center (drawn in frame-id order here) and the
element order within the Schur ordering groups.

**Why.** CLAUDE.md requires deterministic iteration. With a fixed seed COLMAP's own random
starts depend on its hash order, so they cannot be reproduced; the solve converges to the
same positions up to the gauge.

**Evidence.** `GlobalPositioningTests`: `GlobalPositioning_Nominal`,
`GlobalPositioning_MultiCameraRig` and `RefineSensorFromRigFalsePreservesRig` (all 1:1) pass
with COLMAP's `ReconstructionNear` bounds.

## 49. Alignment visits hash containers in a fixed order

**What differs.** Three places in `colmap/estimators/alignment.cc` iterate hash containers
where the order reaches a result; `Estimators/Alignment*.cs` fixes each order:
- `MergeReconstructions` copies the source images missing from the target in the iteration
  order of a `FlatHashSet`, which decides the order their frames are registered in the target
  (`RegFrameIds`). The port copies them in the source's `RegImageIds()` order (registration
  order). The source points are merged in `Points3D` order (ascending id, entry 21), which
  decides the ids the new target points get.
- `AlignReconstructionsViaPoints` pairs each source point with the target point seen most
  often along its track via `std::max_element` over a `FlatHashMap`, so a tie goes to
  whichever id the hash map yields first. The port breaks ties by first appearance along the
  source track.
- `AlignReconstructionToOrigRigScales` sums the per-rig scales in the order of the caller's
  rig map (a `NodeHashMap` in C++, the caller's `IReadOnlyDictionary` here), so the mean scale
  can differ from a given COLMAP build in the last ulp. Within a rig the sensors are visited in
  ascending id order in both (`std::map`).

**Why.** CLAUDE.md requires deterministic iteration. COLMAP relies on no particular order: its
tests check only counts (`MergeReconstructions`) and the recovered transform within 1e-6.

**Evidence.** `AlignmentTests` (alignment_test.cc 1:1) pass: `MergeReconstructions`,
`AlignReconstructionsViaPoints` and `AlignReconstructionToOrigRigScales` with COLMAP's
expectations and tolerances.

## 54. BA covariance factors with our sparse products and Cholesky instead of Eigen's

**What differs.** COLMAP's `EstimateBACovariance` (colmap/estimators/covariance.cc) forms the
Schur complement with `Eigen::SparseMatrix` products, eliminates the other parameters with
`Eigen::SimplicialLLT`, and factors the result with `Eigen::SimplicialLDLT` (AMD ordering)
before inverting L densely. The port (`Estimators/Covariance.Schur.cs`) does the same steps
with `LinearAlgebra/SparseMatrixCsc.cs` and `LinearAlgebra/SimplicialCholesky.cs`, whose AMD
ordering and summation order differ from Eigen's. The damping, the rank test (|D| > 1e-6),
the D^-1/2 row scaling with `DBL_MIN` floor and the permutation are COLMAP's. The results
agree to round-off.

**Why.** Eigen is MPL-2.0 (CLAUDE.md contract 2), so its sparse module is replaced, not ported.

**Evidence.** `BACovarianceTests` (covariance_test.cc 1:1, all seven instantiations) passes at
COLMAP's 1e-8 element tolerance against ceres::Covariance; the largest observed difference is
about 2e-12 on pose covariances of about 8e-5.

## 139. ViewGraphCalibration re-estimates each relative pose from a fresh PRNG

**What differs.** `ReestimateRelativePoses` (colmap/estimators/view_graph_calibration.cc), which
the global mapper runs before `GlobalPipeline`, estimates every pair's calibrated two-view
geometry on a new `ThreadPool`. With `random_seed` -1, RANSAC does not seed, so each worker
draws from its `thread_local` PRNG: a fresh worker starts from the default seed on its first
draw, and a worker that takes several pairs carries its stream from one pair to the next. Which
pair gets which draws therefore depends on the thread count and on scheduling; with one thread
the pairs share one stream in pair order. The port's `Parallel.For` also runs iterations on the
calling thread (the controller's thread, whose PRNG the global mapper then uses), so without
isolation those pairs both inherited and consumed the mapper's stream. The port now runs each
pair on a PRNG freshly seeded with `kDefaultPRNGSeed` and restores the thread's own PRNG
afterwards (`Estimators/ViewGraphCalibration.cs`), as entry 71 does for matching. COLMAP's first
pair on each worker matches exactly; later pairs on a worker draw different samples, a Tier C
difference.

**Why.** CLAUDE.md requires sequential and parallel runs to give the same result, and the same
photos must give the same model whatever ran before in the process. COLMAP's own result here is
timing dependent, so there is no fixed stream to reproduce.

**Evidence.**
`AutomaticReconstructionTests.CSharpOnly_UnseededGlobalModelIgnoresTheCallersPrngAndThreadCount`
(C#-only) runs extraction, matching and the global mapper unseeded twice, once with 1 thread
after seeding the caller's PRNG with 1234 and once with 4 threads after seeding it with 98765
and drawing 1000 values; without the per-pair PRNG `0/cameras.bin` differed, and with it all
sparse files are byte-identical and the caller's PRNG is kept. `ViewGraphCalibrationTests` and
`GlobalPipelineTests` pass.
