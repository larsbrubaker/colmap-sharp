# Third-party notices

colmap-sharp is MIT licensed (see `LICENSE`). Code in it is ported from the
projects below. Their licenses allow commercial use but require these notices
to stay with the source and with binary redistributions. Add a section here in
the same change that ports code from a new upstream. `docs/LICENSE_AUDIT.md`
lists which upstreams may be ported and which are excluded.

## COLMAP (BSD-3-Clause)

Source: https://github.com/colmap/colmap (reference version pinned in `REFERENCE`)

```
Copyright (c) 2016, ETH Zurich and UNC Chapel Hill.
Copyright (c) 2016-2026, The COLMAP Contributors.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
      its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## Ceres Solver (BSD-3-Clause)

Source: http://ceres-solver.org (https://github.com/ceres-solver/ceres-solver)

Ported from Ceres 2.2.0 in `ColmapSharp/Solver/`: the dual-number arithmetic and
elementary-function rules of `ceres::Jet` (`include/ceres/jet.h`) in `Jet.cs`; the
angle-axis/quaternion conversions of `include/ceres/rotation.h` in `Rotation.cs`; the loss
functions of `internal/ceres/loss_function.cc` in `LossFunctions.cs` (with
`loss_function_test.cc` ported in the tests); the Euclidean, subset, quaternion, sphere and
product manifolds of `internal/ceres/manifold.cc`, `include/ceres/sphere_manifold.h`,
`include/ceres/internal/sphere_manifold_functions.h`,
`include/ceres/internal/householder_vector.h` and `include/ceres/product_manifold.h` in
`Manifolds.cs`; the cost-function contract of `include/ceres/cost_function.h` and
`include/ceres/autodiff_cost_function.h` in `AutoDiffCostFunction.cs`; the TinySolver
function adapter of `include/ceres/tiny_solver_autodiff_function.h` in
`TinySolverAutoDiffFunction.cs`; the problem, program,
parameter/residual blocks, loss corrector, program evaluator, dense and block-sparse
Jacobians, the DENSE_QR / DENSE_NORMAL_CHOLESKY / SPARSE_NORMAL_CHOLESKY linear solvers, the
Levenberg-Marquardt strategy, the trust-region minimizer and `ceres::Solve` with its options
and summary (`internal/ceres/problem_impl.cc`, `program.cc`, `parameter_block.h`,
`residual_block.cc`, `corrector.cc`, `program_evaluator.h`, `block_jacobian_writer.cc`,
`block_sparse_matrix.cc`, `dense_sparse_matrix.cc`, `dense_qr_solver.cc`,
`dense_normal_cholesky_solver.cc`, `sparse_normal_cholesky_solver.cc`,
`levenberg_marquardt_strategy.cc`, `trust_region_step_evaluator.cc`,
`trust_region_minimizer.cc`, `trust_region_preprocessor.cc`, `minimizer.cc`, `solver.cc`) in
`Problem.cs`, `Program.cs`, `ParameterBlock.cs`, `ResidualBlock.cs`, `ProgramEvaluator.cs`,
`SparseMatrix.cs`, `BlockSparseMatrix.cs`, `LinearSolvers.cs`, `TrustRegionStrategy.cs`,
`TrustRegionMinimizer.cs`, `SolverTypes.cs` and `LeastSquaresSolver.cs`; the Schur
complement solvers DENSE_SCHUR / SPARSE_SCHUR / ITERATIVE_SCHUR, their elimination ordering,
conjugate gradients and the JACOBI / SCHUR_JACOBI preconditioners
(`internal/ceres/schur_eliminator.h`, `schur_eliminator_impl.h`,
`schur_complement_solver.cc`, `block_random_access_dense_matrix.cc`,
`block_random_access_sparse_matrix.cc`, `block_random_access_diagonal_matrix.cc`,
`small_blas.h`, `invert_psd_matrix.h`, `implicit_schur_complement.cc`,
`partitioned_matrix_view_impl.h`, `conjugate_gradients_solver.h`,
`iterative_schur_complement_solver.cc`, `schur_jacobi_preconditioner.cc`, `preconditioner.h`,
`parameter_block_ordering.cc`, `graph_algorithms.h`, `reorder_program.cc`) in
`SchurEliminator.cs`, `SchurComplementSolvers.cs`, `BlockRandomAccessMatrix.cs`,
`SmallBlas.cs`, `ImplicitSchurComplement.cs`, `IterativeSchurSolver.cs` and
`SchurOrdering.cs`; the gradient checker and Ridders numeric differentiation
(`include/ceres/gradient_checker.h`, `internal/ceres/gradient_checker.cc`,
`internal/ceres/is_close.cc`, `include/ceres/numeric_diff_options.h`, and the RIDDERS path
of `include/ceres/internal/numeric_diff.h` and
`include/ceres/dynamic_numeric_diff_cost_function.h`) in `GradientChecker.cs` (with `corrector_test.cc`, `trust_region_minimizer_test.cc`,
`schur_eliminator_test.cc`, `implicit_schur_complement_test.cc`,
`conjugate_gradients_solver_test.cc`, `schur_complement_solver_test.cc`,
`iterative_schur_complement_solver_test.cc`, `parameter_block_ordering_test.cc`,
`small_blas_test.cc`, `block_random_access_dense_matrix_test.cc`,
`block_random_access_sparse_matrix_test.cc`, `block_random_access_diagonal_matrix_test.cc`,
parts of `graph_algorithms_test.cc` and `reorder_program_test.cc`, and problems 2-4 of
`linear_least_squares_problems.cc` ported in the tests, and the data of
`examples/curve_fitting.cc` in `CeresExampleTests.cs`); parameter bounds, `Problem::Evaluate`
and `CRSMatrix`, the user `ParameterBlockOrdering` (`include/ceres/ordered_groups.h`,
`include/ceres/problem.h`, `include/ceres/crs_matrix.h`, `ApplyOrdering` in
`reorder_program.cc`) and the trust-region minimizer's projected Armijo line search with its
polynomial helpers (`internal/ceres/line_search.cc`, `line_search.h`, `polynomial.cc`,
`polynomial.h`, `function_sample.h`) in `ParameterBlock.cs`, `Problem.cs`,
`Problem.Evaluate.cs`, `ParameterBlockOrdering.cs`, `SchurOrdering.cs`,
`TrustRegionLineSearch.cs` and `CeresPolynomial.cs` (with `ordered_groups_test.cc`,
`polynomial_test.cc`, the
bounds and `ProblemEvaluateTest` cases of `problem_test.cc`, the bounds cases of
`parameter_block_test.cc` and `evaluator_test_utils.cc` ported in the tests).
`ColmapSharp/Optim/TinySolver.cs` ports COLMAP's
`colmap/optim/tiny_solver.h`, which is COLMAP's modified copy of Ceres'
`include/ceres/tiny_solver.h` and carries this notice.

```
Ceres Solver - A fast non-linear least squares minimizer
Copyright 2023 Google Inc. All rights reserved.
http://ceres-solver.org/

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of Google Inc. nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## VLFeat (BSD-2-Clause)

Source: https://www.vlfeat.org, as vendored by COLMAP in `src/thirdparty/VLFeat` (reference
version pinned in `REFERENCE`)

Ported in `ColmapSharp/Feature/VLFeat/`: the SIFT filter of `sift.c`/`sift.h` (including
`vl_sift_calc_raw_descriptor`), the DoG covariant detector of `covdet.c`/`covdet.h` (detection,
affine adaptation, orientations, patch extraction), the Gaussian scale space of
`scalespace.c`/`scalespace.h`, and the parts of `mathop.h`/`mathop.c` (including `vl_svd2`,
`vl_lapack_dlasv2` and `vl_gaussian_elimination`) and `imopv.c` (`vl_imconvcol_vf`,
`vl_imsmooth_f`, `vl_imgradient_f`, `vl_imgradient_polar_f`) they use.

```
Copyright (C) 2007-11, Andrea Vedaldi and Brian Fulkerson
Copyright (C) 2012-13, The VLFeat Team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Boost.Unordered (BSL-1.0)

Source: https://github.com/boostorg/unordered (tag boost-1.87.0, commit
b41c054c66e7c71d6c6105a1e9da096f6731434d). `ColmapSharp/Scene/SceneClustering.cs` ports the
`mulx` hash mixer from `boost/unordered/detail/mulx.hpp`.

```
Copyright 2022 Peter Dimov.
Copyright 2022 Joaquin M Lopez Munoz.

Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

## googletest (BSD-3-Clause)

Source: https://github.com/google/googletest. Used only by the test project:
`ColmapSharp.Tests/GTestDouble.cs` ports `FloatingPoint<double>::AlmostEquals`
from `gtest-internal.h`.

```
Copyright 2008, Google Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## PoseLib (BSD-3-Clause)

Source: https://github.com/PoseLib/PoseLib (commit
fa7280fee27f97aff31ae7f98bab7f583fac7d08, the version COLMAP's
`src/thirdparty/CMakeLists.txt` fetches)

Ported in `ColmapSharp/Estimators/Solvers/PoseLib/`, one file (or partial-class
group) per PoseLib source file it ports: the generalized relative pose solver of
`PoseLib/solvers/gen_relpose_6pt.cc` in `GenRelpose6pt*.cs` (with the helpers it
uses from `misc/quaternion.h` and `misc/essential.cc`); `CameraPose` (`PoseLib/camera_pose.h`,
`misc/quaternion.h`); the camera description of `misc/camera_models.h/.cc`
(`PoseLibCamera.cs`); the Sturm-sequence root finder of `misc/sturm.h`; the
three-quadratics solver of `misc/re3q3.cc`; the P3P solver of `solvers/p3p.cc` with
`solvers/p3p_common.h` and `misc/univariate.cc`'s single-real-root cubic; the P4Pf
solver of `solvers/p4pf.cc`; the generalized P3P solver of `solvers/gp3p.cc`
(`Gp3p.cs`, with `re3q3_rotation`, `rotation_to_3q3` and `quat_multiply` in `Re3q3.cs`);
the five-point essential matrix solver of `solvers/relpose_5pt.cc` (`Relpose5pt.cs`); the
shared-focal relative pose solver of `solvers/relpose_6pt_focal.cc`
(`Relpose6ptSharedFocal*.cs`, with `charpoly_danilevsky_piv` of `misc/sturm.h` in
`Sturm.cs`); the one-sided focal relative pose solver of
`solvers/relpose_6pt_onesided_focal.cc` (`Relpose6ptOnesidedFocal*.cs`, full template);
`motion_from_essential` and `check_cheirality` of `misc/essential.cc` (`Essential.cs`); and
`ImagePair` of `camera_pose.h` and `Camera::focal` of `misc/camera_models.cc`.

```
BSD 3-Clause License

Copyright (c) 2020, Viktor Larsson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## LLVM libc++ (Apache-2.0 WITH LLVM-exception)

Source: https://github.com/llvm/llvm-project (`libcxx/include/__random/*`,
`libcxx/include/__algorithm/shuffle.h`, and the heap algorithms `make_heap.h`,
`push_heap.h`, `pop_heap.h` and `sift_down.h` in `libcxx/include/__algorithm/`, and
`libcxx/include/__hash_table`)

Ported in `ColmapSharp/Mathematics/LibcxxRandom.cs`: the algorithms of
`std::uniform_int_distribution` (with `__independent_bits_engine`), `generate_canonical`,
`uniform_real_distribution`, `normal_distribution` and `std::shuffle`, so seeded draws
match the libc++-built pycolmap oracle. Ported in `ColmapSharp/Mvs/CollapseHeap.cs`:
`std::priority_queue`'s heap (`__sift_down`, `__floyd_sift_down`, `__sift_up`), so mesh
simplification pops tied candidates in the wheel's order. Ported in
`ColmapSharp/Util/LibcxxUnorderedMap.cs`: `std::unordered_map`'s hash table (insertion,
rehashing and iteration order), so PoissonRecon's sparse matrix products order row entries as
libc++ does. Copyright the LLVM Project contributors.

```
==============================================================================
The LLVM Project is under the Apache License v2.0 with LLVM Exceptions:
==============================================================================

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

    1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

    2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

    3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

    4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

    5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

    6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

    7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

    8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

    9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

    END OF TERMS AND CONDITIONS

    APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

    Copyright [yyyy] [name of copyright owner]

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

---- LLVM Exceptions to the Apache 2.0 License ----

As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into an Object form of such source code, you
may redistribute such embedded portions in such Object form without complying
with the conditions of Sections 4(a), 4(b) and 4(d) of the License.

In addition, if you combine or link compiled forms of this Software with
software that is licensed under the GPLv2 ("Combined Software") and if a
court of competent jurisdiction determines that the patent provision (Section
3), the indemnity provision (Section 9) or other Section of the License
conflicts with the conditions of the GPLv2, you may retroactively and
prospectively choose to deem waived or otherwise exclude such Section(s) of
the License, but only in their entirety and only with respect to the Combined
Software.
```

## PoissonRecon (MIT; per-file BSD-3-Clause-style headers)

Source: https://github.com/mkazhdan/PoissonRecon, as vendored by COLMAP in
`src/thirdparty/PoissonRecon` (reference version pinned in `REFERENCE`)

Ported in `ColmapSharp/Mvs/PoissonRecon/` so far: `Polynomial.h`/`.inl`,
`BSplineData.h`/`.inl`, parts of `Geometry.h`/`.inl` (XForm), `PointExtent.h`/`.inl`,
`RegularTree.h`/`.inl`, `Window.h`/`.inl` (the neighbor-window layout), `FEMTree.h`/`.inl`,
`FEMTree.Initialize.inl`, `FEMTree.SortedTreeNodes.inl`, `FEMTree.WeightedSamples.inl`,
`Reconstructors.h` and `Reconstructors.streams.h`. The
folder's `LICENSE` is MIT; the source files also carry the Johns Hopkins BSD-style header
below, whose copyright line varies per file. The distinct lines of the files ported so far:

```
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
Copyright (c) 2016, Michael Kazhdan
Copyright (c) 2022, Michael Kazhdan and Matthew Bolitho
Copyright (c) 2023, Michael Kazhdan
```

Both notices are reproduced; the BSD-style text below is identical in every file apart from
that line.

```
The MIT License (MIT)

Copyright (c) 2015 mkazhdan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
```

## Jonathan Richard Shewchuk's robust geometric predicates (public domain)

Source: J. R. Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust
Geometric Predicates", Discrete & Computational Geometry 18:305-363, 1997, and its
`predicates.c` (https://www.cs.cmu.edu/~quake/robust.html). The author placed that code in
the public domain, so no license terms apply; `ColmapSharp/Geometry/Delaunay/RobustPredicates.cs`
uses its floating-point filter formulas and error bounds and credits it here.
