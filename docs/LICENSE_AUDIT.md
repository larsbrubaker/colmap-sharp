# License audit

colmap-sharp ships under MIT and has to stay usable in closed-source commercial
products (MatterCAD). This file lists which upstream code may be ported and
which may not. Check it before porting any code that is not COLMAP's own, and
before adding a package reference.

**Pure C#, always.** The library is managed code only: no C/C++, no
P/Invoke, no native packages, no linking against COLMAP or anything else. Every
dependency COLMAP has is either ported to C#, replaced by C# written here, or
dropped.

**The license rule:** port or reference only code under a permissive license (MIT, BSD,
Apache-2.0, zlib, Boost, public domain). Anything copyleft (GPL, LGPL, AGPL, or
MPL file-level copyleft) or "non-commercial" is excluded. Excluded code may not
be read as a guide for a line-by-line transcription either. Where COLMAP depends
on excluded code, we write a replacement from the published algorithm or port a
permissively licensed equivalent, and cite the source in the file header.

## COLMAP's own source (BSD-3-Clause): port freely

All of `src/colmap/**` is BSD-3. Keep the notice in `THIRD_PARTY_NOTICES.md`.
The modules that exist only because of an excluded dependency are listed below.

## COLMAP's dependencies

| Upstream | License | Used by COLMAP for | Decision |
|---|---|---|---|
| Eigen | MPL-2.0 | All linear algebra | **Do not port.** Write our own linear algebra (`ColmapSharp.LinearAlgebra`) from textbook algorithms (Golub & Van Loan). Match Eigen's *documented* semantics (e.g. quaternion `w,x,y,z` layout, `AngleAxis` conventions), not its code. Dense decompositions cite their textbook source in each file header; where a sign or pivoting convention must match, it is taken from LAPACK's published documentation (BSD-3; no LAPACK code is ported) and pinned against numpy. |
| libc++ (LLVM) | Apache-2.0 WITH LLVM-exception | `std::uniform_*_distribution`, `normal_distribution` behind `math/random` | **Ported** (`ColmapSharp/Mathematics/LibcxxRandom.cs`: `uniform_int_distribution` with `__independent_bits_engine`, `generate_canonical`, `uniform_real_distribution`, `normal_distribution`, `std::shuffle`) so seeded RANSAC matches the macOS pycolmap oracle. LLVM's notice is in `THIRD_PARTY_NOTICES.md`. |
| Ceres Solver | BSD-3 | Bundle adjustment, all nonlinear refinement | **Port the subset we use:** Levenberg-Marquardt trust region, Schur complement (dense and iterative), Jets for automatic differentiation, loss functions, manifolds. Add Ceres' notice when the first file lands. |
| SuiteSparse / CHOLMOD | LGPL-2.1+ / GPL | `optim/sparse_cholesky`, LAD, rotation averaging | **Do not port.** Write a supernodal- or simplicial-LDLᵀ sparse Cholesky with AMD ordering from the published algorithms (Davis, *Direct Methods for Sparse Linear Systems*, as a description only; CSparse's code is LGPL). Done as `LinearAlgebra/SimplicialCholesky.cs` (simplicial up-looking LLᵀ/LDLᵀ, Liu's elimination tree) and `LinearAlgebra/AmdOrdering.cs` (Amestoy, Davis and Duff, SIAM J. Matrix Anal. Appl. 1996); no SuiteSparse code was read. |
| METIS | Apache-2.0 | Graph partitioning for hierarchical mapping | Allowed. Not needed until hierarchical mapping; a simpler partitioner may suffice. |
| PoseLib | BSD-3 | Minimal solvers (P3P, 5-pt, generalized pose, focal solvers, homography, essential) | **Port freely.** Ported at commit `fa7280fee27f97aff31ae7f98bab7f583fac7d08` (COLMAP 4.2.0's FetchContent pin) into `ColmapSharp/Estimators/Solvers/PoseLib/`, one C# file per PoseLib source; notice in `THIRD_PARTY_NOTICES.md`. |
| VLFeat (`thirdparty/VLFeat`) | BSD-2 | CPU SIFT extraction | **Port freely.** SIFT's patent (US 6,711,293) expired in March 2020. |
| SiftGPU (`thirdparty/SiftGPU`) | UNC, "educational, research and non-profit purposes" only | GPU SIFT | **Excluded.** CPU SIFT (VLFeat) is the only extractor. |
| LSD (`thirdparty/LSD`) | **AGPL-3.0** | Line segment detection (`image/line`, `estimators/coordinate_frame`) | **Excluded.** Coordinate-frame estimation from lines is skipped, or reimplemented from the published LSD paper (von Gioi et al., IPOL 2012) if it is ever needed. |
| CGAL | **GPL-3.0** / commercial | Delaunay meshing, advancing-front meshing, texture mapping | **Excluded.** Delaunay tetrahedralization comes from MIConvexHull (MIT) or MatterCAD's own solution. Graph cut, visibility scoring and surface extraction are COLMAP's own (BSD) and are ported. |
| PoissonRecon (`thirdparty/PoissonRecon`) | MIT | Poisson surface reconstruction | **Port freely.** Add Kazhdan's notice when the first file lands. |
| faiss | MIT | Nearest-neighbor descriptor matching, retrieval | Not used (native). Write a managed kd-tree / brute-force matcher instead. |
| Symforce-Caspar | Apache-2.0 | GPU bundle adjustment | Not needed (GPU only). |
| OpenImageIO | Apache-2.0 | Image decoding, EXIF | Not used (native). The library takes decoded pixel buffers; the host (MatterCAD/agg-sharp) decodes. EXIF focal length is read by a small managed parser written here. |
| SQLite | Public domain | Feature/match database | Not used (native). The database is an in-memory C# store with the same API; persistence, if needed, is a managed format of our own. |
| Boost, gflags, glog | BSL / BSD | Utilities, CLI, logging | Not ported; BCL replacements. |
| Qt | LGPL / commercial | GUI | **Excluded.** No GUI in this library. |
| ONNX Runtime + models (ALIKED, LightGlue, LoMa, AnyCalib) | MIT runtime; model weights vary | Learned features | Out of scope. Check each model's weights license separately if this is ever revisited. |
| CUDA / HIP | Proprietary toolchains | PatchMatch stereo, GPU SIFT | Not used. PatchMatch stereo is ported to managed CPU code from `mvs/patch_match_cuda.cu` (COLMAP's own BSD code). |

## Package references allowed in `ColmapSharp` (the library)

Only the .NET base class library for now. Adding a package needs a row above,
a permissive license, pure managed code (so the library still runs on
browser-wasm), and a pinned version.
