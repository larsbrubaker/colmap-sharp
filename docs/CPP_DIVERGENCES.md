# Deliberate divergences from COLMAP

A divergence entry records a place where ColmapSharp behaves differently from COLMAP on
purpose (a real upstream bug, a replaced dependency, a deterministic order where COLMAP's
comes from a hash container): what differs, why, and the evidence. Code comments cite them
as "divergence N". Remove an entry when the divergence is gone.

Numbers are stable: an entry keeps its number for good, and a removed entry's number is not
reused, so the gaps in the sequence are intentional. Never renumber, because code comments
and open branches cite these numbers.

## Where entries live

The entries are split by module under `docs/divergences/`, each file in ascending number:

- [`divergences/Mathematics.md`](divergences/Mathematics.md): random numbers, graphs and libm
- [`divergences/Geometry.md`](divergences/Geometry.md): rotations, poses and GPS
- [`divergences/Sensor.md`](divergences/Sensor.md): camera models, bitmaps and EXIF
- [`divergences/Scene.md`](divergences/Scene.md): reconstruction, database, synthesis, rigs and clustering
- [`divergences/Optim.md`](divergences/Optim.md): RANSAC, the Ceres replacement and sparse factorizations
- [`divergences/Estimators.md`](divergences/Estimators.md): minimal solvers, bundle adjustment, rotation averaging and alignment
- [`divergences/Feature.md`](divergences/Feature.md): SIFT and matching
- [`divergences/Sfm.md`](divergences/Sfm.md): incremental and global mapping
- [`divergences/Controllers.md`](divergences/Controllers.md): pipelines, readers, undistorters and the automatic reconstruction
- [`divergences/MvsStereo.md`](divergences/MvsStereo.md): MVS files, PatchMatch and fusion
- [`divergences/MvsMeshing.md`](divergences/MvsMeshing.md): simplification, Poisson, Delaunay and texturing

A new entry goes in the file for the module whose code diverges (usually the `ColmapSharp/`
folder the code lives in), takes the next free number below, and gets a row in the table.
Like every file in the repo, a module file stays within 800 lines; one that would grow past
it is split by topic and the table updated. When merging branches, merge entries one by one
rather than by text hunks, in both the module file and this table.

**Next free number: 142.**

## Entries

| # | Divergence | File |
|---|---|---|
| 1 | Real-valued random draws are not fused (no FMA), unlike the macOS pycolmap wheel | [Mathematics](divergences/Mathematics.md) |
| 2 | Hash-container iteration order in UnionFind and connected components | [Mathematics](divergences/Mathematics.md) |
| 3 | Spanning-tree ties between equal edge weights | [Mathematics](divergences/Mathematics.md) |
| 4 | Stoer-Wagner: which min cut, and which side is labeled 1 | [Mathematics](divergences/Mathematics.md) |
| 5 | Boykov-Kolmogorov max-flow with float capacities | [Mathematics](divergences/Mathematics.md) |
| 6 | FMA contraction in the macOS arm64 pycolmap wheel (quaternion-vector rotation, small products, GPS) | [Geometry](divergences/Geometry.md) |
| 7 | sin(a/2) in the angle-axis to quaternion conversion | [Geometry](divergences/Geometry.md) |
| 8 | CameraDatabase iterates the sensor-width table in specs.cc order, not hash order | [Sensor](divergences/Sensor.md) |
| 9 | Bitmap.Rescale is a managed resampler, not OpenImageIO's resize | [Sensor](divergences/Sensor.md) |
| 10 | ExifReader leaves rationals with a zero denominator unset | [Sensor](divergences/Sensor.md) |
| 11 | UTMToEllipsoid latitude can differ by 1 ulp | [Geometry](divergences/Geometry.md) |
| 12 | FMA contraction in the camera models | [Sensor](divergences/Sensor.md) |
| 13 | Sparse Cholesky: simplicial LLT/LDLT with our AMD instead of CHOLMOD and Eigen | [Optim](divergences/Optim.md) |
| 14 | CorrespondenceGraph lists image pairs in insertion order | [Scene](divergences/Scene.md) |
| 15 | ComputeBoundingBoxAndCentroid sorts instead of std::nth_element | [Geometry](divergences/Geometry.md) |
| 16 | PROSAC's out-of-range sample index fails a Check instead of reading past the data | [Optim](divergences/Optim.md) |
| 17 | RANSAC and LO-RANSAC always run their trial loop serially | [Optim](divergences/Optim.md) |
| 18 | Solver cost and gradient are summed in residual-block order for any thread count | [Optim](divergences/Optim.md) |
| 19 | TinySphereManifold's tangent basis uses Hughes & Moller, not Eigen's unitOrthogonal() | [Estimators](divergences/Estimators.md) |
| 20 | CSV number parsing uses .NET's invariant parser | [Scene](divergences/Scene.md) |
| 21 | Reconstruction iterates its objects in ascending id order | [Scene](divergences/Scene.md) |
| 22 | SPARSE_NORMAL_CHOLESKY factors with the simplicial LLT and its own AMD ordering | [Optim](divergences/Optim.md) |
| 23 | The in-memory database applies a failing write completely or not at all | [Scene](divergences/Scene.md) |
| 24 | The 7-point fundamental solver takes its null space from unpivoted Householder QR | [Estimators](divergences/Estimators.md) |
| 25 | Reconstruction text reading takes whole tokens; image names must be UTF-8 | [Scene](divergences/Scene.md) |
| 26 | re3q3's random change of variables uses a fixed-seed mt19937, not std::rand | [Estimators](divergences/Estimators.md) |
| 27 | The five-point solver takes its null space from unpivoted Householder QR | [Estimators](divergences/Estimators.md) |
| 28 | FromTwoVectors handles nearly opposite vectors with its own half-turn construction | [Geometry](divergences/Geometry.md) |
| 29 | re3q3_rotation's pre-rotation uses a fixed-seed mt19937, not std::rand | [Estimators](divergences/Estimators.md) |
| 30 | The six-point focal relative pose solvers take their null space from unpivoted Householder QR | [Estimators](divergences/Estimators.md) |
| 31 | SynthesizeDataset visits points, images and chained pairs in ascending id order | [Scene](divergences/Scene.md) |
| 32 | SynthesizeImages hands bitmaps to a sink instead of writing image files | [Scene](divergences/Scene.md) |
| 33 | DatabaseCache iterates its objects in ascending id order | [Scene](divergences/Scene.md) |
| 34 | ReconstructionManager.Write breaks point-count ties by index | [Scene](divergences/Scene.md) |
| 35 | The Schur solvers eliminate sequentially with dynamic-size kernels and their own sparse ordering | [Optim](divergences/Optim.md) |
| 36 | A user ParameterBlockOrdering keeps each group in insertion order | [Optim](divergences/Optim.md) |
| 37 | Solve does not edit the caller's ordering, and ITERATIVE_SCHUR never falls back to CGNR | [Optim](divergences/Optim.md) |
| 38 | PoseGraph orders equally large frame components by smallest frame id | [Scene](divergences/Scene.md) |
| 39 | Bundle adjustment builds its problem in ascending id order | [Estimators](divergences/Estimators.md) |
| 40 | ExtractTopScaleFeatures keeps equal-scale keypoints in input order | [Feature](divergences/Feature.md) |
| 41 | The macOS arm64 wheel's VLFeat fuses multiply-adds; the SIFT port does not | [Feature](divergences/Feature.md) |
| 42 | FeatureDescriptorIndex is an exact nearest-neighbor search, not faiss's IVF index | [Feature](divergences/Feature.md) |
| 43 | SIFT never reuses a previous image's gradient, and extraction can be cancelled | [Feature](divergences/Feature.md) |
| 44 | Rotation averaging lays out its linear system in ascending id order | [Estimators](divergences/Estimators.md) |
| 45 | Covariance factors a dense Jacobian instead of Ceres' sparse QR | [Estimators](divergences/Estimators.md) |
| 46 | IsPanoramicRig compares rig camera origins to the smallest camera index | [Estimators](divergences/Estimators.md) |
| 47 | Gravity refinement visits error-prone frames and their neighbor pairs in id order | [Estimators](divergences/Estimators.md) |
| 48 | Global positioning keeps frame centers and cameras-in-rig in id order | [Estimators](divergences/Estimators.md) |
| 49 | Alignment visits hash containers in a fixed order | [Estimators](divergences/Estimators.md) |
| 50 | ObservationManager iterates its image pairs in insertion order and prints its graph | [Sfm](divergences/Sfm.md) |
| 51 | IncrementalTriangulator visits points and pairs in a deterministic order | [Sfm](divergences/Sfm.md) |
| 52 | SpatialPairGenerator ranks neighbors by coordinate-difference distances, not faiss's | [Controllers](divergences/Controllers.md) |
| 53 | Problem.GetParameterBlocks lists blocks in insertion order | [Optim](divergences/Optim.md) |
| 54 | BA covariance factors with our sparse products and Cholesky instead of Eigen's | [Estimators](divergences/Estimators.md) |
| 57 | The covariant SIFT extractor orders equal (octave, level) features stably | [Feature](divergences/Feature.md) |
| 58 | IncrementalMapper searches for the initial pair sequentially | [Sfm](divergences/Sfm.md) |
| 59 | IncrementalMapper breaks ranking ties by image id | [Sfm](divergences/Sfm.md) |
| 60 | UndistortReconstruction keeps each camera's id | [Scene](divergences/Scene.md) |
| 62 | Reading a truncated MVS .bin file throws | [MvsStereo](divergences/MvsStereo.md) |
| 63 | The MVS projection matrices can differ from COLMAP's in the last float bits | [MvsStereo](divergences/MvsStereo.md) |
| 64 | GetMaxOverlappingImages orders equal shared-point counts by image index | [MvsStereo](divergences/MvsStereo.md) |
| 66 | The incremental pipeline has no Caspar (GPU) bundle adjustment options | [Controllers](divergences/Controllers.md) |
| 67 | BundleAdjustmentController takes BundleAdjustmentOptions instead of an OptionManager | [Controllers](divergences/Controllers.md) |
| 68 | The incremental and global pipelines read point colors through a host callback, not image_path | [Controllers](divergences/Controllers.md) |
| 69 | ExtractColorsForAllImages sums colors per image and reduces in image-id order | [Scene](divergences/Scene.md) |
| 70 | Feature matching never waits for pairs it did not queue | [Controllers](divergences/Controllers.md) |
| 71 | Unseeded RANSAC in the matching controllers starts every pair from the default seed | [Controllers](divergences/Controllers.md) |
| 72 | SimplifyMesh collapse costs can differ from COLMAP's in the last bits | [MvsMeshing](divergences/MvsMeshing.md) |
| 73 | SimplifyMesh visits boundary edges in vertex-index order | [MvsMeshing](divergences/MvsMeshing.md) |
| 74 | PoissonRecon arithmetic is strict IEEE, not -ffast-math | [MvsMeshing](divergences/MvsMeshing.md) |
| 75 | The Poisson normal transform uses a correctly rounded pow(x, 1./3), not libm's | [MvsMeshing](divergences/MvsMeshing.md) |
| 76 | Poisson meshing runs in memory; the file-to-file call is a wrapper | [MvsMeshing](divergences/MvsMeshing.md) |
| 77 | ComputeNormalizedMinGraphCut partitions with our own multilevel bisection, not METIS | [Mathematics](divergences/Mathematics.md) |
| 80 | ReadRigConfig requires exactly 4 rotation and 3 translation entries | [Scene](divergences/Scene.md) |
| 81 | ApplyRigConfig hands the reconstruction its rigs and frames in id order | [Scene](divergences/Scene.md) |
| 82 | ImageReader reads images through a host-supplied source, not the file system | [Controllers](divergences/Controllers.md) |
| 83 | Feature extraction commits images in reader order | [Controllers](divergences/Controllers.md) |
| 84 | PatchMatchController's "__auto__" source images order equal counts by image index | [MvsStereo](divergences/MvsStereo.md) |
| 85 | PatchMatch problems list their source images in configured order | [MvsStereo](divergences/MvsStereo.md) |
| 86 | PatchMatch runs on the CPU: no GPU index, and its own random numbers | [MvsStereo](divergences/MvsStereo.md) |
| 87 | StereoFusion traverses on one thread | [MvsStereo](divergences/MvsStereo.md) |
| 88 | StereoFusion lists each point's visible images in ascending index order | [MvsStereo](divergences/MvsStereo.md) |
| 89 | StereoFusion throws on a mask it cannot decode | [MvsStereo](divergences/MvsStereo.md) |
| 90 | Texture mapping's occlusion test runs on a BVH instead of CGAL's AABB tree | [MvsMeshing](divergences/MvsMeshing.md) |
| 91 | Texture mapping breaks view-label and atlas-packing ties by index | [MvsMeshing](divergences/MvsMeshing.md) |
| 92 | Texture mapping keeps one bit per (face, image) instead of a dense score table | [MvsMeshing](divergences/MvsMeshing.md) |
| 93 | Scene and reconstruction clustering break sort ties deterministically | [Scene](divergences/Scene.md) |
| 94 | Flat scene clusters are ordered by size, then smallest image id | [Scene](divergences/Scene.md) |
| 95 | PatchMatch samples source images with exact float bilinear weights | [MvsStereo](divergences/MvsStereo.md) |
| 96 | PatchMatch float math uses .NET's MathF and no contraction | [MvsStereo](divergences/MvsStereo.md) |
| 97 | PatchMatch computes window radii 21 to 32 | [MvsStereo](divergences/MvsStereo.md) |
| 98 | The undistorters hand their images to a host sink, and "copy" re-hands the decoded image | [Controllers](divergences/Controllers.md) |
| 99 | The undistorters run images in batches, so a stop finishes the batch in flight | [Controllers](divergences/Controllers.md) |
| 100 | GlobalMapper::EstablishTracks builds tracks in a deterministic order | [Sfm](divergences/Sfm.md) |
| 101 | RotationAveragingPipeline checks for a stop between stages | [Controllers](divergences/Controllers.md) |
| 102 | RotationAveragingPipeline seeds gravity rotations through the prior's corr_data_id | [Controllers](divergences/Controllers.md) |
| 103 | Delaunay meshing uses our own tetrahedralization, not CGAL's | [MvsMeshing](divergences/MvsMeshing.md) |
| 104 | Subsampled Delaunay triangulation inserts every point until the points span 3D | [MvsMeshing](divergences/MvsMeshing.md) |
| 105 | Sparse Delaunay meshing numbers the points in ascending point3D id | [MvsMeshing](divergences/MvsMeshing.md) |
| 106 | Poisson splatting runs sequentially in sample order | [MvsMeshing](divergences/MvsMeshing.md) |
| 107 | GlobalPipeline orders equally large reconstructions by component | [Controllers](divergences/Controllers.md) |
| 108 | HierarchicalPipeline can be cancelled and then returns before merging | [Controllers](divergences/Controllers.md) |
| 109 | Delaunay meshing assembles the graph and the surface in cell and facet order | [MvsMeshing](divergences/MvsMeshing.md) |
| 110 | Delaunay meshing sums per-image weights in image order for any thread count | [MvsMeshing](divergences/MvsMeshing.md) |
| 111 | When the ray leaves the hull at a point, the sink vote goes to the infinite cell behind it | [MvsMeshing](divergences/MvsMeshing.md) |
| 112 | Delaunay meshing explains an empty cut instead of failing a Check | [MvsMeshing](divergences/MvsMeshing.md) |
| 113 | Reading a truncated binary PLY mesh throws inside the texcoord lists | [MvsMeshing](divergences/MvsMeshing.md) |
| 114 | libm and MathF results can differ from COLMAP's in the last ulp | [Mathematics](divergences/Mathematics.md) |
| 115 | Eigen's SIMD evaluation order is not reproduced in norms, reductions and products | [Optim](divergences/Optim.md) |
| 116 | PoissonRecon's log( float ) is the double logarithm rounded to float | [MvsMeshing](divergences/MvsMeshing.md) |
| 117 | Bitmap interpolation treats points beyond int range and NaN as outside the image | [Sensor](divergences/Sensor.md) |
| 120 | SceneClustering.Create hands the image pairs over in Boost hash order | [Scene](divergences/Scene.md) |
| 121 | HierarchicalPipeline reconstructs every cluster from a fresh PRNG | [Controllers](divergences/Controllers.md) |
| 122 | PatchMatchController runs problems one at a time and aborts the one in flight on stop | [MvsStereo](divergences/MvsStereo.md) |
| 123 | Poisson system assembly adds its shared sums sequentially in node order | [MvsMeshing](divergences/MvsMeshing.md) |
| 124 | LO-RANSAC can start its local optimization from a different five-point solution | [Optim](divergences/Optim.md) |
| 125 | A polygon's barycenter vertex gets the mean of its loop's depths | [MvsMeshing](divergences/MvsMeshing.md) |
| 130 | PoissonMeshing carries only red, green and blue from the input PLY | [MvsMeshing](divergences/MvsMeshing.md) |
| 131 | PoissonRecon's command-line flags do not leak between PoissonMeshing calls | [MvsMeshing](divergences/MvsMeshing.md) |
| 132 | PoissonMeshing trims an empty mesh to an empty mesh instead of crashing | [MvsMeshing](divergences/MvsMeshing.md) |
| 134 | AutomaticReconstructionController runs dense stereo and Delaunay meshing on the CPU, and has no vocabulary tree | [Controllers](divergences/Controllers.md) |
| 135 | AutomaticReconstructionController textures each dense mesh | [Controllers](divergences/Controllers.md) |
| 136 | PatchMatch can run on a host-provided WebGPU device | [MvsStereo](divergences/MvsStereo.md) |
| 137 | AutomaticReconstructionController reseeds the PRNG before Delaunay meshing | [Controllers](divergences/Controllers.md) |
| 138 | AutomaticReconstructionController runs the sparse mapper from a fresh PRNG | [Controllers](divergences/Controllers.md) |
| 139 | ViewGraphCalibration re-estimates each relative pose from a fresh PRNG | [Estimators](divergences/Estimators.md) |
| 140 | AutomaticReconstructionController builds dense/<i> from sparse/<i> | [Controllers](divergences/Controllers.md) |
| 141 | AutomaticReconstructionOptions can keep bundle adjustment off a known camera | [Controllers](divergences/Controllers.md) |
