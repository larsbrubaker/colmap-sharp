# Divergences from COLMAP: Sensor: camera models, bitmaps and EXIF

Part of the divergence log: `docs/CPP_DIVERGENCES.md` is the index and explains the
numbering. Entries are in ascending number; each says what differs, why, and the evidence.

## 8. CameraDatabase iterates the sensor-width table in specs.cc order, not hash order

**What differs.** COLMAP's `camera_specs_t` is a `NodeHashMap` (a `std::unordered_map` or
`boost::unordered_node_map`, depending on the build), and `CameraDatabase::QuerySensorWidth`
iterates it, writing the output width on every match and stopping after the second
non-exact match per make. `CameraSpecs.InitializeCameraSpecs` returns a list in `specs.cc`
source order, so when a cleaned EXIF make matches more than one table make (a substring
match either way round, e.g. an empty make matches all of them), which widths are seen,
and so the width left behind and whether a unique match is found, can differ from a given
COLMAP build.

**Why.** Hash iteration order is unspecified and differs between standard libraries and
Boost, so there is no single COLMAP behavior to match; CLAUDE.md asks for deterministic
order here.

**Evidence.** `database_test.cc`'s cases (ported in `CameraDatabaseTests`) match a single
make and pass. Queries whose make matches one table make are unaffected.

## 9. Bitmap.Rescale is a managed resampler, not OpenImageIO's resize

**What differs.** COLMAP's `Bitmap::Rescale` calls `OIIO::ImageBufAlgo::resize` with a
"triangle" (kBilinear) or "box" (kBox) filter. `ColmapSharp/Sensor/BitmapResize.cs`
reimplements the model OIIO's output follows (filter widened by the downsampling ratio,
clamp-to-edge samples, separable, round to nearest). Bilinear results are within one gray
level of pycolmap's; box results agree except where a source pixel center lies exactly on
the box edge at a non-integer ratio, where OIIO's inclusion rule is not reproduced and a
destination pixel can average one source pixel more or fewer.

**Why.** OpenImageIO is native (docs/LICENSE_AUDIT.md). Its resize accumulates in float
with its own filter evaluation, so bit-exact output would need a port of OIIO's
resampling code (Apache-2.0, allowed but not done).

**Evidence.** `oracle/fixture_bitmap_rescale.py` records pycolmap 4.2.0's bilinear output on
seeded grey and RGB images, up and down; `BitmapRescaleOracleTests` checks every pixel
within one gray level. Impulse probes (a single lit pixel, 1-D and 2-D) match pycolmap
exactly for both filters at ratios 4, 8, 3, 1.5, 5/3, 2/3 and 3/5. The box tie case: 23 -> 10
pixels, destination pixel 5 (center 12.65) excludes source pixel 11 (center 11.5, distance
1.15 = half the box) in pycolmap, and a half-open box fails other cases, so the rule is
not simply half-open.

## 10. ExifReader leaves rationals with a zero denominator unset

**What differs.** An EXIF RATIONAL with denominator 0 (FocalLength, FocalPlaneXResolution,
GPSLatitude/Longitude, GPSAltitude) is not stored in the Bitmap's metadata by
`ColmapSharp/Sensor/ExifReader.cs`. OpenImageIO, through which COLMAP reads EXIF, most
likely stores the float quotient (inf, or NaN for 0/0), which COLMAP's getters would see.

**Why.** An inf/NaN focal length or GPS coordinate carries no information, and cameras
write 0/0 to mean "unknown", so "absent" is the faithful reading. It is visible only
through the getters: `ExifLatitude`/`ExifLongitude`/`ExifAltitude` return null here where
COLMAP could return NaN or inf, and `ExifFocalLength` returns null (or a later fallback's
value) where COLMAP could return inf or NaN from a zero-denominator FocalLength.

**Evidence.** Not verified against OIIO: no oracle fixture carries a zero-denominator tag.
The reader's behavior is stated in its file header.

## 12. FMA contraction in the camera models

**What differs.** Camera model projection (`CameraModelImgFromCam`) and ray unprojection
(`CameraModelCamRayFromImg`) of every perspective model, and `CameraModelCamFromImg` of the
fisheye, division, FOV and EUCM models, differ from the pycolmap 4.2.0 macOS arm64 wheel by
a few ulps on part of the inputs (at most 1.6e-14 relative to max(1, |value|) in the
fixture). Which calls succeed or fail never differs.

**Why.** Same cause as entries 1 and 6: the wheel is built with contraction on and fuses
multiply-adds that sit in one C++ statement, e.g. `*x = f * *x + c1` in every model's
`ImgFromCam` and `u * u + v * v + 1.0` in `CamRayFromImg`. ColmapSharp never uses FMA in
math paths (CLAUDE.md, "No FMA"). The iterative undistortion runs its distortion on
`ceres::Jet`, whose operators are separate function calls that clang does not contract, and
it matches the wheel bit for bit.

**Evidence.** `oracle/camera_models.py` prints it: re-deriving SIMPLE_RADIAL's projected x
with ColmapSharp's formula matches the wheel on 64/75 and 60/75 points of the two parameter
sets, and on 75/75 with only `f * x + c1` fused; PINHOLE's ray z matches on 98/101 plain and
101/101 with `u*u + v*v` fused as `fma(u, u, v*v)`.
`CameraModelOracleTests` requires bit-identical `CamFromImg` for the models whose
unprojection is the iterative undistortion or a plain pinhole, and for all of
EQUIRECTANGULAR, and pins everything else at 2e-14 relative to max(1, |value|).

**Related, not observed here.** C++ `EquirectangularCameraModel` evaluates
`2.0 * EIGEN_PI * (...)` with EIGEN_PI a `long double` literal, so on x86-64 Linux (80-bit
long double) its results may differ from both the macOS wheel (where long double is double)
and ColmapSharp, which uses `Math.PI` in double.

## 117. Bitmap interpolation treats points beyond int range and NaN as outside the image

**What differs.** `Bitmap::InterpolateBilinear` computes `x0 = static_cast<int>(std::floor(x))`,
`x1 = x0 + 1` and rejects the point when `x0 < 0 || x1 >= width_` (likewise for y);
`InterpolateNearestNeighbor` casts `std::round(x)`. For a point beyond int range or NaN the
cast is undefined behavior in C++. The port tests the doubles first (`x >= 0 && x < width - 1`,
the same check for every finite x) and returns null for NaN in both methods, so such points
are outside the image.

**Why.** .NET saturates `(int)double` (and maps NaN to 0), so the literal translation turned
`floor(x) = int.MaxValue` into `x1 = int.MinValue`, passed the bounds check and indexed far
outside the pixel array; NaN sampled pixel (0, 0). On x86 COLMAP's cast yields `INT_MIN`, which
the check rejects, so returning null is what COLMAP does there; on arm64 COLMAP reads out of
bounds.

**Evidence.** Rectifying a synthetic stereo pair (`Warp.WarpImageWithHomographyBetweenCameras`)
maps some target pixels to source points beyond int range for about 2.5% of PRNG seeds; this
made `UndistortersTests.StereoImageRectifier_Integration` fail intermittently depending on the
thread-static PRNG state earlier tests left behind. `UndistortionTests.CSharpOnly_RectifyAndUndistortStereoImages_FarSourceSamples`
(seed 25) and `BitmapTests.CSharpOnly_InterpolateFarOutsideOrNaN_ReturnsNull` pin it.
