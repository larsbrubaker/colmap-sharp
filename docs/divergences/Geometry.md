# Divergences from COLMAP: Geometry: rotations, poses and GPS

Part of the divergence log: `docs/CPP_DIVERGENCES.md` is the index and explains the
numbering. Entries are in ascending number; each says what differs, why, and the evidence.

## 6. FMA contraction in the macOS arm64 pycolmap wheel (quaternion-vector rotation, small products, GPS)

**What differs.** `Quaterniond * Vector3d` (Eigen's quaternion-vector rotation) differs from
the pycolmap 4.2.0 macOS arm64 wheel by 1-2 ulps on about half of the inputs. Downstream,
so do the Rigid3d/Sim3d operations built on it (point transform, the translations of
composition and inverse, `TgtOriginInSrc`), `Rigid3d.AdjointInverse` and
`GetCovarianceForRigid3dInverse` (3x3 and 6x6 products), and the `GPSTransform`
ellipsoid/ECEF/ENU/UTM conversions (last bit of the ECEF-scale coordinates, up to
9.3e-10 m).

**Why.** Same cause as entry 1: the wheel is built with contraction on, and it evaluates
the cross products inside the rotation, `a1*b2 - a2*b1`, as `fma(a1, b2, -(a2*b1))`.
ColmapSharp never uses FMA in math paths (CLAUDE.md, "No FMA"), so its results are the
same on every platform. They are expected to match a C++ build that does not contract,
which has not been checked here.

**Evidence.** `oracle/linear_algebra_rotations.py` prints it: re-deriving `q * v` with
ColmapSharp's formula gives 69/138 mismatches against the wheel with plain cross products
and 0/138 with the cross products fused as above.
`RotationOracleTests.ToleranceFields("rotated")` pins the C# result at 1e-14 relative.
The other quaternion operations in that fixture are bit-identical
(`RotationOracleTests.ExactFields`). For the geometry (`oracle/geometry_transforms.py`,
`GeometryOracleTests`): the Rigid3d/Sim3d operations that do not rotate a vector are
bit-identical, and in `EllipsoidToECEF` only the z coordinate, `(N * (1 - e2) + alt) * sin_lat`,
differs; evaluating `N * (1 - e2) + alt` as one FMA takes it from 13/80 mismatches to
3/80 (the script prints this), while x and y (no multiply-add) match on every case. `GeometryOracleTests.ToleranceFields`
pins these at 1e-14 relative (1e-13 for the 6x6 covariance) and 1e-8 m for GPS coordinates.

## 7. sin(a/2) in the angle-axis to quaternion conversion

**What differs.** `AngleAxisd.ToQuaternion` can differ from the wheel by 1 ulp in the vector
part.

**Why.** .NET's `Math.Sin` calls the platform libm `sin`. On some inputs the wheel's
`sin(a/2)` rounds one ulp away from libm `sin`. The cause is not established. One
hypothesis is that the compiler fused the adjacent `sin` and `cos` of the same argument into
a `sincos` call that rounds differently. We do not emulate a compiler's choice of math
routine.

**Evidence.** `oracle/linear_algebra_rotations.py`: all fixture cases but one are
bit-identical with libm `sin`, and that one becomes identical when `sin(a/2)` moves one
ulp. `RotationOracleTests.ToleranceFields("from_axis_angle")` pins it at 1e-14 relative.

## 11. UTMToEllipsoid latitude can differ by 1 ulp

**What differs.** `GPSTransform.UTMToEllipsoid` returns a latitude one ulp away from the
pycolmap 4.2.0 macOS arm64 wheel on 2 of the 80 points in `geometry_transforms.json`;
longitude and altitude, and every other point, are bit-identical.

**Why.** The cause is not established. Re-deriving the conversion in Python with the same
libm (`math.sin`, `math.asin`, `math.cosh`, `math.sinh`) reproduces the C# result exactly,
and fusing the multiply-adds of the xi'/eta' series or of the latitude series into FMAs does
not remove the two mismatches, so it is neither a port bug nor the contraction of entry 6.
It may be another compiler choice in the wheel (as in entry 7). We do not emulate it.

**Evidence.** `GeometryOracleTests.ToleranceFields("gps", "utm_to_ellipsoid")` pins it at
1e-14 relative; the observed gap is 1 ulp (about 7e-15 deg). The Python re-derivation was a
scratch harness following gps.cc term for term.

## 15. ComputeBoundingBoxAndCentroid sorts instead of std::nth_element

**What differs.** `Geometry/Normalization.cs` fully sorts each coordinate list where COLMAP
partitions it with two `std::nth_element` calls. The bounding box (the elements at the two
percentile positions) is the same value, and so is the multiset of elements the centroid
averages, but the order they are summed in differs: COLMAP's is whatever libc++'s
`nth_element` leaves between the two positions. The centroid can therefore differ from
COLMAP's in the last bits.

**Why.** The element order after `nth_element` is an unspecified implementation detail of the
C++ standard library; reproducing it would mean porting libc++'s introselect for one
rounding-level effect. Sorting satisfies every `nth_element` postcondition.

**Evidence.** `normalization_test.cc` passes 1:1 (`NormalizationTests`), including the exact
bounding boxes and the 1e-6 centroid checks.

## 28. FromTwoVectors handles nearly opposite vectors with its own half-turn construction

**What differs.** `Quaterniond.FromTwoVectors` (the replacement for Eigen's
`Quaternion::FromTwoVectors`, used by `Synthetic.SynthesizeDataset` to aim frames) uses
Melax's shortest-arc formula like Eigen does in general, but when the two directions are
nearly opposite (1 + c < 1e-8, c the cosine between them) it composes a half turn about an
axis perpendicular to the first vector (built from the least-aligned coordinate axis) with the
well-conditioned short arc from the negated first vector to the second. Eigen switches branch
at a different threshold (1 + c < 1e-12) and picks its perpendicular axis another way, so for
1 + c < 1e-8 the returned rotation can differ from COLMAP's: for exactly opposite vectors any
half turn about a perpendicular axis is correct and the two libraries pick different ones; for
nearly opposite vectors both map the first direction onto the second, but COLMAP's general
formula there carries errors up to ~1e-8 that ours does not.

**Why.** Eigen is MPL-2.0 and not ported (contract rule 2), so this branch is written here
from first principles; the threshold is where Melax's formula loses more than ~5e-9 relative
accuracy (s = sqrt(2 (1 + c)) with 1 + c known only to ~1e-16 absolute). The general branch,
where all practical inputs land, is unchanged.

**Evidence.** `QuaternionTests.CSharpOnly_FromTwoVectorsOpposite` checks exactly opposite
and nearly opposite inputs (1 + c from 0 to ~5e-9, every least-aligned axis) map the first
direction onto the second within 8e-16 with unit norm. `SyntheticOracleTests` still matches
pycolmap's frame rotations bit for bit (none of those inputs is nearly opposite; view
directions are uniform random, so 1 + c < 1e-8 has probability ~5e-9 per frame).
