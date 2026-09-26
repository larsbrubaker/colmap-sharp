// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// HomographyMatrixEstimator: colmap/estimators/solvers/homography_matrix.h and .cc - the
// direct linear transformation (DLT) homography estimators, between image points
// (HomographyMatrixEstimator) and between bearing rays of calibrated cameras
// (HomographyMatrixRayEstimator, whose residual projects through the second Camera,
// Scene/Camera.cs). Both implement Optim/Estimator.cs for RANSAC / LO-RANSAC; the
// two-view geometry estimator (estimators/two_view_geometry.cc) is their consumer.
// Geometry/HomographyMatrix.cs decomposes the result.
// Tests: ColmapSharp.Tests/Estimators/Solvers/HomographyMatrixTests.cs
// (homography_matrix_test.cc 1:1).
//
// Tier B (tolerance): the solution goes through an LU solve (minimal case) or an SVD.
//
// Sign of the null vector: with more than 4 correspondences H is the SVD's last right
// singular vector, whose sign is arbitrary (and not guaranteed to match Eigen's). The pixel
// estimator is sign-independent: its residual divides by the third row of H x, so H and -H
// transfer every point identically, and the |det H| test does not see the sign. The ray
// estimator fixes the sign itself by a vote over the correspondences (COLMAP's comment in
// Estimate explains why it must), so its output is sign-independent of the SVD too. The
// minimal case fixes h[8] = 1 and has no sign freedom.
//
// Hot path: RANSAC calls Estimate once per hypothesis with the minimal 4 correspondences,
// so that case builds its 8 x 9 system and solves it with PartialPivLU's span kernel on
// stackalloc buffers; only the over-determined (local optimization) case allocates.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>
/// A ray together with the image point it was unprojected from. The ray drives the
/// estimation, the image point is what the residual is measured against. Port of
/// colmap::CamRayWithImgPoint.
/// </summary>
public readonly record struct CamRayWithImgPoint(Vector3d Ray, Vector2d ImgPoint);

/// <summary>
/// Direct linear transformation algorithm to compute the homography between point pairs.
/// This algorithm computes the least squares estimate for the homography from at least 4
/// correspondences. Port of colmap::HomographyMatrixEstimator.
/// </summary>
public readonly struct HomographyMatrixEstimator
	: IEstimator<Vector2d, Vector2d, Matrix3d>, ILocalEstimator<Vector2d, Vector2d, Matrix3d>
{
	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 4;

	/// <summary>
	/// Estimate the projective transformation (homography) from at least 4 corresponding
	/// points. Appends no model for a degenerate configuration.
	/// </summary>
	public void Estimate(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, List<Matrix3d> models)
	{
		Check.Eq(points1.Length, points2.Length);
		Check.Ge(points1.Length, 4);

		models.Clear();

		int numPoints = points1.Length;

		// A minimal homography requires four points in general position (no three
		// collinear) in both images. See Hartley and Zisserman, Multiple View
		// Geometry in Computer Vision, 2nd ed., Sec. 4.1.3, pp. 91-92.
		if (numPoints == 4 && (HasCollinearTriplet(points1) || HasCollinearTriplet(points2)))
		{
			return;
		}

		if (numPoints == 4)
		{
			Span<double> a = stackalloc double[8 * 9];
			for (int i = 0; i < numPoints; ++i)
			{
				SetPointRows(a, 8, i, points1[i], points2[i]);
			}

			HomographyDlt.AddIfValid(HomographyDlt.SolveMinimal(a), models);
		}
		else
		{
			var a = new MatrixXd(2 * numPoints, 9);
			Span<double> values = a.AsSpan();
			for (int i = 0; i < numPoints; ++i)
			{
				SetPointRows(values, 2 * numPoints, i, points1[i], points2[i]);
			}

			HomographyDlt.AddIfValid(HomographyDlt.SolveNullspace(a), models);
		}
	}

	/// <summary>A plain estimator: the local estimate re-estimates from the inliers.</summary>
	public void EstimateLocal(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Matrix3d initialModel, List<Matrix3d> models)
	{
		Estimate(points1, points2, models);
	}

	/// <summary>
	/// The squared transformation error of each pair when transforming the source to the
	/// destination coordinates.
	/// </summary>
	public void Residuals(ReadOnlySpan<Vector2d> points1, ReadOnlySpan<Vector2d> points2, in Matrix3d h, Span<double> residuals)
	{
		Check.Eq(points1.Length, points2.Length);
		Check.Eq(residuals.Length, points1.Length);

		// Note that this code might not be as nice as Eigen expressions,
		// but it is significantly faster in various tests.

		double h00 = h[0, 0];
		double h01 = h[0, 1];
		double h02 = h[0, 2];
		double h10 = h[1, 0];
		double h11 = h[1, 1];
		double h12 = h[1, 2];
		double h20 = h[2, 0];
		double h21 = h[2, 1];
		double h22 = h[2, 2];

		for (int i = 0; i < points1.Length; ++i)
		{
			double s0 = points1[i].X;
			double s1 = points1[i].Y;
			double d0 = points2[i].X;
			double d1 = points2[i].Y;

			double pd0 = h00 * s0 + h01 * s1 + h02;
			double pd1 = h10 * s0 + h11 * s1 + h12;
			double pd2 = h20 * s0 + h21 * s1 + h22;

			double invPd2 = 1.0 / pd2;
			double dd0 = d0 - pd0 * invPd2;
			double dd1 = d1 - pd1 * invPd2;

			residuals[i] = dd0 * dd0 + dd1 * dd1;
		}
	}

	/// <summary>
	/// Writes rows 2i and 2i+1 of the constraint matrix (column-major, <paramref name="rows"/>
	/// rows): [x1^T 0 -x2 x1^T] and [0 x1^T -y2 x1^T] with x1 homogeneous.
	/// </summary>
	private static void SetPointRows(Span<double> a, int rows, int i, Vector2d point1, Vector2d point2)
	{
		int r0 = 2 * i;
		int r1 = r0 + 1;
		Span<double> p1 = [point1.X, point1.Y, 1];
		for (int k = 0; k < 3; ++k)
		{
			a[k * rows + r0] = p1[k];
			a[(3 + k) * rows + r0] = 0;
			a[(6 + k) * rows + r0] = -point2.X * p1[k];
			a[k * rows + r1] = 0;
			a[(3 + k) * rows + r1] = p1[k];
			a[(6 + k) * rows + r1] = -point2.Y * p1[k];
		}
	}

	private static bool HasCollinearTriplet(ReadOnlySpan<Vector2d> points)
	{
		return IsCollinear(points, 0, 1, 2) || IsCollinear(points, 0, 1, 3) ||
			IsCollinear(points, 0, 2, 3) || IsCollinear(points, 1, 2, 3);
	}

	private static bool IsCollinear(ReadOnlySpan<Vector2d> points, int i, int j, int k)
	{
		const double kMinNormalizedAreaSquared = 1e-24;
		Vector2d delta1 = points[j] - points[i];
		Vector2d delta2 = points[k] - points[i];
		double scaleSquared = delta1.SquaredNorm * delta2.SquaredNorm;
		double area = delta1.X * delta2.Y - delta1.Y * delta2.X;
		return scaleSquared == 0.0 || area * area <= kMinNormalizedAreaSquared * scaleSquared;
	}
}

/// <summary>
/// Direct linear transformation algorithm to compute the homography between bearing rays.
/// A world plane induces x2 ~ H x1 on the rays of any central camera, whereas image points
/// are related by a 3x3 matrix only under a pinhole projection, which every model but
/// SIMPLE_PINHOLE and PINHOLE breaks. The rays must come from cameras with known
/// intrinsics. The estimated H maps rays to rays; conjugate it as K2 H K1^-1 for the
/// pixel-space homography, which spherical cameras have no calibration matrix for.
/// Port of colmap::HomographyMatrixRayEstimator.
/// </summary>
/// <remarks>
/// RANSAC copies the estimator per thread, so it holds only a reference to the camera;
/// the camera must outlive the estimator. Residuals needs it; Estimate does not.
/// </remarks>
public readonly struct HomographyMatrixRayEstimator(Camera? camera2)
	: IEstimator<Vector3d, CamRayWithImgPoint, Matrix3d>, ILocalEstimator<Vector3d, CamRayWithImgPoint, Matrix3d>
{
	private readonly Camera? _camera2 = camera2;

	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 4;

	/// <summary>
	/// Estimate the projective transformation (homography) between at least 4 corresponding
	/// bearing rays.
	/// </summary>
	public void Estimate(ReadOnlySpan<Vector3d> camRays1, ReadOnlySpan<CamRayWithImgPoint> camRays2, List<Matrix3d> models)
	{
		Check.Eq(camRays1.Length, camRays2.Length);
		Check.Ge(camRays1.Length, 4);

		models.Clear();

		int numRays = camRays1.Length;
		Matrix3d? solution;
		if (numRays == 4)
		{
			Span<double> a = stackalloc double[8 * 9];
			for (int i = 0; i < numRays; ++i)
			{
				SetRayRows(a, 8, i, camRays1[i], camRays2[i].Ray);
			}

			solution = HomographyDlt.SolveMinimal(a);
		}
		else
		{
			var a = new MatrixXd(2 * numRays, 9);
			Span<double> values = a.AsSpan();
			for (int i = 0; i < numRays; ++i)
			{
				SetRayRows(values, 2 * numRays, i, camRays1[i], camRays2[i].Ray);
			}

			solution = HomographyDlt.SolveNullspace(a);
		}

		if (solution is not Matrix3d h || Math.Abs(h.Determinant()) < 1e-8)
		{
			return;
		}

		// H is defined up to scale, but the residual projects H x1 back into an image
		// that does not contain both a direction and its opposite, so the sign
		// matters. It is global, not per correspondence: a visible plane point has
		// positive depth, so x2 ~ lambda H x1 holds with lambda > 0 throughout and
		// the only freedom is that the solver may return -H. Resolving it per point
		// instead would score each against its nearer antipode, letting a 180 degree
		// error pass as a perfect inlier.
		int signVotes = 0;
		for (int i = 0; i < numRays; ++i)
		{
			signVotes += (h * camRays1[i]).Dot(camRays2[i].Ray) > 0 ? 1 : -1;
		}

		if (signVotes < 0)
		{
			h = -h;
		}

		models.Add(h);
	}

	/// <summary>A plain estimator: the local estimate re-estimates from the inliers.</summary>
	public void EstimateLocal(ReadOnlySpan<Vector3d> camRays1, ReadOnlySpan<CamRayWithImgPoint> camRays2, in Matrix3d initialModel, List<Matrix3d> models)
	{
		Estimate(camRays1, camRays2, models);
	}

	/// <summary>
	/// The squared transformation error in the second camera's pixels: the first ray is
	/// transferred through H and projected back into the image. Unlike the essential matrix,
	/// a homography transfers a point to a point rather than to a line, so this is the exact
	/// geometric error and needs no linearization. Rays with no image point, i.e. transferred
	/// behind a perspective camera, are scored at the maximum residual (double.MaxValue).
	/// </summary>
	public void Residuals(ReadOnlySpan<Vector3d> camRays1, ReadOnlySpan<CamRayWithImgPoint> camRays2, in Matrix3d h, Span<double> residuals)
	{
		Check.Eq(camRays1.Length, camRays2.Length);
		Camera camera2 = Check.NotNull(_camera2);
		Check.Eq(residuals.Length, camRays1.Length);

		// Azimuthal models wrap at the +-pi seam, where a raw pixel difference jumps
		// by about the image width. Wrap it into [-width/2, width/2), as
		// WrapEquirectangularHorizontalSeam does for the reprojection error, which
		// spells the same rounding as a floor since it must stay autodiff-safe.
		bool isPeriodic = camera2.IsSpherical;
		double width = camera2.Width;

		for (int i = 0; i < camRays1.Length; ++i)
		{
			Vector2d? imgPoint = camera2.ImgFromCam(h * camRays1[i]);
			if (imgPoint is not Vector2d projected)
			{
				// Transferred out of the camera's field, so there is nothing to score.
				residuals[i] = double.MaxValue;
				continue;
			}

			Vector2d error = projected - camRays2[i].ImgPoint;
			if (isPeriodic)
			{
				// std::round rounds halfway cases away from zero; .NET's default is to even.
				error = new Vector2d(error.X - width * Math.Round(error.X / width, MidpointRounding.AwayFromZero), error.Y);
			}

			residuals[i] = error.SquaredNorm;
		}
	}

	/// <summary>
	/// Writes the two strongest of the three equations x2 x (H x1) = 0 as rows 2i and 2i+1 of
	/// the constraint matrix (column-major, <paramref name="rows"/> rows).
	/// </summary>
	private static void SetRayRows(Span<double> a, int rows, int i, Vector3d ray1, Vector3d ray2)
	{
		// Setup constraint matrix from x2 x (H x1) = 0. Of the three equations, the
		// rows of [x2]_x, only two are independent, and the weakest is always the one
		// omitting the largest component of x2. For a perspective camera z dominates
		// and this reduces to the pixel estimator's rows.
		Span<double> equations = stackalloc double[3 * 9];
		equations.Clear();
		for (int k = 0; k < 3; ++k)
		{
			// equations is row-major 3 x 9 here: equations[row * 9 + col].
			equations[0 * 9 + 3 + k] = -ray2.Z * ray1[k];
			equations[0 * 9 + 6 + k] = ray2.Y * ray1[k];
			equations[1 * 9 + 0 + k] = ray2.Z * ray1[k];
			equations[1 * 9 + 6 + k] = -ray2.X * ray1[k];
			equations[2 * 9 + 0 + k] = -ray2.Y * ray1[k];
			equations[2 * 9 + 3 + k] = ray2.X * ray1[k];
		}

		// Equation j omits component j of x2, so the one to drop is the argmax (the first
		// one on ties, as Eigen's maxCoeff).
		int droppedEquationIdx = 0;
		double maxAbs = Math.Abs(ray2.X);
		if (Math.Abs(ray2.Y) > maxAbs)
		{
			droppedEquationIdx = 1;
			maxAbs = Math.Abs(ray2.Y);
		}

		if (Math.Abs(ray2.Z) > maxAbs)
		{
			droppedEquationIdx = 2;
		}

		int numKept = 0;
		for (int j = 0; j < 3; ++j)
		{
			if (j == droppedEquationIdx)
			{
				continue;
			}

			int row = 2 * i + numKept++;
			for (int col = 0; col < 9; ++col)
			{
				a[col * rows + row] = equations[j * 9 + col];
			}
		}
	}
}

/// <summary>The DLT solve shared by the pixel and ray homography estimators.</summary>
internal static class HomographyDlt
{
	/// <summary>
	/// The minimal case: fixes h[8] = 1 and solves the 8 x 8 block A(:, 0:8) h = -A(:, 8)
	/// by partial-pivot LU. <paramref name="a"/> is the column-major 8 x 9 constraint matrix;
	/// its first 64 entries are exactly the 8 x 8 block. Null when the solve has a NaN.
	/// </summary>
	public static Matrix3d? SolveMinimal(Span<double> a)
	{
		const int n = 8;
		Span<double> rhs = stackalloc double[n];
		for (int i = 0; i < n; ++i)
		{
			rhs[i] = -a[n * n + i];
		}

		Span<int> permutation = stackalloc int[n];
		Span<double> lu = a[..(n * n)];
		PartialPivLU.FactorInPlace(lu, n, permutation);
		Span<double> h = stackalloc double[9];
		PartialPivLU.SolveInPlace(lu, n, permutation, rhs, h[..n]);
		h[8] = 1;
		foreach (double value in h)
		{
			if (double.IsNaN(value))
			{
				return null;
			}
		}

		return FromRowMajor(h);
	}

	/// <summary>
	/// The over-determined case: the right singular vector of the smallest singular value
	/// (the null space of A). Null when A has rank below 8.
	/// </summary>
	public static Matrix3d? SolveNullspace(MatrixXd a)
	{
		// Solve for the nullspace of the constraint matrix.
		var svd = new JacobiSVD(a, SvdOptions.ComputeFullV);
		if (svd.Rank() < 8)
		{
			return null;
		}

		return FromRowMajor(svd.MatrixV().ColumnSpan(8));
	}

	/// <summary>Appends the pixel estimator's model unless it is missing or singular.</summary>
	public static void AddIfValid(Matrix3d? solution, List<Matrix3d> models)
	{
		// Written as !(|det| < 1e-8) so a NaN determinant passes, as in COLMAP.
		if (solution is Matrix3d h && !(Math.Abs(h.Determinant()) < 1e-8))
		{
			models.Add(h);
		}
	}

	// Eigen::Map<const Matrix3d>(h.data()).transpose(): h holds H row by row.
	private static Matrix3d FromRowMajor(ReadOnlySpan<double> h) =>
		new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
}
