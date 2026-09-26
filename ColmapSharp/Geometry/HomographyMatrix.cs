// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// HomographyMatrix: colmap/geometry/homography_matrix.h/.cc - the analytical
// decomposition of a calibrated homography into its (up to four) rotation, translation and
// plane-normal candidates (Malis and Vargas, "Deeper understanding of the homography
// decomposition for vision-based control", INRIA RR-6303, 2007), picking the candidate by
// cheirality and reprojection error, building H from a pose and plane, and the homography
// transfer error. Depends on Triangulation.cs (TriangulateMidPoint), Rigid3d.cs and
// MathUtils.SignOfNumber. Tests: ColmapSharp.Tests/Geometry/HomographyMatrixTests.cs
// (homography_matrix_test.cc 1:1) and GeometryTwoViewOracleTests (C#-only, pycolmap).
//
// Tiers: HomographyMatrixFromPose and ComputeSquaredHomographyError are scalar products;
// DecomposeHomographyMatrix scales H by its middle singular value (sign-free: singular
// values are non-negative) and is otherwise closed-form, so its results match COLMAP to
// rounding, Tier B. PoseFromHomographyMatrix adds TriangulateMidPoint (Tier B, its SVD null
// vector enters only through a ratio, see Triangulation.cs).
//
// For noise-free correspondences of a plane, two of the four candidates are usually both
// physically valid: every point triangulates in front of both cameras with a reprojection
// sum at rounding level (~1e-16). PoseFromHomographyMatrix then picks between them by
// rounding, in COLMAP as here, so the two can disagree on such exact data; with any noise
// the choice is well defined and matches (GeometryTwoViewOracleTests).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Geometry;

/// <summary>
/// Port of the free functions in colmap/geometry/homography_matrix.h.
/// </summary>
public static class HomographyMatrix
{
	/// <summary>
	/// Decomposes H (between cameras with calibrations K1 and K2) into the possible
	/// relative poses and plane normals: one (rotation, zero translation, zero normal)
	/// for a pure rotation, otherwise four candidates.
	/// Port of colmap::DecomposeHomographyMatrix.
	/// </summary>
	public static void DecomposeHomographyMatrix(
		in Matrix3d h,
		in Matrix3d k1,
		in Matrix3d k2,
		List<Rigid3d> cams2FromCams1,
		List<Vector3d> normals)
	{
		// Remove calibration from homography.
		Matrix3d hNormalized = k2.Inverse() * h * k1;

		// Remove scale from normalized homography.
		Svd3d hmatrixNormSvd = Svd3d.Compute(hNormalized);
		hNormalized /= hmatrixNormSvd.SingularValues.Y;

		// Ensure that we always return rotations, and never reflections.
		//
		// It's enough to take det(H_normalized) > 0.
		//
		// To see this:
		// - In the paper: R := H_normalized * (Id + x y^t)^{-1} (page 32).
		// - Can check that this implies that R is orthogonal: RR^t = Id.
		// - To return a rotation, we also need det(R) > 0.
		// - By Sylvester's idenitity: det(Id + x y^t) = (1 + x^t y), which
		//   is positive by choice of x and y (page 24).
		// - So det(R) and det(H_normalized) have the same sign.
		if (hNormalized.Determinant() < 0)
		{
			hNormalized *= -1.0;
		}

		Matrix3d s = hNormalized.Transpose() * hNormalized - Matrix3d.Identity;

		cams2FromCams1.Clear();
		normals.Clear();

		// Check if H is rotation matrix.
		const double MinInfinityNorm = 1e-3;
		if (MaxAbsCoefficient(s) < MinInfinityNorm)
		{
			cams2FromCams1.Add(new Rigid3d(Quaterniond.FromRotationMatrix(hNormalized), Vector3d.Zero));
			normals.Add(Vector3d.Zero);
			return;
		}

		double m00 = ComputeOppositeOfMinor(s, 0, 0);
		double m11 = ComputeOppositeOfMinor(s, 1, 1);
		double m22 = ComputeOppositeOfMinor(s, 2, 2);

		double rtM00 = Math.Sqrt(Math.Max(m00, 0.0));
		double rtM11 = Math.Sqrt(Math.Max(m11, 0.0));
		double rtM22 = Math.Sqrt(Math.Max(m22, 0.0));

		double m01 = ComputeOppositeOfMinor(s, 0, 1);
		double m12 = ComputeOppositeOfMinor(s, 1, 2);
		double m02 = ComputeOppositeOfMinor(s, 0, 2);

		int e12 = MathUtils.SignOfNumber(m12);
		int e02 = MathUtils.SignOfNumber(m02);
		int e01 = MathUtils.SignOfNumber(m01);

		double nS00 = Math.Abs(s[0, 0]);
		double nS11 = Math.Abs(s[1, 1]);
		double nS22 = Math.Abs(s[2, 2]);

		// std::max_element: the first of equal maxima.
		int idx = 0;
		if (nS11 > nS00)
		{
			idx = 1;
		}

		if (nS22 > (idx == 0 ? nS00 : nS11))
		{
			idx = 2;
		}

		Vector3d np1;
		Vector3d np2;
		if (idx == 0)
		{
			np1 = new Vector3d(s[0, 0], s[0, 1] + rtM22, s[0, 2] + e12 * rtM11);
			np2 = new Vector3d(s[0, 0], s[0, 1] - rtM22, s[0, 2] - e12 * rtM11);
		}
		else if (idx == 1)
		{
			np1 = new Vector3d(s[0, 1] + rtM22, s[1, 1], s[1, 2] - e02 * rtM00);
			np2 = new Vector3d(s[0, 1] - rtM22, s[1, 1], s[1, 2] + e02 * rtM00);
		}
		else
		{
			np1 = new Vector3d(s[0, 2] + e01 * rtM11, s[1, 2] + rtM00, s[2, 2]);
			np2 = new Vector3d(s[0, 2] - e01 * rtM11, s[1, 2] - rtM00, s[2, 2]);
		}

		double traceS = s.Trace();
		double v = 2.0 * Math.Sqrt(Math.Max(1.0 + traceS - m00 - m11 - m22, 0.0));

		double eSii = MathUtils.SignOfNumber(s[idx, idx]);
		double r2 = 2 + traceS + v;
		double nt2 = 2 + traceS - v;

		double r = Math.Sqrt(Math.Max(r2, 0.0));
		double nT = Math.Sqrt(Math.Max(nt2, 0.0));

		Vector3d n1 = np1.Normalized();
		Vector3d n2 = np2.Normalized();

		double halfNt = 0.5 * nT;
		double esiiTimesR = eSii * r;

		Vector3d t1Star = halfNt * (esiiTimesR * n2 - nT * n1);
		Vector3d t2Star = halfNt * (esiiTimesR * n1 - nT * n2);

		Matrix3d r1 = ComputeHomographyRotation(hNormalized, t1Star, n1, v);
		Vector3d t1 = r1 * t1Star;

		Matrix3d rot2 = ComputeHomographyRotation(hNormalized, t2Star, n2, v);
		Vector3d t2 = rot2 * t2Star;

		Quaterniond q1 = Quaterniond.FromRotationMatrix(r1);
		Quaterniond q2 = Quaterniond.FromRotationMatrix(rot2);
		cams2FromCams1.Add(new Rigid3d(q1, t1));
		cams2FromCams1.Add(new Rigid3d(q1, -t1));
		cams2FromCams1.Add(new Rigid3d(q2, t2));
		cams2FromCams1.Add(new Rigid3d(q2, -t2));
		normals.Add(-n1);
		normals.Add(n1);
		normals.Add(-n2);
		normals.Add(n2);
	}

	/// <summary>
	/// The decomposition candidate that triangulates the most ray pairs in front of both
	/// cameras (ties broken by the smaller angular reprojection error), with its plane
	/// normal and those points in camera 1's frame.
	/// Port of colmap::PoseFromHomographyMatrix.
	/// </summary>
	public static void PoseFromHomographyMatrix(
		in Matrix3d h,
		in Matrix3d k1,
		in Matrix3d k2,
		IReadOnlyList<Vector3d> camRays1,
		IReadOnlyList<Vector3d> camRays2,
		out Rigid3d cam2FromCam1,
		out Vector3d normal,
		List<Vector3d> points3D)
	{
		Check.Eq(camRays1.Count, camRays2.Count);

		var cams2FromCams1 = new List<Rigid3d>();
		var normals = new List<Vector3d>();
		DecomposeHomographyMatrix(h, k1, k2, cams2FromCams1, normals);
		Check.Eq(cams2FromCams1.Count, normals.Count);

		cam2FromCam1 = new Rigid3d();
		normal = default;
		points3D.Clear();
		var tentativePoints3D = new List<Vector3d>();
		double bestReprojResidualSum = double.MaxValue;
		for (int i = 0; i < cams2FromCams1.Count; ++i)
		{
			// Note that we can typically eliminate 2 of the 4 solutions using the
			// cheirality check. We can then typically narrow it down to 1 solution by
			// picking the solution with minimal overall reprojection error.
			double reprojResidualSum = CheckCheiralityAndReprojErrorSum(
				cams2FromCams1[i], camRays1, camRays2, tentativePoints3D);
			if (tentativePoints3D.Count > points3D.Count
				|| (tentativePoints3D.Count == points3D.Count && reprojResidualSum < bestReprojResidualSum))
			{
				bestReprojResidualSum = reprojResidualSum;
				cam2FromCam1 = cams2FromCams1[i];
				normal = normals[i];
				points3D.Clear();
				points3D.AddRange(tentativePoints3D);
			}
		}
	}

	/// <summary>
	/// H = K2 (R - t n^T / d) K1^-1 for the plane with (normalized) normal n at distance
	/// d from camera 1. Port of colmap::HomographyMatrixFromPose.
	/// </summary>
	public static Matrix3d HomographyMatrixFromPose(
		in Matrix3d k1, in Matrix3d k2, in Matrix3d r, Vector3d t, Vector3d n, double d)
	{
		Check.Gt(d, 0.0);
		Vector3d nn = n.Normalized();
		Matrix3d tnt = Matrix3d.FromColumns(t * nn.X, t * nn.Y, t * nn.Z);
		return k2 * (r - tnt / d) * k1.Inverse();
	}

	/// <summary>
	/// The squared distance between point2 and the transfer of point1 by H;
	/// double.MaxValue when point1 maps to infinity. Port of colmap::ComputeSquaredHomographyError.
	/// </summary>
	public static double ComputeSquaredHomographyError(Vector2d point1, Vector2d point2, in Matrix3d h)
	{
		Vector3d hp1 = h * point1.Homogeneous();
		if (hp1.Z == 0)
		{
			return double.MaxValue;
		}

		return (point2 - hp1.HNormalized()).SquaredNorm;
	}

	private static double ComputeOppositeOfMinor(in Matrix3d matrix, int row, int col)
	{
		int col1 = col == 0 ? 1 : 0;
		int col2 = col == 2 ? 1 : 2;
		int row1 = row == 0 ? 1 : 0;
		int row2 = row == 2 ? 1 : 2;
		return matrix[row1, col2] * matrix[row2, col1] - matrix[row1, col1] * matrix[row2, col2];
	}

	private static Matrix3d ComputeHomographyRotation(in Matrix3d hNormalized, Vector3d tstar, Vector3d n, double v)
	{
		Vector3d scaled = (2.0 / v) * tstar;
		Matrix3d outer = Matrix3d.FromColumns(scaled * n.X, scaled * n.Y, scaled * n.Z);
		return hNormalized * (Matrix3d.Identity - outer);
	}

	// Eigen's lpNorm<Infinity>(): the largest absolute coefficient.
	private static double MaxAbsCoefficient(in Matrix3d m)
	{
		double max = 0;
		for (int col = 0; col < 3; ++col)
		{
			for (int row = 0; row < 3; ++row)
			{
				max = Math.Max(max, Math.Abs(m[row, col]));
			}
		}

		return max;
	}

	private static double CheckCheiralityAndReprojErrorSum(
		Rigid3d cam2FromCam1,
		IReadOnlyList<Vector3d> camRays1,
		IReadOnlyList<Vector3d> camRays2,
		List<Vector3d> points3D)
	{
		Check.Eq(camRays1.Count, camRays2.Count);
		double reprojResidualSum = 0;
		points3D.Clear();
		for (int i = 0; i < camRays1.Count; ++i)
		{
			if (!Triangulation.TriangulateMidPoint(cam2FromCam1, camRays1[i], camRays2[i], out Vector3d point3DInCam1))
			{
				continue;
			}

			Vector3d point3DInCam2 = cam2FromCam1 * point3DInCam1;
			double error1 = 1 - Math.Clamp(camRays1[i].Dot(point3DInCam1.Normalized()), -1.0, 1.0);
			double error2 = 1 - Math.Clamp(camRays2[i].Dot(point3DInCam2.Normalized()), -1.0, 1.0);
			reprojResidualSum += error1 + error2;
			points3D.Add(point3DInCam1);
		}

		return reprojResidualSum;
	}
}
