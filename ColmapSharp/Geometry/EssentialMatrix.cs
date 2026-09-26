// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// EssentialMatrix: colmap/geometry/essential_matrix.h/.cc - building an essential matrix
// from a relative pose, decomposing it back into the four pose candidates and picking one
// by cheirality, epipoles, optimal (Lindstrom) image corrections, conversions to and from
// fundamental matrices, and the (tangent) Sampson errors the relative-pose estimators
// score with. Depends on Pose.cs (CheckCheirality, CamRayWithJac) and Rigid3d.cs; used
// by Triangulation.cs (TriangulateOptimalPoint). Tests:
// ColmapSharp.Tests/Geometry/EssentialMatrixTests.cs (essential_matrix_test.cc 1:1) and
// GeometryTwoViewOracleTests (C#-only, against pycolmap).
//
// Tiers: EssentialMatrixFromPose, the Sampson errors and the fundamental conversions are
// scalar products (Tier A up to the 3x3 inverse and product grouping, pinned Tier B by the
// oracle test); DecomposeEssentialMatrix, PoseFromEssentialMatrix and
// EpipoleFromEssentialMatrix go through the SVD and are Tier B.
//
// SVD sign independence (the singular vector signs are arbitrary and differ from Eigen's,
// JacobiSvdKernel.cs):
// - DecomposeEssentialMatrix makes U and V proper rotations first, as COLMAP does. For any
//   such SVD of an essential matrix, {U W V^T, U W^T V^T} is the same pair of rotations
//   (Hartley and Zisserman, "Multiple View Geometry", 2nd ed., Result 9.19) and t = +-u3,
//   so the candidate *set* {(R1, t), (R2, t), (R1, -t), (R2, -t)} does not depend on the
//   signs; only which rotation is called R1 and the sign of t can differ from COLMAP.
//   PoseFromEssentialMatrix therefore returns the same pose as COLMAP whenever one
//   candidate has strictly the most points in front of both cameras; on a tie COLMAP keeps
//   the last tied candidate in its order, and that order is sign-dependent.
// - EpipoleFromEssentialMatrix returns the null vector of E (or E^T) as the SVD gives it,
//   so its overall sign is arbitrary, as in COLMAP (an epipole is a homogeneous point).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Geometry;

/// <summary>
/// Port of the free functions in colmap/geometry/essential_matrix.h.
/// </summary>
public static class EssentialMatrix
{
	/// <summary>
	/// Decomposes E into its two possible rotations and the unit translation direction
	/// (up to sign). Port of colmap::DecomposeEssentialMatrix.
	/// </summary>
	public static void DecomposeEssentialMatrix(in Matrix3d e, out Matrix3d r1, out Matrix3d r2, out Vector3d t)
	{
		Svd3d svd = Svd3d.Compute(e);
		Matrix3d u = svd.MatrixU;
		Matrix3d v = svd.MatrixV.Transpose();

		if (u.Determinant() < 0)
		{
			u *= -1;
		}

		if (v.Determinant() < 0)
		{
			v *= -1;
		}

		var w = new Matrix3d(0, 1, 0, -1, 0, 0, 0, 0, 1);

		r1 = u * w * v;
		r2 = u * w.Transpose() * v;
		t = u.Col(2).Normalized();
	}

	/// <summary>
	/// The relative pose encoded by E whose triangulated points lie in front of both
	/// cameras for the most ray pairs, and the indices of those pairs.
	/// Port of colmap::PoseFromEssentialMatrix.
	/// </summary>
	public static void PoseFromEssentialMatrix(
		in Matrix3d e,
		IReadOnlyList<Vector3d> camRays1,
		IReadOnlyList<Vector3d> camRays2,
		out Rigid3d cam2FromCam1,
		List<int> validIndices)
	{
		Check.Eq(camRays1.Count, camRays2.Count);

		DecomposeEssentialMatrix(e, out Matrix3d r1, out Matrix3d r2, out Vector3d t);

		Quaterniond quat1 = Quaterniond.FromRotationMatrix(r1);
		Quaterniond quat2 = Quaterniond.FromRotationMatrix(r2);

		// Generate all possible pose combinations.
		Rigid3d[] cams2FromCams1 =
		[
			new Rigid3d(quat1, t),
			new Rigid3d(quat2, t),
			new Rigid3d(quat1, -t),
			new Rigid3d(quat2, -t),
		];

		cam2FromCam1 = new Rigid3d();
		validIndices.Clear();
		var tentativeValidIndices = new List<int>();
		foreach (Rigid3d candidate in cams2FromCams1)
		{
			Pose.CheckCheirality(candidate, camRays1, camRays2, tentativeValidIndices);
			if (tentativeValidIndices.Count >= validIndices.Count)
			{
				cam2FromCam1 = candidate;
				validIndices.Clear();
				validIndices.AddRange(tentativeValidIndices);
			}
		}
	}

	/// <summary>
	/// E = [t / |t|]x R. Port of colmap::EssentialMatrixFromPose.
	/// </summary>
	public static Matrix3d EssentialMatrixFromPose(Rigid3d cam2FromCam1)
	{
		return Rigid3d.CrossProductMatrix(cam2FromCam1.Translation.Normalized())
			* cam2FromCam1.Rotation.ToRotationMatrix();
	}

	/// <summary>
	/// The observations closest to (point1, point2) that satisfy the epipolar constraint
	/// exactly (Lindstrom, "Triangulation made easy", CVPR 2010, one iteration).
	/// Port of colmap::FindOptimalImageObservations.
	/// </summary>
	public static void FindOptimalImageObservations(
		in Matrix3d e,
		Vector2d point1,
		Vector2d point2,
		out Vector2d optimalPoint1,
		out Vector2d optimalPoint2)
	{
		Vector3d point1Homogeneous = point1.Homogeneous();
		Vector3d point2Homogeneous = point2.Homogeneous();

		// Epipolar lines (S = [I2 | 0] keeps the first two coordinates).
		Vector2d n1 = (e * point2Homogeneous).Head2();
		Vector2d n2 = (e.Transpose() * point1Homogeneous).Head2();

		var eTilde = new Matrix2d(e[0, 0], e[0, 1], e[1, 0], e[1, 1]);

		double a = n1.Dot(eTilde * n2);
		double b = (n1.SquaredNorm + n2.SquaredNorm) / 2.0;
		double c = point1Homogeneous.Dot(e * point2Homogeneous);
		double d = Math.Sqrt(b * b - a * c);
		double lambda = c / (b + d);

		Vector2d delta1 = lambda * n1;
		Vector2d delta2 = lambda * n2;

		n1 -= eTilde * delta2;
		n2 -= eTilde.Transpose() * delta1;

		lambda *= (2.0 * d) / (n1.SquaredNorm + n2.SquaredNorm);

		Vector2d step1 = lambda * n1;
		Vector2d step2 = lambda * n2;
		optimalPoint1 = new Vector3d(point1Homogeneous.X - step1.X, point1Homogeneous.Y - step1.Y, point1Homogeneous.Z).HNormalized();
		optimalPoint2 = new Vector3d(point2Homogeneous.X - step2.X, point2Homogeneous.Y - step2.Y, point2Homogeneous.Z).HNormalized();
	}

	/// <summary>
	/// The epipole of the left (null vector of E) or right (null vector of E^T) image, as a
	/// unit homogeneous vector of arbitrary sign. Port of colmap::EpipoleFromEssentialMatrix.
	/// </summary>
	public static Vector3d EpipoleFromEssentialMatrix(in Matrix3d e, bool leftImage)
	{
		Svd3d svd = Svd3d.Compute(leftImage ? e : e.Transpose());
		return svd.MatrixV.Col(2);
	}

	/// <summary>The essential matrix of the inverse pose, E^T. Port of colmap::InvertEssentialMatrix.</summary>
	public static Matrix3d InvertEssentialMatrix(in Matrix3d e) => e.Transpose();

	/// <summary>F = K2^-T E K1^-1. Port of colmap::FundamentalFromEssentialMatrix.</summary>
	public static Matrix3d FundamentalFromEssentialMatrix(in Matrix3d k2, in Matrix3d e, in Matrix3d k1)
	{
		return k2.Transpose().Inverse() * e * k1.Inverse();
	}

	/// <summary>E = K2^T F K1. Port of colmap::EssentialFromFundamentalMatrix.</summary>
	public static Matrix3d EssentialFromFundamentalMatrix(in Matrix3d k2, in Matrix3d f, in Matrix3d k1)
	{
		return k2.Transpose() * f * k1;
	}

	/// <summary>
	/// The squared Sampson error of a homogeneous correspondence under E (or F);
	/// double.MaxValue when the gradient vanishes. Port of colmap::ComputeSquaredSampsonError.
	/// </summary>
	public static double ComputeSquaredSampsonError(Vector3d point1, Vector3d point2, in Matrix3d e)
	{
		Vector3d epipolarLine1 = e * point1;
		double num = point2.Dot(epipolarLine1);
		var denom = new Vector4d(point2.Dot(e.Col(0)), point2.Dot(e.Col(1)), epipolarLine1.X, epipolarLine1.Y);
		double denomSqNorm = denom.SquaredNorm;
		if (denomSqNorm == 0)
		{
			return double.MaxValue;
		}

		return num * num / denomSqNorm;
	}

	/// <summary>
	/// The squared Sampson errors of image-point correspondences.
	/// Port of colmap::ComputeSquaredSampsonError (Vector2d overload).
	/// </summary>
	public static void ComputeSquaredSampsonError(
		IReadOnlyList<Vector2d> points1, IReadOnlyList<Vector2d> points2, in Matrix3d e, List<double> residuals)
	{
		int numPoints1 = points1.Count;
		Check.Eq(numPoints1, points2.Count);
		residuals.Clear();
		for (int i = 0; i < numPoints1; ++i)
		{
			residuals.Add(ComputeSquaredSampsonError(points1[i].Homogeneous(), points2[i].Homogeneous(), e));
		}
	}

	/// <summary>
	/// The squared Sampson errors of homogeneous correspondences.
	/// Port of colmap::ComputeSquaredSampsonError (Vector3d overload).
	/// </summary>
	public static void ComputeSquaredSampsonError(
		IReadOnlyList<Vector3d> points1, IReadOnlyList<Vector3d> points2, in Matrix3d e, List<double> residuals)
	{
		int numPoints1 = points1.Count;
		Check.Eq(numPoints1, points2.Count);
		residuals.Clear();
		for (int i = 0; i < numPoints1; ++i)
		{
			residuals.Add(ComputeSquaredSampsonError(points1[i], points2[i], e));
		}
	}

	/// <summary>
	/// |J^T g|^2 for a 3x2 Jacobian J, written out as COLMAP does.
	/// Port of colmap::SquaredPixelGradientNorm.
	/// </summary>
	public static double SquaredPixelGradientNorm(in Matrix3x2d j, Vector3d g)
	{
		double gx = j[0, 0] * g.X + j[1, 0] * g.Y + j[2, 0] * g.Z;
		double gy = j[0, 1] * g.X + j[1, 1] * g.Y + j[2, 1] * g.Z;
		return gx * gx + gy * gy;
	}

	/// <summary>
	/// The Sampson error of a ray correspondence with the constraint gradients chained into
	/// pixel space through the ray Jacobians; double.MaxValue when they vanish.
	/// Port of colmap::ComputeSquaredTangentSampsonError.
	/// </summary>
	public static double ComputeSquaredTangentSampsonError(
		Vector3d camRay1, in Matrix3x2d j1, Vector3d camRay2, in Matrix3x2d j2, in Matrix3d e)
	{
		Vector3d eRay1 = e * camRay1;
		Vector3d etRay2 = e.Transpose() * camRay2;
		double num = camRay2.Dot(eRay1);

		// Chain the constraint gradients from ray space into pixel space. The
		// gradient w.r.t. ray1 is E^T ray2 and w.r.t. ray2 is E ray1.
		double denomSqNorm = SquaredPixelGradientNorm(j1, etRay2) + SquaredPixelGradientNorm(j2, eRay1);
		if (denomSqNorm == 0)
		{
			return double.MaxValue;
		}

		return num * num / denomSqNorm;
	}

	/// <summary>Port of colmap::ComputeSquaredTangentSampsonError (CamRayWithJac overload).</summary>
	public static double ComputeSquaredTangentSampsonError(CamRayWithJac camRay1WithJac, CamRayWithJac camRay2WithJac, in Matrix3d e)
	{
		return ComputeSquaredTangentSampsonError(
			camRay1WithJac.Ray, camRay1WithJac.Jacobian, camRay2WithJac.Ray, camRay2WithJac.Jacobian, e);
	}

	/// <summary>Port of colmap::ComputeSquaredTangentSampsonError (vector overload).</summary>
	public static void ComputeSquaredTangentSampsonError(
		IReadOnlyList<CamRayWithJac> camRays1WithJac,
		IReadOnlyList<CamRayWithJac> camRays2WithJac,
		in Matrix3d e,
		List<double> residuals)
	{
		int numRays = camRays1WithJac.Count;
		Check.Eq(numRays, camRays2WithJac.Count);
		residuals.Clear();
		for (int i = 0; i < numRays; ++i)
		{
			residuals.Add(ComputeSquaredTangentSampsonError(camRays1WithJac[i], camRays2WithJac[i], e));
		}
	}

	/// <summary>
	/// The tangent Sampson errors, with double.MaxValue for correspondences that the pose
	/// recovered from E puts behind either camera.
	/// Port of colmap::ComputeSquaredTangentSampsonErrorWithCheirality.
	/// </summary>
	public static void ComputeSquaredTangentSampsonErrorWithCheirality(
		IReadOnlyList<CamRayWithJac> camRays1WithJac,
		IReadOnlyList<CamRayWithJac> camRays2WithJac,
		in Matrix3d e,
		List<double> residuals)
	{
		int numRays = camRays1WithJac.Count;
		Check.Eq(numRays, camRays2WithJac.Count);

		// Recover the relative pose from E (resolving the four-fold decomposition
		// ambiguity by cheirality voting) and flag which correspondences triangulate
		// in front of both cameras. Only the bearings are materialized, since that is
		// all PoseFromEssentialMatrix needs; the Jacobians are read in place below.
		var rays1 = new Vector3d[numRays];
		var rays2 = new Vector3d[numRays];
		for (int i = 0; i < numRays; ++i)
		{
			rays1[i] = camRays1WithJac[i].Ray;
			rays2[i] = camRays2WithJac[i].Ray;
		}

		var validIndices = new List<int>();
		PoseFromEssentialMatrix(e, rays1, rays2, out _, validIndices);
		var isCheiral = new bool[numRays];
		foreach (int idx in validIndices)
		{
			isCheiral[idx] = true;
		}

		// Correspondences behind either camera are not valid inliers for the relative
		// pose regardless of their residual, so they get an infinite residual.
		residuals.Clear();
		for (int i = 0; i < numRays; ++i)
		{
			residuals.Add(isCheiral[i]
				? ComputeSquaredTangentSampsonError(camRays1WithJac[i], camRays2WithJac[i], e)
				: double.MaxValue);
		}
	}
}
