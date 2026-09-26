// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Triangulation: colmap/geometry/triangulation.h/.cc - two-view DLT triangulation from
// image points or bearing vectors, the mid-point method, multi-view DLT through the 4x4
// normal matrix, optimal two-view triangulation (Lindstrom correction first), and the
// triangulation-angle helpers. Depends on EssentialMatrix.cs (TriangulateOptimalPoint),
// Rigid3d.cs, and Svd3d/Svd4d/JacobiSVD/SelfAdjointEigenSolver in LinearAlgebra/. Used by
// HomographyMatrix.cs (the cheirality test of PoseFromHomographyMatrix). Tests:
// ColmapSharp.Tests/Geometry/TriangulationTests.cs (triangulation_test.cc 1:1) and
// GeometryTwoViewOracleTests (C#-only, against pycolmap).
//
// Tiers: CalculateTriangulationAngle(s) and CalculateAngleBetweenVectors are scalar code;
// every triangulation goes through an SVD or a symmetric eigensolver and is Tier B.
//
// Sign independence: each method reads a null vector (the last column of V, or the
// eigenvector of the smallest eigenvalue) whose sign is arbitrary, and then only uses it
// through hnormalized(), i.e. divided by its own last coordinate, which cancels the sign.
// The "== 0" rejections test that same coordinate, so they are sign-independent too.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Geometry;

/// <summary>
/// Port of the free functions in colmap/geometry/triangulation.h.
/// </summary>
public static class Triangulation
{
	/// <summary>
	/// Triangulates a point from two image observations in normalized camera
	/// coordinates by DLT. Returns false for a degenerate (e.g. parallel-ray) geometry.
	/// Port of colmap::TriangulatePoint (Vector2d overload).
	/// </summary>
	public static bool TriangulatePoint(
		in Matrix3x4d cam1FromWorld,
		in Matrix3x4d cam2FromWorld,
		Vector2d camPoint1,
		Vector2d camPoint2,
		out Vector3d xyz)
	{
		var a = Matrix4d.FromRows(
			camPoint1.X * cam1FromWorld.Row(2) - cam1FromWorld.Row(0),
			camPoint1.Y * cam1FromWorld.Row(2) - cam1FromWorld.Row(1),
			camPoint2.X * cam2FromWorld.Row(2) - cam2FromWorld.Row(0),
			camPoint2.Y * cam2FromWorld.Row(2) - cam2FromWorld.Row(1));

		Svd4d svd = Svd4d.Compute(a);
		if (svd.Info != ComputationInfo.Success || svd.MatrixV[3, 3] == 0)
		{
			xyz = default;
			return false;
		}

		xyz = svd.MatrixV.Col(3).HNormalized();
		return true;
	}

	/// <summary>
	/// Triangulates a point from two bearing vectors (unit rays, which may point into the
	/// back hemisphere) by DLT on the projectors I - b b^T.
	/// Port of colmap::TriangulatePoint (Vector3d overload).
	/// </summary>
	public static bool TriangulatePoint(
		in Matrix3x4d cam1FromWorld,
		in Matrix3x4d cam2FromWorld,
		Vector3d camRay1,
		Vector3d camRay2,
		out Vector3d xyz)
	{
		var a = new MatrixXd(6, 4);
		FillRayProjectorRows(a, 0, cam1FromWorld, camRay1);
		FillRayProjectorRows(a, 3, cam2FromWorld, camRay2);

		var svd = new JacobiSVD(a, SvdOptions.ComputeFullV);
		if (svd.Info != ComputationInfo.Success)
		{
			xyz = default;
			return false;
		}

		MatrixXd v = svd.MatrixV();
		if (v[3, 3] == 0)
		{
			xyz = default;
			return false;
		}

		xyz = new Vector3d(v[0, 3] / v[3, 3], v[1, 3] / v[3, 3], v[2, 3] / v[3, 3]);
		return true;
	}

	/// <summary>
	/// The mid-point of the shortest segment between the two rays, in camera 1's frame.
	/// Returns false for parallel rays or a point behind either camera.
	/// Port of colmap::TriangulateMidPoint.
	/// </summary>
	public static bool TriangulateMidPoint(
		Rigid3d cam2FromCam1, Vector3d camRay1, Vector3d camRay2, out Vector3d point3DInCam1)
	{
		Quaterniond cam1FromCam2Rotation = cam2FromCam1.Rotation.Inverse();
		Vector3d camRay2InCam1 = cam1FromCam2Rotation * camRay2;
		Vector3d cam2InCam1 = cam1FromCam2Rotation * -cam2FromCam1.Translation;

		var a = new Matrix3d(
			camRay1.X, -camRay2InCam1.X, -cam2InCam1.X,
			camRay1.Y, -camRay2InCam1.Y, -cam2InCam1.Y,
			camRay1.Z, -camRay2InCam1.Z, -cam2InCam1.Z);

		Svd3d svd = Svd3d.Compute(a);
		if (svd.Info != ComputationInfo.Success || svd.MatrixV[2, 2] == 0)
		{
			point3DInCam1 = default;
			return false;
		}

		Vector2d lambda = svd.MatrixV.Col(2).HNormalized();

		// Check if point is behind cameras.
		if (lambda.X <= LinearAlgebraConstants.MachineEpsilon || lambda.Y <= LinearAlgebraConstants.MachineEpsilon)
		{
			point3DInCam1 = default;
			return false;
		}

		point3DInCam1 = 0.5 * (lambda.X * camRay1 + cam2InCam1 + lambda.Y * camRay2InCam1);
		return true;
	}

	/// <summary>
	/// Triangulates a point from any number of image observations in normalized camera
	/// coordinates (each lifted to the unit ray of (x, y, 1)).
	/// Port of colmap::TriangulateMultiViewPoint (Vector2d overload).
	/// </summary>
	public static bool TriangulateMultiViewPoint(
		ReadOnlySpan<Matrix3x4d> camsFromWorld, ReadOnlySpan<Vector2d> camPoints, out Vector3d xyz)
	{
		Check.Eq(camsFromWorld.Length, camPoints.Length);
		Matrix4d a = Matrix4d.Zero;
		for (int i = 0; i < camPoints.Length; ++i)
		{
			a += TriangulationDltTerm(camsFromWorld[i], camPoints[i].Homogeneous().Normalized());
		}

		return SolveTriangulationDlt(a, out xyz);
	}

	/// <summary>
	/// Triangulates a point from any number of bearing vectors.
	/// Port of colmap::TriangulateMultiViewPoint (Vector3d overload).
	/// </summary>
	public static bool TriangulateMultiViewPoint(
		ReadOnlySpan<Matrix3x4d> camsFromWorld, ReadOnlySpan<Vector3d> camRays, out Vector3d xyz)
	{
		Check.Eq(camsFromWorld.Length, camRays.Length);
		Matrix4d a = Matrix4d.Zero;
		for (int i = 0; i < camRays.Length; ++i)
		{
			a += TriangulationDltTerm(camsFromWorld[i], camRays[i]);
		}

		return SolveTriangulationDlt(a, out xyz);
	}

	/// <summary>
	/// Two-view triangulation after moving both observations onto the epipolar
	/// constraint (FindOptimalImageObservations). Port of colmap::TriangulateOptimalPoint.
	/// </summary>
	public static bool TriangulateOptimalPoint(
		in Matrix3x4d cam1FromWorldMat,
		in Matrix3x4d cam2FromWorldMat,
		Vector2d camPoint1,
		Vector2d camPoint2,
		out Vector3d xyz)
	{
		var cam1FromWorld = new Rigid3d(
			Quaterniond.FromRotationMatrix(cam1FromWorldMat.LeftCols3()), cam1FromWorldMat.Col(3));
		var cam2FromWorld = new Rigid3d(
			Quaterniond.FromRotationMatrix(cam2FromWorldMat.LeftCols3()), cam2FromWorldMat.Col(3));
		Rigid3d cam2FromCam1 = cam2FromWorld * cam1FromWorld.Inverse();
		Matrix3d e = EssentialMatrix.EssentialMatrixFromPose(cam2FromCam1);

		EssentialMatrix.FindOptimalImageObservations(
			e, camPoint1, camPoint2, out Vector2d optimalPoint1, out Vector2d optimalPoint2);

		return TriangulatePoint(cam1FromWorldMat, cam2FromWorldMat, optimalPoint1, optimalPoint2, out xyz);
	}

	/// <summary>
	/// The smaller of the angle between the rays from the two projection centers to the
	/// point and its supplement, in radians. Port of colmap::CalculateTriangulationAngle.
	/// </summary>
	public static double CalculateTriangulationAngle(Vector3d projCenter1, Vector3d projCenter2, Vector3d point3D)
	{
		double angle = CalculateAngleBetweenVectors(point3D - projCenter1, point3D - projCenter2);

		// Triangulation is unstable for acute angles (far away points) and
		// obtuse angles (close points), so always compute the minimum angle
		// between the two intersecting rays.
		return Math.Min(angle, Math.PI - angle);
	}

	/// <summary>
	/// CalculateTriangulationAngle for each point. Port of colmap::CalculateTriangulationAngles.
	/// </summary>
	public static double[] CalculateTriangulationAngles(
		Vector3d projCenter1, Vector3d projCenter2, IReadOnlyList<Vector3d> points3D)
	{
		var angles = new double[points3D.Count];
		for (int i = 0; i < points3D.Count; ++i)
		{
			angles[i] = CalculateTriangulationAngle(projCenter1, projCenter2, points3D[i]);
		}

		return angles;
	}

	/// <summary>
	/// The angle between two vectors in [0, pi]; 0 if either is zero.
	/// Port of colmap::CalculateAngleBetweenVectors.
	/// </summary>
	public static double CalculateAngleBetweenVectors(Vector3d v1, Vector3d v2)
	{
		double squaredNorm1 = v1.SquaredNorm;
		double squaredNorm2 = v2.SquaredNorm;
		if (squaredNorm1 == 0.0 || squaredNorm2 == 0.0)
		{
			return 0.0;
		}

		return Math.Acos(Math.Clamp(v1.Dot(v2) / Math.Sqrt(squaredNorm1 * squaredNorm2), -1.0, 1.0));
	}

	// Rows [row0, row0 + 3) of a = P - b (b^T P): the bearing b's projector applied to P.
	private static void FillRayProjectorRows(MatrixXd a, int row0, in Matrix3x4d p, Vector3d b)
	{
		for (int col = 0; col < 4; ++col)
		{
			Vector3d pc = p.Col(col);
			double btp = b.Dot(pc);
			a[row0, col] = pc.X - b.X * btp;
			a[row0 + 1, col] = pc.Y - b.Y * btp;
			a[row0 + 2, col] = pc.Z - b.Z * btp;
		}
	}

	// Contribution of a single bearing observation to the projector-based DLT
	// normal-equation matrix. With unit bearing b and projection matrix
	// P = cam_from_world, the residual operator is term = P - b b^T P, and the
	// system accumulates term^T term.
	private static Matrix4d TriangulationDltTerm(in Matrix3x4d camFromWorld, Vector3d camRay)
	{
		var bbt = Matrix3d.FromColumns(camRay * camRay.X, camRay * camRay.Y, camRay * camRay.Z);
		Matrix3x4d term = camFromWorld - bbt * camFromWorld;
		Span<double> t = stackalloc double[16];
		for (int i = 0; i < 4; ++i)
		{
			Vector3d ci = term.Col(i);
			for (int j = 0; j < 4; ++j)
			{
				t[j * 4 + i] = ci.Dot(term.Col(j));
			}
		}

		return Matrix4d.FromColumnMajor(t);
	}

	// Solve the DLT system for the homogeneous point (smallest eigenvector of A).
	private static bool SolveTriangulationDlt(in Matrix4d a, out Vector3d xyz)
	{
		var eigenSolver = new SelfAdjointEigenSolver(MatrixXd.From(a));
		if (eigenSolver.Info != ComputationInfo.Success)
		{
			xyz = default;
			return false;
		}

		MatrixXd vectors = eigenSolver.Eigenvectors();
		if (vectors[3, 0] == 0)
		{
			xyz = default;
			return false;
		}

		xyz = new Vector3d(vectors[0, 0] / vectors[3, 0], vectors[1, 0] / vectors[3, 0], vectors[2, 0] / vectors[3, 0]);
		return true;
	}
}
