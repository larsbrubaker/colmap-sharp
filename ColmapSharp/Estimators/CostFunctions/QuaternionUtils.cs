// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// QuaternionUtils: colmap/estimators/cost_functions/quaternion_utils.h. Eigen-order
// (x, y, z, w) wrappers over ceres/rotation.h's [w, x, y, z] conversions
// (Solver/Rotation.cs), and the quaternion multiplication matrices and point-rotation
// Jacobian the analytic cost functions and the TinySolver manifolds (TinyManifold.cs)
// use. Convention: Eigen quaternion storage order (x, y, z, w), Hamilton product. A
// quaternion q = (x, y, z, w) represents R(q) = (w^2 - ||v||^2) I + 2 v v^T + 2 w [v]_x,
// where v = (x, y, z) is the vector part. Tier A: scalar formulas copied term for term.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>Port of colmap/estimators/cost_functions/quaternion_utils.h.</summary>
public static class QuaternionUtils
{
	/// <summary>
	/// colmap::AngleAxisFromEigenQuaternion: the angle-axis vector of an (x, y, z, w)
	/// quaternion, via ceres::QuaternionToAngleAxis.
	/// </summary>
	public static void AngleAxisFromEigenQuaternion<T>(ReadOnlySpan<T> eigenQuaternion, Span<T> angleAxis)
		where T : struct, IScalar<T>
	{
		ReadOnlySpan<T> quaternion = [eigenQuaternion[3], eigenQuaternion[0], eigenQuaternion[1], eigenQuaternion[2]];
		Rotation.QuaternionToAngleAxis(quaternion, angleAxis);
	}

	/// <summary>
	/// colmap::EigenQuaternionFromAngleAxis: the (x, y, z, w) unit quaternion of an
	/// angle-axis vector, via ceres::AngleAxisToQuaternion.
	/// </summary>
	public static void EigenQuaternionFromAngleAxis<T>(ReadOnlySpan<T> angleAxis, Span<T> eigenQuaternion)
		where T : struct, IScalar<T>
	{
		Span<T> quaternion = [default, default, default, default];
		Rotation.AngleAxisToQuaternion(angleAxis, quaternion);
		eigenQuaternion[0] = quaternion[1];
		eigenQuaternion[1] = quaternion[2];
		eigenQuaternion[2] = quaternion[3];
		eigenQuaternion[3] = quaternion[0];
	}

	/// <summary>
	/// Hamilton quaternion left-multiplication matrix (xyzw storage):
	/// QuaternionLeftMultMatrix(q) * p = q * p (as 4-vectors).
	/// </summary>
	public static Matrix4d QuaternionLeftMultMatrix(Quaterniond q)
	{
		double x = q.X, y = q.Y, z = q.Z, w = q.W;
		return new Matrix4d(
			w, -z, y, x,
			z, w, -x, y,
			-y, x, w, z,
			-x, -y, -z, w);
	}

	/// <summary>
	/// Hamilton quaternion right-multiplication matrix (xyzw storage):
	/// QuaternionRightMultMatrix(p) * q = q * p (as 4-vectors).
	/// </summary>
	public static Matrix4d QuaternionRightMultMatrix(Quaterniond q)
	{
		double x = q.X, y = q.Y, z = q.Z, w = q.W;
		return new Matrix4d(
			w, z, -y, x,
			-z, w, x, y,
			y, -x, w, z,
			-x, -y, -z, w);
	}

	/// <summary>
	/// Rotates the point and optionally computes the Jacobian of R(q) * p with respect to the
	/// (x, y, z, w) quaternion q. <paramref name="jacobian"/> receives a 3x4 row-major
	/// matrix; pass an empty span (C++'s nullptr) to skip it.
	/// </summary>
	public static Vector3d QuaternionRotatePointWithJac(ReadOnlySpan<double> q, ReadOnlySpan<double> pt, Span<double> jacobian)
	{
		double qx = q[0], qy = q[1], qz = q[2], qw = q[3];
		double px = pt[0], py = pt[1], pz = pt[2];

		double qxPy = qx * py, qxPz = qx * pz;
		double qyPx = qy * px, qyPz = qy * pz;
		double qzPx = qz * px, qzPy = qz * py;

		// R(q) * p = p + 2*w*(v x p) + 2*(v x (v x p))
		double vxP0 = qyPz - qzPy;
		double vxP1 = qzPx - qxPz;
		double vxP2 = qxPy - qyPx;

		double vxVxP0 = qy * vxP2 - qz * vxP1;
		double vxVxP1 = qz * vxP0 - qx * vxP2;
		double vxVxP2 = qx * vxP1 - qy * vxP0;

		var ptOut = new Vector3d(
			px + 2.0 * (qw * vxP0 + vxVxP0),
			py + 2.0 * (qw * vxP1 + vxVxP1),
			pz + 2.0 * (qw * vxP2 + vxVxP2));

		if (!jacobian.IsEmpty)
		{
			double qxPx = qx * px;
			double qyPy = qy * py;
			double qzPz = qz * pz;
			double qwPx = qw * px;
			double qwPy = qw * py;
			double qwPz = qw * pz;

			jacobian[0] = 2.0 * (qyPy + qzPz);
			jacobian[1] = 2.0 * (-2.0 * qyPx + qxPy + qwPz);
			jacobian[2] = 2.0 * (-2.0 * qzPx - qwPy + qxPz);
			jacobian[3] = 2.0 * (-qzPy + qyPz);

			jacobian[4] = 2.0 * (qyPx - 2.0 * qxPy - qwPz);
			jacobian[5] = 2.0 * (qxPx + qzPz);
			jacobian[6] = 2.0 * (qwPx - 2.0 * qzPy + qyPz);
			jacobian[7] = 2.0 * (qzPx - qxPz);

			jacobian[8] = 2.0 * (qzPx + qwPy - 2.0 * qxPz);
			jacobian[9] = 2.0 * (-qwPx + qzPy - 2.0 * qyPz);
			jacobian[10] = 2.0 * (qxPx + qyPy);
			jacobian[11] = 2.0 * (-qyPx + qxPy);
		}

		return ptOut;
	}
}
