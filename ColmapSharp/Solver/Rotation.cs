// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 include/ceres/rotation.h (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// The ceres/rotation.h functions COLMAP calls: AngleAxisToQuaternion and
// QuaternionToAngleAxis, which colmap/estimators/cost_functions/quaternion_utils.h wraps as
// EigenQuaternionFromAngleAxis / AngleAxisFromEigenQuaternion for the motion-averaging and
// pose-prior cost functions. (COLMAP's reprojection cost functors rotate with Eigen's
// quaternion product, not with ceres::QuaternionRotatePoint, so no other rotation.h entry
// point is reached.) Generic over IScalar (Scalar.cs) so they run on doubles and on Jets.
// Quaternions here are in Ceres' order, [w, x, y, z]; COLMAP's Eigen-order wrappers swap.

namespace ColmapSharp.Solver;

/// <summary>Port of the ceres/rotation.h conversions COLMAP uses.</summary>
public static class Rotation
{
	/// <summary>
	/// ceres::AngleAxisToQuaternion: the unit quaternion [w, x, y, z] for the rotation by
	/// |angleAxis| radians about angleAxis. At the origin it uses the first-order Taylor
	/// expansion (w = 1, xyz = angleAxis / 2), so Jets get correct derivatives there instead
	/// of the NaN that sqrt'(0) would give.
	/// </summary>
	public static void AngleAxisToQuaternion<T>(ReadOnlySpan<T> angleAxis, Span<T> quaternion)
		where T : struct, IScalar<T>
	{
		T a0 = angleAxis[0];
		T a1 = angleAxis[1];
		T a2 = angleAxis[2];
		T theta = T.Hypot(a0, a1, a2);

		if (T.ScalarPart(theta) != 0.0)
		{
			T halfTheta = theta * T.FromDouble(0.5);
			T k = T.Sin(halfTheta) / theta;
			quaternion[0] = T.Cos(halfTheta);
			quaternion[1] = a0 * k;
			quaternion[2] = a1 * k;
			quaternion[3] = a2 * k;
		}
		else
		{
			T k = T.FromDouble(0.5);
			quaternion[0] = T.FromDouble(1.0);
			quaternion[1] = a0 * k;
			quaternion[2] = a1 * k;
			quaternion[3] = a2 * k;
		}
	}

	/// <summary>
	/// ceres::QuaternionToAngleAxis: the angle-axis vector of the rotation by a unit
	/// quaternion [w, x, y, z], normalized to an angle of at most pi. For zero rotation it
	/// uses the first-order Taylor expansion (angleAxis = 2 xyz).
	/// </summary>
	public static void QuaternionToAngleAxis<T>(ReadOnlySpan<T> quaternion, Span<T> angleAxis)
		where T : struct, IScalar<T>
	{
		T q1 = quaternion[1];
		T q2 = quaternion[2];
		T q3 = quaternion[3];
		T sinTheta = T.Hypot(q1, q2, q3);

		if (T.ScalarPart(sinTheta) != 0.0)
		{
			T cosTheta = quaternion[0];

			// If cos_theta is negative, theta is greater than pi/2, which means that angle for
			// the angle_axis vector which is 2 * theta would be greater than pi. While this
			// would still be the correct rotation, it would not be a normalized angle-axis
			// vector. In that case 2 * theta ~ 2 * theta - 2 * pi, which is to say
			// theta - pi = atan(sin(theta - pi), cos(theta - pi)) = atan(-sin(theta), -cos(theta)).
			T twoTheta = T.ScalarPart(cosTheta) < 0.0
				? T.FromDouble(2.0) * T.Atan2(-sinTheta, -cosTheta)
				: T.FromDouble(2.0) * T.Atan2(sinTheta, cosTheta);
			T k = twoTheta / sinTheta;
			angleAxis[0] = q1 * k;
			angleAxis[1] = q2 * k;
			angleAxis[2] = q3 * k;
		}
		else
		{
			T k = T.FromDouble(2.0);
			angleAxis[0] = q1 * k;
			angleAxis[1] = q2 * k;
			angleAxis[2] = q3 * k;
		}
	}
}
