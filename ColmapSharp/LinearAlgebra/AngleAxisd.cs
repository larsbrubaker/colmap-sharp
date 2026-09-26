// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AngleAxisd: a rotation as an angle (radians) about a unit axis, the replacement for
// Eigen::AngleAxisd. COLMAP uses it in geometry/pose.cc (RotationMatrixToAngleAxis,
// AngleAxisToRotationMatrix, EulerAnglesToRotationMatrix). Written here to Eigen's
// documented semantics from the standard formulas; Eigen (MPL-2.0) is not ported
// (docs/LICENSE_AUDIT.md). Siblings: Quaterniond and Matrix3d. Tests:
// ColmapSharp.Tests/LinearAlgebra/QuaternionTests.cs.
//
// Conventions, as Eigen documents them:
// - The axis is expected to be unit length; nothing normalizes it.
// - From a quaternion: angle = 2 atan2(|vec|, |w|) in [0, pi], axis = vec / |vec|, with the
//   axis flipped when w < 0 so the angle stays in [0, pi]. Only an exactly zero vector
//   part gives angle 0 about the x axis; tiny ones keep their angle (oracle-pinned).
// - From a rotation matrix: through Quaterniond.FromRotationMatrix, then as above.
// - ToRotationMatrix is Rodrigues' formula, R = c I + s [a]x + (1 - c) a a^T. pycolmap
//   exposes no Eigen AngleAxis -> matrix call, so its bits are not oracle-pinned (Tier B).

using System.Globalization;

namespace ColmapSharp.LinearAlgebra;

/// <summary>
/// Angle-axis rotation. Replacement for Eigen::AngleAxisd.
/// </summary>
public readonly struct AngleAxisd : IEquatable<AngleAxisd>
{
	/// <summary>Rotation angle in radians.</summary>
	public readonly double Angle;

	/// <summary>Rotation axis; expected unit length.</summary>
	public readonly Vector3d Axis;

	/// <summary>Creates the rotation by angle (radians) about axis (expected unit length).</summary>
	public AngleAxisd(double angle, Vector3d axis)
	{
		Angle = angle;
		Axis = axis;
	}

	/// <summary>
	/// The angle-axis form of a quaternion (see the file header). The quaternion need not
	/// be normalized; the angle comes from atan2, which is scale-invariant.
	/// </summary>
	public static AngleAxisd FromQuaternion(Quaterniond q)
	{
		Vector3d vec = q.Vec;

		// Only an exactly zero vector part has no axis. Anything else, however small, is a
		// real (tiny) rotation: the oracle gives angle 2e-17 for (x=1e-17, w=1) and 2e-200
		// for (x=1e-200, w=1), so the norm must not underflow either.
		if (vec.X == 0 && vec.Y == 0 && vec.Z == 0)
		{
			return new AngleAxisd(0, Vector3d.UnitX);
		}

		double n = VectorPartNorm(vec);
		double angle = 2 * Math.Atan2(n, Math.Abs(q.W));
		Vector3d axis = q.W < 0 ? -vec / n : vec / n;
		return new AngleAxisd(angle, axis);
	}

	// The norm of a nonzero vector part. Found with oracle/linear_algebra_rotations.py,
	// whose 138 cases this reproduces bit for bit:
	// - normally the plain Vector3d.Norm, sqrt(x*x + y*y + z*z);
	// - below machine epsilon, the norm of the vector divided by its largest magnitude,
	//   times that magnitude, so tiny squares cannot underflow. The plain norm is wrong
	//   there even without underflow: for (1e-17, -2e-17, 0) it is 1 ulp off the oracle.
	// Scaling by a power of two, or by the reciprocal of the largest magnitude, each
	// miss at least one fixture case.
	private static double VectorPartNorm(Vector3d v)
	{
		double n = v.Norm;
		if (n >= LinearAlgebraConstants.MachineEpsilon)
		{
			return n;
		}

		double largest = Math.Max(Math.Abs(v.X), Math.Max(Math.Abs(v.Y), Math.Abs(v.Z)));
		return largest * (v / largest).Norm;
	}

	/// <summary>The angle-axis form of a rotation matrix, through its quaternion.</summary>
	public static AngleAxisd FromRotationMatrix(Matrix3d rotation)
	{
		return FromQuaternion(Quaterniond.FromRotationMatrix(rotation));
	}

	/// <summary>The unit quaternion (cos(a/2), sin(a/2) axis).</summary>
	public Quaterniond ToQuaternion() => Quaterniond.FromAngleAxis(this);

	/// <summary>
	/// The rotation matrix by Rodrigues' formula in its textbook element form, with
	/// c = cos(angle), s = sin(angle), t = 1 - c and unit axis (x, y, z):
	/// R = [t x x + c, t x y - s z, t x z + s y; t x y + s z, t y y + c, t y z - s x;
	/// t x z - s y, t y z + s x, t z z + c]. Products evaluate left to right, (t x) y.
	/// </summary>
	public Matrix3d ToRotationMatrix()
	{
		double c = Math.Cos(Angle);
		double s = Math.Sin(Angle);
		double t = 1 - c;
		double x = Axis.X, y = Axis.Y, z = Axis.Z;

		return new Matrix3d(
			t * x * x + c, t * x * y - s * z, t * x * z + s * y,
			t * x * y + s * z, t * y * y + c, t * y * z - s * x,
			t * x * z - s * y, t * y * z + s * x, t * z * z + c);
	}

	/// <summary>
	/// Eigen's AngleAxis isApprox: the axes are approximately equal and the angles are
	/// within precision * min(|a|, |b|) of each other.
	/// </summary>
	public bool IsApprox(AngleAxisd other, double precision = LinearAlgebraConstants.DummyPrecision)
	{
		return Axis.IsApprox(other.Axis, precision)
			&& Math.Abs(Angle - other.Angle) <= precision * Math.Min(Math.Abs(Angle), Math.Abs(other.Angle));
	}

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(AngleAxisd other) => Angle.Equals(other.Angle) && Axis.Equals(other.Axis);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is AngleAxisd other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => HashCode.Combine(Angle, Axis);

	/// <inheritdoc/>
	public override string ToString()
	{
		return string.Create(CultureInfo.InvariantCulture, $"(angle={Angle:R}, axis={Axis})");
	}
}
