// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Pose: colmap/geometry/pose.h/.cc - rotation conversions (angle-axis, Euler, closest
// rotation), projection matrix decomposition, averaging of unit vectors and quaternions,
// pose interpolation, the two-view cheirality test, the SO(3) Jacobians, and the
// gravity-aligned frame helpers. Built on Rigid3d/Sim3d (this folder), the fixed-size
// types and SVD/QR in LinearAlgebra/, and MatrixUtils.DecomposeMatrixRQ. Used by
// EssentialMatrix.cs, HomographyMatrix.cs and Triangulation.cs. pose.h's CamRayWithJac
// struct is in CamRayWithJac.cs. Tests:
// ColmapSharp.Tests/Geometry/PoseTests.cs (pose_test.cc 1:1) and
// GeometryTwoViewOracleTests (C#-only, against pycolmap).
//
// Tiers: the conversions that only run scalar code (Euler angles, the Jacobians, the
// y-axis helpers, CheckCheirality) are Tier A in principle; everything through an SVD or
// QR (ComputeClosestRotationMatrix, DecomposeProjectionMatrix, AverageUnitVectors,
// AverageQuaternions, GravityAlignedRotation) and InterpolateCameraPoses (slerp) is Tier B.
//
// Sign independence: the SVD's singular vector signs are arbitrary (JacobiSvdKernel.cs)
// and differ from Eigen's. ComputeClosestRotationMatrix uses U V^T, in which each pair of
// flipped columns cancels; AverageUnitVectors fixes the sign of the principal vector by a
// weighted majority vote of the inputs, as COLMAP does, so only an exact tie can leave it
// to the SVD.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Geometry;

/// <summary>
/// Port of the free functions in colmap/geometry/pose.h.
/// </summary>
public static class Pose
{
	// Below this angle QuaternionFromAngleAxis and LeftJacobianFromAngleAxis switch to
	// their first-order expansions.
	private const double SmallAngleThreshold = 1e-10;

	/// <summary>
	/// The weighted average direction of the columns of <paramref name="vectors"/> (each
	/// normalized first): the principal left singular vector of
	/// A = N diag(w) N^T / sum(w), signed to agree with the weighted majority of the
	/// inputs. Empty <paramref name="weights"/> means uniform weights.
	/// Port of colmap::AverageUnitVectors.
	/// </summary>
	public static VectorXd AverageUnitVectors(MatrixXd vectors, VectorXd? weights = null)
	{
		int count = vectors.Cols;
		int weightCount = weights?.Length ?? 0;
		Check.Gt(count, 0, "Cannot average empty set of vectors");
		Check.That(weightCount == 0 || weightCount == count, "Weights size must match vectors size");

		if (count == 1)
		{
			return vectors.Col(0).Normalized();
		}

		// Determine weights: use provided weights or uniform weights.
		VectorXd w = weightCount > 0 ? weights! : VectorXd.Ones(count);
		bool allPositive = true;
		for (int i = 0; i < count; ++i)
		{
			allPositive &= w[i] > 0;
		}

		Check.That(allPositive, "Weights must be positive");

		double weightSum = 0;
		for (int i = 0; i < count; ++i)
		{
			weightSum += w[i];
		}

		// Normalize all columns and build weighted outer product sum matrix:
		// A = N * diag(w) * N^T / sum(w)
		int dim = vectors.Rows;
		var normalized = new MatrixXd(dim, count);
		var weighted = new MatrixXd(dim, count);
		for (int j = 0; j < count; ++j)
		{
			VectorXd column = vectors.Col(j);
			double norm = column.Norm();
			for (int i = 0; i < dim; ++i)
			{
				normalized[i, j] = column[i] / norm;
				weighted[i, j] = normalized[i, j] * w[j];
			}
		}

		MatrixXd a = weighted * normalized.Transpose() / weightSum;

		// The first singular vector corresponds to the principal direction.
		var svd = new JacobiSVD(a, SvdOptions.ComputeFullU);
		VectorXd average = svd.MatrixU().Col(0);

		// Ensure consistent sign by aligning with majority of input vectors.
		VectorXd dots = vectors.TransposeTimes(average);
		double negativeWeight = 0;
		for (int i = 0; i < count; ++i)
		{
			negativeWeight += (dots[i] < 0 ? 1.0 : 0.0) * w[i];
		}

		if (negativeWeight > weightSum - negativeWeight)
		{
			average = -average;
		}

		return average;
	}

	/// <summary>
	/// AverageUnitVectors over 3D directions. Port of colmap::AverageDirections.
	/// </summary>
	public static Vector3d AverageDirections(IReadOnlyList<Vector3d> directions, IReadOnlyList<double>? weights = null)
	{
		var mat = new MatrixXd(3, directions.Count);
		for (int i = 0; i < directions.Count; ++i)
		{
			mat.SetCol(i, VectorXd.From(directions[i]));
		}

		VectorXd w = new(weights is null ? [] : weights.ToArray());
		return AverageUnitVectors(mat, w).ToVector3d();
	}

	/// <summary>
	/// The rotation closest to <paramref name="matrix"/> in the Frobenius norm, U V^T of
	/// its SVD, negated if that is a reflection. Port of colmap::ComputeClosestRotationMatrix.
	/// </summary>
	public static Matrix3d ComputeClosestRotationMatrix(in Matrix3d matrix)
	{
		Svd3d svd = Svd3d.Compute(matrix);
		Matrix3d r = svd.MatrixU * svd.MatrixV.Transpose();
		if (r.Determinant() < 0.0)
		{
			r *= -1.0;
		}

		return r;
	}

	/// <summary>
	/// Decomposes P = K [R | T] into the upper-triangular calibration K (positive
	/// diagonal), the rotation R and the translation T. Returns false if K is singular.
	/// Port of colmap::DecomposeProjectionMatrix.
	/// </summary>
	public static bool DecomposeProjectionMatrix(
		in Matrix3x4d p, out Matrix3d k, out Matrix3d r, out Vector3d t)
	{
		MatrixUtils.DecomposeMatrixRQ(p.LeftCols3(), out Matrix3d rr, out Matrix3d qq);

		r = ComputeClosestRotationMatrix(qq);

		double detK = rr.Determinant();
		if (detK == 0)
		{
			k = default;
			t = default;
			return false;
		}

		k = detK > 0 ? rr : -rr;

		Span<double> kValues = stackalloc double[9];
		Span<double> rValues = stackalloc double[9];
		k.CopyToColumnMajor(kValues);
		r.CopyToColumnMajor(rValues);
		for (int i = 0; i < 3; ++i)
		{
			if (kValues[i * 3 + i] < 0.0)
			{
				// Negate column i of K and row i of R (column-major storage).
				for (int row = 0; row < 3; ++row)
				{
					kValues[i * 3 + row] = -kValues[i * 3 + row];
				}

				for (int col = 0; col < 3; ++col)
				{
					rValues[col * 3 + i] = -rValues[col * 3 + i];
				}
			}
		}

		k = Matrix3d.FromColumnMajor(kValues);
		r = Matrix3d.FromColumnMajor(rValues);

		// Back substitution with the upper triangle of K (Eigen's triangularView<Upper>).
		Vector3d b = p.Col(3);
		double t2 = b.Z / k[2, 2];
		double t1 = (b.Y - k[1, 2] * t2) / k[1, 1];
		double t0 = (b.X - k[0, 1] * t1 - k[0, 2] * t2) / k[0, 0];
		t = new Vector3d(t0, t1, t2);
		if (detK < 0)
		{
			t = -t;
		}

		return true;
	}

	/// <summary>
	/// The angle-axis vector (angle times unit axis) of a rotation matrix.
	/// Port of colmap::RotationMatrixToAngleAxis.
	/// </summary>
	public static Vector3d RotationMatrixToAngleAxis(in Matrix3d r)
	{
		AngleAxisd aa = AngleAxisd.FromRotationMatrix(r);
		return aa.Angle * aa.Axis;
	}

	/// <summary>
	/// The rotation matrix of an angle-axis vector; below an angle of 1e-12 the
	/// first-order I + [w]x. Port of colmap::AngleAxisToRotationMatrix.
	/// </summary>
	public static Matrix3d AngleAxisToRotationMatrix(Vector3d w)
	{
		double angle = w.Norm;
		if (angle > 1e-12)
		{
			return new AngleAxisd(angle, w / angle).ToRotationMatrix();
		}

		// Small angle approximation: I + [w]_x.
		return new Matrix3d(
			1, -w.Z, w.Y,
			w.Z, 1, -w.X,
			-w.Y, w.X, 1);
	}

	/// <summary>
	/// The Euler angles of R = Rz * Ry * Rx; NaN angles become 0.
	/// Port of colmap::RotationMatrixToEulerAngles.
	/// </summary>
	public static (double Rx, double Ry, double Rz) RotationMatrixToEulerAngles(in Matrix3d r)
	{
		double rx = Math.Atan2(r[2, 1], r[2, 2]);
		double ry = Math.Asin(-r[2, 0]);
		double rz = Math.Atan2(r[1, 0], r[0, 0]);
		return (double.IsNaN(rx) ? 0 : rx, double.IsNaN(ry) ? 0 : ry, double.IsNaN(rz) ? 0 : rz);
	}

	/// <summary>R = Rz * Ry * Rx. Port of colmap::EulerAnglesToRotationMatrix.</summary>
	public static Matrix3d EulerAnglesToRotationMatrix(double rx, double ry, double rz)
	{
		Matrix3d mx = new AngleAxisd(rx, Vector3d.UnitX).ToRotationMatrix();
		Matrix3d my = new AngleAxisd(ry, Vector3d.UnitY).ToRotationMatrix();
		Matrix3d mz = new AngleAxisd(rz, Vector3d.UnitZ).ToRotationMatrix();
		return mz * my * mx;
	}

	/// <summary>
	/// The weighted average of unit quaternions (sign-invariant, through
	/// AverageUnitVectors on their coefficients). Port of colmap::AverageQuaternions.
	/// </summary>
	public static Quaterniond AverageQuaternions(IReadOnlyList<Quaterniond> quats, IReadOnlyList<double> weights)
	{
		Check.Eq(quats.Count, weights.Count);

		// Convert quaternions to coefficient matrix (each column is a quaternion).
		var qmat = new MatrixXd(4, quats.Count);
		for (int i = 0; i < quats.Count; ++i)
		{
			qmat.SetCol(i, VectorXd.From(quats[i].Normalized().Coeffs));
		}

		VectorXd avg = AverageUnitVectors(qmat, new VectorXd(weights.ToArray()));

		// Convert back to quaternion (coefficient order: x, y, z, w).
		return new Quaterniond(avg[3], avg[0], avg[1], avg[2]);
	}

	/// <summary>
	/// The unit quaternion of an angle-axis vector; below 1e-10 rad the normalized
	/// first-order (1, w / 2), which keeps the rotation direction.
	/// Port of colmap::QuaternionFromAngleAxis.
	/// </summary>
	public static Quaterniond QuaternionFromAngleAxis(Vector3d omega)
	{
		double theta = omega.Norm;
		if (theta < SmallAngleThreshold)
		{
			// First-order Taylor expansion preserving rotation direction.
			return new Quaterniond(1.0, 0.5 * omega.X, 0.5 * omega.Y, 0.5 * omega.Z).Normalized();
		}

		return Quaterniond.FromAngleAxis(new AngleAxisd(theta, omega / theta));
	}

	/// <summary>
	/// The left Jacobian of SO(3) at <paramref name="omega"/>:
	/// sinc(theta) I + (1 - sinc(theta)) a a^T + ((1 - cos theta) / theta) [a]x.
	/// Port of colmap::LeftJacobianFromAngleAxis.
	/// </summary>
	public static Matrix3d LeftJacobianFromAngleAxis(Vector3d omega)
	{
		double theta = omega.Norm;
		if (theta < SmallAngleThreshold)
		{
			return Matrix3d.Identity + 0.5 * Rigid3d.CrossProductMatrix(omega);
		}

		Vector3d a = omega / theta;
		Matrix3d ax = Rigid3d.CrossProductMatrix(a);
		double sinTheta = Math.Sin(theta);
		double sincTheta = sinTheta / theta;
		Matrix3d aat = Matrix3d.FromColumns(a * a.X, a * a.Y, a * a.Z);
		return sincTheta * Matrix3d.Identity + (1.0 - sincTheta) * aat + ((1.0 - Math.Cos(theta)) / theta) * ax;
	}

	/// <summary>
	/// The right Jacobian of SO(3), Jl(-omega). Port of colmap::RightJacobianFromAngleAxis.
	/// </summary>
	public static Matrix3d RightJacobianFromAngleAxis(Vector3d omega) => LeftJacobianFromAngleAxis(-omega);

	/// <summary>
	/// Interpolates two poses: slerp on the rotations, linear on the translations.
	/// Port of colmap::InterpolateCameraPoses.
	/// </summary>
	public static Rigid3d InterpolateCameraPoses(Rigid3d cam1FromWorld, Rigid3d cam2FromWorld, double t)
	{
		Vector3d translation12 = cam2FromWorld.Translation - cam1FromWorld.Translation;
		return new Rigid3d(
			cam1FromWorld.Rotation.Slerp(t, cam2FromWorld.Rotation),
			cam1FromWorld.Translation + translation12 * t);
	}

	/// <summary>
	/// Collects the indices of the ray pairs whose point lies in front of both cameras,
	/// and returns whether there is any. Port of colmap::CheckCheirality.
	/// </summary>
	public static bool CheckCheirality(
		Rigid3d cam2FromCam1,
		IReadOnlyList<Vector3d> camRays1,
		IReadOnlyList<Vector3d> camRays2,
		List<int> validIndices)
	{
		Check.Eq(camRays1.Count, camRays2.Count);
		validIndices.Clear();
		Matrix3d cam2FromCam1Rot = cam2FromCam1.Rotation.ToRotationMatrix();
		Vector3d translation = cam2FromCam1.Translation;
		for (int i = 0; i < camRays1.Count; ++i)
		{
			// Solve the 2x2 system for the depths of the point along both rays; both
			// must be positive for the point to lie in front of both cameras. This
			// assumes unit-norm rays: the common positive factor 1 / (1 - a^2) is
			// dropped since it does not affect the sign (a = cos angle between the
			// rays, so |a| <= 1).
			Vector3d ray1InCam2 = cam2FromCam1Rot * camRays1[i];
			double a = -ray1InCam2.Dot(camRays2[i]);
			double b1 = -ray1InCam2.Dot(translation);
			double b2 = camRays2[i].Dot(translation);
			if (b1 - a * b2 > 0.0 && b2 - a * b1 > 0.0)
			{
				validIndices.Add(i);
			}
		}

		return validIndices.Count > 0;
	}

	/// <summary>
	/// Re-expresses a camera pose after the world is transformed by
	/// <paramref name="newFromOldWorld"/>. Port of colmap::TransformCameraWorld.
	/// </summary>
	public static Rigid3d TransformCameraWorld(Sim3d newFromOldWorld, Rigid3d camFromWorld)
	{
		Sim3d camFromNewWorld =
			new Sim3d(1, camFromWorld.Rotation, camFromWorld.Translation) * newFromOldWorld.Inverse();
		return new Rigid3d(camFromNewWorld.Rotation, camFromNewWorld.Translation * newFromOldWorld.Scale);
	}

	/// <summary>
	/// A right-handed rotation whose second column is the (unit) gravity direction, the
	/// other two columns an orthonormal basis of its complement from a Householder QR.
	/// Port of colmap::GravityAlignedRotation.
	/// </summary>
	public static Matrix3d GravityAlignedRotation(Vector3d gravity)
	{
		Check.Lt(Math.Abs(gravity.Norm - 1.0), 1e-6, "Gravity vector must be normalized");

		// Use Householder QR to find orthonormal basis vectors for the null space.
		var qr = new HouseholderQR(MatrixXd.FromColumnMajor(3, 1, [gravity.X, gravity.Y, gravity.Z]));
		MatrixXd q = qr.HouseholderQ();
		Vector3d col0 = q.Col(1).ToVector3d();
		Vector3d col2 = q.Col(2).ToVector3d();
		Matrix3d r = Matrix3d.FromColumns(col0, gravity, col2);

		// Ensure right-handed coordinate system.
		if (r.Determinant() < 0)
		{
			r = Matrix3d.FromColumns(col0, gravity, -col2);
		}

		return r;
	}

	/// <summary>
	/// The y component of a rotation's angle-axis vector. Port of colmap::YAxisAngleFromRotation.
	/// </summary>
	public static double YAxisAngleFromRotation(in Matrix3d rotation) => RotationMatrixToAngleAxis(rotation).Y;

	/// <summary>
	/// The rotation by <paramref name="angle"/> about the y axis. Port of colmap::RotationFromYAxisAngle.
	/// </summary>
	public static Matrix3d RotationFromYAxisAngle(double angle) => AngleAxisToRotationMatrix(new Vector3d(0, angle, 0));
}
