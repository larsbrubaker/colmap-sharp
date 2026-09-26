// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Rigid3d: colmap/geometry/rigid3.h and rigid3.cc - the 3D rigid transform (rotation
// quaternion + translation) that every camera pose, rig sensor and relative pose in COLMAP
// is. x_in_b = b_from_a * x_in_a is R * x + t. Sibling: Sim3d.cs (adds a scale). Built on
// LinearAlgebra/Quaterniond, Vector3d, Matrix3d, Matrix3x4d and Matrix6d.
// Tests: ColmapSharp.Tests/Geometry/Rigid3dTests.cs (rigid3_test.cc 1:1) and
// GeometryOracleTests.cs (C#-only, against pycolmap).
//
// Tiers (pinned by GeometryOracleTests):
// - Tier A, bit-identical to pycolmap: Inverse's rotation, composition's rotation,
//   ToMatrix, FromMatrix, Adjoint, the equality operators, and ToString.
// - Tier B: everything that rotates a vector with q * v (applying the transform, the
//   translations of Inverse and of composition, TgtOriginInSrc), because the macOS wheel
//   contracts q * v's cross products into FMAs (docs/CPP_DIVERGENCES.md, entry 6), and
//   AdjointInverse and GetCovarianceForRigid3dInverse, whose 3x3 and 6x6 products the
//   wheel contracts the same way. Differences are a few ulps.
//
// Translation notes:
// - COLMAP stores params = [qx, qy, qz, qw, tx, ty, tz] and hands out mutable Eigen::Map
//   views. This is a readonly struct with Rotation and Translation; mutate with `with`
//   (tform with { Translation = ... }). The constructor order (rotation, translation) is
//   COLMAP's.
// - The parameterless constructor is the identity like COLMAP's, but C#'s default(Rigid3d)
//   (and array elements) has an all-zero rotation, which is not a valid transform. Use
//   new Rigid3d() or Rigid3d.Identity.
// - COLMAP's free functions Inverse and CrossProductMatrix become an instance method and a
//   static method here; the covariance helpers are static methods. The 12x12 joint
//   covariances of GetCovarianceForComposedRigid3d / GetCovarianceForRelativeRigid3d are
//   MatrixXd (checked 12x12), and their 6x12 Jacobians are built as MatrixXd, so those
//   two are Tier B through MatrixXd's products (no pycolmap fixture pins them).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Geometry;

/// <summary>
/// 3D rigid transform, x_in_b = R * x_in_a + t. Port of colmap::Rigid3d.
/// </summary>
public readonly struct Rigid3d : IEquatable<Rigid3d>
{
	/// <summary>The identity transform (identity rotation, zero translation), like COLMAP's default constructor.</summary>
	public Rigid3d()
	{
		Rotation = Quaterniond.Identity;
		Translation = Vector3d.Zero;
	}

	/// <summary>Creates the transform from a rotation and a translation.</summary>
	public Rigid3d(Quaterniond rotation, Vector3d translation)
	{
		Rotation = rotation;
		Translation = translation;
	}

	/// <summary>The identity transform.</summary>
	public static Rigid3d Identity => new();

	/// <summary>The rotation, params [qx, qy, qz, qw] in COLMAP.</summary>
	public Quaterniond Rotation { get; init; }

	/// <summary>The translation, params [tx, ty, tz] in COLMAP.</summary>
	public Vector3d Translation { get; init; }

	/// <summary>
	/// The skew-symmetric matrix [v]x with [v]x * w == v x w.
	/// Port of colmap::CrossProductMatrix (rigid3.h).
	/// </summary>
	public static Matrix3d CrossProductMatrix(Vector3d vector)
	{
		return new Matrix3d(
			0, -vector.Z, vector.Y,
			vector.Z, 0, -vector.X,
			-vector.Y, vector.X, 0);
	}

	/// <summary>[R | t] as a 3x4 matrix.</summary>
	public Matrix3x4d ToMatrix()
	{
		return Matrix3x4d.FromBlocks(Rotation.ToRotationMatrix(), Translation);
	}

	/// <summary>
	/// The transform of a 3x4 matrix [R | t]; the rotation is Eigen's
	/// Quaterniond(R).normalized().
	/// </summary>
	public static Rigid3d FromMatrix(Matrix3x4d matrix)
	{
		return new Rigid3d(Quaterniond.FromRotationMatrix(matrix.LeftCols3()).Normalized(), matrix.Col(3));
	}

	/// <summary>
	/// Adjoint matrix to propagate uncertainty on Rigid3d, [R, 0; [t]x R, R].
	/// Reference: https://gtsam.org/2021/02/23/uncertainties-part3.html
	/// </summary>
	public Matrix6d Adjoint()
	{
		Matrix3d rotation = Rotation.ToRotationMatrix();
		// Eigen's R.colwise().cross(-t): each column c becomes c x (-t), which is t x c.
		Vector3d minusT = -Translation;
		Matrix3d tCrossR = Matrix3d.FromColumns(
			rotation.Col(0).Cross(minusT),
			rotation.Col(1).Cross(minusT),
			rotation.Col(2).Cross(minusT));
		return Matrix6d.FromBlocks(rotation, Matrix3d.Zero, tCrossR, rotation);
	}

	/// <summary>The inverse of <see cref="Adjoint"/>, [R^T, 0; -R^T [t]x, R^T].</summary>
	public Matrix6d AdjointInverse()
	{
		Matrix3d rotationT = Rotation.ToRotationMatrix().Transpose();
		return Matrix6d.FromBlocks(rotationT, Matrix3d.Zero, -rotationT * CrossProductMatrix(Translation), rotationT);
	}

	/// <summary>Return the origin position of the target in the source frame, R^-1 * -t.</summary>
	public Vector3d TgtOriginInSrc()
	{
		return Rotation.Inverse() * -Translation;
	}

	/// <summary>The inverse transform, a_from_b from b_from_a. Port of colmap::Inverse(Rigid3d).</summary>
	public Rigid3d Inverse()
	{
		Quaterniond rotation = Rotation.Inverse();
		return new Rigid3d(rotation, rotation * -Translation);
	}

	/// <summary>
	/// Covariance of the inverse transform, Ad^-1 * covar * Ad^-T.
	/// Port of colmap::GetCovarianceForRigid3dInverse.
	/// </summary>
	public static Matrix6d GetCovarianceForRigid3dInverse(Rigid3d rigid3, in Matrix6d covar)
	{
		Matrix6d adjointInv = rigid3.AdjointInverse();
		return adjointInv * covar * adjointInv.Transpose();
	}

	/// <summary>
	/// Covariance of a_from_c = a_from_b * b_from_c from the 12x12 joint covariance of
	/// (a_from_b, b_from_c): J * covar * J^T with J = [I, Ad(a_from_b)].
	/// Port of colmap::GetCovarianceForComposedRigid3d.
	/// </summary>
	public static Matrix6d GetCovarianceForComposedRigid3d(Rigid3d aFromB, MatrixXd covar)
	{
		CheckJointCovariance(covar);
		var j = new MatrixXd(6, 12);
		j.SetBlock(0, 0, MatrixXd.Identity(6));
		j.SetBlock(0, 6, MatrixXd.From(aFromB.Adjoint()));
		return (j * covar * j.Transpose()).ToMatrix6d();
	}

	/// <summary>
	/// Covariance of the relative pose b_from_a = b_from_c * inverse(a_from_c) from the 12x12
	/// joint covariance of (a_from_c, b_from_c): J * covar * J^T with
	/// J = [-Ad(b_from_c) * Ad(a_from_c)^-1, I].
	/// Port of colmap::GetCovarianceForRelativeRigid3d.
	/// </summary>
	public static Matrix6d GetCovarianceForRelativeRigid3d(Rigid3d aFromC, Rigid3d bFromC, MatrixXd covar)
	{
		CheckJointCovariance(covar);
		var j = new MatrixXd(6, 12);
		j.SetBlock(0, 0, MatrixXd.From(-bFromC.Adjoint() * aFromC.AdjointInverse()));
		j.SetBlock(0, 6, MatrixXd.Identity(6));
		return (j * covar * j.Transpose()).ToMatrix6d();
	}

	// COLMAP's parameter is a fixed Eigen::Matrix<double, 12, 12>; the shape is part of the
	// type there and a runtime check here.
	private static void CheckJointCovariance(MatrixXd covar)
	{
		if (covar.Rows != 12 || covar.Cols != 12)
		{
			throw new ArgumentException($"Expected a 12x12 covariance, got {covar.Rows}x{covar.Cols}.", nameof(covar));
		}
	}

	/// <summary>Apply the transform to a point, R * x + t.</summary>
	public static Vector3d operator *(Rigid3d t, Vector3d x)
	{
		return t.Rotation * x + t.Translation;
	}

	/// <summary>
	/// Concatenate transforms, c_from_a = c_from_b * b_from_a. The rotation is renormalized.
	/// </summary>
	public static Rigid3d operator *(Rigid3d cFromB, Rigid3d bFromA)
	{
		return new Rigid3d(
			(cFromB.Rotation * bFromA.Rotation).Normalized(),
			cFromB.Translation + (cFromB.Rotation * bFromA.Translation));
	}

	/// <summary>COLMAP's operator==: rotation coefficients and translation compare equal with double ==.</summary>
	public static bool operator ==(Rigid3d left, Rigid3d right)
	{
		return left.Rotation.Coeffs == right.Rotation.Coeffs && left.Translation == right.Translation;
	}

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(Rigid3d left, Rigid3d right) => !(left == right);

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Rigid3d other) => Rotation.Equals(other.Rotation) && Translation.Equals(other.Translation);

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Rigid3d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => HashCode.Combine(Rotation, Translation);

	/// <summary>
	/// COLMAP's operator&lt;&lt;, e.g. "Rigid3d(rotation_xyzw=[0, 0, 0, 1], translation=[0, 0, 0])",
	/// numbers in the stream's default 6-significant-digit format.
	/// </summary>
	public override string ToString()
	{
		return "Rigid3d(rotation_xyzw=[" + FormatList(Rotation.X, Rotation.Y, Rotation.Z, Rotation.W)
			+ "], translation=[" + FormatList(Translation.X, Translation.Y, Translation.Z) + "])";
	}

	// Eigen::IOFormat(StreamPrecision, DontAlignCols, ", ", ", ") on a vector.
	internal static string FormatList(params ReadOnlySpan<double> values)
	{
		var parts = new string[values.Length];
		for (int i = 0; i < values.Length; i++)
		{
			parts[i] = CppStreamFormat.FormatDouble(values[i]);
		}

		return string.Join(", ", parts);
	}
}
