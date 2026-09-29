// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Sim3d: colmap/geometry/sim3.h and sim3.cc - the 3D similarity transform with 7 degrees
// of freedom (scale, rotation quaternion, translation) that aligns reconstructions to each
// other and to GPS/ground truth. x_in_b = scale * R * x_in_a + t. Sibling: Rigid3d.cs (no
// scale), whose formatting helper it shares. Tests: ColmapSharp.Tests/Geometry/Sim3dTests.cs
// (sim3_test.cc 1:1) and GeometryOracleTests.cs (C#-only, against pycolmap).
//
// Tiers (pinned by GeometryOracleTests):
// - Tier A, bit-identical to pycolmap: Inverse's scale and rotation, composition's scale
//   and rotation, ToMatrix, FromMatrix, the equality operators, ToString, and the
//   ToFile/FromFile text round trip (17 significant digits, lossless).
// - Tier B: everything that rotates a vector with q * v (applying the transform, the
//   translations of Inverse and of composition), for the FMA reason in
//   divergence 6.
//
// Translation notes: as for Rigid3d - a readonly struct mutated with `with`, whose
// parameterless constructor (not default(Sim3d)) is the identity; COLMAP's free Inverse is
// an instance method. COLMAP stores params = [qx, qy, qz, qw, tx, ty, tz, s].

using System.Globalization;

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Geometry;

/// <summary>
/// 3D similarity transform, x_in_b = scale * R * x_in_a + t. Port of colmap::Sim3d.
/// </summary>
public readonly struct Sim3d : IEquatable<Sim3d>
{
	/// <summary>The identity transform (scale 1), like COLMAP's default constructor.</summary>
	public Sim3d()
	{
		Scale = 1;
		Rotation = Quaterniond.Identity;
		Translation = Vector3d.Zero;
	}

	/// <summary>Creates the transform from a scale, a rotation and a translation (COLMAP's argument order).</summary>
	public Sim3d(double scale, Quaterniond rotation, Vector3d translation)
	{
		Scale = scale;
		Rotation = rotation;
		Translation = translation;
	}

	/// <summary>The identity transform.</summary>
	public static Sim3d Identity => new();

	/// <summary>The scale, params[7] in COLMAP.</summary>
	public double Scale { get; init; }

	/// <summary>The rotation, params [qx, qy, qz, qw] in COLMAP.</summary>
	public Quaterniond Rotation { get; init; }

	/// <summary>The translation, params [tx, ty, tz] in COLMAP.</summary>
	public Vector3d Translation { get; init; }

	/// <summary>[scale * R | t] as a 3x4 matrix.</summary>
	public Matrix3x4d ToMatrix()
	{
		return Matrix3x4d.FromBlocks(Scale * Rotation.ToRotationMatrix(), Translation);
	}

	/// <summary>
	/// The transform of a 3x4 matrix [s R | t]: the scale is the norm of the first column,
	/// the rotation Eigen's Quaterniond(M / s).normalized().
	/// </summary>
	public static Sim3d FromMatrix(Matrix3x4d matrix)
	{
		double scale = matrix.Col(0).Norm;
		return new Sim3d(
			scale,
			Quaterniond.FromRotationMatrix(matrix.LeftCols3() / scale).Normalized(),
			matrix.Col(3));
	}

	/// <summary>
	/// Write to a text file without loss of precision: "s qw qx qy qz tx ty tz" with 17
	/// significant digits. Port of Sim3d::ToFile.
	/// </summary>
	public void ToFile(string path)
	{
		string[] values =
		[
			CppStreamFormat.FormatDouble(Scale, 17),
			CppStreamFormat.FormatDouble(Rotation.W, 17),
			CppStreamFormat.FormatDouble(Rotation.X, 17),
			CppStreamFormat.FormatDouble(Rotation.Y, 17),
			CppStreamFormat.FormatDouble(Rotation.Z, 17),
			CppStreamFormat.FormatDouble(Translation.X, 17),
			CppStreamFormat.FormatDouble(Translation.Y, 17),
			CppStreamFormat.FormatDouble(Translation.Z, 17),
		];
		File.WriteAllText(path, string.Join(" ", values) + "\n");
	}

	/// <summary>
	/// Read a transform written by <see cref="ToFile"/>: eight whitespace-separated numbers,
	/// s qw qx qy qz tx ty tz. Port of Sim3d::FromFile.
	/// </summary>
	/// <exception cref="ArgumentException">The file is missing or holds fewer than eight numbers (COLMAP's THROW_CHECK).</exception>
	public static Sim3d FromFile(string path)
	{
		Check.That(File.Exists(path), path, "File.Exists(path)");
		string[] tokens = File.ReadAllText(path).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
		var values = new double[8];
		bool ok = tokens.Length >= 8;
		for (int i = 0; ok && i < 8; i++)
		{
			ok = double.TryParse(tokens[i], NumberStyles.Float, CultureInfo.InvariantCulture, out values[i]);
		}

		Check.That(ok, path, "file >> t.scale() >> t.rotation().w() >> t.rotation().x() >> t.rotation().y() >> t.rotation().z() >> t.translation().x() >> t.translation().y() >> t.translation().z()");
		return new Sim3d(
			values[0],
			new Quaterniond(values[1], values[2], values[3], values[4]),
			new Vector3d(values[5], values[6], values[7]));
	}

	/// <summary>The inverse transform, a_from_b from b_from_a. Port of colmap::Inverse(Sim3d).</summary>
	public Sim3d Inverse()
	{
		Quaterniond rotation = Rotation.Inverse();
		return new Sim3d(1 / Scale, rotation, (rotation * Translation) / -Scale);
	}

	/// <summary>
	/// Apply the transform to a point, scale * (R * x) + t. Note that, as in C++, the
	/// operator is left-associative: d_from_c * c_from_b * b_from_a * x concatenates the
	/// transforms first and then applies the result to x.
	/// </summary>
	public static Vector3d operator *(Sim3d t, Vector3d x)
	{
		return t.Scale * (t.Rotation * x) + t.Translation;
	}

	/// <summary>Concatenate transforms, c_from_a = c_from_b * b_from_a. The rotation is renormalized.</summary>
	public static Sim3d operator *(Sim3d cFromB, Sim3d bFromA)
	{
		return new Sim3d(
			cFromB.Scale * bFromA.Scale,
			(cFromB.Rotation * bFromA.Rotation).Normalized(),
			cFromB.Translation + (cFromB.Scale * (cFromB.Rotation * bFromA.Translation)));
	}

	/// <summary>COLMAP's operator==: scale, rotation coefficients and translation compare equal with double ==.</summary>
	public static bool operator ==(Sim3d left, Sim3d right)
	{
		return left.Scale == right.Scale
			&& left.Rotation.Coeffs == right.Rotation.Coeffs
			&& left.Translation == right.Translation;
	}

	/// <summary>Negation of ==.</summary>
	public static bool operator !=(Sim3d left, Sim3d right) => !(left == right);

	/// <summary>.NET equality (double.Equals per coefficient, so NaN equals NaN).</summary>
	public bool Equals(Sim3d other)
	{
		return Scale.Equals(other.Scale) && Rotation.Equals(other.Rotation) && Translation.Equals(other.Translation);
	}

	/// <inheritdoc/>
	public override bool Equals(object? obj) => obj is Sim3d other && Equals(other);

	/// <inheritdoc/>
	public override int GetHashCode() => HashCode.Combine(Scale, Rotation, Translation);

	/// <summary>
	/// COLMAP's operator&lt;&lt;, e.g.
	/// "Sim3d(scale=1, rotation_xyzw=[0, 0, 0, 1], translation=[0, 0, 0])", numbers in the
	/// stream's default 6-significant-digit format.
	/// </summary>
	public override string ToString()
	{
		return "Sim3d(scale=" + CppStreamFormat.FormatDouble(Scale)
			+ ", rotation_xyzw=[" + Rigid3d.FormatList(Rotation.X, Rotation.Y, Rotation.Z, Rotation.W)
			+ "], translation=[" + Rigid3d.FormatList(Translation.X, Translation.Y, Translation.Z) + "])";
	}
}
