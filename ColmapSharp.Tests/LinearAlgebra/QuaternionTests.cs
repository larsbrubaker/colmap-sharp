// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// QuaternionTests (C#-only; COLMAP has no test for Eigen itself): Quaterniond and
// AngleAxisd against known rotations and round trips (quaternion <-> matrix <-> angle-axis).
// The bit-level comparison against Eigen itself is RotationOracleTests; these pin the
// conventions a reader can check by hand (w-first constructor, xyzw coeffs, Hamilton
// product order, the sign convention of FromRotationMatrix).

using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.LinearAlgebra;

public class QuaternionTests
{
	private static readonly double S = Math.Sqrt(0.5);

	private static IEnumerable<Quaterniond> SampleRotations()
	{
		yield return Quaterniond.Identity;
		yield return new Quaterniond(S, S, 0, 0);
		yield return new Quaterniond(0, 1, 0, 0);
		yield return new Quaterniond(0, 0, 1, 0);
		yield return new Quaterniond(0, 0, 0, 1);
		yield return new Quaterniond(0, S, S, 0);
		yield return new Quaterniond(0.5, -0.5, 0.5, 0.5);
		yield return new Quaterniond(0.9, 0.1, -0.2, 0.3).Normalized();
		yield return new Quaterniond(0.05, 0.7, -0.1, 0.7).Normalized();
		yield return new Quaterniond(0.01, -0.3, 0.9, 0.2).Normalized();
	}

	[Test]
	public async Task Constructor_IsWxyz_CoeffsAreXyzw()
	{
		var q = new Quaterniond(1, 2, 3, 4);
		using (Assert.Multiple())
		{
			await Assert.That(q.W).IsEqualTo(1.0);
			await Assert.That(q.X).IsEqualTo(2.0);
			await Assert.That(q.Coeffs).IsEqualTo(new Vector4d(2, 3, 4, 1));
			await Assert.That(Quaterniond.FromCoeffs(q.Coeffs)).IsEqualTo(q);
			await Assert.That(q.Vec).IsEqualTo(new Vector3d(2, 3, 4));
		}
	}

	[Test]
	public async Task NinetyDegreeRotations()
	{
		var aboutZ = new Quaterniond(S, 0, 0, S);
		var aboutX = new Quaterniond(S, S, 0, 0);
		using (Assert.Multiple())
		{
			await Assert.That((aboutZ * Vector3d.UnitX).IsApprox(Vector3d.UnitY)).IsTrue();
			await Assert.That((aboutX * Vector3d.UnitY).IsApprox(Vector3d.UnitZ)).IsTrue();
			await Assert.That(aboutZ.ToRotationMatrix().IsApprox(new Matrix3d(0, -1, 0, 1, 0, 0, 0, 0, 1))).IsTrue();
			await Assert.That(new AngleAxisd(Math.PI / 2, Vector3d.UnitZ).ToRotationMatrix()
				.IsApprox(new Matrix3d(0, -1, 0, 1, 0, 0, 0, 0, 1))).IsTrue();
			// Composition: (a * b) applies b first. Z then X takes x -> y -> z.
			await Assert.That(((aboutX * aboutZ) * Vector3d.UnitX).IsApprox(Vector3d.UnitZ)).IsTrue();
			await Assert.That(aboutZ.AngularDistance(Quaterniond.Identity)).IsEqualTo(Math.PI / 2).Within(1e-15);
		}
	}

	[Test]
	public async Task RotatingAVector_MatchesTheRotationMatrix()
	{
		var v = new Vector3d(0.3, -1.7, 2.2);
		foreach (Quaterniond q in SampleRotations())
		{
			await Assert.That((q * v).IsApprox(q.ToRotationMatrix() * v, 1e-14)).IsTrue();
		}
	}

	[Test]
	public async Task QuaternionMatrixRoundTrip()
	{
		foreach (Quaterniond q in SampleRotations())
		{
			Quaterniond back = Quaterniond.FromRotationMatrix(q.ToRotationMatrix());
			// q and -q are the same rotation; FromRotationMatrix picks the sign by its branch.
			bool same = back.IsApprox(q) || back.IsApprox(new Quaterniond(-q.W, -q.X, -q.Y, -q.Z));
			await Assert.That(same).IsTrue();
		}
	}

	[Test]
	public async Task FromRotationMatrix_SignConvention()
	{
		using (Assert.Multiple())
		{
			// Positive trace: w is the positive component.
			await Assert.That(Quaterniond.FromRotationMatrix(new Quaterniond(-S, 0, 0, S).ToRotationMatrix()).W).IsGreaterThan(0.0);
			// Half turns (trace -1): the component on the largest diagonal entry is positive.
			await Assert.That(Quaterniond.FromRotationMatrix(new Matrix3d(-1, 0, 0, 0, 1, 0, 0, 0, -1)))
				.IsEqualTo(new Quaterniond(0, 0, 1, 0));
			await Assert.That(Quaterniond.FromRotationMatrix(new Matrix3d(1, 0, 0, 0, -1, 0, 0, 0, -1)))
				.IsEqualTo(new Quaterniond(0, 1, 0, 0));
			await Assert.That(Quaterniond.FromRotationMatrix(new Matrix3d(-1, 0, 0, 0, -1, 0, 0, 0, 1)))
				.IsEqualTo(new Quaterniond(0, 0, 0, 1));
		}
	}

	[Test]
	public async Task AngleAxisRoundTrips()
	{
		foreach (Quaterniond q in SampleRotations())
		{
			AngleAxisd aa = AngleAxisd.FromQuaternion(q);
			using (Assert.Multiple())
			{
				await Assert.That(aa.Angle).IsBetween(0.0, Math.PI);
				await Assert.That(aa.ToRotationMatrix().IsApprox(q.ToRotationMatrix(), 1e-14)).IsTrue();
				await Assert.That(aa.ToQuaternion().ToRotationMatrix().IsApprox(q.ToRotationMatrix(), 1e-14)).IsTrue();
				await Assert.That(AngleAxisd.FromRotationMatrix(q.ToRotationMatrix()).ToRotationMatrix()
					.IsApprox(q.ToRotationMatrix(), 1e-14)).IsTrue();
			}
		}
	}

	[Test]
	public async Task AngleAxis_NegativeWFlipsTheAxis()
	{
		// -q is the same rotation; the angle stays in [0, pi] and the axis flips instead.
		var q = new Quaterniond(-Math.Cos(0.3), Math.Sin(0.3), 0, 0);
		AngleAxisd aa = AngleAxisd.FromQuaternion(q);
		using (Assert.Multiple())
		{
			await Assert.That(aa.Angle).IsEqualTo(0.6).Within(1e-15);
			await Assert.That(aa.Axis).IsEqualTo(-Vector3d.UnitX);
			await Assert.That(AngleAxisd.FromQuaternion(Quaterniond.Identity)).IsEqualTo(new AngleAxisd(0, Vector3d.UnitX));
		}
	}

	[Test]
	public async Task AngleAxis_TinyVectorPartKeepsItsAngleAndAxis()
	{
		// Regression: tiny (but nonzero) vector parts used to collapse to angle 0 about x.
		// Angles are the pycolmap/Eigen values (RotationOracleTests pins them bit for bit).
		AngleAxisd tiny = AngleAxisd.FromQuaternion(new Quaterniond(-1, 1e-17, -2e-17, 0));
		AngleAxisd underflowing = AngleAxisd.FromQuaternion(new Quaterniond(1, 1e-200, 0, 0));
		using (Assert.Multiple())
		{
			await Assert.That(tiny.Angle).IsEqualTo(4.47213595499958e-17);
			// w < 0 flips the axis: -(1, -2, 0) / sqrt(5).
			await Assert.That(tiny.Axis.IsApprox(new Vector3d(-1, 2, 0) / Math.Sqrt(5), 1e-15)).IsTrue();
			await Assert.That(underflowing.Angle).IsEqualTo(2e-200);
			await Assert.That(underflowing.Axis).IsEqualTo(Vector3d.UnitX);
		}
	}

	[Test]
	public async Task NormalizedConjugateInverse()
	{
		var q = new Quaterniond(1, 2, 3, 4);
		using (Assert.Multiple())
		{
			await Assert.That(q.Normalized().Norm).IsEqualTo(1.0).Within(1e-15);
			await Assert.That(new Quaterniond(0, 0, 0, 0).Normalized()).IsEqualTo(new Quaterniond(0, 0, 0, 0));
			await Assert.That((q * q.Inverse()).IsApprox(Quaterniond.Identity)).IsTrue();
			await Assert.That(q.Conjugate()).IsEqualTo(new Quaterniond(1, -2, -3, -4));
			await Assert.That(q.Inverse()).IsEqualTo(new Quaterniond(1.0 / 30, -2.0 / 30, -3.0 / 30, -4.0 / 30));
			await Assert.That(new Quaterniond(0, 0, 0, 0).Inverse()).IsEqualTo(new Quaterniond(0, 0, 0, 0));
		}
	}

	/// <summary>
	/// C#-only: FromTwoVectors maps the first direction onto the second. (The general branch is
	/// pinned bit for bit against pycolmap by SyntheticOracleTests' frame rotations.)
	/// </summary>
	[Test]
	public async Task CSharpOnly_FromTwoVectors()
	{
		var a = new Vector3d(1, 2, 3);
		var b = new Vector3d(-2, 0.5, 1);
		Quaterniond q = Quaterniond.FromTwoVectors(a, b);
		await Assert.That((q * a.Normalized() - b.Normalized()).Norm).IsLessThan(1e-15);
		await Assert.That(Math.Abs(q.Norm - 1)).IsLessThan(1e-15);
		// Shortest arc: the rotation angle is the angle between the vectors.
		double c = a.Normalized().Dot(b.Normalized());
		await Assert.That(Math.Abs(q.W - Math.Sqrt((1 + c) / 2))).IsLessThan(1e-15);
	}

	/// <summary>
	/// C#-only: exactly and nearly opposite vectors take the half-turn branch of
	/// FromTwoVectors (1 + c &lt; 1e-8, divergence 28), which must still map
	/// the first direction onto the second to rounding accuracy and return a unit quaternion.
	/// Covers every choice of the least-aligned coordinate axis and both sides of the
	/// threshold.
	/// </summary>
	[Test]
	public async Task CSharpOnly_FromTwoVectorsOpposite()
	{
		Vector3d[] directions =
		[
			new(1, 0, 0), new(0, -1, 0), new(0, 0, 1), new(1, 2, 3), new(-3, 0.5, 0.25), new(0.1, -4, 2),
		];
		// Perpendicular offsets giving 1 + c = 0, ~5e-13, ~5e-11 and ~5e-9 (inside the branch)
		// and ~2e-8 (just outside it).
		double[] offsets = [0, 1e-6, 1e-5, 1e-4, 2e-4];
		var failures = new List<string>();
		foreach (Vector3d direction in directions)
		{
			Vector3d u = direction.Normalized();
			Vector3d perpendicular = u.Cross(new Vector3d(0.3, -0.7, 0.2)).Normalized();
			foreach (double offset in offsets)
			{
				Vector3d target = -u + offset * perpendicular;
				Quaterniond q = Quaterniond.FromTwoVectors(direction, target);
				double error = (q * u - target.Normalized()).Norm;
				// Outside the branch Melax's formula (the same one COLMAP's Eigen uses there) loses
				// accuracy as 1 + c shrinks, ~1e-8 at 1 + c = 2e-8 (the reason for the branch), so
				// the bound there is looser.
				double tolerance = offset < 2e-4 ? 1e-15 * 8 : 1e-7;
				if (error > tolerance || Math.Abs(q.Norm - 1) > tolerance)
				{
					failures.Add($"{direction} offset {offset}: error {error}, norm {q.Norm}");
				}
			}
		}

		await Assert.That(failures).IsEmpty();
	}
}
