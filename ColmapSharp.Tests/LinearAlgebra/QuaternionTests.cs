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
}
