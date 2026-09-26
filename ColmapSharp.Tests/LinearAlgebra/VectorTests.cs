// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// VectorTests (C#-only; COLMAP has no test for Eigen itself): Vector2d, Vector3d and
// Vector4d in ColmapSharp/LinearAlgebra against hand-computed values and Eigen's
// documented semantics (zero-vector normalized(), homogeneous/hnormalized, isApprox).
// Values are exact where the arithmetic is exact, so they compare with IsEqualTo.

using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.LinearAlgebra;

public class VectorTests
{
	[Test]
	public async Task Vector3d_DotCrossNorm()
	{
		var a = new Vector3d(1, 2, 3);
		var b = new Vector3d(4, -5, 6);
		using (Assert.Multiple())
		{
			await Assert.That(a.Dot(b)).IsEqualTo(12.0);
			await Assert.That(a.Cross(b)).IsEqualTo(new Vector3d(27, 6, -13));
			await Assert.That(Vector3d.UnitX.Cross(Vector3d.UnitY)).IsEqualTo(Vector3d.UnitZ);
			await Assert.That(a.SquaredNorm).IsEqualTo(14.0);
			await Assert.That(new Vector3d(2, 3, 6).Norm).IsEqualTo(7.0);
			await Assert.That(a.Cross(b).Dot(a)).IsEqualTo(0.0);
		}
	}

	[Test]
	public async Task Normalized_ZeroVectorStaysZero()
	{
		using (Assert.Multiple())
		{
			// Eigen's documented behavior: normalized() of a zero vector is the vector itself, not NaN.
			await Assert.That(Vector2d.Zero.Normalized()).IsEqualTo(Vector2d.Zero);
			await Assert.That(Vector3d.Zero.Normalized()).IsEqualTo(Vector3d.Zero);
			await Assert.That(Vector4d.Zero.Normalized()).IsEqualTo(Vector4d.Zero);
			await Assert.That(new Vector3d(0, 3, 4).Normalized()).IsEqualTo(new Vector3d(0, 0.6, 0.8));
			await Assert.That(new Vector2d(3, 4).Normalized()).IsEqualTo(new Vector2d(0.6, 0.8));
		}
	}

	[Test]
	public async Task Homogeneous_And_HNormalized()
	{
		using (Assert.Multiple())
		{
			await Assert.That(new Vector2d(3, 4).Homogeneous()).IsEqualTo(new Vector3d(3, 4, 1));
			await Assert.That(new Vector3d(3, 4, 5).Homogeneous()).IsEqualTo(new Vector4d(3, 4, 5, 1));
			await Assert.That(new Vector3d(4, 6, 2).HNormalized()).IsEqualTo(new Vector2d(2, 3));
			await Assert.That(new Vector4d(4, 6, 8, 2).HNormalized()).IsEqualTo(new Vector3d(2, 3, 4));
		}
	}

	[Test]
	public async Task Operators_And_Indexer()
	{
		var a = new Vector4d(1, 2, 3, 4);
		using (Assert.Multiple())
		{
			await Assert.That(a + a).IsEqualTo(a * 2);
			await Assert.That(2 * a - a).IsEqualTo(a);
			await Assert.That(-a / 2).IsEqualTo(new Vector4d(-0.5, -1, -1.5, -2));
			await Assert.That(a[3]).IsEqualTo(4.0);
			await Assert.That(a.Dot(a)).IsEqualTo(30.0);
			await Assert.That(new Vector3d(1, 2, 3)[1]).IsEqualTo(2.0);
			await Assert.That(() => new Vector3d(1, 2, 3)[3]).Throws<ArgumentOutOfRangeException>();
		}
	}

	[Test]
	public async Task Equality_FollowsEigenOperatorEquals()
	{
		var nan = new Vector3d(double.NaN, 0, 0);
		var sameNan = new Vector3d(double.NaN, 0, 0);
		using (Assert.Multiple())
		{
			// operator== is Eigen's coefficient-wise ==, so NaN is never equal;
			// Equals is .NET's, where NaN equals itself (so vectors work as dictionary keys).
			await Assert.That(nan == sameNan).IsFalse();
			await Assert.That(nan.Equals(sameNan)).IsTrue();
			await Assert.That(new Vector3d(0, 0, 0) == new Vector3d(-0.0, 0, 0)).IsTrue();
		}
	}

	[Test]
	public async Task IsApprox_IsRelative()
	{
		var a = new Vector3d(1, 2, 3);
		using (Assert.Multiple())
		{
			await Assert.That(a.IsApprox(a * (1 + 1e-13))).IsTrue();
			await Assert.That(a.IsApprox(a * (1 + 1e-10))).IsFalse();
			await Assert.That(a.IsApprox(a * (1 + 1e-10), 1e-9)).IsTrue();
			// Relative: only an exact zero is approximately zero, as in Eigen.
			await Assert.That(Vector3d.Zero.IsApprox(new Vector3d(1e-100, 0, 0))).IsFalse();
			await Assert.That(Vector3d.Zero.IsApprox(Vector3d.Zero)).IsTrue();
		}
	}

	[Test]
	public async Task Vector4d_ReductionIsPairedLikeEigen()
	{
		// Lanes {0, 2} and {1, 3} are summed first, as Eigen's 2-lane packet reduction
		// does (RotationOracleTests pins this bit for bit through quaternion norms). Here
		// (1 - 1) + (t^2 + t^2) = 2 t^2, while left to right 1 + t^2 rounds back to 1
		// and the sum would come out as t^2.
		double tiny = Math.Pow(2, -27);
		var a = new Vector4d(1, tiny, 1, tiny);
		var b = new Vector4d(1, tiny, -1, tiny);
		await Assert.That(a.Dot(b)).IsEqualTo(2 * tiny * tiny);
	}
}
