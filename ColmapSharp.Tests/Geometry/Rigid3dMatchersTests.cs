// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Rigid3dMatchersTests: colmap/geometry/rigid3_matchers_test.cc ported 1:1 (suite
// "Rigid3d", named Rigid3d_<Name>), testing the test-side matchers in Rigid3dMatchers.cs.
// The gmock StrictMock halves of Eq and Near check that the matcher accepts each argument
// the mocked method is called with (x matches Rigid3dEq(x), y matches Rigid3dEq(y)); with
// predicates that is the matcher applied to (x, x) and (y, y).
//
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Geometry;

public class Rigid3dMatchersTests
{
	private static Rigid3d WithRotation(Rigid3d t, double dw = 0, double dx = 0)
	{
		Quaterniond q = t.Rotation;
		return t with { Rotation = new Quaterniond(q.W + dw, q.X + dx, q.Y, q.Z) };
	}

	private static Rigid3d WithTranslationX(Rigid3d t, double dx)
	{
		Vector3d v = t.Translation;
		return t with { Translation = new Vector3d(v.X + dx, v.Y, v.Z) };
	}

	[Test]
	public async Task Rigid3d_Eq()
	{
		var x = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());
		Rigid3d y = x;
		bool equal = Rigid3dEq(x, y);
		y = WithRotation(y, dw: 1e-7);
		bool rotationDiffers = !Rigid3dEq(x, y);
		y = x;
		y = WithTranslationX(y, 1e-7);
		bool translationDiffers = !Rigid3dEq(x, y);
		using (Assert.Multiple())
		{
			await Assert.That(equal).IsTrue();
			await Assert.That(rotationDiffers).IsTrue();
			await Assert.That(translationDiffers).IsTrue();
			await Assert.That(Rigid3dEq(x, x)).IsTrue();
			await Assert.That(Rigid3dEq(y, y)).IsTrue();
		}
	}

	[Test]
	public async Task Rigid3d_Near()
	{
		var x = new Rigid3d(RandomEigen.RandomEigenQuaterniond(), RandomEigen.RandomEigenVector3d());
		Rigid3d y = x;
		bool near = Rigid3dNear(x, y, rtol: 1e-8, ttol: 1e-8);
		y = WithRotation(y, dw: 1e-7);
		bool rotationFar = !Rigid3dNear(x, y, rtol: 1e-8, ttol: 1e-8);
		y = x;
		y = WithTranslationX(y, 1e-7);
		bool translationFar = !Rigid3dNear(x, y);
		using (Assert.Multiple())
		{
			await Assert.That(near).IsTrue();
			await Assert.That(rotationFar).IsTrue();
			await Assert.That(translationFar).IsTrue();
			await Assert.That(Rigid3dNear(x, x)).IsTrue();
			await Assert.That(Rigid3dNear(y, y)).IsTrue();
		}
	}

	[Test]
	public async Task Rigid3d_LeftRotationNearIdentity()
	{
		var x = new Rigid3d(Quaterniond.Identity, Vector3d.Zero);
		Rigid3d y = x;
		bool near = Rigid3dNear(x, y, 1e-8);
		x = WithRotation(x, dx: 1e-16);
		bool stillNear = Rigid3dNear(x, y, 1e-8);
		x = WithRotation(x, dx: 1e-7);
		bool far = !Rigid3dNear(x, y, 1e-8);
		using (Assert.Multiple())
		{
			await Assert.That(near).IsTrue();
			await Assert.That(stillNear).IsTrue();
			await Assert.That(far).IsTrue();
		}
	}

	[Test]
	public async Task Rigid3d_RightRotationNearIdentity()
	{
		var x = new Rigid3d(Quaterniond.Identity, Vector3d.Zero);
		Rigid3d y = x;
		bool near = Rigid3dNear(x, y, 1e-8);
		y = WithRotation(y, dx: 1e-16);
		bool stillNear = Rigid3dNear(x, y, 1e-8);
		y = WithRotation(y, dx: 1e-7);
		bool far = !Rigid3dNear(x, y, 1e-8);
		using (Assert.Multiple())
		{
			await Assert.That(near).IsTrue();
			await Assert.That(stillNear).IsTrue();
			await Assert.That(far).IsTrue();
		}
	}

	[Test]
	public async Task Rigid3d_LeftTranslationNearIdentity()
	{
		var x = new Rigid3d(Quaterniond.Identity, Vector3d.Zero);
		Rigid3d y = x;
		bool near = Rigid3dNear(x, y, 1e-8);
		x = WithTranslationX(x, 1e-16);
		bool stillNear = Rigid3dNear(x, y, 1e-8);
		x = WithTranslationX(x, 1e-7);
		bool far = !Rigid3dNear(x, y, 1e-8);
		using (Assert.Multiple())
		{
			await Assert.That(near).IsTrue();
			await Assert.That(stillNear).IsTrue();
			await Assert.That(far).IsTrue();
		}
	}

	[Test]
	public async Task Rigid3d_RightTranslationNearIdentity()
	{
		var x = new Rigid3d(Quaterniond.Identity, Vector3d.Zero);
		Rigid3d y = x;
		bool near = Rigid3dNear(x, y, 1e-8);
		y = WithTranslationX(y, 1e-16);
		bool stillNear = Rigid3dNear(x, y, 1e-8);
		y = WithTranslationX(y, 1e-7);
		bool far = !Rigid3dNear(x, y, 1e-8);
		using (Assert.Multiple())
		{
			await Assert.That(near).IsTrue();
			await Assert.That(stillNear).IsTrue();
			await Assert.That(far).IsTrue();
		}
	}
}
