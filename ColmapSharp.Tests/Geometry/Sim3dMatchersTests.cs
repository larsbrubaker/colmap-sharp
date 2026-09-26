// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Sim3dMatchersTests: colmap/geometry/sim3_matchers_test.cc ported 1:1, one method per
// gtest TEST(Suite, Name) named Suite_Name. Tests ColmapSharp.Tests/Sim3dMatchers.cs.
//
// The gmock StrictMock blocks (EXPECT_CALL(mock, TestMethod(Sim3dEq(x))).Times(1), then
// calling TestMethod with x and y) check that the matcher accepts each argument it was
// built from; they are ported as those two matcher applications. Sim3d_Near perturbs the
// rotation twice where one might expect the scale; that is COLMAP's test as written.
// PrngTestIsolation seeds the PRNG with 0 before every test, as COLMAP's gtest_main does.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Sim3dMatchers;

namespace ColmapSharp.Tests.Geometry;

public class Sim3dMatchersTests
{
	private static Sim3d RandomSim3d()
	{
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d();
		return new Sim3d(2, rotation, translation);
	}

	private static Sim3d Unit() => new(1, Quaterniond.Identity, Vector3d.Zero);

	private static Sim3d AddToScale(Sim3d s, double delta) => s with { Scale = s.Scale + delta };

	private static Sim3d AddToRotationW(Sim3d s, double delta)
	{
		Quaterniond q = s.Rotation;
		return s with { Rotation = new Quaterniond(q.W + delta, q.X, q.Y, q.Z) };
	}

	private static Sim3d AddToRotationX(Sim3d s, double delta)
	{
		Quaterniond q = s.Rotation;
		return s with { Rotation = new Quaterniond(q.W, q.X + delta, q.Y, q.Z) };
	}

	private static Sim3d AddToTranslationX(Sim3d s, double delta)
	{
		Vector3d t = s.Translation;
		return s with { Translation = new Vector3d(t.X + delta, t.Y, t.Z) };
	}

	[Test]
	public async Task Sim3d_Eq()
	{
		Sim3d x = RandomSim3d();
		Sim3d y = x;
		bool equal = Sim3dEq(x, y);
		y = AddToScale(y, 1e-7);
		bool scaleDiffers = !Sim3dEq(x, y);
		y = x;
		y = AddToRotationW(y, 1e-7);
		bool rotationDiffers = !Sim3dEq(x, y);
		y = x;
		y = AddToTranslationX(y, 1e-7);
		bool translationDiffers = !Sim3dEq(x, y);
		using (Assert.Multiple())
		{
			await Assert.That(equal).IsTrue();
			await Assert.That(scaleDiffers).IsTrue();
			await Assert.That(rotationDiffers).IsTrue();
			await Assert.That(translationDiffers).IsTrue();
			await Assert.That(Sim3dEq(x, x)).IsTrue();
			await Assert.That(Sim3dEq(y, y)).IsTrue();
		}
	}

	[Test]
	public async Task Sim3d_Near()
	{
		Sim3d x = RandomSim3d();
		Sim3d y = x;
		bool near = Sim3dNear(x, y, stol: 1e-8, rtol: 1e-8, ttol: 1e-8);
		y = AddToRotationW(y, 1e-7);
		bool rotationFar = !Sim3dNear(x, y, stol: 1e-8, rtol: 1e-8, ttol: 1e-8);
		y = x;
		y = AddToRotationW(y, 1e-7);
		bool rotationFarAgain = !Sim3dNear(x, y, stol: 1e-8, rtol: 1e-8, ttol: 1e-8);
		y = x;
		y = AddToTranslationX(y, 1e-7);
		bool translationFar = !Sim3dNear(x, y);
		using (Assert.Multiple())
		{
			await Assert.That(near).IsTrue();
			await Assert.That(rotationFar).IsTrue();
			await Assert.That(rotationFarAgain).IsTrue();
			await Assert.That(translationFar).IsTrue();
			await Assert.That(Sim3dNear(x, x)).IsTrue();
			await Assert.That(Sim3dNear(y, y)).IsTrue();
		}
	}

	[Test]
	public async Task Sim3d_LeftScaleNearIdentity()
	{
		Sim3d x = Unit();
		Sim3d y = x;
		bool atStart = Sim3dNear(x, y, 1e-8);
		x = AddToScale(x, 1e-16);
		bool tiny = Sim3dNear(x, y, 1e-8);
		x = AddToScale(x, 1e-7);
		bool far = !Sim3dNear(x, y, 1e-8);
		await AssertAll(atStart, tiny, far);
	}

	[Test]
	public async Task Sim3d_RightScaleNearIdentity()
	{
		Sim3d x = Unit();
		Sim3d y = x;
		bool atStart = Sim3dNear(x, y, 1e-8);
		y = AddToScale(y, 1e-16);
		bool tiny = Sim3dNear(x, y, 1e-8);
		y = AddToScale(y, 1e-7);
		bool far = !Sim3dNear(x, y, 1e-8);
		await AssertAll(atStart, tiny, far);
	}

	[Test]
	public async Task Sim3d_LeftRotationNearIdentity()
	{
		Sim3d x = Unit();
		Sim3d y = x;
		bool atStart = Sim3dNear(x, y, 1e-8);
		x = AddToRotationX(x, 1e-16);
		bool tiny = Sim3dNear(x, y, 1e-8);
		x = AddToRotationX(x, 1e-7);
		bool far = !Sim3dNear(x, y, 1e-8);
		await AssertAll(atStart, tiny, far);
	}

	[Test]
	public async Task Sim3d_RightRotationNearIdentity()
	{
		Sim3d x = Unit();
		Sim3d y = x;
		bool atStart = Sim3dNear(x, y, 1e-8);
		y = AddToRotationX(y, 1e-16);
		bool tiny = Sim3dNear(x, y, 1e-8);
		y = AddToRotationX(y, 1e-7);
		bool far = !Sim3dNear(x, y, 1e-8);
		await AssertAll(atStart, tiny, far);
	}

	[Test]
	public async Task Sim3d_LeftTranslationNearIdentity()
	{
		Sim3d x = Unit();
		Sim3d y = x;
		bool atStart = Sim3dNear(x, y, 1e-8);
		x = AddToTranslationX(x, 1e-16);
		bool tiny = Sim3dNear(x, y, 1e-8);
		x = AddToTranslationX(x, 1e-7);
		bool far = !Sim3dNear(x, y, 1e-8);
		await AssertAll(atStart, tiny, far);
	}

	[Test]
	public async Task Sim3d_RightTranslationNearIdentity()
	{
		Sim3d x = Unit();
		Sim3d y = x;
		bool atStart = Sim3dNear(x, y, 1e-8);
		y = AddToTranslationX(y, 1e-16);
		bool tiny = Sim3dNear(x, y, 1e-8);
		y = AddToTranslationX(y, 1e-7);
		bool far = !Sim3dNear(x, y, 1e-8);
		await AssertAll(atStart, tiny, far);
	}

	// EXPECT_THAT(x, Sim3dNear(y, 1e-8)) at the start, after the 1e-16 nudge, and
	// EXPECT_THAT(x, Not(Sim3dNear(y, 1e-8))) after the 1e-7 one.
	private static async Task AssertAll(bool atStart, bool tiny, bool far)
	{
		using (Assert.Multiple())
		{
			await Assert.That(atStart).IsTrue();
			await Assert.That(tiny).IsTrue();
			await Assert.That(far).IsTrue();
		}
	}
}
