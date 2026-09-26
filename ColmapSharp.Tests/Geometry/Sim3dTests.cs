// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Sim3dTests: colmap/geometry/sim3_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Geometry/Sim3d.cs. sim3_test.cc names one of its tests
// TEST(Rigid3d, ApplyChain) although it tests Sim3d; the name is kept. The comparisons
// against pycolmap are in GeometryOracleTests (C#-only), which states each function's tier.
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test, so each test starts with
// RandomUtils.SetPRNGSeed(0) and draws everything before its first await (the PRNG is per
// thread and an await may resume elsewhere).

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Geometry;

public class Sim3dTests
{
	private static Sim3d TestSim3d()
	{
		double scale = RandomUtils.RandomUniformReal(0.1, 10.0);
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d();
		return new Sim3d(scale, rotation, translation);
	}

	[Test]
	public async Task Sim3d_Default()
	{
		var tform = new Sim3d();
		using (Assert.Multiple())
		{
			await Assert.That(tform.Scale).IsEqualTo(1.0);
			await Assert.That(tform.Rotation.Coeffs == Quaterniond.Identity.Coeffs).IsTrue();
			await Assert.That(tform.Translation == Vector3d.Zero).IsTrue();
		}
	}

	[Test]
	public async Task Sim3d_Equals()
	{
		var tform = new Sim3d();
		Sim3d other = tform;
		bool equalAtStart = tform == other;
		tform = tform with { Translation = new Vector3d(1, tform.Translation.Y, tform.Translation.Z) };
		bool notEqualAfterChange = tform != other;
		other = other with { Translation = new Vector3d(1, other.Translation.Y, other.Translation.Z) };
		bool equalAgain = tform == other;
		using (Assert.Multiple())
		{
			await Assert.That(equalAtStart).IsTrue();
			await Assert.That(notEqualAfterChange).IsTrue();
			await Assert.That(equalAgain).IsTrue();
		}
	}

	[Test]
	public async Task Sim3d_Print()
	{
		var tform = new Sim3d();
		await Assert.That(tform.ToString())
			.IsEqualTo("Sim3d(scale=1, rotation_xyzw=[0, 0, 0, 1], translation=[0, 0, 0])");
	}

	[Test]
	public async Task Sim3d_Inverse()
	{
		RandomUtils.SetPRNGSeed(0);
		Sim3d bFromA = TestSim3d();
		Sim3d aFromB = bFromA.Inverse();
		var results = new List<bool>();
		for (int i = 0; i < 100; ++i)
		{
			Vector3d xInA = RandomEigen.RandomEigenVector3d();
			Vector3d xInB = bFromA * xInA;
			results.Add(EigenMatrixNear(aFromB * xInB, xInA, 1e-6));
		}

		await Assert.That(results.All(r => r)).IsTrue();
	}

	[Test]
	public async Task Sim3d_ToMatrix()
	{
		RandomUtils.SetPRNGSeed(0);
		Sim3d bFromA = TestSim3d();
		Matrix3x4d bFromAMat = bFromA.ToMatrix();
		var errors = new List<double>();
		for (int i = 0; i < 100; ++i)
		{
			Vector3d xInA = RandomEigen.RandomEigenVector3d();
			errors.Add((bFromA * xInA - bFromAMat * xInA.Homogeneous()).Norm);
		}

		await Assert.That(errors.Max()).IsLessThan(1e-6);
	}

	[Test]
	public async Task Sim3d_FromMatrix()
	{
		RandomUtils.SetPRNGSeed(0);
		Sim3d b1FromA = TestSim3d();
		Sim3d b2FromA = Sim3d.FromMatrix(b1FromA.ToMatrix());
		var results = new List<bool>();
		for (int i = 0; i < 100; ++i)
		{
			Vector3d xInA = RandomEigen.RandomEigenVector3d();
			results.Add(EigenMatrixNear(b1FromA * xInA, b2FromA * xInA, 1e-6));
		}

		await Assert.That(results.All(r => r)).IsTrue();
	}

	[Test]
	public async Task Sim3d_ApplyScaleOnly()
	{
		var bFromA = new Sim3d(2, Quaterniond.Identity, Vector3d.Zero);
		await Assert.That((bFromA * new Vector3d(1, 2, 3) - new Vector3d(2, 4, 6)).Norm).IsLessThan(1e-6);
	}

	[Test]
	public async Task Sim3d_ApplyTranslationOnly()
	{
		var bFromA = new Sim3d(1, Quaterniond.Identity, new Vector3d(1, 2, 3));
		await Assert.That((bFromA * new Vector3d(1, 2, 3) - new Vector3d(2, 4, 6)).Norm).IsLessThan(1e-6);
	}

	[Test]
	public async Task Sim3d_ApplyRotationOnly()
	{
		var bFromA = new Sim3d(1, new AngleAxisd(Math.PI / 2, Vector3d.UnitX).ToQuaternion(), Vector3d.Zero);
		await Assert.That((bFromA * new Vector3d(1, 2, 3) - new Vector3d(1, -3, 2)).Norm).IsLessThan(1e-6);
	}

	[Test]
	public async Task Sim3d_ApplyScaleRotationTranslation()
	{
		var bFromA = new Sim3d(2, new AngleAxisd(Math.PI / 2, Vector3d.UnitX).ToQuaternion(), new Vector3d(1, 2, 3));
		await Assert.That((bFromA * new Vector3d(1, 2, 3) - new Vector3d(3, -4, 7)).Norm).IsLessThan(1e-6);
	}

	[Test]
	public async Task Rigid3d_ApplyChain()
	{
		RandomUtils.SetPRNGSeed(0);
		Sim3d bFromA = TestSim3d();
		Sim3d cFromB = TestSim3d();
		Sim3d dFromC = TestSim3d();
		Vector3d xInA = RandomEigen.RandomEigenVector3d();
		Vector3d xInB = bFromA * xInA;
		Vector3d xInC = cFromB * xInB;
		Vector3d xInD = dFromC * xInC;
		await Assert.That(dFromC * (cFromB * (bFromA * xInA)) == xInD).IsTrue();
	}

	[Test]
	public async Task Sim3d_Compose()
	{
		RandomUtils.SetPRNGSeed(0);
		Sim3d bFromA = TestSim3d();
		Sim3d cFromB = TestSim3d();
		Sim3d dFromC = TestSim3d();
		Sim3d dFromA = dFromC * cFromB * bFromA;
		Vector3d xInA = RandomEigen.RandomEigenVector3d();
		Vector3d xInB = bFromA * xInA;
		Vector3d xInC = cFromB * xInB;
		Vector3d xInD = dFromC * xInC;
		await Assert.That(EigenMatrixNear(dFromA * xInA, xInD, 1e-6)).IsTrue();
	}

	[Test]
	public async Task Sim3d_ToFromFile()
	{
		// CreateTestDir() in COLMAP: a fresh directory per test.
		string dir = Path.Combine(Path.GetTempPath(), "ColmapSharpTests", nameof(Sim3d_ToFromFile), Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		try
		{
			string path = Path.Combine(dir, "file.txt");
			RandomUtils.SetPRNGSeed(0);
			Sim3d written = TestSim3d();
			written.ToFile(path);
			Sim3d read = Sim3d.FromFile(path);
			using (Assert.Multiple())
			{
				await Assert.That(read.Scale).IsEqualTo(written.Scale);
				await Assert.That(read.Rotation.Coeffs == written.Rotation.Coeffs).IsTrue();
				await Assert.That(read.Translation == written.Translation).IsTrue();
			}
		}
		finally
		{
			Directory.Delete(dir, true);
		}
	}
}
