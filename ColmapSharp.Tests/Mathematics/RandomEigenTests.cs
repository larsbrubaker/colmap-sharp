// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RandomEigenTests: colmap/math/random_eigen_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name. Tests ColmapSharp/Mathematics/RandomEigen.cs.
// The template shapes map to RandomEigen's per-type methods: RandomEigenVectord<4> is
// RandomEigenVector4d, RandomEigenVectorf<2> is RandomEigenVectorf(2),
// RandomEigenMatrixd<3, 4> is RandomEigenMatrix3x4d.
// Each test seeds the PRNG with 0, as COLMAP's gtest_main does.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mathematics;

public class RandomEigenTests
{
	private static bool InRange(double value) => value >= -1 && value <= 1;

	[Test]
	public async Task RandomEigenVectord_Range()
	{
		RandomUtils.SetPRNGSeed(0);
		bool allInRange = true;
		for (int i = 0; i < 1000; ++i)
		{
			Vector3d vector = RandomEigen.RandomEigenVector3d();
			allInRange &= InRange(vector.X) && InRange(vector.Y) && InRange(vector.Z);
		}

		await Assert.That(allInRange).IsTrue();
	}

	[Test]
	public async Task RandomEigenVectord_Deterministic()
	{
		RandomUtils.SetPRNGSeed(42);
		Vector4d vector1 = RandomEigen.RandomEigenVector4d();
		RandomUtils.SetPRNGSeed(42);
		Vector4d vector2 = RandomEigen.RandomEigenVector4d();
		await Assert.That(vector1 == vector2).IsTrue();
	}

	[Test]
	public async Task RandomEigenVectorf_Range()
	{
		RandomUtils.SetPRNGSeed(0);
		bool allInRange = true;
		for (int i = 0; i < 1000; ++i)
		{
			float[] vector = RandomEigen.RandomEigenVectorf(2);
			allInRange &= vector.Length == 2 && vector.All(v => v >= -1 && v <= 1);
		}

		await Assert.That(allInRange).IsTrue();
	}

	[Test]
	public async Task RandomEigenVectorXd_Dynamic()
	{
		RandomUtils.SetPRNGSeed(0);
		VectorXd vector = RandomEigen.RandomEigenVectorXd(7);
		using (Assert.Multiple())
		{
			await Assert.That(vector.Length).IsEqualTo(7);
			await Assert.That(vector.AsSpan().ToArray().All(InRange)).IsTrue();
		}
	}

	[Test]
	public async Task RandomEigenMatrixd_Range()
	{
		RandomUtils.SetPRNGSeed(0);
		Matrix3x4d matrix = RandomEigen.RandomEigenMatrix3x4d();
		var values = new double[12];
		matrix.CopyToColumnMajor(values);
		await Assert.That(values.All(InRange)).IsTrue();
	}

	[Test]
	public async Task RandomEigenMatrixXf_Dynamic()
	{
		RandomUtils.SetPRNGSeed(0);
		float[,] matrix = RandomEigen.RandomEigenMatrixXf(3, 5);
		using (Assert.Multiple())
		{
			await Assert.That(matrix.GetLength(0)).IsEqualTo(3);
			await Assert.That(matrix.GetLength(1)).IsEqualTo(5);
			await Assert.That(matrix.Cast<float>().All(v => v >= -1 && v <= 1)).IsTrue();
		}
	}

	[Test]
	public async Task RandomEigenQuaterniond_Unit()
	{
		RandomUtils.SetPRNGSeed(0);
		double maxError = 0;
		for (int i = 0; i < 1000; ++i)
		{
			Quaterniond quat = RandomEigen.RandomEigenQuaterniond();
			maxError = Math.Max(maxError, Math.Abs(quat.Norm - 1.0));
		}

		await Assert.That(maxError).IsLessThanOrEqualTo(1e-9);
	}

	[Test]
	public async Task RandomEigenQuaterniond_Deterministic()
	{
		RandomUtils.SetPRNGSeed(42);
		Quaterniond quat1 = RandomEigen.RandomEigenQuaterniond();
		RandomUtils.SetPRNGSeed(42);
		Quaterniond quat2 = RandomEigen.RandomEigenQuaterniond();
		await Assert.That(quat1.Coeffs == quat2.Coeffs).IsTrue();
	}
}
