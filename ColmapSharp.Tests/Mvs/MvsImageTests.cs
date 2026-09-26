// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MvsImageTests: colmap/mvs/image_test.cc ported 1:1 (Suite_Name), testing
// ColmapSharp/Mvs/Image.cs. Named MvsImageTests so a --treenode-filter on the class does
// not also select Scene/ImageTests.cs. EXPECT_FLOAT_EQ is gtest's 4-ulp comparison
// (MvsTestUtils.FloatEq). Tier B (docs/CPP_DIVERGENCES.md, entry 63): the products may
// differ from an FMA-contracting C++ build in the last bits, within the 4-ulp checks.

using ColmapSharp.Mvs;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Mvs.MvsTestUtils;

namespace ColmapSharp.Tests.Mvs;

public class MvsImageTests
{
	private static readonly float[] K = [100, 0, 50, 0, 100, 50, 0, 0, 1];
	private static readonly float[] R = [1, 0, 0, 0, 1, 0, 0, 0, 1];
	private static readonly float[] T = [0, 0, 0];

	[Test]
	public async Task Image_DefaultConstructor()
	{
		var image = new Image();
		await Assert.That(image.GetWidth()).IsEqualTo(0);
		await Assert.That(image.GetHeight()).IsEqualTo(0);
		await Assert.That(image.GetPath()).IsEmpty();
	}

	[Test]
	public async Task Image_ParameterizedConstructor()
	{
		var image = new Image("test.jpg", 100, 100, K, R, T);

		await Assert.That(image.GetWidth()).IsEqualTo(100);
		await Assert.That(image.GetHeight()).IsEqualTo(100);
		await Assert.That(image.GetPath()).IsEqualTo("test.jpg");

		float[] imageK = image.GetK().ToArray();
		float[] imageR = image.GetR().ToArray();
		float[] imageT = image.GetT().ToArray();

		for (int i = 0; i < 9; ++i)
		{
			await Assert.That(imageK[i]).IsEqualTo(K[i]);
			await Assert.That(imageR[i]).IsEqualTo(R[i]);
		}

		for (int i = 0; i < 3; ++i)
		{
			await Assert.That(imageT[i]).IsEqualTo(T[i]);
		}
	}

	[Test]
	public async Task Image_Rescale()
	{
		var image = new Image("test.jpg", 100, 100, K, R, T);
		image.Rescale(0.5f);

		await Assert.That(image.GetWidth()).IsEqualTo(50);
		await Assert.That(image.GetHeight()).IsEqualTo(50);

		float[] imageK = image.GetK().ToArray();
		await Assert.That(FloatEq(imageK[0], 50.0f)).IsTrue(); // fx scaled
		await Assert.That(FloatEq(imageK[2], 25.0f)).IsTrue(); // cx scaled
		await Assert.That(FloatEq(imageK[4], 50.0f)).IsTrue(); // fy scaled
		await Assert.That(FloatEq(imageK[5], 25.0f)).IsTrue(); // cy scaled
	}

	[Test]
	public async Task Image_RescaleNonUniform()
	{
		var image = new Image("test.jpg", 100, 100, K, R, T);
		image.Rescale(0.5f, 0.25f);

		await Assert.That(image.GetWidth()).IsEqualTo(50);
		await Assert.That(image.GetHeight()).IsEqualTo(25);

		float[] imageK = image.GetK().ToArray();
		await Assert.That(FloatEq(imageK[0], 50.0f)).IsTrue(); // fx scaled by factor_x
		await Assert.That(FloatEq(imageK[2], 25.0f)).IsTrue(); // cx scaled by factor_x
		await Assert.That(FloatEq(imageK[4], 25.0f)).IsTrue(); // fy scaled by factor_y
		await Assert.That(FloatEq(imageK[5], 12.5f)).IsTrue(); // cy scaled by factor_y
	}

	[Test]
	public async Task Image_Downsize()
	{
		var image = new Image("test.jpg", 100, 100, K, R, T);
		image.Downsize(50, 50);

		await Assert.That(image.GetWidth()).IsEqualTo(50);
		await Assert.That(image.GetHeight()).IsEqualTo(50);

		float[] imageK = image.GetK().ToArray();
		await Assert.That(FloatEq(imageK[0], 50.0f)).IsTrue(); // fx scaled
		await Assert.That(FloatEq(imageK[2], 25.0f)).IsTrue(); // cx scaled
		await Assert.That(FloatEq(imageK[4], 50.0f)).IsTrue(); // fy scaled
		await Assert.That(FloatEq(imageK[5], 25.0f)).IsTrue(); // cy scaled
	}

	[Test]
	public async Task Image_DownsizeNoChange()
	{
		var image = new Image("test.jpg", 100, 100, K, R, T);
		image.Downsize(200, 200);

		await Assert.That(image.GetWidth()).IsEqualTo(100);
		await Assert.That(image.GetHeight()).IsEqualTo(100);
	}

	[Test]
	public async Task Image_GetViewingDirection()
	{
		var image = new Image("test.jpg", 100, 100, K, R, T);
		float[] viewingDir = image.GetViewingDirection().ToArray();

		await Assert.That(viewingDir[0]).IsEqualTo(R[6]);
		await Assert.That(viewingDir[1]).IsEqualTo(R[7]);
		await Assert.That(viewingDir[2]).IsEqualTo(R[8]);
	}

	[Test]
	public async Task ComputeProjectionCenter_Identity()
	{
		float[] r = [1, 0, 0, 0, 1, 0, 0, 0, 1];
		float[] t = [1, 2, 3];
		float[] c = new float[3];
		MvsGeometry.ComputeProjectionCenter(r, t, c);

		await Assert.That(FloatEq(c[0], -1.0f)).IsTrue();
		await Assert.That(FloatEq(c[1], -2.0f)).IsTrue();
		await Assert.That(FloatEq(c[2], -3.0f)).IsTrue();
	}

	[Test]
	public async Task ComposeProjectionMatrix_Identity()
	{
		float[] k = [2, 0, 0, 0, 2, 0, 0, 0, 1];
		float[] r = [0, 1, 0, 1, 0, 0, 0, 0, 1];
		float[] t = [1, 2, 3];
		float[] p = new float[12];
		MvsGeometry.ComposeProjectionMatrix(k, r, t, p);

		float[] expected = [0, 2, 0, 2, 2, 0, 0, 4, 0, 0, 1, 3];
		for (int i = 0; i < 12; ++i)
		{
			await Assert.That(FloatEq(p[i], expected[i])).IsTrue().Because($"{i}");
		}
	}

	[Test]
	public async Task RotatePose_Identity()
	{
		float[] rr = [1, 0, 0, 0, 1, 0, 0, 0, 1];
		float[] r = [1, 0, 0, 0, 1, 0, 0, 0, 1];
		float[] t = [1, 2, 3];

		MvsGeometry.RotatePose(rr, r, t);

		float[] expectedR = [1, 0, 0, 0, 1, 0, 0, 0, 1];
		float[] expectedT = [1, 2, 3];

		for (int i = 0; i < 9; ++i)
		{
			await Assert.That(FloatEq(r[i], expectedR[i])).IsTrue();
		}

		for (int i = 0; i < 3; ++i)
		{
			await Assert.That(FloatEq(t[i], expectedT[i])).IsTrue();
		}
	}

	// C#-only: GetInvP is the top of the inverse of [P; 0 0 0 1] (Tier B, divergence 63),
	// so [P; 0 0 0 1] [InvP; 0 0 0 1] is the identity up to float rounding: each entry within
	// a few float epsilons of the magnitude of the products it sums.
	[Test]
	public async Task Image_InverseProjectionMatrix()
	{
		float[] k = [800, 0, 320, 0, 780, 240, 0, 0, 1];
		float[] r = [0.36f, 0.48f, -0.8f, -0.8f, 0.6f, 0, 0.48f, 0.64f, 0.6f];
		float[] t = [0.5f, -1.25f, 4];
		var image = new Image("a.jpg", 640, 480, k, r, t);
		float[] p = image.GetP().ToArray();
		float[] invP = image.GetInvP().ToArray();
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				double sum = j == 3 ? p[i * 4 + 3] : 0.0;
				double magnitude = Math.Abs(sum);
				for (int m = 0; m < 3; m++)
				{
					double term = (double)p[i * 4 + m] * invP[m * 4 + j];
					sum += term;
					magnitude += Math.Abs(term);
				}

				double expected = i == j ? 1.0 : 0.0;
				await Assert.That(sum).IsEqualTo(expected).Within(8 * 1.1920929E-07 * magnitude).Because($"({i}, {j})");
			}
		}
	}

	[Test]
	public async Task RotatePose_Rotation90DegreesZ()
	{
		// 90 degree rotation around Z axis
		float[] rr = [0, -1, 0, 1, 0, 0, 0, 0, 1];
		float[] r = [1, 0, 0, 0, 1, 0, 0, 0, 1];
		float[] t = [1, 0, 0];

		MvsGeometry.RotatePose(rr, r, t);

		float[] expectedR = [0, -1, 0, 1, 0, 0, 0, 0, 1];
		float[] expectedT = [0, 1, 0];

		for (int i = 0; i < 9; ++i)
		{
			await Assert.That(FloatEq(r[i], expectedR[i])).IsTrue().Because($"{i}");
		}

		for (int i = 0; i < 3; ++i)
		{
			await Assert.That(FloatEq(t[i], expectedT[i])).IsTrue().Because($"{i}");
		}
	}
}
