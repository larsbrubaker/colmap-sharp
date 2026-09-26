// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureTypesTests: colmap/feature/types_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Feature/FeatureMatch.cs, FeatureKeypoint.cs and FeatureDescriptors.cs.
//
// `FeatureMatches matches(1)` value-initializes one match; the C# equivalent is an array of
// one default element, which FeatureMatch decodes as two invalid indices like COLMAP.
// `FeatureKeypoints keypoints(n)` is FeatureKeypoints.Create(n) (a default FeatureKeypoint
// array element would have a zero shape, see FeatureKeypoint.cs). EXPECT_NEAR on floats
// compares the floats promoted to double, like gtest. EXPECT_FLOAT_EQ (4 ulps) in Rot90
// becomes exact equality: those results are exact float negations/subtractions of small
// integers. FeatureDescriptorsData::Random draws Eigen's std::rand bytes; the values do not
// matter to these round trips, so RandomBytes uses a seeded System.Random.

using ColmapSharp.Feature;
using ColmapSharp.LinearAlgebra;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Tests.Feature;

public class FeatureTypesTests
{
	internal static RowMajorMatrix<byte> RandomBytes(int rows, int cols, int seed = 0)
	{
		var data = new byte[rows * cols];
		new Random(seed).NextBytes(data);
		return new RowMajorMatrix<byte>(rows, cols, data);
	}

	[Test]
	public async Task FeatureKeypoints_Nominal()
	{
		using (Assert.Multiple())
		{
			var keypoint = new FeatureKeypoint();
			await ExpectExact(keypoint, 0, 0, 1, 0, 0, 1);

			List<FeatureKeypoint> keypoints = FeatureKeypoints.Create(1);
			await Assert.That(keypoints.Count).IsEqualTo(1);
			await ExpectExact(keypoints[0], 0, 0, 1, 0, 0, 1);

			keypoint = new FeatureKeypoint(1, 2);
			await ExpectExact(keypoint, 1, 2, 1, 0, 0, 1);
			await ExpectShapeParameters(keypoint, 1, 1, 1, 0, 0);

			keypoint = new FeatureKeypoint(1, 2, 0, 0);
			await ExpectExact(keypoint, 1, 2, 0, 0, 0, 0);

			keypoint = new FeatureKeypoint(1, 2, 1, 0);
			await ExpectExact(keypoint, 1, 2, 1, 0, 0, 1);
			await ExpectShapeParameters(keypoint, 1, 1, 1, 0, 0);

			keypoint = new FeatureKeypoint(1, 2, 1, (float)(Math.PI / 2));
			await ExpectLocation(keypoint, 1, 2);
			await ExpectShape(keypoint, 0, -1, 1, 0);
			await ExpectShapeParameters(keypoint, 1, 1, 1, Math.PI / 2, 0);

			keypoint = new FeatureKeypoint(1, 2, 2, (float)(Math.PI / 2));
			await ExpectLocation(keypoint, 1, 2);
			await ExpectShape(keypoint, 0, -2, 2, 0);
			await ExpectShapeParameters(keypoint, 2, 2, 2, Math.PI / 2, 0);

			keypoint = new FeatureKeypoint(1, 2, 2, (float)Math.PI);
			await ExpectLocation(keypoint, 1, 2);
			await ExpectShape(keypoint, -2, 0, 0, -2);
			await ExpectScalesPiOrientation(keypoint, 2, 2, 2);

			keypoint = FeatureKeypoint.FromShapeParameters(1, 2, 2, 2, (float)Math.PI, 0);
			await ExpectLocation(keypoint, 1, 2);
			await ExpectShape(keypoint, -2, 0, 0, -2);
			await ExpectScalesPiOrientation(keypoint, 2, 2, 2);

			keypoint = FeatureKeypoint.FromShapeParameters(1, 2, 2, 3, (float)Math.PI, 0);
			await ExpectLocation(keypoint, 1, 2);
			await ExpectShape(keypoint, -2, 0, 0, -3);
			await ExpectScalesPiOrientation(keypoint, 2.5, 2, 3);

			keypoint = FeatureKeypoint.FromShapeParameters(1, 2, 2, 3, (float)(-Math.PI / 2), (float)(Math.PI / 4));
			await ExpectLocation(keypoint, 1, 2);
			await ExpectShape(keypoint, 0, 2.12132025f, -2, 2.12132025f);
			await ExpectShapeParameters(keypoint, 2.5, 2, 3, -Math.PI / 2, Math.PI / 4);

			keypoint = FeatureKeypoint.FromShapeParameters(1, 2, 2, 3, (float)(Math.PI / 2), (float)(Math.PI / 4));
			await ExpectLocation(keypoint, 1, 2);
			await ExpectShape(keypoint, 0, -2.12132025f, 2, -2.12132025f);
			await ExpectShapeParameters(keypoint, 2.5, 2, 3, Math.PI / 2, Math.PI / 4);

			keypoint.Rescale(2, 2);
			await ExpectLocation(keypoint, 2, 4);
			await ExpectShape(keypoint, 2 * 0.0f, 2 * -2.12132025f, 2 * 2.0f, 2 * -2.12132025f);
			await ExpectShapeParameters(keypoint, 2 * 2.5f, 2 * 2.0f, 2 * 3.0f, Math.PI / 2, Math.PI / 4);

			keypoint.Rescale(1, 0.5f);
			await ExpectLocation(keypoint, 2, 2);
			await ExpectShape(keypoint, 0, -2.12132025f, 4, -2.12132025f);
			await Assert.That((double)keypoint.ComputeScale()).IsEqualTo(3.5).Within(1e-6);
			await Assert.That((double)(keypoint.ComputeScaleX() - 2)).IsEqualTo(2.0).Within(1e-6);
			await Assert.That((double)keypoint.ComputeScaleY()).IsEqualTo(3.0).Within(1e-6);
			await Assert.That((double)keypoint.ComputeOrientation()).IsEqualTo(Math.PI / 2).Within(1e-6);
			await Assert.That((double)keypoint.ComputeShear()).IsEqualTo(Math.PI / 4).Within(1e-6);

			FeatureKeypoint same = keypoint;
			await Assert.That(keypoint == same).IsTrue();
			await Assert.That(keypoint != new FeatureKeypoint(1, 2, 1, 0)).IsTrue();
		}
	}

	[Test]
	public async Task FeatureKeypoint_Rot90()
	{
		var kp = new FeatureKeypoint(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
		int w = 10, h = 20;

		FeatureKeypoint kp1 = kp;
		kp1.Rot90(1, w, h); // 90 CCW
		FeatureKeypoint kp2 = kp;
		kp2.Rot90(2, w, h); // 180 CCW
		FeatureKeypoint kp3 = kp;
		kp3.Rot90(3, w, h); // 270 CCW
		FeatureKeypoint kpIdentity = kp;
		kpIdentity.Rot90(0, w, h);
		FeatureKeypoint kpIdentity4 = kp;
		kpIdentity4.Rot90(4, w, h);
		FeatureKeypoint kpNeg1 = kp;
		kpNeg1.Rot90(-1, w, h); // same as 3

		using (Assert.Multiple())
		{
			await ExpectExact(kp1, 2.0f, 10.0f - 1.0f, 5.0f, 6.0f, -3.0f, -4.0f);
			await ExpectExact(kp2, 10.0f - 1.0f, 20.0f - 2.0f, -3.0f, -4.0f, -5.0f, -6.0f);
			await ExpectExact(kp3, 20.0f - 2.0f, 1.0f, -5.0f, -6.0f, 3.0f, 4.0f);
			await Assert.That(kpIdentity == kp).IsTrue();
			await Assert.That(kpIdentity4 == kp).IsTrue();
			await Assert.That(kpNeg1 == kp3).IsTrue();
		}
	}

	[Test]
	public async Task FeatureDescriptors_Nominal()
	{
		var descriptors = new FeatureDescriptors(FeatureExtractorType.Sift, RandomBytes(2, 3));
		using (Assert.Multiple())
		{
			await Assert.That(descriptors.Type).IsEqualTo(FeatureExtractorType.Sift);
			await Assert.That(descriptors.Data.Rows).IsEqualTo(2);
			await Assert.That(descriptors.Data.Cols).IsEqualTo(3);
			await Assert.That(descriptors.Data[0, 0]).IsEqualTo(descriptors.Data.Data[0]);
			await Assert.That(descriptors.Data[0, 1]).IsEqualTo(descriptors.Data.Data[1]);
			await Assert.That(descriptors.Data[0, 2]).IsEqualTo(descriptors.Data.Data[2]);
			await Assert.That(descriptors.Data[1, 0]).IsEqualTo(descriptors.Data.Data[3]);
			await Assert.That(descriptors.Data[1, 1]).IsEqualTo(descriptors.Data.Data[4]);
			await Assert.That(descriptors.Data[1, 2]).IsEqualTo(descriptors.Data.Data[5]);
		}
	}

	[Test]
	public async Task FeatureDescriptors_SiftConversion()
	{
		// SIFT uses value cast (uint8 <-> float)
		var original = new FeatureDescriptors(FeatureExtractorType.Sift, RandomBytes(10, 128));
		FeatureDescriptorsFloat asFloat = original.ToFloat();
		var expectedFloat = new RowMajorMatrix<float>(10, 128, original.Data.Data.Select(value => (float)value).ToArray());
		FeatureDescriptors recovered = asFloat.ToBytes();
		using (Assert.Multiple())
		{
			await Assert.That(asFloat.Type).IsEqualTo(FeatureExtractorType.Sift);
			await Assert.That(asFloat.Data.Cols).IsEqualTo(original.Data.Cols);
			await Assert.That(asFloat.Data == expectedFloat).IsTrue();
			await Assert.That(recovered.Type).IsEqualTo(FeatureExtractorType.Sift);
			await Assert.That(recovered.Data == original.Data).IsTrue();
		}
	}

	[Test]
	public async Task FeatureDescriptors_AlikedConversion()
	{
		// ALIKED uses reinterpret cast (float32 bytes <-> float)
		var original = new FeatureDescriptors(FeatureExtractorType.AlikedN16Rot, RandomBytes(10, 512));
		FeatureDescriptorsFloat asFloat = original.ToFloat();
		FeatureDescriptors recovered = asFloat.ToBytes();
		using (Assert.Multiple())
		{
			await Assert.That(asFloat.Type).IsEqualTo(FeatureExtractorType.AlikedN16Rot);
			await Assert.That(asFloat.Data.Cols * sizeof(float)).IsEqualTo(original.Data.Cols);
			await Assert.That(recovered.Type).IsEqualTo(FeatureExtractorType.AlikedN16Rot);
			await Assert.That(recovered.Data == original.Data).IsTrue();
		}
	}

	[Test]
	public async Task FeatureMatches_Nominal()
	{
		var match = new FeatureMatch();
		FeatureMatch sameMatch = match;
		var matches = new List<FeatureMatch>(new FeatureMatch[1]);
		using (Assert.Multiple())
		{
			await Assert.That(match.Point2DIdx1).IsEqualTo(InvalidPoint2DIdx);
			await Assert.That(match.Point2DIdx2).IsEqualTo(InvalidPoint2DIdx);
			await Assert.That(matches.Count).IsEqualTo(1);
			await Assert.That(matches[0].Point2DIdx1).IsEqualTo(InvalidPoint2DIdx);
			await Assert.That(matches[0].Point2DIdx2).IsEqualTo(InvalidPoint2DIdx);

			await Assert.That(match == sameMatch).IsTrue();
			await Assert.That(match != new FeatureMatch(0, 1)).IsTrue();
		}
	}

	[Test]
	public async Task KeypointsMatrixConversion_Roundtrip()
	{
		var keypoints = new List<FeatureKeypoint>
		{
			new(1.0f, 2.0f, 3.0f, 0.5f),
			new(4.0f, 5.0f, 1.0f, -0.3f),
			new(10.0f, 20.0f, 0.5f, 0.0f),
		};
		List<FeatureKeypoint> recovered = FeatureKeypoints.KeypointsFromMatrix(FeatureKeypoints.KeypointsToMatrix(keypoints));
		await Assert.That(recovered.Count).IsEqualTo(keypoints.Count);
		using (Assert.Multiple())
		{
			for (int i = 0; i < keypoints.Count; ++i)
			{
				await Assert.That(recovered[i].X).IsEqualTo(keypoints[i].X);
				await Assert.That(recovered[i].Y).IsEqualTo(keypoints[i].Y);
				await Assert.That((double)recovered[i].ComputeScale()).IsEqualTo(keypoints[i].ComputeScale()).Within(1e-5);
				await Assert.That((double)recovered[i].ComputeOrientation()).IsEqualTo(keypoints[i].ComputeOrientation()).Within(1e-5);
			}
		}
	}

	[Test]
	public async Task MatchesMatrixConversion_Roundtrip()
	{
		var matches = new List<FeatureMatch> { new(0, 5), new(3, 7), new(100, 200) };
		List<FeatureMatch> recovered = FeatureKeypoints.MatchesFromMatrix(FeatureKeypoints.MatchesToMatrix(matches));
		await Assert.That(recovered.SequenceEqual(matches)).IsTrue();
	}

	private static async Task ExpectExact(FeatureKeypoint keypoint, float x, float y, float a11, float a12, float a21, float a22)
	{
		await ExpectLocation(keypoint, x, y);
		await Assert.That(keypoint.A11).IsEqualTo(a11);
		await Assert.That(keypoint.A12).IsEqualTo(a12);
		await Assert.That(keypoint.A21).IsEqualTo(a21);
		await Assert.That(keypoint.A22).IsEqualTo(a22);
	}

	private static async Task ExpectLocation(FeatureKeypoint keypoint, float x, float y)
	{
		await Assert.That(keypoint.X).IsEqualTo(x);
		await Assert.That(keypoint.Y).IsEqualTo(y);
	}

	private static async Task ExpectShape(FeatureKeypoint keypoint, double a11, double a12, double a21, double a22)
	{
		await Assert.That((double)keypoint.A11).IsEqualTo(a11).Within(1e-6);
		await Assert.That((double)keypoint.A12).IsEqualTo(a12).Within(1e-6);
		await Assert.That((double)keypoint.A21).IsEqualTo(a21).Within(1e-6);
		await Assert.That((double)keypoint.A22).IsEqualTo(a22).Within(1e-6);
	}

	private static async Task ExpectShapeParameters(
		FeatureKeypoint keypoint, double scale, double scaleX, double scaleY, double orientation, double shear)
	{
		await Assert.That((double)keypoint.ComputeScale()).IsEqualTo(scale).Within(1e-6);
		await Assert.That((double)keypoint.ComputeScaleX()).IsEqualTo(scaleX).Within(1e-6);
		await Assert.That((double)keypoint.ComputeScaleY()).IsEqualTo(scaleY).Within(1e-6);
		await Assert.That((double)keypoint.ComputeOrientation()).IsEqualTo(orientation).Within(1e-6);
		await Assert.That((double)keypoint.ComputeShear()).IsEqualTo(shear).Within(1e-6);
	}

	// The orientation of a shape rotated by pi may come out as pi or -pi.
	private static async Task ExpectScalesPiOrientation(FeatureKeypoint keypoint, double scale, double scaleX, double scaleY)
	{
		await Assert.That((double)keypoint.ComputeScale()).IsEqualTo(scale).Within(1e-6);
		await Assert.That((double)keypoint.ComputeScaleX()).IsEqualTo(scaleX).Within(1e-6);
		await Assert.That((double)keypoint.ComputeScaleY()).IsEqualTo(scaleY).Within(1e-6);
		float orientation = keypoint.ComputeOrientation();
		await Assert.That(Math.Abs(orientation - Math.PI) < 1e-6 || Math.Abs(orientation + Math.PI) < 1e-6).IsTrue();
		await Assert.That((double)keypoint.ComputeShear()).IsEqualTo(0.0).Within(1e-6);
	}
}
