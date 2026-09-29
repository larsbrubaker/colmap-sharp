// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// FeatureKeypoint: the FeatureKeypoint struct of colmap/feature/types.h and .cc (a feature
// location with its affine shape), plus FeatureKeypoints (std::vector<FeatureKeypoint>,
// here List<FeatureKeypoint>) and the KeypointsToMatrix / KeypointsFromMatrix /
// MatchesToMatrix / MatchesFromMatrix conversions. Its neighbors are FeatureMatch.cs and
// FeatureDescriptors.cs (the rest of feature/types.h); Scene/Database.cs stores keypoints.
// Tests: ColmapSharp.Tests/Feature/FeatureTypesTests.cs (feature/types_test.cc 1:1).
//
// Tier A for the location/shape bookkeeping (constructors from a11..a22, Rescale, Rot90,
// equality, the matrix conversions of stored fields). The trigonometric pieces
// (the scale/orientation constructor, FromShapeParameters, ComputeScale*,
// ComputeOrientation, ComputeShear) go through MathF.Sin/Cos/Atan2/Sqrt where COLMAP calls
// the platform's float libm; Sqrt is exact, the others may differ in the last ulp, so those
// are Tier B (types_test.cc compares them with a tolerance; divergence 114).
//
// Default value: COLMAP's default constructor is FeatureKeypoint(0, 0), i.e. the identity
// shape a11 = a22 = 1. A C# struct's parameterless constructor runs for
// `new FeatureKeypoint()`, but default(FeatureKeypoint) and new array elements are all-zero
// (a11 = a22 = 0). So `FeatureKeypoints keypoints(n)` is FeatureKeypoints.Create(n), which
// fills the list with new FeatureKeypoint(); never size a keypoint array and rely on its
// elements. The fields are not offset-encoded like FeatureMatch's ids, because an offset
// would not round-trip float values exactly.

using System.Runtime.InteropServices;

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Util;

namespace ColmapSharp.Feature;

/// <summary>
/// Port of colmap::FeatureKeypoint: location (x, y) with the origin at the upper left image
/// corner (the upper left pixel has the coordinate (0.5, 0.5)) and affine shape a11..a22.
/// A mutable struct like the C++ one.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct FeatureKeypoint : IEquatable<FeatureKeypoint>
{
	/// <summary>Location x.</summary>
	public float X;

	/// <summary>Location y.</summary>
	public float Y;

	/// <summary>Affine shape a11.</summary>
	public float A11;

	/// <summary>Affine shape a12.</summary>
	public float A12;

	/// <summary>Affine shape a21.</summary>
	public float A21;

	/// <summary>Affine shape a22.</summary>
	public float A22;

	/// <summary>FeatureKeypoint(0, 0): the origin with identity shape (see the header).</summary>
	public FeatureKeypoint()
		: this(0, 0)
	{
	}

	/// <summary>A keypoint at (x, y) with identity shape.</summary>
	public FeatureKeypoint(float x, float y)
		: this(x, y, 1, 0, 0, 1)
	{
	}

	/// <summary>A keypoint at (x, y) with an isotropic shape of the given scale and orientation.</summary>
	public FeatureKeypoint(float x, float y, float scale, float orientation)
	{
		Check.Ge(scale, 0.0f);
		X = x;
		Y = y;
		float scaleCosOrientation = scale * MathF.Cos(orientation);
		float scaleSinOrientation = scale * MathF.Sin(orientation);
		A11 = scaleCosOrientation;
		A12 = -scaleSinOrientation;
		A21 = scaleSinOrientation;
		A22 = scaleCosOrientation;
	}

	/// <summary>A keypoint at (x, y) with the affine shape [a11 a12; a21 a22].</summary>
	public FeatureKeypoint(float x, float y, float a11, float a12, float a21, float a22)
	{
		X = x;
		Y = y;
		A11 = a11;
		A12 = a12;
		A21 = a21;
		A22 = a22;
	}

	/// <summary>A keypoint from anisotropic scales, orientation and shear.</summary>
	public static FeatureKeypoint FromShapeParameters(
		float x, float y, float scaleX, float scaleY, float orientation, float shear)
	{
		Check.Ge(scaleX, 0.0f);
		Check.Ge(scaleY, 0.0f);
		return new FeatureKeypoint(
			x,
			y,
			scaleX * MathF.Cos(orientation),
			-scaleY * MathF.Sin(orientation + shear),
			scaleX * MathF.Sin(orientation),
			scaleY * MathF.Cos(orientation + shear));
	}

	/// <summary>Rescales the location and shape size by the given factor.</summary>
	public void Rescale(float scale) => Rescale(scale, scale);

	/// <summary>Rescales the location and shape size by per-axis factors.</summary>
	public void Rescale(float scaleX, float scaleY)
	{
		Check.Gt(scaleX, 0.0f);
		Check.Gt(scaleY, 0.0f);
		X *= scaleX;
		Y *= scaleY;
		A11 *= scaleX;
		A12 *= scaleY;
		A21 *= scaleX;
		A22 *= scaleY;
	}

	/// <summary>
	/// Rotates the location and shape by k * 90 degrees counter-clockwise around the image
	/// center; width and height are the dimensions of the image the keypoint is defined on.
	/// </summary>
	public void Rot90(int k, int width, int height)
	{
		k %= 4;
		if (k < 0)
		{
			k += 4;
		}

		if (k == 0)
		{
			return;
		}

		float newX = X, newY = Y;
		float newA11 = A11, newA12 = A12, newA21 = A21, newA22 = A22;
		float w = width;
		float h = height;

		if (k == 1)
		{
			// 90 CCW
			newX = Y;
			newY = w - X;
			newA11 = A21;
			newA12 = A22;
			newA21 = -A11;
			newA22 = -A12;
		}
		else if (k == 2)
		{
			// 180 CCW
			newX = w - X;
			newY = h - Y;
			newA11 = -A11;
			newA12 = -A12;
			newA21 = -A21;
			newA22 = -A22;
		}
		else if (k == 3)
		{
			// 270 CCW
			newX = h - Y;
			newY = X;
			newA11 = -A21;
			newA12 = -A22;
			newA21 = A11;
			newA22 = A12;
		}

		X = newX;
		Y = newY;
		A11 = newA11;
		A12 = newA12;
		A21 = newA21;
		A22 = newA22;
	}

	/// <summary>Mean of the two axis scales.</summary>
	public readonly float ComputeScale() => (ComputeScaleX() + ComputeScaleY()) / 2.0f;

	/// <summary>Scale along the first shape axis.</summary>
	public readonly float ComputeScaleX() => MathF.Sqrt((A11 * A11) + (A21 * A21));

	/// <summary>Scale along the second shape axis.</summary>
	public readonly float ComputeScaleY() => MathF.Sqrt((A12 * A12) + (A22 * A22));

	/// <summary>Orientation of the first shape axis.</summary>
	public readonly float ComputeOrientation() => MathF.Atan2(A21, A11);

	/// <summary>Shear angle between the shape axes.</summary>
	public readonly float ComputeShear() => MathF.Atan2(-A12, A22) - ComputeOrientation();

	/// <summary>Exact equality of location and shape (float ==).</summary>
	public static bool operator ==(FeatureKeypoint left, FeatureKeypoint right) =>
		left.X == right.X && left.Y == right.Y && left.A11 == right.A11
		&& left.A12 == right.A12 && left.A21 == right.A21 && left.A22 == right.A22;

	/// <summary>Inequality of location or shape.</summary>
	public static bool operator !=(FeatureKeypoint left, FeatureKeypoint right) => !(left == right);

	/// <inheritdoc/>
	public readonly bool Equals(FeatureKeypoint other) => this == other;

	/// <inheritdoc/>
	public override readonly bool Equals(object? obj) => obj is FeatureKeypoint other && this == other;

	/// <inheritdoc/>
	public override readonly int GetHashCode() => HashCode.Combine(X, Y, A11, A12, A21, A22);

	/// <inheritdoc/>
	public override readonly string ToString() => $"FeatureKeypoint({X}, {Y}, {A11}, {A12}, {A21}, {A22})";
}

/// <summary>
/// FeatureKeypoints (std::vector&lt;FeatureKeypoint&gt;) helpers and the keypoint/match
/// matrix conversions of colmap/feature/types.h.
/// </summary>
public static class FeatureKeypoints
{
	/// <summary>kKeypointMatrixCols: the columns [x, y, scale, orientation].</summary>
	public const int KeypointMatrixCols = 4;

	/// <summary>
	/// `FeatureKeypoints keypoints(count)`: <paramref name="count"/> default keypoints (origin,
	/// identity shape), unlike an array of default(FeatureKeypoint) (see the file header).
	/// </summary>
	public static List<FeatureKeypoint> Create(int count)
	{
		var keypoints = new List<FeatureKeypoint>(count);
		for (int i = 0; i < count; ++i)
		{
			keypoints.Add(new FeatureKeypoint());
		}

		return keypoints;
	}

	/// <summary>Port of KeypointsToMatrix: an Nx4 matrix [x, y, scale, orientation].</summary>
	public static RowMajorMatrix<float> KeypointsToMatrix(IReadOnlyList<FeatureKeypoint> featureKeypoints)
	{
		int numFeatures = featureKeypoints.Count;
		var keypoints = new RowMajorMatrix<float>(numFeatures, KeypointMatrixCols);
		for (int i = 0; i < numFeatures; ++i)
		{
			FeatureKeypoint keypoint = featureKeypoints[i];
			keypoints[i, 0] = keypoint.X;
			keypoints[i, 1] = keypoint.Y;
			keypoints[i, 2] = keypoint.ComputeScale();
			keypoints[i, 3] = keypoint.ComputeOrientation();
		}

		return keypoints;
	}

	/// <summary>Port of KeypointsFromMatrix: keypoints from an Nx4 [x, y, scale, orientation] matrix.</summary>
	public static List<FeatureKeypoint> KeypointsFromMatrix(RowMajorMatrix<float> keypoints)
	{
		// Eigen's Ref<const Matrix<float, Dynamic, 4>> fixes the column count at compile time.
		Check.Eq(keypoints.Cols, KeypointMatrixCols);
		var featureKeypoints = new List<FeatureKeypoint>(keypoints.Rows);
		for (int i = 0; i < keypoints.Rows; ++i)
		{
			featureKeypoints.Add(new FeatureKeypoint(keypoints[i, 0], keypoints[i, 1], keypoints[i, 2], keypoints[i, 3]));
		}

		return featureKeypoints;
	}

	/// <summary>Port of MatchesToMatrix: an Nx2 matrix of point2D indices.</summary>
	public static RowMajorMatrix<uint> MatchesToMatrix(IReadOnlyList<FeatureMatch> featureMatches)
	{
		int numMatches = featureMatches.Count;
		var matches = new RowMajorMatrix<uint>(numMatches, 2);
		for (int i = 0; i < numMatches; ++i)
		{
			matches[i, 0] = featureMatches[i].Point2DIdx1;
			matches[i, 1] = featureMatches[i].Point2DIdx2;
		}

		return matches;
	}

	/// <summary>Port of MatchesFromMatrix: matches from an Nx2 matrix of point2D indices.</summary>
	public static List<FeatureMatch> MatchesFromMatrix(RowMajorMatrix<uint> matches)
	{
		Check.Eq(matches.Cols, 2);
		var featureMatches = new List<FeatureMatch>(matches.Rows);
		for (int i = 0; i < matches.Rows; ++i)
		{
			featureMatches.Add(new FeatureMatch(matches[i, 0], matches[i, 1]));
		}

		return featureMatches;
	}
}
