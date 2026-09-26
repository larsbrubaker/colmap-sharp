// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Image: colmap/mvs/image.h and image.cc - the MVS view of an undistorted image: its size,
// single-precision K, R, T, the projection matrix P = K [R | T] and its inverse, and
// optionally its bitmap. Model.cs builds these from a sparse reconstruction; PatchMatch and
// fusion read the matrices. The free functions (ComputeRelativePose .. RotatePose) are in
// MvsGeometry below. Tests: ColmapSharp.Tests/Mvs/MvsImageTests.cs (mvs/image_test.cc
// 1:1).
//
// Tier B: last-bit differences are possible from FMA contraction on the C++ side. The 3x3
// and 3x4 products (P = K [R | T], RotatePose, ComputeRelativePose, ComputeProjectionCenter)
// are summed in Eigen's coefficient order (k = 0, 1, 2) as separate multiplies and adds,
// but Eigen 3.4 on aarch64 may evaluate these fixed-size float products with a fused pmadd,
// and no oracle fixture pins them. GetInvP additionally comes from a 4x4 float inverse
// written here from the textbook adjugate (Laplace expansion over 2x2 minors), which may
// differ from Eigen's inverse in the last float bits. Both are docs/CPP_DIVERGENCES.md,
// entry 63. Everything else (sizes, Rescale's K scaling, Downsize) is exact.
//
// Translation notes:
// - All matrices are row-major float arrays, as COLMAP's Eigen::Map<RowMajor> views are.
// - The accessors return ReadOnlySpan<float> views of the image's own arrays (C++ returns
//   const float*); GetViewingDirection is the third row of R.
// - Bitmap is a reference type: SetBitmap keeps the given bitmap (C++ moves it in).

using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

/// <summary>
/// Port of colmap::mvs::Image: an undistorted image's size, path, pinhole K, pose R, T
/// (camera from world), projection matrices and optional bitmap.
/// </summary>
public sealed class Image
{
	private readonly float[] k = new float[9];
	private readonly float[] r = new float[9];
	private readonly float[] t = new float[3];
	private readonly float[] p = new float[12];
	private readonly float[] invP = new float[12];
	private int width;
	private int height;
	private Bitmap bitmap = new();

	/// <summary>An empty image (size 0 x 0, empty path).</summary>
	public Image()
	{
		Path = "";
	}

	/// <summary>
	/// An image of the given size with row-major 3x3 calibration <paramref name="kMatrix"/>,
	/// row-major rotation <paramref name="rMatrix"/> and translation <paramref name="tVector"/>.
	/// </summary>
	public Image(string path, int width, int height, ReadOnlySpan<float> kMatrix, ReadOnlySpan<float> rMatrix, ReadOnlySpan<float> tVector)
	{
		Path = path;
		this.width = width;
		this.height = height;
		kMatrix[..9].CopyTo(k);
		rMatrix[..9].CopyTo(r);
		tVector[..3].CopyTo(t);
		MvsGeometry.ComposeProjectionMatrix(k, r, t, p);
		MvsGeometry.ComposeInverseProjectionMatrix(k, r, t, invP);
	}

	/// <summary>
	/// A deep copy: its own K, R, T, P, inverse P and a clone of the bitmap (if set), so
	/// Rescale/Downsize on the copy leave this image untouched (C++ copies by value).
	/// </summary>
	public Image Clone()
	{
		var copy = new Image(Path, width, height, k, r, t);
		if (!bitmap.IsEmpty)
		{
			copy.bitmap = bitmap.Clone();
		}

		return copy;
	}

	/// <summary>Width in pixels.</summary>
	public int GetWidth() => width;

	/// <summary>Height in pixels.</summary>
	public int GetHeight() => height;

	/// <summary>The image file path (a key for the host's bitmap source; never opened here).</summary>
	public string Path { get; }

	/// <summary>The image file path (C++ GetPath).</summary>
	public string GetPath() => Path;

	/// <summary>Sets the bitmap, which must have the image's size.</summary>
	public void SetBitmap(Bitmap bitmap)
	{
		Check.Eq(width, bitmap.Width);
		Check.Eq(height, bitmap.Height);
		this.bitmap = bitmap;
	}

	/// <summary>The bitmap (empty unless set).</summary>
	public Bitmap GetBitmap() => bitmap;

	/// <summary>Row-major 3x3 rotation (camera from world).</summary>
	public ReadOnlySpan<float> GetR() => r;

	/// <summary>Translation (camera from world).</summary>
	public ReadOnlySpan<float> GetT() => t;

	/// <summary>Row-major 3x3 calibration matrix.</summary>
	public ReadOnlySpan<float> GetK() => k;

	/// <summary>Row-major 3x4 projection matrix K [R | T].</summary>
	public ReadOnlySpan<float> GetP() => p;

	/// <summary>The top three rows of the inverse of [P; 0 0 0 1], row-major.</summary>
	public ReadOnlySpan<float> GetInvP() => invP;

	/// <summary>The viewing direction: the third row of R.</summary>
	public ReadOnlySpan<float> GetViewingDirection() => r.AsSpan(6, 3);

	/// <summary>Rescales the image (and its bitmap, if set) uniformly.</summary>
	public void Rescale(float factor) => Rescale(factor, factor);

	/// <summary>
	/// Rescales the image to round(width * factorX) x round(height * factorY), scaling K by
	/// the actual size ratio.
	/// </summary>
	public void Rescale(float factorX, float factorY)
	{
		int newWidth = (int)MathF.Round(width * factorX, MidpointRounding.AwayFromZero);
		int newHeight = (int)MathF.Round(height * factorY, MidpointRounding.AwayFromZero);

		if (!bitmap.IsEmpty)
		{
			bitmap.Rescale(newWidth, newHeight);
		}

		float scaleX = newWidth / (float)width;
		float scaleY = newHeight / (float)height;
		k[0] *= scaleX;
		k[2] *= scaleX;
		k[4] *= scaleY;
		k[5] *= scaleY;
		MvsGeometry.ComposeProjectionMatrix(k, r, t, p);
		MvsGeometry.ComposeInverseProjectionMatrix(k, r, t, invP);

		width = newWidth;
		height = newHeight;
	}

	/// <summary>Shrinks the image to fit within maxWidth x maxHeight, keeping its aspect ratio.</summary>
	public void Downsize(int maxWidth, int maxHeight)
	{
		if (width <= maxWidth && height <= maxHeight)
		{
			return;
		}

		float factorX = maxWidth / (float)width;
		float factorY = maxHeight / (float)height;
		Rescale(MinF(factorX, factorY));
	}

	/// <summary>std::min(a, b): (b &lt; a) ? b : a.</summary>
	internal static float MinF(float a, float b) => b < a ? b : a;
}

/// <summary>
/// The free pose functions of colmap/mvs/image.h. Every matrix is a row-major float array.
/// </summary>
public static class MvsGeometry
{
	/// <summary>
	/// The pose of camera 2 relative to camera 1: R = R2 R1ᵀ, T = T2 - R T1.
	/// Port of colmap::mvs::ComputeRelativePose.
	/// </summary>
	public static void ComputeRelativePose(
		ReadOnlySpan<float> r1, ReadOnlySpan<float> t1, ReadOnlySpan<float> r2, ReadOnlySpan<float> t2, Span<float> r, Span<float> t)
	{
		Span<float> rr = stackalloc float[9];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				// (R1ᵀ)(k, j) = R1(j, k).
				rr[i * 3 + j] = r2[i * 3] * r1[j * 3] + r2[i * 3 + 1] * r1[j * 3 + 1] + r2[i * 3 + 2] * r1[j * 3 + 2];
			}
		}

		rr.CopyTo(r);
		for (int i = 0; i < 3; i++)
		{
			t[i] = t2[i] - (rr[i * 3] * t1[0] + rr[i * 3 + 1] * t1[1] + rr[i * 3 + 2] * t1[2]);
		}
	}

	/// <summary>P = K [R | T]. Port of colmap::mvs::ComposeProjectionMatrix.</summary>
	public static void ComposeProjectionMatrix(ReadOnlySpan<float> k, ReadOnlySpan<float> r, ReadOnlySpan<float> t, Span<float> p)
	{
		Span<float> rt = stackalloc float[12];
		for (int i = 0; i < 3; i++)
		{
			rt[i * 4] = r[i * 3];
			rt[i * 4 + 1] = r[i * 3 + 1];
			rt[i * 4 + 2] = r[i * 3 + 2];
			rt[i * 4 + 3] = t[i];
		}

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				p[i * 4 + j] = k[i * 3] * rt[j] + k[i * 3 + 1] * rt[4 + j] + k[i * 3 + 2] * rt[8 + j];
			}
		}
	}

	/// <summary>
	/// The top three rows of the inverse of [K [R | T]; 0 0 0 1].
	/// Port of colmap::mvs::ComposeInverseProjectionMatrix.
	/// </summary>
	public static void ComposeInverseProjectionMatrix(ReadOnlySpan<float> k, ReadOnlySpan<float> r, ReadOnlySpan<float> t, Span<float> invP)
	{
		Span<float> m = stackalloc float[16];
		ComposeProjectionMatrix(k, r, t, m);
		m[12] = 0;
		m[13] = 0;
		m[14] = 0;
		m[15] = 1;
		Span<float> inverse = stackalloc float[16];
		Invert4x4(m, inverse);
		inverse[..12].CopyTo(invP);
	}

	/// <summary>C = -Rᵀ T. Port of colmap::mvs::ComputeProjectionCenter.</summary>
	public static void ComputeProjectionCenter(ReadOnlySpan<float> r, ReadOnlySpan<float> t, Span<float> c)
	{
		for (int i = 0; i < 3; i++)
		{
			// (-Rᵀ)(i, k) = -R(k, i).
			c[i] = -r[i] * t[0] + -r[3 + i] * t[1] + -r[6 + i] * t[2];
		}
	}

	/// <summary>R = RR R, T = RR T. Port of colmap::mvs::RotatePose.</summary>
	public static void RotatePose(ReadOnlySpan<float> rr, Span<float> r, Span<float> t)
	{
		Span<float> newR = stackalloc float[9];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				newR[i * 3 + j] = rr[i * 3] * r[j] + rr[i * 3 + 1] * r[3 + j] + rr[i * 3 + 2] * r[6 + j];
			}
		}

		Span<float> newT = stackalloc float[3];
		for (int i = 0; i < 3; i++)
		{
			newT[i] = rr[i * 3] * t[0] + rr[i * 3 + 1] * t[1] + rr[i * 3 + 2] * t[2];
		}

		newR.CopyTo(r);
		newT.CopyTo(t);
	}

	/// <summary>
	/// Inverse of a row-major 4x4 float matrix by the adjugate: the 2x2 minors of the top
	/// and bottom row pairs give every cofactor (Laplace expansion), divided by the
	/// determinant. A singular matrix gives non-finite values, as Eigen's inverse does.
	/// </summary>
	private static void Invert4x4(ReadOnlySpan<float> m, Span<float> inv)
	{
		float s0 = m[0] * m[5] - m[4] * m[1];
		float s1 = m[0] * m[6] - m[4] * m[2];
		float s2 = m[0] * m[7] - m[4] * m[3];
		float s3 = m[1] * m[6] - m[5] * m[2];
		float s4 = m[1] * m[7] - m[5] * m[3];
		float s5 = m[2] * m[7] - m[6] * m[3];

		float c5 = m[10] * m[15] - m[14] * m[11];
		float c4 = m[9] * m[15] - m[13] * m[11];
		float c3 = m[9] * m[14] - m[13] * m[10];
		float c2 = m[8] * m[15] - m[12] * m[11];
		float c1 = m[8] * m[14] - m[12] * m[10];
		float c0 = m[8] * m[13] - m[12] * m[9];

		float det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
		float invDet = 1.0f / det;

		inv[0] = (m[5] * c5 - m[6] * c4 + m[7] * c3) * invDet;
		inv[1] = (-m[1] * c5 + m[2] * c4 - m[3] * c3) * invDet;
		inv[2] = (m[13] * s5 - m[14] * s4 + m[15] * s3) * invDet;
		inv[3] = (-m[9] * s5 + m[10] * s4 - m[11] * s3) * invDet;

		inv[4] = (-m[4] * c5 + m[6] * c2 - m[7] * c1) * invDet;
		inv[5] = (m[0] * c5 - m[2] * c2 + m[3] * c1) * invDet;
		inv[6] = (-m[12] * s5 + m[14] * s2 - m[15] * s1) * invDet;
		inv[7] = (m[8] * s5 - m[10] * s2 + m[11] * s1) * invDet;

		inv[8] = (m[4] * c4 - m[5] * c2 + m[7] * c0) * invDet;
		inv[9] = (-m[0] * c4 + m[1] * c2 - m[3] * c0) * invDet;
		inv[10] = (m[12] * s4 - m[13] * s2 + m[15] * s0) * invDet;
		inv[11] = (-m[8] * s4 + m[9] * s2 - m[11] * s0) * invDet;

		inv[12] = (-m[4] * c3 + m[5] * c1 - m[6] * c0) * invDet;
		inv[13] = (m[0] * c3 - m[1] * c1 + m[2] * c0) * invDet;
		inv[14] = (-m[12] * s3 + m[13] * s1 - m[14] * s0) * invDet;
		inv[15] = (m[8] * s3 - m[9] * s1 + m[10] * s0) * invDet;
	}
}
