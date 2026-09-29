// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PatchMatchKernel.Geometry: the __device__ geometry helpers of patch_match_cuda.cu - the
// random and perturbed depth/normal hypotheses, depth propagation along a column, viewing
// angles, the plane-induced homography, the geometric-consistency cost and RotateNormalMap's
// vector rotation - run on the CPU. PatchMatchKernel.Photometric.cs holds the NCC and
// ComputeInitialCost, PatchMatchLikelihood.cs the selection-probability model;
// PatchMatchTransforms.cs builds the pose tables these read. Tests:
// ColmapSharp.Tests/Mvs/PatchMatchKernelTests.cs (C#-only; COLMAP tests them only through
// the CUDA kernel).
//
// Float math in COLMAP's operation order with MathF and no fused multiply-add
// (divergence 96): CUDA's rsqrt becomes 1 / MathF.Sqrt, and CUDA's
// float min/max (fminf/fmaxf, which ignore a NaN operand) become CudaMin/CudaMax. Random draws
// come from PatchMatchRandom (entry 86).
//
// Translation notes:
// - COLMAP keeps the reference calibration of the current rotation in __constant__ memory
//   (ref_K, ref_inv_K); here it travels as a PatchMatchFrame value.
// - tex2D(poses_texture, i, image_idx) is poses[image_idx * NumTformParams + i].
// - 3-vectors are Span<float> of length 3, so hot loops can keep them on the stack.

namespace ColmapSharp.Mvs;

/// <summary>
/// The reference calibration of one rotation of the reference image: ref_K as
/// {fx, cx, fy, cy} and ref_inv_K as {1/fx, -cx/fx, 1/fy, -cy/fy}.
/// </summary>
public readonly struct PatchMatchFrame
{
	/// <summary>fx, cx, fy, cy (ref_K).</summary>
	public readonly float K0, K1, K2, K3;

	/// <summary>1/fx, -cx/fx, 1/fy, -cy/fy (ref_inv_K).</summary>
	public readonly float InvK0, InvK1, InvK2, InvK3;

	/// <summary>The frame of rotation <paramref name="rotation"/> of <paramref name="transforms"/>.</summary>
	public PatchMatchFrame(PatchMatchTransforms transforms, int rotation)
	{
		ReadOnlySpan<float> k = transforms.RefK(rotation);
		ReadOnlySpan<float> invK = transforms.RefInvK(rotation);
		(K0, K1, K2, K3) = (k[0], k[1], k[2], k[3]);
		(InvK0, InvK1, InvK2, InvK3) = (invK[0], invK[1], invK[2], invK[3]);
	}
}

/// <summary>The device functions of patch_match_cuda.cu, on the CPU.</summary>
public static partial class PatchMatchKernel
{
	/// <summary>result = mat * vec for a row-major 3x3 matrix.</summary>
	public static void Mat33DotVec3(ReadOnlySpan<float> mat, ReadOnlySpan<float> vec, Span<float> result)
	{
		result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
		result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
		result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
	}

	/// <summary>The dehomogenized mat * (vec[0], vec[1], 1).</summary>
	public static (float X, float Y) Mat33DotVec3Homogeneous(ReadOnlySpan<float> mat, float x, float y)
	{
		float invZ = 1.0f / (mat[6] * x + mat[7] * y + mat[8]);
		return (invZ * (mat[0] * x + mat[1] * y + mat[2]), invZ * (mat[3] * x + mat[4] * y + mat[5]));
	}

	/// <summary>vec1 · vec2 for 3-vectors.</summary>
	public static float DotProduct3(ReadOnlySpan<float> vec1, ReadOnlySpan<float> vec2) =>
		vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];

	/// <summary>A uniform depth in (depthMin, depthMax].</summary>
	public static float GenerateRandomDepth(float depthMin, float depthMax, ref PatchMatchRandom random) =>
		random.NextUniform() * (depthMax - depthMin) + depthMin;

	/// <summary>
	/// A uniformly distributed unit normal facing the camera at pixel (row, col). Unbiased
	/// sampling of normal, according to George Marsaglia, "Choosing a Point from the Surface
	/// of a Sphere", 1972.
	/// </summary>
	public static void GenerateRandomNormal(in PatchMatchFrame frame, int row, int col, ref PatchMatchRandom random, Span<float> normal)
	{
		float v1 = 0.0f;
		float v2 = 0.0f;
		float s = 2.0f;
		while (s >= 1.0f)
		{
			v1 = 2.0f * random.NextUniform() - 1.0f;
			v2 = 2.0f * random.NextUniform() - 1.0f;
			s = v1 * v1 + v2 * v2;
		}

		float sNorm = MathF.Sqrt(1.0f - s);
		normal[0] = 2.0f * v1 * sNorm;
		normal[1] = 2.0f * v2 * sNorm;
		normal[2] = 1.0f - 2.0f * s;

		// Make sure normal is looking away from camera.
		if (DotViewRay(frame, row, col, normal) > 0)
		{
			normal[0] = -normal[0];
			normal[1] = -normal[1];
			normal[2] = -normal[2];
		}
	}

	/// <summary>A uniform depth within ±perturbation (relative) of <paramref name="depth"/>.</summary>
	public static float PerturbDepth(float perturbation, float depth, ref PatchMatchRandom random)
	{
		float depthMin = (1.0f - perturbation) * depth;
		float depthMax = (1.0f + perturbation) * depth;
		return GenerateRandomDepth(depthMin, depthMax, ref random);
	}

	/// <summary>
	/// <paramref name="normal"/> rotated by random angles in ±perturbation/2 about each axis.
	/// If the result faces away from the camera, retries with half the perturbation, up to
	/// three times, and then keeps the input normal.
	/// </summary>
	public static void PerturbNormal(
		in PatchMatchFrame frame,
		int row,
		int col,
		float perturbation,
		ReadOnlySpan<float> normal,
		ref PatchMatchRandom random,
		Span<float> perturbedNormal,
		int numTrials = 0)
	{
		// Perturbation rotation angles.
		float a1 = (random.NextUniform() - 0.5f) * perturbation;
		float a2 = (random.NextUniform() - 0.5f) * perturbation;
		float a3 = (random.NextUniform() - 0.5f) * perturbation;

		float sinA1 = MathF.Sin(a1);
		float sinA2 = MathF.Sin(a2);
		float sinA3 = MathF.Sin(a3);
		float cosA1 = MathF.Cos(a1);
		float cosA2 = MathF.Cos(a2);
		float cosA3 = MathF.Cos(a3);

		// R = Rx * Ry * Rz
		Span<float> r = stackalloc float[9];
		r[0] = cosA2 * cosA3;
		r[1] = -cosA2 * sinA3;
		r[2] = sinA2;
		r[3] = cosA1 * sinA3 + cosA3 * sinA1 * sinA2;
		r[4] = cosA1 * cosA3 - sinA1 * sinA2 * sinA3;
		r[5] = -cosA2 * sinA1;
		r[6] = sinA1 * sinA3 - cosA1 * cosA3 * sinA2;
		r[7] = cosA3 * sinA1 + cosA1 * sinA2 * sinA3;
		r[8] = cosA1 * cosA2;

		// Perturb the normal vector.
		Mat33DotVec3(r, normal, perturbedNormal);

		// Make sure the perturbed normal is still looking in the same direction as the
		// viewing direction, otherwise try again but with smaller perturbation.
		if (DotViewRay(frame, row, col, perturbedNormal) >= 0.0f)
		{
			const int MaxNumTrials = 3;
			if (numTrials < MaxNumTrials)
			{
				PerturbNormal(frame, row, col, 0.5f * perturbation, normal, ref random, perturbedNormal, numTrials + 1);
				return;
			}

			perturbedNormal[0] = normal[0];
			perturbedNormal[1] = normal[1];
			perturbedNormal[2] = normal[2];
			return;
		}

		// Make sure normal has unit norm.
		float invNorm = 1.0f / MathF.Sqrt(DotProduct3(perturbedNormal, perturbedNormal));
		perturbedNormal[0] *= invNorm;
		perturbedNormal[1] *= invNorm;
		perturbedNormal[2] *= invNorm;
	}

	/// <summary>The point at <paramref name="depth"/> along the viewing ray of pixel (row, col).</summary>
	public static void ComputePointAtDepth(in PatchMatchFrame frame, float row, float col, float depth, Span<float> point)
	{
		point[0] = depth * (frame.InvK0 * col + frame.InvK1);
		point[1] = depth * (frame.InvK2 * row + frame.InvK3);
		point[2] = depth;
	}

	/// <summary>
	/// Transfer depth on plane from viewing ray at row1 to row2. The returned depth is the
	/// intersection of the viewing ray through row2 with the plane at row1 defined by the
	/// given depth and normal (in the column's y-z plane); depth1 when the two are nearly
	/// parallel.
	/// </summary>
	public static float PropagateDepth(in PatchMatchFrame frame, float depth1, ReadOnlySpan<float> normal1, float row1, float row2)
	{
		// Point along first viewing ray.
		float x1 = depth1 * (frame.InvK2 * row1 + frame.InvK3);
		float y1 = depth1;

		// Point on plane defined by point along first viewing ray and plane normal1.
		float x2 = x1 + normal1[2];
		float y2 = y1 - normal1[1];

		// Point on second viewing ray (whose origin is (0, 0)); its y is 1.
		float x4 = frame.InvK2 * row2 + frame.InvK3;

		// Intersection of the lines ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4)).
		float denom = x2 - x1 + x4 * (y1 - y2);
		const float Eps = 1e-5f;
		if (MathF.Abs(denom) < Eps)
		{
			return depth1;
		}

		float nom = y1 * x2 - x1 * y2;
		return nom / denom;
	}

	/// <summary>
	/// The cosines of the triangulation angle between the reference and source image
	/// <paramref name="imageIdx"/> at <paramref name="point"/>, and of the incident angle
	/// between the source's viewing direction and <paramref name="normal"/>.
	/// </summary>
	public static (float CosTriangulationAngle, float CosIncidentAngle) ComputeViewingAngles(
		ReadOnlySpan<float> poses, ReadOnlySpan<float> point, ReadOnlySpan<float> normal, int imageIdx)
	{
		// Projection center of source image.
		ReadOnlySpan<float> c = poses.Slice(imageIdx * PatchMatchTransforms.NumTformParams + PatchMatchTransforms.COffset, 3);

		// Ray from point to camera.
		Span<float> sx = [c[0] - point[0], c[1] - point[1], c[2] - point[2]];

		// Length of ray from reference image to point.
		float rxInvNorm = 1.0f / MathF.Sqrt(DotProduct3(point, point));

		// Length of ray from point to source image.
		float sxInvNorm = 1.0f / MathF.Sqrt(DotProduct3(sx, sx));

		float cosIncidentAngle = DotProduct3(sx, normal) * sxInvNorm;
		float cosTriangulationAngle = -DotProduct3(sx, point) * rxInvNorm * sxInvNorm;
		return (cosTriangulationAngle, cosIncidentAngle);
	}

	/// <summary>
	/// The homography H = K * (R - T * n' / d) * Kref^-1 from the reference image to source
	/// image <paramref name="imageIdx"/> induced by the plane through the point at
	/// <paramref name="depth"/> on pixel (row, col) with <paramref name="normal"/>
	/// (row-major, into <paramref name="h"/>).
	/// </summary>
	public static void ComposeHomography(
		ReadOnlySpan<float> poses, in PatchMatchFrame frame, int imageIdx, int row, int col, float depth, ReadOnlySpan<float> normal, Span<float> h)
	{
		ReadOnlySpan<float> pose = poses.Slice(imageIdx * PatchMatchTransforms.NumTformParams, PatchMatchTransforms.NumTformParams);

		// Calibration of source image.
		ReadOnlySpan<float> k = pose.Slice(PatchMatchTransforms.KOffset, 4);

		// Relative rotation between reference and source image.
		ReadOnlySpan<float> r = pose.Slice(PatchMatchTransforms.ROffset, 9);

		// Relative translation between reference and source image.
		ReadOnlySpan<float> t = pose.Slice(PatchMatchTransforms.TOffset, 3);

		// Distance to the plane.
		float dist = depth * (normal[0] * (frame.InvK0 * col + frame.InvK1) + normal[1] * (frame.InvK2 * row + frame.InvK3) + normal[2]);
		float invDist = 1.0f / dist;

		float invDistN0 = invDist * normal[0];
		float invDistN1 = invDist * normal[1];
		float invDistN2 = invDist * normal[2];

		h[0] = frame.InvK0 * (k[0] * (r[0] + invDistN0 * t[0]) + k[1] * (r[6] + invDistN0 * t[2]));
		h[1] = frame.InvK2 * (k[0] * (r[1] + invDistN1 * t[0]) + k[1] * (r[7] + invDistN1 * t[2]));
		h[2] = k[0] * (r[2] + invDistN2 * t[0]) + k[1] * (r[8] + invDistN2 * t[2])
			+ frame.InvK1 * (k[0] * (r[0] + invDistN0 * t[0]) + k[1] * (r[6] + invDistN0 * t[2]))
			+ frame.InvK3 * (k[0] * (r[1] + invDistN1 * t[0]) + k[1] * (r[7] + invDistN1 * t[2]));
		h[3] = frame.InvK0 * (k[2] * (r[3] + invDistN0 * t[1]) + k[3] * (r[6] + invDistN0 * t[2]));
		h[4] = frame.InvK2 * (k[2] * (r[4] + invDistN1 * t[1]) + k[3] * (r[7] + invDistN1 * t[2]));
		h[5] = k[2] * (r[5] + invDistN2 * t[1]) + k[3] * (r[8] + invDistN2 * t[2])
			+ frame.InvK1 * (k[2] * (r[3] + invDistN0 * t[1]) + k[3] * (r[6] + invDistN0 * t[2]))
			+ frame.InvK3 * (k[2] * (r[4] + invDistN1 * t[1]) + k[3] * (r[7] + invDistN1 * t[2]));
		h[6] = frame.InvK0 * (r[6] + invDistN0 * t[2]);
		h[7] = frame.InvK2 * (r[7] + invDistN1 * t[2]);
		h[8] = r[8] + frame.InvK1 * (r[6] + invDistN0 * t[2]) + frame.InvK3 * (r[7] + invDistN1 * t[2]) + invDistN2 * t[2];
	}

	/// <summary>
	/// The forward-backward reprojection error, capped at <paramref name="maxCost"/>, of
	/// the point at <paramref name="depth"/> on pixel (row, col) through source image
	/// <paramref name="imageIdx"/>'s depth map; <paramref name="maxCost"/> when the point
	/// projects outside the source depth map (depth 0).
	/// </summary>
	public static float ComputeGeomConsistencyCost(
		ReadOnlySpan<float> poses,
		PatchMatchSourceDepthMaps srcDepthMaps,
		in PatchMatchFrame frame,
		float row,
		float col,
		float depth,
		int imageIdx,
		float maxCost)
	{
		ReadOnlySpan<float> pose = poses.Slice(imageIdx * PatchMatchTransforms.NumTformParams, PatchMatchTransforms.NumTformParams);

		// Extract projection matrices for source image.
		ReadOnlySpan<float> p = pose.Slice(PatchMatchTransforms.POffset, 12);
		ReadOnlySpan<float> invP = pose.Slice(PatchMatchTransforms.InvPOffset, 12);

		// Project point in reference image to world.
		Span<float> forwardPoint = stackalloc float[3];
		ComputePointAtDepth(frame, row, col, depth, forwardPoint);

		// Project world point to source image.
		float invForwardZ = 1.0f / (p[8] * forwardPoint[0] + p[9] * forwardPoint[1] + p[10] * forwardPoint[2] + p[11]);
		float srcCol = invForwardZ * (p[0] * forwardPoint[0] + p[1] * forwardPoint[1] + p[2] * forwardPoint[2] + p[3]);
		float srcRow = invForwardZ * (p[4] * forwardPoint[0] + p[5] * forwardPoint[1] + p[6] * forwardPoint[2] + p[7]);

		// Extract depth in source image.
		float srcDepth = srcDepthMaps.Sample(srcCol + 0.5f, srcRow + 0.5f, imageIdx);

		// Projection outside of source image.
		if (srcDepth == 0.0f)
		{
			return maxCost;
		}

		// Project point in source image to world.
		srcCol *= srcDepth;
		srcRow *= srcDepth;
		float backwardPointX = invP[0] * srcCol + invP[1] * srcRow + invP[2] * srcDepth + invP[3];
		float backwardPointY = invP[4] * srcCol + invP[5] * srcRow + invP[6] * srcDepth + invP[7];
		float backwardPointZ = invP[8] * srcCol + invP[9] * srcRow + invP[10] * srcDepth + invP[11];
		float invBackwardPointZ = 1.0f / backwardPointZ;

		// Project world point back to reference image.
		float backwardCol = invBackwardPointZ * (frame.K0 * backwardPointX + frame.K1 * backwardPointZ);
		float backwardRow = invBackwardPointZ * (frame.K2 * backwardPointY + frame.K3 * backwardPointZ);

		// Return truncated reprojection error between original observation and the
		// forward-backward projected observation.
		float diffCol = col - backwardCol;
		float diffRow = row - backwardRow;
		return CudaMin(maxCost, MathF.Sqrt(diffCol * diffCol + diffRow * diffRow));
	}

	/// <summary>
	/// Rotates every normal of a 3-slice normal map by 90 degrees counter-clockwise about
	/// the z-axis, (x, y, z) to (y, -x, z), in place; the map itself is then rotated with
	/// Mat.Rotate. Port of the RotateNormalMap kernel.
	/// </summary>
	public static void RotateNormalMap(Mat<float> normalMap, int numThreads = -1)
	{
		Util.Check.Eq(normalMap.GetDepth(), 3);
		float[] data = normalMap.Data;
		int planeSize = normalMap.GetWidth() * normalMap.GetHeight();
		int width = normalMap.GetWidth();
		Parallel.For(0, normalMap.GetHeight(), Mat<float>.ParallelOptionsFor(numThreads), row =>
		{
			for (int i = row * width; i < (row + 1) * width; ++i)
			{
				float x = data[i];
				data[i] = data[planeSize + i];
				data[planeSize + i] = -x;
			}
		});
	}

	/// <summary>
	/// CUDA's min(float, float), i.e. fminf: the smaller operand, or the other one when one
	/// is NaN (MathF.Min would return NaN). The kernel relies on it to turn degenerate
	/// geometry into a bounded cost.
	/// </summary>
	public static float CudaMin(float x, float y)
	{
		if (float.IsNaN(x))
		{
			return y;
		}

		if (float.IsNaN(y))
		{
			return x;
		}

		return MathF.Min(x, y);
	}

	/// <summary>CUDA's max(float, float), i.e. fmaxf: the larger operand, ignoring a NaN one.</summary>
	public static float CudaMax(float x, float y)
	{
		if (float.IsNaN(x))
		{
			return y;
		}

		if (float.IsNaN(y))
		{
			return x;
		}

		return MathF.Max(x, y);
	}

	// normal · (ref_inv_K ray of pixel (row, col)).
	private static float DotViewRay(in PatchMatchFrame frame, int row, int col, ReadOnlySpan<float> normal)
	{
		float rayX = frame.InvK0 * col + frame.InvK1;
		float rayY = frame.InvK2 * row + frame.InvK3;
		return normal[0] * rayX + normal[1] * rayY + normal[2] * 1.0f;
	}
}
