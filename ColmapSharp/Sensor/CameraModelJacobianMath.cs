// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The shared helpers (namespace colmap::internal) of colmap/sensor/models_jacobian.h, used by
// the analytic ImgFromCamWithJac kernels in PinholeCameraModels.Jacobian.cs,
// FisheyeCameraModels.Jacobian.cs and OtherCameraModels.Jacobian.cs. The runtime dispatch
// (CameraModelImgFromCamWithJac) and CamRayFromImgJacobian are in CameraModels.cs.
// Tests: ColmapSharp.Tests/Sensor/ModelsJacobianTests.cs (models_jacobian_test.cc 1:1).
//
// Jacobians are 2xN row-major spans, as COLMAP's double* outputs are; C++'s nullptr (skip
// this Jacobian) is an empty span. Every expression keeps COLMAP's operator order, so the
// kernels are bit-identical to the C++ for the same input up to libm and FMA contraction
// (docs/CPP_DIVERGENCES.md, entry 12).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

namespace ColmapSharp.Sensor;

public static partial class CameraModelMath
{
	/// <summary>
	/// HasProjectableDepth on double, the instantiation the Jacobian kernels use.
	/// </summary>
	internal static bool HasProjectableDepth(double w, bool checkCheirality) =>
		HasProjectableDepth<Real>(w, checkCheirality);

	/// <summary>
	/// Port of internal::FisheyeProjectionWithJac. Fisheye (equidistant) projection: maps
	/// normalized coordinates (a, b) = (u/w, v/w) to the projected fisheye coordinates
	/// (uu, vv). When <paramref name="jFisheye"/> is non-empty it also receives the 2x2
	/// Jacobian d(uu, vv) / d(a, b) in row-major order; pass an empty span on the value-only
	/// path to skip that work. Mirrors BasePerspectiveFisheyeCameraModel::FisheyeFromNormal.
	/// </summary>
	internal static void FisheyeProjectionWithJac(double a, double b, out double uu, out double vv, Span<double> jFisheye)
	{
		double r2 = a * a + b * b;
		double r = Math.Sqrt(r2);
		if (r < LinearAlgebraConstants.MachineEpsilon)
		{
			// Identity in the limit r -> 0 (theta / r -> 1).
			uu = a;
			vv = b;
			if (!jFisheye.IsEmpty)
			{
				jFisheye[0] = 1.0;
				jFisheye[1] = 0.0;
				jFisheye[2] = 0.0;
				jFisheye[3] = 1.0;
			}

			return;
		}

		double theta = Math.Atan(r);
		double s = theta / r;
		uu = s * a;
		vv = s * b;
		if (!jFisheye.IsEmpty)
		{
			// With s = atan(r) / r and r = sqrt(a^2 + b^2), the Jacobian of (uu, vv) =
			// (s * a, s * b) w.r.t. (a, b) is s * I + g * outer((a, b), (a, b)), where
			// g = (ds/dr) / r = (r / (1 + r^2) - atan(r)) / r^3.
			double g = (r / (1.0 + r2) - theta) / (r2 * r);
			jFisheye[0] = s + a * a * g;
			jFisheye[1] = a * b * g;
			jFisheye[2] = a * b * g;
			jFisheye[3] = s + b * b * g;
		}
	}

	/// <summary>
	/// Port of internal::DivisionScaleWithJac. Solves the one-parameter division model's
	/// projection scale r from the camera point (u, v, w): the depth is scaled by
	/// r = 2 / (w + sqrt(w^2 - 4*k*rho2)), with rho2 = u^2 + v^2. Returns false when the
	/// point is behind the model's projection surface (negative discriminant). When
	/// <paramref name="withJac"/>, also returns the derivatives of r w.r.t. (u, v, w, k);
	/// otherwise they are 0.
	/// </summary>
	internal static bool DivisionScaleWithJac(double u, double v, double w, double k, bool withJac, out double r, out double drDu, out double drDv, out double drDw, out double drDk)
	{
		r = 0;
		drDu = 0;
		drDv = 0;
		drDw = 0;
		drDk = 0;
		double rho2 = u * u + v * v;
		double discSq = w * w - 4.0 * rho2 * k;
		if (discSq < 0.0)
		{
			return false;
		}

		double disc = Math.Sqrt(discSq);
		r = 2.0 / (w + disc);
		if (withJac)
		{
			double invDisc = 1.0 / disc;
			double rSq = r * r;
			drDu = 2.0 * rSq * k * u * invDisc;
			drDv = 2.0 * rSq * k * v * invDisc;
			drDw = -0.5 * rSq * (1.0 + w * invDisc);
			drDk = rSq * rho2 * invDisc;
		}

		return true;
	}

	/// <summary>Port of internal::MatMul2x2: out = lhs * rhs for row-major 2x2 matrices.</summary>
	internal static void MatMul2x2(ReadOnlySpan<double> lhs, ReadOnlySpan<double> rhs, Span<double> result)
	{
		result[0] = lhs[0] * rhs[0] + lhs[1] * rhs[2];
		result[1] = lhs[0] * rhs[1] + lhs[1] * rhs[3];
		result[2] = lhs[2] * rhs[0] + lhs[3] * rhs[2];
		result[3] = lhs[2] * rhs[1] + lhs[3] * rhs[3];
	}

	/// <summary>
	/// Port of internal::UvwJacFromAbJac. Given jAb (row-major 2x2) = d(x, y) / d(a, b) with
	/// a = u/w and b = v/w, computes the 2x3 Jacobian jUvw = d(x, y) / d(u, v, w) via the
	/// chain rule through (a, b) = (u/w, v/w).
	/// </summary>
	internal static void UvwJacFromAbJac(ReadOnlySpan<double> jAb, double a, double b, double invW, Span<double> jUvw)
	{
		jUvw[0] = jAb[0] * invW;
		jUvw[1] = jAb[1] * invW;
		jUvw[2] = -(jAb[0] * a + jAb[1] * b) * invW;
		jUvw[3] = jAb[2] * invW;
		jUvw[4] = jAb[3] * invW;
		jUvw[5] = -(jAb[2] * a + jAb[3] * b) * invW;
	}

	/// <summary>
	/// The fisheye models' shared tail: J_ab = diag(fx, fx, fy, fy) * (ipjd * J_fisheye)
	/// (C++ writes it out per model as MatMul2x2 then the f-scaled J_ab array), then
	/// UvwJacFromAbJac.
	/// </summary>
	internal static void FisheyeUvwJac(ReadOnlySpan<double> ipjd, ReadOnlySpan<double> jFisheye, double f1, double f2, double a, double b, double invW, Span<double> jUvw)
	{
		Span<double> m = stackalloc double[4];
		MatMul2x2(ipjd, jFisheye, m);
		ReadOnlySpan<double> jAb = [f1 * m[0], f1 * m[1], f2 * m[2], f2 * m[3]];
		UvwJacFromAbJac(jAb, a, b, invW, jUvw);
	}
}
