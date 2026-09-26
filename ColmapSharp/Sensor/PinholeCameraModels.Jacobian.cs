// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The analytic ImgFromCamWithJac kernels of colmap/sensor/models_jacobian.h for the models
// in PinholeCameraModels.cs: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV,
// FULL_OPENCV and FOV. Shared helpers are in CameraModelJacobianMath.cs; the fisheye and
// remaining models' kernels in FisheyeCameraModels.Jacobian.cs and
// OtherCameraModels.Jacobian.cs. Tests: ColmapSharp.Tests/Sensor/ModelsJacobianTests.cs.
//
// Tier A: COLMAP's operator order is kept expression by expression.

namespace ColmapSharp.Sensor;

public readonly partial struct SimplePinholeCameraModel
{
	/// <inheritdoc/>
	public static bool ImgFromCamWithJac(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, Span<double> jParams, Span<double> jUvw, bool checkCheirality = true)
	{
		x = 0;
		y = 0;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		double f = parameters[0];
		double c1 = parameters[1];
		double c2 = parameters[2];

		double invW = 1.0 / w;
		double uu = u * invW;
		double vv = v * invW;

		x = f * uu + c1;
		y = f * vv + c2;

		if (!jUvw.IsEmpty)
		{
			// J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w)
			// x = f * u / w + c1, y = f * v / w + c2
			double fInvW = f * invW;
			jUvw[0] = fInvW;
			jUvw[1] = 0.0;
			jUvw[2] = -fInvW * uu;
			jUvw[3] = 0.0;
			jUvw[4] = fInvW;
			jUvw[5] = -fInvW * vv;
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x3 matrix (row-major): d(x, y) / d(f, cx, cy)
			jParams[0] = uu;
			jParams[1] = 1.0;
			jParams[2] = 0.0;
			jParams[3] = vv;
			jParams[4] = 0.0;
			jParams[5] = 1.0;
		}

		return true;
	}
}

public readonly partial struct PinholeCameraModel
{
	/// <inheritdoc/>
	public static bool ImgFromCamWithJac(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, Span<double> jParams, Span<double> jUvw, bool checkCheirality = true)
	{
		x = 0;
		y = 0;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		double f1 = parameters[0];
		double f2 = parameters[1];
		double c1 = parameters[2];
		double c2 = parameters[3];

		double invW = 1.0 / w;
		double uu = u * invW;
		double vv = v * invW;

		x = f1 * uu + c1;
		y = f2 * vv + c2;

		if (!jUvw.IsEmpty)
		{
			// J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w)
			// x = fx * u / w + cx, y = fy * v / w + cy
			jUvw[0] = f1 * invW;
			jUvw[1] = 0.0;
			jUvw[2] = -f1 * invW * uu;
			jUvw[3] = 0.0;
			jUvw[4] = f2 * invW;
			jUvw[5] = -f2 * invW * vv;
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x4 matrix (row-major): d(x, y) / d(fx, fy, cx, cy)
			jParams[0] = uu;
			jParams[1] = 0.0;
			jParams[2] = 1.0;
			jParams[3] = 0.0;
			jParams[4] = 0.0;
			jParams[5] = vv;
			jParams[6] = 0.0;
			jParams[7] = 1.0;
		}

		return true;
	}
}

public readonly partial struct SimpleRadialCameraModel
{
	/// <inheritdoc/>
	public static bool ImgFromCamWithJac(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, Span<double> jParams, Span<double> jUvw, bool checkCheirality = true)
	{
		x = 0;
		y = 0;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		double f = parameters[0];
		double c1 = parameters[1];
		double c2 = parameters[2];
		double k = parameters[3];

		double invW = 1.0 / w;
		double uu = u * invW;
		double vv = v * invW;

		double uu2 = uu * uu;
		double vv2 = vv * vv;
		double r2 = uu2 + vv2;
		double kR2 = k * r2;
		double alpha = 1.0 + kR2;
		double xd = alpha * uu;
		double yd = alpha * vv;

		x = f * xd + c1;
		y = f * yd + c2;

		if (!jUvw.IsEmpty)
		{
			// J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w)
			//
			// x = f * alpha * uu + c1, y = f * alpha * vv + c2
			// where alpha = 1 + k * r2, r2 = uu^2 + vv^2, uu = u/w, vv = v/w
			//
			// Using chain rule:
			// dx/du = f/w * (alpha + 2*k*uu^2)
			// dx/dv = f/w * 2*k*uu*vv
			// dx/dw = -f*uu/w * (1 + 3*k*r2)
			// dy/du = f/w * 2*k*uu*vv
			// dy/dv = f/w * (alpha + 2*k*vv^2)
			// dy/dw = -f*vv/w * (1 + 3*k*r2)
			double twoK = 2.0 * k;
			double fInvW = f * invW;
			double beta = 1.0 + 3.0 * kR2;
			double twoKUuVv = twoK * uu * vv;

			jUvw[0] = fInvW * (alpha + twoK * uu2);
			jUvw[1] = fInvW * twoKUuVv;
			jUvw[2] = -fInvW * uu * beta;
			jUvw[3] = fInvW * twoKUuVv;
			jUvw[4] = fInvW * (alpha + twoK * vv2);
			jUvw[5] = -fInvW * vv * beta;
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x4 matrix (row-major): d(x, y) / d(f, cx, cy, k)
			//
			// x = f * alpha * uu + cx, y = f * alpha * vv + cy
			//
			// dx/df = alpha * uu, dx/dcx = 1, dx/dcy = 0, dx/dk = f * uu * r2
			// dy/df = alpha * vv, dy/dcx = 0, dy/dcy = 1, dy/dk = f * vv * r2
			jParams[0] = xd;
			jParams[1] = 1.0;
			jParams[2] = 0.0;
			jParams[3] = f * uu * r2;
			jParams[4] = yd;
			jParams[5] = 0.0;
			jParams[6] = 1.0;
			jParams[7] = f * vv * r2;
		}

		return true;
	}
}

public readonly partial struct RadialCameraModel
{
	/// <inheritdoc/>
	public static bool ImgFromCamWithJac(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, Span<double> jParams, Span<double> jUvw, bool checkCheirality = true)
	{
		x = 0;
		y = 0;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		double f = parameters[0];
		double c1 = parameters[1];
		double c2 = parameters[2];
		double k1 = parameters[3];
		double k2 = parameters[4];

		double invW = 1.0 / w;
		double uu = u * invW;
		double vv = v * invW;

		double uu2 = uu * uu;
		double vv2 = vv * vv;
		double r2 = uu2 + vv2;
		double r4 = r2 * r2;
		double radial = k1 * r2 + k2 * r4;
		double xd = uu * (1.0 + radial);
		double yd = vv * (1.0 + radial);

		x = f * xd + c1;
		y = f * yd + c2;

		if (!jUvw.IsEmpty)
		{
			// J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w).
			// With xd = uu * (1 + radial), yd = vv * (1 + radial),
			// radial = k1 * r2 + k2 * r2^2, r2 = uu^2 + vv^2, the distortion Jacobian
			// in normalized coordinates (uu, vv) is:
			//   d(xd)/d(uu) = 1 + radial + 2 * uu^2 * d_radial_d_r2
			//   d(xd)/d(vv) = 2 * uu * vv * d_radial_d_r2
			//   d(yd)/d(uu) = 2 * uu * vv * d_radial_d_r2
			//   d(yd)/d(vv) = 1 + radial + 2 * vv^2 * d_radial_d_r2
			// where d_radial_d_r2 = k1 + 2 * k2 * r2. The chain rule through
			// (uu, vv) = (u/w, v/w) yields the columns below.
			double dRadialDR2 = k1 + 2.0 * k2 * r2;
			double cross = 2.0 * uu * vv * dRadialDR2;
			double a00 = f * (1.0 + radial + 2.0 * uu2 * dRadialDR2);
			double a01 = f * cross;
			double a10 = f * cross;
			double a11 = f * (1.0 + radial + 2.0 * vv2 * dRadialDR2);

			jUvw[0] = a00 * invW;
			jUvw[1] = a01 * invW;
			jUvw[2] = -(a00 * uu + a01 * vv) * invW;
			jUvw[3] = a10 * invW;
			jUvw[4] = a11 * invW;
			jUvw[5] = -(a10 * uu + a11 * vv) * invW;
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x5 matrix (row-major): d(x, y) / d(f, cx, cy, k1, k2)
			jParams[0] = xd;
			jParams[1] = 1.0;
			jParams[2] = 0.0;
			jParams[3] = f * uu * r2;
			jParams[4] = f * uu * r4;
			jParams[5] = yd;
			jParams[6] = 0.0;
			jParams[7] = 1.0;
			jParams[8] = f * vv * r2;
			jParams[9] = f * vv * r4;
		}

		return true;
	}
}

public readonly partial struct OpenCVCameraModel
{
	/// <inheritdoc/>
	public static bool ImgFromCamWithJac(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, Span<double> jParams, Span<double> jUvw, bool checkCheirality = true)
	{
		x = 0;
		y = 0;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		double f1 = parameters[0];
		double f2 = parameters[1];
		double c1 = parameters[2];
		double c2 = parameters[3];
		double k1 = parameters[4];
		double k2 = parameters[5];
		double p1 = parameters[6];
		double p2 = parameters[7];

		double invW = 1.0 / w;
		double uu = u * invW;
		double vv = v * invW;

		double uu2 = uu * uu;
		double vv2 = vv * vv;
		double uv = uu * vv;
		double r2 = uu2 + vv2;
		double r4 = r2 * r2;
		double radial = k1 * r2 + k2 * r4;

		double du = uu * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * uu2);
		double dv = vv * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * vv2);
		double xd = uu + du;
		double yd = vv + dv;

		x = f1 * xd + c1;
		y = f2 * yd + c2;

		if (!jUvw.IsEmpty)
		{
			// J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w).
			// Partial derivatives of the OpenCV distortion (radial + tangential) in
			// normalized coordinates (uu, vv), with d_radial_d_r2 = k1 + 2 * k2 * r2:
			//   d(du)/d(uu) = radial + 2*uu^2*d_radial_d_r2 + 2*p1*vv + 6*p2*uu
			//   d(du)/d(vv) = 2*uu*vv*d_radial_d_r2 + 2*p1*uu + 2*p2*vv
			//   d(dv)/d(uu) = 2*uu*vv*d_radial_d_r2 + 2*p2*vv + 2*p1*uu
			//   d(dv)/d(vv) = radial + 2*vv^2*d_radial_d_r2 + 2*p2*uu + 6*p1*vv
			// The chain rule through (uu, vv) = (u/w, v/w) yields the columns below.
			double dRadialDR2 = k1 + 2.0 * k2 * r2;
			double cross = 2.0 * uv * dRadialDR2;
			double duDuu = radial + 2.0 * uu2 * dRadialDR2 + 2.0 * p1 * vv + 6.0 * p2 * uu;
			double duDvv = cross + 2.0 * p1 * uu + 2.0 * p2 * vv;
			double dvDuu = cross + 2.0 * p2 * vv + 2.0 * p1 * uu;
			double dvDvv = radial + 2.0 * vv2 * dRadialDR2 + 2.0 * p2 * uu + 6.0 * p1 * vv;

			double a00 = f1 * (1.0 + duDuu);
			double a01 = f1 * duDvv;
			double a10 = f2 * dvDuu;
			double a11 = f2 * (1.0 + dvDvv);

			jUvw[0] = a00 * invW;
			jUvw[1] = a01 * invW;
			jUvw[2] = -(a00 * uu + a01 * vv) * invW;
			jUvw[3] = a10 * invW;
			jUvw[4] = a11 * invW;
			jUvw[5] = -(a10 * uu + a11 * vv) * invW;
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x8 matrix (row-major):
			//   d(x, y) / d(fx, fy, cx, cy, k1, k2, p1, p2)
			jParams[0] = xd;
			jParams[1] = 0.0;
			jParams[2] = 1.0;
			jParams[3] = 0.0;
			jParams[4] = f1 * uu * r2;
			jParams[5] = f1 * uu * r4;
			jParams[6] = f1 * 2.0 * uv;
			jParams[7] = f1 * (r2 + 2.0 * uu2);
			jParams[8] = 0.0;
			jParams[9] = yd;
			jParams[10] = 0.0;
			jParams[11] = 1.0;
			jParams[12] = f2 * vv * r2;
			jParams[13] = f2 * vv * r4;
			jParams[14] = f2 * (r2 + 2.0 * vv2);
			jParams[15] = f2 * 2.0 * uv;
		}

		return true;
	}
}

public readonly partial struct FullOpenCVCameraModel
{
	/// <inheritdoc/>
	public static bool ImgFromCamWithJac(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, Span<double> jParams, Span<double> jUvw, bool checkCheirality = true)
	{
		x = 0;
		y = 0;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		double f1 = parameters[0];
		double f2 = parameters[1];
		double c1 = parameters[2];
		double c2 = parameters[3];
		double k1 = parameters[4];
		double k2 = parameters[5];
		double p1 = parameters[6];
		double p2 = parameters[7];
		double k3 = parameters[8];
		double k4 = parameters[9];
		double k5 = parameters[10];
		double k6 = parameters[11];

		double invW = 1.0 / w;
		double uu = u * invW;
		double vv = v * invW;

		double uu2 = uu * uu;
		double vv2 = vv * vv;
		double uv = uu * vv;
		double r2 = uu2 + vv2;
		double r4 = r2 * r2;
		double r6 = r4 * r2;

		// Rational radial term: radial = num / den.
		double num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
		double den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
		double invDen = 1.0 / den;
		double radial = num * invDen;

		double xd = uu * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * uu2);
		double yd = vv * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * vv2);

		x = f1 * xd + c1;
		y = f2 * yd + c2;

		if (!jUvw.IsEmpty)
		{
			// J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w).
			// With xd = uu * radial + tangential_x, yd = vv * radial + tangential_y,
			// and radial = num / den, the derivative of the rational radial term is
			//   d(radial)/d(r2) = (num' * den - num * den') / den^2
			// with num' = k1 + 2*k2*r2 + 3*k3*r4, den' = k4 + 2*k5*r2 + 3*k6*r4.
			// The distortion Jacobian in normalized coordinates (uu, vv) is:
			//   d(xd)/d(uu) = radial + 2*uu^2*d_radial_d_r2 + 2*p1*vv + 6*p2*uu
			//   d(xd)/d(vv) = 2*uu*vv*d_radial_d_r2 + 2*p1*uu + 2*p2*vv
			//   d(yd)/d(uu) = 2*uu*vv*d_radial_d_r2 + 2*p2*vv + 2*p1*uu
			//   d(yd)/d(vv) = radial + 2*vv^2*d_radial_d_r2 + 2*p2*uu + 6*p1*vv
			// The chain rule through (uu, vv) = (u/w, v/w) yields the columns below.
			double numPrime = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
			double denPrime = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4;
			double dRadialDR2 = (numPrime * den - num * denPrime) * invDen * invDen;
			double cross = 2.0 * uv * dRadialDR2;
			double xdDuu = radial + 2.0 * uu2 * dRadialDR2 + 2.0 * p1 * vv + 6.0 * p2 * uu;
			double xdDvv = cross + 2.0 * p1 * uu + 2.0 * p2 * vv;
			double ydDuu = cross + 2.0 * p2 * vv + 2.0 * p1 * uu;
			double ydDvv = radial + 2.0 * vv2 * dRadialDR2 + 2.0 * p2 * uu + 6.0 * p1 * vv;

			double a00 = f1 * xdDuu;
			double a01 = f1 * xdDvv;
			double a10 = f2 * ydDuu;
			double a11 = f2 * ydDvv;

			jUvw[0] = a00 * invW;
			jUvw[1] = a01 * invW;
			jUvw[2] = -(a00 * uu + a01 * vv) * invW;
			jUvw[3] = a10 * invW;
			jUvw[4] = a11 * invW;
			jUvw[5] = -(a10 * uu + a11 * vv) * invW;
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x12 matrix (row-major):
			//   d(x, y) / d(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
			// The numerator coefficients enter as d(radial)/d(k_i) = r^(2i) / den; the
			// denominator coefficients as d(radial)/d(k_j) = -num * r^(2j) / den^2.
			double numK1 = r2 * invDen;
			double numK2 = r4 * invDen;
			double numK3 = r6 * invDen;
			double negNumInvDen2 = -num * invDen * invDen;
			double denK4 = negNumInvDen2 * r2;
			double denK5 = negNumInvDen2 * r4;
			double denK6 = negNumInvDen2 * r6;

			jParams[0] = xd;
			jParams[1] = 0.0;
			jParams[2] = 1.0;
			jParams[3] = 0.0;
			jParams[4] = f1 * uu * numK1;
			jParams[5] = f1 * uu * numK2;
			jParams[6] = f1 * 2.0 * uv;
			jParams[7] = f1 * (r2 + 2.0 * uu2);
			jParams[8] = f1 * uu * numK3;
			jParams[9] = f1 * uu * denK4;
			jParams[10] = f1 * uu * denK5;
			jParams[11] = f1 * uu * denK6;
			jParams[12] = 0.0;
			jParams[13] = yd;
			jParams[14] = 0.0;
			jParams[15] = 1.0;
			jParams[16] = f2 * vv * numK1;
			jParams[17] = f2 * vv * numK2;
			jParams[18] = f2 * (r2 + 2.0 * vv2);
			jParams[19] = f2 * 2.0 * uv;
			jParams[20] = f2 * vv * numK3;
			jParams[21] = f2 * vv * denK4;
			jParams[22] = f2 * vv * denK5;
			jParams[23] = f2 * vv * denK6;
		}

		return true;
	}
}

public readonly partial struct FOVCameraModel
{
	/// <inheritdoc/>
	public static bool ImgFromCamWithJac(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, Span<double> jParams, Span<double> jUvw, bool checkCheirality = true)
	{
		x = 0;
		y = 0;
		if (!CameraModelMath.HasProjectableDepth(w, checkCheirality))
		{
			return false;
		}

		double f1 = parameters[0];
		double f2 = parameters[1];
		double c1 = parameters[2];
		double c2 = parameters[3];
		double omega = parameters[4];

		double invW = 1.0 / w;
		double a = u * invW;
		double b = v * invW;

		double radius2 = a * a + b * b;
		double omega2 = omega * omega;

		// Chosen to match FOVCameraModel::Distortion.
		const double kEpsilon = 1e-4;

		// The distortion scales (a, b) by a radially symmetric factor. We compute the
		// factor and its partials factor_r = d(factor)/d(radius2) and factor_omega =
		// d(factor)/d(omega), matching whichever branch FOVCameraModel::Distortion
		// selects so that the analytic Jacobian agrees with autodiff everywhere.
		double factor;
		double factorR;
		double factorOmega;
		if (omega2 < kEpsilon)
		{
			factor = (omega2 * radius2) / 3.0 - omega2 / 12.0 + 1.0;
			factorR = omega2 / 3.0;
			factorOmega = 2.0 * omega * radius2 / 3.0 - omega / 6.0;
		}
		else if (radius2 < kEpsilon)
		{
			double t = Math.Tan(omega / 2.0);
			double t2 = t * t;
			// Q = t * (4 * t^2 * radius2 - 3), factor = -2 * Q / (3 * omega).
			double q = t * (4.0 * t2 * radius2 - 3.0);
			factor = -2.0 * q / (3.0 * omega);
			factorR = -8.0 * t * t2 / (3.0 * omega);
			double dtDomega = 0.5 * (1.0 + t2);
			double qOmega = dtDomega * (12.0 * t2 * radius2 - 3.0);
			factorOmega = -2.0 / (3.0 * omega2) * (qOmega * omega - q);
		}
		else
		{
			double radius = Math.Sqrt(radius2);
			double t = Math.Tan(omega / 2.0);
			double arg = 2.0 * radius * t;
			double atanArg = Math.Atan(arg);
			double denomArg = 1.0 + arg * arg;
			// denom_arg divides both derivative numerators; hoist its reciprocal.
			double invDenomArg = 1.0 / denomArg;
			factor = atanArg / (radius * omega);
			factorR = (2.0 * t * radius * invDenomArg - atanArg) / (2.0 * radius2 * radius * omega);
			factorOmega = (radius * omega * (1.0 + t * t) * invDenomArg - atanArg) / (radius * omega2);
		}

		double du = a * factor;
		double dv = b * factor;

		x = f1 * du + c1;
		y = f2 * dv + c2;

		if (!jUvw.IsEmpty)
		{
			// d(du, dv) / d(a, b) with du = a * factor, dv = b * factor and factor a
			// function of radius2 = a^2 + b^2.
			double cross = 2.0 * a * b * factorR;
			double da00 = factor + 2.0 * a * a * factorR;
			double da11 = factor + 2.0 * b * b * factorR;
			ReadOnlySpan<double> jAb = [f1 * da00, f1 * cross, f2 * cross, f2 * da11];
			CameraModelMath.UvwJacFromAbJac(jAb, a, b, invW, jUvw);
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x5 matrix (row-major): d(x, y) / d(fx, fy, cx, cy, omega).
			jParams[0] = du;
			jParams[1] = 0.0;
			jParams[2] = 1.0;
			jParams[3] = 0.0;
			jParams[4] = f1 * a * factorOmega;
			jParams[5] = 0.0;
			jParams[6] = dv;
			jParams[7] = 0.0;
			jParams[8] = 1.0;
			jParams[9] = f2 * b * factorOmega;
		}

		return true;
	}
}
