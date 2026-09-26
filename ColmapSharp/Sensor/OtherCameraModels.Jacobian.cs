// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The analytic ImgFromCamWithJac kernels of colmap/sensor/models_jacobian.h for the models
// in OtherCameraModels.cs: SIMPLE_DIVISION, DIVISION, EUCM and EQUIRECTANGULAR. The
// division models share DivisionScaleWithJac (CameraModelJacobianMath.cs).
// Tests: ColmapSharp.Tests/Sensor/ModelsJacobianTests.cs.
//
// Tier A: COLMAP's operator order is kept expression by expression.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Sensor;

public readonly partial struct SimpleDivisionCameraModel
{
	/// <inheritdoc/>
	/// <remarks>Ignores <paramref name="checkCheirality"/>, as COLMAP does.</remarks>
	public static bool ImgFromCamWithJac(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, Span<double> jParams, Span<double> jUvw, bool checkCheirality = true)
	{
		x = 0;
		y = 0;
		double f = parameters[0];
		double c1 = parameters[1];
		double c2 = parameters[2];
		double k = parameters[3];

		// The derivatives are only written when a Jacobian is requested.
		bool withJac = !jUvw.IsEmpty || !jParams.IsEmpty;
		if (!CameraModelMath.DivisionScaleWithJac(u, v, w, k, withJac, out double r, out double drDu, out double drDv, out double drDw, out double drDk))
		{
			return false;
		}

		x = f * r * u + c1;
		y = f * r * v + c2;

		if (!jUvw.IsEmpty)
		{
			jUvw[0] = f * (r + u * drDu);
			jUvw[1] = f * u * drDv;
			jUvw[2] = f * u * drDw;
			jUvw[3] = f * v * drDu;
			jUvw[4] = f * (r + v * drDv);
			jUvw[5] = f * v * drDw;
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x4 matrix (row-major): d(x, y) / d(f, cx, cy, k).
			jParams[0] = r * u;
			jParams[1] = 1.0;
			jParams[2] = 0.0;
			jParams[3] = f * u * drDk;
			jParams[4] = r * v;
			jParams[5] = 0.0;
			jParams[6] = 1.0;
			jParams[7] = f * v * drDk;
		}

		return true;
	}
}

public readonly partial struct DivisionCameraModel
{
	/// <inheritdoc/>
	/// <remarks>Ignores <paramref name="checkCheirality"/>, as COLMAP does.</remarks>
	public static bool ImgFromCamWithJac(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, Span<double> jParams, Span<double> jUvw, bool checkCheirality = true)
	{
		x = 0;
		y = 0;
		double f1 = parameters[0];
		double f2 = parameters[1];
		double c1 = parameters[2];
		double c2 = parameters[3];
		double k = parameters[4];

		// The derivatives are only written when a Jacobian is requested.
		bool withJac = !jUvw.IsEmpty || !jParams.IsEmpty;
		if (!CameraModelMath.DivisionScaleWithJac(u, v, w, k, withJac, out double r, out double drDu, out double drDv, out double drDw, out double drDk))
		{
			return false;
		}

		x = f1 * r * u + c1;
		y = f2 * r * v + c2;

		if (!jUvw.IsEmpty)
		{
			jUvw[0] = f1 * (r + u * drDu);
			jUvw[1] = f1 * u * drDv;
			jUvw[2] = f1 * u * drDw;
			jUvw[3] = f2 * v * drDu;
			jUvw[4] = f2 * (r + v * drDv);
			jUvw[5] = f2 * v * drDw;
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x5 matrix (row-major): d(x, y) / d(fx, fy, cx, cy, k).
			jParams[0] = r * u;
			jParams[1] = 0.0;
			jParams[2] = 1.0;
			jParams[3] = 0.0;
			jParams[4] = f1 * u * drDk;
			jParams[5] = 0.0;
			jParams[6] = r * v;
			jParams[7] = 0.0;
			jParams[8] = 1.0;
			jParams[9] = f2 * v * drDk;
		}

		return true;
	}
}

public readonly partial struct EUCMCameraModel
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
		double alpha = parameters[4];
		double beta = parameters[5];

		double q = u * u + v * v;
		double rho2 = beta * q + w * w;
		if (rho2 < 0.0)
		{
			return false;
		}

		double rho = Math.Sqrt(rho2);
		double den = alpha * rho + (1.0 - alpha) * w;
		if (!CameraModelMath.HasProjectableDepth(den, checkCheirality))
		{
			return false;
		}

		double xn = u / den;
		double yn = v / den;

		x = f1 * xn + c1;
		y = f2 * yn + c2;

		if (!jUvw.IsEmpty || !jParams.IsEmpty)
		{
			double invRho = 1.0 / rho;
			double invDen = 1.0 / den;
			double invDen2 = invDen * invDen;
			// Derivatives of the denominator den = alpha*rho + (1-alpha)*w.
			double ddenDu = alpha * beta * u * invRho;
			double ddenDv = alpha * beta * v * invRho;
			double ddenDw = alpha * w * invRho + (1.0 - alpha);
			double ddenDalpha = rho - w;
			double ddenDbeta = alpha * q * 0.5 * invRho;

			if (!jUvw.IsEmpty)
			{
				double dxnDu = invDen - u * ddenDu * invDen2;
				double dxnDv = -u * ddenDv * invDen2;
				double dxnDw = -u * ddenDw * invDen2;
				double dynDu = -v * ddenDu * invDen2;
				double dynDv = invDen - v * ddenDv * invDen2;
				double dynDw = -v * ddenDw * invDen2;
				jUvw[0] = f1 * dxnDu;
				jUvw[1] = f1 * dxnDv;
				jUvw[2] = f1 * dxnDw;
				jUvw[3] = f2 * dynDu;
				jUvw[4] = f2 * dynDv;
				jUvw[5] = f2 * dynDw;
			}

			if (!jParams.IsEmpty)
			{
				// J_params is a 2x6 matrix (row-major):
				//   d(x, y) / d(fx, fy, cx, cy, alpha, beta)
				double dxnDalpha = -u * ddenDalpha * invDen2;
				double dxnDbeta = -u * ddenDbeta * invDen2;
				double dynDalpha = -v * ddenDalpha * invDen2;
				double dynDbeta = -v * ddenDbeta * invDen2;
				jParams[0] = xn;
				jParams[1] = 0.0;
				jParams[2] = 1.0;
				jParams[3] = 0.0;
				jParams[4] = f1 * dxnDalpha;
				jParams[5] = f1 * dxnDbeta;
				jParams[6] = 0.0;
				jParams[7] = yn;
				jParams[8] = 0.0;
				jParams[9] = 1.0;
				jParams[10] = f2 * dynDalpha;
				jParams[11] = f2 * dynDbeta;
			}
		}

		return true;
	}
}

public readonly partial struct EquirectangularCameraModel
{
	/// <inheritdoc/>
	/// <remarks>Ignores <paramref name="checkCheirality"/>, as COLMAP does.</remarks>
	public static bool ImgFromCamWithJac(ReadOnlySpan<double> parameters, double u, double v, double w, out double x, out double y, Span<double> jParams, Span<double> jUvw, bool checkCheirality = true)
	{
		x = 0;
		y = 0;
		double width = parameters[0];
		double height = parameters[1];

		double horizontal = Math.Sqrt(u * u + w * w);
		if (horizontal + Math.Abs(v) < LinearAlgebraConstants.MachineEpsilon)
		{
			return false;
		}

		double theta = Math.Atan2(u, w);
		double phi = Math.Atan2(-v, horizontal);

		// EIGEN_PI is M_PI to double precision, the same value as Math.PI.
		const double kInv2Pi = 1.0 / (2.0 * Math.PI);
		const double kInvPi = 1.0 / Math.PI;

		x = (theta * kInv2Pi + 0.5) * width;
		y = (0.5 - phi * kInvPi) * height;

		if (!jUvw.IsEmpty)
		{
			double r2 = horizontal * horizontal; // horizontal^2
			double n2 = r2 + v * v; // full squared norm
			// Hoist the shared reciprocals: R2 and N2*horizontal each divide more than
			// one derivative, and without -ffast-math the compiler cannot factor the
			// repeated runtime division out on its own.
			double invR2 = 1.0 / r2;
			double invN2 = 1.0 / n2;
			double invN2Horizontal = invN2 / horizontal;
			// theta = atan2(u, w).
			double dthetaDu = w * invR2;
			double dthetaDw = -u * invR2;
			// phi = atan2(-v, horizontal), horizontal = sqrt(u^2 + w^2).
			double dphiDu = u * v * invN2Horizontal;
			double dphiDv = -horizontal * invN2;
			double dphiDw = v * w * invN2Horizontal;

			jUvw[0] = width * kInv2Pi * dthetaDu;
			jUvw[1] = 0.0;
			jUvw[2] = width * kInv2Pi * dthetaDw;
			jUvw[3] = -height * kInvPi * dphiDu;
			jUvw[4] = -height * kInvPi * dphiDv;
			jUvw[5] = -height * kInvPi * dphiDw;
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x2 matrix (row-major): d(x, y) / d(width, height).
			jParams[0] = theta * kInv2Pi + 0.5;
			jParams[1] = 0.0;
			jParams[2] = 0.0;
			jParams[3] = 0.5 - phi * kInvPi;
		}

		return true;
	}
}
