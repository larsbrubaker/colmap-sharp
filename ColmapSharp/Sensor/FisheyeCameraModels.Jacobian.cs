// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The analytic ImgFromCamWithJac kernels of colmap/sensor/models_jacobian.h for the models
// in FisheyeCameraModels.cs: SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, OPENCV_FISHEYE,
// THIN_PRISM_FISHEYE, RAD_TAN_THIN_PRISM_FISHEYE, SIMPLE_FISHEYE and FISHEYE. Each goes
// normalized plane (a, b) -> fisheye plane (FisheyeProjectionWithJac) -> distortion ->
// pixels, and chains the 2x2 stage Jacobians before UvwJacFromAbJac
// (CameraModelJacobianMath.cs). Tests: ColmapSharp.Tests/Sensor/ModelsJacobianTests.cs.
//
// Tier A: COLMAP's operator order is kept expression by expression. C++ writes
// `MatMul2x2(ipjd, J_fisheye, m)` and the f-scaled J_ab inline per model; here that tail is
// CameraModelMath.FisheyeUvwJac, the same operations in the same order.

namespace ColmapSharp.Sensor;

public readonly partial struct SimpleRadialFisheyeCameraModel
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
		double a = u * invW;
		double b = v * invW;

		Span<double> jFisheye = stackalloc double[4];
		CameraModelMath.FisheyeProjectionWithJac(a, b, out double uu, out double vv, jUvw.IsEmpty ? default : jFisheye);

		// Single-parameter radial distortion in fisheye coordinates.
		double uu2 = uu * uu;
		double vv2 = vv * vv;
		double t2 = uu2 + vv2;
		double radial = k * t2;
		double uuD = uu + uu * radial;
		double vvD = vv + vv * radial;

		x = f * uuD + c1;
		y = f * vvD + c2;

		if (!jUvw.IsEmpty)
		{
			double twoK = 2.0 * k;
			// I + d(distortion) / d(uu, vv).
			ReadOnlySpan<double> ipjd = [1.0 + radial + twoK * uu2, twoK * uu * vv, twoK * uu * vv, 1.0 + radial + twoK * vv2];
			CameraModelMath.FisheyeUvwJac(ipjd, jFisheye, f, f, a, b, invW, jUvw);
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x4 matrix (row-major): d(x, y) / d(f, cx, cy, k).
			jParams[0] = uuD;
			jParams[1] = 1.0;
			jParams[2] = 0.0;
			jParams[3] = f * uu * t2;
			jParams[4] = vvD;
			jParams[5] = 0.0;
			jParams[6] = 1.0;
			jParams[7] = f * vv * t2;
		}

		return true;
	}
}

public readonly partial struct RadialFisheyeCameraModel
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
		double a = u * invW;
		double b = v * invW;

		Span<double> jFisheye = stackalloc double[4];
		CameraModelMath.FisheyeProjectionWithJac(a, b, out double uu, out double vv, jUvw.IsEmpty ? default : jFisheye);

		double uu2 = uu * uu;
		double vv2 = vv * vv;
		double t2 = uu2 + vv2;
		double t4 = t2 * t2;
		double radial = k1 * t2 + k2 * t4;
		double uuD = uu + uu * radial;
		double vvD = vv + vv * radial;

		x = f * uuD + c1;
		y = f * vvD + c2;

		if (!jUvw.IsEmpty)
		{
			double dRadial = k1 + 2.0 * k2 * t2;
			double cross = 2.0 * uu * vv * dRadial;
			ReadOnlySpan<double> ipjd = [1.0 + radial + 2.0 * uu2 * dRadial, cross, cross, 1.0 + radial + 2.0 * vv2 * dRadial];
			CameraModelMath.FisheyeUvwJac(ipjd, jFisheye, f, f, a, b, invW, jUvw);
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x5 matrix (row-major): d(x, y) / d(f, cx, cy, k1, k2).
			jParams[0] = uuD;
			jParams[1] = 1.0;
			jParams[2] = 0.0;
			jParams[3] = f * uu * t2;
			jParams[4] = f * uu * t4;
			jParams[5] = vvD;
			jParams[6] = 0.0;
			jParams[7] = 1.0;
			jParams[8] = f * vv * t2;
			jParams[9] = f * vv * t4;
		}

		return true;
	}
}

public readonly partial struct OpenCVFisheyeCameraModel
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
		double k3 = parameters[6];
		double k4 = parameters[7];

		double invW = 1.0 / w;
		double a = u * invW;
		double b = v * invW;

		Span<double> jFisheye = stackalloc double[4];
		CameraModelMath.FisheyeProjectionWithJac(a, b, out double uu, out double vv, jUvw.IsEmpty ? default : jFisheye);

		// Radial distortion in the theta-scaled fisheye coordinates.
		double uu2 = uu * uu;
		double vv2 = vv * vv;
		double t2 = uu2 + vv2;
		double t4 = t2 * t2;
		double t6 = t4 * t2;
		double t8 = t4 * t4;
		double radial = k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8;
		double uuD = uu + uu * radial;
		double vvD = vv + vv * radial;

		x = f1 * uuD + c1;
		y = f2 * vvD + c2;

		if (!jUvw.IsEmpty)
		{
			double dRadial = k1 + 2.0 * k2 * t2 + 3.0 * k3 * t4 + 4.0 * k4 * t6;
			double cross = 2.0 * uu * vv * dRadial;
			ReadOnlySpan<double> ipjd = [1.0 + radial + 2.0 * uu2 * dRadial, cross, cross, 1.0 + radial + 2.0 * vv2 * dRadial];
			CameraModelMath.FisheyeUvwJac(ipjd, jFisheye, f1, f2, a, b, invW, jUvw);
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x8 matrix (row-major):
			//   d(x, y) / d(fx, fy, cx, cy, k1, k2, k3, k4)
			jParams[0] = uuD;
			jParams[1] = 0.0;
			jParams[2] = 1.0;
			jParams[3] = 0.0;
			jParams[4] = f1 * uu * t2;
			jParams[5] = f1 * uu * t4;
			jParams[6] = f1 * uu * t6;
			jParams[7] = f1 * uu * t8;
			jParams[8] = 0.0;
			jParams[9] = vvD;
			jParams[10] = 0.0;
			jParams[11] = 1.0;
			jParams[12] = f2 * vv * t2;
			jParams[13] = f2 * vv * t4;
			jParams[14] = f2 * vv * t6;
			jParams[15] = f2 * vv * t8;
		}

		return true;
	}
}

public readonly partial struct ThinPrismFisheyeCameraModel
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
		double sx1 = parameters[10];
		double sy1 = parameters[11];

		double invW = 1.0 / w;
		double a = u * invW;
		double b = v * invW;

		Span<double> jFisheye = stackalloc double[4];
		CameraModelMath.FisheyeProjectionWithJac(a, b, out double uu, out double vv, jUvw.IsEmpty ? default : jFisheye);

		// Radial + tangential + thin-prism distortion in fisheye coordinates.
		double uu2 = uu * uu;
		double vv2 = vv * vv;
		double uv = uu * vv;
		double r2 = uu2 + vv2;
		double r4 = r2 * r2;
		double r6 = r4 * r2;
		double r8 = r4 * r4;
		double radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
		double du = uu * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * uu2) + sx1 * r2;
		double dv = vv * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * vv2) + sy1 * r2;
		double uuD = uu + du;
		double vvD = vv + dv;

		x = f1 * uuD + c1;
		y = f2 * vvD + c2;

		if (!jUvw.IsEmpty)
		{
			double dRadial = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4 + 4.0 * k4 * r6;
			double cross = 2.0 * uv * dRadial;
			double duDuu = radial + 2.0 * uu2 * dRadial + 2.0 * p1 * vv + 6.0 * p2 * uu + 2.0 * sx1 * uu;
			double duDvv = cross + 2.0 * p1 * uu + 2.0 * p2 * vv + 2.0 * sx1 * vv;
			double dvDuu = cross + 2.0 * p2 * vv + 2.0 * p1 * uu + 2.0 * sy1 * uu;
			double dvDvv = radial + 2.0 * vv2 * dRadial + 2.0 * p2 * uu + 6.0 * p1 * vv + 2.0 * sy1 * vv;
			ReadOnlySpan<double> ipjd = [1.0 + duDuu, duDvv, dvDuu, 1.0 + dvDvv];
			CameraModelMath.FisheyeUvwJac(ipjd, jFisheye, f1, f2, a, b, invW, jUvw);
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x12 matrix (row-major):
			//   d(x, y) / d(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1)
			jParams[0] = uuD;
			jParams[1] = 0.0;
			jParams[2] = 1.0;
			jParams[3] = 0.0;
			jParams[4] = f1 * uu * r2;
			jParams[5] = f1 * uu * r4;
			jParams[6] = f1 * 2.0 * uv;
			jParams[7] = f1 * (r2 + 2.0 * uu2);
			jParams[8] = f1 * uu * r6;
			jParams[9] = f1 * uu * r8;
			jParams[10] = f1 * r2;
			jParams[11] = 0.0;
			jParams[12] = 0.0;
			jParams[13] = vvD;
			jParams[14] = 0.0;
			jParams[15] = 1.0;
			jParams[16] = f2 * vv * r2;
			jParams[17] = f2 * vv * r4;
			jParams[18] = f2 * (r2 + 2.0 * vv2);
			jParams[19] = f2 * 2.0 * uv;
			jParams[20] = f2 * vv * r6;
			jParams[21] = f2 * vv * r8;
			jParams[22] = 0.0;
			jParams[23] = f2 * r2;
		}

		return true;
	}
}

public readonly partial struct RadTanThinPrismFisheyeModel
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
		ReadOnlySpan<double> k = parameters.Slice(4, 6); // k0..k5 (radial)
		double p0 = parameters[10];
		double p1 = parameters[11];
		double s0 = parameters[12];
		double s1 = parameters[13];
		double s2 = parameters[14];
		double s3 = parameters[15];

		double invW = 1.0 / w;
		double a = u * invW;
		double b = v * invW;

		Span<double> jFisheye = stackalloc double[4];
		CameraModelMath.FisheyeProjectionWithJac(a, b, out double uu, out double vv, jUvw.IsEmpty ? default : jFisheye);

		// Radial distortion: (xr, yr) = th_radial * (uu, vv). Also accumulate its
		// derivative th_radial' w.r.t. theta2 and the powers theta2^(i+1) used by the
		// per-coefficient parameter Jacobians.
		double theta2 = uu * uu + vv * vv;
		double thRadial = 1.0;
		double dThRadial = 0.0; // d(th_radial) / d(theta2)
		Span<double> thetaPow = stackalloc double[6]; // theta2^(i+1)
		double power = 1.0;
		for (int i = 0; i < 6; ++i)
		{
			double prevPower = power; // theta2^i
			power *= theta2; // theta2^(i+1)
			thetaPow[i] = power;
			thRadial += k[i] * power;
			dThRadial += (double)(i + 1) * k[i] * prevPower;
		}

		double xr = thRadial * uu;
		double yr = thRadial * vv;

		// Tangential + thin-prism distortion applied to (xr, yr).
		double xr2 = xr * xr;
		double yr2 = yr * yr;
		double xyr = xr * yr;
		double r2 = xr2 + yr2;
		double r4 = r2 * r2;

		double dxTang = 2.0 * p1 * xyr + p0 * (r2 + 2.0 * xr2);
		double dyTang = 2.0 * p0 * xyr + p1 * (r2 + 2.0 * yr2);
		double dxTp = s0 * r2 + s1 * r4;
		double dyTp = s2 * r2 + s3 * r4;

		double bigX = xr + dxTang + dxTp;
		double bigY = yr + dyTang + dyTp;

		x = f1 * bigX + c1;
		y = f2 * bigY + c2;

		if (!jUvw.IsEmpty || !jParams.IsEmpty)
		{
			// B = d(X, Y) / d(xr, yr) (tangential + thin-prism stage), used by both the
			// point/pose and the radial-coefficient Jacobians.
			double b00 = 1.0 + 2.0 * p1 * yr + 6.0 * p0 * xr + 2.0 * s0 * xr + 4.0 * s1 * xr * r2;
			double b01 = 2.0 * p1 * xr + 2.0 * p0 * yr + 2.0 * s0 * yr + 4.0 * s1 * yr * r2;
			double b10 = 2.0 * p0 * yr + 2.0 * p1 * xr + 2.0 * s2 * xr + 4.0 * s3 * xr * r2;
			double b11 = 1.0 + 2.0 * p0 * xr + 6.0 * p1 * yr + 2.0 * s2 * yr + 4.0 * s3 * yr * r2;

			if (!jUvw.IsEmpty)
			{
				// A = d(xr, yr) / d(uu, vv) (radial stage).
				double cross = 2.0 * uu * vv * dThRadial;
				ReadOnlySpan<double> matA = [thRadial + 2.0 * uu * uu * dThRadial, cross, cross, thRadial + 2.0 * vv * vv * dThRadial];
				ReadOnlySpan<double> matB = [b00, b01, b10, b11];
				Span<double> m2 = stackalloc double[4];
				CameraModelMath.MatMul2x2(matB, matA, m2); // d(X, Y) / d(uu, vv)
				CameraModelMath.FisheyeUvwJac(m2, jFisheye, f1, f2, a, b, invW, jUvw);
			}

			if (!jParams.IsEmpty)
			{
				// J_params is a 2x16 matrix (row-major):
				//   d(x, y) / d(fx, fy, cx, cy, k0..k5, p0, p1, s0, s1, s2, s3)
				jParams[..(2 * NumParams)].Clear();

				// Focal length and principal point.
				jParams[0] = bigX; // dx/dfx
				jParams[2] = 1.0; // dx/dcx
				jParams[16 + 1] = bigY; // dy/dfy
				jParams[16 + 3] = 1.0; // dy/dcy

				// Radial coefficients k0..k5 (params 4..9): dxr/dk_i = uu * theta2^(i+1),
				// dyr/dk_i = vv * theta2^(i+1), propagated through the tangential/prism
				// stage B.
				for (int i = 0; i < 6; ++i)
				{
					double dxr = uu * thetaPow[i];
					double dyr = vv * thetaPow[i];
					double dX = b00 * dxr + b01 * dyr;
					double dY = b10 * dxr + b11 * dyr;
					jParams[4 + i] = f1 * dX;
					jParams[16 + 4 + i] = f2 * dY;
				}

				// Tangential coefficients p0, p1 (params 10, 11).
				jParams[10] = f1 * (r2 + 2.0 * xr2); // dX/dp0
				jParams[11] = f1 * 2.0 * xyr; // dX/dp1
				jParams[16 + 10] = f2 * 2.0 * xyr; // dY/dp0
				jParams[16 + 11] = f2 * (r2 + 2.0 * yr2); // dY/dp1

				// Thin-prism coefficients s0..s3 (params 12..15).
				jParams[12] = f1 * r2; // dX/ds0
				jParams[13] = f1 * r4; // dX/ds1
				jParams[16 + 14] = f2 * r2; // dY/ds2
				jParams[16 + 15] = f2 * r4; // dY/ds3
			}
		}

		return true;
	}
}

public readonly partial struct SimpleFisheyeCameraModel
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
		double a = u * invW;
		double b = v * invW;

		Span<double> jFisheye = stackalloc double[4];
		CameraModelMath.FisheyeProjectionWithJac(a, b, out double uu, out double vv, jUvw.IsEmpty ? default : jFisheye);

		x = f * uu + c1;
		y = f * vv + c2;

		if (!jUvw.IsEmpty)
		{
			ReadOnlySpan<double> jAb = [f * jFisheye[0], f * jFisheye[1], f * jFisheye[2], f * jFisheye[3]];
			CameraModelMath.UvwJacFromAbJac(jAb, a, b, invW, jUvw);
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x3 matrix (row-major): d(x, y) / d(f, cx, cy).
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

public readonly partial struct FisheyeCameraModel
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
		double a = u * invW;
		double b = v * invW;

		Span<double> jFisheye = stackalloc double[4];
		CameraModelMath.FisheyeProjectionWithJac(a, b, out double uu, out double vv, jUvw.IsEmpty ? default : jFisheye);

		x = f1 * uu + c1;
		y = f2 * vv + c2;

		if (!jUvw.IsEmpty)
		{
			ReadOnlySpan<double> jAb = [f1 * jFisheye[0], f1 * jFisheye[1], f2 * jFisheye[2], f2 * jFisheye[3]];
			CameraModelMath.UvwJacFromAbJac(jAb, a, b, invW, jUvw);
		}

		if (!jParams.IsEmpty)
		{
			// J_params is a 2x4 matrix (row-major): d(x, y) / d(fx, fy, cx, cy).
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
