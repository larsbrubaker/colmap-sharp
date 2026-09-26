// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BasePerspectiveCameraModel::IterativeUndistortion (colmap/sensor/models.h): inverts a
// model's additive distortion x + d(x) = x0 by a trust-region Newton iteration. The
// Jacobian of d comes from evaluating the model's Distortion on Jet<Grad2> (Solver/Jet.cs), as
// COLMAP does with ceres::Jet<double, 2>, and each step solves the 2x2 system with LU
// decomposition and partial (row) pivoting, which is what COLMAP's
// `J.partialPivLu().solve(...)` computes. The 2x2 solve is written here from the textbook
// algorithm (Golub and Van Loan, "Matrix Computations", 3.4.1: Gaussian elimination with
// partial pivoting); Eigen (MPL-2.0) is not ported, docs/LICENSE_AUDIT.md. Part of the
// CameraModelMath helpers declared in CameraModelBase.cs.
//
// Tier A: the iteration count, step constants, trust-region clamp and every operation's
// order match COLMAP, and CameraModelOracleTests pins the result bit for bit against
// pycolmap.

using ColmapSharp.Solver;

namespace ColmapSharp.Sensor;

public static partial class CameraModelMath
{
	/// <summary>
	/// Port of BasePerspectiveCameraModel::IterativeUndistortion. On entry (u, v) is the
	/// distorted point; on exit it is the undistorted one. Returns false if the iteration
	/// did not converge within 100 steps (the last iterate is still written).
	/// </summary>
	/// <param name="extraParams">The model's extra (distortion) parameters only.</param>
	/// <param name="u">Distorted u on entry, undistorted u on exit.</param>
	/// <param name="v">Distorted v on entry, undistorted v on exit.</param>
	public static bool IterativeUndistortion<TModel>(ReadOnlySpan<double> extraParams, ref double u, ref double v)
		where TModel : struct, IDistortedCameraModel<TModel>
	{
		// Parameters for Newton iteration. 100 iterations should be enough for complex
		// camera models with higher order terms.
		const int kNumIterations = 100;
		const double kMinStepSquaredNorm = 1e-10;

		// Trust region: step_x.norm() <= max(x.norm() * kRelStepRadius, kStepRadius)
		const double kRelStepRadius = 0.1;
		const double kStepRadius = 0.1;

		double x00 = u;
		double x01 = v;
		double x0 = u;
		double x1 = v;

		int numExtraParams = TModel.ExtraParamsIdxs.Length;
		Span<Jet<Grad2>> paramsJet = stackalloc Jet<Grad2>[numExtraParams];
		for (int i = 0; i < numExtraParams; i++)
		{
			paramsJet[i] = Jet<Grad2>.FromDouble(extraParams[i]);
		}

		for (int i = 0; i < kNumIterations; i++)
		{
			// Get Jacobian
			TModel.Distortion<Jet<Grad2>>(paramsJet, Jet<Grad2>.Variable(x0, 0), Jet<Grad2>.Variable(x1, 1), out Jet<Grad2> dxJet0, out Jet<Grad2> dxJet1);
			double dx0 = dxJet0.A;
			double dx1 = dxJet1.A;
			double j00 = dxJet0.V[0] + 1;
			double j01 = dxJet0.V[1];
			double j10 = dxJet1.V[0];
			double j11 = dxJet1.V[1] + 1;

			// Update
			SolvePartialPivLu2(j00, j01, j10, j11, x0 + dx0 - x00, x1 + dx1 - x01, out double step0, out double step1);
			double radiusSqr = Math.Max(
				(x0 * x0 + x1 * x1) * kRelStepRadius * kRelStepRadius,
				kStepRadius * kStepRadius);
			double stepNormSqr = step0 * step0 + step1 * step1;
			if (stepNormSqr > radiusSqr)
			{
				double scale = Math.Sqrt(radiusSqr / stepNormSqr);
				step0 *= scale;
				step1 *= scale;
			}

			x0 -= step0;
			x1 -= step1;
			if (step0 * step0 + step1 * step1 < kMinStepSquaredNorm)
			{
				u = x0;
				v = x1;
				return true;
			}
		}

		u = x0;
		v = x1;
		return false;
	}

	/// <summary>
	/// Solves [a00 a01; a10 a11] x = b by LU decomposition with partial pivoting: the pivot
	/// row is the one with the larger |a_i0| (the first on a tie), L is unit lower
	/// triangular, and both triangular solves divide by the pivots. A zero pivot column is
	/// left unscaled, so a singular matrix yields inf/NaN rather than an exception.
	/// </summary>
	internal static void SolvePartialPivLu2(double a00, double a01, double a10, double a11, double b0, double b1, out double x0, out double x1)
	{
		if (Math.Abs(a10) > Math.Abs(a00))
		{
			(a00, a10) = (a10, a00);
			(a01, a11) = (a11, a01);
			(b0, b1) = (b1, b0);
		}

		double l10 = a00 != 0 ? a10 / a00 : a10;
		double u11 = a11 - l10 * a01;

		// Forward substitution with the unit lower factor.
		double y1 = b1 - b0 * l10;

		// Back substitution with the upper factor.
		x1 = y1 / u11;
		x0 = (b0 - x1 * a01) / a00;
	}
}
