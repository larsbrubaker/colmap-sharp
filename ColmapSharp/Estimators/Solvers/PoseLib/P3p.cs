// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08.
//
// P3p: PoseLib/solvers/p3p.cc (poselib::p3p, by Yaqing Ding and Mark Shachkov), the helpers
// of PoseLib/solvers/p3p_common.h it uses (root2real, compute_pq, refine_lambda) and
// univariate::solve_cubic_single_real from PoseLib/misc/univariate.cc. The minimal
// absolute-pose solver behind COLMAP's P3PEstimator (Estimators/Solvers/AbsolutePose.cs).
//
// The method: Y. Ding, J. Yang, V. Larsson, C. Olsson, K. Astrom, "Revisiting the P3P
// Problem", CVPR 2023.
//
// Tier B: goes through a cubic and two quadratics. The only allocation is the output list.

using ColmapSharp.LinearAlgebra;

namespace ColmapSharp.Estimators.Solvers.PoseLib;

/// <summary>The P3P minimal solver. Port of poselib::p3p.</summary>
public static class P3p
{
	/// <summary>
	/// Estimate up to four camera poses from three bearing vectors <paramref name="xCopy"/>
	/// (unit length) and their 3D points <paramref name="bigXCopy"/>. Clears
	/// <paramref name="output"/>, appends the poses and returns their count.
	/// </summary>
	public static int Solve(ReadOnlySpan<Vector3d> xCopy, ReadOnlySpan<Vector3d> bigXCopy, List<CameraPose> output)
	{
		output.Clear();

		Vector3d x01 = bigXCopy[0] - bigXCopy[1];
		Vector3d x02 = bigXCopy[0] - bigXCopy[2];
		Vector3d x12 = bigXCopy[1] - bigXCopy[2];

		double a01 = x01.SquaredNorm;
		double a02 = x02.SquaredNorm;
		double a12 = x12.SquaredNorm;

		Vector3d bigX0 = bigXCopy[0], bigX1 = bigXCopy[1], bigX2 = bigXCopy[2];
		Vector3d x0 = xCopy[0], x1 = xCopy[1], x2 = xCopy[2];

		// Switch X,x so that BC is the largest distance among {X01, X02, X12}
		if (a01 > a02)
		{
			if (a01 > a12)
			{
				(x0, x2) = (x2, x0);
				(bigX0, bigX2) = (bigX2, bigX0);
				(a01, a12) = (a12, a01);
				x01 = -x12;
				x02 = -x02;
			}
		}
		else if (a02 > a12)
		{
			(x0, x1) = (x1, x0);
			(bigX0, bigX1) = (bigX1, bigX0);
			(a02, a12) = (a12, a02);
			x01 = -x01;
			x02 = x12;
		}

		double a12d = 1.0 / a12;
		double a = a01 * a12d;
		double b = a02 * a12d;

		double m01 = x0.Dot(x1);
		double m02 = x0.Dot(x2);
		double m12 = x1.Dot(x2);

		// Ugly parameters to simplify the calculation
		double m12sq = -m12 * m12 + 1.0;
		double m02sq = -1.0 + m02 * m02;
		double m01sq = -1.0 + m01 * m01;
		double ab = a * b;
		double bsq = b * b;
		double asq = a * a;
		double m013 = -2.0 + 2.0 * m01 * m02 * m12;
		double bsqm12sq = bsq * m12sq;
		double asqm12sq = asq * m12sq;
		double abm12sq = 2.0 * ab * m12sq;

		double k3Inv = 1.0 / (bsqm12sq + b * m02sq);
		double k2 = k3Inv * ((-1.0 + a) * m02sq + abm12sq + bsqm12sq + b * m013);
		double k1 = k3Inv * (asqm12sq + abm12sq + a * m013 + (-1.0 + b) * m01sq);
		double k0 = k3Inv * (asqm12sq + a * m01sq);

		bool g = SolveCubicSingleReal(k2, k1, k0, out double s);

		var c = new Matrix3d(
			-a + s * (1 - b), -m02 * s, a * m12 + b * m12 * s,
			-m02 * s, s + 1, -m01,
			a * m12 + b * m12 * s, -m01, -a - b * s + 1);

		ComputePq(c, out Vector3d pq0, out Vector3d pq1);

		Matrix3d xx = Matrix3d.FromColumns(x01, x02, x01.Cross(x02)).Inverse();

		int nSols = 0;

		for (int i = 0; i < 2; ++i)
		{
			// [p0 p1 p2] * [1; x; y] = 0, or [p0 p1 p2] * [d2; d0; d1] = 0
			Vector3d pq = i == 0 ? pq0 : pq1;
			double p0 = pq.X;
			double p1 = pq.Y;
			double p2 = pq.Z;
			// here we run into trouble if p0 is zero,
			// so depending on which is larger, we solve for either d0 or d1
			// The case p0 = p1 = 0 is degenerate and can be ignored
			bool switch12 = Math.Abs(p0) <= Math.Abs(p1);

			double d0, d1, d2;
			if (switch12)
			{
				// eliminate d0
				double w0 = -p0 / p1;
				double w1 = -p2 / p1;
				double ca = 1.0 / (w1 * w1 - b);
				double cb = 2.0 * (b * m12 - m02 * w1 + w0 * w1) * ca;
				double cc = (w0 * w0 - 2 * m02 * w0 - b + 1.0) * ca;
				if (!Root2Real(cb, cc, out double tau0, out double tau1))
				{
					continue;
				}

				for (int k = 0; k < 2; ++k)
				{
					double tau = k == 0 ? tau0 : tau1;
					if (tau <= 0)
					{
						continue;
					}

					// positive only
					d2 = Math.Sqrt(a12 / (tau * (tau - 2.0 * m12) + 1.0));
					d1 = tau * d2;
					d0 = w0 * d2 + w1 * d1;
					if (d0 < 0)
					{
						continue;
					}

					RefineLambda(ref d0, ref d1, ref d2, a01, a02, a12, m01, m02, m12);
					AddPose(d0, d1, d2, x0, x1, x2, bigX0, xx, output);
					++nSols;
				}
			}
			else
			{
				double w0 = -p1 / p0;
				double w1 = -p2 / p0;
				double ca = 1.0 / (-a * w1 * w1 + 2 * a * m12 * w1 - a + 1);
				double cb = 2 * (a * m12 * w0 - m01 - a * w0 * w1) * ca;
				double cc = (1 - a * w0 * w0) * ca;

				if (!Root2Real(cb, cc, out double tau0, out double tau1))
				{
					continue;
				}

				for (int k = 0; k < 2; ++k)
				{
					double tau = k == 0 ? tau0 : tau1;
					if (tau <= 0)
					{
						continue;
					}

					d0 = Math.Sqrt(a01 / (tau * (tau - 2.0 * m01) + 1.0));
					d1 = tau * d0;
					d2 = w0 * d0 + w1 * d1;

					if (d2 < 0)
					{
						continue;
					}

					RefineLambda(ref d0, ref d1, ref d2, a01, a02, a12, m01, m02, m12);
					AddPose(d0, d1, d2, x0, x1, x2, bigX0, xx, output);
					++nSols;
				}
			}

			if (nSols > 0 && g)
			{
				break;
			}
		}

		return output.Count;
	}

	private static void AddPose(
		double d0, double d1, double d2, Vector3d x0, Vector3d x1, Vector3d x2, Vector3d bigX0, Matrix3d xx,
		List<CameraPose> output)
	{
		Vector3d v1 = d0 * x0 - d1 * x1;
		Vector3d v2 = d0 * x0 - d2 * x2;
		Matrix3d yy = Matrix3d.FromColumns(v1, v2, v1.Cross(v2));
		Matrix3d r = yy * xx;
		output.Add(new CameraPose(r, d0 * x0 - r * bigX0));
	}

	/// <summary>
	/// Real roots of x^2 + b x + c. Port of poselib::root2real (p3p_common.h): a slightly
	/// negative discriminant counts as a double root.
	/// </summary>
	internal static bool Root2Real(double b, double c, out double r1, out double r2)
	{
		const double Threshold = -1.0e-12;
		double v = b * b - 4.0 * c;
		if (v < Threshold)
		{
			r1 = r2 = -0.5 * b;
			return v >= 0;
		}

		if (v > Threshold && v < 0.0)
		{
			r1 = -0.5 * b;
			r2 = -2;
			return true;
		}

		double y = Math.Sqrt(v);
		if (b < 0)
		{
			r1 = 0.5 * (-b + y);
			r2 = 0.5 * (-b - y);
		}
		else
		{
			r1 = 2.0 * c / (-b + y);
			r2 = 2.0 * c / (-b - y);
		}

		return true;
	}

	// Port of poselib::compute_pq (p3p_common.h): splits the degenerate conic C into two
	// lines via its adjugate.
	private static void ComputePq(Matrix3d cIn, out Vector3d pq0, out Vector3d pq1)
	{
		Span<double> c = stackalloc double[9];
		cIn.CopyToColumnMajor(c);
		// C(r, k) = c[r + 3k] (column-major).
		double adj00 = c[7] * c[5] - c[4] * c[8];
		double adj11 = c[6] * c[2] - c[0] * c[8];
		double adj22 = c[3] * c[1] - c[0] * c[4];
		double adj01 = c[3] * c[8] - c[6] * c[5];
		double adj02 = c[6] * c[4] - c[3] * c[7];
		double adj10 = adj01;
		double adj12 = c[0] * c[7] - c[6] * c[1];
		double adj20 = adj02;
		double adj21 = adj12;

		Vector3d v;
		if (adj00 > adj11)
		{
			if (adj00 > adj22)
			{
				v = new Vector3d(adj00, adj10, adj20) / Math.Sqrt(adj00);
			}
			else
			{
				v = new Vector3d(adj02, adj12, adj22) / Math.Sqrt(adj22);
			}
		}
		else if (adj11 > adj22)
		{
			v = new Vector3d(adj01, adj11, adj21) / Math.Sqrt(adj11);
		}
		else
		{
			v = new Vector3d(adj02, adj12, adj22) / Math.Sqrt(adj22);
		}

		c[3] -= v.Z; // C(0, 1)
		c[6] += v.Y; // C(0, 2)
		c[7] -= v.X; // C(1, 2)
		c[1] += v.Z; // C(1, 0)
		c[2] -= v.Y; // C(2, 0)
		c[5] += v.X; // C(2, 1)

		pq0 = new Vector3d(c[0], c[1], c[2]);
		pq1 = new Vector3d(c[0], c[3], c[6]);
	}

	// Performs a few newton steps on the equations
	private static void RefineLambda(
		ref double lambda1, ref double lambda2, ref double lambda3, double a12, double a13, double a23,
		double b12, double b13, double b23)
	{
		for (int iter = 0; iter < 5; ++iter)
		{
			double r1 = lambda1 * lambda1 - 2.0 * lambda1 * lambda2 * b12 + lambda2 * lambda2 - a12;
			double r2 = lambda1 * lambda1 - 2.0 * lambda1 * lambda3 * b13 + lambda3 * lambda3 - a13;
			double r3 = lambda2 * lambda2 - 2.0 * lambda2 * lambda3 * b23 + lambda3 * lambda3 - a23;
			if (Math.Abs(r1) + Math.Abs(r2) + Math.Abs(r3) < 1e-10)
			{
				return;
			}

			double x11 = lambda1 - lambda2 * b12;
			double x12 = lambda2 - lambda1 * b12;
			double x21 = lambda1 - lambda3 * b13;
			double x23 = lambda3 - lambda1 * b13;
			double x32 = lambda2 - lambda3 * b23;
			double x33 = lambda3 - lambda2 * b23;
			double detJ = 0.5 / (x11 * x23 * x32 + x12 * x21 * x33); // half minus inverse determinant
			// This uses the closed form of the inverse for the jacobean.
			// Due to the zero elements this actually becomes quite nice.
			lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
			lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
			lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
		}
	}

	/// <summary>
	/// One real root of x^3 + c2 x^2 + c1 x + c0; returns true when it is the only real
	/// root. Port of poselib::univariate::solve_cubic_single_real.
	/// </summary>
	internal static bool SolveCubicSingleReal(double c2, double c1, double c0, out double root)
	{
		double a = c1 - c2 * c2 / 3.0;
		double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
		double c = b * b / 4.0 + a * a * a / 27.0;
		if (c != 0)
		{
			if (c > 0)
			{
				c = Math.Sqrt(c);
				b *= -0.5;
				root = Math.Cbrt(b + c) + Math.Cbrt(b - c) - c2 / 3.0;
				return true;
			}

			c = 3.0 * b / (2.0 * a) * Math.Sqrt(-3.0 / a);
			root = 2.0 * Math.Sqrt(-a / 3.0) * Math.Cos(Math.Acos(c) / 3.0) - c2 / 3.0;
		}
		else
		{
			root = -c2 / 3.0 + (a != 0 ? (3.0 * b / a) : 0);
		}

		return false;
	}
}
