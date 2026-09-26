// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GeneralizedRelativePose.GR8P: the solver half of colmap/estimators/solvers/
// generalized_relative_pose.cc - GR8PEstimator::Estimate and its anonymous-namespace helpers
// (ComposePlueckerData, the Cayley conversions, ComputeRotationBetweenPoints, ComposeG,
// ComputeEigenValue, ComputeCost, ComputeJacobian). The estimator struct is in
// GeneralizedRelativePose.cs.
//
// Kneip and Li (CVPR 2014): the smallest eigenvalue of the 4x4 matrix G(R) vanishes at the
// true rotation, so the Cayley parameters are found by a finite-difference gradient descent
// on a closed-form root of G's characteristic quartic (with random restarts, drawn from the
// thread PRNG like COLMAP's RandomUniformReal), and the translations are read from G's
// eigenvectors at the optimum.
//
// Eigenvectors: G is symmetric (ComposeG mirrors every off-diagonal entry), so all of its
// eigenvalues are real and LinearAlgebra/EigenSolver.cs returns real eigenvectors for them.
// COLMAP's V.real().colwise().hnormalized() then divides each by its last entry, which
// removes the scale and sign that are the only freedom left, so the complex-phase freedom
// the EigenSolver header warns about never reaches the result. The order of the four
// candidates follows the Schur order, which may differ from Eigen's; RANSAC scores all
// four. Tier B.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>The GR8P solver: port of GR8PEstimator::Estimate and its helpers.</summary>
internal static class GR8PSolver
{
	/// <summary>Port of GR8PEstimator::Estimate.</summary>
	public static void Estimate(ReadOnlySpan<GrnpObservation> points1, ReadOnlySpan<GrnpObservation> points2, List<Rigid3d> rigs2FromRigs1)
	{
		Check.Ge(points1.Length, 6);
		Check.Eq(points1.Length, points2.Length);

		rigs2FromRigs1.Clear();

		int n = points1.Length;
		var originsInRig1 = new Vector3d[n];
		var originsInRig2 = new Vector3d[n];
		var rays1 = new Vector3d[n];
		var rays2 = new Vector3d[n];
		for (int i = 0; i < n; ++i)
		{
			ComposePlueckerData(points1[i].CamFromRig.Inverse(), points1[i].RayWithJacInCam.Ray, out originsInRig1[i], out rays1[i]);
			ComposePlueckerData(points2[i].CamFromRig.Inverse(), points2[i].RayWithJacInCam.Ray, out originsInRig2[i], out rays2[i]);
		}

		var terms = new Terms();
		for (int i = 0; i < n; ++i)
		{
			terms.Accumulate(rays1[i], rays2[i], originsInRig1[i], originsInRig2[i]);
		}

		Vector3d initialRotation = ComputeRotationBetweenPoints(rays1, rays2);

		const double kMinLambda = 0.00001;
		const double kMaxLambda = 0.08;
		const double kLambdaModifier = 2.0;
		const int kMaxNumIterations = 50;
		const bool kDisableIncrements = true;

		double perturbationAmplitude = 0.3;
		int numRandomTrials = 0;

		Vector3d rotation = default;
		while (numRandomTrials < 5)
		{
			if (numRandomTrials > 2)
			{
				perturbationAmplitude = 0.6;
			}

			if (numRandomTrials == 0)
			{
				rotation = initialRotation;
			}
			else
			{
				// Separate statements keep the x, y, z draw order of the C++ constructor
				// arguments, which libc++ evaluates left to right here.
				double px = RandomUtils.RandomUniformReal(-perturbationAmplitude, perturbationAmplitude);
				double py = RandomUtils.RandomUniformReal(-perturbationAmplitude, perturbationAmplitude);
				double pz = RandomUtils.RandomUniformReal(-perturbationAmplitude, perturbationAmplitude);
				rotation = initialRotation + new Vector3d(px, py, pz);
			}

			double lambda = 0.01;
			int numIterations = 0;
			double smallestEigenValue = ComputeCost(terms, rotation, 1);

			for (int iter = 0; iter < kMaxNumIterations; ++iter)
			{
				Vector3d jacobian = ComputeJacobian(terms, rotation, smallestEigenValue, 1);

				Vector3d normalizedJacobian = jacobian.Normalized();

				Vector3d samplingPoint = rotation - lambda * normalizedJacobian;
				double samplingEigenValue = ComputeCost(terms, samplingPoint, 1);

				if (numIterations == 0 || !kDisableIncrements)
				{
					while (samplingEigenValue < smallestEigenValue)
					{
						smallestEigenValue = samplingEigenValue;
						if (lambda * kLambdaModifier > kMaxLambda)
						{
							break;
						}

						lambda *= kLambdaModifier;
						samplingPoint = rotation - lambda * normalizedJacobian;
						samplingEigenValue = ComputeCost(terms, samplingPoint, 1);
					}
				}

				while (samplingEigenValue > smallestEigenValue)
				{
					lambda /= kLambdaModifier;
					samplingPoint = rotation - lambda * normalizedJacobian;
					samplingEigenValue = ComputeCost(terms, samplingPoint, 1);
				}

				rotation = samplingPoint;
				smallestEigenValue = samplingEigenValue;

				if (lambda < kMinLambda)
				{
					break;
				}
			}

			if (rotation.Norm < 0.01)
			{
				double eigenValue2 = ComputeCost(terms, rotation, 0);
				if (eigenValue2 > 0.001)
				{
					numRandomTrials += 1;
				}
				else
				{
					break;
				}
			}
			else
			{
				break;
			}
		}

		Matrix3d r = CayleyToRotationMatrix(rotation).Transpose();

		Matrix4d g = ComposeG(terms, rotation);

		var gDynamic = new MatrixXd(4, 4);
		for (int row = 0; row < 4; row++)
		{
			for (int col = 0; col < 4; col++)
			{
				gDynamic[row, col] = g[row, col];
			}
		}

		System.Numerics.Complex[,] v = new EigenSolver(gDynamic, computeEigenvectors: true).Eigenvectors();

		Quaterniond q = Quaterniond.FromRotationMatrix(r);
		for (int i = 0; i < 4; ++i)
		{
			// V.real().colwise().hnormalized()
			double w = v[3, i].Real;
			var vv = new Vector3d(v[0, i].Real / w, v[1, i].Real / w, v[2, i].Real / w);
			rigs2FromRigs1.Add(new Rigid3d(q, -(r * vv)));
		}
	}

	private static void ComposePlueckerData(Rigid3d rigFromCam, Vector3d rayInCam, out Vector3d originInRig, out Vector3d rayInRig)
	{
		// Only the direction half of COLMAP's 6-vector Pluecker coordinate is ever read
		// (head<3>()), so the moment half is not formed.
		rayInRig = (rigFromCam.Rotation * rayInCam).Normalized();
		originInRig = rigFromCam.Translation;
	}

	internal static Matrix3d CayleyToRotationMatrix(Vector3d cayley)
	{
		double cayley0Sqr = cayley[0] * cayley[0];
		double cayley1Sqr = cayley[1] * cayley[1];
		double cayley2Sqr = cayley[2] * cayley[2];
		double cayley01 = cayley[0] * cayley[1];
		double cayley12 = cayley[1] * cayley[2];
		double cayley02 = cayley[0] * cayley[2];

		double scale = 1 + cayley0Sqr + cayley1Sqr + cayley2Sqr;
		double invScale = 1.0 / scale;

		return new Matrix3d(
			invScale * (1 + cayley0Sqr - cayley1Sqr - cayley2Sqr),
			invScale * (2 * (cayley01 - cayley[2])),
			invScale * (2 * (cayley02 + cayley[1])),
			invScale * (2 * (cayley01 + cayley[2])),
			invScale * (1 - cayley0Sqr + cayley1Sqr - cayley2Sqr),
			invScale * (2 * (cayley12 - cayley[0])),
			invScale * (2 * (cayley02 - cayley[1])),
			invScale * (2 * (cayley12 + cayley[0])),
			invScale * (1 - cayley0Sqr - cayley1Sqr + cayley2Sqr));
	}

	private static Vector3d RotationMatrixToCaley(in Matrix3d r)
	{
		Matrix3d c1 = r - Matrix3d.Identity;
		Matrix3d c2 = r + Matrix3d.Identity;
		Matrix3d c = c1 * c2.Inverse();
		return new Vector3d(-c[1, 2], c[0, 2], -c[0, 1]);
	}

	private static Vector3d ComputeRotationBetweenPoints(Vector3d[] rays1, Vector3d[] rays2)
	{
		Check.Eq(rays1.Length, rays2.Length);

		// Compute the center of all observed points.
		Vector3d pointsCenter1 = Vector3d.Zero;
		Vector3d pointsCenter2 = Vector3d.Zero;
		for (int i = 0; i < rays1.Length; i++)
		{
			pointsCenter1 += rays1[i];
			pointsCenter2 += rays2[i];
		}

		pointsCenter1 = pointsCenter1 / rays1.Length;
		pointsCenter2 = pointsCenter2 / rays1.Length;

		Matrix3d hcross = Matrix3d.Zero;
		for (int i = 0; i < rays1.Length; i++)
		{
			Vector3d f1 = rays1[i] - pointsCenter1;
			Vector3d f2 = rays2[i] - pointsCenter2;
			hcross += Terms.Outer(f2, f1);
		}

		Svd3d svd = Svd3d.Compute(hcross);
		Matrix3d v = svd.MatrixV;
		Matrix3d u = svd.MatrixU;

		Matrix3d r = v * u.Transpose();
		if (r.Determinant() < 0)
		{
			Matrix3d vPrime = Matrix3d.FromColumns(v.Col(0), v.Col(1), -v.Col(2));
			r = vPrime * u.Transpose();
		}

		return RotationMatrixToCaley(r);
	}

	private static Matrix4d ComposeG(Terms t, Vector3d rotation)
	{
		Matrix3d r = CayleyToRotationMatrix(rotation);
		Vector3d r0 = r.Row(0);
		Vector3d r1 = r.Row(1);
		Vector3d r2 = r.Row(2);

		ReadOnlySpan<double> rRows = [r[0, 0], r[0, 1], r[0, 2], r[1, 0], r[1, 1], r[1, 2], r[2, 0], r[2, 1], r[2, 2]];
		ReadOnlySpan<double> rCols = [r[0, 0], r[1, 0], r[2, 0], r[0, 1], r[1, 1], r[2, 1], r[0, 2], r[1, 2], r[2, 2]];

		Vector3d xxFr1t = t.XxF * r1;
		Vector3d yyFr0t = t.YyF * r0;
		Vector3d zzFr0t = t.ZzF * r0;
		Vector3d yzFr0t = t.YzF * r0;
		Vector3d xyFr1t = t.XyF * r1;
		Vector3d xyFr2t = t.XyF * r2;
		Vector3d zxFr1t = t.ZxF * r1;
		Vector3d zxFr2t = t.ZxF * r2;

		Vector3d x1PC = Terms.Mul3x9(t.X1P, rCols);
		Vector3d y1PC = Terms.Mul3x9(t.Y1P, rCols);
		Vector3d z1PC = Terms.Mul3x9(t.Z1P, rCols);

		Vector3d x2PR = Terms.Mul3x9(t.X2P, rRows);
		Vector3d y2PR = Terms.Mul3x9(t.Y2P, rRows);
		Vector3d z2PR = Terms.Mul3x9(t.Z2P, rRows);

		Span<double> g = stackalloc double[16];

		// g[row * 4 + col]; a row vector times a matrix times a column is evaluated as
		// (row * matrix) . column, left to right as the C++ expressions read.
		g[0] = Quad(r2, t.YyF, r2);
		g[0] += Quad(-2.0 * r2, t.YzF, r1);
		g[0] += Quad(r1, t.ZzF, r1);

		g[1] = r2.Dot(yzFr0t);
		g[1] += (-1.0 * r2).Dot(xyFr2t);
		g[1] += (-1.0 * r1).Dot(zzFr0t);
		g[1] += r1.Dot(zxFr2t);

		g[2] = r2.Dot(xyFr1t);
		g[2] += (-1.0 * r2).Dot(yyFr0t);
		g[2] += (-1.0 * r1).Dot(zxFr1t);
		g[2] += r1.Dot(yzFr0t);

		g[5] = r0.Dot(zzFr0t);
		g[5] += (-2.0 * r0).Dot(zxFr2t);
		g[5] += Quad(r2, t.XxF, r2);

		g[6] = r0.Dot(zxFr1t);
		g[6] += (-1.0 * r0).Dot(yzFr0t);
		g[6] += (-1.0 * r2).Dot(xxFr1t);
		g[6] += r0.Dot(xyFr2t);

		g[10] = r1.Dot(xxFr1t);
		g[10] += (-2.0 * r0).Dot(xyFr1t);
		g[10] += r0.Dot(yyFr0t);

		g[4] = g[1];
		g[8] = g[2];
		g[9] = g[6];

		g[3] = r2.Dot(y1PC);
		g[3] += r2.Dot(y2PR);
		g[3] += (-1.0 * r1).Dot(z1PC);
		g[3] += (-1.0 * r1).Dot(z2PR);

		g[7] = r0.Dot(z1PC);
		g[7] += r0.Dot(z2PR);
		g[7] += (-1.0 * r2).Dot(x1PC);
		g[7] += (-1.0 * r2).Dot(x2PR);

		g[11] = r1.Dot(x1PC);
		g[11] += r1.Dot(x2PR);
		g[11] += (-1.0 * r0).Dot(y1PC);
		g[11] += (-1.0 * r0).Dot(y2PR);

		g[15] = Terms.Quad9(-1.0, rCols, t.M11P, rCols);
		g[15] += Terms.Quad9(-1.0, rRows, t.M22P, rRows);
		g[15] += Terms.Quad9(-2.0, rRows, t.M12P, rCols);

		g[12] = g[3];
		g[13] = g[7];
		g[14] = g[11];

		return new Matrix4d(
			g[0], g[1], g[2], g[3],
			g[4], g[5], g[6], g[7],
			g[8], g[9], g[10], g[11],
			g[12], g[13], g[14], g[15]);
	}

	/// <summary>(a^T M) . b, Eigen's a.transpose() * M * b for row vector a.</summary>
	private static double Quad(Vector3d a, in Matrix3d m, Vector3d b) => (m.Transpose() * a).Dot(b);

	private static Vector4d ComputeEigenValue(Terms t, Vector3d rotation)
	{
		Matrix4d g = ComposeG(t, rotation);

		// Compute the roots in closed-form.
		double g01_2 = g[0, 1] * g[0, 1];
		double g02_2 = g[0, 2] * g[0, 2];
		double g03_2 = g[0, 3] * g[0, 3];
		double g12_2 = g[1, 2] * g[1, 2];
		double g13_2 = g[1, 3] * g[1, 3];
		double g23_2 = g[2, 3] * g[2, 3];

		double b = -g[3, 3] - g[2, 2] - g[1, 1] - g[0, 0];
		double c = -g23_2 + g[2, 2] * g[3, 3] - g13_2 - g12_2
			+ g[1, 1] * g[3, 3] + g[1, 1] * g[2, 2] - g03_2 - g02_2
			- g01_2 + g[0, 0] * g[3, 3] + g[0, 0] * g[2, 2]
			+ g[0, 0] * g[1, 1];
		double d =
			g13_2 * g[2, 2] - 2.0 * g[1, 2] * g[1, 3] * g[2, 3] + g12_2 * g[3, 3]
			+ g[1, 1] * g23_2 - g[1, 1] * g[2, 2] * g[3, 3] + g03_2 * g[2, 2]
			+ g03_2 * g[1, 1] - 2.0 * g[0, 2] * g[0, 3] * g[2, 3] + g02_2 * g[3, 3]
			+ g02_2 * g[1, 1] - 2.0 * g[0, 1] * g[0, 3] * g[1, 3]
			- 2.0 * g[0, 1] * g[0, 2] * g[1, 2] + g01_2 * g[3, 3] + g01_2 * g[2, 2]
			+ g[0, 0] * g23_2 - g[0, 0] * g[2, 2] * g[3, 3] + g[0, 0] * g13_2
			+ g[0, 0] * g12_2 - g[0, 0] * g[1, 1] * g[3, 3]
			- g[0, 0] * g[1, 1] * g[2, 2];
		double e =
			g03_2 * g12_2 - g03_2 * g[1, 1] * g[2, 2]
			- 2.0 * g[0, 2] * g[0, 3] * g[1, 2] * g[1, 3]
			+ 2.0 * g[0, 2] * g[0, 3] * g[1, 1] * g[2, 3] + g02_2 * g13_2
			- g02_2 * g[1, 1] * g[3, 3] + 2.0 * g[0, 1] * g[0, 3] * g[1, 3] * g[2, 2]
			- 2.0 * g[0, 1] * g[0, 3] * g[1, 2] * g[2, 3]
			- 2.0 * g[0, 1] * g[0, 2] * g[1, 3] * g[2, 3]
			+ 2.0 * g[0, 1] * g[0, 2] * g[1, 2] * g[3, 3] + g01_2 * g23_2
			- g01_2 * g[2, 2] * g[3, 3] - g[0, 0] * g13_2 * g[2, 2]
			+ 2.0 * g[0, 0] * g[1, 2] * g[1, 3] * g[2, 3] - g[0, 0] * g12_2 * g[3, 3]
			- g[0, 0] * g[1, 1] * g23_2 + g[0, 0] * g[1, 1] * g[2, 2] * g[3, 3];

		double bPw2 = b * b;
		double bPw3 = bPw2 * b;
		double bPw4 = bPw3 * b;
		double alpha = -0.375 * bPw2 + c;
		double beta = bPw3 / 8.0 - b * c / 2.0 + d;
		double gamma = -0.01171875 * bPw4 + bPw2 * c / 16.0 - b * d / 4.0 + e;
		double alphaPw2 = alpha * alpha;
		double alphaPw3 = alphaPw2 * alpha;
		double p = -alphaPw2 / 12.0 - gamma;
		double q = -alphaPw3 / 108.0 + alpha * gamma / 3.0 - beta * beta / 8.0;
		double helper1 = -p * p * p / 27.0;
		double theta2 = Math.Pow(helper1, 1.0 / 3.0);
		double theta1 = Math.Sqrt(theta2) * Math.Cos((1.0 / 3.0) * Math.Acos((-q / 2.0) / Math.Sqrt(helper1)));
		double y = -(5.0 / 6.0) * alpha - ((1.0 / 3.0) * p * theta1 - theta1 * theta2) / theta2;
		double w = Math.Sqrt(alpha + 2.0 * y);

		return new Vector4d(
			-b / 4.0 + 0.5 * w + 0.5 * Math.Sqrt(-3.0 * alpha - 2.0 * y - 2.0 * beta / w),
			-b / 4.0 + 0.5 * w - 0.5 * Math.Sqrt(-3.0 * alpha - 2.0 * y - 2.0 * beta / w),
			-b / 4.0 - 0.5 * w + 0.5 * Math.Sqrt(-3.0 * alpha - 2.0 * y + 2.0 * beta / w),
			-b / 4.0 - 0.5 * w - 0.5 * Math.Sqrt(-3.0 * alpha - 2.0 * y + 2.0 * beta / w));
	}

	private static double ComputeCost(Terms t, Vector3d rotation, int step)
	{
		Check.Ge(step, 0);
		Check.Le(step, 1);

		Vector4d roots = ComputeEigenValue(t, rotation);

		if (step == 0)
		{
			return roots.Z;
		}
		else if (step == 1)
		{
			return roots.W;
		}

		return 0;
	}

	private static Vector3d ComputeJacobian(Terms t, Vector3d rotation, double currentCost, int step)
	{
		const double kStepSize = 1e-8;
		Span<double> jacobian = stackalloc double[3];
		for (int j = 0; j < 3; j++)
		{
			Vector3d cayleyJ = new(
				rotation.X + (j == 0 ? kStepSize : 0),
				rotation.Y + (j == 1 ? kStepSize : 0),
				rotation.Z + (j == 2 ? kStepSize : 0));
			double costJ = ComputeCost(t, cayleyJ, step);
			jacobian[j] = costJ - currentCost;
		}

		return new Vector3d(jacobian[0], jacobian[1], jacobian[2]);
	}
	/// <summary>
	/// The sums GR8PEstimator::Estimate accumulates over the correspondences (xxF ... zxF,
	/// x1P ... z2P, m11P, m12P, m22P) that ComposeG combines with a rotation. The 3x9 and
	/// 9x9 matrices are stored row-major.
	/// </summary>
	private sealed class Terms
	{
		public Matrix3d XxF = Matrix3d.Zero;
		public Matrix3d YyF = Matrix3d.Zero;
		public Matrix3d ZzF = Matrix3d.Zero;
		public Matrix3d XyF = Matrix3d.Zero;
		public Matrix3d YzF = Matrix3d.Zero;
		public Matrix3d ZxF = Matrix3d.Zero;

		public readonly double[] X1P = new double[27];
		public readonly double[] Y1P = new double[27];
		public readonly double[] Z1P = new double[27];
		public readonly double[] X2P = new double[27];
		public readonly double[] Y2P = new double[27];
		public readonly double[] Z2P = new double[27];

		public readonly double[] M11P = new double[81];
		public readonly double[] M12P = new double[81];
		public readonly double[] M22P = new double[81];

		/// <summary>Adds one correspondence: rig rays f1, f2 and ray origins t1, t2.</summary>
		public void Accumulate(Vector3d f1, Vector3d f2, Vector3d t1, Vector3d t2)
		{
			Matrix3d f = Outer(f2, f2);
			XxF += f1[0] * f1[0] * f;
			YyF += f1[1] * f1[1] * f;
			ZzF += f1[2] * f1[2] * f;
			XyF += f1[0] * f1[1] * f;
			YzF += f1[1] * f1[2] * f;
			ZxF += f1[2] * f1[0] * f;

			Span<double> ff1 = stackalloc double[9];
			ff1[0] = f1[0] * (f2[1] * t2[2] - f2[2] * t2[1]);
			ff1[1] = f1[1] * (f2[1] * t2[2] - f2[2] * t2[1]);
			ff1[2] = f1[2] * (f2[1] * t2[2] - f2[2] * t2[1]);
			ff1[3] = f1[0] * (f2[2] * t2[0] - f2[0] * t2[2]);
			ff1[4] = f1[1] * (f2[2] * t2[0] - f2[0] * t2[2]);
			ff1[5] = f1[2] * (f2[2] * t2[0] - f2[0] * t2[2]);
			ff1[6] = f1[0] * (f2[0] * t2[1] - f2[1] * t2[0]);
			ff1[7] = f1[1] * (f2[0] * t2[1] - f2[1] * t2[0]);
			ff1[8] = f1[2] * (f2[0] * t2[1] - f2[1] * t2[0]);

			AddScaledOuter(X1P, f1[0] * f2, ff1);
			AddScaledOuter(Y1P, f1[1] * f2, ff1);
			AddScaledOuter(Z1P, f1[2] * f2, ff1);

			Span<double> ff2 = stackalloc double[9];
			ff2[0] = f2[0] * (f1[1] * t1[2] - f1[2] * t1[1]);
			ff2[1] = f2[1] * (f1[1] * t1[2] - f1[2] * t1[1]);
			ff2[2] = f2[2] * (f1[1] * t1[2] - f1[2] * t1[1]);
			ff2[3] = f2[0] * (f1[2] * t1[0] - f1[0] * t1[2]);
			ff2[4] = f2[1] * (f1[2] * t1[0] - f1[0] * t1[2]);
			ff2[5] = f2[2] * (f1[2] * t1[0] - f1[0] * t1[2]);
			ff2[6] = f2[0] * (f1[0] * t1[1] - f1[1] * t1[0]);
			ff2[7] = f2[1] * (f1[0] * t1[1] - f1[1] * t1[0]);
			ff2[8] = f2[2] * (f1[0] * t1[1] - f1[1] * t1[0]);

			AddScaledOuter(X2P, f1[0] * f2, ff2);
			AddScaledOuter(Y2P, f1[1] * f2, ff2);
			AddScaledOuter(Z2P, f1[2] * f2, ff2);

			// m11P -= ff1 ff1^T, m22P -= ff2 ff2^T, m12P -= ff2 ff1^T.
			for (int r = 0; r < 9; r++)
			{
				for (int c = 0; c < 9; c++)
				{
					M11P[r * 9 + c] -= ff1[r] * ff1[c];
					M22P[r * 9 + c] -= ff2[r] * ff2[c];
					M12P[r * 9 + c] -= ff2[r] * ff1[c];
				}
			}
		}

		/// <summary>m (3x9) += a * b^T.</summary>
		private static void AddScaledOuter(double[] m, Vector3d a, ReadOnlySpan<double> b)
		{
			for (int r = 0; r < 3; r++)
			{
				for (int c = 0; c < 9; c++)
				{
					m[r * 9 + c] += a[r] * b[c];
				}
			}
		}

		/// <summary>a * b^T.</summary>
		public static Matrix3d Outer(Vector3d a, Vector3d b) => new(
			a.X * b.X, a.X * b.Y, a.X * b.Z,
			a.Y * b.X, a.Y * b.Y, a.Y * b.Z,
			a.Z * b.X, a.Z * b.Y, a.Z * b.Z);

		/// <summary>The 3x9 matrix <paramref name="m"/> times the 9-vector <paramref name="v"/>.</summary>
		public static Vector3d Mul3x9(double[] m, ReadOnlySpan<double> v)
		{
			Span<double> result = stackalloc double[3];
			for (int r = 0; r < 3; r++)
			{
				double sum = 0;
				for (int c = 0; c < 9; c++)
				{
					sum += m[r * 9 + c] * v[c];
				}

				result[r] = sum;
			}

			return new Vector3d(result[0], result[1], result[2]);
		}

		/// <summary>((scale * a)^T M) . b for 9-vectors a, b and the 9x9 matrix M.</summary>
		public static double Quad9(double scale, ReadOnlySpan<double> a, double[] m, ReadOnlySpan<double> b)
		{
			double result = 0;
			for (int c = 0; c < 9; c++)
			{
				double rowTimesM = 0;
				for (int r = 0; r < 9; r++)
				{
					rowTimesM += scale * a[r] * m[r * 9 + c];
				}

				result += rowTimesM * b[c];
			}

			return result;
		}
	}
}
