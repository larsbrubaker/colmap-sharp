// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Epnp: the EPNPEstimator of colmap/estimators/solvers/absolute_pose.h and .cc (the rest of
// that file is AbsolutePose.cs). EPnP needs four or more 2D-3D correspondences.
// Tests: ColmapSharp.Tests/Estimators/Solvers/AbsolutePoseTests.cs (AbsolutePose.EPNP,
// AbsolutePose.EPNP_BrokenSolveSignCase).
//
// The algorithm is based on the following paper:
//
//    Lepetit, Vincent, Francesc Moreno-Noguer, and Pascal Fua.
//    "Epnp: An accurate o (n) solution to the pnp problem."
//    International journal of computer vision 81.2 (2009): 155-166.
//
// COLMAP's implementation is based on the authors' original open-source release, ported to
// Eigen with several improvements; this is a port of COLMAP's.
//
// Tier B: the pose goes through several SVDs and a QR.
//
// Translation notes:
// - COLMAP keeps the working state (points, control points, alphas, camera-frame points) in
//   members of the estimator, so an EPNPEstimator cannot run two estimates at once. Here it
//   lives in a private EpnpSolver created per Estimate, so the estimator is an immutable
//   struct that RANSAC may copy freely; the arithmetic is unchanged.
// - Eigen::JacobiSVD becomes Svd3d for the 3x3 cases and JacobiSVD for the rest.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators.Solvers;

/// <summary>
/// EPNP solver for the PNP (Perspective-N-Point) problem. Port of colmap::EPNPEstimator.
/// </summary>
public readonly struct EPNPEstimator
	: IEstimator<Point2DWithRay, Vector3d, Matrix3x4d>, ILocalEstimator<Point2DWithRay, Vector3d, Matrix3x4d>
{
	private readonly ImgFromCamFunc _imgFromCamFunc;

	/// <summary>An estimator scoring in pixels through <paramref name="imgFromCamFunc"/>.</summary>
	public EPNPEstimator(ImgFromCamFunc imgFromCamFunc)
	{
		_imgFromCamFunc = imgFromCamFunc;
	}

	/// <summary>The minimum number of samples needed to estimate a model.</summary>
	public static int MinNumSamples => 4;

	/// <summary>
	/// Estimate the most probable solution of the EPNP problem from a set of four or more
	/// 2D-3D point correspondences.
	/// </summary>
	public void Estimate(ReadOnlySpan<Point2DWithRay> points2D, ReadOnlySpan<Vector3d> points3D, List<Matrix3x4d> camsFromWorld)
	{
		Check.Ge(points2D.Length, 4);
		Check.Eq(points2D.Length, points3D.Length);

		camsFromWorld.Clear();

		var solver = new EpnpSolver(points2D.ToArray(), points3D.ToArray(), _imgFromCamFunc);
		if (!solver.ComputePose(out Matrix3x4d camFromWorld))
		{
			return;
		}

		camsFromWorld.Add(camFromWorld);
	}

	/// <summary>
	/// LO-RANSAC's local estimate (EstimateAbsolutePose uses EPnP as P3P's local optimizer):
	/// EPNPEstimator has no Refine, so this re-estimates from the inliers.
	/// </summary>
	public void EstimateLocal(ReadOnlySpan<Point2DWithRay> points2D, ReadOnlySpan<Vector3d> points3D, in Matrix3x4d initialModel, List<Matrix3x4d> camsFromWorld) =>
		Estimate(points2D, points3D, camsFromWorld);

	/// <summary>
	/// Calculate the squared reprojection error given a set of 2D-3D point correspondences
	/// and a projection matrix.
	/// </summary>
	public void Residuals(ReadOnlySpan<Point2DWithRay> points2D, ReadOnlySpan<Vector3d> points3D, in Matrix3x4d camFromWorld, Span<double> residuals)
	{
		AbsolutePose.ComputeSquaredReprojectionError(points2D, points3D, camFromWorld, _imgFromCamFunc, residuals);
	}

	// The working state and steps of one EPnP estimate (COLMAP's private members).
	private sealed class EpnpSolver
	{
		private readonly Point2DWithRay[] _points2D;
		private readonly Vector3d[] _points3D;
		private readonly ImgFromCamFunc _imgFromCamFunc;
		private readonly Vector3d[] _pcs;
		private readonly double[][] _alphas;
		private readonly Vector3d[] _cws = new Vector3d[4];
		private readonly double[][] _ccs = [new double[3], new double[3], new double[3], new double[3]];

		public EpnpSolver(Point2DWithRay[] points2D, Vector3d[] points3D, ImgFromCamFunc imgFromCamFunc)
		{
			_points2D = points2D;
			_points3D = points3D;
			_imgFromCamFunc = imgFromCamFunc;
			_pcs = new Vector3d[points2D.Length];
			_alphas = new double[points2D.Length][];
			for (int i = 0; i < _alphas.Length; ++i)
			{
				_alphas[i] = new double[4];
			}
		}

		public bool ComputePose(out Matrix3x4d camFromWorld)
		{
			camFromWorld = default;
			ChooseControlPoints();

			if (!ComputeBarycentricCoordinates())
			{
				return false;
			}

			MatrixXd m = ComputeM();
			MatrixXd mtM = m.TransposeTimesSelf();

			var svd = new JacobiSVD(mtM, SvdOptions.ComputeFullV | SvdOptions.ComputeFullU);
			MatrixXd ut = svd.MatrixU().Transpose();

			double[,] l6x10 = ComputeL6x10(ut);
			double[] rho = ComputeRho();

			var betas = new double[4][];
			var reprojErrors = new double[4];
			var rs = new Matrix3d[4];
			var ts = new Vector3d[4];

			betas[1] = FindBetasApprox1(l6x10, rho);
			RunGaussNewton(l6x10, rho, betas[1]);
			reprojErrors[1] = ComputeRT(ut, betas[1], out rs[1], out ts[1]);

			betas[2] = FindBetasApprox2(l6x10, rho);
			RunGaussNewton(l6x10, rho, betas[2]);
			reprojErrors[2] = ComputeRT(ut, betas[2], out rs[2], out ts[2]);

			betas[3] = FindBetasApprox3(l6x10, rho);
			RunGaussNewton(l6x10, rho, betas[3]);
			reprojErrors[3] = ComputeRT(ut, betas[3], out rs[3], out ts[3]);

			int bestIdx = 1;
			if (reprojErrors[2] < reprojErrors[1])
			{
				bestIdx = 2;
			}

			if (reprojErrors[3] < reprojErrors[bestIdx])
			{
				bestIdx = 3;
			}

			camFromWorld = Matrix3x4d.FromBlocks(rs[bestIdx], ts[bestIdx]);

			return true;
		}

		private void ChooseControlPoints()
		{
			int n = _points3D.Length;

			// Take C0 as the reference points centroid:
			Vector3d cw0 = Vector3d.Zero;
			for (int i = 0; i < n; ++i)
			{
				cw0 += _points3D[i];
			}

			cw0 /= n;
			_cws[0] = cw0;

			// PW0^T PW0 with PW0 the centered points as rows.
			var pw0 = new MatrixXd(n, 3);
			for (int i = 0; i < n; ++i)
			{
				Vector3d d = _points3D[i] - cw0;
				pw0[i, 0] = d.X;
				pw0[i, 1] = d.Y;
				pw0[i, 2] = d.Z;
			}

			MatrixXd pw0tPw0 = pw0.TransposeTimesSelf();
			Span<double> buffer = stackalloc double[9];
			pw0tPw0.AsSpan().CopyTo(buffer);
			Svd3d svd = Svd3d.Compute(Matrix3d.FromColumnMajor(buffer));
			Vector3d dv = svd.SingularValues;
			Matrix3d ut = svd.MatrixU.Transpose();

			for (int i = 1; i < 4; ++i)
			{
				double k = Math.Sqrt(dv[i - 1] / n);
				_cws[i] = cw0 + k * ut.Row(i - 1);
			}
		}

		private bool ComputeBarycentricCoordinates()
		{
			var cc = new MatrixXd(3, 3);
			for (int i = 0; i < 3; ++i)
			{
				for (int j = 1; j < 4; ++j)
				{
					cc[i, j - 1] = _cws[j][i] - _cws[0][i];
				}
			}

			if (new ColPivHouseholderQR(cc).Rank() < 3)
			{
				return false;
			}

			Span<double> buffer = stackalloc double[9];
			cc.AsSpan().CopyTo(buffer);
			Matrix3d ccInv = Matrix3d.FromColumnMajor(buffer).Inverse();

			for (int i = 0; i < _points3D.Length; ++i)
			{
				double[] alpha = _alphas[i];
				Vector3d p = _points3D[i];
				for (int j = 0; j < 3; ++j)
				{
					alpha[1 + j] = ccInv[j, 0] * (p.X - _cws[0].X) +
						ccInv[j, 1] * (p.Y - _cws[0].Y) +
						ccInv[j, 2] * (p.Z - _cws[0].Z);
				}

				alpha[0] = 1.0 - alpha[1] - alpha[2] - alpha[3];
			}

			return true;
		}

		private MatrixXd ComputeM()
		{
			var m = new MatrixXd(3 * _points2D.Length, 12);
			for (int i = 0; i < _points3D.Length; ++i)
			{
				Vector3d ray = _points2D[i].CameraRay;
				double[] alpha = _alphas[i];
				for (int j = 0; j < 4; ++j)
				{
					m[3 * i, 3 * j] = 0.0;
					m[3 * i, 3 * j + 1] = -alpha[j] * ray.Z;
					m[3 * i, 3 * j + 2] = alpha[j] * ray.Y;

					m[3 * i + 1, 3 * j] = alpha[j] * ray.Z;
					m[3 * i + 1, 3 * j + 1] = 0.0;
					m[3 * i + 1, 3 * j + 2] = -alpha[j] * ray.X;

					m[3 * i + 2, 3 * j] = -alpha[j] * ray.Y;
					m[3 * i + 2, 3 * j + 1] = alpha[j] * ray.X;
					m[3 * i + 2, 3 * j + 2] = 0;
				}
			}

			return m;
		}

		private static double[,] ComputeL6x10(MatrixXd ut)
		{
			var l6x10 = new double[6, 10];

			var dv = new Vector3d[4, 6];
			for (int i = 0; i < 4; ++i)
			{
				int a = 0, b = 1;
				for (int j = 0; j < 6; ++j)
				{
					dv[i, j] = new Vector3d(
						ut[11 - i, 3 * a] - ut[11 - i, 3 * b],
						ut[11 - i, 3 * a + 1] - ut[11 - i, 3 * b + 1],
						ut[11 - i, 3 * a + 2] - ut[11 - i, 3 * b + 2]);

					b += 1;
					if (b > 3)
					{
						a += 1;
						b = a + 1;
					}
				}
			}

			for (int i = 0; i < 6; ++i)
			{
				l6x10[i, 0] = dv[0, i].Dot(dv[0, i]);
				l6x10[i, 1] = 2.0 * dv[0, i].Dot(dv[1, i]);
				l6x10[i, 2] = dv[1, i].Dot(dv[1, i]);
				l6x10[i, 3] = 2.0 * dv[0, i].Dot(dv[2, i]);
				l6x10[i, 4] = 2.0 * dv[1, i].Dot(dv[2, i]);
				l6x10[i, 5] = dv[2, i].Dot(dv[2, i]);
				l6x10[i, 6] = 2.0 * dv[0, i].Dot(dv[3, i]);
				l6x10[i, 7] = 2.0 * dv[1, i].Dot(dv[3, i]);
				l6x10[i, 8] = 2.0 * dv[2, i].Dot(dv[3, i]);
				l6x10[i, 9] = dv[3, i].Dot(dv[3, i]);
			}

			return l6x10;
		}

		private double[] ComputeRho() =>
		[
			(_cws[0] - _cws[1]).SquaredNorm,
			(_cws[0] - _cws[2]).SquaredNorm,
			(_cws[0] - _cws[3]).SquaredNorm,
			(_cws[1] - _cws[2]).SquaredNorm,
			(_cws[1] - _cws[3]).SquaredNorm,
			(_cws[2] - _cws[3]).SquaredNorm,
		];

		// Least-squares solution of L6x10[:, cols] b = rho by a full SVD (JacobiSVD::solve).
		private static VectorXd SolveColumns(double[,] l6x10, double[] rho, ReadOnlySpan<int> cols)
		{
			var l = new MatrixXd(6, cols.Length);
			for (int i = 0; i < 6; ++i)
			{
				for (int j = 0; j < cols.Length; ++j)
				{
					l[i, j] = l6x10[i, cols[j]];
				}
			}

			var svd = new JacobiSVD(l, SvdOptions.ComputeFullV | SvdOptions.ComputeFullU);
			return svd.Solve(new VectorXd(rho));
		}

		// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
		// betas_approx_1 = [B11 B12     B13         B14]
		private static double[] FindBetasApprox1(double[,] l6x10, double[] rho)
		{
			VectorXd b4 = SolveColumns(l6x10, rho, [0, 1, 3, 6]);
			var betas = new double[4];

			if (b4[0] < 0)
			{
				betas[0] = Math.Sqrt(-b4[0]);
				betas[1] = -b4[1] / betas[0];
				betas[2] = -b4[2] / betas[0];
				betas[3] = -b4[3] / betas[0];
			}
			else
			{
				betas[0] = Math.Sqrt(b4[0]);
				betas[1] = b4[1] / betas[0];
				betas[2] = b4[2] / betas[0];
				betas[3] = b4[3] / betas[0];
			}

			return betas;
		}

		// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
		// betas_approx_2 = [B11 B12 B22                            ]
		private static double[] FindBetasApprox2(double[,] l6x10, double[] rho)
		{
			VectorXd b3 = SolveColumns(l6x10, rho, [0, 1, 2]);
			var betas = new double[4];

			if (b3[0] < 0)
			{
				betas[0] = Math.Sqrt(-b3[0]);
				betas[1] = (b3[2] < 0) ? Math.Sqrt(-b3[2]) : 0.0;
			}
			else
			{
				betas[0] = Math.Sqrt(b3[0]);
				betas[1] = (b3[2] > 0) ? Math.Sqrt(b3[2]) : 0.0;
			}

			if (b3[1] < 0)
			{
				betas[0] = -betas[0];
			}

			betas[2] = 0.0;
			betas[3] = 0.0;
			return betas;
		}

		// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
		// betas_approx_3 = [B11 B12 B22 B13 B23                    ]
		private static double[] FindBetasApprox3(double[,] l6x10, double[] rho)
		{
			VectorXd b5 = SolveColumns(l6x10, rho, [0, 1, 2, 3, 4]);
			var betas = new double[4];

			if (b5[0] < 0)
			{
				betas[0] = Math.Sqrt(-b5[0]);
				betas[1] = (b5[2] < 0) ? Math.Sqrt(-b5[2]) : 0.0;
			}
			else
			{
				betas[0] = Math.Sqrt(b5[0]);
				betas[1] = (b5[2] > 0) ? Math.Sqrt(b5[2]) : 0.0;
			}

			if (b5[1] < 0)
			{
				betas[0] = -betas[0];
			}

			betas[2] = b5[3] / betas[0];
			betas[3] = 0.0;
			return betas;
		}

		private static void RunGaussNewton(double[,] l, double[] rho, double[] betas)
		{
			var a = new MatrixXd(6, 4);
			var b = new VectorXd(6);

			const int NumIterations = 5;
			for (int k = 0; k < NumIterations; ++k)
			{
				for (int i = 0; i < 6; ++i)
				{
					a[i, 0] = 2 * l[i, 0] * betas[0] + l[i, 1] * betas[1] + l[i, 3] * betas[2] + l[i, 6] * betas[3];
					a[i, 1] = l[i, 1] * betas[0] + 2 * l[i, 2] * betas[1] + l[i, 4] * betas[2] + l[i, 7] * betas[3];
					a[i, 2] = l[i, 3] * betas[0] + l[i, 4] * betas[1] + 2 * l[i, 5] * betas[2] + l[i, 8] * betas[3];
					a[i, 3] = l[i, 6] * betas[0] + l[i, 7] * betas[1] + l[i, 8] * betas[2] + 2 * l[i, 9] * betas[3];

					b[i] = rho[i] - (l[i, 0] * betas[0] * betas[0] +
						l[i, 1] * betas[0] * betas[1] +
						l[i, 2] * betas[1] * betas[1] +
						l[i, 3] * betas[0] * betas[2] +
						l[i, 4] * betas[1] * betas[2] +
						l[i, 5] * betas[2] * betas[2] +
						l[i, 6] * betas[0] * betas[3] +
						l[i, 7] * betas[1] * betas[3] +
						l[i, 8] * betas[2] * betas[3] +
						l[i, 9] * betas[3] * betas[3]);
				}

				VectorXd x = new ColPivHouseholderQR(a).Solve(b);

				for (int j = 0; j < 4; ++j)
				{
					betas[j] += x[j];
				}
			}
		}

		private double ComputeRT(MatrixXd ut, double[] betas, out Matrix3d r, out Vector3d t)
		{
			ComputeCcs(betas, ut);
			ComputePcs();

			SolveForSign();

			EstimateRT(out r, out t);

			return ComputeTotalError(r, t);
		}

		private void ComputeCcs(double[] betas, MatrixXd ut)
		{
			for (int i = 0; i < 4; ++i)
			{
				_ccs[i][0] = _ccs[i][1] = _ccs[i][2] = 0.0;
			}

			for (int i = 0; i < 4; ++i)
			{
				for (int j = 0; j < 4; ++j)
				{
					for (int k = 0; k < 3; ++k)
					{
						_ccs[j][k] += betas[i] * ut[11 - i, 3 * j + k];
					}
				}
			}
		}

		private void ComputePcs()
		{
			for (int i = 0; i < _points3D.Length; ++i)
			{
				double[] alpha = _alphas[i];
				_pcs[i] = new Vector3d(Pc(alpha, 0), Pc(alpha, 1), Pc(alpha, 2));
			}
		}

		private double Pc(double[] alpha, int j) =>
			alpha[0] * _ccs[0][j] + alpha[1] * _ccs[1][j] + alpha[2] * _ccs[2][j] + alpha[3] * _ccs[3][j];

		private void SolveForSign()
		{
			if (_pcs[0].Z < 0.0)
			{
				for (int i = 0; i < 4; ++i)
				{
					for (int k = 0; k < 3; ++k)
					{
						_ccs[i][k] = -_ccs[i][k];
					}
				}

				for (int i = 0; i < _points3D.Length; ++i)
				{
					_pcs[i] = -_pcs[i];
				}
			}
		}

		private void EstimateRT(out Matrix3d r, out Vector3d t)
		{
			Vector3d pc0 = Vector3d.Zero;
			Vector3d pw0 = Vector3d.Zero;

			for (int i = 0; i < _points3D.Length; ++i)
			{
				pc0 += _pcs[i];
				pw0 += _points3D[i];
			}

			pc0 /= _points3D.Length;
			pw0 /= _points3D.Length;

			Span<double> abt = stackalloc double[9]; // column-major 3x3
			for (int i = 0; i < _points3D.Length; ++i)
			{
				for (int j = 0; j < 3; ++j)
				{
					abt[j] += (_pcs[i][j] - pc0[j]) * (_points3D[i].X - pw0.X);
					abt[j + 3] += (_pcs[i][j] - pc0[j]) * (_points3D[i].Y - pw0.Y);
					abt[j + 6] += (_pcs[i][j] - pc0[j]) * (_points3D[i].Z - pw0.Z);
				}
			}

			Svd3d svd = Svd3d.Compute(Matrix3d.FromColumnMajor(abt));
			Matrix3d abtU = svd.MatrixU;
			Matrix3d abtV = svd.MatrixV;

			r = RowProducts(abtU, abtV);

			if (r.Determinant() < 0)
			{
				Matrix3d abtVPrime = Matrix3d.FromColumns(abtV.Col(0), abtV.Col(1), -abtV.Col(2));
				r = RowProducts(abtU, abtVPrime);
			}

			t = pc0 - r * pw0;
		}

		// R(i, j) = U.row(i) * V.row(j)^T.
		private static Matrix3d RowProducts(Matrix3d u, Matrix3d v)
		{
			Span<double> m = stackalloc double[9];
			for (int i = 0; i < 3; ++i)
			{
				for (int j = 0; j < 3; ++j)
				{
					m[i + 3 * j] = u.Row(i).Dot(v.Row(j));
				}
			}

			return Matrix3d.FromColumnMajor(m);
		}

		private double ComputeTotalError(Matrix3d r, Vector3d t)
		{
			Matrix3x4d camFromWorld = Matrix3x4d.FromBlocks(r, t);

			Span<double> residuals = _points2D.Length <= 256 ? stackalloc double[_points2D.Length] : new double[_points2D.Length];
			AbsolutePose.ComputeSquaredReprojectionError(_points2D, _points3D, camFromWorld, _imgFromCamFunc, residuals);

			double error = 0.0;
			foreach (double residual in residuals)
			{
				error += Math.Sqrt(residual);
			}

			return error;
		}
	}
}
