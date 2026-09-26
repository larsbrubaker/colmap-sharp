// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// EigenSolver.Vectors: eigenvectors from the real Schur form T = Z^T A Z (EigenSolver.cs).
// For each eigenvalue lambda at diagonal block k, the eigenvector y of T is 1 (real) or the
// 2x2 block's own eigenvector (complex pair) at the block, zero below, and above it the
// back-substitution of (T - lambda I) y = 0 block by block (1x1 blocks divide, 2x2 blocks
// solve a 2x2 complex system by Cramer's rule) - Golub & Van Loan, "Matrix Computations",
// 4th ed., §7.6.4 (eigenvectors of a quasi-triangular matrix). The eigenvector of A is
// Z y, normalized to unit norm. A zero pivot (a repeated eigenvalue) is replaced by
// epsilon * ||T||, the perturbation of EISPACK's hqr2 (Smith et al., "Matrix Eigensystem
// Routines - EISPACK Guide", Lecture Notes in Computer Science 6, Springer, 1976; EISPACK
// is freely distributable public software, and only this published idea is used, no
// code). Written from G&VL and that; Eigen (MPL-2.0) is not ported.

using System.Numerics;

namespace ColmapSharp.LinearAlgebra;

public sealed partial class EigenSolver
{
	private static Complex[,] ComputeEigenvectors(MatrixXd t, MatrixXd z, Complex[] eigenvalues)
	{
		int n = t.Rows;
		var vectors = new Complex[n, n];
		// t is scaled to max |entry| = 1, so its norm is 0 only for the zero matrix.
		double tNorm = t.Norm();
		double smallPivot = LinearAlgebraConstants.MachineEpsilon * (tNorm > 0 ? tNorm : 1);
		var y = new Complex[n];
		for (int k = 0; k < n; k++)
		{
			Complex lambda = eigenvalues[k];
			bool complexBlock = k < n - 1 && t[k + 1, k] != 0;
			Array.Clear(y);
			int top;
			if (complexBlock)
			{
				// (B - lambda I) u = 0 for the block B = [a b; c d]: u = (b, lambda - a).
				y[k] = t[k, k + 1];
				y[k + 1] = lambda - t[k, k];
				top = k - 1;
			}
			else
			{
				y[k] = 1;
				top = k - 1;
			}

			int end = complexBlock ? k + 1 : k;
			BackSubstitute(t, lambda, y, top, end, smallPivot);
			Complex[] column = new Complex[n];
			double squaredNorm = 0;
			for (int r = 0; r < n; r++)
			{
				Complex sum = Complex.Zero;
				for (int c = 0; c <= end; c++)
				{
					sum += z[r, c] * y[c];
				}

				column[r] = sum;
				squaredNorm += sum.Real * sum.Real + sum.Imaginary * sum.Imaginary;
			}

			double norm = Math.Sqrt(squaredNorm);
			for (int r = 0; r < n; r++)
			{
				vectors[r, k] = norm > 0 ? column[r] / norm : column[r];
			}

			if (complexBlock)
			{
				for (int r = 0; r < n; r++)
				{
					vectors[r, k + 1] = Complex.Conjugate(vectors[r, k]);
				}

				k++;
			}
		}

		return vectors;
	}

	/// <summary>
	/// Solves rows top..0 of (T - lambda I) y = 0 given y[top+1..end], block by block.
	/// </summary>
	private static void BackSubstitute(MatrixXd t, Complex lambda, Complex[] y, int top, int end, double smallPivot)
	{
		int i = top;
		while (i >= 0)
		{
			bool block = i > 0 && t[i, i - 1] != 0;
			if (!block)
			{
				Complex rhs = -RowSum(t, y, i, i + 1, end);
				Complex pivot = t[i, i] - lambda;
				if (pivot == Complex.Zero)
				{
					pivot = smallPivot;
				}

				y[i] = rhs / pivot;
				i--;
			}
			else
			{
				int r0 = i - 1;
				Complex rhs0 = -RowSum(t, y, r0, i + 1, end);
				Complex rhs1 = -RowSum(t, y, i, i + 1, end);
				Complex a = t[r0, r0] - lambda;
				Complex b = t[r0, i];
				Complex c = t[i, r0];
				Complex d = t[i, i] - lambda;
				Complex det = a * d - b * c;
				if (det == Complex.Zero)
				{
					det = smallPivot;
				}

				y[r0] = (rhs0 * d - b * rhs1) / det;
				y[i] = (a * rhs1 - c * rhs0) / det;
				i -= 2;
			}
		}
	}

	private static Complex RowSum(MatrixXd t, Complex[] y, int row, int from, int to)
	{
		Complex sum = Complex.Zero;
		for (int j = from; j <= to; j++)
		{
			sum += t[row, j] * y[j];
		}

		return sum;
	}
}
