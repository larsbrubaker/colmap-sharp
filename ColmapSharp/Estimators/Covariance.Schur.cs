// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Covariance.Schur: the numeric half of colmap/estimators/covariance.cc (the anonymous
// namespace): Schur elimination of the point parameters from the Jacobian's normal equations,
// optional elimination of the "other" parameters, and the inverse Cholesky factor L_inv of
// the remaining Schur complement S, so that S^-1 = L_inv^T L_inv. The public API is in
// Covariance.cs.
//
// Eigen's sparse matrices and SimplicialLLT/SimplicialLDLT are replaced by
// LinearAlgebra/SparseMatrixCsc.cs and LinearAlgebra/SimplicialCholesky.cs (written from the
// published algorithms; Eigen is MPL-2.0 and not ported). The matrices, the fill-reducing
// AMD ordering's role, the rank test and the damping are COLMAP's; the operation order inside
// the products and the factorization differs, so the results agree to round-off (Tier B,
// divergence 54). As in COLMAP, S is sparse and L_inv is dense
// (pose/other parameters squared).

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;
using ColmapSharp.Util;

namespace ColmapSharp.Estimators;

/// <content>Schur elimination and the inverse factor.</content>
public static partial class BACovarianceEstimation
{
	/// <summary>
	/// Smallest positive normal double (std::numeric_limits&lt;double&gt;::min()), which
	/// ComputeLInverse divides by instead of zero. C#'s double.Epsilon is the smallest
	/// subnormal and would be wrong here.
	/// </summary>
	private const double DoubleMinNormal = 2.2250738585072014E-308;

	/// <summary>
	/// Evaluates the Jacobian over [poses, others, points] (poses and others only when their
	/// covariances are wanted), inverts each point's damped 3x3 Hessian block (the point
	/// covariances conditioned on everything else being constant) and, when pose or other
	/// covariances are wanted, returns the Schur complement S = H_aa - H_ap H_pp^-1 H_pa.
	/// </summary>
	private static bool ComputeSchurComplement(
		bool estimatePointCovs,
		bool estimatePoseCovs,
		bool estimateOtherCovs,
		double damping,
		int pointNumParams,
		List<PointParam> points,
		List<PoseParam> poses,
		List<ArraySegment<double>> others,
		Problem problem,
		Dictionary<ulong, MatrixXd> pointCovs,
		out SparseMatrixCsc? s)
	{
		s = null;
		var evalOptions = new ProblemEvaluateOptions();
		if (estimatePoseCovs || estimateOtherCovs)
		{
			foreach (PoseParam pose in poses)
			{
				evalOptions.ParameterBlocks.Add(pose.CamFromWorld);
			}

			evalOptions.ParameterBlocks.AddRange(others);
		}

		foreach (PointParam point in points)
		{
			evalOptions.ParameterBlocks.Add(point.Xyz);
		}

		var jFullCrs = new CRSMatrix();
		if (!problem.Evaluate(evalOptions, out _, null, null, jFullCrs))
		{
			Log.Warning("Failed to evaluate Jacobian");
			return false;
		}

		// A CRS matrix's arrays are the CSC arrays of its transpose.
		SparseMatrixCsc jFullTransposed = SparseMatrixCsc.FromCsc(
			jFullCrs.NumCols,
			jFullCrs.NumRows,
			jFullCrs.Rows.ToArray(),
			jFullCrs.Cols.ToArray(),
			jFullCrs.Values.ToArray());
		SparseMatrixCsc jFull = jFullTransposed.Transpose();

		if (points.Count == 0)
		{
			s = jFullTransposed * jFull;
			return true;
		}

		// Notice that here "a" refers to pose/other and "p" to point parameters.
		int numAParams = jFull.Cols - pointNumParams;
		SparseMatrixCsc jA = jFull.Block(0, 0, jFull.Rows, numAParams);
		SparseMatrixCsc jP = jFull.Block(0, numAParams, jFull.Rows, pointNumParams);
		SparseMatrixCsc jPTransposed = jP.Transpose();
		SparseMatrixCsc hPp = jPTransposed * jP;

		// COLMAP overwrites each point's diagonal block of H_pp in place with its inverse,
		// leaving any entry outside those blocks as it is. The blocks advance by 3 whatever the
		// tangent size, as in COLMAP; point blocks have no manifold in bundle adjustment, so
		// their tangent size is always 3.
		var blockOfColumn = new int[pointNumParams];
		Array.Fill(blockOfColumn, -1);
		var inverseBlocks = new List<(int Start, MatrixXd Inverse)>(points.Count);
		int pointParamIdx = 0;
		foreach (PointParam point in points)
		{
			int tangentSize = problem.ParameterBlockTangentSize(point.Xyz);
			var hPpIdx = new MatrixXd(tangentSize, tangentSize);
			for (int j = 0; j < tangentSize; j++)
			{
				for (int i = 0; i < tangentSize; i++)
				{
					hPpIdx[i, j] = hPp[pointParamIdx + i, pointParamIdx + j] + (i == j ? damping : 0.0);
				}
			}

			MatrixXd hPpIdxInv = hPpIdx.Inverse();
			for (int i = 0; i < tangentSize; i++)
			{
				blockOfColumn[pointParamIdx + i] = inverseBlocks.Count;
			}

			inverseBlocks.Add((pointParamIdx, hPpIdxInv));
			if (estimatePointCovs)
			{
				// Point covariance conditioned on fixed pose/other parameters.
				pointCovs.TryAdd(point.Point3DId, hPpIdxInv);
			}

			pointParamIdx += 3;
		}

		if (!estimatePoseCovs && !estimateOtherCovs)
		{
			return true;
		}

		var hPpInvTriplets = new List<SparseTriplet>(hPp.NonZeros);
		hPp.ForEach((i, j, v) =>
		{
			if (blockOfColumn[i] < 0 || blockOfColumn[i] != blockOfColumn[j])
			{
				hPpInvTriplets.Add(new SparseTriplet(i, j, v));
			}
		});
		foreach ((int start, MatrixXd inverse) in inverseBlocks)
		{
			for (int j = 0; j < inverse.Cols; j++)
			{
				for (int i = 0; i < inverse.Rows; i++)
				{
					hPpInvTriplets.Add(new SparseTriplet(start + i, start + j, inverse[i, j]));
				}
			}
		}

		SparseMatrixCsc hPpInv = SparseMatrixCsc.FromTriplets(pointNumParams, pointNumParams, hPpInvTriplets);
		SparseMatrixCsc jATransposed = jA.Transpose();
		SparseMatrixCsc hAa = jATransposed * jA;
		SparseMatrixCsc hAp = jATransposed * jP;
		SparseMatrixCsc hPa = hAp.Transpose();

		// Eigen evaluates H_ap * H_pp_inv * H_pa left to right.
		s = hAa - (hAp * hPpInv) * hPa;
		return true;
	}

	/// <summary>
	/// Replaces S (over [poses, others]) with its Schur complement on the pose parameters:
	/// S_cc - S_co (S_oo + damping I)^-1 S_oc. Returns false if the damped S_oo is not
	/// positive definite.
	/// </summary>
	private static bool SchurEliminateOtherParams(double damping, int poseNumParams, int otherNumParams, ref SparseMatrixCsc s)
	{
		// Notice that here "c" refers to pose and "o" to other parameters.
		SparseMatrixCsc sCc = s.Block(0, 0, poseNumParams, poseNumParams);
		if (otherNumParams == 0)
		{
			// Eigen factors the empty S_oo successfully and the product is the zero matrix.
			s = sCc;
			return true;
		}

		SparseMatrixCsc sCo = s.Block(0, poseNumParams, poseNumParams, otherNumParams);
		SparseMatrixCsc sOc = sCo.Transpose();

		// COLMAP sets a structurally or numerically zero diagonal to damping and adds damping to
		// the others; both are the diagonal plus damping.
		SparseMatrixCsc sOo = s.Block(poseNumParams, poseNumParams, otherNumParams, otherNumParams).AddToDiagonal(damping);

		var lltSOo = new SimplicialCholesky(SimplicialCholeskyKind.LLT);
		lltSOo.Compute(sOo);
		if (lltSOo.Info != ComputationInfo.Success)
		{
			Log.Warning("Simplicial LLT for Schur elimination of other parameters failed");
			return false;
		}

		// Eigen's sparse solve with a sparse right-hand side returns a sparse matrix with the
		// exact zeros dropped.
		var solutionTriplets = new List<SparseTriplet>();
		var rhs = new VectorXd(otherNumParams);
		for (int j = 0; j < poseNumParams; j++)
		{
			Span<double> rhsSpan = rhs.AsSpan();
			rhsSpan.Clear();
			ReadOnlySpan<int> colPtr = sOc.ColPtr;
			ReadOnlySpan<int> rowIdx = sOc.RowIndices;
			for (int p = colPtr[j]; p < colPtr[j + 1]; p++)
			{
				rhsSpan[rowIdx[p]] = sOc.Values[p];
			}

			VectorXd x = lltSOo.Solve(rhs);
			for (int i = 0; i < otherNumParams; i++)
			{
				if (x[i] != 0.0)
				{
					solutionTriplets.Add(new SparseTriplet(i, j, x[i]));
				}
			}
		}

		SparseMatrixCsc solution = SparseMatrixCsc.FromTriplets(otherNumParams, poseNumParams, solutionTriplets);
		s = sCc - sCo * solution;
		return true;
	}

	/// <summary>
	/// Factors S = P^T L D L^T P and returns L_inv = D^-1/2 L^-1 P, so that
	/// S^-1 = L_inv^T L_inv. Returns false if the factorization fails or S is rank deficient
	/// (a pivot of magnitude at most 1e-6), which usually means an unfixed gauge.
	/// </summary>
	private static bool ComputeLInverse(SparseMatrixCsc s, out MatrixXd? lInv)
	{
		lInv = null;
		int n = s.Rows;
		if (n == 0)
		{
			lInv = new MatrixXd(0, 0);
			return true;
		}

		var ldltS = new SimplicialCholesky(SimplicialCholeskyKind.LDLT);
		ldltS.Compute(s);
		if (ldltS.Info != ComputationInfo.Success)
		{
			Log.Warning("Simplicial LDLT for computing L_inv failed");
			return false;
		}

		VectorXd dDense = ldltS.VectorD();
		int rank = 0;
		for (int i = 0; i < n; i++)
		{
			if (Math.Abs(dDense[i]) > 1e-6)
			{
				rank++;
			}
		}

		if (rank < n)
		{
			Log.Warning(
				"Unable to compute covariance. The Schur complement on pose/other parameters is rank deficient. "
				+ $"Number of columns: {n}, rank: {rank}. This is likely due to the pose/other parameters being "
				+ "underconstrained with Gauge ambiguity or other degeneracies.");
			return false;
		}

		SparseMatrixCsc lSparse = ldltS.MatrixL();
		ReadOnlySpan<int> lColPtr = lSparse.ColPtr;
		ReadOnlySpan<int> lRowIdx = lSparse.RowIndices;
		ReadOnlySpan<double> lValues = lSparse.Values;
		ReadOnlySpan<int> permutation = ldltS.Permutation;

		var rowScale = new double[n];
		for (int i = 0; i < n; i++)
		{
			rowScale[i] = 1.0 / Math.Max(Math.Sqrt(Math.Max(dDense[i], 0.0)), DoubleMinNormal);
		}

		// Column k of L^-1 is L^-1 e_k, zero above row k. Multiplying by the permutation on the
		// right moves it to column permutation[k], the original index of the permuted k.
		var result = new MatrixXd(n, n);
		for (int k = 0; k < n; k++)
		{
			Span<double> x = result.ColumnSpan(permutation[k]);
			x[k] = 1.0;
			for (int col = k; col < n; col++)
			{
				double xCol = x[col] / lValues[lColPtr[col]];
				x[col] = xCol;
				if (xCol == 0.0)
				{
					continue;
				}

				for (int p = lColPtr[col] + 1; p < lColPtr[col + 1]; p++)
				{
					x[lRowIdx[p]] -= lValues[p] * xCol;
				}
			}

			for (int i = k; i < n; i++)
			{
				x[i] *= rowScale[i];
			}
		}

		lInv = result;
		return true;
	}
}
