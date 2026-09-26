// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// QuaternionUtilsTests: colmap/estimators/cost_functions/quaternion_utils_test.cc ported
// 1:1, one method per gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances.
// Tests ColmapSharp/Estimators/CostFunctions/QuaternionUtils.cs (Tier A formulas; the checks
// are COLMAP's tolerances against Eigen products and finite differences).
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test, so each test starts with
// RandomUtils.SetPRNGSeed(0) and draws all its random inputs before its first await (the
// PRNG is per thread and an await may resume elsewhere); the checks then run in the same
// order as the C++ loop.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class QuaternionUtilsTests
{
	private const int NumTrials = 100;

	private static VectorXd Vec(Quaterniond q) => new([q.X, q.Y, q.Z, q.W]);

	private static Quaterniond FromVec(VectorXd v) => new(v[3], v[0], v[1], v[2]);

	private static (Quaterniond[] Q, Quaterniond[] P) DrawQuaternionPairs()
	{
		RandomUtils.SetPRNGSeed(0);
		var q = new Quaterniond[NumTrials];
		var p = new Quaterniond[NumTrials];
		for (int i = 0; i < NumTrials; ++i)
		{
			q[i] = RandomEigen.RandomEigenQuaterniond();
			p[i] = RandomEigen.RandomEigenQuaterniond();
		}

		return (q, p);
	}

	[Test]
	public async Task QuaternionLeftMultMatrix_Nominal()
	{
		const double Eps = 1e-7;
		(Quaterniond[] qs, Quaterniond[] ps) = DrawQuaternionPairs();

		for (int i = 0; i < NumTrials; ++i)
		{
			Quaterniond q = qs[i];
			VectorXd pVec = Vec(ps[i]);

			// L(q) * p = q * p.
			MatrixXd left = MatrixXd.From(QuaternionUtils.QuaternionLeftMultMatrix(q));
			await Assert.That(EigenMatrixNear(left * pVec, Vec(q * ps[i]), 1e-12)).IsTrue();

			// L(q) = d(q*p)/dp (Jacobian w.r.t. second argument).
			var jNumeric = new MatrixXd(4, 4);
			for (int k = 0; k < 4; ++k)
			{
				VectorXd pPlus = pVec.Clone(), pMinus = pVec.Clone();
				pPlus[k] += Eps;
				pMinus[k] -= Eps;
				jNumeric.SetCol(k, (Vec(q * FromVec(pPlus)) - Vec(q * FromVec(pMinus))) / (2.0 * Eps));
			}

			await Assert.That(EigenMatrixNear(left, jNumeric, 1e-5)).IsTrue();
		}
	}

	[Test]
	public async Task QuaternionRightMultMatrix_Nominal()
	{
		const double Eps = 1e-7;
		(Quaterniond[] qs, Quaterniond[] ps) = DrawQuaternionPairs();

		for (int i = 0; i < NumTrials; ++i)
		{
			Quaterniond p = ps[i];
			VectorXd qVec = Vec(qs[i]);

			// R(p) * q = q * p.
			MatrixXd right = MatrixXd.From(QuaternionUtils.QuaternionRightMultMatrix(p));
			await Assert.That(EigenMatrixNear(right * qVec, Vec(qs[i] * p), 1e-12)).IsTrue();

			// R(p) = d(q*p)/dq (Jacobian w.r.t. first argument).
			var jNumeric = new MatrixXd(4, 4);
			for (int k = 0; k < 4; ++k)
			{
				VectorXd qPlus = qVec.Clone(), qMinus = qVec.Clone();
				qPlus[k] += Eps;
				qMinus[k] -= Eps;
				jNumeric.SetCol(k, (Vec(FromVec(qPlus) * p) - Vec(FromVec(qMinus) * p)) / (2.0 * Eps));
			}

			await Assert.That(EigenMatrixNear(right, jNumeric, 1e-5)).IsTrue();
		}
	}

	[Test]
	public async Task QuaternionRotatePointWithJac_Nominal()
	{
		const double Eps = 1e-7;
		RandomUtils.SetPRNGSeed(0);
		var qs = new Quaterniond[NumTrials];
		var pts = new Vector3d[NumTrials];
		for (int i = 0; i < NumTrials; ++i)
		{
			qs[i] = RandomEigen.RandomEigenQuaterniond();
			pts[i] = RandomEigen.RandomEigenVector3d();
		}

		for (int i = 0; i < NumTrials; ++i)
		{
			Quaterniond q = qs[i];
			Vector3d pt = pts[i];
			double[] qArr = [q.X, q.Y, q.Z, q.W];
			double[] ptArr = [pt.X, pt.Y, pt.Z];

			// R(q) * pt matches Eigen.
			var jAnalyticalData = new double[12];
			Vector3d result = QuaternionUtils.QuaternionRotatePointWithJac(qArr, ptArr, jAnalyticalData);
			await Assert.That(EigenMatrixNear(result, q * pt, 1e-12)).IsTrue();

			// Jacobian d(R(q)*pt)/dq matches numeric.
			var jNumeric = new MatrixXd(3, 4);
			for (int k = 0; k < 4; ++k)
			{
				double[] qPlus = (double[])qArr.Clone();
				double[] qMinus = (double[])qArr.Clone();
				qPlus[k] += Eps;
				qMinus[k] -= Eps;
				Vector3d column =
					(QuaternionUtils.QuaternionRotatePointWithJac(qPlus, ptArr, []) -
					 QuaternionUtils.QuaternionRotatePointWithJac(qMinus, ptArr, [])) / (2.0 * Eps);
				jNumeric.SetCol(k, VectorXd.From(column));
			}

			await Assert.That(EigenMatrixNear(MatrixXd.FromRowMajor(3, 4, jAnalyticalData), jNumeric, 1e-5)).IsTrue();
		}
	}

	[Test]
	public async Task EigenQuaternionAngleAxis_Roundtrip()
	{
		RandomUtils.SetPRNGSeed(0);
		var qs = new Quaterniond[NumTrials];
		for (int i = 0; i < NumTrials; ++i)
		{
			qs[i] = RandomEigen.RandomEigenQuaterniond();
		}

		for (int i = 0; i < NumTrials; ++i)
		{
			Quaterniond q = qs[i];
			double[] qArr = [q.X, q.Y, q.Z, q.W];

			// quaternion -> angle-axis -> quaternion recovers the original rotation.
			var angleAxis = new double[3];
			QuaternionUtils.AngleAxisFromEigenQuaternion(Real.Cast(qArr), Real.CastWritable(angleAxis));
			var qOut = new double[4];
			QuaternionUtils.EigenQuaternionFromAngleAxis(Real.Cast(angleAxis), Real.CastWritable(qOut));

			// Compare as rotations to avoid the quaternion double-cover sign ambiguity.
			var qRecovered = new Quaterniond(qOut[3], qOut[0], qOut[1], qOut[2]);
			await Assert.That(Math.Abs(q.AngularDistance(qRecovered) - 0.0)).IsLessThanOrEqualTo(1e-10);
		}
	}
}
