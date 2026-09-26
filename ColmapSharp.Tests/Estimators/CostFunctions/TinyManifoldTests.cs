// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TinyManifoldTests: colmap/estimators/cost_functions/tiny_manifold_test.cc ported 1:1, one
// method per gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Estimators/CostFunctions/TinyManifold.cs. The static_asserts on the product
// sizes become runtime checks of AmbientSize/TangentSize. Tier B (finite-difference and
// rotation tolerances).

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Optim;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class TinyManifoldTests
{
	// Central finite-difference approximation of the (ambient x tangent) Jacobian of
	// Manifold::Plus at delta = 0.
	private static MatrixXd NumericPlusJacobian<TManifold>(TManifold manifold, double[] x)
		where TManifold : struct, ITinyManifold
	{
		int ambient = manifold.AmbientSize;
		int tangent = manifold.TangentSize;
		const double Eps = 1e-6;
		var j = new MatrixXd(ambient, tangent);
		for (int c = 0; c < tangent; ++c)
		{
			var deltaPlus = new double[tangent];
			var deltaMinus = new double[tangent];
			deltaPlus[c] = Eps;
			deltaMinus[c] = -Eps;
			var xPlus = new double[ambient];
			var xMinus = new double[ambient];
			manifold.Plus(x, deltaPlus, xPlus);
			manifold.Plus(x, deltaMinus, xMinus);
			for (int r = 0; r < ambient; ++r)
			{
				j[r, c] = (xPlus[r] - xMinus[r]) / (2 * Eps);
			}
		}

		return j;
	}

	// Row-major analytic Jacobian returned by Manifold::PlusJacobian, as a matrix.
	private static MatrixXd AnalyticPlusJacobian<TManifold>(TManifold manifold, double[] x)
		where TManifold : struct, ITinyManifold
	{
		int ambient = manifold.AmbientSize;
		int tangent = manifold.TangentSize;
		var data = new double[ambient * tangent];
		manifold.PlusJacobian(x, data);
		return MatrixXd.FromRowMajor(ambient, tangent, data);
	}

	private static double[] Coeffs(Quaterniond q) => [q.X, q.Y, q.Z, q.W];

	private static Quaterniond TestRotation() =>
		new AngleAxisd(0.7, new Vector3d(1, 2, 3).Normalized()).ToQuaternion();

	[Test]
	public async Task EigenQuaternionManifold_PlusAtZeroIsIdentity()
	{
		Quaterniond q = TestRotation();
		var manifold = new TinyEigenQuaternionManifold();
		double[] delta = [0, 0, 0];
		var xPlus = new double[4];
		manifold.Plus(Coeffs(q), delta, xPlus);
		await Assert.That(EigenMatrixNear(new VectorXd(xPlus), new VectorXd(Coeffs(q)), 1e-12)).IsTrue();
	}

	[Test]
	public async Task EigenQuaternionManifold_PlusComposesRotationOnTheRight()
	{
		Quaterniond q = TestRotation();
		var delta = new Vector3d(0.05, -0.1, 0.08);
		var manifold = new TinyEigenQuaternionManifold();
		var xPlus = new double[4];
		manifold.Plus(Coeffs(q), [delta.X, delta.Y, delta.Z], xPlus);

		var qPlus = new Quaterniond(xPlus[3], xPlus[0], xPlus[1], xPlus[2]);
		Matrix3d expected = q.ToRotationMatrix() * new AngleAxisd(delta.Norm, delta.Normalized()).ToRotationMatrix();
		await Assert.That(EigenMatrixNear(MatrixXd.From(qPlus.ToRotationMatrix()), MatrixXd.From(expected), 1e-10)).IsTrue();
	}

	[Test]
	public async Task EigenQuaternionManifold_PlusJacobianMatchesFiniteDiff()
	{
		double[] q = Coeffs(TestRotation());
		var manifold = new TinyEigenQuaternionManifold();
		await Assert.That(EigenMatrixNear(AnalyticPlusJacobian(manifold, q), NumericPlusJacobian(manifold, q), 1e-6)).IsTrue();
	}

	[Test]
	public async Task SphereManifold_PlusStaysOnUnitSphere()
	{
		Vector3d x = new Vector3d(0.5, -1.0, 2.0).Normalized();
		var manifold = new TinySphereManifold();
		double[] delta = [0.15, -0.2];
		var xPlus = new double[3];
		manifold.Plus([x.X, x.Y, x.Z], delta, xPlus);
		await Assert.That(Math.Abs(new VectorXd(xPlus).Norm() - 1.0)).IsLessThanOrEqualTo(1e-12);

		double[] zero = [0, 0];
		var xPlusZero = new double[3];
		manifold.Plus([x.X, x.Y, x.Z], zero, xPlusZero);
		await Assert.That(EigenMatrixNear(new Vector3d(xPlusZero[0], xPlusZero[1], xPlusZero[2]), x, 1e-12)).IsTrue();
	}

	[Test]
	public async Task SphereManifold_PlusJacobianMatchesFiniteDiff()
	{
		Vector3d x = new Vector3d(0.5, -1.0, 2.0).Normalized();
		double[] xData = [x.X, x.Y, x.Z];
		var manifold = new TinySphereManifold();

		// The columns span the tangent plane, i.e. are orthogonal to x.
		MatrixXd j = AnalyticPlusJacobian(manifold, xData);
		await Assert.That(j.TransposeTimes(new VectorXd(xData)).Norm()).IsLessThan(1e-12);
		await Assert.That(EigenMatrixNear(j, NumericPlusJacobian(manifold, xData), 1e-6)).IsTrue();
	}

	[Test]
	public async Task ProductManifold_SizesAndBlockStructure()
	{
		var manifold = new TinyProductManifold<TinyEigenQuaternionManifold, TinySphereManifold>();
		await Assert.That(manifold.AmbientSize).IsEqualTo(7);
		await Assert.That(manifold.TangentSize).IsEqualTo(5);

		Quaterniond q = new AngleAxisd(0.9, new Vector3d(-1, 0.5, 2).Normalized()).ToQuaternion();
		Vector3d t = new Vector3d(1.0, -2.0, 0.5).Normalized();
		double[] x = [q.X, q.Y, q.Z, q.W, t.X, t.Y, t.Z];

		// Plus at zero recovers the point.
		double[] zero = [0, 0, 0, 0, 0];
		var xPlus = new double[7];
		manifold.Plus(x, zero, xPlus);
		await Assert.That(EigenMatrixNear(new VectorXd(xPlus), new VectorXd(x), 1e-12)).IsTrue();

		// The analytic Jacobian is block-diagonal and matches finite differences.
		MatrixXd j = AnalyticPlusJacobian(manifold, x);
		await Assert.That(EigenMatrixNear(j, NumericPlusJacobian(manifold, x), 1e-6)).IsTrue();
		double topRightNorm = j.Block(0, 3, 4, 2).Norm();
		double bottomLeftNorm = j.Block(4, 0, 3, 3).Norm();
		await Assert.That(topRightNorm).IsLessThan(1e-12);
		await Assert.That(bottomLeftNorm).IsLessThan(1e-12);
	}

	// A three-way product exercises the variadic recursion (depth 3). The two manifolds
	// already in this file suffice; the blocks are laid out in argument order and the Plus
	// Jacobian stays block-diagonal.
	[Test]
	public async Task ProductManifold_ThreeWaySizesAndBlockStructure()
	{
		var manifold = new TinyProductManifold<TinyEigenQuaternionManifold, TinySphereManifold, TinyEigenQuaternionManifold>();
		await Assert.That(manifold.AmbientSize).IsEqualTo(4 + 3 + 4);
		await Assert.That(manifold.TangentSize).IsEqualTo(3 + 2 + 3);
		int ambient = manifold.AmbientSize;
		int tangent = manifold.TangentSize;

		Quaterniond q0 = new AngleAxisd(0.9, new Vector3d(-1, 0.5, 2).Normalized()).ToQuaternion();
		Vector3d t = new Vector3d(1.0, -2.0, 0.5).Normalized();
		Quaterniond q1 = new AngleAxisd(0.3, new Vector3d(0.2, -1, 0.7).Normalized()).ToQuaternion();
		double[] x = [q0.X, q0.Y, q0.Z, q0.W, t.X, t.Y, t.Z, q1.X, q1.Y, q1.Z, q1.W];

		// Plus at zero recovers the point.
		var zero = new double[tangent];
		var xPlus = new double[ambient];
		manifold.Plus(x, zero, xPlus);
		await Assert.That(EigenMatrixNear(new VectorXd(xPlus), new VectorXd(x), 1e-12)).IsTrue();

		// The analytic Jacobian matches finite differences.
		MatrixXd j = AnalyticPlusJacobian(manifold, x);
		await Assert.That(EigenMatrixNear(j, NumericPlusJacobian(manifold, x), 1e-6)).IsTrue();

		// Everything outside the three diagonal blocks (4x3, 3x2, 4x3) is zero.
		j.SetBlock(0, 0, new MatrixXd(4, 3));
		j.SetBlock(4, 3, new MatrixXd(3, 2));
		j.SetBlock(7, 5, new MatrixXd(4, 3));
		await Assert.That(j.Norm()).IsLessThan(1e-12);
	}

	// C#-only: TinyEuclideanManifold1 (C++'s EuclideanManifold<1>) is stateless, so the
	// default-constructed pose-plus-focal product the relative-pose-with-focal solvers use
	// has the right sizes and a correct Plus and Jacobian.
	[Test]
	public async Task CSharpOnly_DefaultPoseFocalProductWithEuclidean1()
	{
		var manifold = new TinyProductManifold<TinyEigenQuaternionManifold, TinySphereManifold, TinyEuclideanManifold1>();
		await Assert.That(manifold.AmbientSize).IsEqualTo(8);
		await Assert.That(manifold.TangentSize).IsEqualTo(6);
		await Assert.That(manifold.IsEuclidean).IsFalse();

		Quaterniond q = new AngleAxisd(0.9, new Vector3d(-1, 0.5, 2).Normalized()).ToQuaternion();
		Vector3d t = new Vector3d(1.0, -2.0, 0.5).Normalized();
		double[] x = [q.X, q.Y, q.Z, q.W, t.X, t.Y, t.Z, 0.25];

		// The Euclidean block steps by plain addition.
		double[] delta = [0, 0, 0, 0, 0, 0.5];
		var xPlus = new double[8];
		manifold.Plus(x, delta, xPlus);
		await Assert.That(xPlus[7]).IsEqualTo(0.75);

		// The analytic Jacobian matches finite differences and its last block is 1.
		MatrixXd j = AnalyticPlusJacobian(manifold, x);
		await Assert.That(EigenMatrixNear(j, NumericPlusJacobian(manifold, x), 1e-6)).IsTrue();
		await Assert.That(j[7, 5]).IsEqualTo(1.0);
	}
}
