// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ManifoldTests (C#-only; COLMAP has no manifold test, and Ceres' manifold_test.cc is built
// on its gtest invariant matchers, which are not ported): the manifold invariants Ceres'
// manifold_test_utils.h checks, for every manifold COLMAP uses (ColmapSharp/Solver/
// Manifolds.cs and SphereProductManifolds.cs):
// - Plus(x, 0) = x and Minus(x, x) = 0,
// - Minus(Plus(x, delta), x) = delta for a small tangent step,
// - PlusJacobian and MinusJacobian agree with central finite differences of Plus and Minus,
// - RightMultiplyByPlusJacobian equals the explicit product with PlusJacobian.

using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class ManifoldTests
{
	private const double H = 1e-6;
	private const double Tolerance = 1e-8;

	public static IEnumerable<Func<(string Name, Manifold Manifold, double[] X)>> Cases()
	{
		double invSqrt = 1.0 / Math.Sqrt(0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.9 * 0.9);
		double[] unitQuaternion = [0.1 * invSqrt, 0.2 * invSqrt, 0.3 * invSqrt, 0.9 * invSqrt];
		yield return () => ("Euclidean3", new EuclideanManifold(3), [1.0, -2.0, 0.5]);
		yield return () => ("Subset5", new SubsetManifold(5, [3, 1]), [1.0, 2.0, 3.0, 4.0, 5.0]);
		yield return () => ("EigenQuaternion", QuaternionManifold.EigenOrder(), unitQuaternion);
		yield return () => ("CeresQuaternion", QuaternionManifold.CeresOrder(), unitQuaternion);
		yield return () => ("Sphere3", new SphereManifold(3), [0.3, -0.4, 1.2]);
		yield return () => ("Sphere3NegativePivot", new SphereManifold(3), [0.3, -0.4, -1.2]);
		yield return () => ("Sphere4", new SphereManifold(4), [0.5, 0.5, -0.5, 0.5]);
		yield return () => ("QuaternionTimesR3",
			new ProductManifold(QuaternionManifold.EigenOrder(), new EuclideanManifold(3)),
			[.. unitQuaternion, 1.0, 2.0, 3.0]);
		yield return () => ("QuaternionTimesSphere",
			new ProductManifold(QuaternionManifold.EigenOrder(), new SphereManifold(3)),
			[.. unitQuaternion, 0.0, 0.6, 0.8]);
	}

	[Test]
	[MethodDataSource(nameof(Cases))]
	public async Task PlusMinusInvariantsHold((string Name, Manifold Manifold, double[] X) c)
	{
		Manifold m = c.Manifold;
		double[] x = c.X;
		int n = m.AmbientSize;
		int t = m.TangentSize;

		double[] plusZero = new double[n];
		m.Plus(x, new double[t], plusZero);
		for (int i = 0; i < n; i++)
		{
			await Assert.That(plusZero[i]).IsEqualTo(x[i]);
		}

		double[] minusSelf = new double[t];
		m.Minus(x, x, minusSelf);
		for (int i = 0; i < t; i++)
		{
			await Assert.That(Math.Abs(minusSelf[i])).IsLessThanOrEqualTo(1e-15);
		}

		double[] delta = new double[t];
		for (int i = 0; i < t; i++)
		{
			delta[i] = 0.01 * (i + 1) * (i % 2 == 0 ? 1 : -1);
		}

		double[] y = new double[n];
		m.Plus(x, delta, y);
		double[] recovered = new double[t];
		m.Minus(y, x, recovered);
		for (int i = 0; i < t; i++)
		{
			await Assert.That(Math.Abs(recovered[i] - delta[i])).IsLessThanOrEqualTo(1e-12);
		}
	}

	[Test]
	[MethodDataSource(nameof(Cases))]
	public async Task JacobiansMatchFiniteDifferences((string Name, Manifold Manifold, double[] X) c)
	{
		Manifold m = c.Manifold;
		double[] x = c.X;
		int n = m.AmbientSize;
		int t = m.TangentSize;

		double[] plusJacobian = new double[n * t];
		m.PlusJacobian(x, plusJacobian);
		double[] minusJacobian = new double[t * n];
		m.MinusJacobian(x, minusJacobian);

		double[] forward = new double[n];
		double[] backward = new double[n];
		for (int j = 0; j < t; j++)
		{
			double[] step = new double[t];
			step[j] = H;
			m.Plus(x, step, forward);
			step[j] = -H;
			m.Plus(x, step, backward);
			for (int i = 0; i < n; i++)
			{
				double fd = (forward[i] - backward[i]) / (2 * H);
				await Assert.That(Math.Abs(plusJacobian[i * t + j] - fd)).IsLessThanOrEqualTo(Tolerance);
			}
		}

		// d Minus(y, x) / dy at y = x, stepping y along the ambient axes. Only the component
		// of the step tangent to the manifold is meaningful, so compare MinusJacobian *
		// PlusJacobian (the tangent-space identity) instead of raw ambient columns.
		for (int r = 0; r < t; r++)
		{
			for (int col = 0; col < t; col++)
			{
				double sum = 0.0;
				for (int k = 0; k < n; k++)
				{
					sum += minusJacobian[r * n + k] * plusJacobian[k * t + col];
				}

				await Assert.That(Math.Abs(sum - (r == col ? 1.0 : 0.0))).IsLessThanOrEqualTo(1e-12);
			}
		}

		double[] ambient = new double[2 * n];
		for (int i = 0; i < ambient.Length; i++)
		{
			ambient[i] = 0.1 * i - 0.3;
		}

		double[] tangent = new double[2 * t];
		m.RightMultiplyByPlusJacobian(x, 2, ambient, tangent);
		for (int r = 0; r < 2; r++)
		{
			for (int col = 0; col < t; col++)
			{
				double sum = 0.0;
				for (int k = 0; k < n; k++)
				{
					sum += ambient[r * n + k] * plusJacobian[k * t + col];
				}

				await Assert.That(Math.Abs(tangent[r * t + col] - sum)).IsLessThanOrEqualTo(1e-14);
			}
		}
	}

	// Ceres' SubsetManifold rejects duplicate and out-of-range constant indices.
	[Test]
	public async Task SubsetManifold_RejectsBadIndices()
	{
		await Assert.That(() => new SubsetManifold(3, [1, 1])).Throws<ArgumentException>();
		await Assert.That(() => new SubsetManifold(3, [3])).Throws<ArgumentException>();
	}
}
