// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AutoDiffCostFunctionTests (C#-only; COLMAP's cost-function tests arrive with the cost
// functions in Phase 8): ColmapSharp/Solver/AutoDiffCostFunction.cs on a reprojection
// functor shaped like COLMAP's ReprojErrorCostFunctor (point 3, pose 7 = Eigen quaternion
// x y z w + translation, SIMPLE_RADIAL camera 4, two residuals), plus Rotation.cs.
// - Jacobians agree with central finite differences of the double evaluation.
// - Every gradient width gives bit-identical residuals and Jacobians: chunked passes
//   (widths 2, 4, 7, 16) against a single exact-width pass (14 parameters), the claim
//   that lets a few gradient sizes stand in for Ceres' one Jet of width N.
// - Residuals without Jacobians come from the double path; a block whose Jacobian is null
//   is left pointJacobian.

using ColmapSharp.Sensor;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

/// <summary>A test reprojection functor: point (3), cam_from_world (7), SIMPLE_RADIAL (4).</summary>
internal readonly struct TestReprojectionFunctor(double observedX, double observedY) : IAutoDiffFunctor
{
	public bool Evaluate<T>(ReadOnlySpan<T> p, Span<T> residuals)
		where T : struct, IScalar<T>
	{
		ReadOnlySpan<T> point = p[..3];
		ReadOnlySpan<T> pose = p[3..10];
		ReadOnlySpan<T> camera = p[10..14];

		// R(q) p = p + 2 w (v x p) + 2 v x (v x p), with q = (v, w) stored x, y, z, w.
		T qx = pose[0], qy = pose[1], qz = pose[2], qw = pose[3];
		T c0 = qy * point[2] - qz * point[1];
		T c1 = qz * point[0] - qx * point[2];
		T c2 = qx * point[1] - qy * point[0];
		T d0 = qy * c2 - qz * c1;
		T d1 = qz * c0 - qx * c2;
		T d2 = qx * c1 - qy * c0;
		T u = point[0] + 2.0 * (qw * c0 + d0) + pose[4];
		T v = point[1] + 2.0 * (qw * c1 + d1) + pose[5];
		T w = point[2] + 2.0 * (qw * c2 + d2) + pose[6];

		if (SimpleRadialCameraModel.ImgFromCam(camera, u, v, w, out T x, out T y))
		{
			residuals[0] = x - observedX;
			residuals[1] = y - observedY;
		}
		else
		{
			residuals[0] = T.FromDouble(0.0);
			residuals[1] = T.FromDouble(0.0);
		}

		return true;
	}
}

public class AutoDiffCostFunctionTests
{
	private static readonly int[] BlockSizes = [3, 7, 4];

	private static double[][] Parameters()
	{
		double norm = Math.Sqrt(0.1 * 0.1 + 0.2 * 0.2 + 0.05 * 0.05 + 0.97 * 0.97);
		return
		[
			[0.4, -0.3, 4.0],
			[0.1 / norm, -0.2 / norm, 0.05 / norm, 0.97 / norm, 0.1, 0.2, 0.5],
			[800.0, 320.0, 240.0, -0.05],
		];
	}

	private static double[]?[] NewJacobians() =>
		[new double[2 * 3], new double[2 * 7], new double[2 * 4]];

	// One view per block over whole arrays; a null array becomes the default (skip) view.
	private static ArraySegment<double>[] Views(double[]?[] arrays) =>
		[.. arrays.Select(array => array is null ? default : new ArraySegment<double>(array))];

	// The evaluator's layout: every parameter block and every Jacobian a slice of one shared
	// buffer, at offsets that do not start at zero. Must give the same bits as per-block arrays.
	[Test]
	public async Task SharedBufferViews_MatchPerBlockArrays()
	{
		var costFunction = Create<Grad14>();
		double[][] parameters = Parameters();
		(double[] expectedResiduals, double[]?[] expectedJacobians) = Run(costFunction, parameters);

		double[] parameterBuffer = new double[5 + 14];
		double[] jacobianBuffer = new double[3 + 2 * 14];
		var parameterViews = new ArraySegment<double>[3];
		var jacobianViews = new ArraySegment<double>[3];
		for (int b = 0, p = 5, j = 3; b < 3; p += BlockSizes[b], j += 2 * BlockSizes[b], b++)
		{
			parameters[b].CopyTo(parameterBuffer, p);
			parameterViews[b] = new ArraySegment<double>(parameterBuffer, p, BlockSizes[b]);
			jacobianViews[b] = new ArraySegment<double>(jacobianBuffer, j, 2 * BlockSizes[b]);
		}

		double[] residuals = new double[2];
		await Assert.That(costFunction.Evaluate(parameterViews, residuals, jacobianViews)).IsTrue();
		await Assert.That(residuals[0]).IsEqualTo(expectedResiduals[0]);
		await Assert.That(residuals[1]).IsEqualTo(expectedResiduals[1]);
		for (int b = 0; b < 3; b++)
		{
			for (int k = 0; k < 2 * BlockSizes[b]; k++)
			{
				await Assert.That(jacobianViews[b][k]).IsEqualTo(expectedJacobians[b]![k]);
			}
		}

		for (int k = 0; k < 3; k++)
		{
			await Assert.That(jacobianBuffer[k]).IsEqualTo(0.0);
		}
	}

	private static AutoDiffCostFunction<TestReprojectionFunctor, TGrad> Create<TGrad>()
		where TGrad : unmanaged, IJetGradient =>
		new(new TestReprojectionFunctor(300.0, 260.0), 2, BlockSizes);

	[Test]
	public async Task Jacobians_MatchFiniteDifferences()
	{
		var costFunction = Create<Grad4>();
		double[][] parameters = Parameters();
		double[] residuals = new double[2];
		double[]?[] jacobians = NewJacobians();
		await Assert.That(costFunction.Evaluate(Views(parameters), residuals, Views(jacobians))).IsTrue();

		double[] forward = new double[2];
		double[] backward = new double[2];
		for (int b = 0; b < BlockSizes.Length; b++)
		{
			for (int c = 0; c < BlockSizes[b]; c++)
			{
				double original = parameters[b][c];
				double h = 1e-6 * Math.Max(1.0, Math.Abs(original));
				parameters[b][c] = original + h;
				costFunction.Evaluate(Views(parameters), forward, []);
				parameters[b][c] = original - h;
				costFunction.Evaluate(Views(parameters), backward, []);
				parameters[b][c] = original;
				for (int r = 0; r < 2; r++)
				{
					double fd = (forward[r] - backward[r]) / (2 * h);
					double analytic = jacobians[b]![r * BlockSizes[b] + c];
					await Assert.That(Math.Abs(analytic - fd)).IsLessThanOrEqualTo(1e-5 * Math.Max(1.0, Math.Abs(fd)));
				}
			}
		}
	}

	[Test]
	public async Task EveryGradientWidth_GivesIdenticalBits()
	{
		double[][] parameters = Parameters();
		(double[] fullResiduals, double[]?[] fullJacobians) = Run(Create<Grad14>(), parameters);
		foreach ((double[] residuals, double[]?[] jacobians) in new[]
		{
			Run(Create<Grad2>(), parameters),
			Run(Create<Grad4>(), parameters),
			Run(Create<Grad7>(), parameters),
			Run(Create<Grad16>(), parameters),
		})
		{
			for (int r = 0; r < 2; r++)
			{
				await Assert.That(residuals[r]).IsEqualTo(fullResiduals[r]);
			}

			for (int b = 0; b < BlockSizes.Length; b++)
			{
				for (int k = 0; k < jacobians[b]!.Length; k++)
				{
					await Assert.That(jacobians[b]![k]).IsEqualTo(fullJacobians[b]![k]);
				}
			}
		}
	}

	private static (double[] Residuals, double[]?[] Jacobians) Run<TGrad>(
		AutoDiffCostFunction<TestReprojectionFunctor, TGrad> costFunction, double[][] parameters)
		where TGrad : unmanaged, IJetGradient
	{
		double[] residuals = new double[2];
		double[]?[] jacobians = NewJacobians();
		costFunction.Evaluate(Views(parameters), residuals, Views(jacobians));
		return (residuals, jacobians);
	}

	[Test]
	public async Task ResidualsOnly_UseDoublePath_AndPartialJacobiansMatchFull()
	{
		var costFunction = Create<Grad4>();
		double[][] parameters = Parameters();
		double[] residuals = new double[2];
		await Assert.That(costFunction.Evaluate(Views(parameters), residuals, [])).IsTrue();

		double[] direct = new double[2];
		double[] flat = [.. parameters[0], .. parameters[1], .. parameters[2]];
		new TestReprojectionFunctor(300.0, 260.0).Evaluate(Real.Cast(flat), Real.CastWritable(direct));
		await Assert.That(residuals[0]).IsEqualTo(direct[0]);
		await Assert.That(residuals[1]).IsEqualTo(direct[1]);

		double[] pointJacobian = [7.0, 7.0, 7.0, 7.0, 7.0, 7.0];
		double[]?[] partial = [pointJacobian, null, new double[8]];
		await Assert.That(costFunction.Evaluate(Views(parameters), residuals, Views(partial))).IsTrue();
		await Assert.That(pointJacobian.All(value => value == 7.0)).IsFalse();
		(_, double[]?[] full) = Run(Create<Grad4>(), parameters);
		for (int k = 0; k < 8; k++)
		{
			await Assert.That(partial[2]![k]).IsEqualTo(full[2]![k]);
		}

		double[] pointOnly = new double[2 * 3];
		double[]?[] onlyPoint = [pointOnly, null, null];
		costFunction.Evaluate(Views(parameters), residuals, Views(onlyPoint));
		await Assert.That(pointOnly.Any(value => value != 0.0)).IsTrue();
	}

	[Test]
	public async Task AngleAxisQuaternion_RoundTrip()
	{
		double[] angleAxis = [0.3, -0.5, 0.2];
		double[] quaternion = new double[4];
		Rotation.AngleAxisToQuaternion<Real>(Real.Cast(angleAxis), Real.CastWritable(quaternion));
		double theta = Math.Sqrt(0.3 * 0.3 + 0.5 * 0.5 + 0.2 * 0.2);
		await Assert.That(Math.Abs(quaternion[0] - Math.Cos(theta / 2))).IsLessThanOrEqualTo(1e-15);
		await Assert.That(Math.Abs(quaternion[1] - 0.3 / theta * Math.Sin(theta / 2))).IsLessThanOrEqualTo(1e-15);

		double[] back = new double[3];
		Rotation.QuaternionToAngleAxis<Real>(Real.Cast(quaternion), Real.CastWritable(back));
		for (int i = 0; i < 3; i++)
		{
			await Assert.That(Math.Abs(back[i] - angleAxis[i])).IsLessThanOrEqualTo(1e-15);
		}

		// A quaternion with negative w (angle > pi) maps to the equivalent angle below pi.
		double[] flipped = [-quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]];
		Rotation.QuaternionToAngleAxis<Real>(Real.Cast(flipped), Real.CastWritable(back));
		for (int i = 0; i < 3; i++)
		{
			await Assert.That(Math.Abs(back[i] - angleAxis[i])).IsLessThanOrEqualTo(1e-15);
		}
	}

	// At zero rotation the Taylor branch gives the exact derivatives dq/da = I / 2 and
	// da/dq = 2 I on the vector part, where the general formula would produce NaN.
	[Test]
	public async Task AngleAxisQuaternion_DerivativesAtZero()
	{
		Jet<Grad3>[] angleAxis = [Jet<Grad3>.Variable(0.0, 0), Jet<Grad3>.Variable(0.0, 1), Jet<Grad3>.Variable(0.0, 2)];
		Jet<Grad3>[] quaternion = new Jet<Grad3>[4];
		Rotation.AngleAxisToQuaternion<Jet<Grad3>>(angleAxis, quaternion);
		await Assert.That(quaternion[0].A).IsEqualTo(1.0);
		for (int i = 0; i < 3; i++)
		{
			for (int k = 0; k < 3; k++)
			{
				await Assert.That(quaternion[i + 1].Derivative(k)).IsEqualTo(i == k ? 0.5 : 0.0);
			}
		}

		Jet<Grad3>[] q =
		[
			Jet<Grad3>.FromDouble(1.0), Jet<Grad3>.Variable(0.0, 0), Jet<Grad3>.Variable(0.0, 1), Jet<Grad3>.Variable(0.0, 2),
		];
		Jet<Grad3>[] back = new Jet<Grad3>[3];
		Rotation.QuaternionToAngleAxis<Jet<Grad3>>(q, back);
		for (int i = 0; i < 3; i++)
		{
			for (int k = 0; k < 3; k++)
			{
				await Assert.That(back[i].Derivative(k)).IsEqualTo(i == k ? 2.0 : 0.0);
			}
		}
	}
}
