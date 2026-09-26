// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// MotionAveragingTests: colmap/estimators/cost_functions/motion_averaging_test.cc ported
// 1:1, one method per gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances.
// Tests ColmapSharp/Estimators/CostFunctions/MotionAveragingCostFunctions.cs. The C++ calls
// the functor's operator() with one pointer per block; here the blocks are concatenated
// into one span evaluated on Real (doubles), the functor's IAutoDiffFunctor shape.
// Tier B (1e-10).

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class MotionAveragingTests
{
	private static bool Evaluate<TFunctor>(TFunctor functor, out Vector3d residuals, params double[][] blocks)
		where TFunctor : IAutoDiffFunctor
	{
		double[] parameters = blocks.SelectMany(b => b).ToArray();
		var r = new double[3];
		bool ok = functor.Evaluate(Real.Cast(parameters), Real.CastWritable(r));
		residuals = new Vector3d(r[0], r[1], r[2]);
		return ok;
	}

	private static double[] A(Vector3d v) => [v.X, v.Y, v.Z];

	[Test]
	public async Task BATAPairwiseDirectionCostFunctor_ZeroResidual()
	{
		var pos1 = new Vector3d(1, 2, 3);
		var pos2 = new Vector3d(2, 3, 4);
		const double Scale = 1.0;
		Vector3d direction = pos2 - pos1;

		var costFunctor = new BATAPairwiseDirectionCostFunctor(direction);

		await Assert.That(Evaluate(costFunctor, out Vector3d residuals, A(pos1), A(pos2), [Scale])).IsTrue();
		await Assert.That(EigenMatrixNear(residuals, new Vector3d(0, 0, 0), 1e-10)).IsTrue();
	}

	[Test]
	public async Task BATAPairwiseDirectionCostFunctor_NonZeroResidual()
	{
		var pos1 = new Vector3d(1, 2, 3);
		var pos2 = new Vector3d(4, 5, 6);
		const double Scale = 2.0;
		var direction = new Vector3d(1, 1, 1);

		var costFunctor = new BATAPairwiseDirectionCostFunctor(direction);

		await Assert.That(Evaluate(costFunctor, out Vector3d residuals, A(pos1), A(pos2), [Scale])).IsTrue();

		Vector3d expectedResiduals = direction - Scale * (pos2 - pos1);
		await Assert.That(EigenMatrixNear(residuals, expectedResiduals, 1e-10)).IsTrue();
	}

	[Test]
	public async Task BATAPairwiseDirectionCostFunctor_DifferentScale()
	{
		var pos1 = new Vector3d(1, 2, 3);
		var pos2 = new Vector3d(2, 4, 6);
		const double Scale = 0.5;
		Vector3d direction = Scale * (pos2 - pos1);

		var costFunctor = new BATAPairwiseDirectionCostFunctor(direction);

		await Assert.That(Evaluate(costFunctor, out Vector3d residuals, A(pos1), A(pos2), [Scale])).IsTrue();
		await Assert.That(EigenMatrixNear(residuals, new Vector3d(0, 0, 0), 1e-10)).IsTrue();
	}

	[Test]
	public async Task BATAPairwiseDirectionCostFunctor_Create()
	{
		var direction = new Vector3d(1, 0, 0);
		CostFunction costFunction = BATAPairwiseDirectionCostFunctor.Create(direction);
		await Assert.That(costFunction).IsNotNull();
	}

	[Test]
	public async Task RigBATAPairwiseDirectionConstantRigCostFunctor_ZeroResidual()
	{
		var point3D = new Vector3d(1, 2, 3);
		var rigInWorld = new Vector3d(3, 2, 1);
		const double Scale = 1.5;
		var camFromRigDir = new Vector3d(0.25, 0.5, 0.75);
		Vector3d camFromPoint3DDir = Scale * (point3D - rigInWorld + camFromRigDir);

		var costFunctor = new RigBATAPairwiseDirectionConstantRigCostFunctor(camFromPoint3DDir, camFromRigDir);

		await Assert.That(Evaluate(costFunctor, out Vector3d residuals, A(point3D), A(rigInWorld), [Scale])).IsTrue();
		await Assert.That(EigenMatrixNear(residuals, new Vector3d(0, 0, 0), 1e-10)).IsTrue();
	}

	[Test]
	public async Task RigBATAPairwiseDirectionConstantRigCostFunctor_NonZeroResidual()
	{
		var point3D = new Vector3d(3, 4, 5);
		var rigInWorld = new Vector3d(1, 2, 3);
		const double Scale = 2.0;
		var camFromRigDir = new Vector3d(0.1, 0.2, 0.3);
		var camFromPoint3DDir = new Vector3d(1, 1, 1);

		var costFunctor = new RigBATAPairwiseDirectionConstantRigCostFunctor(camFromPoint3DDir, camFromRigDir);

		await Assert.That(Evaluate(costFunctor, out Vector3d residuals, A(point3D), A(rigInWorld), [Scale])).IsTrue();

		Vector3d expectedResiduals = camFromPoint3DDir - Scale * (point3D - rigInWorld + camFromRigDir);
		await Assert.That(EigenMatrixNear(residuals, expectedResiduals, 1e-10)).IsTrue();
	}

	[Test]
	public async Task RigBATAPairwiseDirectionConstantRigCostFunctor_Create()
	{
		var camFromPoint3DDir = new Vector3d(1, 0, 0);
		var camFromRigDir = new Vector3d(0, 1, 0);
		CostFunction costFunction = RigBATAPairwiseDirectionConstantRigCostFunctor.Create(camFromPoint3DDir, camFromRigDir);
		await Assert.That(costFunction).IsNotNull();
	}

	[Test]
	public async Task RigBATAPairwiseDirectionCostFunctor_ZeroResidual()
	{
		var point3D = new Vector3d(5, 5, 5);
		var rigInWorld = new Vector3d(1, 1, 1);
		var camInRig = new Vector3d(0.5, 0.5, 0.5);
		const double Scale = 1.0;
		Quaterniond rigFromWorldRot = Quaterniond.Identity;
		Vector3d camFromRigDir = rigFromWorldRot.Inverse() * camInRig;
		Vector3d camFromPoint3DDir = Scale * (point3D - rigInWorld - camFromRigDir);

		var costFunctor = new RigBATAPairwiseDirectionCostFunctor(camFromPoint3DDir, rigFromWorldRot);

		await Assert.That(Evaluate(costFunctor, out Vector3d residuals, A(point3D), A(rigInWorld), A(camInRig), [Scale])).IsTrue();
		await Assert.That(EigenMatrixNear(residuals, new Vector3d(0, 0, 0), 1e-10)).IsTrue();
	}

	[Test]
	public async Task RigBATAPairwiseDirectionCostFunctor_NonZeroResidual()
	{
		var point3D = new Vector3d(3, 4, 5);
		var rigInWorld = new Vector3d(1, 2, 3);
		var camInRig = new Vector3d(0.2, 0.3, 0.4);
		const double Scale = 2.0;
		Quaterniond rigFromWorldRot = new Quaterniond(0.707, 0.707, 0, 0).Normalized();
		var camFromPoint3DDir = new Vector3d(1, 1, 1);

		var costFunctor = new RigBATAPairwiseDirectionCostFunctor(camFromPoint3DDir, rigFromWorldRot);

		await Assert.That(Evaluate(costFunctor, out Vector3d residuals, A(point3D), A(rigInWorld), A(camInRig), [Scale])).IsTrue();

		Vector3d camFromRigDir = rigFromWorldRot.ToRotationMatrix().Transpose() * camInRig;
		Vector3d expectedResiduals = camFromPoint3DDir - Scale * (point3D - rigInWorld - camFromRigDir);
		await Assert.That(EigenMatrixNear(residuals, expectedResiduals, 1e-10)).IsTrue();
	}

	[Test]
	public async Task RigBATAPairwiseDirectionCostFunctor_Create()
	{
		var camFromPoint3DDir = new Vector3d(1, 0, 0);
		Quaterniond rigFromWorldRot = Quaterniond.Identity;
		CostFunction costFunction = RigBATAPairwiseDirectionCostFunctor.Create(camFromPoint3DDir, rigFromWorldRot);
		await Assert.That(costFunction).IsNotNull();
	}
}
