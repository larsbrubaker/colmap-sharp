// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// PosePriorTests: colmap/estimators/cost_functions/pose_prior_test.cc ported 1:1, one method
// per gtest TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Estimators/CostFunctions/PosePriorCostFunctions.cs (and the covariance
// wrapper of CostFunctionUtils.cs) through CostFunction.Evaluate. Rigid3d::params is
// [qx, qy, qz, qw, tx, ty, tz]; the C++ tests point parameters straight at it, so a pose
// that the C++ reassigns after the pointer was taken is re-packed here before evaluating.
// Tier B (exact where the C++ uses EXPECT_EQ, else 1e-6).

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.EigenMatchers;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class PosePriorTests
{
	private static void Pack(Rigid3d pose, double[] destination)
	{
		destination[0] = pose.Rotation.X;
		destination[1] = pose.Rotation.Y;
		destination[2] = pose.Rotation.Z;
		destination[3] = pose.Rotation.W;
		destination[4] = pose.Translation.X;
		destination[5] = pose.Translation.Y;
		destination[6] = pose.Translation.Z;
	}

	private static Rigid3d RandomRigid3d()
	{
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d();
		return new Rigid3d(rotation, translation);
	}

	private static double[] NaNs(int n) => Enumerable.Repeat(double.NaN, n).ToArray();

	private static Vector3d V(double[] r) => new(r[0], r[1], r[2]);

	// Eigen::Map<Eigen::Quaterniond>(pose) = rotation_matrix.
	private static void SetRotation(double[] pose, Matrix3d rotationMatrix)
	{
		Quaterniond q = Quaterniond.FromRotationMatrix(rotationMatrix);
		pose[0] = q.X;
		pose[1] = q.Y;
		pose[2] = q.Z;
		pose[3] = q.W;
	}

	[Test]
	public async Task AbsolutePosePositionPriorCostFunctor_Nominal()
	{
		var costFunction = AbsolutePosePositionPriorCostFunctor.Create(new Vector3d(0, 0, 0));

		var sensorFromWorldParams = new double[7];
		Pack(new Rigid3d(Quaterniond.Identity, new Vector3d(0, 0, 0)), sensorFromWorldParams);

		double[] residuals = NaNs(3);
		ArraySegment<double>[] parameters = [new(sensorFromWorldParams)];

		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(EigenMatrixNear(V(residuals), new Vector3d(0, 0, 0), 1e-6)).IsTrue();

		Rigid3d sensorFromWorld = RandomRigid3d();
		Pack(sensorFromWorld, sensorFromWorldParams);
		Vector3d positionInWorld = sensorFromWorld.Inverse().Translation;
		residuals = NaNs(3);
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(EigenMatrixNear(V(residuals), -positionInWorld, 1e-6)).IsTrue();

		costFunction = AbsolutePosePositionPriorCostFunctor.Create(positionInWorld);
		residuals = NaNs(3);
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(EigenMatrixNear(V(residuals), new Vector3d(0, 0, 0), 1e-6)).IsTrue();
	}

	[Test]
	public async Task AbsoluteRigPosePositionPriorCostFunctor_Nominal()
	{
		var costFunction = AbsoluteRigPosePositionPriorCostFunctor.Create(new Vector3d(0, 0, 0));

		var sensorFromRigParams = new double[7];
		var rigFromWorldParams = new double[7];
		Pack(new Rigid3d(Quaterniond.Identity, new Vector3d(0, 0, 0)), sensorFromRigParams);
		Pack(new Rigid3d(Quaterniond.Identity, new Vector3d(0, 0, 0)), rigFromWorldParams);

		double[] residuals = NaNs(3);
		ArraySegment<double>[] parameters = [new(sensorFromRigParams), new(rigFromWorldParams)];
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(EigenMatrixNear(V(residuals), new Vector3d(0, 0, 0), 1e-6)).IsTrue();

		Rigid3d sensorFromRig = RandomRigid3d();
		Rigid3d rigFromWorld = RandomRigid3d();
		Pack(sensorFromRig, sensorFromRigParams);
		Pack(rigFromWorld, rigFromWorldParams);
		Rigid3d sensorFromWorld = sensorFromRig * rigFromWorld;
		Vector3d positionInWorld = sensorFromWorld.Inverse().Translation;
		residuals = NaNs(3);
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(EigenMatrixNear(V(residuals), -positionInWorld, 1e-6)).IsTrue();

		costFunction = AbsoluteRigPosePositionPriorCostFunctor.Create(positionInWorld);
		residuals = NaNs(3);
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(EigenMatrixNear(V(residuals), new Vector3d(0, 0, 0), 1e-6)).IsTrue();
	}

	[Test]
	public async Task AbsolutePosePriorCostFunctor_Nominal()
	{
		var camFromWorldPrior = new Rigid3d();
		var costFunction = AbsolutePosePriorCostFunctor.Create(camFromWorldPrior);

		double[] camFromWorld = [0, 0, 0, 1, 0, 0, 0];
		var residuals = new double[6];
		ArraySegment<double>[] parameters = [new(camFromWorld)];
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		for (int i = 0; i < 6; i++)
		{
			await Assert.That(residuals[i]).IsEqualTo(0);
		}

		camFromWorld[4] = 1;
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(residuals[0]).IsEqualTo(0);
		await Assert.That(residuals[1]).IsEqualTo(0);
		await Assert.That(residuals[2]).IsEqualTo(0);
		await Assert.That(residuals[3]).IsEqualTo(1);
		await Assert.That(residuals[4]).IsEqualTo(0);
		await Assert.That(residuals[5]).IsEqualTo(0);

		// Rotation by 90 degrees around the Y axis.
		SetRotation(camFromWorld, new Matrix3d(0, 0, 1, 0, 1, 0, -1, 0, 0));
		camFromWorld[5] = 2;
		camFromWorld[6] = 3;
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(residuals[0]).IsEqualTo(0).Within(1e-6);
		await Assert.That(residuals[1]).IsEqualTo(MathUtils.DegToRad(90.0)).Within(1e-6);
		await Assert.That(residuals[2]).IsEqualTo(0).Within(1e-6);
		await Assert.That(residuals[3]).IsEqualTo(1).Within(1e-6);
		await Assert.That(residuals[4]).IsEqualTo(2).Within(1e-6);
		await Assert.That(residuals[5]).IsEqualTo(3).Within(1e-6);
	}

	[Test]
	public async Task RelativePosePriorCostFunctor_Nominal()
	{
		var iFromJPrior = new Rigid3d(Quaterniond.Identity, new Vector3d(0, 0, -1));
		var costFunction = RelativePosePriorCostFunctor.Create(iFromJPrior);

		double[] iFromWorld = [0, 0, 0, 1, 0, 0, 0];
		double[] jFromWorld = [0, 0, 0, 1, 0, 0, 1];
		var residuals = new double[6];
		ArraySegment<double>[] parameters = [new(iFromWorld), new(jFromWorld)];
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		for (int i = 0; i < 6; i++)
		{
			await Assert.That(residuals[i]).IsEqualTo(0);
		}

		iFromWorld[6] = 4;
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(residuals[0]).IsEqualTo(0);
		await Assert.That(residuals[1]).IsEqualTo(0);
		await Assert.That(residuals[2]).IsEqualTo(0);
		await Assert.That(residuals[3]).IsEqualTo(0);
		await Assert.That(residuals[4]).IsEqualTo(0);
		await Assert.That(residuals[5]).IsEqualTo(4);

		jFromWorld[4] = 2;
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(residuals[0]).IsEqualTo(0);
		await Assert.That(residuals[1]).IsEqualTo(0);
		await Assert.That(residuals[2]).IsEqualTo(0);
		await Assert.That(residuals[3]).IsEqualTo(-2);
		await Assert.That(residuals[4]).IsEqualTo(0);
		await Assert.That(residuals[5]).IsEqualTo(4);

		// Rotation by 90 degrees around the Y axis.
		SetRotation(jFromWorld, new Matrix3d(0, 0, 1, 0, 1, 0, -1, 0, 0));
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(residuals[0]).IsEqualTo(0).Within(1e-6);
		await Assert.That(residuals[1]).IsEqualTo(MathUtils.DegToRad(-90.0)).Within(1e-6);
		await Assert.That(residuals[2]).IsEqualTo(0).Within(1e-6);
		await Assert.That(residuals[3]).IsEqualTo(0).Within(1e-6);
		await Assert.That(residuals[4]).IsEqualTo(0).Within(1e-6);
		await Assert.That(residuals[5]).IsEqualTo(2).Within(1e-6);
	}

	[Test]
	public async Task CovarianceWeightedCostFunctor_AbsolutePosePositionPriorCostFunctor()
	{
		Rigid3d camFromWorld = RandomRigid3d();
		Rigid3d worldFromCam = camFromWorld.Inverse();

		var residuals = new double[3];
		var camFromWorldParams = new double[7];
		Pack(camFromWorld, camFromWorldParams);
		ArraySegment<double>[] parameters = [new(camFromWorldParams)];

		MatrixXd covariance = MatrixXd.Identity(3);
		for (int i = 0; i < 3; i++)
		{
			covariance[i, i] = 2;
		}

		var costFunction = CovarianceWeightedCostFunctor.Create(
			covariance, new AbsolutePosePositionPriorCostFunctor(new Vector3d(0, 0, 0)));
		await Assert.That(costFunction.Evaluate(parameters, residuals, default)).IsTrue();
		await Assert.That(residuals[0]).IsEqualTo(-0.5 * Math.Sqrt(2) * worldFromCam.Translation.X).Within(1e-6);
		await Assert.That(residuals[1]).IsEqualTo(-0.5 * Math.Sqrt(2) * worldFromCam.Translation.Y).Within(1e-6);
		await Assert.That(residuals[2]).IsEqualTo(-0.5 * Math.Sqrt(2) * worldFromCam.Translation.Z).Within(1e-6);
	}
}
