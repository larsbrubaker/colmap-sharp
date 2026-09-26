// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoseStorageProblemTests: C#-only (no COLMAP counterpart). Pins the Phase 8 contract
// between Scene/Frame.cs's pose storage and the reprojection cost functions: the frame's
// RigFromWorldStorage.Params is registered as ONE 7-value parameter block with the
// EigenQuaternion x Euclidean<3> product manifold, as bundle_adjustment_ceres.cc registers
// `rig_from_world.params.data()`, and solving writes the refined pose back into the frame
// in place.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class PoseStorageProblemTests
{
	[Test]
	public async Task FrameRigFromWorldParams_SolvedInPlaceAsOneBlock()
	{
		var truth = new Rigid3d(
			new AngleAxisd(0.3, new Vector3d(0.2, 1, -0.1).Normalized()).ToQuaternion(),
			new Vector3d(0.1, -0.2, 4));
		double[] cameraParams = [100, 0, 0];

		var frame = new Frame();
		frame.SetRigFromWorld(new Rigid3d(
			(new AngleAxisd(0.05, new Vector3d(1, 0, 0)).ToQuaternion() * truth.Rotation).Normalized(),
			truth.Translation + new Vector3d(0.1, 0.05, -0.2)));
		double[] pose = frame.RigFromWorldStorage.Params;

		var problem = new Problem();
		var points = new List<double[]>();
		for (int i = 0; i < 12; i++)
		{
			double[] point = [(i % 4) - 1.5, (i / 4) - 1.0, 0.5 * (i % 3)];
			points.Add(point);
			Vector3d pointInCam = truth * new Vector3d(point[0], point[1], point[2]);
			var observation = new Vector2d(100 * pointInCam.X / pointInCam.Z, 100 * pointInCam.Y / pointInCam.Z);
			problem.AddResidualBlock(
				CameraCostFunctions.CreateReprojErrorCostFunction(CameraModelId.SimplePinhole, observation),
				null,
				new ArraySegment<double>(point),
				new ArraySegment<double>(pose),
				new ArraySegment<double>(cameraParams));
			problem.SetParameterBlockConstant(point);
		}

		problem.SetParameterBlockConstant(cameraParams);
		problem.SetManifold(
			pose,
			ManifoldHelpers.CreateProductManifold(ManifoldHelpers.CreateEigenQuaternionManifold(), ManifoldHelpers.CreateEuclideanManifold(3)));
		await Assert.That(ManifoldHelpers.ParameterBlockTangentSize(problem, pose)).IsEqualTo(6);

		SolverSummary summary = LeastSquaresSolver.Solve(new SolverOptions(), problem);

		Rigid3d refined = frame.RigFromWorld();
		using (Assert.Multiple())
		{
			await Assert.That(summary.IsSolutionUsable).IsTrue();
			await Assert.That(ReferenceEquals(frame.RigFromWorldStorage.Params, pose)).IsTrue();
			await Assert.That(refined.Rotation.AngularDistance(truth.Rotation)).IsLessThan(1e-8);
			await Assert.That((refined.Translation - truth.Translation).Norm).IsLessThan(1e-8);
		}
	}
}
