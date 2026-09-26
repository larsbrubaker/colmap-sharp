// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// CameraCostFunctionsTests: C#-only tests (COLMAP has no test for CreateCameraCostFunction)
// of ColmapSharp/Estimators/CostFunctions/CameraCostFunctions.cs: every camera model id
// dispatches to the right cost function type with COLMAP's parameter block sizes, and an
// unknown id throws.

using ColmapSharp.Estimators.CostFunctions;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators.CostFunctions;

public class CameraCostFunctionsTests
{
	private static IEnumerable<CameraModelId> AllModelIds() =>
		Enum.GetValues<CameraModelId>().Where(id => id != CameraModelId.Invalid);

	[Test]
	public async Task CreateCameraCostFunction_DispatchesEveryModel()
	{
		var point2D = new Vector2d(1, 2);
		var pose = new Rigid3d();
		foreach (CameraModelId id in AllModelIds())
		{
			int numParams = CameraModels.CameraModelNumParams(id);

			CostFunction reproj = CameraCostFunctions.CreateReprojErrorCostFunction(id, point2D);
			await Assert.That(reproj.GetType().Name).StartsWith("AnalyticalReprojErrorCostFunction");
			await Assert.That(reproj.ParameterBlockSizes.ToArray()).IsEquivalentTo(new[] { 3, 7, numParams });

			CostFunction constantPose = CameraCostFunctions.CreateReprojErrorConstantPoseCostFunction(id, point2D, pose);
			await Assert.That(constantPose.GetType().Name).StartsWith("AnalyticalReprojErrorConstantPoseCostFunction");
			await Assert.That(constantPose.ParameterBlockSizes.ToArray()).IsEquivalentTo(new[] { 3, numParams });

			CostFunction constantPoint = CameraCostFunctions.CreateReprojErrorConstantPoint3DCostFunction(id, point2D, new Vector3d(0, 0, 1));
			await Assert.That(constantPoint.ParameterBlockSizes.ToArray()).IsEquivalentTo(new[] { 7, numParams });

			CostFunction rig = CameraCostFunctions.CreateRigReprojErrorCostFunction(id, point2D);
			await Assert.That(rig.ParameterBlockSizes.ToArray()).IsEquivalentTo(new[] { 3, 7, 7, numParams });

			CostFunction constantRig = CameraCostFunctions.CreateRigReprojErrorConstantRigCostFunction(id, point2D, pose);
			await Assert.That(constantRig.ParameterBlockSizes.ToArray()).IsEquivalentTo(new[] { 3, 7, numParams });
		}
	}

	[Test]
	public async Task CreateCameraCostFunction_UnknownModelThrows()
	{
		await Assert.That(() => CameraCostFunctions.CreateReprojErrorCostFunction(CameraModelId.Invalid, new Vector2d(0, 0)))
			.Throws<ArgumentException>();
	}
}
