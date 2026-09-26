// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// TriangulationEstimationTests: C#-only tests. COLMAP has no estimators/triangulation_test.cc
// (EstimateTriangulation is covered upstream only through the incremental mapper tests and
// pycolmap's binding smoke tests), so these pin the behavior of
// ColmapSharp/Estimators/TriangulationEstimation.cs on noise-free synthetic scenes: exact
// recovery, outlier rejection, and the triangulation-angle requirement. Tier C (RANSAC).

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public class TriangulationEstimationTests
{
	// The observations of one synthetic 3D point: pixels, poses and cameras of its track.
	private static (List<Vector2d> Points, List<Rigid3d> CamsFromWorld, List<Camera> Cameras) TrackObservations(
		Reconstruction reconstruction, Point3D point3D)
	{
		var points = new List<Vector2d>();
		var camsFromWorld = new List<Rigid3d>();
		var cameras = new List<Camera>();
		foreach (TrackElement element in point3D.Track.Elements)
		{
			Image image = reconstruction.Image(element.ImageId);
			points.Add(image.Points2D[(int)element.Point2DIdx].Xy);
			camsFromWorld.Add(image.CamFromWorld());
			cameras.Add(image.CameraPtr);
		}

		return (points, camsFromWorld, cameras);
	}

	private static Reconstruction SynthesizeScene(CameraModelId modelId = CameraModelId.SimpleRadial)
	{
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 4,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 1,
			NumPoints3D = 20,
			CameraModelId = modelId,
		};
		if (modelId == CameraModelId.Equirectangular)
		{
			options.CameraParams =
				Camera.CreateFromModelId(1, modelId, 0.0, options.CameraWidth, options.CameraHeight).Params;
		}

		Synthetic.SynthesizeDataset(options, reconstruction);
		return reconstruction;
	}

	[Test]
	[Arguments(TriangulationEstimator.ResidualType.AngularError)]
	[Arguments(TriangulationEstimator.ResidualType.ReprojectionError)]
	public async Task CSharpOnly_EstimateTriangulation_Nominal(TriangulationEstimator.ResidualType residualType)
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = SynthesizeScene();
		var options = new EstimateTriangulationOptions { ResidualType = residualType };
		if (residualType == TriangulationEstimator.ResidualType.ReprojectionError)
		{
			options.RansacOptions.MaxError = 2.0;
		}

		var log = new ExpectationLog();
		int numTested = 0;
		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			var (points, camsFromWorld, cameras) = TrackObservations(reconstruction, point3D);
			if (points.Count < 2)
			{
				continue;
			}

			numTested++;
			bool success = TriangulationEstimation.EstimateTriangulation(
				options, points, camsFromWorld, cameras, out bool[] inlierMask, out Vector3d xyz);
			log.True(success, $"point {point3DId}: success");
			log.True(inlierMask.Length == points.Count && inlierMask.All(x => x), $"point {point3DId}: all inliers");
			log.Less((xyz - point3D.Xyz).Norm, 1e-6, $"point {point3DId}: xyz");
		}

		await Assert.That(numTested).IsGreaterThan(0);
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task CSharpOnly_EstimateTriangulation_RejectsOutlier()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = SynthesizeScene();
		var options = new EstimateTriangulationOptions();

		var log = new ExpectationLog();
		int numTested = 0;
		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			var (points, camsFromWorld, cameras) = TrackObservations(reconstruction, point3D);
			if (points.Count < 4)
			{
				continue;
			}

			// One grossly wrong observation among at least three correct ones.
			points[0] += new Vector2d(150, -120);
			numTested++;
			bool success = TriangulationEstimation.EstimateTriangulation(
				options, points, camsFromWorld, cameras, out bool[] inlierMask, out Vector3d xyz);
			log.True(success, $"point {point3DId}: success");
			log.True(inlierMask.Length == points.Count && !inlierMask[0] && inlierMask.Skip(1).All(x => x), $"point {point3DId}: mask");
			log.Less((xyz - point3D.Xyz).Norm, 1e-6, $"point {point3DId}: xyz");
		}

		await Assert.That(numTested).IsGreaterThan(0);
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task CSharpOnly_EstimateTriangulation_Equirectangular()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction reconstruction = SynthesizeScene(CameraModelId.Equirectangular);
		var options = new EstimateTriangulationOptions();

		var log = new ExpectationLog();
		int numTested = 0;
		foreach ((ulong point3DId, Point3D point3D) in reconstruction.Points3D)
		{
			var (points, camsFromWorld, cameras) = TrackObservations(reconstruction, point3D);
			if (points.Count < 2)
			{
				continue;
			}

			numTested++;
			bool success = TriangulationEstimation.EstimateTriangulation(
				options, points, camsFromWorld, cameras, out bool[] _, out Vector3d xyz);
			log.True(success, $"point {point3DId}: success");
			log.Less((xyz - point3D.Xyz).Norm, 1e-6, $"point {point3DId}: xyz");
		}

		await Assert.That(numTested).IsGreaterThan(0);
		await Assert.That(log.Failures).IsEmpty();
	}

	[Test]
	public async Task CSharpOnly_EstimateTriangulation_MinTriAngle()
	{
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 100, 200, 200);
		var point3D = new Vector3d(0, 0, 10);
		// Baseline 0.1 at depth 10: a triangulation angle of about 0.57 degrees.
		Rigid3d[] camsFromWorld = [new Rigid3d(), new Rigid3d(Quaterniond.Identity, new Vector3d(-0.1, 0, 0))];
		var points = camsFromWorld.Select(t => camera.ImgFromCam(t * point3D)!.Value).ToList();
		Camera[] cameras = [camera, camera];

		var options = new EstimateTriangulationOptions { MinTriAngle = MathUtils.DegToRad(0.5) };
		bool accepted = TriangulationEstimation.EstimateTriangulation(options, points, camsFromWorld, cameras, out _, out Vector3d xyz);
		options.MinTriAngle = MathUtils.DegToRad(1.0);
		bool rejected = !TriangulationEstimation.EstimateTriangulation(options, points, camsFromWorld, cameras, out _, out _);

		await Assert.That(accepted).IsTrue();
		await Assert.That((xyz - point3D).Norm).IsLessThan(1e-8);
		await Assert.That(rejected).IsTrue();
	}
}
