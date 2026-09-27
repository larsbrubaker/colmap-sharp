// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// GlobalSfmOracleTests (C#-only): global SfM against pycolmap 4.2.0 on the same databases
// (oracle/fixture_global_sfm.py writes TestData/oracle/global_sfm.json). Two entry points:
// RotationAveraging.RunRotationAveraging (pycolmap.run_rotation_averaging) and the whole
// GlobalPipeline (pycolmap.global_mapping: rotation averaging, track establishment, global
// positioning, bundle adjustment, retriangulation). The cases are noise-free, noisy with
// outlier matches and corrupted relative rotations, and two cameras per rig.
//
// Tier C (outcome). The fixture carries the whole database the pipeline reads, so both sides
// start from identical input and the same seed (single-threaded); what can differ is Eigen vs.
// LinearAlgebra/ and Ceres vs. Solver/. Compared per case:
// - the registered images (exact);
// - rotations: the largest relative rotation error between any two registered images of ours
//   and of pycolmap (gauge invariant, so no alignment is needed);
// - positions: our projection centers mapped onto pycolmap's by a Sim3 (EstimateSim3d), the
//   largest residual relative to the scene's radius (RMS distance of pycolmap's centers from
//   their centroid);
// - the number of 3D points after retriangulation (exact on these small scenes, where every
//   synthetic track is recovered).
// The bounds, and the errors measured when the fixture was written, are stated with the
// constants below. Ours is also checked against the ground truth, so a failure both sides
// share cannot pass silently.

using System.Text.Json;

using ColmapSharp.Controllers;
using ColmapSharp.Estimators;
using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public class GlobalSfmOracleTests
{
	private const string FixtureName = "global_sfm.json";

	// Measured when the fixture was written (largest over the cases, all in noisy_outliers):
	// rotation averaging 8.7e-8 deg from pycolmap, the pipeline 5.8e-8 deg and 1.7e-9 of the
	// scene radius; the noise-free and rig cases agree to 1e-13 deg (rotation averaging) and
	// 1e-8 deg / 5e-10 (pipeline). Both sides converge to the same solution, and the bounds
	// leave about two orders of magnitude for the solvers' stopping points.
	private const double RotationAveragingToleranceDeg = 1e-5;
	private const double PipelineRotationToleranceDeg = 1e-5;
	private const double PipelinePositionTolerance = 1e-7;

	// Against the ground truth, so a failure both sides share cannot pass. Measured: on the
	// noise-free and rig cases rotation averaging recovers it to 1e-13 deg (the synthetic
	// relative poses are exact) and the pipeline to 2e-6 deg and 1.1e-7 of the radius; with
	// three corrupted relative rotations rotation averaging stays within 5.4e-4 deg, and the
	// pipeline, bundle adjusted on 0.5 px noise, lands 0.071 deg and 3.0e-3 of the radius away.
	private const double GroundTruthRotationAveragingToleranceDeg = 5e-3;
	private const double GroundTruthPipelineRotationToleranceDeg = 0.2;
	private const double GroundTruthPositionTolerance = 0.01;

	public static IEnumerable<string> CaseNames() =>
		OracleFixture.Load(FixtureName).GetProperty("cases").EnumerateArray().Select(c => c.GetProperty("name").GetString()!);

	[Test]
	[MethodDataSource(nameof(CaseNames))]
	public async Task CSharpOnly_RotationAveragingMatchesPycolmap(string caseName)
	{
		JsonElement expected = Case(caseName);
		JsonElement expectedResult = expected.GetProperty("rotation_averaging");
		uint seed = OracleFixture.Load(FixtureName).GetProperty("random_seed").GetUInt32();

		using InMemoryDatabase database = BuildDatabase(expected.GetProperty("database"));
		var cache = new DatabaseCache();
		cache.Load(database, new DatabaseCache.Options());
		var reconstruction = new Reconstruction();
		reconstruction.Load(cache);
		var poseGraph = new PoseGraph();
		poseGraph.Load(cache.CorrespondenceGraph);
		bool success = RotationAveraging.RunRotationAveraging(
			new RotationEstimatorOptions { RandomSeed = (int)seed }, poseGraph, reconstruction, database.ReadAllPosePriors());

		Dictionary<uint, Quaterniond> ours = reconstruction.RegImageIds().ToDictionary(
			id => id, id => reconstruction.Image(id).CamFromWorld().Rotation);
		Dictionary<uint, Quaterniond> theirs = expectedResult.GetProperty("rotations").EnumerateArray().ToDictionary(
			r => r.GetProperty("image_id").GetUInt32(), r => Quaternion(r.GetProperty("q_xyzw")));
		Dictionary<uint, Quaterniond> groundTruth = Poses(expected.GetProperty("gt_cam_from_world"))
			.ToDictionary(p => p.Key, p => p.Value.Rotation);

		await Assert.That(success).IsEqualTo(expectedResult.GetProperty("success").GetBoolean());
		await Assert.That(ours.Keys.Order()).IsEquivalentTo(theirs.Keys.Order(), TUnit.Assertions.Enums.CollectionOrdering.Matching);
		await Assert.That(MaxRelativeRotationErrorDeg(ours, theirs)).IsLessThanOrEqualTo(RotationAveragingToleranceDeg);
		await Assert.That(MaxRelativeRotationErrorDeg(ours, groundTruth)).IsLessThanOrEqualTo(GroundTruthRotationAveragingToleranceDeg);
	}

	[Test]
	[MethodDataSource(nameof(CaseNames))]
	public async Task CSharpOnly_GlobalPipelineMatchesPycolmap(string caseName)
	{
		JsonElement expected = Case(caseName);
		uint seed = OracleFixture.Load(FixtureName).GetProperty("random_seed").GetUInt32();

		using InMemoryDatabase database = BuildDatabase(expected.GetProperty("database"));
		var reconstructionManager = new ReconstructionManager();
		var options = new GlobalPipelineOptions { NumThreads = 1, RandomSeed = (int)seed };
		new GlobalPipeline(options, database, reconstructionManager).Run();

		List<JsonElement> expectedModels = [.. expected.GetProperty("pipeline").EnumerateArray()];
		Dictionary<uint, Rigid3d> groundTruth = Poses(expected.GetProperty("gt_cam_from_world"));
		await Assert.That(reconstructionManager.Size).IsEqualTo(expectedModels.Count);
		for (int i = 0; i < expectedModels.Count; ++i)
		{
			Reconstruction reconstruction = reconstructionManager.Get(i);
			Dictionary<uint, Rigid3d> ours = reconstruction.RegImageIds().ToDictionary(
				id => id, id => reconstruction.Image(id).CamFromWorld());
			Dictionary<uint, Rigid3d> theirs = Poses(expectedModels[i].GetProperty("cam_from_world"));

			await Assert.That(ours.Keys.Order()).IsEquivalentTo(theirs.Keys.Order(), TUnit.Assertions.Enums.CollectionOrdering.Matching);
			await Assert.That(reconstruction.NumPoints3D).IsEqualTo(expectedModels[i].GetProperty("num_points3D").GetInt32());

			Dictionary<uint, Quaterniond> ourRotations = ours.ToDictionary(p => p.Key, p => p.Value.Rotation);
			await Assert.That(MaxRelativeRotationErrorDeg(ourRotations, theirs.ToDictionary(p => p.Key, p => p.Value.Rotation)))
				.IsLessThanOrEqualTo(PipelineRotationToleranceDeg);
			await Assert.That(MaxRelativeRotationErrorDeg(ourRotations, groundTruth.ToDictionary(p => p.Key, p => p.Value.Rotation)))
				.IsLessThanOrEqualTo(GroundTruthPipelineRotationToleranceDeg);
			await Assert.That(MaxAlignedPositionError(ours, theirs)).IsLessThanOrEqualTo(PipelinePositionTolerance);
			await Assert.That(MaxAlignedPositionError(ours, groundTruth)).IsLessThanOrEqualTo(GroundTruthPositionTolerance);
		}
	}

	private static JsonElement Case(string caseName) =>
		OracleFixture.Load(FixtureName).GetProperty("cases").EnumerateArray()
			.Single(c => c.GetProperty("name").GetString() == caseName);

	// The fixture's database, written with the recorded ids in the order the script read them.
	private static InMemoryDatabase BuildDatabase(JsonElement json)
	{
		var database = new InMemoryDatabase();
		foreach (JsonElement c in json.GetProperty("cameras").EnumerateArray())
		{
			var camera = new Camera
			{
				CameraId = c.GetProperty("camera_id").GetUInt32(),
				ModelId = (CameraModelId)c.GetProperty("model_id").GetInt32(),
				Width = c.GetProperty("width").GetInt32(),
				Height = c.GetProperty("height").GetInt32(),
				Params = OracleFixture.Doubles(c.GetProperty("params")),
				HasPriorFocalLength = c.GetProperty("has_prior_focal_length").GetBoolean(),
			};
			database.WriteCamera(camera, useCameraId: true);
		}

		foreach (JsonElement r in json.GetProperty("rigs").EnumerateArray())
		{
			var rig = new Rig { RigId = r.GetProperty("rig_id").GetUInt32() };
			rig.AddRefSensor(CameraSensor(r.GetProperty("ref_camera_id").GetUInt32()));
			foreach (JsonElement s in r.GetProperty("sensors").EnumerateArray())
			{
				rig.AddSensor(CameraSensor(s.GetProperty("camera_id").GetUInt32()), Pose(s.GetProperty("sensor_from_rig")));
			}

			database.WriteRig(rig, useRigId: true);
		}

		foreach (JsonElement im in json.GetProperty("images").EnumerateArray())
		{
			var image = new Image { ImageId = im.GetProperty("image_id").GetUInt32(), Name = im.GetProperty("name").GetString()! };
			image.SetCameraId(im.GetProperty("camera_id").GetUInt32());
			database.WriteImage(image, useImageId: true);
			float[] xy = [.. im.GetProperty("keypoints").EnumerateArray().Select(v => v.GetSingle())];
			var keypoints = new List<FeatureKeypoint>(xy.Length / 2);
			for (int k = 0; k < xy.Length; k += 2)
			{
				keypoints.Add(new FeatureKeypoint(xy[k], xy[k + 1]));
			}

			database.WriteKeypoints(image.ImageId, keypoints);
		}

		foreach (JsonElement f in json.GetProperty("frames").EnumerateArray())
		{
			var frame = new Frame { FrameId = f.GetProperty("frame_id").GetUInt32() };
			frame.SetRigId(f.GetProperty("rig_id").GetUInt32());
			foreach (JsonElement d in f.GetProperty("data_ids").EnumerateArray())
			{
				frame.AddDataId(new DataId(CameraSensor(d[0].GetUInt32()), d[1].GetUInt32()));
			}

			database.WriteFrame(frame, useFrameId: true);
		}

		foreach (JsonElement p in json.GetProperty("pairs").EnumerateArray())
		{
			var twoViewGeometry = new TwoViewGeometry
			{
				Config = (TwoViewGeometry.ConfigurationType)p.GetProperty("config").GetInt32(),
				E = Matrix(p.GetProperty("E")),
				F = Matrix(p.GetProperty("F")),
				H = Matrix(p.GetProperty("H")),
				Cam2FromCam1 = p.GetProperty("cam2_from_cam1").ValueKind == JsonValueKind.Null
					? null
					: Pose(p.GetProperty("cam2_from_cam1")),
			};
			long[] matches = OracleFixture.Int64s(p.GetProperty("inlier_matches"));
			for (int k = 0; k < matches.Length; k += 2)
			{
				twoViewGeometry.InlierMatches.Add(new FeatureMatch((uint)matches[k], (uint)matches[k + 1]));
			}

			database.WriteTwoViewGeometry(
				p.GetProperty("image_id1").GetUInt32(), p.GetProperty("image_id2").GetUInt32(), twoViewGeometry);
		}

		return database;
	}

	private static SensorId CameraSensor(uint cameraId) => new(SensorType.Camera, cameraId);

	// numpy's reshape(-1) of a 3x3 matrix is row-major.
	private static Matrix3d? Matrix(JsonElement json)
	{
		if (json.ValueKind == JsonValueKind.Null)
		{
			return null;
		}

		double[] v = OracleFixture.Doubles(json);
		return Matrix3d.FromRows(new Vector3d(v[0], v[1], v[2]), new Vector3d(v[3], v[4], v[5]), new Vector3d(v[6], v[7], v[8]));
	}

	private static Quaterniond Quaternion(JsonElement xyzw)
	{
		double[] q = OracleFixture.Doubles(xyzw);
		return new Quaterniond(q[3], q[0], q[1], q[2]);
	}

	private static Rigid3d Pose(JsonElement json)
	{
		double[] t = OracleFixture.Doubles(json.GetProperty("t"));
		return new Rigid3d(Quaternion(json.GetProperty("q_xyzw")), new Vector3d(t[0], t[1], t[2]));
	}

	private static Dictionary<uint, Rigid3d> Poses(JsonElement json) =>
		json.EnumerateArray().ToDictionary(p => p.GetProperty("image_id").GetUInt32(), Pose);

	// The largest angle between the relative rotation of two images in `a` and in `b`, over
	// every pair of images in `a` (degrees). Gauge invariant.
	private static double MaxRelativeRotationErrorDeg(Dictionary<uint, Quaterniond> a, Dictionary<uint, Quaterniond> b)
	{
		List<uint> ids = [.. a.Keys.Order()];
		double maxError = 0;
		for (int i = 0; i < ids.Count; i++)
		{
			for (int j = 0; j < i; j++)
			{
				Quaterniond relA = a[ids[j]] * a[ids[i]].Inverse();
				Quaterniond relB = b[ids[j]] * b[ids[i]].Inverse();
				maxError = Math.Max(maxError, relA.AngularDistance(relB));
			}
		}

		return maxError * 180 / Math.PI;
	}

	// Maps the projection centers of `ours` onto those of `reference` by a Sim3 and returns the
	// largest residual relative to the radius of the reference centers.
	private static double MaxAlignedPositionError(Dictionary<uint, Rigid3d> ours, Dictionary<uint, Rigid3d> reference)
	{
		List<uint> ids = [.. ours.Keys.Order()];
		Vector3d[] src = [.. ids.Select(id => ours[id].TgtOriginInSrc())];
		Vector3d[] tgt = [.. ids.Select(id => reference[id].TgtOriginInSrc())];
		var tgtFromSrc = new Sim3d();
		Check.That(SimilarityTransform.EstimateSim3d(src, tgt, ref tgtFromSrc));

		Vector3d centroid = new();
		foreach (Vector3d t in tgt)
		{
			centroid += t;
		}

		centroid /= tgt.Length;
		double radius = Math.Sqrt(tgt.Sum(t => (t - centroid).SquaredNorm) / tgt.Length);
		double maxError = 0;
		for (int k = 0; k < src.Length; ++k)
		{
			maxError = Math.Max(maxError, (tgtFromSrc * src[k] - tgt[k]).Norm);
		}

		return maxError / radius;
	}
}
