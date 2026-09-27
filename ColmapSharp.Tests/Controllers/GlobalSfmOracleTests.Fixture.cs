// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// GlobalSfmOracleTests.Fixture (C#-only): reads TestData/oracle/global_sfm.json for
// GlobalSfmOracleTests.cs. It rebuilds each case's input database (the whole database
// pycolmap's steps read) as an InMemoryDatabase and converts the recorded poses, and holds the
// gauge-invariant error measures the tests compare with.

using System.Text.Json;

using ColmapSharp.Estimators.Solvers;
using ColmapSharp.Feature;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Tests.Controllers;

public partial class GlobalSfmOracleTests
{
	private const string FixtureName = "global_sfm.json";

	private static JsonElement Fixture => OracleFixture.Load(FixtureName);

	private static int Seed => Fixture.GetProperty("random_seed").GetInt32();

	/// <summary>The names of the cases whose "steps" include <paramref name="step"/>.</summary>
	private static IEnumerable<string> CasesWithStep(string step) =>
		Fixture.GetProperty("cases").EnumerateArray()
			.Where(c => c.GetProperty("steps").EnumerateArray().Any(s => s.GetString() == step))
			.Select(c => c.GetProperty("name").GetString()!);

	private static JsonElement Case(string caseName) =>
		Fixture.GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("name").GetString() == caseName);

	// The fixture's database, written with the recorded ids.
	private static InMemoryDatabase BuildDatabase(JsonElement json)
	{
		var database = new InMemoryDatabase();
		foreach (JsonElement c in json.GetProperty("cameras").EnumerateArray())
		{
			database.WriteCamera(ReadCamera(c), useCameraId: true);
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
			uint imageId1 = p.GetProperty("image_id1").GetUInt32();
			uint imageId2 = p.GetProperty("image_id2").GetUInt32();
			var twoViewGeometry = new TwoViewGeometry
			{
				Config = (TwoViewGeometry.ConfigurationType)p.GetProperty("config").GetInt32(),
				E = Matrix(p.GetProperty("E")),
				F = Matrix(p.GetProperty("F")),
				H = Matrix(p.GetProperty("H")),
				Cam2FromCam1 = MaybePose(p.GetProperty("cam2_from_cam1")),
				InlierMatches = Matches(p.GetProperty("inlier_matches")),
			};
			database.WriteTwoViewGeometry(imageId1, imageId2, twoViewGeometry);
			if (p.TryGetProperty("matches", out JsonElement matches))
			{
				database.WriteMatches(imageId1, imageId2, Matches(matches));
			}
		}

		foreach (JsonElement pp in json.GetProperty("pose_priors").EnumerateArray())
		{
			var posePrior = new PosePrior
			{
				PosePriorId = pp.GetProperty("pose_prior_id").GetUInt32(),
				CorrDataId = new DataId(
					CameraSensor(pp.GetProperty("corr_camera_id").GetUInt32()), pp.GetProperty("corr_image_id").GetUInt32()),
				CoordinateSystem = (PosePriorCoordinateSystem)pp.GetProperty("coordinate_system").GetInt32(),
			};
			if (pp.GetProperty("gravity").ValueKind != JsonValueKind.Null)
			{
				posePrior.Gravity = Vector(pp.GetProperty("gravity"));
			}

			database.WritePosePrior(posePrior, usePosePriorId: true);
		}

		return database;
	}

	private static Camera ReadCamera(JsonElement c) => new()
	{
		CameraId = c.GetProperty("camera_id").GetUInt32(),
		ModelId = (CameraModelId)c.GetProperty("model_id").GetInt32(),
		Width = c.GetProperty("width").GetInt32(),
		Height = c.GetProperty("height").GetInt32(),
		Params = OracleFixture.Doubles(c.GetProperty("params")),
		HasPriorFocalLength = c.GetProperty("has_prior_focal_length").GetBoolean(),
	};

	private static List<FeatureMatch> Matches(JsonElement json)
	{
		long[] flat = OracleFixture.Int64s(json);
		var matches = new List<FeatureMatch>(flat.Length / 2);
		for (int k = 0; k < flat.Length; k += 2)
		{
			matches.Add(new FeatureMatch((uint)flat[k], (uint)flat[k + 1]));
		}

		return matches;
	}

	private static SensorId CameraSensor(uint cameraId) => new(SensorType.Camera, cameraId);

	private static Vector3d Vector(JsonElement json)
	{
		double[] v = OracleFixture.Doubles(json);
		return new Vector3d(v[0], v[1], v[2]);
	}

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

	private static Rigid3d Pose(JsonElement json) =>
		new(Quaternion(json.GetProperty("q_xyzw")), Vector(json.GetProperty("t")));

	private static Rigid3d? MaybePose(JsonElement json) =>
		json.ValueKind == JsonValueKind.Null ? null : Pose(json);

	private static Dictionary<uint, Rigid3d> Poses(JsonElement json) =>
		json.EnumerateArray().ToDictionary(p => p.GetProperty("image_id").GetUInt32(), Pose);

	private static double RadToDeg(double radians) => radians * 180 / Math.PI;

	// The angle between two directions (degrees).
	private static double AngleDeg(Vector3d a, Vector3d b) =>
		RadToDeg(Math.Acos(Math.Clamp(a.Dot(b) / (a.Norm * b.Norm), -1, 1)));

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

		return RadToDeg(maxError);
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
