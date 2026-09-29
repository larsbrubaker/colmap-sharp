// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SyntheticOracleTests (C#-only): ColmapSharp.Scene.Synthetic.SynthesizeDataset against
// pycolmap 4.2.0's synthesize_dataset on the same seed (oracle/fixture_synthetic.py writes
// TestData/oracle/synthetic.json). This pins the PRNG draw order: a single extra or missing
// draw shifts every later value.
//
// Tiers, per value:
// - Tier A (exact): ids, names, counts, camera parameters, 3D point positions (uniform draws),
//   frame rotations (uniform draws + shortest-arc quaternion), and the index and position of
//   every 2D point without a 3D point (uniform draws with min 0, and the shuffle permutation).
// - Tier B: sensor-from-rig poses (1e-12; Gaussian draws, divergence 1);
//   frame translations (1e-14) and 2D projections (1e-9 px), which differ from the wheel in
//   the last ulp where its quaternion rotation is contracted into FMAs (entry 31); 3D point
//   errors (1e-9 absolute, means of tiny reprojection errors).
// - Order-insensitive: which 3D point sits at which 2D index, and so the point2D_idx of track
//   elements. COLMAP projects the points in hash order before the shuffle, ColmapSharp in
//   ascending id order (entry 31); the shuffle permutation is identical, so the set of indices
//   holding a 3D point and each point's image coordinates still match.

using System.Text.Json;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class SyntheticOracleTests
{
	private const double GaussianTolerance = 1e-12;

	// A few ulps of the translation (|t| = 5, ulp 8.9e-16): the wheel's quaternion rotation
	// is contracted into FMAs, ColmapSharp's is not (divergence 31).
	private const double ContractionTolerance = 1e-14;

	// Projections to pixels (~1e3) inherit that last-ulp pose difference.
	private const double ProjectionTolerance = 1e-9;

	// Sums over an image's ~2000 2D points of values carrying ProjectionTolerance-sized
	// rounding (noise deltas, clean projections): a few ulps each, accumulated.
	private const double SummaryTolerance = 1e-8;

	// The same sums weighted by the 3D point id (up to ~2000): SummaryTolerance times the id.
	private const double IdWeightedSummaryTolerance = 1e-4;

	public static IEnumerable<string> CaseNames() =>
		OracleFixture.Load("synthetic.json").GetProperty("cases").EnumerateArray().Select(c => c.GetProperty("name").GetString()!);

	[Test]
	[MethodDataSource(nameof(CaseNames))]
	public async Task CSharpOnly_SynthesizeDatasetMatchesPycolmap(string caseName)
	{
		JsonElement expected = OracleFixture.Load("synthetic.json").GetProperty("cases").EnumerateArray()
			.Single(c => c.GetProperty("name").GetString() == caseName);

		var options = new SyntheticDatasetOptions();
		foreach (JsonProperty option in expected.GetProperty("options").EnumerateObject())
		{
			int value = option.Value.GetInt32();
			switch (option.Name)
			{
				case "num_rigs": options.NumRigs = value; break;
				case "num_cameras_per_rig": options.NumCamerasPerRig = value; break;
				case "num_frames_per_rig": options.NumFramesPerRig = value; break;
				case "num_points3D": options.NumPoints3D = value; break;
				case "num_points2D_without_point3D": options.NumPoints2DWithoutPoint3D = value; break;
				default: throw new InvalidOperationException($"Unknown option {option.Name}");
			}
		}

		var reconstruction = new Reconstruction();
		RandomUtils.SetPRNGSeed(expected.GetProperty("seed").GetUInt32());
		Synthetic.SynthesizeDataset(options, reconstruction);

		var failures = new List<string>();
		CompareCameras(expected, reconstruction, failures);
		CompareRigs(expected, reconstruction, failures);
		CompareFrames(expected, reconstruction, failures);
		ComparePoints3D(expected, reconstruction, failures);
		CompareImages(expected, reconstruction, failures);
		await Assert.That(string.Join("\n", failures.Take(60))).IsEqualTo("");
	}

	public static IEnumerable<string> NoiseCaseNames() =>
		OracleFixture.Load("synthetic.json").GetProperty("noise_cases").EnumerateArray().Select(c => c.GetProperty("name").GetString()!);

	/// <summary>
	/// Track-length pruning followed by SynthesizeNoise, against pycolmap. Points and images are
	/// visited in a different order than COLMAP's hash order (entry 31), so the comparison is
	/// order-insensitive: every track has the target length and the same 3D points exist; frame
	/// poses after noise match per frame (same visiting order; 1e-12, Gaussian draws, entry 1);
	/// and the noise added per image (a 2 * NumPoints2D chunk of the Gaussian stream) and per 3D
	/// point (a 3-draw chunk) matches as a multiset. The chunks only line up if pruning consumed
	/// exactly as many draws as COLMAP's. Deltas are noisy minus clean values, so they carry the
	/// rounding of the addition: 1e-9 px for 2D points (~1e3), 1e-12 for 3D points (~1).
	/// Cases too large to record every 2D point (the fixture's SUMMARIZED_NOISE_CASES, e.g. the
	/// "Nominal" noise of bundle_adjustment_test.cc on 100 frames and 2000 points) match each
	/// image's noise chunk by a summary (count, first three deltas, sums), and pin the clean
	/// dataset by per-image observation summaries and the exact sum of the 3D point positions.
	/// </summary>
	[Test]
	[MethodDataSource(nameof(NoiseCaseNames))]
	public async Task CSharpOnly_TrackLengthAndNoiseMatchPycolmap(string caseName)
	{
		JsonElement expected = OracleFixture.Load("synthetic.json").GetProperty("noise_cases").EnumerateArray()
			.Single(c => c.GetProperty("name").GetString() == caseName);

		JsonElement o = expected.GetProperty("options");
		var options = new SyntheticDatasetOptions
		{
			NumRigs = o.GetProperty("num_rigs").GetInt32(),
			NumCamerasPerRig = o.GetProperty("num_cameras_per_rig").GetInt32(),
			NumFramesPerRig = o.GetProperty("num_frames_per_rig").GetInt32(),
			NumPoints3D = o.GetProperty("num_points3D").GetInt32(),
		};
		if (o.TryGetProperty("track_length", out JsonElement trackLength))
		{
			options.TrackLength = trackLength.GetInt32();
		}

		JsonElement n = expected.GetProperty("noise_options");
		var noiseOptions = new SyntheticNoiseOptions
		{
			RigFromWorldTranslationStddev = n.GetProperty("rig_from_world_translation_stddev").GetDouble(),
			RigFromWorldRotationStddev = n.GetProperty("rig_from_world_rotation_stddev").GetDouble(),
			Point3DStddev = n.GetProperty("point3D_stddev").GetDouble(),
			Point2DStddev = n.GetProperty("point2D_stddev").GetDouble(),
		};

		var reconstruction = new Reconstruction();
		RandomUtils.SetPRNGSeed(expected.GetProperty("seed").GetUInt32());
		Synthetic.SynthesizeDataset(options, reconstruction);
		var trackLengths = reconstruction.Points3D.ToDictionary(kv => kv.Key, kv => kv.Value.Track.Length);
		var clean2D = reconstruction.Images.ToDictionary(kv => kv.Key, kv => kv.Value.Points2D.Select(p => p.Xy).ToList());
		var clean3D = reconstruction.Points3D.ToDictionary(kv => kv.Key, kv => kv.Value.Xyz);
		bool summarized = expected.TryGetProperty("points2D_delta_summaries", out JsonElement expectedSummaries);
		var failures = new List<string>();
		if (summarized)
		{
			CompareCleanSummaries(expected, reconstruction, failures);
		}

		Synthetic.SynthesizeNoise(noiseOptions, reconstruction);

		ulong[] expectedIds = OracleFixture.UInt64s(expected.GetProperty("point3D_ids"));
		long[] expectedLengths = OracleFixture.Int64s(expected.GetProperty("track_lengths"));
		if (!trackLengths.Keys.Order().SequenceEqual(expectedIds)
			|| !trackLengths.OrderBy(kv => kv.Key).Select(kv => (long)kv.Value).SequenceEqual(expectedLengths))
		{
			failures.Add("3D point ids or track lengths differ");
		}

		foreach (JsonElement frame in expected.GetProperty("frames").EnumerateArray())
		{
			uint frameId = frame.GetProperty("frame_id").GetUInt32();
			if (!RigidNear(reconstruction.Frame(frameId).RigFromWorld(), ReadRigid(frame.GetProperty("rig_from_world")), GaussianTolerance))
			{
				failures.Add($"frame {frameId} pose after noise {reconstruction.Frame(frameId).RigFromWorld()}");
			}
		}

		var actual2D = reconstruction.Images.Select(kv => kv.Value.Points2D
			.Select((p, i) => new[] { p.Xy.X - clean2D[kv.Key][i].X, p.Xy.Y - clean2D[kv.Key][i].Y }).ToList()).ToList();
		if (summarized)
		{
			var actualSummaries = actual2D.Select(deltas => new List<double[]> { SummarizeImageNoise(deltas) }).ToList();
			var expectedSummaryChunks = expectedSummaries.EnumerateArray()
				.Select(summary => new List<double[]> { OracleFixture.Doubles(summary) }).ToList();
			MatchChunks("image 2D noise summary", actualSummaries, expectedSummaryChunks, SummaryTolerance, failures);
		}
		else
		{
			var expected2D = expected.GetProperty("points2D_deltas").EnumerateArray()
				.Select(image => image.EnumerateArray().Select(OracleFixture.Doubles).ToList()).ToList();
			MatchChunks("image 2D noise", actual2D, expected2D, ProjectionTolerance, failures);
		}

		var actual3D = reconstruction.Points3D.Select(kv =>
		{
			Vector3d delta = kv.Value.Xyz - clean3D[kv.Key];
			return new List<double[]> { new[] { delta.X, delta.Y, delta.Z } };
		}).ToList();
		var expected3D = expected.GetProperty("points3D_deltas").EnumerateArray()
			.Select(point => new List<double[]> { OracleFixture.Doubles(point) }).ToList();
		MatchChunks("3D point noise", actual3D, expected3D, GaussianTolerance, failures);

		await Assert.That(string.Join("\n", failures.Take(20))).IsEqualTo("");
	}

	// [count, first three (dx, dy), sum dx, sum dy, sum dx^2 + dy^2], summed in index order;
	// oracle/fixture_synthetic.py summarize_image_noise.
	private static double[] SummarizeImageNoise(List<double[]> deltas)
	{
		var summary = new List<double> { deltas.Count };
		summary.AddRange(deltas.Take(3).SelectMany(d => d));
		double sumX = 0, sumY = 0, sumSquares = 0;
		foreach (double[] d in deltas)
		{
			sumX += d[0];
			sumY += d[1];
			sumSquares += d[0] * d[0] + d[1] * d[1];
		}

		summary.AddRange([sumX, sumY, sumSquares]);
		return summary.ToArray();
	}

	// The clean dataset of a summarized noise case, order-insensitively where hash order
	// decides the layout (entry 31): per image the number of 2D points observing a 3D point,
	// every point without one exactly (index and position: same shuffle permutation), and sums
	// x, y, id * x, id * y over the observing points in ascending id order (so which point
	// projects where is pinned, not which index it sits at); the sum of the 3D point positions
	// in id order exactly (uniform draws). Mirrors fixture_synthetic.py summarize_observations.
	private static void CompareCleanSummaries(JsonElement expected, Reconstruction reconstruction, List<string> failures)
	{
		double[] expectedSum = OracleFixture.Doubles(expected.GetProperty("points3D_clean_sum"));
		var sum = new double[3];
		foreach (Point3D point3D in reconstruction.Points3D.OrderBy(kv => kv.Key).Select(kv => kv.Value))
		{
			sum[0] += point3D.Xyz.X;
			sum[1] += point3D.Xyz.Y;
			sum[2] += point3D.Xyz.Z;
		}

		if (!sum.SequenceEqual(expectedSum))
		{
			failures.Add($"clean 3D point sum ({string.Join(", ", sum)})");
		}

		foreach (JsonElement json in expected.GetProperty("observations").EnumerateArray())
		{
			Image image = reconstruction.Image(json.GetProperty("image_id").GetUInt32());
			var observing = image.Points2D.Where(p => p.HasPoint3D).OrderBy(p => p.Point3DId).ToList();
			if (observing.Count != json.GetProperty("num_with_point3D").GetInt32())
			{
				failures.Add($"image {image.ImageId}: {observing.Count} 2D points observe a 3D point");
			}

			var without = image.Points2D.Select((p, idx) => (p, idx)).Where(e => !e.p.HasPoint3D).ToList();
			JsonElement expectedWithout = json.GetProperty("without_point3D");
			if (without.Count != expectedWithout.GetArrayLength()
				|| !without.Zip(expectedWithout.EnumerateArray()).All(pair => pair.First.idx == pair.Second[0].GetInt32()
					&& pair.First.p.Xy == new Vector2d(pair.Second[1].GetDouble(), pair.Second[2].GetDouble())))
			{
				failures.Add($"image {image.ImageId}: 2D points without a 3D point differ");
			}

			double sumX = 0, sumY = 0, sumIdX = 0, sumIdY = 0;
			foreach (Point2D point2D in observing)
			{
				sumX += point2D.Xy.X;
				sumY += point2D.Xy.Y;
				sumIdX += point2D.Point3DId * point2D.Xy.X;
				sumIdY += point2D.Point3DId * point2D.Xy.Y;
			}

			double[] e = OracleFixture.Doubles(json.GetProperty("sums"));
			if (Math.Abs(sumX - e[0]) > SummaryTolerance || Math.Abs(sumY - e[1]) > SummaryTolerance
				|| Math.Abs(sumIdX - e[2]) > IdWeightedSummaryTolerance || Math.Abs(sumIdY - e[3]) > IdWeightedSummaryTolerance)
			{
				failures.Add($"image {image.ImageId}: observation sums ({sumX:R}, {sumY:R}, {sumIdX:R}, {sumIdY:R})");
			}
		}
	}

	// Multiset match: every actual chunk pairs with a distinct expected chunk within tol.
	private static void MatchChunks(string what, List<List<double[]>> actual, List<List<double[]>> expected, double tol, List<string> failures)
	{
		if (actual.Count != expected.Count)
		{
			failures.Add($"{what}: {actual.Count} chunks != {expected.Count}");
			return;
		}

		var unmatched = new List<List<double[]>>(expected);
		foreach (List<double[]> chunk in actual)
		{
			int match = unmatched.FindIndex(candidate => candidate.Count == chunk.Count
				&& candidate.Zip(chunk).All(pair => pair.First.Zip(pair.Second).All(v => Math.Abs(v.First - v.Second) <= tol)));
			if (match < 0)
			{
				failures.Add($"{what}: chunk starting ({string.Join(", ", chunk[0])}) has no pycolmap counterpart");
				continue;
			}

			unmatched.RemoveAt(match);
		}
	}

	private static Rigid3d ReadRigid(JsonElement json)
	{
		double[] q = OracleFixture.Doubles(json.GetProperty("q_xyzw"));
		double[] t = OracleFixture.Doubles(json.GetProperty("t"));
		return new Rigid3d(new Quaterniond(q[3], q[0], q[1], q[2]), new Vector3d(t[0], t[1], t[2]));
	}

	private static bool RigidNear(Rigid3d a, Rigid3d b, double tol) =>
		Math.Abs(a.Rotation.W - b.Rotation.W) <= tol && Math.Abs(a.Rotation.X - b.Rotation.X) <= tol
		&& Math.Abs(a.Rotation.Y - b.Rotation.Y) <= tol && Math.Abs(a.Rotation.Z - b.Rotation.Z) <= tol
		&& (a.Translation - b.Translation).Norm <= tol;

	private static void CompareCameras(JsonElement expected, Reconstruction reconstruction, List<string> failures)
	{
		JsonElement cameras = expected.GetProperty("cameras");
		if (cameras.GetArrayLength() != reconstruction.NumCameras)
		{
			failures.Add($"camera count {reconstruction.NumCameras} != {cameras.GetArrayLength()}");
		}

		foreach (JsonElement json in cameras.EnumerateArray())
		{
			Camera camera = reconstruction.Camera(json.GetProperty("camera_id").GetUInt32());
			if ((int)camera.ModelId != json.GetProperty("model_id").GetInt32()
				|| camera.Width != json.GetProperty("width").GetInt32()
				|| camera.Height != json.GetProperty("height").GetInt32()
				|| !camera.Params.SequenceEqual(OracleFixture.Doubles(json.GetProperty("params"))))
			{
				failures.Add($"camera {camera.CameraId} differs");
			}
		}
	}

	private static void CompareRigs(JsonElement expected, Reconstruction reconstruction, List<string> failures)
	{
		foreach (JsonElement json in expected.GetProperty("rigs").EnumerateArray())
		{
			Rig rig = reconstruction.Rig(json.GetProperty("rig_id").GetUInt32());
			if (rig.RefSensorId.Id != json.GetProperty("ref_camera_id").GetUInt32())
			{
				failures.Add($"rig {rig.RigId} ref sensor differs");
			}

			JsonElement sensors = json.GetProperty("sensors");
			if (sensors.GetArrayLength() != rig.NonRefSensors.Count)
			{
				failures.Add($"rig {rig.RigId} sensor count differs");
				continue;
			}

			foreach (JsonElement sensor in sensors.EnumerateArray())
			{
				var sensorId = new SensorId(SensorType.Camera, sensor.GetProperty("camera_id").GetUInt32());
				if (!RigidNear(rig.SensorFromRig(sensorId), ReadRigid(sensor.GetProperty("sensor_from_rig")), GaussianTolerance))
				{
					failures.Add($"rig {rig.RigId} sensor_from_rig of {sensorId.Id}: {rig.SensorFromRig(sensorId)}");
				}
			}
		}
	}

	private static void CompareFrames(JsonElement expected, Reconstruction reconstruction, List<string> failures)
	{
		JsonElement frames = expected.GetProperty("frames");
		if (frames.GetArrayLength() != reconstruction.NumFrames)
		{
			failures.Add($"frame count {reconstruction.NumFrames} != {frames.GetArrayLength()}");
		}

		foreach (JsonElement json in frames.EnumerateArray())
		{
			Frame frame = reconstruction.Frame(json.GetProperty("frame_id").GetUInt32());
			Rigid3d expectedPose = ReadRigid(json.GetProperty("rig_from_world"));
			if (frame.RigId != json.GetProperty("rig_id").GetUInt32())
			{
				failures.Add($"frame {frame.FrameId} rig differs");
			}

			Rigid3d actualPose = frame.RigFromWorld();
			if (actualPose.Rotation.Coeffs != expectedPose.Rotation.Coeffs)
			{
				failures.Add($"frame {frame.FrameId} rotation {actualPose.Rotation.Coeffs - expectedPose.Rotation.Coeffs}");
			}

			if ((actualPose.Translation - expectedPose.Translation).Norm > ContractionTolerance)
			{
				failures.Add($"frame {frame.FrameId} translation diff {(actualPose.Translation - expectedPose.Translation).Norm:R}");
			}

			ulong[] imageIds = OracleFixture.UInt64s(json.GetProperty("image_ids"));
			if (!frame.ImageIds().Select(d => (ulong)d.Id).Order().SequenceEqual(imageIds))
			{
				failures.Add($"frame {frame.FrameId} image ids differ");
			}
		}
	}

	private static void ComparePoints3D(JsonElement expected, Reconstruction reconstruction, List<string> failures)
	{
		JsonElement points3D = expected.GetProperty("points3D");
		if (points3D.GetArrayLength() != reconstruction.NumPoints3D)
		{
			failures.Add($"3D point count {reconstruction.NumPoints3D} != {points3D.GetArrayLength()}");
		}

		foreach (JsonElement json in points3D.EnumerateArray())
		{
			ulong point3DId = json.GetProperty("point3D_id").GetUInt64();
			Point3D point3D = reconstruction.Point3D(point3DId);
			double[] xyz = OracleFixture.Doubles(json.GetProperty("xyz"));
			if (point3D.Xyz != new Vector3d(xyz[0], xyz[1], xyz[2]))
			{
				failures.Add($"point {point3DId} xyz {point3D.Xyz}");
			}

			if (Math.Abs(point3D.Error - json.GetProperty("error").GetDouble()) > 1e-9)
			{
				failures.Add($"point {point3DId} error {point3D.Error}");
			}

			// Image ids only: point2D_idx depends on the pre-shuffle order (file header).
			IEnumerable<long> expectedImageIds = json.GetProperty("track").EnumerateArray().Select(e => e[0].GetInt64()).Order();
			if (!point3D.Track.Elements.Select(e => (long)e.ImageId).Order().SequenceEqual(expectedImageIds))
			{
				failures.Add($"point {point3DId} track images differ");
			}
		}
	}

	private static void CompareImages(JsonElement expected, Reconstruction reconstruction, List<string> failures)
	{
		foreach (JsonElement json in expected.GetProperty("images").EnumerateArray())
		{
			Image image = reconstruction.Image(json.GetProperty("image_id").GetUInt32());
			if (image.Name != json.GetProperty("name").GetString()
				|| image.CameraId != json.GetProperty("camera_id").GetUInt32()
				|| image.FrameId != json.GetProperty("frame_id").GetUInt32())
			{
				failures.Add($"image {image.ImageId} name/camera/frame differ");
			}

			JsonElement points2D = json.GetProperty("points2D");
			if (points2D.GetArrayLength() != image.NumPoints2D)
			{
				failures.Add($"image {image.ImageId} 2D point count differs");
				continue;
			}

			var observedXy = new Dictionary<ulong, Vector2d>();
			for (int idx = 0; idx < image.NumPoints2D; ++idx)
			{
				Point2D actual = image.Points2D[idx];
				JsonElement point = points2D[idx];
				var expectedXy = new Vector2d(point[0].GetDouble(), point[1].GetDouble());
				long expectedId = point[2].GetInt64();
				if (expectedId < 0)
				{
					if (actual.HasPoint3D || actual.Xy != expectedXy)
					{
						failures.Add($"image {image.ImageId} random point {idx}: {actual.Xy} != {expectedXy}");
					}
				}
				else if (!actual.HasPoint3D)
				{
					failures.Add($"image {image.ImageId} point {idx} should observe a 3D point");
				}
				else
				{
					observedXy[actual.Point3DId] = actual.Xy;
				}
			}

			foreach (JsonElement point in points2D.EnumerateArray().Where(p => p[2].GetInt64() >= 0))
			{
				var expectedXy = new Vector2d(point[0].GetDouble(), point[1].GetDouble());
				if (!observedXy.TryGetValue(point[2].GetUInt64(), out Vector2d xy)
					|| (xy - expectedXy).Norm > ProjectionTolerance)
				{
					failures.Add($"image {image.ImageId} projection of point {point[2]}: {xy} != {expectedXy}");
				}
			}
		}
	}
}
