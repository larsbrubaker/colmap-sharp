// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// CameraModelOracleTests (C#-only; complements the 1:1 port in ModelsTests.cs): every
// camera model against COLMAP through the pycolmap 4.2.0 Camera binding. The fixture is
// written by oracle/camera_models.py into TestData/oracle/camera_models.json: two random
// parameter sets per model, projections of a grid of camera points (plus points behind the
// camera with the cheirality check off), and unprojections (CamFromImg, which runs the
// iterative undistortion, and CamRayFromImg) of a grid of pixels. FOV gets a third set with
// omega = 5e-4 for its small-omega Taylor branch, and every case adds points on and next to
// the optical axis and pixels at and next to the principal point for the small-radius
// branches (FOV's radius^2 < 1e-4 Taylor branch among them).
//
// Tier A, with one documented exception (docs/CPP_DIVERGENCES.md, entry 12, "FMA contraction in the
// camera models"):
// - Which calls fail must match exactly, for every model and field.
// - Bit-identical: CamFromImg of the pinhole models that unproject through the iterative
//   undistortion (SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV) and the plain pinholes, all of
//   EQUIRECTANGULAR, and the pixel threshold. This pins Jet2, the 2x2 LU solve and the
//   Newton iteration bit for bit.
// - Everything else within 2e-14 * max(1, |expected|): the macOS arm64 pycolmap wheel fuses
//   single-statement multiply-adds (f * x + c1, u*u + v*v) into FMAs, which ColmapSharp
//   never does (CLAUDE.md, "No FMA"). The observed maximum is 1.6e-14 (FISHEYE, a pixel
//   near 0 where f * uu cancels c1); oracle/camera_models.py prints the FMA evidence.

using System.Globalization;
using System.Text.Json;

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sensor;

public class CameraModelOracleTests
{
	private const string Fixture = "camera_models.json";

	// Scaled tolerance for the fields the wheel's FMA contraction reaches, see the header.
	private const double ContractionTolerance = 2e-14;

	private static readonly HashSet<string> ExactCamFromImgModels =
		["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV", "EQUIRECTANGULAR"];

	public static IEnumerable<string> ModelNames() =>
		OracleFixture.Load(Fixture).GetProperty("cases").EnumerateArray()
			.Select(c => c.GetProperty("model").GetString()!).Distinct();

	[Test]
	[MethodDataSource(nameof(ModelNames))]
	public async Task MatchesPycolmap(string modelName)
	{
		JsonElement root = OracleFixture.Load(Fixture);
		double[][] camPoints = Rows(root.GetProperty("cam_points"));
		double[][] backPoints = Rows(root.GetProperty("back_points"));
		double[][] pixels = Rows(root.GetProperty("pixels"));
		double[][] nearAxisPoints = Rows(root.GetProperty("near_axis_points"));
		CameraModelId id = CameraModels.CameraModelNameToId(modelName);
		var failures = new List<string>();
		bool allExact = modelName == "EQUIRECTANGULAR";

		foreach (JsonElement testCase in root.GetProperty("cases").EnumerateArray())
		{
			if (testCase.GetProperty("model").GetString() != modelName)
			{
				continue;
			}

			double[] parameters = OracleFixture.Doubles(testCase.GetProperty("params"));
			Compare(failures, true, "threshold", [CameraModels.CameraModelCamFromImgThreshold(id, parameters, 1.5)], [testCase.GetProperty("threshold").GetDouble()]);

			JsonElement[] imgFromCam = testCase.GetProperty("img_from_cam").EnumerateArray().ToArray();
			for (int i = 0; i < camPoints.Length; i++)
			{
				double[] p = camPoints[i];
				Vector2d? xy = CameraModels.CameraModelImgFromCam(id, parameters, new Vector3d(p[0], p[1], p[2]));
				Compare(failures, allExact, $"img_from_cam[{i}]", xy is { } v ? [v.X, v.Y] : null, Optional(imgFromCam[i]));
			}

			JsonElement[] imgFromCamBack = testCase.GetProperty("img_from_cam_back").EnumerateArray().ToArray();
			for (int i = 0; i < backPoints.Length; i++)
			{
				double[] p = backPoints[i];
				Vector2d? xy = CameraModels.CameraModelImgFromCam(id, parameters, new Vector3d(p[0], p[1], p[2]), checkCheirality: false);
				Compare(failures, allExact, $"img_from_cam_back[{i}]", xy is { } v ? [v.X, v.Y] : null, Optional(imgFromCamBack[i]));
			}

			JsonElement[] nearAxis = testCase.GetProperty("near_axis_img_from_cam").EnumerateArray().ToArray();
			for (int i = 0; i < nearAxisPoints.Length; i++)
			{
				double[] p = nearAxisPoints[i];
				Vector2d? xy = CameraModels.CameraModelImgFromCam(id, parameters, new Vector3d(p[0], p[1], p[2]));
				Compare(failures, allExact, $"near_axis_img_from_cam[{i}]", xy is { } v ? [v.X, v.Y] : null, Optional(nearAxis[i]));
			}

			double[][] ppPixels = Rows(testCase.GetProperty("pp_pixels"));
			JsonElement[] ppCamFromImg = testCase.GetProperty("pp_cam_from_img").EnumerateArray().ToArray();
			JsonElement[] ppCamRayFromImg = testCase.GetProperty("pp_cam_ray_from_img").EnumerateArray().ToArray();
			for (int i = 0; i < ppPixels.Length; i++)
			{
				var pixel = new Vector2d(ppPixels[i][0], ppPixels[i][1]);
				Vector2d? uv = CameraModels.CameraModelCamFromImg(id, parameters, pixel);
				Compare(failures, ExactCamFromImgModels.Contains(modelName), $"pp_cam_from_img[{i}]", uv is { } v ? [v.X, v.Y] : null, Optional(ppCamFromImg[i]));
				Vector3d? ray = CameraModels.CameraModelCamRayFromImg(id, parameters, pixel);
				Compare(failures, allExact, $"pp_cam_ray_from_img[{i}]", ray is { } r ? [r.X, r.Y, r.Z] : null, Optional(ppCamRayFromImg[i]));
			}

			JsonElement[] camFromImg = testCase.GetProperty("cam_from_img").EnumerateArray().ToArray();
			JsonElement[] camRayFromImg = testCase.GetProperty("cam_ray_from_img").EnumerateArray().ToArray();
			for (int i = 0; i < pixels.Length; i++)
			{
				var pixel = new Vector2d(pixels[i][0], pixels[i][1]);
				Vector2d? uv = CameraModels.CameraModelCamFromImg(id, parameters, pixel);
				Compare(failures, ExactCamFromImgModels.Contains(modelName), $"cam_from_img[{i}]", uv is { } v ? [v.X, v.Y] : null, Optional(camFromImg[i]));
				Vector3d? ray = CameraModels.CameraModelCamRayFromImg(id, parameters, pixel);
				Compare(failures, allExact, $"cam_ray_from_img[{i}]", ray is { } r ? [r.X, r.Y, r.Z] : null, Optional(camRayFromImg[i]));
			}
		}

		await Assert.That(failures).IsEmpty();
	}

	private static double[][] Rows(JsonElement array) =>
		array.EnumerateArray().Select(OracleFixture.Doubles).ToArray();

	// The script writes NaN and infinities as the strings double.Parse reads back.
	private static double[]? Optional(JsonElement element) =>
		element.ValueKind == JsonValueKind.Null
			? null
			: element.EnumerateArray()
				.Select(e => e.ValueKind == JsonValueKind.String ? double.Parse(e.GetString()!, CultureInfo.InvariantCulture) : e.GetDouble())
				.ToArray();

	private static void Compare(List<string> failures, bool exact, string what, double[]? actual, double[]? expected)
	{
		if (actual is null || expected is null)
		{
			if (actual is not null || expected is not null)
			{
				failures.Add($"{what}: expected {(expected is null ? "failure" : "a value")}, got {(actual is null ? "failure" : "a value")}");
			}

			return;
		}

		for (int k = 0; k < expected.Length; k++)
		{
			bool bothNaN = double.IsNaN(actual[k]) && double.IsNaN(expected[k]);
			if (bothNaN || BitConverter.DoubleToInt64Bits(actual[k]) == BitConverter.DoubleToInt64Bits(expected[k]))
			{
				continue;
			}

			double scaled = Math.Abs(actual[k] - expected[k]) / Math.Max(1.0, Math.Abs(expected[k]));
			if (exact || !(scaled <= ContractionTolerance))
			{
				failures.Add(string.Create(
					CultureInfo.InvariantCulture,
					$"{what}[{k}]: expected {expected[k]:R}, got {actual[k]:R} (scaled difference {scaled:E2}{(exact ? ", must be exact" : "")})"));
			}
		}
	}
}
