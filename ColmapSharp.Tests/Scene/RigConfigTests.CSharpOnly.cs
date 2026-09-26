// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// RigConfigTests (C#-only half): cases rig_test.cc does not have, pinning behavior of
// ColmapSharp/Scene/RigConfig.cs and RigConfig.Apply.cs that COLMAP gets from its helpers:
// StringStartsWith (an empty prefix matches nothing), std::map<std::string> (frame names in
// UTF-8 byte order), and the boost::property_tree / libc++ stream translators (which texts
// read as a bool or a double). The expected translator results were taken from a libc++
// harness (Apple clang, istringstream with ptree's extract logic).

using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public partial class RigConfigTests
{
	// A database with one camera and one image per name, image ids in the order given.
	private static InMemoryDatabase DatabaseWithImages(params string[] names)
	{
		var database = new InMemoryDatabase();
		uint cameraId = database.WriteCamera(Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1.0, 10, 10));
		foreach (string name in names)
		{
			var image = new Image { Name = name };
			image.SetCameraId(cameraId);
			database.WriteImage(image);
		}

		return database;
	}

	private static List<RigConfig> SingleCameraConfig(string imagePrefix)
	{
		var config = new RigConfig();
		config.Cameras.Add(new RigConfig.RigCamera { ImagePrefix = imagePrefix, RefSensor = true });
		return [config];
	}

	[Test]
	public async Task CSharpOnly_ApplyRigConfig_EmptyImagePrefixMatchesNothing()
	{
		// COLMAP's StringStartsWith is false for an empty prefix, so the camera finds no image.
		using InMemoryDatabase database = DatabaseWithImages("a", "b");
		await Assert.That(() => RigConfig.ApplyRigConfig(SingleCameraConfig(""), database))
			.Throws<ArgumentException>().WithMessageContaining("At least one image must exist for each camera in the rig");
	}

	[Test]
	public async Task CSharpOnly_ApplyRigConfig_FrameNamesInUtf8ByteOrder()
	{
		// UTF-8 puts U+FF21 (EF BC A1) before U+1F600 (F0 9F 98 80); UTF-16 ordinal order puts
		// the emoji's high surrogate (D83D) first. std::map<std::string> uses the byte order,
		// so the full-width name's frame is written first.
		using InMemoryDatabase database = DatabaseWithImages("cam_\U0001F600", "cam_Ａ");
		RigConfig.ApplyRigConfig(SingleCameraConfig("cam_"), database);
		List<Frame> frames = database.ReadAllFrames();
		await Assert.That(frames.Count).IsEqualTo(2);
		await Assert.That(frames[0].ImageIds().Select(dataId => dataId.Id)).IsEquivalentTo(new ulong[] { 2 }, CollectionOrdering.Matching);
		await Assert.That(frames[1].ImageIds().Select(dataId => dataId.Id)).IsEquivalentTo(new ulong[] { 1 }, CollectionOrdering.Matching);
	}

	// A two-camera rig whose second camera has the given raw JSON "ref_sensor" value.
	private static string ConfigWithSecondRefSensor(string refSensorJson) => $$"""
		[{"cameras": [
			{"image_prefix": "a/"},
			{"image_prefix": "b/", "ref_sensor": {{refSensorJson}}}
		]}]
		""";

	[Test]
	[Arguments("\"01\"")]
	[Arguments("\"+1\"")]
	[Arguments("\" true \"")]
	[Arguments("\"2true\"")]
	[Arguments("1")]
	[Arguments("true")]
	public async Task CSharpOnly_ReadRigConfig_RefSensorReadsTrueLikePtree(string refSensorJson)
	{
		List<RigConfig> configs = RigConfig.ReadRigConfigFromJson(ConfigWithSecondRefSensor(refSensorJson));
		await Assert.That(configs[0].Cameras[1].RefSensor).IsTrue();
	}

	[Test]
	[Arguments("\"00\"")]
	[Arguments("\"-0\"")]
	[Arguments("false")]
	[Arguments("0")]
	public async Task CSharpOnly_ReadRigConfig_RefSensorReadsFalseLikePtree(string refSensorJson)
	{
		// A false ref_sensor leaves the rig without a reference sensor.
		await Assert.That(() => RigConfig.ReadRigConfigFromJson(ConfigWithSecondRefSensor(refSensorJson)))
			.Throws<ArgumentException>().WithMessageContaining("Rig must have one reference sensor");
	}

	[Test]
	[Arguments("\"2\"")]
	[Arguments("\"-1\"")]
	[Arguments("\"TRUE\"")]
	[Arguments("\"tru\"")]
	[Arguments("\"1x\"")]
	[Arguments("\"0true\"")]
	[Arguments("\"\"")]
	[Arguments("\"99999999999999999999\"")]
	[Arguments("[]")]
	public async Task CSharpOnly_ReadRigConfig_RefSensorRejectsLikePtree(string refSensorJson)
	{
		await Assert.That(() => RigConfig.ReadRigConfigFromJson(ConfigWithSecondRefSensor(refSensorJson)))
			.Throws<ArgumentException>().WithMessageContaining("conversion of data to type \"bool\" failed");
	}

	private static string ConfigWithParam(string paramJson) => $$"""
		[{"cameras": [
			{"image_prefix": "a/", "ref_sensor": true, "camera_model_name": "SIMPLE_PINHOLE", "camera_params": [{{paramJson}}]}
		]}]
		""";

	[Test]
	[Arguments("\"1.5\"", 1.5)]
	[Arguments("\" 2 \"", 2.0)]
	[Arguments("\"+.5\"", 0.5)]
	[Arguments("\"5.\"", 5.0)]
	[Arguments("\"1E5\"", 1e5)]
	[Arguments("\"0x1p3\"", 8.0)]
	[Arguments("\"0x10\"", 16.0)]
	[Arguments("\"0x.8\"", 0.5)]
	[Arguments("\"0e-400\"", 0.0)]
	[Arguments("\"2.2250738585072014e-308\"", 2.2250738585072014e-308)]
	[Arguments("1.7976931348623158e308", 1.7976931348623157e308)]
	[Arguments("-0.25", -0.25)]
	[Arguments("\"-0x10\"", -16.0)]
	[Arguments("\"0x1.00000000000018p0\"", 1.0000000000000004)]
	[Arguments("\"0x1.ffffffffffffffp-1023\"", 2.2250738585072014e-308)]
	[Arguments("\"0x1p-1030\"", 8.6916947597937554e-311)]
	[Arguments("\"0x1p-1074\"", 4.9406564584124654e-324)]
	public async Task CSharpOnly_ReadRigConfig_ParamReadsLikeIostream(string paramJson, double expected)
	{
		List<RigConfig> configs = RigConfig.ReadRigConfigFromJson(ConfigWithParam(paramJson));
		await Assert.That(configs[0].Cameras[0].Camera!.Params).IsEquivalentTo(new[] { expected }, CollectionOrdering.Matching);
	}

	[Test]
	[Arguments("\"inf\"")]
	[Arguments("\"-inf\"")]
	[Arguments("\"nan\"")]
	[Arguments("\"NaN\"")]
	[Arguments("\"Infinity\"")]
	[Arguments("\"∞\"")]
	[Arguments("1e400")]
	[Arguments("-1e400")]
	[Arguments("1.7976931348623159e308")]
	[Arguments("1e-310")]
	[Arguments("1e-400")]
	[Arguments("2.2250738585072011e-308")]
	[Arguments("\"1,5\"")]
	[Arguments("\"1d\"")]
	[Arguments("\"1e5.\"")]
	[Arguments("\"1.2.3\"")]
	[Arguments("\"1e\"")]
	[Arguments("\".\"")]
	[Arguments("\"\"")]
	[Arguments("\"0x\"")]
	[Arguments("\"0xg\"")]
	[Arguments("\"0x1p\"")]
	[Arguments("\"0x1p1024\"")]
	[Arguments("\"0x1.fffffffffffff8p1023\"")]
	[Arguments("\"0x1.8p-1074\"")]
	[Arguments("\"0x1p-1080\"")]
	public async Task CSharpOnly_ReadRigConfig_ParamRejectsLikeIostream(string paramJson)
	{
		await Assert.That(() => RigConfig.ReadRigConfigFromJson(ConfigWithParam(paramJson)))
			.Throws<ArgumentException>().WithMessageContaining("conversion of data to type \"double\" failed");
	}
}
