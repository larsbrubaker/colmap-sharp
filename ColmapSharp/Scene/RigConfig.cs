// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RigConfig: port of colmap/scene/rig.h and the ReadRigConfig half of rig.cc. A rig config
// is a JSON file that groups images into multi-camera rigs by image name prefix; the
// ApplyRigConfig half (RigConfig.Apply.cs) writes the resulting rigs and frames into a
// Database (InMemoryDatabase.cs) and optionally a Reconstruction. The rig itself is
// Sensor/Rig.cs. Tests: ColmapSharp.Tests/Scene/RigConfigTests.cs (rig_test.cc 1:1).
//
// COLMAP parses the file with boost::property_tree, which keeps every value as text and
// converts on access. This port reads the file with System.Text.Json's JsonDocument (no
// reflection serialization, so it stays trim- and AOT-clean) and mirrors the property tree:
// - a node's children are an array's elements or an object's values, and a scalar has none;
// - a lookup by key finds the first property with that key (ptree get_child), not the last
//   (JsonElement.TryGetProperty);
// - a scalar reads as its text (a number's raw text, "true"/"false", a string's value), a
//   container as "" (ptree's empty data), and numbers and booleans parse from that text as
//   the ptree stream translator does (RigConfig.Translators.cs), so "1.5" and 1.5 are both a
//   double.
// Strict JSON (no trailing commas, no comments) matches boost's parser, which rejects both.
// The rotation and translation arrays must have exactly 4 and 3 entries (divergence 80).

using System.Text.Json;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>
/// Port of colmap::RigConfig: one rig of a rig configuration. For each rig, the
/// configuration specifies a list of cameras with exactly one camera specified as the
/// reference sensor as well as all cameras with a specified image prefix. All images with
/// the given name prefix will be associated with the camera. In addition, each camera may
/// specify an optional known pose in the rig - except for the reference camera whose pose
/// is defined as identity. The rotation is expected in the order [w, x, y, z]. Furthermore,
/// each camera may specify a custom camera model and parameters.
/// </summary>
/// <example>
/// Example for eth3d/delivery_area:
/// <code>
/// [
///   {
///     "cameras": [
///       { "image_prefix": "images_rig_cam4_undistorted/", "ref_sensor": true },
///       { "image_prefix": "images_rig_cam5_undistorted/" },
///       { "image_prefix": "images_rig_cam6_undistorted/" },
///       { "image_prefix": "images_rig_cam7_undistorted/" }
///     ]
///   }
/// ]
/// </code>
/// Example for GoPro cubemaps:
/// <code>
/// [
///   {
///     "cameras": [
///       { "image_prefix": "0/", "ref_sensor": true },
///       {
///         "image_prefix": "1/",
///         "cam_from_rig_rotation": [0.7071067811865475, 0.0, 0.7071067811865476, 0.0],
///         "cam_from_rig_translation": [0, 0, 0]
///       },
///       ...
///     ]
///   }
/// ]
/// </code>
/// </example>
public sealed partial class RigConfig
{
	/// <summary>Port of colmap::RigConfig::RigCamera: one camera of a rig configuration.</summary>
	public sealed class RigCamera
	{
		/// <summary>Whether this camera is the rig's reference sensor.</summary>
		public bool RefSensor { get; set; }

		/// <summary>All images whose name starts with this prefix belong to this camera.</summary>
		public string ImagePrefix { get; set; } = "";

		/// <summary>The known cam_from_rig pose, or null to derive it (or leave it unknown).</summary>
		public Rigid3d? CamFromRig { get; set; }

		/// <summary>
		/// Custom intrinsics (model, parameters, prior focal length flag) to apply to the
		/// camera, or null to keep the database's.
		/// </summary>
		public Camera? Camera { get; set; }
	}

	/// <summary>The cameras of the rig.</summary>
	public List<RigCamera> Cameras { get; } = [];

	/// <summary>
	/// Reads the rig configuration from a .json file (see the class summary for the format).
	/// Throws on invalid JSON, a camera without "image_prefix", or a rig without exactly one
	/// reference sensor. Port of colmap::ReadRigConfig.
	/// </summary>
	public static List<RigConfig> ReadRigConfig(string path) => ReadRigConfigFromJson(File.ReadAllText(path));

	/// <summary>
	/// <see cref="ReadRigConfig"/> on JSON text rather than a file, for hosts without a file
	/// system (browser-wasm).
	/// </summary>
	public static List<RigConfig> ReadRigConfigFromJson(string json)
	{
		using JsonDocument document = JsonDocument.Parse(json);

		var configs = new List<RigConfig>();
		foreach (JsonElement rigNode in Children(document.RootElement))
		{
			var config = new RigConfig();
			configs.Add(config);
			bool hasRefSensor = false;
			JsonElement camerasNode = FindChild(rigNode, "cameras")
				?? throw new ArgumentException("No such node (cameras)");
			foreach (JsonElement camera in Children(camerasNode))
			{
				var configCamera = new RigCamera();
				config.Cameras.Add(configCamera);

				configCamera.ImagePrefix = ScalarText(FindChild(camera, "image_prefix")
					?? throw new ArgumentException("No such node (image_prefix)"));

				JsonElement? camFromRigRotationNode = FindChild(camera, "cam_from_rig_rotation");
				JsonElement? camFromRigTranslationNode = FindChild(camera, "cam_from_rig_translation");
				if (camFromRigRotationNode is not null && camFromRigTranslationNode is not null)
				{
					double[] camFromRigWxyz = ReadDoubles(camFromRigRotationNode.Value, 4, "cam_from_rig_rotation");
					double[] translation = ReadDoubles(camFromRigTranslationNode.Value, 3, "cam_from_rig_translation");
					configCamera.CamFromRig = new Rigid3d(
						new Quaterniond(camFromRigWxyz[0], camFromRigWxyz[1], camFromRigWxyz[2], camFromRigWxyz[3]),
						new Vector3d(translation[0], translation[1], translation[2]));
				}

				JsonElement? refSensorNode = FindChild(camera, "ref_sensor");
				if (refSensorNode is not null && ParseBool(ScalarText(refSensorNode.Value)))
				{
					Check.That(camFromRigRotationNode is null && camFromRigTranslationNode is null,
						"Reference sensor must not have cam_from_rig");
					Check.That(!hasRefSensor, "Rig must only have one reference sensor");
					configCamera.RefSensor = true;
					hasRefSensor = true;
				}

				JsonElement? cameraModelNameNode = FindChild(camera, "camera_model_name");
				JsonElement? cameraParamsNode = FindChild(camera, "camera_params");
				if (cameraModelNameNode is not null && cameraParamsNode is not null)
				{
					var paramsList = new List<double>();
					foreach (JsonElement node in Children(cameraParamsNode.Value))
					{
						paramsList.Add(ParseDouble(ScalarText(node)));
					}

					configCamera.Camera = new Camera
					{
						ModelId = CameraModels.CameraModelNameToId(ScalarText(cameraModelNameNode.Value)),
						HasPriorFocalLength = true,
						Params = [.. paramsList],
					};
				}
			}

			Check.That(hasRefSensor, "Rig must have one reference sensor");
		}

		return configs;
	}

	// The children of a property tree node: an array's elements or an object's values.
	private static IEnumerable<JsonElement> Children(JsonElement node)
	{
		switch (node.ValueKind)
		{
			case JsonValueKind.Array:
				foreach (JsonElement element in node.EnumerateArray())
				{
					yield return element;
				}

				break;
			case JsonValueKind.Object:
				foreach (JsonProperty property in node.EnumerateObject())
				{
					yield return property.Value;
				}

				break;
		}
	}

	// ptree get_child_optional: the first child with the key (only objects have keys).
	private static JsonElement? FindChild(JsonElement node, string key)
	{
		if (node.ValueKind != JsonValueKind.Object)
		{
			return null;
		}

		foreach (JsonProperty property in node.EnumerateObject())
		{
			if (property.NameEquals(key))
			{
				return property.Value;
			}
		}

		return null;
	}

	// The text a property tree stores for the node: containers store "".
	private static string ScalarText(JsonElement node) => node.ValueKind switch
	{
		JsonValueKind.String => node.GetString()!,
		JsonValueKind.Number => node.GetRawText(),
		JsonValueKind.True => "true",
		JsonValueKind.False => "false",
		JsonValueKind.Null => "null",
		_ => "",
	};

	private static double[] ReadDoubles(JsonElement node, int count, string key)
	{
		var values = new List<double>();
		foreach (JsonElement element in Children(node))
		{
			values.Add(ParseDouble(ScalarText(element)));
		}

		// COLMAP writes the entries into a fixed-size Eigen vector without a size check.
		Check.That(values.Count == count, $"{key} must have {count} entries");
		return [.. values];
	}
}
