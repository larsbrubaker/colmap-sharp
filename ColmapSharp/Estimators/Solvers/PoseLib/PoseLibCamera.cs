// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
// THIRD_PARTY_NOTICES.md), commit fa7280fee27f97aff31ae7f98bab7f583fac7d08.
//
// PoseLibCamera: the camera description of PoseLib/misc/camera_models.h and .cc (struct
// Camera: model id, size, parameters; id_from_string, name_from_id, init_params, focal,
// set_focal, set_principal_point, and each model's name, id, num_params, focal_idx and
// principal_point_idx). Estimators/Solvers/PoseLibUtils.cs converts COLMAP's Camera to and
// from it.
//
// PoseLib's projection and distortion functions (project/unproject and their Jacobians) are
// not ported: COLMAP only ever converts cameras to PoseLib and back (poselib_utils.cc), and
// every solver COLMAP calls works on bearings or normalized points. Port them with the
// PoseLib code that first needs them.

namespace ColmapSharp.Estimators.Solvers.PoseLib;

/// <summary>
/// A PoseLib camera: model id, image size and model parameters. Port of the data members,
/// constructors and name lookups of poselib::Camera.
/// </summary>
public sealed class PoseLibCamera
{
	private sealed record ModelInfo(string Name, int Id, int NumParams, int[] FocalIdx, int[] PrincipalPointIdx);

	// SWITCH_CAMERA_MODELS, in PoseLib's order (the name lookup returns the first match).
	private static readonly ModelInfo[] Models =
	[
		new("NULL", -1, 0, [], []),
		new("SIMPLE_PINHOLE", 0, 3, [0], [1, 2]),
		new("PINHOLE", 1, 4, [0, 1], [2, 3]),
		new("SIMPLE_RADIAL", 2, 4, [0], [1, 2]),
		new("RADIAL", 3, 5, [0], [1, 2]),
		new("OPENCV", 4, 8, [0, 1], [2, 3]),
		new("FULL_OPENCV", 6, 12, [0, 1], [2, 3]),
		new("OPENCV_FISHEYE", 5, 8, [0, 1], [2, 3]),
		new("FOV", 7, 5, [0, 1], [2, 3]),
		new("SIMPLE_RADIAL_FISHEYE", 8, 4, [0], [1, 2]),
		new("RADIAL_FISHEYE", 9, 5, [0], [1, 2]),
		new("THIN_PRISM_FISHEYE", 10, 12, [0, 1], [2, 3]),
		new("RAD_TAN_THIN_PRISM_FISHEYE", 11, 16, [0, 1], [2, 3]),
		new("1D_RADIAL", 99, 2, [], [0, 1]),
		new("SPHERICAL", 100, 2, [], []),
		new("DIVISION", 101, 5, [0, 1], [2, 3]),
		new("SIMPLE_DIVISION", 102, 4, [0], [1, 2]),
	];

	/// <summary>
	/// Camera(model_name, params, width, height): looks the model up by name (INVALID = -1
	/// when unknown) and runs init_params.
	/// </summary>
	public PoseLibCamera(string modelName, IEnumerable<double> parameters, int width = 0, int height = 0)
		: this(IdFromString(modelName), parameters, width, height)
	{
	}

	/// <summary>Camera(model_id, params, width, height), followed by init_params.</summary>
	public PoseLibCamera(int modelId, IEnumerable<double> parameters, int width = 0, int height = 0)
	{
		ModelId = modelId;
		Params = [.. parameters];
		Width = width;
		Height = height;
		InitParams();
	}

	/// <summary>PoseLib's model id (poselib::CameraModelId; -1 is INVALID).</summary>
	public int ModelId { get; set; }

	/// <summary>Image width in pixels.</summary>
	public int Width { get; set; }

	/// <summary>Image height in pixels.</summary>
	public int Height { get; set; }

	/// <summary>The model parameters.</summary>
	public List<double> Params { get; set; }

	/// <summary>The model name, or "INVALID_MODEL" for an unknown id (model_name()).</summary>
	public string ModelName => NameFromId(ModelId);

	/// <summary>
	/// The mean of the model's focal parameters: 1 for an empty (identity) camera, and also 1
	/// for a model without focal parameters or an unknown id. Port of Camera::focal.
	/// </summary>
	public double Focal()
	{
		if (Params.Count == 0)
		{
			return 1.0; // empty camera assumed to be identity
		}

		double focal = 1.0;
		ModelInfo? model = Find(ModelId);
		if (model is not null && model.FocalIdx.Length > 0)
		{
			focal = 0.0;
			foreach (int idx in model.FocalIdx)
			{
				focal += Params[idx] / model.FocalIdx.Length;
			}
		}

		return focal;
	}

	/// <summary>Port of Camera::id_from_string.</summary>
	public static int IdFromString(string modelName)
	{
		foreach (ModelInfo model in Models)
		{
			if (model.Name == modelName)
			{
				return model.Id;
			}
		}

		return -1;
	}

	/// <summary>Port of Camera::name_from_id.</summary>
	public static string NameFromId(int modelId) => Find(modelId)?.Name ?? "INVALID_MODEL";

	private static ModelInfo? Find(int modelId)
	{
		foreach (ModelInfo model in Models)
		{
			if (model.Id == modelId)
			{
				return model;
			}
		}

		return null;
	}

	/// <summary>
	/// Port of Camera::init_params: resizes the parameters to the model's count (padding with
	/// zeros) and, only when none were given, sets the focal length to 1.2 * max(width,
	/// height) and the principal point to the image center (or 1 and 0 without a size).
	/// </summary>
	private void InitParams()
	{
		bool initParams = Params.Count == 0;
		ModelInfo? model = Find(ModelId);
		if (model is null)
		{
			return;
		}

		if (Params.Count > model.NumParams)
		{
			Params.RemoveRange(model.NumParams, Params.Count - model.NumParams);
		}

		while (Params.Count < model.NumParams)
		{
			Params.Add(0.0);
		}

		if (!initParams)
		{
			return;
		}

		double focal = 1.0, cx = 0.0, cy = 0.0;
		if (Width > 0 && Height > 0)
		{
			focal = Math.Max(Width, Height) * 1.2;
			cx = Width / 2.0;
			cy = Height / 2.0;
		}

		foreach (int k in model.FocalIdx)
		{
			Params[k] = focal;
		}

		if (model.PrincipalPointIdx.Length == 2)
		{
			Params[model.PrincipalPointIdx[0]] = cx;
			Params[model.PrincipalPointIdx[1]] = cy;
		}
	}
}
