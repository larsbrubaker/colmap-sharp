// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ImageReader: port of colmap/controllers/image_reader.h and .cc - iterates over the images
// to import, skips the ones whose features are already in the Database (Scene/Database.cs),
// assigns each image a camera (from EXIF focal length, manual parameters, or an existing
// camera, honoring single_camera / single_camera_per_folder / single_camera_per_image) and a
// rig, and extracts the GPS and gravity pose prior. FeatureExtraction.cs drives it.
// Tests: ColmapSharp.Tests/Controllers/ImageReaderTests.cs (image_reader_test.cc 1:1).
//
// Tier A (exact): pure bookkeeping.
//
// Translation notes:
// - The file system is replaced by IImageSource (ImageSource.cs): image_path, mask_path and
//   camera_mask_path become Images, Masks and CameraMask, and decoding is the host's job
//   (divergence 82). Bitmap::Read's final grey/RGB conversion is kept.
// - Next's six out-pointers become one ImageReaderData, freshly made on each call, so no
//   state from a previous image can leak into the next through a reused Image object.
// - Names sort as COLMAP sorts them. Listed names (empty image_names) are std::sort-ed as
//   std::filesystem::paths, which compare element by element, so "set/0.jpg" comes before
//   "set-2/0.jpg"; explicit image_names are std::strings, compared as whole byte strings.
//   Both compare UTF-8 bytes (UTF-16 ordinal order differs for supplementary-plane
//   characters), as SequentialPairGenerator does.
// - existing_camera_id is an int in C++ cast to camera_t; here it is the camera_t (uint).

using System.Text;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using static ColmapSharp.Util.Types;

namespace ColmapSharp.Controllers;

/// <summary>Port of colmap::ImageReaderOptions.</summary>
public sealed class ImageReaderOptions
{
	/// <summary>The images to read (COLMAP's image_path folder).</summary>
	public IImageSource Images { get; set; } = new InMemoryImageSource();

	/// <summary>
	/// Optional image masks (COLMAP's mask_path). For an image "abc/012.jpg" the mask is
	/// "abc/012.jpg.png", or else "abc/012.png". No features will be extracted in regions
	/// where the mask image is black (pixel intensity value 0 in grayscale).
	/// </summary>
	public IImageSource? Masks { get; set; }

	/// <summary>
	/// Optional mask for all images (COLMAP's camera_mask_path, decoded by the host). No
	/// features will be extracted in regions where the mask is black.
	/// </summary>
	public Bitmap? CameraMask { get; set; }

	/// <summary>
	/// Optional list of images to read, as names in <see cref="Images"/>. Empty means every
	/// image the source lists.
	/// </summary>
	public List<string> ImageNames { get; set; } = [];

	/// <summary>Name of the camera model.</summary>
	public string CameraModel { get; set; } = "SIMPLE_RADIAL";

	/// <summary>
	/// Manual specification of camera parameters. If empty, camera parameters will be
	/// extracted from EXIF, i.e. principal point and focal length.
	/// </summary>
	public string CameraParams { get; set; } = "";

	/// <summary>Whether to use the same camera for all images.</summary>
	public bool SingleCamera { get; set; }

	/// <summary>Whether to use the same camera for all images in the same sub-folder.</summary>
	public bool SingleCameraPerFolder { get; set; }

	/// <summary>Whether to use a different camera for each image.</summary>
	public bool SingleCameraPerImage { get; set; }

	/// <summary>
	/// Whether to explicitly use an existing camera for all images. Note that in this case
	/// the specified camera model and parameters are ignored.
	/// </summary>
	public uint ExistingCameraId { get; set; } = InvalidCameraId;

	/// <summary>
	/// If camera parameters are not specified manually and the image does not have focal
	/// length EXIF information, the focal length is set to the value
	/// `DefaultFocalLengthFactor * max(width, height)`.
	/// </summary>
	public double DefaultFocalLengthFactor { get; set; } = 1.2;

	/// <summary>Whether to read images as grayscale or RGB.</summary>
	public bool AsRgb { get; set; }

	/// <summary>A copy (the C++ value copy); the sources and camera mask are shared.</summary>
	public ImageReaderOptions Clone()
	{
		var copy = (ImageReaderOptions)MemberwiseClone();
		copy.ImageNames = [.. ImageNames];
		return copy;
	}

	/// <summary>Port of ImageReaderOptions::Check (CHECK_OPTION: false on a violation).</summary>
	public bool Check()
	{
		if (!(DefaultFocalLengthFactor > 0.0))
		{
			Log.Error("Option check failed: default_focal_length_factor > 0.0");
			return false;
		}

		if (!CameraModels.ExistsCameraModelWithName(CameraModel))
		{
			Log.Error("Option check failed: ExistsCameraModelWithName(camera_model)");
			return false;
		}

		CameraModelId modelId = CameraModels.CameraModelNameToId(CameraModel);
		if (CameraParams.Length != 0 &&
			!CameraModels.CameraModelVerifyParams(modelId, [.. Misc.CsvToDoubleVector(CameraParams)]))
		{
			Log.Error("Option check failed: CameraModelVerifyParams(model_id, CSVToVector<double>(camera_params))");
			return false;
		}

		return true;
	}
}

/// <summary>What <see cref="ImageReader.Next"/> read for one image.</summary>
public sealed class ImageReaderData
{
	/// <summary>The image's rig (valid on SUCCESS).</summary>
	public Rig Rig { get; set; } = new();

	/// <summary>The image's camera (valid on SUCCESS).</summary>
	public Camera Camera { get; set; } = new();

	/// <summary>The image; its ImageId is set only if it already existed in the database.</summary>
	public Image Image { get; set; } = new();

	/// <summary>The GPS / gravity prior from EXIF (unset fields are NaN).</summary>
	public PosePrior PosePrior { get; set; } = new();

	/// <summary>The decoded image; empty unless it was read.</summary>
	public Bitmap Bitmap { get; set; } = new();

	/// <summary>The decoded mask, or null when no mask was read.</summary>
	public Bitmap? Mask { get; set; }
}

/// <summary>
/// Port of colmap::ImageReader: iterates over the images. Skips an image if it already
/// exists in the database. Extracts the camera intrinsics from EXIF and writes the camera
/// information to the database.
/// </summary>
public sealed class ImageReader
{
	/// <summary>Port of ImageReader::Status.</summary>
	public enum Status
	{
		/// <summary>FAILURE.</summary>
		Failure,
		/// <summary>SUCCESS.</summary>
		Success,
		/// <summary>IMAGE_EXISTS.</summary>
		ImageExists,
		/// <summary>BITMAP_ERROR.</summary>
		BitmapError,
		/// <summary>MASK_ERROR.</summary>
		MaskError,
		/// <summary>CAMERA_SINGLE_DIM_ERROR.</summary>
		CameraSingleDimError,
		/// <summary>CAMERA_EXIST_DIM_ERROR.</summary>
		CameraExistDimError,
		/// <summary>CAMERA_PARAM_ERROR.</summary>
		CameraParamError,
	}

	private readonly ImageReaderOptions options;
	private readonly Database database;
	// Index of previously processed image.
	private int imageIndex;
	// Previously processed rig/camera.
	private Rig prevRig = new();
	private Camera prevCamera = new();
	private readonly Dictionary<string, uint> cameraModelToId = new(StringComparer.Ordinal);
	// Names of image sub-folders.
	private string prevImageFolder = "";
	private readonly HashSet<string> imageFolders = new(StringComparer.Ordinal);

	/// <summary>Port of the ImageReader constructor.</summary>
	public ImageReader(ImageReaderOptions options, Database database)
	{
		this.options = options.Clone();
		this.database = database;
		Check.That(this.options.Check());

		// Get a list of all images, sorted by image name.
		if (this.options.ImageNames.Count == 0)
		{
			// COLMAP sorts the listed std::filesystem::paths (all below the same image_path).
			this.options.ImageNames = [.. this.options.Images.ListNames()];
			this.options.ImageNames.Sort(ComparePathElements);
		}
		else
		{
			this.options.ImageNames.Sort(CompareUtf8);
		}

		if (this.options.ExistingCameraId != InvalidCameraId)
		{
			Check.That(database.ExistsCamera(this.options.ExistingCameraId));
			prevCamera = database.ReadCamera(this.options.ExistingCameraId);
			Rig? rig = database.ReadRigWithSensor(prevCamera.SensorId);
			if (rig is not null)
			{
				prevRig = rig;
			}
			else
			{
				// For backwards compatibility with old databases without rigs.
				prevRig.AddRefSensor(prevCamera.SensorId);
				prevRig.RigId = database.WriteRig(prevRig);
			}
		}
		else
		{
			// Set the manually specified camera parameters.
			prevCamera.CameraId = InvalidCameraId;
			Check.That(CameraModels.ExistsCameraModelWithName(this.options.CameraModel));
			prevCamera.ModelId = CameraModels.CameraModelNameToId(this.options.CameraModel);
			prevCamera.Params = new double[CameraModels.CameraModelNumParams(prevCamera.ModelId)];
			if (this.options.CameraParams.Length != 0)
			{
				Check.That(prevCamera.SetParamsFromString(this.options.CameraParams));
				prevCamera.HasPriorFocalLength = true;
			}
		}
	}

	/// <summary>Port of ImageReader::NextIndex: the number of images already processed.</summary>
	public int NextIndex => imageIndex;

	/// <summary>Port of ImageReader::NumImages.</summary>
	public int NumImages => options.ImageNames.Count;

	/// <summary>
	/// Port of ImageReader::Next: reads the next image into a fresh <paramref name="data"/>.
	/// <paramref name="readMask"/> false is COLMAP's null mask pointer. Throws
	/// ArgumentException past the last image.
	/// </summary>
	public Status Next(out ImageReaderData data, bool readMask = true)
	{
		data = new ImageReaderData();
		imageIndex += 1;
		Check.Le(imageIndex, options.ImageNames.Count);

		string imageName = options.ImageNames[imageIndex - 1];

		using var databaseTransaction = new DatabaseTransaction(database);

		// Set the image name.
		data.Image.Name = imageName;
		string imageFolder = GetParentDir(imageName);

		// Check if image already read.
		bool existsImage = database.ExistsImageWithName(imageName);
		if (existsImage)
		{
			data.Image = database.ReadImageWithName(imageName)!;
			bool existsKeypoints = database.ExistsKeypoints(data.Image.ImageId);
			bool existsDescriptors = database.ExistsDescriptors(data.Image.ImageId);
			if (existsKeypoints && existsDescriptors)
			{
				return Status.ImageExists;
			}
		}

		// Read image.
		Bitmap? bitmap = ReadBitmap(options.Images, imageName, options.AsRgb);
		if (bitmap is null)
		{
			return Status.BitmapError;
		}

		data.Bitmap = bitmap;

		// Read mask.
		if (readMask && options.Masks is not null)
		{
			string maskName = imageName + ".png";
			if (!options.Masks.Exists(maskName))
			{
				bool existsMask = false;
				// Try replacing extension with .png
				int lastDot = imageName.LastIndexOf('.');
				if (lastDot >= 0)
				{
					string altMaskName = imageName[..lastDot] + ".png";
					if (options.Masks.Exists(altMaskName))
					{
						maskName = altMaskName;
						existsMask = true;
					}
				}

				if (!existsMask)
				{
					Log.Error($"Mask at {maskName} does not exist.");
					return Status.MaskError;
				}
			}

			data.Mask = ReadBitmap(options.Masks, maskName, asRgb: false);
			if (data.Mask is null)
			{
				Log.Error($"Failed to read invalid mask file at: {maskName}");
				return Status.MaskError;
			}
		}

		// Check for well-formed data.
		if (existsImage)
		{
			Camera currentCamera = database.ReadCamera(data.Image.CameraId);

			if (options.SingleCamera && prevCamera.CameraId != InvalidCameraId &&
				(currentCamera.Width != prevCamera.Width || currentCamera.Height != prevCamera.Height))
			{
				return Status.CameraSingleDimError;
			}

			if (bitmap.Width != currentCamera.Width || bitmap.Height != currentCamera.Height)
			{
				return Status.CameraExistDimError;
			}

			prevCamera = currentCamera;
			prevRig = ReadOrCreateRig(prevCamera);
		}
		else
		{
			// Check image dimensions.
			if (prevCamera.CameraId != InvalidCameraId &&
				((options.SingleCamera && !options.SingleCameraPerFolder) ||
				 (options.SingleCameraPerFolder && imageFolder == prevImageFolder)) &&
				(prevCamera.Width != bitmap.Width || prevCamera.Height != bitmap.Height))
			{
				return Status.CameraSingleDimError;
			}

			// Read camera model and check for consistency if it exists.
			string? cameraModel = bitmap.ExifCameraModel();
			if (cameraModel is not null && cameraModelToId.TryGetValue(cameraModel, out uint modelCameraId))
			{
				Camera camera = database.ReadCamera(modelCameraId);
				if (camera.Width != bitmap.Width || camera.Height != bitmap.Height)
				{
					return Status.CameraExistDimError;
				}

				prevCamera = camera;
				prevRig = ReadOrCreateRig(prevCamera);
			}

			// Extract camera model and focal length.
			if (prevCamera.CameraId == InvalidCameraId ||
				options.SingleCameraPerImage ||
				(!options.SingleCamera && !options.SingleCameraPerFolder &&
				 options.ExistingCameraId == InvalidCameraId &&
				 (cameraModel is null || !cameraModelToId.ContainsKey(cameraModel))) ||
				(options.SingleCameraPerFolder && !imageFolders.Contains(imageFolder)))
			{
				if (options.CameraParams.Length == 0)
				{
					// Extract focal length.
					double? maybeFocalLength = bitmap.ExifFocalLength();
					double focalLength = maybeFocalLength ??
						options.DefaultFocalLengthFactor * Math.Max(bitmap.Width, bitmap.Height);

					prevCamera = Camera.CreateFromModelId(
						prevCamera.CameraId, prevCamera.ModelId, focalLength, bitmap.Width, bitmap.Height);
					prevCamera.HasPriorFocalLength = maybeFocalLength.HasValue;
				}
				else
				{
					// C++ assigns into the same prev_camera_ object; here the camera handed out
					// for the previous image must not change, so continue on a copy.
					prevCamera = prevCamera.Clone();
				}

				prevCamera.Width = bitmap.Width;
				prevCamera.Height = bitmap.Height;

				if (!prevCamera.VerifyParams())
				{
					return Status.CameraParamError;
				}

				prevCamera.CameraId = database.WriteCamera(prevCamera);

				// By default we create a separate rig per camera. Grouping of different
				// cameras into the same rig is expected to be done with the
				// "rig_configurator" after feature extraction.
				if (database.ReadRigWithSensor(prevCamera.SensorId) is null)
				{
					prevRig = new Rig();
					prevRig.AddRefSensor(prevCamera.SensorId);
					prevRig.RigId = database.WriteRig(prevRig);
				}

				if (cameraModel is not null)
				{
					cameraModelToId[cameraModel] = prevCamera.CameraId;
				}
			}

			data.Image.SetCameraId(prevCamera.CameraId);

			// Extract GPS data.
			PosePrior posePrior = data.PosePrior;
			double? latitude = bitmap.ExifLatitude();
			double? longitude = bitmap.ExifLongitude();
			double? altitude = bitmap.ExifAltitude();
			if (latitude.HasValue && longitude.HasValue && altitude.HasValue)
			{
				posePrior.Position = new Vector3d(latitude.Value, longitude.Value, altitude.Value);
				posePrior.CoordinateSystem = PosePriorCoordinateSystem.Wgs84;
			}

			// Extract Gravity from Orientation.
			int? orientation = bitmap.ExifOrientation();
			if (orientation.HasValue)
			{
				Vector3d? gravity = PosePrior.GravityFromExifOrientation(orientation.Value);
				if (gravity.HasValue)
				{
					posePrior.Gravity = gravity.Value;
				}
			}

			data.PosePrior = posePrior;
		}

		data.Camera = prevCamera.Clone();
		data.Rig = prevRig.Clone();

		imageFolders.Add(imageFolder);
		prevImageFolder = imageFolder;

		return Status.Success;
	}

	/// <summary>Port of ImageReader::StatusToString.</summary>
	public static string StatusToString(Status status) => status switch
	{
		Status.Success => "SUCCESS",
		Status.Failure => "FAILURE: Failed to process the image.",
		Status.ImageExists => "IMAGE_EXISTS: Features for image were already extracted.",
		Status.BitmapError => "BITMAP_ERROR: Failed to read the image file format.",
		Status.MaskError => "MASK_ERROR: Failed to read the mask file.",
		Status.CameraSingleDimError =>
			"CAMERA_SINGLE_DIM_ERROR: Single camera specified, but images have different dimensions.",
		Status.CameraExistDimError =>
			"CAMERA_EXIST_DIM_ERROR: Image previously processed, but current image has different dimensions.",
		Status.CameraParamError => "CAMERA_PARAM_ERROR: Camera has invalid parameters.",
		_ => "Unknown",
	};

	/// <summary>
	/// The host-decoded bitmap converted like the tail of Bitmap::Read: RGB when
	/// <paramref name="asRgb"/>, grey otherwise. Null when the source cannot decode it.
	/// </summary>
	internal static Bitmap? ReadBitmap(IImageSource source, string name, bool asRgb)
	{
		Bitmap? bitmap = source.Read(name);
		if (bitmap is null)
		{
			return null;
		}

		if (asRgb && !bitmap.IsRGB)
		{
			return bitmap.CloneAsRGB();
		}

		if (!asRgb && !bitmap.IsGrey)
		{
			return bitmap.CloneAsGrey();
		}

		return bitmap;
	}

	// std::string's operator<: a byte-wise compare of the UTF-8 encodings.
	private static int CompareUtf8(string a, string b) =>
		Encoding.UTF8.GetBytes(a).AsSpan().SequenceCompareTo(Encoding.UTF8.GetBytes(b));

	// std::filesystem::path::compare of two relative '/'-separated names: element by element,
	// each element compared by its UTF-8 bytes; a path that runs out of elements first is less.
	private static int ComparePathElements(string a, string b)
	{
		string[] elementsA = a.Split('/');
		string[] elementsB = b.Split('/');
		int count = Math.Min(elementsA.Length, elementsB.Length);
		for (int i = 0; i < count; ++i)
		{
			int cmp = CompareUtf8(elementsA[i], elementsB[i]);
			if (cmp != 0)
			{
				return cmp;
			}
		}

		return elementsA.Length.CompareTo(elementsB.Length);
	}

	// std::filesystem::path(name).parent_path() for a '/'-separated relative name.
	private static string GetParentDir(string name)
	{
		int slash = name.LastIndexOf('/');
		return slash < 0 ? "" : name[..slash];
	}

	// The rig holding the camera, or (for backwards compatibility with old databases without
	// rigs) a new single-camera rig written to the database.
	private Rig ReadOrCreateRig(Camera camera)
	{
		Rig? rig = database.ReadRigWithSensor(camera.SensorId);
		if (rig is not null)
		{
			return rig;
		}

		var created = new Rig();
		created.AddRefSensor(camera.SensorId);
		created.RigId = database.WriteRig(created);
		return created;
	}
}
