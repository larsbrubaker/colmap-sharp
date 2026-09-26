// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionIO (continued): ExportBundler, ExportPLY and ExportVRML from
// colmap/scene/reconstruction_io.cc. The shared helpers and the formatting rules (file
// precision 17, track lines and VRML at the default 6, ascending-id walks) are in
// ReconstructionIO.Export.cs.

using System.Text;

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Scene;

public static partial class ReconstructionIO
{
	/// <summary>
	/// Port of colmap::ExportBundler: Bundler v0.3 (https://www.cs.cornell.edu/~snavely/bundler/)
	/// to <paramref name="path"/> and the image list to <paramref name="listPath"/>.
	/// SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL and RADIAL only, unless
	/// <paramref name="skipDistortion"/>. Returns false for an unsupported model. A track
	/// that observes an unregistered image throws (COLMAP's map.at()).
	/// </summary>
	public static bool ExportBundler(Reconstruction reconstruction, string path, string listPath, bool skipDistortion = false)
	{
		using StreamWriter file = OpenText(path);
		using StreamWriter listFile = OpenText(listPath);

		file.Write("# Bundle file v0.3\n");
		file.Write(Int(reconstruction.NumRegImages) + " " + Int(reconstruction.NumPoints3D) + "\n");

		var imageIdToIdx = new Dictionary<uint, long>();
		long imageIdx = 0;

		foreach (uint imageId in reconstruction.RegImageIds())
		{
			Image image = reconstruction.Image(imageId);
			Camera camera = reconstruction.Camera(image.CameraId);

			double k1, k2;
			if (skipDistortion || IsPinhole(camera))
			{
				k1 = 0.0;
				k2 = 0.0;
			}
			else if (camera.ModelId == CameraModelId.SimpleRadial)
			{
				k1 = camera.Params[camera.ExtraParamsIdxs[0]];
				k2 = 0.0;
			}
			else if (camera.ModelId == CameraModelId.Radial)
			{
				k1 = camera.Params[camera.ExtraParamsIdxs[0]];
				k2 = camera.Params[camera.ExtraParamsIdxs[1]];
			}
			else
			{
				return false;
			}

			var camFromWorld = image.CamFromWorld();
			Matrix3d r = camFromWorld.Rotation.ToRotationMatrix();
			var text = new StringBuilder();
			text.Append(Dbl(camera.MeanFocalLength())).Append(' ').Append(Dbl(k1)).Append(' ').Append(Dbl(k2)).Append('\n');
			text.Append(Dbl(r[0, 0])).Append(' ').Append(Dbl(r[0, 1])).Append(' ').Append(Dbl(r[0, 2])).Append('\n');
			text.Append(Dbl(-r[1, 0])).Append(' ').Append(Dbl(-r[1, 1])).Append(' ').Append(Dbl(-r[1, 2])).Append('\n');
			text.Append(Dbl(-r[2, 0])).Append(' ').Append(Dbl(-r[2, 1])).Append(' ').Append(Dbl(-r[2, 2])).Append('\n');
			text.Append(Dbl(camFromWorld.Translation.X)).Append(' ');
			text.Append(Dbl(-camFromWorld.Translation.Y)).Append(' ');
			text.Append(Dbl(-camFromWorld.Translation.Z)).Append('\n');
			file.Write(text.ToString());

			listFile.Write(image.Name + "\n");

			imageIdToIdx[imageId] = imageIdx;
			imageIdx += 1;
		}

		foreach (ulong point3DId in SortedPoint3DIds(reconstruction))
		{
			Point3D point3D = reconstruction.Point3D(point3DId);
			var text = new StringBuilder();
			AppendXyz(text, point3D, " ", " ");
			text.Append('\n');
			AppendColor(text, point3D, " ");
			text.Append('\n');

			var line = new StringBuilder();
			line.Append(Int(point3D.Track.Length)).Append(' ');

			foreach (TrackElement trackEl in point3D.Track.Elements)
			{
				Image image = reconstruction.Image(trackEl.ImageId);
				Camera camera = reconstruction.Camera(image.CameraId);

				// Bundler output assumes image coordinate system origin
				// in the lower left corner of the image with the center of
				// the lower left pixel being (0, 0). Our coordinate system
				// starts in the upper left corner with the center of the
				// upper left pixel being (0.5, 0.5).

				Point2D point2D = Point2DAt(image, trackEl.Point2DIdx);

				// image_id_to_idx_.at(): an unregistered image throws.
				line.Append(Int(imageIdToIdx[trackEl.ImageId])).Append(' ');
				line.Append(Int(trackEl.Point2DIdx)).Append(' ');
				line.Append(Dbl6(point2D.Xy.X - camera.PrincipalPointX())).Append(' ');
				line.Append(Dbl6(camera.PrincipalPointY() - point2D.Xy.Y)).Append(' ');
			}

			text.Append(DropLastChar(line)).Append('\n');
			file.Write(text.ToString());
		}

		return true;
	}

	/// <summary>
	/// Port of colmap::ExportPLY: the 3D points (float positions and RGB, no normals) as a
	/// binary PLY file.
	/// </summary>
	public static void ExportPLY(Reconstruction reconstruction, string path)
	{
		List<PlyPoint> plyPoints = reconstruction.ConvertToPLY();

		const bool WriteNormal = false;
		const bool WriteRgb = true;
		Ply.WriteBinaryPlyPoints(path, plyPoints, WriteNormal, WriteRgb);
	}

	/// <summary>
	/// Port of colmap::ExportVRML: every posed image as a small camera frustum of size
	/// <paramref name="imageScale"/> and color <paramref name="imageRgb"/> to
	/// <paramref name="imagesPath"/>, and the 3D points with their colors to
	/// <paramref name="points3DPath"/>. Both files use the stream's default precision 6.
	/// </summary>
	public static void ExportVRML(Reconstruction reconstruction, string imagesPath, string points3DPath, double imageScale, Vector3d imageRgb)
	{
		using (StreamWriter imagesFile = OpenText(imagesPath))
		{
			WriteVrmlImages(reconstruction, imagesFile, imageScale, imageRgb);
		}

		// Write 3D points

		using StreamWriter points3DFile = OpenText(points3DPath);
		var text = new StringBuilder();
		text.Append("#VRML V2.0 utf8\n");
		text.Append("Background { skyColor [1.0 1.0 1.0] } \n");
		text.Append("Shape{ appearance Appearance {\n");
		text.Append(" material Material {emissiveColor 1 1 1} }\n");
		text.Append(" geometry PointSet {\n");
		text.Append(" coord Coordinate {\n");
		text.Append("  point [\n");

		List<ulong> point3DIds = SortedPoint3DIds(reconstruction);
		foreach (ulong point3DId in point3DIds)
		{
			Point3D point3D = reconstruction.Point3D(point3DId);
			text.Append(Dbl6(point3D.Xyz.X)).Append(", ");
			text.Append(Dbl6(point3D.Xyz.Y)).Append(", ");
			text.Append(Dbl6(point3D.Xyz.Z)).Append('\n');
		}

		text.Append(" ] }\n");
		text.Append(" color Color { color [\n");

		foreach (ulong point3DId in point3DIds)
		{
			Point3D point3D = reconstruction.Point3D(point3DId);
			text.Append(Dbl6(point3D.Color.X / 255.0)).Append(", ");
			text.Append(Dbl6(point3D.Color.Y / 255.0)).Append(", ");
			text.Append(Dbl6(point3D.Color.Z / 255.0)).Append('\n');
		}

		text.Append(" ] } } }\n");
		points3DFile.Write(text.ToString());
	}

	private static void WriteVrmlImages(Reconstruction reconstruction, StreamWriter imagesFile, double imageScale, Vector3d imageRgb)
	{
		double six = imageScale * 0.15;
		double siy = imageScale * 0.1;

		Vector3d[] points =
		[
			new(-six, -siy, six * 1.0 * 2.0),
			new(+six, -siy, six * 1.0 * 2.0),
			new(+six, +siy, six * 1.0 * 2.0),
			new(-six, +siy, six * 1.0 * 2.0),
			new(0, 0, 0),
			new(-six / 3.0, -siy / 3.0, six * 1.0 * 2.0),
			new(+six / 3.0, -siy / 3.0, six * 1.0 * 2.0),
			new(+six / 3.0, +siy / 3.0, six * 1.0 * 2.0),
			new(-six / 3.0, +siy / 3.0, six * 1.0 * 2.0),
		];

		string rgb = Dbl6(imageRgb.X) + " " + Dbl6(imageRgb.Y) + " " + Dbl6(imageRgb.Z);
		foreach (uint imageId in ReconstructionIOUtils.ExtractSortedIds(reconstruction.Images))
		{
			Image image = reconstruction.Image(imageId);
			if (!image.HasPose)
			{
				continue;
			}

			var text = new StringBuilder();
			text.Append("Shape{\n");
			text.Append(" appearance Appearance {\n");
			text.Append("  material DEF Default-ffRffGffB Material {\n");
			text.Append("  ambientIntensity 0\n");
			text.Append("  diffuseColor  ").Append(rgb).Append('\n');
			text.Append("  emissiveColor 0.1 0.1 0.1 } }\n");
			text.Append(" geometry IndexedFaceSet {\n");
			text.Append(" solid FALSE \n");
			text.Append(" colorPerVertex TRUE \n");
			text.Append(" ccw TRUE \n");

			text.Append(" coord Coordinate {\n");
			text.Append(" point [\n");

			// Move camera base model to camera pose.
			Matrix3x4d worldFromCam = image.CamFromWorld().Inverse().ToMatrix();
			foreach (Vector3d basePoint in points)
			{
				Vector3d point = worldFromCam * new Vector4d(basePoint.X, basePoint.Y, basePoint.Z, 1.0);
				text.Append(Dbl6(point.X)).Append(' ').Append(Dbl6(point.Y)).Append(' ').Append(Dbl6(point.Z)).Append('\n');
			}

			text.Append(" ] }\n");

			text.Append("color Color {color [\n");
			for (int p = 0; p < points.Length; p++)
			{
				text.Append(' ').Append(rgb).Append('\n');
			}

			text.Append("\n] }\n");

			text.Append("coordIndex [\n");
			text.Append(" 0, 1, 2, 3, -1\n");
			text.Append(" 5, 6, 4, -1\n");
			text.Append(" 6, 7, 4, -1\n");
			text.Append(" 7, 8, 4, -1\n");
			text.Append(" 8, 5, 4, -1\n");
			text.Append(" \n] \n");

			text.Append(" texCoord TextureCoordinate { point [\n");
			text.Append("  1 1,\n");
			text.Append("  0 1,\n");
			text.Append("  0 0,\n");
			text.Append("  1 0,\n");
			text.Append("  0 0,\n");
			text.Append("  0 0,\n");
			text.Append("  0 0,\n");
			text.Append("  0 0,\n");
			text.Append("  0 0,\n");

			text.Append(" ] }\n");
			text.Append("} }\n");
			imagesFile.Write(text.ToString());
		}
	}
}
