// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Model.Pmvs: the PMVS half of colmap/mvs/model.cc - ReadFromPMVS (Bundler "bundle.rd.out"
// or raw "txt/*.txt" projection matrices + "vis.dat") and GetMaxOverlappingImagesFromPMVS.
// Model.cs holds the COLMAP reader and the queries; Workspace.ImportPMVSWorkspace
// (Workspace.cs) imports a PMVS workspace built from this. model_test.cc has no PMVS
// case; ModelTests.Model_ReadBundlerPMVSNegativeTrackIndex (C#-only) covers one check.
//
// Tier A for the text parsing (Util/CppLineTokens.cs reads tokens like istream >>); the raw
// format decomposes P through Geometry/Pose.cs DecomposeProjectionMatrix (Tier B).
//
// Translation notes:
// - Image sizes come from the image files, which the library does not decode: the host's
//   IBitmapSource reads them (Bitmap::Read in COLMAP), and its Exists replaces ExistsFile
//   for the image files. The text files are read here.
// - vis.dat is read with unchecked extractions in COLMAP; a failed extraction yields 0
//   here, as C++11 streams store 0 on failure, and the range checks then apply.

using System.Globalization;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

namespace ColmapSharp.Mvs;

public sealed partial class Model
{
	private readonly List<List<int>> pmvsVisDat = new();

	/// <summary>
	/// Reads a PMVS model from <paramref name="path"/> (Bundler or raw format), reading image
	/// sizes through <paramref name="bitmaps"/>.
	/// </summary>
	public void ReadFromPMVS(string path, IBitmapSource bitmaps)
	{
		if (ReadFromBundlerPMVS(path, bitmaps) || ReadFromRawPMVS(path, bitmaps))
		{
			return;
		}

		throw new InvalidOperationException("Invalid PMVS format");
	}

	/// <summary>The overlapping images defined in the vis.dat file (empty unless read from raw PMVS).</summary>
	public List<List<int>> GetMaxOverlappingImagesFromPMVS() => pmvsVisDat;

	private static string PmvsImageName(int imageIdx) => imageIdx.ToString("D8", CultureInfo.InvariantCulture) + ".jpg";

	private bool ReadFromBundlerPMVS(string path, IBitmapSource bitmaps)
	{
		string bundleFilePath = System.IO.Path.Combine(path, "bundle.rd.out");
		if (!File.Exists(bundleFilePath))
		{
			return false;
		}

		string text = File.ReadAllText(bundleFilePath);

		// Header line.
		int headerEnd = text.IndexOf('\n');
		var file = new CppLineTokens(headerEnd < 0 ? "" : text[(headerEnd + 1)..]);

		Check.That(file.TryReadInt32(out int numImages) & file.TryReadInt32(out int numPoints));

		Images.Capacity = Math.Max(Images.Capacity, numImages);
		Span<float> r = stackalloc float[9];
		Span<float> t = stackalloc float[3];
		Span<float> k = stackalloc float[9];
		for (int imageIdx = 0; imageIdx < numImages; ++imageIdx)
		{
			string imageName = PmvsImageName(imageIdx);
			string imagePath = System.IO.Path.Combine(path, "visualize", imageName);

			k.Clear();
			k[0] = 1.0f;
			k[4] = 1.0f;
			k[8] = 1.0f;

			Check.That(file.TryReadFloat(out k[0]));
			k[4] = k[0];

			Bitmap bitmap = bitmaps.Read(imagePath, asRgb: true);
			k[2] = bitmap.Width / 2.0f;
			k[5] = bitmap.Height / 2.0f;

			Check.That(file.TryReadFloat(out float k1) & file.TryReadFloat(out float k2));
			Check.Eq(k1, 0.0f);
			Check.Eq(k2, 0.0f);

			for (int i = 0; i < 9; ++i)
			{
				Check.That(file.TryReadFloat(out r[i]));
			}

			for (int i = 3; i < 9; ++i)
			{
				r[i] = -r[i];
			}

			Check.That(file.TryReadFloat(out t[0]) & file.TryReadFloat(out t[1]) & file.TryReadFloat(out t[2]));
			t[1] = -t[1];
			t[2] = -t[2];

			AddImage(new Image(imagePath, bitmap.Width, bitmap.Height, k, r, t), imageName, imageIdx);
		}

		for (int pointId = 0; pointId < numPoints; ++pointId)
		{
			var point = new Point();
			Check.That(file.TryReadFloat(out point.X) & file.TryReadFloat(out point.Y) & file.TryReadFloat(out point.Z));

			Check.That(file.TryReadInt32(out _) & file.TryReadInt32(out _) & file.TryReadInt32(out _));

			Check.That(file.TryReadInt32(out int trackLen));
			point.Track.Capacity = Math.Max(0, trackLen);

			for (int i = 0; i < trackLen; ++i)
			{
				Check.That(
					file.TryReadInt32(out int trackImageIdx) & file.TryReadInt32(out _) &
					file.TryReadFloat(out _) & file.TryReadFloat(out _));
				// COLMAP compares the int index with images.size() as size_t, so a negative
				// index fails there too.
				Check.Ge(trackImageIdx, 0);
				Check.Lt(trackImageIdx, Images.Count);
				point.Track.Add(trackImageIdx);
			}

			Points.Add(point);
		}

		return true;
	}

	private bool ReadFromRawPMVS(string path, IBitmapSource bitmaps)
	{
		string visDatPath = System.IO.Path.Combine(path, "vis.dat");
		if (!File.Exists(visDatPath))
		{
			return false;
		}

		Span<double> pValues = stackalloc double[12];
		Span<float> k = stackalloc float[9];
		Span<float> r = stackalloc float[9];
		Span<float> t = stackalloc float[3];
		for (int imageIdx = 0; ; ++imageIdx)
		{
			string imageName = PmvsImageName(imageIdx);
			string imagePath = System.IO.Path.Combine(path, "visualize", imageName);

			if (!bitmaps.Exists(imagePath))
			{
				break;
			}

			Bitmap bitmap = bitmaps.Read(imagePath, asRgb: true);

			string projMatrixPath = System.IO.Path.Combine(
				path, "txt", imageIdx.ToString("D8", CultureInfo.InvariantCulture) + ".txt");
			var projMatrixFile = new CppLineTokens(File.ReadAllText(projMatrixPath));

			Check.That(projMatrixFile.TryReadString(out string contour));
			Check.That(contour == "CONTOUR", $"Check failed: contour == \"CONTOUR\" ({contour} vs. CONTOUR)");

			for (int i = 0; i < 12; ++i)
			{
				Check.That(projMatrixFile.TryReadDouble(out pValues[i]));
			}

			var p = new Matrix3x4d(
				pValues[0], pValues[1], pValues[2], pValues[3],
				pValues[4], pValues[5], pValues[6], pValues[7],
				pValues[8], pValues[9], pValues[10], pValues[11]);

			Pose.DecomposeProjectionMatrix(p, out Matrix3d kd, out Matrix3d rd, out Vector3d td);

			// The COLMAP patch match algorithm requires that there is no skew.
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 3; col++)
				{
					k[row * 3 + col] = (float)kd[row, col];
					r[row * 3 + col] = (float)rd[row, col];
				}

				t[row] = (float)td[row];
			}

			// The COLMAP patch match algorithm requires that there is no skew.
			k[1] = 0.0f;
			k[3] = 0.0f;
			k[6] = 0.0f;
			k[7] = 0.0f;
			k[8] = 1.0f;

			AddImage(new Image(imagePath, bitmap.Width, bitmap.Height, k, r, t), imageName, imageIdx);
		}

		var visDatFile = new CppLineTokens(File.ReadAllText(visDatPath));

		visDatFile.TryReadString(out string visdata);
		Check.That(visdata == "VISDATA", $"Check failed: visdata == \"VISDATA\" ({visdata} vs. VISDATA)");

		visDatFile.TryReadInt32(out int numImages);
		Check.Ge(numImages, 0);
		Check.Eq(numImages, Images.Count);

		pmvsVisDat.Clear();
		for (int i = 0; i < numImages; ++i)
		{
			pmvsVisDat.Add(new List<int>());
		}

		for (int i = 0; i < numImages; ++i)
		{
			visDatFile.TryReadInt32(out int imageIdx);
			Check.Ge(imageIdx, 0);
			Check.Lt(imageIdx, numImages);

			visDatFile.TryReadInt32(out int numVisibleImages);

			List<int> visibleImageIdxs = pmvsVisDat[imageIdx];
			visibleImageIdxs.Capacity = Math.Max(visibleImageIdxs.Capacity, numVisibleImages);

			for (int j = 0; j < numVisibleImages; ++j)
			{
				visDatFile.TryReadInt32(out int visibleImageIdx);
				Check.Ge(visibleImageIdx, 0);
				Check.Lt(visibleImageIdx, numImages);
				if (visibleImageIdx != imageIdx)
				{
					visibleImageIdxs.Add(visibleImageIdx);
				}
			}
		}

		return true;
	}
}
