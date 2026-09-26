// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionIOTests (continued): the Export* cases of colmap/scene/reconstruction_io_test.cc
// ported 1:1 (ExportNVM, ExportCam, ExportRecon3D, ExportBundler, ExportPLY, ExportVRML),
// testing ColmapSharp/Scene/ReconstructionIO.Export*.cs. COLMAP's cases only check that the
// files exist and that unsupported models return false, so the CSharpOnly_* cases pin the
// exact text on a hand-built one-image scene, with the expected strings derived by hand from
// reconstruction_io.cc (file precision 17, track lines at 6, Eigen's aligned matrices).
// pycolmap exposes only export_PLY, so there is no oracle for the text formats. Tier A.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public partial class ReconstructionIOTests
{
	private static readonly double[] FullOpenCVParams = [1280, 1280, 512, 384, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

	private static string CreateTestDir()
	{
		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-export-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		return dir;
	}

	private static Reconstruction Synthesize(int numFramesPerRig, int numPoints3D, CameraModelId? cameraModelId = null, double[]? cameraParams = null)
	{
		RandomUtils.SetPRNGSeed(0);
		var reconstruction = new Reconstruction();
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = numFramesPerRig,
			NumPoints3D = numPoints3D,
		};
		if (cameraModelId is not null)
		{
			options.CameraModelId = cameraModelId.Value;
		}

		if (cameraParams is not null)
		{
			options.CameraParams = cameraParams;
		}

		Synthetic.SynthesizeDataset(options, reconstruction);
		return reconstruction;
	}

	[Test]
	public async Task ExportNVM_Nominal()
	{
		Reconstruction reconstruction = Synthesize(3, 10, CameraModelId.SimpleRadial);
		string nvmPath = Path.Combine(CreateTestDir(), "export.nvm");

		await Assert.That(ReconstructionIO.ExportNVM(reconstruction, nvmPath)).IsTrue();
		await Assert.That(File.Exists(nvmPath)).IsTrue();
	}

	[Test]
	public async Task ExportNVM_UnsupportedCameraModel()
	{
		Reconstruction reconstruction = Synthesize(1, 5, CameraModelId.FullOpenCV, FullOpenCVParams);
		string nvmPath = Path.Combine(CreateTestDir(), "export.nvm");

		await Assert.That(ReconstructionIO.ExportNVM(reconstruction, nvmPath)).IsFalse();
	}

	[Test]
	public async Task ExportCam_Nominal()
	{
		Reconstruction reconstruction = Synthesize(2, 10, CameraModelId.SimpleRadial);
		string testDir = CreateTestDir();

		await Assert.That(ReconstructionIO.ExportCam(reconstruction, testDir)).IsTrue();

		int numCamFiles = Directory.EnumerateFileSystemEntries(testDir).Count(entry => Path.GetExtension(entry) == ".cam");
		await Assert.That(numCamFiles).IsEqualTo(reconstruction.NumRegImages);
	}

	[Test]
	public async Task ExportCam_UnsupportedCameraModel()
	{
		Reconstruction reconstruction = Synthesize(1, 5, CameraModelId.FullOpenCV, FullOpenCVParams);

		await Assert.That(ReconstructionIO.ExportCam(reconstruction, CreateTestDir())).IsFalse();
	}

	[Test]
	public async Task ExportRecon3D_Nominal()
	{
		Reconstruction reconstruction = Synthesize(3, 15, CameraModelId.SimpleRadial);
		string testDir = CreateTestDir();

		await Assert.That(ReconstructionIO.ExportRecon3D(reconstruction, testDir)).IsTrue();

		string reconDir = Path.Combine(testDir, "Recon");
		await Assert.That(File.Exists(Path.Combine(reconDir, "synth_0.out"))).IsTrue();
		await Assert.That(File.Exists(Path.Combine(reconDir, "urd-images.txt"))).IsTrue();
		await Assert.That(File.Exists(Path.Combine(reconDir, "imagemap_0.txt"))).IsTrue();
	}

	[Test]
	public async Task ExportRecon3D_UnsupportedCameraModel()
	{
		Reconstruction reconstruction = Synthesize(1, 5, CameraModelId.FullOpenCV, FullOpenCVParams);

		await Assert.That(ReconstructionIO.ExportRecon3D(reconstruction, CreateTestDir())).IsFalse();
	}

	[Test]
	public async Task ExportBundler_Nominal()
	{
		Reconstruction reconstruction = Synthesize(3, 20, CameraModelId.SimpleRadial);
		string testDir = CreateTestDir();
		string bundlerPath = Path.Combine(testDir, "bundle.out");
		string listPath = Path.Combine(testDir, "list.txt");

		await Assert.That(ReconstructionIO.ExportBundler(reconstruction, bundlerPath, listPath)).IsTrue();
		await Assert.That(File.Exists(bundlerPath)).IsTrue();
		await Assert.That(File.Exists(listPath)).IsTrue();
	}

	[Test]
	public async Task ExportBundler_UnsupportedCameraModel()
	{
		Reconstruction reconstruction = Synthesize(1, 5, CameraModelId.FullOpenCV, FullOpenCVParams);
		string testDir = CreateTestDir();
		string bundlerPath = Path.Combine(testDir, "bundle.out");
		string listPath = Path.Combine(testDir, "list.txt");

		await Assert.That(ReconstructionIO.ExportBundler(reconstruction, bundlerPath, listPath, false)).IsFalse();
	}

	[Test]
	public async Task ExportPLY_Nominal()
	{
		Reconstruction reconstruction = Synthesize(2, 50);
		string plyPath = Path.Combine(CreateTestDir(), "points.ply");

		ReconstructionIO.ExportPLY(reconstruction, plyPath);
		await Assert.That(File.Exists(plyPath)).IsTrue();
		await Assert.That(new FileInfo(plyPath).Length).IsGreaterThan(0);
	}

	[Test]
	public async Task ExportVRML_Nominal()
	{
		Reconstruction reconstruction = Synthesize(3, 25);
		string testDir = CreateTestDir();
		string imagesPath = Path.Combine(testDir, "images.wrl");
		string points3DPath = Path.Combine(testDir, "points3D.wrl");
		const double ImageScale = 1.0;
		var imageRgb = new Vector3d(1.0, 0.0, 0.0);

		ReconstructionIO.ExportVRML(reconstruction, imagesPath, points3DPath, ImageScale, imageRgb);
		await Assert.That(File.Exists(imagesPath)).IsTrue();
		await Assert.That(File.Exists(points3DPath)).IsTrue();
	}

	// One SIMPLE_RADIAL camera (f = 100, 128 x 80, k = 0.1), one image "a/img.jpg" rotated
	// 180 degrees about z and translated by (0.5, -12.25, 3), and one point at (1, 2, 3)
	// with color (10, 20, 30) observed at (60.5, 30.25). Every value below is exact in
	// binary except k, so the expected text follows from the format alone.
	private static Reconstruction TinyScene()
	{
		var reconstruction = new Reconstruction();
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimpleRadial, 100, 128, 80);
		camera.Params = [100, 64, 40, 0.1];
		reconstruction.AddCameraWithTrivialRig(camera);
		var image = new Image { ImageId = 1, Name = "a/img.jpg" };
		image.SetCameraId(1);
		image.SetPoints2D([new Vector2d(60.5, 30.25), new Vector2d(1, 2)]);
		reconstruction.AddImageWithTrivialFrame(image, new Rigid3d(new Quaterniond(0, 0, 0, 1), new Vector3d(0.5, -12.25, 3)));
		var track = new Track();
		track.AddElement(1, 0);
		reconstruction.AddPoint3D(new Vector3d(1, 2, 3), track, new Vector3ub(10, 20, 30));
		return reconstruction;
	}

	[Test]
	public async Task CSharpOnly_ExportNVM_ExactText()
	{
		string path = Path.Combine(CreateTestDir(), "export.nvm");
		bool ok = ReconstructionIO.ExportNVM(TinyScene(), path);

		await Assert.That(ok).IsTrue();
		await Assert.That(File.ReadAllText(path)).IsEqualTo(
			"NVM_V3 \n \n1  \n"
			+ "a/img.jpg 100 0 0 0 1 0.5 -12.25 -3 -0.10000000000000001 0\n"
			+ "\n1\n"
			+ "1 2 3 10 20 30 1 0 0 60.5 30.25\n");
	}

	[Test]
	public async Task CSharpOnly_ExportCam_ExactText()
	{
		string dir = CreateTestDir();
		bool ok = ReconstructionIO.ExportCam(TinyScene(), dir);

		await Assert.That(ok).IsTrue();
		await Assert.That(File.ReadAllText(Path.Combine(dir, "a", "img.cam"))).IsEqualTo(
			"0.5 -12.25 3 -1 0 0 0 -1 0 0 0 1\n"
			+ "0.78125 0.10000000000000001 1e-10 1 0.5 0.5\n");
	}

	[Test]
	public async Task CSharpOnly_ExportRecon3D_ExactText()
	{
		string dir = CreateTestDir();
		bool ok = ReconstructionIO.ExportRecon3D(TinyScene(), dir);
		string recon = Path.Combine(dir, "Recon");

		await Assert.That(ok).IsTrue();
		// Eigen pads every coefficient to the widest one of the matrix. The track's
		// coordinates are written at precision 6: -3.5 / 128 = -0.02734375 is an exact tie
		// that rounds to even.
		await Assert.That(File.ReadAllText(Path.Combine(recon, "synth_0.out"))).IsEqualTo(
			"colmap 1.0\n1 1\n"
			+ "0.78125 -0.10000000000000001 0\n"
			+ "-1  0  0\n 0 -1  0\n 0  0  1\n"
			+ "   0.5 -12.25      3\n"
			+ "1 2 3\n10 20 30\n"
			+ "1 0 0 -1.0 -0.0273438 -0.0761719\n");
		await Assert.That(File.ReadAllText(Path.Combine(recon, "urd-images.txt"))).IsEqualTo("a/img.jpg\n128 80\n");
		await Assert.That(File.ReadAllText(Path.Combine(recon, "imagemap_0.txt"))).IsEqualTo("0\n");
	}

	[Test]
	public async Task CSharpOnly_ExportBundler_ExactText()
	{
		string dir = CreateTestDir();
		string path = Path.Combine(dir, "bundle.out");
		string listPath = Path.Combine(dir, "list.txt");
		bool ok = ReconstructionIO.ExportBundler(TinyScene(), path, listPath);

		await Assert.That(ok).IsTrue();
		// Negating a zero rotation coefficient writes "-0", as C++ does.
		await Assert.That(File.ReadAllText(path)).IsEqualTo(
			"# Bundle file v0.3\n1 1\n"
			+ "100 0.10000000000000001 0\n"
			+ "-1 0 0\n-0 1 -0\n-0 -0 -1\n"
			+ "0.5 12.25 -3\n"
			+ "1 2 3\n10 20 30\n"
			+ "1 0 0 -3.5 9.75\n");
		await Assert.That(File.ReadAllText(listPath)).IsEqualTo("a/img.jpg\n");
	}

	[Test]
	public async Task CSharpOnly_ExportBundler_UnregisteredObservationThrows()
	{
		// A second, unregistered image that the point also observes: COLMAP's
		// image_id_to_idx_.at() throws std::out_of_range for it.
		Reconstruction reconstruction = TinyScene();
		var unregistered = new Image { ImageId = 2, Name = "b.jpg" };
		unregistered.SetCameraId(1);
		unregistered.SetPoints2D([new Vector2d(3, 4)]);
		reconstruction.AddImageWithTrivialFrame(unregistered);
		reconstruction.AddObservation(1, new TrackElement(2, 0));
		string dir = CreateTestDir();

		await Assert.That(() => ReconstructionIO.ExportBundler(reconstruction, Path.Combine(dir, "b.out"), Path.Combine(dir, "l.txt")))
			.Throws<KeyNotFoundException>();
	}

	[Test]
	public async Task CSharpOnly_ExportVRML_ExactPointsText()
	{
		string dir = CreateTestDir();
		string imagesPath = Path.Combine(dir, "images.wrl");
		string pointsPath = Path.Combine(dir, "points3D.wrl");
		ReconstructionIO.ExportVRML(TinyScene(), imagesPath, pointsPath, 1.0, new Vector3d(1, 0, 0));

		await Assert.That(File.ReadAllText(pointsPath)).IsEqualTo(
			"#VRML V2.0 utf8\nBackground { skyColor [1.0 1.0 1.0] } \nShape{ appearance Appearance {\n"
			+ " material Material {emissiveColor 1 1 1} }\n geometry PointSet {\n coord Coordinate {\n  point [\n"
			+ "1, 2, 3\n ] }\n color Color { color [\n"
			+ "0.0392157, 0.0784314, 0.117647\n ] } } }\n");
		string images = File.ReadAllText(imagesPath);
		await Assert.That(images).StartsWith(
			"Shape{\n appearance Appearance {\n  material DEF Default-ffRffGffB Material {\n  ambientIntensity 0\n"
			+ "  diffuseColor  1 0 0\n  emissiveColor 0.1 0.1 0.1 } }\n geometry IndexedFaceSet {\n solid FALSE \n"
			+ " colorPerVertex TRUE \n ccw TRUE \n coord Coordinate {\n point [\n0.65 -12.15 -2.7\n");
		await Assert.That(images).EndsWith("  0 0,\n ] }\n} }\n");
	}

	[Test]
	[Arguments("img.jpg", "img", ".jpg")]
	[Arguments("dir/a.b.jpg", "dir/a.b", ".jpg")]
	[Arguments("noext", "noext", "")]
	[Arguments("a..jpg", "a", ".jpg")]
	[Arguments("trailing.", "trailing", "")]
	public async Task CSharpOnly_SplitFileExtension(string path, string root, string ext)
	{
		await Assert.That(ReconstructionIO.SplitFileExtension(path)).IsEqualTo((root, ext));
	}
}
