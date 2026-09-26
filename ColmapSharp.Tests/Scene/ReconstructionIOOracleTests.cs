// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ReconstructionIOOracleTests: C#-only oracle tests (not ports of a *_test.cc) for
// ColmapSharp/Scene/ReconstructionIO*.cs and Reconstruction.IO.cs, against the model files
// pycolmap 4.2.0 wrote in TestData/oracle/reconstruction_io (oracle/reconstruction_io.py).
// Tier A: reading gives the values pycolmap reports, binary and text read to the same model,
// and re-writing (through both the Stream and the path APIs) reproduces pycolmap's files byte
// for byte. The rig1cam model's cameras/images/points3D files double as a legacy (pre-rig)
// model, as in reconstruction_io_test.cc's LegacyWithoutRigsAndFrames. COLMAP's own
// reconstruction_io_test.cc cases all need scene/synthetic's SynthesizeDataset and are not
// ported yet; these tests do not stand in for them.

using System.Globalization;
using System.Text.Json;

using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class ReconstructionIOOracleTests
{
	private static readonly string[] FileStems = ["rigs", "cameras", "frames", "images", "points3D"];

	// The fixture directories are binary/ and text/ (a bin/ directory would be git-ignored).
	private static string ModelDir(string model, string format) =>
		OracleFixture.PathOf(Path.Combine("reconstruction_io", model, format == "bin" ? "binary" : "text"));

	private static string CreateTestDir()
	{
		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-io-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		return dir;
	}

	private static Reconstruction ReadModel(string model, string format)
	{
		var reconstruction = new Reconstruction();
		if (format == "bin")
		{
			reconstruction.ReadBinary(ModelDir(model, format));
		}
		else
		{
			reconstruction.ReadText(ModelDir(model, format));
		}

		return reconstruction;
	}

	private static async Task ExpectSameFiles(string expectedDir, string actualDir, string format)
	{
		foreach (string stem in FileStems)
		{
			byte[] expected = File.ReadAllBytes(Path.Combine(expectedDir, $"{stem}.{format}"));
			byte[] actual = File.ReadAllBytes(Path.Combine(actualDir, $"{stem}.{format}"));
			await Assert.That(actual.SequenceEqual(expected)).IsTrue().Because($"{stem}.{format} differs from pycolmap's");
		}
	}

	[Test]
	[Arguments("rig2cam")]
	[Arguments("rig1cam")]
	public async Task ReadBinary_MatchesPycolmapManifest(string model)
	{
		Reconstruction reconstruction = ReadModel(model, "bin");
		JsonElement expected = OracleFixture.Load("reconstruction_io/manifest.json").GetProperty(model);

		await Assert.That(reconstruction.NumRigs).IsEqualTo(expected.GetProperty("num_rigs").GetInt32());
		await Assert.That(reconstruction.NumCameras).IsEqualTo(expected.GetProperty("num_cameras").GetInt32());
		await Assert.That(reconstruction.NumFrames).IsEqualTo(expected.GetProperty("num_frames").GetInt32());
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(expected.GetProperty("num_reg_frames").GetInt32());
		await Assert.That(reconstruction.NumImages).IsEqualTo(expected.GetProperty("num_images").GetInt32());
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(expected.GetProperty("num_reg_images").GetInt32());
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(expected.GetProperty("num_points3D").GetInt32());

		JsonElement cameraJson = expected.GetProperty("first_camera");
		Camera camera = reconstruction.Camera(cameraJson.GetProperty("camera_id").GetUInt32());
		await Assert.That(camera.ModelName).IsEqualTo(cameraJson.GetProperty("model").GetString());
		await Assert.That(camera.Width).IsEqualTo(cameraJson.GetProperty("width").GetInt32());
		await Assert.That(camera.Height).IsEqualTo(cameraJson.GetProperty("height").GetInt32());
		await Assert.That(camera.Params.SequenceEqual(OracleFixture.Doubles(cameraJson.GetProperty("params")))).IsTrue();

		JsonElement imageJson = expected.GetProperty("first_image");
		Image image = reconstruction.Image(imageJson.GetProperty("image_id").GetUInt32());
		await Assert.That(image.Name).IsEqualTo(imageJson.GetProperty("name").GetString());
		await Assert.That(image.CameraId).IsEqualTo(imageJson.GetProperty("camera_id").GetUInt32());
		await Assert.That(image.FrameId).IsEqualTo(imageJson.GetProperty("frame_id").GetUInt32());
		await Assert.That(image.NumPoints2D).IsEqualTo(imageJson.GetProperty("num_points2D").GetUInt32());
		await Assert.That(image.NumPoints3D).IsEqualTo(imageJson.GetProperty("num_points3D").GetUInt32());
		var camFromWorld = image.CamFromWorld();
		double[] q = OracleFixture.Doubles(imageJson.GetProperty("qwxyz"));
		double[] t = OracleFixture.Doubles(imageJson.GetProperty("t"));
		await Assert.That(new[] { camFromWorld.Rotation.W, camFromWorld.Rotation.X, camFromWorld.Rotation.Y, camFromWorld.Rotation.Z }.SequenceEqual(q)).IsTrue();
		await Assert.That(new[] { camFromWorld.Translation.X, camFromWorld.Translation.Y, camFromWorld.Translation.Z }.SequenceEqual(t)).IsTrue();

		foreach (string key in new[] { "first_point3D", "last_point3D" })
		{
			JsonElement pointJson = expected.GetProperty(key);
			Point3D point3D = reconstruction.Point3D(pointJson.GetProperty("id").GetUInt64());
			await Assert.That(new[] { point3D.Xyz.X, point3D.Xyz.Y, point3D.Xyz.Z }.SequenceEqual(OracleFixture.Doubles(pointJson.GetProperty("xyz")))).IsTrue();
			long[] color = OracleFixture.Int64s(pointJson.GetProperty("color"));
			await Assert.That(new long[] { point3D.Color.X, point3D.Color.Y, point3D.Color.Z }.SequenceEqual(color)).IsTrue();
			await Assert.That(point3D.Error).IsEqualTo(pointJson.GetProperty("error").GetDouble());
			long[] track = pointJson.GetProperty("track").EnumerateArray()
				.SelectMany(e => OracleFixture.Int64s(e)).ToArray();
			long[] actualTrack = point3D.Track.Elements.SelectMany(e => new long[] { e.ImageId, e.Point2DIdx }).ToArray();
			await Assert.That(actualTrack.SequenceEqual(track)).IsTrue();
		}
	}

	[Test]
	[Arguments("rig2cam")]
	[Arguments("rig1cam")]
	public async Task ReadText_EqualsReadBinary(string model)
	{
		Reconstruction fromBinary = ReadModel(model, "bin");
		Reconstruction fromText = ReadModel(model, "txt");
		await Assert.That(ReconstructionMatchers.ExplainReconstructionEq(fromText, fromBinary)).IsNull();
	}

	[Test]
	[Arguments("rig2cam", "bin")]
	[Arguments("rig2cam", "txt")]
	[Arguments("rig1cam", "bin")]
	[Arguments("rig1cam", "txt")]
	public async Task Rewrite_IsByteIdenticalToPycolmap(string model, string format)
	{
		Reconstruction reconstruction = ReadModel(model, format);
		string dir = CreateTestDir();
		try
		{
			if (format == "bin")
			{
				reconstruction.WriteBinary(dir);
			}
			else
			{
				reconstruction.WriteText(dir);
			}

			await ExpectSameFiles(ModelDir(model, format), dir, format);

			// Also write the other format from the same model: text from binary-read values and
			// binary from text-read values must match pycolmap too.
			string other = format == "bin" ? "txt" : "bin";
			if (other == "bin")
			{
				reconstruction.WriteBinary(dir);
			}
			else
			{
				reconstruction.WriteText(dir);
			}

			await ExpectSameFiles(ModelDir(model, other), dir, other);
		}
		finally
		{
			Directory.Delete(dir, true);
		}
	}

	[Test]
	public async Task StreamApi_RoundtripsLikeFiles()
	{
		Reconstruction orig = ReadModel("rig2cam", "bin");
		var streams = FileStems.ToDictionary(stem => stem, _ => (Bin: new MemoryStream(), Txt: new MemoryStream()));

		ReconstructionIOBinary.WriteRigsBinary(orig, streams["rigs"].Bin);
		ReconstructionIOBinary.WriteCamerasBinary(orig, streams["cameras"].Bin);
		ReconstructionIOBinary.WriteFramesBinary(orig, streams["frames"].Bin);
		ReconstructionIOBinary.WriteImagesBinary(orig, streams["images"].Bin);
		ReconstructionIOBinary.WritePoints3DBinary(orig, streams["points3D"].Bin);
		ReconstructionIOText.WriteRigsText(orig, streams["rigs"].Txt);
		ReconstructionIOText.WriteCamerasText(orig, streams["cameras"].Txt);
		ReconstructionIOText.WriteFramesText(orig, streams["frames"].Txt);
		ReconstructionIOText.WriteImagesText(orig, streams["images"].Txt);
		ReconstructionIOText.WritePoints3DText(orig, streams["points3D"].Txt);

		foreach (string stem in FileStems)
		{
			byte[] expectedBin = File.ReadAllBytes(Path.Combine(ModelDir("rig2cam", "bin"), stem + ".bin"));
			byte[] expectedTxt = File.ReadAllBytes(Path.Combine(ModelDir("rig2cam", "txt"), stem + ".txt"));
			await Assert.That(streams[stem].Bin.ToArray().SequenceEqual(expectedBin)).IsTrue().Because(stem + ".bin");
			await Assert.That(streams[stem].Txt.ToArray().SequenceEqual(expectedTxt)).IsTrue().Because(stem + ".txt");
			streams[stem].Bin.Position = 0;
			streams[stem].Txt.Position = 0;
		}

		// Read in reconstruction_io_test.cc's Roundtrip order, comparing as it goes.
		foreach (bool binary in new[] { true, false })
		{
			Stream Pick(string stem) => binary ? streams[stem].Bin : streams[stem].Txt;
			var test = new Reconstruction();
			if (binary)
			{
				ReconstructionIOBinary.ReadCamerasBinary(test, Pick("cameras"));
				ReconstructionIOBinary.ReadRigsBinary(test, Pick("rigs"));
				ReconstructionIOBinary.ReadFramesBinary(test, Pick("frames"));
				ReconstructionIOBinary.ReadImagesBinary(test, Pick("images"));
				ReconstructionIOBinary.ReadPoints3DBinary(test, Pick("points3D"));
			}
			else
			{
				ReconstructionIOText.ReadCamerasText(test, Pick("cameras"));
				ReconstructionIOText.ReadRigsText(test, Pick("rigs"));
				ReconstructionIOText.ReadFramesText(test, Pick("frames"));
				ReconstructionIOText.ReadImagesText(test, Pick("images"));
				ReconstructionIOText.ReadPoints3DText(test, Pick("points3D"));
			}

			await Assert.That(ReconstructionMatchers.ExplainReconstructionEq(test, orig)).IsNull();
		}
	}

	[Test]
	[Arguments("bin")]
	[Arguments("txt")]
	public async Task Read_LegacyWithoutRigsAndFrames(string format)
	{
		// Mirrors reconstruction_io_test.cc's LegacyWithoutRigsAndFrames on the oracle model:
		// without rigs and frames files, every camera becomes a trivial rig and every image a
		// frame, which here reproduces pycolmap's own rigs and frames files exactly.
		string dir = CreateTestDir();
		try
		{
			foreach (string stem in new[] { "cameras", "images", "points3D" })
			{
				File.Copy(Path.Combine(ModelDir("rig1cam", format), $"{stem}.{format}"), Path.Combine(dir, $"{stem}.{format}"));
			}

			var legacy = new Reconstruction();
			legacy.Read(dir);
			await Assert.That(ReconstructionMatchers.ExplainReconstructionEq(legacy, ReadModel("rig1cam", format))).IsNull();

			string outDir = Path.Combine(dir, "out");
			Directory.CreateDirectory(outDir);
			legacy.Write(outDir);
			await ExpectSameFiles(ModelDir("rig1cam", "bin"), outDir, "bin");
		}
		finally
		{
			Directory.Delete(dir, true);
		}
	}

	[Test]
	public async Task WriteText_IgnoresCurrentCulture()
	{
		// C#-only counterpart of reconstruction_io_test.cc's TextIO.LocaleIndependentRoundtrip
		// (which needs SynthesizeDataset): a comma-decimal culture must not reach the files.
		Reconstruction orig = ReadModel("rig2cam", "txt");
		CultureInfo saved = CultureInfo.CurrentCulture;
		string dir = CreateTestDir();
		try
		{
			CultureInfo.CurrentCulture = new CultureInfo("de-DE");
			await Assert.That(1.23.ToString(CultureInfo.CurrentCulture)).IsEqualTo("1,23");
			orig.WriteText(dir);
			var test = new Reconstruction();
			test.ReadText(dir);
			CultureInfo.CurrentCulture = saved;

			await ExpectSameFiles(ModelDir("rig2cam", "txt"), dir, "txt");
			await Assert.That(ReconstructionMatchers.ExplainReconstructionEq(test, orig)).IsNull();
		}
		finally
		{
			CultureInfo.CurrentCulture = saved;
			Directory.Delete(dir, true);
		}
	}

	[Test]
	public async Task Read_ThrowsWithoutModelFiles()
	{
		string dir = CreateTestDir();
		try
		{
			await Assert.That(() => new Reconstruction().Read(dir)).Throws<InvalidOperationException>();
		}
		finally
		{
			Directory.Delete(dir, true);
		}
	}
}
