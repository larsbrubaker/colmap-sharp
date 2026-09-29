// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ReconstructionIORobustnessTests: C#-only tests (not ports of a *_test.cc) for how
// ColmapSharp/Scene/ReconstructionIO*.cs handles unusual or broken input: multibyte UTF-8
// image names, names that are not UTF-8 (divergence 25), truncated and
// lying binary files, unknown camera models and sensor types, images no frame holds, and the
// libc++ `istream >>` number spellings CppLineTokens.cs reproduces (probed with a libc++
// harness: "-1" read as an unsigned is its maximum, "0x1p3" is 8, a decimal that underflows
// to a subnormal fails, "inf"/"nan" fail).

using System.Text;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class ReconstructionIORobustnessTests
{
	private static string CreateTestDir()
	{
		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-io-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		return dir;
	}

	// One camera, one rig, one posed frame with image 1 named `name`, two 2D points, one 3D point.
	private static Reconstruction OneImageModel(string name)
	{
		var reconstruction = new Reconstruction();
		Camera camera = Camera.CreateFromModelId(1, CameraModelId.SimpleRadial, 100, 64, 48);
		reconstruction.AddCameraWithTrivialRig(camera);
		var frame = new Frame { FrameId = 1 };
		frame.SetRigId(1);
		frame.AddDataId(new DataId(camera.SensorId, 1));
		frame.SetRigFromWorld(new Rigid3d(new Quaterniond(1, 0, 0, 0), new Vector3d(0.5, -0.25, 3)));
		reconstruction.AddFrame(frame);
		var image = new Image { ImageId = 1, Name = name };
		image.SetCameraId(1);
		image.SetFrameId(1);
		image.SetPoints2D([new Vector2d(1.5, 2.5), new Vector2d(10, 20)]);
		reconstruction.AddImage(image);
		var point3D = new Point3D { Xyz = new Vector3d(0.1, 0.2, 0.3), Color = new Vector3ub(1, 2, 3), Error = 0.5 };
		point3D.Track.AddElement(1, 0);
		reconstruction.AddPoint3D(1, point3D);
		return reconstruction;
	}

	private static MemoryStream Bytes(params byte[][] parts) => new(parts.SelectMany(p => p).ToArray());

	private static MemoryStream Text(string text) => new(Encoding.UTF8.GetBytes(text));

	[Test]
	[Arguments(true)]
	[Arguments(false)]
	public async Task MultibyteUtf8Name_Roundtrips(bool binary)
	{
		Reconstruction orig = OneImageModel("café/画像.jpg");
		string dir = CreateTestDir();
		try
		{
			if (binary)
			{
				orig.WriteBinary(dir);
			}
			else
			{
				orig.WriteText(dir);
			}

			var test = new Reconstruction();
			test.Read(dir);
			await Assert.That(test.Image(1).Name).IsEqualTo("café/画像.jpg");
			await Assert.That(ReconstructionMatchers.ExplainReconstructionEq(test, orig)).IsNull();
		}
		finally
		{
			Directory.Delete(dir, true);
		}
	}

	[Test]
	[Arguments(true)]
	[Arguments(false)]
	public async Task NonUtf8Name_ThrowsNamingFileAndImage(bool binary)
	{
		Reconstruction orig = OneImageModel("Xname.jpg");
		string dir = CreateTestDir();
		try
		{
			string imagesPath = Path.Combine(dir, binary ? "images.bin" : "images.txt");
			if (binary)
			{
				orig.WriteBinary(dir);
			}
			else
			{
				orig.WriteText(dir);
			}

			// Latin-1 'é' (0xE9) alone is not valid UTF-8.
			byte[] bytes = File.ReadAllBytes(imagesPath);
			// (Search for the whole name: the text header's comments contain 'X' too.)
			int at = bytes.AsSpan().IndexOf("Xname.jpg"u8);
			bytes[at] = 0xE9;
			File.WriteAllBytes(imagesPath, bytes);

			var test = new Reconstruction();
			InvalidDataException? e = Assert.Throws<InvalidDataException>(() => test.Read(dir));
			await Assert.That(e!.Message).Contains("Image 1");
			await Assert.That(e.Message).Contains(imagesPath);
			await Assert.That(e.Message).Contains("UTF-8");
		}
		finally
		{
			Directory.Delete(dir, true);
		}
	}

	[Test]
	public async Task TruncatedBinary_Throws()
	{
		string dir = CreateTestDir();
		try
		{
			OneImageModel("a.jpg").WriteBinary(dir);
			string points3DPath = Path.Combine(dir, "points3D.bin");
			byte[] bytes = File.ReadAllBytes(points3DPath);
			File.WriteAllBytes(points3DPath, bytes[..(bytes.Length / 2)]);
			await Assert.That(() => new Reconstruction().ReadBinary(dir)).Throws<EndOfStreamException>();
		}
		finally
		{
			Directory.Delete(dir, true);
		}
	}

	[Test]
	public async Task HugeCountInShortFile_ThrowsFast()
	{
		byte[] huge = BitConverter.GetBytes(1UL << 62);
		byte[] few = [1, 0, 0, 0, 2, 0];

		await Assert.That(() => ReconstructionIOBinary.ReadRigsBinary(new Reconstruction(), Bytes(huge, few)))
			.Throws<EndOfStreamException>();
		await Assert.That(() => ReconstructionIOBinary.ReadCamerasBinary(new Reconstruction(), Bytes(huge, few)))
			.Throws<EndOfStreamException>();
		await Assert.That(() => ReconstructionIOBinary.ReadFramesBinary(new Reconstruction(), Bytes(huge, few)))
			.Throws<EndOfStreamException>();
		await Assert.That(() => ReconstructionIOBinary.ReadImagesBinary(new Reconstruction(), Bytes(huge, few)))
			.Throws<EndOfStreamException>();
		await Assert.That(() => ReconstructionIOBinary.ReadPoints3DBinary(new Reconstruction(), Bytes(huge, few)))
			.Throws<EndOfStreamException>();

		// A huge track length after a valid point header.
		byte[] pointHeader = [.. BitConverter.GetBytes(1UL), .. new byte[24], 0, 0, 0, .. new byte[8]];
		await Assert.That(() => ReconstructionIOBinary.ReadPoints3DBinary(
			new Reconstruction(), Bytes(BitConverter.GetBytes(1UL), pointHeader, huge, few))).Throws<EndOfStreamException>();
	}

	[Test]
	public async Task InvalidCameraModel_Throws()
	{
		byte[] camera = [.. BitConverter.GetBytes(1u), .. BitConverter.GetBytes(999), .. BitConverter.GetBytes(64UL), .. BitConverter.GetBytes(48UL)];
		await Assert.That(() => ReconstructionIOBinary.ReadCamerasBinary(new Reconstruction(), Bytes(BitConverter.GetBytes(1UL), camera)))
			.Throws<ArgumentException>();
		await Assert.That(() => ReconstructionIOText.ReadCamerasText(new Reconstruction(), Text("1 NO_SUCH_MODEL 64 48 100 32 24\n")))
			.Throws<ArgumentException>();
	}

	[Test]
	public async Task UnknownSensorType_Throws()
	{
		var reconstruction = new Reconstruction();
		reconstruction.AddCamera(Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 100, 64, 48));
		InvalidOperationException? e = Assert.Throws<InvalidOperationException>(
			() => ReconstructionIOText.ReadRigsText(reconstruction, Text("1 1 LIDAR 1\n")));
		await Assert.That(e!.Message).IsEqualTo("Unknown string value: LIDAR for enum: SensorType");
	}

	[Test]
	[Arguments(true)]
	[Arguments(false)]
	public async Task ImageInNoFrame_ThrowsNamingImage(bool binary)
	{
		Reconstruction orig = OneImageModel("a.jpg");
		var images = new MemoryStream();
		if (binary)
		{
			ReconstructionIOBinary.WriteImagesBinary(orig, images);
		}
		else
		{
			ReconstructionIOText.WriteImagesText(orig, images);
		}

		// Same cameras and rigs, but the only frame holds image 7, not image 1.
		var test = new Reconstruction();
		Camera camera = orig.Camera(1);
		test.AddCameraWithTrivialRig(camera);
		var frame = new Frame { FrameId = 1 };
		frame.SetRigId(1);
		frame.AddDataId(new DataId(camera.SensorId, 7));
		test.AddFrame(frame);

		images.Position = 0;
		ArgumentException? e = Assert.Throws<ArgumentException>(() =>
		{
			if (binary)
			{
				ReconstructionIOBinary.ReadImagesBinary(test, images);
			}
			else
			{
				ReconstructionIOText.ReadImagesText(test, images);
			}
		});
		await Assert.That(e!.Message).Contains("Image 1 is not in any frame");
	}

	[Test]
	public async Task CppLineTokens_Trim_StripsOnlyStringTrimWhitespace()
	{
		// COLMAP's StringTrim strips ' ', '\n', '\r', '\t' only; '\v' and '\f' stay.
		await Assert.That(CppLineTokens.Trim(" \t\r\n\v a \f\n")).IsEqualTo("\v a \f");
	}

	[Test]
	public async Task CppLineTokens_UnsignedReadsLikeLibcxx()
	{
		var tokens = new CppLineTokens("-1 -4294967295 +7 -0 4294967296 -4294967296");
		await Assert.That(tokens.TryReadUInt32(out uint a) && a == uint.MaxValue).IsTrue();
		await Assert.That(tokens.TryReadUInt32(out uint b) && b == 1).IsTrue();
		await Assert.That(tokens.TryReadUInt32(out uint c) && c == 7).IsTrue();
		await Assert.That(tokens.TryReadUInt32(out uint d) && d == 0).IsTrue();
		await Assert.That(tokens.TryReadUInt32(out _)).IsFalse();

		var tokens64 = new CppLineTokens("-1 -4294967296 18446744073709551616");
		await Assert.That(tokens64.TryReadUInt64(out ulong e) && e == ulong.MaxValue).IsTrue();
		await Assert.That(tokens64.TryReadUInt64(out ulong f) && f == 18446744069414584320UL).IsTrue();
		await Assert.That(tokens64.TryReadUInt64(out _)).IsFalse();
	}

	[Test]
	public async Task CppLineTokens_DoubleReadsLikeLibcxx()
	{
		(string Token, double? Value)[] cases =
		[
			("0x1p3", 8), ("0X1.8P1", 3), ("0x10", 16), ("0x.8", 0.5), ("-0x1p-1074", -double.Epsilon),
			("0x1.fffffffffffff8p0", 2), ("+5", 5), (".5", 0.5), ("5.", 5), ("-0", -0.0),
			("0x1p-1080", null), ("1e-320", null), ("1e400", null), ("inf", null), ("nan", null), ("12abc", null),
		];
		foreach (var (token, value) in cases)
		{
			bool ok = new CppLineTokens(token).TryReadDouble(out double parsed);
			await Assert.That(ok).IsEqualTo(value.HasValue).Because(token);
			if (value.HasValue)
			{
				await Assert.That(BitConverter.DoubleToInt64Bits(parsed)).IsEqualTo(BitConverter.DoubleToInt64Bits(value.Value)).Because(token);
			}
		}
	}
}
