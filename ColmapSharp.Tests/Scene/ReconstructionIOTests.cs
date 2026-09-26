// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionIOTests: colmap/scene/reconstruction_io_test.cc ported 1:1, testing
// ColmapSharp/Scene/ReconstructionIO*.cs. The parameterized suite
// ReaderWriterTests/ParameterizedReaderWriterTests runs Roundtrip and
// LegacyWithoutRigsAndFrames for the four reader/writers (text and binary, string stream and
// file); a std::stringstream is a MemoryStream here, read back from its start. The
// comma-decimal global locale of TextIO.LocaleIndependentRoundtrip is a CurrentCulture whose
// decimal separator is "," (built from the invariant culture, so it works in invariant
// globalization mode too). ReconstructionIOOracleTests.cs holds the C#-only pycolmap checks.
//
// The Export* cases are in ReconstructionIOTests.Export.cs.
// COLMAP's gtest_main seeds the PRNG with 0 before every test; tests seed and synthesize
// before their first await (the PRNG is thread-local).

using System.Globalization;

using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Scene.ReconstructionIOBinary;
using static ColmapSharp.Scene.ReconstructionIOText;

namespace ColmapSharp.Tests.Scene;

public partial class ReconstructionIOTests
{
	private static readonly string[] Parts = ["rigs", "cameras", "frames", "images", "points3D"];

	/// <summary>The four reader/writers of INSTANTIATE_TEST_SUITE_P(ReaderWriterTests, ...).</summary>
	public static IEnumerable<string> ReaderWriters() =>
		["TextStringStream", "BinaryStringStream", "TextFileStream", "BinaryFileStream"];

	/// <summary>
	/// Port of the test's ReaderWriter hierarchy: writes and reads each part through a
	/// string stream (MemoryStream) or a file in a fresh test directory, as text or binary.
	/// Disposing deletes the test directory.
	/// </summary>
	private sealed class ReaderWriter : IDisposable
	{
		private readonly bool text;
		private readonly string? testDir;
		private readonly Dictionary<string, byte[]> streams = [];

		public ReaderWriter(string kind)
		{
			text = kind.StartsWith("Text", StringComparison.Ordinal);
			if (kind.EndsWith("FileStream", StringComparison.Ordinal))
			{
				testDir = Path.Combine(Path.GetTempPath(), "colmapsharp-io-" + Guid.NewGuid().ToString("N"));
				Directory.CreateDirectory(testDir);
			}
		}

		public void Dispose()
		{
			if (testDir != null)
			{
				Directory.Delete(testDir, true);
			}
		}

		private string PathOf(string part) => Path.Combine(testDir!, part + (text ? ".txt" : ".bin"));

		public void Write(string part, Reconstruction reconstruction)
		{
			if (testDir != null)
			{
				WritePath(part, reconstruction, PathOf(part));
				return;
			}

			var stream = new MemoryStream();
			WriteStream(part, reconstruction, stream);
			streams[part] = stream.ToArray();
		}

		public void Read(string part, Reconstruction reconstruction)
		{
			if (testDir != null)
			{
				ReadPath(part, reconstruction, PathOf(part));
				return;
			}

			ReadStream(part, reconstruction, new MemoryStream(streams.GetValueOrDefault(part, [])));
		}

		public string Str(string part) => testDir != null
			? File.ReadAllText(PathOf(part))
			: System.Text.Encoding.UTF8.GetString(streams.GetValueOrDefault(part, []));

		private void WriteStream(string part, Reconstruction r, Stream s)
		{
			switch (part)
			{
				case "rigs": if (text) { WriteRigsText(r, s); } else { WriteRigsBinary(r, s); } break;
				case "cameras": if (text) { WriteCamerasText(r, s); } else { WriteCamerasBinary(r, s); } break;
				case "frames": if (text) { WriteFramesText(r, s); } else { WriteFramesBinary(r, s); } break;
				case "images": if (text) { WriteImagesText(r, s); } else { WriteImagesBinary(r, s); } break;
				default: if (text) { WritePoints3DText(r, s); } else { WritePoints3DBinary(r, s); } break;
			}
		}

		private void WritePath(string part, Reconstruction r, string p)
		{
			switch (part)
			{
				case "rigs": if (text) { WriteRigsText(r, p); } else { WriteRigsBinary(r, p); } break;
				case "cameras": if (text) { WriteCamerasText(r, p); } else { WriteCamerasBinary(r, p); } break;
				case "frames": if (text) { WriteFramesText(r, p); } else { WriteFramesBinary(r, p); } break;
				case "images": if (text) { WriteImagesText(r, p); } else { WriteImagesBinary(r, p); } break;
				default: if (text) { WritePoints3DText(r, p); } else { WritePoints3DBinary(r, p); } break;
			}
		}

		private void ReadStream(string part, Reconstruction r, Stream s)
		{
			switch (part)
			{
				case "rigs": if (text) { ReadRigsText(r, s); } else { ReadRigsBinary(r, s); } break;
				case "cameras": if (text) { ReadCamerasText(r, s); } else { ReadCamerasBinary(r, s); } break;
				case "frames": if (text) { ReadFramesText(r, s); } else { ReadFramesBinary(r, s); } break;
				case "images": if (text) { ReadImagesText(r, s); } else { ReadImagesBinary(r, s); } break;
				default: if (text) { ReadPoints3DText(r, s); } else { ReadPoints3DBinary(r, s); } break;
			}
		}

		private void ReadPath(string part, Reconstruction r, string p)
		{
			switch (part)
			{
				case "rigs": if (text) { ReadRigsText(r, p); } else { ReadRigsBinary(r, p); } break;
				case "cameras": if (text) { ReadCamerasText(r, p); } else { ReadCamerasBinary(r, p); } break;
				case "frames": if (text) { ReadFramesText(r, p); } else { ReadFramesBinary(r, p); } break;
				case "images": if (text) { ReadImagesText(r, p); } else { ReadImagesBinary(r, p); } break;
				default: if (text) { ReadPoints3DText(r, p); } else { ReadPoints3DBinary(r, p); } break;
			}
		}
	}

	// EXPECT_EQ(a.Xs(), b.Xs()): the same key set with operator== equal values.
	private static bool MapEq<TValue>(IReadOnlyDictionary<uint, TValue> a, IReadOnlyDictionary<uint, TValue> b)
		where TValue : IEquatable<TValue> =>
		a.Count == b.Count && a.All(kv => b.TryGetValue(kv.Key, out TValue? other) && kv.Value.Equals(other));

	private static bool Points3DEq(Reconstruction a, Reconstruction b) =>
		a.Points3D.Count == b.Points3D.Count
		&& a.Points3D.All(kv => b.Points3D.TryGetValue(kv.Key, out Point3D? other) && kv.Value.Equals(other));

	[Test]
	[MethodDataSource(nameof(ReaderWriters))]
	public async Task ParameterizedReaderWriterTests_Roundtrip(string kind)
	{
		RandomUtils.SetPRNGSeed(0);
		using var readerWriter = new ReaderWriter(kind);

		var orig = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 3,
			NumCamerasPerRig = 4,
			NumFramesPerRig = 5,
			NumPoints3D = 321,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, orig);

		var written = new List<bool>();
		foreach (string part in Parts)
		{
			readerWriter.Write(part, orig);
			written.Add(readerWriter.Str(part).Length > 0);
		}

		var test = new Reconstruction();
		readerWriter.Read("cameras", test);
		bool camerasEq = MapEq(orig.Cameras, test.Cameras);
		readerWriter.Read("rigs", test);
		bool rigsEq = MapEq(orig.Rigs, test.Rigs);
		readerWriter.Read("frames", test);
		bool framesEq = MapEq(orig.Frames, test.Frames);
		readerWriter.Read("images", test);
		bool imagesEq = MapEq(orig.Images, test.Images);
		readerWriter.Read("points3D", test);
		bool points3DEq = Points3DEq(orig, test);

		await Assert.That(written).DoesNotContain(false);
		await Assert.That(camerasEq).IsTrue();
		await Assert.That(rigsEq).IsTrue();
		await Assert.That(framesEq).IsTrue();
		await Assert.That(imagesEq).IsTrue();
		await Assert.That(points3DEq).IsTrue();
	}

	[Test]
	[MethodDataSource(nameof(ReaderWriters))]
	public async Task ParameterizedReaderWriterTests_LegacyWithoutRigsAndFrames(string kind)
	{
		RandomUtils.SetPRNGSeed(0);
		using var readerWriter = new ReaderWriter(kind);

		var orig = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 3,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 5,
			NumPoints3D = 321,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, orig);

		var written = new List<bool>();
		foreach (string part in new[] { "cameras", "images", "points3D" })
		{
			readerWriter.Write(part, orig);
			written.Add(readerWriter.Str(part).Length > 0);
		}

		var test = new Reconstruction();
		readerWriter.Read("cameras", test);
		bool camerasEq = MapEq(orig.Cameras, test.Cameras);
		readerWriter.Read("images", test);
		bool rigsEq = MapEq(orig.Rigs, test.Rigs);
		bool framesEq = MapEq(orig.Frames, test.Frames);
		bool imagesEq = MapEq(orig.Images, test.Images);
		readerWriter.Read("points3D", test);
		bool points3DEq = Points3DEq(orig, test);

		await Assert.That(written).DoesNotContain(false);
		await Assert.That(camerasEq).IsTrue();
		await Assert.That(rigsEq).IsTrue();
		await Assert.That(framesEq).IsTrue();
		await Assert.That(imagesEq).IsTrue();
		await Assert.That(points3DEq).IsTrue();
	}

	[Test]
	public async Task TextIO_LocaleIndependentRoundtrip()
	{
		RandomUtils.SetPRNGSeed(0);
		// Set global locale to use comma as decimal separator.
		CultureInfo originalCulture = CultureInfo.CurrentCulture;
		var commaDecimal = (CultureInfo)CultureInfo.InvariantCulture.Clone();
		commaDecimal.NumberFormat.NumberDecimalSeparator = ",";
		string formatted;
		var dataLines = new List<string>();
		bool camerasEq, rigsEq, framesEq, imagesEq, points3DEq;
		try
		{
			CultureInfo.CurrentCulture = commaDecimal;

			// Verify that the locale is used correctly.
			formatted = 1.23.ToString();

			var orig = new Reconstruction();
			var syntheticDatasetOptions = new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 2,
				NumFramesPerRig = 3,
				NumPoints3D = 50,
			};
			Synthetic.SynthesizeDataset(syntheticDatasetOptions, orig);

			// Write under comma-decimal locale.
			using var rw = new ReaderWriter("TextStringStream");
			foreach (string part in Parts)
			{
				rw.Write(part, orig);
			}

			// Verify written float data uses dot, not comma, as decimal separator.
			// Check a data line (skip comment lines starting with '#').
			foreach (string line in rw.Str("cameras").Split('\n'))
			{
				if (line.Length > 0 && line[0] != '#')
				{
					dataLines.Add(line);
				}
			}

			// Read back under comma-decimal locale.
			var test = new Reconstruction();
			rw.Read("cameras", test);
			camerasEq = MapEq(orig.Cameras, test.Cameras);
			rw.Read("rigs", test);
			rigsEq = MapEq(orig.Rigs, test.Rigs);
			rw.Read("frames", test);
			framesEq = MapEq(orig.Frames, test.Frames);
			rw.Read("images", test);
			imagesEq = MapEq(orig.Images, test.Images);
			rw.Read("points3D", test);
			points3DEq = Points3DEq(orig, test);
		}
		finally
		{
			// Restore original locale.
			CultureInfo.CurrentCulture = originalCulture;
		}

		await Assert.That(formatted).IsEqualTo("1,23");
		foreach (string line in dataLines)
		{
			await Assert.That(line).Contains(".");
			await Assert.That(line).DoesNotContain(",");
		}

		await Assert.That(camerasEq).IsTrue();
		await Assert.That(rigsEq).IsTrue();
		await Assert.That(framesEq).IsTrue();
		await Assert.That(imagesEq).IsTrue();
		await Assert.That(points3DEq).IsTrue();
	}
}
