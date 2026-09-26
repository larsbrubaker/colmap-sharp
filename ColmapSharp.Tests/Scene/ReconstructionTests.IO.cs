// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionTests.IO: the reconstruction_test.cc cases that synthesize a dataset and go
// through reconstruction_io (Scene/ReconstructionIO*.cs, Reconstruction.IO.cs), 1:1:
// ConstructCopy, AssignCopy and SetRigsAndFrames (with the ExpectEqualSerialization helper,
// which compares Write*Text output) and the ReadWrite/ReadAutoDetect round trips.
// ReconstructionTests.cs holds the other helpers and the translation notes. The C++ copy
// constructor and copy assignment are both Clone(); CreateTestDir() is a fresh temp
// directory, deleted when the test ends.

using ColmapSharp.Mathematics;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Scene.ReconstructionIOText;
using static ColmapSharp.Tests.ReconstructionMatchers;

namespace ColmapSharp.Tests.Scene;

public partial class ReconstructionTests
{
	private static string CreateTestDir()
	{
		string dir = Path.Combine(Path.GetTempPath(), "colmapsharp-recon-" + Guid.NewGuid().ToString("N"));
		Directory.CreateDirectory(dir);
		return dir;
	}

	private static string WriteToString(Action<Reconstruction, Stream> write, Reconstruction reconstruction)
	{
		var stream = new MemoryStream();
		write(reconstruction, stream);
		return System.Text.Encoding.UTF8.GetString(stream.ToArray());
	}

	private static async Task ExpectEqualSerialization(Reconstruction reconstruction1, Reconstruction reconstruction2)
	{
		// compare rigs, cameras, frames, images and point3ds
		Action<Reconstruction, Stream>[] writers = [WriteRigsText, WriteCamerasText, WriteFramesText, WriteImagesText, WritePoints3DText];
		foreach (Action<Reconstruction, Stream> write in writers)
		{
			await Assert.That(WriteToString(write, reconstruction1)).IsEqualTo(WriteToString(write, reconstruction2));
		}
	}

	// ConstructCopy, AssignCopy and SetRigsAndFrames' dataset. COLMAP sets num_rigs = 2, then
	// num_rigs = 3; the second assignment wins.
	private static Reconstruction SynthesizeCopyTestDataset(Database? database = null)
	{
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 3,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 8,
			NumPoints3D = 21,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction, database);
		return reconstruction;
	}

	[Test]
	public async Task Reconstruction_ConstructCopy()
	{
		Reconstruction reconstruction = SynthesizeCopyTestDataset();
		Reconstruction reconstructionCopy = reconstruction.Clone();
		await Assert.That(ExplainReconstructionEq(reconstruction, reconstructionCopy)).IsNull();
		await ExpectEqualSerialization(reconstruction, reconstructionCopy);
		await ExpectValidPtrs(reconstruction);
		await ExpectValidPtrs(reconstructionCopy);
	}

	[Test]
	public async Task Reconstruction_AssignCopy()
	{
		Reconstruction reconstruction = SynthesizeCopyTestDataset();
		var reconstructionCopy = new Reconstruction();
		reconstructionCopy = reconstruction.Clone();
		await Assert.That(ExplainReconstructionEq(reconstruction, reconstructionCopy)).IsNull();
		await ExpectEqualSerialization(reconstruction, reconstructionCopy);
		await ExpectValidPtrs(reconstruction);
		await ExpectValidPtrs(reconstructionCopy);
	}

	[Test]
	public async Task Reconstruction_SetRigsAndFrames()
	{
		var database = new InMemoryDatabase();
		Reconstruction reconstruction = SynthesizeCopyTestDataset(database);
		foreach (uint frameId in reconstruction.Frames.Keys.ToList())
		{
			reconstruction.DeRegisterFrame(frameId);
		}

		Reconstruction origReconstruction = reconstruction.Clone();
		reconstruction.SetRigsAndFrames(database.ReadAllRigs(), database.ReadAllFrames());
		await Assert.That(ExplainReconstructionEq(reconstruction, origReconstruction)).IsNull();
		await ExpectEqualSerialization(reconstruction, origReconstruction);
	}

	private static Reconstruction SynthesizeOneCameraDataset(int numFramesPerRig, int numPoints3D)
	{
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = numFramesPerRig,
			NumPoints3D = numPoints3D,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);
		return reconstruction;
	}

	[Test]
	public async Task Reconstruction_ReadWriteTextRoundtrip()
	{
		Reconstruction reconstruction = SynthesizeOneCameraDataset(numFramesPerRig: 3, numPoints3D: 5);

		string testDir = CreateTestDir();
		try
		{
			reconstruction.WriteText(testDir);

			var loaded = new Reconstruction();
			loaded.ReadText(testDir);

			await Assert.That(ExplainReconstructionEq(loaded, reconstruction)).IsNull();
			await ExpectValidPtrs(loaded);
		}
		finally
		{
			Directory.Delete(testDir, true);
		}
	}

	[Test]
	public async Task Reconstruction_ReadWriteBinaryRoundtrip()
	{
		Reconstruction reconstruction = SynthesizeOneCameraDataset(numFramesPerRig: 3, numPoints3D: 5);

		string testDir = CreateTestDir();
		try
		{
			reconstruction.WriteBinary(testDir);

			var loaded = new Reconstruction();
			loaded.ReadBinary(testDir);

			await Assert.That(ExplainReconstructionEq(loaded, reconstruction)).IsNull();
			await ExpectValidPtrs(loaded);
		}
		finally
		{
			Directory.Delete(testDir, true);
		}
	}

	[Test]
	public async Task Reconstruction_ReadAutoDetectFormat()
	{
		Reconstruction reconstruction = SynthesizeOneCameraDataset(numFramesPerRig: 2, numPoints3D: 3);

		// Write binary and verify Read auto-detects binary format
		{
			string testDir = CreateTestDir();
			try
			{
				reconstruction.WriteBinary(testDir);

				var loaded = new Reconstruction();
				loaded.Read(testDir);

				await Assert.That(ExplainReconstructionEq(loaded, reconstruction)).IsNull();
				await ExpectValidPtrs(loaded);
			}
			finally
			{
				Directory.Delete(testDir, true);
			}
		}

		// Write text and verify Read auto-detects text format
		{
			string testDir = CreateTestDir();
			try
			{
				reconstruction.WriteText(testDir);

				var loaded = new Reconstruction();
				loaded.Read(testDir);

				await Assert.That(ExplainReconstructionEq(loaded, reconstruction)).IsNull();
				await ExpectValidPtrs(loaded);
			}
			finally
			{
				Directory.Delete(testDir, true);
			}
		}
	}
}
