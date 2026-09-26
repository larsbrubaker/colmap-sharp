// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionTests.Synthetic: the reconstruction_test.cc cases that build their scene with
// SynthesizeDataset (Scene/Synthetic.cs) and do not serialize, 1:1. ReconstructionTests.cs
// holds the helpers and translation notes; ReconstructionTests.IO.cs the synthetic cases
// that go through reconstruction_io (copies, SetRigsAndFrames, read/write round trips).

using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public partial class ReconstructionTests
{
	[Test]
	public async Task Reconstruction_Print()
	{
		RandomUtils.SetPRNGSeed(0);
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 2,
			NumPoints3D = 3,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);
		await Assert.That(reconstruction.ToString()).IsEqualTo(
			"Reconstruction(num_rigs=1, num_cameras=1, num_frames=2, "
			+ "num_reg_frames=2, num_images=2, num_points3D=3)");
	}

	[Test]
	public async Task Reconstruction_SetRigsAndFramesResetsNumRegImages()
	{
		RandomUtils.SetPRNGSeed(0);
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 4,
			NumPoints3D = 0,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);
		int numRegImagesBefore = reconstruction.NumRegImages;
		await Assert.That(numRegImagesBefore).IsGreaterThan(0);
		// Copy rigs and frames (with poses) from the reconstruction to re-apply.
		var rigs = new List<Rig>();
		foreach (Rig rig in reconstruction.Rigs.Values)
		{
			rigs.Add(rig.Clone());
		}

		var frames = new List<Frame>();
		foreach (Frame reconstructionFrame in reconstruction.Frames.Values)
		{
			Frame frame = reconstructionFrame.Clone();
			frame.ResetRigPtr();
			frames.Add(frame);
		}

		int numRigsBefore = reconstruction.NumRigs;
		int numFramesBefore = reconstruction.NumFrames;
		int numRegFramesBefore = reconstruction.NumRegFrames;
		// Call SetRigsAndFrames while frames are still registered. Previously this would
		// double-count num_reg_images_ because it was not reset to zero.
		reconstruction.SetRigsAndFrames(rigs, frames);
		// Verify num_reg_images_ is not double-counted.
		await Assert.That(reconstruction.NumRegImages).IsEqualTo(numRegImagesBefore);
		// Verify rigs, frames, and registered frames are preserved.
		await Assert.That(reconstruction.NumRigs).IsEqualTo(numRigsBefore);
		await Assert.That(reconstruction.NumFrames).IsEqualTo(numFramesBefore);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(numRegFramesBefore);
		// Verify every registered frame still has a pose.
		foreach (uint frameId in reconstruction.RegFrameIds)
		{
			await Assert.That(reconstruction.Frame(frameId).HasPose).IsTrue();
		}

		// Verify image-to-frame pointers are correctly re-wired.
		foreach (Image image in reconstruction.Images.Values)
		{
			await Assert.That(image.HasFrameId).IsTrue();
			await Assert.That(image.HasFramePtr).IsTrue();
			await Assert.That(ReferenceEquals(image.FramePtr, reconstruction.Frame(image.FrameId))).IsTrue();
		}
	}

	[Test]
	public async Task Reconstruction_DeleteAllPoints2DAndPoints3D()
	{
		RandomUtils.SetPRNGSeed(0);
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 20,
			NumPoints3D = 50,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);
		reconstruction.DeleteAllPoints2DAndPoints3D();
		await Assert.That(reconstruction.NumPoints3D).IsEqualTo(0);
		await ExpectValidPtrs(reconstruction);
	}

	[Test]
	public async Task Reconstruction_TearDown()
	{
		RandomUtils.SetPRNGSeed(0);
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 3,
			NumPoints3D = 10,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);

		// De-register one frame to create an unregistered frame
		int regFramesBefore = reconstruction.NumRegFrames;
		List<uint> frameIds = [.. reconstruction.RegFrameIds];
		reconstruction.DeRegisterFrame(frameIds[0]);
		await Assert.That(reconstruction.NumRegFrames).IsEqualTo(regFramesBefore - 1);

		// TearDown should remove the unregistered frame and its images
		int numFramesBefore = reconstruction.NumFrames;
		reconstruction.TearDown();
		await Assert.That(reconstruction.NumFrames).IsLessThan(numFramesBefore);
		await Assert.That(reconstruction.NumFrames).IsEqualTo(reconstruction.NumRegFrames);
		await ExpectValidPtrs(reconstruction);
	}
}
