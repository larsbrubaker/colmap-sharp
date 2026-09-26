// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionPruningTests: colmap/scene/reconstruction_pruning_test.cc, one method per
// gtest case named Suite_Name. Tests ColmapSharp/Scene/ReconstructionPruning.cs.
//
// Not ported yet (they build their input with scene/synthetic's SynthesizeDataset, which is
// not ported): FindRedundantPoints3D.VaryingCoverageGain, VaryingTrackLength,
// VaryingSpatialDistribution. CSharpOnly_SameTilePointIsRedundant exercises the selection
// meanwhile; it does not stand in for them.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Scene;

public class ReconstructionPruningTests
{
	[Test]
	public async Task FindRedundantPoints3D_Empty()
	{
		var reconstruction = new Reconstruction();
		await Assert.That(ReconstructionPruning.FindRedundantPoints3D(minCoverageGain: 0, reconstruction)).IsEmpty();
	}

	/// <summary>
	/// C#-only (until the synthetic cases are ported): two single-observation points. In the
	/// same image tile the second one's gain drops from 1 - 1/sqrt(2) to 1/sqrt(2) - 1/sqrt(3)
	/// once the first is selected; in different tiles both keep the full gain. With equal
	/// gains the larger id is selected first, as in COLMAP's (gain, point3D_id) max-heap.
	/// </summary>
	[Test]
	public async Task CSharpOnly_SameTilePointIsRedundant()
	{
		static Reconstruction Build(Vector2d xy1, Vector2d xy2)
		{
			var reconstruction = new Reconstruction();
			reconstruction.AddCameraWithTrivialRig(Camera.CreateFromModelId(1, CameraModelId.SimplePinhole, 1, 80, 80));
			var image = new Image { ImageId = 1, Name = "image" };
			image.SetCameraId(1);
			image.SetPoints2D([xy1, xy2]);
			reconstruction.AddImageWithTrivialFrame(image);
			var track1 = new Track();
			track1.AddElement(1, 0);
			reconstruction.AddPoint3D(Vector3d.Zero, track1);
			var track2 = new Track();
			track2.AddElement(1, 1);
			reconstruction.AddPoint3D(Vector3d.Zero, track2);
			return reconstruction;
		}

		Reconstruction sameTile = Build(new Vector2d(5, 5), new Vector2d(6, 6));
		Reconstruction differentTiles = Build(new Vector2d(5, 5), new Vector2d(75, 5));
		using (Assert.Multiple())
		{
			await Assert.That(ReconstructionPruning.FindRedundantPoints3D(0.2, sameTile)).IsEquivalentTo(new List<ulong> { 1 });
			await Assert.That(ReconstructionPruning.FindRedundantPoints3D(0.1, sameTile)).IsEmpty();
			await Assert.That(ReconstructionPruning.FindRedundantPoints3D(0.2, differentTiles)).IsEmpty();
			await Assert.That(ReconstructionPruning.FindRedundantPoints3D(0.3, differentTiles)).IsEquivalentTo(new List<ulong> { 1, 2 });
		}
	}
}
