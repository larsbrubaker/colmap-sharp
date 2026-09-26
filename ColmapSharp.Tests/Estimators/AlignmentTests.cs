// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AlignmentTests: colmap/estimators/alignment_test.cc ported 1:1, one method per gtest
// TEST(Suite, Name) named Suite_Name, same checks and tolerances. Tests
// ColmapSharp/Estimators/Alignment*.cs. The robust alignments are Tier C (LO-RANSAC on
// noise-free synthetic data, so COLMAP's 1e-6 bars apply to the recovered Sim3d); the error
// summary is Tier B.
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test, so each test starts with
// RandomUtils.SetPRNGSeed(0) and does everything that draws from the PRNG (synthesis, the
// random Sim3d, RANSAC with random_seed = -1) before its first await, since the PRNG is per
// thread and an await may resume elsewhere.

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Optim;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ColmapSharp.Tests.Rigid3dMatchers;

namespace ColmapSharp.Tests.Estimators;

public class AlignmentTests
{
	private static Sim3d TestSim3d()
	{
		// C++ evaluates Sim3d's arguments left to right under clang, as C# does.
		double scale = RandomUtils.RandomUniformReal(0.5, 2.0);
		Quaterniond rotation = RandomEigen.RandomEigenQuaterniond();
		Vector3d translation = RandomEigen.RandomEigenVector3d();
		return new Sim3d(scale, rotation, translation);
	}

	private static async Task ExpectEqualSim3d(Sim3d gtTgtFromSrc, Sim3d tgtFromSrc)
	{
		using (Assert.Multiple())
		{
			await Assert.That(tgtFromSrc.Scale).IsEqualTo(gtTgtFromSrc.Scale).Within(1e-6);
			await Assert.That(gtTgtFromSrc.Rotation.AngularDistance(tgtFromSrc.Rotation)).IsLessThan(1e-6);
			await Assert.That((gtTgtFromSrc.Translation - tgtFromSrc.Translation).Norm).IsLessThan(1e-6);
		}
	}

	private static Reconstruction GenerateReconstructionForAlignment()
	{
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 10,
			NumPoints3D = 50,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);
		return reconstruction;
	}

	// The source and a target moved by a random Sim3d, as every alignment test starts.
	private static (Reconstruction Src, Reconstruction Tgt, Sim3d GtTgtFromSrc) GenerateAlignedPair()
	{
		Reconstruction srcReconstruction = GenerateReconstructionForAlignment();
		Reconstruction tgtReconstruction = srcReconstruction.Clone();

		Sim3d gtTgtFromSrc = TestSim3d();
		tgtReconstruction.Transform(gtTgtFromSrc);
		return (srcReconstruction, tgtReconstruction, gtTgtFromSrc);
	}

	[Test]
	public async Task Alignment_AlignReconstructionToLocations()
	{
		RandomUtils.SetPRNGSeed(0);
		var (srcReconstruction, tgtReconstruction, gtTgtFromSrc) = GenerateAlignedPair();

		var tgtImageNames = new List<string>();
		var tgtImageLocations = new List<Vector3d>();
		foreach (Image image in tgtReconstruction.Images.Values)
		{
			tgtImageNames.Add(image.Name);
			tgtImageLocations.Add(image.ProjectionCenter());
		}

		var ransacOptions = new RansacOptions { MaxError = 1e-2 };

		var tgtFromSrc = new Sim3d();
		bool tooFewCommon = Alignment.AlignReconstructionToLocations(
			srcReconstruction,
			tgtImageNames,
			tgtImageLocations,
			minCommonImages: tgtImageNames.Count + 1,
			ransacOptions,
			ref tgtFromSrc);
		bool aligned = Alignment.AlignReconstructionToLocations(
			srcReconstruction,
			tgtImageNames,
			tgtImageLocations,
			minCommonImages: 3,
			ransacOptions,
			ref tgtFromSrc);

		await Assert.That(tooFewCommon).IsFalse();
		await Assert.That(aligned).IsTrue();
		await ExpectEqualSim3d(gtTgtFromSrc, tgtFromSrc);
	}

	private static List<PosePrior> PosePriorsAtProjectionCenters(Reconstruction tgtReconstruction, double covarianceScale)
	{
		var tgtPosePriors = new List<PosePrior>();
		foreach (Image image in tgtReconstruction.Images.Values)
		{
			tgtPosePriors.Add(new PosePrior
			{
				PosePriorId = (uint)(tgtPosePriors.Count + 1),
				CorrDataId = image.DataId,
				CoordinateSystem = PosePriorCoordinateSystem.Cartesian,
				Position = image.ProjectionCenter(),
				PositionCovariance = covarianceScale * Matrix3d.Identity,
			});
		}

		return tgtPosePriors;
	}

	[Test]
	public async Task Alignment_AlignReconstructionToPosePriors()
	{
		RandomUtils.SetPRNGSeed(0);
		var (srcReconstruction, tgtReconstruction, gtTgtFromSrc) = GenerateAlignedPair();

		List<PosePrior> tgtPosePriors = PosePriorsAtProjectionCenters(tgtReconstruction, 1e-2);

		var ransacOptions = new RansacOptions { MaxError = 1e-2 };

		var tgtFromSrc = new Sim3d();
		bool aligned = Alignment.AlignReconstructionToPosePriors(
			srcReconstruction,
			tgtPosePriors,
			ransacOptions,
			priorPositionFallbackStddev: 1.0,
			ref tgtFromSrc);

		await Assert.That(aligned).IsTrue();
		await ExpectEqualSim3d(gtTgtFromSrc, tgtFromSrc);
	}

	[Test]
	public async Task Alignment_AlignReconstructionToPosePriorsWithAutomaticMaxError()
	{
		RandomUtils.SetPRNGSeed(0);
		var (srcReconstruction, tgtReconstruction, gtTgtFromSrc) = GenerateAlignedPair();

		List<PosePrior> tgtPosePriors = PosePriorsAtProjectionCenters(tgtReconstruction, 1e-4);

		var ransacOptions = new RansacOptions { MaxError = 0.0 };
		var tgtFromSrc = new Sim3d();
		bool aligned = Alignment.AlignReconstructionToPosePriors(
			srcReconstruction,
			tgtPosePriors,
			ransacOptions,
			priorPositionFallbackStddev: 1.0,
			ref tgtFromSrc);

		await Assert.That(aligned).IsTrue();
		await ExpectEqualSim3d(gtTgtFromSrc, tgtFromSrc);
	}

	[Test]
	public async Task Alignment_AlignReconstructionsViaReprojections()
	{
		RandomUtils.SetPRNGSeed(0);
		var (srcReconstruction, tgtReconstruction, gtTgtFromSrc) = GenerateAlignedPair();

		var tgtFromSrc = new Sim3d();
		bool aligned = Alignment.AlignReconstructionsViaReprojections(
			srcReconstruction,
			tgtReconstruction,
			minInlierObservations: 0.9,
			maxReprojError: 2,
			ref tgtFromSrc);

		await Assert.That(aligned).IsTrue();
		await ExpectEqualSim3d(gtTgtFromSrc, tgtFromSrc);
	}

	[Test]
	public async Task Alignment_AlignReconstructionsViaProjCenters()
	{
		RandomUtils.SetPRNGSeed(0);
		var (srcReconstruction, tgtReconstruction, gtTgtFromSrc) = GenerateAlignedPair();

		var tgtFromSrc = new Sim3d();
		bool aligned = Alignment.AlignReconstructionsViaProjCenters(
			srcReconstruction,
			tgtReconstruction,
			maxProjCenterError: 0.1,
			ref tgtFromSrc);

		await Assert.That(aligned).IsTrue();
		await ExpectEqualSim3d(gtTgtFromSrc, tgtFromSrc);
	}

	[Test]
	public async Task Alignment_AlignReconstructionsViaPoints()
	{
		RandomUtils.SetPRNGSeed(0);
		var (srcReconstruction, tgtReconstruction, gtTgtFromSrc) = GenerateAlignedPair();

		var tgtFromSrc = new Sim3d();
		bool aligned = Alignment.AlignReconstructionsViaPoints(
			srcReconstruction,
			tgtReconstruction,
			minCommonObservations: 3,
			maxError: 0.01,
			minInlierRatio: 0.9,
			ref tgtFromSrc);

		await Assert.That(aligned).IsTrue();
		await ExpectEqualSim3d(gtTgtFromSrc, tgtFromSrc);
	}

	// Synthesize a reconstruction which has at least two cameras (3 rigs of 1 camera).
	private static Reconstruction GenerateReconstructionForMerge()
	{
		var srcReconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 3,
			NumCamerasPerRig = 1,
			NumFramesPerRig = 10,
			NumPoints3D = 50,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, srcReconstruction);
		return srcReconstruction;
	}

	private static void RemoveRigFrames(Reconstruction reconstruction, uint rigId)
	{
		List<uint> frameIds = [.. reconstruction.RegFrameIds];
		foreach (uint frameId in frameIds)
		{
			if (reconstruction.Frame(frameId).RigId == rigId)
			{
				reconstruction.DeRegisterFrame(frameId);
			}
		}
	}

	[Test]
	public async Task Alignment_MergeReconstructions()
	{
		RandomUtils.SetPRNGSeed(0);
		Reconstruction srcReconstruction = GenerateReconstructionForMerge();
		Reconstruction origReconstruction = srcReconstruction.Clone();
		Reconstruction tgtReconstruction = srcReconstruction.Clone();

		RemoveRigFrames(srcReconstruction, 1);
		RemoveRigFrames(tgtReconstruction, 2);

		// Remove all unregistered rigs/cameras/frames/images.
		srcReconstruction.TearDown();
		tgtReconstruction.TearDown();
		int[] srcCounts = [srcReconstruction.NumRigs, srcReconstruction.NumCameras, srcReconstruction.NumFrames, srcReconstruction.NumRegFrames, srcReconstruction.NumImages];
		int[] tgtCounts = [tgtReconstruction.NumRigs, tgtReconstruction.NumCameras, tgtReconstruction.NumFrames, tgtReconstruction.NumRegFrames, tgtReconstruction.NumImages];

		// Merge reconstructions.
		bool merged = Alignment.MergeReconstructions(maxReprojError: 1e-4, srcReconstruction, tgtReconstruction);

		// NumRigs, NumCameras, NumFrames, NumRegFrames, NumImages.
		int[] expectedCounts = [2, 2, 20, 20, 20];
		using (Assert.Multiple())
		{
			for (int i = 0; i < expectedCounts.Length; ++i)
			{
				await Assert.That(srcCounts[i]).IsEqualTo(expectedCounts[i]);
				await Assert.That(tgtCounts[i]).IsEqualTo(expectedCounts[i]);
			}
		}

		await Assert.That(merged).IsTrue();
		using (Assert.Multiple())
		{
			await Assert.That(tgtReconstruction.NumRigs).IsEqualTo(3);
			await Assert.That(tgtReconstruction.NumCameras).IsEqualTo(3);
			await Assert.That(tgtReconstruction.NumFrames).IsEqualTo(30);
			await Assert.That(tgtReconstruction.NumRegFrames).IsEqualTo(30);
			await Assert.That(tgtReconstruction.NumImages).IsEqualTo(30);
			await Assert.That(tgtReconstruction.NumPoints3D).IsEqualTo(50);
			await Assert.That(tgtReconstruction.ComputeNumObservations()).IsEqualTo(origReconstruction.ComputeNumObservations());
		}
	}

	[Test]
	public async Task Alignment_MergeReconstructionsInconsistentImageNames()
	{
		// Reconstructions built from independent databases can assign the same image id to
		// distinct physical images. Merging by id would then silently drop images and
		// corrupt tracks, so such an inconsistent id<->name mapping must be detected and
		// rejected rather than merged (see issue #3405).
		RandomUtils.SetPRNGSeed(0);
		Reconstruction srcReconstruction = GenerateReconstructionForMerge();
		Reconstruction tgtReconstruction = srcReconstruction.Clone();

		RemoveRigFrames(srcReconstruction, 1);
		RemoveRigFrames(tgtReconstruction, 2);
		srcReconstruction.TearDown();
		tgtReconstruction.TearDown();

		// Find an image id registered in both reconstructions and give it a different name
		// in the target, simulating an id collision between two distinct images. Enough
		// images still share a consistent name for the alignment step (which matches by
		// name) to succeed, so the merge reaches -- and must fail at -- the id/name
		// consistency check.
		bool foundSharedId = false;
		foreach (uint imageId in srcReconstruction.RegImageIds())
		{
			if (tgtReconstruction.ExistsImage(imageId))
			{
				tgtReconstruction.Image(imageId).Name = "colliding_name.jpg";
				foundSharedId = true;
				break;
			}
		}

		int numTgtImagesBeforeMerge = tgtReconstruction.NumImages;
		bool merged = foundSharedId && Alignment.MergeReconstructions(maxReprojError: 1e-4, srcReconstruction, tgtReconstruction);

		await Assert.That(foundSharedId).IsTrue();
		await Assert.That(merged).IsFalse();
		// The merge must abort before mutating the target reconstruction.
		await Assert.That(tgtReconstruction.NumImages).IsEqualTo(numTgtImagesBeforeMerge);
	}

	[Test]
	public async Task Alignment_AlignReconstructionToOrigRigScales()
	{
		RandomUtils.SetPRNGSeed(0);
		var reconstruction = new Reconstruction();
		var syntheticDatasetOptions = new SyntheticDatasetOptions
		{
			NumRigs = 2,
			NumCamerasPerRig = 4,
			NumFramesPerRig = 10,
			NumPoints3D = 50,
		};
		Synthetic.SynthesizeDataset(syntheticDatasetOptions, reconstruction);
		Dictionary<uint, Rig> origRigs = reconstruction.Rigs.ToDictionary(entry => entry.Key, entry => entry.Value.Clone());

		reconstruction.Transform(TestSim3d());
		Alignment.AlignReconstructionToOrigRigScales(origRigs, reconstruction);

		using (Assert.Multiple())
		{
			foreach (var (rigId, origRig) in origRigs)
			{
				foreach (var (sensorId, sensorFromOrigRig) in origRig.NonRefSensors)
				{
					if (sensorFromOrigRig is not Rigid3d sensorFromOrigRigValue)
					{
						continue;
					}

					await Assert.That(Rigid3dNear(
						reconstruction.Rig(rigId).SensorFromRig(sensorId),
						sensorFromOrigRigValue,
						rtol: 1e-6,
						ttol: 1e-6)).IsTrue();
				}
			}
		}
	}

	[Test]
	public async Task AlignmentErrorSummary_Empty()
	{
		var errors = new List<ImageAlignmentError>();
		AlignmentErrorSummary summary = AlignmentErrorSummary.Compute(errors);
		await Assert.That(summary.RotationErrorsDeg.Min).IsEqualTo(0);
		await Assert.That(summary.ProjCenterErrors.Min).IsEqualTo(0);
	}

	[Test]
	public async Task AlignmentErrorSummary_MultipleErrors()
	{
		var errors = new List<ImageAlignmentError>();
		for (int i = 0; i < 5; ++i)
		{
			errors.Add(new ImageAlignmentError
			{
				RotationErrorDeg = i + 1,
				ProjCenterError = (i + 1) * 0.1,
			});
		}

		AlignmentErrorSummary summary = AlignmentErrorSummary.Compute(errors);
		using (Assert.Multiple())
		{
			await Assert.That(summary.RotationErrorsDeg.Min).IsEqualTo(1.0).Within(1e-10);
			await Assert.That(summary.RotationErrorsDeg.Max).IsEqualTo(5.0).Within(1e-10);
			await Assert.That(summary.RotationErrorsDeg.Mean).IsEqualTo(3.0).Within(1e-10);
			await Assert.That(summary.RotationErrorsDeg.Median).IsEqualTo(3.0).Within(1e-10);
			await Assert.That(summary.RotationErrorsDeg.P90).IsEqualTo(4.6).Within(1e-10);
			await Assert.That(summary.RotationErrorsDeg.P99).IsEqualTo(4.96).Within(1e-10);
			await Assert.That(summary.ProjCenterErrors.Min).IsEqualTo(0.1).Within(1e-10);
			await Assert.That(summary.ProjCenterErrors.Max).IsEqualTo(0.5).Within(1e-10);
			await Assert.That(summary.ProjCenterErrors.Mean).IsEqualTo(0.3).Within(1e-10);
			await Assert.That(summary.ProjCenterErrors.P90).IsEqualTo(0.46).Within(1e-10);
			await Assert.That(summary.ProjCenterErrors.P99).IsEqualTo(0.496).Within(1e-10);
		}
	}
}
