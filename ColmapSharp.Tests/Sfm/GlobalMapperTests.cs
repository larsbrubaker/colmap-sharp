// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// GlobalMapperTests: colmap/sfm/global_mapper_test.cc ported 1:1, one method per gtest TEST
// named Suite_Name, testing ColmapSharp/Sfm/GlobalMapper*.cs and GlobalMapperOptions.cs.
//
// Tier C (outcome): the reconstruction is compared with the synthetic ground truth through
// ReconstructionNear at COLMAP's bounds.
//
// Translation notes: the SQLite database file is InMemoryDatabase. COLMAP's gtest_main seeds
// the PRNG with 0 before every test; the PRNG is per thread, so each test seeds it and does
// all of its mapping before its first await.

using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Sfm;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Sfm;

public class GlobalMapperTests
{
	// Synthesizes the dataset (and noise), runs GlobalMapper.Solve with default options and
	// returns COLMAP's ReconstructionNear explanation (null on a match).
	private static string? SolveAndCompare(
		SyntheticDatasetOptions datasetOptions,
		SyntheticNoiseOptions? noiseOptions,
		bool resetSensorsFromRig,
		double maxRotationErrorDeg,
		double maxProjCenterError,
		double numObsTolerance = 0.0)
	{
		RandomUtils.SetPRNGSeed(0);
		var gtReconstruction = new Reconstruction();
		using var database = new InMemoryDatabase();
		Synthetic.SynthesizeDataset(datasetOptions, gtReconstruction, database);
		if (noiseOptions is not null)
		{
			Synthetic.SynthesizeNoise(noiseOptions, gtReconstruction, database);
		}

		var reconstruction = new Reconstruction();
		var globalMapper = new GlobalMapper(DatabaseCache.Create(database, new DatabaseCache.Options()));
		globalMapper.BeginReconstruction(reconstruction);

		if (resetSensorsFromRig)
		{
			// Set the rig sensors to be unknown
			foreach ((uint rigId, Rig rig) in reconstruction.Rigs)
			{
				foreach ((SensorId sensorId, var sensor) in rig.NonRefSensors.ToList())
				{
					if (sensor is not null)
					{
						reconstruction.Rig(rigId).ResetSensorFromRig(sensorId);
					}
				}
			}
		}

		globalMapper.Solve(new GlobalMapperOptions());

		return ReconstructionMatchers.ExplainReconstructionNear(
			gtReconstruction,
			reconstruction,
			maxRotationErrorDeg,
			maxProjCenterError,
			maxScaleError: null,
			numObsTolerance);
	}

	[Test]
	public async Task GlobalMapper_WithoutNoise()
	{
		string? mismatch = SolveAndCompare(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 7,
				NumPoints3D = 50,
				TwoViewGeometryHasRelativePose = true,
			},
			noiseOptions: null,
			resetSensorsFromRig: false,
			maxRotationErrorDeg: 1e-2,
			maxProjCenterError: 1e-4);

		await Assert.That(mismatch).IsNull();
	}

	[Test]
	public async Task GlobalMapper_WithoutNoiseWithNonTrivialKnownRig()
	{
		string? mismatch = SolveAndCompare(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 2,
				NumFramesPerRig = 7,
				NumPoints3D = 50,
				SensorFromRigTranslationStddev = 0.1, // No noise
				SensorFromRigRotationStddev = 5.0, // No noise
				TwoViewGeometryHasRelativePose = true,
			},
			noiseOptions: null,
			resetSensorsFromRig: false,
			maxRotationErrorDeg: 1e-2,
			maxProjCenterError: 1e-4);

		await Assert.That(mismatch).IsNull();
	}

	[Test]
	public async Task GlobalMapper_WithoutNoiseWithNonTrivialUnknownRig()
	{
		string? mismatch = SolveAndCompare(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 3,
				NumFramesPerRig = 7,
				NumPoints3D = 50,
				SensorFromRigTranslationStddev = 0.1, // No noise
				SensorFromRigRotationStddev = 5.0, // No noise
				TwoViewGeometryHasRelativePose = true,
			},
			noiseOptions: null,
			resetSensorsFromRig: true,
			maxRotationErrorDeg: 1e-2,
			maxProjCenterError: 1e-4);

		await Assert.That(mismatch).IsNull();
	}

	[Test]
	public async Task GlobalMapper_WithNoiseAndOutliers()
	{
		string? mismatch = SolveAndCompare(
			new SyntheticDatasetOptions
			{
				NumRigs = 2,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 4,
				NumPoints3D = 100,
				InlierMatchRatio = 0.7,
				TwoViewGeometryHasRelativePose = true,
			},
			new SyntheticNoiseOptions { Point2DStddev = 0.5 },
			resetSensorsFromRig: false,
			maxRotationErrorDeg: 1e-1,
			maxProjCenterError: 1e-1,
			numObsTolerance: 0.02);

		await Assert.That(mismatch).IsNull();
	}

	[Test]
	public async Task GlobalMapperOptions_RefineSensorFromRigPropagatesToSubOptions()
	{
		var options = new GlobalMapperOptions { RefineSensorFromRig = false };
		// Sub-options keep their own defaults (true) until accessed.
		await Assert.That(options.RotationAveragingOptions.RefineSensorFromRig).IsTrue();
		await Assert.That(options.GlobalPositioningOptions.RefineSensorFromRig).IsTrue();
		await Assert.That(options.BundleAdjustmentOptions.RefineSensorFromRig).IsTrue();
		// Accessors return resolved sub-options with the top-level flag applied.
		await Assert.That(options.RotationAveraging().RefineSensorFromRig).IsFalse();
		await Assert.That(options.GlobalPositioning().RefineSensorFromRig).IsFalse();
		await Assert.That(options.BundleAdjustment().RefineSensorFromRig).IsFalse();
	}

	// Rotation averaging, then track establishment with the given options; the number of
	// established tracks.
	private static int EstablishTracksCount(GlobalMapperOptions options)
	{
		RandomUtils.SetPRNGSeed(0);
		using var database = new InMemoryDatabase();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = 1,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 5,
				NumPoints3D = 50,
				TwoViewGeometryHasRelativePose = true,
			},
			new Reconstruction(),
			database);
		var reconstruction = new Reconstruction();
		var globalMapper = new GlobalMapper(DatabaseCache.Create(database, new DatabaseCache.Options()));
		globalMapper.BeginReconstruction(reconstruction);
		Require(globalMapper.RotationAveraging(options.RotationAveraging()));
		globalMapper.EstablishTracks(options);
		return reconstruction.NumPoints3D;
	}

	private static void Require(bool condition)
	{
		if (!condition)
		{
			throw new InvalidOperationException("Test precondition failed");
		}
	}

	// C#-only: C++ compares the track options through static_cast<size_t>, so -1 wraps to the
	// largest size: a minimum of -1 views per track keeps no track, and -1 required tracks per
	// view never stops adding tracks.
	[Test]
	public async Task CSharpOnly_EstablishTracksNegativeOptionsWrapLikeSizeT()
	{
		int numDefault = EstablishTracksCount(new GlobalMapperOptions());
		int numMinViewsNegative = EstablishTracksCount(new GlobalMapperOptions { TrackMinNumViewsPerTrack = -1 });
		int numRequiredNegative = EstablishTracksCount(new GlobalMapperOptions { TrackRequiredTracksPerView = -1 });

		await Assert.That(numDefault).IsGreaterThan(0);
		await Assert.That(numMinViewsNegative).IsEqualTo(0);
		await Assert.That(numRequiredNegative).IsEqualTo(numDefault);
	}
}
