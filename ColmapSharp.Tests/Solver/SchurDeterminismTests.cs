// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SchurDeterminismTests (C#-only; no COLMAP counterpart): pins the exact output of the Schur
// solvers (ColmapSharp/Solver/SchurEliminator.cs and its callers) on a realistic bundle
// adjustment, so performance work on them cannot change a single bit. 100 images of the
// synthetic dataset, 10k points with tracks of length 8, 2D noise (40 images and 4k points
// for DENSE_SCHUR and ITERATIVE_SCHUR, to keep the suite fast), solved by COLMAP's
// default bundle adjuster with the linear solver forced. The SHA-256 of every optimized
// parameter's bits, the final cost's bits and the iteration count must equal the recorded
// hash, and must be the same with 1 thread and with every hardware thread.
//
// The hashes were recorded before the eliminator's fixed-size kernels, precomputed cell
// offsets and parallel elimination went in; if one changes, the optimization changed the
// arithmetic, which is the bug.

using System.Security.Cryptography;

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Solver;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Solver;

public class SchurDeterminismTests
{
	private static string SolveAndHash(LinearSolverType type, int numThreads, int numImages, int numPoints, int maxNumIterations)
	{
		// Both runs of a test must see the same dataset.
		RandomUtils.SetPRNGSeed(0);
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(
			new SyntheticDatasetOptions
			{
				NumRigs = numImages,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 1,
				NumPoints3D = numPoints,
				TrackLength = 8,
				NumPoints2DWithoutPoint3D = 0,
				MatchConfig = SyntheticMatchConfig.Chained,
			},
			reconstruction);
		Synthetic.SynthesizeNoise(new SyntheticNoiseOptions { Point2DStddev = 1 }, reconstruction);

		var config = new BundleAdjustmentConfig();
		foreach (uint imageId in reconstruction.Images.Keys.Order())
		{
			config.AddImage(imageId);
		}

		config.FixGauge(BundleAdjustmentGauge.TwoCamsFromWorld);
		var options = new BundleAdjustmentOptions();
		CeresBundleAdjustmentOptions ceres = options.Ceres!;
		ceres.AutoSelectSolverType = false;
		ceres.MinNumResidualsForCpuMultiThreading = 0;
		ceres.SolverOptions.LinearSolverType = type;
		ceres.SolverOptions.PreconditionerType = PreconditionerType.SchurJacobi;
		ceres.SolverOptions.NumThreads = numThreads;
		ceres.SolverOptions.MaxNumIterations = maxNumIterations;
		var summary = (CeresBundleAdjustmentSummary)CeresBundleAdjusters
			.CreateDefaultCeresBundleAdjuster(options, config, reconstruction)
			.Solve();

		var bytes = new List<byte>();
		void Add(double value) => bytes.AddRange(BitConverter.GetBytes(value));
		Add(summary.CeresSummary.FinalCost);
		bytes.AddRange(BitConverter.GetBytes(summary.CeresSummary.Iterations.Count));
		foreach (uint cameraId in reconstruction.Cameras.Keys.Order())
		{
			foreach (double p in reconstruction.Camera(cameraId).Params)
			{
				Add(p);
			}
		}

		foreach (uint imageId in reconstruction.Images.Keys.Order())
		{
			Rigid3d camFromWorld = reconstruction.Image(imageId).CamFromWorld();
			Vector4d q = camFromWorld.Rotation.Coeffs;
			for (int k = 0; k < 4; k++)
			{
				Add(q[k]);
			}

			for (int k = 0; k < 3; k++)
			{
				Add(camFromWorld.Translation[k]);
			}
		}

		foreach (ulong pointId in reconstruction.Points3D.Keys.Order())
		{
			Vector3d xyz = reconstruction.Point3D(pointId).Xyz;
			for (int k = 0; k < 3; k++)
			{
				Add(xyz[k]);
			}
		}

		return Convert.ToHexString(SHA256.HashData(bytes.ToArray()));
	}

	private static async Task CheckHash(LinearSolverType type, int numImages, int numPoints, int maxNumIterations, string expected)
	{
		string oneThread = SolveAndHash(type, 1, numImages, numPoints, maxNumIterations);
		string allThreads = SolveAndHash(type, Math.Max(2, Environment.ProcessorCount), numImages, numPoints, maxNumIterations);
		await Assert.That(allThreads).IsEqualTo(oneThread);
		await Assert.That(oneThread).IsEqualTo(expected);
	}

	[Test]
	public async Task SparseSchur_100Images10kPoints_HashIsPinnedAndThreadIndependent() =>
		await CheckHash(LinearSolverType.SparseSchur, 100, 10000, 100, "0FA585B78EE43ED44CC069D2046C09D263E6D7C01A7A0FD7364E4D90D74FC20B");

	[Test]
	public async Task DenseSchur_HashIsPinnedAndThreadIndependent() =>
		await CheckHash(LinearSolverType.DenseSchur, 40, 4000, 100, "34D5326634F1056F2CBD503ADD20580A522BD125AF10A093821F4C4DF0C3206D");

	[Test]
	public async Task IterativeSchur_HashIsPinnedAndThreadIndependent() =>
		await CheckHash(LinearSolverType.IterativeSchur, 40, 4000, 100, "6C1710945476EE6DA67FAEFD41E1DEC497ACBBD2D502F73C13E6446E9F122595");
}
