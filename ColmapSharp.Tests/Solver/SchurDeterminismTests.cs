// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SchurDeterminismTests (C#-only; no COLMAP counterpart): pins the exact output of the Schur
// solvers (ColmapSharp/Solver/SchurEliminator.cs and its callers) on a bundle adjustment
// shaped like COLMAP's, so performance work on them cannot change a single bit. The
// synthetic dataset with 2D noise, tracks of length 8, one camera per image, solved by
// COLMAP's default bundle adjuster with the linear solver forced: every residual block has
// a point (the E block) and two F blocks, the image pose (quaternion manifold and
// translation, 6 tangent parameters) and its camera's intrinsics. The SHA-256 of every
// optimized parameter's bits, the final cost's bits and the iteration count must equal the
// recorded hash, and must be the same with 1 thread and with every hardware thread.
//
// The problems are small so the suite stays fast; the hashes were recorded on code proven
// bit-identical to the eliminator before its fixed-size kernels, precomputed cell offsets
// and cached inverses (a 100-image, 10k-point run pinned that). If one changes, a
// performance change altered the arithmetic, which is the bug.

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
	public async Task SparseSchur_HashIsPinnedAndThreadIndependent() =>
		await CheckHash(LinearSolverType.SparseSchur, 30, 1500, 100, "98B9A3F979F4F2CEE1307351D125D1D384C62ABC799B6AF7AB1B81BD10B4DF6B");

	[Test]
	public async Task DenseSchur_HashIsPinnedAndThreadIndependent() =>
		await CheckHash(LinearSolverType.DenseSchur, 15, 800, 100, "18DAD521D2CE57ED9D5CE1BEAE6C2788844C93EC8A51AF5A6A11272A1F0E4E9C");

	[Test]
	public async Task IterativeSchur_HashIsPinnedAndThreadIndependent() =>
		await CheckHash(LinearSolverType.IterativeSchur, 15, 800, 100, "3827BAA14E5033B162FB0A775ADE9621364B790F33FA379F52CDA3B66954D62A");
}
