// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// RotationAveragingTests, C#-only half: CSharpOnly_* cases (labeled as such, never standing
// in for a ported case) that drive RotationAveragingProblem*.cs and RotationAveragingSolver.cs
// directly, without RotationEstimator, on the synthetic scenes of rotation_averaging_test.cc
// and with its 1e-2 degree bar. They start from ground truth perturbed by a few degrees in
// place of the spanning-tree initialization. The ported cases and the shared helpers are in
// RotationAveragingTests.cs. Tier C (outcome): relative rotations against ground truth.
//
// COLMAP's gtest_main seeds the PRNG with 0 before every test; tests seed and draw before
// their first await (the PRNG is thread-local).

using ColmapSharp.Estimators;
using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Estimators;

public partial class RotationAveragingTests
{
	private static SyntheticDatasetOptions SingleRigOptions(int numCamerasPerRig, int numFramesPerRig) => new()
	{
		NumRigs = 1,
		NumCamerasPerRig = numCamerasPerRig,
		NumFramesPerRig = numFramesPerRig,
		NumPoints3D = 50,
		SensorFromRigRotationStddev = 20.0,
		PriorGravity = true,
		TwoViewGeometryHasRelativePose = true,
	};

	// Stands in for the spanning-tree initialization: every frame starts at its ground-truth
	// rotation perturbed by maxPerturbationDeg about a random axis.
	private static void InitializePerturbed(TestData data, double maxPerturbationDeg)
	{
		foreach ((uint frameId, Frame gtFrame) in data.GtReconstruction.Frames)
		{
			var axis = new Vector3d(
				RandomUtils.RandomGaussian(0.0, 1.0),
				RandomUtils.RandomGaussian(0.0, 1.0),
				RandomUtils.RandomGaussian(0.0, 1.0)).Normalized();
			double angle = MathUtils.DegToRad(RandomUtils.RandomUniformReal(0.0, maxPerturbationDeg));
			Quaterniond perturbed = Quaterniond.FromAngleAxis(new AngleAxisd(angle, axis)) * gtFrame.RigFromWorld().Rotation;
			data.Reconstruction.Frame(frameId).SetRigFromWorld(
				new Rigid3d(perturbed, new Vector3d(double.NaN, double.NaN, double.NaN)));
		}
	}

	private static HashSet<uint> AllImageIds(Reconstruction reconstruction) => [.. reconstruction.Images.Keys];

	private static bool SolveAndApply(TestData data, RotationEstimatorOptions options, Reconstruction reconstruction)
	{
		var problem = new RotationAveragingProblem(
			data.PoseGraph, data.PosePriors, options, AllImageIds(reconstruction), reconstruction);
		if (!new RotationAveragingSolver(options).Solve(problem))
		{
			return false;
		}

		problem.ApplyResultsToReconstruction(reconstruction);
		return true;
	}

	private static RotationEstimatorOptions Options(bool useGravity) => new()
	{
		UseGravity = useGravity,
		RandomSeed = 0,
	};

	/// <summary>
	/// C#-only: the noise-free single-camera scene of WithoutNoise, with and without gravity
	/// (1-DOF and 3-DOF constraints), recovers every relative rotation within 1e-2 degrees.
	/// </summary>
	[Test]
	[Arguments(true)]
	[Arguments(false)]
	public async Task CSharpOnly_ProblemAndSolverWithoutNoise(bool useGravity)
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SingleRigOptions(numCamerasPerRig: 1, numFramesPerRig: 5));
		InitializePerturbed(data, maxPerturbationDeg: 10);

		bool solved = SolveAndApply(data, Options(useGravity), data.Reconstruction);
		double maxError = MaxRelativeRotationError(data.GtReconstruction, data.Reconstruction);

		await Assert.That(solved).IsTrue();
		await Assert.That(maxError).IsLessThanOrEqualTo(MathUtils.DegToRad(1e-2));
	}

	/// <summary>
	/// C#-only: the two-camera known-rig scene of WithoutNoiseWithNonTrivialKnownRig (known
	/// cam_from_rig folded into the constraints, intra-frame pairs skipped).
	/// </summary>
	[Test]
	[Arguments(true)]
	[Arguments(false)]
	public async Task CSharpOnly_ProblemAndSolverWithKnownRig(bool useGravity)
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SingleRigOptions(numCamerasPerRig: 2, numFramesPerRig: 4));
		InitializePerturbed(data, maxPerturbationDeg: 10);

		bool solved = SolveAndApply(data, Options(useGravity), data.Reconstruction);
		double maxError = MaxRelativeRotationError(data.GtReconstruction, data.Reconstruction);

		await Assert.That(solved).IsTrue();
		await Assert.That(maxError).IsLessThanOrEqualTo(MathUtils.DegToRad(1e-2));
	}

	/// <summary>
	/// C#-only: a cam_from_rig with unknown (NaN) translation, the state COLMAP's
	/// InitializeRigRotationsFromImages leaves behind for WithoutNoiseWithNonTrivialUnknownRig,
	/// makes the non-reference camera's rotation an unknown that is recovered along with the
	/// frames; ApplyResultsToReconstruction writes it back with unknown translation. Like the
	/// frames it starts a few degrees off ground truth.
	/// COLMAP's camera update is a fixed-point iteration (UpdateState averages the camera's
	/// step over its frames, since one shared tangent step cannot be exact for every frame),
	/// and its IRLS stops once the average *frame* step is below the threshold, so with the
	/// default 1e-3 threshold a 5 degree camera error only shrinks to about 0.4 degrees here.
	/// COLMAP's pipeline hands this solve an almost exact cam_from_rig (from the expanded
	/// reconstruction); this test instead tightens the threshold to show the iteration
	/// converges to the ground truth.
	/// </summary>
	[Test]
	public async Task CSharpOnly_ProblemAndSolverEstimateUnknownCamFromRig()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SingleRigOptions(numCamerasPerRig: 2, numFramesPerRig: 4));
		foreach ((uint rigId, Rig rig) in data.Reconstruction.Rigs)
		{
			foreach (SensorId sensorId in rig.NonRefSensors.Keys.ToList())
			{
				Quaterniond gtCamFromRig = data.GtReconstruction.Rig(rigId).SensorFromRig(sensorId).Rotation;
				var axis = new Vector3d(
					RandomUtils.RandomGaussian(0.0, 1.0),
					RandomUtils.RandomGaussian(0.0, 1.0),
					RandomUtils.RandomGaussian(0.0, 1.0)).Normalized();
				Quaterniond perturbed = Quaterniond.FromAngleAxis(new AngleAxisd(MathUtils.DegToRad(5.0), axis)) * gtCamFromRig;
				data.Reconstruction.Rig(rigId).SetSensorFromRig(
					sensorId, new Rigid3d(perturbed, new Vector3d(double.NaN, double.NaN, double.NaN)));
			}
		}

		InitializePerturbed(data, maxPerturbationDeg: 5);
		RotationEstimatorOptions options = Options(useGravity: false);
		options.IrlsStepConvergenceThreshold = 1e-9;
		options.MaxNumIrlsIterations = 1000;
		var problem = new RotationAveragingProblem(
			data.PoseGraph, data.PosePriors, options, AllImageIds(data.Reconstruction), data.Reconstruction);
		int numFrames = data.Reconstruction.NumFrames;
		int numParameters = problem.NumParameters;
		bool solved = new RotationAveragingSolver(options).Solve(problem);
		problem.ApplyResultsToReconstruction(data.Reconstruction);
		double maxError = MaxRelativeRotationError(data.GtReconstruction, data.Reconstruction);
		Rigid3d? camFromRig = data.Reconstruction.Rigs.Values.Single().NonRefSensors.Values.Single();

		// One 3-vector per frame plus one for the non-reference camera.
		await Assert.That(numParameters).IsEqualTo(3 * numFrames + 3);
		await Assert.That(solved).IsTrue();
		await Assert.That(maxError).IsLessThanOrEqualTo(MathUtils.DegToRad(1e-2));
		await Assert.That(camFromRig.HasValue).IsTrue();
		await Assert.That(double.IsNaN(camFromRig!.Value.Translation.X)).IsTrue();
	}

	/// <summary>
	/// C#-only: the linear system's layout. Without gravity every pair gives three rows and
	/// every frame three columns plus a 3-row gauge fix; with gravity priors on every frame
	/// the pairs become 1-DOF rows over one column per frame with a 1-row gauge fix.
	/// </summary>
	[Test]
	public async Task CSharpOnly_ConstraintMatrixLayout()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SingleRigOptions(numCamerasPerRig: 1, numFramesPerRig: 5));
		HashSet<uint> imageIds = AllImageIds(data.Reconstruction);
		int numPairs = data.PoseGraph.NumEdges;
		var problem3Dof = new RotationAveragingProblem(
			data.PoseGraph, data.PosePriors, Options(useGravity: false), imageIds, data.Reconstruction);
		var problem1Dof = new RotationAveragingProblem(
			data.PoseGraph, data.PosePriors, Options(useGravity: true), imageIds, data.Reconstruction);
		bool all1Dof = problem1Dof.PairConstraints.Values.All(c => c.Constraint is RotationAveragingProblem.GravityAligned1Dof);
		bool all3Dof = problem3Dof.PairConstraints.Values.All(c => c.Constraint is RotationAveragingProblem.Full3Dof);

		await Assert.That(numPairs).IsEqualTo(10);
		await Assert.That(problem3Dof.NumResiduals).IsEqualTo(3 * numPairs + 3);
		await Assert.That(problem3Dof.NumParameters).IsEqualTo(15);
		await Assert.That(problem3Dof.NumGaugeFixingResiduals).IsEqualTo(3);
		await Assert.That(all3Dof).IsTrue();
		await Assert.That(problem1Dof.NumResiduals).IsEqualTo(numPairs + 1);
		await Assert.That(problem1Dof.NumParameters).IsEqualTo(5);
		await Assert.That(problem1Dof.NumGaugeFixingResiduals).IsEqualTo(1);
		await Assert.That(all1Dof).IsTrue();
		await Assert.That(problem3Dof.ResidualReweighting).IsNull();
	}

	/// <summary>
	/// C#-only: INLIER_MATCH_COUNT reweighting puts num_matches / max(num_matches) on each
	/// pair's rows and 1 on the gauge rows, scales A's rows by it, and (noise-free, as in
	/// WeightedNoiseFreeMatchesInvariant) solves to the same rotations as uniform weighting.
	/// </summary>
	[Test]
	[Arguments(true)]
	[Arguments(false)]
	public async Task CSharpOnly_InlierMatchCountReweighting(bool useGravity)
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SingleRigOptions(numCamerasPerRig: 1, numFramesPerRig: 5));
		int counter = 1;
		foreach (PoseGraph.Edge edge in data.PoseGraph.Edges.Values)
		{
			edge.NumMatches = 10 * (counter++ % 7) + 1;
		}

		InitializePerturbed(data, maxPerturbationDeg: 10);
		Reconstruction reconUniform = data.Reconstruction.Clone();
		Reconstruction reconWeighted = data.Reconstruction.Clone();

		RotationEstimatorOptions weightedOptions = Options(useGravity);
		weightedOptions.Reweighting = RotationAveragingReweighting.InlierMatchCount;
		var problem = new RotationAveragingProblem(
			data.PoseGraph, data.PosePriors, weightedOptions, AllImageIds(reconWeighted), reconWeighted);

		double maxNumMatches = data.PoseGraph.Edges.Values.Max(e => e.NumMatches);
		VectorXd reweighting = problem.ResidualReweighting!;
		var expected = VectorXd.Ones(problem.NumResiduals);
		foreach ((ulong pairId, RotationAveragingProblem.PairConstraint constraint) in problem.PairConstraints)
		{
			int rows = constraint.Constraint is RotationAveragingProblem.GravityAligned1Dof ? 1 : 3;
			for (int i = 0; i < rows; i++)
			{
				expected[constraint.RowIndex + i] = data.PoseGraph.Edges[pairId].NumMatches / maxNumMatches;
			}
		}

		MatrixXd a = problem.ConstraintMatrix.ToDense();
		MatrixXd weightedA = problem.WeightedConstraintMatrix().ToDense();
		double maxRowScaleError = 0;
		for (int r = 0; r < a.Rows; r++)
		{
			for (int c = 0; c < a.Cols; c++)
			{
				maxRowScaleError = Math.Max(maxRowScaleError, Math.Abs(weightedA[r, c] - expected[r] * a[r, c]));
			}
		}

		bool weightedSolved = new RotationAveragingSolver(weightedOptions).Solve(problem);
		problem.ApplyResultsToReconstruction(reconWeighted);
		bool uniformSolved = SolveAndApply(data, Options(useGravity), reconUniform);
		double weightedError = MaxRelativeRotationError(data.GtReconstruction, reconWeighted);
		double weightedVsUniform = MaxRelativeRotationError(reconUniform, reconWeighted);

		await Assert.That((reweighting - expected).MaxAbs()).IsEqualTo(0.0);
		await Assert.That(maxRowScaleError).IsEqualTo(0.0);
		await Assert.That(weightedSolved).IsTrue();
		await Assert.That(uniformSolved).IsTrue();
		await Assert.That(weightedError).IsLessThanOrEqualTo(MathUtils.DegToRad(1e-2));
		await Assert.That(weightedVsUniform).IsLessThanOrEqualTo(MathUtils.DegToRad(1e-2));
	}

	/// <summary>
	/// C#-only: with a fixed RandomSeed two solves of the same problem are bit-identical
	/// (the 1-DOF jitter reseeds per ComputeResiduals, and every iteration order is fixed).
	/// </summary>
	[Test]
	public async Task CSharpOnly_DeterministicWithSeed()
	{
		RandomUtils.SetPRNGSeed(0);
		TestData data = CreateTestData(SingleRigOptions(numCamerasPerRig: 1, numFramesPerRig: 5));
		InitializePerturbed(data, maxPerturbationDeg: 10);
		Reconstruction recon1 = data.Reconstruction.Clone();
		Reconstruction recon2 = data.Reconstruction.Clone();
		SolveAndApply(data, Options(useGravity: true), recon1);
		SolveAndApply(data, Options(useGravity: true), recon2);

		bool identical = recon1.Frames.All(kv =>
			kv.Value.RigFromWorld().Rotation.Equals(recon2.Frame(kv.Key).RigFromWorld().Rotation));

		await Assert.That(identical).IsTrue();
	}
}
