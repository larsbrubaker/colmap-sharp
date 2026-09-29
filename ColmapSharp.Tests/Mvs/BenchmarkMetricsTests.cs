// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BenchmarkMetricsTests: C#-only tests (not ports; COLMAP has no reconstruction benchmark).
// They pin the metric pieces of ColmapSharp/Mvs/Testing/Benchmark on inputs with known answers:
// an identical surface scores zero distance and F = 1, a surface shifted by a known offset gives
// exactly that distance (and F flips at tau), background far outside the object is excluded
// while junk near it counts, a known similarity and a known rotation error are recovered by the
// pose Sim3, identical masks give IoU 1, and a rendered silhouette of the truth mesh agrees with
// the scene's own mask. The end-to-end reconstruction check is in BenchmarkEndToEndTests.

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Mvs.Testing.Benchmark;
using ColmapSharp.Sensor;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class BenchmarkMetricsTests
{
	private static readonly SurfaceMetricOptions Options = new() { SampleCount = 4000 };

	[Test]
	public async Task CSharpOnly_IdenticalMeshScoresZeroErrorAndFullF()
	{
		BenchmarkMesh cube = BenchmarkMeshTestShapes.Cube(0.5);
		SurfaceScores scores = SurfaceMetrics.Compute(cube, cube, null, Options);

		// Samples lie on the mesh up to the BVH's float storage.
		await Assert.That(scores.AccuracyPct).IsLessThan(1e-5);
		await Assert.That(scores.CompletenessPct).IsLessThan(1e-5);
		await Assert.That(scores.Precision).IsEqualTo(1.0);
		await Assert.That(scores.Recall).IsEqualTo(1.0);
		await Assert.That(scores.FScore).IsEqualTo(1.0);
		await Assert.That(scores.ExcludedFraction).IsEqualTo(0.0);
		await Assert.That(scores.TauPct).IsEqualTo(1.0);
	}

	[Test]
	public async Task CSharpOnly_ShiftedSurfaceGivesTheShiftAsDistance()
	{
		// Two parallel squares: every point of one is exactly `shift` from the other.
		BenchmarkMesh truth = BenchmarkMeshTestShapes.Square(1, 0);
		double diagonal = Math.Sqrt(2);
		foreach ((double shiftInTau, double expectedF) in new[] { (0.5, 1.0), (2.0, 0.0) })
		{
			double shift = shiftInTau * Options.TauFraction * diagonal;
			SurfaceScores scores = SurfaceMetrics.Compute(truth, BenchmarkMeshTestShapes.Square(1, shift), null, Options);
			double expectedPct = 100 * shift / diagonal;

			// Up to the BVH's float storage of the shifted square.
			await Assert.That(Math.Abs(scores.AccuracyPct - expectedPct)).IsLessThan(1e-6);
			await Assert.That(Math.Abs(scores.CompletenessPct - expectedPct)).IsLessThan(1e-6);
			await Assert.That(scores.FScore).IsEqualTo(expectedF);
		}
	}

	[Test]
	public async Task CSharpOnly_PointCloudScoresAndBackgroundExclusion()
	{
		BenchmarkMesh cube = BenchmarkMeshTestShapes.Cube(0.5);
		double diagonal = Math.Sqrt(3);
		var points = new List<Vector3d>(cube.SampleSurface(3000, 9))
		{
			// Wall: far outside the grown box, excluded.
			new(0, 0, 5),
			new(3, 3, 3),
		};

		SurfaceScores clean = SurfaceMetrics.Compute(cube, null, points, Options);
		await Assert.That(clean.AccuracyPct).IsLessThan(1e-5);
		await Assert.That(clean.Precision).IsEqualTo(1.0);
		await Assert.That(clean.ExcludedFraction).IsEqualTo(2.0 / 3002);

		// Junk just outside the object but inside the grown box counts against accuracy: one
		// point 0.05 diagonals off the +z face.
		points.Add(new Vector3d(0, 0, 0.5 + 0.05 * diagonal));
		SurfaceScores junk = SurfaceMetrics.Compute(cube, null, points, Options);
		await Assert.That(junk.Precision).IsEqualTo(3000.0 / 3001);
		await Assert.That(Math.Abs(junk.AccuracyPct - 5.0 / 3001)).IsLessThan(1e-5);

		// Recall uses the nearest point: 3000 points on the cube are about 0.045 apart, wider
		// than tau (0.017), so much of the truth is not recalled; a cloud 20 times as dense
		// (about 0.01 apart) recalls nearly all of it.
		await Assert.That(clean.Recall).IsLessThan(0.5);
		SurfaceScores dense = SurfaceMetrics.Compute(cube, null, cube.SampleSurface(60000, 9), Options);
		await Assert.That(dense.Recall).IsGreaterThan(0.95);
	}

	[Test]
	public async Task CSharpOnly_KdTreeNearestDistanceMatchesBruteForce()
	{
		var random = new Mt19937(11);
		var cloud = new Vector3d[3000];
		for (int i = 0; i < cloud.Length; i++)
		{
			// Many duplicate x values so ties in the splits are exercised.
			cloud[i] = new Vector3d(random.Next() % 16, random.Next() / 4294967296.0, random.Next() / 4294967296.0);
		}

		var tree = new PointKdTree(cloud);
		double worst = 0;
		for (int q = 0; q < 300; q++)
		{
			var query = new Vector3d(random.Next() / 4294967296.0 * 18 - 1, random.Next() / 4294967296.0, random.Next() / 4294967296.0 * 2 - 0.5);
			double brute = cloud.Min(p => (p - query).Norm);
			worst = Math.Max(worst, Math.Abs(tree.NearestDistance(query) - brute));
		}

		await Assert.That(worst).IsEqualTo(0.0);
		await Assert.That(double.IsPositiveInfinity(new PointKdTree([]).NearestDistance(Vector3d.Zero))).IsTrue();
	}

	// An object reconstructed 1.3 times too large must lose its precision, not have its
	// oversized parts excluded as background. The object is elongated like the mouse, so its
	// long axis grows by 15% of its length: more than a 10%-of-diagonal box margin, which is why
	// the margin is 25%.
	[Test]
	public async Task CSharpOnly_OversizedReconstructionLosesPrecision()
	{
		BenchmarkMesh truth = BenchmarkMeshTestShapes.Box(0.5, 0.2, 0.15);
		BenchmarkMesh oversized = BenchmarkMeshTestShapes.Box(0.65, 0.26, 0.195);
		SurfaceScores scores = SurfaceMetrics.Compute(truth, oversized, null, Options);
		await Assert.That(scores.ExcludedFraction).IsEqualTo(0.0);
		await Assert.That(scores.Precision).IsLessThan(0.1);
		await Assert.That(scores.FScore).IsLessThan(0.1);
	}

	[Test]
	public async Task CSharpOnly_NoReconstructionScoresZero()
	{
		SurfaceScores scores = SurfaceMetrics.Compute(BenchmarkMeshTestShapes.Cube(0.5), null, null, Options);
		await Assert.That(scores.FScore).IsEqualTo(0.0);
		await Assert.That(double.IsNaN(scores.AccuracyPct)).IsTrue();
		await Assert.That(double.IsNaN(scores.CompletenessPct)).IsTrue();
	}

	[Test]
	public async Task CSharpOnly_PoseSim3RecoversAKnownSimilarityAndRotationError()
	{
		// True cameras on a ring looking at the origin; the estimate is the same rig in a frame
		// related by a known similarity, with one camera turned 5 degrees.
		var truth = new List<Rigid3d>();
		for (int i = 0; i < 10; i++)
		{
			double a = i * 0.3;
			var center = new Vector3d(3 * Math.Cos(a), 0.4 * Math.Sin(3 * a), 3 * Math.Sin(a));
			Quaterniond rotation = Quaterniond.FromAngleAxis(new AngleAxisd(a + 0.1, new Vector3d(0.1, 1, 0.2).Normalized()));
			truth.Add(new Rigid3d(rotation, -(rotation * center)));
		}

		var truthFromEstimate = new Sim3d(2.5, Quaterniond.FromAngleAxis(new AngleAxisd(0.7, new Vector3d(1, 2, 3).Normalized())), new Vector3d(0.3, -1, 2));
		Sim3d estimateFromTruth = truthFromEstimate.Inverse();
		double turn = 5 * Math.PI / 180;
		var estimated = new List<Rigid3d>();
		for (int i = 0; i < truth.Count; i++)
		{
			Quaterniond rotation = truth[i].Rotation * truthFromEstimate.Rotation;
			if (i == 4)
			{
				rotation = Quaterniond.FromAngleAxis(new AngleAxisd(turn, Vector3d.UnitZ)) * rotation;
			}

			Vector3d center = estimateFromTruth * truth[i].TgtOriginInSrc();
			estimated.Add(new Rigid3d(rotation, -(rotation * center)));
		}

		PoseScores scores = PoseMetrics.Compute(estimated, truth, diagonal: 1);
		await Assert.That(scores.Aligned).IsTrue();
		await Assert.That(scores.NumPoses).IsEqualTo(10);
		await Assert.That(Math.Abs(scores.TruthFromEstimate.Scale - 2.5)).IsLessThan(1e-9);
		await Assert.That(scores.TruthFromEstimate.Rotation.AngularDistance(truthFromEstimate.Rotation)).IsLessThan(1e-9);
		await Assert.That((scores.TruthFromEstimate.Translation - truthFromEstimate.Translation).Norm).IsLessThan(1e-9);
		await Assert.That(scores.MedianRotationDeg).IsLessThan(1e-6);
		await Assert.That(Math.Abs(scores.MaxRotationDeg - 5)).IsLessThan(1e-6);
		await Assert.That(scores.MaxPositionPct).IsLessThan(1e-6);
	}

	// Cameras whose centers are nearly collinear (a short, straight track) with noisy centers: a
	// similarity fitted to the centers alone cannot pin the rotation about their line, so the
	// alignment must take its rotation from the orientations. The centers-only RANSAC Sim3 this
	// replaced reported 25 degrees of rotation error on exactly this input.
	[Test]
	public async Task CSharpOnly_PoseSim3TakesRotationFromOrientationsOnCollinearCenters()
	{
		var truthFromEstimate = new Sim3d(0.4, Quaterniond.FromAngleAxis(new AngleAxisd(1.1, new Vector3d(-2, 1, 0.5).Normalized())), new Vector3d(1, 2, -0.5));
		Sim3d estimateFromTruth = truthFromEstimate.Inverse();
		var truth = new List<Rigid3d>();
		var estimated = new List<Rigid3d>();
		for (int i = 0; i < 6; i++)
		{
			// Along x, 1e-4 off the line; the estimate's centers are off by up to 1% of the track.
			var center = new Vector3d(0.1 * i, 1e-4 * (i % 2), 1e-4 * (i % 3));
			Quaterniond rotation = Quaterniond.FromAngleAxis(new AngleAxisd(0.05 * i, Vector3d.UnitY));
			truth.Add(new Rigid3d(rotation, -(rotation * center)));

			var noise = new Vector3d(0, 0.005 * ((i * 7 % 5) - 2), 0.005 * ((i * 3 % 4) - 1.5));
			Quaterniond estRotation = rotation * truthFromEstimate.Rotation;
			Vector3d estCenter = estimateFromTruth * (center + noise);
			estimated.Add(new Rigid3d(estRotation, -(estRotation * estCenter)));
		}

		PoseScores scores = PoseMetrics.Compute(estimated, truth, diagonal: 1);
		await Assert.That(scores.Aligned).IsTrue();
		await Assert.That(scores.TruthFromEstimate.Rotation.AngularDistance(truthFromEstimate.Rotation)).IsLessThan(1e-9);
		await Assert.That(scores.MaxRotationDeg).IsLessThan(1e-6);
		await Assert.That(Math.Abs(scores.TruthFromEstimate.Scale - 0.4)).IsLessThan(0.02);
		await Assert.That(scores.MaxPositionPct).IsLessThan(2.0);
	}

	// Rx(pi) + Ry(pi) + Rz(pi) = -I, whose nearest orthogonal matrix U V^T is -I, a reflection.
	// The chordal mean must flip the last singular direction to stay a proper rotation.
	[Test]
	public async Task CSharpOnly_ChordalMeanOfHalfTurnsIsAProperRotation()
	{
		Quaterniond[] halfTurns =
		[
			Quaterniond.FromAngleAxis(new AngleAxisd(Math.PI, Vector3d.UnitX)),
			Quaterniond.FromAngleAxis(new AngleAxisd(Math.PI, Vector3d.UnitY)),
			Quaterniond.FromAngleAxis(new AngleAxisd(Math.PI, Vector3d.UnitZ)),
		];

		await Assert.That(PoseMetrics.TryChordalMeanMatrix(halfTurns, [true, true, true], out Matrix3d mean)).IsTrue();
		await Assert.That(Math.Abs(mean.Determinant() - 1)).IsLessThan(1e-12);
		Matrix3d gram = mean.Transpose() * mean;
		for (int r = 0; r < 3; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				await Assert.That(Math.Abs(gram[r, c] - (r == c ? 1 : 0))).IsLessThan(1e-12);
			}
		}
	}

	[Test]
	public async Task CSharpOnly_TooFewPosesAreNotAligned()
	{
		Rigid3d[] two = [new Rigid3d(), new Rigid3d(Quaterniond.Identity, Vector3d.UnitX)];
		PoseScores scores = PoseMetrics.Compute(two, two, diagonal: 1);
		await Assert.That(scores.Aligned).IsFalse();
		await Assert.That(double.IsNaN(scores.MedianRotationDeg)).IsTrue();
	}

	[Test]
	public async Task CSharpOnly_IdenticalMasksHaveIouOne()
	{
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(SyntheticObjectKind.DarkObject, 2, 64, 48, seed: 1);
		await Assert.That(SilhouetteMetrics.Iou(scene.Masks[0], scene.Masks[0].Clone())).IsEqualTo(1.0);

		// Half of a full mask: IoU 1/2 exactly.
		var full = new Bitmap(4, 2, asRgb: false);
		var half = new Bitmap(4, 2, asRgb: false);
		for (int i = 0; i < 8; i++)
		{
			full.RowMajorData[i] = 255;
			half.RowMajorData[i] = (byte)(i % 2 == 0 ? 255 : 0);
		}

		await Assert.That(SilhouetteMetrics.Iou(full, half)).IsEqualTo(0.5);
		await Assert.That(SilhouetteMetrics.Iou(new Bitmap(3, 3, false), new Bitmap(3, 3, false))).IsEqualTo(1.0);
	}

	[Test]
	public async Task CSharpOnly_RenderedTruthSilhouetteMatchesTheSceneMask()
	{
		// The renderer here and the scene's rasterizer are independent; they may disagree only
		// on pixels whose centers lie on the silhouette's edge.
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(SyntheticObjectKind.DarkObject, 3, 96, 72, seed: 2);
		BenchmarkMesh mesh = BenchmarkMesh.FromScene(scene);
		for (int k = 0; k < 3; k++)
		{
			Bitmap rendered = SilhouetteMetrics.RenderMask(scene.Camera, scene.CamFromWorld[k], mesh);
			await Assert.That(SilhouetteMetrics.Iou(rendered, scene.Masks[k])).IsGreaterThan(0.98);
		}
	}

	[Test]
	public async Task CSharpOnly_SurfaceSamplingIsSeededAndOnTheSurface()
	{
		BenchmarkMesh cube = BenchmarkMeshTestShapes.Cube(0.5);
		Vector3d[] first = cube.SampleSurface(500, 4);
		Vector3d[] again = cube.SampleSurface(500, 4);
		Vector3d[] other = cube.SampleSurface(500, 5);
		await Assert.That(first.SequenceEqual(again)).IsTrue();
		await Assert.That(first.SequenceEqual(other)).IsFalse();

		// Every sample is on a face, and each of the six faces gets a fair share.
		var perFace = new int[6];
		foreach (Vector3d p in first)
		{
			double[] c = [p.X, p.Y, p.Z];
			int face = Enumerable.Range(0, 3).First(a => Math.Abs(Math.Abs(c[a]) - 0.5) < 1e-12);
			perFace[2 * face + (c[face] > 0 ? 1 : 0)]++;
		}

		await Assert.That(perFace.Sum()).IsEqualTo(500);
		await Assert.That(perFace.Min()).IsGreaterThan(50);
	}

	[Test]
	public async Task CSharpOnly_StageTimerAccumulatesRepeatedStages()
	{
		var timer = new StageTimer();
		timer.Report(new("A", 0, 0, ""));
		timer.Report(new("A", 1, 2, "x"));
		timer.Report(new("B", 0, 0, ""));
		timer.Report(new("A", 0, 0, ""));
		IReadOnlyList<(string Stage, double Seconds)> stages = timer.Stages();
		await Assert.That(string.Join(",", stages.Select(s => s.Stage))).IsEqualTo("A,B");
		await Assert.That(stages.All(s => s.Seconds >= 0)).IsTrue();
	}
}
