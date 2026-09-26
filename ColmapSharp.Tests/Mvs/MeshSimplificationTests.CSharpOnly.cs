// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// MeshSimplificationTests.CSharpOnly: C#-only cases for ColmapSharp/Mvs/MeshSimplification.cs
// that no mesh_simplification_test.cc case stands in for - the same result for every thread
// count, cancellation and progress reporting, and the specialized 4x4 solve agreeing bit for
// bit with Matrix4d's inverse.

using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class MeshSimplificationCSharpOnlyTests
{
	[Test]
	public async Task CSharpOnly_SameResultForEveryThreadCount()
	{
		PlyMesh mesh = MeshSimplificationTests.CreateWavyGridMesh(30);
		var options = new MeshSimplificationOptions { TargetFaceRatio = 0.2, NumThreads = 1 };
		PlyMesh single = MeshSimplification.SimplifyMesh(mesh, options);
		options.NumThreads = 7;
		PlyMesh multi = MeshSimplification.SimplifyMesh(mesh, options);

		await Assert.That(multi.Faces.Count).IsEqualTo(single.Faces.Count);
		await Assert.That(multi.Vertices.Count).IsEqualTo(single.Vertices.Count);
		for (int i = 0; i < single.Faces.Count; i++)
		{
			await Assert.That(multi.Faces[i]).IsEqualTo(single.Faces[i]);
		}

		for (int i = 0; i < single.Vertices.Count; i++)
		{
			await Assert.That(multi.Vertices[i]).IsEqualTo(single.Vertices[i]);
		}
	}

	[Test]
	public async Task CSharpOnly_CancellationStopsTheRun()
	{
		PlyMesh mesh = MeshSimplificationTests.CreateGridMesh(10);
		using var cancellation = new CancellationTokenSource();
		cancellation.Cancel();

		await Assert.That(() => MeshSimplification.SimplifyMesh(mesh, new MeshSimplificationOptions(), null, cancellation.Token))
			.Throws<OperationCanceledException>();
	}

	[Test]
	public async Task CSharpOnly_ReportsProgressUpToCompletion()
	{
		PlyMesh mesh = MeshSimplificationTests.CreateGridMesh(20);
		var reports = new List<double>();
		var progress = new SynchronousProgress(reports.Add);

		PlyMesh result = MeshSimplification.SimplifyMesh(mesh, new MeshSimplificationOptions { TargetFaceRatio = 0.5 }, progress);

		// The target was reached, so the last 10% step reports completion.
		await Assert.That(result.Faces.Count).IsLessThanOrEqualTo(400);
		await Assert.That(reports.Count).IsGreaterThan(1);
		await Assert.That(reports[^1]).IsEqualTo(1.0);
		for (int i = 1; i < reports.Count; i++)
		{
			await Assert.That(reports[i]).IsGreaterThan(reports[i - 1]);
		}
	}

	[Test]
	public async Task CSharpOnly_OptimalPositionSolveMatchesMatrix4dInverse()
	{
		// A symmetric quadric-like matrix; the solve uses rows 0-2 and a (0, 0, 0, 1) last row.
		double[] q =
		[
			2.5, 0.3, -0.7, 1.1,
			0.3, 1.9, 0.2, -0.4,
			-0.7, 0.2, 3.1, 0.6,
			1.1, -0.4, 0.6, 5.0,
		];
		bool solved = MeshSimplifier.TrySolveOptimalPosition(q, out Vector3d position);

		var a = new Matrix4d(
			q[0], q[1], q[2], q[3],
			q[4], q[5], q[6], q[7],
			q[8], q[9], q[10], q[11],
			0, 0, 0, 1);
		Vector4d expected = a.Inverse().Col(3);

		await Assert.That(solved).IsTrue();
		await Assert.That(position.X).IsEqualTo(expected.X);
		await Assert.That(position.Y).IsEqualTo(expected.Y);
		await Assert.That(position.Z).IsEqualTo(expected.Z);
	}

	// Progress<T> posts to the thread pool; this one calls back inline so the list is complete
	// when SimplifyMesh returns.
	private sealed class SynchronousProgress(Action<double> report) : IProgress<double>
	{
		public void Report(double value) => report(value);
	}
}
