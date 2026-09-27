// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PoissonMeshingTests.CSharpOnly (C#-only, not COLMAP tests): the parts of
// ColmapSharp/Mvs/PoissonMeshing.cs that poisson_meshing_test.cc does not reach - the
// std::to_string/atof conversion of pointWeight and trim (Tier A: printf's "%f" rounds the exact
// binary value half to even), PoissonMeshingOptions.Check, colors following the input's
// properties rather than options.Color, a PLY without normals failing as PoissonRecon does, an
// unwritable output path throwing as THROW_CHECK_PATH_OPEN does, and cancellation and
// progress. (A PLY with color properties and no points is PoissonMeshingOracleTests.Exact.cs's
// "empty" case, byte-exact against upstream.)

using ColmapSharp.Mvs;
using ColmapSharp.Mvs.PoissonRecon;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public partial class PoissonMeshingTests
{
	[Test]
	[Arguments(10.0, "10.000000", 10.0f)]
	[Arguments(0.0, "0.000000", 0.0f)]
	[Arguments(1e-7, "0.000000", 0.0f)]
	[Arguments(0.0078125, "0.007812", 0.007812f)] // an exact tie, rounded to even (down)
	[Arguments(0.0234375, "0.023438", 0.023438f)] // an exact tie, rounded to even (up)
	[Arguments(0.1, "0.100000", 0.1f)]
	[Arguments(123456789.123, "123456789.123000", 123456789.123f)]
	[Arguments(-0.0, "-0.000000", -0.0f)]
	public async Task CppToStringAsFloat_MatchesToStringThenAtof(double value, string text, float expected)
	{
		await Assert.That(PoissonMeshing.CppToString(value)).IsEqualTo(text);
		await Assert.That(BitConverter.SingleToInt32Bits(PoissonMeshing.CppToStringAsFloat(value))).IsEqualTo(BitConverter.SingleToInt32Bits(expected));
	}

	[Test]
	public async Task PoissonMeshingOptions_Check()
	{
		await Assert.That(new PoissonMeshingOptions().Check()).IsTrue();
		await Assert.That(new PoissonMeshingOptions { PointWeight = -1 }.Check()).IsFalse();
		await Assert.That(new PoissonMeshingOptions { PointWeight = 0 }.Check()).IsTrue();
		await Assert.That(new PoissonMeshingOptions { Depth = 0 }.Check()).IsFalse();
		await Assert.That(new PoissonMeshingOptions { Trim = -0.5 }.Check()).IsFalse();
		await Assert.That(new PoissonMeshingOptions { NumThreads = 0 }.Check()).IsFalse();
		await Assert.That(new PoissonMeshingOptions { NumThreads = -2 }.Check()).IsFalse();
		await Assert.That(() => PoissonMeshing.Run(new PoissonMeshingOptions { Depth = 0 }, SpherePositions(), SpherePositions())).Throws<ArgumentException>();
	}

	[Test]
	public async Task PoissonMeshing_ColorsFollowTheInputNotTheOption()
	{
		float[] points = SpherePositions();
		byte[] colors = Enumerable.Repeat((byte)200, points.Length).ToArray();
		var options = new PoissonMeshingOptions { Depth = 4, Trim = 0, Color = false };
		PoissonMeshOutput colored = PoissonMeshing.Run(options, points, points, colors);
		await Assert.That(colored.ColorChannels).IsEqualTo(3);
		await Assert.That(colored.Values).IsNull();

		options.Color = true;
		PoissonMeshOutput plain = PoissonMeshing.Run(options, points, points);
		await Assert.That(plain.Colors).IsNull();

		// The colors do not change the surface.
		await Assert.That(plain.Positions.SequenceEqual(colored.Positions)).IsTrue();
		await Assert.That(plain.Triangles.SequenceEqual(colored.Triangles)).IsTrue();
	}

	[Test]
	public async Task PoissonMeshing_PlyWithoutNormals_ReturnsFalse()
	{
		string testDir = MvsTestUtils.CreateTestDir();
		string inputPath = Path.Combine(testDir, "points.ply");
		float[] positions = SpherePositions();
		var points = new List<PlyPoint>();
		for (int i = 0; i < positions.Length / 3; i++)
		{
			points.Add(new PlyPoint { X = positions[3 * i], Y = positions[(3 * i) + 1], Z = positions[(3 * i) + 2] });
		}

		Ply.WriteBinaryPlyPoints(inputPath, points, writeNormal: false, writeRgb: true);
		var options = new PoissonMeshingOptions { Depth = 3, Trim = 0 };
		await Assert.That(PoissonMeshing.Run(options, inputPath, Path.Combine(testDir, "mesh.ply"))).IsFalse();
		await Assert.That(() => PoissonMeshing.Run(options, Path.Combine(testDir, "missing.ply"), Path.Combine(testDir, "mesh.ply"))).Throws<ArgumentException>();
	}

	[Test]
	public async Task PoissonMeshing_UnwritableOutput_Throws()
	{
		// THROW_CHECK_PATH_OPEN runs before COLMAP's try, so it throws rather than returning false.
		string testDir = MvsTestUtils.CreateTestDir();
		string inputPath = Path.Combine(testDir, "points.ply");
		float[] positions = SpherePositions();
		var points = new List<PlyPoint>();
		for (int i = 0; i < positions.Length / 3; i++)
		{
			points.Add(new PlyPoint
			{
				X = positions[3 * i], Y = positions[(3 * i) + 1], Z = positions[(3 * i) + 2],
				Nx = positions[3 * i], Ny = positions[(3 * i) + 1], Nz = positions[(3 * i) + 2],
			});
		}

		Ply.WriteBinaryPlyPoints(inputPath, points, writeNormal: true, writeRgb: true);
		string directoryOutput = Path.Combine(testDir, "mesh.ply");
		Directory.CreateDirectory(directoryOutput);
		var options = new PoissonMeshingOptions { Depth = 3, Trim = 0 };
		await Assert.That(() => PoissonMeshing.Run(options, inputPath, directoryOutput)).Throws<ArgumentException>();
		await Assert.That(() => PoissonMeshing.Run(options, inputPath, Path.Combine(testDir, "missing", "mesh.ply"))).Throws<ArgumentException>();
	}

	[Test]
	public async Task PoissonMeshing_CancellationAndProgress()
	{
		float[] points = SpherePositions();
		var options = new PoissonMeshingOptions { Depth = 5, Trim = 1 };
		using var cancelled = new CancellationTokenSource();
		cancelled.Cancel();
		await Assert.That(() => PoissonMeshing.Run(options, points, points, default, cancelled.Token)).Throws<OperationCanceledException>();

		var fractions = new List<double>();
		PoissonMeshing.Run(options, points, points, default, default, new SynchronousProgress(fractions.Add));
		await Assert.That(fractions.Count).IsGreaterThan(2);
		await Assert.That(fractions.All(f => f >= 0 && f <= 1)).IsTrue();
		await Assert.That(fractions.Zip(fractions.Skip(1)).All(pair => pair.Second >= pair.First)).IsTrue();
		await Assert.That(fractions[^1]).IsEqualTo(1.0);
	}

	// Points on the unit sphere (a Fibonacci lattice); each point is also its own normal.
	private static float[] SpherePositions(int n = 300)
	{
		float[] p = new float[3 * n];
		double golden = Math.PI * (3 - Math.Sqrt(5));
		for (int i = 0; i < n; i++)
		{
			double y = 1 - (2 * (i + 0.5) / n), r = Math.Sqrt(1 - (y * y)), t = golden * i;
			p[3 * i] = (float)(r * Math.Cos(t));
			p[(3 * i) + 1] = (float)y;
			p[(3 * i) + 2] = (float)(r * Math.Sin(t));
		}

		return p;
	}

	private sealed class SynchronousProgress(Action<double> report) : IProgress<double>
	{
		public void Report(double value) => report(value);
	}
}
