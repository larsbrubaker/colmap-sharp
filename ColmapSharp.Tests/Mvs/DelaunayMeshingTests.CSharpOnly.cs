// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// DelaunayMeshingCSharpOnlyTests (C#-only): the building blocks of
// colmap/mvs/delaunay_meshing.cc - DelaunayMeshingOptions.Check, DelaunayMeshingInput (sparse
// and dense in-memory input, full and subsampled triangulation),
// DelaunayMeshingEdgeWeightComputer and DelaunayTriangulationRayCaster. The whole pipeline is
// covered by DelaunayMeshingTests.cs (delaunay_meshing_test.cc 1:1) and
// DelaunayMeshingSceneTests.CSharpOnly.cs. Tier C overall; the ray caster's facet set is
// exact for rays in general position.

using ColmapSharp.Geometry.Delaunay;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mathematics;
using ColmapSharp.Mvs;
using ColmapSharp.Scene;
using ColmapSharp.Tests.Geometry.Delaunay;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class DelaunayMeshingCSharpOnlyTests
{
	[Test]
	public async Task CSharpOnly_OptionsCheck()
	{
		var invalid = new List<Action<DelaunayMeshingOptions>>
		{
			o => o.MaxProjDist = -1,
			o => o.MaxDepthDist = -0.1,
			o => o.MaxDepthDist = 1.1,
			o => o.VisibilitySigma = 0,
			o => o.DistanceSigmaFactor = 0,
			o => o.QualityRegularization = -1,
			o => o.MaxSideLengthFactor = -1,
			o => o.MaxSideLengthPercentile = -1,
			o => o.MaxSideLengthPercentile = 101,
			o => o.NumThreads = -2,
			o => o.NumThreads = 0,
		};
		int accepted = 0;
		foreach (var change in invalid)
		{
			var options = new DelaunayMeshingOptions();
			change(options);
			if (options.Check())
			{
				++accepted;
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(new DelaunayMeshingOptions().Check()).IsTrue();
			await Assert.That(new DelaunayMeshingOptions { MaxProjDist = 0, MaxDepthDist = 1, NumThreads = 4 }.Check()).IsTrue();
			await Assert.That(accepted).IsEqualTo(0);
		}
	}

	[Test]
	public async Task CSharpOnly_InputFromSparseReconstruction()
	{
		var reconstruction = Synthesize(5, 100);
		var input = DelaunayMeshingInput.FromSparseReconstruction(reconstruction);

		var pointIds = reconstruction.Points3D.Keys.Order().ToList();
		int badPoints = 0;
		for (int i = 0; i < pointIds.Count; ++i)
		{
			var point3D = reconstruction.Points3D[pointIds[i]];
			var expected = new Vector3f((float)point3D.Xyz.X, (float)point3D.Xyz.Y, (float)point3D.Xyz.Z);
			if (input.Points[i].Position != expected || input.Points[i].NumVisibleImages != point3D.Track.Length)
			{
				++badPoints;
			}
		}

		// Every observation of a 3D point appears in its image's point list.
		int badImages = 0;
		var regImageIds = reconstruction.RegImageIds();
		for (int k = 0; k < regImageIds.Count; ++k)
		{
			var image = reconstruction.Image(regImageIds[k]);
			var expected = image.Points2D.Where(p => p.HasPoint3D).Select(p => pointIds.IndexOf(p.Point3DId)).ToList();
			if (!expected.SequenceEqual(input.Images[k].PointIdxs) || input.Images[k].CameraId != image.CameraId)
			{
				++badImages;
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(input.Points.Count).IsEqualTo(reconstruction.NumPoints3D);
			await Assert.That(input.Images.Count).IsEqualTo(reconstruction.NumRegImages);
			await Assert.That(input.Cameras.Count).IsEqualTo(reconstruction.NumCameras);
			await Assert.That(badPoints).IsEqualTo(0);
			await Assert.That(badImages).IsEqualTo(0);
		}
	}

	[Test]
	public async Task CSharpOnly_InputFromDense()
	{
		var reconstruction = Synthesize(3, 20);
		var points = new List<PlyPoint>
		{
			new() { X = 1, Y = 2, Z = 3 },
			new() { X = 4, Y = 5, Z = 6 },
			new() { X = 7, Y = 8, Z = 9 },
		};
		var visibility = new List<IReadOnlyList<int>> { new[] { 2, 0 }, new[] { 1 }, new[] { 0, 1, 2 } };
		var input = DelaunayMeshingInput.FromDense(reconstruction, points, visibility);

		var badVisibility = new List<IReadOnlyList<int>> { new[] { 0 }, new[] { 3 }, new[] { 0 } };
		using (Assert.Multiple())
		{
			await Assert.That(input.Images.Count).IsEqualTo(3);
			await Assert.That(input.Images[0].PointIdxs).IsEquivalentTo(new[] { 0, 2 });
			await Assert.That(input.Images[1].PointIdxs).IsEquivalentTo(new[] { 1, 2 });
			await Assert.That(input.Images[2].PointIdxs).IsEquivalentTo(new[] { 0, 2 });
			await Assert.That(input.Points[1]).IsEqualTo(new DelaunayMeshingInput.InputPoint(new Vector3f(4, 5, 6), 1));
			await Assert.That(input.Points[2].NumVisibleImages).IsEqualTo(3u);
			await Assert.That(() => DelaunayMeshingInput.FromDense(reconstruction, points, badVisibility))
				.Throws<ArgumentOutOfRangeException>();
		}
	}

	[Test]
	public async Task CSharpOnly_NonSubsampledTriangulationUsesEveryPoint()
	{
		var input = DelaunayMeshingInput.FromSparseReconstruction(Synthesize(5, 200));
		var triangulation = input.CreateSubSampledDelaunayTriangulation(0, 0.05f);
		using (Assert.Multiple())
		{
			await Assert.That(triangulation.NumberOfVertices).IsEqualTo(200);
			await Assert.That(DelaunayValidator.StructureErrors(triangulation)).IsEmpty();
		}
	}

	[Test]
	public async Task CSharpOnly_SubsampledTriangulationMergesNearbyPoints()
	{
		// A tight cluster of 400 points, a few pixels apart in every image: most of them fall
		// within max_proj_dist of the cell they land in and are skipped.
		var reconstruction = Synthesize(4, 30);
		var random = new Random(4);
		var plyPoints = new List<PlyPoint>();
		var visibility = new List<IReadOnlyList<int>>();
		for (int i = 0; i < 400; ++i)
		{
			plyPoints.Add(new PlyPoint
			{
				X = (float)(random.NextDouble() * 0.05),
				Y = (float)(random.NextDouble() * 0.05),
				Z = (float)(random.NextDouble() * 0.05),
			});
			visibility.Add(new[] { 0, 1, 2, 3 });
		}

		var input = DelaunayMeshingInput.FromDense(reconstruction, plyPoints, visibility);
		RandomUtils.SetPRNGSeed(7);
		var first = input.CreateSubSampledDelaunayTriangulation(20, 0.05f);
		RandomUtils.SetPRNGSeed(7);
		var second = input.CreateSubSampledDelaunayTriangulation(20, 0.05f);
		using (Assert.Multiple())
		{
			await Assert.That(first.Dimension).IsEqualTo(3);
			await Assert.That(first.NumberOfVertices).IsLessThan(400);
			await Assert.That(first.NumberOfVertices).IsGreaterThanOrEqualTo(4);
			await Assert.That(DelaunayValidator.StructureErrors(first)).IsEmpty();
			await Assert.That(DelaunayValidator.Fingerprint(second)).IsEqualTo(DelaunayValidator.Fingerprint(first));
		}
	}

	[Test]
	public async Task CSharpOnly_SubsampledTriangulationInsertsEveryPointWhileFlat()
	{
		// Entry 104: while the points are coplanar there is no cell to test against, so every
		// point goes in (COLMAP's CGAL would locate in its 2D triangulation instead).
		var reconstruction = Synthesize(3, 20);
		var plyPoints = new List<PlyPoint>();
		var visibility = new List<IReadOnlyList<int>>();
		for (int i = 0; i < 50; ++i)
		{
			plyPoints.Add(new PlyPoint { X = (i % 10) * 0.001f, Y = (i / 10) * 0.001f, Z = 0 });
			visibility.Add(new[] { 0, 1, 2 });
		}

		var input = DelaunayMeshingInput.FromDense(reconstruction, plyPoints, visibility);
		var triangulation = input.CreateSubSampledDelaunayTriangulation(20, 0.05f);
		using (Assert.Multiple())
		{
			await Assert.That(triangulation.Dimension).IsEqualTo(2);
			await Assert.That(triangulation.NumberOfVertices).IsEqualTo(50);
		}
	}

	[Test]
	public async Task CSharpOnly_EdgeWeightComputer()
	{
		var triangulation = new DelaunayTriangulation3();
		triangulation.InsertRange(new[] { new Vector3d(0, 0, 0), new Vector3d(1, 0, 0), new Vector3d(0, 2, 0), new Vector3d(0, 0, 3) });

		// Squared edge lengths 1, 4, 5, 9, 10, 13: the 25th percentile interpolates 4 and 5.
		var computer = new DelaunayMeshingEdgeWeightComputer(triangulation, visibilitySigma: 3.0, distanceSigmaFactor: 2.0);
		double sigma = 2.0 * Math.Sqrt(4.25);
		using (Assert.Multiple())
		{
			await Assert.That(computer.DistanceSigma).IsEqualTo(sigma);
			await Assert.That(computer.ComputeDistanceProb(0)).IsEqualTo(0.0);
			await Assert.That(computer.ComputeDistanceProb(1)).IsEqualTo(1.0 - Math.Exp(-0.5 / (sigma * sigma)));
			await Assert.That(computer.ComputeDistanceProb(5 * sigma)).IsEqualTo(1.0);
			await Assert.That(computer.ComputeVisibilityProb(4)).IsEqualTo(1.0 - Math.Exp(4 * (-0.5 / 9.0)));
			await Assert.That(computer.ComputeVisibilityProb(15)).IsEqualTo(1.0);
		}
	}

	[Test]
	public async Task CSharpOnly_RayCasterMatchesBruteForce()
	{
		var random = new Random(8);
		var points = new Vector3d[300];
		for (int i = 0; i < points.Length; ++i)
		{
			points[i] = new Vector3d(random.NextDouble() * 2 - 1, random.NextDouble() * 2 - 1, random.NextDouble() * 2 - 1);
		}

		var triangulation = new DelaunayTriangulation3();
		triangulation.InsertRange(points);
		var caster = new DelaunayTriangulationRayCaster(triangulation);
		var intersections = new List<DelaunayTriangulationRayCaster.Intersection>();
		var traversal = new List<(int Cell, int Index)>();
		int mismatches = 0;
		for (int trial = 0; trial < 100; ++trial)
		{
			// A camera outside the hull (or inside, every fourth ray) looking at a point.
			double scale = trial % 4 == 0 ? 0.5 : 4.0;
			var start = new Vector3d((random.NextDouble() - 0.5) * scale, (random.NextDouble() - 0.5) * scale, (random.NextDouble() - 0.5) * scale);
			var end = new Vector3d(random.NextDouble() - 0.5, random.NextDouble() - 0.5, random.NextDouble() - 0.5);
			var cursor = default(DelaunayTriangulation3.LocateCursor);
			caster.CastRaySegment(start, end, intersections, ref cursor);
			triangulation.TraverseSegment(start, end, traversal);

			var expected = new HashSet<(int, int, int)>();
			foreach (var (cell, i) in triangulation.FiniteFacets())
			{
				var (u, v, w) = triangulation.FacetVertices(cell, i);
				if (DelaunayTriangulation3.SegmentCrossesTriangle(start, end, triangulation.Point(u), triangulation.Point(v), triangulation.Point(w)))
				{
					expected.Add(Sorted(u, v, w));
				}
			}

			var cast = intersections.Select(x => triangulation.FacetVertices(x.Cell, x.Index)).Select(t => Sorted(t.A, t.B, t.C)).ToList();
			bool decreasing = true;
			for (int k = 1; k < intersections.Count; ++k)
			{
				decreasing &= intersections[k].TargetDistanceSquared <= intersections[k - 1].TargetDistanceSquared;
			}

			bool sameAsTraversal = traversal.Count == intersections.Count
				&& traversal.Zip(intersections).All(p => p.First.Cell == p.Second.Cell && p.First.Index == p.Second.Index);
			if (!decreasing || cast.Count != expected.Count || !expected.SetEquals(cast) || !sameAsTraversal)
			{
				++mismatches;
			}
		}

		using (Assert.Multiple())
		{
			await Assert.That(caster.HullFacets.Count).IsEqualTo(triangulation.AllCells().Count(triangulation.IsInfinite));
			await Assert.That(mismatches).IsEqualTo(0);
		}
	}

	private static (int, int, int) Sorted(int a, int b, int c)
	{
		Span<int> s = [a, b, c];
		s.Sort();
		return (s[0], s[1], s[2]);
	}

	private static Reconstruction Synthesize(int numFrames, int numPoints3D)
	{
		RandomUtils.SetPRNGSeed(0);
		var options = new SyntheticDatasetOptions
		{
			NumRigs = 1,
			NumCamerasPerRig = 1,
			NumFramesPerRig = numFrames,
			NumPoints3D = numPoints3D,
		};
		var reconstruction = new Reconstruction();
		Synthetic.SynthesizeDataset(options, reconstruction);
		return reconstruction;
	}
}
