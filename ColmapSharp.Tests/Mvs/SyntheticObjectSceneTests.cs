// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// SyntheticObjectSceneTests: C#-only tests (not ports; COLMAP has no synthetic object scenes).
// They pin ColmapSharp/Mvs/Testing/SyntheticObjectScene*.cs, the benchmark's scene generator:
// it is deterministic, returns the right amount of truth, its mesh is closed and outward, its
// world frame is the object's, and its truth agrees with its pixels - each true mask matches an
// independent ray cast of the true mesh through the true cameras (TriangleBvh, a different code path from the rasterizer), and
// on the dark object the rendered frame separates object from wall exactly where the mask does.
// Images are small so the whole class runs in about a second.

using System.Security.Cryptography;

using ColmapSharp.Geometry;
using ColmapSharp.LinearAlgebra;
using ColmapSharp.Mvs;
using ColmapSharp.Mvs.Testing;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class SyntheticObjectSceneTests
{
	private const int Width = 64;
	private const int Height = 48;

	[Test]
	public async Task CSharpOnly_SameSeedGivesIdenticalFrames()
	{
		foreach (SyntheticObjectKind kind in Enum.GetValues<SyntheticObjectKind>())
		{
			SyntheticObjectScene first = SyntheticObjectScene.Generate(kind, 4, Width, Height, seed: 7);
			SyntheticObjectScene second = SyntheticObjectScene.Generate(kind, 4, Width, Height, seed: 7);
			SyntheticObjectScene other = SyntheticObjectScene.Generate(kind, 4, Width, Height, seed: 8);
			for (int k = 0; k < 4; ++k)
			{
				await Assert.That(Hash(second.Frames[k])).IsEqualTo(Hash(first.Frames[k]));
				await Assert.That(Hash(second.Masks[k])).IsEqualTo(Hash(first.Masks[k]));
				await Assert.That(second.CamFromWorld[k]).IsEqualTo(first.CamFromWorld[k]);

				// The seed reaches the pixels (noise) and the poses (jitter).
				await Assert.That(Hash(other.Frames[k])).IsNotEqualTo(Hash(first.Frames[k]));
				await Assert.That(other.CamFromWorld[k]).IsNotEqualTo(first.CamFromWorld[k]);
			}
		}
	}

	[Test]
	public async Task CSharpOnly_CountsSizesAndCamera()
	{
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(SyntheticObjectKind.TexturedSphere, 5, 40, 30, seed: 1);
		await Assert.That(scene.Kind).IsEqualTo(SyntheticObjectKind.TexturedSphere);
		await Assert.That(scene.Frames.Count).IsEqualTo(5);
		await Assert.That(scene.Masks.Count).IsEqualTo(5);
		await Assert.That(scene.CamFromWorld.Count).IsEqualTo(5);
		for (int k = 0; k < 5; ++k)
		{
			await Assert.That(scene.Frames[k].Width).IsEqualTo(40);
			await Assert.That(scene.Frames[k].Height).IsEqualTo(30);
			await Assert.That(scene.Frames[k].IsRGB).IsTrue();
			await Assert.That(scene.Masks[k].Width).IsEqualTo(40);
			await Assert.That(scene.Masks[k].Height).IsEqualTo(30);
			await Assert.That(scene.Masks[k].IsGrey).IsTrue();
		}

		// One shared camera.
		await Assert.That(scene.Camera.CameraId).IsEqualTo(1u);
		await Assert.That(scene.Camera.ModelId).IsEqualTo(CameraModelId.Pinhole);
		await Assert.That(scene.Camera.Width).IsEqualTo(40);
		await Assert.That(scene.Camera.Height).IsEqualTo(30);
		await Assert.That(scene.Camera.PrincipalPointX()).IsEqualTo(20.0);
		await Assert.That(scene.Camera.PrincipalPointY()).IsEqualTo(15.0);
	}

	// Every directed edge appears exactly once and its reverse too (closed, and neighbors wind
	// consistently), and the signed volume is positive (so the consistent winding is outward)
	// and near the expected one. The mouse has no simple closed form, so it is bounded loosely:
	// its exponents below 1 make it contain the ellipsoid of the same half-extents, and it lies
	// inside their box.
	[Test]
	public async Task CSharpOnly_MeshIsClosedAndOutward()
	{
		const double A = SyntheticObjectScene.MouseHalfLength, B = SyntheticObjectScene.MouseHalfWidth;
		const double Top = SyntheticObjectScene.MouseTopHeight, Bottom = SyntheticObjectScene.MouseBottomHeight;
		(SyntheticObjectKind Kind, double Min, double Max)[] cases =
		[
			(SyntheticObjectKind.TexturedSphere, 0.99 * 4.0 / 3 * Math.PI * 0.125, 4.0 / 3 * Math.PI * 0.125),
			(SyntheticObjectKind.TexturelessBox, 0.9 * 0.6 * 0.7 - 1e-12, 0.9 * 0.6 * 0.7 + 1e-12),
			(SyntheticObjectKind.DarkObject, 2.0 / 3 * Math.PI * A * B * (Top + Bottom), 4 * A * B * (Top + Bottom)),
		];
		foreach ((SyntheticObjectKind kind, double min, double max) in cases)
		{
			SyntheticObjectScene scene = SyntheticObjectScene.Generate(kind, 1, 16, 12, seed: 1);
			var directedEdgeUses = new Dictionary<(int, int), int>();
			double volume = 0;
			foreach (PlyMeshFace face in scene.MeshTriangles)
			{
				int[] corners = [face.VertexIdx1, face.VertexIdx2, face.VertexIdx3];
				for (int e = 0; e < 3; ++e)
				{
					var key = (corners[e], corners[(e + 1) % 3]);
					directedEdgeUses[key] = directedEdgeUses.GetValueOrDefault(key) + 1;
				}

				Vector3d p0 = scene.MeshVertices[face.VertexIdx1];
				volume += p0.Dot(scene.MeshVertices[face.VertexIdx2].Cross(scene.MeshVertices[face.VertexIdx3])) / 6;
			}

			await Assert.That(directedEdgeUses.Values.All(uses => uses == 1)).IsTrue();
			await Assert.That(directedEdgeUses.Keys.All(edge => directedEdgeUses.ContainsKey((edge.Item2, edge.Item1)))).IsTrue();
			await Assert.That(volume).IsGreaterThan(0.0);
			await Assert.That(volume).IsGreaterThanOrEqualTo(min);
			await Assert.That(volume).IsLessThanOrEqualTo(max);
		}
	}

	// The world is the object's frame: undoing the object's known turn at each frame's time
	// leaves the camera's pose in the lab, which is the same for every frame up to the jitter.
	// The poses themselves swing through large angles, since the camera circles the object.
	[Test]
	public async Task CSharpOnly_WorldFrameIsObjectFixed()
	{
		const int NumFrames = 12;
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(SyntheticObjectKind.DarkObject, NumFrames, 16, 12, seed: 4);

		// Each jitter component is within its bound, so two frames differ by at most twice the
		// bound per axis (sqrt(3) times that for the 3-vector).
		double maxRotation = 2 * Math.Sqrt(3) * SyntheticObjectScene.RotationJitter;
		double maxCenterShift = 2 * Math.Sqrt(3) * SyntheticObjectScene.TranslationJitter;
		Rigid3d first = CamFromLab(scene, 0);
		double largestWorldTurn = 0;
		for (int k = 1; k < NumFrames; ++k)
		{
			Rigid3d camFromLab = CamFromLab(scene, k);
			await Assert.That(camFromLab.Rotation.AngularDistance(first.Rotation)).IsLessThanOrEqualTo(maxRotation + 1e-9);
			double centerShift = (camFromLab.TgtOriginInSrc() - first.TgtOriginInSrc()).Norm;
			await Assert.That(centerShift).IsLessThanOrEqualTo(maxCenterShift + 1e-9);
			largestWorldTurn = Math.Max(
				largestWorldTurn, scene.CamFromWorld[k].Rotation.AngularDistance(scene.CamFromWorld[0].Rotation));
		}

		await Assert.That(largestWorldTurn).IsGreaterThan(Math.PI / 2);
	}

	// The mask is the rasterizer's coverage of the pixel center. Cast the pixel-center ray of
	// the true camera against the true mesh with TriangleBvh instead: the two may disagree only
	// where the ray grazes a silhouette edge (float BVH versus double edge tests).
	[Test]
	public async Task CSharpOnly_MasksMatchIndependentRayCastOfMesh()
	{
		foreach (SyntheticObjectKind kind in Enum.GetValues<SyntheticObjectKind>())
		{
			SyntheticObjectScene scene = SyntheticObjectScene.Generate(kind, 4, Width, Height, seed: 3);
			TriangleBvh bvh = BuildBvh(scene);
			double fx = scene.Camera.FocalLengthX(), fy = scene.Camera.FocalLengthY();
			double cx = scene.Camera.PrincipalPointX(), cy = scene.Camera.PrincipalPointY();
			for (int k = 0; k < scene.Frames.Count; ++k)
			{
				var worldFromCam = scene.CamFromWorld[k].Inverse();
				Vector3d eye = worldFromCam.Translation;
				byte[] mask = scene.Masks[k].RowMajorData;
				int area = 0, rayArea = 0, disagreements = 0;
				for (int y = 0; y < Height; ++y)
				{
					for (int x = 0; x < Width; ++x)
					{
						Vector3d direction = worldFromCam.Rotation * new Vector3d((x + 0.5 - cx) / fx, (y + 0.5 - cy) / fy, 1);
						Vector3d end = eye + 20 * direction;
						bool hit = bvh.AnyHit(
							(float)eye.X, (float)eye.Y, (float)eye.Z, (float)end.X, (float)end.Y, (float)end.Z, -1, float.MaxValue);
						bool inMask = mask[y * Width + x] == 255;
						area += inMask ? 1 : 0;
						rayArea += hit ? 1 : 0;
						disagreements += hit != inMask ? 1 : 0;
					}
				}

				// The object fills a real part of the frame and is not cut off everywhere.
				await Assert.That(area).IsGreaterThan(Width * Height / 20);
				await Assert.That(area).IsLessThan(Width * Height / 2);
				await Assert.That(Math.Abs(area - rayArea)).IsLessThanOrEqualTo(2);
				await Assert.That(disagreements).IsLessThanOrEqualTo(2);
			}
		}
	}

	// On the dark object every channel is far below the wall, so thresholding the frame must
	// reproduce the mask everywhere except pixels on the silhouette, whose supersamples mix
	// object and wall: pixels whose 3x3 neighborhood is all object or all wall must agree. The
	// threshold comes from the scene's constants: halfway between the brightest the object can
	// be (the label's top albedo fully lit, plus a full highlight) and the darkest the wall is
	// in this frame (its gradient is linear in the hit point, so the minimum is at an image
	// corner). The gap must exceed 12 noise sigmas, so a change to the scene that closes it
	// fails here by name rather than by chance.
	[Test]
	public async Task CSharpOnly_DarkObjectFrameBoundaryAgreesWithMask()
	{
		const int NumFrames = 6;
		const double NoiseSigma = 1.5;
		SyntheticObjectScene scene = SyntheticObjectScene.Generate(
			SyntheticObjectKind.DarkObject, NumFrames, Width, Height, seed: 11, sensorNoiseSigma: NoiseSigma);
		double objectMax = 255 * (
			(SyntheticObjectScene.LabelMinAlbedo + SyntheticObjectScene.LabelAlbedoRange)
				* (SyntheticObjectScene.Ambient + SyntheticObjectScene.Diffuse)
			+ SyntheticObjectScene.DarkObjectSpecular);
		double fx = scene.Camera.FocalLengthX(), fy = scene.Camera.FocalLengthY();
		double cx = scene.Camera.PrincipalPointX(), cy = scene.Camera.PrincipalPointY();
		for (int k = 0; k < NumFrames; ++k)
		{
			Rigid3d labFromCam = CamFromLab(scene, k).Inverse();
			double wallMin = double.MaxValue;
			foreach ((double u, double v) in new[] { (0.0, 0.0), (Width, 0.0), (0.0, Height), (Width, Height) })
			{
				var ray = new Vector3d((u - cx) / fx, (v - cy) / fy, 1);
				wallMin = Math.Min(wallMin, 255 * SyntheticObjectScene.WallRadiance(labFromCam, ray));
			}

			await Assert.That(wallMin - objectMax).IsGreaterThanOrEqualTo(12 * NoiseSigma);
			double threshold = (objectMax + wallMin) / 2;

			byte[] pixels = scene.Frames[k].RowMajorData;
			byte[] mask = scene.Masks[k].RowMajorData;
			int checkedPixels = 0, disagreements = 0;
			for (int y = 1; y < Height - 1; ++y)
			{
				for (int x = 1; x < Width - 1; ++x)
				{
					bool uniform = true;
					for (int dy = -1; dy <= 1; ++dy)
					{
						for (int dx = -1; dx <= 1; ++dx)
						{
							uniform &= mask[(y + dy) * Width + x + dx] == mask[y * Width + x];
						}
					}

					if (!uniform)
					{
						continue;
					}

					++checkedPixels;
					bool inMask = mask[y * Width + x] == 255;
					for (int channel = 0; channel < 3; ++channel)
					{
						bool dark = pixels[(y * Width + x) * 3 + channel] < threshold;
						disagreements += dark != inMask ? 1 : 0;
					}
				}
			}

			await Assert.That(checkedPixels).IsGreaterThan(Width * Height / 2);
			await Assert.That(disagreements).IsEqualTo(0);
		}
	}

	// The pendulum speeds up, slows down and turns back.
	[Test]
	public async Task CSharpOnly_PendulumReversesDirection()
	{
		int reversals = 0;
		double previousStep = 0;
		for (int i = 1; i <= 200; ++i)
		{
			double step = SyntheticObjectScene.PendulumAngle(i / 200.0) - SyntheticObjectScene.PendulumAngle((i - 1) / 200.0);
			reversals += previousStep * step < 0 ? 1 : 0;
			previousStep = step;
		}

		await Assert.That(reversals).IsGreaterThanOrEqualTo(2);
	}

	private static string Hash(Bitmap bitmap) => Convert.ToHexString(SHA256.HashData(bitmap.RowMajorData));

	// Frame k's camera pose in the lab: its pose in the object-fixed world with the object's
	// known turn at that frame's time undone.
	private static Rigid3d CamFromLab(SyntheticObjectScene scene, int k) =>
		scene.CamFromWorld[k]
		* SyntheticObjectScene.LabFromObject(SyntheticObjectScene.FrameTime(k, scene.Frames.Count)).Inverse();

	private static TriangleBvh BuildBvh(SyntheticObjectScene scene)
	{
		var corners = new float[9 * scene.MeshTriangles.Count];
		var ids = new int[scene.MeshTriangles.Count];
		for (int f = 0; f < ids.Length; ++f)
		{
			PlyMeshFace face = scene.MeshTriangles[f];
			int[] vertexIdxs = [face.VertexIdx1, face.VertexIdx2, face.VertexIdx3];
			for (int c = 0; c < 3; ++c)
			{
				Vector3d v = scene.MeshVertices[vertexIdxs[c]];
				corners[9 * f + 3 * c] = (float)v.X;
				corners[9 * f + 3 * c + 1] = (float)v.Y;
				corners[9 * f + 3 * c + 2] = (float)v.Z;
			}

			ids[f] = f;
		}

		return new TriangleBvh(corners, ids);
	}
}
