// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AutomaticReconstructionTests (continued): C#-only tests, not ports. COLMAP's
// automatic_reconstruction_test.cc turns dense reconstruction off (it needs CUDA there); the
// dense stages run on the CPU here, so this pins that the whole controller, photos in, turns a
// textured scene into a non-empty fused cloud and mesh.
//
// Why a scene of its own: the ported test's SynthesizeImages draws flat image-space patches
// on a black background. Those drive SIFT fine, but PatchMatch has almost nothing to match
// (black is textureless, and the patches are not perspective-consistent surfaces), so its
// filtered depth maps keep only a few percent of the pixels and fusion finds no point seen
// consistently in MinNumPixels (5) images. Here a solid-noise-textured sphere in front of a
// textured wall is ray-traced into a few small views, so every pixel is photo-consistent.
//
// Why Delaunay meshing: a cloud this small (about 2000 points) is too sparse for COLMAP's
// Poisson defaults (depth 13, trim 10) - the surface trimmer removes every triangle. That is
// COLMAP's behavior, not a port bug: on the 4000-point fused.ply of 240x180 views of this
// scene, the port and pycolmap 4.2.0's poisson_meshing both give an empty mesh at trim 10,
// while pycolmap keeps 26325 faces at trim 0. Delaunay meshing has no density trim.
//
// The same run pins the C#-only texturing step (docs/CPP_DIVERGENCES.md entry 135): the mesh
// comes out textured, with one UV in [0, 1] per face corner, most faces assigned a view, a
// non-empty atlas in memory and at the host's sink, and mesh.ply written next to it.
//
// The same run pins the progress contract: a host grouping reports by Stage sees only the
// controller's own stage headings, whatever the sub-stages call themselves.
//
// CSharpOnly_MaskSourceRoutesByReservedKey pins the dense stages' image/mask routing without
// a reconstruction: a workspace whose relative path starts with "masks" must still read its
// undistorted images, not the host's masks.

using ColmapSharp.Controllers;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class AutomaticReconstructionTests
{
	[Test]
	public async Task CSharpOnly_DenseTexturedSceneGivesFusedPointsAndMesh()
	{
		const int NumViews = 4;
		string testDir = CreateTestDir();
		string workspacePath = Path.Combine(testDir, "workspace");
		Directory.CreateDirectory(workspacePath);
		var images = new InMemoryImageSource();
		for (int i = 0; i < NumViews; ++i)
		{
			images.Add($"view{i}.png", RenderTexturedScene(i, NumViews, width: 200, height: 150));
		}

		var textureSink = new InMemoryBitmapStore();
		var options = new AutomaticReconstructionOptions
		{
			WorkspacePath = workspacePath,
			Images = images,
			Data = AutomaticReconstructionOptions.DataType.Individual,
			Quality = AutomaticReconstructionOptions.QualityLevel.Low,
			Dense = true,
			RandomSeed = 1,
			Mesher = AutomaticReconstructionOptions.MesherType.Delaunay,
			TextureSink = textureSink,
		};

		var reconstructionManager = new ReconstructionManager();
		var stages = new StageCollector();
		var controller = new AutomaticReconstructionController(options, reconstructionManager)
		{
			Progress = stages,
		};
		controller.Setup();
		controller.Run();

		int size = reconstructionManager.Size;
		int numRegImages = size == 0 ? 0 : reconstructionManager.Get(0).NumRegImages;
		string densePath = Path.Combine(workspacePath, "dense", "0");
		string fusedPath = Path.Combine(densePath, "fused.ply");
		string meshPath = Path.Combine(densePath, "meshed-delaunay.ply");
		int numFusedPoints = File.Exists(fusedPath) ? Ply.ReadPly(fusedPath).Count : -1;
		int numMeshFaces = File.Exists(meshPath) ? Ply.ReadPlyMesh(meshPath).Mesh.Faces.Count : -1;

		// The textured mesh, in memory, at the sink and on disk.
		int numTextured = controller.TexturedMeshes.Count;
		TexturedModelMesh? textured = numTextured == 1 ? controller.TexturedMeshes[0] : null;
		int texturedFaces = textured?.Mesh.Faces.Count ?? -1;
		int numUvs = textured?.Texture.FaceUvs.Length ?? -1;
		bool uvsInUnitRange = textured is not null && textured.Texture.FaceUvs.All(uv => uv >= 0 && uv <= 1);
		int numAssignedFaces = textured?.Texture.FaceViewIds.Count(v => v >= 0) ?? -1;
		int atlasWidth = textured?.Texture.TextureAtlas.Width ?? 0;
		int atlasHeight = textured?.Texture.TextureAtlas.Height ?? 0;
		string texturedDir = Path.Combine(densePath, "meshed-delaunay-textured");
		string sinkKey = Path.Combine(texturedDir, AutomaticReconstructionController.TextureFileName);
		bool sinkHasAtlas = textureSink.Exists(sinkKey) && ReferenceEquals(textureSink.Get(sinkKey), textured?.Texture.TextureAtlas);
		PlyTexturedMesh? texturedPly = File.Exists(Path.Combine(texturedDir, "mesh.ply"))
			? Ply.ReadPlyMesh(Path.Combine(texturedDir, "mesh.ply"))
			: null;

		try
		{
			Directory.Delete(testDir, recursive: true);
		}
		catch (IOException)
		{
			// A leftover temp folder is harmless.
		}

		await Assert.That(size).IsEqualTo(1);
		await Assert.That(numRegImages).IsEqualTo(NumViews);
		await Assert.That(numFusedPoints).IsGreaterThan(1000);
		await Assert.That(numMeshFaces).IsGreaterThan(0);
		await Assert.That(numTextured).IsEqualTo(1);
		await Assert.That(textured!.ModelIdx).IsEqualTo(0);
		await Assert.That(texturedFaces).IsEqualTo(numMeshFaces);
		await Assert.That(numUvs).IsEqualTo(6 * texturedFaces);
		await Assert.That(uvsInUnitRange).IsTrue();
		await Assert.That(numAssignedFaces).IsGreaterThan(texturedFaces / 2);
		await Assert.That(atlasWidth).IsGreaterThan(0);
		await Assert.That(atlasHeight).IsGreaterThan(0);
		await Assert.That(sinkHasAtlas).IsTrue();
		await Assert.That(texturedPly).IsNotNull();
		await Assert.That(texturedPly!.TextureFile).IsEqualTo(AutomaticReconstructionController.TextureFileName);
		await Assert.That(texturedPly.FaceUvs.Count).IsEqualTo(numUvs);
		await Assert.That(string.Join(" | ", stages.Stages())).IsEqualTo(string.Join(" | ", new[]
		{
			FeatureExtraction.ExtractionStage,
			FeatureMatching.MatchingStage,
			AutomaticReconstructionController.SparseStage,
			AutomaticReconstructionController.DenseStage,
			AutomaticReconstructionController.FusionStage,
			AutomaticReconstructionController.MeshingStage,
			AutomaticReconstructionController.TexturingStage,
		}));
	}

	// Pins divergence 134's fix of an upstream ordering bug: COLMAP sets image_names and then
	// ModifyForVideoData's ResetOptions(false) wipes them, so a video run there processes
	// every image. Here only the selected (consecutive) views are extracted and registered.
	[Test]
	public async Task CSharpOnly_VideoDataKeepsImageNamesSelection()
	{
		const int NumViews = 6;
		string testDir = CreateTestDir();
		string workspacePath = Path.Combine(testDir, "workspace");
		Directory.CreateDirectory(workspacePath);
		var images = new InMemoryImageSource();
		for (int i = 0; i < NumViews; ++i)
		{
			images.Add($"view{i}.png", RenderTexturedScene(i, NumViews, width: 200, height: 150));
		}

		string[] selected = ["view1.png", "view2.png", "view3.png", "view4.png"];
		var options = new AutomaticReconstructionOptions
		{
			WorkspacePath = workspacePath,
			Images = images,
			ImageNames = [.. selected],
			Data = AutomaticReconstructionOptions.DataType.Video,
			Quality = AutomaticReconstructionOptions.QualityLevel.Low,
			Dense = false,
			RandomSeed = 1,
		};

		var database = new InMemoryDatabase();
		var reconstructionManager = new ReconstructionManager();
		var controller = new AutomaticReconstructionController(options, reconstructionManager, database);
		controller.Setup();
		controller.Run();

		string databaseNames = string.Join(",", database.ReadAllImages().Select(image => image.Name).Order());
		int size = reconstructionManager.Size;
		string registeredNames = size == 0 ? "" : string.Join(",", reconstructionManager.Get(0).RegImageIds()
			.Select(id => reconstructionManager.Get(0).Image(id).Name).Order());

		try
		{
			Directory.Delete(testDir, recursive: true);
		}
		catch (IOException)
		{
			// A leftover temp folder is harmless.
		}

		await Assert.That(databaseNames).IsEqualTo(string.Join(",", selected));
		await Assert.That(size).IsEqualTo(1);
		await Assert.That(registeredNames).IsEqualTo(string.Join(",", selected));
	}

	[Test]
	public async Task CSharpOnly_MaskSourceRoutesByReservedKey()
	{
		// The undistorter's key for image a.png of a workspace at the relative path masks/ws.
		string imageKey = Path.Combine("masks", "ws", "dense", "0", "images", "a.png");
		var images = new InMemoryBitmapStore();
		images.Write(imageKey, new Bitmap(4, 3, asRgb: true));
		var masks = new InMemoryImageSource();
		masks.Add("a.png.png", new Bitmap(2, 2, asRgb: false));
		var source = new AutomaticReconstructionController.WorkspaceBitmapSource(images, masks);

		// The fusion's key for a.png's mask (StereoFusion.InitFusedPixelMask).
		string maskKey = Path.Combine(AutomaticReconstructionController.MaskRoot, "a.png.png");

		await Assert.That(source.Exists(imageKey)).IsTrue();
		await Assert.That(source.Read(imageKey, asRgb: true).Width).IsEqualTo(4);
		await Assert.That(source.Exists(maskKey)).IsTrue();
		await Assert.That(source.Read(maskKey, asRgb: false).Width).IsEqualTo(2);
		await Assert.That(source.Exists(Path.Combine("masks", "a.png.png"))).IsFalse();
	}

	// Records the distinct Stage names in first-report order. Synchronous and locked: the
	// stages report from worker threads.
	private sealed class StageCollector : IProgress<ControllerProgress>
	{
		private readonly List<string> stages = [];

		public void Report(ControllerProgress value)
		{
			lock (stages)
			{
				if (!stages.Contains(value.Stage))
				{
					stages.Add(value.Stage);
				}
			}
		}

		public List<string> Stages()
		{
			lock (stages)
			{
				return [.. stages];
			}
		}
	}

	// Ray-traces view viewIdx of numViews: cameras 4 units from the origin on an arc about the
	// y axis (12 degrees apart, alternating slightly up and down), looking at a unit sphere at
	// the origin in front of the wall z = 3. Both surfaces carry the same solid noise texture,
	// sampled at the 3D hit point so it is the same surface in every view; 3x3 supersampling
	// keeps the fine octaves from aliasing.
	private static Bitmap RenderTexturedScene(int viewIdx, int numViews, int width, int height)
	{
		double angle = (viewIdx - (numViews - 1) / 2.0) * 12.0 * Math.PI / 180.0;
		double[] center = [4 * Math.Sin(angle), viewIdx % 2 == 0 ? 0.3 : -0.3, -4 * Math.Cos(angle)];
		double[] forward = Normalized([-center[0], -center[1], -center[2]]);
		double[] right = Normalized(Cross([0, 1, 0], forward));
		double[] down = Cross(forward, right);
		double focal = 1.2 * Math.Max(width, height);

		var bitmap = new Bitmap(width, height, asRgb: true);
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				double sum = 0;
				for (int sy = 0; sy < 3; ++sy)
				{
					for (int sx = 0; sx < 3; ++sx)
					{
						double u = (x + (sx + 0.5) / 3 - width / 2.0) / focal;
						double v = (y + (sy + 0.5) / 3 - height / 2.0) / focal;
						double[] dir = Normalized([
							forward[0] + u * right[0] + v * down[0],
							forward[1] + u * right[1] + v * down[1],
							forward[2] + u * right[2] + v * down[2]]);
						sum += Albedo(Trace(center, dir));
					}
				}

				byte value = (byte)Math.Clamp(sum / 9 * 255, 0, 255);
				bitmap.SetPixel(x, y, new BitmapColor<byte>(value, value, value));
			}
		}

		return bitmap;
	}

	// The first hit of the ray with the unit sphere, else with the wall z = 3.
	private static double[] Trace(double[] origin, double[] dir)
	{
		double b = Dot(origin, dir);
		double c = Dot(origin, origin) - 1;
		double discriminant = b * b - c;
		double t = discriminant >= 0 ? -b - Math.Sqrt(discriminant) : (3 - origin[2]) / dir[2];
		return [origin[0] + t * dir[0], origin[1] + t * dir[1], origin[2] + t * dir[2]];
	}

	// Four octaves of value noise in [0, 1], stretched for contrast.
	private static double Albedo(double[] p)
	{
		double sum = 0;
		double weight = 0.5;
		double frequency = 4;
		for (int octave = 0; octave < 4; ++octave)
		{
			sum += weight * ValueNoise(p[0] * frequency, p[1] * frequency, p[2] * frequency);
			weight *= 0.5;
			frequency *= 2;
		}

		return Math.Clamp((sum / 0.9375 - 0.5) * 2 + 0.5, 0, 1);
	}

	private static double ValueNoise(double x, double y, double z)
	{
		int xi = (int)Math.Floor(x), yi = (int)Math.Floor(y), zi = (int)Math.Floor(z);
		double fx = Smooth(x - xi), fy = Smooth(y - yi), fz = Smooth(z - zi);
		double Lerp(double a, double b, double t) => a + (b - a) * t;
		double Corner(int dx, int dy, int dz) => Hash(xi + dx, yi + dy, zi + dz);
		return Lerp(
			Lerp(Lerp(Corner(0, 0, 0), Corner(1, 0, 0), fx), Lerp(Corner(0, 1, 0), Corner(1, 1, 0), fx), fy),
			Lerp(Lerp(Corner(0, 0, 1), Corner(1, 0, 1), fx), Lerp(Corner(0, 1, 1), Corner(1, 1, 1), fx), fy),
			fz);
	}

	private static double Smooth(double t) => t * t * (3 - 2 * t);

	// A lattice value in [0, 1] from an integer hash (deterministic on every platform).
	private static double Hash(int x, int y, int z)
	{
		uint h = unchecked((uint)x * 73856093u ^ (uint)y * 19349663u ^ (uint)z * 83492791u);
		h ^= h >> 13;
		h = unchecked(h * 0x5bd1e995u);
		h ^= h >> 15;
		return (h & 0xFFFFFF) / (double)0xFFFFFF;
	}

	private static double Dot(double[] a, double[] b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

	private static double[] Cross(double[] a, double[] b) =>
		[a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

	private static double[] Normalized(double[] a)
	{
		double norm = Math.Sqrt(Dot(a, a));
		return [a[0] / norm, a[1] / norm, a[2] / norm];
	}
}
