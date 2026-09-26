// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// TextureMappingTests: colmap/mvs/texture_mapping_test.cc ported 1:1 (Suite_Name), testing
// ColmapSharp/Mvs/TextureMapping*.cs. Tier C (outcome): COLMAP's tests check view
// assignments, UV ranges and baked colors, not bits, and the same checks are made here.
// C#-only tests for the BVH and thread-count determinism are in TriangleBvhTests.cs.

using ColmapSharp.Mvs;
using ColmapSharp.Sensor;
using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class TextureMappingTests
{
	// Create a 2-triangle quad mesh (4 vertices, 2 faces).
	// Quad in XY plane at Z=0, from (0,0,0) to (1,1,0).
	internal static PlyMesh MakeTriangleMesh() => new()
	{
		Vertices =
		[
			new PlyMeshVertex(0.0f, 0.0f, 0.0f),
			new PlyMeshVertex(1.0f, 0.0f, 0.0f),
			new PlyMeshVertex(1.0f, 1.0f, 0.0f),
			new PlyMeshVertex(0.0f, 1.0f, 0.0f),
		],
		Faces = [new PlyMeshFace(0, 1, 2), new PlyMeshFace(0, 2, 3)],
	};

	private static Image MakeImage(string path, int width, int height, float[] t, BitmapColor<byte> color)
	{
		// K: focal length = width, principal point at center.
		float[] k = [width, 0, width / 2.0f, 0, height, height / 2.0f, 0, 0, 1];

		// R: 180° rotation around X-axis. This makes the camera look along -Z in world, with +Y up.
		float[] r = [1, 0, 0, 0, -1, 0, 0, 0, -1];

		var image = new Image(path, width, height, k, r, t);
		var bitmap = new Bitmap(width, height, asRgb: true);
		bitmap.Fill(color);
		image.SetBitmap(bitmap);
		return image;
	}

	// Create a camera looking down -Z at the XY plane.
	// Camera at (0.5, 0.5, 5); T = -R*C = [-0.5, 0.5, 5].
	internal static Image MakeTestImage(int width, int height, BitmapColor<byte> color) =>
		MakeImage("test_image.png", width, height, [-0.5f, 0.5f, 5.0f], color);

	// Create a camera where the mesh at Z=0 is behind the camera.
	// Camera at (0.5, 0.5, -5), looking along -Z in world (away from mesh).
	private static Image MakeBehindCameraImage(int width, int height) =>
		MakeImage("behind.png", width, height, [-0.5f, 0.5f, -5.0f], new BitmapColor<byte>(0));

	// Create a camera that views the mesh at a very grazing angle.
	// Camera nearly in-plane with XY mesh, at (0.5, 0.5, 0.01), looking along -Z.
	private static Image MakeGrazingAngleImage(int width, int height) =>
		MakeImage("grazing.png", width, height, [-0.5f, 0.5f, 0.01f], new BitmapColor<byte>(128));

	// Create a 4-face pyramid mesh.
	private static PlyMesh MakePyramidMesh() => new()
	{
		Vertices =
		[
			new PlyMeshVertex(0.0f, 0.0f, 0.0f),
			new PlyMeshVertex(1.0f, 0.0f, 0.0f),
			new PlyMeshVertex(1.0f, 1.0f, 0.0f),
			new PlyMeshVertex(0.0f, 1.0f, 0.0f),
			new PlyMeshVertex(0.5f, 0.5f, 1.0f),
		],
		Faces = [new PlyMeshFace(0, 1, 4), new PlyMeshFace(1, 2, 4), new PlyMeshFace(2, 3, 4), new PlyMeshFace(3, 0, 4)],
	};

	// Create a 12-face cube mesh (2 triangles per face).
	internal static PlyMesh MakeCubeMesh() => new()
	{
		Vertices =
		[
			new PlyMeshVertex(0, 0, 0),
			new PlyMeshVertex(1, 0, 0),
			new PlyMeshVertex(1, 1, 0),
			new PlyMeshVertex(0, 1, 0),
			new PlyMeshVertex(0, 0, 1),
			new PlyMeshVertex(1, 0, 1),
			new PlyMeshVertex(1, 1, 1),
			new PlyMeshVertex(0, 1, 1),
		],
		Faces =
		[
			// Front (Z=0), Back (Z=1), Left (X=0), Right (X=1), Bottom (Y=0), Top (Y=1).
			new PlyMeshFace(0, 2, 1), new PlyMeshFace(0, 3, 2),
			new PlyMeshFace(4, 5, 6), new PlyMeshFace(4, 6, 7),
			new PlyMeshFace(0, 7, 3), new PlyMeshFace(0, 4, 7),
			new PlyMeshFace(1, 6, 5), new PlyMeshFace(1, 2, 6),
			new PlyMeshFace(0, 5, 4), new PlyMeshFace(0, 1, 5),
			new PlyMeshFace(3, 6, 2), new PlyMeshFace(3, 7, 6),
		],
	};

	private static PlyMesh MakeSingleTriangleMesh() => new()
	{
		Vertices =
		[
			new PlyMeshVertex(0.0f, 0.0f, 0.0f),
			new PlyMeshVertex(1.0f, 0.0f, 0.0f),
			new PlyMeshVertex(0.5f, 1.0f, 0.0f),
		],
		Faces = [new PlyMeshFace(0, 1, 2)],
	};

	private static int CountNonBlack(MeshTextureMappingResult result)
	{
		int count = 0;
		for (int y = 0; y < result.AtlasHeight; ++y)
		{
			for (int x = 0; x < result.AtlasWidth; ++x)
			{
				BitmapColor<byte>? color = result.TextureAtlas.GetPixel(x, y);
				if (color is { } c && (c.R != 0 || c.G != 0 || c.B != 0))
				{
					++count;
				}
			}
		}

		return count;
	}

	[Test]
	public async Task MeshTextureMapping_EndToEnd()
	{
		PlyMesh mesh = MakeTriangleMesh();
		var red = new BitmapColor<byte>(200, 50, 50);
		List<Image> images = [MakeTestImage(256, 256, red)];

		var options = new MeshTextureMappingOptions
		{
			ApplyColorCorrection = false,
			ViewSelectionSmoothingIterations = 0,
			InpaintRadius = 0,
		};

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);

		// Check basic properties.
		await Assert.That(result.AtlasWidth).IsGreaterThan(0);
		await Assert.That(result.AtlasHeight).IsGreaterThan(0);
		await Assert.That(result.FaceUvs.Length).IsEqualTo(mesh.Faces.Count * 6);
		await Assert.That(result.FaceViewIds.Length).IsEqualTo(mesh.Faces.Count);

		// Both faces should be assigned to view 0.
		await Assert.That(result.FaceViewIds[0]).IsEqualTo(0);
		await Assert.That(result.FaceViewIds[1]).IsEqualTo(0);

		// All UVs should be in [0, 1].
		foreach (float uv in result.FaceUvs)
		{
			await Assert.That(uv).IsGreaterThanOrEqualTo(0.0f);
			await Assert.That(uv).IsLessThanOrEqualTo(1.0f);
		}

		// Atlas should contain approximately the red color. Check a pixel that should be baked.
		bool foundColoredPixel = false;
		for (int y = 0; y < result.AtlasHeight && !foundColoredPixel; ++y)
		{
			for (int x = 0; x < result.AtlasWidth && !foundColoredPixel; ++x)
			{
				BitmapColor<byte>? color = result.TextureAtlas.GetPixel(x, y);
				if (color is { } c && c.R > 100 && c.G < 150 && c.B < 150)
				{
					foundColoredPixel = true;
				}
			}
		}

		await Assert.That(foundColoredPixel).IsTrue();
	}

	[Test]
	public async Task MeshTextureMapping_EmptyMesh()
	{
		var mesh = new PlyMesh();
		List<Image> images = [MakeTestImage(64, 64, new BitmapColor<byte>(128))];
		var options = new MeshTextureMappingOptions { ApplyColorCorrection = false };

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);

		await Assert.That(result.AtlasWidth).IsEqualTo(0);
		await Assert.That(result.AtlasHeight).IsEqualTo(0);
		await Assert.That(result.FaceUvs).IsEmpty();
		await Assert.That(result.FaceViewIds).IsEmpty();
	}

	[Test]
	public async Task MeshTextureMapping_NoVisibleFaces()
	{
		PlyMesh mesh = MakeTriangleMesh();
		List<Image> images = [MakeBehindCameraImage(256, 256)];
		var options = new MeshTextureMappingOptions { ApplyColorCorrection = false, ViewSelectionSmoothingIterations = 0 };

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);

		// All faces should be unassigned.
		foreach (int view in result.FaceViewIds)
		{
			await Assert.That(view).IsEqualTo(-1);
		}
	}

	[Test]
	public async Task MeshTextureMapping_SingleFaceSingleView()
	{
		PlyMesh mesh = MakeSingleTriangleMesh();
		List<Image> images = [MakeTestImage(256, 256, new BitmapColor<byte>(100, 200, 50))];
		var options = new MeshTextureMappingOptions
		{
			ApplyColorCorrection = false,
			ViewSelectionSmoothingIterations = 0,
			InpaintRadius = 0,
		};

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);
		await Assert.That(result.FaceViewIds[0]).IsEqualTo(0);
	}

	[Test]
	public async Task MeshTextureMapping_SingleFaceTwoViews()
	{
		PlyMesh mesh = MakeSingleTriangleMesh();

		// Image 0: closer camera -> larger projected area.
		Image imgClose = MakeTestImage(256, 256, new BitmapColor<byte>(255, 0, 0));

		// Image 1: farther camera -> smaller projected area.
		// C = (0.5, 0.5, 20), T = -R*C = [-0.5, 0.5, 20]
		Image imgFar = MakeImage("far.png", 256, 256, [-0.5f, 0.5f, 20.0f], new BitmapColor<byte>(0, 255, 0));

		List<Image> images = [imgClose, imgFar];
		var options = new MeshTextureMappingOptions
		{
			ApplyColorCorrection = false,
			ViewSelectionSmoothingIterations = 0,
			InpaintRadius = 0,
		};

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);

		// Should pick the closer camera (view 0) as it has larger projected area.
		await Assert.That(result.FaceViewIds[0]).IsEqualTo(0);
	}

	[Test]
	public async Task MeshTextureMapping_FaceBehindCamera()
	{
		PlyMesh mesh = MakeSingleTriangleMesh();
		List<Image> images = [MakeBehindCameraImage(256, 256)];
		var options = new MeshTextureMappingOptions { ApplyColorCorrection = false, ViewSelectionSmoothingIterations = 0 };

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);
		await Assert.That(result.FaceViewIds[0]).IsEqualTo(-1);
	}

	[Test]
	public async Task MeshTextureMapping_GrazingAngleRejected()
	{
		PlyMesh mesh = MakeSingleTriangleMesh();
		List<Image> images = [MakeGrazingAngleImage(256, 256)];
		var options = new MeshTextureMappingOptions
		{
			ApplyColorCorrection = false,
			ViewSelectionSmoothingIterations = 0,

			// Set high threshold to reject grazing angles.
			MinCosNormalAngle = 0.9,
		};

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);
		await Assert.That(result.FaceViewIds[0]).IsEqualTo(-1);
	}

	[Test]
	public async Task MeshTextureMapping_NeighborSmoothing()
	{
		// Create a strip mesh centered around (0.5, 0.5) so all faces are visible.
		var mesh = new PlyMesh
		{
			Vertices =
			[
				new PlyMeshVertex(0.0f, 0.0f, 0.0f),
				new PlyMeshVertex(0.5f, 0.0f, 0.0f),
				new PlyMeshVertex(1.0f, 0.0f, 0.0f),
				new PlyMeshVertex(0.0f, 1.0f, 0.0f),
				new PlyMeshVertex(0.5f, 1.0f, 0.0f),
				new PlyMeshVertex(1.0f, 1.0f, 0.0f),
			],
			Faces = [new PlyMeshFace(0, 1, 4), new PlyMeshFace(0, 4, 3), new PlyMeshFace(1, 2, 5), new PlyMeshFace(1, 5, 4)],
		};

		// Both views see all faces.
		Image img0 = MakeTestImage(512, 512, new BitmapColor<byte>(255, 0, 0));
		Image img1 = MakeTestImage(512, 512, new BitmapColor<byte>(0, 255, 0));
		List<Image> images = [img0, img1];

		var options = new MeshTextureMappingOptions
		{
			ApplyColorCorrection = false,
			ViewSelectionSmoothingIterations = 3,
			InpaintRadius = 0,
		};

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);

		// After smoothing, all faces should be assigned since they're all visible.
		int assigned = result.FaceViewIds.Count(v => v >= 0);
		await Assert.That(assigned).IsEqualTo(mesh.Faces.Count);

		// All assigned faces should have the same view (smoothing promotes uniformity).
		int firstView = result.FaceViewIds[0];
		for (int i = 1; i < result.FaceViewIds.Length; ++i)
		{
			await Assert.That(result.FaceViewIds[i]).IsEqualTo(firstView);
		}
	}

	[Test]
	public async Task MeshTextureMapping_CubeAdjacency()
	{
		// Verify the cube mesh produces faces with proper adjacency by running the full pipeline.
		PlyMesh mesh = MakeCubeMesh();

		// Camera looking at front face.
		List<Image> images = [MakeTestImage(512, 512, new BitmapColor<byte>(200, 100, 50))];

		var options = new MeshTextureMappingOptions
		{
			ApplyColorCorrection = false,
			ViewSelectionSmoothingIterations = 0,
			InpaintRadius = 0,
			MinVisibleVertices = 1,
		};

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);

		// At least the front-facing triangles should be assigned.
		await Assert.That(result.FaceViewIds.Length).IsEqualTo(12);
		await Assert.That(result.FaceViewIds.Count(v => v >= 0)).IsGreaterThan(0);
	}

	[Test]
	public async Task MeshTextureMapping_UVRange()
	{
		PlyMesh mesh = MakeTriangleMesh();
		List<Image> images = [MakeTestImage(256, 256, new BitmapColor<byte>(128))];
		var options = new MeshTextureMappingOptions
		{
			ApplyColorCorrection = false,
			ViewSelectionSmoothingIterations = 0,
			InpaintRadius = 0,
		};

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);

		foreach (float uv in result.FaceUvs)
		{
			await Assert.That(uv).IsGreaterThanOrEqualTo(0.0f);
			await Assert.That(uv).IsLessThanOrEqualTo(1.0f);
		}
	}

	[Test]
	public async Task MeshTextureMapping_BakeSolidColor()
	{
		PlyMesh mesh = MakeSingleTriangleMesh();
		var srcColor = new BitmapColor<byte>(180, 90, 45);
		List<Image> images = [MakeTestImage(256, 256, srcColor)];
		var options = new MeshTextureMappingOptions
		{
			ApplyColorCorrection = false,
			ViewSelectionSmoothingIterations = 0,
			InpaintRadius = 0,
		};

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);

		// Find baked pixels and verify they match the source color.
		int bakedCount = 0;
		for (int y = 0; y < result.AtlasHeight; ++y)
		{
			for (int x = 0; x < result.AtlasWidth; ++x)
			{
				BitmapColor<byte>? color = result.TextureAtlas.GetPixel(x, y);
				await Assert.That(color.HasValue).IsTrue();
				BitmapColor<byte> c = color!.Value;
				if (c.R != 0 || c.G != 0 || c.B != 0)
				{
					// Due to bilinear interpolation, allow some tolerance.
					await Assert.That(Math.Abs(c.R - srcColor.R)).IsLessThanOrEqualTo(5);
					await Assert.That(Math.Abs(c.G - srcColor.G)).IsLessThanOrEqualTo(5);
					await Assert.That(Math.Abs(c.B - srcColor.B)).IsLessThanOrEqualTo(5);
					++bakedCount;
				}
			}
		}

		await Assert.That(bakedCount).IsGreaterThan(0);
	}

	[Test]
	public async Task MeshTextureMapping_InpaintFillsNearbyPixels()
	{
		PlyMesh mesh = MakeSingleTriangleMesh();
		List<Image> images = [MakeTestImage(256, 256, new BitmapColor<byte>(200, 100, 50))];
		var options = new MeshTextureMappingOptions
		{
			ApplyColorCorrection = false,
			ViewSelectionSmoothingIterations = 0,
			InpaintRadius = 3,
		};

		MeshTextureMappingResult resultInpaint = TextureMapping.MeshTextureMapping(mesh, images, options);

		// Without inpainting for comparison.
		options.InpaintRadius = 0;
		MeshTextureMappingResult resultNoInpaint = TextureMapping.MeshTextureMapping(mesh, images, options);

		// Count non-black pixels. Inpainted result should have more.
		await Assert.That(CountNonBlack(resultInpaint)).IsGreaterThan(CountNonBlack(resultNoInpaint));
	}

	[Test]
	public async Task MeshTextureMapping_InpaintDoesNotOverwriteBaked()
	{
		PlyMesh mesh = MakeSingleTriangleMesh();
		var srcColor = new BitmapColor<byte>(200, 100, 50);
		List<Image> images = [MakeTestImage(256, 256, srcColor)];

		// Run without inpainting.
		var options = new MeshTextureMappingOptions
		{
			ApplyColorCorrection = false,
			ViewSelectionSmoothingIterations = 0,
			InpaintRadius = 0,
		};
		MeshTextureMappingResult resultBase = TextureMapping.MeshTextureMapping(mesh, images, options);

		// Run with inpainting.
		options.InpaintRadius = 5;
		MeshTextureMappingResult resultInpaint = TextureMapping.MeshTextureMapping(mesh, images, options);

		// Check that baked pixels are unchanged.
		for (int y = 0; y < resultBase.AtlasHeight; ++y)
		{
			for (int x = 0; x < resultBase.AtlasWidth; ++x)
			{
				BitmapColor<byte>? baseColor = resultBase.TextureAtlas.GetPixel(x, y);
				BitmapColor<byte>? inpaintColor = resultInpaint.TextureAtlas.GetPixel(x, y);
				await Assert.That(baseColor.HasValue).IsTrue();
				await Assert.That(inpaintColor.HasValue).IsTrue();
				BitmapColor<byte> b = baseColor!.Value;
				if (b.R != 0 || b.G != 0 || b.B != 0)
				{
					await Assert.That(inpaintColor!.Value).IsEqualTo(b);
				}
			}
		}
	}

	[Test]
	public async Task MeshTextureMapping_GlobalColorCorrectionNoSeams()
	{
		// Single view, single region -> no seams -> no correction applied.
		PlyMesh mesh = MakeTriangleMesh();
		List<Image> images = [MakeTestImage(256, 256, new BitmapColor<byte>(150))];
		var options = new MeshTextureMappingOptions
		{
			ApplyColorCorrection = true,
			ViewSelectionSmoothingIterations = 0,
			InpaintRadius = 0,
		};

		MeshTextureMappingResult resultWithCc = TextureMapping.MeshTextureMapping(mesh, images, options);

		options.ApplyColorCorrection = false;
		MeshTextureMappingResult resultWithoutCc = TextureMapping.MeshTextureMapping(mesh, images, options);

		// Results should be identical since there are no seams.
		await Assert.That(resultWithCc.AtlasWidth).IsEqualTo(resultWithoutCc.AtlasWidth);
		await Assert.That(resultWithCc.AtlasHeight).IsEqualTo(resultWithoutCc.AtlasHeight);
		for (int y = 0; y < resultWithCc.AtlasHeight; ++y)
		{
			for (int x = 0; x < resultWithCc.AtlasWidth; ++x)
			{
				BitmapColor<byte>? c1 = resultWithCc.TextureAtlas.GetPixel(x, y);
				BitmapColor<byte>? c2 = resultWithoutCc.TextureAtlas.GetPixel(x, y);
				await Assert.That(c1.HasValue).IsTrue();
				await Assert.That(c2.HasValue).IsTrue();
				await Assert.That(c1!.Value).IsEqualTo(c2!.Value);
			}
		}
	}

	[Test]
	public async Task MeshTextureMapping_PyramidMultiFace()
	{
		PlyMesh mesh = MakePyramidMesh();
		List<Image> images = [MakeTestImage(512, 512, new BitmapColor<byte>(100, 150, 200))];
		var options = new MeshTextureMappingOptions
		{
			ApplyColorCorrection = false,
			ViewSelectionSmoothingIterations = 0,
			InpaintRadius = 0,
			MinVisibleVertices = 1,
		};

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(mesh, images, options);
		await Assert.That(result.FaceViewIds.Length).IsEqualTo(4);
		await Assert.That(result.FaceUvs.Length).IsEqualTo(24);

		// At least some faces should be assigned.
		await Assert.That(result.FaceViewIds.Count(v => v >= 0)).IsGreaterThan(0);
	}
}
