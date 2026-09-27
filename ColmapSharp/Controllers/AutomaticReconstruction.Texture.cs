// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// AutomaticReconstructionController, texturing part: a C#-only last dense step
// (docs/CPP_DIVERGENCES.md entry 135). COLMAP's automatic reconstruction stops at the mesh;
// texturing is its separate mesh_texturer command (RunMeshTexturer in colmap/exe/mvs.cc). This
// runs that command's body on each model's dense workspace right after meshing
// (AutomaticReconstruction.Dense.cs): read the undistorted model (dense/<i>/sparse and
// dense/<i>/images), load each image as RGB at the model's size, read the mesh just written,
// run Mvs/TextureMapping.cs with default MeshTextureMappingOptions, and write mesh.ply (binary,
// with per-corner UVs and "comment TextureFile texture.png") into
// dense/<i>/<mesh name>-textured/.
//
// Differences from the command, all from where images live here:
// - The undistorted images come from the controller's InMemoryBitmapStore, not files.
// - texture.png is not encoded here (the library leaves image encoding to the host, as
//   BitmapSink.cs describes): the atlas goes to AutomaticReconstructionOptions.TextureSink
//   under that path, and every result is kept in memory in TexturedMeshes.
// - An empty atlas (no face seen by any view) is not handed to the sink; COLMAP's
//   Bitmap::Write of an empty bitmap would fail there.

using ColmapSharp.Mvs;
using ColmapSharp.Util;

namespace ColmapSharp.Controllers;

/// <summary>
/// The textured mesh of one dense model: the mesh, its per-corner UVs, per-face view ids and
/// texture atlas (<see cref="MeshTextureMappingResult"/>), and where mesh.ply was written.
/// </summary>
/// <param name="ModelIdx">The index of the sparse model (dense/&lt;ModelIdx&gt;).</param>
/// <param name="Mesh">The mesh that was textured (the Poisson or Delaunay output).</param>
/// <param name="Texture">The atlas, UVs (6 floats per face) and per-face view ids.</param>
/// <param name="MeshPath">
/// The textured mesh.ply. Its header names texture.png next to it, which the library does not
/// encode: the host writes <c>Texture.TextureAtlas</c> there (or supplies
/// <see cref="AutomaticReconstructionOptions.TextureSink"/>, which receives it under that path).
/// An empty atlas (no face seen by any view) has no texture.png.
/// </param>
public sealed record TexturedModelMesh(int ModelIdx, PlyMesh Mesh, MeshTextureMappingResult Texture, string MeshPath);

public sealed partial class AutomaticReconstructionController
{
	/// <summary>The Stage of the progress reports of mesh texturing (Done of 1000).</summary>
	public const string TexturingStage = "Texturing";

	/// <summary>The texture atlas file name, as COLMAP's mesh_texturer writes it.</summary>
	public const string TextureFileName = "texture.png";

	private readonly List<TexturedModelMesh> texturedMeshes = [];

	/// <summary>
	/// The textured mesh of every dense model this controller textured, in model order
	/// (empty unless <see cref="AutomaticReconstructionOptions.Texture"/> and Dense are on).
	/// </summary>
	public IReadOnlyList<TexturedModelMesh> TexturedMeshes => texturedMeshes;

	private bool NeedsTexturing(int modelIdx) =>
		options.Texture && !texturedMeshes.Any(m => m.ModelIdx == modelIdx);

	// The body of COLMAP's RunMeshTexturer with workspace_path = densePath,
	// input_path = meshingPath and output_path = dense/<i>/<mesh name>-textured.
	private void RunTexturing(int modelIdx, string densePath, string meshingPath, IBitmapSource images)
	{
		IProgress<double>? progress = Forward<double>(
			v => new ControllerProgress(TexturingStage, (int)(v * 1000), 1000, ""));
		progress?.Report(0);

		var model = new Model();
		model.ReadFromCOLMAP(densePath);
		foreach (Image image in model.Images)
		{
			CancellationToken.ThrowIfCancellationRequested();
			Sensor.Bitmap bitmap = images.Read(image.GetPath(), asRgb: true);
			if (bitmap.Width != image.GetWidth() || bitmap.Height != image.GetHeight())
			{
				bitmap.Rescale(image.GetWidth(), image.GetHeight());
			}

			image.SetBitmap(bitmap);
		}

		PlyMesh mesh = Ply.ReadPlyMesh(meshingPath).Mesh;

		MeshTextureMappingResult result = TextureMapping.MeshTextureMapping(
			mesh, model.Images, optionManager.MeshTextureMapping, progress, CancellationToken);

		string outputPath = Path.Combine(densePath, Path.GetFileNameWithoutExtension(meshingPath) + "-textured");
		Directory.CreateDirectory(outputPath);

		if (!result.TextureAtlas.IsEmpty)
		{
			options.TextureSink?.Write(Path.Combine(outputPath, TextureFileName), result.TextureAtlas);
		}

		string meshPath = Path.Combine(outputPath, "mesh.ply");
		var texturedMesh = new PlyTexturedMesh(mesh)
		{
			FaceUvs = [.. result.FaceUvs],
			TextureFile = TextureFileName,
		};
		Ply.WriteBinaryPlyMesh(meshPath, texturedMesh);

		texturedMeshes.Add(new TexturedModelMesh(modelIdx, mesh, result, meshPath));

		// MeshTextureMapping returns early without reaching 1 for an empty mesh or no views.
		progress?.Report(1.0);
	}
}
