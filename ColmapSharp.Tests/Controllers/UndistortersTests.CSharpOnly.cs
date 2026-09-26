// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// UndistortersTests (continued): C#-only tests, not ports. They pin what
// undistorters_test.cc does not: the text outputs against pycolmap 4.2.0's (the expected
// strings were produced by pycolmap.undistort_images on the same synthetic dataset, written
// to disk), the MatterCAD additions (the in-memory hand-off to Mvs.Workspace, progress,
// cancellation through a CancellationToken), and that the thread count does not change the
// images.

using ColmapSharp.Controllers;
using ColmapSharp.ImageProcessing;
using ColmapSharp.Mathematics;
using ColmapSharp.Mvs;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class UndistortersTests
{
	[Test]
	public async Task CSharpOnly_ColmapTextOutputsMatchPycolmap()
	{
		string outputPath = Path.Combine(CreateTestDir(), "output");
		// pycolmap synthesized the scene in a fresh process, whose PRNG starts from the
		// default seed 0, the seed PrngTestIsolation starts every test with.
		(Reconstruction reconstruction, InMemoryImageSource images) = CreateSyntheticReconstructionWithBitmaps();
		new ColmapUndistorter(
			new ColmapUndistorter.Options(), new UndistortCameraOptions(), reconstruction, images, outputPath,
			new InMemoryBitmapStore()).Run();

		await Assert.That(File.ReadAllText(Path.Combine(outputPath, "stereo", "patch-match.cfg"))).IsEqualTo(
			"camera000001_frame000000.png\n__auto__, 20\ncamera000001_frame000001.png\n__auto__, 20\n");
		await Assert.That(File.ReadAllText(Path.Combine(outputPath, "stereo", "fusion.cfg"))).IsEqualTo(
			"camera000001_frame000000.png\ncamera000001_frame000001.png\n");
		await Assert.That(File.ReadAllText(Path.Combine(outputPath, "run-colmap-geometric.sh"))).IsEqualTo(
			"# You must set $COLMAP_EXE_PATH to \n" +
			"# the directory containing the COLMAP executables.\n" +
			"$COLMAP_EXE_PATH/colmap patch_match_stereo \\\n" +
			"  --workspace_path \".\" \\\n" +
			"  --workspace_format COLMAP \\\n" +
			"  --PatchMatchStereo.max_image_size 2000 \\\n" +
			"  --PatchMatchStereo.geom_consistency true\n" +
			"$COLMAP_EXE_PATH/colmap stereo_fusion \\\n" +
			"  --workspace_path \".\" \\\n" +
			"  --workspace_format COLMAP \\\n" +
			"  --input_type geometric \\\n" +
			"  --output_path \"./fused.ply\" \\\n" +
			"$COLMAP_EXE_PATH/colmap poisson_mesher \\\n" +
			"  --input_path \"./fused.ply\" \\\n" +
			"  --output_path \"./meshed-poisson.ply\" \\\n" +
			"$COLMAP_EXE_PATH/colmap delaunay_mesher \\\n" +
			"  --input_path \"./\" \\\n" +
			"  --input_type dense \\\n" +
			"  --output_path \"./meshed-delaunay.ply\" \\\n");
	}

	[Test]
	public async Task CSharpOnly_PmvsTextOutputsMatchPycolmap()
	{
		string outputPath = Path.Combine(CreateTestDir(), "pmvs_output");
		// pycolmap synthesized the scene in a fresh process, whose PRNG starts from the
		// default seed 0, the seed PrngTestIsolation starts every test with.
		(Reconstruction reconstruction, InMemoryImageSource images) = CreateSyntheticReconstructionWithBitmaps();
		new PmvsUndistorter(
			new PmvsUndistorter.Options(), new UndistortCameraOptions(), reconstruction, images, outputPath,
			new InMemoryBitmapStore()).Run();

		await Assert.That(File.ReadAllText(Path.Combine(outputPath, "pmvs", "txt", "00000000.txt"))).IsEqualTo(
			"CONTOUR\n" +
			"1333.35 84.7126 -324.291 2508.8\n" +
			"-107.032 1110.05 -732.363 1881.6\n" +
			"0.456093 0.724214 0.517197 5\n");
		await Assert.That(File.ReadAllText(Path.Combine(outputPath, "pmvs", "vis.dat"))).IsEqualTo(
			"VISDATA\n2\n0 0\n1 0\n");
		string optionAll = File.ReadAllText(Path.Combine(outputPath, "pmvs", "option-all"));
		await Assert.That(optionAll).EndsWith("quad 2.0\ntimages 2 0 1\noimages 0\n");
		await Assert.That(File.ReadAllText(Path.Combine(outputPath, "run-cmvs-colmap-photometric.sh"))).Contains(
			"    rm -rf \"$workspace_path/stereo\"\n" +
			"    $COLMAP_EXE_PATH/colmap patch_match_stereo \\\n" +
			"      --workspace_path \"pmvs\" \\\n" +
			"      --workspace_format PMVS \\\n" +
			"      --pmvs_option_name $option_name \\\n" +
			"      --PatchMatchStereo.max_image_size 2000 \\\n" +
			"      --PatchMatchStereo.geom_consistency false\n" +
			"    $COLMAP_EXE_PATH/colmap stereo_fusion \\\n" +
			"      --workspace_path \"pmvs\" \\\n" +
			"      --workspace_format PMVS \\\n" +
			"      --pmvs_option_name $option_name \\\n" +
			"      --input_type photometric \\\n" +
			"      --output_path \"pmvs/$option_name-fused.ply\" \\\n" +
			"    $COLMAP_EXE_PATH/colmap poisson_mesher \\\n" +
			"      --input_path \"pmvs/$option_name-fused.ply\" \\\n" +
			"      --output_path \"pmvs/$option_name-meshed-poisson.ply\" \\\n" +
			"    $COLMAP_EXE_PATH/colmap delaunay_mesher \\\n" +
			"      --input_path \"pmvs/$option_name-\" \\\n" +
			"      --input_type dense \\\n" +
			"      --output_path \"pmvs/$option_name-meshed-delaunay.ply\" \\\n" +
			"done\n");
	}

	[Test]
	public async Task CSharpOnly_UndistortedWorkspaceFeedsMvsWithoutFiles()
	{
		string outputPath = Path.Combine(CreateTestDir(), "output");
		(Reconstruction reconstruction, InMemoryImageSource images) = CreateSyntheticReconstructionWithBitmaps(
			numImages: 3, imageExtension: ".jpg");
		var store = new InMemoryBitmapStore();
		var progress = new List<ControllerProgress>();
		var undistorter = new ColmapUndistorter(
			new ColmapUndistorter.Options { JpegQuality = 50 }, new UndistortCameraOptions(), reconstruction, images,
			outputPath, store)
		{
			Progress = new SynchronousProgress(progress),
		};
		undistorter.Run();

		await Assert.That(undistorter.UndistortedReconstruction).IsNotNull();
		await Assert.That(progress.Select(p => p.Done)).IsEquivalentTo([1, 2, 3]);
		await Assert.That(progress.All(p => p.Total == 3 && p.Stage == "Image undistortion")).IsTrue();

		// The images never touch the disk, yet the MVS workspace finds and reads them.
		await Assert.That(Directory.EnumerateFiles(Path.Combine(outputPath, "images")).Any()).IsFalse();
		var model = new Model();
		model.ReadFromCOLMAP(undistorter.UndistortedReconstruction!, Path.Combine(outputPath, "images"));
		var workspace = new Workspace(
			new Workspace.Options { WorkspacePath = outputPath, WorkspaceFormat = "COLMAP" }, model, store);
		for (int i = 0; i < model.Images.Count; ++i)
		{
			await Assert.That(workspace.HasBitmap(i)).IsTrue();
			ColmapSharp.Sensor.Bitmap bitmap = store.Read(workspace.GetBitmapPath(i), asRgb: true);
			await Assert.That(bitmap.Width).IsEqualTo(model.Images[i].GetWidth());
			await Assert.That(bitmap.Height).IsEqualTo(model.Images[i].GetHeight());
			await Assert.That(store.Get(workspace.GetBitmapPath(i)).GetMetaData("Compression")).IsEqualTo("jpeg:50");
		}
	}

	[Test]
	public async Task CSharpOnly_CancellationTokenStopsWithoutThrowing()
	{
		string outputPath = Path.Combine(CreateTestDir(), "output");
		(Reconstruction reconstruction, InMemoryImageSource images) = CreateSyntheticReconstructionWithBitmaps(numImages: 4);
		var undistorter = new ColmapUndistorter(
			new ColmapUndistorter.Options(), new UndistortCameraOptions(), reconstruction, images, outputPath,
			new InMemoryBitmapStore());
		using var cancellation = new CancellationTokenSource();
		cancellation.Cancel();
		undistorter.CancellationToken = cancellation.Token;
		undistorter.Run();

		await Assert.That(undistorter.UndistortedReconstruction).IsNull();
		await Assert.That(File.Exists(Path.Combine(outputPath, "stereo", "patch-match.cfg"))).IsFalse();
	}

	[Test]
	public async Task CSharpOnly_ThreadCountDoesNotChangeImages()
	{
		(Reconstruction reconstruction, _) = CreateSyntheticReconstructionWithBitmaps(numImages: 5);

		// A different gradient per image, so a wrong index-to-image mapping or a race between
		// workers changes which pixels land under which name.
		var images = new InMemoryImageSource();
		int imageIdx = 0;
		foreach (ColmapSharp.Scene.Image image in reconstruction.Images.Values)
		{
			var bitmap = new ColmapSharp.Sensor.Bitmap(100, 100, true);
			for (int y = 0; y < 100; ++y)
			{
				for (int x = 0; x < 100; ++x)
				{
					bitmap.SetPixel(x, y, new ColmapSharp.Sensor.BitmapColor<byte>(
						(byte)((x * 7 + y * 3 + imageIdx * 50) % 256), (byte)(x * 2 + imageIdx), (byte)(y * 2)));
				}
			}

			images.Add(image.Name, bitmap);
			++imageIdx;
		}

		var outputs = new List<(string OutputPath, InMemoryBitmapStore Store)>();
		foreach (int numThreads in new[] { 1, 3 })
		{
			var store = new InMemoryBitmapStore();
			string outputPath = Path.Combine(CreateTestDir(), "output");
			new ColmapUndistorter(
				new ColmapUndistorter.Options { NumThreads = numThreads }, new UndistortCameraOptions(), reconstruction,
				images, outputPath, store).Run();
			outputs.Add((outputPath, store));
		}

		await Assert.That(outputs[0].Store.Paths.Count).IsEqualTo(5);
		await Assert.That(outputs[1].Store.Paths.Count).IsEqualTo(5);
		var distinctOutputs = new HashSet<string>();
		foreach (ColmapSharp.Scene.Image image in reconstruction.Images.Values)
		{
			byte[] single = outputs[0].Store.Get(Path.Combine(outputs[0].OutputPath, "images", image.Name)).RowMajorData;
			byte[] multi = outputs[1].Store.Get(Path.Combine(outputs[1].OutputPath, "images", image.Name)).RowMajorData;
			await Assert.That(multi.SequenceEqual(single)).IsTrue();
			distinctOutputs.Add(Convert.ToBase64String(single));
		}

		// The inputs really differ, so the comparison above can tell images apart.
		await Assert.That(distinctOutputs.Count).IsEqualTo(5);
	}

	// Progress<T> posts to the thread pool; the tests need the reports in order and done.
	private sealed class SynchronousProgress(List<ControllerProgress> reports) : IProgress<ControllerProgress>
	{
		public void Report(ControllerProgress value)
		{
			lock (reports)
			{
				reports.Add(value);
			}
		}
	}
}
