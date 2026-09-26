// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// WorkspaceTests: colmap/mvs/workspace_test.cc ported 1:1, testing ColmapSharp/Mvs/
// Workspace.cs and CachedWorkspace.cs. The gtest TEST_P suite ParameterizedWorkspaceTests
// is instantiated for Workspace and CachedWorkspace; here each case takes the workspace kind
// as an [Arguments] parameter (WorkspaceTests_ParameterizedWorkspaceTests_Name).
//
// The fixture writes the image with Bitmap::Write in COLMAP; the library does no image
// decoding, so the same black 10 x 5 RGB bitmap is served by an in-memory IBitmapSource
// under the same path (images/<name>). The sparse model and the depth and normal maps are
// written to disk as in COLMAP. Tier A.

using ColmapSharp.Mvs;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;

using Image = ColmapSharp.Mvs.Image;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Mvs;

public class WorkspaceTests
{
	private sealed class Fixture
	{
		public Fixture()
		{
			TempDir = MvsTestUtils.CreateTestDir();
			Directory.CreateDirectory(Path.Combine(TempDir, "sparse"));
			Directory.CreateDirectory(Path.Combine(TempDir, "images"));
			Directory.CreateDirectory(Path.Combine(TempDir, "stereo"));
			Directory.CreateDirectory(Path.Combine(TempDir, "stereo", "depth_maps"));
			Directory.CreateDirectory(Path.Combine(TempDir, "stereo", "normal_maps"));

			var options = new SyntheticDatasetOptions
			{
				NumRigs = 1,
				NumCamerasPerRig = 1,
				NumFramesPerRig = 1,
				CameraWidth = 10,
				CameraHeight = 5,
			};
			var reconstruction = new Reconstruction();
			Synthetic.SynthesizeDataset(options, reconstruction);
			reconstruction.Write(Path.Combine(TempDir, "sparse"));

			ImageName = reconstruction.Image(1).Name;

			var depthMap = new Mat<float>(options.CameraWidth, options.CameraHeight, 1);
			depthMap.Fill(1.0f);
			depthMap.Write(Path.Combine(TempDir, "stereo", "depth_maps", ImageName + ".geometric.bin"));

			var normalMap = new Mat<float>(options.CameraWidth, options.CameraHeight, 3);
			normalMap.Fill(1.0f);
			normalMap.Write(Path.Combine(TempDir, "stereo", "normal_maps", ImageName + ".geometric.bin"));

			var bitmap = new Bitmap(options.CameraWidth, options.CameraHeight, asRgb: true);
			bitmap.Fill(new BitmapColor<byte>(0, 0, 0));
			Bitmaps.Add(Path.Combine(TempDir, "images", ImageName), bitmap);
		}

		public string TempDir { get; }

		public string ImageName { get; }

		public MvsTestUtils.InMemoryBitmapSource Bitmaps { get; } = new();

		public Workspace.Options GetOptions() => new()
		{
			WorkspacePath = TempDir,
			WorkspaceFormat = "COLMAP",
			InputType = "geometric",
		};

		public Workspace Create(bool cached, Workspace.Options options) =>
			cached ? new CachedWorkspace(options, Bitmaps) : new Workspace(options, Bitmaps);
	}

	[Test]
	[Arguments(false)]
	[Arguments(true)]
	public async Task WorkspaceTests_ParameterizedWorkspaceTests_GetData(bool cached)
	{
		var fixture = new Fixture();
		Workspace workspace = fixture.Create(cached, fixture.GetOptions());
		Model model = workspace.GetModel();
		await Assert.That(model.Images.Count).IsEqualTo(1);
		workspace.Load([fixture.ImageName]);
		await Assert.That(workspace.HasBitmap(0)).IsTrue();
		await Assert.That(workspace.GetBitmapPath(0)).Contains(fixture.ImageName);
		await Assert.That(workspace.GetBitmap(0).IsEmpty).IsFalse();
		await Assert.That(workspace.HasDepthMap(0)).IsTrue();
		await Assert.That(workspace.GetDepthMapPath(0)).Contains(fixture.ImageName);
		await Assert.That(workspace.GetDepthMap(0).GetNumBytes()).IsGreaterThan(0);
		await Assert.That(workspace.HasNormalMap(0)).IsTrue();
		await Assert.That(workspace.GetNormalMapPath(0)).Contains(fixture.ImageName);
		await Assert.That(workspace.GetNormalMap(0).GetNumBytes()).IsGreaterThan(0);
	}

	[Test]
	[Arguments(false)]
	[Arguments(true)]
	public async Task WorkspaceTests_ParameterizedWorkspaceTests_MaxImageSize(bool cached)
	{
		var fixture = new Fixture();
		Workspace.Options options = fixture.GetOptions();
		options.MaxImageSize = 4;
		Workspace workspace = fixture.Create(cached, options);
		workspace.Load([fixture.ImageName]);
		await Assert.That(workspace.GetModel().Images.Count).IsEqualTo(1);
		await Assert.That(workspace.GetModel().Images[0].GetWidth()).IsEqualTo(4);
		await Assert.That(workspace.GetModel().Images[0].GetHeight()).IsEqualTo(2);
		await Assert.That(workspace.GetBitmap(0).Width).IsEqualTo(4);
		await Assert.That(workspace.GetBitmap(0).Height).IsEqualTo(2);
		await Assert.That(workspace.GetDepthMap(0).GetWidth()).IsEqualTo(4);
		await Assert.That(workspace.GetDepthMap(0).GetHeight()).IsEqualTo(2);
		await Assert.That(workspace.GetNormalMap(0).GetWidth()).IsEqualTo(4);
		await Assert.That(workspace.GetNormalMap(0).GetHeight()).IsEqualTo(2);
	}

	[Test]
	[Arguments(false)]
	[Arguments(true)]
	public async Task WorkspaceTests_ParameterizedWorkspaceTests_Load(bool cached)
	{
		var fixture = new Fixture();
		Workspace workspace = fixture.Create(cached, fixture.GetOptions());
		workspace.Load([fixture.ImageName]);
		await Assert.That(workspace.GetModel().Images.Count).IsEqualTo(1);
	}

	// C#-only: a workspace built over an in-memory Model downsizes its own copy of the
	// images, so two workspaces with different MaxImageSize over one Model neither see each
	// other's sizes nor change the caller's Model.
	[Test]
	[Arguments(false)]
	[Arguments(true)]
	public async Task Workspace_InMemoryModelIsNotMutated(bool cached)
	{
		float[] k = [100, 0, 50, 0, 100, 25, 0, 0, 1];
		float[] r = [1, 0, 0, 0, 1, 0, 0, 0, 1];
		float[] t = [0, 0, 0];
		var model = new Model();
		model.Images.Add(new Image("img0.jpg", 100, 50, k, r, t));
		var bitmaps = new MvsTestUtils.InMemoryBitmapSource();

		Workspace Create(int maxImageSize)
		{
			var options = new Workspace.Options { MaxImageSize = maxImageSize, InputType = "geometric" };
			return cached ? new CachedWorkspace(options, model, bitmaps) : new Workspace(options, model, bitmaps);
		}

		Workspace workspace40 = Create(40);
		Workspace workspace20 = Create(20);

		await Assert.That(workspace40.GetModel().Images[0].GetWidth()).IsEqualTo(40);
		await Assert.That(workspace20.GetModel().Images[0].GetWidth()).IsEqualTo(20);
		await Assert.That(model.Images[0].GetWidth()).IsEqualTo(100);
		await Assert.That(model.Images[0].GetHeight()).IsEqualTo(50);
		await Assert.That(model.Images[0].GetK()[0]).IsEqualTo(100.0f);
	}
}
