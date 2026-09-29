// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of what the demo says about which photos a run placed (demo/ColmapDemo/PhotoPlacement.cs,
// ColmapDemoApp.Placement.cs): the summary for one model and for several, which model is shown
// (the one with the most photos, which need not be the mapper's first), the marks on the photo
// list, and which runs give their photos one shared camera. The models are small hand-built
// reconstructions standing in for a real run's, so no pipeline runs here.

using ColmapSharp.Geometry;
using ColmapSharp.Scene;
using ColmapSharp.Sensor;
using MatterHackers.Agg;
using MatterHackers.Agg.Platform;

namespace ColmapDemo.Tests;

[NotInParallel("AggUiThread")]
public class PlacementTests
{
	private sealed class HeadlessOs : IOsInformationProvider
	{
		public OSType OperatingSystem => OSType.Mac;

		public MatterHackers.Agg.Point2D DesktopSize => new MatterHackers.Agg.Point2D(1920, 1080);

		public long PhysicalMemory => 8L << 30;
	}

	// A model with the named images registered, plus the unregistered ones (in the model but
	// without a pose, as the mapper leaves images it tried and could not place).
	private static void AddModel(ReconstructionManager models, IEnumerable<string> registered, IEnumerable<string> unregistered = null)
	{
		Reconstruction model = models.Get(models.Add());
		model.AddCameraWithTrivialRig(Camera.CreateFromModelId(1, CameraModelId.SimpleRadial, 100, 100, 100));
		uint id = 1;
		foreach (string name in registered)
		{
			var image = new Image { ImageId = id++, Name = name };
			image.SetCameraId(1);
			model.AddImageWithTrivialFrame(image, new Rigid3d());
		}

		foreach (string name in unregistered ?? Array.Empty<string>())
		{
			var image = new Image { ImageId = id++, Name = name };
			image.SetCameraId(1);
			model.AddImageWithTrivialFrame(image);
		}
	}

	private static SessionResult FakeResult(IReadOnlyList<string> names, ReconstructionManager models) =>
		new SessionResult { Placement = PhotoPlacement.FromModels(names, models, PhotoPlacement.LargestModelIndex(models)) };

	[Test]
	public async Task AllPhotosPlacedInOneModelSaysSo()
	{
		string[] names = { "a.png", "b.png", "c.png", "d.png", "e.png", "f.png" };
		var models = new ReconstructionManager();
		AddModel(models, names);

		PhotoPlacement placement = FakeResult(names, models).Placement;

		await Assert.That(placement.StatusText).IsEqualTo("Placed 6 of 6 photos.");
		await Assert.That(placement.States.All(s => s == PlacementState.Shown)).IsTrue();
	}

	[Test]
	public async Task SeveralModelsShowTheLargestAndSayHowManyGroups()
	{
		// The mapper's first model is the smaller one here, so "the largest" is model 1.
		string[] names = { "a.png", "b.png", "c.png", "d.png", "e.png", "f.png", "g.png" };
		var models = new ReconstructionManager();
		AddModel(models, new[] { "a.png", "b.png" }, unregistered: new[] { "g.png" });
		AddModel(models, new[] { "c.png", "d.png", "e.png", "f.png" });

		await Assert.That(PhotoPlacement.LargestModelIndex(models)).IsEqualTo(1);
		PhotoPlacement placement = FakeResult(names, models).Placement;

		await Assert.That(placement.StatusText).IsEqualTo(
			"Placed 6 of 7 photos. The photos formed 2 separate groups; showing the largest (4 photos). "
			+ "Photos that could not be placed probably didn't overlap enough or had too little texture.");
		await Assert.That(placement.States.SequenceEqual(new[]
		{
			PlacementState.OtherGroup, PlacementState.OtherGroup,
			PlacementState.Shown, PlacementState.Shown, PlacementState.Shown, PlacementState.Shown,
			PlacementState.NotPlaced,
		})).IsTrue();
	}

	[Test]
	public async Task NoModelPlacesNothing()
	{
		string[] names = { "a.png", "b.png", "c.png" };
		var models = new ReconstructionManager();

		await Assert.That(PhotoPlacement.LargestModelIndex(models)).IsEqualTo(-1);
		await Assert.That(FakeResult(names, models).Placement.StatusText).IsEqualTo(
			"Placed 0 of 3 photos. " + PhotoPlacement.UnplacedAdvice);
	}

	[Test]
	public async Task ThePanelMarksPhotosThatWereNotPlaced()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		var app = new ColmapDemoApp(fileDropSupported: false);
		string dir = Directory.CreateTempSubdirectory("ColmapDemoTests").FullName;
		string[] names = { "a.png", "b.png", "c.png", "d.png" };
		app.AddPhotos(names.Select(n => Path.Combine(dir, n)));
		var models = new ReconstructionManager();
		AddModel(models, new[] { "a.png", "b.png", "c.png" });
		AddModel(models, new[] { "d.png" });
		SessionResult result = FakeResult(names, models);

		app.ShowPlacement(result.Placement, app.PhotoPaths);

		await Assert.That(app.PlacementText).StartsWith("Placed 4 of 4 photos. The photos formed 2 separate groups; showing the largest (3 photos).");
		await Assert.That(app.PhotoListLines).IsEquivalentTo(new[] { "a.png", "b.png", "c.png", "d.png — in a smaller group" });

		// A new run starts from the plain list.
		app.ShowPlacement(null, app.PhotoPaths);
		await Assert.That(app.PlacementText).IsEqualTo(string.Empty);
		await Assert.That(app.PhotoListLines).IsEquivalentTo(names);
		Directory.Delete(dir, recursive: true);
	}

	[Test]
	public async Task OnlyOneVideosFramesShareACamera()
	{
		string[] clip = { "/f/clip-0.png", "/f/clip-1.png", "/f/clip-2.png" };
		string[] other = { "/g/other-0.png", "/g/other-1.png", "/g/other-2.png" };

		await Assert.That(ColmapDemoApp.UsesSingleCamera(clip, new[] { clip })).IsTrue();
		await Assert.That(ColmapDemoApp.UsesSingleCamera(clip, new[] { other, clip })).IsTrue();

		// Photos from files, a photo added to a video's frames, or two videos' frames: a camera each.
		await Assert.That(ColmapDemoApp.UsesSingleCamera(new[] { "/p/a.jpg", "/p/b.jpg", "/p/c.jpg" }, Array.Empty<string[]>())).IsFalse();
		await Assert.That(ColmapDemoApp.UsesSingleCamera(clip.Append("/p/a.jpg").ToList(), new[] { clip })).IsFalse();
		await Assert.That(ColmapDemoApp.UsesSingleCamera(clip.Concat(other).ToList(), new[] { clip, other })).IsFalse();
		await Assert.That(ColmapDemoApp.UsesSingleCamera(Array.Empty<string>(), new[] { clip })).IsFalse();
	}
}
