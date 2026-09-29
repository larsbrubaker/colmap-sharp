// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of the Settings panel (demo/ColmapDemo/ColmapDemoApp.Settings.cs, DemoSettings.cs,
// SessionSettings.cs): each control reaches the right AutomaticReconstructionOptions value through
// the real widgets and the real mapping, a known focal length fixes the camera, Reset restores the
// defaults, and a run locks the panel. No window: the theme's OS question gets a stand-in.

using ColmapSharp.Controllers;
using MatterHackers.Agg;
using MatterHackers.Agg.Platform;
using MatterHackers.Agg.UI;
using static ColmapSharp.Controllers.AutomaticReconstructionOptions;

namespace ColmapDemo.Tests;

[NotInParallel("AggUiThread")]
public class SettingsPanelTests
{
	private sealed class HeadlessOs : IOsInformationProvider
	{
		public OSType OperatingSystem => OSType.Mac;

		public Point2D DesktopSize => new Point2D(1920, 1080);

		public long PhysicalMemory => 8L << 30;
	}

	private static ColmapDemoApp NewApp()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		return new ColmapDemoApp(fileDropSupported: false);
	}

	private static T Find<T>(GuiWidget app, string name)
		where T : GuiWidget => (T)app.FindDescendant(name);

	private static AutomaticReconstructionOptions Options(ColmapDemoApp app, int width = 800, int height = 600, bool oneVideo = false)
	{
		app.CommitSettingsEdits();
		var options = new AutomaticReconstructionOptions();
		app.Settings.ToSessionSettings(photosAreOneVideo: oneVideo).ApplyTo(options, width, height);
		return options;
	}

	[Test]
	public async Task DefaultsMatchTheDemosRunBeforeThePanel()
	{
		AutomaticReconstructionOptions options = Options(NewApp());
		await Assert.That(options.Subject).IsEqualTo(SubjectType.Scene);
		await Assert.That(options.Data).IsEqualTo(DataType.Individual);
		await Assert.That(options.Quality).IsEqualTo(QualityLevel.Low);
		await Assert.That(options.SingleCamera).IsFalse();
		await Assert.That(options.CameraModel).IsEqualTo("SIMPLE_RADIAL");
		await Assert.That(options.CameraParams).IsEqualTo(string.Empty);
		await Assert.That(options.BaRefineFocalLength).IsTrue();
		await Assert.That(options.Dense).IsTrue();
		await Assert.That(options.Texture).IsTrue();
		await Assert.That(options.Mesher).IsEqualTo(MesherType.Poisson);
		await Assert.That(options.Mapper).IsEqualTo(MapperType.Incremental);
		await Assert.That(options.PoissonMeshing.Depth).IsEqualTo(11);
		await Assert.That(options.PoissonMeshing.Trim).IsEqualTo(5.0);
		await Assert.That(options.RandomSeed).IsEqualTo(-1);
	}

	[Test]
	public async Task EachControlMapsToItsOption()
	{
		ColmapDemoApp app = NewApp();
		Find<GuiWidget>(app, "One object Card").InvokeClick();
		Find<SegmentedControl>(app, "Fast Setting").SelectedIndex = 2;
		Find<DropDownList>(app, "Largest photo size Setting").SelectedIndex = Array.IndexOf(ColmapDemoApp.PhotoSizes, 1600);
		Find<CheckBox>(app, "All photos are from the same camera Setting").Checked = true;
		Find<ThemedNumberEdit>(app, "Frames to take from a video Setting").Value = 60;
		Find<CheckBox>(app, "Follow points from frame to frame Setting").Checked = true;
		Find<CheckBox>(app, "Place missed frames from their outlines Setting").Checked = true;
		Find<CheckBox>(app, "Paint the photos' colours onto the mesh Setting").Checked = false;
		Find<DropDownList>(app, "Surface method Setting").SelectedIndex = 1;
		Find<ThemedNumberEdit>(app, "Surface detail (Poisson depth, 9–11 suits most) Setting").Value = 9;
		Find<ThemedNumberEdit>(app, "Trim loose surface (Poisson trim, 0 keeps all) Setting").Value = 0;
		Find<DropDownList>(app, "Camera model Setting").SelectedIndex = 3;
		Find<DropDownList>(app, "How photos are placed Setting").SelectedIndex = 2;
		Find<DropDownList>(app, "How photos are matched Setting").SelectedIndex = 1;
		Find<ThemedNumberEdit>(app, "Random seed (-1 = default) Setting").Value = 7;
		Find<CheckBox>(app, "Use the graphics card for depth Setting").Checked = false;
		Find<CheckBox>(app, "Keep working files after the run Setting").Checked = true;

		AutomaticReconstructionOptions options = Options(app, oneVideo: true);
		await Assert.That(options.Subject).IsEqualTo(SubjectType.Object);
		await Assert.That(options.Quality).IsEqualTo(QualityLevel.High);
		await Assert.That(app.Settings.ToSessionSettings(false).MaxImageSize).IsEqualTo(1600);
		await Assert.That(options.SingleCamera).IsTrue();
		await Assert.That(app.TargetFramesPerVideo).IsEqualTo(60);
		await Assert.That(options.VideoTracking).IsTrue();
		await Assert.That(options.SilhouettePlacement).IsTrue();
		await Assert.That(options.FramesAreTimeOrdered == true).IsTrue();
		await Assert.That(options.Texture).IsFalse();
		await Assert.That(options.Mesher).IsEqualTo(MesherType.Delaunay);
		await Assert.That(options.PoissonMeshing.Depth).IsEqualTo(9);
		await Assert.That(options.PoissonMeshing.Trim).IsEqualTo(0.0);
		await Assert.That(options.CameraModel).IsEqualTo("PINHOLE");
		await Assert.That(options.Mapper).IsEqualTo(MapperType.Hierarchical);
		await Assert.That(options.Data).IsEqualTo(DataType.Video);
		await Assert.That(options.RandomSeed).IsEqualTo(7);
		await Assert.That(app.Settings.ToSessionSettings(false, gpu: null).ComputeDevice).IsNull();
		await Assert.That(app.Settings.UseGpu).IsFalse();
		await Assert.That(app.Settings.ToSessionSettings(false).KeepWorkspace).IsTrue();

		// The quick camera check stops at the sparse points.
		Find<SegmentedControl>(app, "Full mesh Setting").SelectedIndex = 1;
		await Assert.That(Options(app).Dense).IsFalse();
	}

	[Test]
	public async Task VideoOptionsFollowTheLibrarysConditions()
	{
		ColmapDemoApp app = NewApp();
		var tracking = Find<CheckBox>(app, "Follow points from frame to frame Setting");
		var outlines = Find<CheckBox>(app, "Place missed frames from their outlines Setting");
		app.Settings.VideoTracking = true;
		app.Settings.SilhouettePlacement = true;

		// Photos (not one video's frames): neither applies, and neither reaches the library.
		await Assert.That(tracking.Enabled).IsFalse();
		await Assert.That(outlines.Enabled).IsFalse();
		AutomaticReconstructionOptions options = Options(app);
		await Assert.That(options.FramesAreTimeOrdered).IsNull();
		await Assert.That(options.VideoTracking).IsFalse();
		await Assert.That(options.SilhouettePlacement).IsFalse();
		await Assert.That(app.Settings.Summary(app.Settings.ToSessionSettings(false)).First(r => r.Label == "Video").Value).DoesNotContain("follow points");

		// One video's frames: time-ordered, matched exhaustively; tracking runs even for a scene.
		options = Options(app, oneVideo: true);
		await Assert.That(options.FramesAreTimeOrdered == true).IsTrue();
		await Assert.That(options.Data).IsEqualTo(DataType.Individual);
		await Assert.That(options.VideoTracking).IsTrue();
		await Assert.That(options.SilhouettePlacement).IsFalse();

		// Outline placement also needs One object.
		Find<GuiWidget>(app, "One object Card").InvokeClick();
		await Assert.That(Options(app, oneVideo: true).SilhouettePlacement).IsTrue();
		await Assert.That(app.Settings.Summary(app.Settings.ToSessionSettings(true)).First(r => r.Label == "Video").Value).Contains("place from outlines");

		// Choosing recording-order matching makes photos time-ordered too.
		app.Settings.Matching = DataType.Video;
		await Assert.That(Options(app).VideoTracking).IsTrue();
	}

	[Test]
	public async Task KnownFocalSetsCameraParamsAndFixesOnlyTheFocal()
	{
		ColmapDemoApp app = NewApp();
		Find<CheckBox>(app, "I know this camera's focal length Setting").Checked = true;
		Find<ThemedNumberEdit>(app, "Number Setting").Value = 27;

		// COLMAP's 35 mm rule (Bitmap.Exif): f = f35 / 43.27 * diagonal; 800x600 has a 1000 px
		// diagonal. The principal point is the centre; distortion starts at 0 and is still refined.
		AutomaticReconstructionOptions options = Options(app, 800, 600);
		double[] p = options.CameraParams.Split(',').Select(v => double.Parse(v, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
		await Assert.That(p[0]).IsEqualTo(27 / 43.27 * 1000);
		await Assert.That(p.Skip(1)).IsEquivalentTo(new[] { 400.0, 300.0, 0.0 });
		await Assert.That(options.BaRefineFocalLength).IsFalse();
		await Assert.That(options.BaRefineExtraParams).IsTrue();

		// A known focal is one camera's.
		await Assert.That(options.SingleCamera).IsTrue();

		app.Settings.FocalUnit = FocalUnit.Pixels;
		app.Settings.FocalLength = 700;
		options = Options(app, 800, 600);
		await Assert.That(options.CameraParams).IsEqualTo("700, 400, 300, 0");
		await Assert.That(options.BaRefineFocalLength).IsFalse();
	}

	[Test]
	public async Task KnownFocalRefusesPhotosOfDifferentSizesBeforeThePipeline()
	{
		var settings = new SessionSettings { KnownFocal35mm = 26, WorkspaceRoot = Directory.CreateTempSubdirectory("ColmapDemoTests").FullName };
		var images = new InMemoryImageSource();
		images.Add("a.png", new ColmapSharp.Sensor.Bitmap(40, 30, asRgb: true));
		images.Add("b.png", new ColmapSharp.Sensor.Bitmap(30, 40, asRgb: true));
		var session = new ReconstructionSession(settings);
		string stage = null;
		session.ProgressChanged += p => stage ??= p.Stage;

		InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(() => session.RunAsync(images, CancellationToken.None));

		await Assert.That(error.Message).IsEqualTo(SessionSettings.MixedSizesKnownFocalMessage);
		await Assert.That(stage).IsNull();
		await Assert.That(SessionSettings.KnownFocalSizeError(new[] { (40, 30), (40, 30) })).IsNull();
		Directory.Delete(settings.WorkspaceRoot, recursive: true);
	}

	[Test]
	public async Task AllAtOnceIsUnavailableWhileTheFocalIsKnown()
	{
		ColmapDemoApp app = NewApp();
		var placed = Find<DropDownList>(app, "How photos are placed Setting");
		await Assert.That(placed.MenuItems[1].Enabled).IsTrue();
		placed.SelectedIndex = 1;
		await Assert.That(app.Settings.Mapper).IsEqualTo(MapperType.Global);

		Find<CheckBox>(app, "I know this camera's focal length Setting").Checked = true;

		await Assert.That(placed.MenuItems[1].Enabled).IsFalse();
		await Assert.That(app.Settings.Mapper).IsEqualTo(MapperType.Incremental);
		await Assert.That(placed.SelectedIndex).IsEqualTo(0);
	}

	[Test]
	public async Task AClearedNumberFieldKeepsItsValue()
	{
		ColmapDemoApp app = NewApp();
		var frames = Find<ThemedNumberEdit>(app, "Frames to take from a video Setting");
		frames.Text = string.Empty;

		app.CommitSettingsEdits();

		await Assert.That(app.Settings.FramesPerVideo).IsEqualTo(VideoFrameSampler.DefaultTargetFrames);
		await Assert.That(frames.Value).IsEqualTo((double)VideoFrameSampler.DefaultTargetFrames);
	}

	[Test]
	public async Task TheQuickCheckSaysItStoppedAtThePoints()
	{
		var result = new SessionResult { SparsePoints = new ColoredPoint[1234] };
		await Assert.That(ColmapDemoApp.FinishedStatus(result, denseRequested: false)).IsEqualTo($"Camera check done: {1234:N0} points.");
		await Assert.That(ColmapDemoApp.FinishedStatus(result, denseRequested: true)).IsEqualTo("Done, but no mesh came out. Try more photos with more overlap.");
	}

	[Test]
	public async Task ReadMeshFindsTheDelaunayMesh()
	{
		string workspace = Directory.CreateTempSubdirectory("ColmapDemoTests").FullName;
		string folder = Path.Combine(workspace, "dense", "0");
		Directory.CreateDirectory(folder);
		var mesh = new ColmapSharp.Util.PlyMesh();
		mesh.Vertices.AddRange(new[] { new ColmapSharp.Util.PlyMeshVertex(0, 0, 0), new ColmapSharp.Util.PlyMeshVertex(1, 0, 0), new ColmapSharp.Util.PlyMeshVertex(0, 1, 0) });
		mesh.Faces.Add(new ColmapSharp.Util.PlyMeshFace(0, 1, 2));
		ColmapSharp.Util.Ply.WriteBinaryPlyMesh(Path.Combine(folder, "meshed-delaunay.ply"), new ColmapSharp.Util.PlyTexturedMesh(mesh));

		await Assert.That(ReconstructionSession.ReadMesh(workspace, 0, MesherType.Delaunay)?.Faces.Count).IsEqualTo(1);
		await Assert.That(ReconstructionSession.ReadMesh(workspace, 0, MesherType.Poisson)).IsNull();
		Directory.Delete(workspace, recursive: true);
	}

	[Test]
	public async Task ResetRestoresTheHeadsDefaults()
	{
		ColmapDemoApp app = NewApp();
		app.MaxImageSize = 640;
		Find<GuiWidget>(app, "One object Card").InvokeClick();
		Find<SegmentedControl>(app, "Fast Setting").SelectedIndex = 3;
		Find<CheckBox>(app, "Keep working files after the run Setting").Checked = !app.Settings.KeepWorkspace;
		app.Settings.MaxImageSize = 2400;

		Find<GuiWidget>(app, "Reset Settings Button").InvokeClick();

		await Assert.That(app.Settings.Subject).IsEqualTo(SubjectType.Scene);
		await Assert.That(app.Settings.Quality).IsEqualTo(QualityLevel.Low);
		await Assert.That(app.Settings.MaxImageSize).IsEqualTo(640);
		await Assert.That(app.Settings.KeepWorkspace).IsEqualTo(new DemoSettings().KeepWorkspace);
		await Assert.That(Find<SegmentedControl>(app, "Fast Setting").SelectedIndex).IsEqualTo(0);
	}

	[Test]
	public async Task TheSegmentedChoicesAreThemeStripsSharingTheWidth()
	{
		ColmapDemoApp app = NewApp();
		var window = new SystemWindow(1200, 800);
		window.AddChild(app);
		app.SettingsOpen = true;
		window.PerformLayout();

		foreach (string name in new[] { "Fast Setting", "Full mesh Setting" })
		{
			var control = Find<SegmentedControl>(app, name);
			double[] widths = control.Segments.Select(s => s.Width).ToArray();

			await Assert.That(control.Style).IsEqualTo(SegmentedStyle.Strip).Because("the default theme draws segmented choices as a strip");
			await Assert.That(widths.Sum()).IsEqualTo(control.Width).Within(0.5).Because($"{name} stretches across the settings column");
			await Assert.That(widths.Max() - widths.Min()).IsLessThanOrEqualTo(1.0).Because($"{name}'s segments share its width equally");
		}

		window.Close();
	}

	[Test]
	public async Task ARunLocksTheSettingsAndShowsWhatItUses()
	{
		ColmapDemoApp app = NewApp();
		string dir = Directory.CreateTempSubdirectory("ColmapDemoTests").FullName;
		app.AddPhotos(new[] { "a.png", "b.png", "c.png" }.Select(n => Path.Combine(dir, n)));
		app.StartRun();

		await Assert.That(app.SettingsLocked).IsTrue();
		await Assert.That(app.SettingsControls.All(c => !c.Enabled)).IsTrue();
		await Assert.That(Find<GuiWidget>(app, "Reset Settings Button").Enabled).IsFalse();
		await Assert.That(app.RunSummaryRows.Select(r => r.Label)).Contains("Camera");

		app.RequestCancel();
		Directory.Delete(dir, recursive: true);
	}
}
