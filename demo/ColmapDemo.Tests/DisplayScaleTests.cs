// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of the demo's display scale (demo/ColmapDemo/DemoDisplayScale.cs, ColmapDemoApp.Rescale.cs):
// the heads compose GuiWidget.DeviceScale with agg's UiScale before building ColmapDemoApp, so on a 2x
// (Retina) display the panel and its fonts come out at twice the device pixels, and a move to a
// display with another scale rebuilds the app there without losing what the user had set up. The
// policy itself (the window size and its clamp, unusable scales, the startup echo, coalescing) is
// agg's and is pinned by agg-sharp's UiScaleTests.

using System.Diagnostics;
using MatterHackers.Agg;
using MatterHackers.Agg.Platform;
using MatterHackers.Agg.UI;
using static ColmapSharp.Controllers.AutomaticReconstructionOptions;

namespace ColmapDemo.Tests;

// Keyless: these write the process-wide GuiWidget.DeviceScale, which every widget built anywhere in
// the run reads, so nothing may run beside them (agg-sharp's DeviceScaleWriterIsolationTests rule).
[NotInParallel]
public class DisplayScaleTests
{
	private sealed class HeadlessOs : IOsInformationProvider
	{
		public OSType OperatingSystem => OSType.Mac;

		public Point2D DesktopSize => new Point2D(3840, 2160);

		public long PhysicalMemory => 8L << 30;
	}

	[Test]
	public async Task ThePanelAndItsFontsScaleWithTheDisplay()
	{
		await WithScaleRestored(async () =>
		{
			(double panelWidth, double titleEm, double hintEm) atOne = BuildAt(1);
			(double panelWidth, double titleEm, double hintEm) atTwo = BuildAt(2);

			await Assert.That(atOne.panelWidth).IsEqualTo(280);
			await Assert.That(atTwo.panelWidth).IsEqualTo(560);
			await Assert.That(atOne.titleEm).IsEqualTo(16);
			await Assert.That(atTwo.titleEm).IsEqualTo(32);
			await Assert.That(atTwo.hintEm).IsEqualTo(2 * atOne.hintEm);
		});
	}

	[Test]
	public async Task AScaleChangeRebuildsAtTheNewScaleAndKeepsTheSetup()
	{
		await WithScaleRestored(async () =>
		{
			StartAt(1);
			var window = new SystemWindow(1200, 800);
			var app = new ColmapDemoApp(fileDropSupported: false);
			window.AddChild(app);
			DemoDisplayScale.Follow(window, app);

			string[] photos = { "a.png", "b.png", "c.png" };
			app.AddPhotos(photos.Select(n => Path.Combine(Path.GetTempPath(), n)));
			((SegmentedControl)app.FindDescendant("Fast Setting")).SelectedIndex = 2;
			app.FindDescendant("One object Card").InvokeClick();
			app.SettingsOpen = true;
			GuiWidget panelBefore = app.Children[0];

			MoveToDisplay(window, 2);

			await Assert.That(GuiWidget.DeviceScale).IsEqualTo(2.0);
			await Assert.That(app.Children[0]).IsNotSameReferenceAs(panelBefore).Because("the widgets are built again at the new scale");
			await Assert.That(app.Children[0].Width).IsEqualTo(560);
			await Assert.That(app.PhotoPaths.Select(Path.GetFileName)).IsEquivalentTo(photos);
			await Assert.That(app.PhotoListLines).IsEquivalentTo(photos);
			await Assert.That(app.SettingsOpen).IsTrue();
			await Assert.That(app.Settings.Quality).IsEqualTo(QualityLevel.High);
			await Assert.That(app.Settings.Subject).IsEqualTo(SubjectType.Object);
			await Assert.That(((SegmentedControl)app.FindDescendant("Fast Setting")).SelectedIndex).IsEqualTo(2).Because("the new controls show the kept settings");
			await Assert.That(((SelectableCard)app.FindDescendant("One object Card")).Selected).IsTrue();
			await Assert.That(app.FindDescendant("Run Button").Enabled).IsTrue();
			window.Close();
		});
	}

	[Test]
	public async Task AScaleChangeDuringARunWaitsForTheRunToEnd()
	{
		await WithScaleRestored(async () =>
		{
			StartAt(1);
			var window = new SystemWindow(1200, 800);

			// The run shares this thread and parks at its first yield, so it is running until released.
			var gate = new TaskCompletionSource();
			var app = new ColmapDemoApp(fileDropSupported: false) { RunOnUiThread = true, YieldAsync = () => new ValueTask(gate.Task) };
			window.AddChild(app);
			DemoDisplayScale.Follow(window, app);
			app.AddPhotos(new[] { "a.png", "b.png", "c.png" }.Select(n => Path.Combine(Path.GetTempPath(), n)));
			app.StartRun();
			GuiWidget panelBefore = app.Children[0];

			MoveToDisplay(window, 2);

			await Assert.That(app.IsRunning).IsTrue();
			await Assert.That(app.Children[0]).IsSameReferenceAs(panelBefore).Because("a run's progress writes into the widgets it started with");
			await Assert.That(GuiWidget.DeviceScale).IsEqualTo(1.0).Because("the scale waits for the rebuild");

			app.RequestCancel();
			gate.SetResult();

			// The run's end, then UiScale's retry, come through the idle queue; the retry is timed.
			var clock = Stopwatch.StartNew();
			while (app.Children[0] == panelBefore && clock.Elapsed < TimeSpan.FromSeconds(10))
			{
				UiThread.InvokePendingActions();
				await Task.Delay(10);
			}

			await Assert.That(app.IsRunning).IsFalse();
			await Assert.That(GuiWidget.DeviceScale).IsEqualTo(2.0);
			await Assert.That(app.Children[0].Width).IsEqualTo(560);
			await Assert.That(app.StatusText).IsEqualTo("Cancelled.").Because("what the run left on screen is kept");
			window.Close();
		});
	}

	// A head's startup on a display of displayScale: UiScale composed for it, before any widget is built.
	private static void StartAt(double displayScale)
	{
		AggContext.OsInformation ??= new HeadlessOs();
		UiScale.DisplayScaleOverride = displayScale;
		UiScale.ApplyAtStartup();
		UiScale.DisplayScaleOverride = null;
	}

	// The host reporting the window on a display of displayScale, delivered; held for the real host too,
	// so a late report from this machine's own screen cannot move it back mid-test.
	private static void MoveToDisplay(SystemWindow window, double displayScale)
	{
		SystemWindow.SimulatedDisplayScale = displayScale;
		window.SetDisplayScale(displayScale);
		UiThread.InvokePendingActions();
	}

	private static async Task WithScaleRestored(Func<Task> test)
	{
		double savedDeviceScale = GuiWidget.DeviceScale;
		double savedDisplayScale = UiScale.CurrentDisplayScale;
		try
		{
			await test();
		}
		finally
		{
			SystemWindow.SimulatedDisplayScale = null;
			UiScale.DisplayScaleOverride = null;
			UiScale.UpdateForDisplayScale(savedDisplayScale);
			GuiWidget.DeviceScale = savedDeviceScale;
			UiThread.ResetForTests();
		}
	}

	// The left panel's width and the device em sizes of the title and the subtitle, built the way a
	// head builds the app: DeviceScale first (through UiScale), then the app.
	private static (double PanelWidth, double TitleEm, double HintEm) BuildAt(double displayScale)
	{
		StartAt(displayScale);
		var app = new ColmapDemoApp(fileDropSupported: false);
		try
		{
			GuiWidget panel = app.Children[0];
			TextWidget title = app.Descendants<TextWidget>().First(t => t.Text == "ColmapSharp");
			TextWidget hint = app.Descendants<TextWidget>().First(t => t.Text == "Photos or a video to mesh");
			return (panel.Width, title.Printer.TypeFaceStyle.EmSizeInPoints, hint.Printer.TypeFaceStyle.EmSizeInPoints);
		}
		finally
		{
			app.Close();
		}
	}
}
