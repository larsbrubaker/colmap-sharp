// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of the demo's display-scale policy (demo/ColmapDemo/DemoDisplayScale.cs): the heads set
// GuiWidget.DeviceScale to the display's scale before building ColmapDemoApp, so on a 2x (Retina)
// display the window, the panel and its fonts come out at twice the device pixels - the same
// physical size as on a 1x display - instead of half size.

using MatterHackers.Agg;
using MatterHackers.Agg.Platform;
using MatterHackers.Agg.UI;

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
	[Arguments(1.0, 1200, 800)]
	[Arguments(2.0, 2400, 1600)]
	[Arguments(1.5, 1800, 1200)]
	public async Task TheWindowIsTheDesignSizeInPoints(double displayScale, int width, int height)
	{
		double saved = GuiWidget.DeviceScale;
		try
		{
			SystemWindow window = DemoDisplayScale.CreateWindow("test", displayScale, new Point2D(3840, 2160));

			await Assert.That(GuiWidget.DeviceScale).IsEqualTo(displayScale);
			await Assert.That((int)window.Width).IsEqualTo(width);
			await Assert.That((int)window.Height).IsEqualTo(height);
		}
		finally
		{
			GuiWidget.DeviceScale = saved;
		}
	}

	[Test]
	public async Task TheWindowFitsOnASmallDesktop()
	{
		// A 150% laptop: 1800 x 1200 wanted, 1920 x 1040 available.
		await Assert.That(DemoDisplayScale.WindowSize(1.5, new Point2D(1920, 1040))).IsEqualTo((1800, 1040));

		// A host that cannot measure its desktop reports zero, which is no limit.
		await Assert.That(DemoDisplayScale.WindowSize(2, new Point2D(0, 0))).IsEqualTo((2400, 1600));
	}

	[Test]
	[Arguments(0.0)]
	[Arguments(-1.0)]
	[Arguments(double.NaN)]
	[Arguments(double.PositiveInfinity)]
	public async Task AnUnusableDisplayScaleIsOne(double displayScale)
	{
		await Assert.That(DemoDisplayScale.Usable(displayScale)).IsEqualTo(1.0);
	}

	[Test]
	public async Task ThePanelAndItsFontsScaleWithTheDisplay()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		double saved = GuiWidget.DeviceScale;
		try
		{
			(double panelWidth, double titleEm, double hintEm) atOne = BuildAt(1);
			(double panelWidth, double titleEm, double hintEm) atTwo = BuildAt(2);

			await Assert.That(atOne.panelWidth).IsEqualTo(280);
			await Assert.That(atTwo.panelWidth).IsEqualTo(560);
			await Assert.That(atOne.titleEm).IsEqualTo(16);
			await Assert.That(atTwo.titleEm).IsEqualTo(32);
			await Assert.That(atTwo.hintEm).IsEqualTo(2 * atOne.hintEm);
		}
		finally
		{
			GuiWidget.DeviceScale = saved;
		}
	}

	// The left panel's width and the device em sizes of the title and the subtitle, built the way a
	// head builds the app: DeviceScale first (through DemoDisplayScale), then the app.
	private static (double PanelWidth, double TitleEm, double HintEm) BuildAt(double displayScale)
	{
		DemoDisplayScale.CreateWindow("test", displayScale, new Point2D(0, 0));
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
