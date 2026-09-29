// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of ColmapDemoApp's Cancel feedback (demo/ColmapDemo/ColmapDemoApp.Run.cs), without a
// window: the run's own completion is marshalled through agg's idle queue, which nothing pumps
// here, so the panel stays in the state the click left it in. The theme asks agg which OS it is
// on, which normally comes from a platform project the test does not reference, so a stand-in
// answers.

using MatterHackers.Agg;
using MatterHackers.Agg.Platform;

namespace ColmapDemo.Tests;

[NotInParallel("AggUiThread")]
public class CancelTests
{
	private sealed class HeadlessOs : IOsInformationProvider
	{
		public OSType OperatingSystem => OSType.Mac;

		public Point2D DesktopSize => new Point2D(1920, 1080);

		public long PhysicalMemory => 8L << 30;
	}

	[Test]
	public async Task CancelSaysCancellingAndCannotBePressedTwice()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		var app = new ColmapDemoApp(fileDropSupported: false);
		string dir = Directory.CreateTempSubdirectory("ColmapDemoTests").FullName;
		app.AddPhotos(new[] { "a.png", "b.png", "c.png" }.Select(n => Path.Combine(dir, n)));
		app.StartRun();
		await Assert.That(app.CanCancel).IsTrue();

		app.RequestCancel();

		await Assert.That(app.StatusText).IsEqualTo("Cancelling…");
		await Assert.That(app.CanCancel).IsFalse();
		Directory.Delete(dir, recursive: true);
	}
}
