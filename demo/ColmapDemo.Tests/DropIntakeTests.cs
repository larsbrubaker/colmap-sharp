// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// Tests of taking dropped files (demo/ColmapDemo/ColmapDemoApp.Drop.cs): which hovers the window
// accepts, a mac drop used where it is, and a browser drop - staged by agg's BrowserFileStaging as
// the browser host does - whose photos are moved out and whose staging is released, including a
// video where the browser has no reader, which gets a note rather than an error.
// NotInParallel with VideoIntakeTests: both swap AggContext.VideoFrames.

using MatterHackers.Agg;
using MatterHackers.Agg.Platform;
using MatterHackers.Agg.Platform.Browser;
using MatterHackers.Agg.UI;

namespace ColmapDemo.Tests;

[NotInParallel("AggUiThread")]
public class DropIntakeTests
{
	private sealed class HeadlessOs : IOsInformationProvider
	{
		public OSType OperatingSystem => OSType.Mac;

		public Point2D DesktopSize => new Point2D(1920, 1080);

		public long PhysicalMemory => 8L << 30;
	}

	[Test]
	public async Task AHoverIsTakenWhenItCarriesAPhotoOrVideo()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		var app = new ColmapDemoApp(fileDropSupported: true);

		// The browser hovers with stand-ins named from the MIME type.
		foreach ((string name, bool accepted) in new[] { ("dragged-1.mp4", true), ("dragged-1.jpg", true), ("dragged-1.bin", false) })
		{
			var hover = new MouseEventArgs(MouseButtons.None, 0, 10, 10, 0, new List<string> { name });
			app.OnMouseMove(hover);
			await Assert.That(hover.AcceptDrop).IsEqualTo(accepted);
		}
	}

	[Test]
	public async Task ADesktopDropUsesTheFilesWhereTheyAre()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		string root = Directory.CreateTempSubdirectory("ColmapDemoDropTests").FullName;
		try
		{
			string photo = Path.Combine(root, "a.jpg");
			File.WriteAllBytes(photo, new byte[] { 1 });
			var app = new ColmapDemoApp(fileDropSupported: true);

			await app.ReceiveDropAsync(new[] { photo });

			await Assert.That(app.PhotoPaths).IsEquivalentTo(new[] { photo });
			await Assert.That(File.Exists(photo)).IsTrue();
		}
		finally
		{
			Directory.Delete(root, recursive: true);
		}
	}

	[Test]
	public async Task ABrowserDropsPhotosAreMovedOutAndItsStagingReleased()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		string root = Directory.CreateTempSubdirectory("ColmapDemoDropTests").FullName;
		string staging = BrowserFileStaging.CreateRequestDirectory("drop");
		try
		{
			var dropped = new[] { "a.jpg", "b.JPG", "notes.txt" }.Select(n => Path.Combine(staging, n)).ToList();
			foreach (string path in dropped)
			{
				File.WriteAllText(path, Path.GetFileName(path));
			}

			var app = new ColmapDemoApp(fileDropSupported: true)
			{
				VideoFramesRoot = root,
				ReleaseDroppedFile = BrowserFileStaging.Release,
			};

			await app.ReceiveDropAsync(dropped);

			// The staging is gone; the photos live on in the demo's own folder, with their bytes and names.
			await Assert.That(Directory.Exists(staging)).IsFalse();
			await Assert.That(app.PhotoPaths.Select(Path.GetFileName)).IsEquivalentTo(new[] { "a.jpg", "b.JPG" });
			foreach (string photo in app.PhotoPaths)
			{
				await Assert.That(photo.StartsWith(root, StringComparison.Ordinal)).IsTrue();
				await Assert.That(File.ReadAllText(photo)).IsEqualTo(Path.GetFileName(photo));
			}

			await Assert.That(app.NoteText).IsEqualTo("1 file skipped: not a photo or video");

			string folder = Path.GetDirectoryName(app.PhotoPaths[0]);
			app.FindDescendant("Clear Photos Button").InvokeClick();
			await Assert.That(Directory.Exists(folder)).IsFalse();
		}
		finally
		{
			Directory.Delete(root, recursive: true);
			if (Directory.Exists(staging))
			{
				Directory.Delete(staging, recursive: true);
			}
		}
	}

	[Test]
	public async Task ABrowserDropThatCannotBeMovedSaysSoInsteadOfLosingThePhotosSilently()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		string root = Directory.CreateTempSubdirectory("ColmapDemoDropTests").FullName;
		string staging = BrowserFileStaging.CreateRequestDirectory("drop");
		try
		{
			string photo = Path.Combine(staging, "a.jpg");
			File.WriteAllBytes(photo, new byte[] { 1 });

			// A file where the demo's folder would go: making the folder fails as a full MEMFS would.
			string blocked = Path.Combine(root, "not-a-folder");
			File.WriteAllBytes(blocked, new byte[] { 0 });
			var app = new ColmapDemoApp(fileDropSupported: true)
			{
				VideoFramesRoot = blocked,
				ReleaseDroppedFile = BrowserFileStaging.Release,
			};

			await app.ReceiveDropAsync(new[] { photo });

			await Assert.That(app.ErrorText).StartsWith("Couldn't take the dropped photos: ");
			await Assert.That(app.ErrorText).EndsWith("Try dropping them again.");
			await Assert.That(app.PhotoPaths.Count).IsEqualTo(0);
			await Assert.That(Directory.Exists(staging)).IsFalse();
		}
		finally
		{
			Directory.Delete(root, recursive: true);
			if (Directory.Exists(staging))
			{
				Directory.Delete(staging, recursive: true);
			}
		}
	}

	[Test]
	public async Task TheSamePhotoDroppedTwiceInTheBrowserIsListedOnce()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		string root = Directory.CreateTempSubdirectory("ColmapDemoDropTests").FullName;
		var stagings = new List<string>();
		try
		{
			var app = new ColmapDemoApp(fileDropSupported: true)
			{
				VideoFramesRoot = root,
				ReleaseDroppedFile = BrowserFileStaging.Release,
			};

			// Each drop (or picker) stages its own copy, so only the name and size say it is the same photo.
			foreach (string content in new[] { "same", "same", "other bytes" })
			{
				string staging = BrowserFileStaging.CreateRequestDirectory("drop");
				stagings.Add(staging);
				string photo = Path.Combine(staging, "a.jpg");
				File.WriteAllText(photo, content);
				await app.ReceiveDropAsync(new[] { photo });
				await Assert.That(Directory.Exists(staging)).IsFalse();
			}

			// The third differs in size: a different photo that happens to share the name.
			await Assert.That(app.PhotoPaths.Count).IsEqualTo(2);

			// A picker's staged copy of an already-listed photo is not added again either.
			string picked = Path.Combine(root, "open-picker");
			Directory.CreateDirectory(picked);
			File.WriteAllText(Path.Combine(picked, "a.jpg"), "same");
			app.AddPhotos(new[] { Path.Combine(picked, "a.jpg") });
			await Assert.That(app.PhotoPaths.Count).IsEqualTo(2);
		}
		finally
		{
			Directory.Delete(root, recursive: true);
			foreach (string staging in stagings.Where(Directory.Exists))
			{
				Directory.Delete(staging, recursive: true);
			}
		}
	}

	[Test]
	public async Task ABrowserDroppedPhotoWhoseStagedFileIsMissingIsSkippedWithANote()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		string root = Directory.CreateTempSubdirectory("ColmapDemoDropTests").FullName;
		string staging = BrowserFileStaging.CreateRequestDirectory("drop");
		try
		{
			string present = Path.Combine(staging, "a.jpg");
			File.WriteAllBytes(present, new byte[] { 1 });
			var app = new ColmapDemoApp(fileDropSupported: true)
			{
				VideoFramesRoot = root,
				ReleaseDroppedFile = BrowserFileStaging.Release,
			};

			await app.ReceiveDropAsync(new[] { present, Path.Combine(staging, "gone.jpg") });

			await Assert.That(app.PhotoPaths.Select(Path.GetFileName)).IsEquivalentTo(new[] { "a.jpg" });
			await Assert.That(app.NoteText).IsEqualTo("1 dropped photo could not be read and was skipped.");
			await Assert.That(app.ErrorText).IsEqualTo(string.Empty);
		}
		finally
		{
			Directory.Delete(root, recursive: true);
			if (Directory.Exists(staging))
			{
				Directory.Delete(staging, recursive: true);
			}
		}
	}

	[Test]
	public async Task ABrowserDroppedVideoWithNoReaderGetsANoteAndIsReleased()
	{
		AggContext.OsInformation ??= new HeadlessOs();
		IVideoFrameReader savedReader = AggContext.VideoFrames;
		string staging = BrowserFileStaging.CreateRequestDirectory("drop");
		try
		{
			AggContext.VideoFrames = new UnsupportedVideoFrameReader();
			string video = Path.Combine(staging, "clip.mp4");
			File.WriteAllBytes(video, new byte[] { 0 });
			var app = new ColmapDemoApp(fileDropSupported: true)
			{
				ReleaseDroppedFile = BrowserFileStaging.Release,
				NoVideoReaderNote = ColmapDemoApp.BrowserNoVideoNote,
			};

			await app.ReceiveDropAsync(new[] { video });

			await Assert.That(app.NoteText).IsEqualTo(ColmapDemoApp.BrowserNoVideoNote);
			await Assert.That(app.ErrorText).IsEqualTo(string.Empty);
			await Assert.That(app.PhotoPaths.Count).IsEqualTo(0);
			await Assert.That(Directory.Exists(staging)).IsFalse();
		}
		finally
		{
			AggContext.VideoFrames = savedReader;
			if (Directory.Exists(staging))
			{
				Directory.Delete(staging, recursive: true);
			}
		}
	}
}
