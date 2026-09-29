// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColmapDemoApp: the whole demo as one widget, which each head (ColmapDemo.Mac,
// ColmapDemo.Browser) only has to put in a window - the shape of agg-sharp's
// examples/AggSharpDemo/AggSharpDemo/AggSharpDemoApp.cs. The left panel gives capture tips, collects the photos
// and lists the pipeline's stages; the right is the 3D viewport (ModelViewport.cs). Running,
// progress, cancel and saving are in ColmapDemoApp.Run.cs; turning a video into photos is in
// ColmapDemoApp.Video.cs; taking dropped files (and freeing the browser's staged copies) is in
// ColmapDemoApp.Drop.cs; marking which photos a run placed is in ColmapDemoApp.Placement.cs; the
// Settings panel on the right, opened by the Settings button, is in ColmapDemoApp.Settings.cs.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ColmapSharp.Controllers;
using MatterHackers.Agg;
using MatterHackers.Agg.Platform;
using MatterHackers.Agg.UI;

namespace ColmapDemo
{
	/// <summary>
	/// The demo's window content: a left panel (add photos, the photo list, Run, the stage list)
	/// and a 3D viewport on the right.
	/// </summary>
	public partial class ColmapDemoApp : FlowLayoutWidget
	{
		/// <summary>
		/// The photo extensions the demo takes. The mac open panel ignores agg's filter string, and
		/// a drop can carry anything, so every path that comes in is checked against this list.
		/// </summary>
		private static readonly string[] PhotoExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff" };

		/// <summary>
		/// The video extensions the demo takes (ColmapDemoApp.Video.cs cuts them into frames), checked
		/// like <see cref="PhotoExtensions"/>. Here rather than in that file because static fields
		/// initialize in file order and <see cref="PhotoFilter"/> reads it.
		/// </summary>
		private static readonly string[] VideoExtensions = { ".mp4", ".mov", ".m4v", ".avi", ".wmv", ".mkv" };

		/// <summary>The open dialog's filter, built from <see cref="PhotoExtensions"/> and the video extensions.</summary>
		private static readonly string PhotoFilter = "Photos and videos|" + string.Join(";", PhotoExtensions.Concat(VideoExtensions).Select(e => "*" + e));

		private static readonly Color PanelColor = new Color("#f2f2f2");

		private static readonly Color HintColor = new Color("#707070");

		private readonly List<string> photoPaths = new List<string>();

		private readonly TextWidget photoCount;

		private readonly FlowLayoutWidget photoList;

		// Hint-colored, not an error: files skipped from an add, or that video is not read here.
		private readonly WrappedTextWidget noteLine;

		private readonly ThemedTextButton clearButton;

		private readonly ThemedTextButton addButton;

		/// <param name="fileDropSupported">Whether the head delivers dropped files to the window (the
		/// mac and browser hosts do); the drop hint is only shown where a drop works.</param>
		/// <param name="computeDevice">The GPU PatchMatch runs on, or null for the CPU.</param>
		/// <param name="computeNote">One line on where PatchMatch runs and, on the CPU, why.</param>
		public ColmapDemoApp(bool fileDropSupported, ColmapSharp.Compute.IComputeDevice computeDevice = null, string computeNote = "")
			: base(FlowDirection.LeftToRight)
		{
			this.computeDevice = computeDevice;
			this.AnchorAll();
			ThemeConfig theme = ThemeConfig.Current;

			var panel = new FlowLayoutWidget(FlowDirection.TopToBottom)
			{
				HAnchor = HAnchor.Absolute,
				Width = 280 * DeviceScale,
				VAnchor = VAnchor.Stretch,
				Padding = 12,
				BackgroundColor = PanelColor,
			};
			this.AddChild(panel);

			var titleRow = new FlowLayoutWidget(FlowDirection.LeftToRight) { HAnchor = HAnchor.Stretch };
			titleRow.AddChild(new TextWidget("ColmapSharp", pointSize: 16, bold: true) { VAnchor = VAnchor.Center });
			titleRow.AddChild(new HorizontalSpacer());
			var settingsButton = new ThemedTextButton("Settings", theme) { Name = "Settings Button" };
			settingsButton.Click += (sender, e) => this.SettingsOpen = !this.SettingsOpen;
			titleRow.AddChild(settingsButton);
			panel.AddChild(titleRow);
			panel.AddChild(new TextWidget("Photos or a video to mesh", pointSize: 10, textColor: HintColor)
			{
				HAnchor = HAnchor.Left,
				Margin = new BorderDouble(0, 10, 0, 0),
			});

			var photoButtons = new FlowLayoutWidget(FlowDirection.LeftToRight) { HAnchor = HAnchor.Stretch };
			this.addButton = new ThemedTextButton("Add photos…", theme) { Name = "Add Photos Button" };
			this.addButton.Click += (sender, e) => this.ShowAddPhotosDialog();
			photoButtons.AddChild(this.addButton);
			photoButtons.AddChild(new HorizontalSpacer());
			this.clearButton = new ThemedTextButton("Clear", theme)
			{
				Name = "Clear Photos Button",
				Enabled = false,
			};
			this.clearButton.Click += (sender, e) => this.ClearPhotos();
			photoButtons.AddChild(this.clearButton);
			panel.AddChild(photoButtons);

			if (fileDropSupported)
			{
				panel.AddChild(new TextWidget("or drop photos or a video onto this window", pointSize: 9, textColor: HintColor)
				{
					HAnchor = HAnchor.Left,
					Margin = new BorderDouble(0, 0, 0, 4),
				});
			}

			AddCaptureTips(panel);

			this.photoCount = new TextWidget(string.Empty, pointSize: 11, bold: true)
			{
				HAnchor = HAnchor.Left,
				Margin = new BorderDouble(0, 4, 0, 12),
				AutoExpandBoundsToText = true,
			};
			panel.AddChild(this.photoCount);

			// The list can outgrow the panel (a capture is often dozens of photos), so it scrolls and
			// takes whatever height the rest of the panel leaves.
			var photoScroll = new ScrollableWidget(autoScroll: true)
			{
				HAnchor = HAnchor.Stretch,
				VAnchor = VAnchor.Stretch,
				BackgroundColor = Color.White,
			};
			photoScroll.ScrollArea.HAnchor = HAnchor.Stretch;
			this.photoList = new FlowLayoutWidget(FlowDirection.TopToBottom)
			{
				HAnchor = HAnchor.Stretch,
				Padding = 4,
			};
			photoScroll.AddChild(this.photoList);
			panel.AddChild(photoScroll);

			this.noteLine = new WrappedTextWidget(string.Empty, pointSize: 9, textColor: HintColor)
			{
				HAnchor = HAnchor.Stretch,
				Margin = new BorderDouble(0, 0, 0, 4),
				Visible = false,
			};
			panel.AddChild(this.noteLine);

			this.AddRunControls(panel, theme, computeNote);

			this.viewport = new ModelViewport();
			this.AddChild(this.viewport);

			// On the right, over nothing the user needs while choosing settings; the viewport narrows.
			this.settingsColumn = this.BuildSettingsColumn(theme);
			this.AddChild(this.settingsColumn);
			this.RefreshSettingsControls();

			this.UpdatePhotoCount();
		}

		/// <summary>
		/// The pipeline's steps in the order a run goes through them, named by the library's own
		/// progress stages, so each report finds its row.
		/// </summary>
		public static IReadOnlyList<string> Stages { get; } = new[]
		{
			FeatureExtraction.ExtractionStage,
			FeatureMatching.MatchingStage,
			AutomaticReconstructionController.SparseStage,
			AutomaticReconstructionController.DenseStage,
			AutomaticReconstructionController.FusionStage,
			AutomaticReconstructionController.MeshingStage,
			AutomaticReconstructionController.TexturingStage,
		};

		/// <summary>
		/// How to take photos that reconstruct well, shown in the panel before the photo list. What
		/// the mapper needs is texture that stays put while the camera moves: a plain object spinning
		/// on a cable in front of a plain wall places only the frames that see its textured side.
		/// </summary>
		public static IReadOnlyList<string> CaptureTips { get; } = new[]
		{
			"Set the object on a patterned surface (newspaper, a printed mat); don't hang it.",
			"Move the camera around the object; don't spin the object.",
			"Use even lighting.",
			"Take 20–60 photos, each overlapping the last a lot.",
			"Dark, plain or shiny objects need added texture: tape or stickers.",
		};

		/// <summary>The chosen photos' full paths, in the order they were added, without duplicates.
		/// In the browser these are paths in the wasm file system the picker staged the bytes into.</summary>
		public IReadOnlyList<string> PhotoPaths => this.photoPaths;

		/// <summary>Whether <paramref name="path"/> has one of the photo extensions the demo takes
		/// (case-insensitive: cameras write ".JPG").</summary>
		public static bool IsPhotoPath(string path)
		{
			string extension = Path.GetExtension(path);
			return PhotoExtensions.Any(e => string.Equals(e, extension, StringComparison.OrdinalIgnoreCase));
		}

		/// <summary>
		/// Adds the photos among <paramref name="paths"/> to the list, skipping ones already there
		/// and saying how many were neither photos nor videos. Videos are cut into frames in the
		/// background (<see cref="AddVideosAsync"/>).
		/// </summary>
		public void AddPhotos(IEnumerable<string> paths)
		{
			// Not awaited: the video reading reports its own outcome in the panel and never throws.
			_ = this.AddFilesAsync(paths);
		}

		/// <summary>
		/// <see cref="AddPhotos"/>, returning the reading of the videos among <paramref name="paths"/>,
		/// so a caller that must know when the videos are done with (a browser drop) can wait for it.
		/// </summary>
		private Task AddFilesAsync(IEnumerable<string> paths)
		{
			if (this.IsRunning)
			{
				// The run already took its photo list; changing it mid-run would only mislead.
				return Task.CompletedTask;
			}

			int skipped = 0;
			var videoPaths = new List<string>();
			foreach (string path in paths)
			{
				if (IsVideoPath(path))
				{
					videoPaths.Add(path);
					continue;
				}

				if (!IsPhotoPath(path))
				{
					skipped++;
					continue;
				}

				if (this.photoPaths.Contains(path) || this.IsStagedDuplicate(path))
				{
					continue;
				}

				this.photoPaths.Add(path);
				this.AddListEntry(this.AddListLine(Path.GetFileName(path)), new[] { path });
			}

			this.noteLine.Text = skipped == 1 ? "1 file skipped: not a photo or video" : $"{skipped} files skipped: not photos or videos";
			this.noteLine.Visible = skipped > 0;
			this.UpdatePhotoCount();

			return this.AddVideosAsync(videoPaths);
		}

		// Always visible: a first-time user needs them before the first capture, not after a poor run.
		private static void AddCaptureTips(FlowLayoutWidget panel)
		{
			panel.AddChild(new TextWidget("Tips for good results", pointSize: 10, bold: true)
			{
				HAnchor = HAnchor.Left,
				Margin = new BorderDouble(0, 2, 0, 12),
			});
			foreach (string tip in CaptureTips)
			{
				panel.AddChild(new WrappedTextWidget("• " + tip, pointSize: 9, textColor: HintColor)
				{
					HAnchor = HAnchor.Stretch,
					Margin = new BorderDouble(4, 1, 0, 1),
				});
			}
		}

		private TextWidget AddListLine(string text)
		{
			var line = new TextWidget(text, pointSize: 10)
			{
				HAnchor = HAnchor.Left,
				Margin = new BorderDouble(2, 1),
				AutoExpandBoundsToText = true,
			};
			this.photoList.AddChild(line);
			return line;
		}

		private void ShowAddPhotosDialog()
		{
			// The callback can come after this returns (the browser answers from its picker's change
			// event), and never comes on cancel, so everything happens inside it.
			AggContext.FileDialogs.OpenFileDialog(
				new OpenFileDialogParams(PhotoFilter, multiSelect: true, title: "Add photos or a video"),
				openParams =>
				{
					if (openParams.FileNames?.Length > 0)
					{
						this.AddPhotos(openParams.FileNames);
					}
				});
		}

		private void ClearPhotos()
		{
			if (this.IsRunning || this.IsReadingVideo)
			{
				return;
			}

			this.DeleteVideoFrames();
			this.DeleteDroppedPhotos();
			this.photoPaths.Clear();
			this.listEntries.Clear();
			this.photoList.CloseChildren();
			this.noteLine.Visible = false;
			this.UpdatePhotoCount();
		}

		private void UpdatePhotoCount()
		{
			int count = this.photoPaths.Count;
			this.photoCount.Text = count == 0 ? "No photos yet" : count == 1 ? "1 photo" : $"{count} photos";
			this.clearButton.Enabled = count > 0 && !this.IsRunning && !this.IsReadingVideo;
			this.UpdateRunButtons();

			// Whether the photos are one video's frames decides which video options apply.
			if (this.settingsColumn != null)
			{
				this.RefreshSettingsControls();
			}
		}
	}
}
