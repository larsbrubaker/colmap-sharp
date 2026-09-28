// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColmapDemoApp: the whole demo as one widget, which each head (ColmapDemo.Mac,
// ColmapDemo.Browser) only has to put in a window - the shape of agg-sharp's
// examples/AggSharpDemo/AggSharpDemo/AggSharpDemoApp.cs. The left panel collects the photos
// and lists the pipeline's stages; the right is the 3D viewport. This first step only
// collects photos: Run stays disabled and the viewport stays empty until the pipeline and the
// preview are wired in (docs/DEMO_PLAN.md phase 1).

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
	public class ColmapDemoApp : FlowLayoutWidget
	{
		/// <summary>
		/// The photo extensions the demo takes. The mac open panel ignores agg's filter string, and
		/// a drop can carry anything, so every path that comes in is checked against this list.
		/// </summary>
		private static readonly string[] PhotoExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff" };

		/// <summary>The open dialog's filter, built from <see cref="PhotoExtensions"/>.</summary>
		private static readonly string PhotoFilter = "Photos|" + string.Join(";", PhotoExtensions.Select(e => "*" + e));

		private static readonly Color PanelColor = new Color("#f2f2f2");

		private static readonly Color ViewportColor = new Color("#3a3d42");

		private static readonly Color HintColor = new Color("#707070");

		private readonly List<string> photoPaths = new List<string>();

		private readonly TextWidget photoCount;

		private readonly FlowLayoutWidget photoList;

		private readonly TextWidget skippedNote;

		private readonly ThemedTextButton clearButton;

		/// <param name="fileDropSupported">Whether the head delivers dropped files to the window. The
		/// mac host does; the browser host does not yet (docs/DEMO_PLAN.md phase 3), and the drop hint
		/// is only shown where a drop works.</param>
		public ColmapDemoApp(bool fileDropSupported)
			: base(FlowDirection.LeftToRight)
		{
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

			panel.AddChild(new TextWidget("ColmapSharp", pointSize: 16, bold: true) { HAnchor = HAnchor.Left });
			panel.AddChild(new TextWidget("Photos to mesh", pointSize: 10, textColor: HintColor)
			{
				HAnchor = HAnchor.Left,
				Margin = new BorderDouble(0, 10, 0, 0),
			});

			var photoButtons = new FlowLayoutWidget(FlowDirection.LeftToRight) { HAnchor = HAnchor.Stretch };
			var addButton = new ThemedTextButton("Add photos…", theme) { Name = "Add Photos Button" };
			addButton.Click += (sender, e) => this.ShowAddPhotosDialog();
			photoButtons.AddChild(addButton);
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
				panel.AddChild(new TextWidget("or drop photos onto this window", pointSize: 9, textColor: HintColor)
				{
					HAnchor = HAnchor.Left,
					Margin = new BorderDouble(0, 0, 0, 4),
				});
			}

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

			this.skippedNote = new TextWidget(string.Empty, pointSize: 9, textColor: HintColor)
			{
				HAnchor = HAnchor.Left,
				Margin = new BorderDouble(0, 0, 0, 4),
				AutoExpandBoundsToText = true,
				Visible = false,
			};
			panel.AddChild(this.skippedNote);

			// Disabled until the pipeline is wired in; a button that did nothing would lie.
			var runButton = new ThemedTextButton("Run", theme)
			{
				Name = "Run Button",
				HAnchor = HAnchor.Stretch,
				Margin = new BorderDouble(0, 0, 0, 12),
				Enabled = false,
			};
			panel.AddChild(runButton);

			panel.AddChild(new TextWidget("Stages", pointSize: 11, bold: true)
			{
				HAnchor = HAnchor.Left,
				Margin = new BorderDouble(0, 4, 0, 12),
			});
			foreach (string stage in Stages)
			{
				panel.AddChild(new TextWidget(stage, pointSize: 10, textColor: HintColor)
				{
					HAnchor = HAnchor.Left,
					Margin = new BorderDouble(8, 2, 0, 2),
				});
			}

			var viewport = new GuiWidget()
			{
				Name = "Model Viewport",
				HAnchor = HAnchor.Stretch,
				VAnchor = VAnchor.Stretch,
				BackgroundColor = ViewportColor,
			};
			viewport.AddChild(new TextWidget("No model yet", pointSize: 14, textColor: new Color("#b0b0b0"))
			{
				HAnchor = HAnchor.Center,
				VAnchor = VAnchor.Center,
			});
			this.AddChild(viewport);

			this.UpdatePhotoCount();
		}

		/// <summary>
		/// The pipeline's steps in the order a run goes through them, named by the library's own
		/// progress stages so the list will match the progress reports once Run is wired in.
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
		/// and saying how many were not photos.
		/// </summary>
		public void AddPhotos(IEnumerable<string> paths)
		{
			int skipped = 0;
			foreach (string path in paths)
			{
				if (!IsPhotoPath(path))
				{
					skipped++;
					continue;
				}

				if (this.photoPaths.Contains(path))
				{
					continue;
				}

				this.photoPaths.Add(path);
				this.photoList.AddChild(new TextWidget(Path.GetFileName(path), pointSize: 10)
				{
					HAnchor = HAnchor.Left,
					Margin = new BorderDouble(2, 1),
				});
			}

			this.skippedNote.Text = skipped == 1 ? "1 file skipped: not a photo" : $"{skipped} files skipped: not photos";
			this.skippedNote.Visible = skipped > 0;
			this.UpdatePhotoCount();
		}

		/// <summary>A drag over the window is taken when it carries at least one photo.</summary>
		public override void OnMouseMove(MouseEventArgs mouseEvent)
		{
			if (mouseEvent.DragFiles?.Any(IsPhotoPath) == true)
			{
				mouseEvent.AcceptDrop = true;
			}

			base.OnMouseMove(mouseEvent);
		}

		/// <summary>The drop itself arrives as a mouse up carrying the files (agg's FileDropDispatcher).</summary>
		public override void OnMouseUp(MouseEventArgs mouseEvent)
		{
			if (mouseEvent.DragFiles?.Count > 0)
			{
				this.AddPhotos(mouseEvent.DragFiles);
			}

			base.OnMouseUp(mouseEvent);
		}

		private void ShowAddPhotosDialog()
		{
			// The callback can come after this returns (the browser answers from its picker's change
			// event), and never comes on cancel, so everything happens inside it.
			AggContext.FileDialogs.OpenFileDialog(
				new OpenFileDialogParams(PhotoFilter, multiSelect: true, title: "Add photos"),
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
			this.photoPaths.Clear();
			this.photoList.CloseChildren();
			this.skippedNote.Visible = false;
			this.UpdatePhotoCount();
		}

		private void UpdatePhotoCount()
		{
			int count = this.photoPaths.Count;
			this.photoCount.Text = count == 0 ? "No photos yet" : count == 1 ? "1 photo" : $"{count} photos";
			this.clearButton.Enabled = count > 0;
		}
	}
}
