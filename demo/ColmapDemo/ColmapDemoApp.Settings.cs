// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColmapDemoApp.Settings: the Settings panel, a scrolling column on the right of the viewport that
// the "Settings" button at the top of the left panel shows and hides. It groups every option of the
// next run (DemoSettings.cs) under What are you capturing? / Quality / Camera / Video / Result /
// Advanced, in plain words with a hint on when to change each; its controls are built in
// ColmapDemoApp.SettingsControls.cs. While a run goes the controls are disabled and hidden behind a
// read-only "This run" summary and a lock notice; the same summary starts the run log
// (ColmapDemoApp.Run.cs). Reset to defaults puts back the head's defaults (the browser's smaller
// photo size included).
// Settings are not remembered between launches: agg-sharp has no per-app settings store that works
// on both the desktop and the browser (only its demo app's own DemoStateStore).

using System;
using System.Collections.Generic;
using MatterHackers.Agg;
using MatterHackers.Agg.UI;
using static ColmapSharp.Controllers.AutomaticReconstructionOptions;

namespace ColmapDemo
{
	public partial class ColmapDemoApp
	{
		/// <summary>The photo sizes offered (longer side, pixels).</summary>
		public static readonly int[] PhotoSizes = { 480, 640, 800, 1000, 1600, 2400, 3200 };

		private const string LockNotice = "Locked while the mesh is being made. Press Cancel to change them, or wait for the run to finish.";

		private readonly List<GuiWidget> settingsControls = new List<GuiWidget>();

		// Each control's "show the model's value" action, run after every change, Reset and a video's defaults.
		private readonly List<Action> settingsRefreshers = new List<Action>();

		// Each number field's "read what was typed", run on edit complete and before a run starts.
		private readonly List<Action> settingsCommitters = new List<Action>();

		// Controls usable only in some states (the object-only video options), beyond the run lock.
		private readonly List<(GuiWidget Control, Func<bool> EnabledWhen)> conditionalControls = new List<(GuiWidget, Func<bool>)>();

		private DemoSettings settings = new DemoSettings();

		private DemoSettings defaultSettings = new DemoSettings();

		private GuiWidget settingsColumn;

		private GuiWidget settingsEditors;

		private FlowLayoutWidget runSummary;

		/// <summary>The next run's settings, as the panel shows them.</summary>
		public DemoSettings Settings => this.settings;

		/// <summary>Every input control of the settings panel, Reset to defaults included.</summary>
		public IReadOnlyList<GuiWidget> SettingsControls => this.settingsControls;

		/// <summary>Whether the settings are locked (a run is going): controls disabled, summary shown.</summary>
		public bool SettingsLocked => this.runSummary?.Visible == true;

		/// <summary>The "This run" rows shown while locked.</summary>
		public IReadOnlyList<(string Label, string Value)> RunSummaryRows { get; private set; } = Array.Empty<(string, string)>();

		/// <summary>Whether the Settings panel is showing.</summary>
		public bool SettingsOpen
		{
			get => this.settingsColumn?.Visible == true;
			set => this.settingsColumn.Visible = value;
		}

		/// <summary>
		/// The longer side photos are shrunk to, by default and now: a head sets its own (the browser
		/// a smaller one), which Reset to defaults then goes back to.
		/// </summary>
		public int MaxImageSize
		{
			get => this.settings.MaxImageSize;
			set
			{
				this.defaultSettings.MaxImageSize = value;
				this.settings.MaxImageSize = value;
				this.RefreshSettingsControls();
			}
		}

		/// <summary>How many frames each added video is cut into (the "Frames to take from a video" setting).</summary>
		public int TargetFramesPerVideo
		{
			get => this.settings.FramesPerVideo;
			set
			{
				this.settings.FramesPerVideo = value;
				this.RefreshSettingsControls();
			}
		}

		/// <summary>Puts every setting back to the head's defaults (what Reset to defaults does).</summary>
		public void ResetSettings()
		{
			if (this.IsRunning)
			{
				return;
			}

			this.settings = this.defaultSettings.Clone();
			this.RefreshSettingsControls();
		}

		/// <summary>Reads any number typed into a settings field that has not been confirmed yet.</summary>
		public void CommitSettingsEdits()
		{
			foreach (Action commit in this.settingsCommitters)
			{
				commit();
			}
		}

		/// <summary>Switches the settings to a video's defaults (one object, matched in order) and shows them.</summary>
		private void ApplyVideoSettingsDefaults()
		{
			this.settings.ApplyVideoDefaults();
			this.RefreshSettingsControls();
		}

		/// <summary>Locks (a run started) or unlocks the panel: summary and notice instead of the controls.</summary>
		/// <param name="run">The settings the starting run uses, for its summary (null when unlocking).</param>
		private void UpdateSettingsLock(SessionSettings run = null)
		{
			bool locked = this.IsRunning;
			if (locked && run != null)
			{
				this.RunSummaryRows = this.settings.Summary(run);
				this.runSummary.CloseChildren();
				this.runSummary.AddChild(new TextWidget("This run", pointSize: 11, bold: true) { HAnchor = HAnchor.Left });
				foreach ((string label, string value) in this.RunSummaryRows)
				{
					var row = new FlowLayoutWidget(FlowDirection.LeftToRight) { HAnchor = HAnchor.Stretch, Margin = new BorderDouble(0, 2) };
					row.AddChild(new TextWidget(label, pointSize: 10, textColor: HintColor) { Width = 80 * DeviceScale, HAnchor = HAnchor.Absolute });
					row.AddChild(new WrappedTextWidget(value, pointSize: 10) { HAnchor = HAnchor.Stretch });
					this.runSummary.AddChild(row);
				}

				this.runSummary.AddChild(Hint(LockNotice, WarningColor));
			}

			this.runSummary.Visible = locked;
			this.settingsEditors.Visible = !locked;
			foreach (GuiWidget control in this.settingsControls)
			{
				control.Enabled = !locked;
			}

			this.UpdateConditionalControls();
		}

		private void UpdateConditionalControls()
		{
			foreach ((GuiWidget control, Func<bool> enabledWhen) in this.conditionalControls)
			{
				control.Enabled = !this.IsRunning && enabledWhen();
			}
		}

		private void RefreshSettingsControls()
		{
			// A number typed but not yet confirmed (the user clicked another control) is kept, not
			// overwritten by the value it replaces.
			this.CommitSettingsEdits();
			foreach (Action refresh in this.settingsRefreshers)
			{
				refresh();
			}

			this.UpdateConditionalControls();
		}

		// Built hidden with the rest of the UI (again on a rescale); the left panel's Settings button shows and hides it.
		private GuiWidget BuildSettingsColumn(ThemeConfig theme)
		{
			var scroll = new ScrollableWidget(autoScroll: true)
			{
				Name = "Settings Panel",
				HAnchor = HAnchor.Absolute,
				Width = 440 * DeviceScale,
				VAnchor = VAnchor.Stretch,
				BackgroundColor = SettingsPanelColor,
				Visible = false,
			};
			scroll.ScrollArea.HAnchor = HAnchor.Stretch;
			var column = new FlowLayoutWidget(FlowDirection.TopToBottom) { HAnchor = HAnchor.Stretch, Padding = 14 };
			scroll.AddChild(column);

			var top = new FlowLayoutWidget(FlowDirection.LeftToRight) { HAnchor = HAnchor.Stretch };
			var titles = new FlowLayoutWidget(FlowDirection.TopToBottom) { HAnchor = HAnchor.Absolute, Width = 200 * DeviceScale, VAnchor = VAnchor.Fit | VAnchor.Center };
			titles.AddChild(new TextWidget("Settings", pointSize: 16, bold: true) { HAnchor = HAnchor.Left });
			titles.AddChild(new TextWidget("Apply to the next run", pointSize: 9, textColor: HintColor) { HAnchor = HAnchor.Left });
			top.AddChild(titles);
			top.AddChild(new HorizontalSpacer());
			var reset = new ThemedTextButton("Reset to defaults", theme) { Name = "Reset Settings Button", VAnchor = VAnchor.Center };
			reset.Click += (sender, e) => this.ResetSettings();
			top.AddChild(reset);
			this.settingsControls.Add(reset);
			column.AddChild(top);

			this.runSummary = new FlowLayoutWidget(FlowDirection.TopToBottom) { Name = "Run Summary", HAnchor = HAnchor.Stretch, Margin = new BorderDouble(0, 0, 0, 12), Visible = false };
			column.AddChild(this.runSummary);

			var editors = new FlowLayoutWidget(FlowDirection.TopToBottom) { HAnchor = HAnchor.Stretch };
			this.settingsEditors = editors;
			column.AddChild(editors);
			this.AddCaptureSettings(Section(editors, "What are you capturing?", null, theme), theme);
			this.AddQualitySettings(Section(editors, "Quality", null, theme), theme);
			this.AddCameraSettings(Section(editors, "Camera", null, theme), theme);
			this.AddVideoSettings(Section(editors, "Video", null, theme), theme);
			this.AddResultSettings(Section(editors, "Result", null, theme), theme);
			this.AddAdvancedSettings(Section(editors, "Advanced", "for testing and comparisons", theme, collapsible: true), theme);
			return scroll;
		}

		/// <summary>
		/// A titled group of settings: the theme's small-caps section header over the controls, or, for
		/// what most users never need (<paramref name="collapsible"/>), a <see cref="CollapsingHeader"/>
		/// that starts closed. Returns where the controls go.
		/// </summary>
		private static GuiWidget Section(GuiWidget column, string title, string subtitle, ThemeConfig theme, bool collapsible = false)
		{
			GuiWidget body;
			if (collapsible)
			{
				var header = new CollapsingHeader(title, theme, expanded: false) { Name = title + " Section", Margin = new BorderDouble(0, 0, 0, 16) };
				column.AddChild(header);
				body = header.Body;
			}
			else
			{
				body = new FlowLayoutWidget(FlowDirection.TopToBottom) { Name = title + " Section", HAnchor = HAnchor.Stretch, Margin = new BorderDouble(0, 0, 0, 16) };
				TextWidget header = theme.CreateSectionHeader(title);
				header.HAnchor = HAnchor.Left;
				body.AddChild(header);
				column.AddChild(body);
			}

			if (subtitle != null)
			{
				body.AddChild(Hint(subtitle));
			}

			return body;
		}

		private void AddCaptureSettings(GuiWidget section, ThemeConfig theme)
		{
			this.AddCards(
				section,
				theme,
				new[]
				{
					("One object", "On a plain background. Closed, printable shape.", SubjectType.Object),
					("A scene or room", "Everything in view. Open surface.", SubjectType.Scene),
				},
				() => this.settings.Subject,
				v =>
				{
					this.settings.Subject = v;
					this.settings.SubjectPickedForVideo = false;
				});
			WrappedTextWidget videoNote = Hint("A video was dropped, so “One object” was picked for you.", theme.PrimaryAccentColor);
			section.AddChild(videoNote);
			this.settingsRefreshers.Add(() => videoNote.Visible = this.settings.SubjectPickedForVideo);
		}

		private void AddQualitySettings(GuiWidget section, ThemeConfig theme)
		{
			this.AddSegmented(section, null, theme, new[] { ("Fast", QualityLevel.Low), ("Medium", QualityLevel.Medium), ("High", QualityLevel.High), ("Best", QualityLevel.Extreme) },
				() => this.settings.Quality, v => this.settings.Quality = v);
			this.AddChoice(section, "Largest photo size", theme, Array.ConvertAll(PhotoSizes, s => ($"{s} px", s)),
				() => this.settings.MaxImageSize, v => this.settings.MaxImageSize = v);
			section.AddChild(Hint("Bigger photos find more detail but take longer. In the browser, keep this at 640 px or below."));
		}

		private void AddCameraSettings(GuiWidget section, ThemeConfig theme)
		{
			// A known focal is one camera's (the library takes one set of camera parameters), so the
			// box shows ticked and cannot be unticked while the focal is known.
			this.AddCheck(section, "All photos are from the same camera", () => this.settings.SameCamera || this.settings.KnowFocalLength, v => this.settings.SameCamera = v, () => !this.settings.KnowFocalLength);
			this.AddCheck(section, "I know this camera's focal length", () => this.settings.KnowFocalLength, v => this.settings.KnowFocalLength = v);

			// The focal and its unit read as one value, so they sit together in the theme's tinted box.
			var box = new InfoBox(theme) { Name = "Focal Length Box", Margin = new BorderDouble(0, 4, 0, 0) };
			var focalRow = new FlowLayoutWidget(FlowDirection.LeftToRight) { HAnchor = HAnchor.Stretch };
			box.AddChild(focalRow);
			this.AddNumber(focalRow, null, theme, 0.1, 100000, true, () => this.settings.FocalLength, v => this.settings.FocalLength = v).Margin = new BorderDouble(0, 0, 8, 0);
			this.AddChoice(focalRow, null, theme, new[] { ("mm, 35 mm equivalent", FocalUnit.Millimetres35), ("pixels at the chosen photo size", FocalUnit.Pixels) },
				() => this.settings.FocalUnit, v => this.settings.FocalUnit = v).VAnchor = VAnchor.Center;
			section.AddChild(box);
			WrappedTextWidget pixelsHint = Hint("Pixels at the chosen photo size (Largest photo size above), not at the camera's full size.");
			section.AddChild(pixelsHint);
			this.settingsRefreshers.Add(() =>
			{
				box.Visible = this.settings.KnowFocalLength;
				pixelsHint.Visible = this.settings.KnowFocalLength && this.settings.FocalUnit == FocalUnit.Pixels;
			});
			section.AddChild(Hint("Phones list it in the photo details as “26 mm” or similar (35 mm equivalent, measured across the diagonal). It stays fixed while the model is built — the biggest single boost to accuracy. All photos must then be one size."));
		}

		private void AddVideoSettings(GuiWidget section, ThemeConfig theme)
		{
			this.AddNumber(section, "Frames to take from a video", theme, 3, 1000, false, () => this.settings.FramesPerVideo, v => this.settings.FramesPerVideo = (int)v);

			// Enabled by the library's own conditions: tracking needs time-ordered frames (one video's
			// frames, or recording-order matching); outline placement needs that and One object.
			Func<bool> timeOrdered = () => this.settings.TimeOrdered(this.RunUsesSingleCamera);
			this.AddCheck(section, "Follow points from frame to frame", () => this.settings.VideoTracking, v => this.settings.VideoTracking = v, timeOrdered);
			WrappedTextWidget trackingHint = Hint(string.Empty, WarningColor);
			section.AddChild(trackingHint);
			this.AddCheck(section, "Place missed frames from their outlines", () => this.settings.SilhouettePlacement, v => this.settings.SilhouettePlacement = v, () => timeOrdered() && this.settings.IsObject);
			WrappedTextWidget outlineHint = Hint(string.Empty, WarningColor);
			section.AddChild(outlineHint);
			section.AddChild(Hint("Experimental: can misplace frames on round or symmetric objects."));
			this.settingsRefreshers.Add(() =>
			{
				trackingHint.Text = OnlyForVideoFrames;
				trackingHint.Visible = !timeOrdered();
				outlineHint.Text = !timeOrdered() ? OnlyForVideoFrames : "Placing frames from outlines needs ‘One object’";
				outlineHint.Visible = !timeOrdered() || !this.settings.IsObject;
			});
		}

		private const string OnlyForVideoFrames = "Only for frames taken from a video";

		private void AddResultSettings(GuiWidget section, ThemeConfig theme)
		{
			this.AddSegmented(section, null, theme, new[] { ("Full mesh", true), ("Quick camera check", false) },
				() => this.settings.Dense, v => this.settings.Dense = v);
			CheckBox texture = this.AddCheck(section, "Paint the photos' colours onto the mesh", () => this.settings.Texture, v => this.settings.Texture = v);
			WrappedTextWidget quickHint = Hint("Stops after placing the photos and shows the cameras and sparse points. Seconds instead of minutes.");
			section.AddChild(quickHint);
			this.settingsRefreshers.Add(() =>
			{
				texture.Visible = this.settings.Dense;
				quickHint.Visible = !this.settings.Dense;
			});
		}

		private void AddAdvancedSettings(GuiWidget section, ThemeConfig theme)
		{
			this.AddChoice(section, "Surface method", theme, new[] { ("Smooth (Poisson)", MesherType.Poisson), ("Faceted (Delaunay)", MesherType.Delaunay) },
				() => this.settings.Mesher, v => this.settings.Mesher = v);
			this.AddNumber(section, "Surface detail (Poisson depth, 9–11 suits most)", theme, 5, 13, false, () => this.settings.PoissonDepth, v => this.settings.PoissonDepth = (int)v);
			this.AddNumber(section, "Trim loose surface (Poisson trim, 0 keeps all)", theme, 0, 20, true, () => this.settings.PoissonTrim, v => this.settings.PoissonTrim = v);
			this.AddChoice(section, "Camera model", theme,
				new[] { ("Automatic", "SIMPLE_RADIAL"), ("Radial", "RADIAL"), ("Simple pinhole", "SIMPLE_PINHOLE"), ("Pinhole", "PINHOLE"), ("OpenCV (wide lenses)", "OPENCV") },
				() => this.settings.CameraModel, v => this.settings.CameraModel = v);
			this.AddChoice(section, "How photos are placed", theme,
				new[] { ("One by one", MapperType.Incremental), ("All at once", MapperType.Global), ("In groups", MapperType.Hierarchical) },
				() => this.settings.Mapper, v => this.settings.Mapper = v,
				v => v != MapperType.Global || !this.settings.KnowFocalLength);
			section.AddChild(Hint("“All at once” is off while a focal length is known: it always re-estimates the focal."));
			this.AddChoice(section, "How photos are matched", theme,
				new[] { ("Every photo with every other", DataType.Individual), ("In recording order (video)", DataType.Video), ("Internet photos (many cameras)", DataType.Internet) },
				() => this.settings.Matching, v => this.settings.Matching = v);
			section.AddChild(Hint("A video's frames are matched every-with-every and still count as a video; recording order is faster but placed fewer frames of an orbit."));
			this.AddNumber(section, "Random seed (-1 = default)", theme, -1, int.MaxValue, false, () => this.settings.RandomSeed, v => this.settings.RandomSeed = (int)v, allowNegatives: true);
			this.AddCheck(section, "Use the graphics card for depth", () => this.settings.UseGpu, v => this.settings.UseGpu = v);
			this.AddCheck(section, "Keep working files after the run", () => this.settings.KeepWorkspace, v => this.settings.KeepWorkspace = v);
		}
	}
}
