// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColmapDemoApp.Rescale: rebuilding the demo's widgets when the window moves to a display with
// another scale. agg lays widgets out in device pixels fixed when each is built, so a new
// GuiWidget.DeviceScale needs a new widget tree; agg's UiScale.Follow (wired by each head through
// DemoDisplayScale.Follow) sets the scale and calls RebuildUi, and only when no run and no video
// read is going. The app object itself stays - the heads, DevAutoRun and the browser's dev hook hold
// it and its events - and so does its model: the photos and video frames, the settings, the last
// result and the viewport showing it. What the old widgets show (the photo list with its placement
// marks, the stage rows, the status, note and error lines, whether Settings is open) is read off
// them and put back on the new ones. The layout is built in ColmapDemoApp.cs (BuildUi).

using System;
using System.Collections.Generic;
using System.Linq;
using MatterHackers.Agg;
using MatterHackers.Agg.UI;

namespace ColmapDemo
{
	public partial class ColmapDemoApp
	{
		/// <summary>
		/// Builds the whole UI again at the current <see cref="GuiWidget.DeviceScale"/>, showing what it
		/// showed before: the photo list, the settings (and whether their panel is open), the last run's
		/// stages, status and placement, and the model in the viewport as the user turned it.
		/// </summary>
		/// <exception cref="InvalidOperationException">A run or a video read is going: their progress
		/// reports hold widgets of the old tree. <see cref="DemoDisplayScale.Follow"/> waits for both to end.</exception>
		public void RebuildUi()
		{
			if (this.IsRunning || this.IsReadingVideo)
			{
				throw new InvalidOperationException("The UI cannot be rebuilt while a run or a video read is going.");
			}

			// A number typed but not confirmed is the user's; read it before its field goes.
			this.CommitSettingsEdits();

			bool settingsOpen = this.SettingsOpen;
			bool advancedOpen = this.FindDescendant("Advanced Section") is CollapsingHeader { Expanded: true };
			(string Text, bool Visible) note = (this.noteLine.Text, this.noteLine.Visible);
			(string Text, bool Visible) error = (this.errorLine.Text, this.errorLine.Visible);
			(string Text, bool Visible) placement = (this.placementLine.Text, this.placementLine.Visible);
			string status = this.statusLine.Text;
			string computeNote = this.computeNoteLine.Text;
			var stages = this.stageRows.ToDictionary(r => r.Key, r => (r.Value.Text, r.Value.TextColor));

			// Each list line as first added and as shown now (a placement mark, greyed).
			var entries = this.listEntries.Select(e => (e.Text, e.Color, e.Paths, Shown: e.Line.Text, ShownColor: e.Line.TextColor)).ToList();

			// The viewport is kept, not rebuilt: it holds the model and the trackball's view of it.
			this.RemoveChild(this.viewport);
			this.viewport.ClearRemovedFlag();
			this.CloseChildren();
			this.settingsControls.Clear();
			this.settingsRefreshers.Clear();
			this.settingsCommitters.Clear();
			this.conditionalControls.Clear();
			this.stageRows.Clear();
			this.listEntries.Clear();

			this.BuildUi(computeNote);
			this.viewport.RebuildForScale();

			foreach ((string text, Color color, IReadOnlyList<string> paths, string shown, Color shownColor) in entries)
			{
				TextWidget line = this.AddListLine(text);
				line.TextColor = color;
				this.AddListEntry(line, paths);
				line.Text = shown;
				line.TextColor = shownColor;
			}

			foreach (KeyValuePair<string, (string Text, Color Color)> stage in stages)
			{
				this.stageRows[stage.Key].Text = stage.Value.Text;
				this.stageRows[stage.Key].TextColor = stage.Value.Color;
			}

			this.noteLine.Text = note.Text;
			this.noteLine.Visible = note.Visible;
			this.errorLine.Text = error.Text;
			this.errorLine.Visible = error.Visible;
			this.placementLine.Text = placement.Text;
			this.placementLine.Visible = placement.Visible;
			this.statusLine.Text = status;
			if (this.FindDescendant("Advanced Section") is CollapsingHeader advanced)
			{
				advanced.Expanded = advancedOpen;
			}

			this.SettingsOpen = settingsOpen;
			this.UpdatePhotoCount();
		}
	}
}
