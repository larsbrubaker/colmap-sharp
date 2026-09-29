// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColmapDemoApp.Placement: what the panel says about which photos a run placed, and which camera
// setup a run gets. Each line of the photo list (ColmapDemoApp.cs; a video's frames share one line,
// ColmapDemoApp.Video.cs) remembers the photos behind it, so after a run (ColmapDemoApp.Run.cs) a
// line whose photos were not placed says so, and the summary line under the status reads
// "Placed N of M photos." with the groups and the advice PhotoPlacement.cs words.

using System;
using System.Collections.Generic;
using System.Linq;
using MatterHackers.Agg;
using MatterHackers.Agg.UI;

namespace ColmapDemo
{
	public partial class ColmapDemoApp
	{
		private static readonly Color UnplacedColor = new Color("#a0a0a0");

		// Each photo list line, its text and color before any placement mark, and the photos behind it.
		private readonly List<(TextWidget Line, string Text, Color Color, IReadOnlyList<string> Paths)> listEntries = new List<(TextWidget Line, string Text, Color Color, IReadOnlyList<string> Paths)>();

		private WrappedTextWidget placementLine;

		/// <summary>The last run's placement summary ("Placed N of M photos. …"), or empty.</summary>
		public string PlacementText => this.placementLine.Visible ? this.placementLine.Text : string.Empty;

		/// <summary>
		/// Whether a run over the listed photos gives them one shared camera: when they are exactly
		/// one video's frames. Photos from files keep a camera each, since they may come from
		/// different cameras or zoom settings.
		/// </summary>
		public bool RunUsesSingleCamera => UsesSingleCamera(this.photoPaths, this.videoFrameSets);

		/// <summary>
		/// Whether <paramref name="photos"/> are all the frames of one of <paramref name="videoFrames"/>
		/// and nothing else: every frame of one video comes from the same lens at the same zoom.
		/// </summary>
		public static bool UsesSingleCamera(IReadOnlyList<string> photos, IEnumerable<IReadOnlyList<string>> videoFrames)
		{
			return photos.Count > 0
				&& videoFrames.Any(frames => frames.Count == photos.Count && photos.All(new HashSet<string>(frames, StringComparer.Ordinal).Contains));
		}

		/// <summary>
		/// Shows <paramref name="placement"/> for a run over <paramref name="runPhotos"/> (the photo
		/// paths the run was given, in the order its placement lists them): the summary line, and a
		/// mark on each list line with photos that are not in the shown model.
		/// </summary>
		public void ShowPlacement(PhotoPlacement placement, IReadOnlyList<string> runPhotos)
		{
			this.ClearPlacement();
			if (placement == null)
			{
				return;
			}

			var states = new Dictionary<string, PlacementState>(StringComparer.Ordinal);
			for (int i = 0; i < runPhotos.Count && i < placement.States.Count; i++)
			{
				states[runPhotos[i]] = placement.States[i];
			}

			foreach (var entry in this.listEntries)
			{
				int notPlaced = entry.Paths.Count(p => states.TryGetValue(p, out PlacementState s) && s == PlacementState.NotPlaced);
				int otherGroup = entry.Paths.Count(p => states.TryGetValue(p, out PlacementState s) && s == PlacementState.OtherGroup);
				if (notPlaced + otherGroup == 0)
				{
					continue;
				}

				string mark;
				if (entry.Paths.Count == 1)
				{
					mark = notPlaced > 0 ? "not placed" : "in a smaller group";
				}
				else
				{
					var parts = new List<string>();
					if (notPlaced > 0)
					{
						parts.Add($"{notPlaced} not placed");
					}

					if (otherGroup > 0)
					{
						parts.Add($"{otherGroup} in smaller groups");
					}

					mark = string.Join(", ", parts);
				}

				entry.Line.Text = $"{entry.Text} — {mark}";

				// Greyed only when none of its photos made it into the shown model.
				if (notPlaced + otherGroup == entry.Paths.Count)
				{
					entry.Line.TextColor = UnplacedColor;
				}
			}

			this.placementLine.Text = placement.StatusText;
			this.placementLine.Visible = true;
		}

		private void ClearPlacement()
		{
			foreach (var entry in this.listEntries)
			{
				entry.Line.Text = entry.Text;
				entry.Line.TextColor = entry.Color;
			}

			this.placementLine.Visible = false;
		}

		private void AddListEntry(TextWidget line, IReadOnlyList<string> paths)
		{
			this.listEntries.Add((line, line.Text, line.TextColor, paths));
		}

		private void AddPlacementLine(FlowLayoutWidget panel)
		{
			// Wrapped: with several groups the summary runs to a few lines.
			this.placementLine = new WrappedTextWidget(string.Empty, pointSize: 9, textColor: HintColor)
			{
				HAnchor = HAnchor.Stretch,
				Margin = new BorderDouble(0, 0, 0, 4),
				Visible = false,
			};
			panel.AddChild(this.placementLine);
		}
	}
}
