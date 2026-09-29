// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColmapDemoApp.SettingsControls: the building blocks of the Settings panel (ColmapDemoApp.Settings.cs)
// - subject cards, segmented choices, drop-down choices, checkboxes and number fields. Each one reads
// its value from DemoSettings through a getter, writes it back through a setter, and registers a
// refresh (for Reset and a video's defaults) and itself (for locking during a run).

using System;
using System.Collections.Generic;
using System.Linq;
using MatterHackers.Agg;
using MatterHackers.Agg.UI;

namespace ColmapDemo
{
	public partial class ColmapDemoApp
	{
		private static readonly Color CardColor = Color.White;

		private static readonly Color CardSelectedColor = new Color("#e3eefb");

		private static readonly Color CardBorderColor = new Color("#c8c8c8");

		private static readonly Color AmberColor = new Color("#a86400");

		private static WrappedTextWidget Hint(string text, Color? color = null) =>
			new WrappedTextWidget(text, pointSize: 9, textColor: color ?? HintColor) { HAnchor = HAnchor.Stretch, Margin = new BorderDouble(0, 4, 0, 0) };

		private static TextWidget Label(string text) =>
			new TextWidget(text, pointSize: 10) { HAnchor = HAnchor.Left, Margin = new BorderDouble(0, 2, 0, 8), AutoExpandBoundsToText = true };

		private void Register(GuiWidget control, Action refresh)
		{
			refresh();
			this.settingsRefreshers.Add(refresh);
			this.settingsControls.Add(control);
		}

		/// <summary>Large side-by-side cards, one per choice, each a title and a line under it.</summary>
		private void AddCards<T>(GuiWidget section, (string Title, string Line, T Value)[] items, Func<T> get, Action<T> set)
		{
			var row = new FlowLayoutWidget(FlowDirection.LeftToRight) { HAnchor = HAnchor.Stretch, Margin = new BorderDouble(0, 4, 0, 0) };
			var cards = new List<GuiWidget>();
			foreach ((string title, string line, T value) in items)
			{
				var card = new FlowLayoutWidget(FlowDirection.TopToBottom)
				{
					Name = title + " Card",
					HAnchor = HAnchor.Stretch,
					Padding = 8,
					Margin = new BorderDouble(cards.Count == 0 ? 0 : 4, 0, cards.Count == items.Length - 1 ? 0 : 4, 0),
					Border = 1,
					BorderColor = CardBorderColor,
					Cursor = Cursors.Hand,
				};
				card.AddChild(new TextWidget(title, pointSize: 11, bold: true) { HAnchor = HAnchor.Left, Selectable = false });
				card.AddChild(new WrappedTextWidget(line, pointSize: 9, textColor: HintColor) { HAnchor = HAnchor.Stretch, Selectable = false });
				card.Click += (sender, e) =>
				{
					if (card.Enabled)
					{
						set(value);
						this.RefreshSettingsControls();
					}
				};
				cards.Add(card);
				row.AddChild(card);
			}

			section.AddChild(row);
			this.Register(row, () =>
			{
				for (int i = 0; i < cards.Count; i++)
				{
					bool selected = EqualityComparer<T>.Default.Equals(items[i].Value, get());
					cards[i].BackgroundColor = selected ? CardSelectedColor : CardColor;
					cards[i].BorderColor = selected ? ActiveStageColor : CardBorderColor;
				}
			});
		}

		private SegmentedControl AddSegmented<T>(GuiWidget section, string label, ThemeConfig theme, (string Label, T Value)[] items, Func<T> get, Action<T> set)
		{
			if (label != null)
			{
				section.AddChild(Label(label));
			}

			var control = new SegmentedControl(items.Select(i => i.Label), theme) { Name = (label ?? items[0].Label) + " Setting", HAnchor = HAnchor.Stretch, Margin = new BorderDouble(0, 2, 0, 0) };
			bool refreshing = false;
			control.SelectedIndexChanged += (sender, e) =>
			{
				if (!refreshing)
				{
					set(items[control.SelectedIndex].Value);
					this.RefreshSettingsControls();
				}
			};
			section.AddChild(control);
			this.Register(control, () =>
			{
				refreshing = true;
				control.SelectedIndex = Array.FindIndex(items, i => EqualityComparer<T>.Default.Equals(i.Value, get()));
				refreshing = false;
			});
			return control;
		}

		/// <param name="itemEnabled">Which items can be picked right now (null: all).</param>
		private DropDownList AddChoice<T>(GuiWidget section, string label, ThemeConfig theme, (string Label, T Value)[] items, Func<T> get, Action<T> set, Func<T, bool> itemEnabled = null)
		{
			if (label != null)
			{
				section.AddChild(Label(label));
			}

			var list = new DropDownList("Other", theme.TextColor, pointSize: 10) { Name = (label ?? items[0].Label) + " Setting", HAnchor = HAnchor.Stretch };
			foreach ((string itemLabel, T _) in items)
			{
				list.AddItem(itemLabel);
			}

			bool refreshing = false;
			list.SelectionChanged += (sender, e) =>
			{
				if (!refreshing && list.SelectedIndex >= 0)
				{
					set(items[list.SelectedIndex].Value);
					this.RefreshSettingsControls();
				}
			};
			section.AddChild(list);
			this.Register(list, () =>
			{
				refreshing = true;
				list.SelectedIndex = Array.FindIndex(items, i => EqualityComparer<T>.Default.Equals(i.Value, get()));
				refreshing = false;
				for (int i = 0; itemEnabled != null && i < items.Length; i++)
				{
					list.MenuItems[i].Enabled = itemEnabled(items[i].Value);
				}
			});
			return list;
		}

		/// <param name="enabledWhen">Beyond the run lock, when the box can be used (null: always).</param>
		private CheckBox AddCheck(GuiWidget section, string label, Func<bool> get, Action<bool> set, Func<bool> enabledWhen = null)
		{
			var check = new CheckBox(label, textSize: 10) { Name = label + " Setting", HAnchor = HAnchor.Left, Margin = new BorderDouble(0, 0, 0, 8) };
			bool refreshing = false;
			check.CheckedStateChanged += (sender, e) =>
			{
				if (!refreshing)
				{
					set(check.Checked);
					this.RefreshSettingsControls();
				}
			};
			section.AddChild(check);
			if (enabledWhen != null)
			{
				this.conditionalControls.Add((check, enabledWhen));
			}

			this.Register(check, () =>
			{
				refreshing = true;
				check.Checked = get();
				refreshing = false;
			});
			return check;
		}

		private ThemedNumberEdit AddNumber(GuiWidget section, string label, ThemeConfig theme, double min, double max, bool decimals, Func<double> get, Action<double> set, bool allowNegatives = false)
		{
			if (label != null)
			{
				section.AddChild(Label(label));
			}

			var edit = new ThemedNumberEdit(get(), theme, pixelWidth: 90 * DeviceScale, allowNegatives: allowNegatives, allowDecimals: decimals, minValue: min, maxValue: max)
			{
				Name = (label ?? "Number") + " Setting",

				// A left-to-right row only lays out Absolute or Stretch children.
				HAnchor = section is FlowLayoutWidget { FlowDirection: FlowDirection.LeftToRight } ? HAnchor.Absolute : HAnchor.Left,
			};
			// Only a number the user changed is read back: a field showing an old value must not
			// overwrite a setting changed in code (a head's default, a video's defaults).
			double shown = get();
			Action commit = () =>
			{
				// A cleared or unreadable field keeps the value it had; agg would read it as its minimum.
				if (!double.TryParse(edit.Text, out _))
				{
					edit.Value = shown;
					return;
				}

				if (edit.Value != shown)
				{
					set(Math.Clamp(edit.Value, min, max));
					shown = get();
					edit.Value = shown;
				}
			};
			edit.ActuallNumberEdit.EditComplete += (sender, e) => commit();
			this.settingsCommitters.Add(commit);
			section.AddChild(edit);
			this.Register(edit, () =>
			{
				shown = get();
				edit.Value = shown;
			});
			return edit;
		}
	}
}
