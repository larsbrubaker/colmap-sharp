// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColmapDemoApp.Run: the Run / Cancel / Save mesh controls, the stage list's live progress and
// the status line. A run is a ReconstructionSession (ReconstructionSession.cs) started off the
// UI thread; its events arrive on the worker and are marshalled here with UiThread.RunOnIdle.
// The layout of the rest of the panel is in ColmapDemoApp.cs.

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using ColmapSharp.Compute;
using ColmapSharp.Controllers;
using MatterHackers.Agg;
using MatterHackers.Agg.Platform;
using MatterHackers.Agg.UI;

namespace ColmapDemo
{
	public partial class ColmapDemoApp
	{
		/// <summary>The fewest photos a run is offered for: two views make no surface worth showing.</summary>
		public const int MinPhotosToRun = 3;

		private static readonly Color ActiveStageColor = new Color("#1060c0");

		private static readonly Color DoneStageColor = new Color("#208040");

		private static readonly Color ErrorColor = new Color("#c02020");

		private readonly Dictionary<string, TextWidget> stageRows = new Dictionary<string, TextWidget>();

		private readonly IComputeDevice computeDevice;

		private readonly ModelViewport viewport;

		private ThemedTextButton runButton;

		private ThemedTextButton cancelButton;

		private ThemedTextButton saveButton;

		private TextWidget statusLine;

		private TextWidget errorLine;

		private CancellationTokenSource runCancel;

		private SessionResult lastResult;

		private string activeStage;

		/// <summary>Raised on the UI thread once the sparse points are in the viewport.</summary>
		public event Action SparseShown;

		/// <summary>
		/// Raised on the UI thread when a run ends, after the viewport and panel show the outcome:
		/// the result, or null when it was cancelled or failed.
		/// </summary>
		public event Action<SessionResult> RunFinished;

		/// <summary>The status line under the stage list.</summary>
		public string StatusText => this.statusLine.Text;

		/// <summary>Whether the Cancel button can be pressed.</summary>
		public bool CanCancel => this.cancelButton.Enabled;

		/// <summary>Whether a run is in progress.</summary>
		public bool IsRunning => this.runCancel != null;

		/// <summary>Starts a run over the listed photos (what the Run button does).</summary>
		public void StartRun()
		{
			if (this.IsRunning || this.photoPaths.Count < MinPhotosToRun)
			{
				return;
			}

			var cancel = new CancellationTokenSource();
			this.runCancel = cancel;
			this.lastResult = null;
			this.activeStage = null;
			this.errorLine.Visible = false;
			this.viewport.Clear();
			foreach (string stage in Stages)
			{
				this.SetStageRow(stage, string.Empty, HintColor);
			}

			this.statusLine.Text = "Starting…";
			this.UpdatePhotoCount();

			var session = new ReconstructionSession(new SessionSettings { ComputeDevice = this.computeDevice });
			session.ProgressChanged += p => UiThread.RunOnIdle(() => this.OnProgress(p));
			session.SparseReady += points => UiThread.RunOnIdle(() =>
			{
				this.viewport.ShowPoints(points);
				this.SparseShown?.Invoke();
			});
			var photos = new List<string>(this.photoPaths);

			// The pipeline is CPU work that runs synchronously between its awaits, so it goes to the
			// thread pool; the UI thread only draws and handles input.
			Task.Run(async () =>
			{
				try
				{
					SessionResult result = await session.RunAsync(photos, cancel.Token).ConfigureAwait(false);
					UiThread.RunOnIdle(() => this.OnRunFinished(result, null, cancelled: false));
				}
				catch (OperationCanceledException)
				{
					UiThread.RunOnIdle(() => this.OnRunFinished(null, null, cancelled: true));
				}
				catch (Exception e)
				{
					UiThread.RunOnIdle(() => this.OnRunFinished(null, e, cancelled: false));
				}
			});
		}

		/// <summary>Asks the run to stop (what Cancel does). The current step finishes first.</summary>
		public void RequestCancel()
		{
			if (this.runCancel == null || this.runCancel.IsCancellationRequested)
			{
				return;
			}

			this.runCancel.Cancel();

			// A long step (matching, the sparse mapper, meshing) only notices at its next check, so
			// say the click registered rather than leave a live-looking button.
			this.cancelButton.Enabled = false;
			this.statusLine.Text = "Cancelling…";
		}

		private void AddRunControls(FlowLayoutWidget panel, ThemeConfig theme, string computeNote)
		{
			var runButtons = new FlowLayoutWidget(FlowDirection.LeftToRight)
			{
				HAnchor = HAnchor.Stretch,
				Margin = new BorderDouble(0, 0, 0, 12),
			};
			this.runButton = new ThemedTextButton("Run", theme) { Name = "Run Button", Enabled = false };
			this.runButton.Click += (sender, e) => this.StartRun();
			runButtons.AddChild(this.runButton);
			this.cancelButton = new ThemedTextButton("Cancel", theme) { Name = "Cancel Button", Enabled = false };
			this.cancelButton.Click += (sender, e) => this.RequestCancel();
			runButtons.AddChild(this.cancelButton);
			runButtons.AddChild(new HorizontalSpacer());
			this.saveButton = new ThemedTextButton("Save mesh…", theme) { Name = "Save Mesh Button", Enabled = false };
			this.saveButton.Click += (sender, e) => this.ShowSaveDialog();
			runButtons.AddChild(this.saveButton);
			panel.AddChild(runButtons);

			if (!string.IsNullOrEmpty(computeNote))
			{
				panel.AddChild(new TextWidget(computeNote, pointSize: 9, textColor: HintColor)
				{
					HAnchor = HAnchor.Left,
					Margin = new BorderDouble(0, 0, 0, 4),
					AutoExpandBoundsToText = true,
				});
			}

			panel.AddChild(new TextWidget("Stages", pointSize: 11, bold: true)
			{
				HAnchor = HAnchor.Left,
				Margin = new BorderDouble(0, 4, 0, 12),
			});
			foreach (string stage in Stages)
			{
				var row = new TextWidget(stage, pointSize: 10, textColor: HintColor)
				{
					HAnchor = HAnchor.Left,
					Margin = new BorderDouble(8, 2, 0, 2),
					AutoExpandBoundsToText = true,
				};
				this.stageRows[stage] = row;
				panel.AddChild(row);
			}

			this.statusLine = new TextWidget(string.Empty, pointSize: 9, textColor: HintColor)
			{
				HAnchor = HAnchor.Left,
				Margin = new BorderDouble(0, 0, 0, 8),
				AutoExpandBoundsToText = true,
			};
			panel.AddChild(this.statusLine);

			this.errorLine = new TextWidget(string.Empty, pointSize: 9, textColor: ErrorColor)
			{
				HAnchor = HAnchor.Left,
				Margin = new BorderDouble(0, 0, 0, 4),
				AutoExpandBoundsToText = true,
				Visible = false,
			};
			panel.AddChild(this.errorLine);
		}

		private void UpdateRunButtons()
		{
			// Called from the constructor's first UpdatePhotoCount, before the buttons exist.
			if (this.runButton == null)
			{
				return;
			}

			this.runButton.Enabled = !this.IsRunning && this.photoPaths.Count >= MinPhotosToRun;
			this.cancelButton.Enabled = this.IsRunning && !this.runCancel.IsCancellationRequested;
			this.saveButton.Enabled = !this.IsRunning && this.lastResult?.Mesh != null;
			this.addButton.Enabled = !this.IsRunning;
		}

		private void OnProgress(ControllerProgress progress)
		{
			// Late reports from a stopping run must not overwrite "Cancelling…".
			if (!this.IsRunning || this.runCancel.IsCancellationRequested)
			{
				return;
			}

			if (this.stageRows.ContainsKey(progress.Stage) && progress.Stage != this.activeStage)
			{
				if (this.activeStage != null)
				{
					this.SetStageRow(this.activeStage, "done", DoneStageColor);
				}

				this.activeStage = progress.Stage;
			}

			if (progress.Stage == this.activeStage)
			{
				string count = progress.Total > 0 ? $"{progress.Done}/{progress.Total}" : "…";
				this.SetStageRow(progress.Stage, count, ActiveStageColor);
			}

			string message = progress.Message.Length > 0 ? progress.Message : progress.Stage;
			this.statusLine.Text = progress.Total > 0 && !this.stageRows.ContainsKey(progress.Stage)
				? $"{message} ({progress.Done + 1}/{progress.Total})"
				: message;
		}

		private void OnRunFinished(SessionResult result, Exception error, bool cancelled)
		{
			this.runCancel?.Dispose();
			this.runCancel = null;
			this.lastResult = result;
			if (this.activeStage != null)
			{
				this.SetStageRow(this.activeStage, result != null ? "done" : cancelled ? "cancelled" : "failed", result != null ? DoneStageColor : ErrorColor);
			}

			if (result?.PreviewMesh != null)
			{
				this.viewport.ShowMesh(result.PreviewMesh);
				int faces = result.Mesh.Faces.Count;
				this.statusLine.Text = result.IsTextured ? $"Done: {faces:N0} triangles, textured" : $"Done: {faces:N0} triangles (vertex colors; no photo saw the surface)";
			}
			else if (result != null)
			{
				this.statusLine.Text = "Done, but no mesh came out. Try more photos with more overlap.";
			}
			else if (cancelled)
			{
				this.statusLine.Text = "Cancelled.";
			}
			else
			{
				this.statusLine.Text = "The run stopped with an error:";
				this.errorLine.Text = error?.Message ?? "Unknown error";
				this.errorLine.Visible = true;
			}

			this.UpdatePhotoCount();
			this.RunFinished?.Invoke(result);
		}

		private void SetStageRow(string stage, string suffix, Color color)
		{
			TextWidget row = this.stageRows[stage];
			row.Text = suffix.Length > 0 ? $"{stage}  {suffix}" : stage;
			row.TextColor = color;
		}

		private void ShowSaveDialog()
		{
			SessionResult result = this.lastResult;
			if (result?.Mesh == null)
			{
				return;
			}

			AggContext.FileDialogs.SaveFileDialog(
				new SaveFileDialogParams("OBJ mesh|*.obj|PLY mesh|*.ply", title: "Save mesh") { FileName = "mesh.obj" },
				saveParams =>
				{
					if (string.IsNullOrEmpty(saveParams.FileName))
					{
						return;
					}

					string path = Path.HasExtension(saveParams.FileName) ? saveParams.FileName : saveParams.FileName + ".obj";
					try
					{
						ReconstructionSession.SaveMesh(result, path);
						this.statusLine.Text = result.IsTextured && !path.EndsWith(".ply", StringComparison.OrdinalIgnoreCase)
							? $"Saved {Path.GetFileName(path)} with its .mtl and {Path.GetFileNameWithoutExtension(path)}.png"
							: $"Saved {Path.GetFileName(path)}";
					}
					catch (Exception e)
					{
						this.errorLine.Text = "Could not save: " + e.Message;
						this.errorLine.Visible = true;
					}
				});
		}
	}
}
