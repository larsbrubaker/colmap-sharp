// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ColmapDemoApp.Run: the Run / Cancel / Save mesh controls, the stage list's live progress and
// the status line. A run is a ReconstructionSession (ReconstructionSession.cs) started off the
// UI thread on the Mac (RunOnUiThread false), or on the UI thread in the browser, where there
// is no other; its events are marshalled here with UiThread.RunOnIdle either way. Each stage's
// wall time goes to the console as "COLMAP_DEMO stage <name>: <s>" for comparing heads.
// The layout of the rest of the panel is in ColmapDemoApp.cs.

using System;
using System.Collections.Generic;
using System.Diagnostics;
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

		// The stage the timing log is on (any progress stage, "Loading photos" included) and when it began.
		private string timedStage;

		private readonly Stopwatch stageClock = new Stopwatch();

		private readonly Stopwatch runClock = new Stopwatch();

		/// <summary>
		/// Whether a run shares the UI thread instead of going to the thread pool. The browser has one
		/// thread, so there the run awaits <see cref="YieldAsync"/> between units of work to let the
		/// page paint; the Mac leaves this false so the window never waits on the pipeline.
		/// </summary>
		public bool RunOnUiThread { get; set; }

		/// <summary>Passed to the run as <see cref="SessionSettings.YieldAsync"/>; null for Task.Yield.</summary>
		public Func<ValueTask> YieldAsync { get; set; }

		/// <summary>The longer side photos are shrunk to before a run (<see cref="SessionSettings.MaxImageSize"/>).</summary>
		public int MaxImageSize { get; set; } = new SessionSettings().MaxImageSize;

		/// <summary>
		/// Whether Save mesh offers one zip (mesh.obj, mesh.mtl, mesh.png) instead of loose files: the
		/// browser's save is a single download.
		/// </summary>
		public bool SaveMeshAsZip { get; set; }

		/// <summary>Each finished stage of the last run and its wall time in seconds, in order.</summary>
		public List<(string Stage, double Seconds)> StageTimes { get; } = new List<(string Stage, double Seconds)>();

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
			this.StageTimes.Clear();
			this.timedStage = null;
			this.runClock.Restart();

			var session = new ReconstructionSession(new SessionSettings
			{
				ComputeDevice = this.computeDevice,
				MaxImageSize = this.MaxImageSize,
				YieldAsync = this.YieldAsync,
			});
			session.ProgressChanged += p => UiThread.RunOnIdle(() => this.OnProgress(p));
			session.SparseReady += points => UiThread.RunOnIdle(() =>
			{
				this.viewport.ShowPoints(points);
				this.SparseShown?.Invoke();
			});
			var photos = new List<string>(this.photoPaths);

			// The pipeline is CPU work that runs synchronously between its awaits, so on a desktop it
			// goes to the thread pool and the UI thread only draws and handles input. In the browser
			// there is no pool thread to go to; the run starts here and its yields hand the page back.
			Func<Task> run = async () =>
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
			};

			if (this.RunOnUiThread)
			{
				// Not awaited: run catches everything itself, and the click handler must return so
				// the frame that shows "Starting..." can be drawn.
				_ = run();
			}
			else
			{
				Task.Run(run);
			}
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

			if (progress.Stage != this.timedStage)
			{
				this.EndTimedStage();
				this.timedStage = progress.Stage;
				this.stageClock.Restart();
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

		// Closes the stage the timing log is on, if any.
		private void EndTimedStage()
		{
			if (this.timedStage != null)
			{
				double seconds = this.stageClock.Elapsed.TotalSeconds;
				this.StageTimes.Add((this.timedStage, seconds));
				Console.WriteLine($"COLMAP_DEMO stage {this.timedStage}: {seconds:F1} s");
				this.timedStage = null;
			}
		}

		private void OnRunFinished(SessionResult result, Exception error, bool cancelled)
		{
			this.EndTimedStage();
			Console.WriteLine($"COLMAP_DEMO run: {this.runClock.Elapsed.TotalSeconds:F1} s");
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

			SaveFileDialogParams dialogParams = this.SaveMeshAsZip
				? new SaveFileDialogParams("Zipped OBJ mesh|*.zip", title: "Save mesh") { FileName = "mesh.zip" }
				: new SaveFileDialogParams("OBJ mesh|*.obj|PLY mesh|*.ply", title: "Save mesh") { FileName = "mesh.obj" };
			AggContext.FileDialogs.SaveFileDialog(
				dialogParams,
				saveParams =>
				{
					if (string.IsNullOrEmpty(saveParams.FileName))
					{
						return;
					}

					string defaultExtension = this.SaveMeshAsZip ? ".zip" : ".obj";
					string path = Path.HasExtension(saveParams.FileName) ? saveParams.FileName : saveParams.FileName + defaultExtension;
					try
					{
						if (string.Equals(Path.GetExtension(path), ".zip", StringComparison.OrdinalIgnoreCase))
						{
							// The browser downloads the staged file once it stops changing.
							ReconstructionSession.SaveMeshZip(result, path);
							this.statusLine.Text = $"Saved {Path.GetFileName(path)} (OBJ, MTL{(result.IsTextured ? " and texture" : string.Empty)})";
							return;
						}

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
