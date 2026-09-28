// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// The browser head's entry point: ColmapDemoApp (demo/ColmapDemo) on the page's canvas, the
// twin of ColmapDemo.Mac/Program.cs. Set up as agg-sharp's
// examples/AggSharpDemo/AggSharpDemo.Browser/Program.cs is (demo/agg-sharp/examples/BrowserHost/
// README.md says why each piece is here), plus what a pipeline run needs on a page: one thread,
// so the run shares it and yields (BrowserYield.cs); the GPU device made asynchronously after
// the window is up, with a timeout (Run waits for it); small
// photos; saves as one zip download. BrowserDevHook.cs is the scripted-run hook (?demo=autorun)
// and the RunState export demo/scripts/check-site.py polls.

using System;
using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;
using System.Threading.Tasks;
using ColmapDemo.Compute;
using MatterHackers.Agg.Platform;
using MatterHackers.Agg.Platform.Browser;
using MatterHackers.Agg.UI;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.JSInterop;

namespace ColmapDemo
{
	/// <summary>
	/// Blazor is only the loader: it boots the runtime, serves the static assets and, through
	/// <c>RunAsync</c>, keeps the runtime resident after <c>ShowAsSystemWindow</c> returns. agg owns
	/// the canvas and its frame loop.
	/// </summary>
	public static partial class ColmapDemoBrowserProgram
	{
		/// <summary>
		/// The longer side photos are shrunk to in the browser. .NET runs interpreted there (AOT
		/// crashes on .NET 10.0.11) and on one thread, so the CPU stages run tens of times slower than
		/// on a desktop; 640 px keeps a handful of phone photos to minutes rather than an hour, and
		/// still gives SIFT enough texture to match (the Mac default is 1000).
		/// </summary>
		public const int BrowserMaxImageSize = 640;

		/// <summary>How long the GPU request may take before the app goes on without a GPU.</summary>
		private static readonly TimeSpan GpuCheckTimeout = TimeSpan.FromSeconds(10);

		/// <summary>The page's status line writer; null outside a browser.</summary>
		private static IJSInProcessRuntime pageScript;

		/// <summary>The app on the page, for the dev hook and <see cref="RunState"/>.</summary>
		private static ColmapDemoApp app;

		[SupportedOSPlatform("browser")]
		public static async Task Main(string[] args)
		{
			WebAssemblyHostBuilder builder = WebAssemblyHostBuilder.CreateDefault(args);
			string baseAddress = builder.HostEnvironment.BaseAddress;
			WebAssemblyHost host = builder.Build();

			pageScript = host.Services.GetRequiredService<IJSRuntime>() as IJSInProcessRuntime;

			try
			{
				// The window, clipboard and dialogs call into JS modules that must be imported first,
				// and an import is a promise only a head can await.
				await BrowserHostBootstrap.InitializeAsync();

				AggContext.Config.ProviderTypes.OsInformationProvider = AggContext.ProviderSettings.BrowserOsInformationProvider;
				AggContext.Config.ProviderTypes.DialogProvider = AggContext.ProviderSettings.BrowserDialogProvider;
				AggContext.Config.ProviderTypes.SystemWindowProvider = AggContext.ProviderSettings.BrowserSystemWindowProvider;

				// Wasm is one thread (as MatterCAD.Wasm/Program.cs sets up): agg's loops run plain, and
				// nothing needs marshalling to a UI thread when there is only the one.
				MatterHackers.Agg.Parallel.Sequential = true;
				MainThreadDispatcher.MainThreadRequired = false;

				// "This browser cannot run WebGPU" arrives before there is a canvas to draw it on.
				BrowserSystemWindow.ReportStatus = Report;

				var systemWindow = new SystemWindow(1200, 800)
				{
					Title = "ColmapSharp — photos to mesh",
				};
				// The browser host does not deliver dropped files yet (docs/DEMO_PLAN.md phase 3).
				// The window comes up before the GPU check, which can take a while (or never answer),
				// and Run waits for the check.
				app = new ColmapDemoApp(fileDropSupported: false, null, "Checking for a GPU…")
				{
					RunOnUiThread = true,
					YieldAsync = BrowserYield.YieldAsync,
					MaxImageSize = BrowserMaxImageSize,
					SaveMeshAsZip = true,
					ComputeCheckPending = true,
				};
				systemWindow.AddChild(app);
				systemWindow.ShowAsSystemWindow();

				Report(string.Empty);

				(WebGpuComputeDevice gpu, string computeNote) = await CreateGpuAsync();
				Console.WriteLine("COLMAP_DEMO compute: " + computeNote);
				app.SetComputeDevice(gpu, computeNote);

				await BrowserDevHook.AttachAsync(app, baseAddress, pageScript);
			}
			catch (Exception startupException)
			{
				// Reported rather than rethrown: a throw here would take RunAsync with it and leave a
				// page with no runtime and no explanation.
				Report("startup failed: " + startupException);
			}

			await host.RunAsync();
		}

		/// <summary>
		/// The GPU PatchMatch runs on, or null and why not. The browser can never block on the GPU, so
		/// the device is made (and later read back) asynchronously; a request that has not settled in
		/// <see cref="GpuCheckTimeout"/> counts as no GPU, and a device that arrives after that is
		/// released. ?gpu=off (BrowserDevHook.cs) skips the request, to time the CPU path.
		/// </summary>
		[SupportedOSPlatform("browser")]
		private static async Task<(WebGpuComputeDevice Device, string Note)> CreateGpuAsync()
		{
			if (BrowserDevHook.GpuOff(pageScript))
			{
				return (null, "GPU off (?gpu=off); depth maps run on the CPU.");
			}

			Task<WebGpuComputeDevice> request;
			try
			{
				request = WebGpuComputeDevice.CreateAsync(raiseComputeLimits: true, isBrowser: () => true).AsTask();
			}
			catch (Exception e)
			{
				return (null, "No GPU (" + ColmapDemoApp.FirstLine(e.Message) + "); depth maps run on the CPU.");
			}

			if (await Task.WhenAny(request, Task.Delay(GpuCheckTimeout)) != request)
			{
				_ = request.ContinueWith(late => late.Result.Dispose(), TaskContinuationOptions.OnlyOnRanToCompletion);
				return (null, $"No GPU (it did not answer in {GpuCheckTimeout.TotalSeconds:0} s); depth maps run on the CPU.");
			}

			try
			{
				return (await request, "Depth maps run on the GPU.");
			}
			catch (Exception e)
			{
				return (null, "No GPU (" + ColmapDemoApp.FirstLine(e.Message) + "); depth maps run on the CPU.");
			}
		}

		/// <summary>
		/// Whether the page has really drawn, for a script driving it (the Pages smoke check): the
		/// painted frame count and whether the WebGPU device is up, as "paints N, renderer
		/// ready|not ready", or "no window" before there is one. A page screenshot alone cannot
		/// answer this - "loading..." on a dark page is already a picture.
		/// </summary>
		[SupportedOSPlatform("browser")]
		[JSExport]
		internal static string PaintState()
		{
			BrowserSystemWindow window = BrowserSystemWindow.Current;
			if (window == null)
			{
				return "no window";
			}

			return $"paints {window.FrameTick.PaintCount}, renderer {(window.RenderLayerReady ? "ready" : "not ready")}";
		}

		/// <summary>
		/// Where a run is, for a script driving the page: "running|idle", the status line, and the
		/// stage times so far, as one line ("idle | Done: 60,805 triangles, textured | Feature
		/// extraction=12.3; ...").
		/// </summary>
		[SupportedOSPlatform("browser")]
		[JSExport]
		internal static string RunState()
		{
			if (app == null)
			{
				return "no app";
			}

			var times = new System.Text.StringBuilder();
			foreach ((string stage, double seconds) in app.StageTimes)
			{
				times.Append(stage).Append('=').Append(seconds.ToString("F1", System.Globalization.CultureInfo.InvariantCulture)).Append("; ");
			}

			return $"{(app.IsRunning ? "running" : "idle")} | {app.StatusText} | {times}";
		}

		private static void Report(string message)
		{
			if (!string.IsNullOrEmpty(message))
			{
				Console.WriteLine(message);
			}

			pageScript?.InvokeVoid("aggHostStatus", message);
		}
	}
}
