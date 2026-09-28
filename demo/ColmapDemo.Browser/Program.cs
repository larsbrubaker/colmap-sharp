// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// The browser head's entry point: ColmapDemoApp (demo/ColmapDemo) on the page's canvas, the
// twin of ColmapDemo.Mac/Program.cs. Set up exactly as agg-sharp's
// examples/AggSharpDemo/AggSharpDemo.Browser/Program.cs is; demo/agg-sharp/examples/BrowserHost/
// README.md says why each piece is here.

using System;
using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;
using System.Threading.Tasks;
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
		/// <summary>The page's status line writer; null outside a browser.</summary>
		private static IJSInProcessRuntime pageScript;

		[SupportedOSPlatform("browser")]
		public static async Task Main(string[] args)
		{
			WebAssemblyHost host = WebAssemblyHostBuilder.CreateDefault(args).Build();

			pageScript = host.Services.GetRequiredService<IJSRuntime>() as IJSInProcessRuntime;

			try
			{
				// The window, clipboard and dialogs call into JS modules that must be imported first,
				// and an import is a promise only a head can await.
				await BrowserHostBootstrap.InitializeAsync();

				AggContext.Config.ProviderTypes.OsInformationProvider = AggContext.ProviderSettings.BrowserOsInformationProvider;
				AggContext.Config.ProviderTypes.DialogProvider = AggContext.ProviderSettings.BrowserDialogProvider;
				AggContext.Config.ProviderTypes.SystemWindowProvider = AggContext.ProviderSettings.BrowserSystemWindowProvider;

				// "This browser cannot run WebGPU" arrives before there is a canvas to draw it on.
				BrowserSystemWindow.ReportStatus = Report;

				var systemWindow = new SystemWindow(1200, 800)
				{
					Title = "ColmapSharp — photos to mesh",
				};
				// The browser host does not deliver dropped files yet (docs/DEMO_PLAN.md phase 3).
				systemWindow.AddChild(new ColmapDemoApp(fileDropSupported: false));
				systemWindow.ShowAsSystemWindow();

				Report(string.Empty);
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
