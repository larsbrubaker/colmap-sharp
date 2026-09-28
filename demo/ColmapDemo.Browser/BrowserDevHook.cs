// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BrowserDevHook: the browser twin of ColmapDemo.Mac/DevAutoRun.cs, driven by the page's query
// string so demo/scripts/check-site.py can run the pipeline unattended:
//
//   ?demo=autorun     load the bundled sample photos (wwwroot/samples) and press Run
//   ?demo=samples     only load them, for a person to press Run
//   &yield=task       yield with Task.Yield instead of a setTimeout task (BrowserYield.cs)
//   gpu=off           no GPU device: PatchMatch on the CPU (read by Program.cs at startup)
//
// The samples are our own synthetic renders, not photos: views 0-5 of 6 at 320x240 from
// RenderTexturedScene in ColmapSharp.Tests/Controllers/AutomaticReconstructionTests.CSharpOnly.cs
// (a ray-traced unit sphere before a wall, both carrying a 3D value-noise texture; cameras 12
// degrees apart on an arc). To regenerate them, call that method as
// RenderTexturedScene(i, numViews: 6, width: 320, height: 240) for i = 0..5 from a scratch
// program (it is private to the test class, so copy it or make it internal) and write each
// Bitmap as wwwroot/samples/view{i}.png. ColmapDemo.Tests runs the pipeline on them too.
//
// Without a demo parameter it does nothing. Progress is read through Program.cs's RunState export.

using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Runtime.Versioning;
using System.Threading.Tasks;
using MatterHackers.Agg.UI;
using Microsoft.JSInterop;

namespace ColmapDemo
{
	[SupportedOSPlatform("browser")]
	internal static class BrowserDevHook
	{
		/// <summary>The bundled sample photos, under wwwroot/samples.</summary>
		private static readonly string[] SampleNames = { "view0.png", "view1.png", "view2.png", "view3.png", "view4.png", "view5.png" };

		/// <summary>Whether the page's query asks for no GPU device (?gpu=off).</summary>
		public static bool GpuOff(IJSInProcessRuntime page)
			=> (page?.Invoke<string>("colmapDemoQuery") ?? string.Empty).Contains("gpu=off", StringComparison.Ordinal);

		/// <summary>Reads the query and, if asked, loads the samples and starts a run.</summary>
		public static async Task AttachAsync(ColmapDemoApp app, string baseAddress, IJSInProcessRuntime page)
		{
			string query = page?.Invoke<string>("colmapDemoQuery") ?? string.Empty;
			BrowserYield.UseTaskYield = query.Contains("yield=task", StringComparison.Ordinal);
			bool autorun = query.Contains("demo=autorun", StringComparison.Ordinal);
			if (!autorun && !query.Contains("demo=samples", StringComparison.Ordinal))
			{
				return;
			}

			// Fetched into the wasm file system, where the photo list expects paths (as the file
			// picker stages its files).
			string folder = Path.Combine(Path.GetTempPath(), "ColmapDemoSamples");
			Directory.CreateDirectory(folder);
			var paths = new List<string>();
			using (var http = new HttpClient { BaseAddress = new Uri(baseAddress) })
			{
				foreach (string name in SampleNames)
				{
					byte[] bytes = await http.GetByteArrayAsync("samples/" + name);
					string path = Path.Combine(folder, name);
					File.WriteAllBytes(path, bytes);
					paths.Add(path);
				}
			}

			Console.WriteLine($"COLMAP_DEMO loaded {paths.Count} sample photos (yield: {(BrowserYield.UseTaskYield ? "Task.Yield" : "setTimeout")})");
			UiThread.RunOnIdle(() =>
			{
				app.AddPhotos(paths);
				if (autorun)
				{
					app.StartRun();
				}
			});
		}
	}
}
