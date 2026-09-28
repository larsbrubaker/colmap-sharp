// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// BrowserYield: what a run on the page's one thread awaits between units of work (Program.cs
// hands it to ColmapDemoApp.YieldAsync). It resolves on a setTimeout(0) task, from
// colmapDemoYield in wwwroot/index.html, so the browser gets a whole event-loop turn - the
// rendering step and agg's requestAnimationFrame included - before the run goes on.
// ?yield=task (BrowserDevHook.cs) switches back to the library's default Task.Yield, for
// measuring the two against each other.

using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;
using System.Threading.Tasks;

namespace ColmapDemo
{
	/// <summary>The browser's yield to the page between units of pipeline work.</summary>
	[SupportedOSPlatform("browser")]
	internal static partial class BrowserYield
	{
		/// <summary>Whether to await Task.Yield instead of a setTimeout task (the ?yield=task switch).</summary>
		public static bool UseTaskYield { get; set; }

		/// <summary>Resolves after the page has had a turn of its event loop.</summary>
		public static async ValueTask YieldAsync()
		{
			if (UseTaskYield)
			{
				await Task.Yield();
				return;
			}

			await NextTask();
		}

		// A promise from setTimeout(0): a new task, so the browser may render before it runs. Not
		// requestAnimationFrame, which a hidden tab never fires and would stall the run there.
		[JSImport("globalThis.colmapDemoYield")]
		private static partial Task NextTask();
	}
}
