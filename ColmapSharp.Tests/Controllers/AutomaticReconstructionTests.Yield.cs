// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AutomaticReconstructionTests (continued): C#-only tests of the cooperative yields of
// AutomaticReconstructionController.RunAsync (YieldAsync), not a port - COLMAP runs on its own
// thread and never yields. In the browser .NET has one thread, so RunAsync must hand the event
// loop back between units of work. These pin where it yields (after every stage, dense step and
// PatchMatch problem, so no two units run between yields), that it really returns to its
// caller there, that a stop requested during a yield takes effect before the next unit, and
// that the synchronous Run never yields. AutomaticReconstructionTests.RunAsync.cs pins that the
// yields leave the outputs byte-identical to Run's.

using ColmapSharp.Controllers;
using ColmapSharp.Scene;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Controllers;

public partial class AutomaticReconstructionTests
{
	private const string YieldMark = "<yield>";

	[Test]
	public async Task CSharpOnly_RunAsyncYieldsBetweenEveryUnitOfWork()
	{
		string testDir = CreateTestDir();
		InMemoryImageSource images = TexturedSceneImages(NumAsyncViews);
		string workspacePath = Path.Combine(testDir, "yield");
		Directory.CreateDirectory(workspacePath);
		var events = new List<string>();
		AutomaticReconstructionController controller = NewAsyncTestController(
			workspacePath, images, resume: false, device: null, new EventLog(events));
		controller.YieldAsync = () =>
		{
			lock (events)
			{
				events.Add(YieldMark);
			}

			return ValueTask.CompletedTask;
		};
		await controller.RunAsync();
		int numTextured = controller.TexturedMeshes.Count;
		DeleteTestDir(testDir);

		// The segments of work between consecutive yields (and after the last one).
		var segments = new List<List<string>> { new() };
		foreach (string e in events)
		{
			if (e == YieldMark)
			{
				segments.Add([]);
			}
			else
			{
				segments[^1].Add(e);
			}
		}

		int numPatchMatch = events.Count(e => e.Contains("|PatchMatch ", StringComparison.Ordinal));

		// Guard against a vacuous pass: every stage ran, PatchMatch problems included.
		await Assert.That(numTextured).IsEqualTo(1);
		await Assert.That(numPatchMatch).IsGreaterThanOrEqualTo(NumAsyncViews);

		// Extraction, matching, sparse; undistortion; each PatchMatch problem; fusion, meshing,
		// texturing.
		await Assert.That(events.Count(e => e == YieldMark)).IsEqualTo(3 + 1 + numPatchMatch + 3);

		// Between two yields the work is one unit: a single stage, and at most one PatchMatch
		// problem.
		foreach (List<string> segment in segments)
		{
			await Assert.That(segment.Select(e => e[..e.IndexOf('|', StringComparison.Ordinal)]).Distinct().Count())
				.IsLessThanOrEqualTo(1);
			await Assert.That(segment.Count(e => e.Contains("|PatchMatch ", StringComparison.Ordinal))).IsLessThanOrEqualTo(1);
		}

		// Nothing runs after the texturing's yield.
		await Assert.That(segments[^1].Count).IsEqualTo(0);
	}

	// Extraction only, so the test is quick. RunAsync returns to its caller at the yield after
	// extraction without starting matching; a stop requested there takes effect before
	// matching, and RunAsync then returns normally. Run never calls the hook.
	[Test]
	public async Task CSharpOnly_RunAsyncReturnsToCallerAtYieldAndStopsThere()
	{
		string testDir = CreateTestDir();
		InMemoryImageSource images = TexturedSceneImages(NumAsyncViews);

		AutomaticReconstructionController NewController(string name, List<string> events, CancellationToken token)
		{
			string workspacePath = Path.Combine(testDir, name);
			Directory.CreateDirectory(workspacePath);
			AutomaticReconstructionController controller = NewAsyncTestController(
				workspacePath, images, resume: false, device: null, new EventLog(events));
			controller.CancellationToken = token;
			return controller;
		}

		// Run with a hook: it is never called, and every stage runs.
		var syncEvents = new List<string>();
		int syncYields = 0;
		AutomaticReconstructionController syncController = NewController("sync", syncEvents, CancellationToken.None);
		syncController.YieldAsync = () =>
		{
			syncYields++;
			return ValueTask.CompletedTask;
		};
		syncController.Run();

		// RunAsync held at its first yield.
		using var stop = new CancellationTokenSource();
		var asyncEvents = new List<string>();
		var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
		int asyncYields = 0;
		AutomaticReconstructionController asyncController = NewController("async", asyncEvents, stop.Token);
		asyncController.YieldAsync = () =>
		{
			asyncYields++;
			return new ValueTask(release.Task);
		};
		Task run = asyncController.RunAsync();
		bool completedAtYield = run.IsCompleted;
		List<string> eventsAtYield;
		lock (asyncEvents)
		{
			eventsAtYield = [.. asyncEvents];
		}

		stop.Cancel();
		release.SetResult();
		await run;
		long numMatchedPairs = asyncController.Database.NumMatchedImagePairs();
		long numImages = asyncController.Database.NumImages();
		bool haveSparse = Directory.Exists(Path.Combine(testDir, "async", "sparse"));
		DeleteTestDir(testDir);

		await Assert.That(syncYields).IsEqualTo(0);
		await Assert.That(syncController.TexturedMeshes.Count).IsEqualTo(1);

		await Assert.That(completedAtYield).IsFalse();
		await Assert.That(asyncYields).IsEqualTo(1);
		await Assert.That(eventsAtYield.Count).IsGreaterThan(0);
		await Assert.That(eventsAtYield.All(e => e.StartsWith(FeatureExtraction.ExtractionStage + "|", StringComparison.Ordinal))).IsTrue();
		await Assert.That(numImages).IsEqualTo(NumAsyncViews);
		await Assert.That(numMatchedPairs).IsEqualTo(0);
		await Assert.That(haveSparse).IsFalse();
		lock (asyncEvents)
		{
			// Stopped at the yield: nothing after extraction ran.
			eventsAtYield = [.. asyncEvents];
		}

		await Assert.That(eventsAtYield.All(e => e.StartsWith(FeatureExtraction.ExtractionStage + "|", StringComparison.Ordinal))).IsTrue();
	}

	// Records each report as "Stage|Message", synchronously and in order.
	private sealed class EventLog(List<string> events) : IProgress<ControllerProgress>
	{
		public void Report(ControllerProgress value)
		{
			lock (events)
			{
				events.Add(value.Stage + "|" + value.Message);
			}
		}
	}
}
