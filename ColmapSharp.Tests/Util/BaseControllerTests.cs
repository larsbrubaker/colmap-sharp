// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BaseControllerTests: colmap/util/base_controller_test.cc ported 1:1, testing
// ColmapSharp/Util/BaseController.cs. Tier A (exact): callback bookkeeping. The last test,
// CancellationTokenStops, is C#-only: the host's CancellationToken replaces COLMAP's signal
// handler.

using ColmapSharp.Util;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ColmapSharp.Tests.Util;

public class BaseControllerTests
{
	// Concrete implementation of BaseController for testing
	private sealed class TestController : BaseController
	{
		public TestController()
		{
			// Register callbacks during construction
			RegisterCallback(1);
			RegisterCallback(2);
			RegisterCallback(100);
		}

		public override void Run()
		{
			// Simple run implementation that triggers callbacks
			Callback(1);
			if (!CheckIfStopped())
			{
				Callback(2);
			}
		}
	}

	[Test]
	public async Task BaseController_AddCallbackToRegistered()
	{
		var controller = new TestController();
		int counter = 0;
		controller.AddCallback(1, () => counter += 1);
		await Assert.That(counter).IsEqualTo(0);
		controller.Callback(1);
		await Assert.That(counter).IsEqualTo(1);
	}

	[Test]
	public async Task BaseController_AddMultipleCallbacks()
	{
		var controller = new TestController();
		int counter = 0;
		controller.AddCallback(1, () => counter += 1);
		controller.AddCallback(1, () => counter += 10);
		controller.AddCallback(1, () => counter += 100);
		controller.Callback(1);
		await Assert.That(counter).IsEqualTo(111);
	}

	[Test]
	public async Task BaseController_AddCallbackToDifferentIDs()
	{
		var controller = new TestController();
		int counter1 = 0;
		int counter2 = 0;
		controller.AddCallback(1, () => counter1 += 1);
		controller.AddCallback(2, () => counter2 += 2);
		controller.Callback(1);
		await Assert.That(counter1).IsEqualTo(1);
		await Assert.That(counter2).IsEqualTo(0);
		controller.Callback(2);
		await Assert.That(counter1).IsEqualTo(1);
		await Assert.That(counter2).IsEqualTo(2);
	}

	[Test]
	public async Task BaseController_CallbackWithNoCallbacksAdded()
	{
		var controller = new TestController();
		// Calling a registered callback with no functions added should not crash
		await Assert.That(() => controller.Callback(1)).ThrowsNothing();
	}

	[Test]
	public async Task BaseController_CallbackExecutionOrder()
	{
		var controller = new TestController();
		var executionOrder = new List<int>();
		controller.AddCallback(1, () => executionOrder.Add(1));
		controller.AddCallback(1, () => executionOrder.Add(2));
		controller.AddCallback(1, () => executionOrder.Add(3));
		controller.Callback(1);
		await Assert.That(executionOrder.Count).IsEqualTo(3);
		await Assert.That(executionOrder[0]).IsEqualTo(1);
		await Assert.That(executionOrder[1]).IsEqualTo(2);
		await Assert.That(executionOrder[2]).IsEqualTo(3);
	}

	[Test]
	public async Task BaseController_SetCheckIfStoppedFunc()
	{
		var controller = new TestController();
		bool shouldStop = false;

		// Set the check function
		controller.SetCheckIfStoppedFunc(() => shouldStop);

		// Initially not stopped
		await Assert.That(controller.CheckIfStopped()).IsFalse();

		// Change the flag
		shouldStop = true;
		await Assert.That(controller.CheckIfStopped()).IsTrue();
	}

	[Test]
	public async Task BaseController_CheckIfStoppedWithoutSetting()
	{
		var controller = new TestController();
		// Without setting a check function, should return false
		await Assert.That(controller.CheckIfStopped()).IsFalse();
	}

	[Test]
	public async Task BaseController_CheckIfStoppedMultipleCalls()
	{
		var controller = new TestController();
		int callCount = 0;

		controller.SetCheckIfStoppedFunc(() =>
		{
			++callCount;
			return callCount >= 3;
		});

		await Assert.That(controller.CheckIfStopped()).IsFalse(); // call_count = 1
		await Assert.That(controller.CheckIfStopped()).IsFalse(); // call_count = 2
		await Assert.That(controller.CheckIfStopped()).IsTrue(); // call_count = 3
		await Assert.That(controller.CheckIfStopped()).IsTrue(); // call_count = 4
	}

	[Test]
	public async Task BaseController_RunMethod()
	{
		var controller = new TestController();
		int counter1 = 0;
		int counter2 = 0;

		controller.AddCallback(1, () => ++counter1);
		controller.AddCallback(2, () => ++counter2);

		// Run should trigger callbacks
		controller.Run();

		await Assert.That(counter1).IsEqualTo(1);
		await Assert.That(counter2).IsEqualTo(1);
	}

	[Test]
	public async Task BaseController_RunWithStopCheck()
	{
		var controller = new TestController();
		int counter1 = 0;
		int counter2 = 0;

		controller.AddCallback(1, () => ++counter1);
		controller.AddCallback(2, () => ++counter2);

		// Set stop function to return true
		controller.SetCheckIfStoppedFunc(() => true);

		// Run should trigger callback 1 but not callback 2 (due to stop check)
		controller.Run();

		await Assert.That(counter1).IsEqualTo(1);
		await Assert.That(counter2).IsEqualTo(0); // Should not execute due to stop
	}

	[Test]
	public async Task BaseController_ReplaceCheckIfStoppedFunc()
	{
		var controller = new TestController();
		controller.SetCheckIfStoppedFunc(() => true);
		await Assert.That(controller.CheckIfStopped()).IsTrue();
		controller.SetCheckIfStoppedFunc(() => false);
		await Assert.That(controller.CheckIfStopped()).IsFalse();
	}

	[Test]
	public async Task BaseController_CallbackWithException()
	{
		var controller = new TestController();
		int counterBefore = 0;
		int counterAfter = 0;

		controller.AddCallback(1, () => ++counterBefore);
		controller.AddCallback(1, () => throw new InvalidOperationException("test exception"));
		controller.AddCallback(1, () => ++counterAfter);
		await Assert.That(() => controller.Callback(1)).Throws<InvalidOperationException>();
		await Assert.That(counterBefore).IsEqualTo(1);
		await Assert.That(counterAfter).IsEqualTo(0);
	}

	[Test]
	public async Task BaseController_EmptyCallbackList()
	{
		var controller = new TestController();
		// Registered but no callbacks added - should not crash
		await Assert.That(() => controller.Callback(100)).ThrowsNothing();
	}

	// C#-only: a cancelled token stops the controller like COLMAP's signal handler.
	[Test]
	public async Task BaseController_CancellationTokenStops()
	{
		using var cts = new CancellationTokenSource();
		var controller = new TestController { CancellationToken = cts.Token };
		await Assert.That(controller.CheckIfStopped()).IsFalse();
		cts.Cancel();
		await Assert.That(controller.CheckIfStopped()).IsTrue();
	}

	// C#-only: AddCallback / Callback on an unregistered id fail COLMAP's CHECK.
	[Test]
	public async Task CSharpOnly_UnregisteredCallbackIdThrows()
	{
		var controller = new TestController();
		await Assert.That(() => controller.AddCallback(3, () => { })).Throws<ArgumentException>();
		await Assert.That(() => controller.Callback(3)).Throws<ArgumentException>();
	}
}
