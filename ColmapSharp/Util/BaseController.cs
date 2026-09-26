// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// BaseController: port of colmap/util/base_controller.h and .cc, the base of the pipeline
// controllers (Controllers/IncrementalPipeline.cs, Controllers/BundleAdjustmentController.cs).
// It separates a controller's run logic from how it is hosted: numbered callbacks the
// controller fires at milestones, and a stop check the run loop polls. Tests:
// ColmapSharp.Tests/Util/BaseControllerTests.cs (base_controller_test.cc 1:1).
//
// Translation notes:
// - COLMAP's CheckIfStopped also honors ScopedSignalHandler (Ctrl-C in the CLI). The library
//   has no signal handler; its host cancels through CancellationToken instead, which
//   CheckIfStopped reads next to the optional stop function.
// - The NodeHashMap of std::list callbacks is a Dictionary of Lists; callbacks of one id run
//   in insertion order, and an exception from one stops the rest, as in C++.

namespace ColmapSharp.Util;

/// <summary>
/// Port of colmap::BaseController: base class for controllers with registered callbacks and
/// a cooperative stop check.
/// </summary>
public abstract class BaseController
{
	private readonly Dictionary<int, List<Action>> _callbacks = [];
	private Func<bool>? _checkIfStoppedFn;

	/// <summary>
	/// The host's cancellation. When cancelled, <see cref="CheckIfStopped"/> returns true and
	/// the controller stops at its next check (COLMAP's Ctrl-C signal handler).
	/// </summary>
	public CancellationToken CancellationToken { get; set; }

	/// <summary>Adds a callback to a registered id; throws if the id was never registered.</summary>
	public void AddCallback(int id, Action func)
	{
		Check.NotNull(func);
		Check.That(_callbacks.ContainsKey(id), "Callback not registered");
		_callbacks[id].Add(func);
	}

	/// <summary>Calls every callback of the id in the order they were added.</summary>
	public void Callback(int id)
	{
		Check.That(_callbacks.TryGetValue(id, out List<Action>? callbacks), "Callback not registered");
		foreach (Action callback in callbacks)
		{
			callback();
		}
	}

	/// <summary>The main run function implemented by the controller.</summary>
	public abstract void Run();

	/// <summary>Sets (or replaces) the function used to check if the controller is stopped.</summary>
	public void SetCheckIfStoppedFunc(Func<bool>? func) => _checkIfStoppedFn = func;

	/// <summary>True when the host cancelled or the stop function says so.</summary>
	public bool CheckIfStopped() =>
		CancellationToken.IsCancellationRequested || (_checkIfStoppedFn is not null && _checkIfStoppedFn());

	/// <summary>
	/// Registers a callback id. Only registered ids can be added to and called, so a derived
	/// controller registers its ids in its constructor.
	/// </summary>
	protected void RegisterCallback(int id) => _callbacks.TryAdd(id, []);
}
