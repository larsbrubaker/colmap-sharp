// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ReconstructionManager: port of colmap/scene/reconstruction_manager.h and .cc, the list of
// Reconstructions (Reconstruction.cs) the incremental mapper produces, one per connected
// model. Tests: ColmapSharp.Tests/Scene/ReconstructionManagerTests.cs
// (reconstruction_manager_test.cc 1:1).
//
// Not ported yet: Read and Write (they call Reconstruction.Read/Write, which wait on
// scene/reconstruction_io). shared_ptr<Reconstruction> is a plain reference; C++'s
// non-const Get returns a reference to the shared_ptr so callers can swap the model, which
// here is Set.

using ColmapSharp.Util;

namespace ColmapSharp.Scene;

/// <summary>Port of colmap::ReconstructionManager.</summary>
public sealed class ReconstructionManager
{
	private readonly List<Reconstruction> _reconstructions = [];

	/// <summary>The number of reconstructions managed.</summary>
	public int Size => _reconstructions.Count;

	/// <summary>The reconstruction at <paramref name="idx"/>; throws if out of range (C++ at()).</summary>
	public Reconstruction Get(int idx) => _reconstructions[idx];

	/// <summary>Replaces the reconstruction at <paramref name="idx"/> (assignment through C++'s non-const Get).</summary>
	public void Set(int idx, Reconstruction reconstruction) => _reconstructions[idx] = reconstruction;

	/// <summary>Adds a new empty reconstruction and returns its index.</summary>
	public int Add()
	{
		int idx = Size;
		_reconstructions.Add(new Reconstruction());
		return idx;
	}

	/// <summary>Deletes a specific reconstruction.</summary>
	public void Delete(int idx)
	{
		Check.Lt(idx, _reconstructions.Count);
		_reconstructions.RemoveAt(idx);
	}

	/// <summary>Deletes all reconstructions.</summary>
	public void Clear() => _reconstructions.Clear();
}
