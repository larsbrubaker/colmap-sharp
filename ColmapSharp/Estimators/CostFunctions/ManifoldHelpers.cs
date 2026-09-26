// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// ManifoldHelpers: colmap/estimators/cost_functions/manifold.h - the factory and setter
// shims COLMAP writes its Ceres manifold code against (SetManifold, CreateEuclideanManifold,
// CreateEigenQuaternionManifold, CreateSubsetManifold, CreateSphereManifold,
// CreateProductManifold, ParameterBlockTangentSize), mapped onto the Solver/ manifolds
// (Solver/Manifolds.cs, Solver/SphereProductManifolds.cs) and Problem. In C++ they bridge
// Ceres versions before and after 2.1's Manifold API; here only the Manifold side exists,
// and the helpers keep ported call sites (bundle adjustment, pose refinement) 1:1.
// manifold.h has no test file.

using ColmapSharp.Solver;

namespace ColmapSharp.Estimators.CostFunctions;

/// <summary>Port of colmap/estimators/cost_functions/manifold.h.</summary>
public static class ManifoldHelpers
{
	/// <summary>colmap::SetManifold: sets the manifold of a parameter block.</summary>
	public static void SetManifold(Problem problem, ArraySegment<double> parameters, Manifold manifold) =>
		problem.SetManifold(parameters, manifold);

	/// <summary>colmap::CreateEuclideanManifold&lt;size&gt;.</summary>
	public static Manifold CreateEuclideanManifold(int size) => new EuclideanManifold(size);

	/// <summary>colmap::CreateEigenQuaternionManifold: the [x, y, z, w] quaternion manifold.</summary>
	public static Manifold CreateEigenQuaternionManifold() => QuaternionManifold.EigenOrder();

	/// <summary>colmap::CreateSubsetManifold: holds the listed coordinates constant.</summary>
	public static Manifold CreateSubsetManifold(int size, IReadOnlyList<int> constantParams) =>
		new SubsetManifold(size, constantParams);

	/// <summary>colmap::CreateSphereManifold&lt;size&gt;.</summary>
	public static Manifold CreateSphereManifold(int size) => new SphereManifold(size);

	/// <summary>colmap::CreateProductManifold.</summary>
	public static Manifold CreateProductManifold(params Manifold[] manifolds) => new ProductManifold(manifolds);

	/// <summary>colmap::ParameterBlockTangentSize.</summary>
	public static int ParameterBlockTangentSize(Problem problem, ArraySegment<double> parameters) =>
		problem.ParameterBlockTangentSize(parameters);
}
