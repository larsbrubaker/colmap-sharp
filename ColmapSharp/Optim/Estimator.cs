// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Estimator: the C# face of COLMAP's implicit "Estimator" template concept that
// colmap/optim/ransac.h and loransac.h are written against (X_t, Y_t, M_t, kMinNumSamples,
// Estimate, Residuals, and the optional Refine hook of loransac.h). COLMAP never spells the
// concept out; every class in estimators/ and estimators/solvers/ satisfies it by shape.
// Here it is two interfaces: IEstimator for RANSAC's minimal-sample estimator and
// ILocalEstimator for LO-RANSAC's local optimizer. Consumers: Ransac.cs and LoRansac.cs next
// to this file. First implementer: Estimators/Solvers/SimilarityTransform.cs.
//
// Translation notes (these shape every Phase 6 estimator):
// - C# has no associated types, so X_t, Y_t and M_t become the type parameters TX, TY and
//   TModel. RANSAC names them again next to the estimator type
//   (Ransac<TEstimator, TX, TY, TModel>); the constraint ties them together.
// - kMinNumSamples is `static abstract`, so generic code reads TEstimator.MinNumSamples
//   with no instance, exactly like Estimator::kMinNumSamples.
// - Estimate and Residuals are instance methods, because some COLMAP estimators carry
//   state (options, camera, the DEGENSAC estimator). Stateless ones are declared static in
//   C++; here they are instance methods of an empty struct. Implement estimators as
//   structs: RANSAC is generic over the concrete type, so the JIT specializes the whole
//   loop per estimator with direct (inlinable) calls and no boxing. RANSAC copies the
//   estimator for its worker (`Estimator thread_estimator = estimator;`), which is a real
//   copy only for a struct, so an estimator with mutable state must be a struct.
// - The std::vector inputs become ReadOnlySpan, so arrays and lists (via
//   CollectionsMarshal.AsSpan) both work without copying.
// - Residuals writes into a caller-sized span (length == x.Length) instead of resizing a
//   std::vector. Every COLMAP estimator resizes to the number of input pairs, and RANSAC's
//   THROW_CHECK_EQ(residuals.size(), num_samples) after each call only restates that, so the
//   span length is the contract. RANSAC reuses its buffers across calls as COLMAP reuses
//   the vector, so an estimator that (like generalized_absolute_pose's
//   `resize(n, 0)`) leaves some slots untouched sees the previous values there, as in C++.
// - Estimate appends to `models`, which RANSAC has cleared; an implementation may clear it
//   again as COLMAP's do.
// - loransac.h detects at compile time whether the local estimator has
//   `Refine(X, Y, M_t*)` and then refines the current best model instead of calling
//   Estimate on the inliers. C# cannot branch on "has a method" without reflection or
//   boxing, so ILocalEstimator has a single EstimateLocal that receives the current best
//   model: a plain estimator forwards to its Estimate (ignoring the model), a refiner
//   copies the model, refines it and appends it on success. That keeps the choice with the
//   estimator author, where COLMAP's overload detection puts it.

namespace ColmapSharp.Optim;

/// <summary>
/// A model estimator for RANSAC: estimates candidate models from a minimal sample and
/// measures every data pair against a model. The C# face of COLMAP's Estimator concept.
/// </summary>
/// <typeparam name="TX">Independent variable (COLMAP's X_t).</typeparam>
/// <typeparam name="TY">Dependent variable (COLMAP's Y_t).</typeparam>
/// <typeparam name="TModel">Model (COLMAP's M_t).</typeparam>
public interface IEstimator<TX, TY, TModel>
{
	/// <summary>The minimum number of samples needed to estimate a model (kMinNumSamples).</summary>
	static abstract int MinNumSamples { get; }

	/// <summary>
	/// Estimate zero or more models from the sampled pairs and append them to
	/// <paramref name="models"/>, which the caller has cleared.
	/// </summary>
	void Estimate(ReadOnlySpan<TX> x, ReadOnlySpan<TY> y, List<TModel> models);

	/// <summary>
	/// Compute the residual of every pair (x[i], y[i]) under <paramref name="model"/> into
	/// <paramref name="residuals"/>, whose length is <c>x.Length</c>. RANSAC compares the
	/// residuals with the squared max error, so they are squared errors.
	/// </summary>
	void Residuals(ReadOnlySpan<TX> x, ReadOnlySpan<TY> y, in TModel model, Span<double> residuals);
}

/// <summary>
/// The local optimizer of LO-RANSAC (LORANSAC's LocalEstimator parameter): re-estimates the
/// model from the current inlier set, either from scratch (COLMAP's Estimate) or by refining
/// the current best model (COLMAP's optional Refine hook). Its X, Y and model types are
/// those of the minimal estimator, as LORANSAC assigns between them.
/// </summary>
public interface ILocalEstimator<TX, TY, TModel>
{
	/// <summary>
	/// The minimum number of inliers needed for a local estimate (kMinNumSamples); LO-RANSAC
	/// skips local optimization below it.
	/// </summary>
	static abstract int MinNumSamples { get; }

	/// <summary>
	/// Estimate models from the inliers <paramref name="x"/>, <paramref name="y"/> and append
	/// them to <paramref name="models"/>, which the caller has cleared.
	/// <paramref name="initialModel"/> is the current best model: an estimator whose C++ class
	/// has <c>Refine(X, Y, M_t*)</c> refines a copy of it and appends the copy only when
	/// Refine succeeds; any other estimator ignores it and runs its Estimate.
	/// </summary>
	void EstimateLocal(ReadOnlySpan<TX> x, ReadOnlySpan<TY> y, in TModel initialModel, List<TModel> models);

	/// <inheritdoc cref="IEstimator{TX, TY, TModel}.Residuals"/>
	void Residuals(ReadOnlySpan<TX> x, ReadOnlySpan<TY> y, in TModel model, Span<double> residuals);
}
