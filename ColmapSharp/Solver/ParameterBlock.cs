// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from Ceres Solver 2.2.0 internal/ceres/parameter_block.h and
// internal/ceres/array_utils.cc (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// A parameter block is a view of user memory: Ceres keys it by the user's double*, here by
// an (array, offset) pair with the size the cost function (or AddParameterBlock) gives it.
// During a solve its State moves to a slice of the minimizer's own state vector (x or the
// candidate point), so the user's array is only written when the solve ends, as in Ceres.
// Problem.cs owns the blocks; Program.cs orders them and assigns their state and tangent
// offsets; ResidualBlock.cs reads State and PlusJacobian. Bounds (SetParameterLowerBound)
// are not ported yet: they make a problem "constrained", which switches the trust-region
// minimizer to a projected line search that nothing in this step needs.

namespace ColmapSharp.Solver;

/// <summary>ceres::internal::ParameterBlock: one block of parameters the solver optimizes.</summary>
internal sealed class ParameterBlock
{
	private Manifold? manifold;
	private double[]? plusJacobian;

	/// <summary>Creates a block over <paramref name="userState"/>, at position <paramref name="index"/>.</summary>
	public ParameterBlock(ArraySegment<double> userState, int index)
	{
		UserState = userState;
		State = userState;
		Index = index;
	}

	/// <summary>The user's memory for the block (Ceres' user_state()).</summary>
	public ArraySegment<double> UserState { get; }

	/// <summary>Where the block's values are read from during evaluation (Ceres' state()).</summary>
	public ArraySegment<double> State { get; private set; }

	/// <summary>Number of ambient parameters.</summary>
	public int Size => UserState.Count;

	/// <summary>The manifold, or null for plain Euclidean space.</summary>
	public Manifold? Manifold => manifold;

	/// <summary>Set by SetParameterBlockConstant.</summary>
	public bool IsSetConstant { get; set; }

	/// <summary>Constant if set so, or if the manifold leaves no tangent space.</summary>
	public bool IsConstant => IsSetConstant || TangentSize == 0;

	/// <summary>Dimension of the step space.</summary>
	public int TangentSize => manifold?.TangentSize ?? Size;

	/// <summary>Position in the owning program's parameter block list (or a scratch marker).</summary>
	public int Index { get; set; }

	/// <summary>Offset of the block in the program's state vector.</summary>
	public int StateOffset { get; set; } = -1;

	/// <summary>Offset of the block in the program's tangent (delta) vector.</summary>
	public int DeltaOffset { get; set; } = -1;

	/// <summary>Row-major Size x TangentSize Jacobian of Plus at State, or null without a manifold.</summary>
	public double[]? PlusJacobian => plusJacobian;

	/// <summary>
	/// Ceres' SetManifold: the manifold's ambient size must match the block's, and its
	/// Plus Jacobian must be computable at the current state.
	/// </summary>
	public void SetManifold(Manifold? newManifold)
	{
		if (ReferenceEquals(newManifold, manifold))
		{
			return;
		}

		if (newManifold is null)
		{
			manifold = null;
			plusJacobian = null;
			return;
		}

		if (newManifold.AmbientSize != Size)
		{
			throw new ArgumentException(
				$"The parameter block has size = {Size} while the manifold has ambient size = {newManifold.AmbientSize}");
		}

		manifold = newManifold;
		plusJacobian = new double[newManifold.AmbientSize * newManifold.TangentSize];
		if (!UpdatePlusJacobian())
		{
			throw new InvalidOperationException("Manifold::PlusJacobian computation failed for the parameter block's current values.");
		}
	}

	/// <summary>
	/// Points the block at <paramref name="x"/> (a Size-long view) and refreshes the Plus
	/// Jacobian there. Only for variable blocks, as in Ceres.
	/// </summary>
	public bool SetState(ArraySegment<double> x)
	{
		if (IsConstant)
		{
			throw new InvalidOperationException("Tried to set the state of a constant parameter block.");
		}

		State = x;
		return UpdatePlusJacobian();
	}

	/// <summary>Restores State to the user's memory without touching the Plus Jacobian.</summary>
	public void ResetStateToUserState() => State = UserState;

	/// <summary>Copies the current state into <paramref name="destination"/> unless it already is it.</summary>
	public void GetState(ArraySegment<double> destination)
	{
		if (ReferenceEquals(destination.Array, State.Array) && destination.Offset == State.Offset)
		{
			return;
		}

		State.AsSpan().CopyTo(destination.AsSpan(0, Size));
	}

	/// <summary>x_plus_delta = Plus(x, delta), through the manifold if there is one.</summary>
	public bool Plus(ReadOnlySpan<double> x, ReadOnlySpan<double> delta, Span<double> xPlusDelta)
	{
		if (manifold is not null)
		{
			return manifold.Plus(x, delta, xPlusDelta);
		}

		for (int i = 0; i < Size; i++)
		{
			xPlusDelta[i] = x[i] + delta[i];
		}

		return true;
	}

	private bool UpdatePlusJacobian()
	{
		if (manifold is null)
		{
			return true;
		}

		Span<double> jacobian = plusJacobian;
		ArrayValidity.Invalidate(jacobian);
		if (!manifold.PlusJacobian(State.AsSpan(), jacobian))
		{
			return false;
		}

		return ArrayValidity.IsValid(jacobian);
	}
}

/// <summary>
/// Ceres' array_utils: evaluation outputs are pre-filled with an impossible value so a cost
/// function that forgets to write one is caught, along with NaN and infinity.
/// </summary>
internal static class ArrayValidity
{
	/// <summary>Ceres' kImpossibleValue.</summary>
	public const double ImpossibleValue = 1e302;

	/// <summary>Fills <paramref name="x"/> with <see cref="ImpossibleValue"/>.</summary>
	public static void Invalidate(Span<double> x) => x.Fill(ImpossibleValue);

	/// <summary>True if every entry is finite and was written.</summary>
	public static bool IsValid(ReadOnlySpan<double> x)
	{
		foreach (double value in x)
		{
			if (!double.IsFinite(value) || value == ImpossibleValue)
			{
				return false;
			}
		}

		return true;
	}
}
