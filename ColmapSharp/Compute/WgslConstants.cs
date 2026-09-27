// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// WgslConstants: the generated constants header of a composed WGSL kernel. Not a COLMAP port
// (PORTING_PLAN.md Phase 13). WebGPU pipeline override constants are not available to the
// hosts this library targets, so per-problem values (window radius, layer sizes, source
// count, ...) are baked into the kernel text as module-scope `const` declarations instead.
// ColmapSharp/Mvs/PatchMatchShaders.cs puts this header in front of the .wgsl parts it composes.
//
// The output is deterministic: declarations are sorted by name (ordinal), one per line, and
// formatted with invariant rules, so equal constants always give byte-identical WGSL (and a
// host's shader cache can key on the text). Floats are written as WGSL hexadecimal float
// literals, which denote the f32 exactly: no decimal round trip can change a bit. (A
// `bitcast<f32>(0x...u)` would say the same, but naga, the WGSL front end of wgpu, does not
// evaluate bitcast in const expressions.)

using System.Globalization;
using System.Text;

namespace ColmapSharp.Compute;

/// <summary>A set of named WGSL <c>const</c> declarations, written as a deterministic header.</summary>
internal sealed class WgslConstants
{
	private readonly SortedDictionary<string, string> declarations = new(StringComparer.Ordinal);

	/// <summary>The number of constants.</summary>
	public int Count => declarations.Count;

	/// <summary>Adds <c>const name: i32</c>.</summary>
	public WgslConstants Add(string name, int value)
	{
		// -2147483648i does not parse: the literal 2147483648i is out of range before the
		// negation applies.
		string literal = value == int.MinValue
			? "(-2147483647i - 1i)"
			: value.ToString(CultureInfo.InvariantCulture) + "i";
		return Declare(name, "i32", literal, null);
	}

	/// <summary>Adds <c>const name: u32</c>.</summary>
	public WgslConstants Add(string name, uint value)
		=> Declare(name, "u32", value.ToString(CultureInfo.InvariantCulture) + "u", null);

	/// <summary>Adds <c>const name: bool</c>.</summary>
	public WgslConstants Add(string name, bool value) => Declare(name, "bool", value ? "true" : "false", null);

	/// <summary>
	/// Adds <c>const name: f32</c>, exactly. NaN and infinities are rejected: WGSL makes a
	/// const expression that evaluates to one a shader-creation error.
	/// </summary>
	public WgslConstants Add(string name, float value)
		=> Declare(name, "f32", FormatF32(value), value.ToString("R", CultureInfo.InvariantCulture));

	/// <summary>The header: one <c>const</c> line per constant, sorted by name.</summary>
	public string ToWgsl()
	{
		var text = new StringBuilder();
		foreach (string declaration in declarations.Values)
		{
			text.Append(declaration).Append('\n');
		}

		return text.ToString();
	}

	/// <summary>
	/// <paramref name="value"/> as a WGSL hexadecimal float literal with the f32 suffix that
	/// denotes exactly that float: <c>0x1.XXXXXXp±E</c> for normal numbers (the 23 fraction
	/// bits shifted into six hex digits), <c>0x0.XXXXXXp-126</c> for subnormals, and
	/// <c>0.0f</c> or <c>-0.0f</c> for the zeros.
	/// </summary>
	internal static string FormatF32(float value)
	{
		uint bits = BitConverter.SingleToUInt32Bits(value);
		string sign = (bits >> 31) != 0 ? "-" : string.Empty;
		int exponent = (int)((bits >> 23) & 0xFF);
		uint fraction = bits & 0x7FFFFF;
		if (exponent == 0xFF)
		{
			throw new ArgumentException($"A WGSL constant must be finite, but it is {value.ToString(CultureInfo.InvariantCulture)}.", nameof(value));
		}

		if (exponent == 0 && fraction == 0)
		{
			return sign + "0.0f";
		}

		// Six hex digits hold 24 bits: the 23 fraction bits and a trailing 0.
		string digits = (fraction << 1).ToString("X6", CultureInfo.InvariantCulture);
		if (exponent == 0)
		{
			return sign + "0x0." + digits + "p-126f";
		}

		int unbiased = exponent - 127;
		string exponentText = (unbiased >= 0 ? "+" : "-") + Math.Abs(unbiased).ToString(CultureInfo.InvariantCulture);
		return sign + "0x1." + digits + "p" + exponentText + "f";
	}

	private WgslConstants Declare(string name, string type, string literal, string? note)
	{
		if (!IsIdentifier(name))
		{
			throw new ArgumentException($"\"{name}\" is not a WGSL identifier: use letters, digits and '_', not starting with a digit.", nameof(name));
		}

		if (declarations.ContainsKey(name))
		{
			throw new ArgumentException($"The WGSL constant {name} is already defined.", nameof(name));
		}

		string declaration = $"const {name}: {type} = {literal};";
		declarations.Add(name, note == null ? declaration : declaration + " // " + note);
		return this;
	}

	private static bool IsIdentifier(string name)
	{
		if (string.IsNullOrEmpty(name) || char.IsAsciiDigit(name[0]) || name == "_" || name.StartsWith("__", StringComparison.Ordinal))
		{
			return false;
		}

		foreach (char c in name)
		{
			if (!char.IsAsciiLetterOrDigit(c) && c != '_')
			{
				return false;
			}
		}

		return true;
	}
}
