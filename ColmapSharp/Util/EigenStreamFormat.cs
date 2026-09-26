// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// EigenStreamFormat: the text Eigen's operator<< writes for a matrix with its default
// IOFormat, which COLMAP streams straight into files (ExportRecon3D's rotation matrix and
// translation row, Scene/ReconstructionIO.Export.cs). Written from Eigen's documented
// IOFormat defaults, not from Eigen's source (Eigen is MPL-2.0, docs/LICENSE_AUDIT.md):
// precision = StreamPrecision (the stream's own precision), coefficient separator " ", row
// separator "\n", no prefixes or suffixes, and, without the DontAlignCols flag, every
// coefficient right-aligned with spaces to the width of the widest coefficient of the
// matrix. Each coefficient is formatted as the stream would (Util/CppStreamFormat.cs). No
// trailing newline.

using System.Text;

namespace ColmapSharp.Util;

/// <summary>Eigen's default matrix stream output.</summary>
internal static class EigenStreamFormat
{
	/// <summary>
	/// <c>stream &lt;&lt; matrix</c> for a <paramref name="rows"/> x <paramref name="cols"/>
	/// matrix whose coefficient (r, c) is <c>coefficient(r, c)</c>, on a stream with
	/// <c>precision(<paramref name="precision"/>)</c>.
	/// </summary>
	public static string FormatMatrix(int rows, int cols, Func<int, int, double> coefficient, int precision)
	{
		var texts = new string[rows, cols];
		int width = 0;
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				texts[r, c] = CppStreamFormat.FormatDouble(coefficient(r, c), precision);
				width = Math.Max(width, texts[r, c].Length);
			}
		}

		var builder = new StringBuilder();
		for (int r = 0; r < rows; r++)
		{
			if (r > 0)
			{
				builder.Append('\n');
			}

			for (int c = 0; c < cols; c++)
			{
				if (c > 0)
				{
					builder.Append(' ');
				}

				builder.Append(texts[r, c].PadLeft(width));
			}
		}

		return builder.ToString();
	}
}
