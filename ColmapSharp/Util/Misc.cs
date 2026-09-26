// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Misc: the double instantiations of VectorToCSV and CSVToVector from colmap/util/misc.h,
// with the string helpers they call from colmap/util/string.cc (StringSplit with
// token_compress_on, StringTrim, StringToDouble). First user: Scene/Camera.cs
// (ParamsToString / SetParamsFromString), whose camera_test.cc cases pin them; the int,
// float and string instantiations and misc_test.cc's CSVToVector cases come with the rest
// of util/misc when a caller needs them.
//
// VectorToCSV streams each value with an ostream's default precision (6 significant
// digits), so it goes through CppStreamFormat.FormatDouble. StringToDouble parses with
// .NET's invariant double parser instead of std::istringstream (docs/CPP_DIVERGENCES.md,
// entry 20): the two agree on plain decimal and exponent notation.

using System.Globalization;
using System.Text;

namespace ColmapSharp.Util;

/// <summary>Port of the CSV helpers of colmap/util/misc.h.</summary>
public static class Misc
{
	// colmap::IsNotWhiteSpace treats exactly these as white space.
	private static readonly char[] WhiteSpace = [' ', '\n', '\r', '\t'];

	private static readonly char[] CsvDelimiters = [',', ';'];

	/// <summary>
	/// VectorToCSV: the values joined with ", ", each formatted as a C++ ostream does by
	/// default; "" for no values.
	/// </summary>
	public static string VectorToCsv(IReadOnlyList<double> values)
	{
		var builder = new StringBuilder();
		for (int i = 0; i < values.Count; i++)
		{
			if (i > 0)
			{
				builder.Append(", ");
			}

			builder.Append(CppStreamFormat.FormatDouble(values[i]));
		}

		return builder.ToString();
	}

	/// <summary>
	/// CSVToVector&lt;double&gt;: splits on ',' and ';', trims each element, skips empty ones
	/// and parses the rest. Returns an empty list if any element fails to parse, logging the
	/// failure as COLMAP does.
	/// </summary>
	public static List<double> CsvToDoubleVector(string csv)
	{
		// boost::split with token_compress_on merges adjacent delimiters; the empty elements
		// that plain splitting yields instead are skipped below either way.
		string[] elems = csv.Split(CsvDelimiters);
		var values = new List<double>(elems.Length);
		foreach (string rawElem in elems)
		{
			string elem = rawElem.Trim(WhiteSpace);
			if (elem.Length == 0)
			{
				continue;
			}

			if (!TryStringToDouble(elem, out double value))
			{
				Log.Error($"Failed to convert CSV element: {elem}");
				return [];
			}

			values.Add(value);
		}

		return values;
	}

	// StringToDouble without the throw: COLMAP's THROW_CHECK failure is caught by
	// CSVToVector, so only the success flag matters to callers here.
	private static bool TryStringToDouble(string str, out double value) =>
		double.TryParse(str, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
}
