// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Log: where COLMAP's LOG(WARNING) and LOG(ERROR) messages (colmap/util/logging.h, glog)
// go. A library must not write to the console on its host's behalf, so messages go to a
// settable sink that MatterCAD (or a test) points wherever it wants; with no sink they are
// dropped. Progress-style LOG(INFO) output is not routed here: the long-running controllers
// report it through IProgress (Controllers/ControllerProgress.cs) instead. LOG(FATAL) maps to
// an exception at each site (CLAUDE.md), and the THROW_CHECK family lives in Check.cs.

namespace ColmapSharp.Util;

/// <summary>The severity of a <see cref="Log"/> message (glog's WARNING and ERROR).</summary>
public enum LogLevel
{
	/// <summary>LOG(WARNING).</summary>
	Warning,

	/// <summary>LOG(ERROR).</summary>
	Error,
}

/// <summary>Receives COLMAP's LOG(WARNING) / LOG(ERROR) messages.</summary>
public static class Log
{
	/// <summary>
	/// The message sink. Null (the default) drops messages. It may be called from worker
	/// threads, so a sink must be thread safe.
	/// </summary>
	public static Action<LogLevel, string>? Sink { get; set; }

	/// <summary>LOG(WARNING) &lt;&lt; message.</summary>
	public static void Warning(string message) => Sink?.Invoke(LogLevel.Warning, message);

	/// <summary>LOG(ERROR) &lt;&lt; message.</summary>
	public static void Error(string message) => Sink?.Invoke(LogLevel.Error, message);
}
