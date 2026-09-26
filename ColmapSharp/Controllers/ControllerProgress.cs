// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// ControllerProgress: the progress report of the long-running controllers (feature
// extraction, feature matching, ...). It replaces COLMAP's progress-style LOG(INFO) lines
// ("Processed file [3/10]", "Processing block [2/5]") so MatterCAD can show a progress bar;
// warnings and errors go to Util/Log.cs. C#-only: COLMAP has no counterpart.

namespace ColmapSharp.Controllers;

/// <summary>
/// One progress step of a controller: <paramref name="Done"/> of <paramref name="Total"/>
/// units of <paramref name="Stage"/> are finished; <paramref name="Message"/> names the unit
/// just finished (e.g. the image name). A <paramref name="Total"/> of 0 means the total is not
/// known in advance (e.g. transitive matching, whose batches depend on earlier results).
/// </summary>
public readonly record struct ControllerProgress(string Stage, int Done, int Total, string Message);
