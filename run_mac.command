#!/usr/bin/env bash
#
# Double-clickable launcher for the mac Debug build of the ColmapSharp demo (demo/ColmapDemo.Mac).
#
# macOS opens .command files in Terminal, so this is the Finder-friendly way to run the demo
# straight from the working copy: it builds Debug (dotnet build is incremental, so an unchanged
# tree just re-verifies and exits in a few seconds) and then runs what it built. What you get is
# the DEBUG build of the CURRENT SOURCE - whatever is checked out right now, unoptimized and with
# assertions live.
#
# When the app quits cleanly the launcher closes its own Terminal window, so a normal run leaves
# nothing behind; a crash or a build failure holds the window open so the output stays readable.
#
# The demo reads a few developer environment variables (demo/ColmapDemo/DevAutoRun.cs), none of
# which this script sets: COLMAP_DEMO_PHOTOS=<dir> and COLMAP_DEMO_VIDEO=<file> preload input,
# COLMAP_DEMO_AUTORUN=1 presses Run, and COLMAP_DEMO_SCREENSHOT(_SPARSE)=<png> saves screenshots.
# agg's AGG_SMOKE_FRAMES=<n> / AGG_SMOKE_SCREENSHOT=<png> render n frames, save a PNG and exit.
# Set them on the command line, e.g. COLMAP_DEMO_AUTORUN=1 ./run_mac.command

set -euo pipefail
cd "$(dirname "$0")"

# Finder starts Terminal with a bare login PATH, and a bash script does not read ~/.zshenv, which
# is where the interactive shell picks these up. Name them here or dotnet simply will not be found.
export PATH="$HOME/.dotnet:$HOME/.local/bin:$PATH"

if ! command -v dotnet >/dev/null 2>&1; then
	echo "Could not find 'dotnet' on PATH (looked in \$HOME/.dotnet and \$HOME/.local/bin)."
	echo "Install the .NET 10 SDK, or edit the PATH line in this script to point at your dotnet."
	read -n 1 -s -r -p "Press any key to close."
	echo
	exit 1
fi

# The demo's window and AppKit host come from the demo/agg-sharp submodule, which a plain clone
# leaves empty.
if [ ! -f demo/agg-sharp/PlatformMac/PlatformMac.csproj ]; then
	echo "Fetching the agg-sharp submodule (first run only)..."
	if ! git submodule update --init --recursive demo/agg-sharp; then
		echo
		read -n 1 -s -r -p "Could not fetch demo/agg-sharp. Press any key to close."
		echo
		exit 1
	fi
fi

# agg-sharp's AppKit host is plain net10.0, so a plain Debug build of the mac head already
# produces a runnable apphost - no -r osx-arm64 and no workload needed. -clp:ErrorsOnly keeps
# warnings out of the way; errors still print.
echo "Building ColmapDemo.Mac (Debug)..."
if ! dotnet build demo/ColmapDemo.Mac/ColmapDemo.Mac.csproj -c Debug --nologo -v q -clp:ErrorsOnly; then
	echo
	echo "Build failed - see the errors above."
	read -n 1 -s -r -p "Build failed. Press any key to close."
	echo
	exit 1
fi

echo "Starting ColmapDemo..."
status=0
./demo/ColmapDemo.Mac/bin/Debug/net10.0/ColmapDemo.Mac "$@" || status=$?

if [ "$status" -ne 0 ]; then
	echo
	echo "ColmapDemo exited with code $status"
	read -n 1 -s -r -p "Press any key to close."
	echo
	exit "$status"
fi

# A clean quit closes this window and only this window. Terminal sets TERM_SESSION_ID, so its
# absence means we were run from an ordinary shell and have no window of our own to close.
#
# The window is identified by the tty of one of its tabs rather than by "front window", because
# the user may well have focused something else while the app was up. tty is a property of tab,
# not of window, so we walk tabs instead of asking for "first window whose ...".
#
# The close is detached and delayed so this shell exits first: Terminal will not prompt about
# closing a window whose shell has already finished.
if [ -n "${TERM_SESSION_ID:-}" ]; then
	own_tty="$(tty)"
	(
		sleep 0.2
		osascript -e 'on run argv
	set target to item 1 of argv
	tell application "Terminal"
		repeat with w in windows
			repeat with t in tabs of w
				if tty of t is target then
					close w
					return
				end if
			end repeat
		end repeat
	end tell
end run' "$own_tty"
	) >/dev/null 2>&1 &
fi
