#!/bin/bash
# Linework.app entry point (copied to Contents/MacOS/Linework by build_app.sh).
# Code lives inside the app, the Python environment, models and presets in
# ~/Library/Application Support/Linework, drawings in ~/Pictures/Linework.
# Nothing may run from ~/Documents: macOS would stop the app with a privacy prompt.
RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
SRC="$RES/src"
SUPPORT="$HOME/Library/Application Support/Linework"
LOG="$HOME/Library/Logs/Linework.log"
mkdir -p "$SUPPORT" "$HOME/Library/Logs"
export LINEWORK_WEIGHTS="$SUPPORT/weights" LINEWORK_PRESETS="$SUPPORT/presets" \
       LINEWORK_OUTPUT="$HOME/Pictures/Linework" LINEWORK_ICON="$RES/AppIcon.icns"
printf '\n=== %s ===\n' "$(date)" >> "$LOG"

STAMP="$SUPPORT/venv/.linework-requirements"  # written by setup_env.sh
if [ -x "$SUPPORT/venv/bin/python" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$(shasum -a 256 "$SRC/requirements.txt" | cut -d' ' -f1)" ]; then
  cd "$SRC"
  exec "$SUPPORT/venv/bin/python" -u -m linework.desktop >> "$LOG" 2>&1  # exec keeps the Dock icon and name
fi

# First start (or changed requirements): install, then open the app again. After minutes of
# setup macOS has given up on this launch, and Python started now would show up as "python".
osascript -e 'display dialog "Linework is getting ready. The first start takes a few minutes and downloads about 1 GB. The window opens by itself when it is done." with title "Linework" buttons {"OK"} default button 1 giving up after 120' >/dev/null 2>&1 &
if ! "$SRC/macos/setup_env.sh" "$SUPPORT/venv" "$SRC/requirements.txt" >> "$LOG" 2>&1; then
  osascript -e 'display alert "Linework could not be set up" message "Check the internet connection and open Linework again. Details are in ~/Library/Logs/Linework.log." as critical' >/dev/null 2>&1
  exit 1
fi
open -n "$(cd "$RES/../.." && pwd)"  # a fresh launch; this one ends here
