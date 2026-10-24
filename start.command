#!/bin/bash
# Linework launcher for macOS (opens in the browser). Double-click this file.
# For a Dock app instead, build one with macos/build_app.sh.
set -e
cd "$(dirname "$0")"
macos/setup_env.sh .venv requirements.txt
echo "Starting Linework... (close this window to quit)"
exec .venv/bin/python app.py
