#!/bin/bash
# Linework launcher for macOS. Double-click this file.
# First run: sets up a private Python environment in .venv (a few minutes, ~1 GB).
# Later runs: starts in a few seconds and opens the browser.
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "First start: setting up Linework (only happens once)..."
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    PY=""
    for c in python3.13 python3.12 python3.11 python3.10 python3; do
      if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
        PY="$c"; break
      fi
    done
    if [ -z "$PY" ]; then
      echo "No suitable Python found. Installing uv (Python manager from astral.sh)..."
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
      uv venv --python 3.12 .venv
      uv pip install --python .venv/bin/python -r requirements.txt
    else
      "$PY" -m venv .venv
      .venv/bin/python -m pip install --upgrade pip
      .venv/bin/pip install -r requirements.txt
    fi
  fi
fi

echo "Starting Linework... (close this window to quit)"
exec .venv/bin/python app.py
