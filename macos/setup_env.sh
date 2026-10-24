#!/bin/bash
# Makes sure $1 is a Python 3.10-3.13 virtual environment with the packages from $2.
# Creates it on first run (uv if available, else a suitable python3, else installs uv
# from astral.sh first) and reinstalls only when the requirements file has changed.
set -e
VENV="$1"
REQ="$2"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"  # apps start with a bare PATH
STAMP="$VENV/.linework-requirements"
WANT="$(shasum -a 256 "$REQ" | cut -d' ' -f1)"

if [ -x "$VENV/bin/python" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$WANT" ]; then
  exit 0
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "Setting up Linework (only happens once, a few minutes, about 1 GB)..."
  if ! command -v uv >/dev/null 2>&1; then
    for c in python3.13 python3.12 python3.11 python3.10 python3; do
      if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
        PY="$c"; break
      fi
    done
    if [ -z "$PY" ]; then
      echo "No suitable Python found. Installing uv (Python manager from astral.sh)..."
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
  fi
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$VENV"
  else
    "$PY" -m venv "$VENV"
  fi
else
  echo "Updating Linework's Python packages..."
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VENV/bin/python" -r "$REQ"
else
  "$VENV/bin/python" -m pip --version >/dev/null 2>&1 || "$VENV/bin/python" -m ensurepip --upgrade
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -r "$REQ"
fi
echo "$WANT" > "$STAMP"
