#!/usr/bin/env bash
set -euo pipefail

VENV=".venv"

if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r requirements.txt
  echo "Dependencies installed."
fi

exec "$VENV/bin/uvicorn" app.main:app --reload --host 0.0.0.0 --port "${PORT:-8000}"
