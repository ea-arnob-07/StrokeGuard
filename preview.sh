#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/StrokeGuard_App"
echo "StrokeGuard - Clinical Decision Support System"
echo "Product design and application engineering by Estiuk Arafat Arnob"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.12 and try again."
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating an isolated preview environment..."
  python3 -m venv .venv
fi

venv_python=".venv/bin/python"

if ! "$venv_python" -c "import streamlit, sklearn, xgboost, lightgbm, catboost, plotly, reportlab" >/dev/null 2>&1; then
  echo "Installing StrokeGuard dependencies. This may take several minutes..."
  "$venv_python" -m pip install --upgrade pip
  "$venv_python" -m pip install -r requirements.txt
fi

exec "$venv_python" preview.py
