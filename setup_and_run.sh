#!/usr/bin/env bash
set -euo pipefail

if ! command -v git >/dev/null 2>&1; then
    echo "Git is required. Install it with: xcode-select --install"
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required. Install it from https://www.python.org/downloads/macos/ or with Homebrew: brew install python"
    exit 1
fi

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "Run this script from a cloned project repository."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f app.py ]]; then
    echo "Could not find app.py next to this script."
    exit 1
fi

echo "Git: $(git --version)"
echo "Python: $(python3 --version)"

if [[ ! -x .venv/bin/python ]]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Installing project dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install streamlit google-genai pandas plotly graphviz

echo "Starting GreenH2 Studio..."
exec .venv/bin/python -m streamlit run app.py
