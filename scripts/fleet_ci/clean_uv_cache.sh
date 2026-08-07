#!/usr/bin/env bash

set -euo pipefail

uv cache clean 2>/dev/null || true
rm -rf .venv .pytest_cache .ruff_cache .mypy_cache __pycache__
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
