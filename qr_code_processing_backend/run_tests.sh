#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
pytest -q
