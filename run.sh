#!/usr/bin/env bash
set -e
uv sync
uv pip install -e .
source .venv/bin/activate