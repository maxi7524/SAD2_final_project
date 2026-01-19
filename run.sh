#!/usr/bin/env bash
set -e
uv sync
uv pip install -e .
uv source .venv/bin/activate