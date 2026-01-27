#!/usr/bin/env bash
set -euo pipefail

# Resolve the directory where this script is located (works regardless of CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Repository root is one level above scripts/
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Target data directory
DATA_DIR="$REPO_DIR/data"

# Google Drive folder ID
FOLDER_ID="1fbIJiBIGMP9HivtoPEkko7hT6t2mXH0n"

# Create data directory if it does not exist
mkdir -p "$DATA_DIR"

# Download all files from the Google Drive folder into data/
# Requires: gdown (pip install gdown)
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "$DATA_DIR"

# Enable safe globbing (no match -> empty list)
shopt -s nullglob

# Extract all ZIP files and remove them afterwards
for zip_file in "$DATA_DIR"/*.zip; do
    unzip -o "$zip_file" -d "$DATA_DIR"
    rm "$zip_file"
done

# Disable nullglob to avoid side effects
shopt -u nullglob

echo "Data download and extraction completed successfully."
